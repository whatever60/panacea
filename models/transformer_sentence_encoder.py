"""https://github.com/guolinke/TUPE/blob/master/fairseq/modules/transformer_sentence_encoder.py
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm

from .multihead_attention import MultiheadAttention
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)


# this is from T5
def relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(
            torch.long
        ) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )

    ret += torch.where(is_small, n, val_if_large)
    return ret


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.
    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).
    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens
    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        mask_idx: int,
        padding_idx: int,
        cls_idx: int,
        # cls_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        emb_dim: int = 768,
        ffn_emb_dim: int = 3072,
        num_heads: int = 8,
        drop_path: float = 0.0,
        dropout: float = 0.1,
        dropout_attention: float = 0.1,
        dropout_activation: float = 0.1,
        max_seq_len: int = 256,  # max_count
        encoder_normalize_before: bool = False,
        embedding_normalize: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        embed_scale: float = None,
        rel_pos: bool = False,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        export: bool = False,
    ) -> None:

        super().__init__()
        self.mask_idx = mask_idx
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.apply_bert_init = apply_bert_init
        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.emb_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        self.attn_scale_factor = 2
        self.num_heads = num_heads
        # extra 4 for cls-to-others, others-to-cls, mask-to-others, and others-to-mask
        self.pos = nn.Embedding(self.max_seq_len + 4, self.emb_dim, padding_idx=0)
        self.pos_q_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.pos_k_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.pos_scaling = (
            float(self.emb_dim / num_heads * self.attn_scale_factor) ** -0.5
        )
        self.pos_ln = LayerNorm(self.emb_dim, export=export)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path, num_encoder_layers)
        ]  # stochastic depth decay rule
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.emb_dim,
                    ffn_embedding_dim=ffn_emb_dim,
                    num_attention_heads=num_heads,
                    dropout=self.dropout,
                    attention_dropout=dropout_attention,
                    activation_dropout=dropout_activation,
                    activation_fn=activation_fn,
                    attn_scale_factor=self.attn_scale_factor,
                    export=export,
                    encoder_normalize_before=encoder_normalize_before,
                    drop_path=p,
                )
                for p in dpr
            ]
        )

        if embedding_normalize:
            self.emb_layer_norm = LayerNorm(self.emb_dim, export=export)
        else:
            self.emb_layer_norm = None

        if encoder_normalize_before:
            self.emb_out_layer_norm = LayerNorm(self.emb_dim, export=export)
        else:
            self.emb_out_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        self.rel_pos = rel_pos
        if self.rel_pos:
            assert rel_pos_bins % 2 == 0
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.relative_attention_bias = nn.Embedding(
                self.rel_pos_bins + 1 + 2, self.num_heads
            )
            # seq_len = self.max_seq_len
            # context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            # memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            # relative_position = memory_position - context_position
            # self.rp_bucket = relative_position_bucket(
            #     relative_position,
            #     num_buckets=self.rel_pos_bins,
            #     max_distance=self.max_rel_pos,
            # )
            # # others to [CLS]
            # self.rp_bucket[:, 0] = self.rel_pos_bins
            # # [CLS] to others, Note: self.rel_pos_bins // 2 is not used in relative_position_bucket
            # self.rp_bucket[0, :] = self.rel_pos_bins // 2

    def get_rel_pos_bias(self, counts, mask_count=None):
        # counts, mask_count: [b, max_length].
        # return [b, 1+max_length, 1+max_length]

        if not self.rel_pos:
            return None

        counts = self.add_cls(counts, -123)  # doesn't matter
        if mask_count is not None:
            mask_count = self.add_cls(mask_count, False)

        context_position = counts[:, :, None]
        memory_position = counts[:, None, :]
        relative_position = memory_position - context_position
        rp_bucket = relative_position_bucket(
            relative_position,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # ==== others to [CLS] ====
        rp_bucket[:, :, 0] = self.rel_pos_bins
        # [CLS] to others, Note: self.rel_pos_bins // 2 is not used in relative_position_bucket
        rp_bucket[:, 0, :] = self.rel_pos_bins // 2
        # ==== [MASK] to others ====
        if mask_count is not None:
            num_mask = mask_count.sum()
            rp_bucket[mask_count] = torch.tensor(
                self.rel_pos_bins + 1, device=rp_bucket.device
            ).expand(num_mask, rp_bucket.shape[2])
            rp_bucket.transpose(1, 2)[mask_count] = torch.tensor(
                self.rel_pos_bins + 2, device=rp_bucket.device
            ).expand(num_mask, rp_bucket.shape[1])

        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        # if rp_bucket.device != device:
        # rp_bucket = rp_bucket.to(counts.device)
        # rp_bucket = rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        # values = values.permute([2, 0, 1])
        values = values.permute([0, 3, 1, 2])
        return values.contiguous()  # [b, num_heads, max_length, max_length]

    def get_abs_pos_bias(self, counts, mask_count=None):
        # [b, max_length]
        batch_size, seq_len = counts.shape
        # 0 is for other-to-cls 1 is for cls-to-other
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        # [b, max_length, emb_dim]
        counts = counts.clip(0, self.max_seq_len - 1)
        weight = self.pos_ln(self.pos(counts))

        to_head = lambda x: x.view(*x.shape[:-1], self.num_heads, -1)
        # [b, max_length, num_heads, emb_dim // num_heads]
        pos_q = to_head(self.pos_q_linear(weight)) * self.pos_scaling
        pos_k = to_head(self.pos_k_linear(weight))
        # abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
        abs_pos_bias = torch.einsum("bqhd, bkhd -> bqkh", pos_q, pos_k)

        pos_q = self.pos_q_linear(self.pos_ln(self.pos.weight[-4:])) * self.pos_scaling
        pos_k = self.pos_k_linear(self.pos_ln(self.pos.weight[-4:]))
        special_bias = torch.einsum("nhd, nhd -> nh", to_head(pos_q), to_head(pos_k))
        cls_2_other, other_2_cls, mask_2_other, other_2_mask = special_bias

        if mask_count is not None:
            num_mask = mask_count.sum()
            shape = num_mask, seq_len, self.num_heads
            abs_pos_bias[mask_count] = mask_2_other.expand(shape)
            abs_pos_bias.transpose(1, 2)[mask_count] = other_2_mask.expand(shape)
        abs_pos_bias = torch.cat(
            [cls_2_other.expand(batch_size, 1, seq_len, self.num_heads), abs_pos_bias],
            dim=1,
        )
        abs_pos_bias = torch.cat(
            [
                other_2_cls.expand(batch_size, seq_len + 1, 1, self.num_heads),
                abs_pos_bias,
            ],
            dim=2,
        )
        return abs_pos_bias.transpose(1, 3)  # [b, h, l, l]

    def get_abs_pos_bias_sinusoidal(self, counts):
        # [b, max_length]. -3 for pad, -2 for cls, -1 for mask.
        counts = self.add_cls(counts, -2)
        abs_pos_bias_sinusoidal = get_embedding(counts + 2, self.emb_dim)
        # if padding_mask is not None:
        #     abs_pos_bias_sinusoidal[padding_mask] = 0  # padding
        return abs_pos_bias_sinusoidal  # [b, l, d]

    def forward(
        self,
        tokens: torch.Tensor,  # [b, max_length]
        counts: torch.Tensor,  # [b, max_length]
        mask_gene: torch.Tensor = None,  # [b, max_length]
        mask_count: torch.Tensor = None,  # [b, max_length]
        # content_position: torch.Tensor,
        # memory_position: torch.Tensor,
        return_all_tokens: bool = False,
        # padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention

        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        else:
            padding_mask = self.add_cls(padding_mask, False)

        tokens = self.mask_model(tokens, mask_gene, self.mask_idx, self.vocab_size)
        counts = self.mask_model(counts, mask_count, -1, self.max_seq_len)

        # positional encoding
        # [b, 1+max_length, length, length]
        mask_count = counts == -1
        abs_pos_bias = self.get_abs_pos_bias(counts, mask_count)
        abs_pos_bias_sin = self.get_abs_pos_bias_sinusoidal(counts)
        rel_pos_bias = self.get_rel_pos_bias(counts, mask_count)
        if rel_pos_bias is not None:
            abs_pos_bias += rel_pos_bias
        abs_pos_bias = abs_pos_bias.flatten(end_dim=1)

        tokens = self.add_cls(tokens, self.cls_idx)
        # [b, length, emb_dim]
        x = self.embed_tokens(tokens) + abs_pos_bias_sin

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # account for padding while computing the representation

        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if return_all_tokens:
            inner_states.append(x)
        for layer in self.layers:
            x = layer(
                x, self_attn_padding_mask=padding_mask, self_attn_bias=abs_pos_bias
            )
            if return_all_tokens:
                inner_states.append(x)

        if self.emb_out_layer_norm is not None:
            x = self.emb_out_layer_norm(x)
            inner_states[-1] = x

        # sentence_rep = x[:, 0, :]

        # if return_all_tokens:
        #     inner_states = [x]

        # return inner_states

        if return_all_tokens:
            # [b, layer + 1, length, c]
            x = torch.stack([i.transpose(0, 1) for i in inner_states], dim=1)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return x

    def mask_model(self, x, mask, mask_token, max_):
        """80% to mask token, 10% to random token, 10% not modified.
        x: [b, max_length]
        """
        if mask is None:
            return x
        x = x.clone()
        real_mask = torch.bernoulli(torch.full_like(x, 0.8).float()).bool() & mask
        x[real_mask] = mask_token
        random_mask = (
            torch.bernoulli(torch.full_like(x, 0.5).float()).bool() & real_mask
        )
        x[random_mask] = torch.randint_like(x[random_mask], high=max_, dtype=torch.long)
        return x

    def add_cls(self, x, cls_value):
        batch_size = x.size(0)
        return torch.cat(
            [
                torch.full((batch_size, 1), cls_value, device=x.device, dtype=x.dtype),
                x,
            ],
            dim=1,
        )


def get_embedding(counts, embedding_dim):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    batch_size, length = counts.shape
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float, device=counts.device) * -emb
    )
    # emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = counts.float().unsqueeze(2) * emb.view(1, 1, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    # emb = emb.view(batch_size, length, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat(
            [
                emb,
                torch.zeros(batch_size, length, 1, device=emb.device, dtype=emb.dtype),
            ],
            dim=-1,
        )
    # if padding_idx is not None:
    #     emb[padding_idx, :] = 0
    return emb
