from torch import nn
from torch import distributed as dist


from .batchnorms import CSyncBatchNorm, PSyncBatchNorm, CustomSequential


class DINOHead(nn.Module):
    """
    If `bottleneck_dim = 0`, the head has `nlayers` NN layers.
    e.g., in_dim -> hid_dim -> hid_dim -> ... -> out_dim
    Otherwise, the head has `nlayers + 1` NN layers. In this case, feature will be normed before the last layer.
    e.g., in_dim -> hid_dim -> hid_dim -> ... -> bottleneck_dim -> L2-norm -> out_dim
    `norm_last_layer` is only used when `bottleneck_dim > 0`.
    In any case, `self.mlp` will be an MLP with `nlayers` layers.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        norm=None,
        act="gelu",
        dropout_p=0.0,
        last_norm=None,
        nlayers=3,
        hidden_dim=512,
        bottleneck_dim=256,
        norm_last_layer=True,
        **kwargs  # for batchnorm
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)
        dropout = nn.Dropout(dropout_p) if dropout_p else None

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            if dropout is not None:
                layers.append(dropout)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
                if dropout is not None:
                    layers.append(dropout)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)

        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, out_dim, bias=False)
            )
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm: str, hidden_dim: int, **kwargs):
        if norm == "bn":
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == "syncbn":
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == "csyncbn":
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == "psyncbn":
            norm = PSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == "ln":
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == "relu":
            act = nn.ReLU()
        elif act == "gelu":
            act = nn.GELU()
        elif act == "tanh":
            act = nn.Tanh()
        else:
            assert False, "unknown act type {}".format(act)
        return act


class iBOTHead(DINOHead):
    def __init__(
        self,
        num_genes,
        max_count,
        in_dim,
        ibot_cls_dim,
        ibot_patch_dim,
        norm=None,
        act="gelu",
        dropout_p=0.0,
        last_norm=None,
        nlayers=2,
        hidden_dim=512,
        bottleneck_dim=256,
        norm_last_layer=True,
        shared_head=False,
        **kwargs  # for batchnorm
    ):

        super(iBOTHead, self).__init__(
            in_dim=in_dim,
            out_dim=ibot_cls_dim,
            norm=norm,
            act=act,
            dropout_p=dropout_p,
            last_norm=last_norm,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            norm_last_layer=norm_last_layer,
            **kwargs
        )

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(
                    nn.Linear(bottleneck_dim, ibot_patch_dim, bias=False)
                )
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, ibot_patch_dim)
                self.last_layer2 = None

            self.last_norm2 = self._build_norm(
                last_norm, ibot_patch_dim, affine=False, **kwargs
            )
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm

        # reconstruction head
        in_dim_recon = bottleneck_dim if bottleneck_dim > 0 else hidden_dim
        self.pred_gene_head = nn.Linear(in_dim_recon, num_genes, bias=False)
        self.pred_count_head = nn.Linear(in_dim_recon, max_count, bias=False)

    def forward(self, x):
        # x: [batch_size * num_crops, 1+H*W, out_dim]
        if len(x.shape) == 2:  # CLS emb
            return super(iBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])  # [CLS]
            x2 = self.last_layer2(x[:, 1:])  # patch embedding
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])  # patch embedding

        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)

        preds_gene = self.pred_gene_head(x[:, 1:])
        preds_count = self.pred_count_head(x[:, 1:])

        return x, x1, x2, preds_gene, preds_count
