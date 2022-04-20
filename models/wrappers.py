from typing import Tuple
import torch
from torch import nn


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super().__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(
        self,
        x,
        pos,
        mask_gene=None,
        mask_count=None,
        return_all_tokens=None,
    ) -> Tuple[list, list, list]:
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask_gene = [mask_gene] if mask_gene is not None else None
            mask_count = [mask_count] if mask_count is not None else None
        batch_size = x[0].shape[0]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])
            inp_pos = torch.cat(pos[start_idx:end_idx])

            if mask_gene is not None:
                inp_m_g = torch.cat(mask_gene[start_idx:end_idx])
            else:
                inp_m_g = None
            if mask_count is not None:
                inp_m_c = torch.cat(mask_count[start_idx:end_idx])
            else:
                inp_m_c = None

            _out = self.backbone(
                inp_x,
                inp_pos,
                return_all_tokens=return_all_tokens,
                mask_gene=inp_m_g,
                mask_count=inp_m_c,
            )
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # output: [batch_size * num_crops, C]
        # Run the head forward on the concatenated features.
        emb, cls_emb, patch_emb, preds_gene, preds_count = self.head(output)

        num_crops = len(x)
        output_l = output.chunk(num_crops, dim=0)
        emb_l = emb.chunk(num_crops, dim=0)
        cls_emb_l = cls_emb.chunk(num_crops, dim=0)
        patch_emb_l = patch_emb.chunk(num_crops, dim=0)
        preds_gene_l = preds_gene.chunk(num_crops, dim=0)
        preds_count_l = preds_count.chunk(num_crops, dim=0)

        return output_l, emb_l, cls_emb_l, patch_emb_l, preds_gene_l, preds_count_l
