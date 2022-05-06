import json

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from rich import print as rprint
from rich.traceback import install

from models import relative_position_bucket
from train import Panacea
from datamodules import SingleCellDataset, PanaceaDataModule


install()
sns.set_style("ticks")
sns.set_palette("Set2")
plt.rcParams["figure.dpi"] = 300
pl.seed_everything(42)

with open("config_data.yaml") as f:
    config_data = yaml.safe_load(f)
with open("config_model.yaml") as f:
    config_model = yaml.safe_load(f)

DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/panacea"
config_model["gpus"] = []
datamodule = PanaceaDataModule(data_dir=DATA_DIR, **config_data)
datamodule.setup("fit")
dataloader = datamodule.train_dataloader()
config = {**config_model, **config_data}
model = Panacea(**config)

for batch in iter(dataloader):
    if "D28.3_41" in batch["cell_name"]:
        rprint(batch["cell_name"])
        idx = batch["cell_name"].index("D28.3_41")
        break

self = model
stage = "train"
dataloader_idx = None

stage = "train" if self.training else "val"
if stage == "val":
    epoch = -1
    num_crops_g = 1
    ibot_metrics = self.ibot_metrics_val[dataloader_idx]
    ibot_acc_m = self.ibot_acc_val[dataloader_idx]
else:
    epoch = self.current_epoch
    num_crops_g = self.hparams.num_crops_g
    ibot_metrics = self.ibot_metrics_train
    ibot_acc_m = self.ibot_acc_train

# genes, counts, count_raw, *labels, masks = batch
genes = batch["gene"]  # list
counts = batch["count"]  # list
masks = batch["mask"]  # list
target = batch["target"]  # list
count_raw = batch["count_raw"]  # tensor
masks_gene = [mask == 1 for mask in masks]
masks_count = [mask == 2 for mask in masks]

genes_g = genes[:num_crops_g]
counts_g = counts[:num_crops_g]
masks_gene_g = masks_gene[:num_crops_g]
masks_count_g = masks_count[:num_crops_g]

# global views
_, emb_t_l, cls_t_l, patch_t_l, preds_gene_t_l, preds_count_t_l = self.teacher(
    genes_g, counts_g, mask_gene=None, mask_count=None
)
_, emb_s_l, cls_s_l, patch_s_l, preds_gene_s_l, preds_count_s_l = self.student(
    genes_g, counts_g, mask_gene=masks_gene_g, mask_count=masks_count_g
)

# local views (no masking, and patch embedding discarded)
if len(genes) > num_crops_g:
    # self.student.backbone.masked_im_modeling = False
    (
        _,
        emb_s_local_l,
        cls_s_local_l,
        patch_s_local_l,
        preds_gene_local_l,
        preds_count_local_l,
    ) = self.student(
        genes[num_crops_g:],
        counts[num_crops_g:],
        # mask_gene=masks_gene[num_crops_g:],
        # mask_count=masks_count[num_crops_g:],
        mask_gene=None,
        mask_count=None,
    )
    # self.student.backbone.masked_im_modeling = self.hparams.mim
    emb_s_l += emb_s_local_l
    cls_s_l += cls_s_local_l
    patch_s_l += patch_s_local_l
    preds_gene_s_l += preds_gene_local_l
    preds_count_s_l += preds_count_local_l

# ===== cal loss =====
log = {}

# ibot loss
loss_ibot_cls_s, loss_ibot_cls, loss_ibot_patch_gene = self.loss_ibot(
    cls_t_l, patch_t_l, cls_s_l, patch_s_l, masks_gene, epoch
)
_, _, loss_ibot_patch_count = self.loss_ibot(
    cls_t_l,
    patch_t_l,
    cls_s_l,
    patch_s_l,
    masks_count,
    epoch,
    calc_contrastive_loss=False,
)
log.update(
    {
        f"{stage}/loss_ibot_cls_s": loss_ibot_cls_s,
        f"{stage}/loss_ibot_cls": loss_ibot_cls,
        f"{stage}/loss_ibot_patch_gene": loss_ibot_patch_gene,
        f"{stage}/loss_ibot_patch_count": loss_ibot_patch_count,
    }
)

# moco loss
logit_t = cls_t_l[0]
logit_s = cls_s_l[0]
argmax_t = logit_t.argmax(dim=1)
ibot_acc_m(logit_s, argmax_t)
log.update({f"{stage}/ibot_acc": ibot_acc_m})
ibot_metrics(argmax_t, target)

loss_moco, moco_acc = self.loss_moco(
    torch.stack([i[:, 0] for i in emb_s_l], dim=1),
    torch.stack([i[:, 0] for i in emb_t_l], dim=1),
)
log.update({f"{stage}/moco_acc": moco_acc, f"{stage}/loss_moco": loss_moco})

# bert loss
counts = [count.clip(0, self.hparams.max_count - 1) for count in counts]
count_raw = count_raw.clip(0, self.hparams.max_count - 1)

rprint(counts[0][idx])
i = int(input("Please input i: "))

# =========== target ===========
masks_gene[0][idx, i] = True
masks_count[0][idx, i] = True

j_g = masks_gene[0][idx, :i].sum() + masks_gene[0][:idx].sum()
j_c = masks_count[0][idx, :i].sum() + masks_count[0][:idx].sum()
targets_gene = [
    self.get_target_genes(g, count_raw, m) for g, m in zip(genes, masks_gene)
]
targets_count = [self.get_target_counts(c, m) for c, m in zip(counts, masks_count)]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.lineplot(x=np.arange(model.hparams.num_genes), y=count_raw[idx], ax=ax, color="y")
ax_t = ax.twinx()
sns.lineplot(
    x=np.arange(model.hparams.num_genes),
    y=targets_gene[0][j_g],
    ax=ax_t,
    lw=0.1,
    alpha=0.5,
)
# a horizontal line at y=count_raw[0][139]
ax.axhline(
    y=count_raw[idx][genes[0][idx, i] - model.hparams.num_special_tokens],
    color="r",
    linestyle="--",
)
fig.savefig(f"figs/analysis_model/target_gene_all.jpg")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.lineplot(x=np.arange(100), y=count_raw[idx, 5880:5980], ax=ax, color="y")
ax_t = ax.twinx()
sns.lineplot(x=np.arange(100), y=targets_gene[0][j_g, 5880:5980], ax=ax_t)
# a horizontal line at y=count_raw[0][139]
ax.axhline(
    y=count_raw[idx][genes[0][idx, i] - model.hparams.num_special_tokens],
    color="r",
    linestyle="--",
)
fig.savefig(f"figs/analysis_model/target_gene_part.jpg")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.lineplot(
    x=np.arange(model.hparams.max_count),
    y=targets_count[0][j_c],
    ax=ax,
    lw=1,
)
# a horizontal line at y=count_raw[0][139]
ax.axvline(
    x=count_raw[idx][genes[0][idx, i] - model.hparams.num_special_tokens],
    color="r",
    linestyle="--",
)
ax.set_xscale("log")
fig.savefig(f"figs/analysis_model/target_count_all.jpg")


# =========== positional encoding ============
trm_model = model.student.backbone

# sin
# count = torch.arange(100).unsqueeze(0)
# pos_enc = trm_model.get_abs_pos_bias_sinusoidal(count)
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# for i in range(self.hparams.emb_dim // 2):
#     pd.Series(pos_enc[0, :, i]).plot(ax=ax)
# fig.savefig(f"figs/analysis_model/pos_enc_sin.jpg")
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# for i in range(self.hparams.emb_dim // 2, self.hparams.emb_dim):
#     pd.Series(pos_enc[0, :, i]).plot(ax=ax)
# fig.savefig(f"figs/analysis_model/pos_enc_cos.jpg")

# rel
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

    if mask_count is not None:
        num_mask = mask_count.sum()
        # ==== others to [MASK] ====
        rp_bucket.transpose(1, 2)[mask_count] = torch.tensor(
            self.rel_pos_bins + 1, device=rp_bucket.device
        ).expand(num_mask, rp_bucket.shape[1])
        # ==== [MASK] to others ====
        rp_bucket[mask_count] = torch.tensor(
            self.rel_pos_bins + 2, device=rp_bucket.device
        ).expand(num_mask, rp_bucket.shape[2])

    # ==== others to [CLS] ====
    rp_bucket[:, :, 0] = self.rel_pos_bins
    # [CLS] to others, Note: self.rel_pos_bins // 2 is not used in relative_position_bucket
    rp_bucket[:, 0, :] = self.rel_pos_bins // 2
    return rp_bucket


count = (torch.tensor(2) ** torch.arange(14)).unsqueeze(0)
mask = torch.bernoulli(torch.ones_like(count) * 0.2).bool()
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sns.heatmap(
    get_rel_pos_bias(trm_model, count, mask)[0].numpy(),
    annot=True,
    fmt="d",
    linewidths=0.1,
    ax=ax,
)
ax.set_xticklabels(["cls"] + count[0].tolist())
fig.savefig(f"figs/analysis_model/pos_enc_rel.jpg")

# abs
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sns.heatmap(trm_model.get_abs_pos_bias(count, mask)[0, 0].detach().numpy())
ax.set_xticklabels(["cls"] + count[0].tolist())
fig.savefig(f"figs/analysis_model/pos_enc_abs.jpg")
