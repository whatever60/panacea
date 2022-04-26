"""
TODO:
- 接口的参数统一一下
- 改loss
- 开训！
"""

import math
from functools import lru_cache
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datamodules import PanaceaDataModule
from models import (
    iBOTLoss,
    MoCoLoss,
    BERTLoss,
    TransformerSentenceEncoder,
    MultiCropWrapper,
    iBOTHead,
    cosine_scheduler,
)
from optimizers import LARS, LARC
from metrics import iBOTMetrics
from callbacks import LogPredictionsCallback


class Panacea(pl.LightningModule):
    def __init__(self, **config_dict):
        super().__init__()
        self.save_hyperparameters()
        self.get_models()
        self.get_loss()
        self.get_metrics()

    def forward(self, x):
        return self.teacher(x, mask=None)

    def shared_step(self, batch, batch_idx, dataloader_idx=0):
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
        targets_gene = [
            self.get_target_genes(g, count_raw, m) for g, m in zip(genes, masks_gene)
        ]
        targets_count = [
            self.get_target_counts(c, m) for c, m in zip(counts, masks_count)
        ]

        loss_bert_t_gene = self.loss_bert(
            targets_gene[:num_crops_g], preds_gene_t_l, masks_gene[:num_crops_g]
        )
        loss_bert_t_count = self.loss_bert(
            targets_count[:num_crops_g], preds_count_t_l, masks_count[:num_crops_g]
        )
        loss_bert_s_gene_g = self.loss_bert(
            targets_gene[:num_crops_g],
            preds_gene_s_l[:num_crops_g],
            masks_gene[:num_crops_g],
        )
        loss_bert_s_count_g = self.loss_bert(
            targets_count[:num_crops_g],
            preds_count_s_l[:num_crops_g],
            masks_count[:num_crops_g],
        )
        if len(genes) > num_crops_g:
            loss_bert_s_gene_l = self.loss_bert(
                targets_gene[num_crops_g:],
                preds_gene_s_l[num_crops_g:],
                masks_gene[num_crops_g:],
            )
            loss_bert_s_count_l = self.loss_bert(
                targets_count[num_crops_g:],
                preds_count_s_l[num_crops_g:],
                masks_count[num_crops_g:],
            )
        else:
            loss_bert_s_gene_l = loss_bert_s_count_l = 0.0

        log.update(
            {
                f"{stage}/loss_bert_t_gene": loss_bert_t_gene,
                f"{stage}/loss_bert_t_count": loss_bert_t_count,
                f"{stage}/loss_bert_s_gene_g": loss_bert_s_gene_g,
                f"{stage}/loss_bert_s_count_g": loss_bert_s_count_g,
                f"{stage}/loss_bert_s_gene_l": loss_bert_s_gene_l,
                f"{stage}/loss_bert_s_count_l": loss_bert_s_count_l,
            }
        )

        loss = (
            loss_ibot_cls_s * self.hparams.lambda_ibot_ss
            + loss_ibot_cls * self.hparams.lambda_ibot_st
            + loss_ibot_patch_gene * self.hparams.lambda_ibot_gene
            + loss_ibot_patch_count * self.hparams.lambda_ibot_count
            + loss_moco * self.hparams.lambda_ibot_count
            + loss_bert_s_gene_g * self.hparams.lambda_bert_gene_g
            + loss_bert_s_count_g * self.hparams.lambda_bert_count_g
            + loss_bert_s_gene_l * self.hparams.lambda_bert_gene_l
            + loss_bert_s_count_l * self.hparams.lambda_bert_count_l
        )

        log.update({f"{stage}/loss": loss})

        self.log_dict(
            log, batch_size=self.hparams.batch_size, sync_dist=True, on_epoch=True
        )

        if batch_idx == 0:
            return dict(
                loss=loss,
                # images=images[0],
                # generated_images=unpatchify(
                #     recon_s_l[0].detach(), self.hparams.patch_size
                # ),
            )
        else:
            return dict(loss=loss)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.shared_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    @torch.no_grad()
    def training_step_end(self, output):
        # EMA update for the teacher
        m = self.scheduler_mom[self._iter]  # momentum parameter
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in self.student.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in self.teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        params_q = [
            param_q
            for name_q, param_q in zip(names_q, params_q)
            if name_q in names_common
        ]
        params_k = [
            param_k
            for name_k, param_k in zip(names_k, params_k)
            if name_k in names_common
        ]
        for param_q, param_k in zip(params_q, params_k):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        metrics = self.ibot_metrics_train.compute()
        self.ibot_metrics_train.reset()
        self.log_dict(metrics, sync_dist=True)

        flag = True if self.current_epoch >= self.hparams.freeze_last_layer else False
        for n, p in self.student.named_parameters():
            if "last_layer" in n:
                p.requires_grad = flag

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        if not "num_batches" in self.hparams:
            metrics = self.ibot_metrics_val.compute()
            self.ibot_metrics_val.reset()
        else:
            metrics = {
                f"{k}/{idx}": v
                for idx, m in enumerate(self.ibot_metrics_val)
                for k, v in m.compute().items()
            }
            for m in self.ibot_metrics_val:
                m.reset()
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        params_groups = self.get_params_groups()
        if self.hparams.optimizer == "adamw":
            # to use with ViTs
            optimizer = torch.optim.AdamW(params_groups, betas=self.hparams.adam_betas)
        elif self.hparams.optimizer == "sgd":  # lr is set by scheduler
            optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
        elif self.hparams.optimizer == "lars":  # to use with convnet and large batches
            optimizer = LARS(params_groups)

        self.get_schedulers()
        dummy_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=dummy_scheduler,
                interval="step",
                frequency=1,
            ),
        )

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        optimizer = scheduler.optimizer  # self.optimizers[optimizer_idx]
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.schedule_lr[self._iter]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.schedule_wd[self._iter]
        self._iter += 1

    def get_models(self):
        kwargs = dict(
            #
            mask_idx=self.hparams.mask_idx,
            padding_idx=self.hparams.padding_idx,
            cls_idx=self.hparams.cls_idx,
            vocab_size=self.hparams.vocab_size,
            max_seq_len=self.hparams.max_count,
            #
            num_encoder_layers=self.hparams.depth,
            emb_dim=self.hparams.emb_dim,
            ffn_emb_dim=self.hparams.emb_dim * self.hparams.mlp_ratio,
            num_heads=self.hparams.num_heads,
            dropout=0.1,
            dropout_attention=0.1,
            dropout_activation=0.0,
            #
            encoder_normalize_before=self.hparams.encoder_normalize_before,
            embedding_normalize=self.hparams.embedding_normalize,
            drop_path=self.hparams.drop_path,
            #
            apply_bert_init=True,
            activation_fn="gelu",
            rel_pos=True,
            rel_pos_bins=self.hparams.rel_pos_bins,
            max_rel_pos=self.hparams.max_count,
        )
        student = TransformerSentenceEncoder(**kwargs)
        kwargs["drop_path"] = 0
        teacher = TransformerSentenceEncoder(**kwargs)
        student = MultiCropWrapper(
            student,
            iBOTHead(
                self.hparams.vocab_size - self.hparams.num_special_tokens,
                self.hparams.max_count,
                self.hparams.emb_dim,
                self.hparams.out_dim,
                patch_out_dim=self.hparams.patch_out_dim,
                norm=self.hparams.norm_in_head,
                act=self.hparams.act_in_head,
                dropout_p=self.hparams.dropout_in_head,
                hid_dim=self.hparams.head_hid_dim,
                bottleneck_dim=self.hparams.head_bottleneck_dim,
                norm_last_layer=self.hparams.norm_last_layer,
                shared_head=self.hparams.shared_head,
            ),
        )
        teacher = MultiCropWrapper(
            teacher,
            iBOTHead(
                self.hparams.vocab_size - self.hparams.num_special_tokens,
                self.hparams.max_count,
                self.hparams.emb_dim,
                self.hparams.out_dim,
                patch_out_dim=self.hparams.patch_out_dim,
                norm=self.hparams.norm_in_head,
                act=self.hparams.act_in_head,
                dropout_p=self.hparams.dropout_in_head,
                hid_dim=self.hparams.head_hid_dim,
                bottleneck_dim=self.hparams.head_bottleneck_dim,
                shared_head=self.hparams.shared_head_t,
            ),
        )
        # teacher copies student as init.
        teacher.load_state_dict(student.state_dict(), strict=False)
        for p in teacher.parameters():
            p.requires_grad = False  # not update by gradient
        self.student = student
        self.teacher = teacher

    def get_loss(self):
        # loss
        distributed = len(self.hparams.gpus) > 1
        same_dim = self.hparams.shared_head or self.hparams.shared_head_t
        self.loss_ibot = iBOTLoss(
            self.hparams.out_dim,
            self.hparams.out_dim if same_dim else self.hparams.patch_out_dim,
            self.hparams.warmup_teacher_temp,
            self.hparams.teacher_temp,
            self.hparams.warmup_teacher_patch_temp,
            self.hparams.teacher_patch_temp,
            self.hparams.warmup_teacher_temp_epochs,
            self.hparams.epochs,
            mim_start_epoch=self.hparams.pred_start_epoch,
            distributed=distributed,
        )
        self.loss_moco = MoCoLoss(
            self.hparams.head_bottleneck_dim
            if self.hparams.head_bottleneck_dim != 0
            else self.hparams.head_hid_dim,
            self.hparams.num_negs,
            self.hparams.moco_temp,
            distributed=distributed,
        )
        self.loss_bert = BERTLoss()

    def get_schedulers(self):
        true_batch_size = self.hparams.batch_size * len(self.hparams.gpus)
        steps_per_epoch = math.ceil(self.hparams.length_train / true_batch_size)
        self.schedule_lr = cosine_scheduler(
            start_warmup_value=0,
            base_value=self.hparams.lr * true_batch_size / 256,
            final_value=self.hparams.min_lr,
            warmup_steps=self.hparams.warmup_epochs * steps_per_epoch,
            steps=self.hparams.epochs * steps_per_epoch,
        )
        self.schedule_wd = cosine_scheduler(
            start_warmup_value=0,
            base_value=self.hparams.weight_decay,
            final_value=self.hparams.weight_decay_end,
            warmup_steps=0,
            steps=self.hparams.epochs * steps_per_epoch,
        )
        self.scheduler_mom = cosine_scheduler(
            start_warmup_value=0,
            base_value=self.hparams.momentum_t,
            final_value=1,
            warmup_steps=0,
            steps=self.hparams.epochs * steps_per_epoch,
        )
        self._iter = 0

    def get_metrics(self):
        self.ibot_metrics_train = iBOTMetrics(prefix="train")
        self.ibot_acc_train = Accuracy()
        n = 3
        self.ibot_metrics_val = nn.ModuleList(
            [iBOTMetrics(prefix="val") for _ in range(n)]
        )
        self.ibot_acc_val = nn.ModuleList([Accuracy() for _ in range(n)])

    def get_params_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [
            {"params": regularized},
            {"params": not_regularized, "weight_decay": 0.0},
        ]

    def get_target_genes(self, genes, count_raw, mask):
        """masked gene -> predict gene idx
        Prob of genes is given by Negative Binomial distribution
        """
        idx = torch.arange(mask.shape[0]).view(-1, 1).expand_as(mask)[mask]
        means = count_raw[idx, genes[mask] - self.hparams.num_special_tokens]
        targets = nb_pmf(means, self.hparams.temp_gene, count_raw[idx], "gene")
        targets /= targets.sum(dim=1, keepdim=True)
        return targets  # [num_masks_in_batch, num_genes]

    def get_target_counts(self, counts: torch.Tensor, mask: torch.Tensor):
        """masked count -> predict count
        Label smoothing with Negative Binomial distribution
        """
        means = counts[mask]  # 1d
        targets = nb_pmf(
            means,
            self.hparams.temp_count,
            torch.arange(self.hparams.max_count, device=means.device),
            "count",
        )
        targets /= targets.sum(dim=1, keepdim=True)
        return targets  # [num_masks_in_batch, max_count]


# @lru_cache(maxsize=None)
def nb_pmf(
    m: torch.Tensor, p: float, v: torch.Tensor, flavor: Literal["gene", "count"]
) -> torch.Tensor:
    """
    When flavor is `gene`, m is 1d tensor ([m]) and v is 2d tensors ([m, n]). Return shape is [m, n].
    When flavor is `count`, m ([m]) and v ([v]) are 1d tensors, and return shape is [m, v].
    m is nonzero. v can be zero.
    """
    k = (m - 0.5) * (1 - p) / p + 1 / p
    # k = k.clip(1e-8)
    logits = torch.tensor(p / (1 - p), device=k.device).log()
    if flavor == "count":
        k = k.unsqueeze(1)
        v = v.unsqueeze(0)
    elif flavor == "gene":
        k = k.unsqueeze(1)
    else:
        raise NotImplementedError
    log_unnormalized_prob = k * F.logsigmoid(-logits) + v * F.logsigmoid(logits)

    log_normalization = -torch.lgamma(k + v) + torch.lgamma(1.0 + v) + torch.lgamma(k)

    pmf = (log_unnormalized_prob - log_normalization).exp()
    return pmf


def train_func(config_data: dict, config_model: dict):
    datamodule = PanaceaDataModule(data_dir=DATA_DIR, **config_data)

    config = {**config_model, **config_data}
    model = Panacea(**config)

    if FAST_DEV_RUN:
        trainer = pl.Trainer(
            # fast_dev_run=True,
            deterministic=True,
            max_epochs=1,
            strategy="ddp",
            gpus=config["gpus"],
            gradient_clip_val=config["grad_clip"],
        )
        trainer.fit(model, datamodule)
        return

    callbacks = [
        # LogPredictionsCallback(),
        ModelCheckpoint(
            monitor=config["monitor"],
            mode=config["mode"],
            auto_insert_metric_name=False,
            every_n_epochs=10,
            # save_top_k=5,
            # save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    wandb_logger = pl.loggers.WandbLogger(
        project=PROJECT,
        group=GROUP,
        # name=name,
        log_model="all",
        # log_model=True,
        save_code=True,  # this won't work.
    )
    trainer_config = dict(
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        # log_every_n_steps=30,
        max_epochs=config["epochs"],
        gradient_clip_val=config["grad_clip"],
        accumulate_grad_batches=config["grad_acc"],
    )

    if STRATEGY == "ddp":
        wandb_logger.watch(model, log="all", log_freq=500)
        trainer_config["strategy"] = "ddp"
        trainer_config["gpus"] = config["gpus"]
    else:
        raise NotImplementedError
        # trainer_config["num_sanity_val_steps"] = 0
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in config["gpus"]])
        # trainer_config["strategy"] = RayPlugin(
        #     num_workers=len(config["gpus"]), num_cpus_per_worker=4, use_gpu=True
        # )

    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    import argparse
    import yaml
    from rich import print as rprint
    from rich.traceback import install

    install()
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    # parser.add_argument("--num_batches", type=int, default=1, choices=[0, 1, 2, 4])
    parser.add_argument("--test", type=bool, default=False)
    args = parser.parse_args()

    with open("config_data.yaml") as f:
        config_data = yaml.safe_load(f)
    with open("config_model.yaml") as f:
        config_model = yaml.safe_load(f)

    config_model["gpus"] = args.gpus

    DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/panacea"
    PROJECT = "panacea"
    GROUP = "test"
    STRATEGY = "ddp"
    FAST_DEV_RUN = args.test
    train_func(config_data, config_model)
