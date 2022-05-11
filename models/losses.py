from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch import distributed as dist
from torchmetrics.functional import precision, accuracy

from .schedulers import linear_warmup


class iBOTLoss(nn.Module):
    """
    Args:
        out_dim: 8192.
            Dimensionality of output for [CLS] token.
        patch_out_dim: 8192.
            Dimensionality of output for patch tokens.
        ngcrops (global_crops_number): 2.
            Number of global views to generate. Default is to use two global crops.
        nlcrops (local_crops_number): 0.
            Number of smalllocal views to generate. Set this parameter to 0 to disable multi-crop training. When disabling multi-crop we recommend to use "--crop_scale_global 0.14 1."
        warmup_teacher_temp: 0.04.
            Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.
        teacher_temp: 0.04.
            Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.
        warmup_teacher_temp2 (warmup_teacher_patch_temp): 0.04.
            See `--warmup_teacher_temp`.
        teacher_temp2 (teacher_patch_temp): 0.07.
            See `--teacher_temp`.
        warmup_teacher_temp_epochs: 30.
            Number of warmup epochs for the teacher temperature (Default: 30).
        nepochs (epochs): 100.
            Number of epochs of training.

        lambda1: 1.0
            loss weight for dino loss over [CLS] tokens (Default: 1.0).
        lambda2: 1.0
            loss weight for beit loss over masked patch tokens (Default: 1.0).
        mim_start_epoch (pred_start_epoch): 0.
            Start epoch to perform masked image prediction. We typically set this to 50 for swin transformer. (Default: 0)

    """

    def __init__(
        self,
        out_dim,
        patch_out_dim,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        mim_start_epoch=0,
        distributed=False,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.mim_start_epoch = mim_start_epoch
        self.distributed = distributed

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = linear_warmup(
            warmup_teacher_temp, teacher_temp, 0, warmup_teacher_temp_epochs, nepochs
        )
        self.teacher_temp2_schedule = linear_warmup(
            warmup_teacher_temp2,
            teacher_temp2,
            mim_start_epoch,
            warmup_teacher_temp_epochs,
            nepochs,
        )

    def forward(
        self,
        cls_t_l: list,
        patch_t_l: list,
        cls_s_l: list,
        patch_s_l: list,
        mask_l: list,
        epoch: float,
        calc_contrastive_loss=True,
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # cls_*: [[batch_size, out_dim], ...]
        # patch_*: [[batch_size, H*W, out_dim], ...]
        # mask_l: [[batch_size, H, W], ...]

        # [CLS] and patch for global patches
        cls_s_l = [i / self.student_temp for i in cls_s_l]
        patch_s_l = [i / self.student_temp for i in patch_s_l]

        # teacher centering and sharpening
        # why centering???
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        cls_t_l = [((i - self.center) / temp).softmax(dim=-1) for i in cls_t_l]
        patch_t_l = [((i - self.center2) / temp2).softmax(dim=-1) for i in patch_t_l]

        if calc_contrastive_loss:
            # different views between students
            total_loss0, n_loss_terms0 = 0, 0
            for v1 in range(len(cls_s_l)):
                for v2 in range(len(cls_s_l)):
                    if v1 == v2:
                        continue
                    # [batch_size]
                    loss0 = torch.sum(
                        -cls_s_l[v1].softmax(dim=-1) * cls_s_l[v2].log_softmax(dim=-1),
                        dim=-1,
                    )
                    total_loss0 += loss0.mean()
                    n_loss_terms0 += 1
            total_loss0 = total_loss0 / n_loss_terms0 if n_loss_terms0 > 0 else 0.0

            # different views between student and teacher, contrastive loss
            total_loss1, n_loss_terms1 = 0, 0
            for q in range(len(cls_t_l)):
                for v in range(len(cls_s_l)):
                    # [batch_size]
                    loss1 = torch.sum(
                        -cls_t_l[q] * cls_s_l[v].log_softmax(dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            total_loss1 = total_loss1 / n_loss_terms1
        else:
            total_loss0 = total_loss1 = None

        # the same view, mask modeling loss
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(cls_t_l)):
            # [batch_size, H*W]
            loss2 = torch.sum(
                -patch_t_l[q] * patch_s_l[q].log_softmax(dim=-1),
                dim=-1,
            )
            # 1 for mask, 0 for non-mask
            mask = mask_l[q].flatten(start_dim=1).float()
            # [batch_size]
            loss2 = torch.sum(loss2 * mask, dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            total_loss2 += loss2.mean()
            n_loss_terms2 += 1
        total_loss2 = total_loss2 / n_loss_terms2

        # center is not learned.
        self.update_center(torch.cat(cls_t_l, dim=0), torch.cat(patch_t_l, dim=0))
        return total_loss0, total_loss1, total_loss2

    @torch.no_grad()
    def _update_center_distributed(self, cls_t, patch_t):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(cls_t, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(cls_t) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(patch_t.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(patch_t) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
        )

    @torch.no_grad()
    def _update_center_local(self, cls_t, patch_t):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(cls_t, dim=0, keepdim=True)
        cls_center = cls_center / len(cls_t)
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(patch_t.mean(1), dim=0, keepdim=True)
        patch_center = patch_center / len(patch_t)
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
        )

    @torch.no_grad()
    def update_center(self, cls_t, patch_t):
        """
        Update center used for teacher output.
        """
        if self.distributed:
            self._update_center_distributed(cls_t, patch_t)
        else:
            self._update_center_local(cls_t, patch_t)


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.recon_loss = nn.MSELoss()

    def forward(
        self,
        images: list,
        recon_l: list,
        mask_l: list,
    ):
        # Images should have been processed such that its elements are between 0 and 1.
        # Recon should have been sigmoided such that its elements are also between 0 and 1.

        # images: [[batch_size, c, h, w], ...]
        # recon_l: [[batch_size, H*W, patch_size * patch_size * c], ...]
        patch_size = images[0].shape[2] // int(recon_l[0].shape[1] ** 0.5)
        total_loss, n_loss_terms = 0, 0
        for v in range(len(images)):
            mask = mask_l[v].flatten(start_dim=1)
            # [batch_size, num_masks, patch_size * patch_size * c]
            target = patchify(images[v], patch_size)[mask]
            total_loss += self.recon_loss(recon_l[v][mask] * 2 - 1, target * 2 - 1)
            n_loss_terms += 1
        total_loss = total_loss / n_loss_terms
        return total_loss


class BERTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()

    def forward(self, targets_l, preds_l, masks_l, bce=False):
        # targets: [num_masks_in_batch, out_dim]
        # preds: [batch_size, max_length, out_dim]
        # masks: [batch_size, max_length]

        loss = self.loss_bce if bce else self.loss_ce
        # -> [num_crops * num_masks_in_batch, out_dim]
        # preds = torch.stack(preds_l)[torch.stack(masks_l)]
        # both preds and targets are [num_crops * num_masks_in_batch, out_dim].
        # CE loss is computed such that the second dimension is the one-hot vector.

        # other losses deal with each crop separately, and get the mean, here I simply treat all crops as a whole.
        # return loss(preds, torch.cat(targets_l))

        total_loss, n_loss_terms = 0, 0
        for targets, preds, masks in zip(targets_l, preds_l, masks_l):
            total_loss += loss(preds[masks], targets)
            n_loss_terms += 1

        loss = total_loss / n_loss_terms
        return loss


class MoCoLoss(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_negatives: int,
        temp: float,
        p_same_batch: float,
        ignore_label: int,
        distributed: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer(
            "queue_label", torch.full((num_negatives,), -999, dtype=torch.long)
        )
        self.register_buffer(
            "queue_batch", torch.full((num_negatives,), -999, dtype=torch.long)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_val", torch.randn(emb_dim, num_negatives))
        self.queue_val = nn.functional.normalize(self.queue_val, dim=0)
        self.register_buffer(
            "queue_val_label", torch.full((num_negatives,), -999, dtype=torch.long)
        )
        self.register_buffer(
            "queue_val_batch", torch.full((num_negatives,), -999, dtype=torch.long)
        )
        self.register_buffer("queue_ptr_val", torch.zeros(1, dtype=torch.long))

        # self.loss = nn.BCELoss()
        self.loss = nn.CrossEntropyLoss()

        self.emb_dim = emb_dim
        self.num_negatives = num_negatives
        self.temp = temp
        self.p_same_batch = p_same_batch
        self.ignore_label = ignore_label
        self.distributed = distributed
        self._to_enter_queue = None
        self._to_enter_queue_label = None

    @torch.no_grad()
    def _dequeue_and_enqueue(self) -> None:
        if self.training:
            queue = self.queue
            queue_label = self.queue_label
            queue_batch = self.queue_batch
            queue_ptr = self.queue_ptr
        else:
            queue = self.queue_val
            queue_label = self.queue_val_label
            queue_batch = self.queue_val_batch
            queue_ptr = self.queue_ptr_val

        if self._to_enter_queue is None:
            return
        if self.distributed is True:
            self._to_enter_queue = concat_all_gather(self._to_enter_queue)
            self._to_enter_queue_label = concat_all_gather(self._to_enter_queue_label)
            self._to_enter_queue_batch = concat_all_gather(self._to_enter_queue_batch)
        batch_size = self._to_enter_queue.shape[0]
        ptr = int(queue_ptr)
        # if self.num_negatives % batch_size == 0:
        #     return
        ptr_end = ptr + batch_size
        if ptr_end <= self.num_negatives:
            queue[..., ptr : ptr + batch_size] = self._to_enter_queue.t()
            queue_label[ptr : ptr + batch_size] = self._to_enter_queue_label
            queue_batch[ptr : ptr + batch_size] = self._to_enter_queue_batch
            ptr = (ptr + batch_size) % self.num_negatives
        else:
            delta = ptr_end - self.num_negatives
            queue[..., ptr:] = self._to_enter_queue.t()[..., :-delta]
            queue[..., :delta] = self._to_enter_queue.t()[..., -delta:]
            queue_label[ptr:] = self._to_enter_queue_label[:-delta]
            queue_label[:delta] = self._to_enter_queue_label[-delta:]
            queue_batch[ptr:] = self._to_enter_queue_batch[:-delta]
            queue_batch[:delta] = self._to_enter_queue_batch[-delta:]

            ptr = delta

        queue_ptr[0] = ptr

    def forward(
        self,
        cls_s: torch.Tensor,
        cls_t: torch.Tensor,
        label: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # cls_s: [batch_size, num_crops_s, out_dim]
        # cls_t: [batch_size, num_crops_t, out_dim]
        # labels: [batch_size]
        batch_size, num_crops_s, out_dim = cls_s.shape
        _, num_crops_t, _ = cls_t.shape
        # cls_s = nn.functional.normalize(cls_s, dim=2)
        # with torch.no_grad():
        #     cls_t = nn.functional.normalize(cls_t, dim=2)
        crop_idx = np.random.choice(cls_t.shape[1], size=cls_t.shape[0])
        cls_t = cls_t[np.arange(batch_size), crop_idx]
        l_pos = torch.einsum("bsd, bd -> bs", cls_s, cls_t).unsqueeze(-1)
        queue = self.queue if self.training else self.queue_val
        queue_label = self.queue_label if self.training else self.queue_val_label
        queue_batch = self.queue_batch if self.training else self.queue_val_batch
        self._dequeue_and_enqueue()
        self._to_enter_queue = cls_t
        self._to_enter_queue_label = label
        self._to_enter_queue_batch = batch
        l_neg = torch.einsum("bsd, dq -> bsq", cls_s, queue)
        # [batch_size, num_crops_s, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=2) / self.temp
        # logits = logits.softmax(dim=2)
        logits = logits.view(batch_size * num_crops_s, -1)
        # labels_pos = torch.ones(1, device=logits.device).expand(
        #     batch_size, num_crops_s, num_crops_t
        # )
        # labels_neg = torch.zeros(1, device=logits.device).expand(
        #     batch_size, num_crops_s, self.num_negatives
        # )
        # labels = torch.cat([labels_pos, labels_neg], dim=2)
        labels = torch.zeros(
            batch_size * num_crops_s, device=logits.device, dtype=torch.long
        )
        loss = self.loss(logits, labels)
        # p = precision(logits, labels.int(), threshold=0.5 / num_crops_t)
        # p = precision(logits, labels.int())
        a = accuracy(logits, labels)

        # knn loss
        ignored = label.eq(self.ignore_label)  # no sup loss for these samples.
        if ignored.all():
            loss_sup = 0.0
        else:
            label = label[~ignored].view(-1, 1, 1)
            batch = batch[~ignored].view(-1, 1, 1)
            l_neg = l_neg[~ignored]
            queue_label = queue_label.view(1, 1, -1)
            queue_batch = queue_batch.view(1, 1, -1)
            p = l_neg.exp()
            same_label = label == queue_label
            same_batch = batch == queue_batch
            # slsb for "same label same batch", sldb for "same label different batch"
            slsb = same_label & same_batch
            sldb = same_label & ~same_batch
            sim_slsb = (
                p.masked_fill(~slsb, 0) / slsb.sum(dim=2, keepdim=True).clip(1.0)
            ).sum(dim=2)
            sim_sldb = (
                p.masked_fill(~sldb, 0) / sldb.sum(dim=2, keepdim=True).clip(1.0)
            ).sum(dim=2)
            mask_slsb = sim_slsb > 0
            mask_sldb = sim_sldb > 0

            if mask_sldb.any():
                # there have to be different batches in the first place.
                loss_sldb = -sim_sldb[mask_sldb].log().mean()
                if mask_slsb.any():
                    loss_slsb = -sim_slsb[mask_slsb].log().mean()
                else:
                    loss_slsb = 0.0
                loss_sup = (loss_sldb - loss_slsb) + self.p_same_batch * loss_slsb
            else:
                loss_sup = 0.0

        return loss, a, loss_sup


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def patchify(imgs, patch_size):
    """
    imgs: (B, c, h, w)
    x: (B, H*W, patch_size * patch_size * c)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    H = W = imgs.shape[2] // patch_size
    # [B, H, W, p, p, c]
    x = imgs.reshape(*imgs.shape[:2], H, patch_size, W, patch_size).permute(
        0, 2, 4, 3, 5, 1
    )
    # x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], H * W, patch_size * patch_size * imgs.shape[1])
    return x


def unpatchify(x, patch_size):
    """
    imgs: (B, c, h, w)
    x: (B, H*W, patch_size * patch_size * c)
    """
    H = W = int(x.shape[1] ** 0.5)
    assert H * W == x.shape[1]
    c = x.shape[-1] // (patch_size * patch_size)
    assert c * patch_size * patch_size == x.shape[-1]

    x = x.reshape(x.shape[0], H, W, patch_size, patch_size, c).permute(0, 5, 1, 3, 2, 4)
    # x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], c, H * patch_size, W * patch_size)
    return imgs


if __name__ == "__main__":
    import torch
    from pytorch_lightning import seed_everything

    seed_everything(42)

    imgs = torch.randn(2, 3, 24, 24)
    patch_size = 4
    imgs[:, :, 0:4, 4:8] = 0  # the second patch
    x = patchify(imgs, patch_size)
    assert (x[:, 1] == 0).all()
    imgs = unpatchify(x, patch_size)
    assert (imgs[:, :, 0:4, 4:8] == 0).all()
    print("Patchify test passed~")
