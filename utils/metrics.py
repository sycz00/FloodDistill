"""Segmentation metrics in plain PyTorch, no torchmetrics."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_seg_metrics(
    pred_logits: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> dict[str, float]:
    """Metrics from logits=(B,C,H,W) and labels=(B,H,W).

    num_classes includes background. Exclude ignore_index; absent-class IoU is NaN.
    """
    pred = pred_logits.argmax(dim=1)  # (B, H, W)

    pred_flat   = pred.reshape(-1)
    target_flat = target.reshape(-1)

    valid = target_flat != ignore_index
    pred_flat   = pred_flat[valid]
    target_flat = target_flat[valid]

    n_valid = valid.sum().item()
    if n_valid == 0:
        empty = [0.0] * num_classes
        return {"mIoU": 0.0, "iou_per_class": empty, "pixel_acc": 0.0, "f1": 0.0, "f1_per_class": empty}

    # Confusion matrix via bincount: conf[true, pred]
    inds = num_classes * target_flat.long() + pred_flat.long()
    conf = torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    tp = conf.diag().float()                  # (C,)
    fp = conf.sum(0).float() - tp             # predicted as c, actually other
    fn = conf.sum(1).float() - tp             # actually c, predicted as other

    # IoU per class. Classes absent from both pred and target give 0/0 = NaN and are excluded
    union = tp + fp + fn
    iou_per_class = torch.where(union > 0, tp / (union + 1e-7), torch.tensor(float("nan")))
    valid_iou = iou_per_class[~iou_per_class.isnan()]
    miou = valid_iou.mean().item() if len(valid_iou) > 0 else 0.0

    pixel_acc = (pred_flat == target_flat).float().mean().item()

    # F1 per class
    denom = 2 * tp + fp + fn
    f1_per_class = torch.where(denom > 0, 2 * tp / (denom + 1e-7), torch.tensor(float("nan")))
    valid_f1 = f1_per_class[~f1_per_class.isnan()]
    f1 = valid_f1.mean().item() if len(valid_f1) > 0 else 0.0

    return {
        "mIoU":         miou,
        "iou_per_class": iou_per_class.tolist(),
        "pixel_acc":    pixel_acc,
        "f1":           f1,
        "f1_per_class": f1_per_class.tolist(),
    }


def accumulate_metrics(
    pred_logits: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = -1,
    conf_matrix: Tensor | None = None,
) -> Tensor:
    """Accumulate a confusion matrix across batches.

        Returns the updated (num_classes, num_classes) matrix. Pass it back as conf_matrix on
        the next call and call finalize_metrics() at the end.
    """
    if conf_matrix is None:
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pred = pred_logits.argmax(dim=1).reshape(-1)
    tgt  = target.reshape(-1)

    valid = tgt != ignore_index
    pred  = pred[valid]
    tgt   = tgt[valid]

    inds = num_classes * tgt.long() + pred.long()
    conf_matrix += torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu()

    return conf_matrix


def finalize_metrics(conf_matrix: Tensor) -> dict[str, float]:
    """Convert a fully accumulated confusion matrix to metric dict."""
    num_classes = conf_matrix.shape[0]
    conf = conf_matrix.float()

    tp  = conf.diag()
    fp  = conf.sum(0) - tp
    fn  = conf.sum(1) - tp

    union = tp + fp + fn
    iou_per_class = torch.where(union > 0, tp / (union + 1e-7), torch.tensor(float("nan")))
    valid_iou = iou_per_class[~iou_per_class.isnan()]
    miou = valid_iou.mean().item() if len(valid_iou) > 0 else 0.0

    pixel_acc = tp.sum().item() / (conf.sum().item() + 1e-7)

    denom = 2 * tp + fp + fn
    f1_per_class = torch.where(denom > 0, 2 * tp / (denom + 1e-7), torch.tensor(float("nan")))
    valid_f1 = f1_per_class[~f1_per_class.isnan()]
    f1 = valid_f1.mean().item() if len(valid_f1) > 0 else 0.0

    return {
        "mIoU":          miou,
        "iou_per_class": iou_per_class.tolist(),
        "pixel_acc":     pixel_acc,
        "f1":            f1,
        "f1_per_class":  f1_per_class.tolist(),
    }
