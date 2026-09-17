"""OOD evaluation of a trained student (or the teacher) on flood datasets outside Sen1Floods11.

Dataset names: floodplanet (FloodPlanet multi-sensor imagery), worldfloods (WorldFloods S2),
sturm_s2 and sturm_s1 (Sturm et al. Sentinel-2 / Sentinel-1 SAR scenes), deepflood,
urban_sar (urban SAR flood detection), and sen1floods11 (test split, only as an
in-distribution sanity check).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.ood_datasets import (
    get_worldfloods_dataloader,
    get_floodplanet_dataloader,
    get_sturm_flood_dataloader,
    get_deepflood_dataloader,
    get_urban_sar_floods_dataloader,
)
from data.channel_adapter import (
    sturm_s2_adapter,
    sar_2band_adapter,
    urban_sar_adapter,
)


def _get_sen1floods11_test_loader(root, bs, nw, sz):
    from data.sen1floods11 import get_dataloaders
    _, _, test_dl = get_dataloaders(data_root=root, batch_size=bs,
                                    num_workers=nw, eval_img_size=sz)
    return test_dl


_DATASET_FACTORIES = {
    "floodplanet":  lambda root, bs, nw, sz: get_floodplanet_dataloader(
        root=root, batch_size=bs, num_workers=nw, img_size=sz),
    "worldfloods":  lambda root, bs, nw, sz: get_worldfloods_dataloader(
        root=root, batch_size=bs, num_workers=nw, img_size=sz),
    "sturm_s2":     lambda root, bs, nw, sz: get_sturm_flood_dataloader(
        root=root, sensor="S2", batch_size=bs, num_workers=nw, img_size=sz),
    "sturm_s1":     lambda root, bs, nw, sz: get_sturm_flood_dataloader(
        root=root, sensor="S1", batch_size=bs, num_workers=nw, img_size=sz),
    "deepflood":    lambda root, bs, nw, sz: get_deepflood_dataloader(
        root=root, batch_size=bs, num_workers=nw, img_size=sz),
    "urban_sar":    lambda root, bs, nw, sz: get_urban_sar_floods_dataloader(
        root=root, batch_size=bs, num_workers=nw, img_size=sz),
    "sen1floods11": _get_sen1floods11_test_loader,
}

_CHANNEL_ADAPTERS = {
    "sturm_s2":  lambda ic: sturm_s2_adapter(target_channels=ic),
    "sturm_s1":  lambda ic: sar_2band_adapter(target_channels=ic),
    "deepflood": lambda ic: sar_2band_adapter(target_channels=ic),
    "urban_sar": lambda ic: urban_sar_adapter(target_channels=ic),
}


class OODEval:
    """Run OOD evaluation, save per-image predictions and aggregate metrics as CSV in out_dir.

        model_name is just the log label ("student" / "teacher"). img_size has to match the
        training size. Only gpu_ids[0] is used.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        dataset_name: str,
        ood_root: str,
        in_channels: int,
        num_classes: int,
        batch_size: int,
        num_workers: int,
        img_size: int,
        out_dir: str,
        gpu_ids: list[int],
    ) -> None:
        self.model        = model
        self.model_name   = model_name
        self.dataset_name = dataset_name.lower()
        self.ood_root     = ood_root
        self.in_channels  = in_channels
        self.num_classes  = num_classes
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.img_size     = img_size
        self.out_dir      = out_dir
        self.device       = torch.device(
            f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
        )

    def run(self) -> dict:
        os.makedirs(self.out_dir, exist_ok=True)

        if self.dataset_name not in _DATASET_FACTORIES:
            raise ValueError(
                f"Unknown OOD dataset '{self.dataset_name}'. "
                f"Choices: {list(_DATASET_FACTORIES)}"
            )

        dataloader = _DATASET_FACTORIES[self.dataset_name](
            self.ood_root, self.batch_size, self.num_workers, self.img_size
        )

        adapter_fn = _CHANNEL_ADAPTERS.get(self.dataset_name)
        self.adapter = adapter_fn(self.in_channels).to(self.device) if adapter_fn else None

        self.model = self.model.to(self.device)
        self.model.eval()

        metrics = self._evaluate(dataloader)
        self._save_results(metrics)

        print(f"\n[ood_eval] {self.model_name} on {self.dataset_name}")
        print(f"  mIoU  = {metrics['mIoU']:.4f}")
        for k, v in metrics.items():
            if k.startswith("IoU_class"):
                print(f"  {k} = {v:.4f}")
        return metrics

    def _evaluate(self, dataloader: DataLoader) -> dict:
        intersection  = torch.zeros(self.num_classes, device=self.device)
        union         = torch.zeros(self.num_classes, device=self.device)
        true_counts   = torch.zeros(self.num_classes, device=self.device)
        pred_counts   = torch.zeros(self.num_classes, device=self.device)
        n_batches     = 0
        _img_stats_printed = False

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device).float()
                if self.adapter is not None:
                    images = self.adapter(images)
                masks = batch.get("mask") if batch.get("mask") is not None else batch.get("label")
                if masks is None:
                    continue
                masks = masks.to(self.device).long()

                if not _img_stats_printed:
                    _img_stats_printed = True
                    print(f"\n[ood_eval] Input image stats (post-norm, first batch, "
                          f"shape={tuple(images.shape)}):")
                    for b in range(images.shape[1]):
                        ch = images[:, b]
                        print(f"  band{b}: min={ch.min():.3f}  max={ch.max():.3f}  "
                              f"mean={ch.mean():.3f}  std={ch.std():.3f}")

                logits = self.model(images)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:],
                                           mode="bilinear", align_corners=False)
                preds = logits.argmax(dim=1)

                valid = masks >= 0
                for c in range(self.num_classes):
                    pred_c  = (preds == c) & valid
                    label_c = (masks == c) & valid
                    intersection[c] += (pred_c & label_c).sum()
                    union[c]        += (pred_c | label_c).sum()
                    true_counts[c]  += label_c.sum()
                    pred_counts[c]  += pred_c.sum()
                n_batches += 1

        if n_batches == 0:
            raise RuntimeError(
                "No batches with masks found. Check that the dataset returns "
                "'mask' or 'label' keys and --ood_root is correct."
            )

        iou_per_class = (intersection / (union + 1e-6)).cpu()
        miou          = iou_per_class.mean().item()

        denom_f1      = (true_counts + pred_counts).cpu()
        f1_per_class  = torch.where(
            denom_f1 > 0,
            2.0 * intersection.cpu() / (denom_f1 + 1e-7),
            torch.zeros_like(denom_f1),
        )
        f1 = f1_per_class.mean().item()

        total_true = true_counts.sum().item()
        total_pred = pred_counts.sum().item()
        print(f"\n[ood_eval] Class distribution ({self.dataset_name}):")
        for c in range(self.num_classes):
            tc = true_counts[c].item()
            pc = pred_counts[c].item()
            print(f"  class{c}: true={tc:.0f} px ({100*tc/max(total_true,1):.1f}%)  "
                  f"pred={pc:.0f} px ({100*pc/max(total_pred,1):.1f}%)")

        metrics: dict = {"mIoU": round(miou, 6), "f1": round(f1, 6)}
        for c, (iou, f1c) in enumerate(zip(iou_per_class.tolist(), f1_per_class.tolist())):
            metrics[f"IoU_class{c}"] = round(iou, 6)
            metrics[f"f1_class{c}"]  = round(f1c, 6)
        return metrics

    def _save_results(self, metrics: dict) -> None:
        import csv
        csv_path = os.path.join(self.out_dir, f"{self.model_name}_{self.dataset_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
        print(f"[ood_eval] results saved to {csv_path}")
