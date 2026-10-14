"""Training curves, segmentation previews and MAE reconstruction plots.

Paper qualitative panels use figures/ and its shared display conversion.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


# colour maps / legend patches used in prediction plots
_FLOOD_CMAP  = "Blues"
_LEGEND_FLOOD = [
    ("navy",  "Flood"),
    ("white", "No flood"),
    ("grey",  "No data (-1)"),
]


def _rgb_indices(bands: list[str]) -> list[int]:
    """Return [r_idx, g_idx, b_idx] into the band tensor."""
    upper = [b.upper() for b in bands]
    r = next((i for i, b in enumerate(upper) if "RED"   in b and "EDGE" not in b and "NIR" not in b), 2)
    g = next((i for i, b in enumerate(upper) if "GREEN" in b), 1)
    b = next((i for i, b in enumerate(upper) if "BLUE"  in b), 0)
    return [r, g, b]


def _to_rgb(img_chw: np.ndarray, rgb_idx: list[int]) -> np.ndarray:
    """Convert a (C, H, W) float array to (H, W, 3) uint8 for imshow."""
    rgb = img_chw[rgb_idx]                          # (3, H, W)
    rgb = rgb.transpose(1, 2, 0).astype(np.float32) # (H, W, 3)
    lo, hi = rgb.min(), rgb.max()
    if hi > lo:
        rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0, 1)


# 1. Training curves

def plot_training_curves(
    history: dict[str, list[float]],
    out_dir: str,
    filename: str = "training_curves.png",
) -> str:
    """Save train/validation loss and available validation mIoU curves; return the path."""
    import matplotlib.pyplot as plt

    has_miou = "val_mIoU" in history and len(history["val_mIoU"]) > 0
    n_plots  = 2 if has_miou else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Loss subplot
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train loss", color="steelblue")
    if "val_loss" in history:
        # val may be logged less frequently - use the stored epoch indices
        val_epochs = history.get("val_epochs", list(range(len(history["val_loss"]))))
        ax.plot(val_epochs, history["val_loss"], label="val loss",
                color="darkorange", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mIoU subplot
    if has_miou:
        ax2 = axes[1]
        val_epochs = history.get("val_epochs", list(range(len(history["val_mIoU"]))))
        ax2.plot(val_epochs, history["val_mIoU"], color="seagreen", marker="o", markersize=3)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("mIoU")
        ax2.set_title("Validation mIoU")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] training curves → {path}")
    return path


# 2. Prediction grid

@torch.no_grad()
def plot_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    bands: list[str],
    n_samples: int = 5,
    filename: str = "predictions.png",
) -> str:
    """Save an RGB, ground-truth and prediction grid from the first loader batch."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model.eval()
    rgb_idx = _rgb_indices(bands)

    batch   = next(iter(loader))
    images  = batch["image"].to(device, non_blocking=True).float()
    masks   = batch["mask"].cpu().numpy()

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        logits = model(images)
    preds = logits.argmax(dim=1).cpu().numpy()    # (B, H, W)
    images_np = images.cpu().numpy()              # (B, C, H, W)

    n = min(n_samples, images_np.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        rgb  = _to_rgb(images_np[i], rgb_idx)
        gt   = masks[i].astype(float)
        pred = preds[i].astype(float)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("RGB input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt,   cmap=_FLOOD_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap=_FLOOD_CMAP, vmin=0,  vmax=1, interpolation="nearest")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    patches = [mpatches.Patch(color=c, label=l) for c, l in _LEGEND_FLOOD]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=9)
    fig.suptitle("Segmentation results", fontsize=12, y=1.01)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] predictions → {path}")
    return path


# 3. Qualitative split assessment

@torch.no_grad()
def plot_split_samples(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    bands: list[str],
    split: str = "test",
    n_samples: int = 8,
    max_batches: int = 20,
) -> str:
    """Save RGB, ground truth, prediction and error-map rows.

    Errors: white TN, blue TP, red FN, orange FP, grey ignore (mask=-1).
    Samples are drawn from up to max_batches loader batches. Return the PNG path."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model.eval()
    rgb_idx = _rgb_indices(bands)

    # collect a pool of samples
    img_pool, mask_pool = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        img_pool.append(batch["image"])
        mask_pool.append(batch["mask"])

    all_imgs  = torch.cat(img_pool,  dim=0)   # (N, C, H, W)
    all_masks = torch.cat(mask_pool, dim=0)   # (N, H, W)

    total = all_imgs.shape[0]
    idx   = torch.randperm(total)[:min(n_samples, total)]
    imgs_sel  = all_imgs[idx]
    masks_sel = all_masks[idx]

    # run inference
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        logits = model(imgs_sel.to(device).float())
    preds = logits.argmax(dim=1).cpu().numpy()    # (n, H, W)
    imgs_np  = imgs_sel.numpy()                   # (n, C, H, W)
    masks_np = masks_sel.numpy()                  # (n, H, W)

    # plot
    n   = len(idx)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        gt   = masks_np[i]   # (H, W), values: -1, 0, 1
        pred = preds[i]      # (H, W), values: 0, 1
        rgb  = _to_rgb(imgs_np[i], rgb_idx)

        # error map: 4-class RGBA array
        H, W = gt.shape
        err = np.zeros((H, W, 4), dtype=np.float32)
        nodata = gt == -1
        tn = (~nodata) & (gt == 0) & (pred == 0)
        tp = (~nodata) & (gt == 1) & (pred == 1)
        fn = (~nodata) & (gt == 1) & (pred == 0)   # missed flood - dangerous
        fp = (~nodata) & (gt == 0) & (pred == 1)   # false alarm

        err[tn]    = [1.0, 1.0, 1.0, 1.0]          # white
        err[tp]    = [0.2, 0.4, 0.9, 1.0]          # blue
        err[fn]    = [0.9, 0.1, 0.1, 1.0]          # red
        err[fp]    = [1.0, 0.5, 0.0, 1.0]          # orange
        err[nodata]= [0.6, 0.6, 0.6, 1.0]          # grey

        # pixel-level accuracy for title
        valid = ~nodata
        if valid.any():
            acc = ((pred[valid] == gt[valid]).sum() / valid.sum()) * 100
            title_err = f"Err {100-acc:.1f}%"
        else:
            title_err = "no valid px"

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("RGB", fontsize=9)

        axes[i, 1].imshow(gt,   cmap=_FLOOD_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        axes[i, 1].set_title("Ground truth", fontsize=9)

        axes[i, 2].imshow(pred, cmap=_FLOOD_CMAP, vmin=0,  vmax=1, interpolation="nearest")
        axes[i, 2].set_title("Prediction", fontsize=9)

        axes[i, 3].imshow(err, interpolation="nearest")
        axes[i, 3].set_title(f"Error map  ({title_err})", fontsize=9)

        for ax in axes[i]:
            ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=(0.2, 0.4, 0.9), label="TP — correct flood"),
        mpatches.Patch(color="white",           label="TN — correct no-flood", linewidth=0.5,
                       edgecolor="grey"),
        mpatches.Patch(color=(0.9, 0.1, 0.1),  label="FN — missed flood"),
        mpatches.Patch(color=(1.0, 0.5, 0.0),  label="FP — false alarm"),
        mpatches.Patch(color=(0.6, 0.6, 0.6),  label="No data"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Qualitative assessment — {split} split  ({n} random samples)",
                 fontsize=12, y=1.005)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split}_samples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {split} samples → {path}")
    return path


@torch.no_grad()
def plot_all_splits(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    out_dir: str,
    bands: list[str],
    n_samples: int = 8,
) -> list[str]:
    """Run plot_split_samples for train, val, and test and return saved paths."""
    model.eval()
    paths = []
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        p = plot_split_samples(
            model, loader, device, out_dir, bands,
            split=split, n_samples=n_samples,
        )
        paths.append(p)
    return paths


# 4. MAE reconstruction grid

@torch.no_grad()
def plot_mae_reconstruction(
    mae: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    bands: list[str],
    mask_ratio: float = 0.75,
    n_samples: int = 4,
    filename: str = "mae_reconstruction.png",
) -> str:
    """Save original, masked input and reconstruction rows for an MAE batch."""
    import matplotlib.pyplot as plt

    mae_module = mae.module if isinstance(mae, nn.DataParallel) else mae
    mae_module.eval()
    rgb_idx = _rgb_indices(bands)

    batch  = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True).float()

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        loss_dict, pred, mask = mae_module(images, mask_ratio=mask_ratio)

    images_np = images.cpu().float().numpy()      # (B, C, H, W)
    pred_np   = pred.cpu().float().numpy()        # (B, C, H, W)
    mask_np   = mask.cpu().float().numpy()        # (B, H, W) - 1 = masked

    n = min(n_samples, images_np.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        orig    = _to_rgb(images_np[i], rgb_idx)
        # Masked image: zero out masked patch regions
        masked_img = images_np[i].copy()
        masked_img *= (1.0 - mask_np[i])          # zero where mask=1
        masked_rgb  = _to_rgb(masked_img, rgb_idx)
        recon_rgb   = _to_rgb(pred_np[i],  rgb_idx)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masked_rgb)
        axes[i, 1].set_title(f"Masked ({int(mask_ratio*100)}%)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(recon_rgb)
        axes[i, 2].set_title("Reconstruction")
        axes[i, 2].axis("off")

    loss_val = loss_dict["loss"].item() if hasattr(loss_dict["loss"], "item") else loss_dict["loss"]
    fig.suptitle(f"MAE reconstruction  (loss={loss_val:.4f})", fontsize=12)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] MAE reconstruction → {path}")
    return path
