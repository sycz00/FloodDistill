"""Sen1Floods11 without terratorch.

Only for the Jetson Xavier NX, where the JetPack/Torch versions won't install
terratorch. Drop-in for data/sen1floods11.py, needs just rasterio,
albumentations, numpy and torch (no Lightning either).

Band order in S2Hand.tif (13 bands, 0-based):
    0  COASTAL_AEROSOL   1  BLUE       2  GREEN      3  RED
    4  RED_EDGE_1        5  RED_EDGE_2 6  RED_EDGE_3 7  NIR_BROAD
    8  NIR_NARROW        9  WATER_VAPOR 10 CIRRUS    11 SWIR_1  12 SWIR_2

Selected 6 bands: BLUE GREEN RED NIR_NARROW SWIR_1 SWIR_2
                  idx: 1    2    3    8         11     12
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# Per-band normalisation (must match data/sen1floods11.py and training)
NORM_MEAN = [0.1413, 0.1380, 0.1235, 0.3090, 0.2045, 0.1191]
NORM_STD  = [0.0741, 0.0737, 0.0869, 0.1180, 0.0977, 0.0766]

# 0-based indices into the 13-band S2Hand.tif
_BAND_IDX = [1, 2, 3, 8, 11, 12]   # BLUE GREEN RED NIR_NARROW SWIR_1 SWIR_2

CONSTANT_SCALE = 0.0001


class Sen1Floods11Dataset(Dataset):
    """Sen1Floods11 hand-labeled segmentation dataset (no terratorch dependency)."""

    def __init__(
        self,
        data_root: str,
        split: str = "val",          # "train" | "val" | "test"
        transform: Optional[A.Compose] = None,
    ):
        data_root = Path(data_root)
        split_name = {"train": "train", "val": "valid", "test": "test"}[split]

        split_file = (data_root / "v1.1" / "splits" / "flood_handlabeled"
                      / f"flood_{split_name}_data.txt")
        with open(split_file) as f:
            valid_ids = {line.strip() for line in f if line.strip()}

        img_dir   = data_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S2Hand"
        label_dir = data_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"

        all_imgs = sorted(glob.glob(str(img_dir / "*_S2Hand.tif")))
        self.image_files = [
            p for p in all_imgs
            if any(vid in os.path.basename(p) for vid in valid_ids)
        ]
        self.label_files = [
            str(label_dir / os.path.basename(p).replace("_S2Hand.tif", "_LabelHand.tif"))
            for p in self.image_files
        ]

        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
                ToTensorV2(),
            ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        with rasterio.open(self.image_files[idx]) as src:
            img = src.read().astype(np.float32) * CONSTANT_SCALE  # (13, H, W)
        img = img[_BAND_IDX]                                        # (6, H, W)
        img = np.nan_to_num(img, nan=0.0)

        with rasterio.open(self.label_files[idx]) as src:
            mask = src.read(1).astype(np.int32)                    # (H, W)
        mask = np.where(np.isnan(mask.astype(np.float32)), -1, mask).astype(np.int32)

        img_hwc = img.transpose(1, 2, 0)
        out = self.transform(image=img_hwc, mask=mask)
        m = out["mask"]
        mask_t = m if isinstance(m, torch.Tensor) else torch.from_numpy(m)
        return {
            "image":    out["image"].float(),   # (6, H, W)
            "mask":     mask_t.long(),           # (H, W)
            "filename": os.path.basename(self.image_files[idx]),
        }


def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    eval_img_size: Optional[int] = None,
    **kwargs,
):
    """Return (train_dl, val_dl, test_dl), same API as data/sen1floods11.py."""
    if eval_img_size is None:
        eval_img_size = 512

    norm = A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0)

    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        norm,
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(eval_img_size, eval_img_size),
        norm,
        ToTensorV2(),
    ])

    train_ds = Sen1Floods11Dataset(data_root, split="train", transform=train_tf)
    val_ds   = Sen1Floods11Dataset(data_root, split="val",   transform=eval_tf)
    test_ds  = Sen1Floods11Dataset(data_root, split="test",  transform=eval_tf)

    mk = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **mk),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **mk),
        DataLoader(test_ds,  shuffle=False, drop_last=False, **mk),
    )
