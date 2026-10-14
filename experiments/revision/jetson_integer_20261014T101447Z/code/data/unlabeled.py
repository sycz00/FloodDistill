"""Unlabeled Sentinel-2 datasets for pseudo-label generation.

They all return {"image": Tensor[6, H, W]} in teacher band order [B2, B3, B4, B8A, B11, B12].
FloodUnlabeledDataset is the GDACS pool from datasets/download_flood_events.py and is
what generate_pseudo_labels() reads. UnlabeledWrapper drops 'mask' from a labeled
dataset, UnlabeledFolderDataset reads *.tif patches from a directory tree.

Loaders: get_flood_unlabeled_dataloader (GDACS pool), get_unlabeled_dataloader
(folders) and get_unlabeled_from_ood (OOD datasets without labels).
"""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Per-band normalization matching teacher fine-tuning on Sen1Floods11.
# Bands: [B2, B3, B4, B8A, B11, B12]
_NORM_MEAN = [0.1413, 0.1380, 0.1235, 0.3090, 0.2045, 0.1191]
_NORM_STD  = [0.0741, 0.0737, 0.0869, 0.1180, 0.0977, 0.0766]

_BANDS_WANTED = ["B2", "B3", "B4", "B8A", "B11", "B12"]
_DN_SCALE     = 0.0001
_MODE_DIR     = {"S2L1C": "s2c", "S2L2A": "s2a"}


# Generic wrappers

class UnlabeledWrapper(Dataset):
    """Drops the 'mask' key from any labeled dataset that returns a dict."""

    def __init__(self, labeled_dataset: Dataset) -> None:
        self.ds = labeled_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        return {"image": self.ds[idx]["image"]}


class UnlabeledFolderDataset(Dataset):
    """Load all *.tif files under root as unlabeled images.

        Reads the first n_bands raster bands, applies constant_scale, resizes
        to img_size x img_size, and returns {"image": Tensor[C, H, W]}.
    """

    def __init__(
        self,
        root: str,
        n_bands: int = 6,
        img_size: int = 224,
        constant_scale: float = 0.0001,
    ) -> None:
        self.files = sorted(Path(root).rglob("*.tif"))
        if not self.files:
            raise FileNotFoundError(f"No *.tif files found under {root!r}")
        self.n_bands        = n_bands
        self.constant_scale = constant_scale
        self.tf = A.Compose([
            A.Resize(img_size, img_size),
            A.pytorch.transforms.ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with rasterio.open(self.files[idx]) as src:
            bands = min(self.n_bands, src.count)
            img = src.read(range(1, bands + 1)).astype(np.float32)
        img = np.clip(img.transpose(1, 2, 0) * self.constant_scale, 0.0, 1.0)
        return {"image": self.tf(image=img)["image"]}


# GDACS flood-event patches

class FloodUnlabeledDataset(Dataset):
    """Unlabeled GDACS flood-event Sentinel-2 patches.

        Reads B2, B3, B4, B8A, B11, B12, the 6 bands the Prithvi teacher expects.
        B8A (narrow NIR, 20 m) gets resampled onto B2's native 264x264 grid.

        Layout (from datasets/download_flood_events.py):
            root/
                <event_id>_<patch_idx>/
                    <scene_system_id>/
                        B2.tif  B3.tif  B4.tif  B8A.tif  B11.tif  B12.tif

        Not normalised by default, pass normalize=True for teacher inference
        (pseudo-label generation).
    """

    _BAND_ORDER = ["B2", "B3", "B4", "B8A", "B11", "B12"]

    def __init__(
        self,
        root: str,
        img_size: int = 224,
        scale: float = 1e-4,
        normalize: bool = False,
    ) -> None:
        root_path = Path(root)
        self.scenes = sorted(
            p for p in root_path.glob("*/*")
            if p.is_dir() and (p / "B2.tif").exists()
        )
        if not self.scenes:
            why = ("that directory does not exist"
                   if not root_path.is_dir() else
                   "the directory exists but holds no <event>/<scene>/B2.tif")
            raise FileNotFoundError(
                f"No GDACS scenes found under {root!r} - {why}.\n"
                f"Expected the layout datasets/download_flood_events.py writes:\n"
                f"    <root>/<event_id>_<patch_idx>/<scene_id>/B2.tif ... B12.tif\n"
                f"Build the pool with:  EE_PROJECT=<id> bash datasets/prepare.sh\n"
                f"Under Docker, check the pool is inside the mount:\n"
                f"    -v <host>/flood_unlabeled:{root}:ro"
            )
        self.scale = scale
        tf_list = [A.Resize(img_size, img_size)]
        if normalize:
            tf_list.append(A.Normalize(
                mean=_NORM_MEAN, std=_NORM_STD, max_pixel_value=1.0,
            ))
        tf_list.append(A.pytorch.transforms.ToTensorV2())
        self.tf = A.Compose(tf_list)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        scene = self.scenes[idx]
        with rasterio.open(scene / "B2.tif") as ref:
            h, w = ref.height, ref.width

        bands = []
        for b in self._BAND_ORDER:
            with rasterio.open(scene / f"{b}.tif") as src:
                arr = src.read(
                    1,
                    out_shape=(h, w),
                    resampling=rasterio.enums.Resampling.bilinear,
                ).astype(np.float32) * self.scale
                bands.append(arr)
        img = np.clip(np.stack(bands, axis=0), 0.0, 1.0)
        img = img.transpose(1, 2, 0)
        return {"image": self.tf(image=img)["image"]}


# Factory functions

def get_flood_unlabeled_dataloader(
    root: str,
    batch_size: int = 16,
    num_workers: int = 8,
    img_size: int = 224,
) -> DataLoader:
    """DataLoader for GDACS flood-event unlabeled patches (B8A band)."""
    ds = FloodUnlabeledDataset(root, img_size=img_size, normalize=True)
    print(f"[flood_unlabeled] {len(ds)} scenes from {root!r}  "
          f"bands={FloodUnlabeledDataset._BAND_ORDER}  normalize=True")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_unlabeled_dataloader(
    roots: list[str],
    batch_size: int = 16,
    num_workers: int = 8,
    n_bands: int = 6,
    img_size: int = 224,
    constant_scale: float = 0.0001,
) -> DataLoader:
    """DataLoader from one or more directories of pre-processed GeoTIFF patches."""
    datasets = [
        UnlabeledFolderDataset(r, n_bands=n_bands, img_size=img_size,
                               constant_scale=constant_scale)
        for r in roots
    ]
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[unlabeled] {len(combined)} patches from {len(roots)} root(s)")
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


_OOD_DATASET_MAP = {
    "worldfloods": ("data.ood_datasets", "WorldFloodsDataset"),
    "floodplanet": ("data.ood_datasets", "FloodPlanetDataset"),
}


def get_unlabeled_from_ood(
    pairs: list[tuple[str, str]],
    batch_size: int = 16,
    num_workers: int = 8,
    img_size: int = 224,
) -> DataLoader:
    """Unlabeled DataLoader from named OOD datasets (strips mask with UnlabeledWrapper).

        Args:
            pairs: List of (dataset_name, root_path). Supported names: worldfloods, floodplanet.
    """
    import importlib

    datasets = []
    for name, root in pairs:
        root = str(Path(root).expanduser().resolve())
        if name not in _OOD_DATASET_MAP:
            raise ValueError(f"Unknown dataset {name!r}. Choose from: {list(_OOD_DATASET_MAP)}")
        module_path, class_name = _OOD_DATASET_MAP[name]
        cls = getattr(importlib.import_module(module_path), class_name)
        datasets.append(UnlabeledWrapper(cls(root=root, img_size=img_size)))
        print(f"[unlabeled] {name}: {len(datasets[-1])} samples from {root!r}")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[unlabeled] total: {sum(len(d) for d in datasets)} samples")
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
