"""Sen1Floods11 loaders. Wraps the TerraTorch datamodule but returns plain DataLoaders,
so training doesn't need Lightning.
"""

from __future__ import annotations

import albumentations as A
from torch.utils.data import DataLoader

from terratorch.datamodules import Sen1Floods11NonGeoDataModule

# Per-band stats of the Sen1Floods11 training set, after constant_scale=0.0001.
# Under Lightning TerraTorch applies them in on_after_batch_transfer, with plain
# DataLoaders we do it ourselves. Order: BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2
NORM_MEAN = [0.1413, 0.1380, 0.1235, 0.3090, 0.2045, 0.1191]
NORM_STD  = [0.0741, 0.0737, 0.0869, 0.1180, 0.0977, 0.0766]


def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 8,
    img_size: int = 224,
    eval_img_size: int | None = None,
    bands: list[str] | None = None,
    constant_scale: float = 0.0001,
    drop_last: bool = True,
    n_labeled_samples: int | None = None,
    seed: int | None = None,
    train_center_crop_px: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return training, validation and test loaders in the six-band teacher order.

    data_root contains v1.1/; TerraTorch appends that directory. Scale DN by
    constant_scale before applying NORM_MEAN/STD explicitly (no Lightning hook).
    Training resizes to img_size, then flips, rotates and applies per-channel
    multiplicative noise. Validation/test use eval_img_size, default 512.

    train_center_crop_px applies to training image and mask before resizing.
    The geometry control uses 264 native pixels before the 224 resize; validation
    and test are never cropped. n_labeled_samples is a seeded training subset;
    one-less-than-full is rejected. Batches contain image=(B,C,H,W), mask=(B,H,W)
    and filename.
    """
    if bands is None:
        bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

    if eval_img_size is None:
        eval_img_size = 512   # native Sen1Floods11 resolution (official eval uses full 512x512)

    norm = A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0)

    base_tf = [
        A.Resize(eval_img_size, eval_img_size),
        norm,
        A.pytorch.transforms.ToTensorV2(),
    ]
    train_tf = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Per-channel multiplicative noise for band-specific radiometric differences
        # (sensor, atmosphere, viewing angle). Replaces the accidental regularisation
        # the broken projectors gave in earlier experiments. +-20% stays within realistic
        # cross-sensor reflectance variation (Sentinel-2 inter-orbit is about +-5-15%).
        A.MultiplicativeNoise(multiplier=(0.80, 1.20), per_channel=True, p=0.5),
        norm,
        A.pytorch.transforms.ToTensorV2(),
    ]
    if train_center_crop_px is not None:
        # prepended so the other ops keep their position and args; with
        # train_center_crop_px=None the list is unchanged. Albumentations crops image
        # and mask the same way.
        train_tf = [A.CenterCrop(height=train_center_crop_px,
                                 width=train_center_crop_px)] + train_tf
        print(f"[sen1floods11] TRAIN centre-crop {train_center_crop_px}x{train_center_crop_px} "
              f"native px before the {img_size} resize "
              f"(extent {train_center_crop_px*10/1000:.2f} km, "
              f"effective GSD {train_center_crop_px*10/img_size:.1f} m)")

    dm = Sen1Floods11NonGeoDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        bands=bands,
        constant_scale=constant_scale,
        no_data_replace=0,
        no_label_replace=-1,
        train_transform=train_tf,
        val_transform=base_tf,
        test_transform=base_tf,
        drop_last=drop_last,
    )

    dm.setup("fit")
    dm.setup("test")

    train_dl = dm.train_dataloader()

    if n_labeled_samples is not None:
        import random
        from torch.utils.data import DataLoader as _DL, Subset
        full_ds  = train_dl.dataset
        n        = min(n_labeled_samples, len(full_ds))
        # split size minus one is always a mistake. It drops one seed-dependent image,
        # but the run name looks like the ceiling. Matched budget is the full split (252).
        if n == len(full_ds) - 1:
            raise ValueError(
                f"n_labeled_samples={n_labeled_samples} is one short of the full "
                f"Sen1Floods11 train split ({len(full_ds)}). That drops one "
                f"seed-dependent image and is not the label ceiling. Pass "
                f"{len(full_ds)}, or omit the flag entirely to use the whole split."
            )
        rng      = random.Random(seed if seed is not None else 0)
        indices  = rng.sample(range(len(full_ds)), n)
        train_dl = _DL(
            Subset(full_ds, indices),
            batch_size  = train_dl.batch_size,
            shuffle     = True,
            num_workers = train_dl.num_workers,
            pin_memory  = True,
            drop_last   = drop_last,
        )
        print(f"[sen1floods11] labeled subsample: {n}/{len(full_ds)} training images (seed={seed})")

    return train_dl, dm.val_dataloader(), dm.test_dataloader()
