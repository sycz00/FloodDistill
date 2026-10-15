"""Flood dataset loaders and evaluation preprocessing.

The paper reports STURM-S2 and WorldFloods-v2. Other loaders are retained for
existing callers. Band selection, class mappings, normalization, tiling and
perturbations are dataset-specific; the class docstrings describe them.
The paper's evaluation drivers set STURM normalize="original" explicitly.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
from rasterio.windows import Window
from albumentations.pytorch import ToTensorV2
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data.sen1floods11 import NORM_MEAN, NORM_STD


# Haze perturbation (Makarau et al., IEEE TGRS 2014): additive path radiance per band,
# reflectance'_i = reflectance_i + h * w_i, with w_i decreasing with wavelength
# (Angstrom law, alpha ~ 1). h is the blue-band strength in reflectance units. Applied in
# reflectance space before the fixed normalisation. Training only uses multiplicative
# jitter, so haze is outside the training augmentation.
def _haze_weights(wavelengths_nm, ref_nm: float = 490.0, alpha: float = 1.0):
    return ((ref_nm / np.asarray(wavelengths_nm, np.float32)) ** alpha).astype(np.float32)

# STURM S2 9-band order [B2,B3,B4,B5,B6,B7,B8A,B11,B12] (central wavelengths, nm)
_STURM_S2_WL   = [490., 560., 665., 705., 740., 783., 865., 1610., 2190.]
_STURM_HAZE_W  = _haze_weights(_STURM_S2_WL)
# Model / WF 6-band order [B2,B3,B4,B8A,B11,B12]
_S2_6BAND_WL   = [490., 560., 665., 865., 1610., 2190.]
_HAZE_W_6      = _haze_weights(_S2_6BAND_WL)


# T1: per-band radiometric gain, following Gorrono et al. (IEEE TGRS 62, 2024): Sentinel-2
# uncertainty differs per band (visible 5-20%, NIR/SWIR 3-5%) and is correlated within the
# VNIR and SWIR groups. A scalar gain cancels in normalised-difference indices; a per-band
# gain changes band ratios. Applied like sim_gain (reflectance space, before resize and
# normalisation, no clipping) and multiplied with it. None disables it.
def _as_band_gain(v, n_bands: int) -> np.ndarray | None:
    """Validate a per-band gain vector against the loader's NATIVE band count."""
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.size != n_bands:
        raise ValueError(f"sim_gain_bands must have {n_bands} entries (the loader's native "
                         f"band order), got {a.size}")
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"sim_gain_bands must be finite and > 0, got {a.tolist()}")
    if np.allclose(a, 1.0):
        return None                    # identity: no perturbation
    return a


# WorldFloods

# 13-band S2 order: [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12]. The models use
# B2,B3,B4,B8A,B11,B12 = indices [1,2,3,8,11,12] (index 8 is B8A, not B8).
# WorldFloodsv2 adds band 13 (valid mask) and band 14 (cloud probability); both are unused.
_WF_BAND_INDICES = [1, 2, 3, 8, 11, 12]   # B2, B3, B4, B8A, B11, B12

# WorldFloodsv2 test-split statistics for the six model bands (valid, cloud-free pixels).
# They lie about 0.4 sigma below the S11 training means, so normalize="s11" is the default;
# normalize="dataset" uses these values.
WF_NORM_MEAN = [0.1139, 0.1089, 0.0925, 0.2280, 0.1531, 0.0875]
WF_NORM_STD  = [0.0315, 0.0376, 0.0544, 0.1253, 0.1023, 0.0712]


# Evaluation protocols. The default "surface_water" scores every water pixel, permanent water
# included, as Sen1Floods11 does (approved 2026-10-15; experiments/revision/
# surface_water_eval_20261014/PROTOCOL.md). The flood-only protocols remain for the paper tables.
#   WF:    name -> (JRC permanent water, edge-window geometry)
#   STURM: name -> (class mapping, ignore pixels without finite input in the six model bands)
WF_PROTOCOLS = {
    "surface_water": ("water", "pad"),
    "flood_p3": ("ignore", "pad"),
    "flood_paper": ("ignore", "resize"),
    "operational": ("background", "resize"),
}
STURM_PROTOCOLS = {
    "surface_water": ("surface", True),
    "flood_p3": ("paper", True),
    "flood_paper": ("paper", False),
    "operational": ("operational", False),
}
DEFAULT_PROTOCOL = "surface_water"
# STURM 9-band order [B2,B3,B4,B5,B6,B7,B8A,B11,B12] -> model bands B2,B3,B4,B8A,B11,B12
_STURM_MODEL_BANDS = [0, 1, 2, 6, 7, 8]


def _wf_resolve_protocol(protocol, ignore_permanent_water, permanent_water_as):
    """Return (name, permanent-water handling, edge geometry). The old keyword arguments
    reproduce the pre-2026-10-15 behaviour, including resized edge windows."""
    legacy = ignore_permanent_water is not None or permanent_water_as is not None
    if protocol is not None:
        if legacy:
            raise ValueError("pass either protocol or ignore_permanent_water/permanent_water_as")
        if protocol not in WF_PROTOCOLS:
            raise ValueError(f"protocol must be one of {sorted(WF_PROTOCOLS)}, got {protocol!r}")
        return (protocol,) + WF_PROTOCOLS[protocol]
    if not legacy:
        return (DEFAULT_PROTOCOL,) + WF_PROTOCOLS[DEFAULT_PROTOCOL]
    pwa = "ignore" if permanent_water_as is None else permanent_water_as
    if pwa not in ("ignore", "background"):
        raise ValueError(f"permanent_water_as must be 'ignore' or 'background', got {pwa!r}")
    if ignore_permanent_water is False:
        return "legacy_include_permanent", "water", "resize"
    return ("flood_paper", "ignore", "resize") if pwa == "ignore" else ("operational", "background", "resize")


def _wf_detect_layout(root: Path, split: str) -> tuple[str, Path]:
    """Return ("v2", split_dir) for WorldFloodsv2 or ("v1", data_dir) for the old Zenodo subset.

    v2 layout: <root>/{train,val,test}/{S2,gt,meta,...}/<scene>.tif; <root> may be a split dir.
    """
    root = Path(root)
    for base in (root, root / "WorldFloodsv2"):
        if (base / split / "S2").is_dir() and (base / split / "gt").is_dir():
            return "v2", base / split
    # root itself is a split dir (.../test with S2/ and gt/ inside)
    if (root / "S2").is_dir() and (root / "gt").is_dir():
        return "v2", root
    data_dir = root / "world_floods" if (root / "world_floods").is_dir() else root
    return "v1", data_dir


class WorldFloodsDataset(Dataset):
    """WorldFloods Sentinel-2 flood (land/water) segmentation.

    WorldFloodsv2 (Portales-Julia et al. 2023), from Hugging Face isp-uv-es/WorldFloodsv2:
        <root>/{train,val,test}/S2/<scene>.tif   15 bands: 13 L1C bands, valid mask, cloud probability
        <root>/{train,val,test}/gt/<scene>.tif   [0] cloud {0 invalid, 1 clear, 2 cloud},
                                                 [1] land/water {0 invalid, 1 land, 2 water}
        <root>/{split}/PERMANENTWATERJRC/<scene>.tif  0 nodata, 1 no water, 2 seasonal, 3 permanent

    Preprocessing:
      * bands B2,B3,B4,B8A,B11,B12 (indices 1,2,3,8,11,12), DN * 1e-4, clipped to [0, 1]
      * normalize="s11" (default): Sen1Floods11 training statistics
      * scenes are split into non-overlapping `tile_size` windows at 10 m
      * labels: land -> 0, water -> 1, invalid -> ignore; cloud -> ignore if exclude_clouds
      * tiles with valid fraction <= min_valid_frac are dropped

    protocol (see WF_PROTOCOLS; default "surface_water" since 2026-10-15):
      * "surface_water": all gt water (flood + JRC permanent) is water, as in Sen1Floods11;
        partial edge windows are normalized at native size and zero-padded to tile_size
        (mask padded with ignore). 836 test windows at tile_size=512.
      * "flood_p3": JRC permanent water (3) -> ignore, zero-padded edge windows
        (revision protocol P3_WF_NATIVE_ZERO_PAD512_V1, 785 windows).
      * "flood_paper": JRC permanent water -> ignore, edge windows resized (submitted paper).
      * "operational": JRC permanent water -> land, edge windows resized.
    The old arguments ignore_permanent_water / permanent_water_as still work and select
    the old behaviour exactly (edge windows resized); they cannot be combined with protocol.
    partial_windows ("pad" or "resize") overrides the protocol's edge-window geometry.

    The older v1 subset (Zenodo 8298840, *_S2.tif / *_CLASS.tif; 1 -> land, 2 -> water) is
    still read for older results.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        img_size: int = 256,
        augment: bool = False,
        normalize: str = "s11",
        tile_size: int = 256,
        exclude_clouds: bool = True,
        min_valid_frac: float = 0.01,
        sim_gain: float = 1.0,
        sim_offset: float = 0.0,
        sim_haze: float = 0.0,
        sim_gain_bands=None,
        protocol: str | None = None,
        partial_windows: str | None = None,
        ignore_permanent_water: bool | None = None,
        permanent_water_as: str | None = None,
    ) -> None:
        super().__init__()
        self.protocol, permanent, geometry = _wf_resolve_protocol(
            protocol, ignore_permanent_water, permanent_water_as)
        if partial_windows is not None:
            geometry = partial_windows
        if geometry not in ("pad", "resize"):
            raise ValueError(f"partial_windows must be 'pad' or 'resize', got {geometry!r}")
        self.partial_windows = geometry
        # JRC permanent water (3): "water" keeps the gt label, "ignore" masks it (flood-only),
        # "background" scores it as land (operational).
        self.ignore_permanent_water = permanent != "water"
        self.permanent_water_as = "background" if permanent == "background" else "ignore"
        if normalize not in ("s11", "dataset"):
            raise ValueError(f"normalize must be 's11' or 'dataset', got '{normalize}'")
        self.normalize = normalize
        self.tile_size = tile_size
        self.exclude_clouds = exclude_clouds
        # Radiometric perturbation reflectance' = sim_gain*reflectance + sim_offset, applied
        # before the S11 normalisation as in the STURM loader.
        self.sim_gain = float(sim_gain)
        self.sim_offset = float(sim_offset)
        # Haze robustness perturbation (additive per-band path radiance, wavelength-weighted;
        # see _haze_weights). 0.0 = off. Applied in reflectance space, same hook as sim_gain.
        self.sim_haze = float(sim_haze)
        # T1 per-band gain (see _as_band_gain). Six entries in the loader's own band order,
        # which for WorldFloods is already the model's: [B2, B3, B4, B8A, B11, B12].
        self.sim_gain_bands = _as_band_gain(sim_gain_bands, 6)

        self.version, data_dir = _wf_detect_layout(Path(root), split)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"WorldFloods data directory not found: {data_dir}\n"
                f"v2: huggingface-cli download --repo-type dataset "
                f"isp-uv-es/WorldFloodsv2 --local-dir $DATA_ROOT/WorldFloodsv2\n"
                f"v1: download world_floods.7z from https://zenodo.org/records/8298840"
            )

        if self.version == "v2":
            self._init_v2(data_dir, min_valid_frac)
        else:
            self._init_v1(data_dir)

        # normalize="s11": Sen1Floods11 training statistics. "dataset": WorldFloodsv2 statistics.
        _m = list(NORM_MEAN) if normalize == "s11" else WF_NORM_MEAN
        _s = list(NORM_STD)  if normalize == "s11" else WF_NORM_STD
        ops = [A.Resize(img_size, img_size),
               A.Normalize(mean=_m, std=_s, max_pixel_value=1.0)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)
        # Zero-padded edge windows: normalize at native size, pad to tile_size, then resize
        # (only if img_size != tile_size) and flip (same as evaluation.revision_native_border).
        self._normalize = A.Normalize(mean=_m, std=_s, max_pixel_value=1.0)
        pad_ops = [] if img_size == tile_size else [A.Resize(img_size, img_size)]
        if augment:
            pad_ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        self._pad_transform = A.Compose(pad_ops + [ToTensorV2()])

    # v1 (legacy *_S2.tif / *_CLASS.tif)
    def _init_v1(self, data_dir: Path) -> None:
        s2_files  = {Path(p).name.replace("_S2.tif", ""): p
                     for p in sorted(glob.glob(str(data_dir / "*_S2.tif")))}
        cls_files = {Path(p).name.replace("_CLASS.tif", ""): p
                     for p in sorted(glob.glob(str(data_dir / "*_CLASS.tif")))}
        common = sorted(s2_files.keys() & cls_files.keys())
        if not common:
            raise RuntimeError(
                f"No matched *_S2.tif / *_CLASS.tif pairs found in {data_dir}.\n"
                f"  S2 files: {len(s2_files)}, CLASS files: {len(cls_files)}"
            )
        # (s2_path, cls_path, window), window None means read the whole file
        self.samples = [(s2_files[k], cls_files[k], None) for k in common]

    # v2 (509-pair release, tiled)
    def _init_v2(self, split_dir: Path, min_valid_frac: float) -> None:
        s2_files = {p.name: p for p in sorted((split_dir / "S2").glob("*.tif"))}
        gt_files = {p.name: p for p in sorted((split_dir / "gt").glob("*.tif"))}
        common = sorted(s2_files.keys() & gt_files.keys())
        if not common:
            raise RuntimeError(
                f"No matched S2/gt pairs in {split_dir} (S2={len(s2_files)}, gt={len(gt_files)})."
            )

        # Flood-only protocol needs the JRC permanent-water layer; a missing layer is an error.
        self.jrc_files: dict[str, Path] = {}
        if self.ignore_permanent_water:
            jrc_dir = split_dir / "PERMANENTWATERJRC"
            self.jrc_files = {p.name: p for p in sorted(jrc_dir.glob("*.tif"))}
            missing = [n for n in common if n not in self.jrc_files]
            if missing:
                raise FileNotFoundError(
                    f"ignore_permanent_water=True but {len(missing)}/{len(common)} scenes "
                    f"lack a JRC layer in {jrc_dir} (e.g. {missing[0]}). Download the "
                    f"PERMANENTWATERJRC folder or pass ignore_permanent_water=False."
                )

        # Tile index: read each GT once and keep only windows with scorable pixels.
        ts = self.tile_size
        self.samples: list[tuple[Path, Path, tuple[int, int, int, int] | None]] = []
        for name in common:
            gt_path = gt_files[name]
            with rasterio.open(gt_path) as src:
                gt = src.read()                      # (2, H, W)
            H, W = gt.shape[1], gt.shape[2]
            lw = gt[1]
            valid = (lw == 1) | (lw == 2)
            if self.exclude_clouds:
                valid &= gt[0] != 2
            if self.ignore_permanent_water:
                with rasterio.open(self.jrc_files[name]) as src:
                    valid &= src.read(1) != 3        # 3 = JRC permanent water
            for r in range(0, H, ts):
                for c in range(0, W, ts):
                    h = min(ts, H - r); w = min(ts, W - c)
                    if float(valid[r:r + h, c:c + w].mean()) > min_valid_frac:
                        self.samples.append((s2_files[name], gt_path, (c, r, w, h)))

        if not self.samples:
            raise RuntimeError(
                f"No valid tiles in {split_dir} (min_valid_frac={min_valid_frac}). "
                f"Lower min_valid_frac or check the gt/ masks."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _getitem_v1(self, img_path: Path, lbl_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with rasterio.open(img_path) as src:
            all_bands = src.read().astype(np.float32)     # (13, H, W)
        with rasterio.open(lbl_path) as src:
            mask = src.read(1).astype(np.int64)
        image = np.clip(all_bands[_WF_BAND_INDICES] * 0.0001, 0.0, 1.0)  # (6, H, W)
        nodata_mask = (image == 0.0).all(axis=0)
        mapped = np.full_like(mask, fill_value=-1)
        mapped[mask == 1] = 0
        mapped[mask == 2] = 1
        mapped[nodata_mask] = -1
        return image, mapped

    def _getitem_v2(
        self, img_path: Path, gt_path: Path, window: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        c, r, w, h = window
        win = Window(c, r, w, h)
        with rasterio.open(img_path) as src:
            all_bands = src.read(window=win).astype(np.float32)   # (15, h, w)
        with rasterio.open(gt_path) as src:
            gt = src.read(window=win).astype(np.int64)            # (2, h, w)

        image = np.clip(all_bands[_WF_BAND_INDICES] * 0.0001, 0.0, 1.0)  # (6, h, w)
        # Nodata: all 13 spectral bands zero (swath edge). Use spectral bands only -
        # the aux bands (13,14) can be non-zero where the spectrum is empty.
        nodata_mask = (all_bands[:13] == 0.0).all(axis=0)

        lw = gt[1]                                   # land/water: 0=invalid,1=land,2=water
        mapped = np.full((h, w), -1, dtype=np.int64)
        mapped[lw == 1] = 0
        mapped[lw == 2] = 1
        if self.exclude_clouds:
            mapped[gt[0] == 2] = -1                   # cloud band == 2
        if self.ignore_permanent_water:
            with rasterio.open(self.jrc_files[img_path.name]) as src:
                perm = src.read(1, window=win) == 3         # JRC permanent water
            mapped[perm & (mapped != -1)] = -1 if self.permanent_water_as == "ignore" else 0
        mapped[nodata_mask] = -1
        return image, mapped

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_path, lbl_path, window = self.samples[idx]
        if window is None:
            image, mapped = self._getitem_v1(img_path, lbl_path)
            fname = Path(img_path).name
        else:
            image, mapped = self._getitem_v2(img_path, lbl_path, window)
            c, r, w, h = window
            fname = f"{Path(img_path).name}#r{r}_c{c}"

        if self.sim_gain != 1.0 or self.sim_offset != 0.0:
            # Same as the STURM loader: perturb in reflectance space without clipping again,
            # so gains above 1 are not cut off.
            image = image * self.sim_gain + self.sim_offset

        if self.sim_gain_bands is not None:
            # T1: per-band gain, same hook and same no-clip convention as sim_gain above.
            image = image * self.sim_gain_bands[:, None, None]

        if self.sim_haze != 0.0:
            # additive wavelength-weighted haze (Makarau 2014); no clip, mirroring sim_gain
            image = image + self.sim_haze * _HAZE_W_6[:, None, None]

        ts = self.tile_size
        if window is not None and self.partial_windows == "pad" and mapped.shape != (ts, ts):
            h, w = mapped.shape
            image = self._normalize(image=image.transpose(1, 2, 0))["image"]
            image = np.pad(image, ((0, ts - h), (0, ts - w), (0, 0)), constant_values=0.0)
            mapped = np.pad(mapped, ((0, ts - h), (0, ts - w)), constant_values=-1)
            result = self._pad_transform(image=image, mask=mapped)
        else:
            result = self.transform(image=image.transpose(1, 2, 0), mask=mapped)
        return {
            "image":    result["image"].float(),
            "mask":     result["mask"].long(),
            "filename": fname,
        }


def get_worldfloods_dataloader(
    root: str | Path,
    split: str = "test",
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 256,
    shuffle: bool = False,
    normalize: str = "s11",
    tile_size: int = 256,
    exclude_clouds: bool = True,
    min_valid_frac: float = 0.01,
    sim_gain: float = 1.0,
    sim_offset: float = 0.0,
    sim_haze: float = 0.0,
    sim_gain_bands=None,
    protocol: str | None = None,
    partial_windows: str | None = None,
    ignore_permanent_water: bool | None = None,
    permanent_water_as: str | None = None,
) -> DataLoader:
    """Evaluation DataLoader for WorldFloods (v1 subset or v2).

    split: "train" | "val" | "test" (v2 only).
    normalize: "s11" (default, training statistics) or "dataset" (WorldFloodsv2 statistics).
    tile_size: window size for v2 scenes (default 256).
    exclude_clouds: ignore cloud pixels (default True).
    min_valid_frac: drop tiles with a smaller valid fraction (default 0.01).
    protocol: WF_PROTOCOLS key; default "surface_water" (flood + permanent water = water).
    partial_windows: "pad" or "resize" edge windows; None = the protocol's choice.
    ignore_permanent_water / permanent_water_as: old arguments, old behaviour (see class).
    sim_gain_bands: per-band gain in model band order [B2,B3,B4,B8A,B11,B12]; None = off.
    """
    ds = WorldFloodsDataset(
        root=root, split=split, img_size=img_size, augment=False, normalize=normalize,
        tile_size=tile_size, exclude_clouds=exclude_clouds, min_valid_frac=min_valid_frac,
        sim_gain=sim_gain, sim_offset=sim_offset, sim_haze=sim_haze,
        sim_gain_bands=sim_gain_bands, protocol=protocol, partial_windows=partial_windows,
        ignore_permanent_water=ignore_permanent_water,
        permanent_water_as=permanent_water_as,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)




# FloodPlanet

# Indices into FloodPlanet's 10-band S2 stack (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12)
_FP_BAND_INDICES = [0, 1, 2, 7, 8, 9]   # B2, B3, B4, B8A, B11, B12

# FloodPlanet L2A statistics (278 non-Kansas patches, nodata excluded), for normalize="dataset".
FP_NORM_MEAN = [0.1572, 0.1561, 0.1453, 0.2751, 0.1729, 0.0861]
FP_NORM_STD  = [0.1276, 0.1237, 0.1338, 0.1518, 0.1222, 0.0945]

# P2/P98 retained for backwards-compat (normalize="percentile") but no longer recommended.
_FP_P2  = np.array([0.0366, 0.0528, 0.0335, 0.0235, 0.0087, 0.0056], dtype=np.float32)
_FP_P98 = np.array([0.5341, 0.5202, 0.5472, 0.6406, 0.5102, 0.3823], dtype=np.float32)


class FloodPlanetDataset(Dataset):
    """FloodPlanet Sentinel-2 flood segmentation.

    278 S2 patches from 18 flood events (2017-2020); US-Kansas is excluded because of the
    label quality note in the dataset README.

    Download:
        wget -O $DATA_ROOT/FloodPlanet.zip \
            "https://zenodo.org/api/records/15238572/files/FloodPlanet.zip/content"
        cd $DATA_ROOT && unzip FloodPlanet.zip -d FloodPlanet/

    Structure:
        <root>/FloodPlanet/<event>/S2/*.tif      10 bands, float32, raw DN
        <root>/FloodPlanet/<event>/labels/*.tif  uint8, {1, 2}

    Labels: 1 -> 0 (land), 2 -> 1 (flood), nodata -> -1.

    normalize="clip" (default): bands B2,B3,B4,B8A,B11,B12 (indices 0,1,2,7,8,9), DN * 1e-4,
    clip to [0, 1], S11 statistics. "percentile" stretches data that is already reflectance
    and is not recommended; "dataset" uses the FloodPlanet L2A statistics.
    """

    def __init__(
        self,
        root: str | Path,
        img_size: int = 320,
        augment: bool = False,
        exclude_kansas: bool = True,
        normalize: str = "clip",
    ) -> None:
        super().__init__()
        if normalize not in ("dataset", "percentile", "clip"):
            raise ValueError(f"normalize must be 'dataset', 'percentile', or 'clip', got '{normalize}'")
        self.normalize = normalize

        root = Path(root)
        data_dir = root / "FloodPlanet" if (root / "FloodPlanet").is_dir() else root
        if not data_dir.exists():
            raise FileNotFoundError(
                f"FloodPlanet data directory not found: {data_dir}\n"
                f"Download from https://zenodo.org/records/15238572"
            )

        s2_all  = {Path(p).name: p for p in
                   sorted(glob.glob(str(data_dir / "*/S2/*.tif")))
                   if "__MACOSX" not in p}
        lbl_all = {Path(p).name: p for p in
                   sorted(glob.glob(str(data_dir / "*/labels/*.tif")))
                   if "__MACOSX" not in p}
        common = sorted(s2_all.keys() & lbl_all.keys())

        if not common:
            raise RuntimeError(
                f"No matched S2/labels pairs found in {data_dir}.\n"
                f"  S2 files: {len(s2_all)}, label files: {len(lbl_all)}"
            )

        self.samples = []
        for name in common:
            s2_path  = s2_all[name]
            lbl_path = lbl_all[name]
            if exclude_kansas and "US-Kansas" in s2_path:
                continue
            self.samples.append((s2_path, lbl_path))

        _norm_mean = FP_NORM_MEAN if normalize == "dataset" else NORM_MEAN
        _norm_std  = FP_NORM_STD  if normalize == "dataset" else NORM_STD
        ops = [A.Resize(img_size, img_size),
               A.Normalize(mean=_norm_mean, std=_norm_std, max_pixel_value=1.0)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_path, lbl_path = self.samples[idx]
        with rasterio.open(img_path) as src:
            all_bands = src.read().astype(np.float32)   # (10, H, W)
        with rasterio.open(lbl_path) as src:
            mask_raw = src.read(1).astype(np.int64)

        image = all_bands[_FP_BAND_INDICES] * 0.0001   # (6, H, W), reflectance

        # Detect nodata pixels (swath edges - all 6 bands near zero)
        nodata_mask = (image < 0.001).all(axis=0)

        if self.normalize == "percentile":
            p2  = _FP_P2[:, None, None]
            p98 = _FP_P98[:, None, None]
            image = (image - p2) / (p98 - p2)
        image = np.clip(image, 0.0, 1.0)

        h, w = image.shape[1], image.shape[2]
        if mask_raw.shape != (h, w):
            import cv2
            mask_raw = cv2.resize(mask_raw.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int64)

        mapped = mask_raw - 1   # {1,2} to {0,1}
        if nodata_mask.any():
            mapped[nodata_mask] = -1

        result = self.transform(image=image.transpose(1, 2, 0), mask=mapped)
        return {
            "image":    result["image"].float(),
            "mask":     result["mask"].long(),
            "filename": str(Path(img_path).name),
        }


def get_floodplanet_dataloader(
    root: str | Path,
    split: str = "test",
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 320,
    shuffle: bool = False,
    exclude_kansas: bool = True,
    normalize: str = "clip",
) -> DataLoader:
    """Evaluation DataLoader for FloodPlanet; normalize as in FloodPlanetDataset."""
    ds = FloodPlanetDataset(root=root, img_size=img_size, augment=False,
                             exclude_kansas=exclude_kansas, normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# STURM-Flood

# Per-band P2/P98 from 267 STURM-Flood S2 tiles, bands [B2,B3,B4,B5,B6,B7,B8A,B11,B12].
# Used by normalize="percentile" (older protocol, loader default).
_STURM_S2_P2  = np.array([-0.1725,  0.0050, -0.1000,  0.1465,  0.1570,
                            0.1660,  0.1583,  0.0032, -0.1161], dtype=np.float32)
_STURM_S2_P98 = np.array([ 1.6842,  2.1567,  2.2708,  2.3995,  3.0805,
                             3.6797,  3.7275,  2.9681,  2.3909], dtype=np.float32)

# normalize="original" (paper protocol): the STURM paper (Sec. 2.2) scales DN 400-1600 to
# [0,1] for all bands, so reflectance = (400 + 1200*v) * 1e-4 = 0.04 + 0.12*v.
_STURM_S2_DN_OFFSET = 0.04
_STURM_S2_DN_SCALE  = 0.12

_STURM_S2_NORM_MEAN = [NORM_MEAN[0], NORM_MEAN[1], NORM_MEAN[2],   # B2, B3, B4
                       0.1900, 0.2400, 0.2700,                       # B5, B6, B7 (dropped)
                       NORM_MEAN[3], NORM_MEAN[4], NORM_MEAN[5]]    # B8A, B11, B12
_STURM_S2_NORM_STD  = [NORM_STD[0],  NORM_STD[1],  NORM_STD[2],
                       0.0950, 0.1050, 0.1120,
                       NORM_STD[3],  NORM_STD[4],  NORM_STD[5]]


class STURMFloodDataset(Dataset):
    """STURM-Flood Sentinel-1 or Sentinel-2 chips (128x128, 60 flood events).

    Download: https://zenodo.org/records/12748983 (Dataset.zip)

    Structure:
        <root>/Sentinel2/S2/         9 bands, float64
        <root>/Sentinel2/Floodmaps/  labels 0-5
        <root>/Sentinel1/S1/         VV, VH
        <root>/Sentinel1/Floodmaps/

    Labels: 0 background, 1 flooded, 2 river-related features, 3 open water, 4 reservoirs,
    5 lakes, 99 no data.

    protocol (see STURM_PROTOCOLS; default "surface_water" since 2026-10-15):
      * "surface_water": classes 1-5 are water (STURM's own definition), 0 land, 99 ignore;
        pixels whose six model bands are not all finite are ignored (at img_size != 128 the
        bilinear footprint of the non-finite pixels, as in the image resize).
      * "flood_p3": 1-2 -> flood, 3-5 -> ignore, non-finite pixels ignored (revision P3).
      * "flood_paper": 1-2 -> flood, 3-5 -> ignore, no finite-support mask (submitted paper).
      * "operational": 1-2 -> flood, 3-5 -> land.
    The old argument class_mapping ("paper", "published", "operational") selects the old
    behaviour exactly (no finite-support mask); `include_possible` only affects "paper".
    The finite-support mask is evaluation-only and cannot be combined with augment=True.

    normalize: "percentile" (loader default, older fitted P2/P98 stretch), "original"
    (paper protocol, 0.04 + 0.12*v with NaN guard) or "clip".
    """

    N_BANDS: dict[str, int] = {"S1": 2, "S2": 9}

    def __init__(
        self,
        root: str | Path,
        sensor: str = "S2",
        img_size: int = 224,
        augment: bool = False,
        include_possible: bool = True,
        normalize: str = "percentile",
        sim_gain: float = 1.0,
        sim_offset: float = 0.0,
        sim_haze: float = 0.0,
        sim_gain_bands=None,
        protocol: str | None = None,
        class_mapping: str | None = None,
    ) -> None:
        super().__init__()
        root = Path(root)
        if protocol is not None and class_mapping is not None:
            raise ValueError("pass either protocol or class_mapping")
        if class_mapping is not None:
            self.protocol, mask_nonfinite = f"legacy_{class_mapping}", False
        else:
            self.protocol = DEFAULT_PROTOCOL if protocol is None else protocol
            if self.protocol not in STURM_PROTOCOLS:
                raise ValueError(f"protocol must be one of {sorted(STURM_PROTOCOLS)}, "
                                 f"got {self.protocol!r}")
            class_mapping, mask_nonfinite = STURM_PROTOCOLS[self.protocol]
        if mask_nonfinite and augment:
            raise ValueError("the finite-support mask is evaluation-only; use augment=False")
        self.mask_nonfinite = mask_nonfinite
        self.img_size = img_size
        if sensor not in ("S1", "S2"):
            raise ValueError(f"sensor must be 'S1' or 'S2', got '{sensor}'")
        if normalize not in ("percentile", "clip", "original"):
            raise ValueError(
                f"normalize must be 'percentile', 'clip', or 'original', got '{normalize}'"
            )

        self.sensor = sensor
        self.include_possible = include_possible
        self.normalize = normalize
        # Radiometric perturbation reflectance' = sim_gain*reflectance + sim_offset, applied
        # before the S11 normalisation. (1.0, 0.0) disables it.
        self.sim_gain = float(sim_gain)
        self.sim_offset = float(sim_offset)
        # Haze robustness perturbation (additive per-band path radiance, wavelength-weighted;
        # see _haze_weights / Makarau 2014). 0.0 = off. S2 only. Applied like sim_gain.
        self.sim_haze = float(sim_haze)
        # T1 per-band gain in STURM's native order [B2,B3,B4,B5,B6,B7,B8A,B11,B12] (nine
        # entries). B5-B7 are dropped by sturm_s2_adapter; perscene_eval._sturm9 builds the vector.
        self.sim_gain_bands = _as_band_gain(sim_gain_bands, self.N_BANDS[sensor])
        # Class mappings: "paper": 0 -> background, 1-2 -> flood, 3-5 -> ignore.
        # "published"/"surface": STURM's own mapping, classes 1-5 are water. "operational":
        # 3-5 count as background. Do not mix them in one table.
        if class_mapping not in ("paper", "published", "surface", "operational"):
            raise ValueError(f"class_mapping must be 'paper', 'published' or 'operational', "
                             f"got {class_mapping!r}")
        self.class_mapping = "published" if class_mapping == "surface" else class_mapping

        sensor_num = "1" if sensor == "S1" else "2"
        sensor_dir = root / f"Sentinel{sensor_num}"
        img_dir    = sensor_dir / sensor
        lbl_dir    = sensor_dir / "Floodmaps"
        csv_path   = root / f"Sentinel{sensor_num}_metadata.csv"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                f"Download and extract Dataset.zip from https://zenodo.org/records/12748983"
            )

        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            tile_ids = df["tile_id"].astype(str).tolist()
        else:
            tile_ids = [p.name for p in sorted(img_dir.glob("*.tif"))]

        self.samples: list[tuple[Path, Path]] = []
        for tid in tile_ids:
            ip = img_dir / tid
            lp = lbl_dir / tid
            if ip.exists() and lp.exists():
                self.samples.append((ip, lp))

        if not self.samples:
            raise RuntimeError(
                f"No matched image/label pairs found.\n"
                f"  Images: {img_dir}  Labels: {lbl_dir}"
            )

        ops = [A.Resize(img_size, img_size)]
        if sensor == "S2":
            ops.append(A.Normalize(mean=_STURM_S2_NORM_MEAN, std=_STURM_S2_NORM_STD,
                                    max_pixel_value=1.0))
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_path, lbl_path = self.samples[idx]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(lbl_path) as src:
            mask = src.read(1).astype(np.int64)

        nonfinite = None
        if self.mask_nonfinite:
            bands = _STURM_MODEL_BANDS if self.sensor == "S2" else slice(None)
            nonfinite = ~np.isfinite(image[bands]).all(axis=0)
            if nonfinite.shape != (self.img_size, self.img_size):
                import cv2
                nonfinite = cv2.resize(nonfinite.astype(np.float32), (self.img_size, self.img_size),
                                       interpolation=cv2.INTER_LINEAR) > 0

        if self.normalize == "original":
            # About 0.6% of STURM tiles contain NaN pixels; set them to 0 ("original" only).
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.sensor == "S2" and self.normalize == "percentile":
            p2  = _STURM_S2_P2[:, None, None]
            p98 = _STURM_S2_P98[:, None, None]
            image = (image - p2) / (p98 - p2)
        elif self.sensor == "S2" and self.normalize == "original":
            image = _STURM_S2_DN_OFFSET + _STURM_S2_DN_SCALE * image
        image = np.clip(image, 0.0, 1.0)

        # E5 radiometric perturbation in reflectance space (before A.Normalize). Not
        # re-clipped: an over/under-gained reflectance legitimately leaves [0,1].
        if self.sim_gain != 1.0 or self.sim_offset != 0.0:
            image = image * self.sim_gain + self.sim_offset

        if self.sim_gain_bands is not None:
            # T1: per-band gain, same hook and same no-clip convention as sim_gain above.
            image = image * self.sim_gain_bands[:, None, None]

        if self.sensor == "S2" and self.sim_haze != 0.0:
            # additive wavelength-weighted haze (Makarau 2014), 9-band S2; no clip (mirrors gain)
            image = image + self.sim_haze * _STURM_HAZE_W[:, None, None]

        if self.class_mapping == "published":
            # STURM's own binary remap: classes 1-5 are water, 0 is background, 99 no-data.
            mask = np.where(mask == 0, 0,
                   np.where((mask >= 1) & (mask <= 5), 1, -1)).astype(np.int64)
        elif self.class_mapping == "operational":
            mask = np.where(mask == 0, 0,
                   np.where(mask == 1, 1,
                   np.where(mask == 2, 1,
                   np.where((mask >= 3) & (mask <= 5), 0, -1)))).astype(np.int64)
        else:
            flood_val = 1 if self.include_possible else -1
            mask = np.where(mask == 0, 0,
                   np.where(mask == 1, 1,
                   np.where(mask == 2, flood_val, -1))).astype(np.int64)

        result = self.transform(image=image.transpose(1, 2, 0), mask=mask)
        if nonfinite is not None:
            result["mask"][torch.from_numpy(nonfinite)] = -1
        return {
            "image":    result["image"].float(),
            "mask":     result["mask"].long(),
            "filename": str(img_path.name),
        }


def get_sturm_flood_dataloader(
    root: str | Path,
    sensor: str = "S2",
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    shuffle: bool = False,
    include_possible: bool = True,
    normalize: str = "percentile",
    sim_gain: float = 1.0,
    sim_offset: float = 0.0,
    sim_haze: float = 0.0,
    sim_gain_bands=None,
    protocol: str | None = None,
    class_mapping: str | None = None,
) -> DataLoader:
    """Evaluation DataLoader for STURM-Flood.

    normalize: "percentile" (default), "original" (paper protocol) or "clip".
    sim_gain / sim_offset: reflectance' = sim_gain*reflectance + sim_offset; (1.0, 0.0) = off.
    sim_gain_bands: per-band gain in STURM order [B2,B3,B4,B5,B6,B7,B8A,B11,B12]; None = off.
    protocol: STURM_PROTOCOLS key; default "surface_water" (classes 1-5 water).
    class_mapping: old argument, "paper", "published" or "operational" (old behaviour).
    """
    ds = STURMFloodDataset(
        root=root, sensor=sensor, img_size=img_size,
        augment=False, include_possible=include_possible,
        normalize=normalize, sim_gain=sim_gain, sim_offset=sim_offset, sim_haze=sim_haze,
        sim_gain_bands=sim_gain_bands, protocol=protocol, class_mapping=class_mapping,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# DeepFlood

_DF_SAR_DB_MIN = -25.0
_DF_SAR_DB_MAX = 5.0
_DF_OPT_SCALE  = 0.0001


def inspect_deepflood_root(root: str | Path) -> None:
    """Print the directory structure to check the DeepFlood layout."""
    root = Path(root)
    print(f"Inspecting: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > 3:
            continue
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/")
        for f in sorted(filenames)[:3]:
            print(f"{indent}  {f}")
        if len(filenames) > 3:
            print(f"{indent}  ... ({len(filenames)} files total)")


class DeepFloodDataset(Dataset):
    """DeepFlood SAR or optical chips (4,831 chips, 11 events).

    Paper: https://doi.org/10.1038/s41597-025-04554-3

    Structure (from the paper; check with inspect_deepflood_root('<root>')):
        <root>/sar/images/      VV, VH
        <root>/sar/labels/      0 non-flood, 1 flood, 2 inundated vegetation
        <root>/optical/images/
        <root>/optical/labels/
    """

    def __init__(
        self,
        root: str | Path,
        img_dir: str = "images",
        lbl_dir: str = "labels",
        binary: bool = True,
        sar_norm: bool = True,
        img_size: int = 224,
        augment: bool = False,
    ) -> None:
        super().__init__()
        root = Path(root)
        img_root = root / img_dir
        lbl_root = root / lbl_dir

        if not img_root.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_root}\n"
                f"Run inspect_deepflood_root('{root}') to check the actual layout."
            )

        self.binary   = binary
        self.sar_norm = sar_norm

        img_files = {p.stem: p for p in sorted(img_root.glob("*.tif"))}
        lbl_files = {p.stem: p for p in sorted(lbl_root.glob("*.tif"))}
        common = sorted(img_files.keys() & lbl_files.keys())

        if not common:
            raise RuntimeError(f"No matched image/label pairs in {img_root} / {lbl_root}")

        self.samples = [(img_files[k], lbl_files[k]) for k in common]

        ops = [A.Resize(img_size, img_size)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_path, lbl_path = self.samples[idx]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(lbl_path) as src:
            mask = src.read(1).astype(np.int64)

        if self.sar_norm:
            image = np.clip(image, _DF_SAR_DB_MIN, _DF_SAR_DB_MAX)
            image = (image - _DF_SAR_DB_MIN) / (_DF_SAR_DB_MAX - _DF_SAR_DB_MIN)
        else:
            image = np.clip(image * _DF_OPT_SCALE, 0.0, 1.0)

        if self.binary:
            mask = np.where(mask >= 1, 1, 0).astype(np.int64)
        mask = np.where(mask > 2, -1, mask).astype(np.int64)

        result = self.transform(image=image.transpose(1, 2, 0), mask=mask)
        return {
            "image":    result["image"].float(),
            "mask":     result["mask"].long(),
            "filename": str(img_path.name),
        }


def get_deepflood_dataloader(
    root: str | Path,
    modality: str = "sar",
    binary: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    img_dir: str = "images",
    lbl_dir: str = "labels",
) -> DataLoader:
    """Evaluation DataLoader for DeepFlood.

    modality: "sar" or "optical". binary: merge inundated vegetation into flood.
    """
    modality_root = Path(root) / modality
    ds = DeepFloodDataset(
        root=modality_root, img_dir=img_dir, lbl_dir=lbl_dir,
        binary=binary, sar_norm=(modality == "sar"),
        img_size=img_size, augment=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# UrbanSARFloods

_USF_INTENSITY_IDX = [0, 1, 2, 3, 6, 7]
_USF_COHERENCE_IDX = [4, 5]
_USF_S1_DB_MIN = -25.0
_USF_S1_DB_MAX = 5.0
_USF_LABEL_MAP = {"01_NF": 0, "02_FO": 1, "03_FU": 2}


class UrbanSARFloodsDataset(Dataset):
    """UrbanSARFloods, Sentinel-1 SLC flood benchmark (18 events, 512x512 chips).

    Paper: https://arxiv.org/abs/2406.04111

    Structure: <root>/{01_NF,02_FO,03_FU}/{GT,SAR}/, Train_dataset.txt, Valid_dataset.txt.
    Image: 8 bands (VV/VH pre-event, co-event, coherence, post-event).
    Labels: 0 non-flood, 1 open flood, 2 urban flood; binary=True merges 1 and 2.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        binary: bool = True,
        img_size: int = 224,
        augment: bool = False,
    ) -> None:
        super().__init__()
        root = Path(root)
        self.binary = binary

        if split in ("train", "val"):
            txt = root / ("Train_dataset.txt" if split == "train" else "Valid_dataset.txt")
            if not txt.exists():
                raise FileNotFoundError(
                    f"Split file not found: {txt}\n"
                    f"Use split='all' to load all available chips."
                )
            names = set(txt.read_text().splitlines())
            self.samples: list[tuple[Path, Path, int]] = []
            for class_dir, label in _USF_LABEL_MAP.items():
                img_dir = root / class_dir / "SAR"
                lbl_dir = root / class_dir / "GT"
                for img_path in sorted(img_dir.glob("*.tif")):
                    if img_path.stem in names:
                        lbl_path = lbl_dir / img_path.name
                        if lbl_path.exists():
                            self.samples.append((img_path, lbl_path, label))
        else:
            self.samples = []
            for class_dir, label in _USF_LABEL_MAP.items():
                img_dir = root / class_dir / "SAR"
                lbl_dir = root / class_dir / "GT"
                if not img_dir.exists():
                    continue
                for img_path in sorted(img_dir.glob("*.tif")):
                    lbl_path = lbl_dir / img_path.name
                    if lbl_path.exists():
                        self.samples.append((img_path, lbl_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {root}.\n"
                f"Expected subdirectories: 01_NF/, 02_FO/, 03_FU/ with SAR/ and GT/ inside."
            )

        ops = [A.Resize(img_size, img_size)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_path, lbl_path, _class_label = self.samples[idx]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)   # (8, H, W)
        with rasterio.open(lbl_path) as src:
            mask = src.read(1).astype(np.int64)

        norm = image.copy()
        norm[_USF_INTENSITY_IDX] = (
            np.clip(image[_USF_INTENSITY_IDX], _USF_S1_DB_MIN, _USF_S1_DB_MAX)
            - _USF_S1_DB_MIN
        ) / (_USF_S1_DB_MAX - _USF_S1_DB_MIN)
        norm[_USF_COHERENCE_IDX] = np.clip(image[_USF_COHERENCE_IDX], 0.0, 1.0)
        image = norm

        if self.binary:
            mask = np.where(mask >= 1, 1, 0).astype(np.int64)
        mask = np.where(mask > 2, -1, mask).astype(np.int64)

        result = self.transform(image=image.transpose(1, 2, 0), mask=mask)
        return {
            "image":    result["image"].float(),
            "mask":     result["mask"].long(),
            "filename": str(img_path.name),
        }


def get_urban_sar_floods_dataloader(
    root: str | Path,
    split: str = "val",
    binary: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
) -> DataLoader:
    """Evaluation DataLoader for UrbanSARFloods; apply channel_adapter.urban_sar_adapter first."""
    ds = UrbanSARFloodsDataset(
        root=root, split=split, binary=binary, img_size=img_size, augment=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# S1S2-Water

# Structure (Zenodo 11278238):
# <root>/<N>/sentinel12_s2_<N>_img.tif 6 bands (pre-selected B2,B3,B4,B8A,B11,B12)
# <root>/<N>/sentinel12_s2_<N>_msk.tif 1 band: 0=non-water, 1=water
# <root>/<N>/sentinel12_s2_<N>_valid.tif 1 band: 0=nodata, 1=valid
# <root>/<N>/sentinel12_s1_<N>_img.tif 2 bands (VV, VH)
# ...
# S2 images already hold the six model bands; full 10980x10980 tiles are resized at load
# time. L1C data, so the S11 statistics apply.


class S1S2WaterDataset(Dataset):
    """S1S2-Water Sentinel-2 L1C water segmentation (93 scenes, IEEE TGRS 2023).

    Zenodo: https://zenodo.org/records/11278238

    Structure: <root>/<N>/sentinel12_s2_<N>_{img,msk,valid}.tif
    Bands B2,B3,B4,B8A,B11,B12. Labels 0 non-water, 1 water; invalid pixels -> -1.
    """

    def __init__(self, root: str | Path, img_size: int = 512, augment: bool = False) -> None:
        super().__init__()
        root = Path(root)
        data_dir = root / "S1S2Water" if (root / "S1S2Water").is_dir() else root
        if not data_dir.exists():
            raise FileNotFoundError(f"S1S2-Water not found: {data_dir}")

        self._img_size = img_size
        self.samples: list[tuple[Path, Path, Path | None]] = []
        for scene_dir in sorted(data_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            n = scene_dir.name
            img_p = scene_dir / f"sentinel12_s2_{n}_img.tif"
            msk_p = scene_dir / f"sentinel12_s2_{n}_msk.tif"
            vld_p = scene_dir / f"sentinel12_s2_{n}_valid.tif"
            if img_p.exists() and msk_p.exists():
                self.samples.append((img_p, msk_p, vld_p if vld_p.exists() else None))

        if not self.samples:
            raise RuntimeError(f"No S1S2-Water scenes found in {data_dir}")

        # No Resize needed - rasterio reads directly at target resolution
        ops = [A.Normalize(mean=list(NORM_MEAN), std=list(NORM_STD), max_pixel_value=1.0)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_path, lbl_path, vld_path = self.samples[idx]
        # full tiles are 10980x10980, read at target resolution instead of loading 725MB
        sz = self._img_size
        with rasterio.open(img_path) as src:
            image = src.read(
                out_shape=(src.count, sz, sz),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)                           # (6, sz, sz)
        with rasterio.open(lbl_path) as src:
            mask = src.read(
                1, out_shape=(sz, sz),
                resampling=rasterio.enums.Resampling.nearest,
            ).astype(np.int64)

        image = np.clip(image * 0.0001, 0.0, 1.0)

        # Mark invalid pixels (swath edges) as ignore
        if vld_path is not None:
            with rasterio.open(vld_path) as src:
                valid = src.read(
                    1, out_shape=(sz, sz),
                    resampling=rasterio.enums.Resampling.nearest,
                )
            mask = np.where(valid == 0, -1, mask)

        result = self.transform(image=image.transpose(1, 2, 0), mask=mask)
        return {
            "image":    result["image"].float(),
            "mask":     result["mask"].long(),
            "filename": img_path.name,
        }


def get_s1s2water_dataloader(
    root: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 512,
    shuffle: bool = False,
) -> DataLoader:
    """Evaluation DataLoader for S1S2-Water (L1C, S11 normalization, 6 bands pre-selected)."""
    ds = S1S2WaterDataset(root=root, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# SenForFlood

# Structure (SenForFloods-*.tar.gz):
# <root>/CEMS/<EVENT_ID>/<flood_event>/
# s2_during_flood/<id>_s2_during_flood.tif 8 bands: B2,B3,B4,B8A,B11,B12 + QA + fill
# flood_mask/<id>_flood_mask.tif 0=no-flood, 1=flood, 2=uncertain (CEMS)
# <root>/DFO/<country>/<event>/
# s2_during_flood/<id>_s2_during_flood.tif same 8-band format
# flood_mask/<id>_flood_mask.tif 0=no-flood, 1=flood

# Bands 0-5: B2, B3, B4, B8A, B11, B12; band 6: QA/cloud score; band 7: constant fill.
# Patches are 512x512, L1C.


def _suppress_gdal_warnings(worker_id: int = 0) -> None:
    """Silence GDAL warnings about the photometric tag of SenForFlood TIFFs; data reads correctly."""
    import logging
    import os
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    os.environ.setdefault("CPL_LOG_ERRORS", "OFF")


# Suppress in main process too (DataLoader's own iter calls happen here)
_suppress_gdal_warnings()


class SenForFloodDataset(Dataset):
    """SenForFlood Sentinel-2 L1C flood segmentation (about 25k 512x512 patches, 369 events).

    Extract:
        mkdir -p $DATA_ROOT/SenForFlood
        for f in $DATA_ROOT/SenForFloods-*.tar.gz; do tar -xzf "$f" -C $DATA_ROOT/SenForFlood/; done

    Structure:
        <root>/CEMS/<EVENT>/<flood_event>/{s2_during_flood,flood_mask}/*.tif
        <root>/DFO/<country>/<event>/{s2_during_flood,flood_mask}/*.tif

    Bands 0-5 = B2,B3,B4,B8A,B11,B12, band 6 QA, band 7 fill. CEMS label 2 (uncertain) -> -1.
    """

    def __init__(
        self,
        root: str | Path,
        img_size: int = 512,
        augment: bool = False,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        root = Path(root)
        data_dir = root / "SenForFlood" if (root / "SenForFlood").is_dir() else root
        if not data_dir.exists():
            raise FileNotFoundError(
                f"SenForFlood not found: {data_dir}\n"
                f"Extract SenForFloods-*.tar.gz archives into {data_dir}"
            )

        # Collect all (s2_img, flood_mask) pairs recursively
        all_samples: list[tuple[Path, Path]] = []
        for img_p in sorted(data_dir.rglob("s2_during_flood/*_s2_during_flood.tif")):
            stem = img_p.name.replace("_s2_during_flood.tif", "")
            msk_p = img_p.parent.parent / "flood_mask" / f"{stem}_flood_mask.tif"
            if msk_p.exists():
                all_samples.append((img_p, msk_p))

        if not all_samples:
            raise RuntimeError(f"No SenForFlood patches found in {data_dir}")

        # Optional deterministic subsample for faster OOD evaluation
        if max_samples is not None and max_samples < len(all_samples):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(all_samples), size=max_samples, replace=False)
            idx.sort()
            self.samples = [all_samples[i] for i in idx]
        else:
            self.samples = all_samples

        ops = [A.Resize(img_size, img_size),
               A.Normalize(mean=list(NORM_MEAN), std=list(NORM_STD), max_pixel_value=1.0)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        img_path, lbl_path = self.samples[idx]
        with rasterio.open(img_path) as src:
            all_bands = src.read().astype(np.float32)   # (8, H, W)
        with rasterio.open(lbl_path) as src:
            mask = src.read(1).astype(np.int64)

        image = np.clip(all_bands[:6] * 0.0001, 0.0, 1.0)   # bands 0-5: B2,B3,B4,B8A,B11,B12

        # 0=background, 1=flood, CEMS 2 (uncertain) becomes -1 (ignore)
        mapped = np.where(mask > 1, -1, mask).astype(np.int64)

        result = self.transform(image=image.transpose(1, 2, 0), mask=mapped)
        return {
            "image":    result["image"].float(),
            "mask":     result["mask"].long(),
            "filename": img_path.name,
        }


def get_senforflood_dataloader(
    root: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 512,
    shuffle: bool = False,
    max_samples: int | None = None,
) -> DataLoader:
    """Evaluation DataLoader for SenForFlood (L1C, S11 normalisation, bands 0-5).

    max_samples: fixed random subset (seed 0), e.g. 2000 for faster evaluation.
    """
    ds = SenForFloodDataset(root=root, img_size=img_size, max_samples=max_samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False,
                      worker_init_fn=_suppress_gdal_warnings)
