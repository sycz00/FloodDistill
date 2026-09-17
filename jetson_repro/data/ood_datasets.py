"""Out-of-distribution flood segmentation datasets.

    WorldFloodsDataset      16 Sentinel-2 L1C scenes (FloodsNet subset)
    FloodPlanetDataset      298 S2 patches, 19 flood events worldwide (2017-2020)
    STURMFloodDataset       60 events, Sentinel-1 and Sentinel-2, 128x128 chips
    DeepFloodDataset        4,831 SAR/optical chips from 11 flood events
    UrbanSARFloodsDataset   807,500 km2, 18 events, Sentinel-1 SLC 8-band chips
    S1S2WaterDataset        65 hand-labeled S2 L1C patches, 12 flood events (IEEE TGRS 2023)
    SenForFloodDataset      93+ flood events worldwide, Sentinel-2 L1C (ISPRS 2025)

Loaders: get_worldfloods_dataloader, get_floodplanet_dataloader,
get_sturm_flood_dataloader, get_deepflood_dataloader, get_urban_sar_floods_dataloader,
get_s1s2water_dataloader, get_senforflood_dataloader.
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
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data.sen1floods11 import NORM_MEAN, NORM_STD


# Haze robustness perturbation (atmospheric path radiance). Physical model from Makarau et al.,
# "Haze Detection and Removal in Remotely Sensed Multispectral Imagery", IEEE TGARS 2014
# (Chavez dark-object subtraction, Eq. 1-2): haze adds an additive path radiance per band,
#     reflectance'_i = reflectance_i + HTM_i,   HTM_i = h * w_i
# The additive term depends on the band and drops with wavelength (Angstrom law,
# w ~ (lambda_ref/lambda)^alpha, alpha ~ 1 for aerosol haze): strong in blue, almost
# transparent in SWIR, which is why dark-object / SWIR dehazing works. We sweep the
# blue-band strength h (reflectance units), applied in reflectance space before the model's
# fixed normalization (same hook as sim_gain). The training augmentation is a per-channel
# multiplicative jitter (0.8-1.2), while haze is additive and spectrally structured, so it
# really is out of distribution for training.
def _haze_weights(wavelengths_nm, ref_nm: float = 490.0, alpha: float = 1.0):
    return ((ref_nm / np.asarray(wavelengths_nm, np.float32)) ** alpha).astype(np.float32)

# STURM S2 9-band order [B2,B3,B4,B5,B6,B7,B8A,B11,B12] (central wavelengths, nm)
_STURM_S2_WL   = [490., 560., 665., 705., 740., 783., 865., 1610., 2190.]
_STURM_HAZE_W  = _haze_weights(_STURM_S2_WL)
# Model / WF 6-band order [B2,B3,B4,B8A,B11,B12]
_S2_6BAND_WL   = [490., 560., 665., 865., 1610., 2190.]
_HAZE_W_6      = _haze_weights(_S2_6BAND_WL)


# WorldFloods

# 13-band S2 order: [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12] (0-indexed).
# The teacher/student use B2,B3,B4,B8A,B11,B12, i.e. indices [1,2,3,8,11,12].
# Index 8 is B8A (NIR narrow, what the model was trained on), index 7 is B8 (broadband NIR).
# The earlier loaders used index 7, and that band mismatch alone can push land towards the
# water boundary. Fixed here to B8A.
# WorldFloodsv2 has 15 bands: the 13 L1C bands above (indices 0-12, same order, checked:
# index 10 is B10 cirrus with mean ~15 DN), band 13 a constant valid/mask layer and band 14
# the s2cloudless cloud probability (0-100). Both aux bands are ignored, the 6 model bands
# are all in 0-12.
_WF_BAND_INDICES = [1, 2, 3, 8, 11, 12]   # B2, B3, B4, B8A, B11, B12

# WorldFloodsv2 per-band reflectance stats for the 6 model bands (with B8A), measured on the
# valid land/water, cloud-free, non-nodata pixels of the test split (509-pair v2 release).
# The whole distribution sits uniformly ~0.4 sigma below the S11 L1C training means (from
# B2 -0.37 sigma to B8A -0.69 sigma) and the relative spectral structure is intact. That's a
# mild L1C to L1C shift, not a different space, so normalize="s11" (consistent with training)
# is the right default. normalize="dataset" (these stats, per-band whitening / TTA) is only a
# fallback if s11 clearly mis-predicts. The visible stds are small (~0.03-0.05), so "dataset"
# whitening boosts contrast a lot, which can help or hurt. That's why it isn't the default.
WF_NORM_MEAN = [0.1139, 0.1089, 0.0925, 0.2280, 0.1531, 0.0875]
WF_NORM_STD  = [0.0315, 0.0376, 0.0544, 0.1253, 0.1023, 0.0712]


def _wf_detect_layout(root: Path, split: str) -> tuple[str, Path]:
    """("v2", split_dir) for the 509-pair WorldFloodsv2 release, or ("v1", data_dir) for
    the legacy *_S2.tif / *_CLASS.tif Zenodo subset.

    v2 layout: <root>/{train,val,test}/{S2,gt,meta,...}/<scene>.tif (<root> may also
    point at a split dir already).
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

    Detects one of two layouts on disk.

    WorldFloodsv2 (preferred, 509 pairs, Portales-Julia et al. 2023):
        <root>/WorldFloodsv2/{train,val,test}/
            S2/<scene>.tif    15 bands: 13-band L1C (indices 0-12) + valid-mask (13)
                              + cloud-probability (14). int16 raw DN [0, ~10000].
            gt/<scene>.tif    2 bands: [0] cloud mask {0:invalid,1:clear,2:cloud},
                              [1] land/water {0:invalid,1:land,2:water}.
        Download:
            huggingface-cli download --repo-type dataset isp-uv-es/WorldFloodsv2 \\
                --local-dir ~/data/WorldFloodsv2

        Preprocessing (with this a Sen1Floods11 model does well, measured mIoU ~0.80 and
        water-IoU ~0.72 on the test split with an efficientvit_b0 OFA student):
          * bands B2,B3,B4,B8A,B11,B12 (indices 1,2,3,8,11,12) out of the 13 spectral bands,
            DN*1e-4 to reflectance, clipped to [0,1].
          * normalize="s11" (default) applies the model's Sen1Floods11 L1C training stats.
            This matters most: the v2 test bands are uniformly ~0.4 sigma below the training
            means (same L1C space), so s11 works. normalize="dataset" (per-band whitening)
            drops mIoU to ~0.70.
          * scenes are whole AOIs (up to ~10000x7600 px). Each one is cut into native
            resolution `tile_size` x `tile_size` windows (default 256, the paper's sub-image
            size), one sample per tile. Follows the paper and keeps the resolution, which is
            safer for scenes with thin 10 m water. On the WF test set, where the water bodies
            are large rivers/lakes, it scores about the same as resizing the whole scene to
            512 (0.803 vs 0.809), so tiling is a fine default. It does not fix the collapse.
          * labels: land/water band 1 -> 0 (land), 2 -> 1 (water), 0 -> ignore. Cloud pixels
            (cloud band == 2) -> ignore when exclude_clouds=True (default). The 2-class optical
            model can't see through cloud and the land/water GT under cloud is mostly marked
            invalid anyway (scoring clouds costs ~0.01 mIoU). S2 nodata (all spectral bands
            zero) -> ignore.
          * permanent water (ignore_permanent_water=True, default since 2026-07-29): pixels
            marked permanent (value 3) in the co-registered JRC yearly-history layer
            (<split>/PERMANENTWATERJRC/<scene>.tif, 0=nodata, 1=not water, 2=seasonal,
            3=permanent) -> ignore. That matches STURM, whose loader already ignores open
            water / reservoirs / lakes (classes 3-5), so both OOD benchmarks score flood only.
            ~35% of WF test GT-water pixels are JRC-permanent, so WF numbers change a lot;
            ignore_permanent_water=False reproduces results from before 2026-07-29.
            JRC seasonal water (value 2) is still scored, in the same way STURM keeps
            river-related features as flood (rivers swell during floods).
          * empty tiles (valid fraction <= min_valid_frac) are dropped at init.

        The old 3.2 GB v1 subset "collapsed" (mIoU ~0.32, water predicted on 85% of pixels).
        Its raw Copernicus *_CLASS.tif encoding was mapped wrong, so only ~133 k of ~130 M
        pixels were scorable and the metric was broken and nearly empty. The v2 2-channel GT
        is the actual fix.

    Legacy v1 (Zenodo 8298840, *_S2.tif / *_CLASS.tif subset):
        <root>/world_floods/  *_S2.tif (13 bands, raw DN) + *_CLASS.tif (uint8).
        CLASS encoding (Copernicus EMS): 1 -> 0 (land), 2 -> 1 (water), anything else ignored.
        Kept so previously reported numbers can be reproduced.
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
        ignore_permanent_water: bool = True,
    ) -> None:
        super().__init__()
        if normalize not in ("s11", "dataset"):
            raise ValueError(f"normalize must be 's11' or 'dataset', got '{normalize}'")
        self.normalize = normalize
        self.tile_size = tile_size
        self.exclude_clouds = exclude_clouds
        # Flood-only protocol (STURM-aligned): mask JRC permanent water to ignore.
        self.ignore_permanent_water = ignore_permanent_water
        # E5 radiometric perturbation, reflectance' = sim_gain*reflectance + sim_offset, in
        # reflectance space before the fixed S11 normalization. Same as the STURM loader, so
        # the WF robustness sweep is measured on the same physical axis.
        self.sim_gain = float(sim_gain)
        self.sim_offset = float(sim_offset)
        # Haze robustness perturbation (additive per-band path radiance, wavelength-weighted;
        # see _haze_weights). 0.0 = off. Applied in reflectance space, same hook as sim_gain.
        self.sim_haze = float(sim_haze)

        self.version, data_dir = _wf_detect_layout(Path(root), split)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"WorldFloods data directory not found: {data_dir}\n"
                f"v2: huggingface-cli download --repo-type dataset "
                f"isp-uv-es/WorldFloodsv2 --local-dir ~/data/WorldFloodsv2\n"
                f"v1: download world_floods.7z from https://zenodo.org/records/8298840"
            )

        if self.version == "v2":
            self._init_v2(data_dir, min_valid_frac)
        else:
            self._init_v1(data_dir)

        # "s11": the model's Sen1Floods11 L1C training stats, consistent with deployment (no
        # peeking at the WorldFloods distribution). "dataset": WorldFloodsv2 whitening, only
        # as a test-time adaptation fallback.
        _m = list(NORM_MEAN) if normalize == "s11" else WF_NORM_MEAN
        _s = list(NORM_STD)  if normalize == "s11" else WF_NORM_STD
        ops = [A.Resize(img_size, img_size),
               A.Normalize(mean=_m, std=_s, max_pixel_value=1.0)]
        if augment:
            ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
        ops.append(ToTensorV2())
        self.transform = A.Compose(ops)

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

        # The flood-only protocol needs the co-registered JRC permanent-water layer. A missing
        # layer with the flag on is an error for scored scenes (skipping silently would mix
        # protocols across scenes). Turn it off with ignore_permanent_water=False.
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

        # Tile index: read each scene's GT once (2-band int16) and keep only windows with
        # scorable pixels, so empty swath-edge or all-cloud tiles never reach the model. With
        # the flood-only protocol JRC-permanent pixels aren't scorable, so tiles that are all
        # permanent water get dropped as well.
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
        # Nodata: all 13 spectral bands zero (swath edge). Only look at the spectral bands,
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
                mapped[src.read(1, window=win) == 3] = -1   # JRC permanent water
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
            # Same as STURMFloodDataset.__getitem__: perturb in reflectance space and don't
            # re-clip. Over- or under-gained reflectance can leave [0,1] and has to reach the
            # fixed S11 normalization like that. The old re-clip quietly weakened the g>1
            # bright-pixel perturbation, so the WF and STURM E5 axes were not the same and WF
            # looked less gain-sensitive than it is. Removed 2026-07-27 after an audit.
            image = image * self.sim_gain + self.sim_offset

        if self.sim_haze != 0.0:
            # additive wavelength-weighted haze (Makarau 2014); no clip, mirroring sim_gain
            image = image + self.sim_haze * _HAZE_W_6[:, None, None]

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
    ignore_permanent_water: bool = True,
) -> DataLoader:
    """Evaluation DataLoader for WorldFloods (detects v1 subset or v2 509-pair release).

    split: "train" | "val" | "test" (v2 only, ignored for the v1 subset).
    normalize: "s11" (default) uses the model's Sen1Floods11 L1C training stats, consistent
        with deployment. "dataset" is WorldFloodsv2 whitening (TTA).
    tile_size: v2 native window size (default 256 = paper sub-image size). Scenes are split
        into non-overlapping tiles at full 10 m resolution, like the paper. On the test set
        about the same as a whole-scene resize, and safer for scenes with thin water.
    exclude_clouds: v2, cloud pixels (cloud band == 2) are not scored (default True), which
        is right for a 2-class optical water model.
    min_valid_frac: v2, tiles with valid (scorable) fraction <= this are dropped (default
        0.01), so empty swath-edge / all-cloud tiles are skipped.
    ignore_permanent_water: v2, JRC permanent water (value 3) -> ignore, so WF is flood-only
        like STURM (default True since 2026-07-29). False reproduces the older
        flood + permanent-water protocol.
    """
    ds = WorldFloodsDataset(
        root=root, split=split, img_size=img_size, augment=False, normalize=normalize,
        tile_size=tile_size, exclude_clouds=exclude_clouds, min_valid_frac=min_valid_frac,
        sim_gain=sim_gain, sim_offset=sim_offset, sim_haze=sim_haze,
        ignore_permanent_water=ignore_permanent_water,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)




class FloodPlanetDataset(Dataset):
    """FloodPlanet Sentinel-2 flood segmentation.

    278 S2 patches from 18 flood events worldwide (2017-2020). US-Kansas is excluded by
    default, the dataset README warns about its label quality.

    Download:
        wget -O ~/data/FloodPlanet.zip \\
            "https://zenodo.org/api/records/15238572/files/FloodPlanet.zip/content"
        cd ~/data && unzip FloodPlanet.zip -d FloodPlanet/

    Layout:
        <root>/FloodPlanet/<event>/S2/*.tif      10-band, float32, raw DN
        <root>/FloodPlanet/<event>/labels/*.tif  uint8, values {1, 2}

    Labels: 1 -> 0 (land), 2 -> 1 (flood), nodata -> -1 (ignored).

    normalize="clip" (default): take B2, B3, B4, B8A, B11, B12 (indices 0,1,2,7,8,9), raw DN
    * 0.0001 to reflectance, clip to [0, 1], then the Sen1Floods11 (L1C) mean/std like at
    training time. This is the right one. The L2A vs L1C shift is ~0.5-0.9 sigma in NIR/SWIR
    but the relative spectral structure the model learned is kept.
    normalize="percentile": P2/P98 stretch, then S11 stats. Wrong for FP, the data is already
    reflectance, so stretching pushes band means +2-3 sigma above the training distribution.
    normalize="dataset": FP-specific L2A stats. Zero-mean inputs but the wrong space, band
    ratios differ from training and predictions get worse.
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

        # nodata pixels: swath edges, all 6 bands near zero
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
    """Evaluation DataLoader for FloodPlanet.

    normalize="clip" (default) applies the S11 L1C stats to reflectance, same as in training,
    best for inference. "percentile" is wrong for FP (band means end up +2-3 sigma).
    "dataset" uses the FP L2A stats, which is the wrong space for the model.
    """
    ds = FloodPlanetDataset(root=root, img_size=img_size, augment=False,
                             exclude_kansas=exclude_kansas, normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# STURM-Flood

# Per-band P2/P98 from 267 sampled STURM-Flood S2 tiles, bands [B2,B3,B4,B5,B6,B7,B8A,B11,B12].
# Used by the legacy default stretch (normalize="percentile"). Fitted empirically, still the
# default so already reported numbers reproduce. normalize="original" below is the version
# documented in the STURM paper.
_STURM_S2_P2  = np.array([-0.1725,  0.0050, -0.1000,  0.1465,  0.1570,
                            0.1660,  0.1583,  0.0032, -0.1161], dtype=np.float32)
_STURM_S2_P98 = np.array([ 1.6842,  2.1567,  2.2708,  2.3995,  3.0805,
                             3.6797,  3.7275,  2.9681,  2.3909], dtype=np.float32)

# normalize="original": STURM paper sec. 2.2(2) gives the exact forward transform used for the
# released chips, min-max scaling with a fixed DN range of 400-1600, the same for all
# Sentinel-2 channels. Inverting it recovers reflectance from the stored value v:
#     DN = 400 + 1200*v, reflectance = DN * 1e-4 = 0.04 + 0.12*v
# Measured through the legacy P2/P98 stretch, model inputs were +0.33 sigma (B2) up to
# +1.85 sigma (B11) above the S11 training distribution, although the appendix says the band
# means line up. This documented inverse doesn't depend on the data and fixes that
# (paper_todos.md M6/M7, ported from data/ood_datasets.py in the main
# Efficient_satelite_imagery repo).
_STURM_S2_DN_OFFSET = 0.04
_STURM_S2_DN_SCALE  = 0.12

_STURM_S2_NORM_MEAN = [NORM_MEAN[0], NORM_MEAN[1], NORM_MEAN[2],   # B2, B3, B4
                       0.1900, 0.2400, 0.2700,                       # B5, B6, B7 (dropped)
                       NORM_MEAN[3], NORM_MEAN[4], NORM_MEAN[5]]    # B8A, B11, B12
_STURM_S2_NORM_STD  = [NORM_STD[0],  NORM_STD[1],  NORM_STD[2],
                       0.0950, 0.1050, 0.1120,
                       NORM_STD[3],  NORM_STD[4],  NORM_STD[5]]


class STURMFloodDataset(Dataset):
    """STURM-Flood Sentinel-1 or Sentinel-2 tiles.

    60 flood events worldwide, 128x128 chips.

    Download: https://zenodo.org/records/12748983 (Dataset.zip, ~3.8 GB)

    Layout after extraction:
        <root>/Sentinel2/S2/          GeoTIFF 128x128, 9 bands, float64
        <root>/Sentinel2/Floodmaps/   GeoTIFF 128x128, labels 0-5
        <root>/Sentinel1/S1/          GeoTIFF 128x128, 2 bands (VV, VH)
        <root>/Sentinel1/Floodmaps/

    Labels (checked against the STURM paper): 0 = coastline/AOI background, 1 = flooded,
    2 = river-related features (not "possibly flooded", which this codebase used to claim),
    3 = open water, 4 = reservoirs, 5 = lakes, 99 = no-data. STURM's own binary remap counts
    1-5 as water. This loader keeps the original protocol instead, 2 -> flood and
    {3,4,5} -> ignore (~3.3% of pixels), since rivers swell during floods while permanent
    standing water (lakes/reservoirs) is left out of scoring. The parameter is still called
    `include_possible` for backward compatibility, even though class 2 isn't "possibly flooded".

    `normalize` selects the evaluation protocol:
      "percentile" (default): legacy P2/P98 stretch fitted on the data, still the default so
          already reported numbers stay reproducible.
      "original": the inverse documented in the STURM paper (0.04 + 0.12*v) plus a NaN guard
          for the ~0.6% of tiles with NaN pixels (paper_todos.md, M6/M7).
      "clip": no rescaling, just clip to [0,1] (legacy, rarely used).
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
    ) -> None:
        super().__init__()
        root = Path(root)
        if sensor not in ("S1", "S2"):
            raise ValueError(f"sensor must be 'S1' or 'S2', got '{sensor}'")
        if normalize not in ("percentile", "clip", "original"):
            raise ValueError(
                f"normalize must be 'percentile', 'clip', or 'original', got '{normalize}'"
            )

        self.sensor = sensor
        self.include_possible = include_possible
        self.normalize = normalize
        # E5 (input-normalization robustness): radiometric perturbation in reflectance space
        # before the model's fixed S11 normalization, reflectance' = sim_gain * reflectance +
        # sim_offset. Stands in for a real train/deploy mismatch (S2 calibration uncertainty
        # ~3-5%, processing-baseline BOA offset ~0.1) that can't be undone without per-product
        # metadata. (1.0, 0.0) leaves the loader output byte-identical.
        self.sim_gain = float(sim_gain)
        self.sim_offset = float(sim_offset)
        # Haze robustness perturbation (additive per-band path radiance, wavelength-weighted;
        # see _haze_weights / Makarau 2014). 0.0 = off. S2 only. Applied like sim_gain.
        self.sim_haze = float(sim_haze)

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

        if self.normalize == "original":
            # NaN guard (~0.6% of STURM tiles, ~0.004% of pixels): zero-fill so
            # they don't reach the model as literal NaNs. Only applied under the
            # "original" protocol to keep the legacy default byte-identical.
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

        if self.sensor == "S2" and self.sim_haze != 0.0:
            # additive wavelength-weighted haze (Makarau 2014), 9-band S2; no clip (mirrors gain)
            image = image + self.sim_haze * _STURM_HAZE_W[:, None, None]

        flood_val = 1 if self.include_possible else -1
        mask = np.where(mask == 0, 0,
               np.where(mask == 1, 1,
               np.where(mask == 2, flood_val, -1))).astype(np.int64)

        result = self.transform(image=image.transpose(1, 2, 0), mask=mask)
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
) -> DataLoader:
    """Evaluation DataLoader for STURM-Flood.

    normalize="percentile" (default): legacy stretch, still the default so the STURM numbers
        already reported reproduce without changes.
    normalize="original": inverse from the STURM paper (0.04 + 0.12*v) plus NaN guard. Use
        this for the corrected evaluation pass (paper_todos.md).
    normalize="clip": no rescaling, just clip to [0,1] (legacy).
    sim_gain / sim_offset: E5 perturbation reflectance' = sim_gain*reflectance + sim_offset,
        applied before the S11 normalization; (1.0, 0.0) = off.
    """
    ds = STURMFloodDataset(
        root=root, sensor=sensor, img_size=img_size,
        augment=False, include_possible=include_possible,
        normalize=normalize, sim_gain=sim_gain, sim_offset=sim_offset, sim_haze=sim_haze,
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
    """DeepFlood SAR or optical chips, 4,831 chips from 11 flood events.

    Paper: https://doi.org/10.1038/s41597-025-04554-3

    Layout, inferred from the paper (verify against the download):
        <root>/sar/images/       GeoTIFF, 2 bands: VV, VH
        <root>/sar/labels/       GeoTIFF, 0=non-flood, 1=flood, 2=inundated_veg
        <root>/optical/images/
        <root>/optical/labels/

    inspect_deepflood_root('<root>') shows the actual layout.
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

    modality ("sar" or "optical") picks the sub-directory and normalisation. binary merges
    inundated vegetation into the flood class (recommended).
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
    """UrbanSARFloods, flood benchmark based on Sentinel-1 SLC.

    807,500 km2, 18 flood events, 512x512 chips.
    Paper: https://arxiv.org/abs/2406.04111 (CVPR 2024 EarthVision)

    Layout:
        <root>/01_NF/{GT,SAR}/   (non-flood)
        <root>/02_FO/{GT,SAR}/   (open-area flood)
        <root>/03_FU/{GT,SAR}/   (urban flood)
        <root>/Train_dataset.txt, <root>/Valid_dataset.txt

    Image: 8 bands, VV/VH pre-event intensity, VV/VH co-event intensity, VV/VH coherence and
    VV/VH post-event intensity.
    Label: 0=non-flood, 1=FO, 2=FU. binary=True merges FO and FU into 1.
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
    """Evaluation DataLoader for UrbanSARFloods.

    Apply urban_sar_adapter() from channel_adapter.py before the model.
    """
    ds = UrbanSARFloodsDataset(
        root=root, split=split, binary=binary, img_size=img_size, augment=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# S1S2-Water
#
# Layout (Zenodo 11278238, catalog id "s1s2_water"), confirmed:
#   <root>/<N>/sentinel12_s2_<N>_img.tif    6 bands (pre-selected B2,B3,B4,B8A,B11,B12)
#   <root>/<N>/sentinel12_s2_<N>_msk.tif    1 band: 0=non-water, 1=water
#   <root>/<N>/sentinel12_s2_<N>_valid.tif  1 band: 0=nodata, 1=valid
#   <root>/<N>/sentinel12_s1_<N>_img.tif    2 bands (VV, VH)
#   ...
# The S2 images already contain the 6 bands matching Sen1Floods11 training. Full Sentinel-2
# tiles are 10980x10980 px and get resized to img_size at load time. L1C is confirmed (source
# IDs contain "MSIL1C"), so S11 NORM_MEAN/NORM_STD are used directly.


class S1S2WaterDataset(Dataset):
    """S1S2-Water Sentinel-2 L1C water segmentation.

    93 labeled full-tile scenes (flood and water bodies worldwide).
    Paper: "S1S2-Water: A global dataset for semantic segmentation of water
    bodies from Sentinel-1 and Sentinel-2 satellite images", IEEE TGRS 2023.
    Zenodo: https://zenodo.org/records/11278238

    Download:
        wget -O ~/data/files-archive "https://zenodo.org/api/records/11278238/files-archive"
        python3 /tmp/extract_s1s2water.py   # unpacks the nested part*.zip into ~/data/S1S2Water/

    Layout:
        <root>/<N>/sentinel12_s2_<N>_img.tif    (6 bands, L1C DN)
        <root>/<N>/sentinel12_s2_<N>_msk.tif    (0/1)
        <root>/<N>/sentinel12_s2_<N>_valid.tif  (0/1 validity)

    Bands are already [B2, B3, B4, B8A, B11, B12], same as Sen1Floods11.
    Label: 0=non-water, 1=water. Invalid pixels (valid.tif==0) get label -1.
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

        # no Resize needed, rasterio reads directly at the target resolution
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
#
# Layout (checked on partially extracted SenForFloods-*.tar.gz):
#   <root>/CEMS/<EVENT_ID>/<flood_event>/
#       s2_during_flood/<id>_s2_during_flood.tif   8 bands: B2,B3,B4,B8A,B11,B12 + QA + fill
#       flood_mask/<id>_flood_mask.tif              0=no-flood, 1=flood, 2=uncertain (CEMS)
#   <root>/DFO/<country>/<event>/
#       s2_during_flood/<id>_s2_during_flood.tif   same 8-band format
#       flood_mask/<id>_flood_mask.tif              0=no-flood, 1=flood
#
# Bands (checked empirically): 0-5 are B2, B3, B4, B8A, B11, B12 as in Sen1Floods11,
# 6 is a QA/cloud score (46 unique values, ~800 DN, variable), 7 is a constant fill
# (~7800 DN) and gets skipped. Patches are already 512x512, L1C (CEMS Copernicus source).


def _suppress_gdal_warnings(worker_id: int = 0) -> None:
    """Silence the harmless GDAL/libtiff metadata warnings that flood stderr.

    SenForFlood TIFFs have a PhotometricInterpretation mismatch (8 bands tagged as 3-channel
    RGB), and GDAL warns on every file open, ~50K lines per pass. Silenced at the Python
    logging level, the data itself is read correctly.
    """
    import logging
    import os
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    os.environ.setdefault("CPL_LOG_ERRORS", "OFF")


# Suppress in main process too (DataLoader's own iter calls happen here)
_suppress_gdal_warnings()


class SenForFloodDataset(Dataset):
    """SenForFlood Sentinel-2 L1C flood segmentation.

    ~25K 512x512 patches from 369 flood events worldwide (CEMS and DFO sources).
    Paper: ISPRS 2025. Comes as SenForFloods-NNN.tar.gz archives.

    Extract:
        mkdir -p ~/data/SenForFlood
        for f in ~/data/SenForFloods-*.tar.gz; do tar -xzf "$f" -C ~/data/SenForFlood/; done

    Structure:
        <root>/CEMS/<EVENT>/<flood_event>/s2_during_flood/*.tif  (8 bands)
                                          flood_mask/*.tif        (0/1/2)
        <root>/DFO/<country>/<event>/s2_during_flood/*.tif
                                     flood_mask/*.tif             (0/1)

    Bands 0-5 = B2,B3,B4,B8A,B11,B12 (same as training), band 6 = QA, band 7 = fill.
    CEMS label 2 (uncertain) -> -1, ignored in metrics. DFO has no uncertain class.
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
    """Evaluation DataLoader for SenForFlood (L1C, S11 normalization, bands 0-5).

    max_samples: deterministic subsample of that many patches (seed=0) for faster OOD
    evaluation, 2000 is enough for stable mIoU.
    """
    ds = SenForFloodDataset(root=root, img_size=img_size, max_samples=max_samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False,
                      worker_init_fn=_suppress_gdal_warnings)
