"""Pseudo-pool features and spectral corroboration, written to pool_features.csv.

rho_water is the IoU of teacher mask and MNDWI > 0. rho_new compares the mask with
event MNDWI water that is not in the paired baseline. Neither reference uses the
teacher prediction, but they are not flood ground truth either. The stored
normalization is undone first (rho = x*sigma + mu), then the indices are computed.

Band statistics, boundary density and patch_quality.csv metadata are also stored.
The quality join marks event-window acquisitions and paired scenes give the
pre-event reference. Without provenance that interpretation is limited.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.sen1floods11 import NORM_MEAN, NORM_STD
from utils.paths import dataset_root

# Stored band order [B2, B3, B4, B8A, B11, B12]; MNDWI needs green and SWIR-1.
_B3, _B11 = 1, 4

FIELDS = [
    "file", "location", "scene", "is_flood_scene",
    "flood_frac", "mean_conf", "conf_std", "boundary_density",
    "mndwi_water_frac", "rho_water", "rho_new",
    *[f"band_mean_{i}" for i in range(6)],
    *[f"band_std_{i}" for i in range(6)],
    "patch_cloud_prob", "new_water_frac", "permanent_water_frac",
    "candidates_tried", "lon", "lat",
]


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """Invert the fixed S11 normalization to reflectance (L1C for GDACS)."""
    mu = torch.tensor(NORM_MEAN, dtype=torch.float32).view(-1, 1, 1)
    sd = torch.tensor(NORM_STD, dtype=torch.float32).view(-1, 1, 1)
    return x.float() * sd + mu


def _mndwi_water(x: torch.Tensor) -> np.ndarray:
    """Boolean MNDWI water mask. MNDWI = (green - swir1)/(green + swir1) > 0."""
    rho = _denorm(x)
    g, s = rho[_B3], rho[_B11]
    mndwi = (g - s) / (g + s).clamp_min(1e-6)
    return (mndwi > 0.0).numpy()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks. Both empty -> 1.0 (they agree there is no water)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def _boundary_density(mask: np.ndarray) -> float:
    """Fraction of pixels adjacent to a mask boundary."""
    if mask.ndim != 2:
        return 0.0
    v = mask[1:, :] != mask[:-1, :]
    h = mask[:, 1:] != mask[:, :-1]
    return float(v.sum() + h.sum()) / float(mask.size)


def _load_quality(path: Path) -> dict:
    """(location, scene) -> measured per-patch physical quality, for flood-date scenes."""
    out = {}
    if not path.exists():
        print(f"  [features] no quality CSV at {path} — physical columns will be empty")
        return out
    for r in csv.DictReader(open(path)):
        loc = f"{r['event_id']}_{int(r['patch_idx']):04d}"
        out[(loc, r["scene_system_index"])] = r
    return out


def build(pseudo_dir: str, quality_csv: str, out_name: str = "pool_features.csv") -> None:
    base = Path(pseudo_dir)
    manifest_path = base / "source_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. rho_new needs per-tile provenance to pair each "
            f"flood tile with its pre-event baseline. Regenerate the pool with the current "
            f"data/pseudo_labeled.py, which writes the manifest."
        )
    manifest = list(csv.DictReader(open(manifest_path)))
    quality = _load_quality(Path(quality_csv).expanduser())
    print(f"[features] {len(manifest)} tiles | {len(quality)} quality rows")

    # pass 1: per-tile stats + cached masks (packed)
    recs: dict[str, dict] = {}
    by_loc: dict[str, list[str]] = {}
    for row in tqdm(manifest, desc="scanning", unit="tile"):
        fn = row["file"]
        item = torch.load(base / fn, map_location="cpu", weights_only=True)
        mask = item["mask"].numpy().astype(bool)
        img = item["image"]

        if "confidence" in item:
            conf = item["confidence"].float()
            mean_conf, conf_std = conf.mean().item(), conf.std().item()
        else:
            mean_conf, conf_std = 1.0, 0.0

        water = _mndwi_water(img)
        loc, scene = row["location"], row["scene_system_index"]
        recs[fn] = {
            "file": fn, "location": loc, "scene": scene,
            "is_flood_scene": int((loc, scene) in quality),
            "flood_frac": mask.mean(),
            "mean_conf": mean_conf, "conf_std": conf_std,
            "boundary_density": _boundary_density(mask),
            "mndwi_water_frac": float(water.mean()),
            "rho_water": _iou(mask, water),
            "band_mean": img.float().mean(dim=(1, 2)).numpy(),
            "band_std": img.float().std(dim=(1, 2)).numpy(),
            # packed so caching every mask costs ~150 MB rather than ~2.4 GB
            "_mask": np.packbits(mask), "_water": np.packbits(water),
            "_shape": mask.shape,
        }
        by_loc.setdefault(loc, []).append(fn)

    # Pass 2: rho_new. New water = event MNDWI water that is not water in the baseline.

    # Baselines have no new water by definition, so rho_new = IoU(teacher mask, empty):
    # 1.0 if the teacher predicts no flood, 0.0 otherwise.
    n_flood = n_base = 0
    for loc, files in by_loc.items():
        flood = [f for f in files if recs[f]["is_flood_scene"]]
        basel = [f for f in files if not recs[f]["is_flood_scene"]]
        shp = recs[files[0]]["_shape"]
        npx = shp[0] * shp[1]

        def _unpack(key, fn):
            return np.unpackbits(recs[fn][key], count=npx).reshape(shp).astype(bool)

        for f in basel:
            recs[f]["rho_new"] = _iou(_unpack("_mask", f), np.zeros(shp, bool))
            n_base += 1
        if not flood or not basel:
            continue                       # no pre-event reference -> rho_new stays absent
        perm = _unpack("_water", basel[0])
        for f in flood:
            recs[f]["rho_new"] = _iou(_unpack("_mask", f), _unpack("_water", f) & ~perm)
            n_flood += 1
    n_missing = sum(1 for r in recs.values() if "rho_new" not in r)
    print(f"[features] rho_new: {n_flood} flood tiles (vs paired baseline), "
          f"{n_base} baselines (vs empty), {n_missing} unpaired flood tiles left unmeasured")

    out = base / out_name
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for row in manifest:
            r = recs[row["file"]]
            q = quality.get((r["location"], r["scene"]), {})
            rec = {
                "file": r["file"], "location": r["location"], "scene": r["scene"],
                "is_flood_scene": r["is_flood_scene"],
                "flood_frac": f"{r['flood_frac']:.6f}",
                "mean_conf": f"{r['mean_conf']:.6f}",
                "conf_std": f"{r['conf_std']:.6f}",
                "boundary_density": f"{r['boundary_density']:.6f}",
                "mndwi_water_frac": f"{r['mndwi_water_frac']:.6f}",
                "rho_water": f"{r['rho_water']:.6f}",
                # empty for unpaired tiles (0 would mean full disagreement)
                "rho_new": (f"{r['rho_new']:.6f}" if "rho_new" in r else ""),
                "patch_cloud_prob": q.get("patch_cloud_prob", ""),
                "new_water_frac": q.get("new_water_frac", ""),
                "permanent_water_frac": q.get("permanent_water_frac", ""),
                "candidates_tried": q.get("candidates_tried", ""),
                "lon": q.get("lon", ""), "lat": q.get("lat", ""),
            }
            for i in range(6):
                rec[f"band_mean_{i}"] = f"{r['band_mean'][i]:.6f}"
                rec[f"band_std_{i}"] = f"{r['band_std'][i]:.6f}"
            w.writerow(rec)
    print(f"[features] wrote {out}")

    # decorrelation gate: rho is only useful if it tells us something confidence doesn't
    mc = np.array([r["mean_conf"] for r in recs.values()])
    rw = np.array([r["rho_water"] for r in recs.values()])
    rn = np.array([r.get("rho_new", np.nan) for r in recs.values()])
    ok = ~np.isnan(rn)
    print(f"\n[gate] corr(mean_conf, rho_water) = {np.corrcoef(mc, rw)[0,1]:+.3f}")
    print(f"[gate] corr(mean_conf, rho_new)   = {np.corrcoef(mc[ok], rn[ok])[0,1]:+.3f}")
    print(f"[gate] corr(rho_water, rho_new)   = {np.corrcoef(rw[ok], rn[ok])[0,1]:+.3f}")
    print(f"[gate] rho_water  mean {rw.mean():.3f}  p10 {np.percentile(rw,10):.3f}  "
          f"p90 {np.percentile(rw,90):.3f}")
    print(f"[gate] rho_new    mean {rn[ok].mean():.3f}  p10 {np.percentile(rn[ok],10):.3f}  "
          f"p90 {np.percentile(rn[ok],90):.3f}")
    print("[gate] |corr| well below ~0.5 means rho carries signal confidence does not.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    p.add_argument("--quality_csv",
                   default=str(dataset_root("UNLABELED_DIR") / "patch_quality.csv"))
    p.add_argument("--out_name", default="pool_features.csv")
    a = p.parse_args()
    build(a.pseudo_dir, a.quality_csv, a.out_name)


if __name__ == "__main__":
    main()
