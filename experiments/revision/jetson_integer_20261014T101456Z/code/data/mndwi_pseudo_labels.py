# !/usr/bin/env python
"""Swap the teacher masks for MNDWI > 0 threshold masks on the same selected GDACS tiles.

Images, filenames and the index are not touched. Confidence is set to 1 everywhere
and teacher-specific targets are dropped. The tiles were still picked using the
teacher, so this is a labeller control and not a teacher-independent selection.

MNDWI from green (channel 1) and SWIR-1 (channel 4) after undoing the normalization,
denominator clamped at 1e-6. Agreement with pool_features and SpectralIndexPredictor
is checked before anything is written. CE only uses the new hard masks, OFA still
queries the live teacher. See docs/06_controls.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sen1floods11 import NORM_MEAN, NORM_STD          # noqa: E402

_B3, _B11 = 1, 4        # green, SWIR-1 in the 6-band Sen1Floods11 ordering
MU = torch.tensor(NORM_MEAN, dtype=torch.float32).view(-1, 1, 1)
SD = torch.tensor(NORM_STD, dtype=torch.float32).view(-1, 1, 1)


def mndwi_mask(img: torch.Tensor) -> torch.Tensor:
    """img: (6, H, W) normalised -> (H, W) int64 water mask, MNDWI > 0."""
    rho = img.float() * SD + MU
    g, s = rho[_B3], rho[_B11]
    idx = (g - s) / (g + s).clamp_min(1e-6)
    return (idx > 0.0).long()


def _verify_against_audited_implementations(img: torch.Tensor) -> None:
    """Assert bit-equality with the two MNDWI implementations already in the repo."""
    from data.pool_features import _mndwi_water
    from evaluation.perscene_eval import SpectralIndexPredictor

    mine = mndwi_mask(img).numpy().astype(bool)
    theirs = _mndwi_water(img)
    assert np.array_equal(mine, theirs), "disagrees with data.pool_features._mndwi_water"

    pred = SpectralIndexPredictor("mndwi_zero").eval()
    with torch.no_grad():
        ev = pred(img.unsqueeze(0).float()).argmax(1)[0].numpy().astype(bool)
    assert np.array_equal(mine, ev), "disagrees with perscene_eval.SpectralIndexPredictor"
    print("[verify] MNDWI rule matches pool_features._mndwi_water and "
          "perscene_eval.SpectralIndexPredictor on the probe tile")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    ap.add_argument("--index_file", default="diversity_N2500_index.json")
    ap.add_argument("--out_dir", default="data/pseudo_gdacs_mndwi")
    a = ap.parse_args()

    src, out = Path(a.pseudo_dir), Path(a.out_dir)
    if out.resolve() == src.resolve():
        raise SystemExit("refusing to overwrite the teacher-labelled pool")
    idx = json.load(open(src / a.index_file))
    files = idx["files"]
    print(f"[in ] {src}/{a.index_file}: {len(files)} tiles")
    out.mkdir(parents=True, exist_ok=True)

    probe = torch.load(src / files[0], map_location="cpu", weights_only=True)
    _verify_against_audited_implementations(probe["image"])

    ff_teacher, ff_mndwi, iou, agree = [], [], [], []
    for i, fn in enumerate(files):
        d = torch.load(src / fn, map_location="cpu", weights_only=True)
        img, told = d["image"], d["mask"]
        new = mndwi_mask(img)
        torch.save({"image": img,                      # bit-exact copy
                    "mask": new,
                    "source": d.get("source", ""),
                    # constant: a threshold rule has no per-pixel confidence, and this
                    # makes --pixel_conf_threshold a no-op instead of a teacher channel
                    "confidence": torch.ones(new.shape, dtype=torch.float16)},
                   out / fn)
        t, m = told.bool().numpy(), new.bool().numpy()
        ff_teacher.append(float(t.mean())); ff_mndwi.append(float(m.mean()))
        u = np.logical_or(t, m).sum()
        iou.append(1.0 if u == 0 else float(np.logical_and(t, m).sum()) / float(u))
        agree.append(float((t == m).mean()))
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(files)}] mean water-IoU(teacher, MNDWI)={np.mean(iou):.4f}")

    shutil.copy(src / a.index_file, out / a.index_file)

    stats = {
        "source_pool": str(src), "index_file": a.index_file, "n_tiles": len(files),
        "rule": "MNDWI = (rho_B3 - rho_B11) / (rho_B3 + rho_B11) > 0",
        "bands": {"green": _B3, "swir1": _B11},
        "mean_flood_frac_teacher": float(np.mean(ff_teacher)),
        "mean_flood_frac_mndwi": float(np.mean(ff_mndwi)),
        "mean_water_iou_teacher_vs_mndwi": float(np.mean(iou)),
        "median_water_iou_teacher_vs_mndwi": float(np.median(iou)),
        "mean_pixel_agreement": float(np.mean(agree)),
        "n_tiles_mndwi_empty": int(sum(f == 0.0 for f in ff_mndwi)),
        "n_tiles_mndwi_full": int(sum(f == 1.0 for f in ff_mndwi)),
        "confidence_field": "constant 1.0 (teacher confidence deliberately not carried over)",
        "caveat": ("tile SELECTION is still teacher-derived (FPS over teacher-mask features); "
                   "only the LABELS are model-free"),
    }
    json.dump(stats, open(out / "labelling_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
