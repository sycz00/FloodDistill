# !/usr/bin/env python3
"""Collect STURM evaluation-scale controls at 128, 224 and 512 pixels.

The 1.28 km footprint stays fixed. Effective sampling and feature-map size
change together; this is not an isolated GSD intervention. 224 is a non-integer
resize of the native 128 grid. --block margin compares the N=252 training
sources and crop control; --block deployed compares N=2500 models and references.
Separate-size caches are produced by experiments/perscene_controls.sh.
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np

import evaluation.stats_scene_level as S
from evaluation.perscene_collect import load_all
from evaluation.perscene_stats import PERSCENE_ROOT, sign_p, wilc_p

SIZES = (128, 224, 512)
GSD = {128: 10.00, 224: 5.71, 512: 2.50}       # 1.280 km extent / tensor side
SEEDS = ["42", "64", "91", "107"]
# The 512 px column is assembled from the paper-protocol caches only.
CACHE_512 = [str(PERSCENE_ROOT / d) for d in ("perscene", "perscene_B1", "perscene_C1")]
SIZE_CACHE = {sz: str(PERSCENE_ROOT / f"perscene_sturmsize/{sz}") for sz in (128, 224)}

# --block margin: (objective, uncropped S11, cropped S11, pseudo) at the matched N=252 budget.
CELLS = [("CE",  "ncgt_evit_ce_N252",  "b1crop_evit_ce_N252",  "evit_ce_ps_N252"),
         ("LKD", "ncgt_evit_lkd_N252", "b1crop_evit_lkd_N252", "evit_lkd_ps_N252"),
         ("OFA", "ncgt_evit_ofa_N252", "b1crop_evit_ofa_N252", "evit_ofa_ps_N252")]
ARMS = (("S11 uncropped", 1), ("S11 cropped 264", 2), ("pseudo", 3))

# --block deployed: display name, family, seed keys.
ROWS = [("MNDWI > 0 (2-band rule)",     "index_mndwi_zero",     ["-"]),
        ("Teacher Prithvi-EO-2.0 300M", "teacher",              ["-"]),
        ("pseudo-EvIT-CE   N=2500",     "evit_ce_ps_N2500",  SEEDS),
        ("pseudo-EvIT-LKD  N=2500",     "evit_lkd_ps_N2500", SEEDS),
        ("pseudo-EvIT-OFA  N=2500",     "evit_ofa_ps_N2500", SEEDS),
        ("pseudo-UNet-CE   N=2500",     "unet_ce_ps_N2500",  SEEDS),
        ("pseudo-UNet-LKD  N=2500",     "unet_lkd_ps_N2500", SEEDS)]
METRICS = [("micro_mIoU", "micro mIoU"), ("micro_flood", "micro flood IoU"),
           ("macro_scene_mIoU", "macro mIoU"), ("macro_scene_flood", "macro flood IoU")]


def agg(rec, fam, seeds, metric):
    v = [rec[("sturm", fam, s, "clean")][metric] for s in seeds
         if ("sturm", fam, s, "clean") in rec]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def paired(A, B):
    ks = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in ks])
    return d, int((d > 0).sum()), len(d)


def block_margin(rec, R, rows):
    means = {}
    print(f"{'='*104}\nSTURM clean micro mIoU vs EVALUATION TENSOR SIZE  (same checkpoints, "
          f"4 seeds)\n{'='*104}")
    print(f"  {'objective':10s} {'arm':16s} " +
          "".join(f"{f'{sz} px ({GSD[sz]:.2f} m)':>20s}" for sz in SIZES))
    for name, *fams in CELLS:
        for arm, k in ARMS:
            fam = fams[k - 1]
            line = f"  {name:10s} {arm:16s} "
            for sz in SIZES:
                m, sd, n = agg(rec[sz], fam, SEEDS, "micro_mIoU")
                if n:
                    means[(name, arm, sz)] = m
                    line += f"{m:13.4f}+-{sd:.3f}"
                    rows.append([name, arm, fam, sz, GSD[sz], "micro_mIoU_clean",
                                 f"{m:.5f}", f"{sd:.5f}", n])
                else:
                    line += f"{'--':>20s}"
            print(line)

    print(f"\n{'='*104}\nTHE MARGIN (pseudo minus Sen1Floods11) AT EACH EVALUATION SCALE"
          f"\n{'='*104}")
    print(f"  {'objective':10s} {'contrast':22s} " +
          "".join(f"{f'{sz} px':>14s}" for sz in SIZES))
    for name, *_ in CELLS:
        for ref in ("S11 uncropped", "S11 cropped 264"):
            line = f"  {name:10s} {'pseudo - ' + ref:22s} "
            for sz in SIZES:
                a_, b_ = means.get((name, "pseudo", sz)), means.get((name, ref, sz))
                line += f"{a_ - b_:+14.4f}" if (a_ is not None and b_ is not None) else f"{'--':>14s}"
                if a_ is not None and b_ is not None:
                    rows.append([name, f"margin pseudo-{ref}", "", sz, GSD[sz], "margin_micro",
                                 f"{a_-b_:.5f}", "", 4])
            print(line)

    print(f"\n{'='*104}\nPAIRED PER FLOOD MAP (n=46) AT NATIVE 128 px -- does the published "
          f"ordering survive?\n{'='*104}")
    S.R = R[128]
    print(f"  {'objective':10s} {'contrast':26s} {'delta':>9s} {'wins':>8s} {'p_wilc':>10s} "
          f"{'p_sign':>10s}")
    for name, gt, cr, ps in CELLS:
        for lab, A, B in (("pseudo - S11 uncropped", ps, gt), ("pseudo - S11 cropped", ps, cr)):
            Ag, _ = S.group_scores("sturm", A, "clean", "scene")
            Bg, _ = S.group_scores("sturm", B, "clean", "scene")
            if not (Ag and Bg):
                continue
            d, w, n = paired(Ag, Bg)
            print(f"  {name:10s} {lab:26s} {d.mean():+9.4f} {w:4d}/{n:<3d} "
                  f"{wilc_p(d):10.2e} {sign_p(w, n):10.2e}")
            rows.append([name, lab, f"{A} - {B}", 128, GSD[128], "paired_delta",
                         f"{d.mean():.5f}", f"wins={w}/{n}", n])


def block_deployed(rec, R, rows):
    for metric, mlabel in METRICS:
        print(f"\n{'='*106}\nSTURM clean {mlabel} vs EVALUATION TENSOR  (identical checkpoints)"
              f"\n{'='*106}")
        print(f"  {'model':30s}" + "".join(f"{f'{s} px ({GSD[s]:.2f} m)':>22s}" for s in SIZES)
              + f"{'128 - 512':>14s}")
        for name, fam, seeds in ROWS:
            line, vals = f"  {name:30s}", {}
            for sz in SIZES:
                m, sd, n = agg(rec[sz], fam, seeds, metric)
                vals[sz] = m
                rows.append([metric, name, fam, sz, GSD[sz],
                             f"{m:.4f}" if n else "", f"{sd:.4f}" if n else "", n])
                line += (f"{m:15.4f}+-{sd:.3f}" if n > 1 else
                         (f"{m:22.4f}" if n == 1 else f"{'--':>22s}"))
            d = vals[128] - vals[512]
            print(line + (f"{d:+14.4f}" if np.isfinite(d) else f"{'--':>14s}"))

    print(f"\n{'='*106}\nPer-floodmap paired test against the TEACHER (flood IoU, clean, "
          f"46 floodmaps)\n{'='*106}")
    for sz in SIZES:
        print(f"\n  eval tensor {sz} px (effective GSD {GSD[sz]:.2f} m)")
        S.R = R[sz]
        tea, _ = S.group_scores("sturm", "teacher", "clean", "scene", which=1)
        for name, fam, seeds in ROWS:
            if fam == "teacher" or not tea:
                continue
            mod, _ = S.group_scores("sturm", fam, "clean", "scene", which=1)
            if not mod:
                print(f"    {name:30s} --"); continue
            d, w, n = paired(mod, tea)
            print(f"    {name:30s} mean {d.mean():+.4f}  ahead on {w:2d}/{n}  "
                  f"p={sign_p(w, n):.3g}")
            rows.append(["paired_vs_teacher", name, fam, sz, GSD[sz], f"{d.mean():.5f}",
                         f"wins={w}/{n}", f"sign_p={sign_p(w, n):.3e}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="margin", choices=["margin", "deployed"])
    ap.add_argument("--out", default=None,
                    help="CSV path; default <cache root>/perscene_sturmsize/"
                         "<block>_scale_control.csv")
    a = ap.parse_args()
    out = Path(a.out or PERSCENE_ROOT / f"perscene_sturmsize/{a.block}_scale_control.csv")

    # One dict PER SCALE. The cell filename does not record --sturm_img_size, so merging two
    # scales into one dict overwrites cells (last writer wins) without an error.
    R = {512: {}}
    for c in CACHE_512:
        R[512].update(S.load(c))
    for sz in (224, 128):
        R[sz] = S.load(SIZE_CACHE[sz])
    rec = {sz: load_all(SIZE_CACHE[sz]) if sz != 512 else
           {k: v for c in CACHE_512 for k, v in load_all(c).items()} for sz in SIZES}
    print(f"[cache] cells per scale: " + ", ".join(f"{sz}px {len(R[sz])}" for sz in SIZES))

    rows = [["row", "arm_or_model", "family", "eval_tensor", "effective_gsd_m", "metric",
             "mean", "std_or_wins", "n"]]
    (block_margin if a.block == "margin" else block_deployed)(rec, R, rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
