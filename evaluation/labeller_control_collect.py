# !/usr/bin/env python3
"""Compare teacher and MNDWI hard labels on the same N=2500 GDACS images.

Reads teacher-labelled cells and runs/perscene_C4. Selection is still based
on teacher-derived features. CE tests the hard-label source without a live
teacher; OFA retains live-teacher soft supervision and only swaps hard targets.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import evaluation.stats_scene_level as S                           # noqa: E402
from evaluation.perscene_collect import load_all                   # noqa: E402
from evaluation.perscene_stats import PERSCENE_ROOT, iou, sign_p, wilc_p   # noqa: E402
from figures.fig_style import ENVELOPES                            # noqa: E402

SEEDS = ["42", "64", "91", "107"]
# (objective, teacher-labelled family, MNDWI-labelled family)
PAIRS = [("CE  (teacher-free)", "evit_ce_ps_N2500",  "c4mndwi_evit_ce_N2500"),
         ("OFA (teacher in loop)", "evit_ofa_ps_N2500", "c4mndwi_evit_ofa_N2500")]
REFS = [("MNDWI > 0 (the labeller)", "index_mndwi_zero", ["-"]),
        ("Teacher Prithvi 300M",     "teacher",          ["-"])]
DATASETS = [("sturm", "STURM-Flood"), ("wf", "WorldFloods-v2 flood-only")]
METRICS = [("micro_mIoU", "micro mIoU"), ("micro_flood", "micro flood IoU"),
           ("macro_scene_mIoU", "macro mIoU"), ("macro_scene_flood", "macro flood IoU")]


def agg(rec, ds, fam, seeds, metric):
    v = [rec[(ds, fam, s, "clean")][metric] for s in seeds if (ds, fam, s, "clean") in rec]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def wdrop(rec, ds, fam, seeds, env, metric="micro_mIoU"):
    """worst_drop, but with an explicit seed list.

        figures/fig_style.worst_drop takes the seeds from SEEDS_OF(fam), which only hard-codes ["-"]
        for the family literally named "teacher". The index families use seed "-" as well and
        would just return no data. The arithmetic is the same: clean minus envelope minimum per
        seed, then averaged, since min is non-linear.
    """
    vals = []
    for s in seeds:
        v = [rec.get((ds, fam, s, c), {}).get(metric, np.nan) for c in env]
        if any(x != x for x in v):
            continue
        vals.append(v[env.index("clean")] - min(v))
    if not vals:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def per_scene_flood(R, ds, fam, seeds):
    """scene -> flood IoU averaged over available seeds."""
    per: dict[str, list[float]] = {}
    for s in seeds:
        k = (ds, fam, s, "clean")
        if k in R:
            for g, c in R[k]["scene"].items():
                per.setdefault(g, []).append(iou(np.asarray(c, float))[1])
    return {g: float(np.mean(v)) for g, v in per.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PERSCENE_ROOT / "perscene_C4/labeller_control.csv"))
    a = ap.parse_args()

    R, rec = {}, {}
    for c in ("perscene", "perscene_C1", "perscene_C4"):
        c = str(PERSCENE_ROOT / c)
        R.update(S.load(c)); rec.update(load_all(c))
    # The index reference profile over the perturbation grid; the C1 cells are clean-only, so
    # without this the MNDWI row has no envelope. Same evaluator, same grid, same protocol.
    _ref = str(PERSCENE_ROOT / "perscene_T1_ref")
    R.update({k: v for k, v in S.load(_ref).items() if k[1].startswith("index_")})
    rec.update({k: v for k, v in load_all(_ref).items() if k[1].startswith("index_")})
    S.R = R
    have = {k[1] for k in R}
    print(f"[cache] {len(R)} cells | MNDWI-labelled families present: "
          f"{sorted(f for f in have if f.startswith('c4mndwi'))}")

    rows = [["section", "benchmark", "objective", "label_source", "family", "metric",
             "mean", "std", "n_seeds"]]

    # 1. clean accuracy
    for ds, dn in DATASETS:
        print(f"\n{'='*112}\nCLEAN ACCURACY — {dn}  (mean ± std over 4 seeds)\n{'='*112}")
        print(f"  {'model':34s}" + "".join(f"{m[1]:>19s}" for m in METRICS))
        for label, fam, seeds in REFS:
            line = f"  {label:34s}"
            for metric, _ in METRICS:
                m, sd, n = agg(rec, ds, fam, seeds, metric)
                line += f"{m:19.4f}" if n else f"{'--':>19s}"
                rows.append(["clean", ds, "-", label, fam, metric,
                             f"{m:.4f}" if n else "", "", n])
            print(line)
        for obj, fam_t, fam_m in PAIRS:
            for src, fam in (("Prithvi labels", fam_t), ("MNDWI labels", fam_m)):
                line = f"  {obj.split()[0]:4s} / {src:16s} N=2500      "[:36]
                for metric, _ in METRICS:
                    m, sd, n = agg(rec, ds, fam, SEEDS, metric)
                    line += f"{m:12.4f}±{sd:.3f}" if n else f"{'--':>19s}"
                    rows.append(["clean", ds, obj, src, fam, metric,
                                 f"{m:.4f}" if n else "", f"{sd:.4f}" if n else "", n])
                print(line)
            # the difference that is the whole point
            line = f"  {'  -> MNDWI - Prithvi':34s}"
            for metric, _ in METRICS:
                mt, _, nt = agg(rec, ds, fam_t, SEEDS, metric)
                mm, _, nm = agg(rec, ds, fam_m, SEEDS, metric)
                line += f"{mm - mt:+19.4f}" if (nt and nm) else f"{'--':>19s}"
            print(line + "\n")

    # 2. paired per-floodmap tests, MNDWI labels vs Prithvi labels
    print(f"\n{'='*112}\nPAIRED PER-FLOODMAP TEST — MNDWI labels vs Prithvi labels "
          f"(clean flood IoU)\n{'='*112}")
    for ds, dn in DATASETS:
        print(f"\n  {dn}")
        for obj, fam_t, fam_m in PAIRS:
            A, B = per_scene_flood(R, ds, fam_m, SEEDS), per_scene_flood(R, ds, fam_t, SEEDS)
            ks = sorted(set(A) & set(B))
            if not ks:
                print(f"    {obj:22s} --"); continue
            d = np.array([A[k] - B[k] for k in ks])
            w = int((d > 0).sum())
            print(f"    {obj:22s} mean {d.mean():+.4f}  MNDWI-labels ahead on "
                  f"{w:2d}/{len(d)}  sign p={sign_p(w, len(d)):.3g}  "
                  f"wilcoxon p={wilc_p(d):.3g}")
            rows.append(["paired", ds, obj, "MNDWI - Prithvi", fam_m, "clean_flood_IoU",
                         f"{d.mean():.4f}", "", len(d)])

    # 3. robustness envelopes
    print(f"\n{'='*112}\nWORST-CASE DROP (per seed, then averaged) — micro mIoU\n{'='*112}")
    for ds, dn in DATASETS:
        print(f"\n  {dn}")
        print(f"    {'model':34s} {'gain envelope':>20s} {'haze envelope':>20s}")
        todo = [(l, f, s) for l, f, s in REFS] + \
               [(f"{o.split()[0]} / {src}", f, SEEDS)
                for o, ft, fm in PAIRS for src, f in (("Prithvi", ft), ("MNDWI", fm))]
        for label, fam, seeds in todo:
            cells = []
            for axis in ("gain", "haze"):
                env = ENVELOPES[axis]["full"]
                m, sd, n = wdrop(rec, ds, fam, seeds, env)
                cells.append(f"{m:13.4f}±{sd:.3f}" if n else f"{'--':>20s}")
                rows.append(["robustness", ds, axis, label, fam, "worst_drop_micro_mIoU",
                             f"{m:.4f}" if n else "", f"{sd:.4f}" if n else "", n])
            print(f"    {label:34s}" + "".join(cells))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
