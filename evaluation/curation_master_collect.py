# !/usr/bin/env python3
"""Collect S11, STURM and flood-only WorldFloods selector results.

Current clean STURM results use eval_results.json. In the historical source
data, the same filename can mean percentile normalization; there the audited
file is eval_results_STURM_ORIG.json. Do not mix them.
Older WorldFloods flood-only files use wf_mIoU, newer files use wf_fo_mIoU.
Keep recipe-matched controls separate from the unaugmented curation runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEEDS = [42, 64, 91, 107]

# (label, runs_base, dir_pattern, recipe, family)
ARMS = [
    # PRE-FILTERED: 23,720 tiles -> 5,987 candidates
    ("FPS/diversity",  "runs/curation", "fps_ofa_hs_N2500_s{s}",           "amp", "filtered"),
    ("R2-Core",        "runs/curation", "r2core_ofa_hs_N2500_s{s}",        "amp", "filtered"),
    ("GLISTER Alg.2",  "runs/curation", "glister_ofa_hs_N2500_s{s}",       "amp", "filtered"),
    ("confidence",     "runs/curation", "confidence_ofa_hs_N2500_s{s}",    "amp", "filtered"),
    ("R2-Core unif-w", "runs/curation", "r2core_flonly_ofa_hs_N2500_s{s}", "amp", "filtered"),
    ("F-GLISTER-R2",   "runs/curation", "glister_r2_ofa_hs_N2500_s{s}",    "amp", "filtered"),
    ("F-GLISTER live", "runs/curation", "glister_flive_ofa_hs_N2500_s{s}", "amp", "filtered"),
    ("R-GLISTER-OOD",  "runs/curation", "glister_ood_ofa_hs_N2500_s{s}",   "amp", "filtered"),
    ("RANDOM",         "runs/curation", "random_ofa_hs_N2500_s{s}",        "amp", "filtered"),
    # NO PRE-FILTER: selectors rank all 23,720 tiles
    ("FPS raw",        "runs/curation", "fps_raw_ofa_hs_N2500_s{s}",        "amp", "unfiltered"),
    ("R2-Core raw",    "runs/curation", "r2core_raw_ofa_hs_N2500_s{s}",     "amp", "unfiltered"),
    ("GLISTER raw",    "runs/curation", "glister_raw_ofa_hs_N2500_s{s}",    "amp", "unfiltered"),
    ("confidence raw", "runs/curation", "confidence_raw_ofa_hs_N2500_s{s}", "amp", "unfiltered"),
    ("RANDOM raw",     "runs/curation", "random_raw_ofa_hs_N2500_s{s}",     "amp", "unfiltered"),
    # recipe-matched controls: the GT arms' recipe (--no_amp --deterministic)
    ("FPS (r1)",       "runs/curation", "fps_r1_ofa_hs_N2500_s{s}",    "no_amp", "recipe-ctl"),
    ("R2-Core (r1)",   "runs/curation", "r2core_r1_ofa_hs_N2500_s{s}", "no_amp", "recipe-ctl"),
]

# metric key -> list of (filename, key) candidates, in priority order
SOURCES = {
    "s11_test_mIoU":     [("eval_results.json", "s11_test_mIoU")],
    "s11_test_flood_IoU": [("eval_results.json", "s11_test_flood_IoU")],
    "sturm_mIoU":        [("eval_results.json", "sturm_mIoU")],
    "sturm_flood_IoU":   [("eval_results.json", "sturm_flood_IoU")],
    # flood-only WF: unified key first, then the older per-protocol file
    "wf_fo_mIoU":        [("eval_results.json", "wf_fo_mIoU"),
                          ("eval_results_WFv2fo_s11.json", "wf_mIoU")],
    "wf_fo_flood_IoU":   [("eval_results.json", "wf_fo_flood_IoU"),
                          ("eval_results_WFv2fo_s11.json", "wf_flood_IoU")],
}
SHOW = [("s11_test_mIoU", "S11 mIoU"), ("s11_test_flood_IoU", "S11 flood"),
        ("sturm_mIoU", "STURM mIoU"), ("sturm_flood_IoU", "STURM flood"),
        ("wf_fo_mIoU", "WF-fo mIoU"), ("wf_fo_flood_IoU", "WF-fo flood")]


def value(run: Path, metric: str):
    for fname, key in SOURCES[metric]:
        p = run / fname
        if not p.exists():
            continue
        try:
            v = json.load(open(p)).get(key)
        except Exception:
            continue
        if isinstance(v, (int, float)):
            return v
    return None


def vals(base: str, pat: str, metric: str):
    b = Path(base)
    return [value(b / pat.format(s=s), metric) for s in SEEDS]


def main() -> None:
    data = {lab: {m: vals(rb, pat, m) for m, _ in SHOW} for lab, rb, pat, _, _ in ARMS}
    recipe = {lab: r for lab, _, _, r, _ in ARMS}
    fam = {lab: f for lab, _, _, _, f in ARMS}

    print(f"\n{'='*132}")
    print("  CURATION MASTER TABLE — EfficientViT-B0 + OFA, N=2500, seeds 42/64/91/107")
    print(f"{'='*132}")
    hdr = f"{'method':<20}{'recipe':>9}" + "".join(f"{l:>17}" for _, l in SHOW) + f"{'n':>4}"
    print(hdr); print("-" * len(hdr))
    last = None
    for lab, _, _, _, f in ARMS:
        if last is not None and f != last:
            print()
        last = f
        row = f"{lab:<20}{recipe[lab]:>9}"
        n0 = 0
        for m, _ in SHOW:
            v = np.array([x for x in data[lab][m] if x is not None], float)
            n0 = max(n0, len(v))
            row += f"{(f'{v.mean():.4f}±{v.std(ddof=1):.4f}' if len(v) > 1 else (f'{v[0]:.4f}' if len(v) else '--')):>17}"
        print(row + f"{n0:>4}")

    # paired deltas within each family against that family's FPS arm
    print(f"\n{'='*132}\n  PAIRED per-seed delta vs the FPS arm of the SAME family (same pool, same recipe)\n{'='*132}")
    for family, base_lab in [("filtered", "FPS/diversity"),
                             ("unfiltered", "FPS raw"),
                             ("recipe-ctl", "FPS (r1)")]:
        members = [l for l, _, _, _, f in ARMS if f == family and l != base_lab]
        if not members:
            continue
        print(f"\n--- {family} (baseline: {base_lab}) ---")
        print(f"{'method':<20}" + "".join(f"{l:>17}" for _, l in SHOW))
        for lab in members:
            row = f"{lab:<20}"
            for m, _ in SHOW:
                d = [a - b for a, b in zip(data[lab][m], data[base_lab][m])
                     if a is not None and b is not None]
                row += f"{(f'{np.mean(d):+.4f} ({sum(1 for x in d if x>0)}/{len(d)})' if d else '--'):>17}"
            print(row)
    print("\n(x/y) = seeds where the arm beat its family baseline. 4/4 or 0/4 with a "
          "consistent sign is a real effect even when means overlap.")


if __name__ == "__main__":
    main()
