# !/usr/bin/env python3
"""Collect selector results and paired seed differences from runs/curation/.

Read protocol-matched evaluation JSONs and report mean, standard deviation
and paired differences for seeds 42, 64, 91 and 107. Four-seed win counts are
descriptive only. For the paper's inferential comparisons, use group-level tests.
Use --haze for the perturbation table.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ARM_LABEL = {
    "r2core": "R2-Core (new)",
    "fps": "FPS / diversity",
    "glister": "GLISTER Alg.2 r=25",
    "confidence": "confidence (matched)",
    "r2core_flonly": "R2-Core uniform-w (abl)",
    "glister_r2": "F-GLISTER-R2 (A)",
    "glister_flive": "F-GLISTER live-reg (A ctl)",
    "glister_ood": "R-GLISTER-OOD orth (B)",
    "r2core_raw": "R2-Core no-prefilter",
    "fps_raw": "FPS no-prefilter",
    "glister_raw": "GLISTER no-prefilter",
    "confidence_raw": "confidence no-prefilter",
}
# Key names as eval_ablations.py writes them. wf_fo_* is the flood-only WorldFloods
# protocol (JRC permanent water -> ignore), which is the default here.
METRICS = [
    ("s11_test_mIoU", "S11 mIoU (ID)"),
    ("s11_test_flood_IoU", "S11 floodIoU"),
    ("sturm_mIoU", "STURM (OOD)"),
    ("sturm_flood_IoU", "STURM floodIoU"),
    ("wf_fo_mIoU", "WF-fo (OOD)"),
    ("s1s2w_mIoU", "S1S2W (OOD)"),
]


def merged(run_dir: Path, pattern: str = "eval_results*.json") -> dict:
    """Union of every eval json in the run dir. Different invocations write different
    files (clean vs haze vs WF variants); later files win on key collision."""
    out: dict = {}
    for f in sorted(run_dir.glob(pattern)):
        # Skip perturbed evals. They have the same keys as the clean file and sort after it
        # ("_E5_" > ""), so a plain glob merge overwrites sturm_mIoU with a gain/offset
        # perturbed number. That happened with 16 files in an earlier run tree, g=1.30
        # was reported as clean.
        if "_HAZE_" in f.name or "_E5_" in f.name:
            continue
        try:
            out.update(json.load(open(f)))
        except Exception:
            pass
    return out


def haze_series(run_dir: Path) -> dict[float, float]:
    """h -> sturm_mIoU under simulated haze."""
    out = {}
    for f in sorted(run_dir.glob("eval_results*_HAZE_h*.json")):
        m = re.search(r"_HAZE_h([0-9.]+)\.json$", f.name)
        if not m:
            continue
        try:
            out[float(m.group(1))] = float(json.load(open(f))["sturm_mIoU"])
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_base", default="runs/curation")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 64, 91, 107])
    ap.add_argument("--baseline", default="fps")
    ap.add_argument("--haze", action="store_true", help="also print the haze sweep (H1)")
    a = ap.parse_args()
    rb = Path(a.runs_base)

    arms, seen = [], set()
    for d in sorted(rb.glob(f"*_ofa_hs_N{a.n}_s*")):
        arm = d.name.split("_ofa_hs_")[0]
        if arm not in seen:
            seen.add(arm); arms.append(arm)
    arms.sort(key=lambda x: list(ARM_LABEL).index(x) if x in ARM_LABEL else 99)
    if not arms:
        raise SystemExit(f"no runs under {rb}/*_ofa_hs_N{a.n}_s*")

    data = {arm: {s: merged(rb / f"{arm}_ofa_hs_N{a.n}_s{s}") for s in a.seeds}
            for arm in arms}

    def vals(arm, key):
        v = [data[arm][s].get(key) for s in a.seeds]
        return np.array([x for x in v if isinstance(x, (int, float))], float)

    print(f"\n=== curation comparison | N={a.n} | seeds {a.seeds} ===")
    hdr = f"{'arm':<26}" + "".join(f"{lab:>18}" for _, lab in METRICS) + f"{'n':>4}"
    print(hdr); print("-" * len(hdr))
    for arm in arms:
        cells = ""
        for k, _ in METRICS:
            v = vals(arm, k)
            if len(v) > 1:
                txt = f"{v.mean():.4f}±{v.std(ddof=1):.4f}"
            elif len(v):
                txt = f"{v[0]:.4f}"
            else:
                txt = "--"
            cells += f"{txt:>18}"
        print(f"{ARM_LABEL.get(arm, arm):<26}{cells}{len(vals(arm, METRICS[0][0])):>4}")

    if a.baseline in data:
        print(f"\n--- paired per-seed delta vs {ARM_LABEL.get(a.baseline, a.baseline)} ---")
        for arm in arms:
            if arm == a.baseline:
                continue
            line = f"{ARM_LABEL.get(arm, arm):<26}"
            for k, _ in METRICS:
                d = [data[arm][s][k] - data[a.baseline][s][k] for s in a.seeds
                     if isinstance(data[arm][s].get(k), (int, float))
                     and isinstance(data[a.baseline][s].get(k), (int, float))]
                if len(d) > 1:
                    d = np.array(d)
                    line += f"{f'{d.mean():+.4f} ({int((d>0).sum())}/{len(d)})':>18}"
                else:
                    line += f"{'--':>18}"
            print(line)
        print("\n(x/y) = seeds where the arm beat the baseline; 4/4 or 0/4 with a "
              "consistent sign is a real effect even if the means overlap.")

    if a.haze:
        print(f"\n=== haze sweep — STURM mIoU (H1: R2-Core should hold up better) ===")
        hs = sorted({h for arm in arms for s in a.seeds
                     for h in haze_series(rb / f"{arm}_ofa_hs_N{a.n}_s{s}")})
        if not hs:
            print("  (no haze evals yet — run: GPU=0 HAZE=1 bash experiments/curation_eval.sh)")
            return
        hdr = f"{'arm':<26}" + "".join(f"{('h=%s' % h):>13}" for h in hs)
        print(hdr); print("-" * len(hdr))
        base_first = None
        for arm in arms:
            row, first = f"{ARM_LABEL.get(arm, arm):<26}", None
            for h in hs:
                v = np.array([haze_series(rb / f"{arm}_ofa_hs_N{a.n}_s{s}").get(h)
                              for s in a.seeds], dtype=object)
                v = np.array([x for x in v if isinstance(x, (int, float))], float)
                # Show the seed count: an arm averaged over 2 seeds is not comparable to
                # one averaged over 4, and the mean alone does not show this.
                row += f"{(f'{v.mean():.4f}({len(v)})' if len(v) else '--'):>13}"
                if len(v) and first is None:
                    first = v.mean()
            print(row)
        # Retention must be measured over the same haze range for every arm, or an arm
        # that has only finished the mild levels looks artificially robust. Restrict to
        # the h values every arm has actually completed.
        series = {arm: {h: np.mean([x for x in
                                    [haze_series(rb / f"{arm}_ofa_hs_N{a.n}_s{s}").get(h)
                                     for s in a.seeds] if x is not None] or [np.nan])
                        for h in hs} for arm in arms}
        common = [h for h in hs
                  if all(not np.isnan(series[arm][h]) for arm in arms
                         if any(not np.isnan(v) for v in series[arm].values()))]
        if len(common) < 2:
            print("\n(retention needs >=2 haze levels completed by every arm; "
                  f"common so far: {common})")
            return
        lo, hi = min(common), max(common)
        print(f"\nRetention over the COMMON range h={lo} -> h={hi} (H1 statistic):")
        for arm in arms:
            s_lo, s_hi = series[arm][lo], series[arm][hi]
            if not (np.isnan(s_lo) or np.isnan(s_hi)):
                print(f"  {ARM_LABEL.get(arm, arm):<26} {s_hi/s_lo:.4f} "
                      f"({s_lo:.4f} -> {s_hi:.4f})")


if __name__ == "__main__":
    main()
