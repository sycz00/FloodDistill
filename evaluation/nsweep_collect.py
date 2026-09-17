# !/usr/bin/env python3
"""Collect matched-budget pseudo versus Sen1Floods11 results.

Read current eval_results.json files only, excluding perturbation files.
Historical source-data copies use eval_results_STURM_ORIG.json for audited
STURM normalization. Do not read their legacy eval_results.json as current.
Report paired differences over seeds 42, 64, 91 and 107.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [42, 64, 91, 107]
# Full Sen1Floods11 train split: N=252.
NS = [50, 100, 150, 200, 252]
# pseudo arm -> GT counterpart. Architecture, loss and hyperparameters are identical,
# only the label source differs.
PAIRS = [
    ("ofa",       "sen_ofa",       "EvIT-OFA"),
    ("dfkd",      "sen_dfkd",      "EvIT-LogitKD"),
    ("ce",        "sen_ce",        "EvIT-CE"),
    ("dfkd_unet", "sen_dfkd_unet", "UNet-LogitKD"),
    ("ce_unet",   "sen_ce_unet",   "UNet-CE"),
]
METRICS = [("sturm_mIoU", "STURM"), ("sturm_flood_IoU", "STURM-flood"),
           ("wf_fo_mIoU", "WF-fo"), ("s11_test_mIoU", "S11")]
FN = "eval_results.json"


def get(run: Path, key: str):
    p = run / FN
    if not p.exists():
        return None
    try:
        v = json.load(open(p)).get(key)
    except Exception:
        return None
    return v if isinstance(v, (int, float)) else None


def series(base: Path, pat: str, n: int, key: str):
    return [get(base / pat.format(n=n, s=s), key) for s in SEEDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default=None, help="restrict to one metric key")
    ap.add_argument("--ps_base", default="runs/nsweep", help="pseudo-arm runs base dir")
    ap.add_argument("--gt_base", default="runs/ablations_sen1", help="GT-arm runs base dir")
    a = ap.parse_args()
    ps_base, gt_base = Path(a.ps_base), Path(a.gt_base)
    metrics = [m for m in METRICS if a.metric is None or m[0] == a.metric]

    for key, mlabel in metrics:
        print(f"\n{'='*104}\n  {mlabel}  ({key})   pseudo(FPS-curated, GT-parity aug)  vs  GT(Sen1Floods11)\n{'='*104}")
        print(f"{'arm':<15}{'N':>6}{'pseudo':>18}{'GT':>18}{'paired delta':>20}{'seeds':>8}")
        print("-" * 104)
        for parm, garm, label in PAIRS:
            for n in NS:
                pv = series(ps_base, f"pseudo_{parm}_aug_N{{n}}_s{{s}}", n, key)
                gv = series(gt_base, f"{garm}_N{{n}}_s{{s}}", n, key)
                pair = [(p, g) for p, g in zip(pv, gv) if p is not None and g is not None]
                pm = np.array([x for x in pv if x is not None], float)
                gm = np.array([x for x in gv if x is not None], float)
                ptxt = f"{pm.mean():.4f}±{pm.std(ddof=1):.4f}" if len(pm) > 1 else (f"{pm[0]:.4f}" if len(pm) else "--")
                gtxt = f"{gm.mean():.4f}±{gm.std(ddof=1):.4f}" if len(gm) > 1 else (f"{gm[0]:.4f}" if len(gm) else "--")
                if pair:
                    d = np.array([p - g for p, g in pair])
                    dtxt = f"{d.mean():+.4f} ({int((d > 0).sum())}/{len(d)})"
                else:
                    dtxt = "--"
                print(f"{label:<15}{n:>6}{ptxt:>18}{gtxt:>18}{dtxt:>20}{len(pair):>8}")
            print()
        print("delta = pseudo - GT, paired per seed. (x/y) = seeds where pseudo won.")


if __name__ == "__main__":
    main()
