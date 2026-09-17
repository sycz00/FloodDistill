"""Summarise the pairwise gradient cosines in a training grad_conflict.csv.

Training with --grad_conflict writes columns such as focal_vs_ofa0. Reported:
mean cosine, the fraction below zero, and mean negative cosine.
These are gradient diagnostics, not a causal test of generalization.

Usage: python analysis/grad_conflict.py --csv <run_dir>/grad_conflict.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics


def summarise(csv_path: str) -> dict:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[grad_conflict] empty CSV: {csv_path}")
        return {}
    cols = [c for c in rows[0].keys() if c != "step"]
    out: dict[str, dict] = {}
    for c in cols:
        vals = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        if not vals:
            continue
        neg = [v for v in vals if v < 0]
        out[c] = {
            "n":             len(vals),
            "mean_cos":      statistics.fmean(vals),
            "conflict_rate": len(neg) / len(vals),
            "neg_mean_cos":  statistics.fmean(neg) if neg else 0.0,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True,
                   help="Path to grad_conflict.csv (from training with --grad_conflict).")
    p.add_argument("--plot", default=None,
                   help="Optional PNG path for a per-pair mean-cosine bar chart.")
    args = p.parse_args()

    stats = summarise(args.csv)
    if not stats:
        return

    width = max(len(k) for k in stats)
    print(f"\n{'pair':<{width}}  {'n':>6}  {'mean_cos':>9}  {'conflict%':>9}  {'neg_mean':>9}")
    print("-" * (width + 40))
    for k, s in stats.items():
        print(f"{k:<{width}}  {s['n']:>6d}  {s['mean_cos']:>9.4f}  "
              f"{100 * s['conflict_rate']:>8.1f}%  {s['neg_mean_cos']:>9.4f}")
    print("\nInterpretation: mean_cos < 0 ⇒ branches send conflicting backbone updates "
          "(feature-immaturity regime at small N); it self-heals to > 0 as the pseudo "
          "pool grows (see paper §Analysis).")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            keys = list(stats)
            vals = [stats[k]["mean_cos"] for k in keys]
            plt.figure(figsize=(max(6, len(keys) * 0.9), 4))
            plt.bar(keys, vals, color=["#d6680e" if v < 0 else "#1f77b4" for v in vals])
            plt.axhline(0, color="k", lw=0.8)
            plt.ylabel("mean inter-branch cosine")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"[grad_conflict] saved plot → {args.plot}")
        except Exception as e:  # noqa: BLE001
            print(f"[grad_conflict] plot skipped: {e}")


if __name__ == "__main__":
    main()
