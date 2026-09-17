#!/usr/bin/env python3
"""Objective ladder CE -> Logit-KD -> OFA against pseudo-label pool size N (Fig. 2).

Reads pooled micro scores from runs/nsweep/<arm>_N<n>_s<seed>/eval_results.json, as written by
evaluation/eval_ablations.py. Seeds are paired for the OFA - Logit-KD column.
"""
from __future__ import annotations
import csv, glob, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.fig_style import out_dir

OUT = out_dir("matched")
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.color": "0.91",
                     "axes.axisbelow": True, "figure.dpi": 120,
                     "axes.edgecolor": "0.35", "axes.labelcolor": "0.15",
                     "text.color": "0.15", "xtick.color": "0.3", "ytick.color": "0.3"})
INK = "0.15"

_CE, _LKD, _OFA = "#D55E00", "#0072B2", "#009E73"
OBJ = [("pseudo_ce_aug",   "CE",       _CE,  "--", "o"),
       ("pseudo_dfkd_aug", "Logit-KD", _LKD, "-.", "s"),
       ("pseudo_ofa_aug",  "OFA",      _OFA, "-",  "D")]
NS = [20, 40, 50, 100, 150, 200, 252, 2500]
BENCH = [("sturm_mIoU", "STURM-Flood (OOD)"),
         ("wf_fo_mIoU", "WorldFloods-v2 (OOD, flood-only)"),
         ("s11_test_mIoU", "Sen1Floods11 test (in-distribution)")]


def load():
    """(prefix, N) -> list of per-seed pooled result dicts."""
    D = {}
    for pre, *_ in OBJ:
        for n in NS:
            vals = []
            for d in sorted(glob.glob(f"runs/nsweep/{pre}_N{n}_s*")):
                f = os.path.join(d, "eval_results.json")
                if os.path.exists(f):
                    vals.append(json.load(open(f)))
            if vals:
                D[(pre, n)] = vals
    return D


def stat(D, pre, n, key):
    v = [x[key] for x in D.get((pre, n), []) if x.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def main():
    D = load()
    if not D:
        raise SystemExit("no runs/nsweep/*/eval_results.json found; evaluate the N-sweep first")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15))

    for ax, (key, title) in zip(axes, BENCH):
        # N range in which OFA's seed mean leads Logit-KD on all three benchmarks.
        ax.axvspan(50, 252, color="#F2F5EF", zorder=0)
        for pre, lab, col, ls, mk in OBJ:
            m = np.array([stat(D, pre, n, key)[0] for n in NS])
            s = np.array([stat(D, pre, n, key)[1] for n in NS])
            ok = ~np.isnan(m)
            ax.plot(np.array(NS)[ok], m[ok], ls, color=col, marker=mk, ms=3.4, lw=1.5,
                    label=lab, zorder=3)
            ax.fill_between(np.array(NS)[ok], (m - s)[ok], (m + s)[ok], color=col,
                            alpha=0.13, lw=0, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 252, 1000, 2500])
        ax.set_xticklabels(["20", "50", "100", "252", "1000", "2500"])
        ax.set_title(title, fontsize=9, color=INK, pad=5)
        ax.set_xlabel("pseudo-label pool size  $N$", fontsize=8.6)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("clean mIoU (micro)", fontsize=8.8)
    axes[0].legend(fontsize=8, loc="lower right", framealpha=.93, handlelength=2.0)

    fig.tight_layout()
    fig.savefig(OUT / "fig_ladder_vs_n.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_ladder_vs_n.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ladder_span is quoted in the text, so it stays in the CSV.
    with open(OUT / "fig_ladder_vs_n_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "N", "objective", "mean_micro_mIoU", "sd", "n_seeds",
                    "ofa_minus_lkd", "ofa_wins_over_lkd", "ladder_span"])
        for key, _ in BENCH:
            for n in NS:
                pa = {x.get("seed", i): x for i, x in enumerate(D.get(("pseudo_ofa_aug", n), []))}
                pb = {x.get("seed", i): x for i, x in enumerate(D.get(("pseudo_dfkd_aug", n), []))}
                dif = [pa[k2][key] - pb[k2][key] for k2 in sorted(set(pa) & set(pb))
                       if pa[k2].get(key) is not None and pb[k2].get(key) is not None]
                dmean = round(float(np.mean(dif)), 5) if dif else ""
                dwin = f"{sum(1 for x in dif if x > 0)}/{len(dif)}" if dif else ""
                allm = [stat(D, pre, n, key)[0] for pre, *_ in OBJ]
                span = ("" if any(np.isnan(x) for x in allm)
                        else round(float(max(allm) - min(allm)), 5))
                for pre, lab, *_ in OBJ:
                    m, s, k = stat(D, pre, n, key)
                    w.writerow([key, n, lab, "" if np.isnan(m) else round(m, 5),
                                "" if np.isnan(s) else round(s, 5), k,
                                dmean if lab == "OFA" else "", dwin if lab == "OFA" else "",
                                span if lab == "OFA" else ""])

    print("wrote", OUT / "fig_ladder_vs_n.pdf")


if __name__ == "__main__":
    main()
