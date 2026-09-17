"""Plot OFA gradient cosines at N=100, 300 and 500 (supplement Fig. S5).

The six CSVs in analysis/grad_conflict_n/ come from an earlier pool. Keep that
provenance when comparing them with new runs. Magnitudes depend on the averaging
window. The CSV name focal denotes the hard-label term; ofa_s0..s3 are exits.
Use --runs to plot new logs produced with --grad_conflict.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

MYBLUE, MYORANGE, MYGREEN, MYGRAY = "#1F77B4", "#D6680E", "#2CA02C", "#828282"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser()
_ap.add_argument("--runs", default=os.path.join(ROOT, "analysis", "grad_conflict_n",
                                                "N{n}_s{seed}.csv"),
                 help="path template with {n} and {seed} placeholders")
_ap.add_argument("--ns", default="100,300,500")
_ap.add_argument("--seeds", default="42,64")
_args = _ap.parse_args()
RUN = _args.runs
NS = [int(x) for x in _args.ns.split(",")]
SEEDS = [int(x) for x in _args.seeds.split(",")]

GROUPS = {
    "main head$\\leftrightarrow$early branches (s0/s1)": ["focal_vs_ofa_s0", "focal_vs_ofa_s1"],
    "main head$\\leftrightarrow$late branches (s2/s3)":  ["focal_vs_ofa_s2", "focal_vs_ofa_s3"],
    "early$\\leftrightarrow$early (s0$\\leftrightarrow$s1)": ["ofa_s0_vs_ofa_s1"],
    "early$\\leftrightarrow$late branches":          ["ofa_s0_vs_ofa_s2", "ofa_s0_vs_ofa_s3",
                                             "ofa_s1_vs_ofa_s2", "ofa_s1_vs_ofa_s3"],
}


def run_means(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.mean([float(r[c]) for r in rows for c in cols])
            for k, cols in GROUPS.items()}


series = {k: {"mean": [], "lo": [], "hi": []} for k in GROUPS}
print(f"{'N':>4s}  " + "  ".join(f"{k.split(' ')[0]:>14s}" for k in GROUPS))
for n in NS:
    per_seed = [run_means(RUN.format(n=n, seed=s)) for s in SEEDS]
    for k in GROUPS:
        vals = [ps[k] for ps in per_seed]
        series[k]["mean"].append(np.mean(vals))
        series[k]["lo"].append(min(vals))
        series[k]["hi"].append(max(vals))
    print(f"{n:>4d}  " + "  ".join(f"{np.mean([ps[k] for ps in per_seed]):>14.3f}"
                                   for k in GROUPS))

STYLE = [  # (color, marker, linestyle, zorder)
    (MYBLUE,   "o", "-",  4),
    (MYGREEN,  "D", "-",  4),
    (MYORANGE, "^", "-",  4),
    (MYGRAY,   "s", "--", 3),
]

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for (label, s), (color, marker, ls, z) in zip(series.items(), STYLE):
    ax.fill_between(NS, s["lo"], s["hi"], color=color, alpha=0.13, linewidth=0)
    ax.plot(NS, s["mean"], color=color, marker=marker, ls=ls, lw=1.8, ms=6,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
            label=label, zorder=z)

ax.axhline(0.0, color="0.35", lw=1.0)
ax.text(103, 0.006, "cooperation", fontsize=8, color="0.35", va="bottom")
ax.text(103, -0.012, "conflict", fontsize=8, color="0.35", va="top")

ax.set_xticks(NS)
ax.set_xlabel("Pseudo-label pool size $N$")
ax.set_ylabel("Mean pairwise gradient cosine")
ax.grid(True, ls="--", color="0.88", lw=0.6)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="0.7")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "assets", "grad_conflict_n.pdf"))
fig.savefig(os.path.join(ROOT, "assets", "grad_conflict_n.png"), dpi=160)
print("saved assets/grad_conflict_n.{pdf,png}")
