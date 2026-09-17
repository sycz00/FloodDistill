# !/usr/bin/env python3
"""Training-budget curves on STURM and WorldFloods (Figs. 4, 5 and S4).

GT stops at N=252. Pseudo-labels inherit the teacher's labelled training data.
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.fig_style import DS, SEEDS_OF, load_cache, out_dir, pool_arg

POOL = pool_arg()
OUT = out_dir(POOL)
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.color": "0.91",
                     "axes.axisbelow": True, "figure.dpi": 120,
                     "axes.edgecolor": "0.35", "axes.labelcolor": "0.15",
                     "text.color": "0.15", "xtick.color": "0.3", "ytick.color": "0.3"})
MUTED = "0.45"
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#8A8A8A"

GT_NS = [20, 40, 50, 60, 80, 100, 150, 200, 250, 252]
LABEL_CEILING = 252          # Sen1Floods11 hand-labelled train split

# Pseudo arm: pseudo_ofa_aug_N*, each N trained on its own diversity_N<n>_index.json.

AP = argparse.ArgumentParser()
AP.add_argument("--dataset", default="wf", choices=["wf", "sturm"])
AP.add_argument("--metric", default="mIoU", choices=["mIoU", "flood"])
AP.add_argument("--arch", default="evit", choices=["evit", "unet"])
AP.add_argument("--pool", default="matched")          # consumed by fig_style.pool_arg
ARGS = AP.parse_args()
ds, MET, ARCH = ARGS.dataset, ARGS.metric, ARGS.arch
if ARCH == "unet" and ds != "sturm":
    raise SystemExit("--arch unet is the STURM architecture control (Fig. S4); use --dataset sturm")

# Both arms of a panel use the same objective. There is no UNet-OFA run, so UNet uses Logit-KD.
ARCHS = {
    "evit": {"obj": "OFA", "label": "EvIT-OFA", "params": "0.70M",
             "gt": ("ncgt_evit_ofa", [20, 40, 50, 60, 80, 100, 150, 200, 250, 252]),
             "ps": {"matched": ("evit_ofa_ps", [20, 40, 50, 100, 150, 200, 252, 2500, 4000],
                           {}, 4000)}},
    "unet": {"obj": "Logit-KD", "label": "UNet-LKD", "params": "32.5M",
             "gt": ("ncgt_unet_lkd", [20, 40, 50, 60, 80, 100, 150, 200, 252]),
             "ps": {"matched": ("unet_lkd_ps",
                           [20, 40, 50, 100, 150, 200, 252, 2500, 4000], {}, 4000)}},
}
# The curve covers all N, so the full pool uses the same axis.
for _a in ARCHS.values():
    if "matched" in _a["ps"]:
        _a["ps"].setdefault("full", _a["ps"]["matched"])
if POOL not in ARCHS[ARCH]["ps"]:
    raise SystemExit(f"--arch {ARCH} has no pseudo arm in pool {POOL}: the UNet N-sweep exists "
                     f"only in runs/nsweep. Use --pool matched.")
GT_PREFIX, GT_NS = ARCHS[ARCH]["gt"]
PS_PREFIX, PS_NS, PS_XMAP, PSEUDO_MAX_X = ARCHS[ARCH]["ps"][POOL]
ARM = ARCHS[ARCH]["label"]

AGG = [(f"micro_{MET}", "(a) micro — pixel-pooled"),
       (f"macro_scene_{MET}", "(b) macro — per-scene, unweighted")]
MET_LABEL = {"mIoU": "mIoU", "flood": "flood IoU"}[MET]

R = load_cache()
DUMP = [["dataset", "aggregation", "arm", "N", "mean", "std", "n_seeds"]]

def series(prefix, ns, metric, arm):
    xs, ms, ss = [], [], []
    for n in ns:
        f = f"{prefix}_N{n}"
        v = [R.get((ds, f, s, "clean"), {}).get(metric, np.nan) for s in SEEDS_OF(f)]
        v = [x for x in v if x == x]
        if not v:
            continue
        xs.append(n); ms.append(np.mean(v)); ss.append(np.std(v))
        DUMP.append([ds, metric, arm, n, f"{np.mean(v):.4f}", f"{np.std(v):.4f}", len(v)])
    return np.array(xs, float), np.array(ms), np.array(ss)


def teacher(metric):
    r = R.get((ds, "teacher", "-", "clean"))
    return None if r is None else r.get(metric)


fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
for ax, (metric, title) in zip(axes, AGG):
    t = teacher(metric)
    if t is not None:
        ax.axhline(t, color=GRAY, ls=(0, (5, 4)), lw=1.0, zorder=1)
        ax.text(21, t + .004, "Teacher (Prithvi 300M)", color=GRAY, fontsize=7.6, va="bottom")
        DUMP.append([ds, metric, "teacher", "-", f"{t:.4f}", "0.0000", 1])

    x, m, s = series(GT_PREFIX, GT_NS, metric, f"GT-{ARM}")
    ax.fill_between(x, m - s, m + s, color=ORANGE, alpha=.13, lw=0)
    ax.plot(x, m, "--", color=ORANGE, marker="s", ms=4.5, mec="white", mew=.6, lw=1.9,
            label=f"GT-{ARM}  (S11 human labels, capped @{LABEL_CEILING})")

    x, m, s = series(PS_PREFIX, PS_NS, metric, f"pseudo-{ARM}")
    xp = np.array([PS_XMAP.get(int(v), v) for v in x], float)
    ax.fill_between(xp, m - s, m + s, color=BLUE, alpha=.14, lw=0)
    # The teacher was fine-tuned on the same 252 Sen1Floods11 labels, so the pseudo arm is not label-free.
    ax.plot(xp, m, "-", color=BLUE, marker="o", ms=4.5, mec="white", mew=.6, lw=2.2,
            label=f"pseudo-{ARM}  (same 252 labels, routed through the teacher)")

    # Shade the range above N=252 if the pseudo sweep goes beyond it.
    if PSEUDO_MAX_X > LABEL_CEILING:
        ax.axvspan(LABEL_CEILING, PSEUDO_MAX_X, color="0.94", lw=0, zorder=0)
    ax.set_xscale("log")
    # 200 and 252 overlap on the log axis, so only 252 gets a label.
    ticks = sorted({20, 50, 100, 252, 1000, PSEUDO_MAX_X} & {*range(0, PSEUDO_MAX_X + 1)}
                   | {PSEUDO_MAX_X})
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("dataset size $N$  (scenes)", fontsize=8.8)
    ax.set_title(title, fontsize=9.4)
    ax.margins(y=.08)
DS_LABEL = DS[ds]
axes[0].set_ylabel(f"{DS_LABEL.split(' (')[0]} {MET_LABEL}", fontsize=9.0)
if PSEUDO_MAX_X > LABEL_CEILING:
    axes[1].text(np.sqrt(LABEL_CEILING * PSEUDO_MAX_X), .02, "beyond the human-label ceiling",
                 transform=axes[1].get_xaxis_transform(), ha="center", va="bottom",
                 fontsize=7.4, color=MUTED, rotation=0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.2, frameon=False,
           bbox_to_anchor=(.5, -.075), handlelength=2.6)
fig.suptitle(f"E3 — label efficiency at a matched {ARCHS[ARCH]['obj']} recipe "
             f"({ARM}, {ARCHS[ARCH]['params']}), {DS_LABEL}  ·  pseudo pool {POOL}  ·  "
             f"4 seeds (±1 sd)", fontsize=10.4, y=1.0)
CAVEAT = ("Both arms rest on the SAME 252 hand labels: the GT arm trains on them directly, the "
          "pseudo arm inherits them through the\nteacher (fine-tuned on the full 252-scene S11 "
          "train split) applied to 23,720 unlabelled GDACS tiles. The x-axis is training-set "
          "size,\nnot annotation budget — the pseudo arm is not label-free, and the comparison "
          "is 'route the labels through a 300M teacher' vs 'don't'.")
fig.text(.5, -.20, CAVEAT, ha="center", va="top", fontsize=7.2, color=MUTED)
fig.tight_layout()
TAG = f"{ds}" + ("" if ARCH == "evit" else f"_{ARCH}")
for e in ("pdf", "png"):
    fig.savefig(OUT / f"fig_e3_{TAG}.{e}", bbox_inches="tight")
plt.close(fig)

# Table at matched N and the smallest pseudo N that reaches GT at N=252
print(f"[cache] {len(R)} cells   dataset={ds}   arch={ARCH}   metric={MET}")
for metric, title in AGG:
    gx, gm, gs = series(GT_PREFIX, GT_NS, metric, "_")
    px, pm, ps = series(PS_PREFIX, PS_NS, metric, "_")
    G = dict(zip(gx, zip(gm, gs))); P = dict(zip(px, zip(pm, ps)))
    print(f"\n=== {title}  ({metric}) ===")
    print(f"{'N':>5} | {'GT-'+ARM:>16} | {'pseudo-'+ARM:>16} | {'pseudo - GT':>12}")
    for n in sorted(set(G) & set(P)):
        d = P[n][0] - G[n][0]
        print(f"{int(n):>5} | {G[n][0]:.4f}±{G[n][1]:.4f} | {P[n][0]:.4f}±{P[n][1]:.4f} | {d:>+12.4f}")
    win = sum(P[n][0] > G[n][0] for n in set(G) & set(P))
    print(f"pseudo wins at {win}/{len(set(G) & set(P))} matched N")
    if LABEL_CEILING in G:
        best_p = max(pm)
        print(f"GT at the ceiling N={LABEL_CEILING}: {G[LABEL_CEILING][0]:.4f}   "
              f"| best pseudo (any N): {best_p:.4f}   -> gap {best_p - G[LABEL_CEILING][0]:+.4f}")
        reach = [int(n) for n in sorted(P) if P[n][0] >= G[LABEL_CEILING][0]]
        print(f"pseudo N needed to match the full {LABEL_CEILING}-label budget: "
              f"{reach[0] if reach else 'never'}")

with open(OUT / f"fig_e3_{TAG}_values.csv", "w", newline="") as f:
    csv.writer(f).writerows(DUMP)
print(f"\n-> {OUT}/fig_e3_{TAG}.pdf  (+ _values.csv, {len(DUMP)-1} rows)")
