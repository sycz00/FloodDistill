# !/usr/bin/env python3
"""Plot CKA matrices and diagonal summaries over training seeds.

Each JSON must measure all conditions with the same teacher and probe order.
Keep the probe seed fixed across checkpoint seeds. Standard deviations are
over runs, not the four stages. Spatial and global CKA are not interchangeable.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SRC_TMPL = "cka_spatial_gt_vs_pseudo_s{seed}.json"
SRC_DIR  = Path("paper_source_data/Supp_Figure_S3_CKA_Alignment/raw")   # --src_dir
OUT = Path("assets/sturm_orig/figures")

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.color": "0.91",
                     "axes.axisbelow": True, "figure.dpi": 120,
                     "axes.edgecolor": "0.35", "axes.labelcolor": "0.15",
                     "text.color": "0.15", "xtick.color": "0.3", "ytick.color": "0.3"})
INK, MUTED, UNTRAINED = "0.15", "0.45", "0.55"

CE, LKD, OFA = "#D55E00", "#0072B2", "#009E73"
HUE = {"ce": CE, "lkd": LKD, "ofa": OFA}
OBJ = [("ce", "CE"), ("lkd", "Logit-KD"), ("ofa", "OFA")]

SOURCE = {  # key -> (legend label, bar style)
    "gt":     ("Sen1Floods11",                      dict(fill=False, hatch="///", lw=1.4)),
    "ps252":  ("pseudo-labels ($N\\!=\\!252$)",     dict(fill=False, hatch="...", lw=1.4)),
    "ps2500": ("pseudo-labels",                     dict(fill=True,  lw=0.8)),
}
# Budgets are stated once in the LaTeX caption rather than in both legends (2026-08-19).
SHORT = {"gt": "Sen1Floods11", "ps252": "pseudo-labels ($N\\!=\\!252$)",
         "ps2500": "pseudo-labels"}


# data
def load(seeds):
    """-> per_seed[label] = (S, 4, 4) stack, in the order given."""
    stacks, missing = {}, []
    for s in seeds:
        p = SRC_DIR / SRC_TMPL.format(seed=s)
        if not p.exists():
            missing.append(str(p)); continue
        for c in json.load(open(p))["comparisons"]:
            if c.get("skipped"):
                continue
            stacks.setdefault(c["label"], []).append(np.asarray(c["cka_matrix"], float))
    if missing:
        sys.exit("missing input(s): " + ", ".join(missing) +
                 "\ngenerate them with analysis/cka.py --compare_ckpts (one call per seed) "
                 "- see analysis/cka_cache/README.md for the command.")
    n = {k: len(v) for k, v in stacks.items()}
    if len(set(n.values())) != 1:
        sys.exit(f"conditions disagree on seed count: {n}")
    return {k: np.stack(v) for k, v in stacks.items()}, len(seeds)


class Stats:
    """Per-condition seed statistics. Diagonals are extracted per seed, then aggregated,
    so the s.d. is over training runs -- not over the four stages."""

    def __init__(self, stacks):
        self.M = {k: v.mean(0) for k, v in stacks.items()}            # mean matrix
        self.dg = {k: np.diagonal(v, axis1=1, axis2=2) for k, v in stacks.items()}  # (S,4)
        self.diag = {k: d.mean(0) for k, d in self.dg.items()}
        self.diag_sd = {k: d.std(0, ddof=1) if d.shape[0] > 1 else np.zeros(4)
                        for k, d in self.dg.items()}
        self.md = {k: float(d.mean(1).mean()) for k, d in self.dg.items()}
        self.md_sd = {k: float(d.mean(1).std(ddof=1)) if d.shape[0] > 1 else 0.0
                      for k, d in self.dg.items()}


# A
def fig_bars(st, sources, nseed, tag):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.25),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    err = nseed > 1

    w, xs = 0.78 / len(sources), np.arange(len(OBJ))
    for si, pre in enumerate(sources):
        style = SOURCE[pre][1]
        off = (si - (len(sources) - 1) / 2) * w
        for oi, (obj, _) in enumerate(OBJ):
            k = f"{pre}-{obj}"
            v, e = st.md[k], st.md_sd[k]
            fc = HUE[obj] if style.get("fill") else "white"
            axL.bar(xs[oi] + off, v, w * 0.9, facecolor=fc, edgecolor=HUE[obj],
                    hatch=style.get("hatch"), linewidth=style["lw"], zorder=3)
            if err:
                axL.errorbar(xs[oi] + off, v, yerr=e, fmt="none", ecolor="0.3",
                             elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=5)
            axL.text(xs[oi] + off, v + e + 0.014, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=7.0, color=INK, zorder=4)
    axL.axhline(st.md["random"], ls=(0, (4, 3)), lw=1.2, color=UNTRAINED, zorder=2)
    axL.text(0.5, st.md["random"], f"ImageNet-init, untrained ({st.md['random']:.2f})",
             ha="center", va="center", fontsize=7.2, color=UNTRAINED, zorder=6,
             bbox=dict(fc="white", ec="none", pad=1.6))
    axL.set_xticks(xs); axL.set_xticklabels([o[1] for o in OBJ], fontsize=9)
    axL.set_xlabel("training objective", fontsize=8.8)
    axL.set_ylabel("mean diagonal spatial CKA", fontsize=8.8)
    axL.set_ylim(0, 0.80); axL.set_xlim(-0.55, len(OBJ) - 0.45)
    axL.set_title("(a) alignment to the teacher", fontsize=9.4)
    axL.legend(handles=[Patch(facecolor="0.45", edgecolor="0.3", label=SOURCE[p][0])
                        if SOURCE[p][1].get("fill") else
                        Patch(facecolor="white", edgecolor="0.3",
                              hatch=SOURCE[p][1].get("hatch"), label=SOURCE[p][0])
                        for p in sources],
               fontsize=7.3, frameon=False, loc="upper left", handlelength=1.8,
               borderpad=0.2, labelspacing=0.3)

    # (b) per-stage diagonal. Two sources max keeps this readable; with three the middle
    # one is dotted.
    stages = np.arange(4)
    ls_for = {0: "--", 1: "-", 2: ":"}
    axR.plot(stages, st.diag["random"], ls=(0, (4, 3)), lw=1.4, color=UNTRAINED,
             marker="o", ms=4.5, mfc="white", zorder=3)
    for si, pre in enumerate(sources):
        solid = si == len(sources) - 1
        for obj, _ in OBJ:
            k = f"{pre}-{obj}"
            axR.plot(stages, st.diag[k], ls="-" if solid else ls_for.get(si, "--"),
                     lw=1.9 if solid else 1.5, color=HUE[obj], marker="o",
                     ms=5.4 if solid else 5, mfc=HUE[obj] if solid else "white",
                     mec="white" if solid else HUE[obj], mew=0.9 if solid else 1.3,
                     zorder=5 if solid else 4)
            if err:
                axR.fill_between(stages, st.diag[k] - st.diag_sd[k],
                                 st.diag[k] + st.diag_sd[k], color=HUE[obj],
                                 alpha=0.10, lw=0, zorder=2)
    axR.set_xticks(stages)
    axR.set_xticklabels([f"S{i}\n$\\times$T{i}" for i in stages], fontsize=8.2)
    axR.set_xlabel("student stage $\\times$ matched teacher neck level", fontsize=8.8)
    axR.set_ylabel("spatial CKA", fontsize=8.8)
    axR.set_ylim(0.08, 0.80); axR.set_xlim(-0.35, 3.35)
    axR.set_title("(b) resolved by depth", fontsize=9.4)
    hs = [Line2D([], [], color="0.3", ls="-" if si == len(sources) - 1
                 else ls_for.get(si, "--"), marker="o",
                 mfc="0.3" if si == len(sources) - 1 else "white",
                 lw=1.9 if si == len(sources) - 1 else 1.5, ms=5, label=SHORT[p])
          for si, p in enumerate(sources)]
    hs.append(Line2D([], [], color=UNTRAINED, ls=(0, (4, 3)), lw=1.4, marker="o",
                     mfc="white", ms=4.5, label="untrained"))
    axR.legend(handles=hs, fontsize=7.3, loc="lower left", handlelength=2.2,
               borderpad=0.35, labelspacing=0.3, frameon=True, framealpha=0.92,
               edgecolor="0.85", facecolor="white")
    fig.text(.5, -.055,
             "Hue = objective (CE / Logit-KD / OFA). "
             + (f"Mean $\\pm$ s.d. over {nseed} training seeds; " if err
                else "Single seed (s42); ")
             + "100 probe batches, identical teacher and loader for every bar and line.",
             ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cka_bars{tag}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}/fig_cka_bars{tag}.pdf")


# B
def fig_spatial(st, pseudo, nseed, tag):
    """Plot teacher-level by student-stage CKA matrices and their diagonal profiles."""
    panels = [("random", "untrained\n(ImageNet-init)")] + \
             [(f"{pseudo}-{o}", lab + (" (ours)" if o == "ofa" else ""))
              for o, lab in OBJ]
    vmax = max(st.M[k].max() for k, _ in panels)

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, len(panels) + 1, height_ratios=[1.0, 0.88],
                          width_ratios=[1] * len(panels) + [0.055],
                          hspace=0.30, wspace=0.18)
    im = None
    for j, (key, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        m = st.M[key]
        im = ax.imshow(m, cmap="Blues", vmin=0.0, vmax=vmax, aspect="equal")
        for a in range(4):
            for b in range(4):
                ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=6.4,
                        color="white" if m[a, b] > 0.62 * vmax else "0.25")
        ax.set_title(f"{lab}\ndiag {st.md[key]:.3f}", fontsize=8.4, pad=4)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"S{k}" for k in range(4)], fontsize=7.4)
        ax.set_xlabel("student stage", fontsize=7.6)
        if j == 0:
            ax.set_yticks(range(4)); ax.set_yticklabels([f"T{k}" for k in range(4)], fontsize=7.4)
            ax.set_ylabel("teacher neck level", fontsize=7.6)
        else:
            ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color("0.6")
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(im, cax=cax); cb.set_label("spatial CKA", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)

    axl = fig.add_subplot(gs[1, :-1])
    stages = np.arange(4)
    axl.plot(stages, st.diag["random"], ls=(0, (4, 3)), lw=1.5, color=UNTRAINED,
             marker="o", ms=5, mfc="white", label="untrained", zorder=3)
    for obj, olab in OBJ:
        k = f"{pseudo}-{obj}"
        axl.plot(stages, st.diag[k], lw=2.0, color=HUE[obj], marker="o", ms=5.6,
                 mfc=HUE[obj], mec="white", mew=0.9, label=olab, zorder=4)
        if nseed > 1:
            axl.fill_between(stages, st.diag[k] - st.diag_sd[k], st.diag[k] + st.diag_sd[k],
                             color=HUE[obj], alpha=0.12, lw=0, zorder=2)
    axl.set_xticks(stages)
    axl.set_xticklabels([f"stage {k}  ($\\times$ T{k})" for k in stages], fontsize=8.2)
    axl.set_xlim(-0.25, 3.25); axl.set_ylim(0, 0.85)
    axl.set_ylabel("diagonal spatial CKA", fontsize=8.6)
    axl.set_xlabel("student backbone depth $\\rightarrow$", fontsize=8.6)
    # The band between the untrained line (~0.2) and the trained ones (>=0.46) is empty at
    # every stage, so the legend goes there rather than above the axes, where it would
    # collide with the heatmap row's x-labels.
    axl.legend(fontsize=8, ncol=4, loc="center", bbox_to_anchor=(0.5, 0.38),
               handlelength=1.9, columnspacing=1.6, frameon=True, framealpha=0.92,
               edgecolor="0.85", facecolor="white")

    fig.suptitle("Teacher neck level $\\times$ student stage, "
                 + SHORT[pseudo].replace("$\\!", "$\\,").replace("\\!", "\\,"),
                 fontsize=9.4, y=0.995)
    fig.text(.5, -.015,
             "Shared colour scale across panels. "
             + (f"Mean over {nseed} training seeds; band = $\\pm$1 s.d.\n" if nseed > 1
                else "Single seed (s42).\n")
             + "The strongest entry in every trained panel is off-diagonal at "
               "T0$\\times$S2: the student's third stage is what best\nreproduces the "
               "teacher's first neck level.",
             ha="center", va="top", fontsize=7.2, color=MUTED, linespacing=1.5)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cka_spatial_{pseudo}{tag}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}/fig_cka_spatial_{pseudo}{tag}.pdf")


# main
def main():
    global SRC_DIR
    global OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,64,91,107",
                    help="training seeds to average over (comma-separated)")
    ap.add_argument("--sources", default="gt,ps2500",
                    help="label sources in figure A, in bar order: gt, ps252, ps2500")
    ap.add_argument("--pseudo", default="ps2500", choices=["ps252", "ps2500"],
                    help="which pseudo arm figure B draws")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--src_dir", default=str(SRC_DIR),
                    help="directory holding the per-seed cka_spatial_gt_vs_pseudo_s*.json "
                         "(this script chdir's to the repo root, so a relative path is "
                         "resolved from there)")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in SOURCE]
    if bad:
        sys.exit(f"unknown source(s) {bad}; choose from {list(SOURCE)}")

    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    SRC_DIR = Path(a.src_dir)
    stacks, nseed = load(seeds)
    st = Stats(stacks)

    print(f"seeds = {seeds}   sources = {sources}   pseudo panel = {a.pseudo}")
    print(f"{'condition':12s} {'mean_diag':>10} {'sd':>7}   per-stage diagonal")
    for k in ["random"] + [f"{p}-{o}" for p in ["gt", "ps252", "ps2500"] for o, _ in OBJ]:
        if k in st.md:
            print(f"{k:12s} {st.md[k]:10.4f} {st.md_sd[k]:7.4f}   {np.round(st.diag[k], 3)}")

    fig_bars(st, sources, nseed, a.tag)
    fig_spatial(st, a.pseudo, nseed, a.tag)

    dump = [["figure", "condition", "quantity", "cell", "mean", "sd", "n_seeds"]]
    for k in st.md:
        dump.append(["A", k, "mean_diag", "-", f"{st.md[k]:.4f}", f"{st.md_sd[k]:.4f}", nseed])
        for i in range(4):
            dump.append(["A", k, "diag", f"S{i}", f"{st.diag[k][i]:.4f}",
                         f"{st.diag_sd[k][i]:.4f}", nseed])
        for i in range(4):
            for j in range(4):
                dump.append(["B", k, "cka", f"T{i}xS{j}", f"{st.M[k][i, j]:.4f}", "", nseed])
    with open(OUT / f"fig_cka_values{a.tag}.csv", "w", newline="") as f:
        csv.writer(f).writerows(dump)
    print(f"-> {OUT}/fig_cka_values{a.tag}.csv  ({len(dump)-1} rows)")


if __name__ == "__main__":
    main()
