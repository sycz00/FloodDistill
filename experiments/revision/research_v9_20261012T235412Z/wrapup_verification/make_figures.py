#!/usr/bin/env python3
"""Figures for the JSTARS reviewer wrap-up (12 Oct 2026).

Read-only: every input is an existing artifact (paper source data, revision
tables, raw device logs, pool statistics). Writes PNGs to figures/ and the
plotted numbers to figure_data/ (the table view of each figure).

Run:  python3 -I make_figures.py
"""
import csv
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = "/home/schmalstieg/repos/FloodDistill"
REV = f"{REPO}/experiments/revision"
PSD = f"{REPO}/paper_source_data"
OLD_DEVICE = "/home/schmalstieg/repos/OLD/FloodDistill_v2/paper_source_data/Table_III_Deployment_Chain/raw_device"
POOL = "/data1/schmalstieg/flooddistill/data/pseudo_gdacs_v2"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DAT = os.path.join(HERE, "figure_data")

# Validated reference palette (dataviz skill, light mode), fixed entity colours.
C = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    ofa="#2a78d6",      # slot 1 blue   : student trained with OFA (the paper's model)
    mndwi="#eb6834",    # slot 2 orange : MNDWI > 0 spectral rule
    kd="#1baf7a",       # slot 3 aqua   : student trained with logit KD
    hard="#eda100",     # slot 4 yellow : student trained on hard masks only
    teacher="#0b0b0b",  # ink           : Prithvi-EO-2.0 teacher (reference)
    gray="#b9b8b1", gray2="#898781", gray3="#d6d5ce",
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "axes.linewidth": 0.8,
    "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.facecolor": C["surface"], "figure.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": C["ink"], "lines.solid_capstyle": "round",
})
BOX = dict(boxstyle="round,pad=0.15", fc=C["surface"], ec="none")


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, header, rows):
    with open(os.path.join(DAT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def title(ax, text, pad=6):
    ax.set_title(text, loc="left", color=C["ink"], fontweight="semibold", pad=pad)


def headline(fig, text, sub, y=1.02, ysub=0.975):
    fig.suptitle(text, x=0.01, ha="left", fontsize=10, fontweight="semibold", y=y)
    fig.text(0.01, ysub, sub, fontsize=8, color=C["ink2"], ha="left")


def dot(ax, x, y, color, hollow=False, size=6.5, marker="o", z=4):
    ax.plot([x], [y], marker=marker, ms=size, mfc=(C["surface"] if hollow else color),
            mec=color if hollow else C["surface"], mew=1.4 if hollow else 1.0,
            ls="none", zorder=z)


def signed(v, nd=1):
    s = f"{v:+.{nd}f}"
    return s.replace("+0.0", "0.0").replace("-0.0", "0.0") if abs(v) < 0.5 * 10 ** -nd else s


def line_handle(color, lw=1.2):
    return plt.Line2D([], [], color=color, lw=lw)


def dot_handle(color, hollow=False, marker="o", ms=6.5):
    return plt.Line2D([], [], color=color, marker=marker, ls="none", ms=ms,
                      mfc=(C["surface"] if hollow else color), mec=(color if hollow else C["surface"]),
                      mew=1.4 if hollow else 1.0)


# ---------------------------------------------------------------- shared data
SUMMARY = read_csv(f"{REV}/prompt5_closure/tables/current_full_summary.csv")
PROT = {"s11": "S11_TEST512_LEGACY", "sturm": "P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1",
        "wf": "P3_WF_NATIVE_ZERO_PAD512_V1"}
DSNAME = {"s11": "Sen1Floods11 test (in-distribution)", "sturm": "STURM-Flood (external)",
          "wf": "WorldFloods-v2 (external)"}


def summ(arch, variant, budget, ds, metric, protocol=None):
    protocol = protocol or PROT[ds]
    for r in SUMMARY:
        if (r["architecture"], r["variant"], r["budget"], r["dataset"], r["protocol"], r["metric"], r["domain"]) == \
                (arch, variant, budget, ds, protocol, metric, "full"):
            sd = r["sample_SD_ddof1"]
            return 100 * float(r["mean"]), (100 * float(sd) if sd not in ("", "nan", "None") else None)
    raise KeyError((arch, variant, budget, ds, metric, protocol))


TEACHER = ("teacher", "fixed_comparator", "0")
MNDWI = ("index:mndwi_zero", "fixed_comparator", "0")
STUDENT_OFA = ("efficientvit_b0", "historical_ofa_clip5_allvalid", "2500")


# ------------------------------------------------ fig 1: architectures (R1.2, R2.4)
def fig_architectures():
    rows = [
        ("EfficientViT-B0 · 0.7 M", "efficientvit_b0", "matched_focal_g2_clip5", "hard", "hard masks only"),
        ("EfficientViT-B0 · 0.7 M", "efficientvit_b0", "matched_lkd_clip5_valid", "kd", "logit KD"),
        ("EfficientViT-B0 · 0.7 M", "efficientvit_b0", "historical_ofa_clip5_allvalid", "ofa", "OFA"),
        ("MobileNetV3-Small · 3.5 M", "mobilenet_v3_small", "matched_lkd_clip5_valid", "kd", "logit KD"),
        ("MobileNetV3-Small · 3.5 M", "mobilenet_v3_small", "matched_ofa_clip5_valid", "ofa", "OFA"),
        ("LightFormer-ResNet18 · 11.4 M", "lightformer_resnet18", "matched_lkd_clip5_valid", "kd", "logit KD"),
        ("LightFormer-ResNet18 · 11.4 M", "lightformer_resnet18", "matched_ofa_clip5_valid", "ofa", "OFA"),
    ]
    ypos, y, prev = [], 0.0, None
    centers = defaultdict(list)
    for name, *_ in rows:
        if prev is not None and name != prev:
            y += 0.6
        ypos.append(y)
        centers[name].append(y)
        prev = name
        y += 0.42
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=True)
    table = []
    for ax, ds in zip(axes, ["s11", "sturm", "wf"]):
        t, _ = summ(*TEACHER, ds, "mIoU")
        m, _ = summ(*MNDWI, ds, "mIoU")
        vals = []
        for (name, arch, var, key, obj), yy in zip(rows, ypos):
            mu, sd = summ(arch, var, "2500", ds, "mIoU")
            vals += [mu - sd, mu + sd]
            ax.plot([mu - sd, mu + sd], [yy, yy], color=C[key], lw=1.6, zorder=3)
            dot(ax, mu, yy, C[key])
            table.append([ds, name, obj, f"{mu:.2f}", f"{sd:.2f}", f"{t:.2f}", f"{m:.2f}"])
        ax.axvline(t, color=C["teacher"], lw=1.1, zorder=1)
        ax.axvline(m, color=C["mndwi"], lw=1.1, zorder=1)
        ax.set_xlim(min(vals + [t, m]) - 0.6, max(vals + [t, m]) + 0.6)
        ax.set_ylim(ypos[-1] + 0.45, -0.45)
        ax.set_xlabel("mIoU (%)  ·  higher is better")
        title(ax, DSNAME[ds], pad=17)
        ax.text(0.0, 1.012, f"teacher {t:.1f}  ·  MNDWI {m:.1f}", transform=ax.transAxes, fontsize=7.4,
                color=C["ink2"], va="bottom", ha="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks([np.mean(v) for v in centers.values()])
    axes[0].set_yticklabels(list(centers.keys()), color=C["ink"])
    handles = [dot_handle(C[k]) for k in ("hard", "kd", "ofa")] + [line_handle(C["teacher"]), line_handle(C["mndwi"])]
    fig.legend(handles, ["hard teacher masks only (focal loss)", "+ soft teacher targets (logit KD)",
                         "+ OFA side branches", "teacher, 300 M (reference)", "MNDWI > 0 rule (reference)"],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=7.8)
    headline(fig, "Swapping the student architecture changes little: the six distilled students (3 architectures x 2 "
             "losses) lie within 1.3 points of each other outside Sen1Floods11",
             "Mean of 4 training seeds; bars = ±1 seed SD. Note the different x-ranges per panel.", y=1.05, ysub=0.995)
    save(fig, "fig01_architectures.png")
    write_csv("fig01_architectures.csv", ["benchmark", "student", "objective", "mIoU_mean_pct", "seed_sd_pct",
                                          "teacher_mIoU_pct", "mndwi_mIoU_pct"], table)


# ------------------------------------------------ fig 2: OFA minus KD (R2.3)
def fig_ofa():
    eff = read_csv(f"{REV}/prompt5_closure/tables/OFA_matched_effects.csv")
    order = [("efficientvit_b0", "100", "EfficientViT-B0, 100 scenes"),
             ("efficientvit_b0", "2500", "EfficientViT-B0, 2,500 scenes"),
             ("mobilenet_v3_small", "2500", "MobileNetV3-Small, 2,500"),
             ("lightformer_resnet18", "2500", "LightFormer-R18, 2,500")]
    fig = plt.figure(figsize=(11.2, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    y = 0
    yt, yl, table = [], [], []
    for ds in ["s11", "sturm", "wf"]:
        ax.text(-3.25, y - 0.55, DSNAME[ds], fontsize=8.2, color=C["ink"], fontweight="semibold", ha="left")
        for arch, budget, label in order:
            r = next(e for e in eff if e["architecture"] == arch and e["budget"] == budget and e["dataset"] == ds)
            mu, sd = float(r["OFA_minus_KD_pooled_mIoU_pp"]), float(r["paired_seed_SD_pp"])
            em, lo, hi = (float(r[k]) for k in ("event_macro_difference_pp", "event_CI95_low_pp", "event_CI95_high_pp"))
            ax.plot([mu - sd, mu + sd], [y - 0.13, y - 0.13], color=C["ofa"], lw=2.0, zorder=3)
            dot(ax, mu, y - 0.13, C["ofa"])
            ax.plot([lo, hi], [y + 0.13, y + 0.13], color=C["ink2"], lw=1.0, zorder=3)
            dot(ax, em, y + 0.13, C["ink2"], hollow=True, size=5.5, marker="D")
            yt.append(y)
            yl.append(label)
            table.append([ds, label, f"{mu:+.3f}", f"{sd:.3f}", f"{em:+.3f}", f"{lo:+.3f}", f"{hi:+.3f}", r["n_events"]])
            y += 1
        y += 0.9
    ax.axvline(0, color=C["ink"], lw=1.0, zorder=2)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7.6)
    ax.set_ylim(y - 0.6, -1.3)
    ax.set_xlim(-3.3, 2.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("OFA minus logit KD, mIoU (percentage points)\n← KD better        OFA better →")
    title(ax, "a  Matched comparison, 4 seeds each")
    ax.legend([dot_handle(C["ofa"]), dot_handle(C["ink2"], hollow=True, marker="D", ms=5.5)],
              ["all pixels pooled: mean of 4 paired seeds ± seed SD",
               "per flood event: mean with 95% interval (events resampled)"],
              loc="lower left", bbox_to_anchor=(0.0, -0.30), fontsize=7.5)
    curves = read_csv(f"{REV}/research_v8_20261012T123202Z/ofa_learning_curves.csv")
    runs = defaultdict(list)
    for r in curves:
        runs[r["run_id"]].append((int(r["optimizer_updates"]), 100 * float(r["source_validation_mIoU"])))
    ctab = []
    for k, (budget, ttl) in enumerate([("N2500", "b  Training curves, 2,500 scenes (EfficientViT-B0)"),
                                       ("N100", "c  Training curves, 100 scenes (EfficientViT-B0)")]):
        axc = fig.add_subplot(gs[k, 1])
        for obj, key, pat in [("kd", "kd", "efficientvit_b0_matched_lkd_clip5_valid_" + budget),
                              ("ofa", "ofa", ("efficientvit_b0_historical_ofa_clip5_allvalid_" if budget == "N2500"
                                              else "efficientvit_b0_matched_ofa_clip5_valid_") + budget)]:
            ids = sorted(i for i in runs if pat + "_s" in i)
            grid = defaultdict(list)
            for rid in ids:
                pts = sorted(runs[rid])
                axc.plot([p[0] for p in pts], [p[1] for p in pts], color=C[key], lw=0.7, alpha=0.35, zorder=2)
                for u, v in pts:
                    grid[u].append(v)
                    ctab.append([budget, obj, rid.split("_s")[-1], u, f"{v:.2f}"])
            us = sorted(u for u in grid if len(grid[u]) == len(ids))
            axc.plot(us, [st.mean(grid[u]) for u in us], color=C[key], lw=2.0, zorder=3,
                     label=("OFA" if obj == "ofa" else "logit KD") + " (mean of 4 seeds)")
        axc.set_ylabel("S11 validation mIoU (%)")
        axc.set_xlabel("optimizer updates")
        axc.set_ylim((80, 89) if budget == "N2500" else (40, 85))
        title(axc, ttl)
        axc.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("OFA versus plain logit distillation: differences stay within about ±1 point and change sign across "
                 "students and benchmarks; OFA does not learn faster either", x=0.01, ha="left", fontsize=10,
                 fontweight="semibold", y=1.0)
    save(fig, "fig02_ofa_vs_kd.png")
    write_csv("fig02a_ofa_minus_kd.csv", ["benchmark", "student_budget", "pooled_diff_pp", "paired_seed_sd_pp",
                                          "event_mean_diff_pp", "event_ci95_lo_pp", "event_ci95_hi_pp", "n_events"], table)
    write_csv("fig02bc_training_curves.csv", ["budget", "objective", "seed", "optimizer_updates", "s11_val_mIoU_pct"], ctab)


# ------------------------------------------------ helpers for fig 3/4/10
def device_logs():
    res = {}
    for s in ["42", "64", "91", "107"]:
        txt = open(f"{OLD_DEVICE}/N_both_s{s}", errors="replace").read()
        for sec in re.split(r"^=== ", txt, flags=re.M):
            head = sec.split("\n", 1)[0]
            if "Sen1" in head:
                ds = "s11val"
            elif "STURM" in head:
                ds = "sturm"
            elif "WorldFloods" in head:
                ds = "wf"
            else:
                continue
            for line in sec.split("\n"):
                mm = re.match(r"\s+(Float BN-folded|TRT-INT8-engine)\s+micro: mIoU=([0-9.]+) flood=([0-9.]+)", line)
                if mm:
                    res[(ds, "float" if mm.group(1).startswith("Float") else "int8", s)] = 100 * float(mm.group(2))
    return res


def selector_raw():
    out = {}
    for f in glob.glob(f"{PSD}/Supp_Figure_S1_Subset_Selection/raw/*__clean.json"):
        d = json.load(open(f))
        tp, fp, fn = [0, 0], [0, 0], [0, 0]
        for v in d["scene"].values():
            for c in (0, 1):
                tp[c] += v[c][0]
                fp[c] += v[c][1]
                fn[c] += v[c][2]
        iou = [tp[c] / (tp[c] + fp[c] + fn[c]) for c in (0, 1)]
        out.setdefault(d["family"], {})[d["seed"]] = 100 * sum(iou) / 2
    for r in read_csv(f"{PSD}/Supp_Figure_S1_Subset_Selection/fig_curation_decomp_n2500_values.csv"):
        if r["quantity"] == "gain_curve" and r["n_or_key"] == "g=1.0" and r["arm"] in ("v2cur_random_N2500",
                                                                                        "v2cur_random_raw_N2500"):
            out[r["arm"]] = {"mean": 100 * float(r["mean"]), "sd": 100 * float(r["sd_or_ci"])}
    return out


def mean_sd(d):
    if "mean" in d:
        return d["mean"], d["sd"]
    v = list(d.values())
    return st.mean(v), st.stdev(v)


# ------------------------------------------------ fig 3: component effects (R1.4)
def fig_components():
    t1 = {(r["model"], r["labels"], r["benchmark"], r["metric"]): (100 * float(r["mean"]), 100 * float(r["sd_ddof1"]))
          for r in read_csv(f"{PSD}/Table_I_Matched_Budget_N252/Table_I_values.csv")}
    t2 = {(r["model"], r["benchmark"]): (100 * float(r["mean"]), 100 * float(r["sd_ddof1"]))
          for r in read_csv(f"{PSD}/Table_II_Deployment_N2500/Table_II_values.csv")}
    t3 = {(r["stage"], r["benchmark"]): (100 * float(r["mean_micro_mIoU"]), 100 * float(r["sd_ddof1"]))
          for r in read_csv(f"{PSD}/Table_III_Deployment_Chain/Table_III_values.csv")}
    b1 = {(r["cell"], r["arm"], r["benchmark"], r["metric"]): (100 * fnum(r["mean"]), 100 * fnum(r["std"]))
          for r in read_csv(f"{PSD}/Supp_InText_Statistics/S3_harmonisation/B1_geometry_control.csv")}
    paired = read_csv(f"{REV}/prompt4_report_20261007/tables/original_paired_seed_summary.csv")
    sel = selector_raw()
    dev = device_logs()

    def diff(a, b):  # unpaired arms: combined SD as a rough spread
        return a[0] - b[0], math.sqrt(a[1] ** 2 + b[1] ** 2)

    def pair(contrast, arch, ds, cond):
        for r in paired:
            if r["contrast"] == contrast and r["architecture"] == arch and r["dataset"] == ds \
                    and r["condition"] == cond and r["metric"] == "mIoU":
                return 100 * float(r["mean"]), 100 * float(r["sample_SD_ddof1"])
        raise KeyError((contrast, arch, ds, cond))

    of = "EfficientViT-B0, OFA (0.7M)"
    st_c = "P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1:clean"
    wf_c = "P3_WF_NATIVE_ZERO_PAD512_V1"
    rows = [
        ("Teacher labels on GDACS images\ninstead of human labels on S11 (252 scenes)", False, {
            "s11": diff(t1[(of, "pseudo", "S11 test", "mIoU")], t1[(of, "Sen1Floods11", "S11 test", "mIoU")]),
            "sturm": diff(t1[(of, "pseudo", "STURM-Flood", "mIoU")], t1[(of, "Sen1Floods11", "STURM-Flood", "mIoU")]),
            "wf": diff(t1[(of, "pseudo", "WorldFloods-v2", "mIoU")], t1[(of, "Sen1Floods11", "WorldFloods-v2", "mIoU")]),
        }, "images, places and tile size change too"),
        ("same, S11 chips cropped to GDACS tile size", False, {
            "sturm": diff(b1[("OFA", "pseudo 2.64km", "STURM-Flood", "micro_mIoU")],
                          b1[("OFA", "S11 cropped 2.64km", "STURM-Flood", "micro_mIoU")]),
            "wf": diff(b1[("OFA", "pseudo 2.64km", "WorldFloods-v2 flood-only", "micro_mIoU")],
                       b1[("OFA", "S11 cropped 2.64km", "WorldFloods-v2 flood-only", "micro_mIoU")]),
        }, "the crop removes 74% of the S11 water"),
        ("10x more teacher-labelled scenes\n(252 → 2,500)", True, {
            "s11": diff(t2[("EfficientViT-B0, OFA (ours)", "S11")], t1[(of, "pseudo", "S11 test", "mIoU")]),
            "sturm": diff(t2[("EfficientViT-B0, OFA (ours)", "STURM")], t1[(of, "pseudo", "STURM-Flood", "mIoU")]),
            "wf": diff(t2[("EfficientViT-B0, OFA (ours)", "WF-fo")], t1[(of, "pseudo", "WorldFloods-v2", "mIoU")]),
        }, "number of scenes"),
        ("Pre-filter on (with FPS selection)", True, {
            "sturm": diff(mean_sd(sel["v2cur_fps_N2500"]), mean_sd(sel["v2cur_fps_raw_N2500"]))}, "STURM only"),
        ("Pre-filter on (with random selection)", True, {
            "sturm": diff(mean_sd(sel["v2cur_random_N2500"]), mean_sd(sel["v2cur_random_raw_N2500"]))}, "STURM only"),
        ("FPS instead of random selection\n(both after the pre-filter)", True, {
            "sturm": diff(mean_sd(sel["v2cur_fps_N2500"]), mean_sd(sel["v2cur_random_N2500"]))}, "STURM only"),
        ("Soft teacher targets (logit KD)\nvs hard teacher masks only", True, {
            "s11": pair("weighted_KD_minus_full_weight_focal", "efficientvit_b0", "s11", "clean"),
            "sturm": pair("weighted_KD_minus_full_weight_focal", "efficientvit_b0", "sturm", st_c),
            "wf": pair("weighted_KD_minus_full_weight_focal", "efficientvit_b0", "wf", wf_c),
        }, "hard-loss weight halves at the same time"),
        ("OFA side branches vs plain logit KD", True, {
            "s11": pair("OFA_minus_clipping_matched_KD", "efficientvit_b0", "s11", "clean"),
            "sturm": pair("OFA_minus_clipping_matched_KD", "efficientvit_b0", "sturm", st_c),
            "wf": pair("OFA_minus_clipping_matched_KD", "efficientvit_b0", "wf", wf_c),
        }, "—"),
        ("Hardswish → ReLU6 activation\n(retrained from scratch)", False, {
            "sturm": diff(t3[("+ ReLU6", "STURM-Flood")], t3[("EfficientViT-B0, OFA", "STURM-Flood")]),
            "wf": diff(t3[("+ ReLU6", "WorldFloods-v2")], t3[("EfficientViT-B0, OFA", "WorldFloods-v2")]),
        }, "loses the ImageNet starting point"),
    ]
    q = {}
    for ds in ("sturm", "wf"):
        d = [dev[(ds, "int8", s)] - dev[(ds, "float", s)] for s in ("42", "64", "91", "107")]
        q[ds] = (st.mean(d), st.stdev(d))
    rows.append(("8-bit engine on the Jetson\nvs float export (same device)", True, q, "one seed drives most of the loss"))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.6), sharey=True)
    table = []
    for ax, ds in zip(axes, ["s11", "sturm", "wf"]):
        for i, (label, ctrl, vals, also) in enumerate(rows):
            if ds not in vals:
                ax.text(0, i, "not measured", ha="center", va="center", fontsize=6.8, color=C["muted"], bbox=BOX, zorder=5)
                continue
            mu, sd = vals[ds]
            ax.plot([mu - sd, mu + sd], [i, i], color=C["ink2"], lw=1.3, zorder=3)
            dot(ax, mu, i, C["ink"], hollow=not ctrl, size=6.5)
            ax.text(mu, i - 0.27, signed(mu), ha="center", va="bottom", fontsize=6.8, color=C["ink2"], bbox=BOX, zorder=5)
            table.append([label.replace("\n", " "), "one factor" if ctrl else "confounded", ds, f"{mu:+.3f}", f"{sd:.3f}",
                          also])
        ax.axvline(0, color=C["ink"], lw=1.0, zorder=2)
        ax.set_xlim(-4.6, 4.6)
        ax.set_ylim(len(rows) - 0.4, -0.7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("change in mIoU (percentage points)")
        title(ax, DSNAME[ds])
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7.4, color=C["ink"])
    fig.legend([dot_handle(C["ink"]), dot_handle(C["ink"], hollow=True)],
               ["one thing changed (fair comparison)", "other things changed too (read with care)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.04), fontsize=8)
    headline(fig, "What each pipeline step is worth, measured one comparison at a time (do not add the rows up)",
             "Each row is its own controlled comparison, top to bottom in pipeline order. Whiskers: ±1 SD over 4 seeds "
             "(for unpaired arms the two SDs are combined). Selector rows exist for STURM only.")
    save(fig, "fig03_component_effects.png")
    write_csv("fig03_component_effects.csv", ["component", "comparison_type", "benchmark", "diff_mIoU_pp", "spread_sd_pp",
                                              "also_changes"], table)


# ------------------------------------------------ fig 4: filter x selector (R1.4, R1.6)
def fig_filter():
    sel = selector_raw()
    pairs = [("Random", "v2cur_random_N2500", "v2cur_random_raw_N2500"),
             ("FPS (deployed)", "v2cur_fps_N2500", "v2cur_fps_raw_N2500"),
             ("GLISTER", "v2cur_glister_N2500", "v2cur_glister_raw_N2500"),
             ("Confidence ranking", "v2cur_confidence_N2500", "v2cur_confidence_raw_N2500"),
             ("R2-Core", "v2cur_r2core_N2500", None)]
    ref = mean_sd(sel["ncgt_evit_ofa_N252"])
    teacher_legacy, mndwi_legacy = 75.50, 74.67   # paper's STURM protocol (Table II / Supp. S4)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.2), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
    table = []
    for ax, zoom in zip(axes, [False, True]):
        xlo = 74.6 if zoom else 36
        for i, (name, a, b) in enumerate(pairs):
            ma, sa = mean_sd(sel[a])
            mb = sb = None
            if b:
                mb, sb = mean_sd(sel[b])
                if mb >= xlo:
                    ax.plot([mb, ma], [i, i], color=C["gray"], lw=2.2, zorder=2)
                    ax.plot([mb - sb, mb + sb], [i, i], color=C["ink2"], lw=0.9, zorder=3)
                    dot(ax, mb, i, C["ink2"], hollow=True)
                else:
                    ax.text(xlo + 0.03, i - 0.3, f"without filter: {mb:.1f} (off scale)", fontsize=6.8, color=C["ink2"])
            ax.plot([ma - sa, ma + sa], [i, i], color=C["ofa"], lw=1.2, zorder=3)
            dot(ax, ma, i, C["ofa"])
            if not zoom:
                table.append([name, f"{ma:.2f}", f"{sa:.2f}", f"{mb:.2f}" if b else "", f"{sb:.2f}" if b else ""])
        ax.axvline(teacher_legacy, color=C["teacher"], lw=1.0, zorder=1)
        ax.axvline(mndwi_legacy, color=C["mndwi"], lw=1.0, zorder=1)
        ax.axvline(ref[0], color=C["gray2"], lw=1.0, zorder=1)
        ax.set_ylim(len(pairs) - 0.5, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("STURM-Flood mIoU (%), clean images")
        if zoom:
            ax.set_xlim(74.6, 76.3)
            title(ax, "b  Zoom: with the filter, the selection rule barely matters")
        else:
            ax.set_xlim(36, 79)
            title(ax, "a  Without the pre-filter, confidence ranking collapses")
    axes[0].set_yticks(range(len(pairs)))
    axes[0].set_yticklabels([p[0] for p in pairs], color=C["ink"])
    fig.legend([dot_handle(C["ofa"]), dot_handle(C["ink2"], hollow=True), line_handle(C["teacher"]),
                line_handle(C["mndwi"]), line_handle(C["gray2"])],
               ["with pre-filter", "without pre-filter (whole 23,720-tile pool)", "teacher", "MNDWI > 0",
                "student trained on 252 human labels"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12),
               fontsize=7.8)
    headline(fig, "The pre-filter is a safety net for bad selection rules; with FPS it changes almost nothing",
             "2,500 selected tiles, EfficientViT-B0 OFA, 4 seeds (mean ± SD), recomputed from the raw per-scene counts of "
             "the paper's selector study (Supp. Fig. S1; STURM only).", y=1.06, ysub=0.985)
    save(fig, "fig04_filter_selector.png")
    write_csv("fig04_filter_selector.csv", ["selector", "with_filter_mIoU_pct", "with_filter_sd", "no_filter_mIoU_pct",
                                            "no_filter_sd"], table)


# ------------------------------------------------ fig 5: pool funnel (R1.6)
def pool_counts():
    rows = read_csv(f"{POOL}/pool_features.csv")
    ff = [float(r["flood_frac"]) for r in rows]
    cs = [float(r["conf_std"]) for r in rows]
    ev = [r["is_flood_scene"] == "1" for r in rows]
    n = len(rows)
    dry = [i for i in range(n) if ff[i] < 0.05]
    wet = [i for i in range(n) if ff[i] > 0.85]
    inr = [i for i in range(n) if 0.05 <= ff[i] <= 0.85]
    flat = [i for i in inr if cs[i] < 0.05]
    cand = [i for i in inr if cs[i] >= 0.05]
    sel = json.load(open(f"{POOL}/diversity_index.json"))
    assert sel["n_total"] == n and sel["n_candidates"] == len(cand) and sel["n_curated"] == 2500
    return dict(n=n, dry=len(dry), zero=sum(1 for i in dry if ff[i] == 0), dry_pre=sum(1 for i in dry if not ev[i]),
                wet=len(wet), flat=len(flat), cand=len(cand), selected=2500)


def fig_funnel():
    p = pool_counts()
    n = p["n"]
    fig, ax = plt.subplots(figsize=(11.2, 2.9))
    segs1 = [("dropped: teacher sees < 5% water", p["dry"], C["ofa"], "#ffffff"),
             ("> 85% water", p["wet"], C["gray"], C["ink"]),
             ("too uniform", p["flat"], C["gray3"], C["ink"]),
             ("candidates", p["cand"], C["gray2"], "#ffffff")]
    segs2 = [("selected by FPS", 2500, C["gray2"], "#ffffff"), ("not selected", p["cand"] - 2500, C["gray3"], C["ink"])]
    for y, segs, total in [(0, segs1, n), (1, segs2, p["cand"])]:
        x = 0
        for name, v, col, tc in segs:
            ax.barh(y, v - 60, left=x + 30, height=0.5, color=col, zorder=3)
            if v > 2300:
                ax.text(x + v / 2, y, f"{name}\n{v:,} ({100 * v / total:.0f}%)", ha="center", va="center", fontsize=7.6,
                        color=tc, zorder=4)
            x += v
    x0 = p["dry"] + (p["wet"] + p["flat"]) / 2
    ax.annotate(f"small grey blocks: > 85% water ({p['wet']:,}) and too uniform a teacher confidence ({p['flat']:,})",
                xy=(x0, 0.27), xytext=(x0 - 9000, 0.62), fontsize=7.2, color=C["ink2"],
                arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"whole GDACS pool\n{n:,} tiles", f"candidates\n{p['cand']:,} tiles"], color=C["ink"])
    ax.set_ylim(1.55, -0.55)
    ax.set_xlim(0, n)
    ax.set_xlabel("number of tiles")
    ax.grid(axis="y", visible=False)
    ax.text(0, -0.47, f"Of the {p['dry']:,} near-dry tiles, {p['zero']:,} contain no predicted water at all and "
            f"{p['dry_pre']:,} are pre-event images. None of them can enter training.", fontsize=7.8, color=C["ink"],
            ha="left")
    headline(fig, "Before training, 68% of the image pool is discarded for being dry or nearly dry, so the student "
             "never sees an ordinary water-free scene",
             "Counts recomputed from the stored teacher statistics of every pool tile (pool_features.csv) and the saved "
             "selection index.", y=1.07, ysub=0.99)
    save(fig, "fig05_pool_funnel.png")
    write_csv("fig05_pool_funnel.csv", ["stage", "tiles"], [
        ["whole pool", n], ["dropped: teacher water fraction < 5%", p["dry"]],
        ["  of which: zero predicted water", p["zero"]], ["  of which: pre-event images", p["dry_pre"]],
        ["dropped: water fraction > 85%", p["wet"]], ["dropped: confidence spread < 0.05", p["flat"]],
        ["candidates", p["cand"]], ["selected by FPS", 2500]])


# ------------------------------------------------ fig 6: precision / recall (R2.5)
def fig_pr():
    models = [("ofa", STUDENT_OFA, "o", 8, "student, OFA"),
              ("kd", ("efficientvit_b0", "matched_lkd_clip5_valid", "2500"), "D", 6.5, "student, logit KD"),
              ("mndwi", MNDWI, "o", 8, "MNDWI > 0 rule"),
              ("teacher", TEACHER, "s", 9, "teacher (300 M)")]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))
    table = []
    for ax, ds in zip(axes, ["s11", "sturm", "wf"]):
        lo = {"s11": 60, "sturm": 65, "wf": 60}[ds]
        rr = np.linspace(lo, 100, 400)
        for iou in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            R = rr / 100
            P = iou * R / (R * (1 + iou) - iou)   # equal-IoU curve: IoU = PR / (P + R - PR)
            ok = (P > 0) & (100 * P <= 100.5) & (100 * P >= lo)
            ax.plot(rr[ok], 100 * P[ok], color=C["grid"], lw=0.9, zorder=1)
            if 100 * iou >= lo:
                ax.text(100.4, 100 * iou, f"IoU {iou:.2f}", fontsize=6.3, color=C["muted"], va="center", ha="left")
        lines = []
        for key, ident, mk, ms, lab in models:
            p, _ = summ(*ident, ds, "water_precision")
            r, _ = summ(*ident, ds, "water_recall")
            w, _ = summ(*ident, ds, "water_IoU")
            if key == "teacher":
                ax.plot([r], [p], marker=mk, ms=ms, mfc="none", mec=C["teacher"], mew=1.6, ls="none", zorder=6)
            else:
                ax.plot([r], [p], marker=mk, ms=ms, mfc=C[key], mec=C["surface"], mew=1.0, ls="none", zorder=4)
            lines.append(f"{lab}: P {p:.0f} · R {r:.0f} · IoU {w:.1f}")
            table.append([ds, lab, f"{p:.2f}", f"{r:.2f}", f"{w:.2f}"])
        ax.text(0.03, 0.03, "\n".join(lines), transform=ax.transAxes, fontsize=6.7, color=C["ink2"], va="bottom",
                ha="left", bbox=dict(boxstyle="round,pad=0.3", fc=C["surface"], ec=C["grid"]))
        ax.set_xlim(lo, 100)
        ax.set_ylim(lo, 100.5)
        ax.set_xlabel("recall: share of true water found (%)")
        ax.set_ylabel("precision: share of predicted water that is real (%)")
        ax.grid(False)
        title(ax, DSNAME[ds])
    handles = [dot_handle(C["ofa"]), dot_handle(C["kd"], marker="D", ms=6), dot_handle(C["mndwi"]),
               plt.Line2D([], [], marker="s", ls="none", ms=8, mfc="none", mec=C["teacher"], mew=1.6)]
    fig.legend(handles, ["student, OFA (EfficientViT-B0, 2,500 scenes, 4 seeds)", "student, logit KD",
                         "MNDWI > 0 rule", "teacher (300 M)"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=7.8)
    headline(fig, "The learned models and the MNDWI rule make different mistakes: the rule finds more water, the "
             "learned models raise fewer false water alarms",
             "Water class only, all pixels pooled. Grey curves join points of equal water IoU; on the external sets all "
             "four end up on almost the same curve, but at very different precision/recall balances.", y=1.06, ysub=0.99)
    fig.tight_layout()
    save(fig, "fig06_precision_recall.png")
    write_csv("fig06_precision_recall.csv", ["benchmark", "model", "water_precision_pct", "water_recall_pct",
                                             "water_IoU_pct"], table)


# ------------------------------------------------ fig 7: per-event MNDWI vs learned (R2.5)
def event_tables():
    ev = defaultdict(dict)
    wpx = {}
    for r in read_csv(f"{REV}/capacity_v7_20261011T210730Z/event_results.csv"):
        if r["domain"] != "full" or r["family"] not in ("teacher", "MNDWI>0") or r["dataset"] not in ("sturm", "wf"):
            continue
        if r["water_IoU"] in ("", "nan"):
            continue
        ev[(r["dataset"], r["family"])][r["event_id"]] = float(r["water_IoU"])
        wpx[(r["dataset"], r["event_id"])] = int(float(r["water_TP"])) + int(float(r["water_FN"]))
    stu = defaultdict(list)
    target = {"sturm": "P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1:clean", "wf": "P3_WF_NATIVE_ZERO_PAD512_V1"}
    with open(f"{REV}/prompt4_report_20261007/cross_cohort_common_protocols/confusion_counts.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["level"] != "event" or r["variant"] != "historical_ofa_clip5_allvalid_original_data":
                continue
            if r["dataset"] in target and r["condition"] == target[r["dataset"]] and r["water_IoU"] not in ("", "nan"):
                stu[(r["dataset"], r["unit"])].append(100 * float(r["water_IoU"]))
    return ev, wpx, stu


def fig_events():
    ev, wpx, stu = event_tables()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    table = []
    nudge = {("wf", "EMSR342"): (4.0, -1.5), ("sturm", "EMSR501"): (1.5, 2.0)}
    for ax, ds in zip(axes, ["sturm", "wf"]):
        T, M = ev[(ds, "teacher")], ev[(ds, "MNDWI>0")]
        events = sorted(set(T) & set(M))
        n_t = n_s = n_both = 0
        wmax = max(wpx[(ds, k)] for k in events)
        for e in events:
            s = st.mean(stu[(ds, e)]) if len(stu.get((ds, e), [])) == 4 else None
            size = 4 + 10 * math.sqrt(wpx[(ds, e)] / wmax)
            ax.plot([M[e]], [T[e]], "o", ms=size, mfc=C["teacher"], mec=C["surface"], mew=1.0, alpha=0.85, zorder=3)
            if s is not None:
                ax.plot([M[e]], [s], "o", ms=size, mfc=C["ofa"], mec=C["surface"], mew=1.0, alpha=0.85, zorder=4)
                ax.plot([M[e], M[e]], [T[e], s], color=C["gray"], lw=0.8, zorder=2)
                n_s += s > M[e]
                n_both += 1
            n_t += T[e] > M[e]
            table.append([ds, e, wpx[(ds, e)], f"{M[e]:.2f}", f"{T[e]:.2f}", f"{s:.2f}" if s is not None else ""])
            if (ds == "wf" and (abs(T[e] - M[e]) > 18 or wpx[(ds, e)] == wmax)) or (ds == "sturm" and abs(T[e] - M[e]) > 14):
                dx, dy = nudge.get((ds, e), (1.2, 1.0))
                ax.text(M[e] + dx, T[e] + dy, e, fontsize=6.8, color=C["ink2"], bbox=BOX, zorder=6)
        ax.plot([0, 100], [0, 100], color=C["ink2"], lw=0.9, zorder=1)
        ax.text(4, 92, "learned model better ↑", fontsize=7.5, color=C["ink2"])
        ax.text(70, 6, "MNDWI better ↓", fontsize=7.5, color=C["ink2"])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("MNDWI > 0: water IoU per flood event (%)")
        ax.set_ylabel("learned model: water IoU per flood event (%)")
        title(ax, f"{DSNAME[ds]} · {len(events)} events\nbeats MNDWI: teacher on {n_t}, student on {n_s} of {n_both}")
    fig.legend([dot_handle(C["teacher"], ms=7), dot_handle(C["ofa"], ms=7)],
               ["teacher (300 M)", "student (EfficientViT-B0 OFA, mean of 4 seeds)"], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    headline(fig, "Flood by flood: on STURM the learned models and the rule split the events almost evenly; on "
             "WorldFloods the teacher wins 8 of 11, some by a wide margin",
             "Each pair of dots is one flood event (teacher and student share the same x). Dot area ∝ amount of true "
             "water in the event. Above the diagonal: learned model better than MNDWI.")
    save(fig, "fig07_events_vs_mndwi.png")
    write_csv("fig07_events_vs_mndwi.csv", ["benchmark", "event", "reference_water_pixels", "mndwi_water_IoU_pct",
                                            "teacher_water_IoU_pct", "student_ofa_water_IoU_pct_mean4"], table)


# ------------------------------------------------ fig 8: perturbations (R2.5)
def fig_perturb():
    rob = read_csv(f"{REV}/prompt4_report_20261007/tables/robustness_all_conditions_metrics_and_drops.csv")
    val = {}
    for r in rob:
        if r["metric"] == "mIoU":
            val[(r["architecture"], r["variant"], r["dataset"], r["condition"])] = 100 * float(r["mean"])
    comps = [("teacher", "fixed_comparator", "teacher", "teacher (300 M)"),
             ("efficientvit_b0", "historical_ofa_clip5_allvalid", "ofa", "student, OFA"),
             ("index:mndwi_zero", "fixed_comparator", "mndwi", "MNDWI > 0")]
    fams = [("haze", "Haze added (strength h)", [0, 0.05, 0.10, 0.20], lambda v: "clean" if v == 0 else f"h{v:.2f}"),
            ("tilt", "Visible ×(1+d), infrared ×(1−d)", [-30, -20, -10, -5, 0, 5, 10, 20, 30],
             lambda v: "clean" if v == 0 else f"t1tilt{'+' if v > 0 else '-'}{abs(v):02d}"),
            ("vis", "Visible bands only ×(1+d)", [-30, -20, -10, -5, 0, 5, 10, 20, 30],
             lambda v: "clean" if v == 0 else f"t1vis{'+' if v > 0 else '-'}{abs(v):02d}"),
            ("gain", "All bands ×g (brightness)", [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
             lambda v: "clean" if v == 1.0 else f"g{v:.2f}")]
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.4), sharey="row")
    table = []
    for row, (ds, pre, dname) in enumerate([("sturm", "P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1:", "STURM-Flood"),
                                           ("wf", "", "WorldFloods-v2")]):
        for col, (fk, ftitle, xs, cname) in enumerate(fams):
            ax = axes[row, col]
            for arch, var, key, lab in comps:
                ys = [val.get((arch, var, ds, pre + cname(x))) for x in xs]
                xx = [x for x, yv in zip(xs, ys) if yv is not None]
                yy = [yv for yv in ys if yv is not None]
                ax.plot(xx, yy, color=C[key], lw=1.8, zorder=3, label=lab)
                ax.plot(xx[-1:], yy[-1:], "o", ms=5, mfc=C[key], mec=C["surface"], zorder=4)
                for x, yv in zip(xx, yy):
                    table.append([dname, fk, x, lab, f"{yv:.2f}"])
            if fk == "gain":
                ax.axvline(1.0, color=C["axis"], lw=0.9)
            elif fk != "haze":
                ax.axvline(0, color=C["axis"], lw=0.9)
            if row == 0:
                title(ax, ftitle)
            if col == 0:
                ax.set_ylabel(f"{dname}\nmIoU (%)")
            ax.set_xlabel({"haze": "h", "tilt": "d (%)", "vis": "d (%)", "gain": "g"}[fk])
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    headline(fig, "Synthetic stress tests: learned models are far more robust to haze; MNDWI is more robust when the "
             "visible bands get darker relative to the infrared (WorldFloods)",
             "Same images with a computed perturbation applied before the model sees them. Student = EfficientViT-B0 OFA, "
             "2,500 scenes, mean of 4 seeds. STURM: corrected finite-pixel protocol; WorldFloods: legacy protocol.")
    fig.tight_layout()
    save(fig, "fig08_perturbations.png")
    write_csv("fig08_perturbations.csv", ["benchmark", "perturbation", "severity", "model", "mIoU_pct"], table)


# ------------------------------------------------ fig 9: model size ladder (R1.2, R2.6, Discussion)
def fig_ladder():
    v5 = {r["label"]: r for r in read_csv(f"{REV}/capacity_v4_20261008/v5/analysis/cells.csv")}
    v7 = defaultdict(dict)
    for r in read_csv(f"{REV}/capacity_v7_20261011T210730Z/analysis_20261012/external_selected_summary.csv"):
        if r["domain"] == "full":
            v7[r["family"]][r["dataset"]] = r
    ev, _, stu = event_tables()
    t_ev = st.mean(ev[("wf", "teacher")].values())
    m_ev = st.mean(ev[("wf", "MNDWI>0")].values())
    s_ev = st.mean(st.mean(v) for (ds, _e), v in stu.items() if ds == "wf" and len(v) == 4)
    teacher = {"s11": summ(*TEACHER, "s11", "water_IoU")[0], "sturm": summ(*TEACHER, "sturm", "water_IoU")[0],
               "wf": summ(*TEACHER, "wf", "water_IoU")[0], "wf_ev": t_ev}
    mnd = {"s11": summ(*MNDWI, "s11", "water_IoU")[0], "sturm": summ(*MNDWI, "sturm", "water_IoU")[0],
           "wf": summ(*MNDWI, "wf", "water_IoU")[0], "wf_ev": m_ev}
    paper = {"s11": summ(*STUDENT_OFA, "s11", "water_IoU")[0], "sturm": summ(*STUDENT_OFA, "sturm", "water_IoU")[0],
             "wf": summ(*STUDENT_OFA, "wf", "water_IoU")[0], "wf_ev": s_ev}
    pts = []
    for lab, key in [("R1", "R1 LKD @sel lr"), ("R2", "R2 LKD @sel lr"), ("R3", "R3 LKD @sel lr")]:
        r = v5[key]
        pts.append((lab, int(r["params"]), float(r["s11_wIoU_mean"]), float(r["sturm_wIoU_mean"]), float(r["wf_wIoU_mean"]),
                    t_ev + float(r["wf_wIoU_minus_teacher_event_mean"])))
    for lab, fam, p in [("R4", "R4_LKD", 12230), ("A3N", "A3N_LKD", 20250), ("A3", "A3_LKD", 30922),
                        ("A1", "A1_LKD_reanchored", 190870), ("A0", "A0_LKD_reanchored", 704714)]:
        d = v7[fam]
        pts.append((lab, p, float(d["s11"]["water_IoU_mean_pct"]), float(d["sturm"]["water_IoU_mean_pct"]),
                    float(d["wf"]["water_IoU_mean_pct"]), float(d["wf"]["event_macro_water_IoU_mean_pct"])))
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.7), sharex=True)
    panels = [("s11", "Sen1Floods11 test", 2), ("sturm", "STURM-Flood", 3), ("wf", "WorldFloods-v2, all pixels", 4),
              ("wf_ev", "WorldFloods-v2, per flood event", 5)]
    for ax, (k, ttl, idx) in zip(axes, panels):
        for p in pts:
            dot(ax, p[1], p[idx], C["ink2"], size=6)
        dot(ax, 7.0e5, paper[k], C["ofa"], size=7, marker="D", z=5)
        dot(ax, 3.0e8, teacher[k], C["teacher"], size=7.5, marker="s")
        ax.axhline(teacher[k], color=C["teacher"], lw=0.9, zorder=1)
        ax.axhline(mnd[k], color=C["mndwi"], lw=1.1, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(6, 1.2e9)
        title(ax, ttl)
        ax.set_xlabel("parameters (log scale)")
        spec = {"s11": {"R1": ("below", "R1"), "R2": ("right", "R2")},
                "wf_ev": {"R1": ("right", "R1"), "R2": ("below", "R2"), "R3": ("above", "R3"), "R4": ("below", "R4"),
                          "A3": ("below", "A3"), "A0": ("below", "A0")}}.get(k, {})
        ylo, yhi = ax.get_ylim()
        off = 0.035 * (yhi - ylo)
        for p in pts:
            if p[0] in spec:
                where, text = spec[p[0]]
                if where == "right":
                    ax.text(p[1] * 2.2, p[idx], text, fontsize=6.8, color=C["ink2"], ha="left", va="center", bbox=BOX,
                            zorder=6)
                else:
                    up = where == "above"
                    ax.text(p[1], p[idx] + (off if up else -off), text, fontsize=6.8, color=C["ink2"], ha="center",
                            va="bottom" if up else "top", bbox=BOX, zorder=6)
    axes[0].set_ylabel("water IoU (%)")
    fig.legend([dot_handle(C["ink2"], ms=6), dot_handle(C["ofa"], marker="D", ms=7),
                plt.Line2D([], [], marker="s", ls="-", lw=0.9, ms=7, color=C["teacher"], mec=C["surface"]),
                line_handle(C["mndwi"])],
               ["capacity ladder: R1 14 · R2 418 · R3 1,058 · R4 12,230 · A3N 20k · A3 31k · A1 191k · A0 705k parameters",
                "the paper's student (EfficientViT-B0, ImageNet start, OFA)", "teacher (300 M)", "MNDWI > 0 rule"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16), fontsize=7.6)
    headline(fig, "How small can the student be? On Sen1Floods11 a 418-parameter pixel model already reaches the teacher; "
             "on WorldFloods, flood by flood, every student stays below it",
             "Capacity ladder from the follow-up study: scratch-trained ReLU6 graphs, logit KD, learning rate chosen on "
             "S11 validation, 4 seeds each. Different recipe from the paper's student (blue).", y=1.06, ysub=0.99)
    fig.tight_layout()
    save(fig, "fig09_size_ladder.png")
    write_csv("fig09_size_ladder.csv", ["model", "parameters", "s11_test_water_IoU_pct", "sturm_water_IoU_pct",
                                        "wf_water_IoU_pct", "wf_event_mean_water_IoU_pct"],
              [[p[0], p[1], f"{p[2]:.2f}", f"{p[3]:.2f}", f"{p[4]:.2f}", f"{p[5]:.2f}"] for p in pts]
              + [["paper student EfficientViT-B0 OFA (pretrained)", 704714] +
                 [f"{paper[k]:.2f}" for k in ("s11", "sturm", "wf", "wf_ev")],
                 ["teacher", 300_000_000] + [f"{teacher[k]:.2f}" for k in ("s11", "sturm", "wf", "wf_ev")],
                 ["MNDWI>0", 0] + [f"{mnd[k]:.2f}" for k in ("s11", "sturm", "wf", "wf_ev")]])


# ------------------------------------------------ fig 10: INT8 per seed (R2.6)
def fig_int8():
    dev = device_logs()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.8), sharey=True)
    seeds = ["42", "64", "91", "107"]
    table = []
    for ax, (ds, ttl) in zip(axes, [("s11val", "Sen1Floods11 validation split"), ("sturm", "STURM-Flood"),
                                    ("wf", "WorldFloods-v2")]):
        for i, s in enumerate(seeds + ["mean"]):
            if s == "mean":
                f_ = st.mean(dev[(ds, "float", x)] for x in seeds)
                q_ = st.mean(dev[(ds, "int8", x)] for x in seeds)
            else:
                f_, q_ = dev[(ds, "float", s)], dev[(ds, "int8", s)]
            ax.plot([f_, q_], [i, i], color=C["gray"], lw=2.2, zorder=2)
            dot(ax, f_, i, C["ink2"], hollow=True)
            dot(ax, q_, i, C["ofa"])
            ax.text(max(f_, q_) + 0.25, i, signed(q_ - f_), va="center", fontsize=7, color=C["ink2"])
            table.append([ds, s, f"{f_:.2f}", f"{q_:.2f}", f"{q_ - f_:+.2f}"])
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"seed {s}" for s in seeds] + ["mean of 4"], color=C["ink"])
        ax.set_ylim(4.6, -0.6)
        ax.axhline(3.5, color=C["grid"], lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mIoU (%) measured on the Jetson")
        lo = min(dev[(ds, k, s)] for k in ("float", "int8") for s in seeds)
        hi = max(dev[(ds, k, s)] for k in ("float", "int8") for s in seeds)
        ax.set_xlim(lo - 0.8, hi + 1.6)
        title(ax, ttl)
    fig.legend([dot_handle(C["ink2"], hollow=True), dot_handle(C["ofa"])],
               ["float model (ONNX, BN folded)", "deployed TensorRT engine (INT8 with FP16 attention)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    headline(fig, "The 8-bit accuracy cost in Table III is uneven across training seeds: seed 64 loses 3–4 points, "
             "the others 0–1.5",
             "Parsed from the four raw on-device evaluation logs (Jetson Xavier NX, TensorRT 8.4); their means reproduce "
             "Table III exactly (STURM 75.57 → 74.48, WorldFloods 78.61 → 77.02).", y=1.07, ysub=0.99)
    save(fig, "fig10_int8_by_seed.png")
    write_csv("fig10_int8_by_seed.csv", ["benchmark", "seed", "device_float_mIoU_pct", "device_int8_mIoU_pct",
                                         "int8_minus_float_pp"], table)


# ------------------------------------------------ fig 11: geometry control (R2.2)
def fig_geometry():
    b1 = read_csv(f"{PSD}/Supp_InText_Statistics/S3_harmonisation/B1_geometry_control.csv")
    v = {(r["cell"], r["arm"], r["benchmark"], r["metric"]): (100 * fnum(r["mean"]), 100 * fnum(r["std"])) for r in b1}
    arms = [("S11 uncropped 5.12km", "human labels, full S11 chips (5 km)", C["gray2"], True),
            ("S11 cropped 2.64km", "human labels, S11 chips cropped to 2.6 km", C["ink2"], False),
            ("pseudo 2.64km", "teacher labels, GDACS tiles (2.6 km)", C["ofa"], False)]
    panels = [("STURM-Flood", "micro_mIoU", "STURM\nall pixels pooled"),
              ("STURM-Flood", "macro_scene_mIoU", "STURM\naverage of 46 flood maps"),
              ("WorldFloods-v2 flood-only", "micro_mIoU", "WorldFloods\nall pixels pooled"),
              ("WorldFloods-v2 flood-only", "macro_scene_mIoU", "WorldFloods\naverage of 18 scenes")]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1), sharey=True)
    table = []
    for ax, (bm, met, ttl) in zip(axes, panels):
        allx = []
        for i, obj in enumerate(["CE", "Logit-KD", "OFA"]):
            for j, (arm, lab, col, hol) in enumerate(arms):
                mu, sd = v[(obj, arm, bm, met)]
                yy = i + (j - 1) * 0.22
                ax.plot([mu - sd, mu + sd], [yy, yy], color=col, lw=1.2, zorder=3)
                dot(ax, mu, yy, col, hollow=hol, size=6)
                allx += [mu - sd, mu + sd]
                table.append([bm, met, obj, lab, f"{mu:.2f}", f"{sd:.2f}"])
        ax.set_xlim(min(allx) - 0.5, max(allx) + 0.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["hard labels\n(CE)", "logit KD", "OFA"], color=C["ink"])
        ax.set_ylim(2.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mIoU (%)")
        title(ax, ttl)
    fig.legend([dot_handle(c, hollow=h) for _, _, c, h in arms], [a[1] for a in arms], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.13), fontsize=8)
    headline(fig, "Same 252-image budget, three training sets: the teacher-labelled set leads on STURM, but the size of "
             "its lead depends on image footprint and on how scores are averaged",
             "EfficientViT-B0, 4 seeds (mean ± SD). Cropping S11 to the GDACS footprint also deletes 74% of its water "
             "pixels and makes 21 chips completely dry, so it is not a clean control either.", y=1.1, ysub=1.01)
    save(fig, "fig11_geometry_control.png")
    write_csv("fig11_geometry_control.csv", ["benchmark", "aggregation", "objective", "training_set", "mIoU_pct",
                                             "seed_sd_pct"], table)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)
    for fn in (fig_architectures, fig_ofa, fig_components, fig_filter, fig_funnel, fig_pr, fig_events, fig_perturb,
               fig_ladder, fig_int8, fig_geometry):
        fn()
