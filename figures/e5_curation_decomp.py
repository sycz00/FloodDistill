# !/usr/bin/env python3
"""Plot the N=2500 pre-filter and selector comparisons (supplement Fig. S1).

The paper's pool has 23,720 tiles and 5,987 filtered candidates. Compare
unfiltered versus filtered random selection for the pre-filter effect, then
ranked selectors versus filtered random. These unaugmented selector runs
have a different recipe from the matched-budget and deployment tables.
Keep the declared gain/haze envelopes and seed aggregation when redrawing.
"""
from __future__ import annotations
import csv, itertools, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figures.fig_style import (ENVELOPES, apply_pool, curve, load_cache,
                                   out_dir, pool_arg, worst_drop)

RNG = np.random.default_rng(0)
NBOOT = 10000
POOL = pool_arg()
OUT = out_dir(POOL)
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.color": "0.9",
                     "axes.axisbelow": True, "figure.dpi": 120})
MUTED = "0.45"

# Gain levels in the cache. The worst-case drop uses the four-level envelope from
# fig_style.ENVELOPES; 0.70 and 0.90 are drawn but not scored (rank correlation with the
# six-level version: 0.95 micro, 1.00 macro).
GAIN_CONDS = ["g0.70", "g0.80", "g0.90", "clean", "g1.20", "g1.30"]
GAIN_X = [0.70, 0.80, 0.90, 1.00, 1.20, 1.30]

# key, label, colour, marker, linestyle
# All ranked arms select N=2,500 tiles from the same 5,987 filtered candidates
# (flood fraction in [0.05, 0.85], conf_std >= 0.05, out of 23,720 tiles). The other nine arms
# (*_raw, fps_r1, r2core_r1, r2core_flonly, glister_r2/ood/flive) are only written to the CSV.
ARMS = [
    ("cur_random_raw_N2500", "random\n(no filter, 23,720)",     "#999999", "x", ":"),
    ("cur_random_N2500",     "uniform\n(filtered, 5,987)",      "#E69F00", "D", "-."),
    ("cur_fps_N2500",        "FPS diversity\n(filtered, 5,987)", "#0072B2", "o", "-"),
    ("cur_confidence_N2500", "confidence\n(filtered, 5,987)",   "#D55E00", "s", "--"),
    ("cur_glister_N2500",    "GLISTER\n(filtered, 5,987)",      "#7B3294", "^", (0, (3, 1, 1, 1))),
    ("cur_r2core_N2500",     "R2-Core\n(filtered, 5,987)",      "#009E73", "P", (0, (5, 1))),
]

# Contrasts from ARMS: ARMS[0] random over all tiles, ARMS[1] random over the filtered
# candidates, ARMS[2:] ranking rules. Row 1 is the pre-filter effect; the other rows compare
# each ranking rule with ARMS[1].
_RAW, _UNIF = ARMS[0][0], ARMS[1][0]
DECOMP_PAIRS = [(f"PRE-FILTER  {'raw - uniform':<20s}", _RAW, _UNIF)] + [
    (f"RANK  {lab.split(chr(10))[0][:11]:<11s} uniform - it", _UNIF, key)
    for key, lab, *_ in ARMS[2:]]
# N=252 S11 reference. Selector arms use N=2500 and the curation recipe
# (AMP-fp16); GT uses --no_amp --deterministic. See supplement Fig. S1.
REFS = [("ncgt_evit_ofa_N252", "GT-OFA, 252 human labels")]
REF = REFS[0]
METRIC = "micro_mIoU"
# N of the curation arms, taken from the family names (N=2500).
ARM_N = "N=" + ARMS[0][0].rsplit("_N", 1)[1]
ARM_TAG = "n" + ARMS[0][0].rsplit("_N", 1)[1]

R = load_cache()
missing = [k for k, *_ in ARMS
           if worst_drop(R, "sturm", k, ENVELOPES["gain"]["full"], METRIC)[2] == 0]
if missing:
    raise SystemExit(f"no cells for {missing}: train the selector arms with "
                     f"experiments/curation_compare.sh, then run experiments/perscene_sweeps.sh")

# Per-flood-map confusion cells for the scene-level tests (load_cache() only keeps metrics).
SCENE = {}
for p in Path("runs/perscene").glob("sturm__*.json"):
    d = json.load(open(p))
    SCENE[(d["family"], d["seed"], d["cond"])] = d["scene"]


def _iou(cell):
    c = np.asarray(cell, float)
    v = c[:, 0] / np.maximum(c[:, 0] + c[:, 1] + c[:, 2], 1)
    return float(v.mean())


def per_seed_worst(fam, axis):
    """Worst-case drop per seed (not averaged); written to the CSV."""
    conds = ENVELOPES[axis]["full"]
    out = {}
    for s in ("42", "64", "91", "107"):
        v = [R.get(("sturm", fam, s, c), {}).get(METRIC, np.nan) for c in conds]
        if not any(x != x for x in v):
            out[s] = v[conds.index("clean")] - min(v)
    return out


def boot_ci(d, alpha=0.05):
    n = len(d)
    if n == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, n, size=(NBOOT, n))
    mu = d[idx].mean(axis=1)
    return float(np.percentile(mu, 100 * alpha / 2)), float(np.percentile(mu, 100 * (1 - alpha / 2)))


def sign_perm(d):
    """Exact sign-flip permutation p over all 2^n sign patterns.

    With n=4 seeds the smallest two-sided p is 2/16 = 0.125, so the scene-level test is the one to report.
    """
    n = len(d)
    if n == 0:
        return np.nan
    obs = abs(d.mean())
    c = sum(1 for s in itertools.product([1, -1], repeat=n)
            if abs((d * np.array(s)).mean()) >= obs - 1e-12)
    return c / 2 ** n


def per_scene_worst(fam, axis):
    """{flood map: worst-case drop}, seeds averaged within each map first (as in stats_scene_level.py)."""
    conds = ENVELOPES[axis]["full"]
    per = {}
    for c in conds:
        acc = {}
        for s in ("42", "64", "91", "107"):
            cells = SCENE.get((fam, s, c))
            if cells is None:
                continue
            for g, cell in cells.items():
                acc.setdefault(g, []).append(_iou(cell))
        per[c] = {g: float(np.mean(v)) for g, v in acc.items()}
    keys = set.intersection(*(set(per[c]) for c in conds))
    return {g: per["clean"][g] - min(per[c][g] for c in conds) for g in keys}


def scene_test(a, b, axis):
    """Paired scene-level test of worstDrop(a) - worstDrop(b). Returns (mean, win, n, p_sign, p_wilc)."""
    from math import comb
    from scipy.stats import wilcoxon
    A, B = per_scene_worst(a, axis), per_scene_worst(b, axis)
    ks = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in ks])
    w = int((d > 0).sum()); n = len(d)
    k = min(w, n - w)
    p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    try:
        p_w = float(wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        p_w = 1.0
    return float(d.mean()), w, n, p_sign, p_w


# console: the decomposition, on both axes and both aggregations
rows = []
for agg in ("micro_mIoU", "macro_scene_mIoU"):
    for axis in ("gain", "haze"):
        W = {k: {s: v for s, v in per_seed_worst(k, axis).items()} for k, *_ in ARMS} \
            if agg == METRIC else None
        print(f"\n== worst-case drop, {axis}, {agg}  (higher = loses more) ==")
        for k, lab, *_ in ARMS:
            m, sd, n = worst_drop(R, "sturm", k, ENVELOPES[axis]["full"], agg)
            print(f"   {lab.replace(chr(10),' '):38s} {m:.4f} (±{sd:.4f}) n={n}")
            rows.append([f"worstDrop_{axis}", agg, k, f"{m:.4f}", f"{sd:.4f}", n])
        m, sd, n = worst_drop(R, "sturm", REF[0], ENVELOPES[axis]["full"], agg)
        print(f"   {'-- ' + REF[1]:38s} {m:.4f} (±{sd:.4f}) n={n}")
        rows.append([f"worstDrop_{axis}", agg, REF[0], f"{m:.4f}", f"{sd:.4f}", n])
        if agg != METRIC:
            continue
        print("   -- DECOMPOSITION (paired over seeds; positive = first arm loses MORE) --")
        # Ranking rows are compared with the random draw over the same candidates.
        for nm, a, b in DECOMP_PAIRS:
            ks = sorted(set(W[a]) & set(W[b]))
            if not ks:
                print(f"     {nm} (no matched seeds)"); continue
            d = np.array([W[a][k] - W[b][k] for k in ks])
            ci = boot_ci(d); p = sign_perm(d)
            # Seed level: with n=4 the smallest p is 0.125, so the CI alone is not significance.
            seed_v = "CI excludes 0" if (ci[0] > 0 or ci[1] < 0) else "CI spans 0"
            sm, sw, sn, sp, spw = scene_test(a, b, axis)
            star = "SIG" if max(sp, spw) < 0.05 else "n.s."
            print(f"     {nm} seeds Δ={d.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
                  f"p={p:.3f}({seed_v})")
            print(f"     {'':32s} maps  Δ={sm:+.4f} {sw:2d}/{sn} sign p={sp:.2e} "
                  f"wilcoxon p={spw:.2e}  {star}")
            rows.append([f"decomp_{axis}_seedlevel", agg, nm.strip(), f"{d.mean():.4f}",
                         f"{ci[0]:.4f}..{ci[1]:.4f}", f"p={p:.3f} n={len(ks)}"])
            rows.append([f"decomp_{axis}_scenelevel", agg, nm.strip(), f"{sm:.4f}",
                         f"sign={sp:.2e} wilcoxon={spw:.2e}", f"{sw}/{sn}"])

# figure
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4))

# (a) the measured gain curves
ax = axs[0]
lo, hi = 0.80, 1.30
# Shade the gain ranges separately. Training jitter covers [0.80,1.20];
# g=1.30 lies outside it.
AUG_LO, AUG_HI = 0.80, 1.20          # training-time multiplicative jitter, both arms
BANDS = [                             # (lo, hi, colour, short label), widest first
    (0.80, 1.20, "#E7EEF4", "±20% tail"),
    (0.90, 1.10, "#CFE6F5", "±10% (90% px)"),
    (0.95, 1.05, "#B7D9A8", "sensor ±5%"),
]
for _lo, _hi, _c, _ in BANDS:
    ax.axvspan(_lo, _hi, color=_c, zorder=0)
# Short names in a separate key; text inside the narrow bands would overlap.
_keys = [Patch(fc=c, ec="none", label=lab) for _lo, _hi, c, lab in reversed(BANDS)]
ax.add_artist(ax.legend(handles=_keys, fontsize=6.4, loc="upper left", framealpha=.92,
                        handlelength=1.6, borderpad=.35, labelspacing=.28))
_top = ax.secondary_xaxis("top", functions=(lambda g: (g - 1) * 100, lambda p: 1 + p / 100))
_top.set_xlabel("deviation from nominal reflectance  (%)", fontsize=8.2, labelpad=2)
_top.tick_params(labelsize=7.4)
for k, lab, col, mk, ls in ARMS:
    s = curve(R, "sturm", k, GAIN_CONDS, METRIC)
    ax.errorbar(GAIN_X, [v[0] for v in s], yerr=[v[1] for v in s], fmt=mk, ls=ls, color=col,
                ms=5, lw=1.8, capsize=2, elinewidth=0.9, mec="white", mew=0.5,
                label=lab.replace("\n", " "))
    for g, (m, sd, n) in zip(GAIN_X, s):
        rows.append(["gain_curve", METRIC, k, f"{m:.4f}", f"{sd:.4f}", f"g={g}"])
for ri, (rk, rlab) in enumerate(REFS):
    s = curve(R, "sturm", rk, GAIN_CONDS, METRIC)
    if all(v[2] == 0 for v in s):
        print(f"[warn] reference {rk} has no cached cells - skipped"); continue
    # Warn if the reference is missing gain conditions.
    if (_ng := sum(v[2] == 0 for v in s)):
        print(f"[warn] reference {rk}: {_ng}/{len(GAIN_CONDS)} gain conditions missing — "
              f"the line will be broken; markers added so the measured points stay visible")
    ax.plot(GAIN_X, [v[0] for v in s], color="#4FAE92", marker="*", ms=8, mec="white", mew=.6,
            ls=[(0, (4, 2)), (0, (1, 1.6))][ri % 2], lw=2.0, zorder=6, label=rlab)
    for g, (m, sd, n) in zip(GAIN_X, s):
        rows.append(["gain_curve_reference", METRIC, rk, f"{m:.4f}", f"{sd:.4f}", f"g={g}"])
ax.set_xlabel("radiometric gain $g$  (reflectance $\\times g$)")
ax.set_ylabel("STURM mIoU (micro)")
ax.set_title(f"(a) {ARM_N} curation arms under gain   ·   scored over "
             f"$g\\in[{lo:.2f},{hi:.2f}]$", fontsize=9.2)
ax.legend(fontsize=7.0, loc="lower left")

# (b) worst-case drop, BOTH axes
ax = axs[1]
w = 0.38
for j, (axis, hatch) in enumerate((("gain", ""), ("haze", "///"))):
    vals, errs, cols = [], [], []
    for k, lab, col, *_ in ARMS:
        m, sd, n = worst_drop(R, "sturm", k, ENVELOPES[axis]["full"], METRIC)
        vals.append(m); errs.append(sd); cols.append(col)
    x = np.arange(len(ARMS)) + (j - 0.5) * w
    ax.bar(x, vals, width=w, color=cols, edgecolor="white", lw=0.6, hatch=hatch, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="0.35", elinewidth=0.9, capsize=2, zorder=4)
    for ri, (rk, rlab) in enumerate(REFS):
        rm, rsd, rn = worst_drop(R, "sturm", rk, ENVELOPES[axis]["full"], METRIC)
        if rn == 0:
            continue
        ax.hlines(rm, -0.5, len(ARMS) - 0.5, color="#009E73",
                  ls=[(0, (4, 2)), (0, (1, 1.6))][ri % 2], lw=1.4, zorder=5)
        # label text comes from REFS so it can never disagree with the data being plotted
        ax.text(-0.46, rm, f"{rlab} · {axis}", fontsize=6.4, color="#009E73",
                va="center", ha="left", zorder=7,
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.92))
        rows.append([f"worstDrop_{axis}_reference", METRIC, rk, f"{rm:.4f}", f"{rsd:.4f}", rn])
ax.set_xticks(np.arange(len(ARMS)))
ax.set_xticklabels([l for _, l, *_ in ARMS], fontsize=6.8, rotation=18, ha="right")
# "clean" is g=1.0 on the gain axis and h=0 on the haze axis.
ax.set_ylabel("worst-case drop in mIoU\n$f(g{=}1.0,\\,h{=}0)\\;-\\;\\min f$ over the envelope")
ax.set_title("(b) worst-case drop, both axes  (higher = loses more)\n"
             "solid = gain · hatched = haze", fontsize=9.5)
ax.set_ylim(bottom=0)

# Per-seed worst-case drops are only in the CSV (worstDrop_<axis>_per_seed).
for _k, _lab, *_ in ARMS:
    for _axis in ("gain", "haze"):
        for _s, _y in per_seed_worst(_k, _axis).items():
            rows.append([f"worstDrop_{_axis}_per_seed", METRIC, _k, f"{_y:.4f}", "", f"s{_s}"])

fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(OUT / f"fig_curation_decomp_{ARM_TAG}.{e}", bbox_inches="tight")
plt.close(fig)

with open(OUT / f"fig_curation_decomp_{ARM_TAG}_values.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["quantity", "aggregation", "arm", "mean", "sd_or_ci", "n_or_key"])
    wr.writerows(rows)
print(f"\n[ok] {OUT}/fig_curation_decomp_{ARM_TAG}.{{pdf,png}}  + values CSV ({len(rows)} rows)")
