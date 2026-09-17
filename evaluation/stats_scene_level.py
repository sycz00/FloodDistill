# !/usr/bin/env python3
"""Aggregate per-scene confusion cells and run paired group-level tests.

STURM groups chips into 46 flood maps and 29 EMS activations. WorldFloods
groups windows into 18 AOIs and 11 EMS events. AOI slot codes alone are not
unique across activations. For clean comparisons, average each group's IoU
over seeds before pairing groups. Report Wilcoxon with Pratt zeros, win rate,
a 10,000-resample group bootstrap CI and Holm-adjusted p-values per block.
The cache and grouping rules determine which inference these tests support.
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np
from scipy import stats

from figures.fig_style import CACHE_ALIAS, apply_pool, pool_arg

POOL = pool_arg()
RNG = np.random.default_rng(0)
NBOOT = 10000
SEEDS_EXPECTED = 4

LABEL = {
    "teacher":     "Teacher (Prithvi 300M)",
    "evit_ce_gt":  "GT-EvIT-CE",
    "evit_ofa_gt": "GT-EvIT-OFA",
    "evit_ce_ps":  "pseudo-EvIT-CE",
    "evit_ofa_ps": "pseudo-EvIT-OFA (ours)",
    "unet_ce_gt":  "GT-UNet-CE",
    "unet_lkd_gt": "GT-UNet-LKD",
    "unet_lkd_ps": "pseudo-UNet-LKD",
}
ORDER = list(LABEL)
DSN = {"sturm": "STURM-Flood", "wf": "WorldFloods-v2 (flood-only)"}


def iou(cell):
    """cell = 2x3 [[tp,fp,fn] bg, [tp,fp,fn] flood] -> (mIoU, flood_IoU)."""
    c = np.asarray(cell, dtype=np.float64)
    v = c[:, 0] / np.maximum(c[:, 0] + c[:, 1] + c[:, 2], 1)
    return float(v.mean()), float(v[1])


def defined(cell):
    """A group scores flood-IoU only if it actually contains GT flood (tp+fn > 0)."""
    c = np.asarray(cell, dtype=np.float64)
    return (c[1, 0] + c[1, 2]) > 0


_GT_CANON = {v: k for k, v in CACHE_ALIAS.items()}     # ncgt_*_N252 -> *_gt


def load(cache_dir="runs/perscene"):
    """(ds, fam, seed, cond) -> {'scene': {g: 2x3}, 'event': {g: 2x3}}, pool-aliased."""
    R = {}
    for p in glob.glob(f"{cache_dir}/*.json"):
        d = json.load(open(p))
        fam = apply_pool(d["family"], POOL)
        if fam is None:                       # belongs to another pool -> not on this axis
            continue
        fam = _GT_CANON.get(fam, fam)
        rec = {
            "scene": {k: np.asarray(v, float) for k, v in d["scene"].items()},
            "event": {k: np.asarray(v, float) for k, v in d["event"].items()},
        }
        R[(d["dataset"], fam, d["seed"], d["cond"])] = rec
        # Register an aliased family under its raw name too: the control scripts address
        # the N=252 arms as ncgt_*_N252 and *_ps_N252.
        if fam != d["family"]:
            R[(d["dataset"], d["family"], d["seed"], d["cond"])] = rec
    return R


# Filled by set_cache(), which the control scripts call before using the functions below.
R: dict = {}


def set_cache(cache_dir="runs/perscene"):
    """Fill the module-level per-scene cache the analysis functions read."""
    global R
    R = load(cache_dir)
    return R


def group_scores(ds, fam, cond, level, which=0):
    """{group: IoU averaged over seeds}. which=0 -> mIoU, 1 -> flood-IoU."""
    per = {}
    for k, v in R.items():
        if k[0] != ds or k[1] != fam or k[3] != cond:
            continue
        for g, cell in v[level].items():
            if not defined(cell):
                continue
            per.setdefault(g, []).append(iou(cell)[which])
    return {g: float(np.mean(vs)) for g, vs in per.items()}, \
           {g: len(vs) for g, vs in per.items()}


def micro(ds, fam, cond, level="scene", which=0):
    """Pooled confusion over all groups, then IoU, averaged over seeds."""
    vals = []
    for k, v in R.items():
        if k[0] != ds or k[1] != fam or k[3] != cond:
            continue
        tot = np.sum(list(v[level].values()), axis=0)
        vals.append(iou(tot)[which])
    return float(np.mean(vals)) if vals else np.nan, len(vals)


# Aggregation comparison and group-level Wilcoxon / bootstrap tests (main()).

def boot_ci(d, alpha=0.05):
    n = len(d)
    idx = RNG.integers(0, n, size=(NBOOT, n))
    mu = d[idx].mean(axis=1)
    return float(np.percentile(mu, 100 * alpha / 2)), float(np.percentile(mu, 100 * (1 - alpha / 2)))


def holm(ps):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    ps = np.asarray(ps, float)
    m = len(ps)
    o = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(run, 1.0)
    return adj


def paired(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in ks]), ks


def one(name, a, b):
    d, ks = paired(a, b)
    if len(d) < 5:
        return (name, np.nan, np.nan, np.nan, np.nan, len(d), np.nan)
    lo, hi = boot_ci(d)
    try:
        # exact when small and no zero/ties; scipy picks automatically with method='auto'
        w = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided", method="auto")
        p = float(w.pvalue)
    except ValueError:
        p = np.nan
    wins = float((d > 0).mean())
    return (name, float(d.mean()), lo, hi, p, len(d), wins)


def block(title, comps, level, ds, cond, which=0, band=None):
    """band=(lo_cond, hi_cond): compare the band delta per group instead of level accuracy."""
    rows = []
    for nm, fa, fb in comps:
        if band is None:
            A, _ = group_scores(ds, fa, cond, level, which)
            B, _ = group_scores(ds, fb, cond, level, which)
        else:
            lo, hi = band
            A1, _ = group_scores(ds, fa, lo, level, which); A2, _ = group_scores(ds, fa, hi, level, which)
            B1, _ = group_scores(ds, fb, lo, level, which); B2, _ = group_scores(ds, fb, hi, level, which)
            A = {g: A1[g] - A2[g] for g in set(A1) & set(A2)}
            B = {g: B1[g] - B2[g] for g in set(B1) & set(B2)}
        if not A or not B:
            continue
        rows.append(one(nm, A, B))
    if not rows:
        return
    ps = [r[4] for r in rows]
    good = [i for i, p in enumerate(ps) if p == p]
    adj = np.full(len(rows), np.nan)
    if good:
        adj[good] = holm([ps[i] for i in good])
    print(f"\n  {title}")
    print(f"    {'comparison':34s} {'Δ':>8s} {'95% CI (groups)':>21s} {'p_wilc':>8s} "
          f"{'p_holm':>8s} {'n':>4s} {'win%':>6s}  verdict")
    for r, pa in zip(rows, adj):
        nm, m, lo, hi, p, n, wins = r
        if m != m:
            print(f"    {nm:34s}   (n={n} too small)"); continue
        # Wilcoxon (ranks) and bootstrap CI (mean) can disagree, so both are reported.
        ci_sig = (lo > 0 or hi < 0)
        w_sig = (pa == pa and pa < 0.05)
        sig = ("SIG(both)" if (ci_sig and w_sig) else
               "wilc-only" if w_sig else
               "ci-only" if ci_sig else "n.s.")
        print(f"    {nm:34s} {m:+8.4f} [{lo:+7.4f},{hi:+7.4f}] {p:8.4f} "
              f"{pa:8.4f} {n:4d} {100*wins:5.1f}%  {sig}")


HEAD = [
    ("student vs teacher: OFA_ps - Tch", "evit_ofa_ps", "teacher"),
    ("full pipeline: OFA_ps - CE_gt",    "evit_ofa_ps", "evit_ce_gt"),
    ("DATA lever: OFA_ps - OFA_gt",      "evit_ofa_ps", "evit_ofa_gt"),
    ("OBJECTIVE lever: OFA_ps - CE_ps",  "evit_ofa_ps", "evit_ce_ps"),
    ("arch/size: UNet_lkd_gt - OFA_ps",  "unet_lkd_gt", "evit_ofa_ps"),
    ("WF anomaly: UNet_lkd_gt - Tch",    "unet_lkd_gt", "teacher"),
]
BRIT = [
    ("DATA: EvIT-CE  gt - ps",  "evit_ce_gt",  "evit_ce_ps"),
    ("DATA: EvIT-OFA gt - ps",  "evit_ofa_gt", "evit_ofa_ps"),
    ("DATA: UNet-LKD gt - ps",  "unet_lkd_gt", "unet_lkd_ps"),
    ("OBJ: GT-EvIT  CE - OFA",  "evit_ce_gt",  "evit_ofa_gt"),
    ("OBJ: ps-EvIT  CE - OFA",  "evit_ce_ps",  "evit_ofa_ps"),
    ("OBJ: GT-UNet  CE - LKD",  "unet_ce_gt",  "unet_lkd_gt"),
]

def main():
    set_cache()
    print(f"[cache] {len(R)} cells  (pool={POOL})\n")

    # E2 - aggregation ladder

    print("=" * 100)
    print("E2 — AGGREGATION LADDER (clean condition, mIoU, mean over seeds)")
    print("=" * 100)
    for ds in ("sturm", "wf"):
        print(f"\n### {DSN[ds]}")
        print(f"  {'family':24s} {'micro':>8s} {'macro-scene':>12s} {'macro-event':>12s}  "
              f"{'n_sc':>4s} {'n_ev':>4s} {'seeds':>5s}   micro-macro")
        for fam in ORDER:
            mi, nseed = micro(ds, fam, "clean")
            if nseed == 0:
                continue
            sc, _ = group_scores(ds, fam, "clean", "scene")
            ev, _ = group_scores(ds, fam, "clean", "event")
            msc = float(np.mean(list(sc.values()))) if sc else np.nan
            mev = float(np.mean(list(ev.values()))) if ev else np.nan
            print(f"  {LABEL[fam]:24s} {mi:8.4f} {msc:12.4f} {mev:12.4f}  "
                  f"{len(sc):4d} {len(ev):4d} {nseed:5d}   {mi - msc:+.4f}")

    # Is the macro drop a SMALL-GROUP effect? (the "tiny flood chips" hypothesis)
    print("\n" + "=" * 100)
    print("E2b — WHY macro < micro:  is the drop concentrated in SMALL groups?")
    print("      Mechanism test = Spearman(group GT-flood size, group IoU). A positive rho means")
    print("      small floods are harder, so macro (equal weight per group) sits below micro")
    print("      (pixel-dominated). No weighted column is shown on purpose: a weighted mean of")
    print("      per-group IoUs is a mean-of-ratios and does NOT reconstruct micro (ratio-of-sums).")
    print("=" * 100)
    for ds in ("sturm", "wf"):
        for level in ("scene", "event"):
            fam = "evit_ofa_ps"
            sc, _ = group_scores(ds, fam, "clean", level)
            fl, _ = group_scores(ds, fam, "clean", level, which=1)
            # group size = GT flood pixels (tp+fn of the flood row), pooled over seeds
            sizes = {}
            for k, v in R.items():
                if k[0] != ds or k[1] != fam or k[3] != "clean":
                    continue
                for g, cell in v[level].items():
                    if defined(cell):
                        sizes[g] = cell[1, 0] + cell[1, 2]
            gs = sorted(sc)
            y = np.array([sc[g] for g in gs]); w = np.array([sizes[g] for g in gs])
            yf = np.array([fl[g] for g in gs])
            order = np.argsort(w)
            half = len(gs) // 2
            small, large = order[:half], order[half:]
            rho, prho = stats.spearmanr(w, y)
            rhof, prhof = stats.spearmanr(w, yf)
            mi, _ = micro(ds, fam, "clean", level)
            print(f"\n  {ds:6s} {level:6s} n={len(gs):3d}   macro mIoU={y.mean():.4f}  "
                  f"macro flood-IoU={yf.mean():.4f}   micro mIoU={mi:.4f}  (micro-macro {mi-y.mean():+.4f})")
            print(f"         smallest-half mean mIoU={y[small].mean():.4f}  "
                  f"largest-half mean mIoU={y[large].mean():.4f}   "
                  f"Spearman(size, mIoU) rho={rho:+.3f} p={prho:.4f}")
            print(f"         Spearman(size, flood-IoU) rho={rhof:+.3f} p={prhof:.4f}")
            print(f"         GT-flood px per group: min={w.min():.0f} med={np.median(w):.0f} "
                  f"max={w.max():.0f}  (imbalance {w.max()/max(w.min(),1):.0f}x)")

    print("\n" + "=" * 100)
    print("E3 — GROUP-LEVEL TESTS (n = scenes/events, NOT seeds).  Positive Δ = first arg higher.")
    print("     Wilcoxon signed-rank over groups + group-resampling bootstrap CI + Holm within block.")
    print("=" * 100)
    for ds in ("sturm", "wf"):
        for level in ("scene", "event"):
            print(f"\n{'─'*100}\n{DSN[ds]}  ·  level = {level}  ·  CLEAN mIoU\n{'─'*100}")
            block("[headline accuracy deltas]", HEAD, level, ds, "clean")
    for ds in ("sturm", "wf"):
        for axis, lo, hi in (("gain", "g0.80", "g1.30"), ("haze", "clean", "h0.20")):
            print(f"\n{'─'*100}\n{DSN[ds]}  ·  level = scene  ·  band-Δ {axis} "
                  f"({lo}->{hi});  positive Δ = first arg MORE brittle\n{'─'*100}")
            block(f"[brittleness deltas · {axis}]", BRIT, "scene", ds, "clean", band=(lo, hi))

    print("\n" + "=" * 100)
    print("READ-OUT CAUTION")
    print("=" * 100)
    print("""  * Group-level n is large enough for Wilcoxon, but groups are NOT equal-sized; a
    group-level test is a test of the MACRO statistic by construction (macro weights every
    group equally), so pair it with the macro column, not the micro one.
  * Group-level and seed-level tests answer different questions ("another scene?" vs
    "another seed?"). stats_macro.py covers the seed axis; neither supersedes the other.
  * Groups from the same EMS activation are spatially/temporally correlated, so the
    scene-level test mildly overstates independence — this is why the event-level block
    (29 / 11 groups) is reported alongside as the conservative version.""")


if __name__ == "__main__":
    main()
