# !/usr/bin/env python3
"""Paired group tests for matched-budget accuracy and robustness.

Accuracy pairs the five N=252 architecture/objective cells per flood map
(46) and EMS activation (29), with Holm correction within each block of five.
Robustness uses three configurations and two axes, corrected as a block of six.
Compute each seed's worst-case drop before averaging; also report the reverse
order as a sensitivity check. Wilcoxon uses Pratt zeros, with exact sign tests
and a 10,000-resample group bootstrap supplied by the shared statistics code.
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

import evaluation.stats_scene_level as S                 # group_scores, one, holm, boot_ci
from evaluation.perscene_stats import (CORRUPTIONS, PERSCENE_ROOT, load_cache, paired,
                                    per_group, sign_p)

# Table I's five cells, as (label, Sen1Floods11 family, pseudo family).
CELLS = [
    ("EfficientViT-B0 CE",       "ncgt_evit_ce_N252",  "evit_ce_ps_N252"),
    ("EfficientViT-B0 Logit-KD", "ncgt_evit_lkd_N252", "evit_lkd_ps_N252"),
    ("EfficientViT-B0 OFA",      "ncgt_evit_ofa_N252", "evit_ofa_ps_N252"),
    ("UNet-ResNet50 CE",         "ncgt_unet_ce_N252",  "unet_ce_ps_N252"),
    ("UNet-ResNet50 Logit-KD",   "ncgt_unet_lkd_N252", "unet_lkd_ps_N252"),
]
# The three matched pairs (no UNet-OFA run) and their labels in B2_robustness_holm.csv.
B2_NAME = {"EfficientViT-CE": "EvIT CE", "EfficientViT-OFA": "EvIT OFA",
           "UNet-Logit-KD": "UNet Logit-KD"}
B2_HEADER = ["convention", "axis", "pair", "gt_family", "ps_family", "n_groups", "mean_delta",
             "median_delta", "wins", "wilcoxon_p", "sign_p", "holm_p_wilcoxon", "holm_p_sign",
             "survives_holm_both"]
ROBUST_PAIRS = [
    ("EfficientViT-CE",  "ncgt_evit_ce_N252",  "evit_ce_ps_N252"),
    ("EfficientViT-OFA", "ncgt_evit_ofa_N252", "evit_ofa_ps_N252"),
    ("UNet-Logit-KD",    "ncgt_unet_lkd_N252", "unet_lkd_ps_N252"),
]


def test(ds, fa, fb, level, cond="clean"):
    """Paired group-level test, seed-averaged first exactly as stats_scene_level does."""
    A, _ = S.group_scores(ds, fa, cond, level)
    B, _ = S.group_scores(ds, fb, cond, level)
    _, mean, lo, hi, pw, n, _ = S.one("x", A, B)
    wins = int(sum(1 for k in set(A) & set(B) if A[k] - B[k] > 0))
    return dict(n=n, mean=mean, wins=wins, lo=lo, hi=hi, p_wilcoxon=pw,
                p_sign=sign_p(wins, n))


def accuracy_block(ds, rows):
    for level, lname in (("scene", "flood map"), ("event", "EMS activation")):
        print("\n" + "=" * 104)
        print(f"TABLE I's FIVE CELLS, paired per {lname} on {ds.upper()}, clean mIoU")
        print("      pseudo minus Sen1Floods11; positive = the pseudo-label arm is better")
        print("=" * 104)
        # Skip cells without cached data (NaN p-values would pass Holm).
        cells, res = [], []
        for c in CELLS:
            r = test(ds, c[2], c[1], level)
            if r["n"]:
                cells.append(c); res.append(r)
        if not res:
            print(f"  (no cached cells for any of the {len(CELLS)} families)")
            continue
        if len(res) < len(CELLS):
            print(f"  ({len(CELLS) - len(res)} of {len(CELLS)} cells have no cached cells "
                  f"and are excluded from the Holm block)")
        hw = S.holm([r["p_wilcoxon"] for r in res])
        hs = S.holm([r["p_sign"] for r in res])
        print(f"  {'cell':26s} {'delta':>9s} {'wins':>8s} {'95% CI (groups)':>21s} "
              f"{'p_wilc':>9s} {'p_holm':>9s} {'p_sign':>9s} {'sign_holm':>9s}")
        for (lab, gt, ps), r, pw, psg in zip(cells, res, hw, hs):
            print(f"  {lab:26s} {r['mean']:+9.4f} {r['wins']:4d}/{r['n']:<3d} "
                  f"[{r['lo']:+7.4f},{r['hi']:+7.4f}] {r['p_wilcoxon']:9.2e} {pw:9.2e} "
                  f"{r['p_sign']:9.2e} {psg:9.2e}")
            rows.append(["accuracy", ds, lab, level, r["n"], f"{r['mean']:.5f}", r["wins"],
                         f"{r['p_wilcoxon']:.3e}", f"{pw:.3e}", f"{r['p_sign']:.3e}",
                         f"{psg:.3e}", f"{r['lo']:.5f}", f"{r['hi']:.5f}"])
        print(f"  -> {sum(1 for p in hw if p < 0.05)}/{len(res)} survive Holm over the block "
              f"of {len(res)} (Wilcoxon); {sum(1 for p in hs if p < 0.05)}/{len(res)} "
              f"(exact sign test)")


def worst_seed_avg(ds, fam, conds):
    """Worst-case drop per flood map with seeds averaged first (sensitivity check).

    The paper takes the minimum per seed first, because averaging first understates the worst case.
    """
    A = {c: S.group_scores(ds, fam, c, "scene")[0] for c in ["clean"] + list(conds)}
    ks = set.intersection(*(set(v) for v in A.values()))
    return {g: max([0.0] + [A["clean"][g] - A[c][g] for c in conds]) for g in ks}


def robustness_block(R, ds, rows, b2_rows=None):
    """Worst-case drop per flood map, 3 pairs x 2 axes, Holm within the block of six."""
    for per_seed, cname in ((True, "drop per seed then averaged (paper convention)"),
                            (False, "seeds averaged first (sensitivity check)")):
        block = []
        for axis in ("gain", "haze"):
            conds, dss = CORRUPTIONS[axis]
            if ds not in dss:
                continue
            for lab, gf, pf in ROBUST_PAIRS:
                if per_seed:
                    A = per_group(R, ds, gf, conds, "worst")
                    B = per_group(R, ds, pf, conds, "worst")
                else:
                    A, B = worst_seed_avg(ds, gf, conds), worst_seed_avg(ds, pf, conds)
                r = paired(A, B)
                if r is None:
                    continue
                med = float(np.median([A[k] - B[k] for k in set(A) & set(B)]))
                block.append(dict(axis=axis, lab=lab, gf=gf, pf=pf, mean=r[0], med=med,
                                  wins=r[1], n=r[2], pw=r[3], ps=r[4]))
        if not block:
            # No gain/haze cells for these families yet (see experiments/robustness_b2.sh).
            if not per_seed:
                continue                       # both conventions want the same cells; say it once
            want = sorted({c for ax in ("gain", "haze") if ds in CORRUPTIONS[ax][1]
                           for c in CORRUPTIONS[ax][0]})
            print(f"\n[skip] robustness block on {ds.upper()}: no paired corruption cells for "
                  f"{', '.join(p[0] for p in ROBUST_PAIRS)}.")
            print(f"       Needs {ds}__<family>__s<seed>__<cond>.json for cond in "
                  f"{', '.join(want)}\n"
                  f"       over the N=252 families, i.e.\n"
                  f"         bash experiments/perscene_sweeps.sh\n"
                  f"       The accuracy block above needs only the `clean` cells and is "
                  f"unaffected.")
            continue
        hw, hs = S.holm([b["pw"] for b in block]), S.holm([b["ps"] for b in block])
        print("\n" + "=" * 108)
        print(f"BLOCK OF {len(block)} ROBUSTNESS TESTS on {ds.upper()} -- {cname}")
        print("  worst-case drop per flood map; positive delta = the Sen1Floods11 arm is the "
              "MORE brittle one")
        print("=" * 108)
        print(f"  {'axis':6s} {'pair':17s} {'mean':>9s} {'wins':>8s} {'p_wilc':>10s} "
              f"{'holm_w':>10s} {'p_sign':>10s} {'holm_s':>10s}  verdict")
        n_ok = 0
        for b, pw, ps in zip(block, hw, hs):
            ok = pw < 0.05 and ps < 0.05
            n_ok += ok
            print(f"  {b['axis']:6s} {b['lab']:17s} {b['mean']:+9.4f} {b['wins']:4d}/{b['n']:<3d} "
                  f"{b['pw']:10.2e} {pw:10.2e} {b['ps']:10.2e} {ps:10.2e}  "
                  f"{'SURVIVES' if ok else 'fails Holm'}")
            rows.append([f"robustness ({cname.split('(')[0].strip()})", ds, b["lab"], b["axis"],
                         b["n"], f"{b['mean']:.5f}", b["wins"], f"{b['pw']:.3e}", f"{pw:.3e}",
                         f"{b['ps']:.3e}", f"{ps:.3e}", "", ""])
            if b2_rows is not None and ds == "sturm":
                conv = ("per-seed drop then average (paper convention)" if per_seed
                        else "seed-average first (sensitivity check)")
                b2_rows.append([conv, b["axis"], B2_NAME[b["lab"]], b["gf"],
                                b["pf"].replace("_N252", ""), b["n"], f"{b['mean']:.5f}",
                                f"{b['med']:.5f}", b["wins"], f"{b['pw']:.3e}", f"{b['ps']:.3e}",
                                f"{pw:.3e}", f"{ps:.3e}", int(ok)])
        print(f"  -> {n_ok}/{len(block)} survive Holm over the block of {len(block)} under BOTH "
              f"tests; {sum(1 for p in hw if p < 0.05)}/{len(block)} under Wilcoxon alone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sturm,wf")
    ap.add_argument("--cache", default=str(PERSCENE_ROOT / "perscene"))
    ap.add_argument("--out", default=str(PERSCENE_ROOT / "perscene/paired_group_tests.csv"))
    ap.add_argument("--robustness_out", default=None,
                    help="Also write the STURM robustness block in the layout of the "
                         "supplement's B2_robustness_holm.csv")
    a = ap.parse_args()

    S.set_cache(a.cache)
    R = load_cache([a.cache])
    print(f"[cache] {len(R)} cells from {a.cache}")

    rows = [["block", "dataset", "pair", "level_or_axis", "n_groups", "mean_diff", "wins",
             "wilcoxon_p", "holm_p_wilcoxon", "sign_p", "holm_p_sign",
             "boot_ci_lo", "boot_ci_hi"]]
    b2_rows = [B2_HEADER]
    for ds in [d.strip() for d in a.datasets.split(",")]:
        accuracy_block(ds, rows)
        robustness_block(R, ds, rows, b2_rows)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n-> {a.out}  ({len(rows)-1} rows)")
    if a.robustness_out:
        if len(b2_rows) == 1:
            print("[warn] no STURM robustness rows; --robustness_out not written")
        else:
            with open(a.robustness_out, "w", newline="") as f:
                csv.writer(f).writerows(b2_rows)
            print(f"-> {a.robustness_out}  ({len(b2_rows)-1} rows)")


if __name__ == "__main__":
    main()
