# !/usr/bin/env python3
"""Recompute every published table from paper_source_data/ and diff against the printed values.

Only reads inside the folder. Each table directory has a raw/ folder with the per-scene
confusion cells and per-run evaluation JSONs behind its *_values.csv. Reviewers can check
the numbers without checkpoints, datasets or a GPU.

Two reductions, same as when the tables were built:
  per-scene cells   <ds>__<family>__s<seed>__clean.json  ->  sum the 2x3 confusion cells over
                    scenes, then IoU per class, then mean -> micro mIoU, averaged over seeds
  per-run evals     <run>_s<seed>__eval_results*.json    ->  read the metric key directly

Run: python evaluation/verify_paper_source_data.py
     python evaluation/verify_paper_source_data.py --table "Table I" --verbose
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Tolerances. Means are printed to 3-4 dp, so a half-way rounding case can sit one unit out in
# the last place; anything above MEAN_TOL is a real disagreement rather than a rounding tie.
MEAN_TOL = 0.0015
SD_TOL = 0.0025


def micro_miou(paths: list[str]) -> float | None:
    """Pool per-scene 2x3 confusion cells into one micro mIoU."""
    if not paths:
        return None
    tot = np.zeros((2, 3))
    for p in paths:
        for cell in json.load(open(p))["scene"].values():
            tot += np.array(cell)
    ious = [tot[c][0] / max(tot[c][0] + tot[c][1] + tot[c][2], 1e-9) for c in (0, 1)]
    return float(np.mean(ious))


def per_seed_micro(raw: Path, ds: str, family: str) -> list[float]:
    out = []
    for s in (42, 64, 91, 107):
        g = sorted(glob.glob(str(raw / f"{ds}__{family}__s{s}__clean.json")))
        v = micro_miou(g)
        if v is not None:
            out.append(v)
    return out


def per_seed_key(raw: Path, run_prefix: str, key: str) -> list[float]:
    """One metric per seed from the per-run evaluation JSONs.

        Some folders have both STURM protocols next to each other, and the filenames are the
        other way round from the current repo convention: there `eval_results.json` is the
        retired per-tile percentile stretch and `eval_results_STURM_ORIG.json` the audited one.
        A plain glob silently reads the wrong protocol and STURM comes out ~0.06 low. So for
        STURM the audited file wins, for every other key the first file that has it.
    """
    out = []
    for s in (42, 64, 91, 107):
        cands = sorted(glob.glob(str(raw / f"{run_prefix}_s{s}__eval_results*.json")))
        if key.startswith("sturm"):
            cands.sort(key=lambda p: "STURM_ORIG" not in p)
        for p in cands:
            d = json.load(open(p))
            if key in d and d[key] is not None:
                out.append(float(d[key]))
                break
    return out


def stats(vals: list[float]) -> tuple[float, float, float, int] | None:
    if not vals:
        return None
    a = np.array(vals, float)
    return float(a.mean()), float(a.std(ddof=0)), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


class Report:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.rows: list[tuple] = []

    def add(self, table, label, printed_mean, printed_sd, got):
        if got is None:
            self.rows.append((table, label, printed_mean, None, None, None, "NO RAW"))
            return
        m, sd0, sd1, n = got
        dm = abs(printed_mean - m)
        ds = min(abs(printed_sd - sd0), abs(printed_sd - sd1)) if printed_sd is not None else 0.0
        bad = dm > MEAN_TOL or ds > SD_TOL
        self.rows.append((table, label, printed_mean, m, dm, ds, "DIFF" if bad else "ok"))

    def summary(self) -> int:
        tables: dict[str, list] = {}
        for r in self.rows:
            tables.setdefault(r[0], []).append(r)
        worst_any = 0
        for t, rows in tables.items():
            noraw = [r for r in rows if r[6] == "NO RAW"]
            diffs = [r for r in rows if r[6] == "DIFF"]
            ok = [r for r in rows if r[6] == "ok"]
            dmax = max([r[4] for r in ok + diffs] or [0.0])
            worst_any = max(worst_any, len(diffs))
            status = "OK " if not diffs and not noraw else "    "
            print(f"{status}{t:34s} {len(ok)+len(diffs):3d} cells recomputed   "
                  f"max |delta| {dmax:.4f}"
                  + (f"   {len(diffs)} disagree" if diffs else "")
                  + (f"   {len(noraw)} without raw" if noraw else ""))
            for r in diffs:
                print(f"      DIFF  {r[1]:52s} printed {r[2]:.4f}  recomputed {r[3]:.4f}"
                      f"  delta {r[4]:+.4f}")
            if self.verbose:
                for r in ok:
                    print(f"      ok    {r[1]:52s} printed {r[2]:.4f}  recomputed {r[3]:.4f}")
        return worst_any


# Table I -- matched budget, N = 252

_T1_FAMILY = {
    ("EfficientViT-B0, CE (0.7M)", "Sen1Floods11"): ("ncgt_evit_ce_N252", "sen_ce_N252"),
    ("EfficientViT-B0, CE (0.7M)", "pseudo"): ("v2_evit_ce_ps_N252", "pseudo_ce_aug_N252"),
    ("EfficientViT-B0, Logit-KD (0.7M)", "Sen1Floods11"): ("ncgt_evit_lkd_N252", "sen_dfkd_N252"),
    ("EfficientViT-B0, Logit-KD (0.7M)", "pseudo"): ("v2_evit_lkd_ps_N252", "pseudo_dfkd_aug_N252"),
    ("EfficientViT-B0, OFA (0.7M)", "Sen1Floods11"): ("ncgt_evit_ofa_N252", "sen_ofa_N252"),
    ("EfficientViT-B0, OFA (0.7M)", "pseudo"): ("v2_evit_ofa_ps_N252", "pseudo_ofa_aug_N252"),
    ("UNet-ResNet50, CE (32.5M)", "Sen1Floods11"): ("ncgt_unet_ce_N252", "sen_ce_unet_N252"),
    ("UNet-ResNet50, CE (32.5M)", "pseudo"): ("v2_unet_ce_ps_N252", "pseudo_ce_unet_aug_N252"),
    ("UNet-ResNet50, Logit-KD (32.5M)", "Sen1Floods11"): ("ncgt_unet_lkd_N252", "sen_dfkd_unet_N252"),
    ("UNet-ResNet50, Logit-KD (32.5M)", "pseudo"): ("v2_unet_lkd_ps_N252", "pseudo_dfkd_unet_aug_N252"),
}
_BENCH_DS = {"STURM-Flood": "sturm", "WorldFloods-v2": "wf"}
_S11_KEY = {"mIoU": "s11_test_mIoU", "water IoU": "s11_test_flood_IoU"}


def table_I(base: Path, rep: Report):
    d = base / "Table_I_Matched_Budget_N252"
    raw = d / "raw"
    for r in csv.DictReader(open(d / "Table_I_values.csv")):
        key = (r["model"], r["labels"])
        if key not in _T1_FAMILY:
            continue
        fam, run = _T1_FAMILY[key]
        if r["benchmark"] == "S11 test":
            vals = per_seed_key(raw, run, _S11_KEY[r["metric"]])
        elif r["metric"] == "mIoU":
            vals = per_seed_micro(raw, _BENCH_DS[r["benchmark"]], fam)
        else:
            continue          # water IoU on the OOD sets: same cells, not re-derived here
        rep.add("Table I", f"{r['model']} | {r['labels']} | {r['benchmark']} {r['metric']}",
                float(r["mean"]), float(r["sd_ddof0"]), stats(vals))


# Table II -- deployment operating point, N = 2,500

_T2_FAMILY = {
    "EfficientViT-B0, CE": ("v2_evit_ce_ps_N2500", "pseudo_ce_aug_N2500"),
    "EfficientViT-B0, Logit-KD": ("v2_evit_lkd_ps_N2500", "pseudo_dfkd_aug_N2500"),
    "EfficientViT-B0, OFA (ours)": ("v2_evit_ofa_ps_N2500", "pseudo_ofa_aug_N2500"),
    "UNet-ResNet50, CE": ("v2_unet_ce_ps_N2500", "pseudo_ce_unet_aug_N2500"),
    "UNet-ResNet50, Logit-KD": ("v2_unet_lkd_ps_N2500", "pseudo_dfkd_unet_aug_N2500"),
}


def table_II(base: Path, rep: Report):
    d = base / "Table_II_Deployment_N2500"
    raw = d / "raw"
    for r in csv.DictReader(open(d / "Table_II_values.csv")):
        if r["model"] not in _T2_FAMILY:
            continue
        fam, run = _T2_FAMILY[r["model"]]
        if r["benchmark"] == "S11":
            vals = per_seed_key(raw, run, "s11_test_mIoU")
        else:
            vals = per_seed_micro(raw, "sturm" if r["benchmark"] == "STURM" else "wf", fam)
        rep.add("Table II", f"{r['model']} | {r['benchmark']}",
                float(r["mean"]), float(r["sd_ddof0"]), stats(vals))


# Table III -- accuracy through the deployment chain (workstation rows only; the two device
# rows come from the Jetson logs, which are not per-scene cells)

_T3_FAMILY = {
    ("EfficientViT-B0, OFA", "Sen1Floods11, 252"): "ncgt_evit_ofa_N252",
    ("EfficientViT-B0, OFA", "pseudo, 2500"): "v2_evit_ofa_ps_N2500",
    ("+ ReLU6", "pseudo, 2500"): "v2_relu6_ofa_ps_N2500",
}


def table_III(base: Path, rep: Report):
    d = base / "Table_III_Deployment_Chain"
    raw = d / "raw_workstation"
    if not raw.is_dir():
        return
    for r in csv.DictReader(open(d / "Table_III_values.csv")):
        fam = _T3_FAMILY.get((r["stage"], r["labels"]))
        if fam is None:
            continue
        ds = "sturm" if r["benchmark"].startswith("STURM") else "wf"
        rep.add("Table III", f"{r['stage']} | {r['benchmark']}",
                float(r["mean_micro_mIoU"]), float(r["sd_ddof0"]),
                stats(per_seed_micro(raw, ds, fam)))


# Table S3 -- architecture ablation

_S3_RUN = {
    ("MobileNetV3-Small", "CE"): "mnv3_ce", ("MobileNetV3-Small", "Logit-KD"): "mnv3_dfkd",
    ("Small CNN", "CE"): "microm_ce", ("Small CNN", "Logit-KD"): "microm_dfkd",
    ("EfficientViT-B0", "CE"): "ce_hs", ("EfficientViT-B0", "Logit-KD"): "div_dfkd_hs",
    ("EfficientViT-B0", "OFA (ours)"): "div_ofa_hs", ("EfficientViT-B0", "OFA"): "div_ofa_hs",
}
_S3_KEY = {"S11 mIoU": "s11_test_mIoU", "STURM mIoU": "sturm_mIoU"}


def table_S3(base: Path, rep: Report):
    d = base / "Supp_Table_S3_Architecture_Ablation"
    raw = d / "raw"
    if not raw.is_dir():
        return
    avail = {os.path.basename(p) for p in glob.glob(str(raw / "*.json"))}
    for r in csv.DictReader(open(d / "Table_S3_values.csv")):
        run = _S3_RUN.get((r["architecture"], r["loss"]))
        if run is None:
            continue
        # the ablation runs carry an N in the name; find whichever N shipped
        pre = sorted({re.sub(r"_s\d+__eval_results.*", "", f) for f in avail
                      if f.startswith(run + "_N")})
        if not pre:
            continue
        rep.add("Table S3", f"{r['architecture']} | {r['loss']} | {r['benchmark']}",
                float(r["mean"]), float(r["sd_ddof0"]),
                stats(per_seed_key(raw, pre[0], _S3_KEY[r["benchmark"]])))


TABLES = {"Table I": table_I, "Table II": table_II, "Table III": table_III, "Table S3": table_S3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "paper_source_data"),
                    help="The paper_source_data folder to verify.")
    ap.add_argument("--table", default=None, choices=list(TABLES),
                    help="Verify one table instead of all of them.")
    ap.add_argument("--verbose", action="store_true", help="List every cell, not just the bad ones.")
    a = ap.parse_args()

    base = Path(a.root)
    if not base.is_dir():
        print(f"not a directory: {base}", file=sys.stderr)
        return 2

    rep = Report(a.verbose)
    for name, fn in TABLES.items():
        if a.table and name != a.table:
            continue
        try:
            fn(base, rep)
        except FileNotFoundError as e:
            print(f"    {name}: missing input, skipped ({e})")

    print(f"\nRecomputed from {base}/ -- no checkpoints, no datasets, no GPU.")
    print(f"Tolerances: mean {MEAN_TOL}, sd {SD_TOL} (a printed value rounded to 3 dp can sit "
          f"half a unit out in the last place).\n")
    n_bad = rep.summary()
    if n_bad:
        print(f"\n{n_bad} cell(s) disagree beyond tolerance.")
    else:
        print("\nEvery recomputed cell matches the printed value.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
