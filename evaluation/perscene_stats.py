#!/usr/bin/env python3
"""Reader and group-level statistics for the per-scene caches in runs/perscene*.

load_cache() keeps caches with different evaluation protocols (scale, crop, class mapping)
apart. paired_group_tests, protocol_sweep, scale_control_collect and
labeller_control_collect all read through this module.

Robustness is the worst-case absolute drop over a perturbation axis, per seed and then
averaged (paper Sec. IV-F). rtp() is the relative version used by protocol_sweep.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

import numpy as np
from scipy.stats import wilcoxon
from math import comb

# Caches (read-only). All use the paper protocol, so they can be read together. Do not add
# runs/perscene_operational or runs/perscene_sturmsize/*; load_cache() checks for that.
# PERSCENE_ROOT is also used by perscene_eval.py for writing.
from evaluation import PERSCENE_ROOT

CACHES = [str(PERSCENE_ROOT / d)
          for d in ("perscene",         # the approved cache
                    "perscene_B1",      # B1 geometry-matched arm
                    "perscene_C1",      # the non-learned index reference
                    "perscene_T1",      # T1 per-band gain, N=252 and N=2500
                    "perscene_T1_ref")] # T1 for the teacher and the index

# Perturbation groups: name -> (conditions, datasets). `gain` uses the levels that exist on
# both benchmarks; g0.70 and g0.90 (STURM only) are in the separate `gain_ext` group.
CORRUPTIONS = {
    "gain":      (["g0.80", "g1.20", "g1.30"], {"sturm", "wf"}),
    "haze":      (["h0.05", "h0.10", "h0.20"], {"sturm", "wf"}),
    "t1vis":     (["t1vis+05", "t1vis-05", "t1vis+10", "t1vis-10",
                   "t1vis+20", "t1vis-20"], {"sturm", "wf"}),
    "t1tilt":    (["t1tilt+05", "t1tilt-05", "t1tilt+10", "t1tilt-10",
                   "t1tilt+20", "t1tilt-20"], {"sturm", "wf"}),
}
# Stress level: d = 0.30 is the only per-band level outside the training jitter U(0.80, 1.20).
STRESS = {
    "t1vis_x":   ([*CORRUPTIONS["t1vis"][0], "t1vis+30", "t1vis-30"], {"sturm", "wf"}),
    "t1tilt_x":  ([*CORRUPTIONS["t1tilt"][0], "t1tilt+30", "t1tilt-30"], {"sturm", "wf"}),
    "gain_ext":  (["g0.70", "g0.80", "g0.90", "g1.20", "g1.30"], {"sturm"}),
}
# The four groups whose mean is RTP_all. Equal weight per group.
HEADLINE_GROUPS = ["gain", "haze", "t1vis", "t1tilt"]

# admitted arms: label -> (family, budget, checkpoint provenance)
ARMS = [
    ("S11 CE  N=252",        "ncgt_evit_ce_N252",   252,  "runs/ablations_sen1/sen_ce_N252"),
    ("S11 LKD N=252",        "ncgt_evit_lkd_N252",  252,  "runs/ablations_sen1/sen_dfkd_N252"),
    ("S11 OFA N=252",        "ncgt_evit_ofa_N252",  252,  "runs/ablations_sen1/sen_ofa_N252"),
    ("S11crop CE  N=252",    "b1crop_evit_ce_N252",  252, "runs/controls_geometry/sen_ce_crop264_N252"),
    ("S11crop LKD N=252",    "b1crop_evit_lkd_N252", 252, "runs/controls_geometry/sen_dfkd_crop264_N252"),
    ("S11crop OFA N=252",    "b1crop_evit_ofa_N252", 252, "runs/controls_geometry/sen_ofa_crop264_N252"),
    ("pseudo CE  N=252",     "evit_ce_ps_N252",  252,  "runs/nsweep/pseudo_ce_aug_N252"),
    ("pseudo LKD N=252",     "evit_lkd_ps_N252", 252,  "runs/nsweep/pseudo_dfkd_aug_N252"),
    ("pseudo OFA N=252",     "evit_ofa_ps_N252", 252,  "runs/nsweep/pseudo_ofa_aug_N252"),
    ("pseudo CE  N=2500",    "evit_ce_ps_N2500",  2500, "runs/nsweep/pseudo_ce_aug_N2500"),
    ("pseudo LKD N=2500",    "evit_lkd_ps_N2500", 2500, "runs/nsweep/pseudo_dfkd_aug_N2500"),
    ("pseudo OFA N=2500",    "evit_ofa_ps_N2500", 2500, "runs/nsweep/pseudo_ofa_aug_N2500"),
    # UNet-ResNet50 (capacity reference): gain and haze only, no per-band sweep.
    ("S11 UNet CE  N=252",   "ncgt_unet_ce_N252",   252,  "runs/ablations_sen1/sen_ce_unet_N252"),
    ("S11 UNet LKD N=252",   "ncgt_unet_lkd_N252",  252,  "runs/ablations_sen1/sen_dfkd_unet_N252"),
    ("ps UNet CE  N=252",    "unet_ce_ps_N252",  252,  "runs/nsweep/pseudo_ce_unet_aug_N252"),
    ("ps UNet LKD N=252",    "unet_lkd_ps_N252", 252,  "runs/nsweep/pseudo_dfkd_unet_aug_N252"),
    ("ps UNet CE  N=2500",   "unet_ce_ps_N2500", 2500, "runs/nsweep/pseudo_ce_unet_aug_N2500"),
    ("ps UNet LKD N=2500",   "unet_lkd_ps_N2500",2500, "runs/nsweep/pseudo_dfkd_unet_aug_N2500"),
    ("Teacher Prithvi 300M", "teacher",             None, "hf (public checkpoint)"),
    ("MNDWI > 0 (rule)",     "index_mndwi_zero",    None, "(no checkpoint: 0 parameters)"),
]

# Paired contrasts (label, A = S11 arm, B = pseudo arm); a positive delta means A is more brittle.
CONTRASTS = [
    ("CE  · S11 vs pseudo N=252",       "ncgt_evit_ce_N252",   "evit_ce_ps_N252"),
    ("LKD · S11 vs pseudo N=252",       "ncgt_evit_lkd_N252",  "evit_lkd_ps_N252"),
    ("OFA · S11 vs pseudo N=252",       "ncgt_evit_ofa_N252",  "evit_ofa_ps_N252"),
    ("CE  · S11crop vs pseudo N=252",   "b1crop_evit_ce_N252",  "evit_ce_ps_N252"),
    ("LKD · S11crop vs pseudo N=252",   "b1crop_evit_lkd_N252", "evit_lkd_ps_N252"),
    ("OFA · S11crop vs pseudo N=252",   "b1crop_evit_ofa_N252", "evit_ofa_ps_N252"),
    ("CE  · S11 vs pseudo N=2500",      "ncgt_evit_ce_N252",   "evit_ce_ps_N2500"),
    ("LKD · S11 vs pseudo N=2500",      "ncgt_evit_lkd_N252",  "evit_lkd_ps_N2500"),
    ("OFA · S11 vs pseudo N=2500",      "ncgt_evit_ofa_N252",  "evit_ofa_ps_N2500"),
    ("CE  · S11crop vs pseudo N=2500",  "b1crop_evit_ce_N252",  "evit_ce_ps_N2500"),
    ("LKD · S11crop vs pseudo N=2500",  "b1crop_evit_lkd_N252", "evit_lkd_ps_N2500"),
    ("OFA · S11crop vs pseudo N=2500",  "b1crop_evit_ofa_N252", "evit_ofa_ps_N2500"),
    # UNet: gain and haze only (see ARMS).
    ("UNet-CE  · S11 vs pseudo N=252",  "ncgt_unet_ce_N252",  "unet_ce_ps_N252"),
    ("UNet-LKD · S11 vs pseudo N=252",  "ncgt_unet_lkd_N252", "unet_lkd_ps_N252"),
    ("UNet-CE  · S11 vs pseudo N=2500", "ncgt_unet_ce_N252",  "unet_ce_ps_N2500"),
    ("UNet-LKD · S11 vs pseudo N=2500", "ncgt_unet_lkd_N252", "unet_lkd_ps_N2500"),
]

SEEDS = ["42", "64", "91", "107"]
SEEDLESS = ("teacher", "index_")
REL_FLOOR = 0.30          # per-flood-map relative sensitivity check: clean mIoU floor


def _check_budget(fam: str) -> None:
    """Raise for anything that is not an approved budget."""
    if fam == "teacher" or fam.startswith("index_"):
        return
    if not (fam.endswith("_N252") or fam.endswith("_N2500")):
        raise SystemExit(f"REFUSED: {fam} carries no approved budget suffix (N252 / N2500)")


def load_cache(dirs):
    """(ds, fam, seed, cond) -> {'scene': {g: 2x3}, 'event': {...}}. Refuses collisions."""
    R, src = {}, {}
    for cd in dirs:
        for p in sorted(glob.glob(f"{cd}/*.json")):
            d = json.load(open(p))
            k = (d["dataset"], d["family"], d["seed"], d["cond"])
            if k in R:
                a = np.sum([np.asarray(x, float) for x in R[k]["scene"].values()], axis=0)
                b = np.sum([np.asarray(x, float) for x in d["scene"].values()], axis=0)
                if not np.array_equal(a, b):
                    raise SystemExit(
                        f"CELL COLLISION with different content: {k}\n"
                        f"  {src[k]}\n  {p}\nThe cell filename does not record the protocol; "
                        f"two caches with different protocols must never be globbed together.")
                continue
            R[k] = {"scene": {g: np.asarray(v, float) for g, v in d["scene"].items()}}
            src[k] = p
    return R


def iou(cell):
    c = np.asarray(cell, float)
    v = c[:, 0] / np.maximum(c[:, 0] + c[:, 1] + c[:, 2], 1)
    return float(v.mean()), float(v[1])


def seeds_of(fam):
    return ["-"] if fam.startswith(SEEDLESS) else SEEDS


def micro(R, ds, fam, seed, cond, which=0):
    k = (ds, fam, seed, cond)
    if k not in R:
        return None
    tot = np.sum(list(R[k]["scene"].values()), axis=0)
    return iou(tot)[which]


def rtp(R, ds, fam, conds, which=0):
    """(RTP %, sd over seeds, n_seeds). Per seed: mean relative drop over `conds`."""
    vals = []
    for s in seeds_of(fam):
        pc = micro(R, ds, fam, s, "clean", which)
        if pc is None or pc <= 0:
            continue
        ps = [micro(R, ds, fam, s, c, which) for c in conds]
        if any(p is None for p in ps):
            continue
        vals.append(100.0 * float(np.mean([(pc - p) / pc for p in ps])))
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def mean_abs_drop(R, ds, fam, conds, which=0):
    """Absolute mean drop over `conds`, micro, per seed then averaged."""
    vals = []
    for s in seeds_of(fam):
        pc = micro(R, ds, fam, s, "clean", which)
        ps = [micro(R, ds, fam, s, c, which) for c in conds]
        if pc is None or any(p is None for p in ps):
            continue
        vals.append(float(np.mean([pc - p for p in ps])))
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def worst_drop(R, ds, fam, conds, which=0):
    """The manuscript's statistic, for side-by-side reporting."""
    vals = []
    for s in seeds_of(fam):
        pc = micro(R, ds, fam, s, "clean", which)
        ps = [micro(R, ds, fam, s, c, which) for c in conds]
        if pc is None or any(p is None for p in ps):
            continue
        vals.append(pc - min([pc] + ps))
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def per_group(R, ds, fam, conds, mode, which=0, floor=REL_FLOOR):
    """{floodmap: statistic}, per seed and then averaged over seeds.

    mode 'abs':   mean absolute drop over `conds`
    mode 'rel':   mean relative drop in % (flood maps with clean >= floor)
    mode 'worst': worst-case drop over `conds`, at least 0 (clean is part of the envelope)
    """
    per = {}
    for s in seeds_of(fam):
        kc = (ds, fam, s, "clean")
        if kc not in R or any((ds, fam, s, c) not in R for c in conds):
            continue
        base = R[kc]["scene"]
        for g, cell in base.items():
            if cell[1, 0] + cell[1, 2] <= 0:      # no GT flood in this map: undefined
                continue
            pc = iou(cell)[which]
            if mode == "rel" and pc < floor:
                continue
            drops = []
            ok = True
            for c in conds:
                cc = R[(ds, fam, s, c)]["scene"].get(g)
                if cc is None:
                    ok = False
                    break
                drops.append(pc - iou(cc)[which])
            if not ok:
                continue
            d = max([0.0] + drops) if mode == "worst" else float(np.mean(drops))
            per.setdefault(g, []).append(100.0 * d / pc if mode == "rel" else d)
    return {g: float(np.mean(v)) for g, v in per.items()}


def sign_p(k, n):
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def wilc_p(v):
    v = np.asarray(v, float)
    if v.size == 0 or np.allclose(v, 0):
        return 1.0
    try:
        return float(wilcoxon(v, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    o = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(run, 1.0)
    return adj


def paired(A, B):
    ks = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in ks])
    if d.size == 0:
        return None
    w = int((d > 0).sum())
    return float(d.mean()), w, len(d), wilc_p(d), sign_p(w, len(d))
