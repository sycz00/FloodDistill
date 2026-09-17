# !/usr/bin/env python3
"""Compare teacher, students and fixed spectral rules across evaluation protocols.

Load each class mapping, evaluation scale and permanent-water convention into
its own cache. --block conditions evaluates perturbations; --block decomposition
compares objectives and label sources. The GDACS MNDWI-CE control is also free
of live-teacher supervision, although its tile selection is teacher-derived.
"""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

import numpy as np
from evaluation.perscene_stats import (PERSCENE_ROOT, load_cache, iou, seeds_of, micro,
                                    paired, holm)

def _c(*names):
    return [str(PERSCENE_ROOT / n) for n in names]

RULE = "index_mndwi_zero"

# (label, cache dirs, datasets, note), each entry is one consistent protocol
VARIANTS = [
    ("paper 512 px, paper class map", _c("perscene", "perscene_C1", "perscene_B1"),
     ("sturm", "wf"), "the manuscript's protocol"),
    ("operational class map",         _c("perscene_operational"),
     ("sturm", "wf"), "STURM 3-5 -> background; WF JRC-permanent -> background"),
    ("STURM native 128 px",           _c("perscene_sturmsize/128"),
     ("sturm",), "no upsample; 10 m GSD"),
    ("STURM 224 px",                  _c("perscene_sturmsize/224"),
     ("sturm",), "1.75x upsample; teacher's native patch grid"),
]

MODELS = [("Teacher 300M", "teacher"),
          ("pseudo-OFA N=2,500", "evit_ofa_ps_N2500"),
          ("pseudo-OFA N=252", "evit_ofa_ps_N252"),
          ("S11-OFA N=252", "ncgt_evit_ofa_N252"),
          ("S11crop-OFA N=252", "b1crop_evit_ofa_N252")]


def gscores(R, ds, fam, which):
    """{floodmap: IoU}, per seed then averaged over seeds."""
    per = {}
    for s in seeds_of(fam):
        k = (ds, fam, s, "clean")
        if k not in R:
            continue
        for g, cell in R[k]["scene"].items():
            if cell[1, 0] + cell[1, 2] <= 0:
                continue
            per.setdefault(g, []).append(iou(cell)[which])
    return {g: float(np.mean(v)) for g, v in per.items()}


def protocol_block():
    for which, mname in ((0, "micro mIoU"), (1, "micro flood-IoU")):
        print(f"\n{'='*100}\nTEACHER / STUDENT vs MNDWI>0  --  {mname}, clean imagery, "
              f"paired per flood map\n{'='*100}")
        for lab, dirs, dss, note in VARIANTS:
            R = load_cache(dirs)
            print(f"\n-- {lab}   ({note})")
            print(f"   {'model':22s} {'ds':6s} {'model':>8s} {'rule':>8s} {'d_micro':>8s} {'d_pair':>8s} "
                  f"{'wins':>8s} {'p_wilc':>9s} {'p_sign':>9s} {'p_holm':>9s}")
            for ds in dss:
                rows, ps = [], []
                rv = micro(R, ds, RULE, "-", "clean", which)
                if rv is None:
                    print(f"   (no {RULE} cell in this variant)")
                    continue
                for ml, fam in MODELS:
                    mv = [micro(R, ds, fam, s, "clean", which) for s in seeds_of(fam)]
                    mv = [v for v in mv if v is not None]
                    if not mv:
                        continue
                    pr = paired(gscores(R, ds, fam, which), gscores(R, ds, RULE, which))
                    if pr is None:
                        continue
                    rows.append((ml, ds, float(np.mean(mv)), rv, pr))
                    ps.append(pr[3])
                adj = holm(ps) if ps else []
                for (ml, d, mv, rv_, pr), ah in zip(rows, adj):
                    print(f"   {ml:22s} {d:6s} {mv:8.4f} {rv_:8.4f} {mv-rv_:+8.4f} {pr[0]:+8.4f} "
                          f"{pr[1]:4d}/{pr[2]:<3d} {pr[3]:9.3g} {pr[4]:9.3g} {ah:9.3g}")


# Second question: is there any condition where the models beat the rule?
# WorldFloods (Portales-Julia et al., Sci. Rep. 2023) found that MNDWI degrades badly around thin
# or semi-transparent cloud, while their cloud-aware network still finds the water underneath.
# The additive-haze axis is a controlled version of that, so the same paired flood-map contrast
# is run on perturbed imagery instead of clean.
COND_SETS = [("clean", ["clean"]),
             ("gain 1.30", ["g1.30"]),
             ("haze 0.05", ["h0.05"]),
             ("haze 0.10", ["h0.10"]),
             ("haze 0.20", ["h0.20"]),
             ("per-band vis 0.20", ["t1vis+20", "t1vis-20"]),
             ("per-band tilt 0.20", ["t1tilt+20", "t1tilt-20"])]


def gscores_cond(R, ds, fam, conds, which):
    """{floodmap: IoU under `conds`}, averaged over conditions then over seeds."""
    per = {}
    for s in seeds_of(fam):
        if any((ds, fam, s, c) not in R for c in conds):
            continue
        base = R[(ds, fam, s, conds[0])]["scene"]
        for g in base:
            cells = [R[(ds, fam, s, c)]["scene"].get(g) for c in conds]
            if any(c is None for c in cells):
                continue
            if cells[0][1, 0] + cells[0][1, 2] <= 0:
                continue
            per.setdefault(g, []).append(float(np.mean([iou(c)[which] for c in cells])))
    return {g: float(np.mean(v)) for g, v in per.items()}


def condition_sweep():
    from evaluation.perscene_stats import CACHES
    R = load_cache(CACHES)
    for which, mname in ((0, "micro mIoU"), (1, "micro flood-IoU")):
        print(f"\n{'='*100}\nMODEL vs MNDWI>0 UNDER PERTURBATION  --  {mname}, paper protocol, "
              f"paired per flood map\n{'='*100}")
        for ds in ("sturm", "wf"):
            print(f"\n-- {ds}")
            print(f"   {'condition':20s} {'model':22s} {'model':>8s} {'rule':>8s} {'d_micro':>8s} {'d_pair':>8s} "
                  f"{'wins':>8s} {'p_wilc':>9s} {'p_holm':>9s}")
            for cl, conds in COND_SETS:
                rows, ps = [], []
                rmean = [micro(R, ds, RULE, "-", c, which) for c in conds]
                if any(v is None for v in rmean):
                    continue
                rv = float(np.mean(rmean))
                B = gscores_cond(R, ds, RULE, conds, which)
                for ml, fam in MODELS:
                    mv = []
                    for s in seeds_of(fam):
                        vs = [micro(R, ds, fam, s, c, which) for c in conds]
                        if all(v is not None for v in vs):
                            mv.append(float(np.mean(vs)))
                    if not mv:
                        continue
                    pr = paired(gscores_cond(R, ds, fam, conds, which), B)
                    if pr is None:
                        continue
                    rows.append((ml, float(np.mean(mv)), rv, pr))
                    ps.append(pr[3])
                adj = holm(ps) if ps else []
                for (ml, mv, rv_, pr), ah in zip(rows, adj):
                    print(f"   {cl:20s} {ml:22s} {mv:8.4f} {rv_:8.4f} {mv-rv_:+8.4f} "
                          f"{pr[1]:4d}/{pr[2]:<3d} {pr[3]:9.3g} {ah:9.3g}")
                print()


# Third question: does the robustness come from the teacher or from the pool?
# Sen1Floods11-CE is the only arm without a teacher (all other Sen1Floods11 rows load the same
# frozen Prithvi teacher), so the two effects can be separated:
# objective effect = S11-CE vs S11-OFA/LKD (same labels, teacher added to the loss)
# label-source effect = S11-CE vs pseudo-CE (same objective, teacher-written labels,
# no teacher in either loss)
DECOMP = [
    ("objective   S11-CE  vs S11-OFA   N=252",   "ncgt_evit_ce_N252", "ncgt_evit_ofa_N252"),
    ("objective   S11-CE  vs S11-LKD   N=252",   "ncgt_evit_ce_N252", "ncgt_evit_lkd_N252"),
    ("labelsource S11-CE  vs pseudo-CE N=252",   "ncgt_evit_ce_N252", "evit_ce_ps_N252"),
    ("labelsource S11-CE  vs pseudo-CE N=2500",  "ncgt_evit_ce_N252", "evit_ce_ps_N2500"),
    ("both        S11-CE  vs pseudo-OFA N=2500", "ncgt_evit_ce_N252", "evit_ofa_ps_N2500"),
]


def decomposition():
    from evaluation.perscene_stats import CACHES, CORRUPTIONS, per_group, rtp
    R = load_cache(CACHES)
    for grp in ("haze", "gain"):
        conds, _ = CORRUPTIONS[grp]
        for ds in ("sturm", "wf"):
            print(f"\n== {grp} / {ds}  (positive delta = the FIRST arm is more brittle)")
            rows, ps = [], []
            for lab, A, B in DECOMP:
                pr = paired(per_group(R, ds, A, conds, "abs"),
                            per_group(R, ds, B, conds, "abs"))
                ra, rb = rtp(R, ds, A, conds), rtp(R, ds, B, conds)
                if pr is None or ra is None or rb is None:
                    continue
                rows.append((lab, ra[0], rb[0], pr)); ps.append(pr[3])
            if not rows:
                # An empty block used to print as a bare header, which reads as "nothing to
                # report". It means the cache has no corruption cells for these families:
                # runs/perscene ships `clean` only at N=252.
                print(f"  (no paired cells) needs {ds}__<family>__s<seed>__<cond>.json for "
                      f"cond in {', '.join(conds)}")
                print(f"  over {', '.join(sorted({f for _, a_, b_ in DECOMP for f in (a_, b_)}))}")
                print(f"  -> bash experiments/perscene_sweeps.sh")
                continue
            for (lab, ra, rb, pr), a in zip(rows, holm(ps)):
                print(f"  {lab:42s} RTP {ra:5.2f} -> {rb:5.2f}   d={pr[0]:+.4f} "
                      f"{pr[1]:2d}/{pr[2]:<3d} Holm p={a:.3g}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="all",
                    choices=["all", "protocol", "conditions", "decomposition"])
    a = ap.parse_args()
    if a.block in ("all", "protocol"):
        protocol_block()
    if a.block in ("all", "conditions"):
        condition_sweep()
    if a.block in ("all", "decomposition"):
        decomposition()
