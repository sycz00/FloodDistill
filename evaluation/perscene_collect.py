# !/usr/bin/env python3
"""Aggregate runs/perscene/*.json into micro / macro-scene / macro-event tables, plus band deltas."""
import json, glob, sys, os
from collections import defaultdict
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def iou(cell):
    c = np.array(cell); v = c[:, 0] / np.maximum(c[:, 0] + c[:, 1] + c[:, 2], 1)
    return float(v.mean()), float(v[1])

def load_all(cache_dir="runs/perscene"):
    """(ds, family, seed, cond) -> micro / macro records. The control collectors pass one of
    the runs/perscene_* protocol caches; never glob two protocols into one dict."""
    R = {}
    for p in glob.glob(f"{cache_dir}/*.json"):
        d = json.load(open(p))
        tot = np.sum([np.array(v) for v in d["scene"].values()], axis=0)
        mi_m, mi_f = iou(tot)
        out = {"micro_mIoU": mi_m, "micro_flood": mi_f}
        for lvl in ("scene", "event"):
            ms, fs = [], []
            for cell in d[lvl].values():
                c = np.array(cell)
                if c[1, 0] + c[1, 2] <= 0: continue          # no GT flood in this group
                a, b = iou(c); ms.append(a); fs.append(b)
            out[f"macro_{lvl}_mIoU"] = float(np.mean(ms)) if ms else np.nan
            out[f"macro_{lvl}_flood"] = float(np.mean(fs)) if fs else np.nan
            out[f"n_{lvl}"] = len(ms)
        R[(d["dataset"], d["family"], d["seed"], d["cond"])] = out
    return R

def main():
    R = load_all()
    if not R: print("no cache files yet"); sys.exit()
    dss = sorted({k[0] for k in R}); fams = sorted({k[1] for k in R}); conds = sorted({k[3] for k in R})
    METRIC = sys.argv[1] if len(sys.argv) > 1 else "macro_scene_mIoU"
    print(f"metric = {METRIC}\n")
    for ds in dss:
        print(f"=== {ds.upper()} ===")
        print(f"  {'family':14s} " + " ".join(f"{c:>9s}" for c in conds) + "   seeds")
        for f in fams:
            row, ns = [], set()
            for c in conds:
                v = [R[k][METRIC] for k in R if k[0]==ds and k[1]==f and k[3]==c and METRIC in R[k]]
                ns |= {k[2] for k in R if k[0]==ds and k[1]==f and k[3]==c}
                row.append(np.mean(v) if v else np.nan)
            if all(np.isnan(row)): continue
            print(f"  {f:14s} " + " ".join(f"{x:9.4f}" if x==x else "        -" for x in row) + f"   {len(ns)}")
        # band deltas
        for lo, hi, nm in [("g0.80","g1.30","gain Δ(0.80→1.30)"), ("clean","h0.20","haze Δ(0→0.20)")]:
            if lo in conds and hi in conds:
                print(f"  -- {nm} --")
                for f in fams:
                    a = [R[k][METRIC] for k in R if k[0]==ds and k[1]==f and k[3]==lo]
                    b = [R[k][METRIC] for k in R if k[0]==ds and k[1]==f and k[3]==hi]
                    if a and b: print(f"     {f:14s} {np.mean(a)-np.mean(b):+.4f}")
        print()


if __name__ == "__main__":
    main()
