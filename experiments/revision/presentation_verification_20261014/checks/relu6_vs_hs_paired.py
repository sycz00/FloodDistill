"""Single-factor activation effect (ReLU6 - Hardswish), paper Table III workstation caches.
Seed-mean per group (flood map / event), paired across groups; pooled micro mIoU per seed."""
import json, glob, sys, numpy as np
from scipy.stats import wilcoxon
R = sys.argv[1]
def miou(c):
    c = np.asarray(c, float); u = c.sum(1)
    return np.mean([c[i, 0] / u[i] for i in (0, 1) if u[i] > 0])
def wiou(c):
    c = np.asarray(c, float); u = c[1].sum()
    return c[1, 0] / u if u > 0 else np.nan
for ds in ("sturm", "wf"):
    for level in ("scene", "event"):
        per = {}
        micro = {}
        for fam in ("v2_evit_ofa_ps_N2500", "v2_relu6_ofa_ps_N2500"):
            groups, mic = {}, []
            for s in (42, 64, 91, 107):
                d = json.load(open(f"{R}/{ds}__{fam}__s{s}__clean.json"))
                for g, c in d[level].items():
                    groups.setdefault(g, []).append(miou(c))
                mic.append(miou(np.sum([np.asarray(c, float) for c in d["scene"].values()], 0)))
            per[fam] = {g: np.mean(v) for g, v in groups.items()}
            micro[fam] = np.array(mic)
        keys = sorted(set(per["v2_evit_ofa_ps_N2500"]) & set(per["v2_relu6_ofa_ps_N2500"]))
        diff = np.array([per["v2_relu6_ofa_ps_N2500"][k] - per["v2_evit_ofa_ps_N2500"][k] for k in keys])
        p = wilcoxon(diff, zero_method="pratt").pvalue
        md = micro["v2_relu6_ofa_ps_N2500"] - micro["v2_evit_ofa_ps_N2500"]
        print(f"{ds:5s} {level:5s} n={len(keys):2d}  mean d(mIoU)={diff.mean():+.4f}  relu6 wins {np.sum(diff>0)}/{len(keys)}  Wilcoxon p={p:.3f}"
              f" | pooled micro: HS {micro['v2_evit_ofa_ps_N2500'].mean():.4f} ReLU6 {micro['v2_relu6_ofa_ps_N2500'].mean():.4f}"
              f" d={md.mean():+.4f} (seed-unpaired)")
