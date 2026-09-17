# !/usr/bin/env python3
"""Shared figure styles, cache aliases and robustness statistics.

Colour is the objective, line style and fill the label source, and the marker the
architecture. Worst-case drop is taken per seed, then averaged. Moderate and full
haze envelopes are kept separate.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

# Choose the pseudo-label budget; GT uses the full N=252 split in both views.
POOLS = {
    "matched": {"suffix": "_ps",
           "alias": {"evit_ce_ps_N252":  "evit_ce_ps",
                     "evit_lkd_ps_N252": "evit_lkd_ps",
                     "evit_ofa_ps_N252": "evit_ofa_ps",
                     "unet_ce_ps_N252":  "unet_ce_ps",
                     "unet_lkd_ps_N252": "unet_lkd_ps"},
           "out": "assets/sturm_orig/figures"},
    "full": {"suffix": "_ps",
               "alias": {"evit_ce_ps_N2500":  "evit_ce_ps",
                         "evit_lkd_ps_N2500": "evit_lkd_ps",
                         "evit_ofa_ps_N2500": "evit_ofa_ps",
                         "unet_ce_ps_N2500":  "unet_ce_ps",
                         "unet_lkd_ps_N2500": "unet_lkd_ps"},
               "out": "assets/sturm_orig/full_figures"},
}

def pool_arg(argv=None):
    """--pool {matched,full} anywhere on the command line; default matched (the matched-N anchor)."""
    a = sys.argv if argv is None else argv
    if "--pool" in a:
        p = a[a.index("--pool") + 1]
        if p not in POOLS:
            raise SystemExit(f"unknown --pool {p!r}; known: {list(POOLS)}")
        return p
    return "matched"


def out_dir(pool="matched"):
    d = Path(POOLS[pool]["out"]); d.mkdir(parents=True, exist_ok=True)
    return d

# objective hue
_CE, _LKD, _OFA, _T = "#D55E00", "#0072B2", "#009E73", "#000000"
# GT = pale variant of the same hue
_CE_GT, _LKD_GT, _OFA_GT = "#EDA07B", "#7FBEE0", "#7FCEBA"

# family -> (label, colour, linestyle, marker)
STYLE = {
    "teacher":     ("Teacher (Prithvi 300M)",  _T,      ":",  "D"),
    "evit_ce_gt":  ("GT-EvIT-CE",              _CE_GT,  "--", "o"),
    "evit_lkd_gt": ("GT-EvIT-LKD",             _LKD_GT, "--", "o"),
    "evit_ofa_gt": ("GT-EvIT-OFA",             _OFA_GT, "--", "o"),
    "unet_ce_gt":  ("GT-UNet-CE",              _CE_GT,  "--", "s"),
    "unet_lkd_gt": ("GT-UNet-LKD",             _LKD_GT, "--", "s"),
    "evit_ce_ps":  ("pseudo-EvIT-CE",          _CE,     "-",  "o"),
    "evit_lkd_ps": ("pseudo-EvIT-LKD",         _LKD,    "-",  "o"),
    "evit_ofa_ps": ("pseudo-EvIT-OFA (ours)",  _OFA,    "-",  "o"),
    "unet_ce_ps":  ("pseudo-UNet-CE",          _CE,     "-",  "s"),
    "unet_lkd_ps": ("pseudo-UNet-LKD",         _LKD,    "-",  "s"),
}
ORDER = list(STYLE)
IS_UNET = {f: f.startswith("unet") for f in STYLE}
SHORT = {"teacher": "Teacher",
         "evit_ce_gt": "GT-EvIT-CE",   "evit_lkd_gt": "GT-EvIT-LKD",
         "evit_ofa_gt": "GT-EvIT-OFA", "unet_ce_gt": "GT-UNet-CE",
         "unet_lkd_gt": "GT-UNet-LKD", "evit_ce_ps": "ps-EvIT-CE",
         "evit_lkd_ps": "ps-EvIT-LKD", "evit_ofa_ps": "ps-EvIT-OFA\n(ours)",
         "unet_ce_ps": "ps-UNet-CE",   "unet_lkd_ps": "ps-UNet-LKD"}
DS = {"sturm": "STURM-Flood", "wf": "WorldFloods-v2 (flood-only)"}

# axis -> (conditions in sweep order, x values, x label). clean is always the anchor.
AXES = {
    "gain": (["g0.80", "clean", "g1.20", "g1.30"], [0.80, 1.00, 1.20, 1.30],
             "radiometric gain  $g$   (reflectance $\\times g$)"),
    "haze": (["clean", "h0.05", "h0.10", "h0.20"], [0.0, 0.05, 0.10, 0.20],
             "haze  $h$   (blue-band additive reflectance)"),
}
# Robustness envelopes. The first is the headline one and physically uncontested (Chavez
# dark-object blue-band path radiance of 0.02-0.10 is ordinary to heavy haze). The second
# is a stress point, h=0.20 is roughly thick haze / thin cirrus.
ENVELOPES = {
    "gain": {"full": ["g0.80", "clean", "g1.20", "g1.30"]},
    "haze": {"moderate": ["clean", "h0.05", "h0.10"], "full": ["clean", "h0.05", "h0.10", "h0.20"]},
}
SEEDS_OF = lambda fam: ["-"] if fam == "teacher" else ["42", "64", "91", "107"]


def iou(cell):
    c = np.asarray(cell, float)
    v = c[:, 0] / np.maximum(c[:, 0] + c[:, 1] + c[:, 2], 1)
    return float(v.mean()), float(v[1])


def apply_pool(family, pool):
    """Resolve selected pseudo families to canonical names.

    Controls and N-sweep families keep their raw names. Bare pseudo keys from
    the previous pool are ignored."""
    alias = POOLS[pool]["alias"]
    if family in alias:
        return alias[family]
    if family.startswith(("noaug_", "cur_")):
        # Extras (aug controls, selector arms) keep their own names and are visible to both
        # sizes -- the E3 curve and the curation figure need every N, not just the anchor.
        # Only the ALIASED families above differ between matched and full.
        return family
    if family.endswith("_ps"):
        # A bare canonical pseudo key can only be a leftover cell of the removed pool: here the
        # pseudo half arrives via the alias above. Dropping it is what keeps a stale
        # runs/perscene/ from reintroducing that pool through the back door.
        return None
    return family


# Resolve all GT arms to their N=252 per-scene records.
CACHE_ALIAS = {f"{m}_{o}_gt": f"ncgt_{m}_{o}_N252"
               for m, objectives in (("evit", ("ce", "lkd", "ofa")), ("unet", ("ce", "lkd")))
               for o in objectives}

def cache_fam(family):
    """Canonical family -> the name it is actually stored under in the perscene cache."""
    return CACHE_ALIAS.get(family, family)


def disk_family(family, pool):
    'Resolve a canonical pseudo family to its budget-specific cache name.'
    for raw, canon in POOLS[pool]["alias"].items():
        if canon == family:
            return raw
    return family


def load_cache(pattern="runs/perscene/*.json", pool=None):
    """(ds, family, seed, cond) -> {metric: value}."""
    pool = pool_arg() if pool is None else pool
    R = {}
    for p in glob.glob(pattern):
        d = json.load(open(p))
        raw = d["family"]
        fam = apply_pool(raw, pool)
        if fam is None:
            continue
        d["family"] = fam
        tot = np.sum([np.asarray(v) for v in d["scene"].values()], axis=0)
        mi_m, mi_f = iou(tot)
        rec = {"micro_mIoU": mi_m, "micro_flood": mi_f}
        for lvl in ("scene", "event"):
            ms, fs = [], []
            for cell in d[lvl].values():
                c = np.asarray(cell, float)
                if c[1, 0] + c[1, 2] <= 0:       # no GT flood -> flood-IoU undefined
                    continue
                a, b = iou(c); ms.append(a); fs.append(b)
            rec[f"macro_{lvl}_mIoU"] = float(np.mean(ms)) if ms else np.nan
            rec[f"macro_{lvl}_flood"] = float(np.mean(fs)) if fs else np.nan
        R[(d["dataset"], d["family"], d["seed"], d["cond"])] = rec
        # An aliased family is registered under BOTH names. The canonical key is what the
        # factorial figures read; the original key is what the N-sweep figure reads, and
        # evit_ofa_ps_N252 is a member of both -- aliasing it away silently deleted the
        # N=252 point from the E3 curve, which is the one N where GT and pseudo are matched.
        if fam != raw:
            R[(d["dataset"], raw, d["seed"], d["cond"])] = rec
    return R


def worst_drop(R, ds, fam, envelope, metric):
    """f(clean) - min f over `envelope`, per seed and then averaged.

        min is non-linear, so taking it after averaging would understate the worst case.
        Seeds with missing conditions are skipped; (nan, 0, 0) if none are left.
    """
    vals = []
    for s in SEEDS_OF(fam):
        v = [R.get((ds, fam, s, c), {}).get(metric, np.nan) for c in envelope]
        if any(x != x for x in v):
            continue
        vals.append(v[envelope.index("clean")] - min(v))
    if not vals:
        return float("nan"), 0.0, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def curve(R, ds, fam, conds, metric):
    """mean, std, n over seeds for each condition."""
    out = []
    for c in conds:
        v = [R[k][metric] for k in R
             if k[0] == ds and k[1] == fam and k[3] == c and R[k].get(metric, np.nan) == R[k].get(metric, np.nan)]
        out.append((float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), 0.0, 0))
    return out
