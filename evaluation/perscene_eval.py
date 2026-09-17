# !/usr/bin/env python3
"""Cache per-scene TP/FP/FN for STURM and WorldFloods.

STURM groups tiles into 46 flood maps and 29 events. WorldFloods groups
windows into 18 AOIs, with EMS codes parsed from scene names. Micro and macro
scores are computed from these cells without further inference.

Files are <dataset>__<family>__s<seed>__<condition>.json. Class mapping and evaluation scale are not encoded.
Use a separate --out_dir for non-default protocols.
perscene_stats.load_cache also checks collisions.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np, pandas as pd, torch

from evaluation import PERSCENE_ROOT
from data.ood_datasets import get_sturm_flood_dataloader, get_worldfloods_dataloader
from data.channel_adapter import sturm_s2_adapter
from models.student import build_student
from models.teacher import build_teacher

# Dataset roots: $DATA_ROOT/<dataset>, see utils/paths.py.
from utils.paths import dataset_root
STURM = str(dataset_root("STURM_ROOT"))
WF = str(dataset_root("WF_ROOT"))
OUT = PERSCENE_ROOT / "perscene"      # default; --out_dir overrides it

# family -> (arch, ckpt template)

# Frozen public teacher and full-split Sen1Floods11 students.
FAMILIES = {
    # "hf": public Prithvi-EO-2.0-300M-TL Sen1Floods11 checkpoint; a local .pt also works.
    "teacher":     ("teacher",         "hf"),
}

# Run-name templates for the labelled budget sweep.
_GT_CEIL_N = [20, 40, 50, 60, 80, 100, 150, 200, 252]
_SWEEPS_GT = {
    "ncgt_evit_ofa": ("efficientvit_b0", "sen_ofa", [20, 40, 50, 60, 80, 100, 150, 200, 250, 252]),
    "ncgt_evit_ce":  ("efficientvit_b0", "sen_ce", _GT_CEIL_N),
    "ncgt_evit_lkd": ("efficientvit_b0", "sen_dfkd", _GT_CEIL_N),
    "ncgt_unet_ce":  ("unet_resnet50", "sen_ce_unet", _GT_CEIL_N),
    "ncgt_unet_lkd": ("unet_resnet50", "sen_dfkd_unet", _GT_CEIL_N),
}
# nsweep uses the GT recipe (--no_amp --deterministic) and matched geometric
# and multiplicative augmentation. Each N uses its own FPS index.
# curation uses AMP-fp16 and non-deterministic kernels; compare selectors within
# that tree. GT references in curation plots are cross-recipe comparisons.
_PS_N = [20, 40, 50, 100, 150, 200, 252, 2500, 4000]
_PS_EVIT_N = [20, 40, 50, 100, 150, 200, 252, 2500]   # CE/Logit-KD: no N=4000 run
_PS_UNET_N = [20, 40, 50, 100, 150, 200, 252, 2500, 4000]
_GT_UNET_N = [20, 40, 50, 60, 80, 100, 150, 200, 252]  # human labels stop at the full split
_SWEEPS_PSN = {   # family prefix -> (arch, run-name prefix, list of N), base runs/nsweep
    "evit_ce_ps":    ("efficientvit_b0", "pseudo_ce_aug",         _PS_EVIT_N),
    "evit_lkd_ps":   ("efficientvit_b0", "pseudo_dfkd_aug",       _PS_EVIT_N),
    "evit_ofa_ps":   ("efficientvit_b0", "pseudo_ofa_aug",        _PS_N),
    "unet_ce_ps":    ("unet_resnet50",   "pseudo_ce_unet_aug",    _PS_UNET_N),
    "unet_lkd_ps":   ("unet_resnet50",   "pseudo_dfkd_unet_aug",  _PS_UNET_N),
    "noaug_evit_ce_ps":  ("efficientvit_b0", "pseudo_ce_noaug",       [252]),
    "noaug_evit_lkd_ps": ("efficientvit_b0", "pseudo_dfkd_noaug",     [252]),
    "noaug_evit_ofa_ps": ("efficientvit_b0", "pseudo_ofa_noaug",      [252]),
}
_CUR_ARMS = ["random", "random_raw", "fps", "fps_r1", "fps_raw", "confidence",
                "confidence_raw", "glister", "glister_r2", "glister_ood", "glister_flive",
                "glister_raw", "r2core", "r2core_r1", "r2core_flonly", "r2core_raw"]
_SWEEPS_CUR = {f"cur_{a}": ("efficientvit_b0", f"{a}_ofa_hs", [2500])
               for a in _CUR_ARMS}
# Matched-budget selector controls use the GT recipe and augmentation.
# The corresponding pseudo OFA sweep supplies the FPS reference.
_CUR_N252_ARMS = ["random_raw", "random", "confidence", "glister", "r2core"]
_SWEEPS_CUR.update({f"cur252_{a}": ("efficientvit_b0", f"{a}_aug_ofa_hs", [252])
                    for a in _CUR_N252_ARMS})

# Evaluate the ReLU6 deployment arm on the same perturbation grid.
FAMILIES["relu6_ofa_ps_N2500"] = (
    "efficientvit_b0_relu6", "runs/repro/repro_ofa_relu6_N2500_s{seed}/best.pt")

# B1 geometry control: Sen1Floods11 arms trained on 264 px centre crops (same extent and GSD
# as the GDACS tiles), compared with ncgt_*_N252. See experiments/b1_crop_train.sh.
_SWEEPS_B1 = {
    "b1crop_evit_ce":  ("efficientvit_b0", "sen_ce_crop264",   [252]),
    "b1crop_evit_lkd": ("efficientvit_b0", "sen_dfkd_crop264", [252]),
    "b1crop_evit_ofa": ("efficientvit_b0", "sen_ofa_crop264",  [252]),
}

# C4 labeller control: same tiles, images, recipe and seeds as evit_{ce,ofa}_ps_N2500, but
# MNDWI > 0 masks (PSEUDO_DIR=data/pseudo_gdacs_mndwi TAG=mndwi). `ce` is teacher-free;
# `ofa` still distils from the teacher and only changes the hard target.
_SWEEPS_C4 = {
    "c4mndwi_evit_ce":  ("efficientvit_b0", "pseudo_ce_mndwi",  [2500]),
    "c4mndwi_evit_ofa": ("efficientvit_b0", "pseudo_ofa_mndwi", [2500]),
}

# C1 spectral-index families: no checkpoint, load() handles the `index:` arch.
_THR_SWEEP = [f"mndwi_{k}" for k in ("m10", "m05", "p05", "p10", "p15", "p20", "p25")]
for _v in ("mndwi_otsu", "ndwi_otsu", "mndwi_zero", "ndwi_zero", *_THR_SWEEP):
    FAMILIES[f"index_{_v}"] = (f"index:{_v}", "")
del _v

for _base, _sw in (("ablations_sen1", _SWEEPS_GT),
                   ("nsweep", _SWEEPS_PSN), ("curation", _SWEEPS_CUR),
                   ("controls_geometry", _SWEEPS_B1), ("controls_labeller", _SWEEPS_C4)):
    for _fam, (_arch, _pre, _ns) in _sw.items():
        for _n in _ns:
            FAMILIES[f"{_fam}_N{_n}"] = (_arch, f"runs/{_base}/{_pre}_N{_n}_s{{seed}}/best.pt")
del _base, _sw, _fam, _arch, _pre, _ns, _n
# Default set: the teacher and the five N=252 Sen1Floods11 arms.
CORE_FAMILIES = ["teacher"] + [f"ncgt_{m}_{o}_N252" for m, o in
                               (("evit", "ce"), ("evit", "lkd"), ("evit", "ofa"),
                                ("unet", "ce"), ("unet", "lkd"))]
# --families group names, expanded before lookup
GROUPS = {"core": CORE_FAMILIES,
          "ncurve_gt": [f for f in FAMILIES if f.startswith("ncgt_")],
          # matchedcore: pseudo arms at N=252 plus the no-augmentation control; ncurve_ps: pseudo
          # OFA over N; curation: the 16 selector arms at N=2500.
          "matchedcore":     [f"{m}_{o}_ps_N252" for m, o in
                         (("evit", "ce"), ("evit", "lkd"), ("evit", "ofa"),
                          ("unet", "ce"), ("unet", "lkd"))]
                        + [f"noaug_evit_{o}_ps_N252" for o in ("ce", "lkd", "ofa")]
                        + ["evit_ofa_ps_N4000"],
          "ncurve_ps":   [f"evit_ofa_ps_N{n}" for n in _PS_N],
          "curation": [f for f in FAMILIES if f.startswith("cur_")],
          "curation252": [f"cur252_{a}_N252" for a in _CUR_N252_ARMS],
          # UNet curves, both objectives and label sources (GT stops at N=252).
          "unetcurve":  [f"ncgt_unet_{o}_N{n}" for o in ("ce", "lkd") for n in _GT_UNET_N]
                        + [f"unet_{o}_ps_N{n}" for o in ("ce", "lkd") for n in _PS_UNET_N],
          # Full-split comparison, including both label sources.
          "n252":       [f"ncgt_evit_{o}_N252" for o in ("ce", "lkd", "ofa")]
                        + [f"ncgt_unet_{o}_N252" for o in ("ce", "lkd")]
                        + [f"evit_{o}_ps_N252" for o in ("ce", "lkd", "ofa")]
                        + [f"unet_{o}_ps_N252" for o in ("ce", "lkd")],
          # The activation swap, on the same grid as everything else.
          "relu6":      ["relu6_ofa_ps_N2500"],
          # B1: the three cropped Sen1Floods11 arms.
          "b1crop":     [f"b1crop_evit_{o}_N252" for o in ("ce", "lkd", "ofa")],
          # T1 (per-band gain): N=252 EfficientViT arms (S11, cropped S11, pseudo) and the
          # N=2,500 pseudo arms.
          "t1arms":     [f"ncgt_evit_{o}_N252" for o in ("ce", "lkd", "ofa")]
                        + [f"b1crop_evit_{o}_N252" for o in ("ce", "lkd", "ofa")]
                        + [f"evit_{o}_ps_N252" for o in ("ce", "lkd", "ofa")],
          "t1arms_n2500": [f"evit_{o}_ps_N2500" for o in ("ce", "lkd", "ofa")],
          # T1 references: teacher and the two index rules on the same grid.
          "t1ref":      ["teacher", "index_mndwi_zero", "index_ndwi_zero"],
          "index":      ["index_mndwi_otsu", "index_ndwi_otsu", "index_mndwi_zero",
                         "index_ndwi_zero"],
          # Index threshold sweep.
          "indexthr":   [f"index_mndwi_{k}" for k in
                         ("m10", "m05", "zero", "p05", "p10", "p15", "p20", "p25")],
          # C4: MNDWI-labelled arms and the matching teacher-labelled arms.
          "c4":         ["c4mndwi_evit_ce_N2500", "c4mndwi_evit_ofa_N2500",
                         "evit_ce_ps_N2500", "evit_ofa_ps_N2500"],
          # T3: STURM at native 128 px (--sturm_img_size 128) for the N=2,500 arms, teacher and index.
          "sturmnative": ["teacher", "index_mndwi_zero",
                          "evit_ce_ps_N2500", "evit_lkd_ps_N2500",
                          "evit_ofa_ps_N2500", "unet_ce_ps_N2500",
                          "unet_lkd_ps_N2500"]}
# condition -> (gain, haze); clean is gain=1, haze=0.
# STURM also uses g=0.70 and 0.90 to cover non-monotone gain responses.
CONDS = {"clean": (1.0, 0.0),
         "g0.70": (0.70, 0.0), "g0.80": (0.80, 0.0), "g0.90": (0.90, 0.0),
         "g1.20": (1.20, 0.0), "g1.30": (1.30, 0.0),
         "h0.05": (1.0, 0.05), "h0.10": (1.0, 0.10), "h0.20": (1.0, 0.20)}
# Conditions restricted to one dataset (everything not listed runs on all datasets).
COND_DATASETS = {"g0.70": {"sturm"}, "g0.90": {"sturm"}}

# T1 changes band ratios, unlike scene-wide gain. Model order:
# B2, B3, B4 (VIS), B8A, B11, B12 (NIR/SWIR).
# vis: VIS*(1+-d), NIR/SWIR unchanged.
# tilt: VIS*(1+-d), NIR/SWIR*(1-+d).
# Keep d<=0.20 separate from the d=0.30 stress point: training uses
# per-channel multiplicative jitter U(0.80,1.20). See supplement Sec. S3.
_T1_D = (0.05, 0.10, 0.20, 0.30)
BAND_CONDS: dict[str, list[float]] = {}
for _d in _T1_D:
    _t = f"{int(round(_d * 100)):02d}"
    BAND_CONDS[f"t1vis+{_t}"]  = [1 + _d] * 3 + [1.0] * 3
    BAND_CONDS[f"t1vis-{_t}"]  = [1 - _d] * 3 + [1.0] * 3
    BAND_CONDS[f"t1tilt+{_t}"] = [1 + _d] * 3 + [1 - _d] * 3
    BAND_CONDS[f"t1tilt-{_t}"] = [1 - _d] * 3 + [1 + _d] * 3
del _d, _t
# Identity control: all-ones gain must give the same result as `clean`.
BAND_CONDS["t1unit"] = [1.0] * 6
# Scalar gain/haze conditions (the runs/perscene cache and the --conds default). The T1
# conditions below go to a separate cache.
SCALAR_CONDS = tuple(CONDS)
# Every T1 condition is gain 1.0 / haze 0.0 on the scalar axes; the vector carries it all.
CONDS.update({k: (1.0, 0.0) for k in BAND_CONDS})


def _sturm9(g6):
    """Model 6-vector [B2,B3,B4,B8A,B11,B12] -> STURM 9-vector [B2,B3,B4,B5,B6,B7,B8A,B11,B12].

    B5-B7 are not used by the model and get 1.0.
    """
    return [g6[0], g6[1], g6[2], 1.0, 1.0, 1.0, g6[3], g6[4], g6[5]]


# Spectral-index reference using the same loader, class map and aggregation.
# Inputs are S11-normalized B2/B3/B4/B8A/B11/B12. Recover reflectance before
# computing MNDWI=(B3-B11)/(B3+B11), as in data.pool_features._mndwi_water.
class SpectralIndexPredictor(torch.nn.Module):
    """MNDWI / NDWI water mask as a two-class segmenter.

    "mndwi_otsu" / "ndwi_otsu": Otsu threshold per chip.
    "mndwi_zero": index > 0, used for all reported index rows.
    "mndwi_p05" / "mndwi_m05" etc.: fixed thresholds +0.05 / -0.05 (threshold sweep).
    """

    _BANDS = {"mndwi": (1, 4),   # green B3, SWIR-1 B11
              "ndwi":  (1, 3)}   # green B3, NIR-narrow B8A

    def __init__(self, variant: str):
        super().__init__()
        self.index, self.rule = variant.rsplit("_", 1)
        if self.index not in self._BANDS:
            raise ValueError(f"unknown index {self.index!r}")
        if self.rule in ("zero", "otsu"):
            self.thr_value = 0.0
        elif self.rule[0] in "pm" and self.rule[1:].isdigit():
            self.thr_value = (1.0 if self.rule[0] == "p" else -1.0) * int(self.rule[1:]) / 100.0
        else:
            raise ValueError(f"unknown rule {self.rule!r}")
        from data.sen1floods11 import NORM_MEAN, NORM_STD
        self.register_buffer("mu", torch.tensor(NORM_MEAN).view(1, -1, 1, 1))
        self.register_buffer("sd", torch.tensor(NORM_STD).view(1, -1, 1, 1))

    @staticmethod
    def _otsu(v: torch.Tensor, nbins: int = 256) -> torch.Tensor:
        """Per-image Otsu threshold on a (B, H, W) map. Returns (B,)."""
        out = []
        for x in v:
            x = x[torch.isfinite(x)]
            if x.numel() == 0:
                out.append(torch.tensor(0.0, device=v.device)); continue
            lo, hi = x.min(), x.max()
            if not torch.isfinite(lo) or hi <= lo:
                out.append(lo); continue
            h = torch.histc(x, bins=nbins, min=float(lo), max=float(hi))
            p = h / h.sum()
            centres = lo + (hi - lo) * (torch.arange(nbins, device=v.device) + 0.5) / nbins
            w0 = torch.cumsum(p, 0)
            w1 = 1.0 - w0
            m0 = torch.cumsum(p * centres, 0) / w0.clamp_min(1e-12)
            mt = (p * centres).sum()
            m1 = (mt - torch.cumsum(p * centres, 0)) / w1.clamp_min(1e-12)
            var_b = torch.nan_to_num(w0 * w1 * (m0 - m1) ** 2, nan=-1.0)
            out.append(centres[int(torch.argmax(var_b))])
        return torch.stack(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho = x * self.sd + self.mu                     # back to surface reflectance
        a, b = self._BANDS[self.index]
        idx = (rho[:, a] - rho[:, b]) / (rho[:, a] + rho[:, b]).clamp_min(1e-6)
        if self.rule == "otsu":
            thr = self._otsu(idx).view(-1, 1, 1)
        else:
            thr = torch.full((idx.shape[0], 1, 1), self.thr_value, device=idx.device)
        water = (idx > thr).float()
        return torch.stack([1.0 - water, water], dim=1)  # argmax == water mask


def load(arch, ckpt, dev):
    if arch.startswith("index:"):
        return SpectralIndexPredictor(arch.split(":", 1)[1]).to(dev).eval()
    if arch == "teacher":
        m = build_teacher(variant="prithvi_eo_v2_300_tl", num_classes=2, pretrained=False,
                          freeze=True, official_ckpt=ckpt)
        return m.to(dev).eval()          # build_teacher already loaded the weights
    m = build_student(arch, num_classes=2, pretrained=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(dev).eval()


def sturm_maps():
    md = pd.read_csv(f"{STURM}/Sentinel2_metadata.csv")
    return (dict(zip(md["tile_id"].astype(str), md["floodmap_id"].astype(str))),
            dict(zip(md["tile_id"].astype(str), md["ems_code"].astype(str))))


@torch.no_grad()
def run(model, dl, dev, adapter, key_fns):
    """key_fns: (scene_of(filename), event_of(filename)). Returns {scene: 2x3}, {event: 2x3}."""
    sc = defaultdict(lambda: np.zeros((2, 3))); ev = defaultdict(lambda: np.zeros((2, 3)))
    n_tiles = n_drop = 0
    for b in dl:
        img = b["image"].to(dev)
        if adapter is not None:
            img = adapter(img)
        out = model(img)
        out = out["out"] if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)
        pred = out.argmax(1).cpu(); m = b["mask"]; v = m != -1
        for i, fn in enumerate(b["filename"]):
            cell = np.zeros((2, 3))
            for c in (0, 1):
                pc = (pred[i] == c) & v[i]; gc = (m[i] == c) & v[i]
                cell[c] = [(pc & gc).sum().item(), (pc & ~gc & v[i]).sum().item(), (~pc & gc).sum().item()]
            s, e = key_fns[0](str(fn)), key_fns[1](str(fn))
            n_tiles += 1
            # Micro scores are summed from the scene dict, so unmapped tiles are an error.
            if s: sc[s] += cell
            else: n_drop += 1
            if e: ev[e] += cell
    return sc, ev, n_tiles, n_drop


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--datasets", default="sturm,wf")
    ap.add_argument("--families", default=",".join(CORE_FAMILIES),
                    help="family names, or a group: " + ", ".join(GROUPS))
    ap.add_argument("--seeds", default="42,64,91,107")
    ap.add_argument("--conds", default=",".join(SCALAR_CONDS),
                    help="Conditions to evaluate (default: scalar gain/haze). Per-band conditions ("
                         + ", ".join(list(BAND_CONDS)[:3]) + ", ...) need --out_dir runs/perscene_T1.")
    ap.add_argument("--num_workers", type=int, default=6,
                    help="DataLoader workers; does not change results")
    ap.add_argument("--operational_protocol", action="store_true",
                    help="Count permanent water as background (STURM classes 3-5, WorldFloods "
                         "JRC permanent). Needs a separate --out_dir.")
    ap.add_argument("--sturm_class_mapping", default="paper", choices=["paper", "published"],
                    help="'paper' (default: 1-2 flood, 3-5 ignore) or 'published' (1-5 water, "
                         "needs a separate --out_dir)")
    ap.add_argument("--sturm_img_size", type=int, default=512,
                    help="Input size for STURM's 128 px chips: 512 (paper protocol) or 128 "
                         "(native). Other values need a separate --out_dir.")
    ap.add_argument("--out_dir", default=str(OUT),
                    help="Output directory for the per-scene cells (default runs/perscene)")
    ap.add_argument("--runs_base", default="runs",
                    help="Base dir the FAMILIES checkpoint templates are rooted at (default runs)")
    args = ap.parse_args()
    # File names do not store the protocol, so non-default protocols need their own --out_dir.
    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    _approved = OUT == PERSCENE_ROOT / "perscene"
    if args.sturm_class_mapping != "paper" and _approved:
        raise SystemExit("--sturm_class_mapping published requires a separate --out_dir")
    if args.operational_protocol and _approved:
        raise SystemExit("--operational_protocol requires a separate --out_dir")
    if args.sturm_img_size != 512 and _approved:
        raise SystemExit("--sturm_img_size != 512 requires a separate --out_dir")
    if any(c.strip() in BAND_CONDS for c in args.conds.split(",")) and _approved:
        raise SystemExit(
            "T1 per-band-gain conditions require a separate --out_dir, e.g.\n"
            "  python evaluation/perscene_eval.py --datasets sturm,wf \\\n"
            "      --families t1arms,t1arms_n2500 --conds <t1 conds> --out_dir runs/perscene_T1\n"
            "Use  bash experiments/perscene_controls.sh t1  instead: it fills both\n"
            "runs/perscene_T1 (the arms) and runs/perscene_T1_ref (teacher and index).")
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dss = [d.strip() for d in args.datasets.split(",")]
    fams = []
    for f in args.families.split(","):
        fams.extend(GROUPS.get(f.strip(), [f.strip()]))
    unknown = [f for f in fams if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}\nknown groups: {list(GROUPS)}")
    seeds = [s.strip() for s in args.seeds.split(",")]
    conds = [c.strip() for c in args.conds.split(",")]
    t2s, t2e = sturm_maps()
    ad = sturm_s2_adapter(target_channels=6).to(dev)
    t0 = time.time(); n_done = n_skip = 0

    for ds in dss:
        for cond in conds:
            if ds not in COND_DATASETS.get(cond, {ds}):
                continue                      # cond is scoped to another dataset (see COND_DATASETS)
            g, h = CONDS[cond]
            gb = BAND_CONDS.get(cond)          # T1 per-band vector, or None on every old cond
            if ds == "sturm":
                dl = get_sturm_flood_dataloader(STURM, sensor="S2", batch_size=8,
                                                num_workers=args.num_workers,
                                                img_size=args.sturm_img_size,
                                                normalize="original", sim_gain=g, sim_haze=h,
                                                sim_gain_bands=None if gb is None else _sturm9(gb),
                                                class_mapping=("operational" if args.operational_protocol
                                                               else args.sturm_class_mapping))
                keys = (lambda fn: t2s.get(fn), lambda fn: t2e.get(fn)); adapter = ad
            else:
                dl = get_worldfloods_dataloader(root=WF, split="test", batch_size=4,
                                                num_workers=args.num_workers,
                                                img_size=512, tile_size=512, normalize="s11",
                                                sim_gain=g, sim_haze=h, sim_gain_bands=gb,
                                                permanent_water_as=("background" if args.operational_protocol
                                                                    else "ignore"))
                # WF scene name embeds the EMS code: 'EMSR347_07ZOMBA_DEL_v2.tif#r0_c0'
                keys = (lambda fn: fn.split("#")[0],
                        lambda fn: fn.split("#")[0].split("_")[0]); adapter = None
            for fam in fams:
                arch, tmpl = FAMILIES[fam]
                # All real checkpoint templates start with "runs/"; --runs_base swaps that root.
                if args.runs_base != "runs" and tmpl.startswith("runs/"):
                    tmpl = args.runs_base + tmpl[len("runs"):]
                for sd in (["-"] if fam.startswith(("teacher", "index_")) else seeds):
                    out_p = OUT / f"{ds}__{fam}__s{sd}__{cond}.json"
                    if out_p.exists(): n_skip += 1; continue
                    ck = tmpl if sd == "-" else tmpl.format(seed=sd)
                    if not arch.startswith("index:") and ck != "hf" and not Path(ck).exists():
                        print(f"[miss] {ck}"); continue
                    sc, ev, n_t, n_d = run(load(arch, ck, dev), dl, dev, adapter, keys)
                    if n_d:
                        raise RuntimeError(f"{n_d}/{n_t} tiles had no scene key in {ds} — micro and "
                                           f"macro would both be biased. Fix the key mapping first.")
                    if n_t != len(dl.dataset):
                        raise RuntimeError(f"processed {n_t} tiles but dataset has {len(dl.dataset)}")
                    # gain_bands only for T1 conditions, so other cells stay unchanged.
                    json.dump({"dataset": ds, "family": fam, "seed": sd, "cond": cond,
                               "gain": g, "haze": h, **({"gain_bands": gb} if gb else {}),
                               "n_tiles": n_t,
                               "scene": {k: v.tolist() for k, v in sc.items()},
                               "event": {k: v.tolist() for k, v in ev.items()}},
                              open(out_p, "w"))
                    n_done += 1
                    print(f"[ok] {out_p.name}  ({n_t} tiles, {len(sc)} scenes, {n_done} done, "
                          f"{time.time()-t0:.0f}s)", flush=True)
    print(f"\n[done] {n_done} new, {n_skip} cached | {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
