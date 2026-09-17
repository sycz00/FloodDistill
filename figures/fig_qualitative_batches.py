# !/usr/bin/env python3
"""Candidate STURM and WorldFloods panels for gain and haze perturbations."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import json
import numpy as np, torch, torch.nn.functional as F
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from data.ood_datasets import get_sturm_flood_dataloader, get_worldfloods_dataloader
from data.channel_adapter import sturm_s2_adapter
from figures.qual_rgb import TRUE_COLOUR_CEILING, input_quality, to_rgb
from models.student import build_student
from models.teacher import build_teacher
from utils.paths import dataset_root

STURM = str(dataset_root("STURM_ROOT"))
WF = str(dataset_root("WF_ROOT"))
OUT = Path("assets/qual_candidates"); OUT.mkdir(parents=True, exist_ok=True)

# Pseudo arms use the N=2,500 pool, GT arms the full Sen1Floods11 split (N=252). All seed 42.
PSEUDO_N, GT_N, SEED = 2500, 252, 42
# key -> (column title, checkpoint, kind, arch)
MODELS = {
    "teacher": ("Teacher\nPrithvi 300M", "hf", "teacher", None),
    "unet":    ("UNet-KD 32.5M",
                f"runs/nsweep/pseudo_dfkd_unet_aug_N{PSEUDO_N}_s{SEED}/best.pt", "student",
                "unet_resnet50"),
    "gt":      (f"EViT 0.7M\n{GT_N} human labels",
                f"runs/ablations_sen1/sen_ofa_N{GT_N}_s{SEED}/best.pt", "student",
                "efficientvit_b0"),
    "ofa":     (f"EViT-OFA 0.7M (ours)\n{PSEUDO_N:,} routed labels",
                f"runs/nsweep/pseudo_ofa_aug_N{PSEUDO_N}_s{SEED}/best.pt", "student",
                "efficientvit_b0"),
}
COLS = ["teacher", "unet", "gt", "ofa"]                  # left-to-right after input | GT

# Header of the --jetson_npz column. The deployed model comes from the ReLU6 retraining
# (runs/repro/repro_ofa_relu6_N2500_s42), not from the OFA checkpoint in the "ofa" column,
# so the header names both the retraining and INT8.
DEV_TITLE = "EViT-OFA INT8 on Xavier\nReLU6 re-train + QAT"

# (label, baseline arm, weight). Main criterion: pseudo > GT with the same architecture and
# objective. A scene also needs OFA >= UNet; the weights only break ties.
CRITERIA = [("pseudo_vs_gt", "gt", 1.00),
            ("evit_vs_unet", "unet", 0.40)]

LEVELS = {"haze": dict(sim_haze=0.20), "gain": dict(sim_gain=1.30)}
LEVEL_LABEL = {"haze": "haze $h$=0.20", "gain": "gain $g$=1.30"}


def load(ck, kind, arch, dev):
    if kind == "teacher":
        if ck == "hf":
            # Public checkpoint: build_teacher downloads and loads it, as in perscene_eval.load.
            return build_teacher(variant="prithvi_eo_v2_300_tl", num_classes=2, pretrained=False,
                                 freeze=True, official_ckpt="hf").to(dev).eval()
        m = build_teacher(variant="prithvi_eo_v2_300_tl", num_classes=2,
                          pretrained=False, freeze=True)
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        m.load_state_dict(sd.get("model", sd), strict=True)
        return m.to(dev).eval()
    m = build_student(arch, in_channels=6, num_classes=2, pretrained=False).to(dev).eval()
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=False)
    return m


@torch.no_grad()
def predict(m, x, dev):
    lo = m(x.to(dev))
    lo = lo["out"] if isinstance(lo, dict) else (lo[0] if isinstance(lo, (list, tuple)) else lo)
    if lo.shape[-2:] != x.shape[-2:]:
        lo = F.interpolate(lo, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return lo.argmax(1).cpu()


def flood_iou(pred, gt):
    v = gt != -1
    inter = ((pred == 1) & (gt == 1) & v).sum()
    union = (((pred == 1) | (gt == 1)) & v).sum()
    return float(inter) / max(float(union), 1.0)


def make_loaders(ds, axis, img_size, display_size):
    """-> (clean_dl, corrupt_dl, clean_display_dl, corrupt_display_dl, scene_of, need_adapter)

    The display loaders only differ in resize; they draw the input column at native size.
    """
    kw = LEVELS[axis]
    if ds == "sturm":
        f = lambda sz, **k: get_sturm_flood_dataloader(STURM, sensor="S2", batch_size=8,
                                                       num_workers=6, img_size=sz,
                                                       normalize="original", **k)
        md = pd.read_csv(f"{STURM}/Sentinel2_metadata.csv")
        scene_of = dict(zip(md["tile_id"].astype(str), md["floodmap_id"].astype(str)))
        return (f(img_size), f(img_size, **kw), f(display_size), f(display_size, **kw),
                (lambda fn: scene_of.get(fn, fn)), True)
    f = lambda sz, **k: get_worldfloods_dataloader(root=WF, split="test", batch_size=4,
                                                   num_workers=6, img_size=sz, tile_size=sz,
                                                   normalize="s11", **k)
    return (f(img_size), f(img_size, **kw), f(img_size), f(img_size, **kw),
            (lambda fn: fn.split("#")[0]), False)


# plot_models.py's dataset keys, so the manifest works with plot_models.py --picks_from and
# jetson/qualitative_jetson.py --manifest.
MANIFEST_KEY = {"sturm": "STURM-S2", "wf": "WorldFloods-v2"}


def dataset_index_map(ds, key) -> dict[str, int]:
    """filename -> dataset index, without reading rasters.

    Uses the same names as __getitem__ (WorldFloods adds '#r<row>_c<col>'). Consumers look up
    tiles by name first; the index depends on the loader configuration.
    """
    if key == "sturm":
        return {img.name: i for i, (img, _lbl) in enumerate(ds.samples)}
    out = {}
    for i, (img, _gt, win) in enumerate(ds.samples):
        out[img.name if win is None else f"{img.name}#r{win[1]}_c{win[0]}"] = i
    return out


def collect(ds, axis, models, dev, adapter, max_tiles, img_size, display_size,
            ceiling, max_blown):
    dl0, dlc, dd0, ddc, scene_of, need_ad = make_loaders(ds, axis, img_size, display_size)
    cands, rejected, seen = [], {}, 0
    for b0, bc, d0, dc in zip(dl0, dlc, dd0, ddc):
        # Four unshuffled loaders (clean/perturbed, model/display size); check per batch
        # that they still return the same files.
        assert list(b0["filename"]) == list(bc["filename"]) == list(d0["filename"]) \
            == list(dc["filename"]), (
                "clean/corrupted/display loaders are out of step:\n"
                f"  model-clean   {list(b0['filename'])[:3]}\n"
                f"  model-corrupt {list(bc['filename'])[:3]}\n"
                f"  disp-clean    {list(d0['filename'])[:3]}\n"
                f"  disp-corrupt  {list(dc['filename'])[:3]}")
        x0, xc = b0["image"].to(dev), bc["image"].to(dev)
        if need_ad:
            x0, xc = adapter(x0), adapter(xc)
        x0, xc = x0.cpu(), xc.cpu()
        if seen == 0:
            # Check that the perturbed branch differs from the clean one.
            d = float((xc - x0).abs().mean())
            assert d > 1e-4, (f"{ds}/{axis}: corrupted batch is identical to the clean one "
                              f"(mean |diff| = {d:.2e}); LEVELS[{axis!r}] = {LEVELS[axis]} "
                              f"did not reach the dataset")
            print(f"  perturbation reaches the model: mean |x_corrupt - x_clean| = {d:.4f} "
                  f"(in z-score units)")
        m = b0["mask"]
        # every arm on both conditions
        p0 = {k: predict(models[k], x0, dev) for k in COLS}
        pc = {k: predict(models[k], xc, dev) for k in COLS}
        for i in range(x0.shape[0]):
            gt = m[i].numpy()
            v = gt != -1
            wf_frac = float((gt == 1)[v].mean()) if v.sum() else 0.0
            # Enough valid pixels and a water fraction between 3% and 60%.
            if v.sum() < 20000 or not (0.03 <= wf_frac <= 0.60):
                continue
            keep, _c, why = input_quality(x0[i].numpy(),
                                          "STURM-S2" if ds == "sturm" else "wf", gt)
            if not keep:
                rejected[why.split(" (")[0]] = rejected.get(why.split(" (")[0], 0) + 1
                continue
            # Skip tiles whose perturbed input saturates to white (bright scenes under
            # strong haze); the display mapping itself stays fixed.
            rgbc, _ = to_rgb(dc["image"][i], fixed=ceiling)
            blown = float((rgbc >= 0.995).all(axis=-1).mean())
            if blown > max_blown:
                rejected["perturbed input saturated"] = \
                    rejected.get("perturbed input saturated", 0) + 1
                continue
            iou0 = {k: flood_iou(p0[k][i].numpy(), gt) for k in COLS}
            iouc = {k: flood_iou(pc[k][i].numpy(), gt) for k in COLS}
            # margins on the perturbed condition
            marg = {lab: iouc["ofa"] - iouc[a] for lab, a, _w in CRITERIA}
            cands.append(dict(
                scene=scene_of(str(b0["filename"][i])), fn=str(b0["filename"][i]),
                wf=wf_frac, gt=m[i], iou0=iou0, iouc=iouc, marg=marg,
                disp0=d0["image"][i], dispc=dc["image"][i],
                pred0={k: p0[k][i] for k in COLS}, predc={k: pc[k][i] for k in COLS},
                qualifies=all(marg[lab] > 0 for lab, _a, _w in CRITERIA),
                score=sum(w * marg[lab] for lab, _a, w in CRITERIA)))
        seen += x0.shape[0]
        if seen >= max_tiles:
            break
    if rejected:
        print("  input-quality gate dropped "
              + ", ".join(f"{v} {k}" for k, v in sorted(rejected.items())))
    return cands, dataset_index_map(dl0.dataset, ds)


DS_LABEL = {"sturm": "STURM-Flood", "wf": "WorldFloods-v2"}


def load_jetson(paths, label, title):
    """Device predictions -> {axis: {tile filename: row}} for the INT8 column.

    The .npz comes from jetson/qualitative_jetson.py --dump_npz with this figure's manifest.
    Checks: format qual_jetson_v2, `preds_corrupt` present, and axis and level equal to the
    drawn ones. Keyed by (axis, level), so `--axes haze,gain` takes one .npz per axis.
    """
    by_axis = {}
    for path in paths:
        z = np.load(path, allow_pickle=False)
        if "format" not in z or str(z["format"][0]) != "qual_jetson_v2":
            raise SystemExit(
                f"{path}: not a qual_jetson_v2 dump. Before 2026-08-17 the device wrote a "
                f"single `preds` array that held the PERTURBED condition whenever an axis "
                f"was active, so it cannot fill the clean rows and must not be merged as if "
                f"it could. Re-run jetson/qualitative_jetson.py --dump_npz on the device.")
        if z["preds_corrupt"].size == 0:
            raise SystemExit(f"{path}: no perturbed predictions (the device run had no "
                             f"--haze/--gain); it can only fill the clean rows")
        labels = [str(s) for s in z["labels"]]
        if label not in labels:
            raise SystemExit(f"{path}: no device column named {label!r}. Available: "
                             + ", ".join(repr(x) for x in labels))
        m = labels.index(label)
        key = (str(z["axis"][0]), round(float(z["level"][0]), 6))
        dev = by_axis.setdefault(key, {"title": title, "label": label, "rows": {}})["rows"]
        for i, name in enumerate(z["names"]):
            dev[str(name)] = dict(pred0=z["preds"][i][m], predc=z["preds_corrupt"][i][m],
                                  iou0=float(z["ious"][i][m]),
                                  iouc=float(z["ious_corrupt"][i][m]),
                                  gt=z["gt"][i], src=Path(path).name,
                                  model_dir=str(z["model_dir"][0]), pool=str(z["pool"][0]))
        print(f"[jetson] {Path(path).name}: {len(z['names'])} tile(s) at {key[0]}={key[1]:g}, "
              f"column {label!r} from {str(z['model_dir'][0])} ({str(z['pool'][0])})")
    return by_axis


def rank(cands, ds, axis):
    """Qualifying scenes first (every margin positive), then by weighted score."""
    cands.sort(key=lambda c: (not c["qualifies"], -c["score"]))
    nq = sum(c["qualifies"] for c in cands)
    gt_m = np.array([c["marg"]["pseudo_vs_gt"] for c in cands]) if cands else np.zeros(1)
    print(f"  {ds}/{axis}: {nq}/{len(cands)} candidates win on every criterion at "
          f"{LEVEL_LABEL[axis]}   OFA−GT margin: median {np.median(gt_m):+.3f}, "
          f"mean {gt_m.mean():+.3f}, p90 {np.percentile(gt_m, 90):+.3f}")
    return cands


def _device_row(dev, c, ds):
    """Device entry for a drawn tile, or SystemExit naming the tile.

    Tiles are matched by file name; the ground-truth masks must also be equal, which catches a
    different image size, STURM normalisation, permanent-water rule or tiling.
    """
    row = dev["rows"].get(c["fn"])
    if row is None:
        raise SystemExit(
            f"the device dump has no tile {c['fn']!r} ({DS_LABEL[ds]}), which this panel "
            f"drew. Run that tile on the device -- the manifest this figure writes names "
            f"it -- or drop --jetson_npz. Present in the dump: "
            + ", ".join(sorted(dev["rows"])[:6]) + (" ..." if len(dev["rows"]) > 6 else ""))
    gt_here = c["gt"].numpy()
    if row["gt"].shape != gt_here.shape or not np.array_equal(row["gt"], gt_here):
        n = int((row["gt"] != gt_here).sum()) if row["gt"].shape == gt_here.shape else -1
        raise SystemExit(
            f"{c['fn']}: the device and this figure decoded DIFFERENT ground truth "
            f"(shapes {row['gt'].shape} vs {gt_here.shape}, {n} differing pixels). The two "
            f"runs are not the same experiment; check --img_size, --sturm_normalize and the "
            f"WorldFloods permanent-water flag on the device side.")
    return row


def emit_paired(by_ds, axis, n_batches, ceiling, only=None, drawn=None, dev=None):
    """One figure per batch with four rows: STURM clean/perturbed, WorldFloods clean/perturbed.

    `dev` (from load_jetson) adds the INT8 column on the right. Selection does not depend on
    the device tiles; a drawn tile missing from the .npz is an error.
    """
    order = ["sturm", "wf"]
    picked = {ds: [] for ds in order}
    used_scene = {ds: {} for ds in order}
    water = ListedColormap(["#C8C8C8", "#0072B2"])
    titles = ["input", "ground truth"] + [MODELS[k][0] for k in COLS] \
        + ([dev["title"]] if dev else [])
    nc = 2 + len(COLS) + (1 if dev else 0)

    for b in range(n_batches):
        # one tile per dataset per batch; a scene may return at most twice across the set
        chosen = {}
        for ds in order:
            for c in by_ds[ds]:
                if any(c is p for p in picked[ds]) or used_scene[ds].get(c["scene"], 0) >= 2:
                    continue
                chosen[ds] = c
                picked[ds].append(c)
                used_scene[ds][c["scene"]] = used_scene[ds].get(c["scene"], 0) + 1
                break
        if len(chosen) < len(order):
            break
        # Selection runs over all batches so the numbering matches a full run.
        if only is not None and (b + 1) not in only:
            continue

        fig, axes = plt.subplots(2 * len(order), nc, figsize=(2.05 * nc, 2.20 * 2 * len(order)))
        for d, ds in enumerate(order):
            c = chosen[ds]
            gtn = c["gt"].numpy().astype(float)
            gtm = np.where(gtn == -1, np.nan, gtn)
            dv = _device_row(dev, c, ds) if dev else None
            # One fixed display mapping for all rows (see qual_rgb).
            rgb0, _ = to_rgb(c["disp0"], fixed=ceiling)
            rgbc, _ = to_rgb(c["dispc"], fixed=ceiling)
            for sub, (img, pred, iou, tag) in enumerate(
                    [(rgb0, c["pred0"], c["iou0"], "clean"),
                     (rgbc, c["predc"], c["iouc"], LEVEL_LABEL[axis])]):
                rr = 2 * d + sub
                ims = [img, gtm] + [pred[k].numpy() for k in COLS] \
                    + ([dv["pred0" if sub == 0 else "predc"]] if dv else [])
                for k, im in enumerate(ims):
                    ax = axes[rr, k]
                    # Column 0 is the RGB input, the others are water maps (ignore pixels are NaN).
                    ax.imshow(im, **({} if k == 0 else dict(cmap=water, vmin=0, vmax=1)),
                              interpolation="nearest")
                    ax.set_xticks([]); ax.set_yticks([])
                    if rr == 0:
                        ax.set_title(titles[k], fontsize=8.5)
                for k, key in enumerate(COLS):
                    axes[rr, 2 + k].set_xlabel(f"flood-IoU {iou[key]:.2f}", fontsize=7.5)
                if dv:
                    axes[rr, 2 + len(COLS)].set_xlabel(
                        f"flood-IoU {dv['iou0' if sub == 0 else 'iouc']:.2f}", fontsize=7.5)
                axes[rr, 0].set_ylabel(f"{DS_LABEL[ds]}\n{c['scene'][:18]}\n{tag}\n"
                                       f"water {100*c['wf']:.0f}%", fontsize=6.2)
            # Margins below the OFA column of the perturbed row (indexed from the left
            # because --jetson_npz adds a last column).
            axes[2 * d + 1, 1 + len(COLS)].set_xlabel(
                f"flood-IoU {c['iouc']['ofa']:.2f}\n"
                + "\n".join(f"−{a.upper()}  {c['marg'][lab]:+.2f}" for lab, a, _w in CRITERIA),
                fontsize=6.8, color="#00703C" if c["qualifies"] else "0.35")
            for sub in (0, 1):                       # frame the pair so the rows group
                for ax in axes[2 * d + sub]:
                    for sp in ax.spines.values():
                        sp.set_color("0.75" if sub == 0 else "#B07AA1")
                        sp.set_linewidth(1.1)
        fig.legend(handles=[Patch(facecolor="#C8C8C8", edgecolor="0.5", label="land"),
                            Patch(facecolor="#0072B2", edgecolor="0.5", label="water/flood"),
                            Patch(facecolor="white", edgecolor="0.5", label="ignore")],
                   loc="lower center", ncol=3, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.012))
        fig.suptitle(
            f"batch {b+1}  ·  rows 1–2 STURM-Flood, rows 3–4 WorldFloods-v2  ·  "
            f"clean above, {LEVEL_LABEL[axis]} below\n"
            f"true colour = Sentinel-2 True Color, $\\min(\\rho/{ceiling:g},\\,1)$ on B4/B3/B2 "
            f"— zero-referenced and identical in every row, so the corruption is shown at "
            f"its true magnitude", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.015, 1, 0.955))
        # The level is part of the file name so h=0.10 and h=0.20 do not overwrite each other.
        lvl = (LEVELS[axis].get("sim_haze") or LEVELS[axis].get("sim_gain"))
        p = OUT / f"qual_{axis}{round(100 * lvl):03d}_batch{b+1:02d}.png"
        fig.savefig(p, bbox_inches="tight", dpi=140)
        plt.close(fig)
        print(f"  [ok] {p.name}  " + " | ".join(
            f"{ds} {chosen[ds]['scene'][:12]} {'*' if chosen[ds]['qualifies'] else ' '}"
            + " ".join(f"{k}{chosen[ds]['iouc'][k]:.2f}" for k in COLS) for ds in order))
        if drawn is not None:
            dvs = {ds: (_device_row(dev, chosen[ds], ds) if dev else None) for ds in order}
            drawn.append({"panel": p.name, "batch": b + 1,
                          **({"jetson_column": {
                              "title": dev["title"], "device_model": dev["label"],
                              "npz": sorted({v["src"] for v in dvs.values()}),
                              "model_dir": sorted({v["model_dir"] for v in dvs.values()}),
                              "pool": sorted({v["pool"] for v in dvs.values()}),
                              "flood_iou": {ds: {"clean": round(dvs[ds]["iou0"], 4),
                                                 "perturbed": round(dvs[ds]["iouc"], 4)}
                                            for ds in order}}} if dev else {}),
                          "rows": [{"dataset": ds, "id": chosen[ds]["fn"],
                                    "scene": chosen[ds]["scene"],
                                    "water_frac": round(chosen[ds]["wf"], 4),
                                    "flood_iou_clean": {k: round(chosen[ds]["iou0"][k], 4)
                                                        for k in COLS},
                                    "flood_iou_perturbed": {k: round(chosen[ds]["iouc"][k], 4)
                                                            for k in COLS}}
                                   for ds in order]})


def write_manifest(axis, drawn, index_of, a):
    """Write the drawn scenes in the plot_models.py manifest format.

    `perturbation` adds the axis and level that the device run has to apply.
    """
    if not drawn:
        return
    picks = {}
    for panel in drawn:
        for row in panel["rows"]:
            key = MANIFEST_KEY[row["dataset"]]
            idx = index_of[row["dataset"]].get(row["id"])
            if idx is None:
                raise SystemExit(f"{row['id']!r} is not in the {row['dataset']} sample list; "
                                 f"the index map and the loader disagree")
            entry = {"idx": idx, "id": row["id"]}
            if entry not in picks.setdefault(key, []):
                picks[key].append(entry)
    lvl = LEVELS[axis].get("sim_haze") or LEVELS[axis].get("sim_gain")
    man = {
        "generated_by": "figures/fig_qualitative_batches.py",
        "perturbation": {"axis": axis, "level": lvl,
                         "apply_with": f"--{axis} {lvl:g}"},
        "loader": {"eval_img_size": a.img_size, "sturm_normalize": "original",
                   "wf": f"split=test, tile_size={a.img_size}, normalize=s11",
                   "s11_split": "n/a (not evaluated by this figure)"},
        "rgb": {"map": "min(rho/K, 1) on B4/B3/B2", "ceiling": a.ceiling},
        "picks": picks,
        "panels": drawn,
    }
    out = OUT / f"qual_robust_manifest_{axis}{round(100 * lvl):03d}.json"
    out.write_text(json.dumps(man, indent=2))
    print(f"  manifest -> {out}  ("
          + ", ".join(f"{k} {len(v)}" for k, v in picks.items()) + ")")


def main():
    global OUT                      # must precede the first read of OUT, or SyntaxError
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--axes", default="haze,gain")
    ap.add_argument("--max_tiles", type=int, default=900)
    ap.add_argument("--img_size", type=int, default=512, help="model input size")
    ap.add_argument("--display_size", type=int, default=128,
                    help="size the input column is RENDERED at; STURM chips are natively "
                         "128x128, so 512 would show a 4x bilinear upsample")
    ap.add_argument("--ceiling", type=float, default=TRUE_COLOUR_CEILING,
                    help="fixed true-colour reflectance ceiling (rho/ceiling, clipped)")
    ap.add_argument("--haze", type=float, default=LEVELS["haze"]["sim_haze"],
                    help="Haze level of the perturbed row (0.10 in Fig. S2, 0.20 is the stress level)")
    ap.add_argument("--gain", type=float, default=LEVELS["gain"]["sim_gain"])
    ap.add_argument("--max_blown", type=float, default=0.20,
                    help="Reject tiles whose perturbed input has more white pixels than this "
                         "fraction; 1.0 disables it")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--only_batches", default=None,
                    help="Comma-separated batches to draw, e.g. '1,5'; numbering as in a full run")
    ap.add_argument("--jetson_npz", default=None,
                    help="Comma-separated .npz files from jetson/qualitative_jetson.py --dump_npz; "
                         "adds the device column. Use with --only_batches.")
    ap.add_argument("--jetson_model", default="TRT-INT8",
                    help="Device column from the .npz `labels`: 'TRT-INT8', 'W8-INT8 (ORT)' "
                         "or 'Float BN-folded'")
    ap.add_argument("--jetson_title", default=None,
                    help="Header of the device column (default names the ReLU6 retraining and INT8)")
    a = ap.parse_args()
    LEVELS["haze"]["sim_haze"], LEVELS["gain"]["sim_gain"] = a.haze, a.gain
    LEVEL_LABEL["haze"] = f"haze $h$={a.haze:.2f}"
    LEVEL_LABEL["gain"] = f"gain $g$={a.gain:.2f}"
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    # Check the device dumps before loading models; every requested axis needs one.
    jetson = (load_jetson([s.strip() for s in a.jetson_npz.split(",")], a.jetson_model,
                          a.jetson_title or DEV_TITLE) if a.jetson_npz else None)
    if jetson:
        want = {(ax.strip(),
                 round(LEVELS[ax.strip()]["sim_haze" if ax.strip() == "haze" else "sim_gain"],
                       6)) for ax in a.axes.split(",")}
        if not want <= set(jetson):
            raise SystemExit(
                "--jetson_npz does not cover every --axes level. Requested "
                + ", ".join(f"{k[0]}={k[1]:g}" for k in sorted(want)) + "; the dump(s) carry "
                + ", ".join(f"{k[0]}={k[1]:g}" for k in sorted(jetson))
                + ". Run the device at the missing level or narrow --axes.")
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    adapter = sturm_s2_adapter(target_channels=6).to(dev)
    # Check checkpoints first: build_student(strict=False) accepts wrong state dicts.
    # "hf" is the public teacher on the Hugging Face hub, not a local file.
    missing = [ck for _t, ck, kind, _ar in MODELS.values()
               if not (kind == "teacher" and ck == "hf") and not Path(ck).exists()]
    if missing:
        raise SystemExit("checkpoint(s) not found:\n  " + "\n  ".join(missing))
    models = {k: load(ck, kind, arch, dev) for k, (_t, ck, kind, arch) in MODELS.items()}
    # Each panel pairs a STURM and a WorldFloods scene, so collect both first.
    only = ({int(x) for x in a.only_batches.split(",")} if a.only_batches else None)
    index_of = {}
    for axis in a.axes.split(","):
        axis = axis.strip()
        print(f"\n=== {axis} ===")
        by_ds = {}
        for ds in ("sturm", "wf"):
            cands, index_of[ds] = collect(
                ds, axis, models, dev, adapter, a.max_tiles, a.img_size,
                a.display_size if ds == "sturm" else a.img_size, a.ceiling, a.max_blown)
            by_ds[ds] = rank(cands, ds, axis)
        # Not `dev`, which is the torch device.
        devcol = None
        if jetson:
            lvl = round(LEVELS[axis]["sim_haze" if axis == "haze" else "sim_gain"], 6)
            devcol = jetson.get((axis, lvl))
            if devcol is None:
                raise SystemExit(
                    f"--jetson_npz has no dump at {axis} = {lvl:g}; it carries "
                    + ", ".join(f"{k[0]}={k[1]:g}" for k in jetson)
                    + f". A panel column is one axis at one level, so run the device at "
                      f"{axis} {lvl:g} or drop '{axis}' from --axes.")
        drawn = []
        emit_paired(by_ds, axis, a.batches, a.ceiling, only=only, drawn=drawn, dev=devcol)
        write_manifest(axis, drawn, index_of, a)
    print(f"\n[done] -> {OUT}")


if __name__ == "__main__":
    main()
