"""Qualitative panels at selected training budgets and deployment precisions."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import compute_seg_metrics
from utils.paths import dataset_root
from figures.qual_rgb import TRUE_COLOUR_CEILING, input_quality, to_rgb

# Project constants

BANDS           = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
ABLATIONS_ROOT  = ROOT / "runs" / "nsweep"
TEACHER_CKPT    = "hf"      # the public Prithvi checkpoint; a local .pt path also works
GDACS_DIR       = str(ROOT / "data" / "pseudo_gdacs")

SEEDS         = [42, 64, 91, 107]
# Two budgets: few labels and the full pool. Selected scenes must hold at both.
LOW_N_VALUES  = [50]
FULL_N        = 2500                       # the full curated pool
N_VALUES      = LOW_N_VALUES + [FULL_N]

OFA_PREFIX,  OFA_ARCH  = "pseudo_ofa_aug",       "efficientvit_b0"
# Logit-KD with the same architecture and labels as OFA; only the objective differs.
LKD_PREFIX,  LKD_ARCH  = "pseudo_dfkd_aug",      "efficientvit_b0"
UNET_PREFIX, UNET_ARCH = "pseudo_dfkd_unet_aug", "unet_resnet50"
# Hand-labelled EfficientViT arm (same architecture and objective). It stops at N=252, the
# full Sen1Floods11 train split, so the N=2,500 panel compares against N=252.
GT_ROOT                = ROOT / "runs" / "ablations_sen1"
GT_PREFIX,   GT_ARCH   = "sen_ofa",              "efficientvit_b0"
GT_MAX_N               = 252
# STURM scores with normalize="original"; results from the older percentile protocol are not read.
EVAL_JSON_NAMES = ["eval_results.json"]


_AGREE_BOTH    = (1.0, 1.0, 1.0)
_AGREE_B_ONLY  = (0.20, 0.75, 0.20)   # model_b (OFA / INT8) right, model_a wrong
_AGREE_A_ONLY  = (0.85, 0.25, 0.10)   # model_a (UNet-KD / Float) right, model_b wrong
_AGREE_NEITHER = (0.15, 0.15, 0.15)
_AGREE_NODATA  = (0.60, 0.60, 0.60)


def n_label(n: int) -> str:
    return "2500 (full pool)" if n == FULL_N else str(n)


def n_filename(n: int) -> str:
    return "N2500_fullpool" if n == FULL_N else f"N{n}"


# Dataset construction

@dataclass
class DatasetSpec:
    key: str
    dataset: Any
    adapter: Any            # ChannelAdapter or None
    mask_is_gt: bool         # False only for GDACS (pseudo-label, not real GT)


def build_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    from data.sen1floods11 import get_dataloaders
    from data.ood_datasets import get_sturm_flood_dataloader, get_worldfloods_dataloader
    from data.channel_adapter import sturm_s2_adapter
    from data.pseudo_labeled import PseudoLabeledDataset

    specs = []

    _, _, s11_test_dl = get_dataloaders(
        data_root=args.s11_root, batch_size=1, num_workers=0,
        eval_img_size=args.eval_img_size,
    )
    specs.append(DatasetSpec("S11", s11_test_dl.dataset, None, True))

    # Set normalize="original" (0.04 + 0.12*v) explicitly; the loader default is "percentile".
    sturm_dl = get_sturm_flood_dataloader(
        root=args.sturm_root, sensor="S2", batch_size=1, num_workers=0,
        img_size=args.eval_img_size, normalize="original", protocol="flood_paper",
    )
    specs.append(DatasetSpec("STURM-S2", sturm_dl.dataset, sturm_s2_adapter(6), True))

    # WorldFloods-v2 with the table protocol: flood-only, JRC permanent water ignored,
    # 512 px tiles, normalize="s11".
    wf_dl = get_worldfloods_dataloader(
        root=args.wf_root, split="test", batch_size=1, num_workers=0,
        img_size=args.eval_img_size, tile_size=args.eval_img_size, normalize="s11",
        ignore_permanent_water=True,
    )
    specs.append(DatasetSpec("WorldFloods-v2", wf_dl.dataset, None, True))

    gdacs_ds = PseudoLabeledDataset(
        args.gdacs_dir, n_samples=None, use_curated=True, index_file=args.gdacs_index,
    )
    specs.append(DatasetSpec("GDACS", gdacs_ds, None, False))

    if args.datasets:
        wanted = set(args.datasets)
        specs = [s for s in specs if s.key in wanted]
    return specs


def get_sample_id(spec: DatasetSpec, idx: int) -> str:
    """File name of a sample, so manifests do not depend on loader indices."""
    item = spec.dataset[idx]
    if "filename" in item:
        return item["filename"]
    if spec.key == "GDACS":
        return spec.dataset.files[idx].name
    for attr in ("image_files", "files", "image_paths"):
        seq = getattr(spec.dataset, attr, None)
        if seq is not None and len(seq) == len(spec.dataset):
            return Path(str(seq[idx])).name
    return f"{spec.key}_{idx}"


# Candidate scanning (mask-only, no model)

_VIS_BANDS = (0, 1, 2)   # BLUE, GREEN, RED

# Fixed z-score thresholds for bright (cloudy) visible pixels. With the S11 normalisation,
# z=2.1 corresponds to about 30% mean visible reflectance. STURM is stretched to [0,1]
# before normalisation, so it has its own threshold (3.0, above its p95 of 2.9).
_CLOUD_BRIGHT_THRESH = {
    "S11":            2.1,
    "WorldFloods-v2": 2.1,   # normalize="s11", so the same z-score cutoff applies
    "GDACS":          2.1,
    "STURM-S2":       3.0,
}


def cloud_fraction(img_chw: np.ndarray, bright_thresh: float,
                   vis_bands: tuple[int, ...] = _VIS_BANDS) -> float:
    """Fraction of pixels whose mean visible-band value exceeds `bright_thresh`."""
    vis = img_chw[list(vis_bands)].mean(axis=0)
    return float((vis > bright_thresh).mean())


def scan_candidates(
    dataset,
    key: str,
    max_scan: int,
    pool_size: int,
    min_valid_frac: float,
    flood_frac_range: tuple[float, float],
    bright_thresh: float,
    max_cloud_frac: float = 0.25,
    seed: int = 0,
) -> dict[int, float]:
    """Find scenes with mostly valid pixels, moderate flood extent and little cloud.

    Scans up to `max_scan` items in dataset order and subsamples `pool_size` with a fixed
    seed. Returns {idx: cloud_frac}; the cloud fraction is used again as a tie-breaker.
    """
    satisfying: dict[int, float] = {}
    dropped: dict[str, int] = {}
    n = min(len(dataset), max_scan)
    for i in range(n):
        item = dataset[i]
        mask = item["mask"]
        mask_np = mask.numpy() if torch.is_tensor(mask) else np.asarray(mask)
        valid = mask_np != -1
        valid_frac = float(valid.mean())
        if valid_frac < min_valid_frac:
            continue
        flood_frac = float((mask_np[valid] == 1).mean()) if valid.any() else 0.0
        if not (flood_frac_range[0] <= flood_frac <= flood_frac_range[1]):
            continue
        img = item["image"]
        img_np = img.numpy() if torch.is_tensor(img) else np.asarray(img)
        cf = cloud_fraction(img_np, bright_thresh)
        if cf > max_cloud_frac:
            continue
        ok, _contrast, why = input_quality(img_np, key, mask_np)
        if not ok:
            dropped[why.split(" (")[0].split(" ")[0]] = dropped.get(
                why.split(" (")[0].split(" ")[0], 0) + 1
            continue
        satisfying[i] = cf

    if dropped:
        print(f"[plot_models] {key}: input-quality gate dropped "
              + ", ".join(f"{v} {k}" for k, v in sorted(dropped.items())))
    if len(satisfying) > pool_size:
        rng = random.Random(seed)
        keep = rng.sample(list(satisfying), pool_size)
        satisfying = {i: satisfying[i] for i in sorted(keep)}
    return satisfying


# Model builders

def build_student_model(arch: str, ckpt_path, device: torch.device):
    from models.student import build_student
    model = build_student(arch=arch, in_channels=6, num_classes=2, pretrained=False)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("model", payload), strict=True)
    return model.to(device).eval()


def build_teacher_model(device: torch.device, ckpt_path: str = TEACHER_CKPT):
    from models.teacher import build_teacher
    model = build_teacher(
        variant="prithvi_eo_v2_300_tl", num_classes=2, pretrained=False,
        official_ckpt=ckpt_path,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()


def pick_representative_seed(
    prefix: str, n: int, seeds: list[int] = SEEDS, metric: str = "sturm_mIoU",
    root: Path | None = None,
) -> tuple[int, Path]:
    """Pick the seed whose `metric` is closest to the median over the four seeds."""
    vals: dict[int, float] = {}
    for s in seeds:
        for name in EVAL_JSON_NAMES:
            p = (root or ABLATIONS_ROOT) / f"{prefix}_N{n}_s{s}" / name
            if p.exists():
                v = json.loads(p.read_text()).get(metric)
                if v is not None:
                    vals[s] = v
                break
    if not vals:
        raise FileNotFoundError(f"No eval_results.json for {prefix} N={n} "
                                f"(seeds {seeds}) — run evaluation/eval_ablations.py first")
    med  = statistics.median(vals.values())
    seed = min(vals, key=lambda s: abs(vals[s] - med))
    ckpt = (root or ABLATIONS_ROOT) / f"{prefix}_N{n}_s{seed}" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    return seed, ckpt


# Inference over a bounded candidate set

@torch.no_grad()
def evaluate_model_on_indices(
    model, dataset, indices: list[int], adapter, device: torch.device, batch_size: int = 8,
) -> dict[int, dict]:
    """Returns {idx: {"pred", "flood_iou", "miou", "flood_pred_frac"}}.

    flood_pred_frac is the predicted flood share; it filters out empty predictions.
    """
    from torch.utils.data import Subset, DataLoader

    if not indices:
        return {}
    dl  = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                     shuffle=False, num_workers=0)
    out: dict[int, dict] = {}
    ptr = 0
    for batch in dl:
        images = batch["image"].float().to(device)
        masks  = batch["mask"].long()
        if adapter is not None:
            images = adapter.to(device)(images)
        logits = model(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1).cpu()
        for b in range(images.shape[0]):
            idx = indices[ptr]
            m = compute_seg_metrics(logits[b:b + 1].cpu(), masks[b:b + 1],
                                    num_classes=2, ignore_index=-1)
            iou = m["iou_per_class"]
            pred_b = preds[b].numpy().astype(np.uint8)
            out[idx] = {
                "pred":            pred_b,
                "flood_iou":       iou[1] if len(iou) > 1 else float("nan"),
                "miou":            m["mIoU"],
                "flood_pred_frac": float((pred_b == 1).mean()),
            }
            ptr += 1
    return out


# Hero-sample selection

# Selection criteria with decreasing weights: 1. OFA > Logit-KD (objective),
# 2. pseudo > GT (label source), 3. EfficientViT > UNet (architecture).
# A scene qualifies only if all three margins are positive.
CRITERIA = [("ofa_vs_lkd",  "lkd",  1.00),
            ("pseudo_vs_gt", "gt",  0.60),
            ("evit_vs_unet", "unet", 0.30)]


def select_hero_samples(
    dataset_key: str,
    candidates: dict[str, dict[int, float]],
    results: dict[int, dict],
    n_values: list[int],
    k: int,
    teacher_preds: dict[str, dict],
    min_gain: float = 0.0,
    cloud_weight: float = 1.5,
    min_pred_flood_frac: float = 0.01,
) -> list[int]:
    """Rank candidates by the CRITERIA above, using the smallest margin over `n_values`.

    Margin = OFA flood IoU minus the other arm's flood IoU. Scenes where an arm predicts
    almost no flood are dropped. Returns up to `k` indices, best first.
    """
    arms_needed = ["ofa"] + [a for _lab, a, _w in CRITERIA]
    scored: dict[int, dict] = {}
    for idx, cloud_frac in candidates[dataset_key].items():
        if teacher_preds.get(dataset_key, {}).get(idx, {}).get(
                "flood_pred_frac", 1.0) <= min_pred_flood_frac:
            continue                      # teacher predicts no flood here

        per_crit: dict[str, list[float]] = {lab: [] for lab, _a, _w in CRITERIA}
        ok = True
        for n in n_values:
            res = results[n][dataset_key]
            arms = {a: res[a][idx] for a in arms_needed if a in res and idx in res[a]}
            if len(arms) < len(arms_needed):
                ok = False; break
            if any(e["flood_pred_frac"] <= min_pred_flood_frac for e in arms.values()):
                ok = False; break         # empty prediction for one arm
            if any(e["flood_iou"] != e["flood_iou"] for e in arms.values()):
                ok = False; break         # NaN: flood class absent from pred AND gt
            for lab, a, _w in CRITERIA:
                per_crit[lab].append(arms["ofa"]["flood_iou"] - arms[a]["flood_iou"])
        if not ok:
            continue

        worst = {lab: min(v) for lab, v in per_crit.items()}
        scored[idx] = dict(worst,
                           qualifies=all(worst[lab] > min_gain for lab, _a, _w in CRITERIA),
                           cloud_frac=cloud_frac,
                           score=sum(w * worst[lab] for lab, _a, w in CRITERIA)
                                 - cloud_weight * cloud_frac)

    if not scored:
        print(f"[plot_models] {dataset_key}: no scene with all four arms evaluated.")
        return []
    strict = {i: s for i, s in scored.items() if s["qualifies"]}
    if not strict:
        print(f"[plot_models] WARNING: no {dataset_key} scene satisfies all three criteria "
              f"(OFA>LKD, pseudo>GT, EViT>UNet) at every N in {n_values}; ranking on the "
              f"weighted score anyway -- some margin below is negative.")
    pool = strict or scored
    ranked = sorted(pool.items(), key=lambda kv: kv[1]["score"], reverse=True)
    for rank, (idx, s) in enumerate(ranked[:k], 1):
        print(f"[plot_models]   {dataset_key} #{rank}: idx {idx}  "
              + "  ".join(f"{lab} {s[lab]:+.3f}" for lab, _a, _w in CRITERIA)
              + f"  | score {s['score']:+.3f}  cloud {100*s['cloud_frac']:.0f}%"
              + ("" if s["qualifies"] else"   [does NOT satisfy all three]"))
    return [idx for idx, _ in ranked[:k]]


# Plotting

_CAT_NEITHER, _CAT_BOTH, _CAT_B_ONLY, _CAT_A_ONLY, _CAT_NODATA = 0, 1, 2, 3, 4
_CAT_COLOR = {
    _CAT_NEITHER: _AGREE_NEITHER,
    _CAT_BOTH:    _AGREE_BOTH,
    _CAT_B_ONLY:  _AGREE_B_ONLY,
    _CAT_A_ONLY:  _AGREE_A_ONLY,
    _CAT_NODATA:  _AGREE_NODATA,
}


def _denoise_categorical(label_map: np.ndarray, kernel: int) -> np.ndarray:
    """Median filter on the agreement map to remove single-pixel noise at edges."""
    if kernel <= 1:
        return label_map
    from scipy.ndimage import median_filter
    return median_filter(label_map, size=kernel, mode="nearest")


def build_agreement_map(
    gt: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, denoise_kernel: int = 3,
) -> np.ndarray:
    """Per-pixel agreement between two predictions against `gt`. pred_b is the
    "highlighted" model (OFA / INT8); pred_a is the baseline (UNet-KD / Float)."""
    valid     = gt != -1
    a_correct = (pred_a == gt) & valid
    b_correct = (pred_b == gt) & valid

    cat = np.full(gt.shape, _CAT_NEITHER, dtype=np.int64)
    cat[a_correct & b_correct]  = _CAT_BOTH
    cat[b_correct & ~a_correct] = _CAT_B_ONLY
    cat[a_correct & ~b_correct] = _CAT_A_ONLY
    cat[~valid]                 = _CAT_NODATA

    if denoise_kernel > 1:
        nodata = cat == _CAT_NODATA
        cat = _denoise_categorical(cat, denoise_kernel)
        cat[nodata] = _CAT_NODATA   # never let despeckle bleed over real no-data

    out = np.empty(gt.shape + (3,), dtype=np.float32)
    for c, color in _CAT_COLOR.items():
        out[cat == c] = color
    return out


def _agreement_legend(label_a: str, label_b: str):
    import matplotlib.patches as mpatches
    return [
        mpatches.Patch(color=_AGREE_B_ONLY,  label=f"{label_b} correct, {label_a} wrong"),
        mpatches.Patch(color=_AGREE_A_ONLY,  label=f"{label_a} correct, {label_b} wrong"),
        mpatches.Patch(facecolor=_AGREE_BOTH, edgecolor="grey", label="both correct"),
        mpatches.Patch(color=_AGREE_NEITHER, label="both wrong"),
        mpatches.Patch(color=_AGREE_NODATA,  label="no data"),
    ]


def _to_rgb_display(
    img_chw: np.ndarray, rgb_idx: list[int] | None = None,
    p_low: float = 2.0, p_high: float = 98.0, gamma: float = 1.0, sat: float = 1.0,
    fixed: float | None = None,
) -> np.ndarray:
    """Render with qual_rgb.to_rgb. `rgb_idx` is ignored; RGB_BANDS selects the channels."""
    return to_rgb(img_chw, p_low=p_low, p_high=p_high, gamma=gamma, sat=sat,
                  per_channel=False, fixed=fixed)[0]


def save_figure(
    n: int,
    specs: list[DatasetSpec],
    picks: dict[str, list[int]],
    results: dict[int, dict],
    teacher_preds: dict[str, dict],
    out_path: Path,
    show_agreement: bool = True,
    denoise_kernel: int = 3,
    font_scale: float = 1.9,
    dpi: int = 300,
    paper_clean: bool = False,
    rgb_p_low: float = 2.0,
    rgb_p_high: float = 98.0,
    rgb_gamma: float = 1.0,
    rgb_sat: float = 1.0,
    rgb_fixed: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fs_title = round(9 * font_scale)    # per-panel column titles
    fs_supt  = round(13 * font_scale)   # overall suptitle
    fs_leg   = round(8 * font_scale)    # agreement legend

    spec_by_key = {s.key: s for s in specs}
    rows = [(key, idx) for key in picks for idx in picks[key] if key in spec_by_key]
    if not rows:
        print(f"[plot_models] N={n_label(n)}: nothing to plot, skipping")
        return

    ncols = (8 if show_agreement else 7)   # RGB, GT, teacher, GT-EViT, LKD, UNet, OFA
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.7 * ncols, 3.6 * len(rows)))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for r, (key, idx) in enumerate(rows):
        spec = spec_by_key[key]
        img  = spec.dataset[idx]["image"].numpy()
        gt   = spec.dataset[idx]["mask"].numpy()
        rgb  = _to_rgb_display(img, None, rgb_p_low, rgb_p_high, rgb_gamma, rgb_sat,
                               fixed=rgb_fixed)

        teacher = teacher_preds[key][idx]
        unet    = results[n][key]["unet"][idx]
        ofa     = results[n][key]["ofa"][idx]
        gtm     = results[n][key]["gt"][idx]
        gt_n    = results[n].get("_gt_n", min(n, GT_MAX_N))
        gt_label = "Ground truth" if spec.mask_is_gt else "Pseudo-label (teacher)"

        ax = axes[r]
        ax[0].imshow(rgb)
        ax[0].set_title(f"{key}\nRGB", fontsize=fs_title)
        ax[1].imshow(gt, cmap="Blues", vmin=-1, vmax=1, interpolation="nearest")
        ax[1].set_title(gt_label, fontsize=fs_title)
        ax[2].imshow(teacher["pred"], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax[2].set_title(f"Teacher 300M\nflood-IoU={teacher['flood_iou']:.2f}", fontsize=fs_title)
        ax[3].imshow(gtm["pred"], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax[3].set_title(f"EViT, {gt_n} human labels\nflood-IoU={gtm['flood_iou']:.2f}",
                        fontsize=fs_title)
        lkd = results[n][key]["lkd"][idx]
        ax[4].imshow(lkd["pred"], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax[4].set_title(f"EViT-LogitKD 0.7M\nflood-IoU={lkd['flood_iou']:.2f}",
                        fontsize=fs_title)
        ax[5].imshow(unet["pred"], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax[5].set_title(f"UNet-logitKD 32.5M\nflood-IoU={unet['flood_iou']:.2f}",
                        fontsize=fs_title)
        ax[6].imshow(ofa["pred"], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax[6].set_title(f"EViT-OFA 0.7M (ours)\nflood-IoU={ofa['flood_iou']:.2f}",
                        fontsize=fs_title)

        if show_agreement:
            agree = build_agreement_map(gt, unet["pred"], ofa["pred"], denoise_kernel)
            delta = ofa["flood_iou"] - unet["flood_iou"]
            ax[7].imshow(agree, interpolation="nearest")
            ax[7].set_title(f"OFA − UNet-KD  Δflood-IoU={delta:+.2f}", fontsize=fs_title)

        for a in ax:
            a.axis("off")

    if show_agreement:
        fig.legend(handles=_agreement_legend("UNet-logitKD", "OFA"),
                   loc="lower center", ncol=5, fontsize=fs_leg, bbox_to_anchor=(0.5, -0.02))
    if not paper_clean:
        fig.suptitle(f"EViT-OFA vs UNet-logit-KD — N={n_label(n)}", fontsize=fs_supt, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_models] saved {out_path}")


# Workstation branch

def run_workstation(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    specs = build_datasets(args)

    # --picks_from: draw the scenes of an earlier manifest without scanning and ranking.
    # Each stored `id` is checked against the current dataset.
    reuse = None
    if args.picks_from:
        reuse = json.loads(Path(args.picks_from).read_text()).get("picks", {})
        print(f"[plot_models] --picks_from {args.picks_from}: reusing stored scenes, "
              f"skipping scan and ranking")

    candidates = {}
    for spec in specs:
        bright_thresh = _CLOUD_BRIGHT_THRESH[spec.key]
        if reuse is not None:
            if spec.key not in reuse:
                raise SystemExit(f"--picks_from has no picks for {spec.key}; drop it from "
                                 f"--datasets or use a manifest that covers it")
            idxs = {}
            for p in reuse[spec.key]:
                # Same function that wrote the manifest; not every dataset has "filename".
                got = get_sample_id(spec, p["idx"])
                if got != p["id"]:
                    raise SystemExit(
                        f"--picks_from is stale for {spec.key}: index {p['idx']} was "
                        f"{p['id']!r} when the manifest was written but is {got!r} now. "
                        f"The dataset or its ordering changed; re-select instead of reusing.")
                idxs[p["idx"]] = 0.0        # cloud fraction, unused on this path
            candidates[spec.key] = idxs
            print(f"[plot_models] {spec.key}: {len(idxs)} scene(s) reused from manifest")
            continue
        idxs = scan_candidates(
            spec.dataset, spec.key, args.max_scan, args.candidate_pool,
            args.min_valid_frac, (args.flood_frac_min, args.flood_frac_max),
            bright_thresh=bright_thresh, max_cloud_frac=args.max_cloud_frac,
        )
        candidates[spec.key] = idxs
        print(f"[plot_models] {spec.key}: {len(idxs)} candidate scenes "
              f"(scanned {min(len(spec.dataset), args.max_scan)}/{len(spec.dataset)}, "
              f"cloud bright_thresh={bright_thresh}, max_cloud_frac={args.max_cloud_frac})")

    print("[plot_models] running teacher …")
    teacher = build_teacher_model(device)
    teacher_preds = {
        spec.key: evaluate_model_on_indices(teacher, spec.dataset, list(candidates[spec.key]),
                                            spec.adapter, device)
        for spec in specs
    }
    del teacher
    torch.cuda.empty_cache()

    results: dict[int, dict] = {}
    for n in args.n_values:
        ofa_seed,  ofa_ckpt  = pick_representative_seed(OFA_PREFIX,  n)
        unet_seed, unet_ckpt = pick_representative_seed(UNET_PREFIX, n)
        # Human labels stop at N=252; above that the GT arm stays at N=252 (see column title).
        gt_n = min(n, GT_MAX_N)
        gt_seed, gt_ckpt = pick_representative_seed(GT_PREFIX, gt_n, root=GT_ROOT)
        lkd_seed, lkd_ckpt = pick_representative_seed(LKD_PREFIX, n)
        print(f"[plot_models] N={n_label(n)}: OFA seed={ofa_seed}  LKD seed={lkd_seed}  "
              f"UNet-KD seed={unet_seed}  GT seed={gt_seed} (N={gt_n})")

        ofa_model  = build_student_model(OFA_ARCH,  ofa_ckpt,  device)
        lkd_model  = build_student_model(LKD_ARCH,  lkd_ckpt,  device)
        unet_model = build_student_model(UNET_ARCH, unet_ckpt, device)
        gt_model   = build_student_model(GT_ARCH,   gt_ckpt,   device)
        results[n] = {"_gt_n": gt_n}
        for spec in specs:
            idxs = list(candidates[spec.key])
            results[n][spec.key] = {
                "ofa":  evaluate_model_on_indices(ofa_model,  spec.dataset, idxs, spec.adapter, device),
                "lkd":  evaluate_model_on_indices(lkd_model,  spec.dataset, idxs, spec.adapter, device),
                "unet": evaluate_model_on_indices(unet_model, spec.dataset, idxs, spec.adapter, device),
                "gt":   evaluate_model_on_indices(gt_model,   spec.dataset, idxs, spec.adapter, device),
            }
        del ofa_model, lkd_model, unet_model, gt_model
        torch.cuda.empty_cache()

    # Variant v uses the v-th best scene per dataset; N=50 and N=2500 show the same scenes.
    ranked = {}
    for spec in specs:
        if reuse is not None:
            # Manifest ORDER is the variant order, so qual_*_1.png stays qual_*_1.png.
            ranked[spec.key] = [p["idx"] for p in reuse[spec.key]]
            print(f"[plot_models] {spec.key}: {len(ranked[spec.key])} scene(s) in manifest order")
            continue
        ranked[spec.key] = select_hero_samples(
            spec.key, candidates, results, args.rank_n, args.variants,
            teacher_preds=teacher_preds, min_gain=args.min_gain,
            cloud_weight=args.cloud_weight, min_pred_flood_frac=args.min_pred_flood_frac,
        )
        print(f"[plot_models] {spec.key}: ranked {len(ranked[spec.key])} scene(s)")

    n_var = max((len(v) for v in ranked.values()), default=0)
    if not n_var:
        raise SystemExit("no scene passed the selection criteria on any dataset")
    n_var = min(n_var, args.variants)
    for v in range(n_var):
        picks = {k: [idxs[v]] for k, idxs in ranked.items() if len(idxs) > v}
        for n in args.n_values:
            save_figure(n, specs, picks, results, teacher_preds,
                        out_dir / f"qual_{n_filename(n)}_{v + 1}.png",
                        show_agreement=args.show_agreement,
                        denoise_kernel=args.agreement_denoise_kernel,
                        font_scale=args.font_scale, dpi=args.dpi,
                        paper_clean=args.paper_clean,
                        rgb_p_low=args.rgb_p_low, rgb_p_high=args.rgb_p_high,
                        rgb_gamma=args.rgb_gamma, rgb_sat=args.rgb_sat,
                        rgb_fixed=args.rgb_fixed)
    picks = {k: idxs[:n_var] for k, idxs in ranked.items()}

    # Indices only hold for one loader configuration, so it is stored with the picks.
    # jetson/qualitative_jetson.py refuses indices if its configuration differs.
    manifest = {
        "generated_by":          "figures/plot_models.py",
        "n_values":              args.n_values,
        "samples_per_dataset":   args.samples_per_dataset,
        "loader": {
            "s11_split":         "test",
            "eval_img_size":     args.eval_img_size,
            "sturm_normalize":   "original",
            "wf":                "n/a (not evaluated by this branch)",
            "gdacs_index":       args.gdacs_index,
        },
        "picks": {
            spec.key: [{"idx": idx, "id": get_sample_id(spec, idx)} for idx in picks[spec.key]]
            for spec in specs
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[plot_models] manifest → {manifest_path}")
    print(f"[plot_models] device predictions for the same scenes: "
          f"jetson/qualitative_jetson.py --manifest {manifest_path}")


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir",  default=str(ROOT / "runs" / "qualitative_figures"))
    p.add_argument("--manifest", default=None,
                   help="Default: <out_dir>/qual_samples_manifest.json")

    p.add_argument("--picks_from", default=None,
                   help="Earlier qual_samples_manifest.json: draw its scenes in order, without "
                        "scanning and ranking. Stored ids are checked against the dataset.")
    p.add_argument("--samples_per_dataset", type=int, default=1,
                   help="Rows per dataset in each figure")
    p.add_argument("--candidate_pool",      type=int, default=40)
    p.add_argument("--max_scan",            type=int, default=800,
                   help="Maximum number of scenes scanned per dataset")
    p.add_argument("--min_valid_frac", type=float, default=0.5)
    p.add_argument("--flood_frac_min", type=float, default=0.02)
    p.add_argument("--flood_frac_max", type=float, default=0.6)
    p.add_argument("--max_cloud_frac", type=float, default=0.15,
                   help="Reject scenes with more bright (cloud) pixels than this fraction")
    p.add_argument("--cloud_weight", type=float, default=1.5,
                   help="Score penalty per unit cloud fraction; 0 disables it")
    p.add_argument("--min_pred_flood_frac", type=float, default=0.01,
                   help="Drop scenes where teacher, OFA or UNet predict less flood than this "
                        "fraction at any N")

    p.add_argument("--rank_n", default=None,
                   help="N values at which the margins must hold (default: all --n_values), "
                        "e.g. '2500'")
    p.add_argument("--variants", type=int, default=4,
                   help="Number of alternative figures per N; variant v uses the v-th best scene")
    p.add_argument("--min_gain", type=float, default=0.0,
                   help="Required flood-IoU margin of pseudo EViT-OFA over UNet and GT EViT "
                        "at every N")
    p.add_argument("--fade_weight", type=float, default=0.5,
                   help="Weight for scenes where the OFA-UNet gap shrinks at large N; 0 disables it")

    p.add_argument("--show_agreement", dest="show_agreement", action="store_true", default=True,
                   help="Draw the agreement column (default)")
    p.add_argument("--no_agreement", dest="show_agreement", action="store_false",
                   help="Leave out the agreement column")
    p.add_argument("--agreement_denoise_kernel", type=int, default=3,
                   help="Median filter size for the agreement map; 0 or 1 disables it")

    # Rendering
    p.add_argument("--font_scale", type=float, default=1.9,
                   help="Scale factor for titles and legends")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output resolution of the PNG files")
    p.add_argument("--paper_clean", action="store_true",
                   help="No suptitle; with --no_agreement this gives the paper layout")
    p.add_argument("--rgb_p_low", type=float, default=2.0,
                   help="Lower percentile of the RGB stretch (only with --rgb_fixed 0)")
    p.add_argument("--rgb_p_high", type=float, default=98.0,
                   help="Upper percentile of the RGB stretch (only with --rgb_fixed 0)")
    p.add_argument("--rgb_gamma", type=float, default=1.0,
                   help="Display gamma after the stretch; 1.0 is off")
    p.add_argument("--rgb_fixed", type=float, default=TRUE_COLOUR_CEILING,
                   help="True-colour ceiling K: min(rho/K, 1) on B4/B3/B2 (0.40 as in Sentinel "
                        "Hub). Must match jetson/qualitative_jetson.py. 0 uses a percentile stretch.")
    p.add_argument("--rgb_sat", type=float, default=1.0,
                   help="Colour saturation; values below 1 desaturate")

    p.add_argument("--n_values", default=None,
                   help="Comma-separated N values, e.g. '50,2500'")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated subset of the datasets (default: all)")

    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eval_img_size", type=int, default=512)

    p.add_argument("--s11_root",   default=str(dataset_root("S11_ROOT")))
    p.add_argument("--sturm_root", default=str(dataset_root("STURM_ROOT")))
    p.add_argument("--s1s2w_root", default=str(dataset_root("S1S2W_ROOT")),
                   help="Not used; kept for older command lines")
    p.add_argument("--wf_root", default=str(dataset_root("WF_ROOT")))
    p.add_argument("--gdacs_dir",   default=GDACS_DIR)
    # The N=2,500 training index of the plotted students.
    p.add_argument("--gdacs_index", default="diversity_N2500_index.json")


    args = p.parse_args()
    # --rgb_fixed 0 means off; to_rgb expects None for that.
    args.rgb_fixed = args.rgb_fixed if args.rgb_fixed and args.rgb_fixed > 0 else None
    args.n_values = ([int(x) for x in args.n_values.split(",")]
                     if args.n_values else N_VALUES)
    args.rank_n = ([int(x) for x in args.rank_n.split(",")] if args.rank_n
                   else list(args.n_values))
    if not set(args.rank_n) <= set(args.n_values):
        p.error(f"--rank_n {args.rank_n} must be a subset of --n_values {args.n_values}")
    args.datasets = args.datasets.split(",") if args.datasets else None
    if args.manifest is None:
        args.manifest = str(Path(args.out_dir) / "qual_samples_manifest.json")
    return args


def main() -> None:
    run_workstation(args=parse_args())


if __name__ == "__main__":
    main()
