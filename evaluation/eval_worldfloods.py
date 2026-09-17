# !/usr/bin/env python3
"""Evaluate checkpoints on WorldFloods; writes runs/<run>/eval_results_WF_<normalize>.json.

Run selection works as in evaluation/eval_ablations.py:
    python evaluation/eval_worldfloods.py --runs_base runs/nsweep --runs_reg pseudo_ofa_aug_N --n_filter 2500 --normalize s11 --gpu 1
    python evaluation/eval_worldfloods.py --runs_base runs/ablations_sen1 --run sen_ce_unet_N252_s42 --arch unet_resnet50 --normalize s11 --gpu 1

normalize="s11" (default) uses the Sen1Floods11 training statistics; "dataset" uses
WorldFloods statistics. Existing result files are skipped unless --force.
"""
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import torch
import eval_ablations as E   # reuse load_model / detect_arch / DEVICE
from data.ood_datasets import get_worldfloods_dataloader
from utils.paths import dataset_root


def eval_one(model, dl, device):
    # 2-class confusion over valid pixels
    tp = np.zeros(2); fp = np.zeros(2); fn = np.zeros(2)
    pred_flood = valid = gt_flood = 0
    with torch.no_grad():
        for b in dl:
            img = b["image"].to(device); m = b["mask"]
            out = model(img)
            out = out["out"] if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)
            pred = out.argmax(1).cpu()
            v = m != -1
            for c in (0, 1):
                pc = (pred == c) & v; gc = (m == c) & v
                tp[c] += (pc & gc).sum().item()
                fp[c] += (pc & ~gc & v).sum().item()
                fn[c] += (~pc & gc).sum().item()
            pred_flood += ((pred == 1) & v).sum().item()
            gt_flood += ((m == 1) & v).sum().item()
            valid += v.sum().item()
    iou = tp / np.maximum(tp + fp + fn, 1)
    return {
        "wf_mIoU": float(iou.mean()),
        "wf_bg_IoU": float(iou[0]),
        "wf_flood_IoU": float(iou[1]),
        "wf_pred_flood_frac": pred_flood / max(valid, 1),
        "wf_gt_flood_frac": gt_flood / max(valid, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_base", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--runs_reg", default=None)
    ap.add_argument("--anchor", action="store_true",
                    help="Match --runs_reg as a NAME PREFIX (startswith) instead of a substring, "
                         "so e.g. 'div_ofa_hs_N' does not also catch 'clean_div_ofa_hs_N'.")
    ap.add_argument("--n_filter", type=int, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--normalize", default="s11", choices=["s11", "dataset"])
    ap.add_argument("--wf_root", default=str(dataset_root("WF_ROOT")))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="WorldFloodsv2 split (ignored for the legacy v1 subset).")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--tile_size", type=int, default=256,
                    help="v2 native tile size (256 = paper sub-image). Scenes are tiled "
                         "at full 10 m resolution rather than whole-scene resized.")
    ap.add_argument("--score_clouds", action="store_true",
                    help="Score cloud pixels too (default: exclude — correct for a "
                         "2-class optical water model).")
    ap.add_argument("--include_permanent", action="store_true",
                    help="Count JRC permanent water as water (default: flood-only). Use a "
                         "different --out_tag for each protocol.")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_tag", default="",
                    help="Tag in the output name eval_results_WF<tag>_<norm>.json, e.g. 'v2'")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="E5 radiometric robustness on WF: multiply reflectance by this before "
                         "the s11 normalization (1.0 = off). Same physical axis as the STURM E5.")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="E5 additive reflectance offset before normalization (0.0 = off).")
    ap.add_argument("--haze", type=float, default=0.0,
                    help="Haze robustness: additive wavelength-weighted path radiance "
                         "(blue-band strength, reflectance units; Makarau 2014). 0.0 = off.")
    args = ap.parse_args()

    E.GPU = args.gpu
    E.DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    device = E.DEVICE

    base = Path(args.runs_base)
    if args.run:
        runs = [base / args.run]
    else:
        runs = sorted(p for p in base.iterdir() if p.is_dir())
        if args.runs_reg:
            runs = [r for r in runs if (r.name.startswith(args.runs_reg) if args.anchor
                                        else args.runs_reg in r.name)]
        if args.n_filter is not None:
            runs = [r for r in runs if f"_N{args.n_filter}_" in r.name]

    dl = get_worldfloods_dataloader(root=args.wf_root, split=args.split, img_size=args.img_size,
                                    tile_size=args.tile_size, normalize=args.normalize,
                                    exclude_clouds=not args.score_clouds,
                                    batch_size=8, num_workers=6,
                                    sim_gain=args.gain, sim_offset=args.offset, sim_haze=args.haze,
                                    ignore_permanent_water=not args.include_permanent)
    print(f"[WF] protocol: {'legacy flood+permanent' if args.include_permanent else 'FLOOD-ONLY (JRC permanent -> ignore)'}")
    if args.gain != 1.0 or args.offset != 0.0:
        print(f"[WF] E5 perturbation: reflectance -> {args.gain}*x + {args.offset}")
    if args.haze != 0.0:
        print(f"[WF] HAZE perturbation: reflectance += {args.haze}*w_band (blue→SWIR)")
    unit = "tiles" if getattr(dl.dataset, "version", "v1") == "v2" else "scenes"
    print(f"[WF] split={args.split} | {len(dl.dataset)} {unit} | normalize={args.normalize} "
          f"| {len(runs)} run(s) | dev={device}")
    out_name = f"eval_results_WF{args.out_tag}_{args.normalize}.json"
    if getattr(dl.dataset, "version", "v1") != "v2":
        print(f"[WARN] loader resolved WorldFloods {getattr(dl.dataset,'version','?')} (not v2) "
              f"at {args.wf_root} — check the dataset path.")
    for r in runs:
        ckpt = r / "best.pt"
        if not ckpt.exists():
            print(f"[miss] {r.name}"); continue
        out = r / out_name
        if out.exists() and not args.force:
            print(f"[skip] {r.name}"); continue
        arch = E.detect_arch(r.name, arch_override=args.arch)
        model = E.load_model(ckpt, arch)
        res = eval_one(model, dl, device)
        res.update({"arch": arch, "normalize": args.normalize, "n_scenes": len(dl.dataset),
                    "wf_version": getattr(dl.dataset, "version", "v1"),
                    "tile_size": args.tile_size, "img_size": args.img_size,
                    "protocol": ("flood+permanent" if args.include_permanent
                                 else "flood_only_jrc")})
        json.dump(res, open(out, "w"), indent=2)
        print(f"[ok]   {r.name:38s} mIoU={res['wf_mIoU']:.4f} flood={res['wf_flood_IoU']:.4f} "
              f"predflood={res['wf_pred_flood_frac']:.3f}")


if __name__ == "__main__":
    main()
