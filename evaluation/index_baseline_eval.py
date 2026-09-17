# !/usr/bin/env python3
"""Evaluate MNDWI and NDWI on the Sen1Floods11 test split.

Uses the same SpectralIndexPredictor as external evaluation, 512x512 inputs,
fixed S11 normalization and ignore label -1. Reports pooled and per-chip
scores, including separate background and water IoU. The teacher was
fine-tuned on the training split, not this test split.
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np, torch

from data.sen1floods11 import get_dataloaders
from utils.paths import dataset_root
from evaluation.perscene_eval import FAMILIES, load

ROWS = [
    ("MNDWI > 0  (2-band rule, 0 params)", "index_mndwi_zero", ["-"]),
    ("NDWI > 0   (2-band rule, 0 params)", "index_ndwi_zero",  ["-"]),
    ("Prithvi 300M teacher",               "teacher",          ["-"]),
    ("pseudo-OFA  N=2500  (deployed)",     "evit_ofa_ps_N2500",  ["42","64","91","107"]),
    ("pseudo-OFA  N=252",                  "evit_ofa_ps_N252",   ["42","64","91","107"]),
    ("S11-OFA uncropped  N=252",           "ncgt_evit_ofa_N252",    ["42","64","91","107"]),
    ("S11-OFA cropped    N=252",           "b1crop_evit_ofa_N252",  ["42","64","91","107"]),
]


@torch.no_grad()
def run(model, dl, dev):
    """Returns (pooled 2x3 confusion, [per-chip mIoU])."""
    tot = np.zeros((2, 3)); per = []
    for b in dl:
        out = model(b["image"].to(dev))
        out = out["out"] if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)
        pred = out.argmax(1).cpu(); m = b["mask"]; v = m != -1
        for i in range(pred.shape[0]):
            cell = np.zeros((2, 3))
            for c in (0, 1):
                pc = (pred[i] == c) & v[i]; gc = (m[i] == c) & v[i]
                cell[c] = [(pc & gc).sum().item(), (pc & ~gc & v[i]).sum().item(), (~pc & gc).sum().item()]
            tot += cell
            iou = cell[:, 0] / np.maximum(cell.sum(1), 1)
            per.append(float(iou.mean()))
    return tot, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_root",
                    default=str(dataset_root("S11_ROOT")),
                    help="Directory containing v1.1/. Default: $DATA_ROOT/sen1floods11.")
    ap.add_argument("--out", default="runs/perscene_C1/s11_test_index_baseline.csv")
    ap.add_argument("--families", default=None,
                    help="Comma-separated families instead of the ROWS table (e.g. threshold sweep)")
    a = ap.parse_args()
    rows_spec = ROWS
    if a.families:
        rows_spec = [(f, f, ["-"] if (f == "teacher" or f.startswith("index_"))
                     else ["42", "64", "91", "107"])
                     for f in a.families.split(",")]
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    # eval_img_size defaults to 512 = native; batch 4 keeps the 300M teacher inside memory.
    _, _, test_dl = get_dataloaders(a.data_root, batch_size=4, num_workers=6, drop_last=False)
    n = len(test_dl.dataset)
    print(f"Sen1Floods11 hand-labelled TEST split: {n} chips at native 512x512\n")
    print(f"  {'model':38s} {'mIoU':>7s} {'bg IoU':>8s} {'water IoU':>10s} {'macro':>8s}")
    print("  " + "-" * 76)

    rows = [["model", "family", "n_seeds", "micro_mIoU", "bg_IoU", "water_IoU", "macro_mIoU"]]
    for label, fam, seeds in rows_spec:
        arch, tmpl = FAMILIES[fam]
        mi, bg, wa, ma = [], [], [], []
        for sd in seeds:
            ck = tmpl if sd == "-" else tmpl.format(seed=sd)
            if sd != "-" and not Path(ck).exists():
                continue
            tot, per = run(load(arch, ck, dev), test_dl, dev)
            iou = tot[:, 0] / np.maximum(tot.sum(1), 1)
            mi.append(float(iou.mean())); bg.append(float(iou[0])); wa.append(float(iou[1]))
            ma.append(float(np.mean(per)))
        if not mi:
            print(f"  {label:38s}   (no checkpoints)"); continue
        sd_txt = f" ±{np.std(wa):.3f}" if len(mi) > 1 else ""
        print(f"  {label:38s} {np.mean(mi):7.4f} {np.mean(bg):8.4f} {np.mean(wa):10.4f}{sd_txt}"
              f" {np.mean(ma):8.4f}")
        rows.append([label, fam, len(mi), f"{np.mean(mi):.5f}", f"{np.mean(bg):.5f}",
                     f"{np.mean(wa):.5f}", f"{np.mean(ma):.5f}"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
