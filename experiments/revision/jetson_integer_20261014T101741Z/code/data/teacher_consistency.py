"""Measure frozen-teacher label consistency under radiometric perturbations.

c(x) = mean_k IoU(argmax T(x), argmax T(P_k(x))).
Use reflectance-space gain rho'=g*rho and haze rho'=rho+h*(490/lambda).
Stored images are normalized, so invert with rho=x*sigma+mu, perturb, then
normalize again. Do not re-clip after gain. Definitions match the OOD loaders.
Consistency is a stability diagnostic, not proof that a pseudo-label is correct.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sen1floods11 import NORM_MEAN, NORM_STD  # noqa: E402

# 6-band order [B2,B3,B4,B8A,B11,B12] central wavelengths (nm) -> Angstrom haze weights.
_WL = np.array([490., 560., 665., 865., 1610., 2190.], dtype=np.float32)
_HAZE_W = torch.tensor((490.0 / _WL) ** 1.0, dtype=torch.float32).view(-1, 1, 1)

# Inside the evaluated sweeps (gain 0.80-1.30, haze 0.00-0.20), not at the extremes:
# c(x) should measure everyday stability, not survival of the worst corruption.
DEFAULT_PERTURBS = [("gain", 0.85), ("gain", 1.15), ("haze", 0.05), ("haze", 0.10)]


class _Tiles(Dataset):
    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        d = torch.load(self.files[i], map_location="cpu", weights_only=True)
        return {"image": d["image"], "idx": i}


def _perturb(x: torch.Tensor, kind: str, amount: float,
             mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Perturb in reflectance space and return the re-normalised tensor."""
    rho = x * sd + mu
    if kind == "gain":
        rho = rho * amount
    elif kind == "haze":
        rho = rho + amount * _HAZE_W.to(rho.device)
    else:
        raise ValueError(f"unknown perturbation {kind!r}")
    return (rho - mu) / sd


@torch.no_grad()
def run(pseudo_dir: str, official_ckpt: str, gpu: int = 0, batch_size: int = 32,
        num_workers: int = 8, limit: int = 0, out_name: str = "consistency.csv") -> None:
    # The train CLI is scripts/train.py here (repo root in the research repo). Use its
    # builder so we get exactly the teacher that labelled the pool. c(x) from some other
    # checkpoint of the same architecture would be meaningless.
    from scripts.train import _build_teacher

    base = Path(pseudo_dir)
    files = sorted(base.glob("pseudo_*.pt"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no pseudo_*.pt under {pseudo_dir!r}")

    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # same defaults as the train.py parser
    args = argparse.Namespace(official_ckpt=official_ckpt, teacher_ckpt=None,
                              teacher_variant="prithvi_eo_v2_300_tl", num_classes=2,
                              bridge_teacher_ckpts=None)
    teacher = _build_teacher(args, freeze=True).to(dev).eval()

    mu = torch.tensor(NORM_MEAN, dtype=torch.float32).view(-1, 1, 1).to(dev)
    sd = torch.tensor(NORM_STD, dtype=torch.float32).view(-1, 1, 1).to(dev)

    dl = DataLoader(_Tiles(files), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    print(f"[consistency] {len(files)} tiles x {len(DEFAULT_PERTURBS)} perturbations on {dev}")

    per = {f"{k}_{v}": np.zeros(len(files), np.float32) for k, v in DEFAULT_PERTURBS}
    cmean = np.zeros(len(files), np.float32)

    done = 0
    for batch in dl:
        x = batch["image"].to(dev, dtype=torch.float32)
        idx = batch["idx"].numpy()
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            clean = teacher(x).float().argmax(1)                      # (B,H,W)

        acc = torch.zeros(x.shape[0], device=dev)
        for kind, amount in DEFAULT_PERTURBS:
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                pm = teacher(_perturb(x, kind, amount, mu, sd)).float().argmax(1)
            # IoU over the flood class, per tile; both-empty counts as agreement
            inter = ((pm == 1) & (clean == 1)).flatten(1).sum(1).float()
            union = ((pm == 1) | (clean == 1)).flatten(1).sum(1).float()
            iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter))
            per[f"{kind}_{amount}"][idx] = iou.cpu().numpy()
            acc += iou
        cmean[idx] = (acc / len(DEFAULT_PERTURBS)).cpu().numpy()

        done += x.shape[0]
        if done % (batch_size * 40) < batch_size:
            print(f"  [consistency] {done}/{len(files)}", flush=True)

    out = base / out_name
    cols = [f"{k}_{v}" for k, v in DEFAULT_PERTURBS]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "c_mean", *cols])
        for i, f in enumerate(files):
            w.writerow([f.name, f"{cmean[i]:.6f}", *[f"{per[c][i]:.6f}" for c in cols]])
    print(f"\n[consistency] wrote {out}")
    print(f"  c_mean  mean {cmean.mean():.3f}  p10 {np.percentile(cmean,10):.3f}  "
          f"p50 {np.median(cmean):.3f}  p90 {np.percentile(cmean,90):.3f}")
    for c in cols:
        print(f"  {c:12s} mean {per[c].mean():.3f}  p10 {np.percentile(per[c],10):.3f}")
    # Haze should hurt more than gain - if it does not, the perturbation is too weak
    # to separate tiles and c(x) will not discriminate.
    g = np.mean([per[c].mean() for c in cols if c.startswith("gain")])
    h = np.mean([per[c].mean() for c in cols if c.startswith("haze")])
    print(f"  gain-stability {g:.3f} vs haze-stability {h:.3f}  "
          f"({'haze hurts more, as expected' if h < g else 'WARNING: haze not harder than gain'})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    p.add_argument("--official_ckpt", default="hf",
                   help="Teacher used for the consistency pre-pass. \"hf\" is the public "
                        "Prithvi checkpoint the paper reports; a path loads a local .pt.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N tiles only")
    p.add_argument("--out_name", default="consistency.csv")
    a = p.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    run(a.pseudo_dir, a.official_ckpt, a.gpu, a.batch_size, a.num_workers, a.limit,
        a.out_name)


if __name__ == "__main__":
    main()
