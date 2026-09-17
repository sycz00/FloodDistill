"""Farthest-point sampling of the pseudo-label pool (paper Sec. III-B).

Filter by flood fraction and within-tile confidence standard deviation. Each
candidate has 15 features: flood_frac, mean_conf, conf_std, six band means and
six band standard deviations. Scale each dimension to [0,1] after clipping at
the 1st and 99th percentiles, then apply greedy Euclidean FPS.

Writes diversity_N<top_k>_index.json. Build an index for each training budget.
PseudoLabeledDataset samples uniformly from that index, not by rank. The paper
does not establish a ranking advantage over other filtered selectors at N=2500.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Feature extraction

def _extract_features(pt_path: Path) -> dict | None:
    """Load one pseudo_*.pt and return scalar/vector features.

        Returns None if the file lacks required keys or is outside the ff range.
    """
    try:
        item = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"  [diversity] warning: could not load {pt_path.name}: {e}")
        return None

    mask = item["mask"].float()         # (H, W)
    img  = item["image"].float()        # (6, H, W)

    flood_frac = mask.mean().item()

    if "confidence" in item:
        conf      = item["confidence"].float()
        mean_conf = conf.mean().item()
        conf_std  = conf.std().item()
    else:
        # confidence not stored - fall back; scenes will pass conf_std filter
        # only if we use min_conf_std=0.0
        mean_conf = 1.0
        conf_std  = 0.0

    # Per-band statistics in normalized space (z-score, stored as-is)
    band_mean = img.mean(dim=(1, 2)).numpy()   # (6,)
    band_std  = img.std(dim=(1, 2)).numpy()    # (6,)

    return {
        "name":       pt_path.name,
        "flood_frac": flood_frac,
        "mean_conf":  mean_conf,
        "conf_std":   conf_std,
        "band_mean":  band_mean,
        "band_std":   band_std,
    }


def _build_feature_matrix(records: list[dict]) -> np.ndarray:
    """Stack per-scene scalar + vector features into (N, 15) array."""
    rows = []
    for r in records:
        row = np.concatenate([
            [r["flood_frac"], r["mean_conf"], r["conf_std"]],   # 3 scalars
            r["band_mean"],                                       # 6 band means
            r["band_std"],                                        # 6 band stds
        ])
        rows.append(row)
    return np.stack(rows, axis=0).astype(np.float32)             # (N, 15)


def _normalise(X: np.ndarray) -> np.ndarray:
    """Normalise each feature dimension to [0, 1] using 1st/99th percentile clipping."""
    lo  = np.percentile(X, 1,  axis=0)
    hi  = np.percentile(X, 99, axis=0)
    denom = np.where(hi - lo > 1e-8, hi - lo, 1.0)
    X_norm = (X - lo) / denom
    return np.clip(X_norm, 0.0, 1.0)


# Farthest-Point Sampling

def farthest_point_sampling(X: np.ndarray, k: int, seed_idx: int = 0) -> np.ndarray:
    """Greedy farthest-point sampling, returns (k,) int64 indices into X.

        Each step takes the point farthest away from its nearest selected point.
        O(N*k) time, O(N) memory. X is the (N, D) float32 feature matrix and should
        be normalised. seed_idx is the first pick.
    """
    N = len(X)
    if k >= N:
        return np.arange(N)

    selected   = np.empty(k, dtype=np.int64)
    min_dists  = np.full(N, np.inf, dtype=np.float32)

    # Seed from the first record (already sorted by conf_std desc before calling)
    current = seed_idx
    selected[0] = current

    for i in tqdm(range(1, k), desc="FPS", unit="pt", leave=False):
        # Distance from all points to the last selected point
        diff       = X - X[current]              # (N, D)
        d_current  = (diff * diff).sum(axis=1)   # (N,) squared Euclidean
        np.minimum(min_dists, d_current, out=min_dists)
        min_dists[current] = -1.0                # exclude already-selected

        current      = int(np.argmax(min_dists))
        selected[i]  = current

    return selected


# Main selection function

def select_diverse(
    pseudo_dir: str,
    top_k:        int   = 2500,
    min_ff:       float = 0.05,
    max_ff:       float = 0.85,
    min_conf_std: float = 0.05,
    out_name:     str | None = None,
) -> None:
    """Filter, scale 15-D features and write a top_k FPS index. Keep gates identical for matched-pool controls."""
    base  = Path(pseudo_dir)
    files = sorted(base.glob("pseudo_*.pt"))
    if not files:
        raise FileNotFoundError(f"No pseudo_*.pt files found in {pseudo_dir!r}")

    print(f"[diversity] Scanning {len(files)} files in {pseudo_dir!r} ...")
    records: list[dict] = []
    n_skipped = 0
    for pt in tqdm(files, desc="extracting", unit="patch"):
        feat = _extract_features(pt)
        if feat is None:
            n_skipped += 1
            continue
        records.append(feat)

    print(f"  {len(records)} loadable  ({n_skipped} skipped/corrupt)")

    # pre-filter, SPL-D compatible
    candidates = [
        r for r in records
        if min_ff <= r["flood_frac"] <= max_ff
        and r["conf_std"] >= min_conf_std
    ]
    n_ff_only  = sum(1 for r in records if min_ff <= r["flood_frac"] <= max_ff)
    print(f"\n  ff ∈ [{min_ff}, {max_ff}]:                   {n_ff_only}")
    print(f"  ff-range AND conf_std ≥ {min_conf_std}:  {len(candidates)}")

    if len(candidates) < top_k:
        print(f"\n  [diversity] WARNING: only {len(candidates)} candidates available "
              f"for top_k={top_k}.  Reducing top_k to {len(candidates)}.")
        top_k = len(candidates)

    if len(candidates) == 0:
        raise ValueError(
            f"No candidates satisfy the filters. "
            f"Try reducing min_conf_std (current: {min_conf_std}) or widening ff range."
        )

    # highest conf_std first, so FPS starts from the most boundary-rich scene
    candidates.sort(key=lambda r: r["conf_std"], reverse=True)

    X      = _build_feature_matrix(candidates)         # (N_cand, 15)
    X_norm = _normalise(X)
    print(f"\n  Feature matrix: {X_norm.shape}  (15 dims: ff, mean_conf, conf_std, band×6 mean+std)")

    print(f"  Running FPS: selecting {top_k} from {len(candidates)} candidates ...")
    idx = farthest_point_sampling(X_norm, top_k, seed_idx=0)

    selected = [candidates[i] for i in idx]

    # report + save
    sel_ffs    = np.array([r["flood_frac"] for r in selected])
    sel_mcs    = np.array([r["mean_conf"]  for r in selected])
    sel_css    = np.array([r["conf_std"]   for r in selected])

    print(f"\n  Selected {len(selected)} scenes:")
    print(f"    flood_frac  — mean={sel_ffs.mean():.3f}  std={sel_ffs.std():.3f}"
          f"  p10={np.percentile(sel_ffs,10):.3f}  p90={np.percentile(sel_ffs,90):.3f}")
    print(f"    mean_conf   — mean={sel_mcs.mean():.3f}  std={sel_mcs.std():.3f}")
    print(f"    conf_std    — mean={sel_css.mean():.3f}  std={sel_css.std():.3f}"
          f"  p10={np.percentile(sel_css,10):.3f}  p90={np.percentile(sel_css,90):.3f}")

    # score per scene is required by the index format; reporting only, not a ranking
    scores = [
        round(4.0 * r["flood_frac"] * (1.0 - r["flood_frac"]) * r["conf_std"], 6)
        for r in selected
    ]

    index = {
        "n_total":      len(records),
        "n_in_range":   n_ff_only,
        "n_candidates": len(candidates),
        "n_curated":    len(selected),
        "min_ff":       min_ff,
        "max_ff":       max_ff,
        "min_conf_std": min_conf_std,
        "score_fn":     (
            "farthest_point_sampling on [ff,mean_conf,conf_std,band_mean×6,band_std×6], "
            f"pre-filtered ff∈[{min_ff},{max_ff}] AND conf_std≥{min_conf_std}"
        ),
        "files":        [r["name"] for r in selected],
        "scores":       scores,
    }

    out = base / (out_name or f"diversity_N{top_k}_index.json")
    with open(out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved → {out}")
    print(f"Use with: --curated_pseudo --curated_index_file {out.name} "
          f"--n_samples {top_k}")


# CLI

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pseudo_dir",    default="./data/pseudo_gdacs",
                   help="Directory containing pseudo_*.pt files.")
    p.add_argument("--top_k",         type=int,   default=2500,
                   help="Number of diverse scenes to select.")
    p.add_argument("--out_name",      default=None,
                   help="Output index filename. Default: diversity_N<top_k>_index.json. "
                        "PseudoLabeledDataset draws n_samples uniformly at random from the "
                        "index and never re-reads the ranking, so every training budget needs "
                        "its own index built at --top_k N; naming it after N is what keeps the "
                        "two from drifting apart. Pass a distinct name when relaxing the "
                        "pre-filter.")
    p.add_argument("--min_ff",        type=float, default=0.05,
                   help="Minimum flood fraction (proxy contrast floor).")
    p.add_argument("--max_ff",        type=float, default=0.85,
                   help="Maximum flood fraction.")
    p.add_argument("--min_conf_std",  type=float, default=0.05,
                   help="Minimum within-scene confidence std (boundary-pixel floor). "
                        "Scenes below this have no meaningful uncertainty variation for SPL-D.")
    args = p.parse_args()

    select_diverse(
        pseudo_dir    = args.pseudo_dir,
        top_k         = args.top_k,
        min_ff        = args.min_ff,
        max_ff        = args.max_ff,
        min_conf_std  = args.min_conf_std,
        out_name      = args.out_name or f"diversity_N{args.top_k}_index.json",
    )


if __name__ == "__main__":
    main()
