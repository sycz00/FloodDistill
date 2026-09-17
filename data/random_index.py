# !/usr/bin/env python3
"""Uniform random subsets of the filtered and unfiltered pseudo pools (controls).

Uses random.Random(seed).sample over the sorted pseudo_*.pt filenames. The default
seed is 12345, not the training seed. To reproduce an existing index keep RNG,
sample call and file order the same. Build exactly N entries. Prefer the
full-precision .pt scan, CSV rounding can push a tile over a filter boundary.

filtered-random vs unfiltered-random tests the pre-filter. A ranking rule vs
filtered-random tests selection within the same candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm


def _scan_pt(base: Path) -> list[dict]:
    """Full-precision scan of the pool (standard path)."""
    files = sorted(base.glob("pseudo_*.pt"))
    if not files:
        raise FileNotFoundError(f"No pseudo_*.pt files found in {str(base)!r}")
    records = []
    for pt in tqdm(files, desc="scanning", unit="patch"):
        item = torch.load(pt, map_location="cpu", weights_only=True)
        ff = item["mask"].float().mean().item()
        if "confidence" in item:
            conf = item["confidence"].float()
            mean_conf, conf_std = conf.mean().item(), conf.std().item()
        else:
            mean_conf, conf_std = 1.0, 0.0
        records.append({"file": pt.name, "flood_frac": ff,
                        "mean_conf": mean_conf, "conf_std": conf_std})
    return records


def _read_features_csv(path: Path) -> list[dict]:
    """Faster: read the pool_features.csv from data/pool_features.py instead.

        Same rows in the same order (both use sorted(glob)). The CSV rounds flood_frac and
        conf_std to 6 decimals though, so a tile right on a filter boundary could land on
        the other side. For this pool it gave the same 5,987 candidates and the same draw.
        If unsure, use the .pt scan.
    """
    rows = list(csv.DictReader(open(path)))
    return [{"file": r["file"], "flood_frac": float(r["flood_frac"]),
             "mean_conf": float(r["mean_conf"]), "conf_std": float(r["conf_std"])}
            for r in rows]


def build(pseudo_dir: str, top_k: int = 2500, seed: int = 12345,
          min_ff: float = 0.05, max_ff: float = 0.85, min_conf_std: float = 0.05,
          out_name: str = "random_filtered_index.json",
          features_csv: str | None = None) -> Path:
    base = Path(pseudo_dir)
    records = (_read_features_csv(base / features_csv) if features_csv
               else _scan_pt(base))

    # same pre-filter as diversity_index / confidence_filtered_index / glister / r2core,
    # so differences between arms come from the ranking rule only
    cand = [r for r in records
            if min_ff <= r["flood_frac"] <= max_ff and r["conf_std"] >= min_conf_std]
    print(f"\n{len(records)} patches scanned")
    print(f"  ff in [{min_ff}, {max_ff}] AND conf_std >= {min_conf_std}: {len(cand)}")
    if len(cand) < top_k:
        raise SystemExit(f"only {len(cand)} candidates for top_k={top_k}")

    pick = random.Random(seed).sample(cand, top_k)
    filtered = not (min_ff <= 0.0 and max_ff >= 1.0 and min_conf_std <= 0.0)
    where = (f"the SAME {len(cand):,} pre-filtered candidates every other selector ranks "
             f"— the control that says whether ranking beats chance"
             if filtered else
             f"ALL {len(cand):,} tiles (no pre-filter) — the control for the "
             f"full-catalogue arms")

    out = base / out_name
    json.dump({
        "n_total": len(records),
        "n_candidates": len(cand),
        "n_curated": len(pick),
        "min_ff": min_ff, "max_ff": max_ff, "min_conf_std": min_conf_std,
        "score_fn": f"RANDOM baseline (seed {seed}): uniform random draw from {where}",
        "files": [r["file"] for r in pick],
    }, open(out, "w"), indent=2)
    print(f"\nSaved {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    ap.add_argument("--top_k", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=12345,
                    help="selection seed; 12345 reproduces the published control arms")
    ap.add_argument("--out_name", default="random_filtered_index.json")
    ap.add_argument("--min_ff", type=float, default=0.05)
    ap.add_argument("--max_ff", type=float, default=0.85)
    ap.add_argument("--min_conf_std", type=float, default=0.05)
    ap.add_argument("--features_csv", default=None,
                    help="reuse pool_features.csv instead of re-reading every .pt "
                         "(see _read_features_csv for the rounding caveat)")
    a = ap.parse_args()
    build(a.pseudo_dir, a.top_k, a.seed, a.min_ff, a.max_ff, a.min_conf_std,
          a.out_name, a.features_csv)


if __name__ == "__main__":
    main()
