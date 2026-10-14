"""Deployed/main curation: 4*f*(1-f)*mean_conf after a flood-fraction filter.

The balance term peaks at f=0.5. Writes curated_index.json. For the paper's
matched-pool confidence baseline use scripts/curate.py confmatch instead, that
one also applies the confidence-spread gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


def _score(flood_frac: float, mean_conf: float,
           min_ff: float = 0.05, max_ff: float = 0.85) -> float:
    if flood_frac < min_ff or flood_frac > max_ff:
        return 0.0
    return 4.0 * flood_frac * (1.0 - flood_frac) * mean_conf


def curate(pseudo_dir: str, top_k: int = 2500,
           min_ff: float = 0.05, max_ff: float = 0.85) -> None:
    base = Path(pseudo_dir)
    files = sorted(base.glob("pseudo_*.pt"))
    if not files:
        raise FileNotFoundError(f"No pseudo_*.pt files found in {pseudo_dir!r}")

    records: list[tuple[float, str]] = []
    n_zero = 0

    for pt in tqdm(files, desc="scoring", unit="patch"):
        item = torch.load(pt, map_location="cpu", weights_only=True)

        mask  = item["mask"].float()            # (H, W) 0/1 hard labels
        ff    = mask.mean().item()

        if "confidence" in item:
            mc = item["confidence"].float().mean().item()
        else:
            # confidence key absent - fall back to flood_frac balance only
            mc = 1.0

        s = _score(ff, mc, min_ff, max_ff)
        if s == 0.0:
            n_zero += 1
        records.append((s, pt.name))

    records.sort(key=lambda x: x[0], reverse=True)
    top = records[:top_k]

    n_nonzero = len(records) - n_zero
    print(f"\n{len(files)} patches scanned")
    print(f"  in-range (ff ∈ [{min_ff}, {max_ff}]): {n_nonzero}")
    print(f"  filtered out (score=0): {n_zero}")
    print(f"  selected top-{top_k}")
    if top:
        scores = [s for s, _ in top]
        print(f"  score range in curated set: [{min(scores):.4f}, {max(scores):.4f}]")

    index = {
        "n_total":   len(files),
        "n_in_range": n_nonzero,
        "n_curated": len(top),
        "min_ff":    min_ff,
        "max_ff":    max_ff,
        "score_fn":  "4*ff*(1-ff)*mean_conf, ff in [min_ff, max_ff]",
        "files":     [fname for _, fname in top],
        "scores":    [round(s, 6) for s, _ in top],
    }

    out = base / "curated_index.json"
    with open(out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pseudo_dir", default="./data/pseudo_gdacs",
                   help="Directory containing pseudo_*.pt files")
    p.add_argument("--top_k", type=int, default=2500,
                   help="Number of top patches to keep in the curated index")
    p.add_argument("--min_ff", type=float, default=0.05,
                   help="Minimum flood fraction to include")
    p.add_argument("--max_ff", type=float, default=0.85,
                   help="Maximum flood fraction to include")
    args = p.parse_args()

    curate(args.pseudo_dir, top_k=args.top_k, min_ff=args.min_ff, max_ff=args.max_ff)


if __name__ == "__main__":
    main()
