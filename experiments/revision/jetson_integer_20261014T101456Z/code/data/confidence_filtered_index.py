# !/usr/bin/env python3
"""Rank the filtered pseudo pool by mean teacher confidence.

Use the same flood-fraction and confidence-spread gates as FPS when comparing
ranking rules. This differs from the legacy confidence selector, which uses
4*f*(1-f)*mean_conf and only the flood-fraction gate.
Writes confidence_filtered_index.json; build it at the intended training N.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from tqdm import tqdm


def build(pseudo_dir: str, top_k: int = 2500, min_ff: float = 0.05,
          max_ff: float = 0.85, min_conf_std: float = 0.05,
          out_name: str = "confidence_filtered_index.json") -> None:
    base = Path(pseudo_dir)
    files = sorted(base.glob("pseudo_*.pt"))
    if not files:
        raise FileNotFoundError(f"No pseudo_*.pt files found in {pseudo_dir!r}")

    records = []
    for pt in tqdm(files, desc="scoring", unit="patch"):
        item = torch.load(pt, map_location="cpu", weights_only=True)
        ff = item["mask"].float().mean().item()
        if "confidence" in item:
            conf = item["confidence"].float()
            mean_conf, conf_std = conf.mean().item(), conf.std().item()
        else:
            mean_conf, conf_std = 1.0, 0.0
        records.append({"file": pt.name, "flood_frac": ff,
                        "mean_conf": mean_conf, "conf_std": conf_std})

    # same pre-filter as diversity_index / influence_v3b
    cand = [r for r in records
            if min_ff <= r["flood_frac"] <= max_ff and r["conf_std"] >= min_conf_std]
    print(f"\n{len(records)} patches scanned")
    print(f"  ff in [{min_ff}, {max_ff}] AND conf_std >= {min_conf_std}: {len(cand)}")
    if len(cand) < top_k:
        raise SystemExit(f"only {len(cand)} candidates for top_k={top_k}")

    cand.sort(key=lambda r: r["mean_conf"], reverse=True)
    top = cand[:top_k]
    print(f"  selected top-{top_k} by mean_conf")
    print(f"  mean_conf range in selection: "
          f"[{top[-1]['mean_conf']:.5f}, {top[0]['mean_conf']:.5f}]")

    out = base / out_name
    json.dump({
        "n_total": len(records),
        "n_candidates": len(cand),
        "n_curated": len(top),
        "min_ff": min_ff, "max_ff": max_ff, "min_conf_std": min_conf_std,
        "score_fn": "PURE mean teacher confidence, ranked over the SAME 3,476 pre-filtered "
                    "candidates as diversity_index and influence_v3b_index "
                    "(matched-pool confidence baseline)",
        "files": [r["file"] for r in top],
        "scores": [round(r["mean_conf"], 6) for r in top],
    }, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


# in main() so scripts/curate.py can call it like the other selectors
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    ap.add_argument("--top_k", type=int, default=2500)
    ap.add_argument("--out_name", default="confidence_filtered_index.json",
                    help="output index filename (use a distinct name when relaxing the pre-filter)")
    ap.add_argument("--min_ff", type=float, default=0.05)
    ap.add_argument("--max_ff", type=float, default=0.85)
    ap.add_argument("--min_conf_std", type=float, default=0.05)
    a = ap.parse_args()
    build(a.pseudo_dir, a.top_k, a.min_ff, a.max_ff, a.min_conf_std, a.out_name)


if __name__ == "__main__":
    main()
