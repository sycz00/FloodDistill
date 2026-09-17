# !/usr/bin/env python3
"""Map a selected index onto a rebuilt pseudo pool using source manifests.

Join on source (location/scene ID), preserving source-index order. Continue
down that ranking to fill N when scenes are missing from the target pool.
The output contains exactly N files for the training recipe's index check.

python experiments/map_index_to_pool.py --src_pool data/pseudo_gdacs \
    --src_index diversity_N2500_index.json --dst_pool data/pseudo_gdacs_regen \
    --n 252 --out_name mapped_diversity_N252_index.json"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_manifest(pool: Path) -> list[dict]:
    with open(pool / "source_manifest.csv") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_pool", required=True, type=Path)
    ap.add_argument("--src_index", default="diversity_N2500_index.json")
    ap.add_argument("--dst_pool", required=True, type=Path)
    ap.add_argument("--n", required=True, type=int)
    ap.add_argument("--out_name", required=True)
    args = ap.parse_args()

    src_rows = read_manifest(args.src_pool)
    dst_rows = read_manifest(args.dst_pool)
    src_file2src = {r["file"]: r["source"] for r in src_rows}
    dst_src2file = {r["source"]: r["file"] for r in dst_rows}

    ranking = json.load(open(args.src_pool / args.src_index))["files"]

    picked, ranks, missed = [], [], 0
    for rank, f in enumerate(ranking):
        if len(picked) == args.n:
            break
        tgt = dst_src2file.get(src_file2src.get(f, ""))
        if tgt is None:
            missed += 1
            continue
        picked.append(tgt)
        ranks.append(rank)

    if len(picked) != args.n:
        raise SystemExit(
            f"[FAIL] only {len(picked)} of {args.n} ranked entries exist in {args.dst_pool}; "
            f"widen --src_index or lower --n"
        )

    # How faithful is this to the shipped cut at the same N?
    kept = sum(1 for r in ranks if r < args.n)
    depth = ranks[-1] + 1

    out = {
        "files": picked,
        "n_curated": len(picked),
        "note": (
            f"mapped from {args.src_pool.name}/{args.src_index} onto {args.dst_pool.name} by "
            f"source_manifest 'source' join; walked FPS ranks 0..{depth-1} to fill N={args.n}; "
            f"{kept}/{args.n} tiles come from the shipped top-{args.n} cut, "
            f"{args.n - kept} are FPS top-ups from ranks {args.n}..{depth-1}; "
            f"{missed} ranked scenes absent from the regenerated pool"
        ),
        "provenance": {
            "src_pool": str(args.src_pool),
            "src_index": args.src_index,
            "dst_pool": str(args.dst_pool),
            "n": args.n,
            "ranks_used": ranks,
            "n_from_original_cut": kept,
            "n_topped_up": args.n - kept,
            "n_ranked_scenes_missing": missed,
            "max_rank_used": depth - 1,
        },
    }
    dst = args.dst_pool / args.out_name
    json.dump(out, open(dst, "w"), indent=2)
    print(f"[ok] {dst}  N={len(picked)}")
    print(f"     {kept}/{args.n} from the shipped top-{args.n} cut "
          f"({100*kept/args.n:.1f}% overlap); {args.n-kept} topped up from ranks "
          f"{args.n}..{depth-1}; {missed} ranked scenes missing from target pool")


if __name__ == "__main__":
    main()
