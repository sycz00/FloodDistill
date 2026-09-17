# !/usr/bin/env python3
"""Check index size, candidate-pool settings and overlap between selectors.

Each index must contain exactly the training N: the loader samples uniformly
from it. Filtered selectors must share their candidate counts and gates.
Report measured overlap alongside the forced floor max(0,2*N/C-1).
Identical selected files cannot provide a distinct selection control.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def load(base: Path, patterns: list[str]) -> dict[str, dict]:
    out = {}
    for pat in patterns:
        for p in sorted(base.glob(pat)):
            try:
                out[p.name] = json.load(open(p))
            except Exception as e:                      # corrupt file is a finding, not a crash
                print(f"[FAIL] {p.name}: unreadable ({e})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    ap.add_argument("--n", type=int, default=2500,
                    help="training budget N the indices must hold exactly")
    ap.add_argument("--patterns", nargs="+",
                    default=["*_index.json"],
                    help="which indices to check (default: every *_index.json)")
    ap.add_argument("--all_pairs", action="store_true",
                    help="print the full pairwise-overlap matrix, not only notable rows")
    a = ap.parse_args()

    base = Path(a.pseudo_dir)
    idx = load(base, a.patterns)
    if not idx:
        raise SystemExit(f"no indices matched {a.patterns} under {base}")

    # The per-N sweep indices are built at their own N on purpose, so check them against the
    # N in their filename rather than against the single --n budget.
    def expected_n(name: str, d: dict) -> int:
        stem = name.replace("_index.json", "")
        for part in stem.split("_"):
            if part.startswith("N") and part[1:].isdigit():
                return int(part[1:])
        return a.n

    print(f"{'index':38s} {'files':>6s} {'want':>6s} {'cand':>7s} "
          f"{'ff-range':>13s} {'cstd':>6s}  status")
    fails = 0
    for name, d in sorted(idx.items()):
        files, want = len(d.get("files", [])), expected_n(name, d)
        cand = d.get("n_candidates", -1)
        rng = f"[{d.get('min_ff','?')},{d.get('max_ff','?')}]"
        ok = files == want
        fails += not ok
        print(f"{name:38s} {files:>6d} {want:>6d} {cand:>7d} {rng:>13s} "
              f"{str(d.get('min_conf_std','?')):>6s}  {'ok' if ok else 'SIZE MISMATCH'}")

    # Matched-pool check, filtered arms only (the *_raw family relaxes the filter by design).
    filt = {n: d for n, d in idx.items() if "_raw_" not in n
            and (d.get("min_conf_std") or 0) > 0}
    pools = {(d.get("n_candidates"), d.get("min_ff"), d.get("max_ff"), d.get("min_conf_std"))
             for d in filt.values()}
    print()
    if len(pools) <= 1:
        print(f"[ok] all {len(filt)} filtered arms share one candidate pool: {pools or '—'}")
    else:
        fails += 1
        print("[FAIL] filtered arms do NOT share a candidate pool — differences between them "
              "are pool differences, not ranking differences:")
        for n, d in sorted(filt.items()):
            print(f"   {n:38s} cand={d.get('n_candidates')} ff=[{d.get('min_ff')},"
                  f"{d.get('max_ff')}] conf_std>={d.get('min_conf_std')}")

    # Pairwise overlap, same-N arms only (an N=50 and an N=2500 index share trivially).
    print("\npairwise overlap (same budget only; 'forced' = pigeonhole floor 2N/C-1):")
    by_n: dict[int, list[str]] = {}
    for name, d in idx.items():
        by_n.setdefault(len(d.get("files", [])), []).append(name)
    shown = 0
    for n, names in sorted(by_n.items()):
        if len(names) < 2:
            continue
        for x, y in combinations(sorted(names), 2):
            sx, sy = set(idx[x]["files"]), set(idx[y]["files"])
            ov = len(sx & sy) / max(n, 1)
            c = max(idx[x].get("n_candidates", 0), idx[y].get("n_candidates", 0))
            forced = max(0.0, 2 * n / c - 1) if c else 0.0
            # An identical pair is only a FINDING when the two arms claim to be different
            # methods. Two names for one arm (diversity_index / diversity_N2500_index) are
            # identical by construction and would otherwise cry wolf on every run.
            same_method = idx[x].get("score_fn") == idx[y].get("score_fn")
            flag = ""
            if ov == 1.0:
                flag = ("  <-- alias: same score_fn, one arm under two names" if same_method
                        else "  <-- IDENTICAL: cannot produce a different model")
                fails += not same_method
            elif forced and ov <= forced + 0.02:
                flag = "  <-- ~forced by budget"
            if flag or a.all_pairs:
                print(f"  N={n:<5d} {x:34s} vs {y:34s} {ov:6.1%}  forced {forced:5.1%}{flag}")
                shown += 1
    if not shown:
        print("  (nothing notable — pass --all_pairs for the full matrix)")

    print(f"\n{'PASS' if not fails else f'{fails} PROBLEM(S)'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
