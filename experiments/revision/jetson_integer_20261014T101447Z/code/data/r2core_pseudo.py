"""R2-Core: weighted facility-location selection of pseudo-labelled tiles.

F(S) = sum_x w(x) * max_{s in S} sim(x,s), with |S| = N.
w combines reliability (teacher consistency, MNDWI agreement, mean confidence),
informativeness (4*f*(1-f) and boundary density) and cloud quality, so the
reliability part does contain teacher-derived signals.

Only coverage changes with the weights; the candidate pool stays the same.
Marginal gains are cached and updated lazily. Region quotas cap the maximum.
No minimum per region. --uniform_weights gives the unweighted facility-location
control. See experiments/build_indices.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# pre-filter constants shared with diversity_pseudo.py / confidence_filtered_index.py.
# All selectors have to rank the same candidates, otherwise a difference between arms
# could come from the pool and not from the ranking rule.
MIN_FF, MAX_FF, MIN_CONF_STD = 0.05, 0.85, 0.05

_PHI = ["flood_frac", "mean_conf", "conf_std",
        *[f"band_mean_{i}" for i in range(6)], *[f"band_std_{i}" for i in range(6)]]


def _f(row: dict, key: str, default: float = np.nan) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else default


def _normalise(X: np.ndarray) -> np.ndarray:
    """Clip each dim at its 1st/99th percentile and scale to [0,1], same as diversity_pseudo."""
    lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    den = np.where(hi - lo > 1e-8, hi - lo, 1.0)
    return np.clip((X - lo) / den, 0.0, 1.0)


def _continent(lon: float, lat: float) -> str:
    """Coarse region bucket for the matroid quota. Matches the strategy doc's buckets."""
    if np.isnan(lon) or np.isnan(lat):
        return "Unknown"
    if -25 <= lon < 45 and 35 <= lat <= 72:
        return "Europe"
    if -20 <= lon < 55 and -40 <= lat < 35:
        return "Africa"
    if 25 <= lon < 60 and 12 <= lat < 43:
        return "MiddleEast"
    if 60 <= lon < 95 and 5 <= lat <= 60:
        return "SouthAsia"
    if 95 <= lon <= 180 and -12 <= lat <= 60:
        return "EastSEAsia"
    if 45 <= lon < 180 and 35 < lat <= 80:
        return "NorthAsia"
    if -170 <= lon < -30 and 12 < lat <= 75:
        return "NorthAmerica"
    if -85 <= lon < -30 and -60 <= lat <= 12:
        return "SouthAmerica"
    if 110 <= lon <= 180 and -50 <= lat < -12:
        return "Oceania"
    return "Other"


# scores

def reliability(rows: list[dict]) -> np.ndarray:
    """Combine consistency, MNDWI agreement and mean confidence geometrically.

    Keep confidence spread in the informativeness/filter terms, not in this score.
    """
    c = np.array([_f(r, "c_mean", np.nan) for r in rows])
    rw = np.array([_f(r, "rho_water", 0.0) for r in rows])
    rn = np.array([_f(r, "rho_new", np.nan) for r in rows])
    mc = np.array([_f(r, "mean_conf", 0.0) for r in rows])

    # Unpaired tiles (no pre-event baseline) have no rho_new measurement; fall back
    # to rho_water rather than scoring them 0, which would be a measurement claim.
    rn = np.where(np.isnan(rn), rw, rn)
    c = np.where(np.isnan(c), np.nanmedian(c) if np.isfinite(c).any() else 1.0, c)

    # Don't average rho_water and rho_new, they measure different things. rho_water asks
    # whether there is water at all (catches hallucinations), rho_new whether it is flood
    # water. With an average a permanent river counts as reliable. That's how pre-event
    # baselines ended up looking more reliable than flood scenes on every teacher-side
    # signal (measured rho_water 0.506 vs 0.420, c_mean 0.817 vs 0.773; permanent water is
    # spectrally easy, so stable and confident). The flood-only protocol ignores permanent
    # water, so rho_new is the right signal. Multiplying means either one near zero vetoes
    # the tile. That zeroes all baselines (rho_new == 0 by construction) but only 7.9% of
    # flood candidates, and those are the tiles whose "flood" is not new water.

    # Scaled by p99 rather than rank-normalised: rank normalisation would spread the
    # mass of exact zeros across arbitrary ranks (up to 0.41 here) and destroy the
    # zero semantics this term depends on.
    rho_raw = np.sqrt(np.clip(rw, 0, None) * np.clip(rn, 0, None))
    p99 = np.percentile(rho_raw, 99)
    rho = np.clip(rho_raw / (p99 if p99 > 1e-8 else 1.0), 0.0, 1.0)
    conf = _rank01(mc)
    cons = _rank01(c)
    eps = 1e-3
    return np.cbrt((cons + eps) * (rho + eps) * (conf + eps))


def informativeness(rows: list[dict]) -> np.ndarray:
    """Flood-fraction balance 4*f*(1-f) multiplied by boundary density."""
    ff = np.array([_f(r, "flood_frac", 0.0) for r in rows])
    bd = np.array([_f(r, "boundary_density", 0.0) for r in rows])
    return np.sqrt(np.clip(4.0 * ff * (1.0 - ff), 0, 1) * _rank01(bd))


def img_quality(rows: list[dict]) -> np.ndarray:
    """Convert measured cloud probability into a within-pool quality weight."""
    cp = np.array([_f(r, "patch_cloud_prob", np.nan) for r in rows])
    cp = np.where(np.isnan(cp), np.nanmedian(cp) if np.isfinite(cp).any() else 20.0, cp)
    return np.clip(1.0 - cp / 100.0, 0.05, 1.0)


def _rank01(v: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0,1], all these scores have heavy tails.

        Ties get average ranks. argsort of argsort would order ties by index, and c_mean
        has lots of values at exactly 1.0 (p90 = 1.000), so that matters here.
    """
    from scipy.stats import rankdata
    return (rankdata(v, method="average") - 1.0) / max(len(v) - 1, 1)


# weighted facility location, lazy greedy

def facility_location_greedy(sim: np.ndarray, w: np.ndarray, k: int,
                             groups: np.ndarray | None = None,
                             quotas: dict | None = None) -> np.ndarray:
    """Maximise sum_x w(x)*max_s sim(x,s) with lazy greedy updates.

    Optional groups/quotas impose per-region upper bounds, not lower bounds.
    """
    import heapq

    n = len(sim)
    k = min(k, n)
    # best[x] = w(x) * max_{s in S} sim(x,s); starts at 0 (empty set covers nothing)
    best = np.zeros(n, dtype=np.float32)
    selected: list[int] = []
    taken = {g: 0 for g in (quotas or {})}

    # Initial marginal gains: F({e}) - F(empty) = sum_x w(x)*sim(x,e)
    gains = (sim * w[:, None]).sum(axis=0)
    heap = [(-float(g), int(i)) for i, g in enumerate(gains)]
    heapq.heapify(heap)
    in_set = np.zeros(n, bool)

    while len(selected) < k and heap:
        neg, e = heapq.heappop(heap)
        if in_set[e]:
            continue
        if quotas is not None and groups is not None:
            g = groups[e]
            if taken.get(g, 0) >= quotas.get(g, k):
                continue                      # matroid: this region is full
        # Re-evaluate the stale gain; if it is still the best, it is exact.
        cand = np.maximum(best, w * sim[:, e])
        gain = float(cand.sum() - best.sum())
        if heap and gain < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-gain, e))  # stale -> re-queue with the true value
            continue
        selected.append(e)
        in_set[e] = True
        best = cand
        if quotas is not None and groups is not None:
            taken[groups[e]] = taken.get(groups[e], 0) + 1
    return np.array(selected, dtype=np.int64)


# main

def select(pseudo_dir: str, top_k: int = 2500, out_name: str = "r2core_index.json",
           uniform_weights: bool = False, region_quota: float = 0.0,
           features_name: str = "pool_features.csv",
           consistency_name: str = "consistency.csv",
           exclude_baselines: bool = False,
           min_ff: float = MIN_FF, max_ff: float = MAX_FF,
           min_conf_std: float = MIN_CONF_STD) -> None:
    base = Path(pseudo_dir)
    feats = base / features_name
    if not feats.exists():
        raise FileNotFoundError(f"{feats} not found — run: python -m data.pool_features "
                                f"--pseudo_dir {pseudo_dir}")
    rows = list(csv.DictReader(open(feats)))

    cons_path = base / consistency_name
    if cons_path.exists():
        cmap = {r["file"]: r["c_mean"] for r in csv.DictReader(open(cons_path))}
        for r in rows:
            r["c_mean"] = cmap.get(r["file"], "")
        print(f"[r2core] consistency c(x) joined for {sum(1 for r in rows if r['c_mean'])} tiles")
    else:
        print(f"[r2core] WARNING: {cons_path} missing — running WITHOUT c(x), the haze term")

    # candidate pre-filter, identical to every other selector
    cand = [r for r in rows
            if min_ff <= _f(r, "flood_frac", -1) <= max_ff
            and _f(r, "conf_std", -1) >= min_conf_std]
    print(f"[r2core] {len(rows)} tiles -> {len(cand)} candidates "
          f"(ff in [{min_ff},{max_ff}] and conf_std >= {min_conf_std})")
    if exclude_baselines:
        # H3 ablation: hard-drop pre-event acquisitions instead of letting rho_new
        # down-weight them. Breaks the matched candidate pool on purpose, so this arm
        # is only interpretable against another exclude_baselines arm.
        n0 = len(cand)
        cand = [r for r in cand if r.get("is_flood_scene") == "1"]
        print(f"[r2core] H3: dropped {n0-len(cand)} pre-event baselines -> {len(cand)}")
    if len(cand) < top_k:
        raise SystemExit(f"only {len(cand)} candidates for top_k={top_k}")

    # weights
    r_sc, i_sc, q_sc = reliability(cand), informativeness(cand), img_quality(cand)
    if uniform_weights:
        w = np.ones(len(cand), np.float32)
        print("[r2core] ABLATION: uniform weights — pure facility location, no reliability")
    else:
        w = (r_sc * i_sc * q_sc).astype(np.float32)
        w = np.clip(w / max(w.mean(), 1e-8), 0.02, 20.0)   # mean-1, tails bounded

    # similarity on the same 15-dim phi as FPS
    X = _normalise(np.array([[_f(r, k, 0.0) for k in _PHI] for r in cand], np.float32))
    d2 = (
        (X * X).sum(1)[:, None] + (X * X).sum(1)[None, :] - 2.0 * X @ X.T
    ).clip(min=0.0)
    sim = np.exp(-d2 / (2.0 * np.median(d2[d2 > 0]))).astype(np.float32)  # RBF, median heuristic

    # optional per-region matroid quotas
    groups = quotas = None
    if region_quota > 0:
        groups = np.array([_continent(_f(r, "lon"), _f(r, "lat")) for r in cand])
        uniq, counts = np.unique(groups, return_counts=True)
        # cap = region_quota x the region's proportional share; >1 loosens the cap
        quotas = {g: max(1, int(np.ceil(region_quota * top_k * c / len(cand))))
                  for g, c in zip(uniq, counts)}
        print(f"[r2core] matroid quotas over {len(uniq)} regions "
              f"(factor {region_quota}): {dict(sorted(quotas.items()))}")

    print(f"[r2core] greedy facility location: {top_k} of {len(cand)} ...")
    idx = facility_location_greedy(sim, w, top_k, groups, quotas)
    sel = [cand[i] for i in idx]
    print(f"[r2core] selected {len(sel)}")

    # report
    def _stat(name, arr_all, arr_sel):
        print(f"    {name:16s} pool {arr_all.mean():.3f} -> selected {arr_sel.mean():.3f}")

    for nm, v in [("reliability r", r_sc), ("informativeness i", i_sc), ("quality q", q_sc)]:
        _stat(nm, v, v[idx])
    for nm in ("flood_frac", "mean_conf", "rho_water", "boundary_density"):
        a = np.array([_f(r, nm, np.nan) for r in cand])
        print(f"    {nm:16s} pool {np.nanmean(a):.3f} -> selected {np.nanmean(a[idx]):.3f}")
    if any(r.get("c_mean") for r in cand):
        a = np.array([_f(r, "c_mean", np.nan) for r in cand])
        print(f"    {'c_mean (haze)':16s} pool {np.nanmean(a):.3f} -> selected {np.nanmean(a[idx]):.3f}")
    nb = sum(1 for r in sel if r.get("is_flood_scene") == "0")
    nbp = sum(1 for r in cand if r.get("is_flood_scene") == "0")
    print(f"    pre-event baselines: {nb}/{len(sel)} selected ({nb/len(sel):.1%}) "
          f"vs {nbp/len(cand):.1%} of candidates")

    index = {
        "n_total": len(rows), "n_candidates": len(cand), "n_curated": len(sel),
        "min_ff": min_ff, "max_ff": max_ff, "min_conf_std": min_conf_std,
        "score_fn": (
            "R2-Core: weighted facility location (lazy greedy, 1-1/e) over the same "
            "15-dim phi and pre-filter as diversity_index; coverage weight "
            "w = g(r)*h(i)*q with r = cbrt(consistency * rho_MNDWI * conf), "
            "i = sqrt(4f(1-f) * boundary_density), q = 1 - cloud_prob"
            + (" [ABLATION uniform w]" if uniform_weights else "")
            + (f" [region matroid x{region_quota}]" if region_quota > 0 else "")
        ),
        "files": [r["file"] for r in sel],
        "scores": [round(float(w[i]), 6) for i in idx],
    }
    out = base / out_name
    json.dump(index, open(out, "w"), indent=2)
    print(f"\n[r2core] wrote {out}")

    div = base / f"diversity_N{top_k}_index.json"
    if div.exists():
        d = set(json.load(open(div))["files"])
        ov = len(d & set(index["files"]))
        print(f"[r2core] overlap with FPS-{len(d)}: {ov}/{len(sel)} ({ov/len(sel):.1%})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    p.add_argument("--top_k", type=int, default=2500)
    p.add_argument("--out_name", default="r2core_index.json")
    p.add_argument("--uniform_weights", action="store_true",
                   help="ablation: w == 1, isolating the facility-location swap from "
                        "the reliability weighting")
    p.add_argument("--region_quota", type=float, default=0.0,
                   help="partition-matroid cap as a multiple of each region's "
                        "proportional share (0 = off, 1.5 is a reasonable start)")
    p.add_argument("--min_ff", type=float, default=MIN_FF,
                   help="pre-filter floor on flood_frac. 0.0 disables it, letting the "
                        "selector rank the FULL pool including negatives.")
    p.add_argument("--max_ff", type=float, default=MAX_FF)
    p.add_argument("--min_conf_std", type=float, default=MIN_CONF_STD)
    p.add_argument("--features_name", default="pool_features.csv")
    p.add_argument("--consistency_name", default="consistency.csv")
    p.add_argument("--exclude_baselines", action="store_true",
                   help="H3 ablation: hard-drop pre-event baseline acquisitions")
    a = p.parse_args()
    select(a.pseudo_dir, a.top_k, a.out_name, a.uniform_weights, a.region_quota,
           a.features_name, a.consistency_name, a.exclude_baselines,
           a.min_ff, a.max_ff, a.min_conf_std)


if __name__ == "__main__":
    main()
