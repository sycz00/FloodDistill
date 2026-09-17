"""Download the GDACS Sentinel-2 pool used for pseudo-labelling (paper Sec. IV-A).

Event locations come from the GDACS alert API. Each event gives patches within the event
window and a pre-event baseline of the same location 30-395 days earlier with a stricter
cloud limit. The paper pool has 23,720 tiles (11,895 event and 11,825 baseline
acquisitions, 3,603 events).

A patch that fails one of these checks is sampled again within the event radius
(up to --max_candidates times):
  1. Cloud probability of the patch (COPERNICUS/S2_CLOUD_PROBABILITY) above
     --patch_cloud_prob_max. The scene cloud percentage covers ~110 km, the patch ~2.6 km.
  2. Permanent water fraction (JRC/GSW1_4/GlobalSurfaceWater) above
     --max_permanent_water_frac.
  3. New water (MNDWI > 0 outside JRC permanent water) below --min_new_water_frac
     (0 disables it).

Bands, crop sizes and output layout come from datasets/gee_download.py; the pool is read
by data/unlabeled.py::FloodUnlabeledDataset. patch_quality.csv stores the measured values
per patch (see datasets/flood_pool_quality.py).

    python datasets/download_flood_events.py \
        --save_path $DATA_ROOT/flood_unlabeled \
        --project "$EE_PROJECT" \
        --fetch_gdacs --gdacs_from 2017-01-01 --gdacs_to 2026-07-31 \
        --gdacs_alert 'orange;red' --max_gdacs_events 100000 \
        --patches_per_event 4 --add_baseline --num_workers 8 \
        --patch_cloud_prob_max 40 --max_permanent_water_frac 0.5 \
        --min_new_water_frac 0.01 --max_candidates 8

Short test with a few events:
    python datasets/download_flood_events.py \
        --save_path /tmp/flood_smoke --project "$EE_PROJECT" --smoke --num_workers 4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
import zlib
from datetime import date, timedelta
from multiprocessing.dummy import Lock, Pool

warnings.simplefilter("ignore", UserWarning)

import ee
import numpy as np
import urllib3

# building blocks from gee_download, used unchanged
from gee_download import (
    BAND_CROPS,
    DOWNLOAD_BANDS,
    PATCH_RADIUS_M,
    FloodEvent,
    SEED_EVENTS,
    _download_patch_s2,
    _get_s2_l1c_collection,
    _save_patch,
    fetch_gdacs_events,
    load_events_csv,
)

# GEE assets only needed for the extra filters
_JRC_OCCURRENCE = "JRC/GSW1_4/GlobalSurfaceWater"          # permanent water
_S2_CLOUD_PROB  = "COPERNICUS/S2_CLOUD_PROBABILITY"        # per-pixel cloud prob

# Sidecar filename for per-patch quality measurements.
QUALITY_CSV = "patch_quality.csv"
QUALITY_FIELDS = [
    "event_id", "patch_idx", "lon", "lat", "status",
    "scene_system_index", "tile_cloud_pct",
    "patch_cloud_prob", "water_frac", "permanent_water_frac", "new_water_frac",
    "candidates_tried",
]



def _task_rng(seed: int, event_id: str, patch_idx: int) -> np.random.Generator:
    """RNG per (seed, event, patch), so resampling is reproducible for a given --seed."""
    ent = (seed & 0xFFFFFFFF, zlib.crc32(event_id.encode()) & 0xFFFFFFFF, patch_idx)
    return np.random.default_rng(np.random.SeedSequence(ent))


def _sample_one(event: FloodEvent, radius_km: float,
                rng: np.random.Generator) -> tuple[float, float]:
    """One uniform (lon, lat) inside a radius_km circle around the event centre."""
    spread = radius_km / 111.0
    while True:
        dlon = rng.uniform(-spread, spread)
        dlat = rng.uniform(-spread, spread)
        if dlon * dlon + dlat * dlat <= spread * spread:
            return (event.lon + dlon, event.lat + dlat)


# Server-side quality evaluation (one getInfo per candidate)

def _pick_collection(coords, start, end, cloud_limit):
    """Filtered S2 collection over the window; None if no scenes."""
    col = (
        _get_s2_l1c_collection(cloud_limit)
        .filterDate(start, end)
        .filterBounds(ee.Geometry.Point(coords))
    )
    try:
        if col.size().getInfo() == 0:
            return None
    except (ee.EEException, urllib3.exceptions.HTTPError):
        return None
    return col


def _evaluate(col, coords, cfg) -> dict | None:
    """Cloud, water and new-water fractions of the image that would be downloaded (lowest cloud first).

    Returns a dict (one getInfo call) or None on a GEE error.
    """
    region = ee.Geometry.Point(coords).buffer(PATCH_RADIUS_M).bounds()
    sel = col.sort("CLOUDY_PIXEL_PERCENTAGE", True).first()
    idx = ee.String(sel.get("system:index"))

    # 1) per-patch cloud probability (join S2_CLOUD_PROBABILITY by system:index)
    cp_col = ee.ImageCollection(_S2_CLOUD_PROB).filter(ee.Filter.eq("system:index", idx))
    cloud_prob = ee.Algorithms.If(
        cp_col.size().gt(0),
        ee.Image(cp_col.first()).select("probability")
            .reduceRegion(ee.Reducer.mean(), region, 20, maxPixels=1e9)
            .get("probability"),
        None,
    )

    # 2) open water via MNDWI (green B3 / SWIR B11), also works for turbid flood water
    mndwi = sel.normalizedDifference(["B3", "B11"])
    water = mndwi.gt(cfg["mndwi_thr"]).rename("water")

    # 3) permanent water (JRC occurrence) and new (flood) water = water & !permanent
    perm = (ee.Image(_JRC_OCCURRENCE).select("occurrence")
            .gt(cfg["perm_occ_thr"]).unmask(0).rename("perm"))
    new_water = water.And(perm.Not()).rename("newwater")

    stats = ee.Dictionary({
        "scene_system_index": idx,
        "tile_cloud_pct": sel.get("CLOUDY_PIXEL_PERCENTAGE"),
        "patch_cloud_prob": cloud_prob,
        "water_frac": water.reduceRegion(
            ee.Reducer.mean(), region, 20, maxPixels=1e9).get("water"),
        "permanent_water_frac": perm.reduceRegion(
            ee.Reducer.mean(), region, 30, maxPixels=1e9).get("perm"),
        "new_water_frac": new_water.reduceRegion(
            ee.Reducer.mean(), region, 20, maxPixels=1e9).get("newwater"),
    })
    try:
        return stats.getInfo()
    except (ee.EEException, urllib3.exceptions.HTTPError):
        return None


def _passes_gates(m: dict, cfg) -> bool:
    """Apply the three opt-in quality gates to a measured candidate."""
    cp = m.get("patch_cloud_prob")
    if cp is not None and cp > cfg["patch_cloud_prob_max"]:
        return False
    pw = m.get("permanent_water_frac")
    if pw is not None and pw > cfg["max_permanent_water_frac"]:
        return False
    nw = m.get("new_water_frac")
    if cfg["min_new_water_frac"] > 0 and (nw is None or nw < cfg["min_new_water_frac"]):
        return False
    return True


def _download_from_col(col, coords, location_dir) -> bool:
    """Download + save the collection's best image (same path as the original)."""
    try:
        patch = _download_patch_s2(col, coords, PATCH_RADIUS_M, DOWNLOAD_BANDS, BAND_CROPS)
        os.makedirs(location_dir, exist_ok=True)
        _save_patch(patch["raster"], patch["coords"], patch["metadata"], location_dir)
        return True
    except (ee.EEException, urllib3.exceptions.HTTPError, Exception):
        return False


# Per-patch worker

def _worker(task: tuple) -> tuple:
    """Download one event patch with resampling; returns (event_id, patch_idx, lon, lat, status, metrics)."""
    event, patch_idx, save_path, cfg = task
    location_id  = f"{event.event_id}_{patch_idx:04d}"
    location_dir = os.path.join(save_path, location_id)

    # Resume: skip if this location already has at least one scene on disk.
    if os.path.isdir(location_dir) and any(
        os.path.isdir(os.path.join(location_dir, d)) for d in os.listdir(location_dir)
    ):
        return (event.event_id, patch_idx, None, None, 1, {"resumed": True})

    rng = _task_rng(cfg["seed"], event.event_id, patch_idx)

    # Windows to try per candidate: primary, then a widened window at the fallback
    # cloud threshold (S2 revisit ~5 days) - same escalation as the original.
    s_ext = (date.fromisoformat(event.start_date) - timedelta(days=7)).isoformat()
    e_ext = (date.fromisoformat(event.end_date)   + timedelta(days=7)).isoformat()
    windows = [
        (event.start_date, event.end_date, cfg["cloud_pct"]),
        (s_ext, e_ext, cfg["cloud_pct_fallback"]),
    ]

    last_lon = last_lat = None
    for attempt in range(1, cfg["max_candidates"] + 1):
        lon, lat = _sample_one(event, cfg["event_radius_km"], rng)
        last_lon, last_lat = lon, lat
        coords = [lon, lat]

        for start, end, cloud_limit in windows:
            col = _pick_collection(coords, start, end, cloud_limit)
            if col is None:
                continue
            m = _evaluate(col, coords, cfg)
            if m is None:
                continue
            if not _passes_gates(m, cfg):
                break  # image exists but fails a gate -> resample a new location
            if not _download_from_col(col, coords, location_dir):
                continue
            if cfg["add_baseline"]:
                _download_baseline(coords, event, location_dir, cfg)
            m.update({"candidates_tried": attempt})
            return (event.event_id, patch_idx, lon, lat, 1, m)

    return (event.event_id, patch_idx, last_lon, last_lat, 0,
            {"candidates_tried": cfg["max_candidates"]})


def _download_baseline(coords, event: FloodEvent, location_dir, cfg) -> None:
    """Download one clear pre-event reference (about one year earlier) with a stricter cloud limit; errors are not fatal."""
    pre_start = (date.fromisoformat(event.start_date) - timedelta(days=395)).isoformat()
    pre_end   = (date.fromisoformat(event.start_date) - timedelta(days=30)).isoformat()
    col = _pick_collection(coords, pre_start, pre_end, cfg["baseline_cloud_pct"])
    if col is not None:
        _download_from_col(col, coords, location_dir)


# Main

def main() -> None:
    p = argparse.ArgumentParser(
        description="GDACS flood-event Sentinel-2 pool builder (the paper's training pool).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--events_csv",  default=None,
                     help="CSV: event_id,lon,lat,start_date,end_date[,description]")
    src.add_argument("--fetch_gdacs", action="store_true",
                     help="Fetch flood events from the GDACS public API.")

    # GDACS options. Event window; baselines go up to 395 days further back.
    p.add_argument("--gdacs_from",  default="2017-01-01")
    p.add_argument("--gdacs_to",    default="2026-07-31")
    p.add_argument("--gdacs_alert", default="orange;red",
                   help="Alert levels, semicolon-separated (quote it in bash!).")
    p.add_argument("--max_gdacs_events", type=int, default=2000,
                   help="Fetch cap the original hard-codes at 2000; raise to pull more.")

    # Download settings (inherited defaults from the original)
    p.add_argument("--save_path", required=True)
    p.add_argument("--project",   default=None,
                   help="Google Cloud project ID with the Earth Engine API enabled. "
                        "Falls back to the $EE_PROJECT env var if omitted.")
    p.add_argument("--patches_per_event", type=int, default=8)
    p.add_argument("--event_radius_km",   type=float, default=30.0)
    p.add_argument("--cloud_pct",          type=float, default=50.0,
                   help="Tile-level primary cloud threshold (pre-filter).")
    p.add_argument("--cloud_pct_fallback", type=float, default=80.0)
    p.add_argument("--baseline_cloud_pct", type=float, default=20.0)
    p.add_argument("--add_baseline", action="store_true")
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--resume",       default=None)
    p.add_argument("--seed",         type=int, default=42)

    # optional quality filters
    p.add_argument("--patch_cloud_prob_max", type=float, default=40.0,
                   help="Reject patch if mean S2 cloud probability over the patch "
                        "exceeds this %% (per-PATCH, not per-tile). Default 40.")
    p.add_argument("--max_permanent_water_frac", type=float, default=0.5,
                   help="Reject patch if JRC permanent-water fraction exceeds this "
                        "(0-1). Default 0.5.")
    p.add_argument("--min_new_water_frac", type=float, default=0.0,
                   help="Require flood (new) water fraction >= this (0-1). 0 = off. "
                        "Try 0.01-0.05 to actively select flooded patches.")
    p.add_argument("--mndwi_thr", type=float, default=0.0,
                   help="MNDWI water threshold (>thr = water). Default 0.0.")
    p.add_argument("--perm_occ_thr", type=float, default=50.0,
                   help="JRC occurrence %% above which a pixel is 'permanent' water.")
    p.add_argument("--max_candidates", type=int, default=8,
                   help="Resample this many locations per patch slot before giving "
                        "up when candidates keep failing the gates. Default 8.")

    # Smoke test: a few seed events, few patches, small budget.
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run (3 seed events x 2 patches) to validate the path.")

    args = p.parse_args()

    # Fall back to $EE_PROJECT and give a clear error if no project is set.
    project = args.project or os.environ.get("EE_PROJECT")
    if not project:
        sys.exit(
            "[error] no Earth Engine project ID. Pass --project YOUR_PROJECT_ID "
            "or set EE_PROJECT. (Register/find the ID at "
            "https://console.cloud.google.com/earth-engine ; see datasets/README.md.)"
        )
    ee.Initialize(project=project)
    print(f"[gee] initialised  project={project}")

    # Load events
    if args.smoke:
        events = SEED_EVENTS[:3]
        args.patches_per_event = min(args.patches_per_event, 2)
        print(f"[smoke] {len(events)} seed events x {args.patches_per_event} patches")
    elif args.events_csv:
        events = load_events_csv(args.events_csv)
    elif args.fetch_gdacs:
        events = fetch_gdacs_events(args.gdacs_from, args.gdacs_to,
                                    args.gdacs_alert, limit=args.max_gdacs_events)
    else:
        events = SEED_EVENTS
        print(f"[events] using {len(events)} built-in seed flood events")

    cfg = {
        "seed": args.seed,
        "event_radius_km": args.event_radius_km,
        "cloud_pct": args.cloud_pct,
        "cloud_pct_fallback": args.cloud_pct_fallback,
        "baseline_cloud_pct": args.baseline_cloud_pct,
        "add_baseline": args.add_baseline,
        "patch_cloud_prob_max": args.patch_cloud_prob_max,
        "max_permanent_water_frac": args.max_permanent_water_frac,
        "min_new_water_frac": args.min_new_water_frac,
        "mndwi_thr": args.mndwi_thr,
        "perm_occ_thr": args.perm_occ_thr,
        "max_candidates": args.max_candidates,
    }

    # Resume state
    done_patches: set[tuple] = set()
    checked_csv = args.resume or os.path.join(args.save_path, "checked_patches.csv")
    if args.resume and os.path.exists(checked_csv):
        with open(checked_csv, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 5:
                    done_patches.add((row[0], int(row[1])))
        print(f"[resume] skipping {len(done_patches)} already-attempted patches")

    # Task list
    tasks = []
    for event in events:
        for patch_idx in range(args.patches_per_event):
            if (event.event_id, patch_idx) in done_patches:
                continue
            tasks.append((event, patch_idx, args.save_path, cfg))

    print(f"[download] {len(tasks)} patch slots "
          f"({len(events)} events x {args.patches_per_event}); "
          f"gates: cloud_prob<={args.patch_cloud_prob_max} "
          f"perm_water<={args.max_permanent_water_frac} "
          f"new_water>={args.min_new_water_frac} "
          f"resample<={args.max_candidates}")
    if not tasks:
        print("[download] nothing to do.")
        return

    os.makedirs(args.save_path, exist_ok=True)
    quality_csv = os.path.join(args.save_path, QUALITY_CSV)
    write_header = not os.path.exists(quality_csv)

    lock = Lock()
    n_success = n_fail = 0
    t0 = time.perf_counter()

    def _record(result: tuple) -> None:
        nonlocal n_success, n_fail
        event_id, patch_idx, lon, lat, status, m = result
        with lock:
            if status:
                n_success += 1
            else:
                n_fail += 1
            with open(checked_csv, "a", newline="") as f:
                csv.writer(f).writerow([event_id, patch_idx, lon, lat, status])
            if not m.get("resumed"):
                with open(quality_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=QUALITY_FIELDS)
                    nonlocal write_header
                    if write_header:
                        w.writeheader()
                        write_header = False
                    w.writerow({
                        "event_id": event_id, "patch_idx": patch_idx,
                        "lon": lon, "lat": lat, "status": status,
                        "scene_system_index": m.get("scene_system_index", ""),
                        "tile_cloud_pct": m.get("tile_cloud_pct", ""),
                        "patch_cloud_prob": m.get("patch_cloud_prob", ""),
                        "water_frac": m.get("water_frac", ""),
                        "permanent_water_frac": m.get("permanent_water_frac", ""),
                        "new_water_frac": m.get("new_water_frac", ""),
                        "candidates_tried": m.get("candidates_tried", ""),
                    })

    with Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, tasks), 1):
            _record(result)
            if i % 10 == 0 or i == len(tasks):
                el = (time.perf_counter() - t0) / 60
                print(f"  [{i}/{len(tasks)}] success={n_success} "
                      f"failed={n_fail} elapsed={el:.1f}min")

    print(f"\n[done] {n_success} patches kept  {n_fail} exhausted candidates  "
          f"total={(time.perf_counter()-t0)/60:.1f}min")
    print(f"[done] data:          {args.save_path}")
    print(f"[done] progress log:  {checked_csv}")
    print(f"[done] quality sidecar: {quality_csv}")


if __name__ == "__main__":
    main()
