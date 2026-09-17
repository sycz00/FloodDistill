"""Summarise geographic, temporal and spectral properties of a GDACS download.

Reports event and acquisition distributions, scene/patch cloud estimates,
MNDWI water coverage and event-minus-baseline water. These are quality proxies,
not flood labels.

Usage: python datasets/flood_pool_quality.py --root $DATA_ROOT/flood_unlabeled
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

BANDS = ("B2", "B3", "B4", "B8A", "B11", "B12")


# small stats helpers

def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, q))


def _summary(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(float(np.mean(xs)), 4),
        "std": round(float(np.std(xs)), 4),
        "min": round(float(np.min(xs)), 4),
        "p10": round(_pct(xs, 10), 4),
        "median": round(_pct(xs, 50), 4),
        "p90": round(_pct(xs, 90), 4),
        "max": round(float(np.max(xs)), 4),
    }


def _shannon_evenness(counts: list[int]) -> float:
    """Normalised Shannon entropy, 1 = perfectly even, close to 0 = concentrated."""
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in ps)
    return round(h / math.log(len(counts)), 4)


def _continent(lon: float, lat: float) -> str:
    """Rough continent from lon/lat, only for a coverage overview."""
    if lat < -60:
        return "Antarctica"
    if -170 <= lon < -30 and lat >= 12:
        return "N.America"
    if -90 <= lon < -30 and lat < 12:
        return "S.America"
    if -30 <= lon < 60 and lat >= 35:
        return "Europe"
    if -20 <= lon < 55 and lat < 35:
        return "Africa"
    if 25 <= lon < 60 and lat >= 12:
        return "MiddleEast"
    if 60 <= lon < 150 and lat >= 5:
        return "Asia"
    if 90 <= lon < 180 and lat < 5:
        return "SE.Asia/Oceania"
    if lon >= 110 and lat < -10:
        return "Oceania"
    return "Other"


# load scene index from the pool

def _load_coords(root: Path) -> dict[tuple[str, int], tuple[float, float]]:
    coords: dict[tuple[str, int], tuple[float, float]] = {}
    ck = root / "checked_patches.csv"
    if not ck.exists():
        return coords
    with open(ck, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 5 or row[4].strip() != "1":
                continue
            try:
                coords[(row[0], int(row[1]))] = (float(row[2]), float(row[3]))
            except ValueError:
                continue
    return coords


def _scan_scenes(root: Path, coords) -> list[dict]:
    """One record per scene dir with a metadata.json."""
    scenes = []
    for loc in sorted(p for p in root.iterdir() if p.is_dir()):
        event_id, _, pidx = loc.name.rpartition("_")
        if not event_id or not pidx.isdigit():
            continue
        patch_idx = int(pidx)
        lon, lat = coords.get((event_id, patch_idx), (None, None))
        for sd in sorted(p for p in loc.iterdir() if p.is_dir()):
            meta = sd / "metadata.json"
            if not meta.exists():
                continue
            try:
                props = json.load(open(meta)).get("properties", {})
            except (json.JSONDecodeError, OSError):
                continue
            ts = props.get("system:time_start")
            dt = (datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                  if ts is not None else None)
            if (lon is None) and (sd / "B3.tif").exists():
                try:
                    with rasterio.open(sd / "B3.tif") as s:
                        b = s.bounds
                        lon, lat = (b.left + b.right) / 2, (b.bottom + b.top) / 2
                except rasterio.errors.RasterioIOError:
                    pass
            scenes.append({
                "event_id": event_id, "patch_idx": patch_idx,
                "location": loc.name, "scene_dir": sd,
                "lon": lon, "lat": lat,
                "mgrs": props.get("MGRS_TILE", ""),
                "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
                "spacecraft": props.get("SPACECRAFT_NAME", ""),
                "dt": dt,
            })
    return scenes


# flood-signal (MNDWI) computed offline from the GeoTIFFs

def _mndwi_water_frac(scene_dir: Path, thr: float = 0.0) -> float | None:
    """Water fraction with MNDWI = (B3 - B11)/(B3 + B11) > thr, on the 20 m grid (B3 averaged 2x2)."""
    try:
        with rasterio.open(scene_dir / "B3.tif") as s:
            g = s.read(1).astype("float32")
        with rasterio.open(scene_dir / "B11.tif") as s:
            sw = s.read(1).astype("float32")
    except (rasterio.errors.RasterioIOError, OSError):
        return None
    if g.shape != sw.shape:  # downsample the 10 m green to the 20 m SWIR grid
        h = (g.shape[0] // 2) * 2
        w = (g.shape[1] // 2) * 2
        g = g[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        g = g[:sw.shape[0], :sw.shape[1]]
        sw = sw[:g.shape[0], :g.shape[1]]
    denom = g + sw
    valid = denom > 0
    if not valid.any():
        return None
    mndwi = np.zeros_like(g)
    mndwi[valid] = (g[valid] - sw[valid]) / denom[valid]
    return float((mndwi > thr).mean())


def _flood_signal(scenes: list[dict], sample_n: int, thr: float, rng) -> dict:
    """Per-location dry->wet new-water where a baseline scene is present."""
    by_loc: dict[str, list[dict]] = defaultdict(list)
    for s in scenes:
        by_loc[s["location"]].append(s)

    locs = list(by_loc)
    if sample_n and len(locs) > sample_n:
        locs = [locs[i] for i in rng.choice(len(locs), sample_n, replace=False)]

    water_fracs, new_water, only_event = [], [], []
    n_paired = n_allwater = n_dry = 0
    for loc in locs:
        ss = sorted(by_loc[loc], key=lambda s: (s["dt"] or datetime.min.replace(tzinfo=timezone.utc)))
        fracs = [(_mndwi_water_frac(s["scene_dir"], thr)) for s in ss]
        fracs = [f for f in fracs if f is not None]
        if not fracs:
            continue
        event_frac = fracs[-1]          # latest scene ~ during/after event
        water_fracs.append(event_frac)
        if len(fracs) >= 2:
            baseline_frac = fracs[0]    # earliest ~ pre-event baseline
            nw = max(0.0, event_frac - baseline_frac)
            new_water.append(nw)
            n_paired += 1
            if min(fracs) > 0.6:
                n_allwater += 1         # already water in both -> low flood value
            if nw < 0.01 and event_frac < 0.05:
                n_dry += 1              # no visible water change -> weak signal
        else:
            only_event.append(event_frac)

    return {
        "locations_evaluated": len(locs),
        "event_water_frac": _summary(water_fracs),
        "paired_locations": n_paired,
        "new_water_frac_dry_to_wet": _summary(new_water),
        "pct_locations_all_water_low_value": round(100 * n_allwater / n_paired, 1) if n_paired else None,
        "pct_locations_no_flood_signal": round(100 * n_dry / n_paired, 1) if n_paired else None,
    }


# per-patch sidecar (measured at download time)

def _sidecar_quality(root: Path) -> dict | None:
    sc = root / "patch_quality.csv"
    if not sc.exists():
        return None
    cp, wf, pw, nw, tried = [], [], [], [], []
    with open(sc, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "1":
                continue
            for key, box in (("patch_cloud_prob", cp), ("water_frac", wf),
                             ("permanent_water_frac", pw), ("new_water_frac", nw),
                             ("candidates_tried", tried)):
                v = r.get(key, "")
                if v not in ("", "None", None):
                    try:
                        box.append(float(v))
                    except ValueError:
                        pass
    return {
        "patch_cloud_prob_pct": _summary(cp),
        "water_frac": _summary(wf),
        "permanent_water_frac": _summary(pw),
        "new_water_frac": _summary(nw),
        "candidates_tried": _summary(tried),
    }


# report

def assess(root: Path, flood_sample: int, mndwi_thr: float, seed: int) -> dict:
    coords = _load_coords(root)
    scenes = _scan_scenes(root, coords)
    rng = np.random.default_rng(seed)

    events = Counter(s["event_id"] for s in scenes)
    patches = {(s["event_id"], s["patch_idx"]) for s in scenes}
    tiles = Counter(s["mgrs"] for s in scenes if s["mgrs"])
    geo = [(s["lon"], s["lat"]) for s in scenes if s["lon"] is not None]

    cells = Counter((math.floor(lon), math.floor(lat)) for lon, lat in geo)
    continents = Counter(_continent(lon, lat) for lon, lat in geo)
    years = Counter(s["dt"].year for s in scenes if s["dt"])
    months = Counter(s["dt"].month for s in scenes if s["dt"])
    craft = Counter(s["spacecraft"] for s in scenes if s["spacecraft"])
    clouds = [float(s["cloud_pct"]) for s in scenes if s["cloud_pct"] is not None]
    patches_per_event = list(events.values())

    lons = [g[0] for g in geo]; lats = [g[1] for g in geo]

    report = {
        "root": str(root),
        "totals": {
            "scenes": len(scenes),
            "locations": len(patches),
            "unique_events": len(events),
            "unique_mgrs_tiles": len(tiles),
            "scenes_with_geo": len(geo),
        },
        "geographic_diversity": {
            "lon_range": [round(min(lons), 3), round(max(lons), 3)] if lons else None,
            "lat_range": [round(min(lats), 3), round(max(lats), 3)] if lats else None,
            "occupied_1deg_cells": len(cells),
            "cell_evenness_0to1": _shannon_evenness(list(cells.values())),
            "mgrs_tile_evenness_0to1": _shannon_evenness(list(tiles.values())),
            "top5_mgrs_tiles": tiles.most_common(5),
            "continent_scene_counts": dict(continents.most_common()),
        },
        "event_diversity": {
            "patches_per_event": _summary([float(x) for x in patches_per_event]),
            "event_evenness_0to1": _shannon_evenness(patches_per_event),
            "top5_events_by_scene": events.most_common(5),
        },
        "temporal_diversity": {
            "scenes_per_year": dict(sorted(years.items())),
            "scenes_per_month": dict(sorted(months.items())),
            "spacecraft": dict(craft),
        },
        "cloud_quality_tile_level_pct": {
            **_summary(clouds),
            "pct_under_10": round(100 * sum(c < 10 for c in clouds) / len(clouds), 1) if clouds else None,
            "pct_under_20": round(100 * sum(c < 20 for c in clouds) / len(clouds), 1) if clouds else None,
            "pct_over_50": round(100 * sum(c > 50 for c in clouds) / len(clouds), 1) if clouds else None,
        },
        "flood_signal_offline_mndwi": _flood_signal(scenes, flood_sample, mndwi_thr, rng),
    }
    side = _sidecar_quality(root)
    if side is not None:
        report["per_patch_measured"] = side
    return report


def _print(rep: dict) -> None:
    t = rep["totals"]; g = rep["geographic_diversity"]
    e = rep["event_diversity"]; c = rep["cloud_quality_tile_level_pct"]
    fs = rep["flood_signal_offline_mndwi"]
    print(f"\n{'='*70}\nFLOOD POOL QUALITY  —  {rep['root']}\n{'='*70}")
    print(f"scenes={t['scenes']}  locations={t['locations']}  "
          f"events={t['unique_events']}  MGRS_tiles={t['unique_mgrs_tiles']}")

    print(f"\n── Geographic diversity ──")
    print(f"  lon {g['lon_range']}  lat {g['lat_range']}")
    print(f"  occupied 1-deg cells: {g['occupied_1deg_cells']}  "
          f"(cell evenness {g['cell_evenness_0to1']}, tile evenness {g['mgrs_tile_evenness_0to1']})")
    print(f"  continents: {g['continent_scene_counts']}")
    print(f"  top tiles: {g['top5_mgrs_tiles']}")

    print(f"\n── Event diversity ──")
    print(f"  patches/event: {e['patches_per_event']}")
    print(f"  event evenness {e['event_evenness_0to1']}  "
          f"(1=uniform, low=few mega-events dominate)")
    print(f"  biggest events: {e['top5_events_by_scene']}")

    td = rep["temporal_diversity"]
    print(f"\n── Temporal diversity ──")
    print(f"  years:  {td['scenes_per_year']}")
    print(f"  months: {td['scenes_per_month']}")

    print(f"\n── Cloud quality (tile-level CLOUDY_PIXEL_PERCENTAGE) ──")
    print(f"  mean={c.get('mean')} median={c.get('median')} p90={c.get('p90')} max={c.get('max')}")
    print(f"  <10%: {c.get('pct_under_10')}%   <20%: {c.get('pct_under_20')}%   "
          f">50%: {c.get('pct_over_50')}%")

    print(f"\n── Flood signal (offline MNDWI, {fs['locations_evaluated']} locations) ──")
    print(f"  event-scene water frac: {fs['event_water_frac']}")
    print(f"  paired (baseline present): {fs['paired_locations']}")
    print(f"  new (dry->wet) water frac: {fs['new_water_frac_dry_to_wet']}")
    print(f"  low-value all-water patches: {fs['pct_locations_all_water_low_value']}%   "
          f"no-flood-signal patches: {fs['pct_locations_no_flood_signal']}%")

    if "per_patch_measured" in rep:
        v = rep["per_patch_measured"]
        print(f"\n── per-PATCH measured (download-time sidecar) ──")
        print(f"  patch cloud prob %: {v['patch_cloud_prob_pct']}")
        print(f"  permanent water frac: {v['permanent_water_frac']}")
        print(f"  new water frac: {v['new_water_frac']}")
        print(f"  candidates tried/patch: {v['candidates_tried']}")
    print(f"{'='*70}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Pool root directory to assess.")
    p.add_argument("--flood_sample", type=int, default=300,
                   help="Locations to sample for the (slow) offline MNDWI signal "
                        "(0 = all). Default 300.")
    p.add_argument("--mndwi_thr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="Also write the full report as JSON.")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    rep = assess(root, args.flood_sample, args.mndwi_thr, args.seed)
    _print(rep)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(out, "w"), indent=2, default=str)
        print(f"[quality] JSON report written to {out}")


if __name__ == "__main__":
    main()
