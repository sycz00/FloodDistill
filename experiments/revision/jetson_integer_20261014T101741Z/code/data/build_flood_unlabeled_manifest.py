"""Extract exact GEE scene identities from the raw GDACS download metadata.

The input tree is <root>/<event_id>_<patch_idx>/<scene_id>/ with band TIFFs
and metadata.json. checked_patches.csv supplies patch coordinates where present.
Pseudo-label filenames encode save order, not scene identity. Their source key
and source_manifest.csv link to this raw-scene manifest for exact re-downloads.

Usage: python -m data.build_flood_unlabeled_manifest --root $DATA_ROOT/flood_unlabeled --out data/flood_unlabeled_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import rasterio

from utils.paths import dataset_root

FIELDS = [
    "event_id", "patch_idx", "lon", "lat",
    "scene_system_index", "gee_asset_id", "product_id",
    "mgrs_tile", "granule_id", "datatake_identifier", "spacecraft_name",
    "acquisition_datetime_utc", "cloudy_pixel_percentage",
    "local_path",
]


def _load_patch_coords(checked_csv: Path) -> dict[tuple[str, int], tuple[float, float]]:
    coords: dict[tuple[str, int], tuple[float, float]] = {}
    with open(checked_csv, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            event_id, patch_idx, lon, lat, status = row
            if status.strip() != "1":
                continue
            if not lon.strip() or not lat.strip():
                continue  # no centre stored; build_manifest falls back to the GeoTIFF bounds
            coords[(event_id, int(patch_idx))] = (float(lon), float(lat))
    return coords


def _center_from_tif(scene_dir: Path) -> tuple[float, float] | tuple[None, None]:
    """Approximate patch centre (lon, lat) from the GeoTIFF bounds (written in lon/lat by gee_download.py).

    Used for scenes that were downloaded but are missing in checked_patches.csv.
    """
    for band in ("B2", "B3", "B4", "B8A", "B11", "B12"):
        tif = scene_dir / f"{band}.tif"
        if not tif.exists():
            continue
        try:
            with rasterio.open(tif) as src:
                b = src.bounds
                return ((b.left + b.right) / 2, (b.bottom + b.top) / 2)
        except rasterio.errors.RasterioIOError:
            continue
    return (None, None)


def build_manifest(root: str, out_path: str) -> None:
    root_p = Path(root).expanduser()
    checked_csv = root_p / "checked_patches.csv"
    coords = _load_patch_coords(checked_csv) if checked_csv.exists() else {}
    print(f"[manifest] loaded {len(coords)} patch coordinates from {checked_csv}")

    rows: list[dict] = []
    n_missing_coords = 0
    n_recovered_coords = 0
    n_missing_meta = 0
    n_unrecoverable = 0

    location_dirs = sorted(p for p in root_p.iterdir() if p.is_dir())
    for loc_dir in location_dirs:
        event_id, _, patch_idx_str = loc_dir.name.rpartition("_")
        if not event_id or not patch_idx_str.isdigit():
            continue
        patch_idx = int(patch_idx_str)
        lon, lat = coords.get((event_id, patch_idx), (None, None))
        if lon is None:
            n_missing_coords += 1

        for scene_dir in sorted(p for p in loc_dir.iterdir() if p.is_dir()):
            meta_path = scene_dir / "metadata.json"
            if not meta_path.exists():
                n_missing_meta += 1
                continue

            scene_lon, scene_lat = lon, lat
            if scene_lon is None:
                # no row in checked_patches.csv (e.g. killed between _save_patch()
                # and the CSV write), use the GeoTIFF bounds instead of dropping it
                scene_lon, scene_lat = _center_from_tif(scene_dir)
                if scene_lon is not None:
                    n_recovered_coords += 1
                else:
                    n_unrecoverable += 1
                    continue

            with open(meta_path) as f:
                meta = json.load(f)
            props = meta.get("properties", {})
            ts_ms = props.get("system:time_start")
            acq_dt = (
                datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
                if ts_ms is not None else ""
            )
            rows.append({
                "event_id": event_id,
                "patch_idx": patch_idx,
                "lon": scene_lon,
                "lat": scene_lat,
                "scene_system_index": props.get("system:index", scene_dir.name),
                "gee_asset_id": meta.get("id", ""),
                "product_id": props.get("PRODUCT_ID", ""),
                "mgrs_tile": props.get("MGRS_TILE", ""),
                "granule_id": props.get("GRANULE_ID", ""),
                "datatake_identifier": props.get("DATATAKE_IDENTIFIER", ""),
                "spacecraft_name": props.get("SPACECRAFT_NAME", ""),
                "acquisition_datetime_utc": acq_dt,
                "cloudy_pixel_percentage": props.get("CLOUDY_PIXEL_PERCENTAGE", ""),
                "local_path": str(scene_dir.relative_to(root_p)),
            })

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[manifest] {len(location_dirs)} locations scanned, {len(rows)} scenes written")
    print(f"[manifest] {n_missing_coords} locations missing patch coords in checked_patches.csv")
    if n_missing_coords:
        print(f"[manifest]   -> {n_recovered_coords} recovered from GeoTIFF bounds, "
              f"{n_unrecoverable} scenes unrecoverable (no tif + no CSV row, skipped)")
    print(f"[manifest] {n_missing_meta} scene dirs missing metadata.json (skipped)")
    print(f"[manifest] saved {out_p}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(dataset_root("UNLABELED_DIR")),
                   help="Raw flood-event download root (default $DATA_ROOT/flood_unlabeled)")
    p.add_argument("--out", default="data/flood_unlabeled_manifest.csv",
                   help="Output manifest CSV path")
    args = p.parse_args()
    build_manifest(args.root, args.out)


if __name__ == "__main__":
    main()
