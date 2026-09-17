"""Fetch Sentinel-2 scenes by the GEE asset IDs in a raw-scene manifest.

The manifest comes from data/build_flood_unlabeled_manifest.py. Required
columns: event_id, patch_idx, lon, lat, gee_asset_id, local_path.
Asset IDs avoid repeating the date/cloud search. Verify downloaded bytes and
grids against the original data before claiming an exact reproduction.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from multiprocessing.dummy import Pool

import ee
import numpy as np

from gee_download import (
    BAND_CROPS,
    DOWNLOAD_BANDS,
    PATCH_RADIUS_M,
    _center_crop,
    _save_patch,
)


def _download_exact(asset_id: str, coords: list[float], bands: list[str],
                     crops: dict[str, tuple]) -> dict:
    """Fetch one scene by exact GEE asset id (no date/cloud search)."""
    image  = ee.Image(asset_id)
    region = ee.Geometry.Point(coords).buffer(PATCH_RADIUS_M).bounds()
    patch  = image.select(*bands).sampleRectangle(region, defaultValue=0)
    feats  = patch.getInfo()

    raster = OrderedDict()
    for band in bands:
        arr = np.atleast_3d(feats["properties"][band]).astype("uint16")
        arr = _center_crop(arr, crops[band])
        raster[band] = arr

    raw_coords = np.array(feats["geometry"]["coordinates"][0])
    bbox = [
        [raw_coords[:, 0].min(), raw_coords[:, 1].max()],
        [raw_coords[:, 0].max(), raw_coords[:, 1].min()],
    ]
    ref_band = bands[0]
    old_size = (
        len(feats["properties"][ref_band]),
        len(feats["properties"][ref_band][0]),
    )
    new_size = raster[ref_band].shape[:2]
    from gee_download import _adjust_coords
    bbox = _adjust_coords(bbox, old_size, new_size)

    return {"raster": raster, "coords": bbox, "metadata": image.getInfo()}


def _worker(task: tuple) -> tuple:
    row, save_path = task
    location_dir = os.path.join(save_path, f"{row['event_id']}_{int(row['patch_idx']):04d}")
    scene_dir = os.path.join(location_dir, row["scene_system_index"])

    if os.path.isdir(scene_dir) and os.path.exists(os.path.join(scene_dir, "metadata.json")):
        return (row["gee_asset_id"], "skipped")

    try:
        coords = [float(row["lon"]), float(row["lat"])]
        patch = _download_exact(row["gee_asset_id"], coords, DOWNLOAD_BANDS, BAND_CROPS)
        os.makedirs(location_dir, exist_ok=True)
        _save_patch(patch["raster"], patch["coords"], patch["metadata"], location_dir)
        return (row["gee_asset_id"], "ok")
    except Exception as e:
        return (row["gee_asset_id"], f"failed: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="Path to flood_unlabeled_manifest.csv")
    p.add_argument("--save_path", required=True, help="Output root directory")
    p.add_argument("--project", required=True, help="Google Cloud project ID with Earth Engine API enabled")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    ee.Initialize(project=args.project)
    print(f"[gee] initialised  project={args.project}")

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[redownload] {len(rows)} scenes in manifest")

    os.makedirs(args.save_path, exist_ok=True)
    tasks = [(row, args.save_path) for row in rows]

    n_ok = n_skip = n_fail = 0
    with Pool(args.num_workers) as pool:
        for i, (asset_id, status) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  [fail] {asset_id}: {status}")
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}]  ok={n_ok}  skipped={n_skip}  failed={n_fail}")

    print(f"\n[done] ok={n_ok}  skipped={n_skip}  failed={n_fail}")
    print(f"[done] data saved to: {args.save_path}")


if __name__ == "__main__":
    main()
