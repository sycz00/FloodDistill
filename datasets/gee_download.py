"""Google Earth Engine primitives for flood-event Sentinel-2 L1C downloads.

Shared library behind datasets/download_flood_events.py (the pool builder). Holds the
event sources (GDACS alert API, CSV, the built-in seed list), the S2 L1C collection
query, patch sampling and the GeoTIFF writer.

Output structure:
    <save_path>/
        <event_id>_<patch_idx>/    location directory
            <scene_system_id>/     date-stamped scene directory
                B2.tif ... B12.tif
                metadata.json
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
from multiprocessing.dummy import Lock

warnings.simplefilter("ignore", UserWarning)

import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine

# B8A (narrow NIR, 20 m) rather than B8 (broad NIR, 10 m) to match the 6-band
# set used when fine-tuning Prithvi on Sen1Floods11.
DOWNLOAD_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]

# crop size per native resolution, for a 1320 m radius buffer
_CROP_10M = (264, 264)   # B2 B3 B4
_CROP_20M = (132, 132)   # B8A B11 B12

BAND_CROPS = {
    "B2":  _CROP_10M,
    "B3":  _CROP_10M,
    "B4":  _CROP_10M,
    "B8A": _CROP_20M,
    "B11": _CROP_20M,
    "B12": _CROP_20M,
}

PATCH_RADIUS_M = 1320   # 264 px x 10 m / 2; the largest window the EE request
                        # budget allowed for every event in the catalog


# Flood event catalogue

@dataclass
class FloodEvent:
    event_id:    str
    lon:         float   # event centre longitude
    lat:         float   # event centre latitude
    start_date:  str     # ISO YYYY-MM-DD (inclusive)
    end_date:    str     # ISO YYYY-MM-DD (inclusive)
    description: str = ""


# Built-in list of major flood events 2017-2023 on all continents. Coordinates are
# rough centroids, patches get sampled around them anyway.

# Original 15 events (Copernicus EMS / known major events):
SEED_EVENTS: list[FloodEvent] = [
    FloodEvent("EMSR271_FARKADONA",    22.07,  39.48, "2018-02-20", "2018-03-10", "Greece Karditsa riverine flood"),
    FloodEvent("EMSR279_ZARAGOZA",     -0.92,  41.65, "2018-04-08", "2018-04-22", "Spain Ebro flood"),
    FloodEvent("EMSR333_SICILY",       14.60,  37.20, "2018-11-01", "2018-11-15", "Italy Sicily flash flood"),
    FloodEvent("EMSR342_EASTMIDLANDS", -1.10,  52.90, "2019-11-08", "2019-11-22", "UK East Midlands flood"),
    FloodEvent("EMSR347_MALAWI",       35.30, -14.80, "2019-03-04", "2019-03-18", "Malawi/Mozambique Idai flood"),
    FloodEvent("EMSR479_GREECE",       22.40,  38.90, "2020-09-18", "2020-10-02", "Greece Karditsa 2020 flood"),
    FloodEvent("EMSR517_GERMANY",       7.00,  50.50, "2021-07-13", "2021-07-27", "Germany Ahr valley flood"),
    FloodEvent("EMSR547_BCCANDA",    -121.50,  49.00, "2021-11-14", "2021-11-28", "British Columbia floods"),
    FloodEvent("KERALA2018",           76.50,  10.00, "2018-08-08", "2018-08-22", "India Kerala monsoon flood"),
    FloodEvent("PAKISTAN2022",         68.00,  26.00, "2022-08-20", "2022-09-10", "Pakistan Sindh monsoon flood"),
    FloodEvent("BANGLADESH2022",       91.90,  24.90, "2022-06-15", "2022-06-30", "Bangladesh Sylhet monsoon flood"),
    FloodEvent("NIGERIA2022",           6.80,   6.10, "2022-10-01", "2022-10-20", "Nigeria Anambra riverine flood"),
    FloodEvent("INDONESIA2021",       114.00,  -0.50, "2021-01-08", "2021-01-22", "Kalimantan Indonesia flood"),
    FloodEvent("BRAZIL2021",          -40.00, -14.00, "2021-12-20", "2022-01-05", "Brazil Bahia flood"),
    FloodEvent("AUSTRALIA2022",       152.50, -27.50, "2022-02-26", "2022-03-10", "Australia Queensland flood"),

    # 17 more events for geographic and flood-type diversity, mostly riverine and flash
    # floods in underrepresented regions. All after the Sentinel-2 launch (Dec 2015).

    # East Asia - riverine & flash
    FloodEvent("JAPAN2018_MABI",       133.48,  34.67, "2018-07-06", "2018-07-18", "Japan Okayama-Hiroshima riverine flood"),
    FloodEvent("LAOS2018_ATTAPEU",     107.52,  15.01, "2018-07-23", "2018-08-05", "Laos Attapeu dam-breach flood"),
    FloodEvent("JAPAN2019_HAGIBIS",    138.90,  35.80, "2019-10-12", "2019-10-20", "Japan Typhoon Hagibis Chikuma River"),
    FloodEvent("CHINA2020_YANGTZE",    116.00,  29.50, "2020-06-15", "2020-09-01", "China Yangtze-Poyang Lake 2020 flood"),
    FloodEvent("JAPAN2020_KUMAMOTO",   130.72,  32.52, "2020-07-04", "2020-07-14", "Japan Kumamoto Kuma River flood"),
    FloodEvent("CHINA2021_ZHENGZHOU",  113.65,  34.75, "2021-07-20", "2021-07-31", "China Zhengzhou flash flood"),

    # Southeast Asia - monsoon & riverine
    FloodEvent("VIETNAM2020_CENTRAL",  107.50,  15.50, "2020-10-05", "2020-11-15", "Vietnam central highlands monsoon flood"),
    FloodEvent("PHILIPPINES2021_RAI",  124.00,   9.80, "2021-12-16", "2021-12-25", "Philippines Typhoon Odette/Rai flood"),

    # Middle East & Central Asia - flash floods
    FloodEvent("IRAN2019_KHUZESTAN",    48.52,  31.50, "2019-03-19", "2019-04-10", "Iran Khuzestan flash flood"),
    FloodEvent("AFGHANISTAN2022",       65.00,  34.50, "2022-03-25", "2022-04-10", "Afghanistan spring flash floods"),

    # Africa - riverine & pluvial
    FloodEvent("SUDAN2020_KHARTOUM",    32.56,  15.55, "2020-08-01", "2020-09-15", "Sudan Khartoum Nile riverine flood"),
    FloodEvent("SOUTHAFRICA2022_KZN",   30.90, -29.87, "2022-04-10", "2022-04-22", "South Africa KwaZulu-Natal flood"),
    FloodEvent("LIBYA2023_DERNA",       22.64,  32.77, "2023-09-10", "2023-09-20", "Libya Derna flash flood Storm Daniel"),

    # Americas - flash & riverine
    FloodEvent("PERU2017_ELNINO",      -77.50,  -8.50, "2017-03-01", "2017-03-25", "Peru El Nino coastal-riverine flood"),
    FloodEvent("TENNESSEE2021",        -87.63,  36.07, "2021-08-21", "2021-08-26", "USA Tennessee Humphreys flash flood"),
    FloodEvent("BRAZIL2022_PETROPOLIS",-43.18, -22.51, "2022-02-15", "2022-02-26", "Brazil Petropolis flash flood"),

    # Europe - riverine
    FloodEvent("ITALY2023_EMILIA",      11.90,  44.30, "2023-05-15", "2023-05-31", "Italy Emilia-Romagna riverine flood"),
    FloodEvent("SLOVENIA2023",          14.52,  46.17, "2023-08-04", "2023-08-16", "Slovenia 2023 historic flood"),
]


def load_events_csv(path: str) -> list[FloodEvent]:
    """Load flood events from a CSV with columns:
            event_id, lon, lat, start_date, end_date[, description]
    """
    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(FloodEvent(
                event_id    = row["event_id"].strip(),
                lon         = float(row["lon"]),
                lat         = float(row["lat"]),
                start_date  = row["start_date"].strip(),
                end_date    = row["end_date"].strip(),
                description = row.get("description", "").strip(),
            ))
    print(f"[events] loaded {len(events)} events from {path}")
    return events


def fetch_gdacs_events(
    from_date:   str,
    to_date:     str,
    alert_level: str = "orange;red",
    limit:       int = 2000,
) -> list[FloodEvent]:
    """Flood events from the public GDACS API (no auth needed).

        GDACS v2 SEARCH endpoint, paged with 100 events per page. Dates are ISO strings
        like "2017-01-01". alert_level is semicolon separated ("green", "orange", "red"),
        orange + red by default. limit caps the total over all pages.
        Returns FloodEvents with centroid and date window.
    """
    import urllib.request

    base = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
    events: list[FloodEvent] = []
    page = 1

    while len(events) < limit:
        url = (
            f"{base}?eventlist=FL"
            f"&fromdate={from_date}&todate={to_date}"
            f"&alertlevel={alert_level}&pagenumber={page}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "climatecompression/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
            data = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            break   # empty or non-JSON body, so no more pages
        except Exception as e:
            raise RuntimeError(f"GDACS fetch failed (page {page}): {e}") from e

        features = data.get("features", [])
        if not features:
            break   # no more pages

        for feat in features:
            if len(events) >= limit:
                break
            props = feat.get("properties", {})
            geom  = feat.get("geometry", {})
            if geom.get("type") != "Point":
                continue
            lon, lat = geom["coordinates"][:2]
            raw_from = (props.get("fromdate") or "")[:10]
            raw_to   = (props.get("todate")   or "")[:10]
            if not raw_from:
                continue
            end = raw_to if raw_to else raw_from
            eid = f"GDACS_{props.get('eventid', 'UNK')}_{props.get('episodeid', '0')}"
            events.append(FloodEvent(
                event_id    = eid,
                lon         = lon,
                lat         = lat,
                start_date  = raw_from,
                end_date    = end,
                description = f"{props.get('country', '')} ({props.get('alertlevel', '')})",
            ))

        print(f"[gdacs] page {page}: +{len(features)} events  total so far: {len(events)}")
        page += 1

    print(f"[gdacs] fetched {len(events)} flood events ({from_date} → {to_date})")
    return events


# GEE helpers

def _get_s2_l1c_collection(cloud_pct: float) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )


def _center_crop(img: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    h, w  = img.shape[:2]
    oh, ow = out_size
    t = int((h - oh + 1) * 0.5)
    l = int((w - ow + 1) * 0.5)
    return img[t:t+oh, l:l+ow]


def _adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + xoff * xres,         coords[0][1] - yoff * yres],
        [coords[0][0] + (xoff + new_size[1]) * xres,
         coords[0][1] - (yoff + new_size[0]) * yres],
    ]


def _download_patch_s2(
    collection: ee.ImageCollection,
    coords: list[float],
    radius: int,
    bands: list[str],
    crops: dict[str, tuple],
) -> dict:
    """Download a single S2 patch from a filtered collection.

        Returns a dict with raster (OrderedDict of band arrays), coords and metadata.
        Raises ee.EEException / urllib3.exceptions.HTTPError on failure.
    """
    # use the least cloudy image in the collection
    image  = collection.sort("CLOUDY_PIXEL_PERCENTAGE", True).first()
    region = ee.Geometry.Point(coords).buffer(radius).bounds()
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
    # Adjust coords after crop
    ref_band  = bands[0]
    old_size  = (
        len(feats["properties"][ref_band]),
        len(feats["properties"][ref_band][0]),
    )
    new_size  = raster[ref_band].shape[:2]
    bbox      = _adjust_coords(bbox, old_size, new_size)

    return {"raster": raster, "coords": bbox, "metadata": image.getInfo()}


def _save_geotiff(img: np.ndarray, coords: list, path: str) -> None:
    h, w, c = img.shape
    xres = (coords[1][0] - coords[0][0]) / w
    yres = (coords[0][1] - coords[1][1]) / h
    transform = (
        Affine.translation(coords[0][0] - xres / 2, coords[0][1] + yres / 2)
        * Affine.scale(xres, -yres)
    )
    profile = {
        "driver": "GTiff", "width": w, "height": h, "count": c,
        "crs": "+proj=latlong", "transform": transform,
        "dtype": img.dtype, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as f:
        f.write(img.transpose(2, 0, 1))


def _save_patch(
    raster:   OrderedDict,
    coords:   list,
    metadata: dict,
    location_dir: str,
) -> None:
    """Save all bands + metadata.json.  Directory is <location_dir>/<scene_id>/."""
    scene_id   = metadata["properties"]["system:index"]
    scene_path = os.path.join(location_dir, scene_id)
    os.makedirs(scene_path, exist_ok=True)

    for band, arr in raster.items():
        _save_geotiff(arr, coords, os.path.join(scene_path, f"{band}.tif"))

    with open(os.path.join(scene_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


# Per-patch worker

_lock = Lock()

def _worker(task: tuple) -> tuple:
    """Download one event-patch.  Returns (event_id, patch_idx, lon, lat, status)."""
    event, patch_idx, lon, lat, save_path, cloud_pct, cloud_pct_fallback, add_baseline = task

    location_id  = f"{event.event_id}_{patch_idx:04d}"
    location_dir = os.path.join(save_path, location_id)

    # Skip if already downloaded (directory exists with at least one scene)
    if os.path.isdir(location_dir) and any(
        os.path.isdir(os.path.join(location_dir, d))
        for d in os.listdir(location_dir)
    ):
        return (event.event_id, patch_idx, lon, lat, 1)

    coords = [lon, lat]

    def _try_download(start: str, end: str, cloud_limit: float) -> bool:
        """Try to download a patch for a given date range + cloud limit."""
        col = (
            _get_s2_l1c_collection(cloud_limit)
            .filterDate(start, end)
            .filterBounds(ee.Geometry.Point(coords))
        )
        try:
            n = col.size().getInfo()
        except (ee.EEException, urllib3.exceptions.HTTPError):
            return False
        if n == 0:
            return False
        try:
            patch = _download_patch_s2(col, coords, PATCH_RADIUS_M, DOWNLOAD_BANDS, BAND_CROPS)
            os.makedirs(location_dir, exist_ok=True)
            _save_patch(patch["raster"], patch["coords"], patch["metadata"], location_dir)
            return True
        except (ee.EEException, urllib3.exceptions.HTTPError, Exception):
            return False

    success = False
    # Primary: event window at primary cloud threshold
    success = _try_download(event.start_date, event.end_date, cloud_pct)

    # Fallback: widen cloud threshold
    if not success and cloud_pct_fallback > cloud_pct:
        success = _try_download(event.start_date, event.end_date, cloud_pct_fallback)

    # Fallback: extend window by 7 days on each side (S2 revisit ~5 days)
    if not success:
        start_ext = (date.fromisoformat(event.start_date) - timedelta(days=7)).isoformat()
        end_ext   = (date.fromisoformat(event.end_date)   + timedelta(days=7)).isoformat()
        success = _try_download(start_ext, end_ext, cloud_pct_fallback)

    if success and add_baseline:
        # Optional: one pre-event clear-sky image (same location, ~1 year before)
        pre_start = (date.fromisoformat(event.start_date) - timedelta(days=395)).isoformat()
        pre_end   = (date.fromisoformat(event.start_date) - timedelta(days=30)).isoformat()
        # Use a stricter cloud threshold for the baseline (we want a clear reference)
        _try_download(pre_start, pre_end, 20.0)
        # still count as success if the baseline fails

    return (event.event_id, patch_idx, lon, lat, 1 if success else 0)


# Spatial sampling within event area

def _sample_patch_coords(
    event: FloodEvent,
    n_patches: int,
    radius_km: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Sample n_patches random (lon, lat) within radius_km of event centre."""
    coords = []
    deg_per_km = 1.0 / 111.0   # rough conversion
    spread     = radius_km * deg_per_km

    while len(coords) < n_patches:
        dlon = rng.uniform(-spread, spread)
        dlat = rng.uniform(-spread, spread)
        if dlon ** 2 + dlat ** 2 <= spread ** 2:  # keep within circle
            coords.append((event.lon + dlon, event.lat + dlat))
    return coords
