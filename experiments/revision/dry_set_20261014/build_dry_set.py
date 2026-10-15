"""Independent dry-tile evaluation set (P5, approved 14 Oct 2026).

Selection uses ONLY static or scene-metadata maps, never a water index or a model output:
  JRC GSW v1.4 max_extent (never water in tile + 100 m), ESA WorldCover v200 land cover,
  Copernicus GLO-30 DEM slope, Sentinel-2 L2A scene classification (only for the snow and
  cloud-shadow stress strata), Sentinel-2 cloud probability (only to reject cloudy tiles).
Tiles are downloaded with the GDACS pool code (datasets/gee_download.py: S2_HARMONIZED L1C,
264 px at 10 m, 20 m bands at 132 px) so the input contract equals the training pool.
A human then confirms each tile visually (render phase -> rating page). Only confirmed-dry tiles are scored.

Phases:  select   -> selection.json (frozen before download)
         download -> /data1/schmalstieg/flooddistill/data/dry_set_v1/<tile_id>/<scene>/B*.tif
         render   -> MajorRev/dry_rating_20261014/ (quicklooks, rating.html, ratings_template.csv)
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/schmalstieg/repos/FloodDistill")
sys.path.insert(0, str(REPO))
OUT_DATA = Path("/data1/schmalstieg/flooddistill/data/dry_set_v1")
OUT_RATE = Path("/home/schmalstieg/repos/MajorRev/dry_rating_20261014")
PROJECT = "gdacs-502709"
TARGET_PER_STRATUM = 18          # oversampled; >= 15 per stratum expected to survive visual review
SEED = 20261014

STRATA = {
    "desert_bare":      dict(wc=[60], frac=0.6, desc="bare / sparse vegetation (deserts, dry plains)"),
    "urban":            dict(wc=[50], frac=0.3, desc="built-up areas (>= 30 % of the tile built-up)"),
    "cropland":         dict(wc=[40], frac=0.6, desc="cropland"),
    "forest":           dict(wc=[10], frac=0.7, desc="tree cover"),
    "grass_shrub":      dict(wc=[20, 30], frac=0.6, desc="grassland and shrubland"),
    "dry_floodplain":   dict(wc=[20, 30, 40], frac=0.6, near_water_m=3000,
                             desc="dry land within 3 km of mapped surface water (flood-plain context)"),
    "mountain_shadow":  dict(wc=[10, 20, 30, 60], frac=0.6, slope=20, sza=55,
                             desc="steep terrain imaged at low sun (terrain shadow)"),
    "snow":             dict(wc=[10, 20, 30, 60, 70], frac=0.6, snow=0.3,
                             desc="snow- or ice-covered land (L2A scene class snow >= 30 %)"),
    "cloud_shadow":     dict(wc=[10, 20, 30, 40, 50, 60], frac=0.6, shadow=0.05, cloudmax=0.4,
                             desc="land with cloud shadows (L2A scene class shadow >= 5 %)"),
}


def ee_init():
    import ee
    ee.Initialize(project=PROJECT)
    return ee


def candidates(ee, name, cfg, n, seed):
    """Random land points drawn per WorldCover class of the stratum (WorldCover is masked over the ocean).
    Mountain/snow strata additionally require slope >= 12 deg at 1 km (coarse) or snow/ice / >= 1,500 m."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    cls = wc
    if "slope" in cfg or name == "snow":
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().setDefaultProjection("EPSG:3857", None, 30)
        if "slope" in cfg:
            cls = cls.updateMask(ee.Terrain.slope(dem).gte(12))
        if name == "snow":
            cls = cls.updateMask(wc.eq(70).Or(dem.gte(1500)))
    per = int(math.ceil(n / len(cfg["wc"])))
    pts = cls.rename("c").stratifiedSample(numPoints=0, classBand="c", region=ee.Geometry.BBox(-180, -56, 180, 72),
                                           scale=1000, seed=seed, classValues=cfg["wc"], classPoints=[per] * len(cfg["wc"]),
                                           geometries=True, tileScale=16)
    return pts.randomColumn("r", seed).sort("r").limit(n)


def evaluate(ee, cfg, name, pts):
    """Tile-level checks for every candidate, server side; one dict per candidate."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().setDefaultProjection("EPSG:3857", None, 30)
    slope = ee.Terrain.slope(dem)
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    good = ee.List(cfg["wc"])

    def one(f):
        g = f.geometry()
        tile = g.buffer(1320).bounds()
        buf = g.buffer(1420).bounds()
        frac_target = wc.remap(good, ee.List.repeat(1, good.length()), 0).reduceRegion(
            ee.Reducer.mean(), tile, 10, maxPixels=1e8).get("remapped")
        wet_wc = wc.remap([80, 90, 95], [1, 1, 1], 0).reduceRegion(ee.Reducer.max(), tile, 10, maxPixels=1e8).get("remapped")
        jrc_tile = jrc.reduceRegion(ee.Reducer.max(), buf, 30, maxPixels=1e8).get("max_extent")
        props = dict(frac_target=frac_target, wet_wc=wet_wc, jrc_tile=jrc_tile,
                     slope_mean=slope.reduceRegion(ee.Reducer.mean(), tile, 30, maxPixels=1e8).get("slope"))
        if "near_water_m" in cfg:
            props["jrc_near"] = jrc.reduceRegion(ee.Reducer.max(), g.buffer(cfg["near_water_m"]).bounds(), 30,
                                                 maxPixels=1e9).get("max_extent")
        col = s2.filterBounds(g).filterDate("2019-01-01", "2026-01-01")
        if name == "cloud_shadow":
            col = col.filter(ee.Filter.rangeContains("CLOUDY_PIXEL_PERCENTAGE", 10, 50))
        elif name == "snow":
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        else:
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        if "sza" in cfg:
            col = col.filter(ee.Filter.gte("MEAN_SOLAR_ZENITH_ANGLE", cfg["sza"]))
        col = col.sort("CLOUDY_PIXEL_PERCENTAGE").limit(4)

        def stats(img):
            idx = img.get("system:index")
            scl_c = sr.filter(ee.Filter.eq("system:index", idx))
            cp_c = prob.filter(ee.Filter.eq("system:index", idx))
            scl = ee.Image(ee.Algorithms.If(scl_c.size().gt(0), ee.Image(scl_c.first()).select("SCL"),
                                            ee.Image.constant(0).rename("SCL")))
            cp = ee.Image(ee.Algorithms.If(cp_c.size().gt(0), ee.Image(cp_c.first()).select("probability"),
                                           ee.Image.constant(100).rename("probability")))
            fr = lambda m: m.reduceRegion(ee.Reducer.mean(), tile, 20, maxPixels=1e8).values().get(0)
            return ee.Feature(None, dict(
                idx=idx, cloudy=img.get("CLOUDY_PIXEL_PERCENTAGE"), sza=img.get("MEAN_SOLAR_ZENITH_ANGLE"),
                date=ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
                has_scl=scl_c.size(), snow=fr(scl.eq(11)), shadow=fr(scl.eq(3)),
                cloud=fr(scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))),
                cloudprob=cp.reduceRegion(ee.Reducer.mean(), tile, 20, maxPixels=1e8).get("probability")))
        props["scenes"] = ee.FeatureCollection(col.map(stats)).toList(4).map(lambda x: ee.Feature(x).toDictionary())
        return ee.Feature(g, props)
    return pts.map(one)


def accept(name, cfg, p):
    if p.get("frac_target") is None or p["frac_target"] < cfg["frac"]:
        return None, "land cover"
    if p.get("wet_wc") or p.get("jrc_tile"):
        return None, "mapped water/wetland in tile"
    if "near_water_m" in cfg and not p.get("jrc_near"):
        return None, "no water within 3 km"
    if "slope" in cfg and (p.get("slope_mean") or 0) < cfg["slope"]:
        return None, "slope"
    for sc in p.get("scenes") or []:
        if name == "snow":
            if sc.get("has_scl") and (sc.get("snow") or 0) >= cfg["snow"] and (sc.get("cloud") or 1) <= 0.2:
                return sc, "ok"
        elif name == "cloud_shadow":
            if sc.get("has_scl") and (sc.get("shadow") or 0) >= cfg["shadow"] and (sc.get("cloud") or 1) <= cfg["cloudmax"]:
                return sc, "ok"
        else:
            if sc.get("cloudprob") is not None and sc["cloudprob"] <= 10:
                return sc, "ok"
    return None, "no suitable scene"


def haversine_km(a, b):
    lo1, la1, lo2, la2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def select(args):
    ee = ee_init()
    gdacs = [(float(r["lon"]), float(r["lat"])) for r in csv.DictReader(
        open("/data1/schmalstieg/flooddistill/data/flood_unlabeled_v2/patch_quality_v2.csv")) if r["status"] == "1"]
    cells, chosen, log = set(), [], []
    rng = random.Random(SEED)
    for si, (name, cfg) in enumerate(STRATA.items()):
        if args.strata and name not in args.strata.split(","):
            continue
        got = []
        for rnd in range(args.rounds):
            pts = candidates(ee, name, cfg, args.batch, SEED + 1000 * si + rnd)
            try:
                plist = pts.toList(args.batch)
                npts = plist.size().getInfo()
            except Exception as exc:
                log.append(dict(stratum=name, round=rnd, error=str(exc)[:300])); print(name, rnd, "error", str(exc)[:200]); continue
            feats = []
            for start in range(0, npts, 6):          # small requests: Earth Engine per-request memory limit
                for size in (6, 3):
                    ok = True
                    for s0 in range(start, min(start + 6, npts), size):
                        chunk = ee.FeatureCollection(plist.slice(s0, min(s0 + size, start + 6, npts)))
                        try:
                            feats += evaluate(ee, cfg, name, chunk).getInfo()["features"]
                        except Exception as exc:
                            ok = False; print(name, rnd, s0, "chunk error", str(exc)[:120], flush=True)
                            if size == 3:
                                log.append(dict(stratum=name, round=rnd, chunk=s0, error=str(exc)[:300]))
                            else:
                                break
                    if ok or size == 3:
                        break
            rng.shuffle(feats)
            reasons = {}
            for f in feats:
                lon, lat = f["geometry"]["coordinates"]
                p = f["properties"]
                sc, why = accept(name, cfg, p)
                cell = (int(math.floor(lon / 2)), int(math.floor(lat / 2)))
                if sc is not None and cell in cells:
                    why, sc = "2-degree cell already used", None
                if sc is not None and any(haversine_km((lon, lat), g) < 25 for g in gdacs):
                    why, sc = "within 25 km of a GDACS pool tile", None
                log.append(dict(stratum=name, lon=lon, lat=lat, decision=why))
                reasons[why] = reasons.get(why, 0) + 1
                if sc is None:
                    continue
                cells.add(cell)
                got.append(dict(tile_id=f"{name}_{len(got):02d}", stratum=name, lon=lon, lat=lat, scene=sc["idx"],
                                date=sc["date"], scene_cloudy_pct=sc["cloudy"], sza=sc["sza"],
                                frac_target_landcover=p["frac_target"], slope_mean=p.get("slope_mean"),
                                scl_snow=sc.get("snow"), scl_shadow=sc.get("shadow"), scl_cloud=sc.get("cloud"),
                                cloudprob=sc.get("cloudprob")))
                if len(got) >= args.target:
                    break
            print(f"{name}: round {rnd}, {len(feats)} evaluated, {len(got)} accepted, reasons {reasons}", flush=True)
            if len(got) >= args.target:
                break
        chosen += got
    out = dict(created=time.time(), seed=SEED, criteria=STRATA, project=PROJECT,
               rule="static maps only; no water index or model output; >= 25 km from GDACS pool; one tile per 2-degree cell",
               tiles=chosen)
    path = HERE / ("selection.json" if not args.strata else f"selection_{args.strata.replace(',', '_')}.json")
    path.write_text(json.dumps(out, indent=1))
    (HERE / ("selection_log.json" if not args.strata else f"selection_log_{args.strata.replace(',', '_')}.json")).write_text(json.dumps(log, indent=1))
    print("selected", len(chosen), "->", path)


def merge(args):
    """Combine per-stratum selections: one tile per 2-degree cell across strata (fixed stratum order), <= TARGET each."""
    cells, tiles = set(), []
    for name in STRATA:
        f = HERE / f"selection_{name}.json"
        if not f.exists():
            print("missing", name); continue
        k = 0
        for t in json.loads(f.read_text())["tiles"]:
            cell = (int(math.floor(t["lon"] / 2)), int(math.floor(t["lat"] / 2)))
            if cell in cells or k >= TARGET_PER_STRATUM:
                continue
            cells.add(cell); t = dict(t, tile_id=f"{name}_{k:02d}"); tiles.append(t); k += 1
        print(name, k)
    (HERE / "selection.json").write_text(json.dumps(dict(created=time.time(), seed=SEED, criteria=STRATA, project=PROJECT,
        rule="static maps only; no water index or model output; >= 25 km from GDACS pool; one tile per 2-degree cell",
        tiles=tiles), indent=1))
    print("merged", len(tiles))


def download(args):
    ee = ee_init()
    import importlib.util                      # the HF `datasets` package shadows the repo folder name
    spec = importlib.util.spec_from_file_location("gee_download", REPO / "datasets/gee_download.py")
    g = importlib.util.module_from_spec(spec); sys.modules["gee_download"] = g; spec.loader.exec_module(g)
    _download_patch_s2, _save_patch = g._download_patch_s2, g._save_patch
    DOWNLOAD_BANDS, BAND_CROPS, PATCH_RADIUS_M = g.DOWNLOAD_BANDS, g.BAND_CROPS, g.PATCH_RADIUS_M
    sel = json.loads((HERE / "selection.json").read_text())
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    for t in sel["tiles"]:
        loc = OUT_DATA / t["tile_id"]
        if loc.exists() and any(loc.iterdir()):
            continue
        col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filter(ee.Filter.eq("system:index", t["scene"]))
        for attempt in range(3):
            try:
                patch = _download_patch_s2(col, [t["lon"], t["lat"]], PATCH_RADIUS_M, DOWNLOAD_BANDS, BAND_CROPS)
                loc.mkdir(parents=True, exist_ok=True)
                _save_patch(patch["raster"], patch["coords"], patch["metadata"], str(loc))
                print("ok", t["tile_id"], flush=True); break
            except Exception as exc:
                print("retry", t["tile_id"], str(exc)[:150], flush=True); time.sleep(5)


def render(args):
    import numpy as np
    from PIL import Image, ImageDraw
    from data.unlabeled import FloodUnlabeledDataset
    sel = json.loads((HERE / "selection.json").read_text())
    meta = {t["tile_id"]: t for t in sel["tiles"]}
    ds = FloodUnlabeledDataset(str(OUT_DATA), img_size=264, normalize=False)
    OUT_RATE.mkdir(parents=True, exist_ok=True); (OUT_RATE / "tiles").mkdir(exist_ok=True)
    def stretch(x, hi):
        return (np.clip(x / hi, 0, 1) ** (1 / 1.6) * 255).astype(np.uint8)
    rows = []
    for k, scene in enumerate(ds.scenes):
        tid = scene.parent.name
        x = ds[k]["image"].numpy()                                   # (6, 264, 264) reflectance, clipped [0,1]
        rgb = np.stack([stretch(x[2], 0.30), stretch(x[1], 0.30), stretch(x[0], 0.30)], -1)
        fc = np.stack([stretch(x[4], 0.50), stretch(x[3], 0.50), stretch(x[2], 0.30)], -1)
        big = lambda a: Image.fromarray(a).resize((528, 528), Image.NEAREST)
        canvas = Image.new("RGB", (528 * 2 + 12, 528 + 30), "white")
        canvas.paste(big(rgb), (0, 30)); canvas.paste(big(fc), (540, 30))
        d = ImageDraw.Draw(canvas)
        d.text((4, 8), f"{tid}  true colour (B4 B3 B2)", fill="black")
        d.text((544, 8), "false colour (B11 B8A B4): water = dark blue/black", fill="black")
        canvas.save(OUT_RATE / "tiles" / f"{tid}.png")
        t = meta[tid]
        rows.append(dict(tile_id=tid, stratum=t["stratum"], date=t["date"], lon=round(t["lon"], 4), lat=round(t["lat"], 4)))
    rows.sort(key=lambda r: r["tile_id"])
    with open(OUT_RATE / "ratings_template.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["rating", "notes"]); w.writeheader()
        for r in rows: w.writerow({**r, "rating": "", "notes": ""})
    (OUT_RATE / "tiles.json").write_text(json.dumps(rows))
    (OUT_RATE / "rating.html").write_text(RATING_HTML.replace("__TILES__", json.dumps(rows)))
    print("rendered", len(rows), "tiles ->", OUT_RATE)


RATING_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Dry-tile rating</title>
<style>
:root{--bg:#fff;--fg:#1e1e1e;--muted:#5f6b73;--line:#d0d5d9;--accent:#005b7f}
body{margin:0;font:15px/1.45 system-ui,Arial,sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
main{max-width:1160px;margin:0 auto;padding:16px}
.tile{border:1px solid var(--line);border-radius:8px;padding:12px;margin:0 0 18px}
.tile img{width:100%;height:auto;image-rendering:pixelated}
.meta{color:var(--muted);font-size:13px;margin:4px 0 8px}
label{margin-right:16px;cursor:pointer} input[type=text]{width:60%;padding:4px}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:7px 12px;cursor:pointer}
.done{border-color:#1baf7a}
</style></head><body>
<header><b>Dry-tile rating</b> · <span id="count"></span> ·
<button onclick="exportCsv()">Export ratings CSV</button>
<span class="meta"> Rate "dry" only if no water at all is visible (rivers, ponds, flooded fields, puddles).
Ratings are kept in this browser until you export.</span></header>
<main id="list"></main>
<script>
const TILES = __TILES__; const KEY = "dryrating_20261014";
let R = {}; try { R = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { R = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} update(); }
function update(){ const n = TILES.filter(t => R[t.tile_id] && R[t.tile_id].rating).length;
  document.getElementById("count").textContent = n + " of " + TILES.length + " rated";
  TILES.forEach(t => document.getElementById("t_" + t.tile_id).classList.toggle("done", !!(R[t.tile_id]||{}).rating)); }
const list = document.getElementById("list");
TILES.forEach(t => { const r = R[t.tile_id] || {}; const d = document.createElement("div"); d.className = "tile"; d.id = "t_" + t.tile_id;
  d.innerHTML = `<b>${t.tile_id}</b><div class="meta">${t.stratum} · ${t.date} · ${t.lat}, ${t.lon}</div>
  <img loading="lazy" src="tiles/${t.tile_id}.png" alt="${t.tile_id}"><div style="margin-top:8px">
  ${["dry","water_present","unsure"].map(v => `<label><input type="radio" name="r_${t.tile_id}" value="${v}" ${r.rating===v?"checked":""}> ${v.replace("_"," ")}</label>`).join("")}
  <input type="text" placeholder="notes (optional)" value="${(r.notes||"").replace(/"/g,"&quot;")}"></div>`;
  d.querySelectorAll("input[type=radio]").forEach(i => i.onchange = () => { R[t.tile_id] = {...(R[t.tile_id]||{}), rating: i.value}; save(); });
  d.querySelector("input[type=text]").oninput = e => { R[t.tile_id] = {...(R[t.tile_id]||{}), notes: e.target.value}; save(); };
  list.appendChild(d); });
function exportCsv(){ const head = "tile_id,stratum,date,lat,lon,rating,notes\\n";
  const body = TILES.map(t => { const r = R[t.tile_id] || {}; return [t.tile_id,t.stratum,t.date,t.lat,t.lon,r.rating||"",
    '"' + (r.notes||"").replace(/"/g,'""') + '"'].join(","); }).join("\\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([head + body], {type: "text/csv"}));
  a.download = "dry_ratings_20261014.csv"; a.click(); }
update();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "merge", "download", "render"])
    ap.add_argument("--target", type=int, default=22)
    ap.add_argument("--strata", default="")
    ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    {"select": select, "merge": merge, "download": download, "render": render}[a.phase](a)
