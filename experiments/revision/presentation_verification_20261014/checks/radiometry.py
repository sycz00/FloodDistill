"""Per-band reflectance distributions as each loader delivers them BEFORE normalisation.
S11 train/test (TerraTorch path: DN*1e-4, NaN->0, no clip), GDACS pool (event vs baseline; DN*1e-4,
20 m bands bilinear to 10 m, clip [0,1]), WF-v2 TEST only (DN*1e-4, clip), STURM S2 (0.04+0.12 v, clip).
Valid = not all-zero spectrum. Read-only; WF-v2 train/val are never opened."""
import sys, glob, random, json, numpy as np, rasterio
sys.path.insert(0, "/home/schmalstieg/repos/FloodDistill")
random.seed(0)
D = "/data1/schmalstieg/flooddistill/data"
B = ["B2", "B3", "B4", "B8A", "B11", "B12"]
def summarize(name, arrs):
    x = np.concatenate([a.reshape(6, -1) for a in arrs], 1)
    v = ~(x == 0).all(0); x = x[:, v]
    g, s = x[1], x[4]
    m = (g - s) / np.maximum(g + s, 1e-6)
    sub = x[:, np.random.default_rng(0).choice(x.shape[1], min(x.shape[1], 2_000_000), replace=False)]
    p = np.percentile(sub, [5, 50, 95], axis=1)
    out = {"n_images": len(arrs), "valid_px": int(v.sum()),
           "median": [round(float(t), 4) for t in p[1]], "p5": [round(float(t), 4) for t in p[0]],
           "p95": [round(float(t), 4) for t in p[2]],
           "frac_gt1": [round(float((x[i] > 1).mean()), 5) for i in range(6)],
           "MNDWI_gt0_frac": round(float((m > 0).mean()), 4)}
    print(name, json.dumps(out)); return out
res = {}
# S11 (exact TerraTorch semantics: masked=True -> NaN -> 0, *1e-4, no clip)
def s11(split, n):
    ids = [l.strip().split(",")[0] for l in open(f"{D}/sen1floods11/v1.1/splits/flood_handlabeled/flood_{split}_data.txt")]
    ids = random.sample(ids, min(n, len(ids))); arrs = []
    for i in ids:
        f = f"{D}/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/{i}_S2Hand.tif"
        with rasterio.open(f) as src:
            a = src.read(masked=True).astype(np.float32).filled(np.nan)
        a = np.nan_to_num(a, nan=0.0)[[1, 2, 3, 8, 11, 12]] * 1e-4
        arrs.append(a)
    return arrs
res["s11_train"] = summarize("s11_train", s11("train", 80))
res["s11_test"] = summarize("s11_test", s11("test", 90))
# GDACS through the real loader (no normalisation, no resize: img_size=264 is identity on 264 tiles)
from data.unlabeled import FloodUnlabeledDataset
import csv
pf = list(csv.DictReader(open(f"{D}/pseudo_gdacs_v2/pool_features.csv")))
ds = FloodUnlabeledDataset(f"{D}/flood_unlabeled_v2", img_size=264, normalize=False)
idx = {(p.parent.name, p.name): k for k, p in enumerate(ds.scenes)}
for kind, flag in (("gdacs_event", "1"), ("gdacs_baseline", "0")):
    rows = random.sample([r for r in pf if r["is_flood_scene"] == flag], 150); arrs = []
    for r in rows:
        k = idx.get((r["location"], r["scene"]))
        if k is None: continue
        arrs.append(ds[k]["image"].numpy())
    res[kind] = summarize(kind, arrs)
# WF-v2 TEST through the real loader's raw reader
from data.ood_datasets import WorldFloodsDataset, STURMFloodDataset
wf = WorldFloodsDataset(f"{D}/WorldFloodsv2", split="test", img_size=512, tile_size=512, normalize="s11")
arrs = []
for k in random.sample(range(len(wf)), 120):
    ip, lp, w = wf.samples[k]; img, _ = wf._getitem_v2(ip, lp, w); arrs.append(img)
res["wf_test"] = summarize("wf_test", arrs)
st = STURMFloodDataset(f"{D}/STURM-Flood/Dataset", sensor="S2", img_size=128, normalize="original")
arrs = []
for k in random.sample(range(len(st)), 400):
    with rasterio.open(st.samples[k][0]) as src:
        a = src.read().astype(np.float32)
    a = np.clip(0.04 + 0.12 * np.nan_to_num(a, nan=0.0), 0, 1)[[0, 1, 2, 6, 7, 8]]
    arrs.append(a)
res["sturm"] = summarize("sturm", arrs)
json.dump(res, open(sys.argv[1], "w"), indent=1)
