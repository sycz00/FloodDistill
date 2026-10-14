"""MNDWI>0 detection rate per STURM label class at native 128 px (paper inverse 0.04+0.12v; MNDWI is a ratio so the
affine offset matters, the scale does not). Pixels with any non-finite selected band are skipped. Read-only."""
import glob, numpy as np, rasterio, collections
R = "/data1/schmalstieg/flooddistill/data/STURM-Flood/Dataset/Sentinel2"
hit = collections.Counter(); tot = collections.Counter()
for f in sorted(glob.glob(R + "/S2/*.tif")):
    with rasterio.open(f) as s: x = s.read().astype(np.float32)
    with rasterio.open(f.replace("/S2/", "/Floodmaps/")) as s: m = s.read(1)
    ok = np.isfinite(x[[1, 7]]).all(0)
    g = 0.04 + 0.12 * x[1]; w = 0.04 + 0.12 * x[7]
    pred = (g - w) / np.where(np.abs(g + w) < 1e-6, 1e-6, g + w) > 0
    for k in (0, 1, 2, 3, 4, 5):
        sel = ok & (m == k)
        tot[k] += int(sel.sum()); hit[k] += int((pred & sel).sum())
for k in sorted(tot):
    print(f"class {k}: MNDWI>0 on {100*hit[k]/max(tot[k],1):5.1f}% of {tot[k]} px")
