"""WF-v2 TEST dry windows (legacy 512 tiling, flood-only protocol): reproduce the dry-window count and MNDWI>0 alarm
count, then split MNDWI false-water pixels by distance to ignored pixels (JRC permanent water, cloud, invalid).
Read-only, CPU, test split only."""
import sys, numpy as np, rasterio
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, "/home/schmalstieg/repos/FloodDistill")
from data.ood_datasets import WorldFloodsDataset
wf = WorldFloodsDataset("/data1/schmalstieg/flooddistill/data/WorldFloodsv2", split="test", img_size=512, tile_size=512,
                        normalize="s11", permanent_water_as="ignore")
n_dry = alarms = 0; fp_tot = fp_near_perm = fp_near_any = fp_far = 0; with_perm = 0; events = {}
for ip, lp, w in wf.samples:
    img, mask = wf._getitem_v2(ip, lp, w)
    valid = mask != -1
    if not valid.any() or (mask == 1).any():
        continue
    n_dry += 1
    c, r, ww, hh = w
    with rasterio.open(wf.jrc_files[ip.name]) as s: perm = s.read(1, window=rasterio.windows.Window(c, r, ww, hh)) == 3
    if perm.any(): with_perm += 1
    g, s11 = img[1], img[4]
    m = (g - s11) / np.where(g + s11 == 0, 1e-6, g + s11) > 0
    fp = m & valid
    if fp.sum() >= 0.01 * valid.sum():
        alarms += 1; ev = ip.name.split("_")[0]; events[ev] = events.get(ev, 0) + 1
    d_perm = distance_transform_edt(~perm) if perm.any() else np.full(mask.shape, 1e9)
    d_ign = distance_transform_edt(valid)          # distance to nearest ignored pixel
    fp_tot += int(fp.sum())
    fp_near_perm += int((fp & (d_perm <= 3)).sum())
    fp_near_any += int((fp & (d_ign <= 3)).sum())
print(f"dry windows {n_dry}; containing JRC permanent water {with_perm}; MNDWI alarms (>=1% of valid) {alarms}")
print(f"MNDWI false-water px {fp_tot}: within 3 px (30 m) of JRC permanent water {100*fp_near_perm/fp_tot:.1f}%; "
      f"within 3 px of any ignored pixel (perm/cloud/invalid) {100*fp_near_any/fp_tot:.1f}%")
print("alarm windows by event:", dict(sorted(events.items(), key=lambda x: -x[1])))
