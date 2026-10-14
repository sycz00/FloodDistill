"""STURM S2 label census: pixel counts per class, chips without any class-1/2 pixel (dry under the paper mapping),
chips with zero scorable pixels under each mapping. Read-only."""
import glob, collections, numpy as np, rasterio, json, sys
root = "/data1/schmalstieg/flooddistill/data/STURM-Flood/Dataset/Sentinel2/Floodmaps"
files = sorted(glob.glob(root + "/*.tif"))
tot = collections.Counter(); chips = collections.Counter()
dry_paper = zero_valid_paper = dry_pub = zero_valid_pub = only_class2_water = 0
for f in files:
    with rasterio.open(f) as s: m = s.read(1)
    u, c = np.unique(m, return_counts=True)
    for a, b in zip(u, c): tot[int(a)] += int(b); chips[int(a)] += 1
    valid_paper = np.isin(m, [0, 1, 2]); water_paper = np.isin(m, [1, 2])
    valid_pub = np.isin(m, [0, 1, 2, 3, 4, 5]); water_pub = np.isin(m, [1, 2, 3, 4, 5])
    if valid_paper.sum() == 0: zero_valid_paper += 1
    elif water_paper.sum() == 0: dry_paper += 1
    if valid_pub.sum() == 0: zero_valid_pub += 1
    elif water_pub.sum() == 0: dry_pub += 1
    if (m == 2).any() and not (m == 1).any(): only_class2_water += 1
n = sum(tot.values())
print("chips", len(files), "pixels", n)
for k in sorted(tot): print(f"class {k:3d}: {tot[k]:10d} px ({100*tot[k]/n:5.2f}%) in {chips[k]} chips")
print("paper mapping {1,2}=water, {3,4,5,99}=ignore: zero-valid chips", zero_valid_paper, "| dry chips (valid but no water)", dry_paper)
print("published mapping {1..5}=water: zero-valid chips", zero_valid_pub, "| dry chips", dry_pub)
print("chips whose only scored water is class 2 (river-related)", only_class2_water)
