"""Dry-scene funnel of the GDACS pool, by stage and acquisition type (event window vs pre-event baseline).
Read-only: pool_features.csv, diversity_N2500_index.json, diversity_N252_index.json, patch_quality_v2.csv."""
import csv, json, sys, collections
POOL = "/data1/schmalstieg/flooddistill/data/pseudo_gdacs_v2"
rows = list(csv.DictReader(open(f"{POOL}/pool_features.csv")))
print("pool rows", len(rows))
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan", "None") else float("nan")
sel = {n: set(json.load(open(f"{POOL}/diversity_N{n}_index.json"))["files"]) for n in (252, 2500)}
def stage(r):
    ff, cs, mc = f(r, "flood_frac"), f(r, "conf_std"), f(r, "mean_conf")
    return dict(conf80=mc >= 0.80, cand=(0.05 <= ff <= 0.85) and cs >= 0.05)
def drycls(r):
    ff = f(r, "flood_frac")
    return "teacher=0" if ff == 0 else ("<0.1%" if ff < 0.001 else ("<1%" if ff < 0.01 else ("<5%" if ff < 0.05 else ">=5%")))
tab = collections.Counter(); cand = collections.Counter(); s2500 = collections.Counter(); s252 = collections.Counter()
mnd = collections.Counter()
for r in rows:
    kind = "event" if r["is_flood_scene"] == "1" else "baseline"
    c = drycls(r); st = stage(r)
    tab[(kind, c)] += 1
    if st["cand"]: cand[(kind, c)] += 1
    if r["file"] in sel[2500]: s2500[(kind, c)] += 1
    if r["file"] in sel[252]: s252[(kind, c)] += 1
    m = f(r, "mndwi_water_frac")
    mc = "mndwi=0" if m == 0 else ("<1%" if m < 0.01 else (">=1%"))
    mnd[(kind, c, mc)] += 1
cls = ["teacher=0", "<0.1%", "<1%", "<5%", ">=5%"]
print(f"{'kind':9s} {'teacher water frac':18s} {'pool':>6s} {'candidates':>10s} {'N2500':>6s} {'N252':>5s}")
for kind in ("event", "baseline"):
    for c in cls:
        print(f"{kind:9s} {c:18s} {tab[(kind,c)]:6d} {cand[(kind,c)]:10d} {s2500[(kind,c)]:6d} {s252[(kind,c)]:5d}")
    print(f"{kind:9s} {'total':18s} {sum(tab[(kind,c)] for c in cls):6d} {sum(cand[(kind,c)] for c in cls):10d} {sum(s2500[(kind,c)] for c in cls):6d} {sum(s252[(kind,c)] for c in cls):5d}")
print("\nteacher-dry (<5%) tiles by MNDWI water fraction (scene-level, as stored):")
for kind in ("event", "baseline"):
    for mc in ("mndwi=0", "<1%", ">=1%"):
        n = sum(mnd[(kind, c, mc)] for c in cls[:4])
        print(f"  {kind:9s} {mc:8s} {n}")
q = list(csv.DictReader(open("/data1/schmalstieg/flooddistill/data/flood_unlabeled_v2/patch_quality_v2.csv")))
ok = [x for x in q if x["status"] == "1"]
print(f"\nacquisition: {len(q)} event-patch slots attempted, {len(ok)} downloaded; every downloaded event tile has "
      f"MNDWI>0 new-water >= 1%: {all(float(x['new_water_frac']) >= 0.01 for x in ok)}; "
      f"slots that exhausted 8 candidates without passing: {sum(x['status']=='0' for x in q)}")
