"""Model registry for the surface-water re-evaluation (14 Oct 2026). Read-only on checkpoints.
Writes models.json: id, family, arch, loader, checkpoint, sha256, seed, provenance."""
import hashlib, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REV = Path("/home/schmalstieg/repos/FloodDistill/experiments/revision")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
models = [dict(id="teacher", family="teacher", arch="teacher", loader="pe", seed=None,
               checkpoint="/home/schmalstieg/.cache/huggingface/hub/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-TL-Sen1Floods11/blobs/3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60",
               provenance="public Prithvi-EO-2.0-300M-TL-Sen1Floods11 (pinned blob used by v7/V9)")]
# 1) Recovered original cohort N2500 + matched controls (prompt4/5 O4 = paper student and slide-7 table)
O4 = Path("/data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006/training")
for d in sorted(O4.iterdir()):
    m = re.match(r"O4_train_(efficientvit_b0|mobilenet_v3_small|lightformer_resnet18)_(.+)_N(\d+)_s(\d+)$", d.name)
    if not m or not (d / "best.pt").exists(): continue
    arch, variant, n, seed = m.groups()
    models.append(dict(id=d.name.replace("O4_train_", "O4_"), family=f"O4_{arch}_{variant}_N{n}", arch=arch, loader="pe",
                       seed=int(seed), checkpoint=str(d / "best.pt"), provenance="prompt4/5 recovered original cohort (O4)"))
# 2) Table I matched-budget N252 arms + ReLU6 N2500 (September reproduction, runs_repro)
RR = Path("/data1/schmalstieg/flooddistill/runs_repro")
ARCH = {"evit": "efficientvit_b0", "unet": "unet_resnet50", "evit_relu6": "efficientvit_b0_relu6"}
for d in sorted(RR.iterdir()):
    m = re.match(r"(s11|pseudo)_(evit_relu6|evit|unet)_(ce|lkd|ofa)_N(\d+)_s(\d+)$", d.name)
    if not m or not (d / "best.pt").exists(): continue
    src, a, obj, n, seed = m.groups()
    if not (n == "252" or (a == "evit_relu6" and n == "2500")): continue
    models.append(dict(id="RR_" + d.name, family=f"RR_{src}_{a}_{obj}_N{n}", arch=ARCH[a], loader="pe", seed=int(seed),
                       checkpoint=str(d / "best.pt"), provenance="runs_repro (Sept 2026 reproduction of the paper recipe)"))
# 3) Capacity ladder (v5/v7 selected checkpoints), loaded with the frozen rungs builder
import csv
rows = list(csv.DictReader(open(REV / "capacity_v7_20261011T210730Z/results.csv")))
want = {"R4_LKD", "R4_CE", "A0_LKD_reanchored", "A3_LKD", "A3_CE", "A3N_LKD"}
seen = set()
for r in rows:
    if r["family"] in want and r["dataset"] == "s11" and r["domain"] == "full" and r["checkpoint_kind"] == "selected":
        key = (r["family"], r["seed"])
        if key in seen: continue
        seen.add(key)
        models.append(dict(id=f"V7_{r['family']}_s{r['seed']}", family="V7_" + r["family"], arch=r["graph"], loader="rungs",
                           seed=int(r["seed"]), checkpoint=r["selected_checkpoint_path"], expected_sha256=r["selected_checkpoint_sha256"],
                           provenance="capacity v5/v7 selected checkpoint (results.csv)"))
for m in models:
    m["sha256"] = sha(m["checkpoint"])
    if m.get("expected_sha256") and m["expected_sha256"] != m["sha256"]:
        sys.exit(f"checkpoint bytes changed: {m['id']}")
json.dump(dict(created="2026-10-14", models=models), open(HERE / "models.json", "w"), indent=1)
import collections
c = collections.Counter(m["family"] for m in models)
for k, v in sorted(c.items()): print(f"{v:2d} {k}")
print("total", len(models))
