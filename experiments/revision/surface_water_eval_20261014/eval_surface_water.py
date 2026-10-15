"""Surface-water re-evaluation of all approaches (approved by the author 14 Oct 2026).

One forward pass per (model, image); every label mapping is scored from that pass:
  S11 / Bolivia : S11 hand labels as distributed (surface water)               -> "s11"
  WorldFloods   : "surface"   gt water (flood + permanent) vs land; cloud/invalid/no-data ignored
                  "floodonly" current revision protocol P3_WF_NATIVE_ZERO_PAD512_V1 (JRC permanent ignored),
                              only on the 785 windows of that protocol
  STURM         : "surface"   classes 1-5 water, 0 land, 99/NaN-support ignored  (STURM's own water definition)
                  "paper"     1-2 water, 3-5 ignored (= P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1)
                  "strict"    1 water, 2-5 ignored (flood polygons only)
                  "paper_legacy" (512 only) paper mapping without the NaN-support exclusion (submitted protocol)
Spectral rules (MNDWI, NDWI-B8A, AWEInsh, AWEIsh, WI2015) at the published threshold 0 and at one threshold
calibrated on S11 train+val ONLY (max pooled water IoU), frozen in calibration.json before any test image is read.
Outputs per dataset: results/<dataset>.npz with per-window confusion counts [TP, FP, FN, TN] for every model x
mapping, and per-event margin/index histograms (for AP and source-selected operating points). Aggregation is
offline (collect.py). WorldFloods-v2 train/val are never opened.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/schmalstieg/repos/FloodDistill")
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/schmalstieg/repos/_deps/efficientvit")
sys.path.append("/home/schmalstieg/repos/_deps/flooddistill_revision_python")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("LIGHTFORMER_ROOT", "/home/schmalstieg/repos/_deps/lightformer_574577dd")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
DATA = Path("/home/schmalstieg/data")
NB = 4096                                   # histogram bins
MARGIN_RANGE = (-32.0, 32.0)
RULES = OrderedDict([("MNDWI", (-1.0, 1.0)), ("NDWI", (-1.0, 1.0)), ("AWEInsh", (-3.0, 3.0)),
                     ("AWEIsh", (-1.5, 1.5)), ("WI2015", (-60.0, 60.0))])
MAPPINGS = {"s11val": ["s11"], "s11test": ["s11"], "bolivia": ["s11"], "s11train": ["s11"],
            "wf": ["surface", "floodonly"],
            "sturm512": ["surface", "paper", "strict", "paper_legacy"],
            "sturm128": ["surface", "paper", "strict"]}
HIST_MAPPINGS = {"wf": ["surface", "floodonly"], "sturm512": ["surface", "paper"], "sturm128": ["surface", "paper"]}
STURM_SEL = [0, 1, 2, 6, 7, 8]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- datasets
class S11Split:
    """TerraTorch Sen1Floods11NonGeo with the evaluation transform of data/sen1floods11.get_dataloaders."""

    def __init__(self, split):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from terratorch.datasets import Sen1Floods11NonGeo
        from data.sen1floods11 import NORM_MEAN, NORM_STD

        class _DS(Sen1Floods11NonGeo):
            splits = {"train": "train", "val": "valid", "test": "test", "bolivia": "bolivia"}

        tf = A.Compose([A.Resize(512, 512), A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
                        ToTensorV2()])
        self.ds = _DS(data_root=str(DATA / "sen1floods11"), split=split,
                      bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"], transform=tf,
                      constant_scale=0.0001, no_data_replace=0, no_label_replace=-1)
        self.names = [Path(p).name for p in self.ds.image_files]
        self.maps = self.names
        self.events = [n.split("_")[0] for n in self.names]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import torch
        item = self.ds[i]
        m = item["mask"].to(torch.int8).unsqueeze(0)
        return item["image"].float(), m, i


class WFTest:
    def __init__(self):
        import rasterio  # noqa
        from data.ood_datasets import WorldFloodsDataset
        root = DATA / "WorldFloodsv2"
        self.ds = WorldFloodsDataset(root, split="test", img_size=512, tile_size=512, normalize="s11",
                                     ignore_permanent_water=False)
        legacy = WorldFloodsDataset(root, split="test", img_size=512, tile_size=512, normalize="s11",
                                    permanent_water_as="ignore")
        assert len(legacy) == 785
        self.legacy = {(p.name, tuple(w)) for p, _, w in legacy.samples}
        mine = {(p.name, tuple(w)) for p, _, w in self.ds.samples}
        assert self.legacy <= mine, "flood-only windows must be a subset of surface-water windows"
        self.jrc_dir = self.ds.samples[0][0].parent.parent / "PERMANENTWATERJRC"
        self.names = [f"{p.name}#r{w[1]}_c{w[0]}" for p, _, w in self.ds.samples]
        self.maps = [p.name for p, _, _ in self.ds.samples]
        self.events = [p.name.split("_")[0] for p, _, _ in self.ds.samples]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import numpy as np, rasterio, torch
        from evaluation.revision_native_border import native_item
        ip, _, w = self.ds.samples[i]
        item = native_item(self.ds, i)                     # padded normalized image, unpadded surface mask
        c, r, ww, hh = w
        surface = item["mask"].astype(np.int8)
        flood = np.full_like(surface, -1)
        if (ip.name, tuple(w)) in self.legacy:
            with rasterio.open(self.jrc_dir / ip.name) as s:
                perm = s.read(1, window=rasterio.windows.Window(c, r, ww, hh)) == 3
            flood = surface.copy()
            flood[perm & (flood != -1)] = -1
        masks = np.full((2, 512, 512), -1, np.int8)
        masks[0, :hh, :ww] = surface
        masks[1, :hh, :ww] = flood
        return item["image"].float(), torch.from_numpy(masks), i


class STURMSet:
    def __init__(self, size):
        import pandas as pd
        from data.ood_datasets import STURMFloodDataset
        root = DATA / "STURM-Flood/Dataset"
        self.size = size
        self.ds = STURMFloodDataset(root, sensor="S2", img_size=size, normalize="original", class_mapping="published")
        assert len(self.ds) == 2675
        md = pd.read_csv(root / "Sentinel2_metadata.csv")
        to_map = dict(zip(md["tile_id"].astype(str), md["floodmap_id"].astype(str)))
        to_ev = dict(zip(md["tile_id"].astype(str), md["ems_code"].astype(str)))
        self.names = [p.name for p, _ in self.ds.samples]
        self.maps = [to_map[n] for n in self.names]
        self.events = [to_ev[n] for n in self.names]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import cv2, numpy as np, rasterio, torch
        item = self.ds[i]
        ip, lp = self.ds.samples[i]
        with rasterio.open(ip) as s:
            raw = s.read().astype(np.float32)
        with rasterio.open(lp) as s:
            lab = s.read(1).astype(np.int64)
        S = self.size
        lab_s = lab if S == 128 else cv2.resize(lab, (S, S), interpolation=cv2.INTER_NEAREST)
        surface = np.where(lab_s == 0, 0, np.where((lab_s >= 1) & (lab_s <= 5), 1, -1)).astype(np.int8)
        assert np.array_equal(surface, item["mask"].numpy().astype(np.int8)), "label resize mismatch"
        paper = np.where(lab_s == 0, 0, np.where(np.isin(lab_s, [1, 2]), 1, -1)).astype(np.int8)
        strict = np.where(lab_s == 0, 0, np.where(lab_s == 1, 1, -1)).astype(np.int8)
        invalid6 = ~np.isfinite(raw[STURM_SEL]).all(axis=0)
        support = invalid6 if S == 128 else cv2.resize(invalid6.astype(np.float32), (S, S),
                                                        interpolation=cv2.INTER_LINEAR) > 0
        stack = [surface, paper, strict] + ([paper.copy()] if S == 512 else [])
        for m in stack[:3]:
            m[support] = -1
        masks = torch.from_numpy(np.stack(stack))
        return item["image"][STURM_SEL].float(), masks, i


def make_dataset(name):
    if name in ("s11val", "s11test", "s11train", "bolivia"):
        return S11Split({"s11val": "val", "s11test": "test", "s11train": "train", "bolivia": "bolivia"}[name])
    if name == "wf":
        return WFTest()
    return STURMSet(512 if name == "sturm512" else 128)


# ------------------------------------------------------------------------------------------------- models
def load_model(entry, device):
    import torch
    import evaluation.perscene_eval as pe
    if entry["loader"] == "pe":
        return pe.load(entry["arch"], entry["checkpoint"], device)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v5_rungs", REPO / "experiments/revision/capacity_v4_20261008/code/v5/rungs.py")
    rungs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rungs)
    model = rungs.build(entry["arch"])
    payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def rule_maps(x):
    """Index maps from S11-normalized 6-band input (same reflectance recovery as SpectralIndexPredictor)."""
    import torch
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    mu = torch.tensor(NORM_MEAN, device=x.device).view(1, -1, 1, 1)
    sd = torch.tensor(NORM_STD, device=x.device).view(1, -1, 1, 1)
    r = x * sd + mu
    b, g, red, n, s1, s2 = (r[:, k] for k in range(6))
    return OrderedDict([
        ("MNDWI", (g - s1) / (g + s1).clamp_min(1e-6)),
        ("NDWI", (g - n) / (g + n).clamp_min(1e-6)),
        ("AWEInsh", 4 * (g - s1) - (0.25 * n + 2.75 * s2)),
        ("AWEIsh", b + 2.5 * g - 1.5 * (n + s1) - 0.25 * s2),
        ("WI2015", 1.7204 + 171 * g + 3 * red - 70 * n - 45 * s1 - 71 * s2)])


# ------------------------------------------------------------------------------------------- accumulation
class Acc:
    def __init__(self, n_windows, n_map, n_events, hist_maps, rng, device):
        import torch
        self.conf = torch.zeros((n_windows, n_map, 4), dtype=torch.int64, device=device)
        self.hist = {k: torch.zeros(n_events * 2 * NB, dtype=torch.int64, device=device) for k in hist_maps}
        self.rng, self.n_events = rng, n_events

    def add(self, pred, masks, idx, score=None, ev=None, map_index=None):
        import torch
        for k in range(masks.shape[1]):
            m = masks[:, k]
            v = m != -1
            y = m == 1
            p = pred & v
            tp = (p & y).sum((1, 2)); fp = (p & ~y).sum((1, 2))
            fn = (~pred & y & v).sum((1, 2)); tn = (~pred & ~y & v).sum((1, 2))
            self.conf[idx, k] += torch.stack([tp, fp, fn, tn], 1)
        if score is None:
            return
        lo, hi = self.rng
        b = ((score.clamp(lo, hi - 1e-6) - lo) / (hi - lo) * NB).long().clamp(0, NB - 1)
        for name, k in map_index.items():
            m = masks[:, k]
            v = m != -1
            key = ((ev.view(-1, 1, 1).expand_as(m) * 2 + (m == 1).long()) * NB + b)[v]
            self.hist[name] += torch.bincount(key, minlength=self.n_events * 2 * NB)


def run_dataset(name, models, rules_thr, device, workers, log, rules_only=False):
    import numpy as np, torch
    from torch.utils.data import DataLoader
    ds = make_dataset(name)
    maps = MAPPINGS[name]
    hist_maps = HIST_MAPPINGS.get(name, maps)
    hist_index = {m: maps.index(m) for m in hist_maps}
    events = sorted(set(ds.events))
    ev_id = torch.tensor([events.index(e) for e in ds.events], device=device)
    n = len(ds)
    accs = OrderedDict()
    if not rules_only:
        for mid in models:
            accs[mid] = Acc(n, len(maps), len(events), hist_maps, MARGIN_RANGE, device)
    for r, rng in RULES.items():
        accs["rule:" + r] = Acc(n, len(maps), len(events), hist_maps, rng, device)
        if rules_thr:
            accs["rulecal:" + r] = Acc(n, len(maps), len(events), [], rng, device)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=workers, pin_memory=True)
    t0 = time.time(); seen = 0
    with torch.no_grad():
        for x, masks, idx in dl:
            x = x.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
            idx = idx.to(device); ev = ev_id[idx]
            if not torch.isfinite(x).all():
                raise RuntimeError(f"non-finite input in {name}")
            for mid, model in models.items():
                if rules_only:
                    break
                out = model(x)
                out = out["out"] if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)
                assert out.shape[-2:] == x.shape[-2:] and out.shape[1] == 2, (mid, out.shape)
                margin = (out[:, 1] - out[:, 0]).float()
                pred = out.argmax(1) == 1
                accs[mid].add(pred, masks, idx, margin, ev, hist_index)
            for r, value in rule_maps(x).items():
                accs["rule:" + r].add(value > 0, masks, idx, value, ev, hist_index)
                if rules_thr:
                    accs["rulecal:" + r].add(value > rules_thr[r], masks, idx)
            seen += x.shape[0]
            if seen % 400 < 8:
                log(f"{name}: {seen}/{n} images, {time.time() - t0:.0f}s")
    out = {"names": np.array(ds.names), "maps": np.array(ds.maps), "events": np.array(ds.events),
           "event_list": np.array(events), "mappings": np.array(maps), "hist_mappings": np.array(hist_maps),
           "model_ids": np.array(list(accs))}
    for mid, a in accs.items():
        out["conf__" + mid] = a.conf.cpu().numpy()
        for k, h in a.hist.items():
            out[f"hist__{mid}__{k}"] = h.view(len(events), 2, NB).cpu().numpy()
    log(f"{name}: done {n} images in {time.time() - t0:.0f}s")
    return out


def calibrate(dsets, log):
    """Pooled-water-IoU-optimal threshold per rule on S11 train+val (permitted calibration data)."""
    import numpy as np
    thr = {}
    for r, (lo, hi) in RULES.items():
        h = sum(d[f"hist__rule:{r}__s11"].sum(0) for d in dsets)       # (2 labels, NB)
        land, water = h[0], h[1]
        # predict water if value > edge_k; edges at bin boundaries
        tp = np.cumsum(water[::-1])[::-1]; fp = np.cumsum(land[::-1])[::-1]
        fn = water.sum() - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        k = int(np.argmax(iou))
        edge = lo + (hi - lo) * k / NB
        thr[r] = float(edge)
        log(f"calibration {r}: threshold {edge:+.4f}, S11 train+val water IoU {iou[k]:.4f} "
            f"(threshold 0: {iou[int(round((0 - lo) / (hi - lo) * NB))]:.4f})")
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", default="s11val,s11test,bolivia,wf,sturm512,sturm128")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--only", default="", help="comma-separated model ids (smoke)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images only")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    import numpy as np, torch
    torch.set_num_threads(4)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "eval.log", "a")
    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    reg = json.loads(Path(args.models).read_text())["models"]
    if args.only:
        keep = set(args.only.split(","))
        reg = [m for m in reg if m["id"] in keep]
    identity = {"script_sha256": sha(__file__), "models_sha256": sha(args.models), "only": args.only,
                "limit": args.limit, "torch": torch.__version__,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    device = torch.device(args.device)
    if args.limit:
        global make_dataset
        _orig = make_dataset
        def make_dataset(name):                                         # noqa: F811 (smoke subset)
            ds = _orig(name)
            class Sub:
                def __init__(s): s.ds = ds; s.names = ds.names[:args.limit]; s.maps = ds.maps[:args.limit]; s.events = ds.events[:args.limit]
                def __len__(s): return min(args.limit, len(ds))
                def __getitem__(s, i): return ds[i]
            return Sub()
    # phase 0: rule thresholds from S11 train+val only, frozen before any test image is read
    cal_path = out_dir / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text()); assert cal["identity"]["script_sha256"] == identity["script_sha256"]
        thr = cal["thresholds"]
    else:
        parts = [run_dataset(n, {}, None, device, args.workers, log, rules_only=True) for n in ("s11train", "s11val")]
        thr = calibrate(parts, log)
        atomic_json(cal_path, {"identity": identity, "thresholds": thr, "criterion": "max pooled water IoU",
                               "data": "Sen1Floods11 train (252) + validation (89) hand labels; no test data",
                               "created": time.time()})
    log(f"loading {len(reg)} models")
    models = OrderedDict()
    for m in reg:
        if sha(m["checkpoint"]) != m["sha256"]:
            raise RuntimeError(f"checkpoint changed: {m['id']}")
        models[m["id"]] = load_model(m, device)
    log(f"loaded; GPU memory {torch.cuda.memory_allocated() / 2**30:.2f} GiB" if device.type == "cuda" else "loaded")
    for name in args.datasets.split(","):
        target = out_dir / f"{name}.npz"
        done = out_dir / f"{name}.done.json"
        if done.exists() and json.loads(done.read_text())["identity"] == identity:
            log(f"{name}: already complete, skipped"); continue
        size = 128 if name == "sturm128" else 512
        usable = OrderedDict()
        with torch.no_grad():
            for mid, model in models.items():
                try:
                    model(torch.zeros(1, 6, size, size, device=device))
                    usable[mid] = model
                except Exception as exc:                       # e.g. LightFormer window padding at 128 px
                    log(f"{name}: {mid} not applicable at {size} px ({type(exc).__name__}); excluded")
        res = run_dataset(name, usable, thr, device, args.workers, log)
        tmp = out_dir / f"{name}.tmp.npz"
        np.savez_compressed(tmp, **res); os.replace(tmp, target)
        atomic_json(done, {"identity": identity, "npz_sha256": sha(target), "finished": time.time(),
                           "n_images": int(len(res["names"]))})
    atomic_json(out_dir / "COMPLETE.json", {"identity": identity, "finished": time.time()})
    log("all datasets complete")


if __name__ == "__main__":
    main()
