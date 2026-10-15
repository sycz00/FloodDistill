"""Qualitative panels from ONNX and TensorRT models on the Jetson.

--select accepts dataset indices, filenames or unique substrings. Indices
refer to the current loader configuration; changing tile size or masking
can change them. WorldFloods IDs include the window as scene.tif#r..._c....

--manifest uses selections from figures/plot_models.py. Names are resolved
first. Index fallback requires matching loader metadata; use --split test
for workstation test selections. Unsupported datasets are skipped.

Gain/haze panels contain a clean and a perturbed row per tile. --dump_npz
stores both prediction sets for merging into workstation panels. The merge
checks filenames and ground truth. Defaults match eval_jetson.py: STURM
original normalization and flood-only WorldFloods. See jetson/README.md."""

import argparse
import os
import sys

# Import order matters on Jetson (see eval_jetson.py): torch before albumentations/skimage,
# tensorrt before numpy.
import torch
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as _pycuda
    import pycuda.autoinit           # noqa: F401 - required side-effect (CUDA context)
    _TRT_AVAILABLE = True
except Exception as _trt_err:
    trt = None                       # type: ignore[assignment]
    _pycuda = None                   # type: ignore[assignment]
    _TRT_AVAILABLE = False
    print(f"[WARNING] TRT Python API unavailable: {_trt_err}")
    print("  Set PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH and retry.")
    print("  ONNX columns will still be drawn.")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# terratorch-free Sen1Floods11 - must be installed in sys.modules before data.ood_datasets,
# which imports NORM_MEAN/NORM_STD from data.sen1floods11 at module level.
import data.sen1floods11_nodep as _s11_nodep
sys.modules["data.sen1floods11"] = _s11_nodep    # type: ignore[assignment]

import onnxruntime as ort

from data.sen1floods11 import NORM_MEAN, NORM_STD
from data.channel_adapter import sturm_s2_adapter

IMG_SIZE_DEFAULT = 512
IN_CHANNELS = 6

# Arguments

p = argparse.ArgumentParser(
    description="Qualitative panels from the deployed ONNX / TRT models",
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
p.add_argument("--model_dir", default=os.path.join(_REPO_ROOT, "runs/deploy_qat/docker_qat"),
               help="Dir with model_float/qdq_trt*.onnx and any *.engine")
p.add_argument("--model", action="append", default=None,
               help="Draw this single ONNX instead of the standard set (repeatable)")
p.add_argument("--engine", default=None, help="Extra TRT .engine to draw a column for")
p.add_argument("--no_onnx", action="store_true", help="Engine columns only")
p.add_argument("--dataset", default="sturm", choices=["s11", "sturm", "wf"])
p.add_argument("--split", default="val", choices=["train", "val", "test"],
               help="s11 split (default val — the split eval_jetson.py scores)")
p.add_argument("--s11_root", default=os.path.expanduser("~/data/sen1floods11"))
p.add_argument("--sturm_root", default=os.path.expanduser("~/data/STURM-Flood/Dataset"))
p.add_argument("--wf_root", default=os.path.expanduser("~/data/WorldFloodsv2"))
p.add_argument("--sturm_normalize", default="original",
               choices=["percentile", "clip", "original"],
               help="Must match the eval run this panel illustrates (default original)")
p.add_argument("--wf_include_permanent", action="store_true",
               help="Score JRC permanent water as water (legacy). Default: flood-only.")
p.add_argument("--img_size", type=int, default=IMG_SIZE_DEFAULT)
p.add_argument("--select", default=None,
               help="Comma-separated dataset indices and/or filenames (see epilog)")
p.add_argument("--manifest", default=None,
               help="qual_samples_manifest.json written by figures/plot_models.py. "
                    "Reproduces THAT figure's scenes and layout with the deployed models "
                    "substituted for the workstation ones, so qual_N2500_fullpool.png and the "
                    "device panel show identical imagery. Rows may span datasets; picks from "
                    "datasets this device does not carry (S1S2Water, GDACS) are dropped with a "
                    "warning rather than silently.")
p.add_argument("--per_dataset", action="store_true",
               help="Write ONE figure per dataset instead of one tall sheet. A 14-pick "
                    "manifest is 12 device rows in a single column strip otherwise.")
p.add_argument("--max_per_dataset", type=int, default=0,
               help="Keep at most N picks per dataset (0 = all). Applied in manifest order.")
p.add_argument("--paper_layout", action="store_true",
               help="Draw in the qual_N*.png style: axes off, flood-IoU in the panel title, "
                    "Blues colormap, no suptitle/legend. Implied by --manifest.")
p.add_argument("--font_scale", type=float, default=1.9,
               help="--paper_layout font multiplier; matches plot_models.py's default.")
# Literal 0.40, not figures.qual_rgb.TRUE_COLOUR_CEILING: that module is imported below,
# after the sys.path setup, so it does not exist yet here. The two are asserted equal at
# import time (see the check next to the import), so they cannot drift.
p.add_argument("--rgb_fixed", type=float, default=0.40,
               help="Sentinel-2 True Color ceiling: the input is rendered as min(rho/K, 1) "
                    "on B4/B3/B2. K=0.40 is the Sentinel Hub / Copernicus Browser "
                    "`2.5 * reflectance` evalscript. Zero-referenced and the SAME map for "
                    "every tile and every corruption level, so hue does not drift between "
                    "tiles and a gained or hazed input is shown at its true magnitude. "
                    "0 restores the old per-scene percentile stretch. Must match the "
                    "workstation panel this figure is compared against "
                    "(plot_models.py --rgb_fixed).")
p.add_argument("--list", action="store_true", help="Print index -> filename and exit")
p.add_argument("--stats", action="store_true",
               help="--list: also read each tile's GT for water/scorable fraction (slow)")
p.add_argument("--list_limit", type=int, default=0, help="--list: stop after N entries")
p.add_argument("--gain", type=float, default=None,
               help="Also run every tile at this radiometric gain (e.g. 1.30). Each tile "
                    "becomes TWO figure rows, clean above perturbed, and both predictions "
                    "go into the .npz — the layout the workstation panel uses.")
p.add_argument("--haze", type=float, default=None,
               help="Also run every tile at this haze level (e.g. 0.20). Two rows per tile, "
                    "as --gain.")
p.add_argument("--out", default=None,
               help="Output PNG (default assets/qual_jetson_<dataset>[_<axis>].png)")
p.add_argument("--dpi", type=int, default=140)
p.add_argument("--dump_npz", action="store_true",
               help="Also write the raw arrays next to the PNG (workstation re-plotting)")
args = p.parse_args()

if args.manifest:
    args.paper_layout = True


def pool_of(model_dir):
    """Which pseudo pool the checkpoint in `model_dir` was distilled from."""
    base = os.path.basename(os.path.normpath(model_dir))
    if base == "docker_qat":
        return "current (data/pseudo_gdacs, N=2,500)"
    if base.startswith("n_qat_"):
        return "current (data/pseudo_gdacs, N=2,500)"
    if base.startswith("qat_s") or "div_ofa_hs" in base:
        return "earlier pool — REMOVED from this repo 2026-08-13; do not publish beside current results"
    return "UNKNOWN — check the parent checkpoint before publishing this panel"


if args.gain is not None and args.haze is not None:
    p.error("--gain and --haze are separate axes; pass one (a panel column is one axis)")

_AXIS = "gain" if args.gain is not None else ("haze" if args.haze is not None else None)
_LEVEL = args.gain if _AXIS == "gain" else (args.haze if _AXIS == "haze" else None)
_SIM = ({"sim_gain": args.gain} if _AXIS == "gain" else
        {"sim_haze": args.haze} if _AXIS == "haze" else {})

# Datasets
# num_workers=0, direct ds[i] access. Selecting tile 982 should not decode
# 981 earlier tiles. All loaders here are map-style.


def build_dataset(dataset=None, **sim):
    """Return (dataset, needs_9to6_adapter). `sim` = the perturbation kwargs, or empty.

        `dataset` defaults to --dataset; --manifest passes it explicitly because one panel can
        then carry rows from several datasets at once, exactly as qual_N*.png does.
    """
    dataset = dataset or args.dataset
    if dataset == "sturm":
        from data.ood_datasets import get_sturm_flood_dataloader
        dl = get_sturm_flood_dataloader(
            args.sturm_root, sensor="S2", batch_size=1, num_workers=0,
            img_size=args.img_size, normalize=args.sturm_normalize, protocol="flood_paper", **sim)
        return dl.dataset, True
    if dataset == "wf":
        from data.ood_datasets import get_worldfloods_dataloader
        dl = get_worldfloods_dataloader(
            args.wf_root, split="test", batch_size=1, num_workers=0,
            img_size=args.img_size, tile_size=args.img_size, normalize="s11",
            ignore_permanent_water=not args.wf_include_permanent, **sim)
        return dl.dataset, False
    if sim:
        p.error("--gain/--haze are OOD-loader features; s11 has no perturbation hook")
    from data.sen1floods11 import get_dataloaders
    dls = dict(zip(("train", "val", "test"), get_dataloaders(
        args.s11_root, batch_size=1, num_workers=0, eval_img_size=args.img_size)))
    return dls[args.split].dataset, False


def sample_names(ds, dataset=None):
    """index -> filename, without reading any raster.

        Mirrors each dataset's own __getitem__ 'filename' exactly (WF appends the window as
        '#r<row>_c<col>'), so a name printed by --list is a name --select accepts.
    """
    dataset = dataset or args.dataset
    if dataset == "sturm":
        return [img.name for img, _ in ds.samples]
    if dataset == "wf":
        out = []
        for img, _gt, win in ds.samples:
            if win is None:
                out.append(img.name)
            else:
                c, r, _w, _h = win
                out.append(f"{img.name}#r{r}_c{c}")
        return out
    return [os.path.basename(f) for f in ds.image_files]


def scene_of(name, dataset=None):
    """Group a tile into its scene, the same grouping eval_jetson.py aggregates macro over."""
    dataset = dataset or args.dataset
    if dataset == "wf":
        return name.split("#")[0]
    if dataset == "sturm":
        return _sturm_scene_map().get(name, name)
    return name


_STURM_MAP = {}


def _sturm_scene_map():
    """tile_id -> floodmap_id from the dataset's own CSV (stdlib csv, no pandas needed)."""
    global _STURM_MAP
    if _STURM_MAP:
        return _STURM_MAP
    import csv
    path = os.path.join(args.sturm_root, "Sentinel2_metadata.csv")
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("tile_id") and row.get("floodmap_id"):
                    _STURM_MAP[row["tile_id"].strip()] = row["floodmap_id"].strip()
    except Exception as e:
        print(f"[WARNING] STURM scene map unavailable ({e}); scenes shown as tile names.")
    return _STURM_MAP


def resolve(entries, names):
    """'417' | exact filename | unique substring  ->  dataset index.

        On ambiguity, raise an error with the candidates. Taking the first match could
        draw a different tile without this being visible in the finished figure.
    """
    by_name = {n: i for i, n in enumerate(names)}
    idxs = []
    for raw in entries:
        e = raw.strip()
        if not e:
            continue
        if e.lstrip("-").isdigit():
            i = int(e)
            if not 0 <= i < len(names):
                p.error(f"--select {i}: out of range (dataset has {len(names)} samples)")
            idxs.append(i)
            continue
        if e in by_name:
            idxs.append(by_name[e])
            continue
        hits = [i for i, n in enumerate(names) if e in n]
        if len(hits) == 1:
            print(f"[select] '{e}' -> [{hits[0]}] {names[hits[0]]}")
            idxs.append(hits[0])
        elif not hits:
            p.error(f"--select '{e}': no such tile. Run --list to see the names.")
        else:
            preview = "\n    ".join(f"[{i}] {names[i]}" for i in hits[:12])
            p.error(f"--select '{e}': matches {len(hits)} tiles — narrow it:\n    {preview}"
                    + ("\n    ..." if len(hits) > 12 else ""))
    return idxs


# Manifest -> rows
# plot_models.py names datasets S11 / STURM-S2 / S1S2Water / GDACS; this script speaks
# s11 / sturm / wf. Only the first two exist on both sides. This limitation is reported:
# qual_N*.png has six rows; the device panel can show the three from datasets evaluated
# on the device.
# WorldFloods-v2 was missing here until 2026-08-13: --dataset wf and build_dataset("wf")
# both existed, but a manifest naming it fell through to "SKIP: unknown dataset", so the
# robustness panels (which are STURM + WorldFloods) lost half their rows silently.
_MANIFEST_DS = {"S11": "s11", "STURM-S2": "sturm", "WorldFloods-v2": "wf"}
_MANIFEST_SKIP = {"S1S2Water": "not in this script's dataset roster",
                  "GDACS": "training pool, not shipped to the device"}


def manifest_plan():
    """-> [(dataset_key, [dataset indices])] for the picks this device can actually draw.

        Resolves by filename first, the index is only a checked fallback. An index only means
        something for the loader configuration that produced it, and by default the two sides
        differ: plot_models.py reads the Sen1Floods11 test split, this script defaults to val.
        Falling back to the index across that difference would draw a different scene without
        any visible warning, so the fallback is refused unless the manifest's recorded
        `loader` block matches.
    """
    import json
    man = json.loads(open(args.manifest).read())
    loader = man.get("loader", {})
    plan = []
    for key, picks in man.get("picks", {}).items():
        if key in _MANIFEST_SKIP:
            print(f"[manifest] SKIP {key} ({len(picks)} pick(s)): {_MANIFEST_SKIP[key]}")
            continue
        dkey = _MANIFEST_DS.get(key)
        if dkey is None:
            print(f"[manifest] SKIP {key}: unknown dataset")
            continue
        names = sample_names(build_dataset(dkey)[0], dkey)
        by_name = {n: i for i, n in enumerate(names)}
        idxs = []
        for pk in picks:
            sid, idx = pk.get("id"), pk.get("idx")
            if sid in by_name:
                j = by_name[sid]
                if idx is not None and j != idx:
                    print(f"[manifest] {key} '{sid}': index moved {idx} -> {j} "
                          f"(different tiling/split); using the NAME, which is stable.")
                idxs.append(j)
                continue
            # No name match. Only trust the raw index if the configuration it was recorded
            # under is the configuration we are reading now.
            why = _config_mismatch(dkey, loader, len(names), man)
            if why:
                print(f"[manifest] DROP {key} '{sid}' (idx {idx}): not found by name, and "
                      f"the index is not transferable — {why}")
                continue
            if idx is None or not 0 <= idx < len(names):
                print(f"[manifest] DROP {key} '{sid}': idx {idx} out of range "
                      f"(dataset has {len(names)})")
                continue
            print(f"[manifest] {key} '{sid}': no name match; falling back to idx {idx} "
                  f"-> {names[idx]} (loader config matches)")
            idxs.append(idx)
        if args.max_per_dataset and len(idxs) > args.max_per_dataset:
            print(f"[manifest] {key}: keeping the first {args.max_per_dataset} of "
                  f"{len(idxs)} pick(s) (--max_per_dataset)")
            idxs = idxs[:args.max_per_dataset]
        if idxs:
            plan.append((dkey, idxs))
    return plan


def _config_mismatch(dkey, loader, n_here, man):
    """Reason the manifest's indices are not transferable to the current args, or ''."""
    if not loader:
        return ("the manifest predates the `loader` block, so the configuration it was "
                "written under is unknown (regenerate it with figures/plot_models.py)")
    if loader.get("eval_img_size") not in (None, args.img_size):
        return f"img_size {loader['eval_img_size']} != {args.img_size}"
    if dkey == "s11" and loader.get("s11_split") not in (None, args.split):
        return (f"s11 split '{loader['s11_split']}' != '{args.split}' "
                f"(pass --split {loader['s11_split']})")
    if dkey == "sturm" and loader.get("sturm_normalize") not in (None, args.sturm_normalize):
        return f"sturm normalize '{loader['sturm_normalize']}' != '{args.sturm_normalize}'"
    return ""


# Models: the same set eval_jetson.py evaluates

_STANDARD_NAMES = [
    ("model_float.onnx", "Float BN-folded"),
    # Prefer the no-output-QDQ export; see eval_jetson.py for why the plain
    # model_qdq_trt.onnx collapses flood_IoU under per-tensor output scales.
    ("model_qdq_trt_no_out.onnx", "W8-INT8 (ORT)"),
    ("model_qdq_trt.onnx", "W8-INT8 (ORT)"),
]
_AUTO_ENGINE_NAMES = [
    ("model_fp32.engine", "TRT-FP32"),
    ("model_fp16.engine", "TRT-FP16"),
    ("model_int8_qat.engine", "TRT-INT8"),
]


class _TRTBackend:
    """TRT 8.4 (JetPack 5.x) execute_async_v2 path, not the TRT 10 v3 API.

        Duplicated from eval_jetson.py rather than imported: that file evaluates at import
        time, so importing it here would run a full evaluation before drawing anything.
    """

    def __init__(self, engine_path, label, img_size):
        if not _TRT_AVAILABLE:
            raise RuntimeError("TRT Python API not available")
        self._cuda = _pycuda
        self.path = engine_path
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self._ctx = engine.create_execution_context()
        self._stream = _pycuda.Stream()
        in_idx = engine.get_binding_index("image")
        out_idx = engine.get_binding_index("logits")
        self._h_in = _pycuda.pagelocked_empty((1, IN_CHANNELS, img_size, img_size), np.float32)
        self._h_out = _pycuda.pagelocked_empty((1, 2, img_size, img_size), np.float32)
        self._d_in = _pycuda.mem_alloc(self._h_in.nbytes)
        self._d_out = _pycuda.mem_alloc(self._h_out.nbytes)
        self._bindings = [None, None]
        self._bindings[in_idx] = int(self._d_in)
        self._bindings[out_idx] = int(self._d_out)
        print(f"  {label:<18} {engine_path}  ({os.path.getsize(engine_path)/1e6:.1f} MB)")

    def __call__(self, x_1chw):
        np.copyto(self._h_in, x_1chw)
        self._cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
        self._ctx.execute_async_v2(self._bindings, self._stream.handle)
        self._cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
        self._stream.synchronize()
        return self._h_out.copy()


class _ORTBackend:
    def __init__(self, path, label):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.path = path
        print(f"  {label:<18} {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    def __call__(self, x_1chw):
        return self.sess.run(None, {self.in_name: x_1chw})[0]


def build_models():
    models = []
    if not args.no_onnx:
        if args.model:
            for m in args.model:
                models.append((os.path.basename(m), _ORTBackend(os.path.abspath(m),
                                                                os.path.basename(m))))
        else:
            seen = set()
            for fname, label in _STANDARD_NAMES:
                path = os.path.join(args.model_dir, fname)
                if label not in seen and os.path.isfile(path):
                    models.append((label, _ORTBackend(path, label)))
                    seen.add(label)
    if _TRT_AVAILABLE:
        loaded = set()
        for fname, label in _AUTO_ENGINE_NAMES:
            path = os.path.join(args.model_dir, fname)
            if os.path.isfile(path):
                try:
                    models.append((label, _TRTBackend(path, label, args.img_size)))
                    loaded.add(os.path.realpath(path))
                except Exception as e:
                    print(f"[WARNING] Could not load {label}: {e}")
        if args.engine and os.path.realpath(args.engine) not in loaded:
            try:
                models.append(("TRT-INT8" if not loaded else os.path.basename(args.engine),
                               _TRTBackend(args.engine, "TRT (--engine)", args.img_size)))
            except Exception as e:
                print(f"[WARNING] Could not load --engine: {e}")
    elif args.engine:
        print("[WARNING] --engine given but TRT unavailable; skipping engine columns.")
    if not models:
        p.error(f"No models to draw. Nothing in --model_dir {args.model_dir}, no --engine.")
    return models


# Rendering

# The colour transform comes from figures/qual_rgb.py, same as in
# figures/fig_qualitative_batches.py. This file used to have its own copy, and that's how a
# device panel and a workstation panel of the same tile ended up looking different: the local
# copy stretched each channel on its own percentiles, a per-scene white balance that destroys
# the band ratios carrying the colour and invents casts on scenes with unusual band statistics.
# qual_rgb uses one black/white point for all three bands in reflectance space, which is true
# colour. Its docstring explains the three variants.
from figures.qual_rgb import TRUE_COLOUR_CEILING, to_rgb  # noqa: E402

# The --rgb_fixed default is a literal up in the parser (this module is not importable that
# early). If the shared ceiling ever moves, the device panel must move with it, or it stops
# being comparable to the workstation panel it exists to reproduce.
assert p.get_default("rgb_fixed") == TRUE_COLOUR_CEILING, (
    f"--rgb_fixed default {p.get_default('rgb_fixed')} != "
    f"figures.qual_rgb.TRUE_COLOUR_CEILING {TRUE_COLOUR_CEILING}")


def flood_iou(pred, gt):
    v = gt != -1
    inter = ((pred == 1) & (gt == 1) & v).sum()
    union = (((pred == 1) | (gt == 1)) & v).sum()
    return float(inter) / max(float(union), 1.0)


def predict(model, x_6hw):
    logits = model(x_6hw[None].numpy().astype(np.float32))
    return np.asarray(logits)[0].argmax(0)


def main():
    ds, needs_adapter = build_dataset()
    names = sample_names(ds)
    adapter = sturm_s2_adapter(target_channels=6) if needs_adapter else None

    if args.list:
        lim = args.list_limit or len(names)
        print(f"# {args.dataset}: {len(names)} samples "
              f"(STURM normalize={args.sturm_normalize}, "
              f"WF={'flood+permanent' if args.wf_include_permanent else 'flood-only'})")
        hdr = f"{'idx':>6}  {'filename':<52} {'scene':<26}"
        print(hdr + ("  water%  scorable%" if args.stats else ""))
        for i, n in enumerate(names[:lim]):
            line = f"{i:>6}  {n:<52} {scene_of(n)[:26]:<26}"
            if args.stats:
                m = ds[i]["mask"].numpy()
                v = m != -1
                water = float((m[v] == 1).mean()) * 100 if v.any() else 0.0
                line += f"  {water:6.1f}  {100*float(v.mean()):9.1f}"
            print(line)
        if lim < len(names):
            print(f"# ... {len(names) - lim} more (raise --list_limit)")
        return

    if not args.select and not args.manifest:
        p.error("nothing to draw: pass --select <indices/filenames>, --manifest <json>, "
                "or --list to browse")

    print(f"Model dir: {args.model_dir}")
    print(f"  pool:    {pool_of(args.model_dir)}")
    models = build_models()

    plan = manifest_plan() if args.manifest else [(args.dataset,
                                                   resolve(args.select.split(","), names))]
    rows = []
    for dkey, idxs in plan:
        d, needs_ad = build_dataset(dkey)
        ad = sturm_s2_adapter(target_channels=6) if needs_ad else None
        nm = sample_names(d, dkey)
        d_c = build_dataset(dkey, **_SIM)[0] if _AXIS else None
        print(f"{dkey}: {len(idxs)} tile(s)")
        for i in idxs:
            s = d[i]
            x = ad(s["image"][None])[0] if ad else s["image"]
            xc = None
            if d_c is not None:
                sc = d_c[i]
                xc = ad(sc["image"][None])[0] if ad else sc["image"]
            gt = s["mask"].numpy()
            # Always predict both conditions, not just the corrupted one. Until 2026-08-17 an
            # active axis meant predicting on the corrupted input only, which can't be merged:
            # the workstation robustness panel puts each model's clean prediction right above
            # its perturbed one (rows 1/3 clean, rows 2/4 perturbed), so a device column with only
            # the perturbed half fills two of the four rows and leaves two empty. The device's
            # own panel also couldn't show any degradation, and that's what the panel is for.
            preds = [(label, predict(m, x)) for label, m in models]
            preds_c = ([(label, predict(m, xc)) for label, m in models]
                       if xc is not None else None)
            rows.append(dict(idx=i, name=nm[i], scene=scene_of(nm[i], dkey), dataset=dkey,
                             x=x, xc=xc, gt=gt, preds=preds, preds_c=preds_c,
                             ious=[(label, flood_iou(pr, gt)) for label, pr in preds],
                             ious_c=(None if preds_c is None else
                                     [(label, flood_iou(pr, gt)) for label, pr in preds_c])))
            print(f"  [{i}] {nm[i]}")
            print("      clean      " +
                  "  ".join(f"{l}: {v:.3f}" for l, v in rows[-1]["ious"]))
            if rows[-1]["ious_c"] is not None:
                print(f"      {_AXIS}={_LEVEL:<5.2f}" + "  ".join(
                    f"{l}: {v:.3f}" for l, v in rows[-1]["ious_c"]))
    if not rows:
        p.error("no drawable rows (see the warnings above)")

    suffix = f"_{_AXIS}{_LEVEL:.2f}" if _AXIS else ""
    stem = "manifest" if args.manifest else args.dataset
    out = args.out or os.path.join(_REPO_ROOT, "assets",
                                   f"qual_jetson_{stem}{suffix}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if args.per_dataset:
        # One figure per dataset. A 14-pick manifest drawn as one sheet is 12 device rows
        # (GDACS is not on the device) x 5 columns, which is a page nobody reads; the same
        # picks as three 4-row sheets are three usable figures. The .npz is written per
        # sheet too, so each merges independently.
        base, ext = os.path.splitext(out)
        for dkey in dict.fromkeys(r["dataset"] for r in rows):
            draw([r for r in rows if r["dataset"] == dkey], f"{base}_{dkey}{ext}")
    else:
        draw(rows, out)


def _display_rows(rows):
    """-> [(tile index, row, condition)], one figure row per condition and not per tile.

        With an axis active each tile gets two rows, clean above perturbed, like the workstation
        robustness layout (figures/fig_qualitative_batches.py). Only that layout shows the
        degradation. The previous one had both inputs side by side and a single prediction
        column, so you could see the input change but not what it did to the models.
    """
    out = []
    for i, r in enumerate(rows):
        out.append((i, r, "clean"))
        if r.get("preds_c") is not None:
            out.append((i, r, _AXIS))
    return out


def draw(rows, out):
    """Panel via matplotlib; .npz fallback so a headless device still produces something."""
    npz_path = os.path.splitext(out)[0] + ".npz"
    rgbs = []
    # Default (--rgb_fixed on) is the standard Sentinel-2 True Color map min(rho/K, 1) with
    # K = 0.40, the same for every tile and every corruption level. It replaced the per-scene
    # percentile stretch on 2026-08-13 (see figures/qual_rgb.TRUE_COLOUR_CEILING). A per-scene
    # black point doesn't keep band ratios, so hue drifted between tiles, and the stretch
    # limits depended on the rest of the row, so the same clean tile looked different in a haze
    # panel than in a gain panel. --rgb_fixed 0 gives the old behaviour.
    # K has to equal the workstation figure's ceiling, otherwise the device panel isn't
    # comparable to the one it reproduces. The assert next to the import above checks that.
    K = args.rgb_fixed if args.rgb_fixed > 0 else None
    for r in rows:
        if K is not None:
            rgbs.append((to_rgb(r["x"], fixed=K)[0],
                         None if r["xc"] is None else to_rgb(r["xc"], fixed=K)[0]))
            continue
        rgb0, pc0 = to_rgb(r["x"])
        if r["xc"] is not None:
            _cc, pcc = to_rgb(r["xc"])
            # SHARED stretch over clean+corrupted: the clean tile's own stretch blows the
            # corrupted column out, a per-image stretch hides the shift entirely.
            pcts = [(min(pc0[k][0], pcc[k][0]), max(pc0[k][1], pcc[k][1])) for k in range(3)]
            rgbs.append((to_rgb(r["x"], pcts)[0], to_rgb(r["xc"], pcts)[0]))
        else:
            rgbs.append((rgb0, None))

    if args.dump_npz:
        # Dump format 2 (2026-08-17): `preds` is the clean prediction and `preds_corrupt` the
        # perturbed one. Format 1 had a single `preds` holding whatever condition was active,
        # so merging such a file as clean would put hazed predictions in the clean row and
        # still look plausible. figures/stitch/fig_int8_qualitative_merge.py requires this key,
        # so an old file gives an error instead of a wrong figure.
        np.savez_compressed(
            npz_path,
            format=np.array(["qual_jetson_v2"]),
            names=np.array([r["name"] for r in rows]),
            scenes=np.array([r["scene"] for r in rows]),
            datasets=np.array([r["dataset"] for r in rows]),
            idx=np.array([r["idx"] for r in rows]),
            labels=np.array([l for l, _ in rows[0]["preds"]]),
            rgb=np.stack([a for a, _ in rgbs]),
            rgb_corrupt=(np.stack([b for _, b in rgbs]) if rgbs[0][1] is not None
                         else np.zeros(0)),
            gt=np.stack([r["gt"] for r in rows]),
            preds=np.stack([np.stack([pr for _, pr in r["preds"]]) for r in rows]),
            preds_corrupt=(np.stack([np.stack([pr for _, pr in r["preds_c"]])
                                     for r in rows]) if rows[0]["preds_c"] else np.zeros(0)),
            ious=np.array([[v for _, v in r["ious"]] for r in rows], dtype=np.float32),
            ious_corrupt=(np.array([[v for _, v in r["ious_c"]] for r in rows],
                                   dtype=np.float32) if rows[0]["ious_c"] else np.zeros(0)),
            axis=np.array([_AXIS or "none"]), level=np.array([_LEVEL if _LEVEL else 0.0]),
            # Provenance, so the merge can refuse a mismatch instead of assuming one.
            model_dir=np.array([os.path.basename(os.path.normpath(args.model_dir))]),
            pool=np.array([pool_of(args.model_dir)]),
            manifest=np.array([os.path.basename(args.manifest) if args.manifest else ""]),
            img_size=np.array([args.img_size]),
            sturm_normalize=np.array([args.sturm_normalize]),
            wf_include_permanent=np.array([bool(args.wf_include_permanent)]),
            rgb_ceiling=np.array([float(args.rgb_fixed)]))
        print(f"[saved] {npz_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception as e:
        print(f"[WARNING] matplotlib unavailable ({e}) — no PNG written.")
        if not args.dump_npz:
            np.savez_compressed(npz_path, **{"note": np.array(["rerun with --dump_npz"])})
        print("  Re-run with --dump_npz and draw the panel on the workstation.")
        return

    if args.paper_layout:
        _draw_paper(rows, rgbs, out, plt)
        return

    water = ListedColormap(["#C8C8C8", "#0072B2"])
    n_model = len(rows[0]["preds"])
    lvl = f"{_AXIS} = {_LEVEL:.2f}" if _AXIS else None
    disp = _display_rows(rows)
    titles = ["input", "ground truth"] + [l for l, _ in rows[0]["preds"]]
    ncol = len(titles)
    fig, axes = plt.subplots(len(disp), ncol, figsize=(2.24 * ncol, 2.35 * len(disp)),
                             squeeze=False)
    for r_i, (t_i, r, cond) in enumerate(disp):
        gtm = np.where(r["gt"] == -1, np.nan, r["gt"].astype(float))
        rgb = rgbs[t_i][0 if cond == "clean" else 1]
        preds = r["preds"] if cond == "clean" else r["preds_c"]
        ious = r["ious"] if cond == "clean" else r["ious_c"]
        ims = [rgb, gtm] + [pr for _, pr in preds]
        for k, im in enumerate(ims):
            ax = axes[r_i][k]
            ax.imshow(im, cmap=None if k == 0 else water,
                      vmin=None if k == 0 else 0, vmax=None if k == 0 else 1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r_i == 0:
                ax.set_title(titles[k], fontsize=8.5)
        for m_i, (_l, v) in enumerate(ious):
            axes[r_i][ncol - n_model + m_i].set_xlabel(f"flood-IoU {v:.2f}", fontsize=8)
        tag = "clean" if cond == "clean" else lvl
        axes[r_i][0].set_ylabel(f"[{r['idx']}] {r['scene'][:22]}\n{tag}", fontsize=6.5)
        if cond != "clean":                       # frame the perturbed row of each pair
            for ax in axes[r_i]:
                for sp in ax.spines.values():
                    sp.set_color("#B07AA1"); sp.set_linewidth(1.1)
    fig.legend(handles=[Patch(facecolor="#C8C8C8", edgecolor="0.5", label="land"),
                        Patch(facecolor="#0072B2", edgecolor="0.5", label="water/flood"),
                        Patch(facecolor="white", edgecolor="0.5", label="ignore")],
               loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    prot = (f"STURM normalize={args.sturm_normalize}" if args.dataset == "sturm" else
            (f"WF {'flood+permanent' if args.wf_include_permanent else 'flood-only'}"
             if args.dataset == "wf" else f"Sen1Floods11 {args.split}"))
    fig.suptitle(f"{args.dataset.upper()} · Jetson deployed models · {prot}", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"[saved] {out}")


def _draw_paper(rows, rgbs, out, plt):
    """The qual_N*.png layout, with the deployed models instead of the workstation ones.

        Column order and styling follow figures/plot_models.py::save_figure with
        --no_agreement --paper_clean, so the device panel and qual_N2500_fullpool.png can sit
        next to each other and only the model columns differ:

            <dataset>
    RGB | Ground truth | <one column per deployed model, flood-IoU in title>

        Differences to the device's own layout, taken over from plot_models: axes fully off
        (no ticks or frame), flood-IoU in the title instead of an xlabel, the "Blues" colormap
        with vmin=-1 on the ground-truth column so the ignore class shows as a third tone, and
        no suptitle or legend so the PNG doesn't need cropping.
    """
    fs_title = round(9 * args.font_scale)
    disp = _display_rows(rows)
    ncol = 2 + len(rows[0]["preds"])
    fig, axes = plt.subplots(len(disp), ncol, figsize=(3.7 * ncol, 3.6 * len(disp)),
                             squeeze=False)
    lvl = f"{_AXIS} = {_LEVEL:.2f}" if _AXIS else None
    for r_i, (t_i, r, cond) in enumerate(disp):
        ax = axes[r_i]
        clean = cond == "clean"
        at = "" if clean else f" @ {lvl}"
        col = 0
        rgb_tag = "" if not _AXIS else (" (clean)" if clean else at)
        ax[col].imshow(rgbs[t_i][0 if clean else 1])
        ax[col].set_title(f"{r['dataset'].upper()}\nRGB{rgb_tag}", fontsize=fs_title)
        col += 1
        ax[col].imshow(r["gt"], cmap="Blues", vmin=-1, vmax=1, interpolation="nearest")
        ax[col].set_title("Ground truth", fontsize=fs_title); col += 1
        for (label, pred), (_l, iou) in zip(r["preds"] if clean else r["preds_c"],
                                            r["ious"] if clean else r["ious_c"]):
            ax[col].imshow(pred, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax[col].set_title(f"{label}{at}\nflood-IoU={iou:.2f}", fontsize=fs_title)
            col += 1
        for a in ax:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
