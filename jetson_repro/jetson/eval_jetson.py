"""Evaluate ONNX and TensorRT models on the Jetson Xavier NX.

S11 defaults to the test split (--s11_split val selects source validation) and uses
the rasterio-only loader. STURM uses its
original DN inverse and the six-band adapter. WorldFloods ignores clouds. Labels follow
--protocol: surface_water (default; all water, as Sen1Floods11) or the flood-only tables.
Both external datasets use 512x512 inputs and S11 stats.

Micro scores pool confusion cells. Macro scores average flood-containing
scenes (STURM flood maps; WorldFloods AOIs). Robustness applies gain and haze
in reflectance space before normalization and measures clean-minus-minimum
over the selected envelope. Keep the clean anchor in that envelope.

TensorRT uses JetPack's Python bindings; see jetson/README.md for setup,
engine building and benchmarking. --quick limits the number of batches."""

import gc
import os
import statistics
import sys
import time

# Import order matters on Jetson
# torch must come before albumentations/skimage to prevent libcublas ordering
# issues. tensorrt (if used) must come before numpy (patches ABCMeta).
import torch
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as _pycuda
    import pycuda.autoinit   # initialises CUDA context - required side-effect
    _TRT_AVAILABLE = True
except Exception as _trt_err:
    trt            = None       # type: ignore[assignment]
    _pycuda        = None       # type: ignore[assignment]
    _TRT_AVAILABLE = False
    print(f"[WARNING] TRT Python API unavailable: {_trt_err}")
    print("  Set PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH and retry.")
    print("  ORT accuracy evaluation will still run.")

# Project root on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Point data.sen1floods11 at the terratorch-free version. Has to happen before anything
# imports data.sen1floods11, including data.ood_datasets (which imports NORM_MEAN/NORM_STD
# from it at module level).
import data.sen1floods11_nodep as _s11_nodep
sys.modules["data.sen1floods11"] = _s11_nodep  # type: ignore[assignment]

import argparse

import onnxruntime as ort

from utils.metrics import accumulate_metrics, finalize_metrics
from data.channel_adapter import sturm_s2_adapter, adapt_batch

# Argument parsing

_DEFAULT_DIR = os.path.join(_REPO_ROOT, "runs/deploy_qat/docker_qat")

parser = argparse.ArgumentParser(
    description="Jetson Xavier evaluation — ORT (always) + optional TRT engine",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("--model_dir", default=_DEFAULT_DIR,
                    help="Dir containing model_float/qdq/qdq_trt.onnx")
parser.add_argument("--model", default=None,
                    help="Evaluate a single ONNX file instead of the full set")
parser.add_argument("--engine", default=None,
                    help="Path to compiled TRT .engine (built with trtexec --int8)")
parser.add_argument("--quick", type=int, default=0,
                    help="Evaluate only first N batches per dataset (0 = full)")
parser.add_argument("--n_latency", type=int, default=50,
                    help="Timed runs for latency measurement (default 50)")
parser.add_argument("--warmup", type=int, default=10,
                    help="Warm-up runs discarded before timing (default 10)")
parser.add_argument("--no_latency", action="store_true",
                    help="Skip latency measurement — accuracy only")
parser.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader workers (default 2; use 0 where worker IPC is unavailable)")
# Data roots - defaults match the standard Jetson layout from deploy_jetson.sh
parser.add_argument("--s11_root",    default=os.path.expanduser("~/data/sen1floods11"),
                    help="Sen1Floods11 root (default ~/data/sen1floods11)")
parser.add_argument("--s11_split", choices=["test", "val"], default="test",
                    help="S11 split: test for reporting (default), val for source selection/checks")
parser.add_argument("--sturm_root",  default=os.path.expanduser("~/data/STURM-Flood/Dataset"),
                    help="STURM-Flood Dataset dir (default ~/data/STURM-Flood/Dataset)")
parser.add_argument("--sturm_normalize", default="original",
                    choices=["percentile", "clip", "original"],
                    help="STURM normalization protocol (data/ood_datasets.py). "
                         "'original' (default since 2026-08-11) is the STURM-paper-documented "
                         "DN inverse (0.04+0.12*v) + NaN guard, and is the ONLY protocol the published "
                         "results use (eval_results.json, evaluation/README.md) — the "
                         "Jetson default used to be 'percentile', which silently produced numbers "
                         "not comparable to any table here. Pass 'percentile' only to reproduce "
                         "older Jetson numbers.")
parser.add_argument("--s1s2w_root",  default=os.path.expanduser("~/data/S1S2Water"),
                    help="S1S2Water root (default ~/data/S1S2Water)")
parser.add_argument("--wf_root",     default=os.path.expanduser("~/data/WorldFloodsv2"),
                    help="WorldFloods-v2 root (default ~/data/WorldFloodsv2)")
parser.add_argument("--protocol", default="surface_water",
                    choices=["surface_water", "flood_p3", "flood_paper"],
                    help="Label protocol for STURM and WF (data/ood_datasets.py). 'surface_water' "
                         "(default since 2026-10-15): every water pixel counts, as in Sen1Floods11 "
                         "(WF ground-truth water including permanent water; STURM classes 1-5), non-finite STURM "
                         "pixels ignored, WF edge windows zero-padded. 'flood_p3': flood only "
                         "(WF permanent water and STURM classes 3-5 ignored), same geometry, the "
                         "revision P3 tables. 'flood_paper': flood only with resized WF edge windows "
                         "and no STURM finite-support mask, the submitted paper's tables.")
# Backward-compat alias: --data_root <path> acts as --s11_root <path>
parser.add_argument("--data_root",   default=None,
                    help="[deprecated] alias for --s11_root")
# Dataset selection
parser.add_argument("--datasets", default="s11,sturm,wf",
                    help="Comma-separated subset of {s11,sturm,wf,s1s2w} to evaluate. "
                         "Default 's11,sturm,wf' (S1S2Water is saturated and no longer reported).")
# Robustness sweep: does the robustness measured on the workstation survive INT8
# quantization? Perturbations are applied in reflectance space before the model's fixed S11
# normalization, the same way for STURM and WorldFloods, so both OOD sets are on the same
# physical axis (data/ood_datasets.py).
parser.add_argument("--robustness", action="store_true",
                    help="Run the gain + haze robustness sweep on each OOD dataset.")
parser.add_argument("--gains", default="0.80,1.00,1.20,1.30",
                    help="Radiometric gain levels for --robustness (reflectance multiplier).")
parser.add_argument("--hazes", default="0.00,0.05,0.10,0.20",
                    help="Haze levels for --robustness (blue-band additive path radiance, "
                         "wavelength-weighted; Makarau TGARS 2014).")
parser.add_argument("--robust_datasets", default="sturm,wf",
                    help="Which OOD sets the robustness sweep runs on.")
parser.add_argument("--json_out", default=None,
                    help="Write all accuracy/latency/robustness results to this JSON path.")
args = parser.parse_args()
if args.num_workers < 0:
    parser.error("--num_workers must be nonnegative")

_SEL = {s.strip() for s in args.datasets.split(",") if s.strip()}
_ROBUST_SEL = {s.strip() for s in args.robust_datasets.split(",") if s.strip()}
_GAINS = [float(x) for x in args.gains.split(",") if x.strip()]
_HAZES = [float(x) for x in args.hazes.split(",") if x.strip()]

if args.data_root is not None:
    args.s11_root = args.data_root

IMG_SIZE    = 512
IN_CHANNELS = 6
_LAT_SHAPE  = (1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


# Unified memory (RSS delta)

def _rss_mb() -> float:
    """Process RSS in MB from /proc/self/status.  Approximates unified memory."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


# Model list

_STANDARD_NAMES = [
    ("model_float.onnx",           "Float BN-folded    "),
    # Prefer model_qdq_trt_no_out.onnx: per-tensor activation QDQ (TRT 8.4 compatible) but
    # no Conv output QDQ. The standard model_qdq_trt.onnx has per-tensor output QDQ, and its
    # background-dominated scale crushes the minority-class channels, flood_IoU drops to ~0.
    # TRT 8.4 doesn't need output QDQ, it infers the Conv output precision from the next
    # input QDQ. Falls back to model_qdq_trt.onnx when only old exports exist (_seen_labels
    # dedup means only the first match is used).
    ("model_qdq_trt_no_out.onnx",  "W8-INT8 (ORT)      "),
    ("model_qdq_trt.onnx",         "W8-INT8 (ORT)      "),
]

if args.model:
    onnx_models = [(os.path.basename(args.model), os.path.abspath(args.model))]
else:
    onnx_models = []
    _seen_labels: set = set()
    for fname, label in _STANDARD_NAMES:
        if label in _seen_labels:
            continue  # prefer first match (static before original)
        path = os.path.join(args.model_dir, fname)
        if os.path.isfile(path):
            onnx_models.append((label, path))
            _seen_labels.add(label)

_AUTO_ENGINE_NAMES = [
    ("model_fp32.engine",     "TRT-FP32-engine"),
    ("model_fp16.engine",     "TRT-FP16-engine"),
    ("model_int8_qat.engine", "TRT-INT8-engine"),
]


def _any_auto_engine_exists() -> bool:
    return any(
        os.path.isfile(os.path.join(args.model_dir, fname))
        for fname, _ in _AUTO_ENGINE_NAMES
    )


if not onnx_models and not args.engine and not _any_auto_engine_exists():
    raise FileNotFoundError(
        f"No ONNX files found in {args.model_dir} and no --engine given.\n"
        "Pass --model <path>, --model_dir <dir>, or --engine <path>."
    )

print("Models to evaluate:")
for label, path in onnx_models:
    print(f"  {label}  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


# TRT engine backend (TRT 8.4 API - execute_async_v2 + bindings list)
# The Jetson Xavier ships with JetPack 5.0 / TRT 8.4.x.
# Do not use execute_async_v3 / set_tensor_address here - that is the TRT 10 API
# used by quantization/trt_eval.py on the workstation.

class _TRTBackend:
    """Wraps a trtexec-compiled .engine for batch evaluation.

        The engine was built for input shape [1, 6, 512, 512] (batch=1 fixed).
        eval() loops over images in each DataLoader batch one at a time so the
        caller sees the same interface as eval_onnx.
    """

    def __init__(self, engine_path: str, label: str = "TRT-engine"):
        if not _TRT_AVAILABLE:
            raise RuntimeError("TRT Python API not available (see warning above)")
        self._cuda = _pycuda
        self._engine_path = engine_path
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        self._ctx    = engine.create_execution_context()
        self._stream = _pycuda.Stream()

        # ONNX exported with input_names=["image"], output_names=["logits"]
        in_idx  = engine.get_binding_index("image")
        out_idx = engine.get_binding_index("logits")

        in_shape  = (1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
        out_shape = (1, 2, IMG_SIZE, IMG_SIZE)
        self._h_in  = _pycuda.pagelocked_empty(in_shape,  dtype=np.float32)
        self._h_out = _pycuda.pagelocked_empty(out_shape, dtype=np.float32)
        self._d_in  = _pycuda.mem_alloc(self._h_in.nbytes)
        self._d_out = _pycuda.mem_alloc(self._h_out.nbytes)
        self._bindings = [None, None]
        self._bindings[in_idx]  = int(self._d_in)
        self._bindings[out_idx] = int(self._d_out)

        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"  {label}  {engine_path}  ({size_mb:.1f} MB)")

    def infer_one(self, image_1chw: np.ndarray) -> np.ndarray:
        """Run one [1, 6, H, W] float32 image, returns [1, 2, H, W] logits."""
        np.copyto(self._h_in, image_1chw)
        self._cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
        self._ctx.execute_async_v2(self._bindings, self._stream.handle)
        self._cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
        self._stream.synchronize()
        return self._h_out.copy()


_trt_backends: list = []    # list of (label: str, backend: _TRTBackend)
_loaded_engine_realpaths: set = set()

if _TRT_AVAILABLE:
    print("TRT engines (auto-detected from model_dir):")
    for _fname, _elabel in _AUTO_ENGINE_NAMES:
        _epath = os.path.join(args.model_dir, _fname)
        if os.path.isfile(_epath):
            try:
                _be = _TRTBackend(_epath, _elabel)
                _trt_backends.append((_elabel, _be))
                _loaded_engine_realpaths.add(os.path.realpath(_epath))
            except Exception as _e:
                print(f"[WARNING] Could not load {_elabel} ({_fname}): {_e}")
        else:
            print(f"  {_elabel:<20}  not found — skipping ({_fname})")
    if not any(os.path.isfile(os.path.join(args.model_dir, f)) for f, _ in _AUTO_ENGINE_NAMES):
        print("  (none found)")

    if args.engine:
        _explicit_real = os.path.realpath(os.path.abspath(args.engine))
        if _explicit_real not in _loaded_engine_realpaths:
            _existing_labels = {l for l, _ in _trt_backends}
            _elabel = "TRT-INT8-engine"
            if _elabel in _existing_labels:
                _elabel = f"TRT-engine ({os.path.basename(args.engine)})"
            try:
                _be = _TRTBackend(args.engine, _elabel)
                _trt_backends.append((_elabel, _be))
                _loaded_engine_realpaths.add(_explicit_real)
            except Exception as _e:
                print(f"[WARNING] Could not load --engine {args.engine}: {_e}")
elif args.engine:
    print("[WARNING] --engine given but TRT Python API unavailable; skipping TRT eval.")


# Latency helpers

def _latency_ort(sess) -> float:
    in_name = sess.get_inputs()[0].name
    dummy   = np.zeros(_LAT_SHAPE, dtype=np.float32)
    times: list = []
    for i in range(args.warmup + args.n_latency):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sess.run(None, {in_name: dummy})
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _latency_trt(backend) -> float:
    dummy = np.zeros(_LAT_SHAPE, dtype=np.float32)
    times: list = []
    for i in range(args.warmup + args.n_latency):
        backend._stream.synchronize()
        t0 = time.perf_counter()
        backend.infer_one(dummy)
        backend._stream.synchronize()
        if i >= args.warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


# Evaluation helpers

def _scene_ious(conf):
    """Per-scene (mIoU, flood_IoU, has_GT_flood), same convention as
        evaluation/perscene_collect.py, so Jetson macro numbers compare to the workstation cache.

          * IoU_c = TP_c / max(TP_c + FP_c + FN_c, 1). An empty class scores 0, it isn't
            dropped. finalize_metrics() returns NaN instead and averages the remaining classes,
            which gives mIoU=1.0 on an all-flood scene where the workstation reports 0.5. Only
            the macro path uses this function, micro still uses finalize_metrics.
          * a scene only counts if it has GT flood pixels (row 1 of the confusion matrix is
            not empty). A flood prediction alone isn't enough: scoring a scene with only false
            positives as 0.0 would pull macro down against a workstation number that never
            included that scene.
    """
    c  = conf.double()          # float64: a scene can hold 10^8 pixels, and float32 costs
    tp = c.diag()               # ~1e-9 of agreement against the numpy-float64 workstation
    union = (tp + (c.sum(0) - tp) + (c.sum(1) - tp)).clamp(min=1.0)
    v = tp / union
    return float(v.mean()), float(v[1]), bool(c[1].sum() > 0)


def _finalize_both(conf, per_scene):
    """Metrics dict with both aggregations.

        micro_*: from the single pooled confusion matrix (the historical numbers, unchanged)
        macro_*: mean of per-scene IoUs, only scenes with GT flood (see _scene_ious)
        Top-level 'mIoU'/'iou_per_class' stay micro so existing tables and logs mean the same.
    """
    out = finalize_metrics(conf) if conf is not None else {}
    if out:
        out["micro_mIoU"] = out.get("mIoU")
        out["micro_flood_IoU"] = (out.get("iou_per_class") or [float("nan")] * 2)[1]
    ms, fs = [], []
    for c in per_scene.values():
        m, f, has_gt_flood = _scene_ious(c)
        if has_gt_flood:
            ms.append(m); fs.append(f)
    if ms:
        out["macro_scene_mIoU"] = float(np.mean(ms))
        out["macro_scene_flood_IoU"] = float(np.mean(fs))
        out["n_scenes_scored"] = len(ms)
        out["n_scenes_seen"] = len(per_scene)
    return out


def _acc_scene(per_scene, key, logits, mask, num_classes, ignore_index):
    per_scene[key] = accumulate_metrics(logits, mask, num_classes, ignore_index,
                                        per_scene.get(key))


def eval_onnx(onnx_path, batches, num_classes=2, ignore_index=-1, keyfn=None):
    """Evaluate an ONNX model on a stream of batches with onnxruntime (CPU).

        Accumulates the pooled (micro) confusion matrix and, when keyfn is given, one matrix
        per scene so macro-per-scene metrics come out of the same single pass.
    """
    sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    # model_qdq_trt.onnx has static batch=1; others have dynamic batch.
    _static_batch = sess.get_inputs()[0].shape[0]
    _per_image    = isinstance(_static_batch, int)
    conf = None; per_scene: dict = {}
    for batch in batches:
        imgs  = batch["image"].float().numpy()   # [B, 6, H, W]
        masks = batch["mask"].long()
        fns   = batch.get("filename", [""] * imgs.shape[0])
        if _per_image:
            for i in range(imgs.shape[0]):
                logits = torch.from_numpy(sess.run(None, {in_name: imgs[i: i + 1]})[0])
                conf = accumulate_metrics(logits, masks[i: i + 1], num_classes, ignore_index, conf)
                if keyfn is not None:
                    _acc_scene(per_scene, keyfn(fns[i]), logits, masks[i: i + 1],
                               num_classes, ignore_index)
        else:
            logits = torch.from_numpy(sess.run(None, {in_name: imgs})[0])
            conf = accumulate_metrics(logits, masks, num_classes, ignore_index, conf)
            if keyfn is not None:
                for i in range(imgs.shape[0]):
                    _acc_scene(per_scene, keyfn(fns[i]), logits[i: i + 1], masks[i: i + 1],
                               num_classes, ignore_index)
    return _finalize_both(conf, per_scene)


def eval_trt_engine(backend, dl_stream, num_classes=2, ignore_index=-1, keyfn=None):
    """Evaluate a TRT engine on a stream of batches (micro + macro in one pass)."""
    conf = None; per_scene: dict = {}
    for batch in dl_stream:
        imgs  = batch["image"].float().numpy()
        masks = batch["mask"].long()
        fns   = batch.get("filename", [""] * imgs.shape[0])
        for i in range(imgs.shape[0]):
            logits = torch.from_numpy(backend.infer_one(imgs[i: i + 1]))
            conf = accumulate_metrics(logits, masks[i: i + 1], num_classes, ignore_index, conf)
            if keyfn is not None:
                _acc_scene(per_scene, keyfn(fns[i]), logits, masks[i: i + 1],
                           num_classes, ignore_index)
    return _finalize_both(conf, per_scene)


def _row(label, m):
    """One line per model: MICRO (pixel-pooled) then MACRO (per-scene) when available."""
    iou   = m.get("iou_per_class", [0.0, 0.0])
    flood = iou[1] if len(iou) > 1 else 0.0
    bg    = iou[0] if len(iou) > 0 else 0.0
    line = (f"  {label}  micro: mIoU={m.get('mIoU', 0.0):.4f} flood={flood:.4f} bg={bg:.4f}")
    if "macro_scene_mIoU" in m:
        line += (f"   |  macro/scene: mIoU={m['macro_scene_mIoU']:.4f} "
                 f"flood={m['macro_scene_flood_IoU']:.4f} (n={m.get('n_scenes_scored','?')})")
    return line


def _stream(dl, adapter=None, max_batches=0):
    """Yields one batch at a time, the full dataset is never held in RAM.

        Peak memory is one batch (~12 MB at batch_size=2, 512x512). That matters on the
        Xavier, where CPU and GPU share the same physical memory. "filename" is passed along
        so the per-scene (macro) metrics are accumulated in the same pass as micro.
    """
    for i, b in enumerate(dl):
        if max_batches and i >= max_batches:
            break
        if adapter is not None:
            b = adapt_batch(b, adapter)
        yield {"image": b["image"].float(), "mask": b["mask"].long(),
               "filename": b.get("filename", [""] * b["image"].shape[0])}


# Macro (per-scene) metrics need scene groups. IoU is a ratio: the pooled micro
# number cannot recover the mean of scene IoUs. Accumulate per scene during the pass.
# STURM: tile 'EMSR279_03ALFARO_17_16_1_1.tif' -> floodmap_id (scene) from the dataset's
# Sentinel2_metadata.csv (stdlib csv, no pandas on the Jetson).
# WF: 'EMSR347_07ZOMBA_DEL_v2.tif#r0_c0' -> text before '#'.
# S11: one scene per file, the filename is the group.

_STURM_TILE2SCENE: dict = {}


def _load_sturm_scene_map(sturm_root):
    """tile_id -> floodmap_id from Sentinel2_metadata.csv. Empty dict if unavailable."""
    global _STURM_TILE2SCENE
    if _STURM_TILE2SCENE:
        return _STURM_TILE2SCENE
    import csv as _csv
    path = os.path.join(sturm_root, "Sentinel2_metadata.csv")
    try:
        with open(path, newline="") as f:
            for row in _csv.DictReader(f):
                if row.get("tile_id") and row.get("floodmap_id"):
                    _STURM_TILE2SCENE[row["tile_id"].strip()] = row["floodmap_id"].strip()
    except Exception as e:
        print(f"[WARNING] STURM scene map unavailable ({e}); macro metrics for STURM "
              f"will fall back to per-tile grouping.")
    return _STURM_TILE2SCENE


class _SceneKey:
    """filename -> scene id, and counts the tiles it couldn't map.

        An unmapped tile becomes its own group, and that does change the result: one macro
        sample turns into many and the macro average stops matching the workstation
        (evaluation/perscene_eval.py raises in this case). Here it's only reported, because on
        the device a missing metadata CSV shouldn't block the micro numbers, which are the
        ones the deployment table uses.
    """

    def __init__(self, fn, ds_key):
        self._fn = fn
        self._ds = ds_key
        self.n = self.miss = 0
        self._reported = False

    def __call__(self, fn):
        self.n += 1
        k = self._fn(str(fn))
        if k is None:
            self.miss += 1
            return str(fn)
        return k

    def report(self):
        if self._reported or not self.miss:
            return
        self._reported = True
        print(f"[WARNING] {self.miss}/{self.n} {self._ds} tiles had no scene key — each became "
              f"its own macro group, so macro/scene here is NOT comparable to the workstation "
              f"cache. Micro is unaffected. Check Sentinel2_metadata.csv on the device.")


def _scene_keyfn(ds_key):
    """filename -> scene grouping used for the MACRO aggregation."""
    if ds_key == "sturm":
        m = _load_sturm_scene_map(args.sturm_root)
        return _SceneKey(lambda fn: m.get(fn), ds_key)
    if ds_key == "wf":
        return _SceneKey(lambda fn: fn.split("#")[0], ds_key)
    return _SceneKey(lambda fn: fn, ds_key)


# _acc_results[label][dataset_key] = finalize_metrics() dict
_acc_results: dict = {}


def eval_all(dl_factory, adapter=None, ds_key="?"):
    """Evaluate every model by streaming the dataloader once per model.

        dl_factory() returns a fresh DataLoader per call so the dataset can be
        iterated multiple times without pre-loading into RAM.  Each model's ORT
        session and batch tensors are freed before the next model starts.
    """
    kf = _scene_keyfn(ds_key)
    for label, path in onnx_models:
        m = eval_onnx(path, _stream(dl_factory(), adapter, args.quick), keyfn=kf)
        m["protocol"] = args.protocol if ds_key in ("sturm", "wf") else "native_surface_water"
        if ds_key == "s1f":
            m["split"] = args.s11_split
        kf.report()
        print(_row(label, m))
        _acc_results.setdefault(label.strip(), {})[ds_key] = m
        gc.collect()
    for _lbl, _be in _trt_backends:
        m = eval_trt_engine(_be, _stream(dl_factory(), adapter, args.quick), keyfn=kf)
        m["protocol"] = args.protocol if ds_key in ("sturm", "wf") else "native_surface_water"
        if ds_key == "s1f":
            m["split"] = args.s11_split
        kf.report()
        print(_row(_lbl, m))
        _acc_results.setdefault(_lbl, {})[ds_key] = m
        gc.collect()


# Dataset factories. Same protocol as the workstation evaluation so the Jetson numbers can
# be compared to the paper directly: both OOD sets at IMG_SIZE=512 and normalized with the
# model's Sen1Floods11 L1C training stats. STURM uses normalize="original" (the DN inverse
# from the paper, 0.04+0.12*v) and sturm_s2_adapter for 9 to 6 bands. WorldFloods is the v2
# 509-pair release with tile_size=512, normalize="s11", cloud pixels excluded, bands
# [B2,B3,B4,B8A,B11,B12]. Gain/haze perturbations are applied in reflectance space before
# normalization on both sets, so the robustness axis is physically the same.

from data.sen1floods11 import get_eval_dataloader as _s11_get_eval_dataloader
from data.ood_datasets import get_sturm_flood_dataloader as _sturm_dl_fn
from data.ood_datasets import get_worldfloods_dataloader as _wf_dl_fn

_sturm_adapter = sturm_s2_adapter(target_channels=6)


def _s11_dl():
    return _s11_get_eval_dataloader(
        args.s11_root, split=args.s11_split, batch_size=2, num_workers=args.num_workers,
        eval_img_size=IMG_SIZE)


def _sturm_dl(gain=1.0, haze=0.0):
    # root = <STURM-Flood>/Dataset/ (contains Sentinel1/, Sentinel2/, *.csv)
    return _sturm_dl_fn(
        args.sturm_root, sensor="S2", batch_size=2, num_workers=args.num_workers,
        img_size=IMG_SIZE, normalize=args.sturm_normalize,
        sim_gain=gain, sim_haze=haze, protocol=args.protocol)


def _wf_dl(gain=1.0, haze=0.0):
    # WorldFloods-v2: scenes are tiled at native 10 m into 512x512 windows; --protocol sets
    # whether JRC permanent water counts as water (surface_water) or is ignored (flood_*).
    return _wf_dl_fn(
        args.wf_root, split="test", batch_size=2, num_workers=args.num_workers,
        img_size=IMG_SIZE, tile_size=IMG_SIZE, normalize="s11",
        sim_gain=gain, sim_haze=haze, protocol=args.protocol)


if "s11" in _SEL:
    print("\n=== Sen1Floods11 (%s split) ===" % args.s11_split)
    eval_all(_s11_dl, ds_key="s1f")

if "sturm" in _SEL:
    print("\n=== STURM S2 (OOD) @512, normalize=%s ===" % args.sturm_normalize)
    eval_all(_sturm_dl, adapter=_sturm_adapter, ds_key="sturm")

if "wf" in _SEL:
    print("\n=== WorldFloods-v2 (OOD) @512, normalize=s11 ===")
    eval_all(_wf_dl, ds_key="wf")

if "s1s2w" in _SEL:
    # S1S2Water is already 6-band Sentinel-2 L1C - no channel adapter needed.
    print("\n=== S1S2Water (OOD) ===")
    from data.ood_datasets import get_s1s2water_dataloader as _s1s2w_dl_fn
    eval_all(lambda: _s1s2w_dl_fn(args.s1s2w_root, batch_size=1, num_workers=args.num_workers,
                                  img_size=IMG_SIZE), ds_key="s1s2w")


# INT8 robustness sweep. On the workstation distillation improved radiometric/haze
# robustness, this checks whether that survives INT8 quantization. Reports the band delta
# (best - worst) per model and axis; a small |delta| means the deployed model is robust.

_rob_results: dict = {}


_CLEAN_LEVEL = {"gain": "1.00", "haze": "0.00"}


def _worst_drop(row, axis, agg_key="mIoU"):
    """Clean score minus the envelope minimum. If clean is absent, return the flagged max-minus-min fallback."""
    vals = {k: v.get(agg_key, float("nan")) for k, v in row.items()}
    vals = {k: v for k, v in vals.items() if v == v}
    if not vals:
        return float("nan"), False
    anchor = _CLEAN_LEVEL[axis]
    if anchor not in vals:
        return max(vals.values()) - min(vals.values()), True
    return vals[anchor] - min(vals.values()), False


def _sweep(ds_key, dl_for, adapter, axis, levels):
    print(f"\n--- {ds_key.upper()} · {axis} sweep: {levels} ---")
    kf = _scene_keyfn(ds_key)
    for label, path in onnx_models:
        row = {}
        for lv in levels:
            kw = {"gain": lv} if axis == "gain" else {"haze": lv}
            m = eval_onnx(path, _stream(dl_for(**kw), adapter, args.quick), keyfn=kf)
            row[f"{lv:.2f}"] = m
            iou = m.get("iou_per_class", [0.0, 0.0])
            print(f"  {label} {axis}={lv:.2f}  mIoU={m.get('mIoU', 0.0):.4f}  "
                  f"flood={iou[1] if len(iou) > 1 else 0.0:.4f}")
            gc.collect()
        wd, fell_back = _worst_drop(row, axis)
        print(f"  -> {label} worstDrop({axis}) = {wd:+.4f}"
              + ("   [no clean anchor in sweep — this is signed band-Δ]" if fell_back else ""))
        _rob_results.setdefault(label.strip(), {}).setdefault(ds_key, {})[axis] = row
    for _lbl, _be in _trt_backends:
        row = {}
        for lv in levels:
            kw = {"gain": lv} if axis == "gain" else {"haze": lv}
            m = eval_trt_engine(_be, _stream(dl_for(**kw), adapter, args.quick), keyfn=kf)
            row[f"{lv:.2f}"] = m
            print(f"  {_lbl} {axis}={lv:.2f}  mIoU={m.get('mIoU', 0.0):.4f}")
            gc.collect()
        wd, fell_back = _worst_drop(row, axis)
        print(f"  -> {_lbl} worstDrop({axis}) = {wd:+.4f}"
              + ("   [no clean anchor in sweep — this is signed band-Δ]" if fell_back else ""))
        _rob_results.setdefault(_lbl, {}).setdefault(ds_key, {})[axis] = row


if args.robustness:
    print("\n" + "=" * 70)
    print("INT8 ROBUSTNESS SWEEP (gain + haze, reflectance-space, pre-normalization)")
    print("=" * 70)
    if "sturm" in _ROBUST_SEL:
        _sweep("sturm", _sturm_dl, _sturm_adapter, "gain", _GAINS)
        _sweep("sturm", _sturm_dl, _sturm_adapter, "haze", _HAZES)
    if "wf" in _ROBUST_SEL:
        _sweep("wf", _wf_dl, None, "gain", _GAINS)
        _sweep("wf", _wf_dl, None, "haze", _HAZES)


# Latency + unified memory

_lat_results: dict = {}

if not args.no_latency:
    print(f"\n=== Latency (batch=1, {IMG_SIZE}×{IMG_SIZE}, "
          f"warmup={args.warmup}, n={args.n_latency}) ===")
    print("  (For more stable TRT numbers: "
          "trtexec --loadEngine=... --warmUp=100 --iterations=300 --percentile=99)")

    for label, path in onnx_models:
        lbl = label.strip()
        size_mb    = os.path.getsize(path) / 1024 / 1024
        rss_before = _rss_mb()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        rss_after  = _rss_mb()
        rss_delta  = max(0.0, rss_after - rss_before)
        lat_ms = _latency_ort(sess)
        thr    = 1000.0 / lat_ms
        print(f"  {lbl:<26}  {size_mb:5.1f} MB  "
              f"RSS +{rss_delta:5.0f} MB  "
              f"{lat_ms:7.2f} ms/img  {thr:7.1f} img/s")
        _lat_results[lbl] = dict(size_mb=size_mb, lat_ms=lat_ms, rss_mb=rss_delta)
        del sess
        gc.collect()

    for _lbl, _be in _trt_backends:
        _esize_mb = os.path.getsize(_be._engine_path) / 1024 / 1024
        _elat_ms  = _latency_trt(_be)
        _ethr     = 1000.0 / _elat_ms
        print(f"  {_lbl:<26}  {_esize_mb:5.1f} MB  "
              f"RSS    n/a  "
              f"{_elat_ms:7.2f} ms/img  {_ethr:7.1f} img/s")
        _lat_results[_lbl] = dict(
            size_mb=_esize_mb, lat_ms=_elat_ms, rss_mb=float("nan"))


# Summary table

W = 110
all_labels = list(dict.fromkeys(
    list(_acc_results.keys()) + list(_lat_results.keys())
))


def _miou(label, ds):
    return _acc_results.get(label, {}).get(ds, {}).get("mIoU", float("nan"))


def _fmt(v, fmt, na="   n/a"):
    return format(v, fmt) if v == v else na


def _get(label, ds, key):
    return _acc_results.get(label, {}).get(ds, {}).get(key, float("nan"))


_PROTOCOL = (f"PROTOCOL: STURM normalize={args.sturm_normalize} · "
             f"labels {args.protocol} · "
             f"S11 split={args.s11_split} · "
             f"{IMG_SIZE}×{IMG_SIZE}" + (f" · QUICK {args.quick} batches/dataset" if args.quick else ""))


def _summary_table(agg_key, agg_name):
    """One accuracy table per aggregation. Skipped when nothing populated that key."""
    if not any(_get(l, d, agg_key) == _get(l, d, agg_key)
               for l in all_labels for d in ("s1f", "sturm", "wf", "s1s2w")):
        return
    print("\n" + "=" * W)
    print(f"AGGREGATION: {agg_name}")
    print(_PROTOCOL)
    print(f"{'Model':<26}  {'Size':>6}  {'Latency':>9}  {'Throughput':>10}"
          f"    {'S11-' + args.s11_split:>8}  {'STURM':>7}  {'WFv2':>7}  {'S1S2W':>7}")
    print(f"{'':26}  {'(MB)':>6}  {'(ms)':>9}  {'(img/s)':>10}"
          f"    {'mIoU':>7}  {'mIoU':>7}  {'mIoU':>7}  {'mIoU':>7}")
    print("-" * W)
    for lbl in all_labels:
        lat  = _lat_results.get(lbl, {})
        s_mb = _fmt(lat.get("size_mb", float("nan")), "6.1f")
        l_ms = _fmt(lat.get("lat_ms",  float("nan")), "9.2f")
        thr  = _fmt(
            1000.0 / lat["lat_ms"] if lat.get("lat_ms", 0) > 0 else float("nan"),
            "10.1f",
        )
        s1f   = _fmt(_get(lbl, "s1f",   agg_key), "7.4f")
        stu   = _fmt(_get(lbl, "sturm", agg_key), "7.4f")
        wf    = _fmt(_get(lbl, "wf",    agg_key), "7.4f")
        s1s2w = _fmt(_get(lbl, "s1s2w", agg_key), "7.4f")
        print(f"{lbl:<26}  {s_mb}  {l_ms}  {thr}    {s1f}  {stu}  {wf}  {s1s2w}")
    print("=" * W)


_summary_table("mIoU", "MICRO (pixel-pooled)")
_summary_table("macro_scene_mIoU", "MACRO (per-scene mean) — comparable to runs/perscene")


# Robustness summary (band delta per model, dataset and axis)

def _robustness_table(agg_key, agg_name):
    cells_by_label = {}
    for lbl in _rob_results:
        cells = []
        for ds in ("sturm", "wf"):
            for ax in ("gain", "haze"):
                row = _rob_results.get(lbl, {}).get(ds, {}).get(ax)
                cells.append(_worst_drop(row, ax, agg_key)[0] if row else float("nan"))
        cells_by_label[lbl] = cells
    if not any(c == c for cells in cells_by_label.values() for c in cells):
        return                                    # this aggregation was never populated
    print("\n" + "=" * W)
    print(f"ROBUSTNESS SUMMARY [{agg_name}] — worstDrop = clean − min over the swept "
          f"envelope (SMALLER = more robust)")
    print("-" * W)
    print(f"{'Model':<26}  {'STURM gain':>11}  {'STURM haze':>11}  "
          f"{'WF gain':>9}  {'WF haze':>9}")
    for lbl, cells in cells_by_label.items():
        print(f"{lbl:<26}  {_fmt(cells[0],'11.4f')}  {_fmt(cells[1],'11.4f')}  "
              f"{_fmt(cells[2],'9.4f')}  {_fmt(cells[3],'9.4f')}")
    print("=" * W)


if args.robustness and _rob_results:
    _robustness_table("mIoU", "micro")
    _robustness_table("macro_scene_mIoU", "macro/scene")

# JSON dump
if args.json_out:
    import json
    # Preserve the historical s1f key, but identify its split in both each result and config.
    # Split, label protocol and preprocessing must match before combining result files.
    payload = {"accuracy": _acc_results, "latency": _lat_results,
               "robustness": _rob_results,
               "config": {"img_size": IMG_SIZE, "sturm_normalize": args.sturm_normalize,
                          "s11_split": args.s11_split,
                          "num_workers": args.num_workers,
                          "protocol": args.protocol,
                          "wf_flood_only": args.protocol != "surface_water",
                          "datasets": sorted(_SEL), "gains": _GAINS, "hazes": _HAZES,
                          "robust_datasets": sorted(_ROBUST_SEL) if args.robustness else [],
                          "robustness_metric": "worstDrop = clean - min over envelope",
                          "model_dir": os.path.abspath(args.model_dir),
                          "engines": [os.path.abspath(b._engine_path) for _, b in _trt_backends],
                          "quick": args.quick}}
    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n[saved] {args.json_out}")
