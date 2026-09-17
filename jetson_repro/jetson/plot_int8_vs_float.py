"""Float vs INT8 panels for fixed scenes, run on the Jetson.

The --jetson branch of figures/plot_models.py, separated to run without the
workstation models. Scenes come from a manifest (manifests/qual_samples_manifest.json).
Each stored id is checked against the local dataset before drawing.

    python jetson/plot_int8_vs_float.py \
        --picks_from manifests/qual_samples_manifest.json \
        --model_dir runs/deploy_qat/docker_qat \
        --engine runs/deploy_qat/docker_qat/model_int8_qat.engine \
        --no_agreement --paper_clean --rows_per_figure 4 --out_dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import compute_seg_metrics
from figures.qual_rgb import TRUE_COLOUR_CEILING, to_rgb

JETSON_DATASETS = ["S11", "STURM-S2", "WorldFloods-v2"]

_AGREE_BOTH    = (1.0, 1.0, 1.0)
_AGREE_B_ONLY  = (0.20, 0.75, 0.20)   # INT8 right, Float wrong
_AGREE_A_ONLY  = (0.85, 0.25, 0.10)   # Float right, INT8 wrong
_AGREE_NEITHER = (0.15, 0.15, 0.15)
_AGREE_NODATA  = (0.60, 0.60, 0.60)
_CAT_NEITHER, _CAT_BOTH, _CAT_B_ONLY, _CAT_A_ONLY, _CAT_NODATA = 0, 1, 2, 3, 4
_CAT_COLOR = {
    _CAT_NEITHER: _AGREE_NEITHER,
    _CAT_BOTH:    _AGREE_BOTH,
    _CAT_B_ONLY:  _AGREE_B_ONLY,
    _CAT_A_ONLY:  _AGREE_A_ONLY,
    _CAT_NODATA:  _AGREE_NODATA,
}

_QUANT_MODEL_NAMES = [
    ("model_float.onnx",          "Float"),
    ("model_qdq_trt_no_out.onnx", "INT8"),
    ("model_qdq_trt.onnx",        "INT8"),
]


@dataclass
class DatasetSpec:
    key: str
    dataset: Any
    adapter: Any
    mask_is_gt: bool


def get_sample_id(spec: DatasetSpec, idx: int) -> str:
    item = spec.dataset[idx]
    if "filename" in item:
        return item["filename"]
    for attr in ("image_files", "files", "image_paths"):
        seq = getattr(spec.dataset, attr, None)
        if seq is not None and len(seq) == len(spec.dataset):
            return Path(str(seq[idx])).name
    return f"{spec.key}_{idx}"


def build_datasets(args) -> list[DatasetSpec]:
    # Same loader settings as eval_jetson.py, except S11 uses the TEST split here
    # because that is what the manifest was written against.
    import data.sen1floods11_nodep as nodep
    sys.modules["data.sen1floods11"] = nodep
    from data.ood_datasets import get_sturm_flood_dataloader, get_worldfloods_dataloader
    from data.channel_adapter import sturm_s2_adapter

    _, _, s11_test_dl = nodep.get_dataloaders(
        data_root=args.s11_root, batch_size=1, num_workers=0,
        eval_img_size=args.eval_img_size)
    sturm_dl = get_sturm_flood_dataloader(
        root=args.sturm_root, sensor="S2", batch_size=1, num_workers=0,
        img_size=args.eval_img_size, normalize="original")
    wf_dl = get_worldfloods_dataloader(
        root=args.wf_root, split="test", batch_size=1, num_workers=0,
        img_size=args.eval_img_size, tile_size=args.eval_img_size, normalize="s11",
        ignore_permanent_water=True)
    return [
        DatasetSpec("S11", s11_test_dl.dataset, None, True),
        DatasetSpec("STURM-S2", sturm_dl.dataset, sturm_s2_adapter(6), True),
        DatasetSpec("WorldFloods-v2", wf_dl.dataset, None, True),
    ]


def discover_quant_models(model_dir) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for fname, label in _QUANT_MODEL_NAMES:
        if label in found:
            continue
        p = Path(model_dir) / fname
        if p.exists():
            found[label] = p
    missing = {"Float", "INT8"} - found.keys()
    if missing:
        raise FileNotFoundError(f"Missing {missing} ONNX file(s) in {model_dir}")
    return found


def build_trt_backend(engine_path: str):
    """Same TRT 8.4 binding code as eval_jetson.py. Returns None without TRT."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except Exception as e:
        print(f"TRT Python API unavailable ({e}); INT8 column falls back to onnxruntime.")
        return None

    class _TRTBackend:
        def __init__(self, path: str, img_size: int = 512, in_channels: int = 6):
            logger = trt.Logger(trt.Logger.WARNING)
            with open(path, "rb") as f:
                engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to load TRT engine: {path}")
            self._ctx    = engine.create_execution_context()
            self._stream = cuda.Stream()
            in_idx  = engine.get_binding_index("image")
            out_idx = engine.get_binding_index("logits")
            self._h_in  = cuda.pagelocked_empty((1, in_channels, img_size, img_size), dtype=np.float32)
            self._h_out = cuda.pagelocked_empty((1, 2, img_size, img_size), dtype=np.float32)
            self._d_in  = cuda.mem_alloc(self._h_in.nbytes)
            self._d_out = cuda.mem_alloc(self._h_out.nbytes)
            self._bindings = [None, None]
            self._bindings[in_idx]  = int(self._d_in)
            self._bindings[out_idx] = int(self._d_out)

        def infer_one(self, image_1chw: np.ndarray) -> np.ndarray:
            np.copyto(self._h_in, image_1chw)
            cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
            self._ctx.execute_async_v2(self._bindings, self._stream.handle)
            cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
            self._stream.synchronize()
            return self._h_out.copy()

    return _TRTBackend(engine_path)


def build_agreement_map(gt, pred_a, pred_b, denoise_kernel=3):
    valid     = gt != -1
    a_correct = (pred_a == gt) & valid
    b_correct = (pred_b == gt) & valid

    cat = np.full(gt.shape, _CAT_NEITHER, dtype=np.int64)
    cat[a_correct & b_correct]  = _CAT_BOTH
    cat[b_correct & ~a_correct] = _CAT_B_ONLY
    cat[a_correct & ~b_correct] = _CAT_A_ONLY
    cat[~valid]                 = _CAT_NODATA

    if denoise_kernel > 1:
        from scipy.ndimage import median_filter
        nodata = cat == _CAT_NODATA
        cat = median_filter(cat, size=denoise_kernel, mode="nearest")
        cat[nodata] = _CAT_NODATA

    out = np.empty(gt.shape + (3,), dtype=np.float32)
    for c, color in _CAT_COLOR.items():
        out[cat == c] = color
    return out


def agreement_legend(label_a: str, label_b: str):
    import matplotlib.patches as mpatches
    return [
        mpatches.Patch(color=_AGREE_B_ONLY,  label=f"{label_b} correct, {label_a} wrong"),
        mpatches.Patch(color=_AGREE_A_ONLY,  label=f"{label_a} correct, {label_b} wrong"),
        mpatches.Patch(facecolor=_AGREE_BOTH, edgecolor="grey", label="both correct"),
        mpatches.Patch(color=_AGREE_NEITHER, label="both wrong"),
        mpatches.Patch(color=_AGREE_NODATA,  label="no data"),
    ]


def save_figure(rows, spec_by_key, preds, out_path: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs_title = round(11 * args.font_scale)
    fs_supt  = round(13 * args.font_scale)
    fs_leg   = round(9 * args.font_scale)

    # Only one INT8 column: the TRT engine if it ran, otherwise the ORT QDQ model.
    trt = "TRT-INT8-engine" in preds
    labels = ["Float"] + ([("TRT-INT8-engine" if trt else "INT8")] if trt or "INT8" in preds
                          else [])
    display = {"Float": "Float", "TRT-INT8-engine": "INT8 (TensorRT)", "INT8": "INT8 (ORT)"}
    int8_label = labels[-1] if len(labels) > 1 else None
    if int8_label == "INT8":
        print("WARNING: no TRT engine ran, the INT8 column is onnxruntime, not the deployed engine.")
    show_agreement = args.show_agreement and int8_label is not None
    ncols = 2 + len(labels) + (1 if show_agreement else 0)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.7 * ncols, 3.6 * len(rows)))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for r, (key, idx, _sample_id) in enumerate(rows):
        item = spec_by_key[key].dataset[idx]
        img  = item["image"].numpy()
        gt   = item["mask"].numpy()
        rgb  = to_rgb(img, p_low=args.rgb_p_low, p_high=args.rgb_p_high, gamma=args.rgb_gamma,
                      sat=args.rgb_sat, per_channel=False, fixed=args.rgb_fixed)[0]

        ax = axes[r]
        ax[0].imshow(rgb)
        ax[0].set_title(f"{key}\nRGB", fontsize=fs_title)
        ax[1].imshow(gt, cmap="Blues", vmin=-1, vmax=1, interpolation="nearest")
        ax[1].set_title("Ground truth", fontsize=fs_title)

        pred_by_label = {}
        for c, label in enumerate(labels, start=2):
            logits = torch.from_numpy(preds[label][(key, idx)])
            gt_t   = torch.from_numpy(gt).long().unsqueeze(0)
            if logits.shape[-2:] != gt_t.shape[-2:]:
                logits = F.interpolate(logits, size=gt_t.shape[-2:],
                                       mode="bilinear", align_corners=False)
            m = compute_seg_metrics(logits, gt_t, num_classes=2, ignore_index=-1)
            pred = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
            pred_by_label[label] = pred
            iou = m["iou_per_class"]
            flood_iou = iou[1] if len(iou) > 1 else float("nan")
            ax[c].imshow(pred, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax[c].set_title(f"{display[label]}\nflood-IoU={flood_iou:.2f}", fontsize=fs_title)

        if show_agreement:
            agree = build_agreement_map(gt, pred_by_label["Float"], pred_by_label[int8_label],
                                        args.agreement_denoise_kernel)
            ax[-1].imshow(agree, interpolation="nearest")
            ax[-1].set_title("Float vs INT8 agreement", fontsize=fs_title)

        for a in ax:
            a.axis("off")

    if show_agreement:
        fig.legend(handles=agreement_legend("Float", "INT8"),
                   loc="lower center", ncol=5, fontsize=fs_leg, bbox_to_anchor=(0.5, -0.02))
    if not args.paper_clean:
        fig.suptitle("Deployed EViT-OFA — Float vs INT8 (Jetson quantized engine)",
                     fontsize=fs_supt, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def run(args) -> None:
    manifest = json.loads(Path(args.picks_from).read_text())
    picks_all = manifest.get("picks", {})
    skipped = [k for k in picks_all if k not in JETSON_DATASETS]
    if skipped:
        print(f"dropping manifest picks for {skipped} (not on the device)")

    import onnxruntime as ort

    onnx_paths = discover_quant_models(args.model_dir)
    sess = {label: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            for label, p in onnx_paths.items()}
    print("models: " + "  ".join(f"{l}={p.name}" for l, p in onnx_paths.items()))

    trt_backend = build_trt_backend(args.engine) if args.engine else None

    specs = build_datasets(args)
    spec_by_key = {s.key: s for s in specs}

    rows = []
    for key in JETSON_DATASETS:
        if key not in picks_all:
            continue
        for p in picks_all[key]:
            want = p.get("id")
            got = get_sample_id(spec_by_key[key], p["idx"])
            if want is not None and got != want:
                raise SystemExit(
                    f"manifest does not match local data for {key}: index {p['idx']} was "
                    f"{want!r} in the manifest but is {got!r} here. Check --eval_img_size "
                    f"and that the dataset copy is complete.")
            rows.append((key, p["idx"], want or got))
    if not rows:
        raise SystemExit("No manifest picks for S11 / STURM-S2 / WorldFloods-v2.")

    # The ORT INT8 column is not drawn when an engine is present, so skip that pass.
    if trt_backend is not None:
        sess.pop("INT8", None)
    preds: dict[str, dict] = {label: {} for label in sess}
    if trt_backend is not None:
        preds["TRT-INT8-engine"] = {}

    for key, idx, _id in rows:
        spec  = spec_by_key[key]
        image = spec.dataset[idx]["image"].unsqueeze(0).float()
        if spec.adapter is not None:
            image = spec.adapter(image)
        image_np = image.numpy()
        for label, s in sess.items():
            preds[label][(key, idx)] = s.run(None, {s.get_inputs()[0].name: image_np})[0]
        if trt_backend is not None:
            preds["TRT-INT8-engine"][(key, idx)] = trt_backend.infer_one(image_np)

    # Split into sheets, breaking on dataset boundaries first.
    n_per = args.rows_per_figure or len(rows)
    chunks: list[list] = []
    for key in dict.fromkeys(k for k, _i, _s in rows):
        block = [r for r in rows if r[0] == key]
        for i in range(0, len(block), n_per):
            part = block[i:i + n_per]
            if chunks and len(chunks[-1]) + len(part) <= n_per:
                chunks[-1].extend(part)
            else:
                chunks.append(part)
    for i, chunk in enumerate(chunks, start=1):
        name = ("qual_jetson_int8_vs_float.png" if len(chunks) == 1
                else f"qual_jetson_int8_vs_float_{i}.png")
        save_figure(chunk, spec_by_key, preds, Path(args.out_dir) / name, args)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--picks_from", default=str(ROOT / "manifests" / "qual_samples_manifest.json"))
    p.add_argument("--model_dir", default=str(ROOT / "runs" / "deploy_qat" / "docker_qat"))
    p.add_argument("--engine", default=None)
    p.add_argument("--out_dir", default=str(ROOT / "results" / "figures"))
    p.add_argument("--eval_img_size", type=int, default=512,
                   help="must match the manifest (WorldFloods picks are tile indices)")
    p.add_argument("--s11_root",   default=str(Path.home() / "data" / "sen1floods11"))
    p.add_argument("--sturm_root", default=str(Path.home() / "data" / "STURM-Flood" / "Dataset"))
    p.add_argument("--wf_root",    default=str(Path.home() / "data" / "WorldFloodsv2"))

    p.add_argument("--show_agreement", dest="show_agreement", action="store_true", default=True)
    p.add_argument("--no_agreement", dest="show_agreement", action="store_false")
    p.add_argument("--agreement_denoise_kernel", type=int, default=3)
    p.add_argument("--rows_per_figure", type=int, default=0)
    p.add_argument("--paper_clean", action="store_true", help="no suptitle")
    p.add_argument("--font_scale", type=float, default=1.9)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--rgb_fixed", type=float, default=TRUE_COLOUR_CEILING,
                   help="true-colour ceiling (0 = per-scene percentile stretch)")
    p.add_argument("--rgb_p_low", type=float, default=2.0)
    p.add_argument("--rgb_p_high", type=float, default=98.0)
    p.add_argument("--rgb_gamma", type=float, default=1.0)
    p.add_argument("--rgb_sat", type=float, default=1.0)
    args = p.parse_args()
    args.rgb_fixed = args.rgb_fixed if args.rgb_fixed and args.rgb_fixed > 0 else None
    return args


if __name__ == "__main__":
    run(parse_args())
