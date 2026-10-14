"""TensorRT INT8 engine builder and mIoU evaluator.

build_engine() compiles a QDQ ONNX into a TensorRT INT8 engine; eval_trt() runs the engine
on a DataLoader and returns mIoU and per-class IoU. The activation scales come from the
QuantizeLinear/DequantizeLinear nodes, so no calibrator is needed (TensorRT 10).

    from quantization.trt_eval import build_engine, eval_trt
    engine_path = build_engine(onnx_path="runs/export/model_qdq_trt.onnx",
                               engine_path="runs/export/model_int8.trt", fp16_fallback=True)
    metrics = eval_trt(engine_path, val_dl, num_classes=2)

CLI:
    python scripts/quantize.py eval-trt \
        --onnx_path runs/export/model_qdq_trt.onnx --engine_path runs/export/model_int8.trt \
        --data_root $DATA_ROOT/sen1floods11 --student_arch efficientvit_b0_relu6

Requires tensorrt (10.x) and pycuda.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch



# Engine build


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16_fallback: bool = True,
    workspace_gb: float = 4.0,
    verbose: bool = False,
) -> str:
    """Compile a QDQ ONNX (per-tensor symmetric Q-DQ, static input shape) into a TensorRT INT8 engine.

    onnx_path is the exported model_qdq_trt.onnx, the serialized engine goes to engine_path,
    which is also returned. fp16_fallback lets layers without INT8 support run in FP16
    (recommended). workspace_gb is the TensorRT workspace (default 4).
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError(
            "tensorrt Python package not found. "
            "Install with: pip install tensorrt"
        )

    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(log_level)

    print(f"[TRT] building INT8 engine from {onnx_path!r}")
    print(f"      TRT version {trt.__version__}  fp16_fallback={fp16_fallback}")

    t0 = time.perf_counter()

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with (
        trt.Builder(logger) as builder,
        builder.create_network(explicit_batch) as network,
        trt.OnnxParser(network, logger) as parser,
    ):
        config = builder.create_builder_config()

        # Memory pool for tactic search (TRT 10 API)
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gb * (1 << 30)),
        )

        # INT8: scales come from the Q/DQ nodes
        config.set_flag(trt.BuilderFlag.INT8)
        if fp16_fallback:
            # FP16 used for non-quantizable ops (Resize, Sigmoid, etc.)
            config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_path, "rb") as f:
            raw = f.read()

        ok = parser.parse(raw)
        if not ok:
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"TRT ONNX parse failed ({len(errors)} error(s)):\n"
                + "\n".join(errors)
            )

        print(f"  ONNX parsed: {network.num_layers} layers")
        print(f"  input:  {network.get_input(0).shape}")
        print(f"  output: {network.get_output(0).shape}")

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                "TRT engine build returned None -- check logs above for errors"
            )

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(engine_path) / (1 << 20)
    print(f"  engine saved -> {engine_path}  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")
    return engine_path



# Inference and evaluation


def _load_engine(engine_path: str):
    """Deserialize a saved .trt engine plan."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime, open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def eval_trt(
    engine_path: str,
    dataloader,
    num_classes: int = 2,
    ignore_index: int = -1,
    device_id: int = 0,
) -> dict:
    """Run a TensorRT engine on a DataLoader and return segmentation metrics.

    The engine input size has to match the loader's image size (512 by default). The loader
    yields {'image': Tensor, 'mask': Tensor}, pixels labelled ignore_index (default -1) are
    skipped. Returns a dict with mIoU, iou_per_class and pixel_acc.
    """
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 - sets up CUDA context
    except ImportError:
        raise RuntimeError(
            "pycuda not found. Install with: pip install pycuda\n"
            "Or use eval_onnx() from quantization.onnx_qdq for onnxruntime-based eval."
        )

    from utils.metrics import accumulate_metrics, finalize_metrics

    engine = _load_engine(engine_path)
    context = engine.create_execution_context()

    # Allocate buffers based on the engine's expected I/O shapes.
    # TRT 10 API uses get_tensor_name / set_tensor_address.
    in_name  = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    in_shape  = tuple(engine.get_tensor_shape(in_name))
    out_shape = tuple(engine.get_tensor_shape(out_name))

    print(f"[TRT eval] engine  input={in_name}{in_shape}  output={out_name}{out_shape}")
    print(f"           device=cuda:{device_id}")

    # Pre-allocate pinned host + device buffers for a single batch.
    in_nbytes  = int(np.prod(in_shape))  * 4   # float32
    out_nbytes = int(np.prod(out_shape)) * 4   # float32

    in_host  = cuda.pagelocked_empty(in_nbytes  // 4, dtype=np.float32)
    out_host = cuda.pagelocked_empty(out_nbytes // 4, dtype=np.float32)
    in_dev   = cuda.mem_alloc(in_host.nbytes)
    out_dev  = cuda.mem_alloc(out_host.nbytes)

    stream = cuda.Stream()

    # TRT 10: use set_tensor_address
    context.set_tensor_address(in_name,  int(in_dev))
    context.set_tensor_address(out_name, int(out_dev))

    conf = None
    n_batches = 0
    lat_times: list[float] = []

    for batch in dataloader:
        img  = batch["image"].float().numpy()  # [B, C, H, W]
        mask = batch["mask"].long()

        np.copyto(in_host, img.ravel())
        cuda.memcpy_htod_async(in_dev, in_host, stream)

        t_start = time.perf_counter()
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(out_host, out_dev, stream)
        stream.synchronize()
        t_end = time.perf_counter()

        if n_batches >= 5:  # skip first 5 as warmup
            lat_times.append((t_end - t_start) * 1000.0)

        logits = torch.from_numpy(out_host.reshape(out_shape).copy())
        conf = accumulate_metrics(logits, mask, num_classes, ignore_index, conf)
        n_batches += 1

    metrics = finalize_metrics(conf) if conf is not None else {}

    if lat_times:
        lat_sorted = sorted(lat_times)
        mean_lat   = sum(lat_sorted) / len(lat_sorted)
        p99_idx    = min(len(lat_sorted) - 1, int(0.99 * len(lat_sorted)))
        p99_lat    = lat_sorted[p99_idx]
        bs         = in_shape[0] if in_shape[0] > 0 else 1
        fps        = bs * 1000.0 / (mean_lat + 1e-9)
        metrics["lat_ms_mean"]    = mean_lat
        metrics["lat_ms_p99"]     = p99_lat
        metrics["fps"]            = fps
        metrics["engine_size_mb"] = os.path.getsize(engine_path) / (1 << 20)
        print(f"[TRT eval] {n_batches} batches  mIoU={metrics.get('mIoU', 0.0):.4f}  "
              f"lat_mean={mean_lat:.1f}ms  lat_p99={p99_lat:.1f}ms  "
              f"FPS={fps:.1f}  engine={metrics['engine_size_mb']:.1f}MB")
    else:
        print(f"[TRT eval] {n_batches} batches  mIoU={metrics.get('mIoU', 0.0):.4f}")

    return metrics
