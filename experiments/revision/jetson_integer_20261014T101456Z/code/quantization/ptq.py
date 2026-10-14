"""Quantization driver using the custom fixed-scale QAT implementation.

The published path uses per-tensor symmetric activations for TensorRT and
per-output-channel symmetric weights. Per-channel activation variants remain
for comparison and are not the Jetson deployment recipe. Keep the original
calibration scales for QDQ export. Stripped QAT weights are not a float baseline.
"""

from __future__ import annotations

import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable

from config import QuantConfig
from utils.metrics import accumulate_metrics, finalize_metrics




# Public: qat()


def qat(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    cfg: QuantConfig,
    task_loss_fn: Callable,
    num_classes: int = 2,
    ignore_index: int = -1,
    teacher: nn.Module | None = None,
    kd_alpha: float = 0.0,
    kd_temperature: float = 4.0,
    calibration_type: str = "per_tensor",
) -> tuple[nn.Module, dict]:
    """QAT with fixed-scale QATConv2d fake-quant.

        1. calibrate activation + weight scales (per_tensor or per_channel)
        2. replace Conv2d with QATConv2d (fake-quant, STE gradients, fixed scales)
        3. fine-tune: warmup with fake-quant off, then QAT with it on
        4. export ONNX with the original QAT calibration scales; re-calibrating a stripped
           checkpoint would give mismatched scales

        calibration_type:
            "per_tensor" (default): global amax/127, zp=0. This is the correct one for
                TRT/Jetson Xavier deployment. Needs ReLU6 activations (efficientvit_b0_relu6)
                so all channels are bounded to [0,6] and the single scale has full resolution.
            "per_channel_symmetric": per-channel abs-max/127, all zp=0. Full precision and no
                STE dead zone, but research only and not deployable (TRT needs per-tensor act).
            "per_channel": per-channel asymmetric MSE. Can collapse on ReLU6 models (zp=-128
                dead zone). Research only.

        `model` is the float32 model with pretrained / distilled weights, test_dl is only for
        the float baseline comparison, cfg a QuantConfig (qat_epochs, qat_lr,
        calibration_batches, ...), task_loss_fn(logits, masks) -> Tensor. The optional frozen
        teacher adds a KL-div KD term weighted by kd_alpha (0 = CE only) at kd_temperature
        (default 4.0). Pixels labelled ignore_index are left out of the metrics.

        Returns (stripped_float_model, results_dict) with float_val_mIoU, qat_val_mIoU,
        float_test_mIoU, qat_test_mIoU and mIoU_drop.
    """
    from quantization.qat import prepare_qat, train_qat, export_qat_qdq_onnx

    t0 = time.perf_counter()
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device(f"cuda:{cfg.gpu_ids[0]}"
                          if cfg.gpu_ids and torch.cuda.is_available()
                          else "cpu")
    print(f"[QAT] device={device}  qat_epochs={cfg.qat_epochs}  lr={cfg.qat_lr}")

    # Float32 baseline on val
    float_model = copy.deepcopy(model).to(device).eval()
    print("[QAT] evaluating float32 baseline on val set ...")
    conf = None
    with torch.no_grad():
        for batch in val_dl:
            logits = float_model(batch["image"].to(device).float())
            conf   = accumulate_metrics(logits.cpu(), batch["mask"].long(),
                                        num_classes, ignore_index, conf)
    float_val = finalize_metrics(conf) if conf is not None else {}
    print(f"  float32 val mIoU = {float_val.get('mIoU', 0.0):.4f}")

    # Calibrate + prepare QATConv2d.
    # Calibration uses val_dl (labeled S11 at eval_img_size), not train_dl. The pseudo-labeled
    # GDACS images in train_dl inflate activation scales 5-9x compared to S11 (checked
    # empirically), large scales mean a coarse INT8 grid, and TRT INT8 accuracy on the Jetson
    # collapses. CC's QAT pipeline also always calibrates on labeled data.
    # val_dl uses eval_img_size (512px by default), the deployment resolution. The resolution
    # difference between calibration and training doesn't matter for ReLU6 models (ratio
    # ~1.0 across all layers, verified).
    cal_batches = []
    for b in val_dl:
        cal_batches.append({"image": b["image"].float(), "mask": b["mask"].long()})
        if len(cal_batches) >= cfg.calibration_batches:
            break
    print(f"[QAT] calibrating on {len(cal_batches)} batches "
          f"(calibration_type={calibration_type!r}, S11 val data @ eval resolution) ...")
    qat_model = prepare_qat(copy.deepcopy(model), cal_batches,
                            calibration_type=calibration_type)

    # Move teacher to device if provided
    if teacher is not None and kd_alpha > 0.0:
        teacher = teacher.to(device).eval()
        print(f"[QAT] KD loss enabled  kd_alpha={kd_alpha}  temperature={kd_temperature}")

    # closures for the train_qat interface
    def loss_fn(m: nn.Module, batch: dict) -> torch.Tensor:
        images = batch["image"].float()
        logits = m(images)
        loss = task_loss_fn(logits, batch["mask"].long())
        if teacher is not None and kd_alpha > 0.0:
            with torch.no_grad():
                t_logits = teacher(images)
            T = kd_temperature
            kd = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean",
            ) * (T ** 2)
            loss = loss + kd_alpha * kd
        return loss

    def evaluate_fn(m: nn.Module) -> dict:
        m.eval()
        c = None
        with torch.no_grad():
            for b in val_dl:
                logits = m(b["image"].to(device).float())
                c = accumulate_metrics(logits.cpu(), b["mask"].long(),
                                       num_classes, ignore_index, c)
        return finalize_metrics(c) if c is not None else {}

    # QAT training loop
    stripped = train_qat(
        qat_model,
        train_dl,
        loss_fn       = loss_fn,
        evaluate_fn   = evaluate_fn,
        device        = device,
        epochs        = cfg.qat_epochs,
        lr            = cfg.qat_lr,
        weight_decay  = cfg.weight_decay,
        warmup_epochs = 1,
        grad_clip     = cfg.grad_clip,
        scheduler     = cfg.scheduler,
        val_every_n_epochs = cfg.val_every_n_epochs,
        out_dir       = cfg.out_dir,
    )
    # After train_qat, qat_model has the best QATConv2d weights (needed for export).
    # No re-calibration here. QAT adapted the weights to the fixed calibration scales, and
    # running prepare_qat again after training would give new scales the weights were never
    # trained with, i.e. a different quantization grid. The cc pipeline doesn't re-calibrate
    # for the same reason.

    # QAT val mIoU (fake-quant on, best checkpoint)
    qat_val = evaluate_fn(qat_model)
    print(f"[QAT] QAT int8 (fake-quant) val mIoU = {qat_val.get('mIoU', 0.0):.4f}")

    # Float test baseline
    conf_f = None
    with torch.no_grad():
        for batch in test_dl:
            logits = float_model(batch["image"].to(device).float())
            conf_f = accumulate_metrics(logits.cpu(), batch["mask"].long(),
                                        num_classes, ignore_index, conf_f)
    float_test = finalize_metrics(conf_f) if conf_f is not None else {}

    # QAT test mIoU
    conf_q = None
    qat_model.eval()
    from quantization.qat import set_fake_quant
    set_fake_quant(qat_model, True)
    with torch.no_grad():
        for batch in test_dl:
            logits = qat_model(batch["image"].to(device).float())
            conf_q = accumulate_metrics(logits.cpu(), batch["mask"].long(),
                                        num_classes, ignore_index, conf_q)
    qat_test = finalize_metrics(conf_q) if conf_q is not None else {}

    miou_drop = float_test.get("mIoU", 0.0) - qat_test.get("mIoU", 0.0)
    print(f"\n[QAT] done in {_elapsed(t0)}")
    print(f"  float32 test mIoU  = {float_test.get('mIoU', 0.0):.4f}")
    print(f"  QAT int8 test mIoU = {qat_test.get('mIoU', 0.0):.4f}  (drop={miou_drop:+.4f})")

    # Export ONNX with QAT calibration scales
    # Use eval_img_size (512) for ONNX export so onnxsim can fold LiteMLA
    # padding ops (512*512 % 8 = 0 constant-folds away in jp50 mode).
    # Zero tensor is fine - tracing only needs the correct shape.
    dummy_batch = next(iter(val_dl))["image"]
    _n, _c, _h, _w = dummy_batch.shape
    dummy = torch.zeros(1, _c, _h, _w)   # eval_img_size shape, no actual pixel data
    try:
        qdq_no_out_path = export_qat_qdq_onnx(
            qat_model, dummy, cfg.out_dir,
            trt_compat        = cfg.trt_compat,
            cal_batches       = cal_batches,
            output_qdq_scale  = getattr(cfg, "output_qdq_scale", "calibrated"),
            matmul_qdq        = getattr(cfg, "matmul_qdq", False),
        )
        print(f"[QAT] ONNX exported: {qdq_no_out_path}")
    except Exception as exc:
        print(f"[QAT] WARNING: ONNX export failed: {exc}")

    stripped_path = os.path.join(cfg.out_dir, "model_qat_stripped.pt")
    torch.save({"student": stripped.state_dict()}, stripped_path)
    print(f"[QAT] stripped float checkpoint -> {stripped_path}")
    print("[QAT] NOTE: these weights were trained against fixed fake-quant activation "
          "scales.\n"
          "      Run in plain float they are OUTSIDE their trained regime and lose accuracy\n"
          "      (measured on a 2500-scene ReLU6 OFA run: 0.700 float vs 0.878 through the\n"
          "      QDQ graph, S11 test mIoU). model_float*.onnx has the same property.\n"
          "      Deploy/evaluate model_qdq_trt_no_out.onnx; for an FP32/FP16 engine use the\n"
          "      PRE-QAT checkpoint, not this one.")

    results = {
        "float_val_mIoU":  float_val.get("mIoU", 0.0),
        "qat_val_mIoU":    qat_val.get("mIoU", 0.0),
        "float_test_mIoU": float_test.get("mIoU", 0.0),
        "qat_test_mIoU":   qat_test.get("mIoU", 0.0),
        "mIoU_drop":       miou_drop,
    }
    return stripped, results



# Utility


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s/60:.1f}min" if s > 90 else f"{s:.1f}s"
