"""Export and measure the actual desktop inference graph, without training state.

Fake quantization retains float convolution arithmetic and float master weights.
Its TorchScript file is an executable simulation artifact, not a packed low-bit
CNN engine. MAC counts cover convolutions and both attention matrix products;
normalization, interpolation, activation, bias, and QDQ overhead are excluded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import warnings

import torch
from torch import nn

from models.compression_quant import FixedQATConv2d, precision_map, quantization_mode
from quantization.qat import QATMatMul


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # No change to the caller's training/data RNG. These are synthetic export
    # checks, not extra source or external accuracy observations.
    generator = torch.Generator(device="cpu").manual_seed(1729 + size)
    return torch.randn(1, 6, size, size, generator=generator).to(device=device, dtype=dtype)


def _finite_logits(value: torch.Tensor, size: int) -> None:
    if value.shape != (1, 2, size, size) or not torch.isfinite(value).all():
        raise RuntimeError(f"Export probe requires finite two-class {size}x{size} logits")


def count_inference_macs(model: nn.Module, image: torch.Tensor) -> dict:
    """Count operators actually executed at image.shape, avoiding MLA double counts."""
    rows, handles = [], []

    def convolution(name):
        def hook(module, inputs, output):
            macs = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            rows.append({"name": name, "op": "Conv2d", "macs": int(macs),
                         "arithmetic_dtype": str(output.dtype), "output_shape": list(output.shape),
                         "weight_bits": module.weight_bits if isinstance(module, FixedQATConv2d) else
                                        module.weight.element_size() * 8,
                         "activation_bits": module.activation_bits if isinstance(module, FixedQATConv2d) else
                                            inputs[0].element_size() * 8,
                         "groups": module.groups})
        return hook

    def matmul(name):
        def hook(module, inputs, output):
            # torch.matmul performs M*N dot products of length K for each
            # broadcast batch; output captures the actual broadcast dimensions.
            macs = math.prod(output.shape[:-2]) * output.shape[-2] * output.shape[-1] * inputs[0].shape[-1]
            rows.append({"name": name, "op": "MatMul", "macs": int(macs),
                         "arithmetic_dtype": str(output.dtype), "output_shape": list(output.shape),
                         "activation_bits": output.element_size() * 8,
                         "fake_quant_enabled": module.fake_quant_enabled})
        return hook

    def native_attention(name):
        def hook(module, inputs, output):
            batch, _, height, width = inputs[0].shape
            positions = height * width
            heads = module.qkv.conv.out_channels // (3 * module.dim)
            effective_heads = heads * (1 + len(module.aggreg))
            # Linear attention pads V by one normalization row. The two products
            # are [dim+1,N]@[N,dim], then [dim+1,dim]@[dim,N].
            if positions > module.dim:
                macs = 2 * batch * effective_heads * positions * module.dim * (module.dim + 1)
                branch = "linear_with_padded_normalizer_row"
            else:
                macs = 2 * batch * effective_heads * positions * positions * module.dim
                branch = "quadratic"
            dtype = torch.float32 if inputs[0].dtype == torch.float16 and positions > module.dim else inputs[0].dtype
            rows.append({"name": name, "op": "LiteMLA_MatMul_pair", "macs": int(macs),
                         "arithmetic_dtype": str(dtype), "activation_bits": 32 if dtype == torch.float32 else 16,
                         "branch": branch, "effective_heads": effective_heads,
                         "spatial_positions": positions, "attention_dim": module.dim})
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(convolution(name)))
        elif isinstance(module, QATMatMul):
            handles.append(module.register_forward_hook(matmul(name)))
        elif type(module).__name__ == "LiteMLA":
            # LiteMLAQAT calls its nested LiteMLA children directly, bypassing
            # LiteMLA.__call__; this hook therefore never double-counts its mm1/mm2.
            handles.append(module.register_forward_hook(native_attention(name)))
    try:
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    conv = sum(row["macs"] for row in rows if row["op"] == "Conv2d")
    attention = sum(row["macs"] for row in rows if row["op"] != "Conv2d")
    arithmetic = {}
    for row in rows:
        arithmetic[row["arithmetic_dtype"]] = arithmetic.get(row["arithmetic_dtype"], 0) + row["macs"]
    return {"input_shape": list(image.shape), "total_macs": conv + attention,
            "convolution_macs": conv, "attention_matmul_macs": attention,
            "macs_by_actual_arithmetic_dtype": arithmetic, "operators": rows,
            "count_convention": "one scalar multiply-accumulate counts as one MAC",
            "excluded_operations": ["bias addition", "BatchNorm", "activation", "interpolation",
                                    "residual addition", "attention normalization", "fake quantization/QDQ"]}


def export_and_validate(model: nn.Module, path: Path) -> tuple[dict, torch.jit.ScriptModule]:
    """Serialize only the supplied inference module and check a loaded artifact."""
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    if any(module.training for module in model.modules()):
        raise ValueError("Inference export requires evaluation mode on every module")
    for key in model.state_dict():
        if any(part in key.split(".") for part in
               ("teacher", "loss_fn", "adapter", "transform", "exit_branches", "feat_projectors")):
            raise RuntimeError(f"Training-only module present in export state: {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    checks, notices = [], []
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample = _probe(512, device, dtype)
        traced = torch.jit.trace(model, sample, strict=True, check_trace=False)
        # Shape-dependent attention branches are traced at the deployment shape.
        # Explicit loaded-artifact checks below cover both fixed project sizes.
        temporary = path.with_name(path.name + ".tmp")
        torch.jit.save(traced, str(temporary))
        os.replace(temporary, path)
        loaded = torch.jit.load(str(path), map_location=device).eval()
        # Validate the same warmed execution plan used by the runtime benchmark.
        # CUDA graph fusion may only activate after the first profiling calls.
        for _ in range(20):
            loaded(sample)
        for size in (224, 512):
            image = _probe(size, device, dtype)
            expected, actual = model(image), loaded(image)
            _finite_logits(expected, size); _finite_logits(actual, size)
            difference = (actual.float() - expected.float()).abs()
            # Half precision may use different valid fused implementations.
            tolerance = 1e-3 if dtype == torch.float16 else 1e-5
            close = torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)
            agreement = float((actual.argmax(1) == expected.argmax(1)).float().mean())
            checks.append({"input_shape": list(image.shape), "max_absolute_logit_difference": float(difference.max()),
                           "argmax_agreement_fraction": agreement, "allclose": bool(close),
                           "absolute_relative_tolerance": tolerance})
            if not close or agreement != 1.:
                raise RuntimeError(f"Loaded TorchScript prediction mismatch at {size}: {checks[-1]}")
        # Zero input is a second content check at the actual deployment shape.
        zeros = torch.zeros(1, 6, 512, 512, device=device, dtype=dtype)
        expected, actual = model(zeros), loaded(zeros)
        _finite_logits(expected, 512); _finite_logits(actual, 512)
        if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance) or not torch.equal(actual.argmax(1), expected.argmax(1)):
            raise RuntimeError("Loaded TorchScript zero-input prediction mismatch")
        checks.append({"input_shape": list(zeros.shape), "content": "zero", "allclose": True,
                       "argmax_agreement_fraction": 1.,
                       "max_absolute_logit_difference": float((actual.float() - expected.float()).abs().max())})
        notices = list(dict.fromkeys(str(notice.message) for notice in caught))
    return {"status": "exported_prediction_validated", "path": str(path.resolve()),
            "sha256": _sha256(path), "serialized_bytes": path.stat().st_size,
            "format": "TorchScript", "contains_optimizer_or_loss": False,
            "contains_training_feature_adapter": False, "validation": checks,
            "validation_scope": "synthetic finite probes; frozen checkpoint identity is supplied by terminal receipt",
            "validation_runtime_warmups": 20,
            "tracing_warnings": notices}, loaded


def benchmark_artifact(module: torch.jit.ScriptModule, image: torch.Tensor) -> dict:
    """Synchronized batch-one latency of the loaded exported runtime graph."""
    device = image.device
    if device.type != "cuda":
        return {"status": "unavailable", "reason": "CUDA runtime measurement requires the authorized GPU; CPU export test only"}
    times = []
    with torch.no_grad():
        for _ in range(20):
            module(image)
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(200):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            module(image)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - started) * 1000)
    samples = torch.tensor(times, dtype=torch.float64)
    return {"status": "measured", "backend": "loaded TorchScript / PyTorch CUDA",
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "visible_gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "input_shape": list(image.shape),
            "input_dtype": str(image.dtype), "warmups": 20, "repeats": 200,
            "synchronization": "before and after every timed batch",
            "latency_ms_median": float(samples.median()),
            "latency_ms_mean": float(samples.mean()), "latency_ms_p95": float(torch.quantile(samples, .95)),
            "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
            "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "process_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "additional_peak_allocated_bytes": max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated),
            "memory_scope": "this process; baseline includes loaded artifact, probe, and any already resident model; additional peak is forward allocations"}


def measure_artifact(model: nn.Module, output_dir, graph_id: str, quantized: bool) -> dict:
    """Export/validate the current graph, then measure its real runtime on CUDA."""
    output_dir = Path(output_dir)
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    actual_quantized = any(isinstance(module, FixedQATConv2d) for module in model.modules())
    if actual_quantized != quantized:
        raise ValueError("quantized label differs from the actual supplied inference graph")
    if dtype != torch.float32:
        raise ValueError("The matched primary inference artifact uses float32 master parameters")
    result = {"graph_id": graph_id, "quantized": quantized,
              "main_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "auxiliary_training_parameters_in_artifact": 0,
              "logical_bitwidth_does_not_determine_artifact_bytes": True,
              "packed_lowbit_kernel": False if quantized else None}
    with quantization_mode(model, True):
        image = _probe(512, device, torch.float32)
        result["precision_map"] = precision_map(model)
        result["macs"] = count_inference_macs(model, image)
        artifact_path = output_dir / ("model_fakequant.ts" if quantized else "model_fp32.ts")
        try:
            artifact, module = export_and_validate(model, artifact_path)
            result["artifact"] = artifact
            try:
                result["runtime"] = benchmark_artifact(module, image)
            except Exception as error:
                result["runtime"] = {"status": "blocked", "reason": f"{type(error).__name__}: {error}"}
            del module
        except Exception as error:
            # Model accuracy is a separate terminal outcome. An unavailable or
            # mismatching export must remain explicit without discarding counts.
            result["artifact"] = {"status": "blocked", "reason": f"{type(error).__name__}: {error}",
                                  "prediction_validation_passed": False}
            if artifact_path.exists():
                unvalidated = artifact_path.with_name(artifact_path.stem + ".unvalidated.ts")
                if not unvalidated.exists():
                    artifact_path.rename(unvalidated)
                else:
                    unvalidated = artifact_path
                result["artifact"]["unvalidated_file_path"] = str(unvalidated.resolve())
            result["runtime"] = {"status": "blocked", "reason": "No prediction-validated inference artifact"}
        result["status"] = "simulation_only" if quantized else (
            "deployed_measured" if result["runtime"]["status"] == "measured" else
            "exported_unbenchmarked" if result["artifact"]["status"] == "exported_prediction_validated" else "blocked")
        if quantized:
            result["float16_reference"] = {"status": "pending_matching_native_FP_anchor",
                "reason": "FixedQAT uses float32 master parameters; the matching native FP anchor supplies a separate FP16 reference"}
        elif device.type != "cuda":
            result["float16_reference"] = {"status": "unavailable", "reason": "CPU test does not establish authorized GPU FP16 support"}
        else:
            try:
                half = copy.deepcopy(model).half().eval()
                half_image = image.half()
                with torch.no_grad():
                    reference, output = model(image), half(half_image)
                    _finite_logits(output, 512)
                half_artifact, half_module = export_and_validate(half, output_dir / "model_fp16.ts")
                result["float16_reference"] = {"status": "deployed_measured", "artifact": half_artifact,
                    "macs": count_inference_macs(half, half_image),
                    "runtime": benchmark_artifact(half_module, half_image),
                    "precision": "FP16 conv/BN master graph with native LiteMLA FP32 linear MatMuls",
                    "synthetic_difference_from_FP32": {
                        "max_absolute_logit_difference": float((reference - output.float()).abs().max()),
                        "argmax_agreement_fraction": float((reference.argmax(1) == output.argmax(1)).float().mean())},
                    "accuracy_scope": "synthetic export/reference check only; no claim of benchmark accuracy equivalence"}
            except Exception as error:
                result["float16_reference"] = {"status": "blocked", "reason": f"{type(error).__name__}: {error}"}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "artifact_measurement.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    result["measurement_receipt"] = str(path.resolve())
    result["measurement_receipt_sha256"] = _sha256(path)
    return result
