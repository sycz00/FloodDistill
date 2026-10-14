# Jetson Xavier NX handover

Objective: resolve reviewer concern about INT8/FP16 inference for EfficientViT-B0 OFA and the 12,230-parameter R4 LKD student. Determine actual integer coverage and accuracy; retain failures. Work exclusively on the Jetson. No paper claims of fully integer execution are supported yet.

## Current workstation evidence

All 89 Sen1Floods11 validation scenes were evaluated at 512 pixels. The same normalized arrays, ignore masks and original bilinear output geometry are included. Calibration used 32 frozen TRAIN images; GDACS and S11 TRAIN alternatives were compared. Ten source event groups, pooled/equal-event metrics and paired whole-event bootstrap intervals are provided. No external benchmarks were used for selection.

| Model | Float source mIoU | Recommended device probe | Probe source mIoU |
|---|---:|---|---:|
| R4, 12,230 parameters | 0.880144 | `R4/percentile_pc.onnx` | 0.881789 |
| EfficientViT-B0 OFA ReLU6, 704,714 parameters | 0.859494 | `B0/qat128/model_qdq.onnx` | 0.867043 |

R4 PTQ includes every convolution and its head, symmetric scalar activation scales, per-channel weights, percentile 99.99 calibration. Remaining CPU float kernels and resize operations are documented; this is not proof of full INT8. Its one-update QAT smoke verifies gradients/reload/export; it is not a completed accuracy-optimization experiment.

B0's aggressive Conv/MatMul/Add PTQ collapsed source mIoU to approximately 0.055–0.071. Conv-only and first-attention-MatMul variants reached approximately 0.570 and 0.582. Those negative artifacts are retained. The QAT candidate used 128 AdamW updates, batch4, LR1e-4, weight decay0.05, focal gamma2 on cached GDACS teacher hard masks, frozen folded BN, no live teacher/KD/OFA retraining. All convolution input/weight rounding is trained, including the head; residual, attention normalization and resize arithmetic remains floating in the workstation path.

B0 QAT strict logit parity **failed** (up to ~0.474 unoptimized and ~0.471 optimized on a fixed calibration image). The unrounded export matches the same master weights (max error2.73e-5). Early convolution differences initially near3e-6 become sparse rounding jumps. The actual exported runtime was evaluated separately over all89 scenes: mIoU0.867043 vs PyTorch fake-quant0.866661. Do not alias their predictions or claim exact parity. Whole-event paired drift is recorded in `B0/qat128/runtime_validation.json`.

## Environment: verify, preserve

Historical `jetson_repro/` specifies JetPack5.0.2/L4TR35.1, CUDA11.4, cuDNN8.4.1, TensorRT8.4.1.5, Python3.8.20, NVIDIA torch1.13.0a0+d0d6b1f2.nv22.10, NumPy1.24.4, ONNX1.17.0 and ORT1.19.2. These are historical specifications, not live observations of your device. Start by recording actual versions, `/etc/nv_tegra_release`, platform and GPU/tegrastats availability. Use the existing `flood` environment. Do not reinstall/upgrade libraries, change clocks/power modes, or replace the old deployment.

Workstation TensorRT10.16.0.72 failed builder initialization with CUDA35 before parsing/calibration; no native engines, Jetson latency, engine size or device-memory figures were measured. PyTorch CUDA allocation succeeded, but experiments stayed CPU-only.

## Transfer layout

Extract the workstation `jetson_bundle.tar` into a NEW directory. Run all commands from `bundle/`:

```text
bundle/
  SHA256.json
  protocol.json, execution_recipe_audit.json, assets_manifest.json
  deployment_manifest.json, final_status.json, results_final.csv
  HANDOVER_JETSON.md, WORKSTATION_REPORT.md
  tools/integer_tools.py, tools/verify_bundle.py
  R4/model_float.onnx, percentile_pc.onnx, checkpoint_float.pt
  R4/qat_smoke/checkpoint_qat.pt, model_qdq.onnx, repaired_validation.json
  B0/model_float.onnx, checkpoint_float.pt
  B0/qat128/checkpoint_qat.pt, model_qdq.onnx, runtime_validation.json
  calibration/*.npy, calibration_s11/*.npy
  source_val/*.npz
  outputs/                 # create on the Jetson
```

The bundle also contains `workstation_sources/` (executed snapshot and repair/current source files) and512 cached224-pixel training arrays in `workstation_qat_training/`. These are provenance/workstation reproduction inputs; device PyTorch1.13 training/export compatibility has not been asserted.

Other ONNX files and failures are diagnostic controls. Never build from `*.optimized.onnx` (workstation-specific ORT graphs), `*_prepared.onnx`, `*_ort.onnx` (INT32 bias-DQ), or `asymmetric_ort.onnx` (unsupported TensorRT zero points). Use only the candidates in `deployment_manifest.json`. No workstation engine is portable to Xavier.

## Execution order and commands

1. Verify transfer before generating local outputs:

```bash
python tools/verify_bundle.py
python -c 'import sys,numpy,onnx,onnxruntime,tensorrt; print(sys.version,numpy.__version__,onnx.__version__,onnxruntime.__version__,tensorrt.__version__)'
mkdir -p outputs/R4/attempt1 outputs/B0/attempt1
```

2. Reproduce float/QDQ CPU accuracy to distinguish an environment problem from TensorRT conversion. `source_val` arrays are already normalized float32 `[6,512,512]`; never normalize or resize them again. Mask -1 is ignored. Event names are embedded.

```bash
python tools/integer_tools.py run --backend ort --model R4/percentile_pc.onnx --assets source_val --receipt outputs/R4/attempt1/ort.json
python tools/integer_tools.py run --backend ort --model B0/qat128/model_qdq.onnx --assets source_val --receipt outputs/B0/attempt1/ort.json
```

Run corresponding `model_float.onnx` baselines with distinct receipt names. Expect integer confusion matrices and mIoU near the workstation values; investigate any material discrepancy.

3. Attempt strict native builds first, retaining exit2 receipts and full logs:

```bash
python tools/integer_tools.py build --mode strict --model R4/percentile_pc.onnx --engine outputs/R4/attempt1/strict.engine --receipt outputs/R4/attempt1/strict_build.json > outputs/R4/attempt1/strict_build.log 2>&1
python tools/integer_tools.py build --mode strict --model B0/qat128/model_qdq.onnx --engine outputs/B0/attempt1/strict.engine --receipt outputs/B0/attempt1/strict_build.json > outputs/B0/attempt1/strict_build.log 2>&1
```

The builder uses TensorRT8.4 binding APIs, INT8, detailed profiling, and `OBEY_PRECISION_CONSTRAINTS`; it forces INT8 for computational layers. QDQ precision restrictions may take precedence: inspector evidence is decisive. A successful build alone is insufficient. Inspect `*.engine.layers.json`: list floating compute, boundary reformats and unknown layers. `full_integer_tensor_path` includes host-IO reformats and is intentionally conservative; separately disclose boundary conversion, INT32 biases/accumulation and shape indices. Tensor formats alone cannot certify internal arithmetic. Confirm tactics/kernels before making a fully integer claim.

4. Independently try removing FP16 with allowed FP32 fallback and then the mixed reference:

```bash
python tools/integer_tools.py build --mode no_fp16 --model R4/percentile_pc.onnx --engine outputs/R4/attempt1/no_fp16.engine --receipt outputs/R4/attempt1/no_fp16_build.json > outputs/R4/attempt1/no_fp16_build.log 2>&1
python tools/integer_tools.py build --mode no_fp16 --model B0/qat128/model_qdq.onnx --engine outputs/B0/attempt1/no_fp16.engine --receipt outputs/B0/attempt1/no_fp16_build.json > outputs/B0/attempt1/no_fp16_build.log 2>&1
```

Repeat using `--mode mixed` and distinct engine/receipt/log paths for matched INT8+FP16 controls. Do not describe no-FP16 as fully integer; FP32 may remain. Do not substitute another backend after a failed build. The tools reject existing output paths: create attempt2 for retries and record why.

Equivalent diagnostic trtexec command (no strict certificate):

```bash
/usr/src/tensorrt/bin/trtexec --onnx=R4/percentile_pc.onnx --int8 --workspace=1024 --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=outputs/R4/attempt1/trtexec_layers.json --saveEngine=outputs/R4/attempt1/trtexec.engine
```

QDQ contains fixed scales, so do not attach a random-range calibrator. `--fp16` is deliberately omitted in this example. The native Python builder is preferred for fail-closed receipts and inspection.

5. Evaluate ONLY successfully deserialized engines, with float host bindings (integer IO requires explicit scale encoding and another runner):

```bash
python tools/integer_tools.py run --backend trt --model outputs/R4/attempt1/no_fp16.engine --assets source_val --receipt outputs/R4/attempt1/trt_eval.json --warmup 20 --iterations 100
python tools/integer_tools.py run --backend trt --model outputs/B0/attempt1/no_fp16.engine --assets source_val --receipt outputs/B0/attempt1/trt_eval.json --warmup 20 --iterations 100
```

The runner checks shape/finite outputs and reports all89 per-scene confusion matrices. Source gates are max0.01 pooled/equal-event mIoU drop and0.02 water-IoU drop vs the corresponding float parent. Compare per-scene counts, and carry whole-event uncertainty into final tables. Source gates are engineering tolerances, not statistical equivalence.

## Benchmarking and debugging

Batch1, fixed1x6x512x512, warmup20, iterations100. Report median/mean/p95 kernel time via CUDA events separately from wall inference including H2D/D2H; both exclude preprocessing/disk. Save engine bytes, engine device memory, binding-buffer sizes and peak device/process memory where measured. Repeat under a recorded stable temperature/load and existing power mode, without changing global settings. Measure power with existing tegrastats facilities only if available; no substitute power estimate. Retain raw logs and device/version/engine/input hashes.

Check every Conv/head, residual Add, bilinear Resize, dynamic attention MatMul and normalization Div. Both models use BN and ReLU6; there is no LayerNorm to replace. B0 attention division is an important precision/numerical bottleneck. Never change interpolation or apply argmax before upsampling: that changes the registered model. Known historical FP16 attention overflow may require FP32 mm1/mm2/normalization for the mixed control; generate layer names from the NEW graph, not old hardcoded names, and mark the repair as a separate precision recipe.

Exports use static batch1/spatial512, opset13, IR10, symmetric signedINT8/scalar zero points, float-weight Q/DQ workaround and float rounded-bias initializers to avoid TRT8.4 INT32 bias-DQ. Engine kernels may still use INT32 bias/accumulation. Legacy per-channel activations and asymmetric zero points are ORT research controls, not this TensorRT deployment path. Workstation export calls (`dynamo=False`) require torch2.x; do not run them under device torch1.13. Checkpoints permit further workstation QAT, but the device continuation starts from supplied ONNX.

INT4 remains CLOSED until R4 has demonstrably native integer execution with minor degradation. Do not launch simulated INT4 or apply modern TensorRT INT4 support statements to Xavier/TRT8.4. Any full-integer graph/kernel rewrite, DLA route or integer output/interface is a new variant requiring source validation; do not silently amend these candidates.

Deliver a device ledger distinguishing built/evaluated, failed, unsupported, mixed, no-FP16-with-FP32, fully-integer-unverified, and genuinely certified integer cases. Include the original negative results and every incompatibility. All required context is in this bundle.
