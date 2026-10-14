# Compact Jetson model candidates

Only the two selected model checkpoints and their deployment ONNX exports are included. Full experiments, calibration/evaluation arrays and transfer archives remain local and ignored by Git.

- `r4_float.pt`: 12,230-parameter R4 LKD model, original float weights; PTQ scales/quantized graph are in `r4_int8_ptq.onnx`.
- `evit_b0_qat.pt`: EfficientViT-B0 OFA ReLU6 checkpoint after128 QAT updates, including fixed quantizer buffers; export is `evit_b0_int8_qat.onnx`.
- `r4_architecture.py`: original R4 construction definition; EfficientViT uses the existing repository student factory and pinned upstream implementation.

`MANIFEST.json` records SHA256, source files and measured source-validation results. Both exports use static1x6x512x512 input, opset13/IR10, scalar symmetric activation scales and per-channel weights. They are candidates prepared for Jetson; fully integer execution has not been verified. B0 strict logit parity failed, and its actual exported-runtime accuracy was evaluated separately. Floating attention/residual/resize arithmetic remains a known concern.

Preserve the existing JetPack5.0.2/TensorRT8.4 environment. Build locally on the Jetson using `jetson_repro/integer_tools.py`; workstation engines are not portable. From the repository root:

```bash
python jetson_repro/integer_tools.py build --mode no_fp16 --model jetson_repro/integer_models/r4_int8_ptq.onnx --engine /tmp/r4_attempt1.engine --receipt /tmp/r4_attempt1.json
python jetson_repro/integer_tools.py build --mode no_fp16 --model jetson_repro/integer_models/evit_b0_int8_qat.onnx --engine /tmp/b0_attempt1.engine --receipt /tmp/b0_attempt1.json
```

Use `--mode strict` with new output names for a strict precision attempt. No-FP16 can retain FP32 fallback; inspect layers and kernels before making integer claims. Device accuracy/latency/memory checks are pending. Calibration and fixed-array validation assets are in the local full Jetson bundle when needed for transfer.
