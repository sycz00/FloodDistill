# Four-seed W8A8 QAT candidates (15 Oct 2026)

EfficientViT-B0 (ReLU6 OFA, N = 2,500) and the 12,230-parameter R4 LKD student, seeds 42 / 64 / 91 / 107,
quantization-aware trained with the Jetson integer recipe of `integer_models/` (BN folded, every convolution
including the head W8A8, per-tensor symmetric activations, per-channel weights, 128 updates). Seed 42 of B0 is
byte-identical to `integer_models/evit_b0_qat.pt`. R4 additionally ships its per-seed PTQ export
(`r4_s*_int8_ptq.onnx`, percentile calibration, per-channel weights), the variant selected on 14 Oct.

| File | Content |
|---|---|
| `b0_s<seed>_qat128.pt`, `r4_s<seed>_qat128.pt` | QAT checkpoint with fixed quantizer buffers (`{"model", "recipe", "updates"}`) |
| `*_int8_qat128.onnx` | QDQ export, static 1×6×512×512, opset 13 |
| `r4_s<seed>_int8_ptq.onnx` | R4 PTQ export |
| `MANIFEST.json` | SHA-256, source paths, parents and source-validation results |

Source-validation (S11 val, exported ONNX runtime) mIoU change vs each seed's float parent, points:

| Seed | B0 QAT | R4 QAT | R4 PTQ |
|---|---|---|---|
| 42 | +0.75 ✔ | −1.45 ✘ | +0.16 ✔ |
| 64 | −2.07 ✘ | −0.86 ✔ | +0.01 ✔ |
| 91 | −1.27 ✘ | −1.68 ✘ | −35.96 ✘ (collapse) |
| 107 | −0.69 ✔ | −1.00 ✔ | −0.44 ✔ |

✔ = within the 14 Oct tolerances (pooled and equal-event ≥ −1 point, water IoU ≥ −2 points). Use per seed:
B0 → QAT (seeds 42, 107 pass); R4 → PTQ except seed 91, where QAT is required. Fully integer execution is not
verified; build on the Jetson with `jetson_repro/integer_tools.py` as described in `integer_models/README.md`.
Details: `experiments/revision/qat_seeds_20261015/README.md`.
