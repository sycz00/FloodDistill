> Current execution handoff: [device_handoff/README.md](device_handoff/README.md) and [the Jetson prompt](../../prompts/jetson_prompt.md). The section below records the earlier preparation stage. QAT/PTQ candidates are now supplied in the current handoff; do not rerun its legacy training plan.

# Jetson follow-up prepared on 22 October 2026

This directory adds four existing R4-micro float ONNX exports, their complete
source-parity receipts, a separate four-seed QAT plan, and a smaller-transfer
runtime. It changes no historical checkpoint, engine, protocol or frozen source.
No QAT training or native Jetson inference was run for this addition.

## What the received bundle establishes

Current512 R4 has 16 native engines: four seeds at FP32, FP16, mixed INT8 and
full INT8. Full benchmark inventories were evaluated, but the supplementary dry
sets and 97-event panel were not evaluated on-device. Frozen source acceptance
passes seeds 64/91/107 and rejects 42 in both INT8 modes. The later relaxed rule
is post-hoc. QAT was never run on these current512 checkpoints. The older R4
seed91 PTQ collapse and QAT recovery belong to another cohort.

EfficientViT-B0 has only partial original-protocol device evidence. Older QAT
weights are not the current512 B0 parents. Do not combine their float and INT8
rows into a matched four-seed current512 precision comparison.

## R4-micro: exported and qualified for float device checks

`micro_exports/MANIFEST.json` identifies all four 1,266-parameter parents and
exports. Each seed passed ORT semantic and optimized versus PyTorch logit parity
on all 89 frozen source-validation images, at atol=rtol=0.001. The models retain
the trained bilinear interpolation. These receipts certify the workstation
float export; PTQ, TensorRT build, native parity, accuracy and speed remain unrun.

From the repository root on Jetson, build each float export with the existing
builder, using a fresh output directory. For example:

```bash
python3 jetson_repro/integer_tools.py build \
  --model jetson_repro/finalize_f_20261022/micro_exports/s42/R4_micro_s42_float.onnx \
  --engine /tmp/R4_micro_s42_fp32.engine --mode fp32 \
  --receipt /tmp/R4_micro_s42_fp32_build.json
```

Repeat all four seeds at FP32 and FP16. Check full validation against the float
parent before any PTQ recipe is admitted. PTQ must reuse the frozen current512
32-image calibration and `percentile_pc` recipe with the original accuracy limits;
there are no calibrated R4-micro QDQ exports in this addition. Benchmark every
repeat and retain failed builds/accuracy gates. Do not choose a seed from tests.

## Consistent QAT: prepared, not run

`qat_plan/PLAN.json` pins the existing recipe and four current R4 parents. It
does not mix successful PTQ seeds with a QAT rescue in one purported QAT mean.
The recipe is 128 updates per seed, batch4, AdamW 1e-4, weight decay .05,
cached current512 training hard masks, focal gamma2, seed1729, final update.
This is the inherited hard-target recipe, not probability-map distillation;
therefore a negative result cannot establish that every QAT method is ineffective.

Preparation already ran. If this separate experiment is later commissioned,
execute on the workstation in its existing `esi` environment with a fresh output:

```bash
/home/schmalstieg/miniconda3/envs/esi/bin/python \
  jetson_repro/finalize_f_20261022/qat_followup.py \
  --output /tmp/r4_qat_new_attempt --execute
```

This is CPU-only, two threads, 512 total updates, no new teacher inference.
The script checks full89 float construction parity before each fit and full89
QAT export parity plus the frozen accuracy gate before device admission. Failed
attempts remain failed; no LR sweep, extra updates or retry for accuracy. It does
not launch automatically. All four external evaluations and the dry readouts
would follow source admission; the 97-event panel stays closed to further tuning.

## Transfers: native runner ready for validation

`mask_transfer.py` keeps the TensorRT engine unchanged, reuses input quantization
buffers and emits the binary class decision in a CUDA kernel into a uint8 buffer.
Ties go to background, exactly like two-channel argmax. It returns 262,144 bytes
instead of 2,097,152 bytes of FP32 logits; integer input is 1,572,864 bytes instead
of 6,291,456 float bytes. Float-I/O engines still transfer float input. Quantizing
host float data on the GPU would still require the float H2D transfer, so it does
not by itself reduce input traffic.

TensorRT 8.4 has no UINT8 tensor datatype. The mask is produced by the separate
CUDA kernel, not a claimed native UINT8 engine binding. See NVIDIA's
[8.4 datatype API](https://github.com/NVIDIA/TensorRT/blob/release/8.4/include/NvInferRuntimeCommon.h)
and the [8.4 ONNX parser operator list](https://github.com/onnx/onnx-tensorrt/blob/8.4-GA/docs/operators.md).

```bash
python3 jetson_repro/finalize_f_20261022/mask_transfer.py --self-test
python3 jetson_repro/finalize_f_20261022/mask_transfer.py \
  --engine /path/to/R4_s64_ptq_full_int8.engine \
  --build-receipt /path/to/R4_s64_pure/build.json \
  --io /path/to/R4_s64_io.json \
  --assets /path/to/complete/s11val_assets --expected-images 89 \
  --out /tmp/R4_s64_mask_transfer_attempt1.json
```

It requires exact mask agreement with the same engine's original logits path on
every supplied image before writing a success receipt. It reports host preparation,
H2D, engine, CUDA mask, D2H and wall time, with repeated-image timing identified.
CPU encoding/tie fixtures pass here; CUDA compilation and execution remain untested
until Jetson. Use all four seeds and all existing benchmarks/dry sets for a final
performance claim. Compare equally instrumented boundaries; sensor normalization,
disk IO and the host acquisition pipeline are excluded.

## Other compression ideas

- **Nearest/pixel shuffle:** defer the model change. The 8.4 parser lists Resize
  and DepthToSpace without INT8 support; neither name guarantees an integer path.
  Nearest also changes the trained function. The existing fixed ConvTranspose
  variant is already measured and slower. A replacement needs a separate source
  accuracy gate, native layer proof and, if fine-tuned, a new bounded experiment.
- **DLA:** deferred as requested.
- **INT4 and structured sparsity:** no implementation in this package; the
  installed Xavier/Volta TensorRT8.4 deployment remains the target. No benefit is
  claimed from fake quantization, smaller weight counts or a newer runtime's API.

No push, deployment, power-mode change or foreign-process modification is performed.
