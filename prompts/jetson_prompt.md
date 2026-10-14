# Objective

You are working inside the existing research project and must act as an experienced ML systems researcher, quantization expert, and software engineer. Your goal is to address the remaining reviewer concern regarding mixed precision inference while producing artifacts that are directly usable for the paper revision and the upcoming Jetson Xavier NX validation.

This is **not** exploratory research. The goal is to produce evidence, reproducible pipelines, and practical implementations while minimizing unnecessary engineering. Reuse the existing project infrastructure whenever possible.

Do not introduce new frameworks unless there is a strong technical justification. Respect the existing project structure and previous experiments.

---

# Research Goal

Demonstrate whether our proposed compressed models can operate with fully integer inference while maintaining acceptable accuracy.

The reviewer concern is specifically about mixed precision (INT8 + FP16). We therefore want to investigate whether mixed precision can be removed or significantly reduced.

This work should generate publishable evidence regardless of outcome.

---

# Tasks

## Task 1 — EfficientViT-B0 OFA Full INT8

Investigate how to obtain a fully INT8 EfficientViT-B0 OFA model.

Current goal:

- remove FP16 computations from remaining layers
- eliminate mixed precision wherever possible
- preserve accuracy as much as possible

Study existing implementations and known deployment techniques before modifying the code.

Possible approaches include (when applicable):

- integer output heads
- integer LayerNorm replacements
- BatchNorm folding
- activation rescaling
- quantized residual paths
- quantization-aware graph transformations
- TensorRT compatible integer operators
- alternative calibration procedures
- QAT if PTQ alone is insufficient

Do not blindly implement everything.

Instead:

- identify promising methods
- justify why they should work
- implement only the most promising ones
- compare them experimentally

Document:

- unsupported operators
- precision bottlenecks
- TensorRT limitations
- ONNX export issues
- accuracy changes
- latency changes

---

## Task 2 — Fully Quantize the 12k Parameter Network

Take the smallest (~12k parameter) student network.

Attempt full INT8 inference.

Start with PTQ using best practices.

If degradation is too large, investigate QAT.

Experiment with:

- calibration datasets
- calibration methods
- per-channel vs per-tensor quantization
- symmetric vs asymmetric quantization
- activation calibration strategies
- weight equalization if appropriate

Only add complexity when required.

Produce:

- accuracy comparison
- latency comparison
- model size
- memory footprint
- operator compatibility

---

## Task 3 — Investigate INT4

Only continue to INT4 if Task 2 demonstrates that INT8 is successful with only minor degradation.

Do not waste time on INT4 if INT8 remains unsolved.

For INT4:

Investigate realistic deployment options supported by Jetson Xavier NX.

Evaluate:

- partial INT4 quantization
- mixed INT4/INT8
- weight-only INT4
- activation INT8 + weight INT4
- TensorRT support
- simulated execution on workstation

The objective is not perfect performance but determining feasibility.

Clearly distinguish between:

- simulated results
- deployable results
- unsupported configurations

---

## Task 4 — Prepare Jetson Deployment

The workstation should prepare everything required for later execution on the Jetson Xavier NX.

Prepare reproducible pipelines for both:

- EfficientViT-B0 OFA
- 12k parameter student

Support:

- PTQ
- QAT

Ensure the generated artifacts can later be transferred to Jetson with minimal modification.

This includes:

- exported checkpoints
- ONNX models
- calibration assets
- TensorRT conversion scripts if needed
- evaluation scripts
- inference scripts
- documentation of required commands

Avoid Jetson-specific assumptions unless verified.

---

## Task 5 — Respect Jetson Environment

Inspect the existing

jetson_repro/

folder.

Understand:

- library versions
- TensorRT version
- CUDA version
- PyTorch version
- ONNX version
- existing deployment scripts
- unsupported APIs

Do not overwrite or modernize the environment.

Instead adapt the workstation outputs to remain compatible with the Jetson environment.

Document every incompatibility discovered.

---

## Task 6 — Handover for Jetson Agent

Produce a concise but complete handover document for another autonomous agent working exclusively on the Jetson Xavier NX.

The handover should contain:

- project objective
- current status
- completed workstation work
- exported artifacts
- remaining Jetson tasks
- execution order
- expected outputs
- validation checklist
- debugging hints
- TensorRT conversion procedure
- benchmarking protocol
- expected directory structure
- known limitations
- known incompatibilities

The Jetson agent should be able to continue without requiring additional context.

---

# Experimental Protocol

Every experiment must include:

- motivation
- implementation details
- configuration
- reproducibility information
- quantitative results
- failures
- interpretation

Negative results are valuable.

Do not discard failed approaches.

Explain why they failed and what that implies.

---

# Success Criteria

Highest priority:

1. Fully INT8 EfficientViT-B0 OFA.
2. Fully INT8 12k parameter network.
3. Robust PTQ pipeline.
4. Robust QAT pipeline.
5. Jetson-ready deployment artifacts.
6. Complete Jetson handover.

Only investigate INT4 after INT8 has been demonstrated to work with acceptable degradation.

---

# Constraints

- Reuse existing project code whenever possible.
- Avoid unnecessary refactoring.
- Avoid introducing new dependencies unless clearly justified.
- Keep implementations reproducible.
- Prefer robust engineering over experimental hacks.
- Preserve compatibility with the existing project and previous experiments.
- Keep detailed notes suitable for inclusion in the reviewer response and supplementary material.