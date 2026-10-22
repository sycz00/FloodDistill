# Workstation repair and evidence boundaries — 22 October 2026

The user relayed a Jetson report that micro seed 107 pure INT8 falls back to floating kernels and predicts no water. The actual new failing build/inspector/source receipts were not supplied to this workstation. Thus the narrow-channel diagnosis is a reported hypothesis supported by inspecting the source graph, not a directly reproduced native failure. The other three micro pure models are not declared failed merely because they share the architecture.

The source graphs have feature widths 4/6/8, seven depthwise convolutions and narrow decoder layers. NVIDIA's [TensorRT8.4 API documentation](https://developer.nvidia.com/docs/drive/drive-os/6.0.5/public/drive-os-tensorrt/api-reference/docs/python/infer/Graph/Layers.html) documents channel-per-group restrictions for INT8 grouped convolution. Exact availability and tactic choice on Xavier NX still require the target build/inspector. Floating fallback alone does not prove the cause of the all-background output; IO, actual accuracy and precision certification are independently checked.

The fixed repair embeds every micro feature map in 16 channels, expands depthwise weights to a dense block-diagonal matrix (group 1), and zero-pads kernels/biases. It retains the original real weights and scales; extra per-channel scales are positive placeholders for identically zero channels. The first layer retains six input bands, and an INT8 Slice returns only the two original class channels after the final shared-scale quantizer. There is no new training, calibration, validation search, output-threshold selection or external readout. Physical stored coefficients and executed work increase; original learned parameter counts must not be confused with deployed zero-filled tensor sizes. Re-measure size, memory, time and energy.

All four graphs passed 89-source-image optimized CPU accuracy against their genuine original float parents. Absolute losses (percentage points, a negative loss is an improvement):

| Seed | Method retained | Pooled mIoU | Equal-event mIoU | Water IoU |
|---|---|---:|---:|---:|
|42|PTQ|0.574139|0.247804|1.040951|
|64|PTQ|-0.095550|-0.176666|-0.192540|
|91|PTQ|0.273427|0.291991|0.477802|
|107|QAT|0.922800|0.471975|1.621649|

All three losses are <=2 points per seed. Input/mask hashes and all 89 raw confusion rows are included. Original inputs, masks, split, labels and acceptance thresholds are unchanged.

The unrounded old-versus-repaired graph outputs pass atol=rtol=.001 on all 89 images for each seed (maximum absolute errors below 2.8e-5). This is structural preservation, not exact QDQ equivalence. Optimized INT8 logits are byte-identical on 89/89 images for42 and91,72/89 for64,70/89 for107. Seed107 preserves optimized argmax on every pixel of all 89 images; seed 64 has small quantization-boundary changes (minimum per-image argmax agreement 0.999946594). Semantic (optimizations disabled) logits are also recorded separately, including failures. The inherited QAT107 PyTorch/export strict-equivalence failure stays false; this repair does not relabel it as passing. Per-image evidence and metrics are in each REPAIR_VALIDATION.json.

Native integer placement, IO, source accuracy, external accuracy and benchmarks are PENDING_JETSON. A separate finite package with four paired native FP32 baselines and four pure repair candidates leaves the old 48-slot handoff and every failed engine intact. Floating/unknown inspector formats or an all-background source result block the repair. Source accuracy qualification and strict equivalence are separate fields. Original tiny/B0 entries and all datasets are unchanged.

Workstation sources and logs are retained under MajorRev/revision_package_20261021/13_micro_int8_repair_20261022. Parent Git commit:2f6640ce134dc2bbe4a8de4d3591b1741037a193. The final repair commit/export identity is supplied beside the transfer package. This folder contains all deployable inputs except the exact 353 prepared arrays and existing raw datasets, reused only by frozen hashes.

The repair builder also fixes both public bindings to LINEAR NCHW and records their vectorized dimension, element components, dtype and format. The runner refuses packed bindings, and pure admission requires linear INT8 input/output. This prevents interpreting a padded CHW4/CHW32 host buffer as dense six-band/two-class arrays; it is a defensive IO guard, not evidence that packing caused the original failure. Internal INT8 layouts remain available. See NVIDIA's [TensorRT8.4 buffer-format guidance](https://developer.nvidia.com/docs/drive/drive-os/archives/6.0.4/tensorrt/developer-guide/index.html).
