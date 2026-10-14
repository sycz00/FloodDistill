# Workstation integer-inference evidence

The workstation preparation and bounded experiments are complete. Fully integer inference remains **unproven** for both models. R4 has a PTQ candidate with minor source degradation under the declared pooled/equal-event/water-IoU tolerances. EfficientViT requires QAT to obtain an acceptable exported candidate in this bounded package. Native TensorRT/Jetson performance remains blocked by workstation builder CUDA35 and pending device execution. INT4 was not started.

## Protocol and provenance

Branch `research`, dirty checkout preserved. Exact prompt and `AGENT.md` were read; no historical checkpoint, experiment or Jetson environment was replaced. Sources were frozen before scientific runs; changes/failures were retained in separate run/repair directories. `protocol.json`, `execution_recipe_audit.json`, manifests, receipts, logs, source snapshots, original checkpoint hashes, architecture records and transfer checksums provide provenance.

R4 is the 12,230-parameter context-convolution LKD student, seed42, original checkpoint SHA256 `182c64cce20f01630f1a1dc7ae792ff8a45c38154d0cfafaf2a4275bb8a49ab9`. EfficientViT-B0 OFA ReLU6 has704,714 parameters, originalN2500 seed42, checkpoint SHA256 `d2a6c75ae464ebf5198b9bd4ac4bb0e9937c5676fd0e8a73b09697081d8b078b`. These are one-parent engineering comparisons, not new four-seed training evidence.

Calibration:32 hash-selected GDACS TRAIN224 tiles bilinearly upsampled to512, plus a32-image S11 TRAIN alternative at native512. Labels were not consumed for scale calibration. All89 S11 validation scenes were evaluated at512 with original masks, ignore-1, six bands B2/B3/B4/B8A/B11/B12, and registered mean/std normalization. The lightweight Jetson loader matched the established workstation loader exactly on three prespecified scenes and the complete split-ID order. Actual array/label hashes are included.

A metadata discrepancy is retained: the original protocol called calibration order 'smallest SHA256 filename'; this describes selection, while execution iterates selected asset filenames lexicographically. `execution_recipe_audit.json` records the exact executed order, and future launcher metadata is corrected. Scientific inputs/results were not silently amended.

PTQ compared MinMax/Percentile99.99, per-output-channel/per-tensor symmetric weights, GDACS/S11 TRAIN calibration, and asymmetric-activation ORT controls. Activations in TensorRT candidates are per-tensor symmetric signedINT8. Existing BN folding and ONNX infrastructure were reused; no new dependency/environment was installed. Static batch1/spatial512, opset13, IR10, and float-weight Q/DQ representation were used. ORT INT32 bias-DQ was repaired to a float initializer containing exactly the same rounded bias; original versus repaired output was checked for exact equality. These transformations describe quantization and compatibility, not actual device kernel precision.

## Complete source results

Fraction units. Failed rows and their repairs remain separate. CSV and JSON contain per-scene confusion matrices, event results, CPU operator inventories, memory and paired bootstrap intervals. Asymmetric controls are unsuitable for TensorRT8.4.

| Model | Variant | mIoU | Water IoU | Equal-event mIoU | Status |
|---|---|---:|---:|---:|---|
| R4 | float parent | 0.880144 | 0.787798 | 0.870600 | float |
| R4 | asymmetric_ort | 0.881344 | 0.789927 | 0.871076 | evaluated |
| R4 | minmax_pc | 0.874092 | 0.777470 | 0.863591 | evaluated |
| R4 | minmax_pt | — | — | — | failed |
| R4 | percentile_pc | 0.881789 | 0.791468 | 0.868475 | evaluated |
| R4 | repair_minmax_pt | 0.871986 | 0.773443 | 0.862898 | evaluated |
| R4 | s11_minmax_pc | 0.880426 | 0.788962 | 0.868146 | evaluated |
| R4 | QAT1-update smoke | 0.876310 | 0.780709 | 0.868350 | measured exported runtime |
| B0 | float parent | 0.859494 | 0.754017 | 0.853150 | float |
| B0 | asymmetric_ort | 0.113964 | 0.119677 | 0.114007 | evaluated |
| B0 | conv_only_percentile | 0.570156 | 0.227735 | 0.548556 | evaluated |
| B0 | minmax_pc | 0.055417 | 0.110275 | 0.061590 | evaluated |
| B0 | minmax_pt | — | — | — | failed |
| B0 | mm1_only_percentile | 0.581959 | 0.249313 | 0.564430 | evaluated |
| B0 | percentile_pc | 0.056700 | 0.110174 | 0.062876 | evaluated |
| B0 | repair_minmax_pt | 0.071248 | 0.106086 | 0.079345 | evaluated |
| B0 | s11_minmax_pc | 0.055258 | 0.110266 | 0.061441 | evaluated |
| B0 | QAT128 updates | 0.867043 | 0.766488 | 0.857336 | measured exported runtime |

R4 Percentile per-channel-weight PTQ scored0.881789 vs float0.880144 (pooled delta+0.001645; equal-event delta-0.002125). Its paired whole-event pooled95% interval is[-0.010629,0.016189]; equal-event interval[-0.011745,0.010204]. It passes the predeclared point-estimate engineering tolerances; this does not establish equivalence or superiority.

EfficientViT aggressive attention PTQ failed badly despite alternative datasets, calibration and symmetry. Preserving attention during Conv-only PTQ was insufficient; retaining only attention mm1 quantization was also insufficient. These are genuine negative quantization results, not builder failures. No activation swap, LayerNorm replacement, weight-equalization search or new integer plugin was added: the model already uses ReLU6/BN and has no LayerNorm, and numerical/export bottlenecks warranted a bounded QAT attempt first.

QAT used frozen folded BN, fixed all-convolution input/weight quantizers including the head, AdamW LR1e-4/weight decay0.05, batch4, seed1729, gradient clip5, focal gamma2 on512 frozen GDACS TRAIN cached-teacher hard-mask tiles. EfficientViT completed128 updates; R4 needed only a one-update gradient/reload/export smoke because PTQ met its gate. This is separately declared quantization adaptation, not continuation of original OFA/live-KD training. Residual, attention and resize operations remain floating in the simulation; there are no fully integer arithmetic claims.

EfficientViT exported QAT mIoU0.867043 vs parent0.859494 (pooled delta+0.007548, equal-event delta+0.004186). Paired whole-event pooled95% interval[-0.001216,0.017962] crosses zero; no accuracy-improvement claim is supported. The actual exported model passes all source point-estimate tolerances. Further QAT convergence/seeds were not investigated.

## Export/runtime failures and reconciliation

Both original model jobs failed strict optimized-logit parity checks after their PTQ experiments; their queue receipts remain failed and the original dependent collector remains blocked. R4 semantic parity passed with ORT optimization disabled (max error9.54e-7), while optimized execution changed logits by0.104900 and99.9668% of probe predictions agreed. Its actual source runtime was evaluated separately.

EfficientViT QAT strict parity still fails: unoptimized max error0.474149, optimized0.471155, optimized probe argmax agreement99.6456%. Removing Q/DQ from the export matches the same unrounded QAT master weights at max error2.73e-5 and100% probe argmax agreement. Layer diagnostics show initial convolution errors of order3e-6 followed by sparse larger rounding jumps. Quantization discontinuities amplifying floating-kernel differences are a supported inference, not a proven universal explanation. No tolerance was relaxed or failed parity relabeled passed.

The complete exported-runtime source evaluation gave0.867043 versus PyTorch fake-quant0.866661, paired pooled difference0.000382. The artifact is recommended for a native **probe** using its own runtime results and hashes. It must not be described as an exact replacement for PyTorch fake-quant predictions. `runtime_validation.json` preserves strict parity failure alongside structural checks and source results.

The per-tensor PTQ arm initially failed a scalar-bias reshape compatibility bug; repaired arms were run additively. An initial initialization-path error launched no model experiments. An initial collector imported an older frozen helper API; its log is retained, and completed evaluations were reused rather than repeated. Reconciled final evidence is `final_status.json`, `receipts/*_final.json`, and `results_final.csv`; original unsuccessful queues were not rewritten.

## Execution precision, latency, model size and memory

Optimized ORT graphs demonstrate partial integer kernels, not fully integer execution: R4 selected PTQ has12 QLinearConv and8 floating Conv nodes plus three Resize operations; EfficientViT QAT has4 QLinearConv,15 Conv,38 FusedConv, floating MatMuls/Div/Resize. Every learned convolution is surrounded by weight/input QDQ in the candidate graph, but the backend decides kernel execution and fusion. Float host inputs/outputs are disclosed. INT32 biases/accumulators and shape arithmetic must be distinguished from FP16/FP32 compute in the eventual engine audit.

The original Jetson build log explicitly used INT8+FP16. The live workstation TensorRT10.16 builder failed with CUDA35 before ONNX parsing or calibration. This is an environment blocker, not a calibrated/measured INT8 model failure. No native latency, engine size, GPU-memory footprint or Jetson power result exists for these new candidates.

The following isolated-process CPU diagnostic uses2 threads pinned to2 cores, batch1x6x512x512,20 warmups/100 timed `session.run` calls on a fixed source array. It excludes preprocessing/disk and ran under ordinary unrelated host load. The models were timed sequentially after the experiment workers ended. RSS includes Python/NumPy/ORT imports, session and intermediates; it is not model-only/device memory. Earlier per-arm timings varied with contention and resource admission and remain diagnostic records, not deployment comparisons.

| Model/variant | CPU mean ms | CPU median ms | CPU p95 ms | ONNX bytes | Peak process RSS MiB |
|---|---:|---:|---:|---:|---:|
| R4 float | 3.060 | 2.838 | 4.259 | 58892 | 81.20 |
| R4 PTQ percentile | 5.916 | 5.785 | 6.916 | 101639 | 73.54 |
| B0 float | 25.315 | 25.366 | 25.888 | 2897102 | 92.93 |
| B0 QAT128 | 40.167 | 35.111 | 66.148 | 3068929 | 95.60 |

QDQ ONNX can be larger than float ONNX because the JetPack5.0 float-weight workaround and quantization metadata are retained. File-size savings or the historical1.5MB engine figure cannot be assigned to these candidates before device compilation. CPU results do not support a Jetson speedup claim.

## Jetson compatibility and remaining work

Historical target: Python3.8.20/torch1.13 NVIDIA build, CUDA11.4/TensorRT8.4.1.5, ONNX1.17/ORT1.19.2. Device scripts use binding-index/execute_async_v2 APIs and legacy workspace properties, not TensorRT10 IO APIs. Workstation `dynamo=False` exports are not commands for device torch1.13. Standalone device CLI was parsed and invoked under Python3.8, but actual TensorRT8.4 import/build/run is pending. Its version gate, missing-backend receipts, refusal to overwrite attempts and precision audit fail closed.

Per-tensor activation scaling is required for the target TensorRT quantization route; asymmetric/per-channel-activation research controls must not be treated as deployable. See [TensorRT8.4 quantization guidance](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/pdf/TensorRT-Developer-Guide.pdf). IR10 was selected for the preserved ORT1.19 reader; see [ORT compatibility](https://onnxruntime.ai/docs/reference/compatibility.html). QDQ alone describes quantization boundaries and does not guarantee integer kernels; see [ORT quantization formats](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

The new handover gives reproducible strict/no-FP16/mixed builds, fixed-array evaluation, kernel/wall timing boundaries, inspector validation, debugging, expected outputs and original-environment constraints. Removing FP16 can leave FP32 fallback; only a layer/kernel audit can support fully integer inference. Native-bias accumulation, boundary IO conversion and shape indices require explicit reporting. No-FP16 and strict builds are prepared, not executed on Xavier.

INT4 remains closed because fully integer INT8 execution is unverified. No four-bit simulation/deployment experiment or modern-runtime substitution was added. Remaining device tasks are native builds, precision/kernel audit, source parity, accuracy gates, latency, engine size, memory and optional power. Do not change geometry or normalization to improve results.

## Workstation controls and persistence

The existing detached queue was used with closed stdin, logs, process identity, atomic receipts and finite ceilings. A two-job sentinel completed after its launching process exited; stdin was `/dev/null`, and children had independent sessions. Pause and stop were exercised. Editor closure/logout/reboot were not tested. Final owned workers have ended; no jobs are currently queued for launch, and original failed/blocked records remain intact.

For audit/status (from the repository):

```bash
python experiments/revision/jetson_integer_20261014T101741Z/code/experiments/revision_queue.py status --manifest experiments/revision/jetson_integer_20261014T101741Z/queue_manifest.json
python experiments/revision/jetson_integer_20261014T101741Z/code/experiments/revision_queue.py status --manifest experiments/revision/jetson_integer_20261014T101741Z/repairs/cpu_20261014T102439Z/queue_manifest.json
```

The same wrapper accepts `pause`, `resume`, and `stop`. Do not resume a historical failed scientific job as an unrecorded retry; start a new attempt with reconciled inputs. The current launcher source contains the scalar-bias/resource-limit/parity-reporting repairs; executed snapshots and their original failures remain unchanged.

The transfer bundle additionally retains the executed source snapshot, subsequent repair/current source files and512 cached QAT TRAIN arrays. Modern workstation training/export sources are not assumed compatible with device torch1.13.
