# Jetson: validate the four repaired micro pure-INT8 candidates

Use this separate `micro_int8_repair_20261022` package. It repairs the micro representation without retraining or recalibrating. The existing `device_handoff` and its 48-slot run are unchanged. Retain the reported seed-107 failure and every other old result. This is an additive four-seed repair readout, not a replacement claim that the old models succeeded.

First let the existing heavy controller finish, or use its documented PAUSE control and wait for its current child to exit. Preserve its attempt, logs and raw-data check. Do not run two heavy jobs together, kill a worker, alter clocks/power, install software, merge/reset/clean the dirty research checkout, or commit/push results. An independent unpacked repair export is allowed; record its GIT_EXPORT_IDENTITY.json. Keep all four local device commits and unrelated files. A repair-package lock prevents duplicate repair controllers but does not lock the old handoff: verify that the old controller is finished or paused with no child before launch and before resume.

The source graphs have feature widths 4/6/8. These candidates embed them in 16 channels with zero padding and replace depthwise kernels with dense block-diagonal kernels. Learned nonzero weights, original quantization scales, normalization and labels are retained. The classifier's extra channels are discarded by an INT8 Slice, preserving six-band input and two-class INT8 output. This is a new deployment representation with extra stored zeros/work; do not assume the original micro speed, size, energy or memory advantage survives. All four pass actual optimized CPU source accuracy under the user's absolute 0.02 limit on pooled mIoU, equal-event mIoU and pooled water IoU. Unrounded all-89 structural parity passes at atol=rtol=.001. Integer logits are exactly preserved on all 89 images only for seeds 42/91; seeds 64/107 have explicit kernel-rounding differences. Seed107 also retains its pre-existing QAT export-equivalence failure. Accuracy and strict equivalence remain separate. See provenance/WORKSTATION_REPAIR.md and each REPAIR_VALIDATION.json. Native recovery is PENDING, not established by these CPU checks. The user-relayed device report is not an independently inspected layer receipt; return the original failing build/layers/evaluation files too.

## Prepare and launch

Use the existing flood environment, normally `~/miniforge3/envs/flood/bin/python` with the system TensorRT PYTHONPATH. Confirm interpreter, Python3.8, TensorRT8.4.x and actual device state. No environment upgrades. Run from this repair directory:

```bash
export JETSON_PYTHON="$HOME/miniforge3/envs/flood/bin/python"
export PYTHONPATH="/usr/lib/python3.8/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$JETSON_PYTHON" -c 'import sys, torch, tensorrt; print(sys.executable, sys.version); print(torch.__version__, tensorrt.__version__)'
sha256sum -c SHA256SUMS
```

Set EXISTING_HANDOFF to the already prepared original handoff directory containing `assets/`. No new imagery needs transferring. Link the exact 353 arrays without moving/overwriting the original files; that directory must remain in place:

```bash
"$JETSON_PYTHON" prepare_inputs.py --bundle-root "$EXISTING_HANDOFF" --link
"$JETSON_PYTHON" prepare_inputs.py --check
"$JETSON_PYTHON" verify_repair.py
```

If the original handoff is not fully prepared, finish its frozen-hash input recovery first. Do not regenerate/recompress arrays or bypass mismatches. `--bundle-root` here admits each exact frozen file hash; the complete package name/date is not input identity. The fixed raw roots remain required. All353 prepared inputs, 836 WorldFloods windows and 2675 STURM512 chips are verified by preflight. Preserve the 97-event panel.

After confirming the old heavy job has drained, start one new immutable repair attempt:

```bash
ATTEMPT="micro_repair_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results_logs
setsid "$JETSON_PYTHON" runtime/run_remaining.py --attempt "$ATTEMPT" \
  --sturm-root "$HOME/data/STURM-Flood/Dataset" \
  --wf-root "$HOME/data/WorldFloodsv2" \
  </dev/null >"results_logs/$ATTEMPT.log" 2>&1 &
printf '%s\n' "$!" >"results_logs/$ATTEMPT.pid"
printf '%s\n' "$ATTEMPT" >results_logs/latest_attempt.txt
```

Replace only actual dataset-root locations as necessary. Verify controller PID, advancing results/$ATTEMPT/HEARTBEAT.json and the child log; shell success is not launch proof.

The finite job has 8 builds: four exact original FP32 micro parents and four aligned16 pure candidates, starting with 107 then 42/64/91. It covers all five full datasets (89 validation, 90 test, 15 Bolivia, 836 WF, 2675 STURM512), dry159 (author49 central264 primary; 28 ordinary / 21 stress; other views sensitivity), and the 200 fixed WF dry windows. It benchmarks batch-one512 kernel/wall p50/p95/p99, throughput/layer profiles, engine/device memory and steady power/energy. Do not describe159 scenes as independent expert ground truth.

Each FP32 baseline must first pass the existing 0.005 source-wiring check. Each pure engine must have both INT8 bindings and an inspector certificate containing no floating/unknown tensor formats. Unknown formats, float fallback, no INT8 evidence or wrong model identity fail the repair cell. These checks certify tensor formats, not every kernel's internal arithmetic; INT32 bias/accumulation and shape operations remain distinct. Never enable FP16 or relax the certificate to make a cell pass. Retain the engine and inspector even on certificate failure. A built but uncertified engine is a failed pure candidate, never a mixed substitute.

After certification, all 89 source images are evaluated and the three native source losses against the same-seed native FP32 parent must each be <=0.02; zero true-positive water fails explicitly. A failed source gate blocks external evaluation for that engine. Collect the failure and continue unrelated seeds. Do not tune from external scores, retrain, recalibrate, change IO scales, graphs, masks or acceptance thresholds. The controller performs no model selection.

## Control and return

From this directory, `touch results/$ATTEMPT/PAUSE` pauses after the current model completes its evaluation/benchmarks; remove PAUSE to continue. `touch results/$ATTEMPT/STOP` stops after the current model. Neither interrupts a child. To resume a controlled stop, first ensure no old or repair child is active, remove STOP/PAUSE, then repeat the exact launch with `--resume` and a fresh `.resume.log`; keep the same attempt and roots. Completed failures remain terminal. Never resume the original 48-slot attempt with these new models.

Collection runs automatically and can be repeated without model work:

```bash
"$JETSON_PYTHON" runtime/collect_native.py --attempt "$ATTEMPT"
```

Return the complete repair results/$ATTEMPT directory and controller/resume logs, including failed builds/inspectors, NATIVE_SOURCE_GATE.json, raw per-image counts, timing/power samples, coverage_matrix.csv, accuracy.csv, four-seed summaries, paired_event_intervals.csv (whole-event bootstrap 10000 / seed 1729, seed SD separate, Bolivia descriptive), dry_scene_summary.csv, wf_dry_per_image.csv, benchmark.csv and COLLECTION_RECEIPT.json. Retain all four seeds, including failures; missing seeds block a four-seed aggregate. The repair coverage receipt has 4 slots / 8 attempts and is explicitly an addendum; it never certifies the original 48 slots or B0. Return the original failed micro107 build/log/layers/source report so the workstation can confirm the exact device cause. Do not retry a failed scientific build/accuracy arm with changed flags or silently overwrite results.

The repair builder also fixes both public bindings to LINEAR NCHW and records their vectorized dimension, element components, dtype and format. The runner refuses packed bindings, and pure admission requires linear INT8 input/output. This prevents interpreting a padded CHW4/CHW32 host buffer as dense six-band/two-class arrays; it is a defensive IO guard, not evidence that packing caused the original failure. Internal INT8 layouts remain available. See NVIDIA's [TensorRT8.4 buffer-format guidance](https://developer.nvidia.com/docs/drive/drive-os/archives/6.0.4/tensorrt/developer-guide/index.html).
