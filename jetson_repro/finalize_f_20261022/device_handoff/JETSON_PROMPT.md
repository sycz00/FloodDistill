# 23-October received-failure follow-up

For the current Jetson window, use [JETSON_WINDOW_PROMPT_20261023.md](JETSON_WINDOW_PROMPT_20261023.md). This version integrates the four aligned16 micro pure candidates, the audited B0 mixed PREFER build policy, and STURM numerical preprocessing tolerance. Preserve old failures and obtain a fresh handoff directory. Use runtime/chain_evaluations.py to wait for existing evaluations and reuse compatible completed work. For the compact follow-up archive, first link all 353 prepared arrays with prepare_inputs.py --bundle-root OLD_HANDOFF --link, then check SHA256SUMS; images are intentionally omitted from that compact transport.

# Jetson task: finish the current 512-pixel model matrix

Use the existing `research` checkout and the latest provided `device_handoff`, either under `jetson_repro/finalize_f_20261022/` or in a separate verified export. Read this prompt and that folder's `README.md`, `SOURCE_PACKAGE.json`, `RUN_MANIFEST.json` and `DATASETS.json`. Execute the remaining work and collect results; do not stop at a plan.

This prompt includes the corrected mixed B0 handoff from `10_b0_mixed_repair_20261022` and supersedes the earlier handoff in commit `cf315e5`, which contained failed B0 QAT candidates. The 159 supplementary dry arrays are now Git-tracked at `assets/dry159/` and need no archive import. Use this updated package with a new attempt name; never resume an older attempt against these changed package hashes. The workstation has now prepared all supplied ONNX candidates. **Do not repeat PTQ calibration, QAT training, checkpoint selection or architecture searches on Jetson.** Use the exact committed graphs. Preserve existing work, failed attempts and the closed 97-event panel. Do not install or upgrade software, change clocks or power mode, use DLA, or commit/push results.

Read [the user-relayed Jetson status](provenance/JETSON_AGENT_STATUS_20261022.md). It describes an older `cabfc13` handoff on device HEAD `175c98c`, before the dry-data commit `5cd7c9e` arrived. That report records `INPUTS_MISSING` and zero new runs. Preserve its four local commits, deleted/relocated repository prompt and untracked folders. Use this folder's `JETSON_PROMPT.md` as the execution copy. The result of the reported background raw-data check has not been supplied; recheck its process/log before treating it as completed or starting another full scan.

## 1. Establish the checkout and inputs

Work from the existing FloodDistill repository. Confirm that its branch is `research`, record its HEAD and dirty state, and preserve unrelated changes. Do not reset, clean, force-checkout or pull over local changes. Use the updated committed handoff or a separately unpacked, hash-verified workstation export beside the existing checkout. For an export, set `HANDOFF_ROOT` to its `device_handoff` directory and record the supplied `GIT_EXPORT_IDENTITY.json` from its parent. Keep the existing dirty checkout untouched; do not merge, reset or restore its intentionally deleted prompt just to obtain these files. If neither updated handoff is available, report that exact missing prerequisite.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
git branch --show-current
git rev-parse HEAD
git status --short
HANDOFF_ROOT="${HANDOFF_ROOT:-$REPO_ROOT/jetson_repro/finalize_f_20261022/device_handoff}"
cd "$HANDOFF_ROOT"
sha256sum -c SHA256SUMS
```

Use the already established Python 3.8 environment with PyTorch, TensorRT 8.4.x, PyCUDA, ONNX, NumPy, rasterio, albumentations and the existing dataset-loader dependencies. The received native session used JetPack 5.0.2 / TensorRT 8.4.1.5 on Xavier NX. Activate that existing environment and retain any required system TensorRT `PYTHONPATH`; do not create another environment. Resolve and record the actual interpreter:

```bash
export JETSON_PYTHON="$(command -v python3)"
"$JETSON_PYTHON" -c 'import sys, torch, tensorrt; print(sys.executable); print(sys.version); print(torch.__version__, tensorrt.__version__)'
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
```

The Git handoff contains all models, integer-I/O scales, inventories, reference scores, source gates, the existing 16 tiny engines/native accuracy records, and the 159 supplementary dry tiles with boolean validity masks at `assets/dry159/*.npz`. Verify the dry inputs immediately after checkout:

```bash
"$JETSON_PYTHON" prepare_inputs.py --check-dry
```

This check needs no archive, source arrays or raw external datasets. The committed arrays are the exact prepared six-band 512x512 model inputs. Original raw-scene absolute paths in `inventories/dry159_manifest.json` are provenance only. Keep the 49 author-retained primary scenes (28 ordinary, 21 stress), their masks and all 159 sensitivity tiles exactly as declared. Do not describe all 159 as independently verified dry ground truth.

For a dry-only extension of an existing certified engine, `runtime/evaluate_native.py --only-dry` needs `--engine`, a fresh `--out`, and the exact `--io` for integer-I/O engines; it does not require WF/STURM roots. The folder README gives a complete runnable example. The full controller below still evaluates every declared model/dataset and benchmarks it.

For the full matrix, the other 194 prepared S11 validation/test/Bolivia arrays remain external. The reported 14-October copies may be reused **only when every file matches the current frozen `ASSET_SHA256SUMS`**. Package name/date alone neither qualifies nor disqualifies identical bytes. No regeneration, recompression, label changes or hash bypass is allowed.

Prefer Route C when the 194 existing files are available. `SOURCE_INPUT_ROOT` contains either `assets/source_val`, `assets/source_test`, `assets/source_bolivia`, or those three `source_*` directories directly. If the old layout differs, create a separate staging directory of symlinks using these exact names, preserving the originals. The importer verifies all 194 candidates and existing destinations before writing, requires the committed dry159 check to pass, and then verifies all 353 arrays. A missing/mismatching input remains blocked. Otherwise use the original complete bundle/archive. Run **one** route; matching committed dry files are reused without replacement:

```bash
# Route C: exact existing source arrays, including older copies only if hashes match.
"$JETSON_PYTHON" prepare_inputs.py --source-root "$SOURCE_INPUT_ROOT"
# Add --link only if this verified source directory will remain in place.

# Route A: an already unpacked transfer directory.
"$JETSON_PYTHON" prepare_inputs.py --bundle-root "$TRANSFER_BUNDLE_ROOT"

# Route B: the original transfer archive; its pinned SHA256 is checked first.
"$JETSON_PYTHON" prepare_inputs.py --archive "$TRANSFER_ARCHIVE"

# After the chosen route, verify all 353 arrays.
"$JETSON_PYTHON" prepare_inputs.py --check
```

If the complete repaired handoff archive already supplied all of `assets/`, run `prepare_inputs.py --check` and skip all import routes after it passes. The `--archive` importer accepts the original pinned archive only; unpack the complete repaired handoff normally. The 200 water-free WorldFloods windows are derived separately from full WF evaluation and still require the existing WorldFloods dataset.

`TRANSFER_BUNDLE_ROOT` means the directory containing `assets/`, not `assets/` itself. `--link` may be added to Route A to reuse verified arrays without copying; that source directory must then remain in place. The pinned transfer-archive SHA256 is `37acef126307b7bec899728b3d95e609cfa68d5514a9cd8d892deb40e95fca5c`. The original-archive importer accepts only that pinned archive; Route C admits individual source files solely by their frozen hashes. Do not recompress arrays, regenerate labels or bypass a hash mismatch. If no route supplies matching inputs, report `INPUTS_MISSING` with the missing paths or mismatching hashes; finish the checkout/environment checks and do not claim the runs started.

Locate the complete existing raw datasets. Defaults are `$HOME/data/STURM-Flood/Dataset` and `$HOME/data/WorldFloodsv2`; pass their actual roots if different. No workstation absolute path is required. The controller checks every dataset before model work. STURM image hash equality is not required: use absolute tolerance 1e-4 in normalized inputs against a float64 bilinear reference, recording the maximum error and actual hash. Keep STURM masks, event IDs, image count, band order, geometry and normalization fixed. Other dataset checks stay as specified. The 97-event panel is outside this task.

## 2. Launch one durable finite run

Inspect current device work, memory, hardware identity and observed clock/power state. Wait if another heavy job is running. Do not kill, suspend or reconfigure another worker. First resolve the reported read-only raw-dataset check from its existing log/process; reuse a completed matching result as context, but the controller still performs its own mandatory preflight. Do not launch duplicate raw scans while the existing check is active, or claim its result from the earlier running report. Preserve the reported environment, clocks and power mode. Run one heavy job at a time. A package-wide lock prevents two controllers from this handoff running concurrently; it does not lock other projects.

From the handoff directory, start a new immutable attempt, replacing dataset-root arguments only if their actual locations differ:

```bash
ATTEMPT="native_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results_logs
setsid "$JETSON_PYTHON" runtime/run_remaining.py --attempt "$ATTEMPT" \
  --sturm-root "$HOME/data/STURM-Flood/Dataset" \
  --wf-root "$HOME/data/WorldFloodsv2" \
  </dev/null >"results_logs/$ATTEMPT.log" 2>&1 &
CONTROLLER_PID=$!
printf '%s\n' "$CONTROLLER_PID" >"results_logs/$ATTEMPT.pid"
printf '%s\n' "$ATTEMPT" >results_logs/latest_attempt.txt
```

Verify that the controller PID is alive, `results/<attempt>/HEARTBEAT.json` advances and its recorded child log grows. Do not infer launch success from the shell exit code. Initial preflight checks all 353 prepared arrays, the full external inventories, target TensorRT compatibility, and deserialization/precision receipts of the 16 reused engines. A failed preflight admits no model work; retain its receipt and log.

Controls, from the same folder, using the recorded attempt name:

```bash
# Status and current child.
cat "results/$ATTEMPT/HEARTBEAT.json"
cat "results/$ATTEMPT/STATUS.json"

# Pause after the current model finishes its evaluation/benchmarks.
touch "results/$ATTEMPT/PAUSE"
# Continue.
rm -f "results/$ATTEMPT/PAUSE"
# Gracefully stop after the current model.
touch "results/$ATTEMPT/STOP"
```

To resume a controlled stop, remove `STOP` and any `PAUSE`, then rerun the original launch command with the same `--attempt` and add `--resume`. Preserve the earlier controller log; write the resumed controller output to a new `.resume.log`. Resume checks package, dataset-root and engine identities and reuses complete atomic datasets/builds. It refuses a previously owned child that is still running. Completed build/evaluation/benchmark failures remain terminal; do not silently retry a scientific failure or change its graph. Report such failures and retain the remaining successful work.

## 3. Execute the declared matrix and preserve distinctions

The primary matrix has **48 slots**: R4 tiny, R4_micro and current-protocol EfficientViT-B0 × seeds 42, 64, 91, 107 × FP32, FP16, mixed INT8/float and pure integer. There are 44 supplied primary candidates; **four B0 pure-integer slots have no eligible candidate**. The manifest contains 51 engine entries, comprising 16 existing tiny engines and **35 new build attempts**, including diagnostic controls.

- Reuse the exact 16 committed tiny engines and their previous complete five-dataset measurements/benchmarks. Extend each with dry159. Verify engine hashes and the original build/precision receipts.
- Build and evaluate all 35 new entries from their supplied ONNX and I/O specifications. Never substitute FP32 after a requested integer build fails.
- Tiny mixed retains PTQ for all four seeds. Tiny pure retains PTQ for 64/91/107 and measures QAT42 as its fallback; the old PTQ42 pure failure remains a diagnostic. Tiny QAT42 mixed is an additional diagnostic.
- Micro mixed uses PTQ for 42/64/91 and QAT107 fallback; micro pure uses the four new aligned16 zero-embedded graphs under distinct IDs, retaining PTQ/QAT weights and all prior failures. Failed micro107 mixed PTQ remains a diagnostic baseline.
- B0 mixed uses the four newly repaired QAT exports, with fresh IDs `B0_s<seed>_qat_mixed_repaired`. All four pass the 0.02 source-accuracy limit on the actual optimized deployment graph. Their strict QDQ logit-equivalence checks still fail; all 89 unrounded structure checks pass. Build these candidates and measure actual native accuracy. The four failed B0 PTQ graphs remain diagnostic controls. The four old failed QAT graphs and their source receipts remain in `retired_diagnostics` as historical records; do not schedule them as additional native entries. All completed QAT recipes, the bounded continuation and failed precision diagnostics are retained. Do not train another recipe here.

The current parents use the real 512-pixel original-selected 2,500-image LOGIT_KD cohort. They are not the untrained broader GDACS-SW512-v3 arm, nor the distinct update-scheduled data-policy comparison controls. Genuine FP32/FP16 graphs retain their own original parent weights. Do not create float baselines by removing quantizers from a QAT graph.

B0 uses explicitly scored cubic-to-linear Resize variants for TensorRT 8.4; retain that graph identity. **B0 is a mixed model: the final two-class classifier and attention remain floating.** The repaired QAT graphs all retain 56 Conv input/weight QDQ sites and one unquantized classifier. There are no extra floating CNN exceptions in the selected graphs. Seeds 42/64/107 recovered with corrected CE QAT; seed 91 additionally required 384 updates matching the frozen genuine parent logits on the same 512 training images. No new foundation-teacher inference or external selection was used. Per-seed updates and scores are listed in `provenance/B0_MIXED_REPAIR.json`. These are QAT-trained weights with a declared mixed-precision placement, not pure integer B0. Their channel rescaling preserves the parent float function within the frozen full-source tolerance before fitting; the native Hardswish architecture was retained. Do not add classifier/output QDQ or quantize attention.

For all 16 B0 entries, the builder must apply the graph-specific `FP32_ISLANDS.json`: constrain all 8 attention MatMuls, their downstream normalization through four Div nodes, and the classifier to FP32 with `PREFER_PRECISION_CONSTRAINTS` in mixed builds and `OBEY_PRECISION_CONSTRAINTS` in float builds; disable TF32. The mandatory post-build inspector must still certify every required FP32 output; PREFER alone is never admission. Other eligible layers may use INT8/FP16 in the mixed arm. B0 FP16 therefore has explicit FP32 exceptions. A missing/unmatched constraint, FP16 normalizer, wrong inspector format or stale policy hash fails that build; never bypass the protection to obtain an engine. Preserve `fp32_constraints`, `fp32_island_audit`, the policy SHA256 and raw layer inspector. Verify actual INT8 placements separately; zero INT8 compute evidence cannot support an INT8 deployment claim. The inspector checks requested compute constraints and tensor formats, not every kernel internal operation. The controller runs a target-version synthetic attention capability probe. Preserve its full output. A synthetic success does not establish a complete all-integer B0 graph: leave all four pure B0 slots unavailable. Do not label mixed attention full INT8.

For supplied pure R4/micro graphs, require the exact integer-I/O scales and an actual inspector certificate with no floating or unknown tensor formats. A missing certificate fails that cell. It certifies tensor formats, not every kernel's internal arithmetic. Keep INT32 bias/accumulation and shape operations distinct. Do not enable FP16 to make a pure build succeed.

## 4. Evaluate every image and benchmark every available cell

Every new engine must evaluate the complete frozen surface-water protocol:

| Dataset | Images/windows/chips | Events |
|---|---:|---:|
| S11 validation | 89 | 10 |
| S11 test | 90 | 10 |
| Bolivia | 15 | 1 |
| WorldFloods test | 836 | 11 |
| STURM512 | 2,675 | 29 |

Input images, masks, event identities, valid/water totals and all names must match the inventories. Do not change normalization, channel order, interpolation, label mapping, split or mask to improve a result. New FP32 engines must match the corresponding CPU deployment-graph reference on full source validation within 0.005 pooled mIoU before external evaluation; a failure is retained and blocks that engine's external readout.

Evaluate all 159 supplementary dry tiles. Report author-retained 49 central-264 results as primary (28 ordinary, 21 stress), author49 full512 sensitivity and map159 central264 sensitivity. Preserve ordinary/stress groups, uncertain/zero-support masks and undefined denominators. These are author-reviewed scenes, not independent expert ratings. Also derive the fixed 200 water-free WorldFloods windows from nine events from the full WorldFloods masks. Do not use the erroneous historical STURM dry-window list. The scene alarm is false-water fraction strictly greater than 1%.

Report raw per-image confusion counts, summed per-event counts, equal-event/pooled water IoU and mIoU, precision and recall. Keep all four seeds, including negative results; a missing seed prevents a four-seed aggregate. Use paired whole-event bootstrap intervals with 10,000 draws and seed 1729, conditional on the four fitted seeds. Report seed SD separately. Bolivia is one descriptive event with no event-bootstrap interval. Undefined water scores stay undefined.

The user's 22-October accuracy limit is an absolute drop of at most **0.02 on each** source-validation pooled mIoU, equal-event mIoU and pooled water IoU against the same-seed genuine FP32 parent. Preserve the original .01/.01/.02 verdict separately; do not call the amended rule preregistered. No external result selects a model. Tiny42/micro107 and all four repaired B0 QAT candidates passed CPU deployment accuracy but failed strict QDQ logit parity at atol=rtol=.001. For the fallback models, CPU pooled mIoU/water IoU losses in percentage points are tiny42 mixed 0.394/0.667, tiny42 pure 0.434/0.747, micro107 mixed 0.825/1.440 and micro107 pure 0.923/1.622; equal-event mIoU losses are also below 2 points. These source results do not replace native checks. See `provenance/B0_MIXED_REPAIR.json` for the B0 per-seed results. Keep those failures explicit even if actual native accuracy passes. No accuracy check proves strict export equivalence.

For batch-one 512-pixel inference, retain GPU kernel p50/p95/p99, host wall p50/p95/p99, trtexec throughput and layer profiles, engine size, device/activation memory and steady power/energy with raw samples. Wall timing includes host INT8 encoding and both transfers; it excludes disk, normalization and argmax. Output remains two-channel INT8 logits for integer-I/O models, not a uint8 mask. Optional mask-transfer and alternate-upsampling experiments are outside this finite task.

Use the steady inference power window; startup-diluted averages remain secondary. Retain rail identity uncertainty and read labels if permitted. Do not infer matched hardware conditions from the old bundle's inaccurate clock summary. Record actual states, disclose differences between sessions and avoid unsupported cross-architecture speed/energy claims. Failed trtexec commands, missing steady samples or inconsistent energy receipts must remain failed benchmarks.

## 5. Collect, verify and return

The controller collects automatically after all declared attempts resolve. Collection can be rerun without model execution:

```bash
"$JETSON_PYTHON" runtime/collect_native.py --attempt "$ATTEMPT"
```

Return the entire `results/<attempt>/` directory and all controller/resume logs, including preflight, environment, probe, failed builds, layer inspectors, raw evaluations, timing/power samples and these tables:

- `accuracy.csv` and `accuracy_four_seed_summary.csv`
- `benchmark.csv` and `benchmark_four_seed_summary.csv`
- `dry_scene_summary.csv` and `wf_dry_per_image.csv`
- `coverage_matrix.csv`, `paired_event_intervals.csv` and `wiring_check.csv`
- `COLLECTION_RECEIPT.json` and `ATTEMPTS_RESOLVED.json`

Check all 48 primary slots and all 51 declared attempts. Enumerate every failed, missing or unavailable model/seed/precision/dataset/benchmark. `ALL_SUPPORTED_CELLS_OBSERVED` describes measured coverage of the 44 supplied primary cells, not accuracy/export qualification. Report observation, accuracy, strict export parity and integer-format certification separately. Report whether each repaired B0 candidate passes native source accuracy and its required FP32 island audit. Keep its strict export-equivalence failure visible even if native accuracy passes. Four unavailable pure B0 slots remain; do not claim 48 successful cells or native recovery before measurement.

Archive the returned results/logs outside the input folder or in ignored `results_logs/`, add a SHA256, and provide its path plus a short completed/failed/unavailable summary. Do not push, edit immutable manifests or overwrite old results.
