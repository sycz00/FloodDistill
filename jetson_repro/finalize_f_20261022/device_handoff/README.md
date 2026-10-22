# Current Jetson handoff for research

Start with [JETSON_PROMPT.md](JETSON_PROMPT.md). This folder is the runnable Git handoff at `jetson_repro/finalize_f_20261022/device_handoff/` on `research`. It supersedes the older exploratory prompt and the parent directory's prepared-only QAT plan.

It contains all 51 declared ONNX entries and integer-I/O scales, full source gates/references, twelve float-export parity records, five frozen dataset inventories, the 16 exact existing tiny engines/build receipts/raw accuracy records, and the native runner/collector. The primary matrix has 48 cells: 44 supplied candidates and four unavailable B0 pure-integer cells. Thirty-five new engine attempts remain, plus dry159 extensions for the 16 old engines. Historical B0 failures and the corrected mixed QAT receipts are included under `workstation/`. Original checkpoints and the first two recipes remain in the transfer bundle; corrected fits, selected checkpoints, frozen code and diagnostics are preserved under `MajorRev/revision_package_20261021/10_b0_mixed_repair_20261022/`. Read [RUN_MANIFEST.json](RUN_MANIFEST.json), [MODEL_COVERAGE.csv](MODEL_COVERAGE.csv) and [B0_QAT_ATTEMPTS.csv](B0_QAT_ATTEMPTS.csv).

The earlier two B0 QAT recipes failed. Seeds 42/64/107 recovered with corrected CE QAT; seed 91 required a further 384 updates matching the frozen original student logits on the same 512 training images, after its bounded CE continuation failed. All selected models retain 56 QDQ Convs out of 57, with floating attention/classifier and no additional floating CNN exceptions. The corrected mixed candidates now pass all three source-accuracy limits for all four seeds on the actual optimized deployment graphs. See [provenance/B0_MIXED_REPAIR.json](provenance/B0_MIXED_REPAIR.json) for per-seed scores, QDQ Conv counts and any additional floating Conv exceptions. Attention and the final classifier remain floating. All 16 B0 builds now require explicit FP32 attention/normalization/classifier constraints and an inspector certificate; B0 FP16 therefore includes FP32 exceptions. The four old failed QAT graphs remain historical records rather than extra scheduled builds; failed PTQ controls remain scheduled diagnostics. Tiny42, micro107 and repaired B0 QAT candidates retain strict QDQ parity failures despite passing actual CPU accuracy and unrounded structure checks. Native execution is pending. A completed queue or a smaller model does not establish device accuracy, integer arithmetic, latency or energy.

The 159 supplementary dry tiles are now committed directly in [assets/dry159/](assets/dry159/), with their validity masks and [selection manifest](inventories/dry159_manifest.json). These are the exact six-band normalized 512x512 inputs consumed by the native runner; the original raw-scene paths in the manifest are provenance, not required Jetson paths. The primary dry set remains 49 author-retained scenes (28 ordinary, 21 stress); all 159 are retained for the declared sensitivity readout. These are author-reviewed/map-selected scenes, not independent expert ground truth.

From this folder, dry inputs can be verified immediately after checkout:

```bash
python3 prepare_inputs.py --check-dry
```

A dry-only extension for an existing certified float engine needs no S11 arrays or raw WF/STURM directories. On the target TensorRT8.4 device, use a fresh output folder, for example:

```bash
python3 runtime/evaluate_native.py \
  --engine prior_device_bundle/02_engines/tiny_R4_512protocol/fp32/R4_s42_fp32.engine \
  --out "results/dry_only_R4_s42_fp32_$(date -u +%Y%m%dT%H%M%SZ)" --only-dry
```

For integer-I/O engines also pass their exact `--io` JSON. Follow [JETSON_PROMPT.md](JETSON_PROMPT.md) for the complete multi-model run, environment checks and result collection.

The [user-relayed device report](provenance/JETSON_AGENT_STATUS_20261022.md) records an older handoff with missing dry data and zero new runs. The dry inputs were added in `5cd7c9e`, after the reported `cabfc13` files. Preserve the Jetson's local commits and deleted/relocated prompt. A separate verified handoff export may be used by setting `HANDOFF_ROOT` in the prompt, leaving the existing checkout untouched.

The remaining 194 prepared S11 validation/test/Bolivia inputs and workstation training history remain in the unified `MajorRev/revision_package_20261021/` transfer bundle, identified by [SOURCE_PACKAGE.json](SOURCE_PACKAGE.json). Existing source arrays from any dated package may be reused only if all 194 exact file hashes match the frozen manifest. `--source-root` accepts a directory containing `source_val/`, `source_test/`, `source_bolivia/`, directly or beneath `assets/`. It checks all candidates/destinations before writing, then verifies all 353 arrays. The committed 159 dry inputs must already pass. These checks preserve the controller's full preflight. Choose one import route:

```bash
# Exact existing source arrays; directory age/name is not proof of compatibility.
python3 prepare_inputs.py --source-root /actual/existing/source_root
# Optional: add --link if that verified directory will remain in place.
# Alternatively, import the original complete bundle:
python3 prepare_inputs.py --bundle-root /actual/path/to/8_jetson_export_20261022
# Alternatively: python3 prepare_inputs.py --archive /actual/path/to/Jetson_export_20261022.tar.gz
python3 prepare_inputs.py --check
```

The import still checks all 353 file hashes and preserves conflicting existing assets. `--link` reuses verified unpacked inputs without copying. Other source inputs and new results remain ignored by Git; `assets/dry159/*.npz` is tracked. Raw WorldFloods/STURM datasets must exist for the full run. The 200 water-free WorldFloods windows are derived from the complete WorldFloods evaluation and are separate from the 159 committed supplementary tiles.

The runner additionally verifies reused integer certificates, full result inventories and benchmark return codes/steady samples. New FP32 source wiring is checked before external evaluation. The corrected B0 graphs have new identities. The 48-slot matrix, tiny/micro graphs, prepared-data archive, source tolerances and historical observations are preserved. Use a new native attempt for this updated package; the runner rejects resuming results from the earlier package hash.

The complete repaired export `MajorRev/revision_package_20261021/Jetson_mixed_repair_20261022.tar.gz` includes the 353 arrays; after unpacking, run `prepare_inputs.py --check` directly. The `--archive` importer accepts only the original pinned transfer archive, not this repaired handoff archive.

For the reported device with an older dirty checkout, the separate `Jetson_input_recovery_20261022.tar.gz` export contains the current models/runtime and all 159 dry inputs, plus its Git export identity. Unpack it into a fresh directory and set `HANDOFF_ROOT` to its `device_handoff` directory. Reuse the 194 local source files through `--source-root` only after their frozen hashes match; no full archive is required for that route. The source-root importer was verified on the canonical workstation cache, not on the user-reported older Jetson files.
