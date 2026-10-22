# Current Jetson handoff for research

Start with [JETSON_PROMPT.md](JETSON_PROMPT.md). This folder is the runnable Git handoff at `jetson_repro/finalize_f_20261022/device_handoff/` on `research`. It supersedes the older exploratory prompt and the parent directory's prepared-only QAT plan.

It contains all 51 declared ONNX entries and integer-I/O scales, full source gates/references, twelve float-export parity records, five frozen dataset inventories, the 16 exact existing tiny engines/build receipts/raw accuracy records, and the native runner/collector. The primary matrix has 48 cells: 44 supplied candidates and four unavailable B0 pure-integer cells. Thirty-five new engine attempts remain, plus dry159 extensions for the 16 old engines. The eight B0 collection receipts and both recipe records are included under `workstation/`; fitting and original checkpoints remain in the transfer bundle. Read [RUN_MANIFEST.json](RUN_MANIFEST.json), [MODEL_COVERAGE.csv](MODEL_COVERAGE.csv) and [B0_QAT_ATTEMPTS.csv](B0_QAT_ATTEMPTS.csv).

B0 QAT was performed under two fixed recipes for all four current512 seeds; none met the0.02 source-accuracy rule. Its supplied quantized graphs are diagnostics. Tiny42 and micro107 QAT fallback candidates pass CPU source accuracy but retain strict QDQ parity failures. Native execution is pending. A completed queue or a smaller model does not establish device accuracy, integer arithmetic, latency or energy.

Large datasets and workstation training history remain in the unified `MajorRev/revision_package_20261021/` transfer bundle, identified by [SOURCE_PACKAGE.json](SOURCE_PACKAGE.json). To import the exact 194 prepared S11 arrays and 159 dry arrays:

```bash
python3 prepare_inputs.py --bundle-root /actual/path/to/8_jetson_export_20261022
# Alternatively: python3 prepare_inputs.py --archive /actual/path/to/Jetson_export_20261022.tar.gz
python3 prepare_inputs.py --check
```

The import checks all 353 file hashes, preserves conflicting existing assets and never extracts unrelated archive members. `--link` reuses a verified unpacked bundle without copying. Assets and new results are ignored by Git. Raw WorldFloods/STURM datasets must already exist on the target; the runner verifies every transformed image/mask hash before admitting model work. Follow the prompt to select the existing environment, launch detached, verify heartbeat, pause/stop/resume, and return raw results.

The runner additionally verifies reused integer certificates, full result inventories and benchmark return codes/steady samples. New FP32 source wiring is checked before external evaluation. These are execution/collection safeguards; models, scientific tolerances, datasets and previously observed results are unchanged.
