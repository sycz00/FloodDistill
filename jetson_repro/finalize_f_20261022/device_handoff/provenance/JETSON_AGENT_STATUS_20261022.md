# Jetson agent status relayed by the user — 22 October 2026

This is a reported device snapshot, not a live check performed on this workstation. No new matrix runs started. The raw WF/STURM hash-check outcome has not been supplied; its reported running state must be rechecked on Jetson before reuse or a duplicate launch.

I could not start the 48-slot run. The 159 dry arrays (`assets/dry159`) are not on the Jetson, and the controller's preflight refuses to start without all 353 prepared arrays. Status is **`INPUTS_MISSING`**, and no runs have started.

**What's missing:** none of the three accepted sources is on the device. I searched the whole NVMe disk, and no USB drive or other mount is attached.

- the unpacked `8_jetson_export_20261022/` directory (Route A)
- `Jetson_export_20261022.tar.gz` with SHA256 `37acef12…5fca5c` (Route B)
- the complete repaired archive `Jetson_mixed_repair_20261022.tar.gz` (unpack, then `prepare_inputs.py --check`)

The 194 S11/Bolivia arrays are already here as exact copies from the 14 October package. The prompt forbids substitute bundles, so I'm not using them, and nothing was written into `device_handoff/`.

**Checked and fine:**

- **Checkout:** branch `research`. HEAD is your merge `175c98c`, 4 commits ahead of origin and not pushed. The handoff folder matches `cabfc13` exactly and all 401 `SHA256SUMS` entries pass. Leftover local changes are kept as they were:
  - `prompts/jetson_prompt.md` is deleted; it is the same text as your `jetson_prompts/jetson_prompt.md`.
  - two untracked earlier experiment folders.
- **Environment:** `~/miniforge3/envs/flood/bin/python` (Python 3.8.20) plus `PYTHONPATH=/usr/lib/python3.8/dist-packages`. TensorRT 8.4.1.5, torch 1.13 (nv22.10), PyCUDA, ONNX 1.17, NumPy 1.24.4, rasterio and albumentations are all present, as is `trtexec`.
- **Reused engines:** all 16 match their hashes and build receipts, the pure ones have their integer certificates, and all 16 load on this device.
- **Device:** Xavier NX on L4T R35.1, idle with no other heavy jobs. Power mode is `MODE_15W_2CORE`, GPU pinned at 1109.25 MHz, CPU at 1907.2 MHz, about 4 GiB memory free. None of this was changed.

**Still running:** a background check that streams all 836 WorldFloods and 2,675 STURM images through the handoff's own hash checks. It is read-only and writes only to `/tmp/claude-1003/-home-schmalstieg/d59c0ba7-3141-4a19-b61c-f1db0753dd0c/scratchpad/precheck_raw.log`. It will show whether the raw datasets match before the bundle arrives. I'll report the result when it finishes.


## Workstation clarification and recovery

The reported handoff matches `cabfc13`, which predates the dry-data commit `5cd7c9e23abb4d3fdc32948cd01ec21ea41631a5`. That later commit contains all 159 dry NPZ inputs. It was committed locally; no push was performed by this agent. Deploy the latest provided research handoff and verify its actual file hashes; an ahead-of-origin count or a merge HEAD alone does not establish which files arrived. Do not reset, clean or overwrite the Jetson's four local commits, deleted prompt or untracked folders. Read `device_handoff/JETSON_PROMPT.md` if the repository-level prompt is locally deleted or relocated.

The original archive remains an accepted transport, but it is not mandatory if all frozen inputs are already available. The 194 existing source arrays may be reused **only if every file matches the current `ASSET_SHA256SUMS`**. Their package date or a claim of exact copying is not sufficient evidence. `prepare_inputs.py --source-root <existing-root>` validates all 194 sources and existing destinations before importing them, then verifies all 353 arrays. It requires the 159 committed dry inputs to pass first. `--link` is optional when the original verified source directory will remain in place. A mismatch remains `INPUTS_MISSING`/identity failure; do not regenerate, recompress or silently replace inputs.

The controller's full 353-input preflight remains mandatory. The 16 engine deserialization checks and device/environment readings above are user-relayed observations, not new model accuracy/benchmark evidence. Preserve the running raw check and its log; collect its completed result without starting another full read while the existing check is active. Do not change the reported clocks, power mode or environment.
