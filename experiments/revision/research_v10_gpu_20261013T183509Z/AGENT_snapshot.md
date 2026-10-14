# Research experiment operating instructions

Read this file explicitly before executing `prompt1.md`. Also read applicable existing `AGENTS.md` files. Do not assume this singular filename will be discovered automatically, rename it, or overwrite existing repository instructions. User instructions and established repository requirements take precedence.

## Mission

Prototype the JSTARS major-revision experiments for **Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation** in the existing project on the `research` branch. Reuse the project. Reestablish comparable local baselines on the RTX 4090, then add controlled MobileNetV3-Small and LightFormer-ResNet18 OFA experiments, audit evaluation provenance, evaluate independently verified dry scenes, and compare learned models with fixed spectral rules under the existing robustness protocol.

The requested deliverable is working experiments and an unattended queue, not prose documentation. Do not rewrite the paper, draft responses, expand READMEs, create a new framework, or perform unrelated cleanup. Minimal configuration, manifests, logs, metrics, diagnostic figures and checkpoint artifacts are necessary experimental bookkeeping and are allowed.

## Start and resume

- Confirm repository root, branch and dirty state. Preserve user changes and historical results. Never silently reset, clean, stash, force-checkout, overwrite checkpoints, push or merge.
- Read existing launchers, configs, data manifests, model/loss/evaluation code, environment metadata, active process state and run ledger before changing or launching anything.
- Reconcile completed and running runs by configuration/data/code/environment identity, not by filenames alone. Reuse compatible outputs; never launch duplicate jobs after a restart.
- Work from actual local paths and installed facilities. If the project or GPU is inaccessible, state the blocker; do not fabricate execution or substitute an unrelated checkout.
- Continue routine authorized work without repeatedly asking permission. Ask only about unresolved material ambiguity or missing access/input, and keep independent tasks moving.

## Keep implementation small

- Prefer existing entry points, registries, feature adapters, losses, augmentation, plotting, checkpoint and tracking code. Add narrow adapters/configs where needed.
- For LightFormer, pin the published implementation and import only necessary model components. Avoid a new training stack or global environment upgrades.
- Test concrete risks: feature/output shapes, six-band handling, frozen teacher, finite losses, intended gradients, removable OFA heads, checkpoint reload, metric validity and queue recovery. Reuse existing tests or short smoke checks.
- Do not write broad tests that merely mirror configuration, build speculative abstractions, or refactor working code for style.
- Keep training sources immutable while jobs use them. Preserve the code/config snapshot or checksum for every run; do not modify running jobs' inputs.

## Scientific contract

- Freeze splits, subset IDs, band order, normalization, geometry, label/ignore mapping, augmentation, loss definitions, optimizer schedule, checkpoint rule and seed set before new comparisons.
- Recover the exact executable original recipe. Treat numbers in `prompt1.md` as cross-checks. Resolve conflicting code/paper/config evidence explicitly; do not silently change semantics to reproduce a score.
- Original results came from another machine/software environment. Evaluate old checkpoints separately from retraining. Use reestablished same-environment controls for new comparisons and preserve original results as references.
- Keep the frozen Prithvi teacher and its checkpoint identity fixed. Teacher inference remains in eval/no-gradient mode. Preserve live soft targets on augmented inputs if the original method uses them.
- S11 hard-only training uses human targets. S11 KD/OFA uses human hard targets plus teacher soft outputs. GDACS hard-only uses teacher hard masks; GDACS KD/OFA adds teacher soft outputs. Do not collapse these distinct conditions into ambiguous labels.
- “CE” may denote a focal hard-label term in this project. Check the actual loss. Any correction to a suspected OFA formula mismatch is a separate variant, never a silent baseline change.
- S11 validation may legitimately serve checkpoint selection and calibration. Do not use STURM, WorldFloods or S11 test outcomes to tune the new experiments.
- The user says STURM and WorldFloods were untouched. Audit evidence with an open mind: fixed evaluation is not adaptive tuning; absence of a code path is not proof of historical non-use. Label facts as verified, user-reported, inferred or unknown.
- Do not change evaluation masks, resolution or normalization to improve rankings. Preserve the main 512-pixel STURM protocol and distinguish any native-resolution diagnostic.
- Compare new students under the same data/objective/budget/selection recipe. Record pretraining and input adaptation. Count adapted model and auxiliary parameters separately.
- OFA need not improve accuracy. A valid port with no gain is a result. An incompatible or missing port is a feasibility blocker, not an accuracy result.
- Do not automatically add models, budgets, seeds, threshold sweeps or full deployment ports beyond the bounded matrix in `prompt1.md`.

## Negative scenes and robustness

- Independently verified absence of water defines a dry reference. Teacher-empty or MNDWI-empty predictions alone do not. Preserve cloud/nodata/uncertain-region masks and geographic/event independence.
- Measure dry-scene behavior before changing training. Do not automatically add negative scenes to training or alter the 5% filter.
- On empty reference masks, prioritize false-water pixel fraction and prespecified scene alarm criteria. Do not hide undefined water IoU behind all-background accuracy.
- Apply identical existing perturbations to every comparator in reflectance space before fixed normalization. Preserve original severities, interpolation, clipping behavior and masks.
- Keep MNDWI threshold fixed at zero for the primary comparison. Common positive scalar gain should cancel from the ratio; investigate numerical exceptions instead of suppressing this rule advantage.
- Report clean and corrupted accuracy, absolute and relative degradation, all predeclared severities and failed/null results. No cherry-picking scenes or searching for a perturbation that makes the learned model win.
- Distinguish teacher results from student results, float checkpoints from actual quantized execution, and synthetic haze from independently observed atmospheric conditions.
- Use scene/event units for uncertainty and paired analyses where supported. Four training seeds are not thousands of independent geographic replications. Nonsignificance is not proof of equivalence.

## RTX 4090 resource and queue rules

- Reuse an existing scheduler. If absent, add only a small persistent queue wrapper and machine-readable manifest around established commands.
- Allow one GPU-heavy job at a time across all agents, including substantial evaluation and teacher inference. Use a shared process-safe lock and inspect foreign compute processes before launch.
- Never kill, suspend or reconfigure other work. Retain measured GPU-memory headroom, conservative CPU/BLAS threads and DataLoader workers, and avoid overlapping heavy disk operations.
- Do not silently change scientific batch size, precision, optimizer steps or normalization to fit a run. Gradient accumulation is not automatically equivalent when BatchNorm or update schedules change. Diagnose OOM and block unresolved configurations.
- Use existing user-level resource controls where available. Do not change clocks, global power limits, display settings, sleep policy or system services. A single training job still consumes resources; never promise zero slowdown.
- Run the worker outside the editor's terminal lifecycle using an available existing supervisor, user-level service or detached session. Detect the OS; do not assume systemd/tmux exists.
- Test a tiny two-job sentinel for parent-process independence. Record what was actually tested. Editor closure, logout, reboot and host sleep are different guarantees. No job runs while the host is asleep or powered off.
- Queue finite predeclared work with dependencies: protocol/evaluator checks, local baselines, architecture checks/controls, dry-scene evaluation and robustness. CPU audits may overlap conservatively. Use measured ETAs, not invented durations.
- Persist exact commands, frozen configurations, fingerprints, outputs, logs, owned process identities, timestamps, attempts and exit codes. Update state atomically. Reconcile on restart before resuming or launching a duplicate.
- Provide status, pause-new-launches, resume and graceful worker-stop commands. Pause does not kill the active job. Explicit job cancellation is a separate action with existing checkpoint-safe handling.
- Validate expected outputs before marking a run complete. A failed gate blocks descendants; unrelated safe tasks may continue. Allow at most one recorded retry of a genuinely transient failure, with no silent recipe changes.
- The next valid queued run must launch without a live Codex conversation. Stop or block on unresolved scientific decisions rather than guessing. Stop after the bounded workload completes.

## Result integrity and handoff

Keep historical checkpoints, manifests and metrics intact. Store new work under distinct existing-style run IDs. Never mix different recipes/environments into a supposedly matched aggregate. Do not discard an unfavorable seed or relabel a failed run as a missing datapoint without recording it.

Use the existing tracker or one compact ledger for completed/running/pending/blocked state, provenance, errors, decisions and unresolved references. This operational record is permitted; narrative documentation is not requested.

At handoff, give a concise factual update: what was reused/changed, what has completed, what is running or queued, blockers, queue persistence mechanism, actual control commands and measured remaining-time estimate. Never claim a queue has started, survived editor closure, reproduced a result or finished an experiment without evidence.
