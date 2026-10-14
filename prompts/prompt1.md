# Prototype the flood-distillation revision experiments

Work in the existing project folder on the `research` Git branch. Read the companion `AGENT.md` explicitly, as well as applicable existing `AGENTS.md` instructions, before doing anything. This is an instruction to inspect, implement narrowly scoped changes, run experiments, and establish an unattended local experiment queue on the project machine. Do not stop after proposing a plan.

## Scope and working style

Understand the repository and its actual status first. Reuse its datasets, manifests, training entry points, model factory, losses, evaluation adapters, plotting utilities, checkpoint handling, experiment tracking, and launchers. Prefer a small adapter or configuration change over new infrastructure. Do not create a new training framework, dataset pipeline, scheduler platform, dashboard, or broad refactor.

This phase is experimental prototyping only. Do not edit the manuscript, write a reviewer response, expand READMEs, or produce narrative documentation. Necessary configs, queue state, logs, provenance records, checkpoints, CSV/JSON metrics, and diagnostic plots are allowed. Use existing locations and formats whenever possible. These two instruction files are not a request for more documentation.

Make routine reversible choices independently. Do not request approval for every run or wait for the user between successful queue jobs. Ask only for a real missing input, destructive operation, access requirement, or scientific ambiguity that cannot be resolved from the project. Continue independent unblocked tasks.

## Context and current scientific status

The project supports the paper **Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation**, submitted to the JSTARS special issue **Generalist–Specialist Model Synergy for Remote Sensing: Theories, Methods, and Applications**. The decision is major revision. Reviewers value the experimental coverage and actual deployment, but question novelty, incomplete architecture controls, evaluation provenance, negative-scene coverage, and practical advantages over a spectral rule.

The pipeline uses the public Prithvi-EO-2.0-300M-TL-Sen1Floods11 task checkpoint as a frozen teacher, generates supervision on GDACS Sentinel-2 imagery, curates a training pool, trains a compact EfficientViT-B0 student, and has a separate ReLU6/QAT/TensorRT deployment path. It uses both hard teacher masks and live soft teacher outputs, depending on the objective. OFA adds intermediate prediction heads during training; those heads are discarded for inference.

Known manuscript findings, to verify against local artifacts:

- Primary student approximately 0.7M parameters; alternative MobileNetV3-Small already exists in the architecture ablation. Resolve the exact implementations rather than relying on ambiguous model names.
- GDACS pool: 23,720 tiles, 5,987 surviving the reported filter, with a 2,500-image FPS-selected operating subset and smaller nested budgets. Reuse the exact stored IDs where available, not a regenerated approximation.
- Filtering matters, but FPS has no clearly resolved advantage over reasonable filtered alternatives at the operating budget.
- OFA has an intermediate-data benefit in the reported experiments, but is not consistently better than Logit-KD at 2,500 images. A null OFA gain is a valid outcome, not evidence that implementation must be adjusted until it wins.
- Fixed MNDWI is competitive on clean external benchmarks. The strongest documented learned-model advantage over it is under synthetic additive haze. Common multiplicative gain cancels from MNDWI mathematically.
- The labels chiefly describe surface water. A pre-event river prediction need not be a water-segmentation error, although it is not new inundation from the target event.
- The final Jetson engine is mixed INT8/FP16. The reported deployment measurements belong to Jetson Xavier NX, not the RTX 4090. Do not transplant those latency/power claims onto this machine.
- Original results came from a different machine and software environment. Reestablish comparable local baselines before interpreting new-model differences. Do not tune against the published test scores to force numerical agreement.
- The user states that STURM and WorldFloods were untouched external evaluation sets. The supplied manuscript nevertheless contains a sentence saying method choices were refined using reported benchmarks. Investigate this discrepancy factually; do not assume contamination or silently invent a clean history.

If `main.pdf`, `supplement.pdf`, and the review are in the project, read the relevant sections and reconcile them with code/configuration history. If not, use the context here and available experiment artifacts. Do not invent missing file paths or recreate whole datasets merely because an artifact is absent.

## 1. Establish the repository and experiment state

1. Confirm the repository root, current branch, worktree changes, relevant remotes, environment, OS, GPU, existing processes, available disk, and existing detached-run facilities. Do not expose secrets from configs or environment variables.
2. Work on the existing `research` branch. Switch only when safe. Preserve uncommitted work and checkpoints. If another checkout is needed, reuse an existing worktree or use a non-destructive checkout strategy; never reset, clean, overwrite, or stash user work silently. Do not push or merge as part of this task.
3. Read relevant instructions and inspect the training/evaluation/configuration code with targeted searches. Inventory completed, running, failed, and planned runs. Identify reusable checkpoints, dataset manifests, cached predictions, selector outputs, and previous architecture experiments.
4. Identify the smallest changes required for each task. Do not duplicate already implemented functionality or relaunch completed compatible runs.
5. Capture an environment fingerprint: Python, PyTorch, torchvision, timm, CUDA runtime, driver, cuDNN, GPU, relevant model packages, precision/TF32/determinism settings, Git revision and dirty-state fingerprint. Compare with available original metadata. Use the existing environment; add only necessary isolated dependencies and avoid global package upgrades.
6. Give a brief progress message identifying what exists, what is reusable, what needs code, and the initial queue order. Persist operational state using the existing tracker or one compact machine-readable ledger, not a new report.

## 2. Recover and freeze the recipe

Treat these as cross-checks against the original executable configs, not permission to overwrite them blindly:

| Item | Reported setting or constraint |
|---|---|
| Teacher | Public frozen Prithvi-EO-2.0-300M-TL-Sen1Floods11 checkpoint; no new teacher fine-tuning |
| Optical bands | B2, B3, B4, B8A, B11, B12 in the existing order |
| S11 split | 252 train / 89 validation / 90 test; 15 Bolivia chips separately reserved |
| Training geometry | Resize to 224 x 224; preserve existing interpolation, validity masks and transforms |
| Evaluation geometry | Main protocol 512 x 512; STURM originates at 128 x 128 and is upsampled under this protocol |
| Normalization | Recover reflectance correctly, then apply fixed S11-training statistics; no fitting to external data |
| Reported mean | [0.1413, 0.1380, 0.1235, 0.3090, 0.2045, 0.1191] |
| Reported standard deviation | [0.0741, 0.0737, 0.0869, 0.1180, 0.0977, 0.0766] |
| Optimizer | AdamW, learning rate 5e-5, weight decay 0.05, cosine schedule |
| Base training budget | 50 epochs, batch size 16; verify actual step, warmup and scheduler conventions |
| Augmentation | Existing dihedral transforms and per-channel multiplicative jitter in [0.80, 1.20] |
| Main loss | Existing hard-label focal term, with the reported hard/soft mixture 0.5 where KD is enabled |
| KD | Reported temperature 4.0; recover exact normalization and temperature scaling from code |
| OFA | Reported branch weight 0.3, modulation exponent 1.2; reuse existing branch loss and aggregation |
| Seeds | Four reported seeds; deployment lists 42, 64, 91, 107. Verify the relevant experiments used these |
| Checkpoint selection | Recover the exact S11-validation metric, direction, tie rule and selection frequency |

Preserve dataset-specific adapters. In particular, verify STURM's inverse normalization, B8 substitution for B8A, original validity handling, class mapping {1,2} positive and {3,4,5} ignored; verify the WorldFloods reference-water mask and seasonal-water treatment. Do not silently switch to published STURM's different evaluation protocol. Record any current-code discrepancy rather than patching until scores look right.

Recover actual GDACS hard masks, the frozen teacher identity, and the live-teacher path on correspondingly augmented training images. Do not replace live augmented soft targets with cached unaugmented logits. Do not count S11 Logit-KD/OFA as teacher-free: those configurations use human hard masks plus teacher soft outputs.

The main text labels some hard-only objectives “CE” despite using focal modulation. Preserve executable loss semantics and state their identity in run metadata. The printed claim that OFA Equation 6 becomes standard KD at exponent zero may not match the displayed equation: inspect the implementation, but do not silently change the original objective during reproduction. Any correction requires an explicitly separate variant and an identified reason.

Predeclare the comparison matrix and freeze all choices before inspecting new external results. STURM, WorldFloods and S11 test scores must not determine thresholds, exits, loss weights, epochs, checkpoints, or seed inclusion. It is normal to evaluate fixed ablations and discuss their outcomes; distinguish that from optimizing a recipe against external scores.

## 3. Reestablish local baselines first

Separate two checks:

**Evaluation reproduction:** If archived checkpoints exist, run the existing evaluator on the same data and protocol. Compare with archived per-scene outputs/metrics. This isolates evaluator/environment drift from retraining variation. Include the teacher and fixed MNDWI where outputs are missing. Reuse correctly fingerprinted predictions.

**Training reproduction:** Retrain the necessary EfficientViT-B0 controls on the current RTX 4090 using the frozen recipe. Start with the first original seed and then the remaining original seeds after correctness checks pass. The primary budget is N=2,500. Reproduce hard-only, Logit-KD, and OFA so new-model comparisons share a local baseline and environment.

Paper Table II reference means, for context only; all are clean micro mIoU, not target values to optimize toward:

| N=2,500 model | S11 | STURM | WorldFloods flood-only |
|---|---:|---:|---:|
| Teacher | 0.897 | 0.755 | 0.771 |
| EfficientViT hard-only | 0.873 | 0.754 | 0.754 |
| EfficientViT Logit-KD | 0.875 | 0.758 | 0.767 |
| EfficientViT OFA | 0.877 | 0.756 | 0.764 |
| Fixed MNDWI > 0 | 0.789 | 0.747 | 0.763 |

Investigate material discrepancies in this order: data/checkpoint identity, split and subset IDs, preprocessing/bands/masks, geometry, metric aggregation, loss/augmentation/selection, then runtime precision and software differences. Set gate tolerances from archived repeatability and numerical checks before new-model results; do not require bitwise agreement across different environments or invent an arbitrary universal IoU cutoff.

If a semantic mismatch remains unresolved, block dependent expensive runs. If the recipe is verified but retraining shifts modestly across environments, preserve both result sets, disclose the shift in the ledger, and use the new same-environment controls for comparisons. Do not erase the old results or tune external performance to match them.

## 4. Complete the architecture controls

Implement or reuse these students:

1. Existing EfficientViT-B0 reference.
2. Existing MobileNetV3-Small segmentation implementation, retaining its main decoder.
3. LightFormer with ResNet18, based on the published 2026 implementation rather than an unrelated model with the same name.

LightFormer source context:

- Published paper: https://doi.org/10.1016/j.jag.2026.105331
- Author repository: https://github.com/Chen-XiaoLv/LightFormer
- The repository reports roughly 11.41M parameters for its ResNet18 configuration; measure your adapted full segmentation model instead of copying this count. Its original optical/UAV experiments are not evidence of six-band Sentinel-2 performance. Use the published version; do not base the experiment on the withdrawn earlier preprint.

Inspect upstream code, dependency compatibility, licensing, pretrained-weight provenance and intermediate feature shapes. Pin the source revision and import only necessary components. Reuse the project's six-band input adaptation and output interface. Do not install a second training ecosystem or replace the existing evaluator to accommodate LightFormer.

For each added student, test whether the existing OFA mechanism can attach to encoder feature stages corresponding as closely as possible to strides 4, 8, 16 and 32. Use the same lightweight projection/upsampling and branch loss where compatible. Record actual channels, strides, branch count, parameter overhead and loss normalization. Do not assume EfficientViT's channels apply to another backbone, tap arbitrary decoder features without disclosure, or redesign a model merely to force four exits.

OFA branches must provide gradients to the intended backbone and must disappear from the main inference path. Verify output geometry, teacher freezing/eval mode, loss finiteness, gradients, checkpoint save/reload, and identical main-head inference before/after removing auxiliary heads. Use existing smoke tests or tiny training/validation samples; do not build a large generic test suite.

The default full-budget comparison is:

| Student | Hard-only | Logit-KD | OFA |
|---|---|---|---|
| EfficientViT-B0 | Reestablish locally | Reestablish locally | Reestablish locally |
| MobileNetV3-Small | Reuse a matching current-environment run or run | Reuse or run | Add missing exits and run |
| LightFormer-ResNet18 | Reuse if genuinely available or run | Reuse or run | Assess exits, implement narrowly, and run if valid |

Use N=2,500, identical selected IDs, original seed set, data geometry, augmentation, optimizer recipe and validation selection. This is at most 9 configurations x 4 seeds = 36 full training runs before deduplication, not a request for an open-ended search. Complete one seed per configuration first for functional checks, then expand all valid predeclared configurations to four seeds regardless of whether they beat OFA or MNDWI. Historical runs from a different environment are reference results, not interchangeable local replicates.

Keep pretraining provenance and input-weight adaptation explicit. A common fixed recipe is not proof that every architecture has reached its own optimum. No per-architecture test-guided tuning. Start with N=2,500; do not automatically reproduce the entire data-size sweep. If an existing approved smaller-budget experiment already exists, preserve/reuse it. Any later N=100 comparison must use the existing subset, paired Logit-KD/OFA controls and a predeclared budget; do not add it merely to find a setting where OFA wins.

If an OFA port is fundamentally incompatible or needs extensive rewriting, save the concrete feasibility finding in the run ledger, mark that cell blocked, and continue the other tasks. Never label an unavailable implementation as an empirical negative result or silently substitute another backbone. Do not port every new architecture to QAT/TensorRT; that is outside this phase.

## 5. Audit protocol provenance

Use Git history, exact configs, command lines, logs, selection code and saved artifacts to reconstruct these choices:

| Choice | Required evidence |
|---|---|
| Checkpoint selection | Dataset/split, metric, checkpoint candidates and selection rule |
| Calibration | Images/split, whether labels were consumed, PTQ/QAT role, exported artifact identity |
| Filtering thresholds | Actual f_min/f_max/variation thresholds, when set, and available origin evidence |
| OFA hyperparameters/exits | Source defaults or adaptations, any search space and selection data |
| Architecture selection | Deployment constraints, prior work, validation evidence and any role of external scores |

Add a compact provenance table to the existing tracker or a CSV/JSON with choice, value, data role, evidence path/revision and confidence status. Distinguish verified, user-reported, inferred and unknown. A source-code scan alone cannot prove that an external score was never viewed or used historically.

S11 validation use for selection/calibration is legitimate and does not itself contaminate STURM/WorldFloods. Merely running fixed evaluations is also not training or tuning. Conversely, no training gradients from an external set does not prove its metrics never influenced method choices. If evidence conflicts with the user's account or the manuscript, preserve the discrepancy and ask one precise question; do not rewrite the manuscript or assert misconduct. Continue independent experiments using a clearly frozen forward protocol.

## 6. Evaluate fully dry scenes without changing training first

Search existing authorized project datasets/manifests for genuinely held-out scenes with independently labeled or verified zero water over valid pixels. Prefer existing data and readers. Do not expand into a new acquisition/annotation platform. Exclude overlap with training, calibration and selection data; preserve event/location identifiers and reference provenance. Keep the canonical S11/STURM/WorldFloods evaluations unchanged.

Teacher-empty tiles and MNDWI-empty tiles may help identify candidates, but neither is independent dry-scene ground truth. Do not select and score only negatives that a tested model already declares dry. If only such candidates exist, produce a small existing-format inspection set and mark this task as awaiting independent verification. You may still report student-teacher disagreement, clearly named as such, but not true false-alarm rate. Do not invent annotations or silently treat cloud/nodata as dry land.

Use a fixed, representative negative set spanning available landscapes, including confusing surfaces if independently verified. Record how scenes were selected and any exclusions. A small pilot can validate the evaluator; report actual sample/event counts rather than claiming operational reliability from an arbitrary small sample.

Evaluate the teacher, local EfficientViT controls, completed new students and fixed MNDWI on exactly the same valid pixels. Use existing class decisions. Report per-scene predicted water fraction, aggregate false-positive pixel rate, and scene false-alarm frequency with prespecified criteria. Include any-positive-pixel incidence plus area thresholds such as 0.1% and 1% as explicitly diagnostic criteria; do not tune them against outcomes or present them as established operational alarm thresholds. Report the distribution and event-level summaries where available. Treat water IoU/F1 on empty references carefully; do not substitute all-background accuracy as the main result.

This is evaluation first. Do not automatically change the 5% training filter or add dry scenes to training. A training intervention would be a separate controlled follow-up after measuring the actual behavior.

## 7. Compare learned models and spectral rules under the existing robustness protocol

Test the hypothesis of a learned-model advantage honestly. A result favoring a rule is valid. Reuse the existing reflectance-space perturbation implementation and evaluation pipeline; verify transforms occur before fixed normalization and are identical for every comparator.

Include clean data and all existing axes:

- Common scalar gain: reported g in {0.80, 0.90, 1.00, 1.10, 1.20, 1.30}.
- Wavelength-dependent additive haze: reported h in {0, 0.05, 0.10, 0.20}. Recover the exact band weights from the original implementation; do not invent replacements.
- Existing per-band gain variants: visible-only and opposing visible/infrared gains, reported d in {0.05, 0.10, 0.20, 0.30}. Preserve the implemented directions and severity definitions.

Primary rule: MNDWI = (B3 - B11)/(B3 + B11), fixed threshold > 0, with the existing documented treatment of invalid denominators. Reuse NDWI only if already implemented as a secondary baseline. No per-dataset, per-severity or external-label threshold optimization. Preserve validity masks and reference labels across perturbations; do not clip, stretch or renormalize away the perturbation unless this is part of the original declared protocol.

First validate common-gain invariance of MNDWI within numerical tolerance; inspect epsilon, clipping, interpolation or invalid-value effects if it fails. Under differential gains and haze, measure the actual ranking rather than assuming the rule must fail. Distinguish pseudo-versus-S11 training comparisons from learned-versus-rule comparisons.

Evaluate the frozen teacher, reestablished EfficientViT Logit-KD/OFA and completed MobileNet/LightFormer controls on the same inputs. Include an existing final/deployed checkpoint only through an available valid execution path. Do not pretend a float RTX evaluation is Jetson INT8 execution; do not rebuild unavailable deployment engines as a dependency of this task.

Export clean scores, scores by severity, worst-case absolute and relative drops, water IoU, mIoU, precision/recall or false-water area where supported, and both pixel-pooled and scene/event summaries consistent with the project. Save enough per-scene data for paired comparisons. Show moderate severities as well as extremes and retain failures and null results. Reuse existing plotting code for matched curves and a small fixed set of qualitative examples, including rule wins. Do not cherry-pick scenes or severity ranges based on the desired conclusion.

The existing supplement reports large MNDWI losses under the strongest haze, particularly versus the teacher. Explicitly establish what holds for the actual students. Synthetic haze establishes sensitivity to that defined perturbation, not performance under every real atmosphere. Use independent event units for uncertainty, not pixels as independent replicates. Do not infer training-procedure superiority from one training seed; distinguish checkpoint-conditional event uncertainty from variation across training seeds. Do not equate nonsignificance with equivalence.

## 8. Build and start a bounded persistent queue on the RTX 4090

Reuse the existing launcher/scheduler first. If none can do this, implement only a thin wrapper around existing commands plus durable queue state; avoid a general scheduling system. CPU provenance/data inspection may proceed while the GPU queue runs, within conservative resource limits.

Default priority and dependencies:

| Phase | Work | Launch condition |
|---|---|---|
| P0 | Repository/state/protocol audit; queue persistence sentinel; model CPU checks | Immediately, without disturbing current jobs |
| P1 | Archived-checkpoint evaluation and first-seed EfficientViT baseline controls | Protocol resolved and GPU available |
| P2 | MobileNet/LightFormer forward-backward checks and first full-seed controls | Evaluator correctness and assessed first-seed EfficientViT training-reproduction gates passed; relevant adapters valid |
| P3 | Remaining baseline and architecture seeds, round-robin across valid configurations | Both baseline gates and each configuration's first-seed functional checks passed; frozen matrix and resource estimate recorded |
| P4 | Dry-scene evaluations of completed checkpoints | Independent negative references ready; otherwise block only this task |
| P5 | Matched robustness evaluations and aggregation | Corresponding clean checkpoint/evaluator checks passed |

Small evaluation jobs may run between training jobs, never concurrently with GPU-heavy training. Estimate wall time from measured runs, report pending GPU hours and revise ETAs as measurements arrive. Use actual queue dependencies rather than guessed calendar start times. Preserve existing user schedules; if none exist, launch opportunistically when sufficient headroom exists. Do not invent an overnight-only preference.

Persist both baseline gates as explicit dependencies, including the evidence that cleared them. Unresolved semantic discrepancies block new-model full training and expansion; a documented acceptable environment shift may clear the reproduction gate without matching the old means exactly. Do not bypass a gate merely because the initiating Codex session ends.

Requirements:

- At most **one GPU-heavy job** at a time, including teacher inference, training, calibration, or substantial evaluation. A training process containing its own teacher forward pass is one job. CPU agents must not start independent GPU work.
- Use a process-safe lock, check foreign GPU processes and available memory before every launch, and wait when busy. Do not kill or suspend existing work. Display processes are not automatically blockers; distinguish them from heavy compute and retain measured headroom.
- Start with modest DataLoader workers and bounded CPU/BLAS threads; avoid duplicate dataset scans and I/O-heavy parallel jobs. Use user-level CPU/I/O priority controls where available. CPU niceness does not reserve GPU capacity, so do not promise zero slowdown. No changes to GPU clocks, global power limits or system services.
- Preserve scientific batch size, precision, augmentation and optimization budget. If a model does not fit, diagnose first. Do not silently replace batch 16 by microbatches/gradient accumulation, which can alter BatchNorm and optimization semantics. Record and block unresolved OOM cases rather than retry indefinitely.
- Run the worker independently of the VSCode terminal. Prefer an existing supervisor; otherwise use an available user-level service or detached process/session appropriate to the detected OS. Do not assume Linux tools exist. Use explicit working directory and environment paths.
- Verify that the worker launches a second tiny CPU-only sentinel after its initiating terminal/process exits. This verifies parent-process independence; separately inspect supervisor settings. Do not claim an actual VSCode-close/logout test occurred unless it did. State the achieved persistence guarantee.
- Editor closure must not stop queued jobs. Machine sleep, power-off and reboot are separate conditions. Use existing authorized persistence facilities if available; do not use sudo or silently modify global boot/sleep settings. State any unfulfilled logout/reboot guarantee precisely.
- Persist run IDs, dependency/status state, exact commands, resolved config hashes, code/environment/data/checkpoint fingerprints, timestamps, log paths, exit codes and output paths atomically. Reconcile live PIDs/process identity on restart before resuming; never launch duplicates.
- Save checkpoints with existing safe handling and sufficient resume state. Do not edit source/configuration under a running job: use an existing immutable snapshot mechanism or finish the change before launching that cohort. Freeze queue inputs so later edits cannot silently change pending experiments.
- Provide minimal status, pause-new-launches, resume and graceful-stop controls. Pausing leaves the current job running. Separate explicit cancellation from stopping the worker. Track only owned processes.
- Failed provenance/data/shape/baseline gates block dependents. Independent safe jobs may continue. Allow at most one recorded retry for a clearly transient failure; repeat OOM or invalid metrics must block the run. Completion requires expected outputs and valid metrics, not exit code alone.
- The queue executes already specified commands; it must not require an active Codex chat to choose the next run. Scientific ambiguities may produce explicit blocked states rather than autonomous recipe changes. Once the finite matrix/evaluations finish, stop launching jobs.

## Completion for this phase

Actually start and verify the persistent queue when the project machine is available. Do not claim experiments are complete merely because they are queued. If a long queue remains, return a short operational update and allow it to continue without VSCode or an active chat.

At handoff, report only: branch/root used; minimal code changes; reused artifacts; baseline reproduction status; completed/running/pending/blocked counts; queue mechanism and persistence limitations; concrete status/pause/resume/stop commands; approximate remaining GPU time; and any precise missing input. No manuscript revisions or narrative report.

Success means the project has a verified baseline path, valid OFA feasibility/controls for the requested architectures or concrete blockers, a factual protocol-provenance record, an independent dry-scene evaluation or honestly identified reference gap, matched robustness outputs including unfavorable results, and a durable bounded queue that continues automatically. Preserve all original results.
