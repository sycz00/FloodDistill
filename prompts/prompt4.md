# Reproduce the paper with recovered original data and complete the reviewer comparisons

Execute this in the existing FloodDistill research checkout on the new research machine, on the `research` branch. Respect applicable AGENTS.md/AGENT.md, preserve unrelated work and reuse existing loaders, training code, evaluators, ledgers and the detached scheduler. Execute eligible work; do not stop at proposing a plan.

## 1. Updated authorization and supplied inputs

The user has now supplied:

| Input | Location |
|---|---|
| Original paper checkpoints and evaluations | `runs/original_runs/`, relative to the project root |
| Copied original source imagery | `/data1/schmalstieg/flooddistill/data/flood_unlabeled_v2` |
| Original indices and saved `.pt` artifacts | `/data1/schmalstieg/flooddistill/data/pseudo_gdacs_v2` |

Verify these paths locally. The repository was previously `/home/schmalstieg/repos/FloodDistill`. These paths describe the user's research machine, not a document-review workspace.

**This update supersedes prompt4's previous zero-training restriction and its two-evaluation limit. It authorizes original-checkpoint reproduction, the finite original-data retraining/comparison matrix below, and automatic scheduling after correctness gates pass.** Do not ask again whether those specific runs may start. Missing data or ambiguous scientific identities must be reported, not guessed. Continue independent work while an affected task is blocked.

Only address reproduction and the existing reviewers: architecture/OFA controls, modest gains, data roles and leakage, geometry confounding, MNDWI trade-offs, dry imagery, and bounded deployment evidence. Do not introduce new teachers, datasets, architectures, threshold searches, budget sweeps, QAT training, few-shot/multimodal/XAI research or speculative novelty experiments. No manuscript editing or full response-letter drafting yet; maintain the existing technical outputs and a concise handoff.

## 2. Preserve completed work and refresh live state

**Latest user status: the full programme is still running.** Treat this as an active continuation, not a completed programme or a fresh start. Refresh the live state to identify which training, evaluation and collection jobs are actually unfinished. Preserve all healthy active jobs and their frozen inputs. New original-data tasks supplement the running programme through its existing queue; they must not replace, restart or duplicate it.

Read prompt1, prompt2, the previous experiment prompt and the latest records in `experiments/revision/`, especially:

- `prompt3_handoff.txt`, `prompt3_run_status.json`, `prompt3_current_gate_handoff.json`
- `prompt3_completed_training_audit.json` and its text summary
- `prompt3_reviewer_status.json`, `prompt3_reviewer_handoff.txt`
- `prompt3_evaluation_plan.json`, `prompt3_evaluation_summary.json`, `prompt3_sensitivity_preflight.json`
- `prompt3_deployment_provenance.json` and the dry-reference package
- The supplied paper, supplement and actual reviewer letter where available.

The 2026-10-06 11:57 Berlin snapshot reported 60 completed revision training runs and canonical student readouts. Its corrected WorldFloods/STURM queue was still progressing after a missing-package-file repair. These are historical counts. Inspect actual worker/lock/heartbeat, queue manifests and outputs before scheduling anything. The first 36 revision runs are not automatically the historical paper runs now copied into `runs/original_runs/`.

The user's newer running-status update takes precedence over any inference of completion from that snapshot. Do not assume that current active jobs have the same identities or counts. This also applies if an earlier version of this prompt has already queued original-data work: reconcile its actual manifests and reuse those jobs.

Finish and collect already authorized pending sensitivity work for the regenerated cohort. Reuse `analysis_prompt3_complete` if it now exists and is valid. Never rerun completed cells merely because this prompt arrived. Do not modify an active job's paths, data, code snapshot or environment. Preserve the regenerated cohort under `/data1/schmalstieg/flooddistill/revision_20261005/regen_N2500_approved_20261005` and its results as a separate study.

## 3. Establish whether the original data and checkpoints are complete

### Original run inventory

Inspect `runs/original_runs/` using existing code and artifact readers. Map each relevant checkpoint and evaluation to paper table/figure, architecture, objective, budget, seed, selected epoch, dataset manifest, code/config identity and precision/activation variant. Do not infer scientific identity from filename alone. Distinguish `best` from `last`, main student from OFA heads, pre-QAT float from QAT, original run from copied duplicate, and raw per-seed measurements from summary means.

Check required tensor keys/shapes, loading compatibility, finite tensors, and main model versus auxiliary-head separation. Reuse existing loading and smoke checks. Record SHA256 and exact missing artifacts. Do not silently load with ignored missing/unexpected model keys. An inference-ready main checkpoint can be complete for rescoring without being a training-resume checkpoint; optimizer state is not needed for a fresh reproduction run. If weights lack configs or selection records, label the corresponding identity as unverified rather than declaring the whole experiment complete.

Recover original configs, evaluation logs/counts/predictions, model/evaluator source and environment exports from the supplied folder/project. Do not assume the new machine has the original environment. Preserve original files unchanged; write new outputs elsewhere. No network scanning, invented SSH access or changes to the original machine's environment.

### Original dataset identity gate

Use the actual existing loaders to inspect both supplied dataset directories and determine what each `.pt` stores: inputs, hard labels, soft outputs, descriptors, indices or other tensors. Do not guess semantics, renormalize already normalized tensors, or regenerate original pseudo-labels merely because they can be generated again.

Reconstruct the exact historical source-to-cache-to-selected-ID mapping from original indices, manifests and configs. Numeric indices may refer to an old ordered source list: do not apply them to a newly sorted filesystem listing. Preserve selected order, scene/acquisition identity and source/label pairing.

Check every selected N=2500 record for:

- Source and required cache existence/readability, unique ID and correct image/label pairing.
- Shape, band order, radiometry/normalization, label values, validity masks and finite inputs.
- Teacher/cache identity and whether original training used saved hard masks, live soft targets or cached soft targets.
- Agreement with historical hashes where available. A hash computed now identifies the copied bytes but alone cannot prove they match historical bytes.

Resolve each of the previously missing 127 original source IDs explicitly. Produce counts/lists of restored, still missing, mismatched and ambiguous records. **Do not declare the 127 replacements eliminated because the folders exist or contain 2,500 files.** Use the original selected IDs and original associated pseudo artifacts, without substituting replacement scenes or freshly generated labels. Missing nonselected pool images need not block selected-N2500 training; missing selected images or required labels do.

Freeze an original-data manifest with indices, source/cache hashes, loader/config identity and the provenance limits. Verify split disjointness against the existing S11 train/validation/test memberships and repeat existing overlap screens only where the restored IDs make them necessary. Reuse unchanged audit evidence. Disjoint IDs do not prove independent geography or historical non-use of test scores.

Recover the actual historical N=100 subset if available. Otherwise, a new matched N=100 subset may use the first 100 entries of the verified original FPS ordering, frozen before evaluation, but must be labelled a new original-data control, not an exact historical subset reproduction. If the ordering/selection rule cannot be verified, request the specific missing indices and block that subset; do not choose another favourable budget.

### Paths and compatibility

Prefer explicit data-root configuration or a small path-mapping change over moving data. If a legacy path is hardcoded, a minimal backward-compatible loader/config patch, documented alias, or necessary directory rename is authorized. Keep the mapping recorded and verify identical loaded content. Do not globally replace paths, overwrite existing directories, rename roots used by live jobs, or redirect the regenerated cohort to the originals. Freeze each queued run's resolved paths and manifest so a later path change cannot alter its inputs.

## 4. Reproduce scores from the original checkpoints first

This phase checks evaluation reproducibility; it does not by itself reproduce training.

Predeclare the relevant original checkpoint set from paper claims and available run records before reading new scores. Include all reported seeds, not a favourable subset:

1. The primary EfficientViT N=2500 hard-only, Logit-KD and OFA rows.
2. The original compact-backbone comparisons and the single N=100 KD/OFA comparison where the actual corresponding runs exist.
3. The fixed teacher and MNDWI once per distinct evaluation protocol, reusing outputs with matching identities.
4. Original deployment artifacts through the separate device gate, not as desktop substitutes.

Do not rescore every exploratory checkpoint or every budget. Inventory historical null experiments if relevant receipts exist; do not launch new null-method searches.

Start with one primary seed selected by fixed availability order 42, 64, 91, 107 for a functional check, then evaluate every available declared seed regardless of its score. Use S11 clean as the initial short check. Reuse old counts/predictions only when checkpoint, dataset, masks and evaluator identities match.

For original score reproduction, use the executed historical evaluation protocol recovered from code/configs, including original geometry, class definitions, ignore masks and aggregation. Keep the published printed mean/SD as a separate reference, recomputed historical aggregate beside it, and each per-seed discrepancy. Agreement after rounding is not proof of bitwise equivalence. Distinguish water IoU from two-class mIoU and sample SD from other summaries. Never force the old numbers by changing thresholds, inputs, seeds or selection.

Also evaluate the retained original checkpoint comparisons under the already established corrected WorldFloods/STURM protocols where necessary for reviewer conclusions. Reuse unchanged interior/count cells and fixed baselines only when their identity contracts pass. Legacy and corrected rows must remain separately named. New precision/recall must come from the same count domain as that row's IoU, not be appended to unmatched historical means.

If old and new evaluations disagree, check data/order, labels, bands, normalization, resize/crop, masks, model state, metric pooling, precision and source versions before attributing the discrepancy to hardware. Compare fixed input tensors/predictions where available. If a true old-environment reference is missing, report that limit. Use an original-machine invocation only when an already authorized workflow exists; do not block all work on remote access. Preserve a separate evaluation-only environment/source snapshot if required, without downgrading active jobs' environment.

## 5. Reproduce training and fill the original-data control matrix

Checkpoint rescoring and training reproduction answer different questions. The user explicitly requests reproduction with the original data. After the data gate and functional checks pass, run the following bounded programme. Reuse a completed training cell only if its actual data, recipe, initialization/pretraining, selection policy and declared environment/hardware stratum qualify for that comparison. Copying an old-machine checkpoint does not make it a new-machine training replicate.

Before launch, freeze a deduplicated manifest with reviewer purpose, data digest, objective, seed, updates, selection rule, environment, GPU assignment and reuse decision. Eligibility is based on missing controlled evidence, not whether earlier results favour OFA.

| Group | Original-data training cells before reuse | Maximum | Purpose |
|---|---|---:|---|
| A: historical primary recipe | EfficientViT N=2500: executed true-CE, historical KD, historical OFA; four seeds each | 12 | Reproduce the main training results on recovered original data |
| B: primary matched comparison | EfficientViT N=2500: clipping/mask-matched KD and OFA; four seeds each | 8 | Isolate the OFA comparison on the original cohort |
| C: hard-loss control | EfficientViT N=2500: full-weight focal-only, clipping matched; four seeds | 4 | Separate the weighted KD recipe from the hard-loss family |
| D: alternative backbones | MobileNetV3-Small and LightFormer-ResNet18 N=2500: matched KD/OFA; four seeds each | 16 | Address reviewers' architecture/OFA-control concern on original data |
| E: one small budget | EfficientViT N=100: matched KD/OFA; four seeds each | 8 | Check the small-budget conclusion on a declared original-data subset |

The ceiling is **48 unique new training cells before reuse**. Typically the four Group A OFA runs can also fill Group B, giving **44 or fewer**, but only if actual recipes, initialization and mask domains match. This is a ceiling, not a target. Reuse every compatible completed cell. Do not add alternative-backbone historical CE reproduction or other budget/teacher/model runs to this matrix.

Use seeds 42, 64, 91 and 107 for matched controls. Confirm the historical paper's seed list from the newly supplied records. If it differs, report the difference and do not call these four current seeds exact historical seed reproduction; do not silently add another seed grid. All predeclared valid seeds must finish regardless of accuracy. Functional gates concern correctness, finite loss, gradients, teacher freezing and memory, not a minimum test score.

Group A preserves the actual historical recipe, including its known limitations; do not silently correct its objective or clipping and still call it historical reproduction. Recover its original training geometry, data order/augmentation, batch size, optimizer, LR schedule, step/epoch budget, initialization, teacher, checkpoint selection and loss algebra. Where old runtime versions cannot be recovered, state original-recipe reproduction in the current environment rather than exact historical execution. If a scientifically essential configuration is unknown, block only that historical cell and request the specific record; matched controls with fully known recipes may proceed.

For Groups B-E, retain the already frozen controlled recipe from the previous revision programme:

- Same original IDs and teacher/input transforms within each budget; main-model initialization paired by architecture and seed, verified with initial hashes where possible.
- Full-weight focal-only gamma 2 for C; recovered 0.5*focal + 0.5*KL for KD; existing OFA branch algebra and heads added to that same main objective.
- Gradient clipping norm 5 for matched focal/KD/OFA, with the same AMP-unscale convention.
- The recovered branch temperatures and `0.3 * mean(four exit losses)`, without redesigning the equation to match incorrect manuscript prose.
- The same declared valid domain for matched main/branch losses. Scan original masks through augmentation: the regenerated cohort's zero-ignore finding does not establish that original masks are all valid. Reuse Group A OFA only if equivalence is demonstrated. Do not treat an absent class label as proven cloud/sensor validity.
- Frozen teacher; live/cached soft-target use consistent with the recovered training semantics. No pseudo-label refresh or new teacher fine-tuning.
- Existing matched optimizer, LR, batch 16, 224 training geometry and 50-epoch policy if confirmed in the current receipts; preserve actual loader drop/selection behaviour. Report completed and selected optimizer steps. Match compute policy within each method pair; N100 versus N2500 is not an equal-compute comparison.
- Unchanged six-band adaptation and pretraining; LightFormer's native auxiliary objective uniformly disabled; OFA exits discarded at inference.
- Checkpoint selection using the existing S11 validation rule only. No test-guided early stopping, retuning or selecting a seed by external performance.

If historical and controlled implementations differ, retain explicit variants and separate outputs. Add only the minimal compatibility/configuration changes needed to execute them using existing components. Do not rebuild the pipeline.

For every new cell, collect clean S11/STURM/WorldFloods metrics with the fixed evaluation definitions. Reuse existing correction adapters to retain separately identified legacy/corrected external results. For robustness, repeat only the previously declared primary EfficientViT-versus-teacher/MNDWI conditions needed for the reviewer argument on original data; do not extend a full corruption sweep to every architecture and budget. Use actual student outputs, not teacher behaviour as a proxy. Freeze this evaluation matrix before inspecting scores.

Compare original-data versus regenerated-data retraining only where recipe, environment, initialization policy, budget and scoring protocol match. If cached pseudo-label generation or another factor also differs, name the contrast as a joint cohort/cache change; it cannot isolate the effect of exactly 127 images. Never average seeds across the two datasets. Preserve the earlier null/negative results and report whether conclusions persist.

## 6. Scheduling and bounded execution

Reuse the current detached scheduler, locks, immutable snapshots and atomic state updates. No second competing worker or new framework. Preserve authorized GPU identities/concurrency; at most one heavy job per approved GPU with CPU/I/O headroom. Pair hardware classes where possible and record assignment; do not assign devices based on observed scores. Never stop unrelated workloads or change global clocks/power policies.

Priority order, allowing independent work on available resources:

1. Refresh state and finish/collect the pending existing evaluation corrections.
2. Inventory the copied originals and freeze verified original-data/run identities.
3. Original checkpoint functional check, then missing declared original rescoring.
4. First-seed functional checks for Group A and missing matched variants; complete A/B/C across all seeds.
5. Complete eligible D/E cells and their readouts, deduplicating the shared OFA cells.
6. Required matched robustness, eligible dry/device work, then the final collector/handoff.

### Evaluations must start automatically after their training dependencies finish

Register the missing evaluation and collection dependencies now using the existing scheduler, so they do not require another user message or an open VSCode session. Do not wait for the whole training programme to end before arranging them.

- For each already running, queued or newly authorized relevant training cell, first check whether the required evaluations are already queued, running or complete. Add only missing cells, deduplicated by checkpoint/run identity, data, evaluator, mask, condition and protocol.
- Release a final evaluation only after its training job succeeds and its validation-selected checkpoint is finalized, readable and hashed. A `best.pt` file that exists while training is still updating it is not a final-checkpoint receipt. Do not silently use `last.pt`, an intermediate checkpoint or a checkpoint selected by test accuracy. A failed or incomplete training job blocks its own dependent evaluations; independent jobs continue.
- As each final checkpoint becomes available, automatically run its authorized clean S11/STURM/WorldFloods evaluations and the separately specified correction/robustness jobs, reusing compatible results. Keep tasks conditional on the original-data, reference and device gates defined above. Schedule evaluations with the same GPU/headroom limits as training, without killing or oversubscribing active training jobs.
- Attach per-comparison collectors to all required method/seed/condition outputs, not merely to training completion. Tables produced while any required seed or condition is missing must be explicitly labelled partial, with missing cells shown; never silently average only completed or favourable seeds.
- Attach the final aggregate collector to the entire declared runnable evaluation matrix. Verify completion from successful receipts and valid output/count files, not an empty process list or the mere existence of result paths. It must regenerate paired tables, corrected-versus-legacy comparisons and the reviewer-status handoff when those dependencies finish.
- Keep the running and newly added original-data programmes separately identifiable. A finished regenerated-cohort table can be final for that cohort while original-data reproduction remains running; it must not be presented as the final original-data result.
- A collector blocked on independent reference/device inputs must not suppress completed numerical results. Produce an explicitly bounded computation-complete handoff listing those gates. Use the existing dependency/resume mechanism so newly supplied valid inputs trigger their already authorized jobs and refresh the aggregate outputs, without fabricating references or claiming that the entire programme is finished.

Before leaving the interactive session, verify the actual dependency links, worker heartbeat and durable queue state. Report what is running, which evaluations will follow automatically, and the location of their outputs. Do not report final scientific results while those evaluations are still pending. Reuse the scheduler's existing waiting/dependency support; do not create a separate polling daemon, another scheduler or a long blocking chat wait.

Estimate GPU-hours from actual previous durations before launching the finite matrix and update the estimate as it runs. The user has authorized this matrix; presenting its cost is not an extra approval gate. Preserve settings rather than reducing batch size or changing seeds after OOM. Record a bounded transient retry; repeated resource/correctness failures block affected dependents with the exact reason.

Keep eligible jobs running after VSCode/editor closure using the existing detached mechanism. Check heartbeat/state; do not promise survival of reboot/sleep/shutdown or install services. If original-data work is gated, do not replace missing scenes to keep GPUs busy.

## 7. Existing dry-reference, deployment and history gates

Use genuine completed dispositions for the 11 dry candidates if now supplied; import through the existing validator and run the prepared scoring only after it passes. Do not invent annotations, reviewer identities or assessability. Newly copied training data do not themselves verify dry references. Keep the within-benchmark, three-event and full-acquisition limitations explicit. Reuse existing verified-reference jobs; score original-data primary candidates only on that same frozen reference and criteria where required for the retained practical claim, without expanding the reference set or tuning thresholds.

Reinspect deployment provenance against the newly supplied originals: items previously missing may now be available. Prefer verified original logs/engines and recovered per-seed ancestry. Any necessary remeasurement remains on the actually available, already authorized Jetson Xavier NX, with matching artifacts and a distinct identity. RTX timing cannot validate Xavier performance. No new QAT training; no assumed TensorRT engine portability. Verify that device evaluation uses its named split (the prepared S11 path previously selected validation), and do not use a latency-only QAT-stripped float export as a genuine float-accuracy baseline. Record mixed precision, engine hashes, timing scope, device settings and actual named power rails; peak watts are not energy per image.

Use any supplied factual clarification about historical benchmark influence. Dataset restoration and new experiments cannot establish whether STURM/WorldFloods scores guided past choices. Do not repeatedly ask the same already recorded history question or block independent computation on it.

## 8. Deliverables, missing-data request and stop condition

Use existing artifact directories/collectors, adding an explicitly original-data run namespace and manifest. Do not overwrite `runs/original_runs/`, the original dataset/cache, regenerated-cohort results, or paper-source summaries.

Deliver:

1. Original checkpoint completeness and paper-row mapping, with missing seeds/configs/evaluations explicitly identified.
2. Original dataset reconciliation, including all 127 former missing IDs, exact N2500/N100 manifests and the level of historical identity that is actually verified.
3. Original checkpoint score reproduction: paper values, recovered original raw aggregates, new legacy-protocol evaluations, corrected-protocol evaluations and differences. Keep incomplete seed groups incomplete rather than substituting runs.
4. The deduplicated finite training/evaluation matrix and live completed/running/pending/blocked states; commands for status/pause/resume/stop and a measured remaining-cost estimate.
5. Four-seed original-data reproduction and matched comparisons, with paired differences, sample SD, water IoU/mIoU/precision/recall, and separately identified event-level uncertainty. Fixed teacher/MNDWI evaluations are not repeated independent seeds.
6. A comparison with regenerated-data findings only for genuinely comparable rows. Separate original reported results, original checkpoint rescoring, historical-recipe retraining, corrected matched training, and device results.
7. Reviewer concern -> evidence -> supported conclusion -> remaining writing/input action. Retain actual manuscript corrections and existing qualifications; do not manufacture problems from earlier audit overstatements. No claim of general OFA superiority, exact reproduction, no leakage or dry-scene reliability without suitable evidence.

Every handoff must carry its capture time and distinguish `running/partial`, `computation-complete with explicit external-input blockers`, and `complete for the declared scope`. Preserve intermediate captures and identify the latest final collector output. Training completion alone does not satisfy completion: the required evaluations and aggregate checks must also finish.

If more original data are needed, tell the user exactly what to provide: missing scene/selected IDs and expected paths; source versus pseudo-cache versus indices; missing checkpoint seeds/configs/teacher version; evaluation labels/masks; or device artifacts. State which jobs depend on each item and which continue. Ask for only the missing items, not the whole dataset again. Report newly discovered critical missing inputs promptly while the detached eligible queue continues.

Do not reopen the completed external RSE/MNDWI diagnostic search, chase its reported number, or run experiments until OFA wins. Stop after this finite reproduction/reviewer programme and give an honest account of agreement, disagreement and unresolved evidence. A reproducible negative result is a valid outcome.
