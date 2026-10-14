# Finish the reviewer evidence with bounded, automatically scheduled experiments

Work in the existing FloodDistill project on the `research` branch. Read applicable `AGENT.md`/`AGENTS.md`, `prompt1.md`, `prompt2.md`, the latest experiment ledger and the attached `prompt2_report.pdf` before changing anything. Reuse existing code, data, checkpoints, evaluators and the persistent queue. Execute the authorized work; do not stop at a proposed plan.

This prompt follows the report captured on **2026-10-05 at 19:21:34 Europe/Berlin**. Its counts and running process IDs are a historical snapshot, not current state. At that capture, 14 of the original 36 training jobs had succeeded, one was running and 21 were pending. Refresh the actual state first.

The report identifies the repository as `/home/schmalstieg/repos/FloodDistill` and the approved current cohort under `/data1/schmalstieg/flooddistill/revision_20261005/regen_N2500_approved_20261005`. Verify these paths locally. This prompt is intended for that research machine, not for a document-review workspace without its data or GPUs.

## 1. Authorization, scope and scientific purpose

Only address the existing reviewers:

- R1.1 / R2.1: clarify the contribution beyond established distillation, pseudo-labelling, selection and quantization methods.
- R1.2 / R2.4: compare compact architectures and test OFA with alternative backbones under controlled conditions.
- R1.3: explain training, validation, calibration, selection and final-evaluation roles; assess possible leakage.
- R1.4: separate component contributions and their confounders.
- R1.5: use consistent surface-water, flood and supervision terminology.
- R1.6: characterize or explicitly limit claims about false alarms on fully dry imagery.
- R2.2: address image, geometry and geography confounding in the S11-versus-GDACS comparison.
- R2.3: delimit OFA's modest, budget-dependent advantage.
- R2.5: show when the actual student offers a useful trade-off against MNDWI.
- R2's deployment comment and the editor: qualify mixed precision/hardware claims and prepare evidence for a Discussion and point-by-point response.

**This prompt now authorizes the specific additional controls below.** It supersedes prompt2's no-new-training restriction only for that finite matrix. It does not authorize an open-ended method search, another dataset programme, new teachers, SAR/multimodal/XAI/few-shot experiments, another full budget sweep, new negative-scene training, or new QAT training. Preserve all original results. Do not edit the manuscript or write the reviewer response in this phase; produce the evidence needed for that later pass.

Reuse completed compatible work. Do not regenerate the approved pool or rerun the original 36 configurations just because their recipe has limitations. Keep their true-CE, historical-KD and historical-OFA identities explicit.

## 2. Establish the current state and repair the evidence classification

1. Reconcile the existing worker, locks, manifests, live processes, outputs and dependency states. Avoid launching another scheduler. Preserve dirty work and immutable inputs of active jobs; pause new launches briefly only if needed to update the queue safely, then resume eligible work. Never kill an unrelated process or silently change a running job.
2. Verify the approved pool receipt and digest. The report found 2,373 original sources and 127 replacements among the current 2,500 images. Use that same frozen current cohort for every new matched comparison. It is not an exact reconstruction of the paper's original training data. Keep original-data, regenerated-data and different-environment results separate.
3. Match report findings to actual code/artifacts and manuscript sentences. The report's `Claim` fields sometimes describe hypothetical assertions, not quotations from the paper. For example, the supplied manuscript already identifies mixed INT8/FP16 execution, acknowledges inconsistent full-budget OFA gains, and the supplement rejects an accuracy-ceiling interpretation. Do not create new errors or experiments from such hypothetical assertions. Classify each record as a demonstrated mismatch, an already acknowledged limitation, a source/provenance gap, or an unsupported assertion in the report.
4. Separate work completion from concern resolution. A completed audit that finds a missing reference set or a confounded comparison does not close that concern. Record `evidence-ready`, `needs-experiment`, `needs-author-history`, `needs-reference-verification`, `needs-device-artifact`, or `writing-only`, with evidence paths.
5. Reconcile archived-checkpoint rescoring with the exact original run identities where possible. Some fresh count-derived learned-model scores differ from the paper-source means. Do not attach fresh precision/recall to old IoU means as though they were one evaluation. Use the same checkpoint, mask, evaluator and aggregation for each reported metric row. Do not force agreement with published targets or block all new controls indefinitely because an old execution hash is unavailable.

Source context: the report says the actual reviewer letter and reference article PDFs were absent from the research checkout. Read supplied copies if available. The reviewer scope above is authoritative for this task; missing article PDFs are not permission to invent their methods or launch a broader literature benchmark.

## 3. Freeze the minimal corrected comparison before launching training

The report establishes three different issues:

- The hard-only baseline is **true CE**; KD and OFA use focal loss with gamma 2. Their difference is not solely the addition of soft targets.
- Historical OFA uses gradient clipping at norm 5, while Logit-KD uses no clipping. An OFA-versus-KD difference also includes clipping.
- Main KL includes ignored pixels, unlike the hard and OFA branch losses. A two-pixel diagnostic establishes the code behaviour, but not its prevalence or empirical effect in the actual training pool.

Before using any completed run as a matched control, compare resolved loss, clipping, mask reduction, data, initialization/pretraining, augmentation, update budget and selection configurations. Do not infer compatibility from an objective name alone.

### A. Resolve mask scope first, without training

Inspect the masks actually reaching loss computation for the approved N=2,500 cohort and the proposed N=100 subset, including augmentation/padding. Reuse the existing dataset scan if it already establishes this. Count ignored pixels/images and distinguish missing human annotation from sensor-invalid/nodata/cloud regions where metadata permits. Missing annotation and invalid imagery are not the same thing. Training-time teacher supervision on an unlabelled pixel is not automatically test leakage.

For the new controlled variants, use the same declared valid-pixel domain for the main hard loss, main KL and auxiliary branch losses. Preserve the recovered KL temperature, temperature multiplier and per-pixel/class reduction; make only the mask restriction explicit. If a batch is entirely ignored, handle it deterministically without NaNs or fabricated background labels and record the skipped/zero-contribution behaviour.

Reuse the existing two-pixel diagnostic to require zero ignored-pixel gradient in the valid-only variant and equivalence to the inherited loss on an all-valid batch, within the existing numerical tolerance. This is a narrow regression check, not a new testing framework.

If every relevant pseudo-training pixel is valid, valid-only and all-pixel reductions are equivalent and an otherwise matching existing OFA run is reusable. If not, do not compare a newly masked KD run to an unmasked OFA run and call that an OFA ablation. Use the conditional replacement budget below. Record the legacy S11 mask semantics separately; this prompt does not authorize retraining the entire S11 study. Retain its results as the executed recipe and narrow claims of an isolated label-source effect.

### B. Shared settings for the new matched controls

- Architectures: existing EfficientViT-B0, MobileNetV3-Small and LightFormer-ResNet18 implementations only.
- Seeds: 42, 64, 91 and 107. Complete all valid predeclared seeds regardless of accuracy.
- Data: identical frozen approved IDs within each budget; identical teacher and input/label transformations.
- Main focal loss: gamma 2 with the recovered implementation and other parameters unchanged.
- KD: recovered `0.5 * focal + 0.5 * KL` implementation, with the explicit mask policy above.
- OFA: the same main KD objective plus the existing `0.3 * mean(branch losses)` and actual existing heads. Preserve the executed teacher/student branch temperatures and lack/presence of branch scaling as recovered; do not redesign the objective to make it fit an incorrect printed equation.
- Gradient clipping: **norm 5 for both KD and OFA and the new focal-only control**, with the same norm/AMP-unscale convention. Existing no-clipping controls remain separate.
- Retain the frozen optimizer, LR/scheduler, batch 16, 224 training geometry, augmentation and validation-selection rules. Match epoch/update and early-stopping policy within each comparison; record actual updates and selected epoch. No test-guided tuning.
- Match main-model pretraining and six-band weight adaptation within each architecture. Use stored initialization hashes where available; do not invent proof of identical initial tensors when historical hashes are missing.
- LightFormer's native auxiliary loss remains uniformly disabled. Preserve its existing main decoder and distinguish active inference parameters, unused native parameters and OFA training-only parameters.
- Keep teacher parameters frozen and teacher outputs aligned with the augmented input. Discard OFA exits at inference. Record pre-clipping gradient norms/clipped-step frequency only through a small existing logging hook if available; no gradient-diagnostic research programme.

Give the new variants explicit names/configuration identities, e.g. `matched_focal_g2_clip5`, `matched_lkd_clip5_valid` and `matched_ofa_clip5_valid`. Never overwrite historical CE/KD/OFA artifacts or relabel their semantics.

## 4. Finite training matrix and reuse budget

| Task | Architecture and data | Cells before reuse | Reviewer purpose |
|---|---|---:|---|
| T1: clipping-matched KD | All three existing architectures, N=2,500; KD at clip 5 | 3 architectures × 4 seeds = **12 full-budget runs maximum** | Compare against otherwise identical OFA to answer R1.4/R2.4 without the clipping confound |
| T2: focal-only control | EfficientViT-B0, N=2,500; full-weight focal-only, clip 5 | **4 full-budget runs maximum** | Compare the weighted KD recipe against the same hard-loss family; retain true CE as an additional historical/current-cohort baseline |
| T3: one small-budget check | EfficientViT-B0 only, N=100; matched KD and matched OFA, both clip 5 | 2 objectives × 4 seeds = **8 short runs maximum** | Check R2.3 at one predeclared small budget after removing the clipping confound |
| T4: conditional matching OFA | N=2,500, only architecture/seed cells whose existing OFA is incompatible with §3 | **Up to 12 full-budget runs**, only for documented incompatibility | Supply valid partners for T1 when loss masks or another required control prevent reuse |

**Default added budget: at most 16 full-budget runs plus 8 N=100 runs, before deduplication. Absolute ceiling: 36 additional training runs including the 12 conditional OFA partners.** The already authorized original matrix is separate. The conditional allowance is not a request to retrain all OFA models. Use compatible completed or already pending partners first. If the ceiling cannot produce valid pairs, mark the specific comparison blocked instead of expanding it.

T2 uses full-weight focal-only, not a deliberately downweighted hard baseline. The focal-only-to-KD contrast evaluates the stated weighted KD recipe: its hard coefficient changes from 1 to 0.5 as the soft term is introduced. Do not describe this as a derivative experiment that holds every loss coefficient fixed. No extra coefficient sweep or half-weight baseline is required for the narrower reviewer response.

Use the existing compatible frozen N=100 subset from the current cohort when available. Otherwise freeze the first 100 entries of the existing recorded FPS ordering of the approved current pool before any new evaluation. Preserve its ordered manifest/digest and label it a current-cohort subset, not a reproduction of historical N=100 data. If no trustworthy ordering/subset exists, block T3 only; do not rerun curation. N=100 is selected here in advance, not because it is the largest historical gain. Do not add N=50/150/200 or other budgets later.

T3 tests a sample-budget-specific difference under the same training policy within N=100. It is not few-shot domain adaptation or an equal-compute comparison across N=100 and N=2,500. Report actual optimizer steps. Launch its complete four-seed pair regardless of whether T1 finds an OFA gain.

Start with the first seed for functional validation of each genuinely new path, then complete all predeclared seeds after those checks. Functional gates concern valid shapes, masks, finite losses, gradients, checkpoint reload and output identity; they must not require OFA or any model to win. Do not rerun completed original CE/KD controls merely for symmetry.

## 5. Finish necessary inference and check evaluation geometry

### A. Existing canonical clean and robustness evaluation

First reuse or finish the teacher, MNDWI and student evaluations already queued. Evaluate newly matched checkpoints through the same canonical clean S11/STURM/WorldFloods path and add confusion counts during that pass. Use validation-selected checkpoints only. Freeze model lists and condition matrices before inspecting new external results.

For practical MNDWI comparisons, the required student robustness set is the EfficientViT matched focal-only, KD and OFA N=2,500 four-seed controls, the teacher and fixed MNDWI. Use only the already defined scalar-gain, per-band-gain and haze conditions. Reuse exact compatible predictions; do not rerun the rule's completed diagnostics. Existing MobileNet/LightFormer robustness jobs may finish, but no new robustness sweep for conditional replacement models is required merely to enlarge the table. T3 needs clean evaluation only.

Report water IoU, two-class mIoU, precision and recall on identical valid pixels, both appropriate pooled and map/event summaries. Keep event uncertainty separate from seed variability. For empty references, disclose undefined metrics and use false-positive area/scene-alarm diagnostics. Preserve moderate corruptions, rule wins and null results. Any student advantage must be demonstrated for the student, not inferred from the teacher.

Use a single declared seed-SD convention and identify whether archived summaries used population or sample SD. Do not treat that convention change as a new performance finding. Never pool predictions across seeds into an unlabelled ensemble.

### B. WorldFloods border geometry: one necessary evaluation sensitivity

The report found **148 partial boundary windows among 785 windows**. Resizing each boundary window to 512×512 changes both geometry and the number of scored pixels per unit of ground. Correcting the written description alone does not measure whether the reported comparison depends on this choice.

Create a narrowly scoped **native-grid border evaluation variant**, separate from the preserved legacy evaluator:

1. Keep full 512×512 interior windows unchanged. For each partial border window, retain native pixel size; pad the input to the model's required shape using a fixed, documented existing padding rule, then crop predictions back to the original extent. Do not resize its categorical reference or count padding. Use the same original class/cloud/JRC masks, geographic extent and threshold.
2. Fix the padding rule from the existing implementation before scoring; do not compare multiple padding methods to select a winner. Verify original affine coordinates, coverage, one count per valid native pixel and zero contribution from padding. Do not interpolate nodata into valid input.
3. Reuse interior counts/predictions only after verifying their evaluator/input identity. Run only the missing boundary inference where possible. The teacher and fixed rule are evaluated once; students use all valid seeds.
4. Apply this check to the teacher, MNDWI and the final matched full-budget student comparisons from §4, including the existing true-CE rows if they are retained in the architecture table. Do not extend it to every historical sweep/checkpoint or every corruption severity.
5. Export paired legacy/native-grid pooled and per-map/event results, valid-pixel counts and differences. The legacy table remains available. Name the native-grid result by its actual protocol and report it irrespective of whether rankings improve or worsen; never select the headline protocol by which model wins. A geometry/area-weight sensitivity need not identify each effect's isolated causal contribution.

This is inference only. Do not retrain students at another resolution or redefine the WorldFloods split.

### C. STURM: target only an actual unresolved validity issue

Reuse the existing 128/224/512 evaluation-scale results. Record the executed inverse transform, clipping, nonfinite replacement and retained chips accurately. The current paper's description of no clipping/NaN-chip removal disagrees with the code.

Use existing audit data to determine whether nonfinite input pixels were counted as valid and how many scored pixels actually change under clipping. If the discrepancy is only a documented preprocessing choice with consistently valid scoring, it needs a methods correction, not a new training programme. If invalid input locations entered scores, add one explicitly versioned evaluation-only mask correction on affected chips, applied to every comparator. Reuse unaffected counts. If a clipping sensitivity is necessary to check a central claim, predeclare the executed versus unclipped path on the same finite-input chip set and evaluate only affected inputs; do not select a transform using scores. No new resolution/normalization sweep.

## 6. Dry-scene evidence with an explicit reference gate

The six S11 dry test chips are useful diagnostics, but are too narrow to establish unattended dry-scene reliability, and their location/event groups overlap the training/validation groups. Reuse those counts; do not call them a new independent-event benchmark.

Do not repeat an unrestricted search already completed by prompt2. Check whether the current project now contains a separately verified negative set or new ground-truth artifacts. Existing reference-labelled full scenes may support additional **dry-chip** evaluation, provided water absence is established from the original all-water labels on the valid extent, not by ignoring permanent-water pixels or selecting model-empty predictions. Label a dry chip as a chip, not a fully dry acquisition or operational landscape sample. Check known training/selection/event overlaps and state residual uncertainty.

If no suitable independently referenced negatives exist, create one bounded verification package from existing imagery only:

- Select at most **60 candidate tiles**, before looking at any tested model's predictions, using a fixed seed and location/event metadata. Prefer distinct locations outside known training/validation/selection coverage and spread across the available landscapes. Record the sampling frame and exclusions; 60 is a workload cap, not a claim of statistical adequacy.
- Include true-colour and suitable infrared views, geolocation/acquisition metadata and any independent reference evidence. Withhold teacher/student/MNDWI predictions from the reference-review view. Do not call uncertain/cloudy imagery dry or infer no water from a teacher-empty mask.
- Reuse an existing annotation/inspection format for `verified_dry`, `water_present`, `unassessable`, valid-region information, reference source and reviewer. Automated metadata checks may reject unusable cases but must not invent ground truth. If valid-region masks are unavailable, accept only fully assessable tiles; do not mark hidden pixels as land.
- Freeze the verified set after reference review and before model scoring. Schedule the existing dry evaluator behind a real reference-file gate; it may start automatically when verified annotations are supplied. Use the teacher, MNDWI and primary full-budget matched EfficientViT controls, all four available seeds, with existing fixed 0/0.1%/1% diagnostic alarm criteria. No threshold tuning or negative-scene training.
- If independent verification is unavailable, mark the task `needs-reference-verification`, keep other jobs running, and state that the operational claim must remain restricted. Do not replace this gap with more synthetic haze or S1S2Water runs.

## 7. Close source and writing gaps without inventing experiments

### MNDWI and the RSE paper

The report reproduced canonical S11 MNDWI water IoU 65.0336%, precision 67.5402% and recall 94.6013%. All 446 chips gave 59.3781% pooled water IoU; per-chip means were lower again. Raw-radiometry and spatial checks did not produce a material explanation for the RSE 74.39% value. The initial split/aggregation hypothesis is therefore not demonstrated by the tested variants.

Stop the completed MNDWI diagnostic loop. Retain the unknown official grouping, masks and input preprocessing. Do not tune the index to match the published number, call our method inferior from that cross-protocol comparison, or score our teacher/students on all 446 as a new holdout. The official `kipoju/udl4fl` release at `8f56cc7970ac9e561bfcd04a9ae0aa8e58c1c709` lacked the benchmark MNDWI evaluator. The RSE paper provides a useful comparison of evaluation design, not proof of a matched performance ranking. Contacting authors requires separate explicit authorization; do not send messages.

### Protocol history and manuscript corrections

Consolidate the existing data-role records and bounded overlap checks. Seek missing GDACS footprints or acquisition metadata only in existing manifests/export requests/georeferenced files; do not fabricate native footprints from unverified nominal sizes. Keep same-location S11 dependence and shared STURM/WorldFloods events explicit.

Check whether an existing author statement already resolves historical benchmark use. Otherwise record one precise question: did STURM/WorldFloods scores influence architecture, filtering, loss, checkpoint, calibration or other choices before the final evaluations? Neither a current code scan nor new matched runs can answer that historical question. Preserve the conflict with the manuscript sentence; do not assert either contamination or an untouched history without evidence. Continue independent work.

Prepare evidence-backed writing actions for the real mismatches: true CE versus focal; clipping; KL masks; branch temperatures and mean aggregation; 11,368 rather than 488 EfficientViT auxiliary parameters; maximum-class confidence versus water-probability descriptors; STURM preprocessing; WorldFloods border geometry; crop removal of 73.6% of native pooled training-water pixels; and QAT weights adapting to fixed scale buffers. Do not redesign these components just to match old prose. Reused-component novelty, limited OFA gains, label semantics and geometry confounds mainly require accurate interpretation and presentation after the controlled results are available.

### Deployment provenance

Search only the project's existing export/evaluation/log locations and explicitly mounted original-artifact locations once for the four device seeds and timing/power identities. Recover checkpoint→ONNX→engine hashes, evaluator/masks, actual hardware and precision metadata where possible. Distinguish reported summary CSVs from raw-verified measurements and from the shipped engine's separate build timing. Verify power-rail names; combined CPU/GPU/CV rail measurements are not automatically GPU-only power.

If the original artifacts are missing, do not manufacture four-seed device evidence. If the **already authorized target Jetson and its existing execution workflow are actually available**, schedule bounded inference/timing remeasurement of existing matching float/FP16/mixed artifacts there, with frozen settings and clear new-measurement identities. No new QAT training or claim of original reproduction; no network/device discovery or new remote access setup. If matching artifacts or target access are unavailable, prepare the existing commands and mark the task blocked on those exact inputs. RTX inference cannot verify a Xavier-specific engine or latency. The later paper must limit device claims to the evidence that can be supported.

## 8. Integrate into the existing automatic schedule

The report locates the current worker at `queue_dual_gpu/revision_queue.py` with `manifest.json` and `manifest.state.json` under the approved cohort directory. Verify the active pointer and exact command syntax rather than copying PDF-wrapped command lines. Reuse this scheduler and its status/pause/resume/graceful-stop controls.

Preserve its actually authorized GPU configuration. The report records prior approval for two GPUs; verify the local receipt/configuration and GPU identities. Do not assume that both devices are RTX 4090s. Use at most one heavy job per approved GPU, account for foreign processes and maintain CPU/I/O headroom. If device types differ, keep matched pairs on the same hardware class where possible and record assignment; do not make hardware assignment depend on results. No global power/clock changes or new DDP framework.

| Priority | Automatically scheduled work | Gate |
|---|---|---|
| Q0 | Refresh state, deduplicate, classify report findings, inspect actual mask scope, freeze configurations and estimate cost | CPU work; preserve active jobs |
| Q1 | Finish missing canonical teacher/primary outputs and verify necessary baseline/evaluator identities | Existing dependencies; no duplicate inference |
| Q2 | First-seed functional checks, then T1/T2 and only necessary T4 partners, round-robin across valid comparisons | Valid data/loss/architecture gate; OFA partners compatible or explicitly queued |
| Q3 | T3 fixed N=100 KD/OFA pairs across all four seeds | Frozen trustworthy N=100 manifest and valid loss path |
| Q4 | Missing clean readouts as checkpoints finish; native-grid WorldFloods border evaluation | Matching checkpoint/input/mask fingerprints |
| Q5 | Existing matched robustness jobs plus required new primary EfficientViT readouts | Clean evaluator correct; reuse completed fixed-rule and compatible model cells |
| Q6 | Independent-reference dry evaluation and available target-device remeasurement | Their separate explicit reference/artifact/device gates |
| Q7 | Final paired summaries, reviewer evidence status and bounded operational handoff | All runnable work complete; blocked work identified rather than hidden |

This ordering is dependency-driven, not a rule to keep a free GPU idle. Allow short clean evaluations between training jobs, prioritize complete method pairs, and keep unaffected original jobs valid. Do not change the meaning of a pending legacy run in place; add a distinct variant or explicitly mark a redundant pending cell with its reason, preserving its identity and history.

Use the existing detached worker; editor closure must not terminate eligible jobs. Verify a live worker and durable state. Reuse the already proven parent-process sentinel rather than rerunning it gratuitously. Do not promise survival of reboot, sleep or power-off. Do not install system services or alter machine sleep policies.

Before launch, estimate costs from current measured durations and show original versus additional remaining GPU-hours. The report's old 6.03 GPU-hour estimate is not a current ETA. Track code/config/data/checkpoint/environment hashes, GPU, command, timestamps, exit status and expected outputs. Use existing locks and atomic queue updates. One recorded retry for a transient failure is enough; repeat OOM, invalid metrics or a failed correctness gate blocks dependents. Never change batch size, masking, clipping or seeds to get through a failing run silently.

Reference/device gates must not require an active chat to launch already specified eligible work. They must also not hallucinate approvals, annotations or hardware availability. A drained queue with unresolved human/source gates ends in a clear blocked state rather than an endless search or new experiment generation.

## 9. Deliverables and completion criteria

Use existing artifact directories and ledgers, not another dashboard or report generator. Produce:

1. A deduplicated finite job matrix with reused/completed/running/pending/blocked states and the reason for every conditional replacement.
2. Matched architecture/objective tables by seed and paired OFA-minus-KD differences; one N=100 paired table; a clearly named focal-only-versus-weighted-KD comparison.
3. Consistent clean/robustness confusion counts and error summaries; paired legacy/native-grid WorldFloods results; only the necessary STURM validity sensitivity.
4. A verified dry-scene result or a bounded independent-reference package with an honest blocker; no claim that six related S11 chips establish deployment reliability.
5. Recovered deployment provenance or precise missing-artifact limits, plus actual target measurements only if the separate gate was satisfied.
6. A concise reviewer-status handoff separating evidence-ready items, writing corrections, pending computations and author/reference/device inputs. Include exact live status/pause/resume/stop commands and updated remaining time.

Success does not require OFA, LightFormer or the student to beat a comparator. It requires that each retained comparison means what its label says, each central conclusion has suitable evidence, and every remaining limitation is explicit. Once this finite programme completes, stop adding experiments and hand the evidence to the manuscript/response stage.
