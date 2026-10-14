# Reviewer-only addendum to prompt1.md

Read and apply this file **after `prompt1.md`** in the existing research project. This addendum closes gaps in reviewer evidence; it does not start another research programme. Read the available manuscript, supplement and review as evidence, together with the actual code and experiment artifacts. The supplied files may be named `main(1).pdf`, `supplement(1).pdf` and `review(1).docx`; locate their actual project equivalents rather than inventing paths.

**This addendum authorizes no additional training, fine-tuning, QAT or hyperparameter searches.** Training already authorized by prompt1 may continue under its existing gates. All new tasks below use code/configuration inspection, saved outputs, small correctness checks, or narrowly specified inference on existing checkpoints. No additional architectures, teachers, datasets, budgets or seeds enter the training queue through this file.

Inherit prompt1's branch, worktree protection, immutable job inputs, one-GPU-job limit, detached queue, provenance and no-manuscript-editing rules. Reuse the same tracker and evaluator. Do the work, rather than returning only a plan. Continue independent tasks when one task is blocked. A completed compatible artifact satisfies its task: do not rerun it merely because this file mentions it. An instruction in prompt1 is not evidence that the work has actually finished.

## 1. Scope, deduplication and evaluation decisions

First map each task below to the existing prompt1 task, run IDs and artifact fingerprints. Use statuses such as `verified-complete`, `needs-analysis-only`, `needs-inference`, `blocked` and `excluded-from-revision`. Extend existing records rather than creating a second tracking system.

| Already assigned in prompt1 | What this addendum adds |
|---|---|
| §§1–3: inventory, recipe recovery, baseline reproduction | Trace paper claims to executable artifacts; verify actual split membership and metric semantics in greater depth |
| §4: bounded architecture/objective matrix | Audit native auxiliary losses and the fairness of the adapted LightFormer comparison |
| §5: choice/calibration provenance | Trace all data roles and concrete overlap mechanisms, while retaining uncertainty about historical benchmark use |
| §6: independent dry-scene evaluation | Require interpretable error counts; retain its reference-quality gate |
| §7: clean and robustness comparisons | Obtain missing precision/recall from the same predictions; investigate the S11 MNDWI discrepancy |
| §8: persistent queue | Add only missing inference dependencies to that queue; no second worker or expanded training matrix |

### Decision: the following evaluations are sufficient for this addendum

| Evaluation | Exact models/runs | Data and readout | New work allowed and stopping rule |
|---|---|---|---|
| Primary clean error analysis | Frozen teacher; fixed MNDWI; EfficientViT hard-only, Logit-KD and OFA at N=2,500, all four original seeds where valid checkpoints exist | Canonical S11 90-chip test split, STURM and WorldFloods protocols; confusion counts, water IoU, mIoU, precision and recall | Reaggregate saved counts or predictions first. Infer only missing outputs, once per checkpoint/data/protocol combination. Stop when the principal comparisons have matching counts. |
| Architecture controls already in prompt1 | Completed compatible MobileNetV3-Small and LightFormer-ResNet18 hard-only/Logit-KD/OFA runs at N=2,500 | The same clean evaluator and readout | Export counts during their already planned evaluation. Do not retrain, add models or wait for every pending run before completing independent audits. |
| S11 MNDWI discrepancy | Fixed spectral rule only; no training seeds | Our 90-chip test split; additionally all 446 manually labelled S11 chips if available through existing data/readers | Run the bounded protocol comparison in §3. This is a diagnostic, not a new held-out learned-model benchmark. |
| Dry scenes and spectral robustness | Exactly the models and severities already assigned in prompt1 §§6–7 | Same independent references, masks and frozen decisions | Reuse that work. Add missing counts to its outputs; do not create another negative set, perturbation sweep or threshold search. |
| Budget, geometry and evaluation-scale evidence | Existing N=252 S11-versus-GDACS controls, existing cropped-S11 controls, existing OFA/Logit-KD size sweep, and existing STURM 128/224/512 results | Existing artifacts supporting R2.2 and R2.3 | Analysis and provenance recovery only by default. Re-score an existing checkpoint only if a specific missing quantity is necessary to check a central published claim; name the claim and checkpoint before launch. No new size sweep or training. |
| Deployment claims | Existing float/export/engine artifacts underlying Table III, where their actual execution path is available | Existing accuracy, precision mode and timing evidence | Inspect/reaggregate first. Do not substitute RTX float inference for Jetson mixed-precision inference or rebuild unavailable engines. |

For learned models, use the frozen validation-selected checkpoint for each seed, not every epoch or the best test checkpoint. Keep archived and current-environment cohorts distinct. Missing seeds remain missing; do not fill them by training under this addendum. Never omit a valid seed because it performs poorly.

**Do not open a new S1S2Water evaluation by default.** The reviewers ask about protocol validity, controlled distillation, dry-scene errors and practical advantages over MNDWI; adding a permanent-water benchmark does not directly resolve those gaps. First recover any existing project S1S2Water result to identify its metric, split and provenance. The remembered “over 98%” is unverified and must not be treated as established IoU saturation. For context, the RSE paper's Table 2 reports 94.86% water IoU for its S2-only model; values above 98% occur in its recall column. Those are their results, not ours. High in-domain performance there also does not predict our model's transfer performance. Reuse an existing relevant result if available, but do not acquire, adapt to or train on this dataset for this revision.

Do not run the RSE neural models merely to populate another comparison table. Inspect their public code to understand the protocol. Prompt1 already supplies the additional architecture comparison requested by the reviewers.

### Metrics needed to answer the reviewers

IoU remains the primary segmentation score. Add **precision and recall at the existing fixed decision rule**, because they distinguish excess predicted water from missed water. This directly supports R1.6 and R2.5 and requires no retraining. No new PR-curve sweep, threshold optimization or redundant metric collection is required.

For each valid image/map, export integer `TP, FP, FN, TN`, valid/reference-water pixel counts, dataset/split, image/map/event IDs, checkpoint/seed, protocol and mask identity. Derive:

- Water IoU = `TP / (TP + FP + FN)`.
- Precision = `TP / (TP + FP)` — how much predicted water is correct.
- Recall = `TP / (TP + FN)` — how much reference water is found.
- Dry-scene false-positive pixel rate = `FP / (FP + TN)`.

Neither precision nor recall can be uniquely recovered from an IoU value. Compute them from predictions/counts, not by guessing or converting a published mIoU. Separate water-class IoU from two-class mIoU. “Micro” means summing counts before division; “macro” means calculating a score for each stated unit and averaging. Do not silently average chips and call them flood events. Use the same aggregation when comparing methods.

For zero denominators, preserve and document the existing convention, expose undefined values and contributing sample counts, and do not silently turn undefined recall or empty-reference IoU into perfect performance. Use the already specified dry-scene area/scene-alarm diagnostics. Group uncertainty by independent events where possible and keep training-seed variability separate, following prompt1. No new significance fishing or claim of equivalence from a nonsignificant result.

## 2. Implementation audit without retraining

Create or extend a compact **claim-to-evidence ledger**, not a narrative report. Each record needs: reviewer ID; paper/supplement location and claim; relevant code/configuration revision; run/data/checkpoint evidence; finding; affected results; severity; smallest remedy; and status. Distinguish `verified`, `inconsistent`, `user-reported`, `inferred` and `unknown`. A check of current code does not verify an older checkpoint unless its executed revision and configuration are recoverable.

Complete the relevant checks below by reusing prompt1's results. Inspect all implementation-dependent statements needed for the reviewer concerns, including footnotes and supplement qualifications. Do not expand this into a general software-quality review.

### A. Data lineage, splitting and possible leakage — R1.3, R2.2

1. **Membership, not just counts.** Resolve the actual S11 manifests for 252 training, 89 validation, 90 test and 15 separately reserved Bolivia chips. Verify IDs, duplicates, reference-mask identities and any exceptional exclusions. Confirm loader/sampler arguments use the intended manifests in every experiment, including supervised controls, cached pseudo-label generation and calibration. Do not assume a folder called `test` was the data actually evaluated.
2. **All paths into learning and selection.** Trace data used for student gradients, teacher task fine-tuning/checkpoint selection, normalization statistics, pseudo-label/filter/descriptor construction, subset selection, early stopping, architecture/loss/threshold choices, calibration and QAT. Separate image use from label use. In particular, record GLISTER's S11-validation use already described in the supplement. Validation use is not automatically leakage; test-guided fitting or selection is a different issue. Verify that normalization or descriptor scaling was not fitted on a concatenation containing evaluation samples.
3. **Duplicates and shared acquisitions.** Check stable scene IDs and file/content identities where available, then geographic footprints and acquisition dates. Inspect intersections between training, validation, calibration and test sources, including GDACS pre-event/event pairs and overlapping tiles from the same acquisition. Group related samples by event/location; different filenames or crops do not establish independence. Use available spatial indexing and metadata rather than expensive indiscriminate pairwise scans.
4. **Geographic dependence versus exact leakage.** Extend the existing centroid proximity screen to footprint intersection where geometry is available. Report acquisition-time separation and overlap area; do not treat a 5 km/2-day or 25 km/30-day screen as proof of independence. Separate exact duplicate imagery, spatially overlapping near-date observations, and shared events with disjoint footprints. Where geometry is missing, report the audit's coverage and limitation. An official chip split can have spatial dependence without a local implementation bug; qualify the claim accordingly rather than inventing a new split and retraining.
5. **External-set dependence.** Reuse the known STURM/WorldFloods shared-activation analysis and sensitivity outputs. They cannot be counted as wholly independent replications when events overlap. Confirm that external data did not enter filters, validation selection, calibration or train-mode BatchNorm/normalization updates. Evaluation must not update model state or apply test-time fitting implicitly.
6. **Teacher provenance.** Check the released teacher's stated task-training/selection data against our S11 split. Distinguish task-label overlap from possible overlap in large-scale unlabelled foundation pretraining. Do not claim an exhaustive pretraining-contamination audit if the source data are unavailable; this does not authorize a second teacher or new fine-tuning.
7. **Historical benchmark use.** Build on prompt1 §5 to reconcile the user's untouched-test account with the manuscript's “benchmark-guided development” statement. Code cannot establish that nobody inspected an external score. Preserve evidence and unknowns; ask one precise factual question only if the conflict cannot be resolved. Four seeds, corrected p-values and a forward-frozen protocol do not retrospectively erase test-guided development.

Output both the specific intersections found and the limits of the search. Use “no overlap found by these checks,” never “100% leakage-free.” If a test set influenced method development, identify it as such for later writing. Do not silently relabel it untouched, delete inconvenient cases, or manufacture a fresh test claim.

### B. Preprocessing, reference semantics and evaluator — R1.3, R1.5, R2.2, R2.5

- Trace band names through each raw reader to the actual tensor positions; units, processing level, scale/offset, missing bands and fixed normalization must agree with the executed configuration. Check STURM's inverse transform and B8 substitution, and the WorldFloods adapter, using prompt1's existing audit. A distribution-based suspicion about STURM's processing level is not verified product metadata. Applying identical preprocessing to different models does not prove that a domain shift affects them equally.
- Check input, prediction and reference grids separately: physical footprint, pixel size, interpolation, padding, cropping, resampling order and geospatial alignment. Confirm categorical labels/ignore masks retain their intended classes and that no padded pixel enters scoring. Upsampling a 128-pixel chip does not create new physical detail. Reuse the already completed native-grid control; teacher parity at 512 must not become a resolution-independent claim.
- Verify signed labels, nodata, clouds, invalid reflectance and NaNs are handled consistently in losses and metrics. Inspect masks before/after resizing and integer conversion; ignored pixels must not become background. Verify retained/excluded pixel counts for every comparator.
- Resolve surface-water versus event-inundation labels. Audit STURM's positive/ignored class mapping and WorldFloods' exact reference-water product, year, class mapping and resampling. Confirm seasonal-water treatment and which permanent-water categories remain. Do not assume another paper's GSW-derived mask is equivalent to ours. A permanent-water exclusion mask restricts the evaluation; it does not turn a single-image water model into a detector of newly flooded land. Record whether a reference layer is an evaluation mask or a deployed input and whether its timing supports the claimed use case.
- Check class-channel ordering, argmax/probability thresholds, de-tiling without duplicated border counts, valid masks, scene/event grouping, per-seed aggregation and table-generation scripts. Do not average seed probability maps and call that the mean performance of individual runs. Trace displayed scores back to the correct checkpoints and evaluator revisions.

### C. Losses, training controls and ablation interpretation — R1.1, R1.2, R1.4, R2.1–4

Use prompt1 §2 for the already assigned CE/focal, OFA-equation and live-teacher checks. Extend the evidence to establish:

- Every comparison uses its declared hard-label source, soft teacher source, augmentations and valid-pixel loss reduction. Cached hard masks must receive the same geometric transforms as images; live teacher soft targets must correspond to the actual augmented image. Check teacher freezing/eval mode, detached targets, temperature scaling, focal settings and summed/averaged auxiliary weights against the executed code. Correct a mathematical description without silently changing the trained objective.
- Hard-only, Logit-KD and OFA cells have the intended shared initialization/pretraining, six-band adaptation, main decoder, IDs, optimizer/scheduler, updates, augmentation and validation selection. At different dataset sizes, a fixed epoch count changes the number of optimizer steps; include this when interpreting sample efficiency. Reconstruct selection provenance rather than assuming a currently saved `best` checkpoint used the stated rule.
- **LightFormer has native auxiliary supervision in its published recipe.** Record whether it is retained or disabled in our adaptation, its exact loss/weight, and whether the policy is identical across its hard-only, Logit-KD and OFA cells. Avoid adding native auxiliary supervision only to the OFA cell and then attributing the whole difference to OFA. Freeze the policy before pending runs, following the already authorized common-objective design. Completed incompatible variants stay labelled as such. Our common-recipe six-band adaptation is not a reproduction of LightFormer's published best-tuned results; count the adapted full model, not only its decoder.
- Link existing filtering, selector, hard/soft/OFA, activation and QAT comparisons in one component ledger. Identify the two runs forming each contrast and every changed factor. Different augmentation recipes, pretraining, input geometry, N, epochs or evaluation paths prevent a one-factor causal interpretation. Do not chain unrelated ablations into an invented additive contribution total.
- Check the existing pool lineage from raw to filtered to selected IDs, including event/pre-event membership and nested budgets. Verify that reported water fractions use the intended valid-pixel denominator and that descriptors/selector scores match their names. Spatial variation of water probability is not automatically model uncertainty; mean water probability is not automatically prediction confidence. Link any mismatch to the filtering/selection interpretation, without regenerating subsets or repeating completed selector experiments.
- Reuse the existing cropped-S11 experiment to address R2.2. Verify its implementation and lost-water/dry-chip counts. Cropping changes scene content as well as field of view, so it does not isolate label quality or geography perfectly. Do not commission a replacement geometry experiment under this addendum.
- Recover all existing OFA-versus-Logit-KD budget results, seed spreads and paired comparisons. Preserve the limited/intermediate-budget gain and the lack of consistent N=2,500 superiority. Do not select only a favourable budget, generate a new sweep, or treat a null effect as an implementation failure. Separate the practical compression contribution from a claim that OFA is a new or universally better algorithm.

### D. Deployment and practical claims — R1.4, R1.6, R2.5 and R2's final comment

Audit the existing chain only: original activation model → ReLU6 training → QAT → float export → mixed INT8/FP16 engine. Recover checkpoint ancestry, trainable/frozen parameters, calibration inputs and label use, export/build flags, layer precisions, masks and evaluation paths. Check whether activation comparisons also change initialization or training conditions. Freezing BatchNorm does not by itself mean only quantization scales are trained; inspect actual optimizer parameters.

Do not attach the undeployed model's best accuracy to a different deployed engine. Keep per-seed accuracy separate from a representative engine's timing. Verify full-model parameter counts, engine-file size versus runtime memory, kernel-only latency versus end-to-end latency, and device/software/precision/batch/input settings. The reported table has FP16 faster than the mixed engine; do not claim mixed INT8 is the fastest. Peak power alone is not energy per image, and no measured teacher/student device speedup exists when the teacher was not benchmarked there. No new device benchmarking is required.

For dry imagery, distinguish independent false alarms from disagreement with a teacher-empty pseudo-label. Reuse prompt1 §6; if independently verified negatives are missing, mark the operational-generalization concern unresolved. Do not convert teacher agreement, pre-event imagery or permanent-water predictions into proof of correct flood alarms.

### Handling findings

An incorrect evaluator can be repaired narrowly and re-run on affected existing checkpoints. Preserve original outputs, version the corrected protocol and identify affected claims. An error in training, split use or historical selection cannot generally be repaired by rescoring. Mark the affected evidence and block dependent new launches; **do not initiate replacement training outside prompt1's authorized matrix** or claim a clean result from an unchanged contaminated checkpoint. Continue unblocked audits. Never modify code underneath a running job.

## 3. Diagnose the Sen1Floods11 MNDWI discrepancy

This task supports R2.5 and the protocol concern, not a new leaderboard claim.

### Facts already established; verify the local implementation against them

| Item | Our manuscript | Portalés-Julià et al., RSE 2025 |
|---|---|---|
| S11 MNDWI water IoU | 65.0%; separate two-class mIoU 78.9% | 74.39% all-water IoU in Table 3 |
| S11 population | Canonical 90-chip test split | All 446 manually labelled chips, treated as external data for their models |
| Aggregation | Pixel-pooled for the quoted 65.0% | Described as averaging scores per flood map; the text also uses “event,” so verify the exact grouping if recoverable |
| Spectral rule | `(B3 - B11) / (B3 + B11) > 0` | Zero-threshold MNDWI |
| Learned optical inputs | B2, B3, B4, B8A, B11, B12 | B2, B3, B4, B8, B11, B12 |

The 9.39 percentage-point difference is **not a measured implementation deficit on matched data**. Split and aggregation differences are plausible contributors, not a proven explanation. B8 versus B8A cannot directly explain MNDWI because that index uses neither band. Comparing our 78.9% mIoU with their 74.39% water IoU is also invalid.

Official source: <https://github.com/kipoju/udl4fl>. The inspected main revision on 2026-10-05 was `8f56cc7970ac9e561bfcd04a9ae0aa8e58c1c709`. Pin and record the revision used locally; do not assume an old README hyperlink identifies current HEAD.

Inspection of that revision found inference/model-loading code and a demonstration notebook, **no MNDWI implementation or benchmark-table evaluator**. In `udl4fl/worldfloods_model.py`, training and validation methods are stubs; `predict()` uses checkpoint-supplied normalization and a tiling path whose defaults are not the paper's stated 256-pixel/32-pixel-overlap evaluation recipe. `udl4fl/data_configs.py` and `notebooks/run_in_gee_image.ipynb` establish the six-band optical input ordering. Trace active configuration rather than trusting unused metadata. The released `sm_unet_s2` is described as trained on WorldFloods plus S1S2Water; it must not be identified with every single-modality row in the paper.

Therefore the repository alone does not establish the exact cloud/nodata, denominator, map grouping or GSW-mask implementation behind 74.39%. Check any directly supplied official evaluation artifacts if available; otherwise retain those unknowns. A paper-specification reconstruction must be labelled as such, not as an exact official-code reproduction. Do not search indefinitely or download model weights for this index-only audit.

### Bounded diagnostic sequence

1. **Recover our result.** Find the exact MNDWI function, inputs, checkpoint-independent configuration, 90 IDs, masks and aggregation that produced 0.650 water IoU. Reuse archived predictions/counts. If unavailable, run the current frozen canonical evaluator and distinguish current reproduction from verified historical reproduction.
2. **Check index arithmetic on a few diagnostic arrays and real valid pixels.** Use floating-point green B3 and SWIR1 B11 before subtraction/division, with verified tensor positions. Check unsigned underflow, swapped channels, threshold direction, zero/negative/near-zero denominators, epsilon, nonfinite values and ignore masks. MNDWI must use recovered radiometric values, not per-band z-scores. A shared positive multiplicative scale cancels; different per-band standardization does not. A shared additive offset does not change the numerator and is not automatically an explanation for a changed zero-threshold decision when the denominator stays positive. Diagnose actual behaviour rather than assuming any scale/offset difference explains the score.
3. **Check spatial operations.** Compare raw-band-to-index-to-mask order with resampling-band-to-index order, alignment and mask interpolation. Thresholding before versus after interpolation can change edge pixels. A per-pixel spectral rule has no neural receptive-field benefit from overlapping tiles; tiling should not change valid interior decisions when alignment and counting agree. Reuse prompt1's common-gain check; do not implement it twice.
4. **Separate population from aggregation.** Compute MNDWI once per available labelled S11 chip with the verified fixed rule and save counts. From those same counts, produce a table for (a) the canonical 90 chips and (b) all 446 chips, each with pooled water IoU, per-chip mean water IoU and, where a verified mapping exists, per-map/event means. State exactly how counts are combined within an event. These are jointly specified diagnostic views, not competing scores from which to choose the most favourable. If the RSE grouping cannot be reconstructed, mark that comparison approximate rather than relabelling per-chip means.
5. **Limit protocol variants.** Compare our verified protocol with a paper-consistent variant only for identified differences in available input products, validity masks, resampling or aggregation. Predeclare each variant and its reason before examining its score. Start from the table above; do not generate every possible preprocessing combination. Change one documented factor at a time for diagnostics, but acknowledge interactions rather than claiming the deltas uniquely explain the entire gap. Do not tune the threshold, ignore difficult pixels, substitute a different index or modify masks to reach 74.39%.
6. **Protect the holdout distinction.** All 446 chips include our training/validation material and material used by the task teacher. The fixed-rule result on that population is useful for protocol comparison. Do not evaluate our trained teacher/students on all 446 and present the score as independent test performance. Retain the canonical learned-model comparison on the valid held-out split.
7. **Stop with evidence.** Classify the gap as explained, partly explained or unresolved, with measured changes and remaining unknowns. An implementation bug requires concrete evidence; a remaining mismatch is not permission to keep adjusting the rule. Extend to WorldFloods only if the same discovered bug affects its already assigned evaluation or a central reviewer-facing comparison requires a missing matched-mask readout. Otherwise reuse its existing permanent-water sensitivity and aggregation results.

Required output: an index-protocol comparison table, paired counts/pixel-disagreement summaries for the necessary variants, exact source/input identities and a short structured root-cause finding. A missing official evaluator or unavailable raw product is a valid explicit limitation, not a reason to assert exact reproduction.

## 4. Novelty: use existing evidence; do not launch few-shot/XAI work

R1.1 and R2.1 ask us to clarify the contribution beyond established components. They do not require a new distillation algorithm, multimodal training, multiclass benchmark or few-shot adaptation study. For this revision, **do not launch the proposed one-/few-shot + multimodal + XAI experiments**. They introduce new data, controls and claims, and would violate this addendum's no-new-training scope.

Instead, inspect only existing relevant logs/artifacts for feature matching, gradient modifications and other attempted extensions. Preserve verified null results and distinguish them from the user's recollection or unrecorded trials. Use them to delimit what was tested, not to prove that teacher-derived auxiliary methods cannot work. Reuse the supplement's existing analyses; no new XAI/CKA/gradient experiment is required.

Audit the information argument carefully. A binary hard class is one bit, but a binary soft target has one continuous probability degree of freedom per pixel. A structured mask, teacher features and spatial relationships are not merely one bit for the whole image. The supplement's statement that a deterministic signal adds no independent label source conditional on the fixed teacher and inputs does **not** prove a student accuracy ceiling, identical optimization optima, or futility of XAI guidance. More classes might change the distillation problem; that remains an untested hypothesis here.

Existing small-N training is not automatically few-shot adaptation. A credible future adaptation study would require a clearly defined new event/domain, a small labelled support set disjoint from its evaluation/query set, plain fine-tuning and distillation controls, multiple independent episodes and actual time-to-accuracy measurements. New modalities require temporal/spatial alignment and missing-modality controls. This is a separate future study, not a task to queue now or a contribution to claim in the revision.

For the current novelty response, recover evidence for the limited contribution actually demonstrated: the foundation-model-to-compact-model transfer and measured deployment, with controlled applicability and limitations. Identify which components are reused and which adaptations are supported by code/evidence. Do not claim invention of OFA, filtering, pseudo-labelling or QAT, and do not promise that additional evaluations will guarantee acceptance.

## 5. Priority and remaining reviewer work

The table below orders the work; it does not replace prompt1's queue phases. Audit dependencies must clear before dependent new jobs start. These priorities refer to scientific necessity, not whether a desired result is likely.

| Priority | Concern | How to handle it within the authorized work | Completion condition |
|---|---|---|---|
| Highest: validity | R1.3 protocol/selection/leakage; implementation errors affecting any comparison | Complete §2A–B and reuse prompt1's provenance audit; repair evaluation bugs only where evidenced | Data-role records and specific checks verified, or precisely bounded unknowns/affected claims recorded |
| High: practical evidence | R1.6 dry-scene errors; R2.5 MNDWI competitiveness | Complete existing dry/robustness work plus §3 and fixed-rule precision/recall | Matched evidence available, or independent-reference/protocol gap explicitly left unresolved |
| High: controlled comparisons | R1.2 and R2.4 architecture controls; R1.4 component contributions | Finish prompt1's existing bounded matrix; audit loss/pretraining comparability and link existing ablations | Required cells valid or concretely blocked; each contribution has an identifiable comparison |
| High: honest interpretation | R2.2 geometry/geography confounds; R2.3 limited OFA benefit | Reuse geometry, native-grid and size-sweep evidence; retain nulls and residual confounds | Supported claims separated from causal or universal claims not established by the experiments |
| Required at writing stage | R1.1/R2.1 novelty; R1.5 terminology; R2 deployment qualification; editor's suggested Discussion | Record the evidence-backed corrections and limitations in the same ledger | Evidence ready for a later manuscript/response pass; this is not marked resolved merely because experiments are queued |
| Outside this revision | New S1S2Water programme; new teacher/SAR/multiclass/few-shot/XAI methods; new hardware | Do not start; retain only already available relevant evidence | No additional task or hidden queue expansion |

Prompt1's no-manuscript/no-reviewer-response constraint remains in force. The later writing pass must still consolidate component contributions, state the actual data roles, clarify surface-water terminology and novelty, qualify hardware/precision claims, and develop the Discussion and point-by-point response. Record those pending actions with reviewer IDs and manuscript locations; do not draft them now. A verified limitation can justify narrowing a claim, but do not label a concern resolved if evidence necessary for that claim is missing.

## 6. Deliverables and stopping criteria

Reuse existing formats and paths. Produce only the additions needed beyond prompt1:

1. Extended reviewer/claim-to-evidence ledger, including unresolved data-history questions and affected results.
2. Missing confusion-count exports and consistent metric summaries for the finite evaluation matrix in §1.
3. Bounded S11 MNDWI protocol comparison and supported root-cause status.
4. Updated existing queue/tracker entries for genuinely missing inference, with no added training configuration.
5. A concise operational handoff: what was reused, what was checked, errors found, evaluations completed/pending/blocked, and what remains for the later writing stage.

Stop optional evaluation once the relevant concern has sufficient evidence or an explicit irreducible blocker. Do not keep adding runs to obtain significance, force OFA or the student to win, or reproduce a paper number by tuning. State the limits of the findings rather than claiming 100% experimental certainty.

## Source anchors

- `prompt1.md`: the attached base instructions; preserve them and deduplicate against their actual completion state.
- Supplied manuscript, supplement and reviewer letter for JSTARS-2026-05386, *Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation*. Numeric anchors above are reported values, not newly reproduced results.
- Portalés-Julià et al., *Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets*, RSE 328 (2025), 114882: <https://doi.org/10.1016/j.rse.2025.114882>. Use the supplied PDF, especially dataset/preprocessing/evaluation sections and Tables 2–3.
- Official RSE repository inspected at a fixed revision: <https://github.com/kipoju/udl4fl/tree/8f56cc7970ac9e561bfcd04a9ae0aa8e58c1c709>. Relevant files: `udl4fl/worldfloods_model.py`, `udl4fl/data_configs.py`, `udl4fl/model_setup.py`, `udl4fl/utils.py`, and `notebooks/run_in_gee_image.ipynb`.
- Published LightFormer paper: <https://doi.org/10.1016/j.jag.2026.105331>; official code: <https://github.com/Chen-XiaoLv/LightFormer>. Use the supplied PDF's loss/implementation section when checking native auxiliary supervision. Its published cross-dataset scores are not direct rankings against our S11/STURM/WorldFloods results.
