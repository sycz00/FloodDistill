# Reviewer Revision — Pipeline Attribution, Dataset Confounding, Dry Scenes, and MNDWI

You are acting as a rigorous remote-sensing researcher, experimental scientist, and senior ML developer.

Your task is to resolve the remaining reviewer concerns concerning:

1. **R1.4 — What is each pipeline step worth?**
2. **Dataset and label-source confounding**
3. **Dry-scene exclusion and false alarms**
4. **Teacher-to-GDACS domain and preprocessing differences**
5. **Whether learned models genuinely outperform MNDWI**

Work directly within the existing research branch and project repository. Read the original paper, reviewer comments, revision reports (including V1–V10), experiment manifests, current results, protocols, and relevant source code before proposing new work.

Do not assume that the reviewer's criticism is correct or that our current implementation is correct. Investigate both.

## General working principles

- Reuse existing implementations, checkpoints, datasets, evaluation scripts, and experiment infrastructure.
- Do not introduce new pipelines when existing ones can be extended.
- Respect the established experimental recipe, seeds, splits, calibration policies, and checkpoint-selection rules.
- Distinguish results from the original paper, reproduced results, new revision experiments, and exploratory findings.
- Avoid data leakage, test-set tuning, unfair threshold calibration, and incompatible class definitions.
- Do not manufacture measurements or turn inconclusive findings into positive claims.
- Prioritize experiments with direct reviewer relevance and meaningful scientific information gain.
- Schedule approved independent experiments using existing GPU infrastructure without disrupting running jobs.
- Record reproducible commands, configurations, outputs, and experimental provenance.
- **Do not change the official evaluation protocol or launch protocol-dependent experiments before my approval.**

# Task 1 — R1.4: What is each pipeline step worth?

## 1.1 Audit existing evidence

Reconstruct the complete experimental pipeline, including all components that contribute to the final student model.

Identify which components have already been measured through controlled ablations and which remain unmeasured.

For every component, establish:

- Its purpose and claimed benefit.
- Whether an isolated experiment exists.
- Whether an experiment is missing, incomplete, confounded, or already sufficient.
- The appropriate baseline and controlled comparison.
- Required datasets, architectures, metrics, and computational cost.
- Its importance for R1.4 and the paper's novelty claim.

Create a traceability matrix linking reviewer concerns, paper claims, pipeline components, existing evidence, and missing experiments.

**Do not blindly run every possible ablation.** Identify the smallest defensible set of experiments needed to quantify the pipeline's contributions.

Consider component interactions where necessary. One-at-a-time ablations may not adequately establish attribution when components depend on each other.

## 1.2 Activation replacement and learning rate

Investigate the activation replacement experiments conducted during the revisions.

Specifically determine:

- Was the learning rate updated?
- Why was it changed?
- Which learning rates, schedules, optimizers, and training settings were tested?
- Did the change improve convergence, stability, or final performance?
- Were the comparisons controlled and computationally fair?
- Is the activation replacement independently beneficial, or does its apparent benefit depend on different optimization settings?

Inspect actual code, configuration files, logs, and checkpoints rather than relying only on reports.

If optimization remains unresolved, design a small, controlled experiment that separates the effect of activation replacement from the effect of learning-rate tuning.

Do not make an architectural conclusion from an inadequately optimized comparison.

## 1.3 Implementation and scheduling

Implement missing high-priority experiments using the existing infrastructure.

Run protocol-independent, scientifically justified experiments where possible. Queue approved experiments through the existing scheduler.

Produce a prioritized R1.4 experiment matrix with clear stopping criteria.

# Task 2 — Dataset homogeneity and confounded label-source evaluation

This task requires a comprehensive investigation before experiments are launched.

## 2.1 Audit Sen1Floods11, GDACS, WorldFloods, and STURM

Reconstruct exactly how each dataset enters the current training and evaluation pipeline.

Investigate:

- Original dataset provenance and annotation source.
- Label semantics and definitions of water, floodwater, permanent water, background, and invalid pixels.
- Class distributions and class remapping.
- Band selection, channel ordering, scaling, normalization, clipping, and spectral transformations.
- Spatial resolution, reprojection, resampling, alignment, and interpolation.
- Sensor and acquisition differences, including relevant seasonal, geographical, and temporal shifts.
- NoData, cloud, cloud-shadow, snow, vegetation, and other ambiguous regions.
- Image dimensions, tiling, padding, filtering, and scene selection.
- Differences between teacher inputs during fine-tuning and GDACS inputs during pseudo-label generation.
- Potential geographic, event, or scene overlap and leakage.
- Differences in teacher/student/threshold-baseline input handling.

Inspect both the actual data and the code. Use representative numerical distributions, metadata checks, visual inspections, and controlled tests.

Do not confuse differences in imagery with differences in annotation semantics.

## 2.2 Original versus modified evaluation protocols

We currently remove or merge certain classes in WorldFloods and STURM.

Determine exactly what modifications are applied, where they occur, and why they were originally introduced.

Investigate the original dataset protocols and class definitions using the available repository material and, where necessary, authoritative dataset documentation.

Compare three possibilities:

1. Our current harmonized protocol.
2. Each dataset's original protocol.
3. A carefully defined common-label protocol that preserves the largest scientifically defensible shared evaluation target.

Do not assume that the original protocol is automatically better.

For each option, explain:

- What prediction task is being evaluated.
- Which classes are included, excluded, or merged.
- Whether teacher, student, and MNDWI predictions can be evaluated fairly.
- How ambiguous and invalid pixels are handled.
- Whether metrics remain comparable across datasets.
- Whether any evaluation changes would alter the paper's claims.
- Which protocol should be primary and which should be supplementary.

**STOP FOR USER APPROVAL:** Present the exact class-mapping tables, recommended protocol, alternatives, tradeoffs, and affected experiments before implementing protocol changes or running protocol-dependent evaluations.

After approval, evaluate the relevant architectures, threshold-based methods, and teacher-margin methods under the approved protocol.

Use identical valid-pixel masks, clearly defined target labels, and fair calibration rules. Report per-dataset and per-class performance rather than hiding differences through inappropriate aggregation.

# Task 3 — Dry scenes and false alarms

The reviewer argues that fully dry scenes are excluded during GDACS training, leaving false-alarm behavior insufficiently demonstrated.

First verify the premise.

## 3.1 Reconstruct dry-scene handling

Determine whether fully dry scenes are removed:

- From GDACS teacher pseudo-label generation.
- From GDACS student training.
- From validation or checkpoint selection.
- From Sen1Floods11 evaluation.
- From WorldFloods evaluation.
- From STURM evaluation.
- From any other relevant evaluation.

Differentiate between a scene with no water, a scene with no floodwater, and a scene with no valid positive annotations.

These are not necessarily equivalent.

Inspect selection rules, preprocessing, filtering, class remapping, and scene-level metadata.

Quantify the number and fraction of dry scenes retained or excluded at every stage.

## 3.2 Design a reviewer-relevant false-alarm evaluation

Determine whether the current results already demonstrate false-alarm behavior on independently annotated dry scenes.

If not, design a controlled evaluation measuring:

- False-positive water pixels per dry scene.
- Fraction of dry scenes containing at least one false alarm.
- False-positive area or coverage per scene.
- Pixel-level false-positive rate.
- Scene-level false-alarm rate under explicitly defined alarm criteria.
- False alarms across relevant geographical and environmental conditions.

Compare teacher, student architectures, MNDWI, and other relevant threshold-based baselines.

Investigate whether MNDWI predicts substantially more water in genuinely dry scenes, as suggested by existing presentation results.

Verify this using the original data, valid masks, and correct labels. Distinguish genuine false positives from permanent water, ambiguous annotation, or excluded classes.

Use paired scene-level comparisons and uncertainty estimates where appropriate.

## 3.3 Training-selection sensitivity

Assess whether excluding dry GDACS scenes materially biases the learned student.

If existing evidence is insufficient, propose a minimal experiment comparing the current training selection with a defensible dry-scene-inclusive alternative.

Control training budget, data volume, pseudo-label quality, and class imbalance where feasible.

Do not automatically add dry scenes to training: first determine whether they provide reliable supervision and whether the experiment directly addresses the reviewer's concern.

# Task 4 — Teacher-to-GDACS confounding

The teacher was fine-tuned on Sen1Floods11 and subsequently used to generate pseudo-labels for GDACS.

The key question is whether teacher predictions on GDACS are affected by inconsistent preprocessing, dataset shift, or both.

## 4.1 Reconstruct teacher preprocessing

Trace the complete teacher input pipeline during:

1. Sen1Floods11 fine-tuning.
2. Sen1Floods11 validation.
3. GDACS pseudo-label generation.
4. External-dataset inference, where applicable.

Compare every preprocessing operation, its fitted parameters, and its implementation.

Determine whether GDACS currently follows the same preprocessing contract as Sen1Floods11.

Pay special attention to normalization statistics, spectral scaling, missing channels, image resampling, clipping, and invalid-pixel handling.

## 4.2 Determine whether harmonization is justified

Do not automatically force GDACS to resemble Sen1Floods11.

Separate:

- **Implementation inconsistencies:** incorrect or unintended preprocessing differences that should be fixed.
- **Legitimate domain differences:** sensor, geographical, seasonal, spectral, or acquisition differences that must be measured rather than artificially removed.
- **Label-source differences:** differences introduced by teacher pseudo-labels versus human or dataset annotations.

If preprocessing inconsistencies exist, implement a controlled comparison of the current and corrected pipelines.

If the pipelines are already consistent, investigate domain shift and pseudo-label reliability rather than inventing unnecessary preprocessing.

Where independent annotations permit, compare teacher predictions directly with ground truth.

## 4.3 Isolate label-source confounding

Develop a minimal, defensible experiment that distinguishes improvements due to the learned student from effects attributable to:

- Teacher-generated pseudo-labels.
- Training-data selection.
- Different annotation conventions.
- Preprocessing differences.
- Evaluation class mappings.

Consider matched-data comparisons, label-source stratification, teacher–student agreement analysis, and limited controlled training experiments.

Be explicit about which causal questions are identifiable from the available data.

If no clean experiment can isolate a particular effect, state that limitation and propose a proportionate alternative rather than making unsupported causal claims.

# Task 5 — Is the learned model genuinely better than MNDWI?

Existing presentation material and robustness experiments may already contain important evidence.

Locate and verify that evidence before creating new experiments.

We need a scientifically balanced comparison, not simply another aggregate IoU table.

## 5.1 Consolidate existing evidence

Collect relevant results from:

- Original paper.
- Revision experiments.
- Presentation figures.
- Dry-scene analyses.
- Robustness evaluations.
- External-dataset results.
- Existing threshold and teacher-margin comparisons.

Identify which comparisons are already valid and which require rerunning due to protocol differences.

## 5.2 Compare at multiple aggregation levels

Evaluate MNDWI against teacher and students at:

- Pixel level.
- Chip level.
- Full-scene level.
- Dataset level.
- Relevant geographical or event-level groupings.

Measure where each method wins or loses, not only average performance.

Consider distributions of paired performance differences, worst-case behavior, false-positive area, false-negative area, and variability across scenes.

Do not treat chips from the same scene or event as independent observations when estimating uncertainty.

Investigate whether a small number of difficult scenes disproportionately affects aggregate results.

## 5.3 Fair threshold comparison

Determine how the MNDWI threshold and other threshold-based baselines were selected.

Ensure thresholds are calibrated only using permitted calibration data and not optimized on external test labels.

Compare:

- Existing fixed-threshold results.
- Legitimately calibrated thresholds.
- Threshold sensitivity.
- Relevant teacher-margin alternatives.

Clearly distinguish threshold-free neural predictions from predictions requiring operating-point selection.

If a learned model wins only against a poorly chosen MNDWI threshold, that is not sufficient evidence of superiority.

## 5.4 Robustness and dry-scene evidence

Revisit existing robustness experiments and investigate whether the learned models maintain performance more consistently than MNDWI under relevant distribution shifts.

Combine robustness, dry-scene false alarms, and per-scene results into a coherent analysis.

Identify conditions where MNDWI is better, conditions where students are better, and conditions where neither is reliable.

Use representative qualitative examples only when supported by quantitative evidence. Avoid cherry-picked scenes.

## 5.5 Scientific conclusion

Answer explicitly:

- Does the learned student outperform MNDWI on the primary evaluation task?
- Does the advantage persist across datasets and aggregation levels?
- Does it persist under fair threshold calibration?
- Does it reduce false alarms on dry scenes?
- Is it more robust under distribution shift?
- What performance advantage, if any, remains after accounting for compute, model complexity, and annotation-source confounding?
- Which claims can be made confidently in the revised paper, and which must be weakened?

Do not assume the learned approach is superior.

# Execution order and deliverables

## Phase A — Evidence reconstruction

Start by auditing existing code, data, experiments, and reports.

Produce:

1. `reviewer_evidence_matrix.md` — reviewer concern → existing evidence → missing evidence → action.
2. `pipeline_ablation_plan.md` — R1.4 component map, missing experiments, and activation replacement learning-rate findings.
3. `dataset_protocol_audit.md` — preprocessing, annotation semantics, class mappings, dataset homogeneity, and confounding risks.
4. `dry_scene_audit.md` — actual dry-scene handling across all datasets and pipeline stages.
5. `mndwi_evidence_audit.md` — verified existing comparisons, limitations, and missing measurements.

Keep documents concise and evidence-based, with references to exact files, configurations, and result artifacts.

## Phase B — Approval checkpoint

Present a concise decision report before any protocol-dependent experiment.

Explain:

- Whether the current dataset protocol is scientifically defensible.
- Exactly which class mappings should be retained or changed.
- Whether GDACS preprocessing requires correction.
- Whether dry-scene training inclusion is necessary.
- Which experiments are essential versus optional.
- Expected GPU cost and dependencies.
- Which actions require my approval.

**Do not proceed past this checkpoint on protocol-dependent changes until I approve.**

Independent audits, existing-result reanalysis, and protocol-independent R1.4 experiments may continue.

## Phase C — Implementation and execution

Implement approved experiments using existing infrastructure.

For every experiment, record the hypothesis, controlled comparison, config, checkpoint, dataset version, evaluation mask, calibration rule, seed, compute budget, and output path.

Schedule jobs according to GPU availability and existing workload. Avoid duplicate experiments and unnecessary retraining when existing checkpoints can be reevaluated.

Use explicit stopping criteria for experiments with negligible information gain.

## Phase D — Reviewer-ready synthesis

Produce a final evidence-based summary for each concern:

- What the reviewer questioned.
- What was previously known.
- What the audit or new experiments established.
- Whether the concern is resolved, partially resolved, or still open.
- Which paper figures, tables, and claims should change.
- Suggested precise wording for the manuscript and reviewer response.

Separate measured evidence from interpretation and unresolved uncertainty.

# Final priority

The objective is not to maximize experiment count. It is to make the revised paper more defensible, reproducible, and scientifically informative.

Prioritize experiments that can change a scientific conclusion, reveal a real weakness, or resolve an important reviewer objection.

Preserve the paper's central contribution where supported, but do not protect it from contradictory evidence.

**Begin with repository inspection and evidence reconstruction. Do not start by writing new training code.**