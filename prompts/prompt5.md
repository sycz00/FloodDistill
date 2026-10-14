# Close the actual reviewer concerns after V4

Work in the existing FloodDistill project on the `research` branch. Read applicable AGENTS.md/AGENT.md and preserve original sources, checkpoints, data, results and unrelated work. Execute the bounded work below using the existing evaluators, collectors and scheduler. This is a reviewer-closure phase, not another research programme.

## 1. Source boundary and authorization

Use only:

1. Our supplied manuscript and supplement, or their corresponding editable project sources.
2. The actual editor/reviewer letter.
3. Existing project implementation, recorded experiments and V1-V4 artifacts as evidence for those manuscript claims and reviewer concerns.

The paper and letter define the requirements. Earlier prompts and audit reports do not create additional reviewer requirements. Do not browse new literature, use the supplied competing papers to invent new obligations, add an unrelated benchmark, or redesign the contribution. Check the actual manuscript sentence before saying a claim needs removal: it already qualifies geometry confounding, inconsistent OFA gains, synthetic perturbations, surface-water semantics and mixed INT8/FP16 execution in several places.

This prompt authorizes focused analysis, count-based sensitivity summaries, narrowly necessary inference from fixed checkpoints, and reviewer-focused revision/rebuttal drafts. **No new training, new QAT, seed expansion, budget sweep, architecture, teacher, dataset collection or hyperparameter/threshold search.** Preserve previously approved running work if newer live state exists, but do not relaunch completed V4 jobs. A demonstrable issue requiring training beyond this scope must be explained as a specific remaining decision, not silently scheduled.

Do not submit, publish, send messages to reviewers, or contact other authors. Produce reviewable drafts without overwriting the original manuscript.

## 2. Establish the current evidence and what is already finished

Read the latest local `prompt4_report_20261007/` package, especially:

- `prompt4_report.txt` or PDF; `verified_completion_status.json`
- `reviewer_evidence_conclusions.csv`
- `tables/clean_all_metrics.csv`, original per-seed/paired summaries, event summaries and original-score reproduction tables
- `tables/training_all_44.csv`, `tables/queue_all_642_jobs.csv`
- `cross_cohort_collection_resolution.json`, `unavailable_cross_cohort_cells.json`, and `cross_cohort_common_protocols/`
- `source_index.json` and the referenced original-data, checkpoint, overlap and deployment records.

The report capture at 2026-10-07 11:35 Berlin states:

- All 2,500 selected original source/cache pairs are restored and verified, including all 127 formerly missing IDs. No further selected source imagery or pseudo cache is requested by this result.
- All 28 declared copied main checkpoints load strictly. Original score rescoring differs from recovered old aggregate mIoU by at most about 0.0192 percentage points; this is not exact historical training reproduction.
- All 44 original-data training runs, their declared numerical readouts/corrections and original-data collectors finished. The 60-run regenerated-cohort study is separate and complete in its declared numerical scope.
- The worker finished; zero runnable GPU hours remained. The raw historical queue has 611 successful, 30 externally blocked and one failed collector node. The failed node requested undeclared regenerated-N100 native-WF values. A separate common-protocol collector repaired the usable comparison scope without inventing those scores.

Refresh live state before acting; do not treat that capture as a current process listing. Preserve original receipts. The missing two regenerated-N100 native-WF cross-cohort comparisons are NOT automatically a reviewer gap. Retain them as unavailable and use the completed common-protocol tables; do not schedule eight extra evaluations solely to make an old collector green.

## 3. Map every actual comment to a concrete response

Build one compact closure table with: reviewer ID; exact request; manuscript section/claim; existing evidence; scientific limitation; remaining action; final draft location; status. Use statuses that distinguish evidence-ready, writing-needed, requires-specific-analysis, author-input-needed, device-input-needed, and explicitly-limited-in-manuscript. Completion of code or an audit is not itself completion of a reviewer response.

Cover all of these actual requests:

| Comment | Required scope |
|---|---|
| R1.1, R2.1 | Explain the contribution beyond established KD, pseudo-labels, selection and quantization |
| R1.2, R2.4 | Present/discuss recent compact architecture comparison and OFA on alternative backbones |
| R1.3 | State data roles, selection/calibration/development choices and possible information leakage |
| R1.4 | Make filtering, FPS, KD/OFA, activation replacement and QAT ablations understandable together |
| R1.5 | Make surface-water/flood and teacher/pseudo-label terminology consistent |
| R1.6 | Acknowledge removal of dry scenes and limits on unattended false alarms |
| R2.2 | Bound the S11/GDACS comparison by image, geometry and geography confounders |
| R2.3 | Explain modest and budget-dependent OFA gains |
| R2.5 | Explain when learned models help relative to MNDWI |
| R2 final | Qualify hardware-specific and mixed INT8/FP16 deployment results |
| Editor | Provide a Discussion and a point-by-point response identifying changes and locations |

Do not require every method to win, a new theoretical novelty, a new untouched dataset, exhaustive pretraining ancestry, or a complete rerun of Table I merely because a previous prompt discussed them.

## 4. Highest-priority remaining analysis: external footprint intersections

Purpose: address R1.3 and qualify the external-generalization claims. V4 reports zero selected-source footprint intersections with S11 validation/test/Bolivia, but three STURM chips intersect two selected training sources (two pairs within 30 days), and one WorldFloods map intersects two selected training sources. Whole-source intersection does not establish overlap with scored pixels or supervised-label leakage, but cannot be treated as proof of an independent test either.

First read the existing original-data gate/geographic-screen outputs for the actual IDs; do not rescan all experiments. Identify the source-level intersection records, dates, native footprints, original N2500/N100 membership, training crops/augmentations and corresponding evaluation windows/masks. Check that coordinates and CRS are correct and that an entire scene bounding box has not been mistaken for the actual chip. Distinguish:

- Duplicate acquisition/input exposure.
- Geographic overlap on different dates, with the actual date separation.
- Evaluation-label use versus exposure only to imagery/teacher-derived targets.
- Source-footprint intersection versus actual scored-region intersection.
- Unknown effective training support where the historical crop path cannot be recovered.

Use full potentially sampled training support for stochastic crops, not an invented single crop. If exact historical support is unknown, retain a conservative overlap classification. Count affected evaluation chips/windows/maps/events and valid/water pixels where existing masks permit. Derive the exclusion list from metadata only, before viewing comparative scores.

### Bounded overlap sensitivity

If actual scored-region overlap exists OR cannot be ruled out with the recovered evidence, produce one conservative held-out sensitivity excluding the affected complete STURM chips and affected WorldFloods map(s), using a common exclusion set for every comparator. Preserve the full-benchmark scores separately. Do not change the training pool, select exclusions to improve results, or call this a newly untouched benchmark.

Default scope: clean evaluation for the fixed teacher, fixed MNDWI and all four seeds of the original-data EfficientViT N2500 focal/KD/OFA, N100 KD/OFA, and N2500 MobileNet/LightFormer KD/OFA comparisons used in the reviewer tables. Start from existing per-map/window confusion counts or saved predictions. Recompute pooled metrics and existing event summaries on the retained groups; do not average chip IoUs to approximate pooled IoU. State changed event counts and that seed variation differs from geographic uncertainty. Apply the already fixed corrected STURM validity and native-WF geometry definitions, with legacy scores kept as historical references.

If the directly claimed N=252 historical S11-versus-GDACS results involve different training subsets, first establish whether the same overlap concern affects those actual subsets. Reuse their existing counts for the same metadata-defined sensitivity only when membership and count-domain identities are available. Do not substitute N2500 overlap findings for unverified N252 membership, or retrain Table I. If these artifacts are missing, qualify that historical claim explicitly.

This sensitivity should normally be CPU collection, not a new GPU programme. If fixed-checkpoint inference is genuinely necessary because required counts/predictions are absent, record the exact missing cells and finite job list; reuse the existing evaluator/scheduler and unaffected counts wherever possible. No stochastic inference or model changes. Reuse existing per-map robustness counts to check retained primary diagnostic conclusions on the same exclusion set where available; do not run a new full corruption sweep to fill absent counters.

If all potential intersections are demonstrably outside scored content, record that evidence and skip unnecessary rescoring. In either case, state what is and is not established. Excluding known overlap does not erase historical benchmark-guided method choices or prove all geographically correlated examples have been eliminated.

## 5. Close protocol/history and numerical presentation gaps

Prepare a single data-role table covering teacher task adaptation, direct S11 hard labels, pseudo hard targets, live teacher soft targets, normalization, pre-filter/FPS choice, validation checkpoint selection, activation/quantization calibration, external readouts and any development feedback. State where each threshold/hyperparameter came from when existing records establish it. Do not silently describe empirically chosen values as predeclared.

Use any already supplied author answer about whether STURM/WorldFloods scores influenced method choices. Otherwise carry that one factual question forward as an author input, with the exact conflicting manuscript wording. Do not claim either historical non-use or contamination from code inspection. Retain conservative disclosure rather than inventing an untouched-test history, and continue all other work.

Keep original printed results, original checkpoints rescored under the historical evaluator, corrected-protocol scores, recovered-recipe retraining and corrected matched controls as separate evidence groups. Original-data restoration establishes availability and current source/cache matching, not a complete historical byte-level chain. Exact historical runtime/cache-generation records can qualify claims if recovered; their absence does not automatically require more training.

Reconcile Table II display values against its underlying seed records and declared rounding. V4 identifies UNet-ResNet50 KD/STURM printed 0.747 versus recovered raw about 0.746492 and rescored about 0.746483. Inspect the other printed/raw/rescored rounding crossings as well, including EfficientViT CE/WF and KD/WF. State whether each is a source/display discrepancy or a small evaluation drift; regenerate consistent display rows from a named source, without mixing old SD with new means. Do not tune the evaluator to recover a rounded digit. Fresh precision/recall must share the exact checkpoint/mask/protocol/count domain with the row's IoU.

**Reconcile the teacher baseline separately.** The near-reproduction maximum of 0.0192 percentage points covers the declared student checkpoint comparisons, not all numbers in the paper. The actual Table II teacher row is S11/STURM/WF mIoU 0.897/0.755/0.771. V4's fixed teacher under its legacy clean protocol is approximately 0.894572/0.753917/0.774820. These differences are not all explained by display rounding. Inspect the existing historical teacher results, checkpoint identity, model implementation, input/evaluation/mask/aggregation settings and source of the printed row. Reuse current fixed-teacher counts and existing source receipts; do not train or tune a teacher. Explain any proven protocol/model difference; do not guess that hardware caused it. If the historical teacher row cannot be traced, preserve it as an unverified historical value and base revised numerical comparisons on the verified teacher and student readouts from the same declared evaluation protocol. Recompute affected teacher-gap statements and paired analyses from those matching counts; do not silently retain the old teacher row beside new student scores. Additional frozen-teacher inference is allowed only for an identified missing historical comparison, with no new training.

Do not call recovered-recipe training bitwise reproduction. Keep all seeds, including the increased CE variation. Report actual completed/selected steps and common early-stopping policy rather than claiming every control completed exactly the same number of updates. This does not justify another sweep.

## 6. Turn completed evidence into the requested scientific explanation

### Architecture, OFA and novelty

Use the original-data matched comparisons, all four seeds and declared protocol labels. Expected clean OFA-minus-matched-KD mIoU differences in percentage points are approximately:

| Student/budget | S11 | Corrected STURM | Native WF |
|---|---:|---:|---:|
| EfficientViT N2500 | -0.335 | -0.023 | +0.438 |
| MobileNetV3-Small N2500 | -0.291 | -0.052 | +0.353 |
| LightFormer-ResNet18 N2500 | +0.152 | -0.183 | -0.887 |
| EfficientViT N100 | +0.695 | +0.116 | +0.697 |

Verify against the final source tables; these anchors are not target scores to reproduce again. Attach seed variation and distinguish event-macro evidence from pooled-pixel effects. N100 supports a limited positive effect, strongest in the supplied S11/WorldFloods evidence; STURM is not resolved by its event interval. The regenerated-cohort null result remains a separate result. Do not attribute every change to the 127 images: original hard-label caches and other cohort semantics also differ.

The additional backbones address the architecture-control request without needing to win. A common controlled recipe does not establish a globally optimal LightFormer configuration, matched parameter size or measured edge performance for every backbone. Do not invent an architecture-size-equivalence claim the manuscript never makes.

Explain the contribution using what our manuscript and evidence actually support: applying a frozen task teacher to additional imagery without new manual annotation, controlled compact-model transfer and its limits, and the measured deployment configuration to the extent its evidence is supportable. Attribute established components. Avoid an unsupported universal novelty/first-in-literature assertion; no new literature search or speculative few-shot/XAI theory is required here.

### Ablation clarity and implementation corrections

Prepare one concise ablation map: component -> existing comparison -> what is held fixed -> what also changes -> supported result -> source location. Include filtering, FPS versus other selectors, KD/OFA, activation replacement and QAT. Reuse the existing main/supplement evidence. Do not turn sequential or confounded deltas into an additive decomposition, or infer an FPS benefit where the supplement reports no clear advantage.

Correct only demonstrated manuscript mismatches: true CE versus focal; historical clipping differences; loss mask reductions; branch teacher/student temperatures and normalization; 11,368 EfficientViT training-only OFA head parameters rather than 488; stored maximum-class-confidence/spatial-dispersion descriptors; STURM transforms and invalid support; partial WorldFloods border resizing; QAT weights adapting with fixed scales/BN. Correct the crop diagnostic from 86 to 200 chips losing over half their native water, with 73.6263% pooled water removal and 21 positive chips becoming dry. Preserve that this does not isolate label quality or geography.

In the STURM dataset description, replace the assertion that a shift common to both inputs makes a paired comparison insensitive to that shift. Shared input processing makes the evaluation protocol consistent; differently trained models can still respond differently to the same shift. Keep the comparative conclusions conditional on the protocol. This is a wording correction supported by the existing diagnostics, not a request for another experiment.

### MNDWI and practical meaning

Use actual student, teacher and fixed-threshold MNDWI outputs; clean precision/recall and all already declared perturbation outcomes are available. Explain the trade-off between false water detections and missed water. For example, native-WF N2500 OFA has approximately 91.48% precision/68.33% recall versus MNDWI 77.82%/77.99%; this is not a dry-acquisition false-alarm estimate.

Retain both favourable and unfavourable perturbations. Haze h0.20 gives a large conditional learned-model advantage (legacy WF OFA about 74.76% mIoU versus MNDWI 25.67%), while other existing spectral perturbations favour MNDWI. Keep these as synthetic diagnostics, as the manuscript already does; they do not prove robustness to real observed atmosphere or remove the need to qualify clean performance. Use no cross-paper score target or new threshold tuning.

### Terminology and Discussion

Use surface-water segmentation consistently for what the labels establish. Explain that detecting newly inundated land needs temporal/reference-water information, and distinguish teacher-supervised hard labels from live soft targets and human validation labels. Draft a focused Discussion tying the results to each actual review concern, without adding new claims of climate, hydrological or operational validation.

## 7. Dry-scene and deployment evidence: close the actual request without inventing requirements

### Dry scenes: R1.6

The reviewer asks for clearer acknowledgement that filtering removes dry scenes and false alarms are inadequately evaluated. A new dry dataset is not an explicit mandatory experiment.

Check the existing 11-candidate/3-event package for genuine independent review dispositions. If supplied and valid, reuse the existing import gate and already prepared dry jobs, with fixed criteria and all appropriate original-data primary comparisons. Do not invent labels, fill unknown regions as land, or use teacher/MNDWI predictions as independent truth. Keep undefined empty-water IoU/recall explicit and report false-water fraction and existing alarm criteria instead.

If the dispositions are absent, do not block the manuscript indefinitely or collect another dataset. Prepare the clear limitation requested in Methods/Discussion/Conclusion and remove only any actual unsupported unattended-operation claim. Mark independent dry reliability as unmeasured while marking the clarification as addressed once incorporated. Existing related S11 dry-chip results remain limited diagnostics.

### Deployment: R2 final and the paper's Tables III-IV

The body already states Xavier-specific mixed INT8/FP16 execution and that FP16 is faster. Make Abstract, contributions, headings and tables consistent where needed; do not present those acknowledgements as newly discovered implementation faults.

Separately inspect whether retained numerical deployment claims have adequate supporting artifacts. V4 says the copied files did not add the missing engine/export/device-log chain. Check only newly supplied or already identified project locations once. Four-seed device-accuracy claims need corresponding seed evidence; Table IV's representative timing measurement does not require four independent timing seeds.

Where matching original artifacts and an already authorized accessible Xavier workflow exist, reuse logs or perform the existing bounded fixed-artifact accuracy/timing measurements. Keep checkpoint->ONNX->engine identity, matched float comparator, dataset split, actual power rail and measurement scope explicit. The prepared S11 path selects validation; do not call it test. Do not use a QAT-stripped latency-only float export as a validated pre-QAT float accuracy comparator. No RTX substitution, assumed engine portability or new QAT training.

Where evidence remains unavailable, prepare the precise claim/table qualification or removal needed to avoid unsupported precision. Do not label historical results false solely because raw logs are absent. Keep a specific author/device input list for claims that cannot yet be substantiated. Do not extrapolate peak watts into energy per image or relabel a combined CPU/GPU/CV rail as GPU-only power. Do not attach the new desktop model accuracies to an old deployed engine.

## 8. Deliver reviewer-focused drafts and final status

Reuse existing project locations and collectors. Produce only:

1. The reviewer-closure table, covering every actual numbered comment and the editor's Discussion/response requirement.
2. The bounded overlap assessment and, where needed, its common-domain count-based sensitivity tables.
3. Updated, consistently sourced manuscript tables/figures needed for those responses; retain historical/corrected and cohort distinctions.
4. Proposed manuscript/supplement edits and a point-by-point response draft, with actual section/table/figure locations. If editable sources exist, prepare a reviewable patch or separate revised copy and compile/check it using existing tools; preserve the submitted original. If only PDFs are available, provide exact replacement passages and table data with section anchors, not invented line numbers or an invented source tree.
5. A short remaining-input list distinguishing author history, optional dry-reference validation, and the deployment evidence needed for retained device claims.

In the response draft, distinguish a proposed change from an incorporated change. Do not claim manuscript edits or independent verification are complete unless they are. For unavailable inputs, give the supported conservative wording and mark the unresolved author choice.

If any bounded inference or newly unblocked dry/device work is scheduled, attach its dependent collector using the existing detached worker. Reuse completed receipts; a collector must wait for all required outputs, not merely process exit. Label partial tables explicitly. No duplicate worker, repeated training, or long chat polling. If no runnable computation remains, finish the prose/table work rather than keeping GPUs busy.

Success means that each reviewer request has a specific evidence-backed answer or a clearly explained limit, and that remaining numerical manuscript claims are traceable. It does not mean forcing favourable results, erasing historical failed queue records, or proving every possible form of generalization. Stop when this scope is complete.
