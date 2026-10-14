# V9 Codex prompt: learn from disagreement without damaging unsupported cases

Prepared 12 October 2026, Europe/Berlin.
Purpose: implement and run a bounded overnight follow-up to the audited V8 results, incorporating Opus's reviewer wrap-up supplied on 12 October.
This is an experimental instruction, not a claim that the proposed method is novel or successful.

## 1. Mission and constraints

Work in the existing FloodDistill research branch and reuse its loaders, graphs, evaluators, quantizers, caches, queue, and reporting code. Inspect AGENTS.md and the working tree first. Preserve user changes and immutable historical runs. Avoid a new framework.

Finish high-value reviewer controls and investigate two linked research questions:

1. Can support imagery selected using complementary Sentinel-2 predictions enable a useful, inexpensive student correction, while preserving predictions in cases the support does not represent?
2. How much quality does the established tiny student retain under competent low-bit compression, including difficult events?

Default to S2-only inputs. Use the existing Prithvi teacher, R4 student and existing MNDWI implementation. An already installed, independently human-supervised S2 UNet is admitted only for the controlled teacher comparison below. No S1 requirement, new foundation-model fine-tuning, NAS, new dataset acquisition, driver changes, paid service, message to another person, or manuscript submission. No subagents.

Implement and validate the finite experiment, start the detached worker on the authorized host, and verify it is actually running. Do not stop after writing a plan when the repository/data/device are available. Conversely, do not claim a launch when the current environment only contains reports.

## 2. Ground truth from V8 and historical context

Locate and read the audited V8 report and original run directory. The report identifies:
- /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v8_20261012T123202Z
- analysis_20261012/audit_analysis.py and audited CSVs inside that run.
- Original host environment: /home/schmalstieg/miniconda3/envs/esi/bin/python.

These are observed historical paths; verify rather than assuming they exist on this host. The supplied PDF's SHA256 is:
1d05fef4a19b25dc41f3eddafbf974f0bb0d33a25f50ad0ee180caa35eac62c4

Read reviewer_wrapup_20261012.md, the actual reviewer letter and test.txt if available, earlier revision closure tables, V6/V7 audits, and newer live state. The supplied wrap-up identifies paper/review.docx on the research host and newly recovered raw Jetson logs. V8's statement that its agent could not locate the letter is an input-availability limitation, not proof the letter does not exist. Its 11 figures and verification script are referenced host artifacts, not automatically present merely because the Markdown is attached.

Known facts to reconcile against receipts:
- V8 completed eight jobs and 180 adaptation rows, with no failed jobs. This was a three-event pilot, not 180 independent experiments.
- Target pilot: EMSR264, EMSR286, EMSR333; two support draws and parents 42/64; 92 query tiles. Keep these episodes as diagnostic data.
- R4 has 12,230 parameters, frozen extractor/BN, a 16-feature affine two-class head, and 17 effective margin coordinates.
- Relative to frozen FP: F2_20 ordinary soft-KD +0.079 pp query mIoU; uniform ridge -0.378; confidence/stability-weighted ridge -0.552.
- One-chip supervised ridge -1.771 pp; teacher+label ridge -1.238; supervised SGD20 -0.353. Supervised hyperparameters were inherited from teacher-only tuning, so this was not a strong independent supervised baseline.
- On six support chips, 8,741 of 9,648 examined confident/stable reference-water pixels were incorrectly predicted as land. This is 90.60% conditional water error in that diagnostic subset, not overall teacher error.
- Some support samples had no water. Source-selected lambda=0.1 was at the grid boundary. All teacher-only ridge steps had eta=1.
- Uniform ridge and soft-KD differed in objective, not merely solver. Frozen-feature integrity and solve/folding checks passed; repeating those entire audits is not the next scientific question.
- WF equal-event water IoU: R4 47.95, teacher 50.54, their unattainable oracle 54.00. Teacher/MNDWI oracle: 61.40 equal-event and 73.86 pooled. Do not invent a three-expert oracle from two pairwise tables.
- Quantized parent: W4A8 body, W8A8 stem, FP32 head, fake quantization. Mean query mIoU difference from FP -0.016 pp, but event differences -1.240/-0.148/+1.339 pp. Q3 failed against simpler Q controls.
- Inner fit medians: ridge 0.731 ms, SGD20 6.267 ms. Partial component sums 58.1 versus 63.9 ms omit true cold-start work.
- FP16 artifact 156,695 bytes versus FP32 183,255; almost unchanged source score but slower observed workstation timing. No native INT4 deployment result.
- Corrected OFA curves show some budget-dependent update behavior, not universal speedup or measured time-to-quality. Parser repair is complete.
- V7 C3 continuation failed; even ordinary continuation degraded the parent. Earlier low-bit teacher-excess allocation did not beat a simple structural allocation.
- CE in the current GDACS runs uses teacher hard masks. The fixed teacher inherits 252 labelled source-training scenes.
- Existing external benchmarks are research-exposed. Re-freezing them does not make them untouched.

Retire unchanged F4 and Q3 as candidate recipes. Reuse them only as historical controls. Do not repeat the same +2% stability weighting in search of a favorable seed. Keep ordinary head KD, useful teacher/rule complementarity, tiny-model compression, and properly controlled few-shot learning open.

### 2.1 Incorporate the wrap-up critically, using CPU evidence first

Keep its [V] re-derived, [R] inherited and [J] judgement distinctions. Verify accessible raw files before adopting a newly recovered number. Produce revision_corrections.csv with claim, original value, proposed correction, protocol/graph/seed, evidence path/hash and unresolved qualification. Do not rewrite the manuscript automatically.

Prioritize the teacher row and dependent comparisons; FP32 mean GPU rail power (reported correction 3.95 to 3.25 W); pre-filter effects by selector; crop accounting (200 chips, 73.6% removed water, 21 dry); OFA training-head count (11,368); teacher-referenced versus true false alarms; and the newly recovered per-seed Jetson chain. Reuse the wrap-up's CPU verification script if present, inspect it before running, and write new outputs rather than modifying earlier evidence.

Resolve these overstatements instead of copying them:
- Four-bit "lossless" is unsupported by the matched -0.25 pp interval or V8 event cancellations; non-significance is not equivalence.
- Better LR supports under-optimization as a contributor to the apparent activation cost; it does not isolate an intrinsic activation-only effect across different training/initialization histories.
- OFA has no demonstrated universal/end-to-end speed advantage. Preserve N100's bounded update evidence, censored thresholds, whole-wrapper timing, and the audited full-budget masking distinction.
- Do not mix legacy WorldFloods teacher values with corrected native-geometry values in one gap calculation.
- Do not use "effect smaller than seed SD" as a significance rule; use the paired design and its uncertainty.
- Do not assume every external protocol removes permanent water; verify the specific full-water/flood-only mask definition.

Use the author's existing STURM clarification: STURM did not determine the choice of OFA or EfficientViT; it was used to investigate additional teacher-derived/OOD criteria, including saliency-related ideas, that gave null results. Mark this as author-reported history, reconcile retained code/comments, and do not extrapolate it to WorldFloods or every selection threshold. Preserve development exposure even for null exploratory tests. Leave only genuinely missing details as author-input items; do not ask again for the clarification already supplied.

### 2.2 Required reviewer control: identical images, different labels

This addresses R2.2 directly. Use the same 252 S11 training images, split, valid support, footprint, augmentations, initialization per seed, optimizer, stopping/budget and evaluation code. Compare human hard masks with frozen Prithvi hard masks. The only difference within each paired contrast is label source.

Primary: four paired seeds of EfficientViT-B0 with true cross-entropy in both arms. Reuse a human-labelled anchor only if its entire recipe/input lineage matches; otherwise train the pair. Do not compare true CE with focal loss or change clipping only for one arm. Apply the same reference-valid mask to both target sources, without learning from validation/test labels. Make new teacher masks immutable; preserve original caches.

Optional extension after the primary pairs: cross label source with the same added soft-KD term, using L=CE(hard target)+0.5*T^2*KL(teacher||student), T=4, for both human-hard and teacher-hard arms. Keep the CE coefficient fixed across presence/absence of KD. This isolates the hard-label source within an objective and provides a controlled soft-target contrast. Existing anchors with another objective are not interchangeable. No extra OFA retraining is needed.

These are labels on images the task-adapted teacher itself saw during source training. The control isolates label-source effects on that source dataset; it does not prove pseudo-label superiority on unseen imagery, remove all original S11/GDACS confounds, or establish label-free learning.

### 2.3 Conditional reviewer control: is the foundation-model teacher necessary?

Inventory the existing UNet-ResNet50 checkpoint suggested by Opus. Verify that it was trained on the 252 human-labelled S11 training images, not Prithvi pseudo-labels or an ensemble trained on those pseudo-labels. Verify six-band handling, preprocessing, output convention, task head and selected checkpoint. If that independent provenance is absent, mark the comparison unavailable and continue; do not spend the budget creating a new teacher.

If available, use this S2-only teacher on the same fixed GDACS N2500 cohort. Do not re-filter/reselect the cohort for each teacher; disclose that the original selection was based on Prithvi, so this is conditional on that cohort. Evaluate teacher/student/rule joint errors and teacher task quality. Then train four EfficientViT-B0 logit-KD student seeds against the existing compatible Prithvi-taught anchors, with identical student recipe and budgets and teacher-specific hard/soft targets. If compatible anchors cannot be established, admit complete matched pairs or defer the comparison.

Admission depends on provenance and cost, not whether the alternative teacher wins its screen. Report poorer, equal or complementary results. This tests the specific Prithvi-versus-supervised-UNet comparison, not the universal necessity of foundation models. Teacher pretraining and adaptation costs remain different and must be disclosed. A fixed teacher plus four student seeds is not four independent teacher trainings.

### 2.4 Dry-scene evidence without invented verification

Locate the six reference-labelled dry S11 chips and the 11 unreviewed candidate windows described by Opus. A cheap readout of the already verified six chips for compatible parents is useful, with any-water/0.1%/1% area thresholds, but it remains a tiny exposed diagnostic. The reported 43% false-water case belongs to one historical KD checkpoint, not automatically R4 or all KD students.

Run the prepared 11-window evaluation only if independent human verification records actually exist. Otherwise keep it blocked and provide the existing review template/contact-sheet path. Never fill "verified dry" from teacher predictions, MNDWI or the need to finish overnight. Do not call teacher-referenced disagreement a true dry false-alarm rate.

## 3. Budget, schedule and job admission

V8 cumulative spend was 25.969689 GPU-command hours under the existing 30-hour cap, leaving 4.030311 then. Recompute current spend and reservations before doing GPU work.

Incremental cap = min(4.0 hours, max(0, 30 - verified spend - other reservations)).
This prompt does not reset that cumulative cap. Count failed attempts, teacher inference, smokes, adaptation, quantization and evaluation consistently. CPU work is separately timed.

Use two explicit clocks: allow up to eight hours from worker launch to wait for the first GPU command, then allow up to eight hours from that first GPU command (including a smoke) to finish the finite run. The absolute wall limit is sixteen hours from worker launch. Record launch, wait, execution and absolute deadlines in UTC and Europe/Berlin; persist them so a restart never resets them. Waiting after the first GPU command counts against the execution window. Reserve the final ten minutes of the applicable execution/absolute window for collection, and run CPU collection immediately if the initial wait limit expires. Idle polling and CPU-only work consume wall time, not GPU-command hours. The host must remain awake. Do not reopen a finished V8 manifest; create a timestamped V9 directory. If another worker is active, respect its locks and reservations and wait automatically under Section 9.

Suggested admission ceilings if four hours remain:
- Same-image label-source control: 0.65 h.
- Provenance-qualified S2 conventional-teacher comparison: 1.50 h, conditional on existing checkpoint and compatible anchors.
- Causal replay, support pools, source development and fixed adaptation matrix: 0.75 h.
- Full low-bit readout and bounded compression follow-up: 0.50 h.
- Protected final evaluation/collection and bounded retry reserve: 0.60 h.

CPU reconciliation starts immediately. GPU priority is the primary label-swap pairs, the bounded adaptation core, the full unchanged-PTQ readout, then the qualified alternate-teacher comparison. Optional label-swap soft-target extension, QAT pairs and extra adaptation seeds use verified freed capacity only. If the alternate teacher is unavailable, its reservation may fund complete FP/QAT pairs or the label-swap extension; this fallback is predeclared and cannot depend on external accuracy. Opus's 0.5/2 GPU-hour estimates are estimates; measure smoke throughput before admitting work. Do not fit an optimistic schedule by dropping controls or exceeding the cap.

These are maximums, not targets. Stop when the frozen work is complete. Preserve paired blocks and final collection before adding optional jobs. If the cumulative budget is exhausted, finish CPU analysis, code and a deferred manifest with exact blockers. If capacity is only temporarily reserved by another active worker, wait and reconcile its verified ledger at the next ten-minute check; never cancel its reservations or treat unverified spend as zero.

All conditional branches and criteria must be in the frozen manifest before target-query scores are read. Source-development results may select settings; target outcomes never alter admission, hyperparameters or support selection.

## 4. Minimal causal replay: distinguish objective, solver and update magnitude

Reuse V8 cached support features and teacher logits wherever their identities match. Saved teacher hard masks are sufficient for oracle counts, not soft-KD targets; never manufacture soft targets from them.

On the existing V8 episodes, conduct a labelled-as-diagnostic replay:
- Compare uniform ridge's exact solution with a numerically converged iterative solution of exactly its same squared-logit objective, same lambda, same sampling/normalization.
- Compare one damped Newton step, 20-step SGD, and a bounded converged convex solver on exactly the same probability-KD objective below.
- Show both native updates and a source-prespecified common RMS margin-change cap of 0.1. This is a magnitude control, not evidence that one solver is inherently better.
- Reuse F3/F4 historical outcomes. No broad retry of confidence/stability weights.
- Separate offline true-water diagnostics from label-free fitting and selection.

For convergence checks, use objective value and stationarity; do not select solver iterations on target labels. For the ridge equality check, a deterministic tensor check plus one real support episode is enough. Query predictions across old episodes can diagnose causes but do not become a validation set for the new method.

Outputs: objective_solver_replay.csv with objective, solver, lambda, update RMS, predicted-water shift, water precision/recall and query differences.

Possible interpretations:
- Same ridge solution across solvers: the closed-form implementation was not the cause.
- Probability KD outperforms squared logits at matched movement: objective choice matters.
- Very small steps avoid losses: support-induced movement was too aggressive.
- Every teacher-driven update worsens a case: teacher targets or support relevance may be inadequate; do not assume numerical failure.

## 5. Main hypothesis: use disagreement for coverage, protect unrepresented cases

Working description: support-aware protected head adaptation.

This is a proposed combination, not a first-in-literature claim. Support selection and anti-forgetting are established ideas. The specific test is whether teacher/student/spectral-rule disagreement identifies missing support regimes, and whether protecting those regimes improves adaptation at a fixed deployed graph.

### 5.1 Episode construction and data roles

Keep the three V8 events for diagnosis. For the new fixed pilot, select up to four other eligible WF events deterministically by metadata, without reading outcome scores. If fewer are eligible, report fewer. Require verified spatial separation and common valid masks.

Per event/draw, define:
- a pool of at most 16 unlabelled candidate support chips;
- a fixed query region, disjoint from every candidate chip and with at least the model's spatial-support buffer;
- two deterministic pool/draw assignments;
- parents 42/64.

Do not use query inputs for selection, fitting, anchors or normalization. This is an inductive comparison, not an undeclared transductive one. With four events there are 16 event/parent/draw episodes per arm; these are not 16 independent events.

Hold out query regions first by spatial metadata. Never change them after seeing whether the proposed selector finds useful support. If valid spatial separation cannot be verified, block claims of new transfer and complete diagnostic/source/compression work.

For source development use at least four eligible S11-validation event-separated support/query episodes if available; otherwise disclose the reduced count. Record every unique human-labelled source query pixel used to tune settings. Reuse TRAIN-only feature normalization. No extra source labels may silently become training labels.

### 5.2 Fixed prediction regimes

On candidate support inputs, compute:
s = 1[p_student(water)>=0.5],
t = 1[p_teacher(water)>=0.5],
r = existing MNDWI binary water prediction.

Use the existing physically correct reflectance/band handling. Never compute MNDWI directly from independently standardized bands without reversing that transformation. Respect denominator/mask rules and the inherited zero threshold.

The triple (s,t,r) defines eight prediction regimes. Agreement is not truth; these groups describe model behavior only. Two regimes have unanimous predictions; the other six expose complementary hypotheses.

Sample at most 4096 spatially distributed valid pixels per candidate chip for scoring. A regime counts as represented when it has at least 64 sampled pixels in at least two distinct cells of a fixed 4x4 spatial grid. Freeze these design constants; do not tune them on WF.

Support policies:
- H0: deterministic metadata/hash-selected chip, independent of predictions.
- H1: coverage-only control. Let u=max(s,t,r); choose the chip maximizing min(count(u=0),count(u=1))/N. This balances predicted water/land coverage without identifying disagreement regimes.
- H2: disagreement-regime coverage. Maximize sum_g omega_g*I[g represented], with omega=2 for disagreement regimes and 1 for unanimous regimes. Tie-break by the number of occupied regime/spatial-grid cells, then a stable chip hash.

Each policy selects exactly one fitting chip. Preserve the pool and selection scores before any target references are read. H2 can still select a bad chip; that is a scientific outcome.

Use the same within-chip spatial sampling distribution for all methods; do not add class balancing in only the proposed arm. Log realized regime counts and, offline only, true-class coverage.

### 5.3 Anchors and preservation

From the unselected candidate pool, cache up to 128 spatially diverse features per nonempty regime, equally weighting regimes. These are unlabelled target context; report their images, features, bytes and teacher queries in the access budget. They are not free support.

First distinguish selecting an informative chip from believing its teacher. Selecting more teacher/rule disagreements does not make the teacher right.

Use two teaching policies:
- D0: ordinary teacher-only targets on all sampled support pixels.
- D1: teacher targets only where teacher and MNDWI agree (t=r). Treat their agreement as a hypothesis, not ground truth. Where they disagree, retain the parent unless support reference labels are explicitly available in a labelled arm. If D1 admits no pixels, return no-update and record the reason.

Normalize the fitting alpha weights only over admitted pixels. All methods have the same initial sampled support. Report admitted pixel counts and offline teacher error by regime; shared teacher/rule mistakes remain possible. D1 is a simple agreement-gating baseline, not a novelty claim.

Let G_B be the regimes represented by pixels actually admitted to the fitting loss. Presence in the chip alone is insufficient. In a supervised arm, human support labels determine targets and override teacher/rule disagreement; labels from anchors or queries are never used. Compare:
- P0: no prediction-preservation penalty;
- P1: global preservation over all nonempty anchor regimes;
- P2: preservation only over anchor regimes absent/underrepresented in G_B.

P1 is the conventional anti-forgetting control. P2 tests the specific missing-coverage mechanism. Both may preserve wrong parent predictions; neither is a correctness guarantee.

If no unselected support chip remains, do not take anchors from the query region. Mark P1/P2 unavailable. If G_B covers all available regimes, P2 has no protected groups and reduces to P0; report the equality, do not count it as independent evidence.

### 5.4 Objective and mathematically matched solvers

Keep the 17-coordinate head representation and all frozen extractor/BN states. Let phi include the intercept; m0 is the parent water-minus-land logit margin; m_delta=m0+phi^T delta.

Use temperature T=4 and
p_delta^T = sigmoid(m_delta/T), q_T^T = teacher water probability at T.

With alpha_i summing to one on the fitting chip, minimize:

L(delta) =
  sum_i alpha_i T^2 KL(Ber(q_T,i^T) || Ber(p_delta,i^T))
  + (lambda/2)||delta||^2
  + gamma sum_j beta_j KL(Ber(p0,j) || Ber(p_delta,j)).

The anchor term uses T=1, frozen parent p0, and weights beta with equal mass per protected nonempty regime. Set gamma=0 for P0. Its absence is not a different target-label budget.

At delta=0, the gradient and Hessian are:

g = sum_i alpha_i T (p0,i^T-q_T,i^T) phi_i

H = sum_i alpha_i p0,i^T(1-p0,i^T) phi_i phi_i^T
    + lambda I
    + gamma sum_j beta_j p0,j(1-p0,j) phi_j phi_j^T.

The anchor gradient is zero at the parent. Solve H delta_N = -g in float64 with Cholesky. Lambda>0 makes H positive definite. This is one Newton step for a convex objective, not its exact minimizer.

For a conservative Newton arm, use deterministic backtracking factors 1,1/2,1/4,...,1/64 and accept the largest that reduces the actual fitting+anchor objective and respects the source-selected RMS movement cap. If none pass, return delta=0. No target labels enter that decision.

Use the same full objective for 20-step SGD and the diagnostic converged solver. Match penalties and data; give every solver the same cached features and teacher targets. Fold the accepted delta into the original two classifier rows with plus/minus half corrections, undoing feature normalization as in V8.

For supervised one-chip arms, use ordinary binary cross-entropy on human support labels at T=1. Tune their lambda/LR independently using source-labelled episodes. Do not inherit teacher-only hyperparameters as V8 did. Count the full accessible labelled chip separately from sampled fit pixels.

### 5.5 Required matrix and source tuning

Core teacher-only matrix:
- Frozen parent.
- H0/P0 conventional head KD, 20 steps.
- H1/P0 coverage-only selection.
- H2/P0 disagreement selection.
- H0/P2 preservation without improved selection.
- H1/P2 coverage-only selection plus preservation.
- H2/P1 disagreement selection plus global preservation.
- H2/P2 represented-regime protection, 20 steps.

The arms above use D0. Also run H2/P0/D1 (agreement-gating control), H2/P1/D1 (global protection), and H2/P2/D1 (full conservative candidate). This separates support selection from which teacher corrections are admitted. Compare one Newton step with 20-step SGD on exactly the H2/P2/D1 objective, penalties and admitted pixels.

Add one H2/P2/D0 variant that uses only (s,t) regimes, omitting MNDWI from selection/protection. Compare it with H2/P2/D0 to isolate the rule's coverage role without also changing D1.

Within each objective family, use a bounded source-only grid and select one shared configuration across both parents. Default grid lambda={0.1,1,10}, gamma={0.1,1,10}; P0 varies only lambda. Select SGD LR from {0.001,0.01,0.1} in an initial source pass and then freeze it. Evaluate movement caps {0.05,0.1,0.25} once on source for the selected objective, not across an unconstrained Cartesian sweep. Record the actual number of configurations; controls receive comparable tuning opportunity. Include no-update as a source-selectable conservative policy.

Use event-mean source water IoU for selection with source mIoU as a recorded secondary measure. Record every source event and worst-event loss; do not select solely on land-dominated accuracy. Source-selected settings are frozen before target evaluation.

Add two supervised controls using the same H0 and H2 labelled fitting chips and separately tuned supervised SGD. Include H2 supervised plus P2 preservation if budget allows. This tests whether support selection solves the earlier failed one-chip recipe. Do not claim a label-budget advantage by hiding that a full chip contains many pixel annotations.

Run the fixed core matrix even if the proposed source result is null; it is a bounded diagnostic. Optional expansion to parents 91/107 or quantized adaptation requires, on source development only:
- at least +0.25 pp mean water IoU versus both frozen and H0/P0;
- nonnegative mean mIoU change;
- no source event water-IoU loss worse than 1 pp.
These are declared promotion criteria, not significance tests or application safety guarantees. Keep all failed gates.

### 5.6 Access and cost accounting

H2 may inspect 16 teacher-predicted chips to fit on one. Call this "one fitting chip with unlabelled context", not "one image total". For supervised arms, call it "one labelled chip selected from an unlabelled pool".

Show both:
1. a controlled comparison sharing the same available pool/cache;
2. actual operational cost per method, where H0 need not run the teacher on unselected chips.

Teacher queries, context acquisition and anchor storage can erase a solver-time benefit. Count them. Existing cached inputs may be reused for execution efficiency, but their acquisition cost must be explicitly measured or marked unavailable for a deployment claim.

## 6. Compression branch independent of adaptation success

Reuse R4 FP parents for all available four seeds. Make a full source-validation and full compatible S11-test/STURM/WF readout for the unchanged V8 PTQ precision point; the original three-event cancellation is insufficient. Reuse counts where identities match.

Report pooled and equal-event water IoU/mIoU, precision/recall, every event, and source loss tolerances 1 pp primary, 2/3 pp secondary relative to each matched FP parent. These are exploratory declared tolerances, not a clinical/operational requirement or proof of equivalence. Do not pool training-seed and event uncertainty.

Build a fixed-graph size frontier from existing compatible R2/R3/R4/A3/A0 anchors. Do not retrain all graphs or assume prior early-stopped runs completed the same updates. Mark incomparable histories.

Use the established adopted integer-arithmetic fake-quant simulator and its verified STE path. For a conventional R4 QAT pilot:
- Use existing competent QAT/LSQ code if verified for this graph; do not claim faithful LSQ if it is not implemented.
- Keep W4A8 body/W8A8 stem/FP head as the first point. Do not add W4A4 until this baseline is understood.
- Run parents 42/64 with exactly matched floating-point continuation and QAT continuation: same initialization, batches, valid support, duration, optimizer schedule and clipping.
- Use an existing compatible tuned recipe; if absent, admit a small source-only LR pilot rather than copying a known weak recipe. Protect the complete FP/QAT control pairs. This training is conditional on the available compression reservation or freed alternate-teacher budget; the full unmodified PTQ readout comes first.
- Default maximum 1,560 updates per continuation. Preflight timings determine admission; no incomplete unmatched pair should become a positive result.
- If either continuation fails technically, diagnose once and preserve failure receipts. If both scientifically worsen, do not rescue only QAT by changing its schedule.

Do not repeat the V8 quantization-consistency penalty; it did not beat simple controls. Only if the adaptation source promotion gate passes, apply the selected adaptation and its P0/P1 controls to the same quantized parent. This is a conditional transfer check, not a new low-bit mechanism search.

Real INT8/INT4 deployment remains conditional on an already working compatible backend. No driver repair or new hardware dependency tonight. Fake-quant accuracy, theoretical packed weights, serialized engine bytes and measured kernel speed must have separate columns. Count FP heads, scales and metadata.

Repeat FP32/FP16 timings in three fresh process sessions only if affordable. Record flags and dtypes at the actual timing boundary and actual accuracy boundary. Do not compare new latency to V7's different measurement route or call workstation measurements edge results.

## 7. Metrics and decision rules

Primary adaptation outcome: equal-event query water IoU change versus the same frozen parent. Secondary: mIoU, water precision/recall, false-positive water area, lower-tail event outcomes, and fraction of no-update episodes.

Mechanism metrics:
- fraction of prediction regimes represented in support;
- true-water support coverage, offline only;
- teacher-correct/student-wrong corrections captured;
- student-correct/teacher-wrong errors introduced;
- water-to-land and land-to-water changes;
- prediction movement on protected and represented anchor regimes;
- proposed method minus global-preservation and coverage-only controls;
- gain per measured adaptation second and teacher query, with zero/negative gains retained.

Compute a three-way teacher/student/MNDWI oracle from actual joint counts if available; label it diagnostic and unattainable. Do not fit a gate using query oracle choices.

Report all event/parent/draw rows. Event is the transfer unit; many pixels and repeated support draws do not create independent events. With four new events, show paired event effects and descriptive uncertainty limits; do not market a population guarantee. Four trained-parent results, if obtained, have a separate seed-variation summary.

No-update is a valid fallback but not a novel accuracy improvement. Reduced harm alone must be described as such. If global preservation or simple coverage matches the proposed mechanism, attribute the benefit to that simpler explanation.

Stopping interpretations:
- H1 matches H2: coverage, not disagreement structure, explains the benefit.
- P1 matches P2: generic regularization explains the benefit.
- D1 alone matches the full candidate: simple agreement gating explains the benefit. If D1 rejects useful teacher corrections, report lost opportunities as well as avoided harm.
- MNDWI-free matches full: no evidence the rule helps this mechanism.
- One Newton step matches SGD quality: potential solver efficiency, requiring full cost measurement.
- Supervised source-tuned controls still fail with covered support: this episode/representation remains hard; no claim that annotation is worthless.
- Full proposed method cannot beat frozen/simple controls on new events: stop this recipe. Do not add more weights overnight.
- Low-bit mean hides large event losses: keep the frontier qualified; do not claim lossless compression.

## 8. Minimal verification and reproducibility

Reuse V8 verified code and add targeted checks for actual changes:
- analytic gradient/Hessian versus finite differences or autograd, including T factors;
- no frozen tensor/BN changes;
- folding parity;
- anchor groups and support/query disjointness;
- no target reference read by selector or teacher-only optimizer;
- matched objective for Newton/SGD;
- delta=0/no-update path;
- actual precision map and identical FP/QAT starting hashes.

A deterministic tensor check and a real-batch smoke are sufficient where applicable. No repeated full historical audit unless identity mismatches arise. Preserve originals; fixes go in a new frozen snapshot.

Verify the scheduling changes with a fake clock and mocked GPU/process telemetry; do not spend real ten-minute waits or GPU hours on scheduler tests. Cover these meaningful failure cases:
- busy at startup, still busy at 600 seconds, idle at 1,200 seconds: no early launch, then exactly one eligible launch;
- a foreign compute process with near-zero utilization, insufficient memory, an unallocated device, or failed/unknown telemetry: all remain unavailable;
- a device becomes occupied between the initial check and lock acquisition: the under-lock recheck prevents launch;
- simultaneous workers cannot acquire the same device lock or over-reserve the shared budget; a paired block is deferred if the whole block plus collection cannot fit;
- restart while waiting, during a surviving job, and after a completed receipt: preserve deadlines/spend and do not duplicate a job;
- budget, timeout, pause and stop handling affect only owned processes; final collection still runs after failures or a wait deadline with no free GPU.

Finally perform a live read-only availability check and verify the detached worker's state. A healthy WAITING_GPU state is a valid startup outcome on an occupied host. Run the real-batch smoke only after actual GPU admission, and charge it to the budget.

## 9. Autonomous worker and overnight survival

Implement, test and launch a finite host-side worker that checks GPU availability immediately, then every 600 seconds while jobs are waiting, and automatically starts eligible jobs when an authorized GPU becomes idle. This is an execution requirement, not just a suggested shell command. Do not stop after printing the queue or asking the user to check the GPUs. If this environment only contains reports and cannot access the research host, finish the implementation/manifest and state the exact launch blocker; never claim a worker was launched elsewhere.

### 9.1 Freeze the queue and make it independent of the agent session

Reuse the established detached queue, shared budget ledger and GPU UUID locks. Freeze code/config/input hashes and a finite dependency graph before launch. Each node must have a stable job ID, exact argv/environment, inputs, output paths and validation criteria, dependencies, seed, paired-block ID, conservative GPU cost, memory requirement, timeout, retry policy and declared scientific gate. Gates may use the source evidence explicitly permitted above, never target-query accuracy. Implement all queue decisions and collection in executable code; an LLM session, open VSCode window, browser, or ongoing agent credits must not be needed to continue.

Prefer the research host's existing batch scheduler and respect its allocation rules. Otherwise use an already installed scheduler or a detached setsid/nohup worker with stdin disconnected, absolute paths/interpreter, redirected stdout/stderr, a single-worker lock, atomic state writes and a heartbeat. Do not install a system service, change power policy, request new paid resources or extend an allocation automatically.

Persist a small scheduler configuration with these defaults:
- availability_poll_seconds = 600;
- heartbeat_seconds = 30;
- initial_gpu_wait_hours = 8;
- execution_wall_hours = 8;
- absolute_wall_hours = 16;
- final_collection_reserve_minutes = 10;
- max_jobs_per_gpu = 1;
- max_technical_retries_per_job = 1;
- GPU-command cap and protected collection allowance from Section 3.

Use a monotonic clock for in-process polling and elapsed-time accounting, with persisted UTC deadlines for restart reconciliation. Maintain the 600-second polling cadence while waiting; do not add ten minutes after the end of each scan. Schedule from the prior poll's start, without burst catch-up scans after a delay. A fresh admission check after one of our own jobs finishes is allowed. Heartbeats and responsiveness to pause/stop are separate from GPU scans: do not hide the worker in an uninterruptible ten-minute sleep.

### 9.2 Decide whether a GPU is available

Discover eligible devices by stable GPU UUID, using the existing host configuration and allocation. An idle device is eligible only if this project is authorized to use it; on a managed cluster, obtain the normal scheduler allocation before running GPU commands. Do not assume a device index is stable or that another user's idle allocation is free to use.

Use the host's existing NVML/nvidia-smi facilities. An availability scan may admit a device only when all of the following hold:
1. There is no foreign compute process, no active job of this worker on the device, and no incompatible reservation. Near-zero utilization alone is insufficient: a sleeping training process still owns its GPU.
2. Utilization is below 5% in three observations spanning roughly 30 seconds, taken only when the initial observation suggests availability.
3. Free memory covers the job's measured or conservative peak plus max(2 GiB, 20% of that peak). Establish this requirement from existing comparable jobs or a bounded admitted smoke, not an unbounded speculative run. Account for persistent display/graphics use if present.
4. Device health, allocation, process inventory and memory telemetry are readable and consistent. Unknown/N/A telemetry, unresolved MPS ownership, inaccessible process inventory or stale reservations mean wait and log the reason; they do not imply an idle GPU.

After a device appears available, acquire its existing exclusive UUID lock non-blockingly, acquire the shared ledger mutex, reconcile spend/reservations, and recheck device availability before admission. Use one documented lock ordering throughout. Reserve the full remaining paired block and protected collection cost atomically; avoid double-reserving jobs already covered by that block. Release the ledger mutex promptly. If any check fails, release newly acquired locks/reservations and return to waiting. Hold the GPU lock for the running job and release it when the job ends; never hold a GPU lock merely while waiting for a dependency or another user's job. Advisory locks protect cooperating workers only; the host scheduler remains the authority for exclusive allocations.

Run at most one job per GPU. Independent jobs may use multiple authorized idle GPUs only if dependencies, total reservations and host resources permit. Sum elapsed GPU-command hours across concurrent jobs; parallel execution does not make the cumulative cost equal to wall time. Keep paired speed measurements on the same GPU model/device and under comparable conditions. Do not introduce multi-GPU training just to use spare devices.

### 9.3 Run, wait and recover without intervention

On startup, reconcile existing job receipts and process identities, then run CPU-only reconciliation/collection tasks immediately. Scan for available GPUs. If none is available, enter WAITING_GPU and set next_check_at to the next 600-second tick. Keep checking until work becomes admissible, the queue completes, the user stops it, or a Section 3 deadline expires. Re-evaluate verified spend, external reservations, dependencies and the frozen priority order at each admission. Do not launch a block whose conservative remaining runtime, shutdown margin and collection cannot fit the remaining budget/deadline. A bounded smoke can be a separately budgeted admission to refine the estimate.

Waiting for a busy GPU is not an experiment failure. Keep distinct states for READY, WAITING_GPU, WAITING_DEPENDENCY, PAUSED, RUNNING, COMPLETED, TECHNICAL_FAILED, SCIENTIFIC_GATE_FAILED, UNAVAILABLE, DEFERRED_BUDGET, DEFERRED_BUSY and DEFERRED_DEADLINE. A failed scientific gate has no automatic retry. Allow at most one retry after a diagnosed transient technical failure, with both attempts charged and sufficient remaining budget. Do not restart a poor-performing experiment with changed settings.

Monitor owned jobs frequently enough to enforce timeouts and budget limits, independently of the ten-minute idle polling. Reserve a termination grace margin before a hard limit, stop new admissions in time, and checkpoint/terminate only this worker's own verified processes if needed. Identify a process by PID plus start identity and, where available, boot identity; a reused PID is not one of our jobs. If foreign work appears on a shared device after admission, do not kill or change it. Stop further admissions there, checkpoint/pause our own job where supported, or stop only our own job if safe coexistence cannot be established. Record contention and exclude contaminated latency/energy measurements from performance claims.

Resume must reconcile the ledger, live process identities, output hashes and completion receipts before doing anything. Reattach monitoring to surviving owned jobs instead of launching duplicates. Mark genuinely interrupted jobs and resume from a validated checkpoint only under the declared retry policy. Preserve the original caps and deadlines. On partial failure, budget exhaustion, stop or timeout, run final CPU collection and leave a complete manifest of completed and deferred experiments. If no GPU becomes available within the initial wait window, collect the CPU results and mark GPU jobs DEFERRED_BUSY; do not fabricate a scientific result or silently wait forever.

### 9.4 Visibility, controls and launch verification

Write an append-only gpu_availability.jsonl with check time, eligible UUIDs, utilization, free/required memory, process/reservation availability, decision and reason. Maintain scheduler_status.json with heartbeat, last/next GPU check, active/pending jobs, completed receipts, verified/reserved/remaining GPU hours, launch/first-GPU/deadline timestamps and run directory. Never copy credentials or unrelated command lines into these reports.

Provide working status, pause, resume and stop commands for this exact run. Pause prevents new admissions and lets an active job reach a safe checkpoint/completion; stop prevents new admissions, safely ends only owned work and invokes collection. Logs and these controls must work from a fresh terminal after closing VSCode. Report exact commands, not placeholders. Preserve originals and write the launch receipt into the V9 run directory.

Compile/import, run Section 8's targeted checks, launch the detached worker, then verify its identity, fresh heartbeat and persisted state after the launcher exits. If an eligible GPU is already idle, verify the first admitted job or its completion receipt. If GPUs are busy, verify WAITING_GPU, the recorded blocking reason and the next check scheduled within ten minutes; do not force a job onto a busy GPU merely to prove startup. Report the actual PID/service or scheduler job ID, eligible GPU UUIDs, run directory, budgets/deadlines, status and controls. Report a GPU experiment as started only when its own launch receipt exists.

The worker must continue after closing VSCode while the host and any required allocation remain active. It cannot run through host sleep/shutdown, container destruction or allocation termination. This is a finite local polling worker for this V9 queue; stop it after collection when work is complete or a deadline is reached. Do not create an unrelated ChatGPT reminder or an indefinitely recurring experiment schedule.

## 10. Deliverables and reviewer relevance

Write concise research outputs, not an extensive replacement documentation system:
- frozen_protocol.md, manifest/state/ledger and launch receipts;
- scheduler configuration, scheduler_status.json, gpu_availability.jsonl, targeted scheduler-test results and exact status/pause/resume/stop commands;
- evidence_index.json and episode/pool/selection manifests;
- revision_corrections.csv, reviewer_closure_status.md, same_image_label_swap.csv and alternative_teacher_comparison.csv (with explicit unavailable/deferred states);
- objective_solver_replay.csv;
- adaptation_results.csv and per-event/control summaries;
- support_coverage.csv and correction_harm.csv;
- full_precision_frontier.csv, FP/QAT paired results and backend/timing receipts;
- negative_results.md;
- report_v9.md and a compact reviewer_response_map.md.

The morning report must state what ran, what was reused, what was skipped and why, with observation versus explanation separated. Include an experiment checklist covering every required and conditional block, its paired controls, admitted/completed seed counts, gate or deferral reason, and evidence path. Report GPU wait duration separately from execution time and summed GPU-command hours. If the worker is still within its declared window, publish a clearly labelled interim status rather than claiming the run is finished. Do not count deferred or unavailable experiments as negative scientific evidence.

Reviewer mapping, grounded in the actual letter:
- R1.1/R2.1: technical novelty requires controlled evidence for a specific contribution; familiar ingredients alone are insufficient.
- R1.2/R2.4: existing MobileNet/LightFormer KD/OFA comparisons already address the alternative-student gap; use them rather than restarting.
- R2.3: retain the corrected budget-dependent OFA results without inventing wall-clock time-to-quality.
- R2.5: teacher/MNDWI complementarity and implementable use of disagreement address when learning adds value over a spectral rule.
- R1.3/R2.2: use the existing author clarification, disclose source labels, support selection, unlabelled context, benchmark exposure and geometry/acquisition confounds; add the same-image label-source control.
- R1.4: consolidate the relevant component controls.
- R1.5: distinguish surface-water segmentation, flood-event application, teacher-derived supervision, zero target-label adaptation and one labelled chip.
- R1.6/deployment: independent dry-scene reliability remains unresolved unless suitable verified references actually exist; qualify hardware and mixed precision.

Do not claim this follow-up alone closes all reviewer concerns. Do not alter submitted manuscripts or send anything externally.

## 11. Prior art and claim boundary

Relevant starting points, not an exhaustive novelty clearance:
- EATA, Efficient Test-Time Model Adaptation without Forgetting, ICML 2022: https://arxiv.org/abs/2204.02610 — active sample selection and anti-forgetting regularization.
- SelectNAdapt, ICCV Workshops 2023: https://arxiv.org/abs/2308.04946 — deliberate support selection for few-shot adaptation.
- Meta-learning with differentiable closed-form solvers, ICLR 2019: https://arxiv.org/abs/1805.08136 — fast closed-form adaptation.
- Towards Stable Test-Time Adaptation in Dynamic Wild World, ICLR 2023: https://arxiv.org/abs/2302.12400 — instability and class-imbalance failure cases.

Our candidate hypothesis is specific: heterogeneous S2 expert disagreements can expose missing support regimes; admitting supported corrections and protecting unrepresented or unresolved cases can improve the quality/cost trade-off of an extremely small student. V9 must test that hypothesis against ordinary coverage, agreement gating, global regularization and conventional KD. No first-ever claim and no guaranteed improvement.

