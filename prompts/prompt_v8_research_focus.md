# Codex follow-up: useful teacher corrections, rapid adaptation, and extreme compression

Date: 2026-10-12. This is a research-priority revision of the earlier V8 plan, not a claim that V8 experiments have already run.

## 1. Mission and authority

Work in the existing FloodDistill repository and its established data/compute environment. Implement, validate, launch, and collect a bounded follow-up without depending on this conversation or an interactive VSCode process staying open.

The central question is: **Can a very small student absorb useful, trustworthy teacher corrections with much less adaptation work, and retain those benefits at low precision?** A second teacher is valuable when it supplies complementary correct information, not merely because it is larger or newer.

Prioritize the user's two branches: complementary teaching signals and extreme compression. Preserve established anchors and failed experiments. Distinguish a failed configuration from a disproved research direction. Do not repeat the V7 C3 recipe unchanged. Do not run NAS.

This prompt authorizes ordinary implementation and bounded experiments in the existing environment. Respect repository instructions, existing compute limits, and live processes. Do not install drivers, expand a compute allocation, start paid services, download large datasets/checkpoints, submit a paper, or contact anyone. Do not kill unrelated jobs. Do not spawn other agents.

If a previous V8 worker is running, inspect its immutable manifest and ledger. Reuse its results, avoid conflicting resources, and queue this work after its completion if budget permits. Do not silently edit or restart an already launched manifest.

## 2. Evidence to reconcile before implementation

Locate actual files using repository paths and rg; paths in old reports are hints, not proof of availability. Likely original host repository: /home/schmalstieg/repos/FloodDistill. Read current AGENTS.md files, reviewer letter, manuscript/supplement, prompt5 closure tables, audited V7 report, test.txt, and any newer completed runs.

Create evidence_index.json with absolute paths, hashes, graph identifiers, seed, training budget, data/split identities, checkpoint-selection rules, and known limitations. Do not merge historical original-activation EfficientViT results with the later repaired/capacity graphs without matching identities.

Known audited anchors, to verify against receipts:

- V7 R4 LKD: 12,230 parameters; selected source-validation mIoU 87.94 ± 0.43 across four seeds. R4 cached-hard CE: 88.34 ± 0.38.
- Repaired A0 LKD: 704,714 parameters, 87.32 ± 0.21. Repaired A1: 190,870, 86.74 ± 0.29. A3: 30,922; A3N: 20,250.
- Selected R4-minus-A0 source difference: +0.62 pp, paired-seed 95% CI [-0.04,+1.28]. All four observed paired differences are positive. The interval is uncertainty in the mean difference, not the range of absolute model scores.
- WorldFloods pooled water IoU: R4 LKD 65.95 versus teacher 64.32. Equal-event water IoU: R4 47.59 versus teacher 50.54. These measure different averages; report both.
- All V7 C0–C5 continuation arms selected update zero. C3 final source mIoU was below the matched C0, C4 and C5 controls. C3's weighting was active; normalization did not eliminate changes in gradient magnitude. Retain this negative result.
- CE is focal loss on teacher-generated hard masks, not teacher-free learning.
- The task-adapted Prithvi teacher inherits 252 labelled source-training scenes. Do not label the whole pipeline annotation-free or zero-shot.
- Earlier matched EfficientViT N100 OFA-minus-LKD seed-mean mIoU effects were approximately +0.695 pp S11-test, +0.116 STURM, +0.697 WorldFloods. Full-budget effects varied by dataset/backbone. N100 and N2500 also differed in optimizer work: this is evidence of regime dependence, not yet proof of isolated sample efficiency or wall-clock acceleration.
- All current external benchmarks have influenced research. Preserve that disclosure. No existing external score is newly untouched simply because this prompt freezes a new protocol.
- V7 memory measurements were contaminated by retained tensors, and actual TF32 settings differed from a prose label. INT8 engine construction failed in the recorded CUDA/driver environment. Simulated low bits do not demonstrate deployment speed.

The actual reviewer letter is authoritative, including requests about novelty, budget-dependent OFA benefits, alternative backbones, MNDWI, data roles, component ablations, dry-scene false alarms, terminology, and mixed/hardware-specific deployment.

## 3. Budget and execution priorities

Recompute the current ledger and reservations first. V7 ended at 25.871253 GPU-command hours against a cumulative 30-hour limit, leaving 4.128747 then. Newer work may have consumed that remainder.

Set incremental cap = min(4.0 hours, max(0, 30 - verified spent - other reserved hours)). Count training, teacher inference, evaluation, failed attempts, and GPU preflights consistently with the inherited ledger. This prompt does not reset the cumulative limit. With no remaining budget, complete CPU analysis, implementation, tests, and a blocked/deferred manifest; do not invent extra authorization.

Set an eight-hour wall deadline from this worker's launch. Before launch, freeze a manifest and allow-list of conditional jobs. Suggested maximum GPU-command allocations when four hours are available:

| Bundle | Cap | Priority |
|---|---:|---|
| Teacher/student joint-error analysis and baseline parity | 0.50 h | Required |
| Existing learning-curve recovery and optional ready-teacher screen | 0.40 h | Required CPU; teacher screen conditional |
| Frozen-head rapid-adaptation pilot, including controls | 1.60 h | Main experiment |
| Fixed-graph compression frontier and conditional quantization-aware update | 0.90 h | Main second branch |
| Evaluation, retries, final collection reserve | 0.60 h | Protected |

These are ceilings, not spending targets. Preserve complete control/candidate comparisons before adding more arms. If estimates do not fit, reduce prespecified event/seed blocks symmetrically or defer whole optional comparisons. Never selectively omit a poor arm. Every admitted job needs a timeout and collection reserve. No full foundation-model fine-tuning within this budget.

## 4. First question: is there useful teaching headroom?

Use compatible saved predictions whenever possible; otherwise run one costed common inference pass. Use actual reference labels only for offline diagnosis and evaluation.

For teacher T and student S, tabulate per event and class:

1. both correct;
2. teacher correct, student wrong;
3. student correct, teacher wrong;
4. both wrong.

Report G(T|S) = P(T correct and S wrong) and H(T|S) = P(T wrong and S correct). On the same pixels, teacher-minus-student pixel accuracy equals G-H. This identity does not apply directly to IoU; recompute IoU from confusion counts.

Build an explicitly labelled, unattainable oracle choosing the correct prediction whenever either T or S is correct. Recompute its water IoU and mIoU. Also assess T/MNDWI and, if available, T/T2 and S/T/T2. This oracle measures available complementary information; it is not a deployment method.

Always show equal-event and pooled-pixel results, water recall, precision, and valid support. Use class-conditional counts so land dominance cannot hide water misses. Report source and external datasets separately. Do not infer teacher/student joint errors from teacher/rule tables or aggregate means.

Measure teacher stability under the same prespecified mild radiometric perturbation on TRAIN/support inputs. Compare confident/stable errors with confident/stable correct predictions. Treat stability as a proposed reliability proxy, not evidence of correctness.

Deliver teacher_headroom.csv, joint_confusions.json, event_headroom.csv, and headroom_interpretation.md. Answer:

- Does the teacher know corrections the student has not learned?
- How much harmful imitation is possible?
- Is complementary headroom concentrated in a few events or difficult classes?
- Is an improved router potentially useful, or do all experts fail together?

Do not use external reference labels to choose a deployable gate, threshold, hyperparameter, or support sample.

## 5. Recover the existing OFA story before buying new training

Collect the original matched LKD/OFA learning curves for N100 and N2500 and the alternative-backbone ablations already run. Separate:

- quality versus optimizer updates;
- quality versus total measured training time;
- quality versus number of distinct training scenes.

Time curves require actual training timestamps/receipts. Some prompt5 jobs with names containing "train" are inference/counting commands: inspect argv. Never use inference receipt duration as training time or invent time from update counts.

At prespecified source-validation mIoU thresholds 70, 75, 80 and 85%, report first observed crossing and its measurement interval. Mark unreached thresholds as censored; do not extrapolate. Show the complete curve and selected/final checkpoints. Include teacher forward work and OFA head overhead in measured training cost. OFA's training-only heads must be removed from deployment comparisons.

Existing N100/N2500 results confound data quantity and total updates across N. Within-N matched effects remain useful. If the log coverage cannot establish wall-clock or sample efficiency, say so. Do not spend the remaining budget recreating every historical run.

OFA means the repository's established training method, not Once-for-All architecture search. R4 has no established OFA mapping: do not attach a new head and silently call it the same controlled OFA method.

## 6. Main experiment: one-pass residual correction of a frozen student

### Hypothesis

The mature student may have little remaining benefit from whole-network continuation. A small correction to its final decision layer may absorb useful teacher information while avoiding backbone and batch-normalization drift. We will also test whether that correction can be made insensitive to quantization changes.

This is a new testable combination, not a claim to have invented ridge regression, few-shot distillation, trust weighting, or closed-form adaptation.

Here “one-pass” means one accumulation of cached support statistics followed by one linear solve. It does not mean one total neural-network forward: teacher perturbation views and FP/quantized feature extraction are separately counted. Prefer “single-solve adaptation” in results when that is clearer.

### Architecture check

Inspect the actual R4 graph. If its final two-class head is affine in d feature channels, represent water-minus-land margin as

m(x) = v^T phi(x),

where phi includes an intercept. Verify the feature location, normalization, upsampling and padding. For a linear upsampler, the affine head can be represented using consistently upsampled features. Do not assume a 17-dimensional solve until the actual 16-channel head is verified.

Freeze all feature-extractor parameters and buffers; use eval mode. Assert BN running statistics do not change. Keep the base checkpoint immutable and reset each episode from it.

### Sampling and targets

Use at most 4096 spatially distributed valid support pixels per chip, fixed by the manifest, with equal total chip mass. Record sampling rules and selected indices. Do not treat sampled pixels as independent experimental repetitions.

For zero target-label adaptation, teacher p_T supplies target margin t = clip(logit(clip(p_T,1e-4,1-1e-4)),-8,8). This is a bounded logit-regression target, not an exact KL-distillation objective.

For the teacher reliability proxy, use

w_i = 0.1 + 0.9 [1-H_2(p_T,i)] exp(-|p_T,i-p'_T,i|/0.1),

where H_2 is binary entropy in bits, and p'_T is the teacher probability under one common mild augmentation. Use a training-pipeline gain of +2% on valid reflectance before declared normalization, clipped only to the pipeline's valid reflectance bounds; if this transformation is not scientifically valid for the stored input representation, disable the stability factor and label the arm confidence-only. Preserve masks and geometry. Do not silently choose a different perturbation based on external results.

This proxy can confidently weight wrong predictions. Measure that failure; do not describe it as calibrated trust.

Let alpha_i give each chip equal mass and sum to one over sampled pixels. Normalize alpha_i*w_i to sum to one, giving diagonal W. Let M = diag(alpha_i). No mean-gradient scaling argument is required because the head objective is solved explicitly, and all arms share the same regularization and damping rules.

### Closed-form update

Compute source-training feature means/scales once without labels; apply the same transform to FP and quantized features. Clamp near-zero feature scales to a declared positive floor and absorb the transformation when folding the update into the original head.

Let Phi_q be the frozen deployed-backbone feature matrix, Phi_f its floating-point counterpart, D = Phi_q-Phi_f, and r = t-m_S,q. For ordinary FP adaptation, Phi_q=Phi_f and D=0.

Solve

J(delta) = (Phi_q delta-r)^T W (Phi_q delta-r)
           + lambda ||delta||_2^2
           + gamma (D delta)^T M (D delta).

The unique solution for lambda>0 satisfies

(Phi_q^T W Phi_q + lambda I + gamma D^T M D) delta
    = Phi_q^T W r.

Use float64 accumulation and a Cholesky solve, not explicit matrix inversion. The quantization penalty discourages corrections that change substantially when the backbone is quantized; it does not prove overall prediction invariance or guarantee correctness.

Preselect lambda using source-only development episodes from {0.001,0.01,0.1}; default 0.01 if no eligible development split exists. Use gamma=1 for the proposed quantization arm and 0 for its control. Source development labels, if used to select lambda, must be disclosed as a selection budget, with the same opportunity for controls. Do not train a new reliability predictor on them without opening a separate labelled-data comparison.

Apply common damping eta=min(1, 1/max(RMS(Phi_q delta),1e-12)) so the sampled RMS margin change is at most one. This is a declared post-solve damping operation, not an exact solution to a constrained optimization problem. Record undamped and applied updates.

Fold eta*delta into the existing classifier: add half to water weights/bias and subtract half from land weights/bias, after undoing feature standardization. There are no additional deployed feature channels or extra teacher calls after adaptation. Verify folded predictions against the unfused calculation.

### Required controls and order

First implement FP experiments:

- F0: frozen original student.
- F1: two-parameter margin calibration a*m+b with teacher targets, regularized around a=1,b=0.
- F2: conventional head-only soft KD using 1, 5 and 20 optimizer steps.
- F3: uniform residual ridge, w=1.
- F4: reliability-weighted residual ridge using the formula above.

Use identical support examples, base checkpoint, cached teacher targets, feature tensors and valid masks. Give SGD the same caching opportunity; do not compare cached ridge against unnecessarily repeated teacher inference. Choose its learning rate only on source development episodes from a small declared grid {0.001,0.01,0.1}; freeze it before external evaluation. Use the existing documented KD temperature, or T=1 if absent.

For low precision, run a 2x2 ablation on one admitted R4 precision configuration:

| Arm | Reliability weights | Quantization penalty |
|---|---|---|
| Q0 | uniform | gamma=0 |
| Q1 | proposed proxy | gamma=0 |
| Q2 | uniform | gamma=1 |
| Q3 | proposed proxy | gamma=1 |

Keep the head in FP16/FP32 for this experiment and disclose the mixed precision. All arms use the same quantized frozen backbone and corresponding unquantized checkpoint. Report the extra floating-point feature pass in adaptation cost. Do not pretend this mechanism adapts fully integer weights.

Use these as mechanism controls, not a large hyperparameter search. If uniform ridge provides no useful correction, still complete its paired reliability control when affordable; classify whether teacher targets, head expressiveness, or reliability failed. Do not discard the whole idea simply because one objective did not improve final mean mIoU.

### Meaningful implementation checks

Verify: unchanged frozen features/BN; correct water/land channel orientation; objective gradient close to zero before damping; equivalence of folded and unfused margins; gamma has no effect when D=0; quantizer uses the declared bit ranges; no support/query overlap; invalid pixels do not enter the solve. Use small deterministic tensors and one real-batch parity check. Stop broad testing once these concrete risks are covered.

## 7. Adaptation episodes and honest terminology

Use up to three eligible target events, selected deterministically by metadata/availability before loading their outcome scores. Require at least two geographically disjoint chips, reliable geometry, and no overlapping support/query pixels. Freeze event IDs, support IDs, query IDs and two support draws. For the main pilot use one support chip per event, seeds 42 and 64 where compatible parents exist. This gives up to 12 parent/support/event episodes. Reuse teacher predictions/features across methods. If three eligible events do not exist, report the actual smaller pilot; do not manufacture independent samples.

Track two budgets separately:

A. Zero target-label adaptation: teacher queries on unlabelled support only. No target reference labels used for fitting or hyperparameter selection.

B. One-shot target adaptation: exactly one manually labelled support chip, with its valid labelled pixel count and area disclosed. It is many pixel annotations, not one labelled pixel. Compare direct supervised head fitting against teacher-plus-label fitting under the same annotation budget. Use supervised target t_y=logit(0.95) for water and logit(0.05) for land; teacher-plus-label target is 0.5*t_y+0.5*t_T. Use the same uniform ridge solver first so the teacher's incremental contribution is identifiable. Also include a supervised head-only SGD baseline if the block fits.

A frozen model evaluated directly is zero-shot transfer with respect to that target. A model updated from unlabelled target images is zero target-label adaptation. Neither makes the source-trained teacher label-free. One-pass is a computational property; it is not synonymous with one-shot.

Do not choose query checkpoints or early stops using query labels. All current target sets are research-exposed, so call these exploratory adaptation episodes. Check prior training overlap and report it. If valid disjoint episodes cannot be established, run source-only method checks and explicitly block target-transfer claims.

For each method report teacher query count, target labels/area, trainable parameters, feature extraction time, solve/optimization time, cold and cached end-to-end adaptation time, memory, and query event metrics. A fast linear solve alone does not establish faster end-to-end adaptation.

Treat event as the main transfer unit. Seed/support variability is a separate axis. With only three events, show every event result and descriptive paired differences; do not claim a precise population confidence interval from millions of pixels. Retain all failures and worsening episodes.

## 8. Complementary foundation-model branch

Inventory existing co-registered Sentinel-1/Sentinel-2 data, acquisition dates, preprocessing, licences, and task-adapted teacher checkpoints.

TerraMind with a flood/water head is a sensible first candidate; CROMA is an alternative radar/optical backbone. A pretrained encoder alone is not a water-segmentation teacher. A repository training configuration is not proof a usable task checkpoint is installed.

Admit one second-teacher screen only when compatible data, a usable task head, and bounded inference already exist. Otherwise write exact missing inputs and estimated preparation work, then continue the main student experiments. Do not spend this four-hour remainder training a second foundation model from scratch.

Measure T2's useful corrections relative to T1 and S, not just its mean score. Compare individual experts, a fixed uniform probability average, and a fixed confidence-only mixture before a more elaborate trust method. Diagnose an oracle router separately. Learn any supervised gate only within an explicitly accounted source/support label budget, with an equal-label direct-student control.

If T2 sees SAR while the student only sees optical images, label this privileged-modality teaching. It cannot guarantee recovery of information absent from the student's input. Distinguish optical-only and fused teacher comparisons where available. Generated "SAR-like" features from optical inputs are not independent SAR observations.

A second teacher's failure to improve its own mean does not automatically eliminate it: complementary correct predictions may remain useful. Conversely, disagreement alone is not useful if both teachers are wrong.

## 9. Extreme compression branch

Reuse verified R2/R3/R4/A3/A0 anchors where available and scientifically comparable. Derive actual parameter and serialized-byte counts; retain historical early-stopping exceptions. Do not declare a universal minimum network size from this finite set.

First make a frontier of query/source quality versus parameter count, actual deployed bytes, measured latency and adaptation cost. Keep selected/final and pooled/event summaries separate.

For a prospective source-development comparison, declare quality-loss tolerances of 1 pp primary and 2/3 pp secondary relative to the matched R4 FP parent. These are operational reporting tolerances, not an accepted application's requirements. Show all margins, not whichever makes the candidate pass. For independent later evaluation, paired uncertainty must support the margin; four-seed selected-validation comparisons remain exploratory. Never retroactively rename previous superiority tests as preregistered noninferiority.

Admit FP16 and real INT8 only where the existing backend works. Keep W4A8/fake-quant results explicitly simulation_only unless real kernels are executed. Count scales, zero-points, full-precision heads and file overhead. No theoretical bit packing may substitute for an actual deployed artifact size.

Use the strongest existing conventional calibration/QAT recipe as a control, including documented clipping, stem precision, and faithful LSQ when present. Do not introduce a weak four-bit baseline to make the proposed update look good. If competent QAT would exceed budget, report it as a future prerequisite and run a clearly labelled PTQ/fake-quant adaptation pilot instead.

Do not run every architecture x bit width x method combination. First establish one informative R4 precision point, then the four Q arms in Section 6 if affordable. Retain source Pareto candidates for full event reporting; avoid choosing a different winner per external benchmark.

Any latency/memory comparison used in the new report requires fresh processes, actual precision/TF32 settings, common inputs, synchronization, warmups and repeated measurements. Edge hardware absent means edge latency unmeasured. Do not let infrastructure repair consume the central research experiment.

## 10. Interpretation gates

Predeclare what different outcomes mean:

- Oracle headroom small: this teacher cannot provide many additional hard-label corrections in that evaluation regime; inspect calibration/learning speed and new modalities rather than repeating uniform KD.
- Oracle headroom large but all head updates fail: investigate whether corrections are representable in frozen features, support/query shift, target bias and proxy reliability. It does not prove no useful teacher information exists.
- Uniform ridge helps, reliability weighting does not: retain rapid adaptation; reject this reliability proxy in this setting.
- Reliability helps FP but disappears after quantization: test the quantization penalty and actual feature distortion; do not silently keep only FP results.
- Q3 helps beyond Q0/Q1/Q2 at equal adaptation cost: evidence for the interaction, requiring replication before a strong method claim.
- Calibration F1 matches the proposed method: describe the gain as an operating-point correction; do not claim new representation learning.
- One labelled chip beats teacher-only correction: report the value of annotation honestly.
- OFA reaches useful quality earlier but has equal final quality: a potentially valuable efficiency result, provided end-to-end cost supports it.
- T2 has complementary errors but no reliable implementable gate: potential identified, deployable gain not demonstrated.
- All small models retain source quality but fail some events: a bounded source compression frontier, not universal teacher-equivalent flood performance.

Do not require every exploratory seed to be significant. Also do not use "ceiling" to protect a mechanism from unfavorable evidence. Continue an idea only when the next experiment changes a plausible cause and can distinguish explanations.

## 11. Detached execution and handover

Implement the work as deterministic scripts plus a finite queue. Do not rely on an LLM staying active overnight.

Reuse a verified existing scheduler/queue if present. Otherwise use an already available scheduler or a detached Linux worker (setsid/nohup with stdin disconnected, logs redirected, durable state and an exclusive lock). Do not install services or change login/power policy. A detached local worker can survive VSCode closure; it cannot survive host shutdown, sleep, loss of its container, or allocation termination.

Provide immutable job identities, argv, input hashes, output paths, dependencies, timeouts, budget reservations, atomic status writes, heartbeat, failure reasons and a final collector. At most one automatic retry for an identifiable transient failure; no automatic scientific retries until a positive result appears. Keep failed attempts in the ledger.

Compile/import and smoke-test the scripts, freeze the manifest, launch on the authorized available device, and verify the detached worker actually starts and writes a heartbeat after the launcher exits. If the host/data/GPU is unavailable, leave fully prepared code and explicit blockers, without claiming jobs are running.

Provide exact absolute-path commands for status, logs, pause/resume and stop of this worker only. Record PID plus process identity/start time to prevent killing a reused PID. Make collection run after partial failure and budget exhaustion. No recurring automation is needed.

## 12. Required deliverables and reviewer mapping

Write a new timestamped run directory; preserve earlier reports.

Required outputs:

- evidence_index.json, frozen_protocol.md, manifest.json, ledger.json, status.json and launch receipts;
- teacher_headroom.csv and event-level joint-error counts;
- ofa_learning_efficiency.csv/plots with missing timing clearly marked;
- adaptation_results.csv with every arm/event/seed/support and separate cost components;
- compression_frontier.csv and precision/backend provenance;
- negative_results.md separating implementation problems, protocol problems and unsupported mechanisms;
- reviewer_response_map.md linking exact reviewer points to old evidence, new evidence and remaining gaps;
- report.md explaining what ran, what changed, what did not work, and why the next causal test follows.

Reviewer mapping:

| Concern | Evidence/action |
|---|---|
| R1.1 / R2.1 novelty | State a falsifiable contribution, nearest prior art, controlled solver/trust/quantization ablations; no "first" claim without a much broader review |
| R2.3 OFA only helps at small budget | Existing matched low-data results and measured learning curves; distinguish sample, update and time budgets |
| R1.2 / R2.4 alternative students | Use already completed MobileNet/LightFormer OFA controls, and controlled tiny-graph frontier; disclose graph changes |
| R2.5 MNDWI/practical benefit | Joint errors, difficult-event recall/precision, teacher/rule failure modes |
| R1.3 / R2.2 protocol/confounds | Data-role/overlap table, support/query separation, teacher source labels, geometry and budget confounds |
| R1.4 component ablations | One consolidated table distinguishing inherited and new experiments |
| R1.5 terminology | Surface-water labels versus event flood use; teacher-derived versus teacher-free; one-shot definitions |
| R1.6 deployment/dry imagery | Explicit unresolved dry-scene limitation unless verified independent dry references exist; qualify actual hardware and mixed precision |

Do not claim a new adaptation result resolves dry-scene evaluation or removes previously exposed benchmarks. Draft precise manuscript/response additions when supported; do not submit them.

The immediate user handoff must state what is actually running, the device, cap/deadline, run directory, controls, expected deliverables and exact monitoring commands. The morning report must distinguish observation, explanation, hypothesis and next experiment.

## 13. Primary literature anchors for positioning

These are prior-art starting points, not an exhaustive novelty clearance:

- Bertinetto et al., Meta-learning with differentiable closed-form solvers, ICLR 2019: https://arxiv.org/abs/1805.08136 — fast ridge-based adaptation already exists.
- Gao et al., Enrich, Distill and Fuse: Generalized Few-Shot Semantic Segmentation in Remote Sensing Leveraging Foundation Model's Assistance, CVPR Workshops 2024: https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Gao_Enrich_Distill_and_Fuse_Generalized_Few-Shot_Semantic_Segmentation_in_Remote_CVPRW_2024_paper.pdf — foundation-assisted few-shot remote sensing is existing work; verify full text before making detailed comparisons.
- TerraMind, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html ; official code/configurations: https://github.com/IBM/terramind .
- CROMA, NeurIPS 2023: https://papers.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html ; official code: https://github.com/antofuller/CROMA .
- Tulbure et al., Leveraging AI multimodal geospatial foundation models for improved near-real-time flood mapping at a global scale, 2025 preprint: https://arxiv.org/abs/2512.02055 — TerraMind with paired SAR/optical flood data has already been studied.

The candidate contribution is the experimentally demonstrated relationship between useful teacher corrections, one-pass adaptation, quantization stability and an extremely small deployed student. Merely combining familiar components or changing the teacher name does not establish novelty.

