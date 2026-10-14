# V11: repair what quantization loses; test a tiny model that adjusts without retraining

Prepared 13 October 2026, Europe/Berlin. Execute this prompt in the existing FloodDistill research checkout. This is a bounded experimental add-on to the running V9/V10 jobs, not their replacement. Implement, test, launch the detached queue and verify its live state. Do not stop at a plan. If this environment has reports but no research-host access, produce the implementation and exact launch blocker without claiming remote execution.

## 1. Goal and scope

The research goal remains an extremely small Sentinel-2 water/flood model that retains useful teacher knowledge, survives low precision, and needs little deployment-time adaptation. We need evidence for a specific contribution, not another large benchmark grid or a renamed combination of standard methods.

Run two separate work packages, in this order:

1. **Main bet: quantization-specific teacher guidance.** Does prioritizing the *extra* teacher mismatch introduced by quantization recover accuracy faster or more safely than uniform KD, ordinary hard-pixel KD and ordinary FP/quantized disagreement?
2. **Smaller, higher-risk bet: a source-trained, image-conditioned classifier.** Can a fixed tiny backbone use inexpensive image statistics to adjust its final classifier, with no teacher query, target label or optimizer at deployment? Run only the bounded probe and its controls in Section 6. Never combine it with package 1 during this run.

The first changes training only and leaves the deployed graph unchanged. The second adds a small explicitly counted conditioner. Neither is presumed successful or unprecedented.

Read AGENTS.md, current git status, the active worker manifests and these local documents:
- `prompt_v10_cpu_followup.md`: its supplied header is **Revision 3**, newer than `prompt_v10_cpu_followup.rev2_20261013.md` and `.rev1_20261013.md`;
- the actual frozen V9/V10 protocols, run receipts, source-selection records and ledgers;
- the V8 audited report and Opus reviewer wrap-up, where available.

Use active frozen manifests to establish what is actually running. The V9 prompt in this chat and the later host-side V9 may differ. Do not infer completed experiments from a prompt or filename.

Reuse loaders, evaluators, quantizers, label guard, teacher logit cache, queue code and `experiments/revision/v10_reference/`. Put new code/results in `experiments/revision/research_v11_<UTC>/`, with a small reusable implementation module if needed. Preserve current code and all older evidence. Do not reset the working tree, amend running jobs, alter V10's manifests, install infrastructure, download data/models, train another teacher, write manuscript text or send messages. No NAS, new OFA sweep or backend-repair project.

## 2. First produce a dependency and evidence ledger

### 2.1 Avoid duplicate work

Inventory completed, active, queued, reserved, failed and deferred jobs. Fingerprint scientific identity using architecture/graph, parent hash, cohort, seed, augmentation, objective coefficients, optimizer/schedule, quantizer/scales, precision map, update count and evaluation protocol. Reuse a result only when the relevant fingerprint matches. A differently named job is not a different experiment. A planned V10 optional arm is still overlapping work; wait for its receipt or report it unavailable rather than launch it again.

V10 Revision 3 already covers:
- size × SUP / teacher-hard / offline-KD, including R4_micro;
- R2 log inputs, local context means, hard versus easy negatives;
- the size/bit-width frontier, range calibration and layer protection;
- FP continuation, parent-KD QAT and teacher-KD QAT;
- mixed-pixel shoreline targets, stronger-index edge decoding and threshold controls;
- supervised large-model comparators and strict INT8 runtime checks.

Do not expand any of those grids here. Use them as references and prerequisites.

### 2.2 Carry forward the actual lessons

V8 found useful teacher/student complementarity but unsuccessful transfer in its small pilot: uniform ridge about −0.378 pp query mIoU, reliability-weighted ridge about −0.552 pp, and 20-step head KD about +0.079 pp. These are not reasons to discard all selective transfer. They are reasons to avoid treating confidence as correctness, avoid unsupported global updates, separate objectives from solvers, and measure harmful corrections.

V8 W4A8's small mean penalty hid opposite event effects. A mean near zero does not establish preservation on every event. V10's reported early readouts remain reported context until matched receipts are found. OFA's historical controls should be retained with their budget/protocol qualifications; do not restart OFA experiments.

Write an **analysis sidecar**, without changing the ongoing run, for these issues:
1. The prompt5 `current_full_summary.csv` distinguishes B0 matched KD N2500 S11 mIoU **0.8781847837** from N100 **0.7840466290**. The latest V10 prose assigns the latter to N2500. Verify the rows on the host; do not discard a valid N2500 comparator because of this mix-up.
2. A teacher need not be rotation/flip equivariant. Jointly transforming a cached target is a well-defined training target, but need not equal a fresh teacher prediction of the transformed input. Measure the mismatch; do not enforce a false equality or silently regenerate the running cache. FP16 cache checks need dtype-aware tolerances.
3. Shared mistakes and reference-based oracle removals do not prove a fundamental ceiling of six-band optical sensing. Call them shared-error and reference-conditional repair diagnostics.
4. Retain joint teacher/FP/quantized correctness codes. Teacher-inherited and compression-induced error categories can overlap.
5. Test-score disappointments trigger provenance investigation, not seed deletion, score-based exclusion or retraining until a target score is restored.
6. Similar scores after threshold matching are compatible with an operating-point explanation, not proof of causal equivalence. SUP versus GDACS teacher training changes several factors; a null soft-KD increment does not isolate data as the sole cause.

Do not spend the night re-auditing the entire project. Resolve these from existing tables and small parity checks, then proceed.

## 3. Data roles, freeze and inference limits

Freeze code, matrix, parent choices, seeds, source gates, resource caps, selection rules and evaluation membership before V11 training. Log what previous external results were already known.

- GDACS N2500 TRAIN and its verified teacher logits: optimization, feature statistics and fixed pixel sampling. Use the same teacher and clean-target augmentation convention as V10.
- S11 validation: source-only development, gates, checkpoint and regularization selection. Disclose this label use. Group by event; do not invent millions of independent samples from its pixels.
- S11 test, STURM and WF exposed tests: evaluation only.
- V9/V10's WorldFloods sealed membership: reuse unchanged if available, but assume **another non-confirmatory research readout** unless an access audit establishes that it was still untouched by all relevant design decisions. A new protocol name cannot reseal data. Do not inspect previous sealed scores to choose a method, precision or continuation.

Training/selection functions must not open target labels or evaluator output. Log evaluation label access. Start one final V11 evaluation phase only after all admitted models/settings are frozen; stream all selected variants over the same tiles where practical. Source diagnostics may run earlier. Do not externally evaluate every intermediate checkpoint: retain the fixed 600-step primary endpoint, a source-selected endpoint, and the 1560-step endpoint only if the complete extension ran. Deduplicate identical states and protect the complete primary matrix before admitting secondary readouts.

Honor native dataset water definitions, valid masks, event grouping and interpolation. Do not call an index/teacher disagreement a true false alarm without human/reference labels. Use only deployable input masks for any runtime descriptor.

## 4. Main method: teacher loss added by quantization

### 4.1 Intervention and score

For augmented input x, frozen cached teacher logits z_T, current trainable student weights theta, and T=4, define

    q   = softmax(z_T / T)
    p_F = softmax(z_theta,quantization_bypassed(x) / T)
    p_Q = softmax(z_theta,quantization_enabled(x) / T)

The two student paths must use **the same current weights and input**, with identical normalization and stochastic state. The FP shadow is not the original frozen parent. Bypass both activation and weight fake quantizers. Scoring must not update observers, BN buffers, dropout state, optimizer state or the teacher cache. Reuse the normal QAT path for gradients; detach the scoring shadow and all selection/weight calculations. If the existing code cannot perform a clean paired intervention, fix that small interface before claiming this score measures quantization.

For each valid pixel i:

    d_i = T^2 * [ KL(q_i || p_Q,i) - KL(q_i || p_F,i) ].

Positive d means quantization increased teacher-matching loss; negative d means it decreased it. This is a controlled effect on that loss, **not evidence that the teacher is correct**, and not a detector of every ground-truth compression error.

For binary softened margins a_F, a_Q and teacher water probability q, verify the equivalent stable expression:

    d / T^2 = softplus(a_Q) - softplus(a_F) - q * (a_Q - a_F).

For a small margin change delta=a_Q-a_F:

    d / T^2 ≈ (p_F - q)*delta + 0.5*p_F*(1-p_F)*delta^2.

The subtraction tests whether baseline teacher mismatch should be removed before allocating extra supervision. Do not market ordinary disagreement weighting as the contribution.

### 4.2 Loss allocation, fixed in advance

Within each image's valid pixels, select at most the highest-scoring 25% with d>1e-8. Use deterministic ties. Let N be its valid count, k the selected count, and m_i its binary selection mask. Set

    w_i = (1 + 3*m_i) / (1 + 3*k/N).

Thus the valid-pixel mean weight is exactly 1, every pixel retains positive KD, and max/min weight ratio is at most 4. If k=0, use uniform KD. Keep V10's ordinary valid-pixel aggregation across images; weight normalization must not accidentally reweight images with different valid counts.

Replace only the KL term:

    L = 0.5 * focal_gamma2(hard_teacher_mask, student_Q)
        + 0.5 * T^2 * mean_valid[ w_i * KL(q_i || p_Q,i) ].

Use V10's exact focal implementation, preprocessing, T, augmentation, optimizer, clipping and quantizer conventions. Keep the hard-loss coefficient unchanged across arms. Do not add a confidence gate, boundary gate, anti-forgetting loss or extra hyperparameter search. Positive supervision outside the prioritized subset is not a mathematical guarantee of unchanged predictions there.

### 4.3 Minimal, necessary controls

Core point: **R4 W4A4 body / W8A8 stem / FP32 head**, four seeds 42,64,91,107. Use the same FP parents, starting quantizer state and schedule as V10's corresponding continuation where verified. Do not substitute a different parent because it performs better externally.

| ID | Training rule | What it rules out |
|---|---|---|
| Q0 | unchanged starting PTQ state | change caused merely by continued training |
| F0 | FP continuation with U's same teacher-KD objective | improvement available without quantization |
| U | uniform teacher KD | ordinary V10 teacher-QAT explanation |
| D | proposed positive extra-loss priority | candidate mechanism |
| A | priority by absolute KL(q || p_Q) | ordinary hard-pixel distillation |
| L | priority by ||p_Q-q||_2 | a simple near-prior disagreement score |
| N | priority by ||p_Q-p_F||_2 | numerical displacement without external-teacher direction |
| P | matched randomized priority | generic reweighting and class-balance effects |

L is a **score control inspired by QAT-ACS**, not a reproduction of its full sample-coreset algorithm. Existing V10 parent-KD QAT is an additional reused reference, not another required new sweep.

Run D first for each seed and persist, per step/image, its prioritized counts within teacher-predicted water and land, plus batch/augmentation identity. A/L/N choose their highest scores within each such class using those exact counts; P draws a deterministic random subset within each image/class with those counts. All use the same weight formula. This matching uses TRAIN traces only, no labels or performance scores. It prevents a water-prevalence shift from masquerading as a superior score. Own-model scores may evolve in each arm; record that clearly.

Independent arms must share initial weights, optimizer initialization, input order, augmentation stream and precision map, with all four seeds retained. Exact V10 controls can be reused only at matching updates; a 1,560-step final state is not a 600-step matched baseline. If additional new control continuations are needed, keep them short and explain the mismatch rather than silently reuse incompatible results.

### 4.4 Schedule and bounded extension

First reproduce source FP/PTQ scores. Admit the core training block only if the starting point has either:
- seed-mean S11-validation equal-event water-IoU loss of at least 0.3 pp versus its FP parent; or
- at least +1 pp absolute dark-land false-water rate or bright-water miss rate, with a nonzero relevant denominator and at least 1,000 contributing reference pixels overall.

These source criteria test whether there is meaningful damage to repair. If neither holds, retain the diagnostic and mark the point NO_MATERIAL_SOURCE_DEFICIT, not FAILED. Do not create a harsher quantizer merely to manufacture a result. The fixed second point below may still qualify.

Core training: at most **600 updates**, with source checkpoints at 0,50,200,600. This is 24 QAT continuations (six training rules × four seeds), plus four F0 controls only if missing. Profile a 50-update smoke and reserve the complete matched block plus evaluation before admission. Preserve the existing learning-rate schedule by original update index; do not compress a longer cosine schedule into 600 steps.

For the matched-update main comparison use step 600 for every arm and seed. Separately, each arm may select one common update across its four seeds by maximum four-seed mean S11-validation equal-event water IoU over its scheduled source checkpoints; exact ties choose the earliest update. Never compare best-D with final-U under a matched-update label. The 1560-step extension is a secondary endpoint and does not replace the fixed 600-step primary result. An incomplete four-seed block is reported as incomplete, not silently redefined.

One predeclared extension to 1,560 total updates is allowed only if all required source 600-step results exist, D or U improves by at least 0.25 pp from step 200 to 600, neither D nor U reaches the 1 pp source tolerance, and the complete all-arm/all-seed extension fits the budget. Extend every QAT rule, plus its matched F0 if needed; no winner-only extension. Otherwise stop at 600 and distinguish a bounded negative pilot from a proof of impossibility.

Optional transfer point: V10's already source-selected smallest viable model at its declared W4A8 precision; if its identity is R4, this means R4 W4A8. Wait for the valid selection receipt rather than make a competing size choice. Apply the same deficit gate and full control matrix at 600 updates. Admit it only after the core is complete or was inapplicable and after reserving final evaluation. No architecture/precision search beyond these two fixed points. Do not choose this branch from external scores.

### 4.5 Mechanism and efficiency readout

On S11 validation, diagnose whether d prioritizes actual quantization damage using human labels **only in this readout**. Preserve the joint correctness code of teacher, current FP shadow, starting FP parent and quantized model. Quantization damage relative to the same current weights and regression relative to the starting parent are different comparisons.

Report:
- selection coverage and positive/negative d mass by teacher class, dark/bright/shoreline strata and event;
- selection precision/recall for FP-correct/Q-wrong pixels and for FP-wrong/Q-correct pixels, with denominators;
- source-only oracle selection as an explicitly undeployable diagnostic, never a training arm;
- before/after corrected and newly broken reference pixels, false-water and missed-water separately;
- teacher-wrong selected cases, so confidence/teacher imitation cannot hide harm.

For each arm report quality versus updates **and actual cumulative CPU seconds/core-hours**, including scoring, the extra FP forward, sorting, data access and teacher-cache generation cost with amortization shown separately. Do not pad U with unnecessary FP forwards in the practical timing comparison. Training-side FP shadows add cost; pixel weighting does not reduce network FLOPs.

Use first evaluated checkpoint reaching the same source target: within 1 pp water IoU of the starting FP parent. Report interval censoring between evaluations and NOT_REACHED; never fabricate an interpolated crossing time. If Q0 already meets the target, time-to-target is zero and offers no efficiency claim. Also report quality at equal measured time using completed checkpoints, not extrapolation.

Distinguish observed learning-curve time-to-target from total cost of this completed experiment. Source-checkpoint selection requires its validation/selection cost; do not report only the selected checkpoint's prefix as the total cost of a run that actually continued to 600 or1560.

Package 1 warrants follow-up only if D beats the relevant simple controls, rather than merely Q0. A source-screening signal is D−U and D−A at least +0.25 pp, signs positive in at least 3/4 seeds, and no greater than +1 percentage-point dry-window-alarm increase. Report N/L/P regardless. These rules cannot select or extend anything after external evaluation. A claimed speed benefit additionally requires lower end-to-end time at matched quality; a quality-only result is reported as such.

## 5. Meaningful tests before the main block

Reuse passing V10 tests, adding only tests for actual changes:
- vector KL difference equals the binary softplus expression, including T^2;
- identity/bypassed quantization yields d=0 and exact uniform-loss fallback;
- construct one perturbation toward q and one away from q and verify signed scores;
- weights have mean1 on valid pixels, bounded ratio, correct empty/single-class behavior and deterministic ties;
- invalid/nodata pixels cannot enter scores, masks or denominators;
- D and all controls reproduce per-image/per-class counts and augmentation hashes;
- no gradients enter scores/FP shadow; Q receives finite gradients; scoring leaves BN/observer/RNG state unchanged;
- paired FP/Q paths have identical current weights and normalization state; shadow bypass covers every fake quantizer;
- all branches share declared starts and precision maps; final runtime artifact uses the declared quantizer rather than disabling it for evaluation;
- cache alignment/interpolation is correct; distinguish cache serialization error from teacher augmentation non-equivariance.

If a source-parity discrepancy is genuine, fix it in the new snapshot and rerun only affected smokes. Never modify a running V10 cache in place.

## 6. Independent optional bet: zero-optimizer image conditioning

This is a source-trained dynamic classifier, not one-shot labelled learning and not test-time optimization. Its appeal is zero target labels, zero target teacher queries and no iterative adaptation. Dynamic filters/hypernetworks are prior art; the proposed test is whether a very small conditioner moves our actual tiny-model accuracy/cost frontier.

### 6.1 Fixed prototype, not another architecture grid

Use **R4_micro KDoff**, four seeds, from V10 A5 if completed and verified. Its small decoder makes the conditioner inexpensive. Use FP32 for this first representational test. Do not substitute whichever architecture won externally, retrain the backbone, or combine with package 1. If these checkpoints are unavailable, mark the probe WAITING_DEPENDENCY/DEFERRED and continue the main work.

Keep the backbone frozen. Let h_i be the native final-head feature vector and m0_i its original water-minus-land margin. On GDACS TRAIN, estimate fixed feature means/scales and define phi_i=[(h_i−mu)/s;1]. Let z_t be the six input-band means and six standard deviations of tile t, standardized using TRAIN statistics and clipped to[-3,3]. Use only the existing input nodata/validity mask. Never use reference cloud masks, true water fractions, labels or target-fitted statistics in z.

Freeze descriptor geometry: compute input summaries on each canonical inference tile, with the same field-of-view/resize/padding policy used for its TRAIN descriptors. In particular, verify whether V10's 224px training views are crops or resizes of 512px tiles. Do not pretend summary statistics from different geographic fields of view are identical. Include a source 224-versus-512 descriptor/output audit under the deployable policy. If aligning geometry would require replacing the dataset pipeline, defer this probe.

Define

    m_i = m0_i + phi_i^T (a + H z_t).

With 12 descriptor components, the extra trainable coefficient count is13*(d+1), where d is feature width. Require this to be≤ 512; include stored statistics, buffers and all extra runtime operations in size/cost. No PCA, rank or descriptor sweep.

### 6.2 Fit once on source; controls that identify the mechanism

Use one fixed hash-selected subset of up to 512 GDACS TRAIN tiles, at most 256 valid sampled pixels per tile, with identical rows for all controls and seeds where geometry permits. Sample by coordinates/input validity only, never by target labels. Weight tiles equally, pixels equally within tile. Extract each parent's features once and reuse. No query-time fitting.

| ID | Form | Question |
|---|---|---|
| C0 | original frozen model | reference |
| C1 | m0 + phi^T a | ordinary static head refit |
| C2 | m0 + phi^T a + beta^T z | image-dependent threshold plus static refit |
| C3 | m0 + phi^T(a+Hz) | full feature/context interaction |
| C4 | same full form, TRAIN descriptors deterministically shuffled among tiles | extra capacity without the correct image/context pairing |

C2 already equals an additive [phi,z] model; do not create a duplicate additive arm. For C4, train with a fixed nonidentity permutation of descriptors, then use the actual input's descriptor at inference; disclose this negative-control distribution mismatch. It is an equal-parameter association control, not a complete static-architecture benchmark.

Fit every nonbase form to the same soft teacher targets with regularized convex logistic KD. With normalized row weights r_j, psi_j the appropriate fixed design row, theta its coefficients and T=4:

    J(theta) = T^2 * sum_j r_j * BCE(q_j, sigmoid((m0_j+psi_j^T theta)/T))
               + lambda/2 * ||theta||^2
    gradient = T * sum_j r_j*(p_j-q_j)*psi_j + lambda*theta
    Hessian  = sum_j r_j*p_j*(1-p_j)*psi_j*psi_j^T + lambda*I.

Teacher entropy is constant, so BCE gives the same optimizer as KL here. Use lambda in{0.001,0.1,10}, select separately for each form using S11-validation equal-event water IoU averaged over the four seeds; ties choose stronger regularization. Fit with≤ 10 damped Newton iterations and backtracking; record gradient norm, objective decrease and any nonconvergence. This is an ordinary convex readout fit, not a novel solver. Do not describe it as an exact closed-form probability-KD solution.

The TRAIN-only fit differs from V8's one-support-chip squared-residual update. Still, its failure is possible and informative. Preserve all controls and seeds. Budget this entire probe, including cached features, fits and source selection, at **12 reserved CPU-core-hours**; if estimated to exceed that, defer it instead of dropping controls.

For deployment, let c=a+Hz. The exact native-feature correction is

    delta_w = c_features / s
    delta_b = c_bias - dot(delta_w, mu).

For a two-logit head, add half this correction to the water row and subtract half from the land row. Verify explicit versus folded margins. The folded coefficients vary by tile; this is **not one globally constant head**. Benchmark descriptor computation and per-tile head construction plus inference. An FP conditioner is not a strict-integer model. Do not export/rewrite a backend overnight.

Essential tests: zero coefficients reproduce C0; H=0 reproduces C1; bias-row-only H reproduces C2; finite-difference gradient/Hessian check; damped-step objective decrease; folded/explicit margin parity; immutable backbone/BN state; TRAIN-only statistics and input-only masks.

Promotion criterion for future work: C3 improves source equal-event water IoU by at least 0.5 pp over C0 and0.25 pp over both C1/C2, signs consistent in at least 3/4 seeds, with no greater than +1pp dry-window-alarm increase. Report C4, total bytes/MACs and measured end-to-end latency. If C2 matches C3, interpret the effect as conditional calibration. If C4 matches C3, there is no convincing evidence that correct image conditioning matters. Do not add target adaptation, new descriptors or a second conditioner overnight. An accuracy improvement that costs more than a larger V10 model is not a new efficiency frontier.

## 7. Final evaluation and claims

Use the existing evaluator and reference definitions. Read all required arms on identical event membership. Report:
- equal-event water IoU (primary), pooled water IoU and mIoU;
- per-seed means/SD and paired seed differences;
- event-bootstrap 95% intervals of seed-mean paired differences, 10,000 draws with seed 1729; keep events clustered and distinguish this uncertainty from training-seed variation;
- dry-window alarms at the established 0.1/1/5% area thresholds, dark-land false-water rate, dark/bright-water recall, shoreline performance and denominators;
- per-event correction/harm, all unfavorable events, worst event and a prespecified lower-tail summary;
- calibration/threshold-matched baselines from source-only selection, plus AP to help distinguish ranking from operating-point changes;
- actual bytes, precision map, operations, working memory and end-to-end CPU time, distinguishing simulator quality from genuine low-bit kernel speed.

For package 1, the two main external contrasts are D−U and D−A at the fixed 600-step endpoint. Source-selected and extended checkpoints are secondary, explicitly labelled comparisons. Report L/N/P as indispensable mechanism controls. If making a positive multi-contrast statement, use paired event-bootstrap simultaneous intervals (or Bonferroni 97.5% intervals for the two co-primary contrasts), alongside ordinary 95% descriptive intervals. Package 2 is an exploratory independent probe. No discovery label should override previous benchmark exposure.

A high average accuracy and a negative *difference* interval are compatible: the interval compares methods, not absolute performance. Show raw scores and paired differences side by side. Non-significance is not equivalence, and a small mean is not evidence of no event-specific damage.

Classify results by cause:
- **No repair headroom:** existing compression is already good under the source gate; do not push further bits opportunistically.
- **Good score, bad learning:** d detects FP-correct/Q-wrong pixels but training fails; inspect gradient/update trajectory and target mismatch, not a new weighting grid.
- **Bad score:** prioritization tracks teacher errors or irrelevant numerical changes; retain the falsification.
- **Learning gain, no speed gain:** FP-shadow overhead cancels fewer updates; claim quality only, if supported.
- **No gain over ordinary disagreement:** subtraction adds no demonstrated value here; stop this recipe.
- **Conditioner reduces to thresholding:** keep the simpler interpretation.
- **Insufficient execution/data:** unavailable/deferred/nonconverged is not a negative scientific result.

Do not promise reviewer closure. Package 1 can support a narrow training-metric contribution only with controlled gains. Package 2 can support a small dynamic-model result only with a real accuracy/cost improvement. Existing V10 remains the main evidence for what distillation buys, alternative models, and the size/precision frontier.

## 8. Autonomous resource-aware execution

Use the existing detached worker framework. Write a frozen finite DAG whose source-only gates can execute without an LLM session or remaining agent credits. Reuse control commands and receipts; do not create a replacement scheduling system.

**CPU-first budget:** at most 8 V11 CPU threads, preferably one 8-thread job or two 4-thread jobs. Keep combined V9+V10+V11 allocations below 40 of the reported 48 cores, and reduce to the actual host's safe capacity if different. Account for dataloader/BLAS/OpenMP subprocesses, RAM and IO, not just a process's nominal thread setting. Give existing jobs priority; do not change their thread counts or kill them. Set thread limits explicitly for V11.

Admission must be atomic through the existing shared CPU-reservation lock/ledger or scheduler allocation, with dependencies/RAM/reservations rechecked before launch. If V9/V10 do not participate in a common reservation mechanism, a new V11-only lock cannot protect against their launches: conservatively reserve their configured maximum active-worker thread allocations, or wait for their terminal receipts. Do not assume their currently low utilization will persist. Release resource reservations while merely waiting, while keeping the planned experiment's internal cost reservation distinguishable from hardware occupancy.

Cap this add-on at 64 reserved CPU-core-hours (sum of job elapsed hours × allocated threads), including teacher inference if any, and record measured process CPU time separately. Suggested maximum reservations: 6 for reconciliation/smokes,28 for the core,12 for the independent conditioner,14 for evaluation,4 for collection/retries. These are ceilings, not spending targets. Optional extension/second point may use verified freed capacity only after final evaluation is protected.

Enforce `completed charged cost + active-job accrued cost + reserved remaining cost <= 64` atomically, without double-counting a block reservation when it becomes an active job. Charge smokes, retries and interrupted attempts; reconcile on restart and completion. Reserve each whole all-arm/all-seed block and protected evaluation before admission. Monitor owned jobs independently of idle polling; stop new admissions before limits and checkpoint/terminate only verified V11 work with a reserved shutdown/collection margin. An outstanding-reservation total alone is not the cumulative budget.

Allow up to 8 hours to obtain resources/prerequisites, then up to 8 hours from the first compute-heavy job, with an absolute 16-hour limit from worker launch. Persist all deadlines; restarts never reset them. Lightweight inventory runs immediately. Waiting after the first compute-heavy job consumes its execution window. Reserve the final 30 minutes for collection. If everything finishes earlier, collect and exit.

**Check availability immediately and every 600 seconds while waiting.** Check V9/V10 worker status, compatible reservations, CPU load, free RAM and completion/hash receipts for dependencies. Copy/read only atomically completed inputs; never open a partially written checkpoint or teacher cache. WAITING_RESOURCES and WAITING_DEPENDENCY are valid healthy states. Persist last/next check and reason. Heartbeat every 30 seconds; respond promptly to pause/stop instead of sleeping uninterruptibly for ten minutes.

**GPU:** no GPU training is needed or permitted by this prompt. If existing authorized inference is genuinely necessary and CPU/cache reuse cannot finish it in the bounded allocation, queue only that inference under V9's unchanged shared ledger and admission rules. No reset of the 30 GPU-command-hour cumulative cap. Recompute verified spend/reservations; never assume the old 4.03 hours remain. While waiting, inspect authorized GPU UUIDs every 10 minutes, require no foreign compute process, adequate memory plus margin, sustained low utilization and readable telemetry, acquire the existing UUID lock and recheck before launch. An idle unallocated device is not authorized. Count concurrent GPU-command hours additively. Do not seize a device from V9/V10, cancel their reservations or introduce a second ledger. If no GPU capacity remains, continue CPU work or defer inference explicitly.

Order:
1. inventory, independent sidecar corrections, freeze, targeted tests and source baseline gate;
2. main complete paired block, using dependency-ready inputs only;
3. source-only extension/transfer-point decision and the separately budgeted conditioner; skip/defer optional work that threatens evaluation;
4. freeze selected/final checkpoints and execute the single final external evaluation phase;
5. collection, report and exit, also after partial failure, deadline or budget exhaustion.

Use an already installed scheduler or setsid/nohup with stdin disconnected, absolute paths/interpreter, redirected logs, exclusive worker lock and atomic state files. Verify current interpreter/dependency paths rather than assuming the older machine path exists. At most one retry for a diagnosed transient technical failure; failed science is not retryable. Resume by PID plus start/boot identity, receipts and hashes, reattaching to surviving owned jobs rather than duplicating them. Stop/pause affects only verified V11 processes. Never terminate another user's job. No power-policy changes or allocation extensions.

Verify launch after the launcher exits: a live heartbeat, correct identity, immutable manifest, and either an admitted job receipt or a healthy WAITING state with next check≤ 10 minutes away. Report exact status/pause/resume/stop/log commands usable from a fresh terminal. Closing VSCode must not stop the detached worker while the host/allocation remains active. Host sleep/shutdown or loss of the allocation can stop it; do not claim otherwise.

Add mocked-clock tests only for scheduler changes: busy→600-second recheck→idle admission; unknown telemetry→wait; lock/recheck race; no double reservation; no duplicate after restart; deadline and stop collect partial results. Reuse existing tested behavior rather than rewrite it.

## 9. Deliverables and morning answer

Keep outputs compact and reviewable:
- `frozen_protocol_v11.md`, fingerprints/dependencies, code hash and launch receipt;
- `history_corrections.md` with verified row/cache/claim issues and evidence paths;
- source-gate/selection receipts, tests and input/cache identities;
- `priority_diagnostics.csv`, `repair_learning_curves.csv`, `repair_by_event.csv`, `correction_harm.csv`;
- optional `conditioner_results.csv`, coefficient/statistics inventory and full cost measurements;
- `contrasts.csv`, `resource_ledger.json`, worker status, availability log and controls;
- `negative_results.md` and `report_v11.md`.

The report must answer in plain language: what was tried, what it newly tests beyond V10, what ran versus waited/deferred, what improved and worsened, whether the cause was quantization-specific selection or a simpler control, and what should happen next. Include all seeds, absolute scores and paired differences, actual resource costs, benchmark exposure and precise evidence paths. Clearly mark an interim report if the worker is still active. Do not write manuscript claims or modify V9/V10 outputs.

## 10. Closest prior work and novelty boundary

These are starting references, not exhaustive clearance:
- Huang et al., **Efficient and Robust Quantization-aware Training via Adaptive Coreset Selection**, TMLR 2024: https://arxiv.org/html/2306.07215v3 and https://openreview.net/forum?id=4c2pZzG94y. Disagreement-based QAT selection is established. Our proposed distinction is subtracting teacher mismatch to the same current FP shadow; the near-prior score and absolute-KL controls are required.
- Zhang et al., **GHOST**, TGRS 2023: https://arxiv.org/abs/2301.00131. Guided quantization and selective self-teaching already exist in remote sensing. Do not claim their general combination as new.
- Kim et al., **Teacher Intervention**, EACL 2023: https://aclanthology.org/2023.eacl-main.64/. Teacher assistance for faster low-precision convergence is established; our tested intervention reallocates output-loss weight.
- **Dynamic Filter Networks**, NeurIPS 2016: https://arxiv.org/abs/1605.09673. Input-conditioned weights are established. Package 2 tests a tightly bounded use in our small S2 models, with conditional-threshold and static-head controls.

No first-ever claims. A successful specific subtraction rule or a measured new tiny-model frontier may justify a contribution after further validation. A clean negative result should tell us which mechanism failed without erasing the strong existing compression results.
