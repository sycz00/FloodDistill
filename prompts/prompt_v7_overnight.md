# FloodDistill v7 — a bounded, autonomous overnight investigation

You are working in the existing FloodDistill repository on its training host. Implement this plan, validate it, and **launch the experiments**. Do not stop at proposing commands or writing a handover. Make ordinary implementation decisions autonomously within this contract. Read applicable repository instructions first. Do not request additional compute, external services, credentials, or new data.

The user calls the preceding package “v6.” Its authoritative report is named `report_v5_codex_audit_20261011.md`, inside `experiments/revision/capacity_v4_20261008/`. Preserve those historical names. Call this new investigation v7 and create a unique timestamped output directory. Never overwrite old protocols, source capsules, receipts, checkpoints, queues, or failed attempts.

## 1. Objective and scientific position

Answer three questions, in priority order:

1. Does the small-model result survive a better-trained large baseline, and does R4 still benefit from soft distillation under a competent common recipe?
2. Can R4 benefit from a cheap spectral-rule contribution and extra teacher supervision where that rule disagrees with the teacher? Separate the effects and compare against shuffled and uncertainty-based weighting.
3. What accuracy, storage, runtime memory, and latency do the selected models actually achieve on a supported inference backend?

This is a fixed set of declared architectures and interventions, **not NAS**. A negative result that rules out a mechanism is a successful research outcome. Do not chase external benchmark scores or invent replacement architectures during the night.

Build all code, scheduling decisions, gates, collection, and report generation before detaching. Overnight work must run as ordinary Python/shell processes without further Codex calls, tokens, chat messages, or an open VSCode window. A prompt alone does not keep an agent alive. Detachment survives closing the editor or disconnecting its terminal; the execution host must remain powered, awake, and available.

## 2. Evidence to inherit, not rerun indiscriminately

Read the latest audit and these artifacts before changing code:

- `audit_20261011/t0_verification.json`
- `audit_20261011/report_completion_receipt.json`
- `audit_20261011/supplement_summary.json`
- `audit_20261011/selected_and_final_source.csv`
- `v5/analysis/summary.json` and `cells.csv`
- `protocol_v4.yaml`, `protocol_v5_addendum.yaml`, `simulator_adoption.json`
- The adopted simulator, `code/v5/train_v5c.py`, `code/v5/rungs.py`, the frozen loss/data code, and `v5c/revision_queue.py`.

Resolve paths against the real checkout; do not assume archived absolute paths still exist. The archive refers to `/home/schmalstieg/repos/FloodDistill` and an `esi` environment, but verify both locally. Do not execute imported scripts that mutate the frozen capsule as a side effect.

The October 11 audit reports:

- All historical queues finished. T1–T4 follow-ups were not queued. Older handover text and budget ledgers can be stale. Inspect live state before deciding whether to attach to an existing v7 worker or start one.
- 21.3327738197 GPU-command hours spent; approximately 8.6672261803 left under the cumulative 30-hour cap. This is summed elapsed time of GPU-tagged commands, not utilization or wall time.
- Four seeds: 42, 64, 91, 107. There are 23 populated analysis cells; 110 terminal receipts alias only 103 distinct checkpoint hashes.
- A3 learning-rate repair improves source validation by +4.14 percentage points, paired-seed 95% CI [1.32, 6.96]. A0/A1 did not receive that LR search.
- Optimized R4: 12,230 parameters, source mIoU 87.94; optimized A3: 30,922 parameters, 86.57. Their paired difference is +1.37 [0.41, 2.33]. This does not establish that small models generally beat well-trained large ones.
- R2: 418 parameters, source mIoU 85.64. It meets the registered mean-only source floor, but no rung establishes joint external teacher non-inferiority. Do not call the external capacity floor found.
- Q48 W4A8 versus matched FP continuation F48: source difference −0.25 [−0.81, 0.32]. Extra optimization explained the earlier apparent QAT accuracy gain. Q4b versus Q48 loses 1.14 source points in the mean and about 3.00 WF event-macro water-IoU points, conditional on the four checkpoints.
- Integer-domain simulation repaired the catastrophic gradients. All existing low-bit results remain `simulation_only`. The adopted forward is not identical to the old simulator on augmented training inputs.
- CE means focal loss on cached teacher hard pseudo-labels, not teacher-free learning. LKD also queries the teacher live on augmented inputs. SUP uses a different, much smaller human-labelled training set. Keep these distinctions explicit.
- LKD versus CE improves pooled STURM water IoU by +2.72 points for A0 and +5.42 for R2 under their tested recipes. The universal “smaller models benefit more from KD” hypothesis failed its registered primary test.
- One WF event, EMSR342, supplies 82.06% of reference-water pixels. Pooled WF scores and equally weighted event scores can reverse the teacher comparison.
- Six unique historical tiny-rung runs stopped before 7,800 updates under inherited patience. Keep their selected results and disclose the fixed-budget exception. Do not silently relabel them or rerun the entire matrix to tidy it up.

Recompute a small set of key contrasts from receipts. Verify hashes of reused code, cohort, teacher, checkpoint bytes, normalization, graph, loss, seed, selected LR, augmentation, and update schedule. A receipt path alone is not proof of reuse compatibility. Do not count aliases as independent seeds. If a scientific identity cannot be established, block only the affected branch and explain why.

## 3. Data and selection contract

- Use the established six bands, restored 2,500-tile pseudo-labelled TRAIN cohort, frozen teacher, and source validation split. Maintain the original band order and normalization.
- Use only TRAIN for calibration, sampling diagnostics, weight design, and quantizer calibration. Use source validation for declared checkpoint/LR selection and the prospective pilot gates below.
- S11-test, STURM, and WF are already research-exposed external readouts. They are not untouched confirmation sets. Do not use them to tune thresholds, choose arms, change losses, or trigger additional training.
- Preserve main evaluation at its registered resolution and original valid masks. Training at 224 and evaluating at 512 does not justify resizing evaluation inputs silently.
- For new full-length training, disable patience-based early stopping and complete exactly 7,800 updates unless a technical or budget stop occurs. Technical stops are incomplete runs. Use the inherited source checkpoint-selection rule and frequency; save both selected and final checkpoints. For new continuations, complete exactly 1,560 updates.
- Keep batch size, optimizer family, augmentation, weight decay, gradient clipping, precision, and validation frequency matched within each comparison. Record deliberate amendments. Construct the teacher before the student, or explicitly restore identical RNG states, to preserve initialization parity across methods.

## 4. Hard budget and admission rules

At launch recompute the cumulative GPU-command ledger from current receipts, including failed attempts and any work started since October 11. Never assume the archived remainder is still available.

Set the incremental cap to `min(8.0 GPU-hours, 30.0 minus verified cumulative GPU-hours)`, floored at zero. Use one available GPU by default. Do not double the budget if another GPU is idle. Do not rent compute, change host power policy, or kill unrelated processes.

Set a wall-clock deadline eight hours after worker launch. Do not admit a job whose hard timeout and reserved collection work cannot fit before the deadline. After the deadline, run only bounded CPU collection/reporting. Temporary contention is a wait/defer condition, not permission to overrun the deadline.

Indicative reservations at the full 8-hour allowance:

| Bundle | GPU-command hours | Admission priority |
|---|---:|---|
| Preflight, target/disagreement diagnostics, smoke timing | 0.35 | First |
| A0 re-anchor | 2.00 | Required if a complete bundle fits |
| R4 CE comparison | 0.80 | Required if a complete bundle fits |
| Six-arm R4 novelty pilot, two seeds | 1.30 | Main exploratory bundle |
| Conditional confirmation and target-refresh controls | 1.30 | Gates below |
| Terminal inference and supported deployment checks | 1.00 | Protect collection first |
| Diagnosed technical retry reserve | 0.75 | At most one retry per failed job |
| Optional extension | 0.50 | Only after required reservations |

These are reservations, not promised durations. Measure smoke throughput, include validation/teacher overhead, and use conservative hard timeouts. A full A0 re-anchor can require five new jobs at up to 1,400 seconds each. Full method bundles must reserve all their seeds before admission. If a bundle cannot fit, record `deferred_budget`; do not shorten training and call it the same experiment. If the available budget is smaller than eight hours, reduce lower-priority bundles before touching the protected analysis reserve.

Permit releasing unused reservations to the next declared priority. Never spend the same released reservation twice. Maintain one transactional budget ledger across all child manifests, including running-job reservations, subprocesses, calibration, teacher inference, retries, and terminal inference. CPU-only work is separate; GPU inference is not free. Count elapsed command time once, including nested GPU work, rather than double-counting parent and child commands.

## 5. Stage A — sanity checks and cheap diagnostics

First reproduce the reported A3 LR contrast, R4–A3 contrast, and Q48–F48 contrast from the four paired source receipts within 0.01 percentage points. Verify reuse hashes. Do not run historical training again.

On a deterministic, hash-selected sample of up to 128 TRAIN tiles, replay the actual augmentation, including per-band reflectance gain. Compare:

1. Geometrically transformed cached teacher hard masks.
2. Current teacher hard predictions on the geometrically transformed image without photometric jitter.
3. Current teacher hard predictions on the same image with the actual photometric jitter.

Record valid-pixel disagreement, per-class rates, teacher confidence, and tile variability. The implementation contains per-band gain `x' = k*x + (k−1)*mean/std`; masks are not relabelled by that gain. Distinguish teacher non-equivariance under geometry from changes after photometric augmentation. Do not assume refreshed teacher answers are always more correct.

Compute MNDWI from **denormalized reflectance**, using verified green and SWIR1 bands. For auxiliary rule calculations only, require finite, nonnegative band values and a sum greater than a documented epsilon in reflectance units; flag other pixels as rule-unavailable. Let `r = clip((green−SWIR1)/(green+SWIR1), −1, 1)` on available pixels and zero elsewhere. The fixed rule is `R = 1[r > 0]`. Never tune its threshold on external data.

Record teacher–rule agreement and disagreement frequencies on TRAIN. On source validation, use reference labels to describe teacher-correct/rule-wrong, rule-correct/teacher-wrong, both-correct, and both-wrong regions. No reference correctness is available from unlabelled TRAIN alone. Save counts, not just colorful examples.

Check that the rule feature aligns exactly with the augmented image and prediction grid. Avoid a new cache whose labels and images silently refer to different augmented views.

## 6. Stage B — repair the most important comparisons

### B1. Re-anchor A0

At seed 42 compare LRs `{5e-5, 5e-4, 5e-3}` using the original full FP/LKD recipe. Reuse the compatible old 5e-5 run. Select by the frozen source-validation rule, with deterministic tie breaking documented before running. Reuse a compatible selected-LR seed-42 run, then complete seeds 64, 91, 107 at that LR. If 5e-5 wins, reuse all compatible old runs.

This is equal search effort for the original large anchor and the optimized tiny models. It does not settle A1/A2/X1/X2. Do not quietly generalize beyond the re-anchored graphs.

### B2. R4 hard-target versus soft-target training

Reuse the four compatible optimized R4 LKD runs. Train four R4 CE runs at the same LR 5e-3 and otherwise matched 7,800-update recipe. CE keeps its historical definition: focal loss on cached teacher hard labels. Call the conclusion a comparison at a common LKD-selected recipe, not independently optimized superiority of either loss.

If useful budget is released, the first optional full bundle is four optimized A3 CE runs at its existing selected LR, reusing four optimized A3 LKD anchors. No new OFA grid tonight. A1 LR re-anchoring is lower priority and requires reservation of its complete bundle; otherwise defer it explicitly.

## 7. Stage C — one controlled research idea

### Hypothesis

A compact spatial model may use a cheap water-sensitive rule for easy spectral structure and spend its limited learning capacity on teacher–rule disagreements. The teacher may also be wrong there. Test correction and harm separately; never assume disagreement is useful knowledge.

This is a prospective **continuation experiment**, not evidence that scratch training has improved. Start every arm for a seed from the identical existing selected R4 checkpoint. Use a fresh AdamW state, LR 5e-4 with the same cosine schedule over 1,560 updates, weight decay 0.05, clip 5, FP32, batch 16, and the same augmentation/data ordering. Preserve inherited focal-loss settings and valid masks. Evaluate every 312 updates, include the start checkpoint as a candidate for every arm, and report both selected and final outcomes. Use identical treatment of batch-normalization state across arms.

### Additive rule contribution

Let the original R4 logits be `(z_land, z_water)` and `m = z_water − z_land`. Add only two trainable scalars:

`a = 8*tanh(alpha)`, `b = 8*tanh(beta)`, with `alpha = beta = 0` initially.

Set `c = a*r + b`, `z'_water = z_water + c/2`, and `z'_land = z_land − c/2`.

Then the new binary margin is exactly `m + c`; at initialization it is exactly the old R4 margin. The network can learn to ignore or reverse the rule. This is an additive rule contribution, not a proof of a pure residual decomposition or a physical constraint. Count both parameters and the runtime cost of computing the rule. Do not claim zero inference overhead.

### Distillation and weighting

For valid pixel `i`, let `p_i = softmax(t_i/T)`, `q_i = softmax(z'_i/T)`, and `T=4`. The teacher is frozen. Define teacher hard predictions at temperature 1, and:

`d_i = 1[teacher_hard_i != R_i]` on rule-available pixels; use zero otherwise.

For `N` valid pixels in the minibatch:

`u_i = 1 + 2*d_i`

`w_i = u_i / ((1/N) * sum_valid(u))`

`L = 0.5 * focal(z', cached_hard_mask) + 0.5 * T^2 * (1/N) * sum_valid(w_i * KL(p_i || q_i))`.

All masks and weights are detached. Use `log_softmax` and numerically stable KL with the correct teacher-to-student direction. Return a safe zero contribution for an empty valid mask; never divide by zero. Focal weighting is unchanged. Baseline arms set `w=1`.

Mean weight is one. This preserves coefficient/weight mass, **not the realized KL value or gradient norm**. Log unweighted/weighted KL, effective sample size `(sum w)^2/sum(w^2)`, gradient norms, and agreement/disagreement loss contributions. Do not silently renormalize the loss by its observed KL or gradient after seeing outcomes.

### Fixed pilot arms

Run all six arms at seeds 42 and 64, with paired initialization, data order, augmentations, schedule, and selection:

| Arm | Rule contribution | KD weighting |
|---|---|---|
| C0 | Absent | Uniform |
| C1 | Present | Uniform |
| C2 | Absent | Teacher–rule disagreement |
| C3 | Present | Teacher–rule disagreement |
| C4 | Present | Shuffled disagreement control |
| C5 | Present | Teacher-uncertainty control |

C4: permute the binary disagreement mask within each image, teacher class, and fixed teacher-confidence bin `[0.5,0.7), [0.7,0.9), [0.9,1]`, preserving counts. Use a dedicated deterministic RNG stream that cannot alter data augmentation or initialization. Log strata where permutation is degenerate. This matches class/confidence composition and weight count while removing the original pixel assignment.

C5: within each image and teacher class, assign the same number of emphasized pixels as C3 to the highest teacher-entropy pixels; break ties deterministically. This tests generic uncertainty weighting at the same per-class count. Use only valid rule-available pixels for both candidates. Log empty or fully selected strata.

The C0/C1/C2/C3 factorial separates the rule contribution and weighting. C4/C5 test whether targeting rule disagreement adds anything beyond generic extra emphasis. Only C3 is the preregistered combined candidate; do not select whichever pilot arm happens to win and rename it the method.

### Prospective promotion rule

After both pilot seeds complete, promote C3 only if all are true on selected **source-validation** scores:

- Mean C3−C0 mIoU is at least +0.50 percentage points.
- C3−C0 mIoU is nonnegative in each pilot seed.
- Mean C3−C4 and C3−C5 mIoU are each at least +0.20 points.
- Mean C3−C0 water IoU is no worse than −0.50 points.
- All finite-loss, label-alignment, initialization-parity, and budget checks pass.

These are practical screening thresholds, not a power calculation or significance test. Freeze them in the protocol before training. Promotion schedules C0/C3/C4/C5 at seeds 91 and 107 if their entire confirmation bundle plus collection fits. Replicate C1/C2 only if separately reserved; otherwise their ablations remain two-seed exploratory results.

If the gate fails, report which comparison failed and stop expanding this recipe. C1 alone improving would favor a useful rule contribution without evidence for the proposed weighting. C3 tying C4/C5 means a specialized disagreement mechanism is unsubstantiated. A noisy small miss is an inconclusive pilot, not proof the whole research direction is impossible. No gamma sweep, threshold changes, new graph, or external-score-triggered rescue tonight.

## 8. Stage D — disentangle target refresh from target softness

Use the Stage A audit. If cached versus live hard targets disagree on at least 1% of valid augmented TRAIN pixels, allocate a bounded diagnostic before optional architecture/quantizer extensions. If the mismatch is below 1%, preserve the rates and defer this diagnostic; that threshold does not prove refresh has zero effect.

Reuse C0 as the live-soft continuation. At seeds 42 and 64, train two matched additional continuations from the same R4 checkpoints:

- HC: `0.5*focal(cached) + 0.5*T^2*CE(onehot(cached), softmax(student/T))`.
- HL: same expression with the current teacher argmax on that augmented input replacing `onehot(cached)` in the second term.

Use `T=4`, no rule contribution, uniform weights, and exactly the C0 schedule. Apply identical valid masks. C0 uses the live-soft KL in that second term. KL from a one-hot target equals this cross-entropy; implement it consistently.

HC→HL isolates replacing cached targets with refreshed hard targets at the same loss construction. HL→C0 changes hard targets to soft targets. Gradient magnitudes can change as a consequence of target entropy; log them instead of claiming equal gradient strength. Existing historical CE is not HC and is not a substitute for this control.

This two-seed continuation diagnostic does not retroactively prove the cause of old scratch-training gains. Preserve all three outcomes. If it cannot fit, record the attribution gap explicitly.

## 9. Stage E — measurements, low-bit work, and what to defer

Protect terminal inference for admitted scientific bundles first. Reuse verified existing per-image counts where applicable. Every admitted training bundle must reserve selected-checkpoint terminal inference. Save final checkpoints and source scores; score final checkpoints externally only if separately reserved, otherwise label those readouts `not_run`.

For supported deployment checks, use existing selected R4/A3N/A3 artifacts, first seed 42, and a fixed TRAIN-only calibration set. Benchmark the same input shape, batch size 1, device, and preprocessing policy. Measure FP32, FP16, and a real INT8 engine only where the installed backend actually supports the graph. Existing W4A8 simulation can be reported separately. Do not describe quantize/dequantize nodes or floating-point execution as packed INT4/INT8 deployment.

Record runtime/version, operator fallback, artifact bytes, actual parameter/buffer bytes, peak inference memory with the teacher unloaded, warmup, synchronized p50/p95 latency, and prediction parity/accuracy of each compiled artifact. A minimum of 20 warmups and 100 timed iterations is enough unless there is a concrete instability. State whether timings include input transfer/preprocessing; optionally report both boundaries. Do not infer energy from MACs or latency. If FP16 overflows, diagnose it within the time allowance or mark it unsupported; do not silently score FP32 outputs under an FP16 label.

If a native low-bit convolution backend is unavailable, stop that export branch and report the blocker. Do not install or build a large new stack overnight. An honest unsupported result is preferable to a fake speedup.

No broad QAT matrix tonight. Preserve the remaining low-bit research question: 4-bit activation damage is unresolved and appears important at the stem. The next dedicated quantization phase must start with a competent FP anchor, matched FP continuation, structural stem precision, graph-verified signed/unsigned ranges, clipping, and a faithful LSQ baseline before a log-domain stem or flip-aware loss. Keep the exact adopted integer simulator and its native STE contract. Do not reintroduce the old cancellation defect.

Optional extensions use only released reservations, in this order: optimized A3 CE full bundle; complete A1 re-anchor bundle if it truly fits; additional fixed artifact deployment readouts. Do not admit partial bundles merely to consume the budget.

Explicitly defer:

- Repeating the failed hard-target U1 log-index recipe. Its poor WF result is real; its soft-target variant remains untested. Do not optimize its threshold on WF.
- Claiming U6 allocation is new/superior when a cheaper structural rule tied it.
- More R1 linear-rung training under the unstable current recipe.
- Broad OFA searches, the unsupported universal capacity/KD law, new architecture sampling, temporal flood-increment claims without temporal references, and unrelated exploration branches.
- Treating resized evaluation images as a clean context test. Any future context experiment must separate resampling from native-resolution cropping/tiling.

## 10. Evaluation that can explain success and failure

Report mIoU, water IoU, precision, and recall. Explain that IoU is overlap between predicted and reference regions, not pixel accuracy. Keep these estimands separate:

1. Pooled pixel-count scores.
2. Equal-event mean water IoU, using established event IDs, valid masks, and empty-union convention. Declare the convention; do not invent event IDs from tiles.
3. Four-seed paired differences with sample SD and a df=3 Student-t 95% CI when all four paired seeds exist. Two-seed pilots are descriptive and do not justify strong significance claims.
4. Event-bootstrap CIs conditional on the trained checkpoints, using the established 10,000 whole-event resamples and seed 1729. They quantify event uncertainty, not training uncertainty. Do not bootstrap pixels as independent samples.

Preserve full and overlap-excluded external readouts. Exclusion does not undo prior research exposure. Report all completed arms, including failed novelty candidates; do not hide external weaknesses because source selection passed.

For teacher/rule/student mechanisms, aggregate integer counts first and report per-event summaries:

- Correction rate: `P(student correct | rule wrong, teacher correct)`.
- Teacher-following harm: `P(student wrong | rule correct, teacher wrong)`.
- Rescue rate where both teacher and rule are wrong.
- Frequency and sample size of each conditioning group; undefined rates for empty groups, never zero-filled.

Also compute the signed benefit on teacher–rule disagreement: `(number teacher-correct/rule-wrong − number teacher-wrong/rule-correct) / number disagreements`, against real reference labels. On a valid binary disagreement exactly one of teacher/rule is correct; on unlabelled TRAIN, this quantity is unknown. This diagnostic can reveal whether extra emphasis transfers knowledge or copies teacher mistakes.

If a model improves only pooled WF, say that. If a method improves generic uncertainty pixels equally well, say that. If a source gain costs recall on small events, preserve it as a tradeoff. Do not infer turbid-water, shadow, or boundary mechanisms from scores without an appropriate labelled/stratified diagnostic.

Novelty language must be conservative. Known ingredients include spectral-index integration, selective/disagreement KD, and learned quantizer step sizes. Useful primary starting points:

- Trio-ViT: https://arxiv.org/abs/2405.03882
- LSQ: https://arxiv.org/abs/1902.08153
- Spectral-index integration for water segmentation: https://doi.org/10.1007/s41976-026-00290-3
- Disagreement-guided KD for segmentation: https://doi.org/10.3390/app16115573

The candidate contribution is a tested way to preserve the teacher's useful corrections beyond a cheap rule under severe model/storage constraints, with mechanism controls and event-aware transfer evidence. Whether it is novel or effective remains to be established. Do not claim first-ever priority from a bounded search or invent citations.

## 11. Implement durable autonomy before launch

Reuse the audited queue runner where suitable; extend it in new files if it lacks conditional orchestration. Starting or resuming the old finished v5c manifest does not schedule this work.

Create these operational artifacts in the new run directory:

- `protocol_v7.yaml`: frozen hypotheses, arms, exact formulas, data roles, gates, budget, seeds, selection, and deviations.
- An immutable initial manifest and deterministic code that materializes child manifests after preregistered CPU gates. Hash each before submission.
- `run_overnight.py` (or equivalent), `start_overnight.sh`, and `status`, `pause`, `resume`, `stop` controls.
- A single budget ledger, job/state database or atomic JSON files, failure ledger, append-only decision log, heartbeat, worker log, per-job logs, and per-job completion receipts.
- An idempotent collector/report builder that runs when work completes, a gate fails, the budget closes, or the worker stops normally.

Required behavior:

1. Lock by experiment identity and GPU UUID; detect an already-running matching worker and attach/status it. Prevent duplicate launch and duplicate spend.
2. Check foreign GPU processes and host resource headroom. Keep the workstation responsive; respect inherited resource limits. Never terminate unrelated jobs.
3. Launch the worker detached from the editor's terminal using the established `start_new_session=True`/closed-stdin/log-redirection pattern or an existing compatible service. Verify the actual process group/session and an advancing heartbeat. Do not rely solely on shell `&`.
4. Launch job process groups with explicit timeouts. On a controlled stop or timeout terminate only owned descendants, save the ledger, and collect surviving artifacts. A stop must not leave an orphan GPU job spending time outside the ledger.
5. Save checkpoint, optimizer, scheduler, update counter, RNG/data-position state when feasible. If exact resumption cannot be established, use a fresh, separately identified attempt instead of pretending it is exact continuation. Count all failed work.
6. Allow one diagnosed technical retry, only if its reservation fits. Do not retry low scientific scores. Mark dependants blocked or deferred with explicit causes. Do not let one invalid optional export prevent valid training/analysis from finishing.
7. No dependency on future agent messages, automated approval dialogs, interactive shells, or LLM-written decisions after detachment. All gates must be executable deterministic code already present.
8. Status must show running/completed/deferred/failed jobs, GPU UUID, elapsed and reserved hours, deadline, active subprocess, last heartbeat, and report path.

Before the real launch, perform meaningful focused checks:

- Zero-initialized rule contribution reproduces base logits; class ordering and MNDWI band/normalization handling are correct.
- Valid-mask handling, KL direction, empty masks, detached weights, mean weight one, and shuffled-control count preservation.
- Identical initial student state and first augmented minibatches across paired arms; extra diagnostics do not consume their RNG streams.
- Manifest dependency, retry, budget refusal, and conditional-promotion behavior using short mock jobs; a detached mock survives its launching shell exiting.
- One brief real training smoke per new code path, with finite losses/gradients, correct checkpoint metadata, and timing for budget admission. Smoke receipts are not scientific runs.

Then freeze, launch, and verify the real worker has progressed into a substantive job or has a clearly logged legitimate resource wait. Do not end after `manifest written`. If required local data/checkpoints/GPU access are absent, complete the runnable implementation and produce `BLOCKED.md` naming the exact missing dependency; never claim launch succeeded.

## 12. Morning deliverables and immediate handoff

The worker must automatically write:

- `report_v7.md`: plain-language answer to what changed, what was shown, what failed, and what remains unknown; then concise technical evidence.
- `results.csv`: every admitted arm and seed, training budget, selected/final checkpoint identities, source/external scores, and explicit missing values.
- `comparisons.csv`, `event_results.csv`, and `mechanism_counts.csv` with units, denominators, and uncertainty type.
- `deployment.csv` separating measured engines, unsupported attempts, and simulation-only results.
- `decision_log.jsonl`, `failure_ledger.json`, `budget_ledger.json`, `completion_receipt.json`, and `handover.md`.
- A small set of plots: accuracy versus actual artifact bytes/latency when measured; per-event differences; teacher–rule correction versus harm. Do not use parameter count as a substitute for runtime cost.

The report must distinguish `completed`, `reused`, `technical_failure`, `inconclusive_pilot`, `failed_scientific_gate`, `deferred_budget`, `unsupported`, and `not_run`. Include exact commands to reproduce analysis. State whether the candidate passed its prospective gate and whether its gain exceeded both controls. Do not turn a failed pilot into a retrospective success criterion.

Your immediate response after launching must contain: absolute run directory; worker PID/service and GPU UUID; incremental budget and wall deadline; admitted and deferred bundles; status/pause/stop commands; log/report locations; and confirmation of the observed heartbeat and first real job state. Remind the user briefly that the host must stay awake. Do not claim overnight results before they exist.

Proceed now: inspect, implement, validate, freeze, launch, verify, and leave the executable worker and reporting pipeline in control.
