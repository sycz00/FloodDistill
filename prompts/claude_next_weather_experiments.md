# Claude Opus 5.5: bounded autonomous weather research campaign

You are the implementation and research agent in the existing `weather_mod_v1/` project. Implement, test, launch, monitor and analyse the campaign below. Do not stop after writing a plan. Use the model selected by the user in Claude Code; inspect the installed CLI and account-supported model identifier rather than inventing a literal model slug for “Opus 5.5.”

This is a **new campaign**, provisionally `weather_frontier_v13_<actual-start-date>`. Existing experiments and their original conclusions are immutable evidence. Reconcile actual repository status before deciding whether any requested item has already completed.

## 1. Goal and two distinct claims

Our primary research question is: **can a genuinely cheaper compressed weather model preserve useful rainfall during its own free rollout, and does recovery aimed at later harmful errors outperform ordinary immediate-error recovery at the same budget?**

A supporting product question is: **how cheaply can we produce the requested rainfall totals while keeping the dense weather trajectory, and can adaptive queries preserve important events better than simple equal-cost schedules?**

Do not replace the primary question with a rainfall-head optimisation. Conversely, do not dismiss a useful rainfall-service result just because it is not whole-model compression. Keep a separate table of claims, evidence and measured resource contract for each branch.

No novel-method, native-four-bit deployment, independent-generalisation or flood-safety claim without the corresponding evidence. A negative result is an output. An implementation failure is not a negative scientific comparison.

## 2. Mandatory initial reading and evidence reconciliation

Find the actual workspace root using `pwd`, version-control metadata and existing project files. Work in `weather_mod_v1/`, not an old sibling checkout. Internal Python imports named `weather_v1` are not proof that the workspace is wrong: preserve the existing package name unless a real import defect requires changing it.

Read existing `CLAUDE.md`, `AGENTS.md`, local permission rules and resource policies before edits. Do not overwrite them. Read the revised `weather_project_history_explained_2026-10-09` document and the earlier history. Search `reports/`, `research_help/`, `ideas/`, `paper/`, `prompts/` and the historically used `promts/` only where present. Prioritise:

- `reports/autonomous_compression_2026-10-07/` and its quantizer, topology, cases, scorer, remedy and resource receipts;
- `reports/decoder_generalization_v11_2026-10-07/` and the best qualified exact dense product;
- `reports/protect_state_v12_2026-10-08/`, particularly repaired four-bit rank results, exact prefix reuse, bypass, interpolation, data-ledger and training-feasibility receipts;
- `reports/v12_followup_2026-10-09/`, including narratives, `J45/`, diagnostic outputs if present live, failure ledger, waiter/service logs and the latest `FU_J1_R2` status;
- original prompts only to interpret intended comparisons, never as evidence that they ran.

Produce `EVIDENCE_STATUS.md` with claim, exact source, cohort, implementation hash, execution recipe, result status and missing evidence. Obtain exact meanings for shorthand such as A1, D4 and `skipU25` from their local experiment registries. Unresolved labels stay unresolved. Do not infer bits or query counts from ambiguous names. Preserve a campaign-qualified alias registry.

The uploaded follow-up archive contains completed J.4/J.5 CPU replay results but lacks the successful crop/perturbation diagnostic directory and final noise-ensemble retry receipt. Its waiter log records the retry launching at 11:05 UTC on 9 October, later than the narrative that still says “waiting.” Reconcile **live** status before continuing. Never relaunch a live or already completed job because a stale summary sounds incomplete.

Current evidence that must inform the campaign:

1. Aggressive simulated four-bit weights damaged development rainfall; native eight-bit effects were much smaller and cohort-dependent.
2. Dense state-updating calls recovered most aggressive damage, but retaining a dense model is not compact deployment.
3. Four-bit low-rank residual construction was repaired and reportedly worsened rollouts on six starts despite approximately 1.3% smaller layer error. Do not rerun it as an unresolved construction bug unless the live receipts contradict that report.
4. Window alignment reportedly reduced crop error to about 0.79 times the three-query comparator, still above the frozen 0.5 pass bar. The diagnostic covered six saved blocks, not six independent full rollouts.
5. Simple latent interpolation was correctly implemented and missed its frozen target. Preserve the negative rather than repeating it unchanged.
6. J.4's magnitude-10 adaptive rule used 78 mean queries per 120 hours, +0.003226 mm added daily reference error, zero introduced and nine recovered heavy-rain misses on six exposed starts. This is CPU archive replay, not measured latency or zero total reference misses.
7. J.5's heterogeneous compression ensemble is underdispersed in the reported diagnostics; simulated members do not supply native deployment savings.

## 3. Bounded execution authority and proposed new-campaign budgets

Honor any stricter pre-existing project/site policy. Do not extend an old campaign's budget. For this new campaign, use these conservative proposed ceilings unless the user has already supplied a different explicit approval:

- At most **16 GPU-hours total**, including failed attempts, warmup, native qualification and training; at most one GPU worker at a time.
- At most **48 wall-clock hours** after campaign admission. Queued time counts toward the wall deadline, not active GPU time. Idle waiting is finite, at most 12 hours for any job.
- At most **40 GiB of newly generated campaign artifacts and private compiler caches**, with a minimum 20 GiB free-space floor or the site's stricter floor. Large archives needed for confirmation require explicit admission before writing.
- **No new paid cloud compute, no new subscriptions and no new external-data downloads by default.** Existing locally/authorised-remotely cached assets may be used according to their exposure and permission status. Use an existing explicit download approval only within its remaining budget; otherwise report admission blocked rather than bypassing a cap.
- Existing workstation access only. Inspect authorised host configuration; do not assume a hostname, credentials or a second machine from old prose alone. Never change GPU power limits, driver state or administrator settings without existing permission.
- One diagnosed implementation repair per distinct failure path, plus one occupancy retry when interruption came from another user's job. Each retry gets a new attempt ID and preserves the old receipt. A scientific quality failure does not justify an automatic retry with relaxed gates.

Allocate approximately 2 GPU-hours to reconciliation/qualification and a possibly unfinished approved noise baseline, 2 to live adaptive-query and cost controls, 8 to primary matched recovery, 1 to creative boundary/bypass probes, 2 to frozen confirmation and 1 contingency. Reserve the main recovery and confirmation budgets before spending on peripheral ensembles or broad literature-driven sweeps. These are ceilings, not a request to consume all available compute. Small admissible cohorts may yield inconclusive results; do not manufacture power by treating cells as independent.

Write `CAMPAIGN_FREEZE.json` with actual start/deadline, remaining resource caps, permitted host and GPU UUID, root and artifact paths, precision/product contracts, selected phases, seeds and stop rules. Do not silently increase any cap. Wait/blocked branches must allow unrelated admissible CPU or development work to progress.

## 4. A persistent queue that does not depend on Visual Studio Code

Implement or extend a **durable experiment state machine on the compute host**. Ordinary Python/shell code, not an active chat turn, must be able to run the entire predetermined job graph, compute metrics, apply frozen numerical promotion gates and write reports. The optional model-assisted repair controller is separate.

### Required architecture

Create a versioned controller, job manifest and event journal under the new campaign. Prefer the site's existing scheduler when present. Otherwise use a user-owned systemd **service**, not a terminal-bound scope. The controller should invoke workers directly with explicit absolute interpreter paths and working directory, immutable configuration files, sanitized environment and redirected logs. Do not rely on the lifetime of the editor, terminal, notebook, SSH connection or Claude's background-task mechanism.

Use a real persistent service unit plus resumable queue for automatic restart only when already authorised. A transient `systemd-run --user` service may survive closing the editor but is not, by itself, a reboot-resumable campaign. Verify the actual systemd version, service behaviour, `loginctl show-user ... -p Linger`, logout policy and existing permissions. Do not silently enable lingering or install a system-wide service with elevated privileges. If approved persistence is unavailable, use the permitted site scheduler or report the exact operational blocker; still implement and test the queue and run allowable short work.

The user may close the editor before the first GPU becomes free. Therefore the **waiter, controller, workers and post-run analysis** must all be detached, not just the final GPU worker. Verify a harmless detached heartbeat job before launching expensive work. Confirm it continues after its spawning shell exits; where a safe separate test session is available, verify disconnect and reconnect. Report exactly which persistence scope was tested. Do not terminate the user's current session or restart the machine for a test. Machine shutdown/sleep still stops computation; resumption requires valid checkpoints and an authorised restart mechanism.

### Required queue behaviour

Implement states `planned`, `waiting`, `ready`, `running`, `complete`, `failed`, `interrupted`, `blocked`, `skipped` and `cancelled`. A job can be complete only with exit code, expected outputs, valid completion receipt and successful output validation, not simply because its PID vanished. Use atomic manifest writes, an append-only event journal and single-controller locking. On restart, reconcile job IDs, ownership and live processes before resuming. Never duplicate a worker or replace a running output directory.

Checkpoint at forecast-start boundaries and at safe training steps. If mid-rollout resumption changes warmup, compiler state or numeric history, restart that forecast in a new attempt rather than claiming bitwise continuity. Resume completed starts by their input, model and scorer hashes. A new model or recipe invalidates old quality/cache identities. On service restart, preserve completed work and charge all consumed resources to the same campaign budget.

Use the stable GPU UUID in admission records and verify device-index mapping after `CUDA_VISIBLE_DEVICES` remapping. Prefer exclusive scheduler reservation where supported. Poll device health and other users' occupancy. A failed or unparseable `nvidia-smi` call means **unknown/unavailable**, never “idle.” Require at least ten continuous idle minutes measured with a monotonic elapsed-time clock, not merely two five-minute samples. Recheck immediately before launch; a local lock does not reserve the GPU against other users. If foreign activity appears, stop/checkpoint only this campaign's own process tree and mark timing contaminated. Never send signals to unrelated PIDs.

Enforce remaining GPU, wall-clock, host-memory and storage budgets before every job and periodically within long jobs. Count failed work and warmup. Collect device telemetry without expensive tree scans inside timed inference. Use private compiler cache namespaces compatible with existing guards; record their path and size, never clear another campaign's cache. Do not parse stale absolute deadlines from copied worker manifests as fresh authority.

A campaign `STOP_REQUESTED` file must prevent new launches and request graceful checkpointing of owned active work. Produce `status.sh`, `stop.sh` and `resume.sh` plus an exact user-facing runbook. Use bounded service restart policy and explicit retry counters; no `Restart=always` loop that perpetually consumes compute on an invalid experiment.

### Optional autonomous code-repair agent

A durable numerical queue is mandatory; ongoing model/API calls are not. The initial Claude session should implement all routine branches before detaching. Only enable a headless Claude repair/analysis callback if noninteractive authentication, narrow tool permissions and an **explicit separate spending cap** are already authorised. Otherwise fail closed on unanticipated errors and continue independent jobs, writing a repair request.

Where approved, feature-detect `claude --help` and the installed CLI. Use bounded noninteractive calls with exact campaign context, structured output, finite turns/runtime and a shared remaining cost ledger. Do not invent unsupported flags or a model identifier; do not use a global permission-bypass flag. Do not expose credentials in logs, manifests or generated archives. A repair agent may inspect and edit only permitted campaign paths and narrowly necessary code; it cannot open sealed data, alter frozen metrics, relax gates, raise budgets or kill other jobs.

Before ending your interactive session, start the validated controller with the admitted job graph and show its real unit/job ID, status, log path and stop command. Distinguish “controller running,” “waiting for GPU,” “worker running” and “scientific result complete.” Do not say the campaign is running if only files were written.

## 5. Dataset admission, validation and extreme-event handling

Build `COHORT_LEDGER.csv` before loading evaluation outcomes. Record exact start time, both input-history timestamps, full forecast/reference windows, source/checksum, region masks, event family, season, original role, every known prior exposure, authorisation and current role.

Use separate roles for training/calibration, exposed development validation, locked representative evaluation and locked extreme-event evaluation. An event stress subset may overlap the representative evaluation only if declared as such and not double-counted. Split by non-overlapping forecast windows/storm families, not random cells. Every fit, threshold choice, architecture choice and selector comparison consumes development data, not the final test.

Resolve the archive's references to 16 new 2022 forecasts, locked 2024 dates and a sealed 2022 reserve against actual manifests. Year alone is not a split. Do not open the sealed reserve. Do not assume locally cached inputs imply complete references. Verify end-to-end assets and their costs before allocating GPU jobs.

When an existing explicitly authorised evaluation cohort is available, choose its exact release rule **before** scores are inspected. Freeze candidates, objective and analysis code first; the controller can then open that cohort under the existing authority. If no authorised untouched cohort can be admitted, finish development work and mark independent confirmation blocked. Never relabel six exposed non-calibration dates as independent test.

Define extreme events from the reference or a prespecified external catalogue, not from the candidate's errors. Use the existing 25 mm/day rule only for its established product and regions; do not invent 48/72-hour thresholds. Report how many starts/storms and reference-heavy cell-days support each result. January 31 dominates heavy-cell support in the current six-start replay; January 11 has zero such support. Zero misses on a zero-event date is not heavy-rain evidence.

Primary ordinary scoring stays the frozen per-start mean of area-weighted regional/day errors. Keep the archived heavy-weighted task objective and its fixed normalisers. Save the scorer hash. Report raw unscaled rain as primary; the historical bias coefficient is frozen secondary only. Separate dense disagreement, added reference error and practical event transitions. Compare equal-width early/late accumulations; wider accumulations are not evidence of propagation.

Use paired start/storm-level uncertainty with predeclared resampling and seeds; report absolute changes, denominators, all failures and missing support. Do not use a pooled cell-level sample size. Prespecify primary contrasts and handle multiple tested candidates transparently. Report estimation without a binary acceptability claim when a justified non-inferiority margin is unavailable. A minimum small-cohort screen does not establish robust seasonal skill.

## 6. P0 — baseline, construction and deployment qualification

Recover the **best qualified exact dense stack** under the same requested product contract: dense bfloat16 storage, qualified compilation, decoder tiling, global or regional rain-only intermediate decoding, and encoder-prefix reuse **only if its actual receipts prove correctness and resource benefit**. Keep global-hourly, regional-hourly and block-total services as separate contracts.

Qualify same-instance cold/postwarm and fresh-process repeatability on permitted development starts. Compare actual fields, full two-time handoffs, model values, dtypes, coordinates, preprocessing and timestamp joins. Zero-change and no-op wrappers must match; test all fine leads relevant to a candidate, not only lead 1. Record any compiler fallback and recompile-limit behaviour. A graph-break warning is not an identified performance bottleneck until measured.

Construct the native eight-bit and intended aggressive candidate from the frozen quantizer/topology. Confirm which of the 192 backbone modules are quantized, protected, repaired or unchanged. Record exact group size/axis, codebook, scales, rounding, source checkpoint dtype, actual weight bytes, native kernel path and all dequantization buffers.

An aggressive decoded-weight simulator can be admitted for numerical mechanism tests but must be labelled `SIMULATED_NO_NATIVE_SAVING`. Do not count its hypothetical four-bit bytes as real process memory. Before long recovery runs, test whether a supported native low-bit path can actually execute the relevant shapes on this machine with the required fidelity and memory. Do not spend the campaign implementing an unbounded new kernel stack. If only native eight-bit deployment is viable, retain the aggressive simulator as a mechanism branch and qualify one recovery on the native route; do not claim that proves native four-bit compression.

Measure complete 120-hour products in at least three clean matched native repetitions where budget permits. Report construction/cold start, warmup, warmed inference, I/O boundary, parameters, allocated/reserved/process peaks and actual coverage. Exclude hashing and diagnostics from inference timing. Do not pool hardware, power, precision, compiler or coverage strata. A method requiring both dense and compressed networks must count both, including loading/offloading costs.

## 7. P1 — confirm adaptive query escalation, with fair cost controls

Implement a **live** version of the existing frozen J.4 magnitude-10 rule; do not retune its threshold. Baselines are full six queries, fixed two queries and fixed three queries. The magnitude trigger is based only on the cheap six-hour field `3*(p3+p6)`, in millimetres, taking the maximum over either frozen regional halo. The decision escalates the **whole block**, not selected cells. Keep the original rule's global-versus-regional service contract.

For each block, retain the original two-time history. Query hours 3 and 6, preserving the original hour-6 weather update exactly once. Decide whether to request missing hours 1, 2, 4 and 5. All queries must still use the same pre-update history. For an escalated block, accumulate native hourly fields in the original 1-through-6 order and dtype; querying in a different order must not silently change the sum. Advance the weather history only after all selected calls are complete. No future reference, archived unqueried rainfall or dense oracle may enter the deployed trigger.

Unit tests: timestamp unit normalisation; exact coordinates/masks; nonfinite handling; deterministic trigger; equality at the inclusive threshold; no mutation of input/history; exact dense replay on fully escalated blocks; correct two-query total otherwise; query-count receipts; identical dense state trajectory throughout 120 hours. Reproduce the archived six-start frontier before new evaluation. Detect any real-time cache/context difference from the CPU replay.

Add one **simple cost control** and one **creative audit probe**, both fixed on development only:

- Simple control: full-query escalation on a predeclared fixed 10-of-20 block schedule, giving 80 total calls, and/or a fixed-seed random schedule with a development-frozen escalation rate. Compare measured costs rather than pretending 78 and 60 queries are equal. Report a secondary count-matched replay control clearly as such if per-start matching uses the adaptive method's realised count.
- Audit probe: start with the magnitude-10 policy, allocate a small fixed sentinel-query budget on some otherwise non-escalated blocks, and use a predeclared discrepancy rule to escalate. Use only quantities available at that point. Choose sentinel hour/schedule and the discrepancy threshold once on development under an explicit query ceiling; compare with random added queries at the same cost. Its purpose is to detect sparse-trigger blind spots, not to keep adding queries until error disappears.

One compact development screen only. Freeze at most the original magnitude-10 candidate and one clearly justified audit candidate for untouched confirmation. The original candidate is confirmable even if the novel audit probe fails.

Report actual latency and memory, calls per start, escalated blocks, same-history identities, ordinary and heavy-weighted reference errors, **total** misses/false alarms and introduced/recovered/repaired transitions with denominators. Zero introduced misses must never be described as zero total misses. Report policy-by-start results and common support. CPU replay on existing authorised full hourly archives is a useful cheap confirmation of scoring, but does not replace native timing of the live control flow.

## 8. P2 — the primary matched compressed-rollout recovery test

Do not launch another 192-layer sweep. Use a fixed small architecture and a bounded training/reconstruction feasibility test. Recover any already completed V12 training probe before designing a replacement.

### 8.1 Compact common correction architecture

Prefer the smallest existing native-compatible low-rank adapter or state-correction implementation. Freeze its insertion sites, ranks, trainable parameter count, dtypes and deployment operator before comparing objectives. Bias occurs exactly once; zero factors reproduce the quantized base; no hidden full dense residual or dense teacher remains at inference. Match storage and correction operations across recovery arms. If you use a state-output corrector, all trained arms must use the same input features and parameterised output map.

The untrained compressed base is C0. The existing plain weight-residual repair, already reported negative, is a historical comparator; recover its archived fields rather than rerunning it blindly. Dense-state hybrid is an oracle/performance bound, never a same-budget deployment method.

### 8.2 Recovery arms

C1 — **one-step clean-history recovery**: train the common adapter on dense/clean development histories, using dense teacher outputs for the exact same input history. Optimise a frozen normalised immediate state/output loss. This is the ordinary recovery control.

C2 — **student-history recovery**: train the same adapter on histories generated by the compressed student, with the dense teacher queried on those same histories. Freeze or explicitly version the history-generation/replay protocol. Detach stored histories where truncated training is needed; do not claim gradients passed through a full rollout if they did not. This isolates deployment-input shift from adapter size.

C3 — **future-harm-sensitive student-history recovery**: same architecture and student-history procedure as C2, but use one predeclared additional weighting/short unrolled objective targeting rainfall at later equal-width windows. Candidate targets may emphasise wind/transport-sensitive state directions supported by development diagnostics. Match training GPU work and teacher work as closely as possible; report rather than conceal any unavoidable difference. The deployed model receives only its current permitted weather history, not dense histories, future reference rain or an oracle rescue.

The key contrasts are C2 versus C1 and C3 versus C2. A C3 advantage over an untrained base alone is insufficient to support rollout-aware novelty. Include an equal-size generic/direction-randomised loss-control screen on a small subset if needed to show that targeting, not extra parameters or work, drives the gain; choose the final contrast before final outcomes.

### 8.3 Feasibility and training control

First run a small permitted development batch: correct forward identity, finite loss/gradients, zero initial correction, bounded memory, one optimizer update, checkpoint/resume and native export/reload. Use fixed train/development separation, seeds and total optimizer-step/teacher-call/GPU-work budgets. Avoid full-backbone retraining. Prefer stopped-gradient replay or short truncated rollouts when full-horizon training is infeasible, clearly describing the approximation.

Quantized native kernels may not support the needed gradients. In that case, use a precisely matched training surrogate, document its discrepancy and requalify the exported native model before any deployment claim. A surrogate training result cannot inherit native latency from an unrelated model instance. Do not silently dequantize the entire inference model after training.

If gradient training is genuinely infeasible within the cap, record the specific failure and switch once to a prespecified **activation-aware reconstruction** control using stored development activations and student histories, at the same factor budget as ordinary weight reconstruction. Record this as a changed, narrower question; do not relabel it as a completed C1/C2/C3 training comparison.

### 8.4 Evaluation and promotion

Use free rollouts for every deployable compressed arm. Use a small shared-history factorial on early and late blocks for mechanism diagnosis, recording current-call and inherited-history components and the signed cross-term. Teacher/shared-history queries are diagnostic cost, not the deployment path.

Score paired reference-error changes on fixed-width 6/24-hour windows, early versus late, plus the existing 48/72-hour amount products without invented extreme thresholds. Retain atmospheric sanity checks, wet support, event transitions, bias and severe case maps. Study magnitude versus direction: a smaller local reconstruction error is not a promotion gate by itself.

Promote at most one recovery candidate plus its strongest simple matched control after a development freeze. Require a valid native export and a credible complete-product resource frontier before expensive final confirmation. A simulated aggressive candidate may be reported as mechanism evidence, but the native route must be evaluated separately. Do not force a claim of statistical superiority from an underpowered cohort or replace missing deployment savings with theoretical bit ratios.

## 9. P3 — one creative, bounded boundary-consistent rain probe

This is an exploratory branch capped at one GPU-hour after the primary budget is reserved. First recover the reported successful crop diagnostic receipts and verify the exact `skipU25` mask and compilation status. Do not run many alternatives.

Priority: **window-aligned regional computation with explicit boundary context**. Start from the existing aligned West Coast crop. Preserve absolute coordinates, patch/window phase, fine-lead conditioning, padding and longitude conventions. Test whole-globe-through-crop equality at all intermediate leads as the zero-change control. Keep Central Europe blocked unless a separately qualified coordinate/boundary solution exists; do not “repair” it with arbitrary longitude rolls that change positional encodings.

Compare aligned crop alone, the existing larger aligned crop, a narrowly defined boundary-ring/context reuse from an already paid dense anchor, full global rain-only calls and sparse-query controls. The dense anchor and the cache's memory/lifetime must be charged. A cached backbone feature is potentially lead-dependent; using it is an approximation unless exact dependence is proved. Do not call a halo method exact because the output decoder is exact.

State the hypothesis and buffer ownership before implementation. Select one boundary-ring design based on code dependency analysis, not on test outcomes. The remaining crop error is not proven to be a lack of context, so inspect window-origin, shifted-window, positional and periodic-boundary effects separately. Report six-hour disagreement on exactly the same saved blocks and the same three-query denominator as the old 0.79 result.

Preserve the original `<=0.5 × T3 disagreement` gate for continuity. A method that misses it can still be placed on an explicitly exploratory cost/quality frontier, but must not be declared a pass after changing the threshold. Only time complete products after the operator/identity and preliminary quality checks pass.

Alternative, not additional: if boundary reuse is not implementable within the cap and an exact `skipU25` configuration already exists, qualify and time that existing bypass against a genuinely matched compiled/eager baseline. Do not expand its name into an invented percentage or create a new block sweep. The branch can finish with a well-diagnosed negative.

## 10. P4 — pending ensemble diagnostics, lower priority

Inspect the existing perturbed-initial ensemble retry and finish already authorised postprocessing when its complete receipts exist. Do not duplicate or extend the old two-GPU-hour follow-up budget. If fresh compute is necessary and worthwhile, admit it explicitly under this campaign's P0 allowance with one frozen perturbation rule; otherwise mark pending and proceed.

Retain the original descriptive checks, but separately explain that different-input ensemble members are not repeatability controls. The optional full-precision-backbone sweep was removed after a dtype failure, so the low-dose response plateau supports but does not uniquely prove the rounding explanation. Do not fit perturbation size to make a compressed candidate pass an acceptance band.

For compression-versus-noise ensembles, use the same starts, region/day support and member count, and report **total measured member-generation cost**. Equal member count is not equal compute. Include ordinary empirical continuous ranked probability score alongside the previously reported finite-ensemble-corrected variant. Explain that the heterogeneous deterministic compression members do not automatically meet that correction's sampling assumptions. Include Brier score, reliability/threshold support and row versus pooled spread/error aggregation. Do not claim calibrated uncertainty from a favourable mean score.

## 11. P5 — frozen confirmation and reporting

After the development candidate freeze, evaluate the admitted untouched representative cohort and separately reported reference-defined extreme set. Use the same deployed product, coverage, checkpoint, native execution, scorer and baseline. A dense state hybrid, a globally decoded forecast and a regional crop are different cost contracts and must not be mixed into an unexplained ranking.

Show a start/event-by-method matrix, seasonal and heavy-support summaries, paired intervals, failure ledger and measured resource frontier. Analyse severe regressions even when the average passes. Report where different methods fail only when per-event results establish that difference; otherwise label the mechanism hypotheses as hypotheses.

Primary scientific report questions:

1. Does the live magnitude-10 product reproduce the CPU frontier and beat simple cost-matched schedules on untouched starts?
2. Does student-history recovery outperform ordinary clean-history recovery at the same native budget?
3. Does future-harm-sensitive recovery add value beyond the student-history control, or was ordinary recovery sufficient?
4. What happens to dense disagreement versus reference harm, especially at late equal-width lead windows and on heavy events?
5. Is there a genuine native memory/latency advantage against the best qualified exact dense stack, with no teacher at inference?
6. Did the boundary or sentinel probe identify a useful unconventional direction, a correct-but-unhelpful design, or an implementation blocker?

Do not declare a winner from one metric while hiding another metric's severe regression. Preserve raw and secondary corrected products separately. Do not call the historical 5% global gate a flood-risk tolerance.

## 12. Required outputs

Within the new campaign folder, write:

- `README.md` and `RESULTS.md`: plain-language research questions, what changed, how tested, status, support, results, limits and next decisions. Every campaign/policy abbreviation gets a compact inline reminder outside literal paths/code.
- `EVIDENCE_STATUS.md`, `EXPERIMENT_REGISTRY.csv`, `COHORT_LEDGER.csv`, `CAMPAIGN_FREEZE.json`, `CANDIDATE_FREEZE.json`, `SCORER_FREEZE.json`, `CLAIMS_LEDGER.csv` and append-only failure/repair records.
- Versioned executable queue/controller, worker manifests, service/scheduler definition, runbook, status/stop/resume commands and disconnect/restart test receipt.
- Unit/integration tests, source/environment hashes, exact model/configuration and input/reference identities, per-start and per-lead score tables, whole-product resource repetitions and provenance for reused results.
- A small set of useful plots: per-start paired differences, fixed-width error versus lead, cost/quality frontier, event transitions with denominators and representative failure maps. No chart should combine incompatible cohorts or native/simulated costs.
- `NEXT_DECISION.md`: which hypothesis is supported, contradicted, unresolved or blocked; what independent/native evidence is still missing. Do not turn an early screen into a novelty claim.

Before leaving the interactive session, present the verified controller/job status and stop command. On campaign termination, the durable controller must write a final summary even when no candidate wins or confirmation is blocked. Package scientific receipts and reports, not huge compiler caches, credentials or unrelated data.

Begin now with evidence and live-job reconciliation, then implement the durable controller and execute the admitted phases. Ask only for genuinely non-resolvable permissions or resources; otherwise use the frozen rules and keep working within the stated scope and budgets.
