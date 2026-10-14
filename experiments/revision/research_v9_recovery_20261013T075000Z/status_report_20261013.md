# V9 experiment status and recovery audit — scientific evaluation incomplete

Verified 2026-10-13T08:01:13.449724+00:00 (2026-10-13T10:01:13.449724+02:00).
Original campaign: `research_v9_20261012T235412Z`. Recovery: `research_v9_recovery_20261013T075000Z`.

## Outcome

All20 requested training runs completed successfully:8 same-image label-source controls and12 edge-arm runs. Final scientific evaluation did not complete. The original worker collected its failures at03:48:56 CEST on13October. Its original report's “Final collection” title describes worker termination, not complete scientific results. The original report and all historical artifacts are preserved.

The recovery worker is WAITING_GPU, with no active GPU command at this snapshot. It is detached from its launcher and VSCode, with stdin disconnected. Both eligible GPU UUIDs are occupied by foreign compute work. Their7548MiB required safe free memory exceeds the3529MiB observed per device. Waiting is not an experimental failure. No foreign process is altered.

The main edge-distillation hypothesis, same-image label-source OOD comparison, failure-stratum analysis, full unchanged-PTQ readout, and sealed-event scientific gates remain unresolved. No sealed OOD scores exist for these new arms yet.

## Completed training and source-selection evidence

| Arm | Completed seeds | Updates per seed | Mean selected source water IoU (%) | Across-seed SD (pp) |
|---|---:|---:|---:|---:|
| human | 4/4 | 750 | 60.029 | 5.515 |
| teacher | 4/4 | 750 | 59.944 | 5.353 |
| E1 | 4/4 | 7800 | 79.443 | 0.696 |
| E2 | 4/4 | 7800 | 77.092 | 1.072 |
| E3 | 4/4 | 7800 | 79.665 | 0.430 |

These are pooled S11 validation scores from the original training/checkpoint-selection receipts, averaged over four seeds. They are not test scores, equal-event scores, or the repaired TF32-off final evaluator. Validation selected the checkpoints, so these values cannot establish OOD gains or statistical significance. Seeds are42,64,91,107.

The paired human-minus-teacher-label selected source water-IoU mean is +0.086pp, with seed differences +0.072, +1.431, -1.475, +0.315pp. This small descriptive source difference is not evidence that human labels improve unseen-event transfer. E3's source mean exceeds E1 by +0.222pp; the confidence control therefore remains necessary for attribution. B0 and R4 use different training cohorts, so comparing their source means does not isolate architecture quality.

The human/teacher B0 arms trained on identical252 source images, common valid support, paired initial weights and first input minibatches, true CE,50epochs/750updates. E1/E2/E3 each used the fixed2500-image pseudo-label cohort, focal hard-label loss under the historical “CE” name,50epochs/7800updates. E1 changes edge labels using source-selected spectral rules, E2 ignores the edge band, E3 globally thresholds teacher confidence to match the changed-pixel count. E0 reuses four compatible historical teacher-hard R4 checkpoints. Four existing LKD R4, four original OFA B0, and four PTQ checkpoints are retained in the36-model frozen inventory.

All36 checkpoint byte hashes were reverified. Every label-source pair has matching initialization and input hashes. Every edge initialization matches its E0 anchor, and input hashes match across E1/E2/E3. The historical E0 receipt does not expose separate image-only hashes; the earlier smoke independently checked the frozen minibatch construction. All20 training runs recorded finite gradient checks within the5.0 clipping limit. Two primary-checkpoint metadata descriptions retain legacy156-updates-per-epoch wording; authoritative epochs/updates show50/750. This is disclosed metadata debt, not a failed training run.

## Per-seed receipts

| Arm | Seed | Epochs | Updates | Selected epoch | Source mIoU (%) | Source water IoU (%) | Final-update water IoU (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| human | 42 | 50 | 750 | 40 | 79.421 | 64.187 | 63.832 |
| human | 64 | 50 | 750 | 38 | 74.664 | 56.218 | 54.975 |
| human | 91 | 50 | 750 | 46 | 73.790 | 54.408 | 53.052 |
| human | 107 | 50 | 750 | 28 | 80.140 | 65.305 | 62.614 |
| teacher | 42 | 50 | 750 | 40 | 79.384 | 64.115 | 63.744 |
| teacher | 64 | 50 | 750 | 38 | 73.627 | 54.787 | 53.372 |
| teacher | 91 | 50 | 750 | 46 | 74.629 | 55.883 | 54.537 |
| teacher | 107 | 50 | 750 | 28 | 79.932 | 64.990 | 63.107 |
| E1 | 42 | 50 | 7800 | 20 | 88.011 | 78.764 | 77.529 |
| E1 | 64 | 50 | 7800 | 36 | 88.575 | 79.841 | 77.277 |
| E1 | 91 | 50 | 7800 | 24 | 88.101 | 78.954 | 76.240 |
| E1 | 107 | 50 | 7800 | 26 | 88.779 | 80.213 | 76.223 |
| E2 | 42 | 50 | 7800 | 26 | 86.405 | 76.024 | 75.853 |
| E2 | 64 | 50 | 7800 | 18 | 86.590 | 76.317 | 74.507 |
| E2 | 91 | 50 | 7800 | 24 | 87.553 | 77.954 | 75.157 |
| E2 | 107 | 50 | 7800 | 38 | 87.619 | 78.073 | 75.629 |
| E3 | 42 | 50 | 7800 | 20 | 88.267 | 79.369 | 78.111 |
| E3 | 64 | 50 | 7800 | 22 | 88.729 | 80.148 | 76.663 |
| E3 | 91 | 50 | 7800 | 32 | 88.251 | 79.242 | 78.163 |
| E3 | 107 | 50 | 7800 | 14 | 88.564 | 79.902 | 77.749 |

`completed_training_audit.csv` gives absolute checkpoint paths, SHA256, training elapsed time and gradient summaries. Source selection was fixed before sealed labels and is not being repeated with changed training settings.

## Diagnosis and retry scope

The original `normal_teacher_B0_final03` GPU command completed all3639 teacher windows, then exited1 while recording backend metadata. PyTorch2.10's `get_float32_matmul_precision()` raised after the process used incompatible backend precision-control APIs. The failed command consumed150.408GPU-command seconds. Since no full completion manifest existed, dependent source calibration and sealed evaluation did not run. The later sealed CPU analysis also exited1 because its prerequisite cache was absent.

The repair changes only backend telemetry: it omits the incompatible legacy getter, records actual matmul/cuDNN flags and precision overrides, and preserves the executed model constructors and inference arithmetic. The same crash was reproduced on CPU; the repaired record serializes successfully with both TF32 flags false. The installed-version reference is [PyTorch2.10 CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices). `backend_repair_test.json` contains the reproduction and repaired record.

Earlier failures remain recorded: standalone smoke omitted the Trainer's seed reset; its one retry then found numerical CPU/GPU differences. A later floating-model gate passed with TF32 disabled, while PTQ parent64 still differed on128 pixels and therefore requires authoritative GPU inference. Earlier stopped partial inference attempts remain preserved and excluded. These are technical diagnostics, not unfavorable scientific outcomes to rerun until they improve. None of the20 successful trainings is rerun.

The finite retry dependency graph is:

1. Full normal teacher/B0/PTQ GPU prediction matrix, fresh output directory,17 predictor rows.
2. S11-validation-only source operating-point selection for every frozen model.
3. One final sealed reference-cache read after all checkpoints/settings are frozen.
4. Full sealed-matrix cost admission using runtime and window count only.
5. Full sealed teacher/B0/PTQ GPU prediction matrix if admitted.
6. Complete normal CPU analysis with all frozen R4 models and fixed spectral rules.
7. Complete sealed CPU analysis and report collection.

All descendants now require completed prerequisites. A failed dependency cannot start the sealed reader or the sealed CPU evaluator. Partial model coverage is discarded from comparable results. No seed or model is dropped to fit the budget. The source and target scientific recipes, thresholds, cohorts and selection rules remain frozen.

## Sealed evaluation and claim limits

After overlap exclusions, the sealed metadata contains468 scenes from116 events,23excluded scenes, and30458 input-only candidate windows before final validity screening. It is not the nominal130-event population. WorldFloods-v2 train/val raw references remain unopened in both campaigns at this snapshot. Prior retained code/log audits found no executed historical label read, but there was no comprehensive historical OS access trace; “untouched” is qualified by that limitation.

The final reader is the only authorized sealed-label reader, and writes an access log and derived cache. Recovery cannot reopen raw references after an incomplete final-reader attempt. Any such failure must be diagnosed against retained progress/cache instead of blindly retrying. Permanent-water and cloud/nodata masks, native geometry and registered six-band normalization remain fixed. These data are evaluation-only.

Full results must include pooled and equal-event metrics, whole-event uncertainty, seed SD separately, approximate4096-bin AP, precision at matched MNDWI recall, source-only operating points, label-free EM/anchor policies, boundary/thin-water/cloud/spectral strata, independently referenced dry-window alarms, and per-event PTQ-versus-FP differences. The primary E1 gate requires a0.5pp sealed equal-event gain, positive whole-event interval, at least3/4positive seeds,10pp S11 thin-water recall gain, no more than0.5pp STURM loss and no more than1pp dry-alarm increase. Unavailable gates are not negative scientific evidence.

## Budget, clocks and controls

Verified cumulative GPU-command spend is 27.554129h; remaining unreserved allowance is 2.445871h under the30h cap. The inherited4h V9 incremental cap is enforced across both original and recovery receipts, not reset for the new directory. GPU retries reserve1500s normal plus7200s full sealed readout atomically before admission; actual elapsed GPU-command time, including failures, is charged. CPU work and GPU waiting are separate.

The original execution deadline remains10:36:29 CEST today; new work and active commands stop in time to preserve the10min collection allowance. The original absolute deadline is18:35:59 CEST today. Full recovery needs an explicitly authorized extension of the execution window; this was requested because §9.3 says “Preserve the original caps and deadlines.” No extension has been applied at this snapshot. A lack of response does not authorize an extension. If the current window cannot admit the full commands, the worker records deferrals and collects an incomplete report.

The worker's PID is3809974, with verified independent session and PPID1. It checks both authorized UUIDs every600s, writes a30s heartbeat, uses UUID locks and shared atomic budget reservations, and leaves busy foreign jobs unchanged. Editor closure is supported; host sleep/power-off prevents computation. `scheduler_status.json`, `gpu_availability.jsonl`, `recovery_scheduler_test.json`, `detached_launch.json` and `CONTROLS.txt` provide live evidence and controls.

```
/home/schmalstieg/miniconda3/envs/esi/bin/python /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_recovery_20261013T075000Z/scheduler_recovery.py status
/home/schmalstieg/miniconda3/envs/esi/bin/python /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_recovery_20261013T075000Z/scheduler_recovery.py pause
/home/schmalstieg/miniconda3/envs/esi/bin/python /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_recovery_20261013T075000Z/scheduler_recovery.py resume
/home/schmalstieg/miniconda3/envs/esi/bin/python /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_recovery_20261013T075000Z/scheduler_recovery.py stop
```

## Optional work and remaining author tasks

- B0 E1 transfer: UNAVAILABLE. Original B0 hard-label checkpoints lack full embedded matched input/initialization lineage; eight new N2500 B0 runs exceed protected freed-budget allowance; required normal+sealed inference remains protected; no benchmark scores are used
- hard negatives A1/A2: DEFERRED_BUDGET. Eight complete N2500 trainings plus pool-feature/candidate-overlap preparation require3360s beyond core and full inference reservations; required normal+sealed inference remains protected; no benchmark scores are used
- human S2 UNet teacher students: UNAVAILABLE. Located candidate lacks verified252-human-source training provenance; no new teacher training authorized; required normal+sealed inference remains protected; no benchmark scores are used
- same-image soft-KD extension: DEFERRED_BUDGET. Eight full live-teacher CE+KL trainings need a conservative6000s paired allowance beyond protected core/inference; primary true-CE pairs take priority; required normal+sealed inference remains protected; no benchmark scores are used
- R4 FP/QAT continuation: DEFERRED_BUDGET. No competent verified tuned continuation recipe; bounded source pilot plus complete FP/QAT readouts need a separate1800s reservation; unchanged PTQ has priority; required normal+sealed inference remains protected; no benchmark scores are used
- fresh FP32/FP16 timing sessions: COMPLETED: three fresh sessions. Verified freed budget admits all three fresh FP32/FP16 paired sessions; required normal+sealed inference remains protected; no benchmark scores are used

These provenance and budget deferrals remain separate from scientific negatives. The corrected OFA component evidence and old V7/V8 reports remain historical results; no adaptation or native low-bit claim is reinstated.

First finish the repaired readouts and confirm complete model/seed/event coverage before interpreting the main hypothesis. Then apply the frozen event-level mechanism and robustness gates. If E1 fails, retain the inherited-failure characterization and negative control results. If simpler E2/E3 explain any gain, qualify attribution accordingly; nonsignificance is not equivalence. Surface-water segmentation, event-level flood application and pseudo-label training must remain distinct claims.

Reviewer work still requires verified numerical/terminology corrections, consolidated component controls, consistent teacher protocols, transparent exposure and label-source disclosure, independently verified ordinary dry imagery, conventional-UNet training provenance if that optional comparison is pursued, and author-written Discussion and point-by-point response. Native low-bit speed/energy remains unsupported by fake quantization. Ordinary independently verified dry imagery and the11-window human review remain unresolved. Flooded vegetation's poor recall alone does not establish spectral invisibility. No manuscript was changed, submitted, or sent externally.

## Evidence

Original completion receipts: `/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_20261012T235412Z/scientific_receipts` and `command_receipts`.
Frozen model inventory: `/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_20261012T235412Z/final_models_final03.json`.
Original error: `/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_20261012T235412Z/logs/normal_teacher_B0_final03.log`.
New replay/audit: `/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_recovery_20261013T075000Z/manifest_final03.json`, `recovery_audit.json`, `backend_repair_test.json`, `recovery_scheduler_test.json`.
The original Markdown/PDF and their failure state are retained; this document is an interim audit, not the final scientific report.
