# FloodDistill v8: audited results, detailed analysis and required next work

12 October 2026, Europe/Berlin. This report preserves the frozen worker outputs and corrects two reporting problems in separate audit tables. It distinguishes observations, plausible explanations and proposed next experiments.

**Status. All scheduled work is complete.** The worker launched at 14:55:27 CEST and finished at 15:01:05 CEST; final collection completed at 15:01:06 CEST. A live host check at 19:23 CEST confirmed that PID 1493699 had exited and there were no GPU compute processes on either RTX4090. All eight worker jobs completed, all 180 adaptation rows are present, and the failure ledger is empty. No experiment remains queued by this finished worker.

The programme spent **0.098437 GPU-command hours** (5.91 minutes), including both real smokes. Cumulative capacity-programme spend is **25.969689 hours**, leaving **4.030311 hours** under the 30-hour limit. Worker wall duration was 5.63 minutes. This is elapsed GPU-tagged command time, not utilization, energy or training-only time. Shared saved teacher hard predictions, tiny student forwards and small cached head fits explain the short workload; no new foundation-model fine-tuning was run. The remaining budget is not an instruction to restart the finished manifest or spend it automatically.

**Research conclusion. Useful complementary teacher information remains, but the proposed reliability-weighted single-solve update did not absorb it successfully in this pilot.** Uniform ridge loses 0.38 pp mean query mIoU; reliability weighting increases the loss to 0.55 pp. Twenty cached head-only soft-KD steps give a small descriptive gain of 0.079 pp, with worsening episodes retained. The quantization penalty reduces some harm but does not make the combined method beat its simpler controls. R4 remains an established small deployed graph; v8 supplies a reproducible negative adaptation result, not a successful new selective-distillation method.

## Execution and integrity

The CPU audit verified **286 frozen input hashes**, all six child-manifest hashes and eight terminal job attempts. All **180 adapted checkpoint hashes** match their rows; 25,200 extractor/buffer tensor comparisons confirm that only the final classifier changed. It independently recomputed 108 calibration/ridge solves and all 180 scores from integer confusion counts. It checked the pixel-accuracy identity in 480 event/comparator cells. No integrity mismatch was found. Real FP/Q folding errors were at most 2.87e-6 margin units, and normal-equation gradient residuals were below 1e-8.

The checks validate their stated numerical/artifact properties. They do not establish population generalization, novelty, field readiness, or the correctness of every prose label. Two reporting problems were found despite successful worker receipts:

- The learning-curve parser searched for `ep` without a word boundary and matched it inside `step`. Every recorded validation epoch was one epoch too early; selected/final flags were consequently wrong. Corrected tables parse the zero-based epoch `step` explicitly and add one. The 798 observations now identify all 32 selected and 32 final checkpoints. Original outputs are retained.

- Deployment backend metadata was captured after the source loader reset matmul precision to `medium` and matmul TF32 to true. The earlier timing code configured TF32 false and matmul `highest`; CPU replay verified that the reset happens after tracing and before source accuracy. There was no contemporaneous GPU flag receipt at timing. The FP16 backend helper also hard-coded FP32 dtype labels. Saved artifact parameters independently verify FP16 execution weights. The audited report separates the configured timing path, post-accuracy flags and actual artifact dtype.

See `analysis_20261012/audit_summary.json`, `solver_audit.csv`, `deployment_audit.csv` and `backend_phase_replay.json` for evidence. Frozen code, manifests and the automatic report are unchanged.

## What was tested and what the experimental units are

The teacher is the fixed task-adapted Prithvi checkpoint; it inherits 252 labelled source-training scenes. R4 has 12,230 parameters and a 16-channel affine final two-class head. The experiment optimizes 17 effective margin coordinates, or two calibration coordinates, and folds the correction into existing weights. No deployed feature channel or post-adaptation teacher call is added. Extractors and BN buffers stay frozen in eval mode.

Three WorldFloods events were chosen deterministically by metadata: EMSR264, EMSR286 and EMSR333. One map per event supplies 51, 34 and 7 query tiles, respectively: **92 query tiles**, not a full adapted-benchmark evaluation. Each event has two geographically disjoint one-chip support draws. Parents 42/64 yield12 event/parent/support episodes per arm. Frozen baselines repeated across draws are the same parent predictions, not additional trained models. Tables average within event across the four paired seed/support outcomes, then weight the three events equally. No population CI is estimated from millions of correlated pixels.

Source-only selection used two disjoint S11-validation development pairs, Ghana and India. Both parents selected lambda 0.1; SGD selected LR 0.001 for seed 42 and 0.1 for seed 64. The two development query chips supplied 517,329 unique valid labelled pixels for hyperparameter selection, shared across arms/parents; support reference values were used only for offline diagnosis. Source labels are a disclosed selection budget. Query labels from the target events never fit an update or choose a checkpoint. Every external set is already research-exposed, so this is exploratory transfer.

The restored original N2500 membership has zero replacements and SHA256 `1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60`. Historical generation-time byte identity is unavailable. TRAIN-only images set feature statistics and quantizer scales. R4 parents, teacher, split/order, normalization, masks, original activation versus repaired graph identities and the earlier V7 C3 negative result are preserved.

## Does the teacher know useful corrections?

Yes, it has correct predictions on some pixels where R4 is wrong. It also makes errors where R4 is correct. The unattainable label-aware oracle identifies remaining complementary information; it is not a deployable router. All table scores below are percentages and are descriptive means across the two student parents; the teacher is one fixed model. G/H are fractions of the same valid pixels, expressed here as percentages. G-H equals teacher-minus-student pixel accuracy, not their IoU difference.

| Dataset | Average | G % | H % | Teacher water IoU | R4 water IoU | Oracle water IoU |
| --- | --- | --- | --- | --- | --- | --- |
| source_validation | Pooled | 1.014 | 0.629 | 81.71 | 78.17 | 86.46 |
| source_validation | Equal event | 0.905 | 0.563 | 80.71 | 76.86 | 84.86 |
| s11 | Pooled | 0.666 | 0.768 | 81.67 | 82.43 | 87.08 |
| s11 | Equal event | 0.716 | 0.936 | 79.06 | 80.19 | 85.21 |
| sturm | Pooled | 1.222 | 1.575 | 64.87 | 64.71 | 68.43 |
| sturm | Equal event | 1.063 | 1.441 | 38.84 | 37.82 | 41.90 |
| wf | Pooled | 0.629 | 0.982 | 64.32 | 65.98 | 68.24 |
| wf | Equal event | 0.684 | 0.512 | 50.54 | 47.95 | 54.00 |

On WF, the oracle increases pooled water IoU from 65.98 to 68.24, about 2.26 pp; equal-event water IoU rises from 47.95 to 54.00, about 6.06 pp. Thus the teacher is neither a universally superior prediction nor devoid of useful corrections. WF pooled G is below H, while equal-event G is above H, and the teacher/R4 water-IoU ranking also reverses. Large events/pixel counts and equal-event means answer different questions. The STURM teacher has useful water corrections even though pooled pixel accuracy favors R4: class-conditional water correction rates are 3.30–4.51%, versus 0.63–0.84% harmful imitation on reference-water pixels. Precision, recall, IoU and pixel accuracy must remain separate.

The teacher/MNDWI oracle on WF reaches73.86 pooled and 61.40 equal-event water IoU, versus teacher 64.32/50.54 and rule 63.81/40.06. This points to considerable expert complementarity already available without a second foundation model. It does not show that an implementable confidence gate can find the correct expert, or that a tiny optical student can represent every correction. The two pairwise oracle tables do not establish a three-expert joint oracle. Full and known-overlap-excluded event/class counts are retained, with undefined empty unions omitted from event means.

![Pooled and equal-event WF headroom must be distinguished.](analysis_20261012/headroom_estimands.png)

## FP adaptation: controls and every worsening outcome

mIoU is region overlap averaged across land/water classes, not pixel accuracy. Effects below are percentage points relative to the corresponding frozen parent. Positive/negative counts are descriptive episode counts; their twelve entries contain only three transfer events.

| Arm | Method | Query mIoU % | mIoU change pp | Water-IoU change pp | Better/worse episodes |
| --- | --- | --- | --- | --- | --- |
| F0 | Frozen FP parent | 70.049 | +0.000 | +0.000 | 0/0 |
| F1 | Two-coordinate calibration | 69.763 | -0.287 | -0.576 | 4/8 |
| F2_1 | Head-only soft KD, 1 step | 70.071 | +0.021 | +0.045 | 7/3 |
| F2_5 | Head-only soft KD, 5 steps | 70.117 | +0.067 | +0.141 | 9/3 |
| F2_20 | Head-only soft KD, 20 steps | 70.128 | +0.079 | +0.162 | 7/5 |
| F3 | Uniform residual ridge | 69.671 | -0.378 | -0.782 | 5/7 |
| F4 | Reliability-weighted ridge | 69.498 | -0.552 | -1.188 | 4/8 |
| B_supervised | One-chip supervised ridge | 68.278 | -1.771 | -3.732 | 3/9 |
| B_teacher_label | One-chip teacher+label ridge | 68.812 | -1.238 | -2.627 | 3/9 |
| B_supervised_SGD20 | One-chip supervised SGD, 20 steps | 69.696 | -0.353 | -0.769 | 4/8 |
| Qbase | Frozen PTQ simulation | 70.033 | +0.000 | +0.000 | 0/0 |
| Q0 | Uniform, gamma=0 | 69.559 | -0.474 | -0.910 | 4/8 |
| Q1 | Reliability, gamma=0 | 69.223 | -0.810 | -1.640 | 4/8 |
| Q2 | Uniform, gamma=1 | 69.586 | -0.447 | -0.849 | 4/8 |
| Q3 | Reliability, gamma=1 | 69.316 | -0.717 | -1.443 | 4/8 |

The reliability candidate F4 loses 0.173 pp against uniform F3,0.265 pp against two-coordinate calibration, and 0.630 pp against F2_20. It does not exceed either a simple correction or conventional head-only KD. F2_20 improves seven episodes and worsens five; its maximum gain is0.597 pp and minimum is −0.191 pp. The modest average benefit is useful pilot evidence, not a demonstrated robust population gain. F2_5 improves nine episodes with a smaller average gain. There is no prospective promotion to larger training hidden in these outcomes.

F4's event mean mIoU changes are −1.013 pp on EMSR264,−0.804 pp on EMSR286 and +0.162 pp on EMSR333. The last event's small mIoU increase coexists with a water-IoU decrease of 0.010 pp; background improvement alone is insufficient for the flood/water objective. On EMSR264, F4 increases precision from 64.89 to 65.62% while reducing recall from 44.55 to 41.23%. On EMSR286 recall falls from 82.44 to 80.51%. All event means and paired controls are in the audited CSVs.

![Mean paired effects and all three event effects. Q arms use their quantized parent comparator.](analysis_20261012/adaptation_control_effects.png)

## Why the reliability proxy is a plausible cause of failure

An offline diagnostic on the six target support chips found **8,741 errors among 9,648 confident/stable reference-water pixels: 90.60% were confidently predicted as land**. Confidence means max probability>=0.9; stability means absolute probability change<=0.05 under the frozen+2% gain. These are counted observations on six correlated chips, not independent trials. Source-development confident/stable water error is only 184/219,915, about 0.084%; this proxy changes reliability sharply under domain shift.

The actual sampled reliability weights reinforce this concern. On EMSR264 support draw 0, the average weight for incorrect water predictions exceeds the correct-water weight in both parents:0.631 versus 0.439 for seed 42,0.702 versus 0.392 for64. Incorrect sampled water predictions receive81.6–82.2% of that chip's water-class weight. On EMSR333 draw 0, corresponding weighted error shares are 62.4–65.7%. Low sensitivity to a small radiometric change measures invariance, not truth. Land dominance can hide water misses in an overall confidence-accuracy summary.

A second issue is support representativeness. Both EMSR286 support samples contain zero labelled water pixels despite query water IoU near 81.61% for the parent. EMSR333's second support draw is also sampled entirely as land; the sparse EMSR264 second draw supplies only two water samples for seed 42 and none for64. Support chips were selected without target outcomes, as required; these observations explain a possible support/query coverage problem without retroactively changing selection. Teacher-only and supervised fits can both learn the local operating point of mostly-land support and lose query recall. The observations support this hypothesis; they do not prove that label-free support selection is impossible or that all teacher corrections are unrepresentable.

Numerical failure is not the explanation: cached features are finite, gradients/linear systems are valid, normal equations solve correctly and folded heads match their intended updates. Lambda 0.1 was selected at the grid's upper boundary on source development. All teacher-only ridge/Q updates had eta 1; the declared damping did not rescue or cause their negative effects. The supervised ridge arm alone required eta 0.61–0.78. Ridge optimizes bounded logit residuals; SGD optimizes soft probability KD at temperature 4. A comparison of the two does not isolate the linear solve from the objective or update magnitude. A future causal test must separate those factors.

## One labelled chip: the tested fitting recipe failed

The supervised ridge arm loses 1.771 pp mean mIoU; teacher+label ridge loses 1.238 pp. Mixing teacher targets improves the supervised-ridge control by 0.533 pp, yet the mixture remains0.884 pp worse than supervised 20-step SGD and below the frozen parent. No gain from the currently tested labelled-head recipe is established. This is not evidence that target annotation is worthless. Labelled fits inherit the teacher-only source-selected lambda/LR; they were not independently optimized for their supervised objective. A strong annotation-value claim therefore requires a separately source-tuned supervised baseline with equal selection/annotation opportunity.

Each episode receives one existing manually labelled benchmark support chip, with 134,889–252,905 valid reference pixels, covering 13.49–25.29 km² on the 10 m grid. The six unique alternative support chips offer 1,158,767 labelled pixels total, reused across parents/arms; do not multiply them into independent annotation datasets. Fitting uses 2,474–4,013 spatially sampled pixels per chip, not every available labelled pixel. Report the accessible full-chip annotation budget and the sampled fit support separately. This is one labelled chip, not one labelled pixel; no new manual annotation was purchased.

## Quantization: small penalty benefit, unsuccessful combined mechanism

Qbase is fixed PTQ/fake-quant W4A8 backbone, W8A8 stem and FP32 head, calibrated on 16 TRAIN chips. Q0–Q3 form the frozen 2x2 reliability/penalty ablation. Scales are fixed; master arithmetic/tensors remain float32. No real low-bit kernels, packed integer artifact, QAT-trained backbone or low-bit latency result was established.

Relative to Qbase, Q0/Q1/Q2/Q3 mean mIoU changes are −0.474/−0.810/−0.447/−0.717 pp. Gamma 1 helps uniform by 0.027 pp and reliability by 0.093 pp, but Q3 remains worse than Q0 by 0.243 pp and Q2 by 0.270 pp. The combined-arm claim fails its stated interpretation criterion. A small positive factorial interaction of about 0.067 pp is descriptive and does not override losses against simpler controls.

The frozen quantized parent itself is only−0.016 pp below FP in the three-event average, yet event changes are −1.240 pp on EMSR264,−0.148 pp on EMSR286 and +1.339 pp on EMSR333. This cancellation shows why the average is insufficient. It is a simulation outcome and possible operating-point effect, not proof of trustworthy severe compression. The new Q point has no complete source-validation 1/2/3 pp frontier and no competent full-QAT comparison; those are prerequisites for a strong compression-method claim.

## Adaptation cost and actual deployment

Median solver/optimizer times are 0.731 ms for uniform ridge,0.737 ms for weighted ridge and 6.267 ms for20-step KD. Shared cached teacher targets and features give SGD the same caching opportunity. The roughly 8.5-times inner-optimizer difference is not an end-to-end acceleration claim. Recorded median component sums are 58.1 ms for ridge versus 63.9 ms for20-step KD, so teacher/features largely dominate that comparison.

The field named `cold_end_to_end_seconds` sums measured reused-engine components. It omits teacher construction/checkpoint loading, input preparation/transfer before teacher timing, some feature transformations, head folding and artifact serialization; shared normalization/calibration are separately reported. Per-method memory also comes from a shared command process. Preserve the measured component times, but do not present them as a truly cold wall-clock or isolated memory comparison. A future full-pipeline benchmark must include those boundaries.

| Engine | Actual bytes | p50 / p95 ms | Incremental allocator MiB | Source mIoU % |
| --- | --- | --- | --- | --- |
| FP32 | 183,255 | 0.4746 / 0.4844 | 6.000 | 88.014430 |
| FP16 | 156,695 | 0.6359 / 0.6418 | 3.001 | 88.014194 |

FP16 saves **14.49% actual serialized bytes**, with source mIoU change−0.000236 pp, but its observed p50 forward latency is **33.98% slower** in these sessions. FP16 passes the descriptive1/2/3 pp source quality-loss tolerances for this seed. It does not have a demonstrated throughput advantage. Both points are Pareto-relevant depending on storage versus workstation latency; repeated-session/edge performance and energy remain unmeasured. Static parameter counts or theoretical packing are not artifact/latency measurements.

These are separate fresh processes, teacher unloaded, common1x6x512x512 input,20 warmups and 100 synchronized forward timings. CPU-loaded artifact parameters verify pure FP16 versusFP32 storage. The incremental allocator peaks are isolated timing-phase measurements, unlike retained-tensor-contaminated V7 peaks. Post-source-score flags are cuDNN TF32false, matmul TF32true and precisionmedium; the timing code's configured strict phase has only code/CPU-replay evidence. Neither a comparison with earlier V7 latency nor an edge speedup should be inferred. Native INT8 remains unsupported under the inherited TensorRT/runtime-driver combination; no repair or new install was attempted.

## OFA efficiency recovered from real training, with corrected epochs

The32 actual training wrappers provide798 validation observations, with original logs/checkpoint identities preserved. Inference commands containing `train` in their names were excluded by inspecting effective nested argv. Source-validation thresholds70/75/80/85% were fixed in advance. First crossings are observed at validation intervals; they are not interpolated or exact continuous crossing updates. Unreached thresholds stay censored. Logged scores have four-decimal precision, so threshold equality has that rounding limitation.

For EfficientViT N100, all four seeds reach70% and 75%: average first observed crossings are 108/144 updates for LKD versus 96/132 for OFA. Neither reaches80% or85%. Thus the observed low-data curves show a12-update earlier mean threshold crossing for OFA, within the 12-update validation spacing. This is bounded update-efficiency evidence, not isolated sample efficiency or measured time-to-quality acceleration.

At N2500, both methods first exceed80% at 312 updates, the first source validation. At85%, LKD reaches in three of four seeds, with one censored; OFA reaches all four. Mean936 versus 1482 among reached seeds is not a fair unconditional ranking because censoring differs. LightFormer 85% crossings average2184LKD/2886OFA; MobileNet averages1794LKD/1482OFA. These mixed curves do not support universal OFA acceleration across backbones/budgets. Full-budget EfficientViT OFA retains its legacy all-pixels KL label, while the LKD wrapper uses valid-only; do not silently erase that policy distinction.

Mean complete training-wrapper elapsed time is176.7 s LKD versus 182.1 s OFA atN100;924.6 versus 1073.7s for EfficientViT N2500;946.2 versus 1084.2s forMobileNet;1064.4 versus 1200.5s forLightFormer. Wrappers include teacher/head work plus terminal evaluation/checkpoint checks. Per-validation time stamps and isolated training phase timing are absent, so observed time-to-threshold is unavailable. AcrossN100/N2500, distinct scenes and total update budgets change together (300 versus 7800 maximum updates); the curves cannot isolate sample efficiency.

![Corrected update curves; dots mark the actual selected checkpoints. Timing at intermediate observations is unavailable.](analysis_20261012/ofa_learning_curves_corrected.png)

## What must be done next

1. **Use the corrected reporting now.** Cite the audited epoch/crossing tables, actual deployment dtypes and explicit timing phases. Keep original reports/results as historical artifacts. Add selection/label budgets, map-level pilot scope, exposed external-set disclosure and all failed scientific effects. Record the V7 continuation failure and the V8 reliability/combined-Q failures together; do not promote either into a positive novelty claim. Parser correction and the present integrity audit are complete.

2. **Make the next experiment diagnose a cause.** Before more full-network KD, compare confidence-only versus stability-weighted correction, equal objective/update-size controls, and label-free support-coverage rules designed on source development only. Separate loss objective, linear solver and update magnitude. Include a source-selected conservative/no-update control. The current gain+2% proxy is demonstrably overconfident on some water misses; repeating F4 unchanged is not justified. A support set selected after looking at these query outcomes would create new tuning bias and cannot be called prospective.

3. **Strengthen the small-student deployment branch.** Retain R4 FP32/FP16 as working artifacts, then require a competent existing QAT control, source-quality tolerances, all-event accuracy and a compatible native engine before claiming extreme low-bit deployment. Log flags/dtypes at both timing and accuracy boundaries. Benchmark true cold/reused-engine adaptation cost and repeat workstation timings; measure target edge hardware only when it exists. Avoid spending the whole remainder on driver/backend repair. The severely poor EMSR333 parent water IoU (~8.43%) remains an application limitation despite strong source validation.

4. **Acquire missing evidence before manuscript closure.** The actual reviewer letter and `test.txt` were not located; concern IDs currently inherit prompt summaries. Independent verified dry scenes are still needed for real dry false-alarm claims. A second-teacher screen needs an installed task-adapted water head, acquisition dates/licence, verified co-registration and compatible SAR/optical preprocessing. Matching names or a pretrained encoder are insufficient. Any new teacher/data preparation or compute reallocation needs a separately bounded task; none was scheduled here.

5. **Confirm on valid independent transfer units.** Use new disjoint support/query episodes and, if possible, a prospectively reserved event set. Keep the original 3-event pilot descriptive, inspect every worsening case and distinguish training-seed from event uncertainty. All current external benchmarks are research-exposed; no re-freeze makes them untouched. A reliability or compression interaction needs replication before a strong method claim. Extra labelled selection must have equal-label direct-student controls.

The most defensible present contribution is a carefully audited tiny-student accuracy/storage tradeoff plus evidence that useful teacher corrections and harmful imitation coexist. V8 identifies why a plausible confidence/stability update is unsafe in this regime and supplies matched negative controls. The next method must solve that demonstrated failure and beat conventional KD/calibration at equal cost before it can support a new accuracy-method contribution. Known ridge-based adaptation and multimodal foundation-assisted flood mapping remain prior art; a bounded literature check is not novelty clearance.

## Deliverables and reproduction

The immutable initial manifest, status/ledger, job/launch receipts and original180 rows remain in the run directory. New audited outputs are `analysis_20261012/audit_summary.json`, `adaptation_summary.csv`, `adaptation_events.csv`, `paired_controls.csv`, `joint_errors_by_class.csv`, `headroom_summary.csv`, `support_audit.csv`, `reliability_audit.csv`, `solver_audit.csv`, `ofa_learning_curves_corrected.csv`, `ofa_learning_efficiency_corrected.csv`, `deployment_audit.csv` and the three figures. `reviewer_response_map.md` retains point-to-evidence mapping with the missing-letter qualification.

CPU reproduction uses the installed esi environment; it performs no GPU job or experiment relaunch:

```bash
R=/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v8_20261012T123202Z
P=/home/schmalstieg/miniconda3/envs/esi/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$P" "$R/analysis_20261012/audit_analysis.py"
CUDA_VISIBLE_DEVICES='' "$P" "$R/analysis_20261012/build_report.py"
```

No further experiments are running, reserved or automatically queued. Remaining research work above is an outline, not a claim of completed validation.
