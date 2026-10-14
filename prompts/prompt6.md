# FloodDistill: investigate the failed compression pilot and test a narrower mechanism
Version 3 — 8 October 2026

## Task

Work in the existing FloodDistill research repository. Audit the executed compression-v2 implementation, implement the bounded experiments below, run preflight, and start the existing detached job runner once its checks pass. Complete useful implementation and diagnostic work without waiting for routine implementation decisions. Do not stop at a plan or unexecuted launch command when the repository, data and authorized GPU are available.

The scientific objective is to explain the observed failures and test whether quantization-induced attention instability and task-specific quantization damage are better intervention targets than a global teacher-feature metric. This is a hypothesis, not a guaranteed novel method or expected positive result. Keep the negative v2 evidence.

Use a new research worktree/branch when needed. Preserve existing results, historical failures, immutable capsules, manifests and unrelated jobs. Do not rewrite the manuscript in this task. No NAS, supernet, performance-driven graph generation, broad hyperparameter search, new scheduling framework or multi-agent delegation is required.

## 1. Recover the actual execution context

Read applicable AGENTS.md, git status and the current research code. Verify rather than assume these provenance pointers:

- Repository: /home/schmalstieg/repos/FloodDistill
- Python: /home/schmalstieg/miniconda3/envs/esi/bin/python
- Main data root: /home/schmalstieg/data
- Experiment directory: experiments/revision/compression_v2_20261007/
- Original cohort: /data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006
- Student checkpoints: /data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007/
- Frozen training capsule identifier: 140df39f28974301
- v2 report: report_20261008/compression_v2_full_report.txt
- Machine-readable evidence: pilot_collection_verified.json, completion_audit.json, verified_collection_spec.json and repair_terminal_*.json
- Prior specification: codex_compression_prompt_v2.txt

Discover current source paths with rg. Likely reuse points include training/trainer.py, training/losses.py, models/student.py, models/qblocks.py, scripts/quantize.py, data/pseudo_labeled.py, evaluation/revision_eval.py and experiments/revision_queue.py. Inspect the exact EfficientViT dependency version and LiteMLA implementation. Do not infer its epsilon, quantizer placement or derivative from a generic online implementation.

The report bundle contains results/receipts, not the original frozen trainer, weights or actual exports. Access and hash those on this machine. A receipt claiming verification is not a substitute for checking accessible bytes. If the execution environment is absent, finish the audit/code that is possible and identify missing paths explicitly; do not invent a launch or training outcome.

## 2. Evidence to preserve and reproduce

The completed pilot was one training seed, not 98 independent experiments. It completed three FP students, nine QAT continuations, three PTQ maps, four interaction fits and twelve terminal checkpoints. Most successful queue nodes were explicit skips. The collection failure was an omitted figures/fig_style.py dependency; export-only recovery repaired warmed TorchScript execution mismatches without retraining. FP16 remained blocked. Recorded summed GPU-command wall time was approximately 1.576 hours.

Architecture parameter counts:
A0 704714; A1 190870; A2 62277 (untrained); A3 30922.
A1 uses width .5 and full repeat depth; A3 uses width .25 and half repeat depth. Their difference cannot be attributed to width alone. The resolved A2 and A3 repeat structures are identical despite nominal depth .75 versus .5; inspect actual block indices.

Source validation mIoU, percent:
- A0 FP32 84.155
- A1 FP32 83.595; PTQ W4A8 78.952; QAT B1 79.635; B2 79.907
- A3 FP32 82.699; PTQ W4A8 82.200; QAT B1 82.252; B2 81.863
- A3 PTQ W8A8 82.335; QAT B1 82.133; B2 82.235
- A3 PTQ W4A4 68.990; QAT B1 69.270; B2 70.545; B3 70.269

B1 is task/logit KD QAT; B2 adds isotropic feature MSE; B3 adds the v2 interaction metric.
After uncertainty shrinkage M was exactly identity in A1 W4A8, A3 W4A8 and A3 W8A8. Duplicate B3 runs were correctly skipped. A3 W4A4 retained one direction, condition number 1.093432 and ||M-I||F .091693. The B3-B2 source difference was -0.276 percentage points. S11 test/STURM/WF differences were -1.530/-.560/+.156 points. These heterogeneous one-seed observations do not establish benefit, equivalence or impossibility.

A1 FP32 was the smallest measured artifact within the prespecified one-point SOURCE tolerance from A0, with 1183268 vs 3247524 serialized bytes. A3 FP32 was 381984 bytes but missed that source tolerance. All W4/W8 artifacts were floating-weight fake-quant TorchScript, not packed low-bit engines.

Critical training-log facts to verify independently:
- Almost all QAT updates were globally clipped at norm 1.
- Across epochs, A1 W4A8 B2 mean pre-clip norms ranged roughly 9.56e5 to 3.405e8.
- A3 W4A4 B2 epoch means ranged roughly 5.297e9 to 1.300e10.
- Even A3 W8A8 B2 had epoch means roughly 1.082e3 to 3.298e5.
- Large means can reflect heavy tails; recover per-step distributions.
- This does NOT prove zero optimizer movement: Adam's moments and normalization matter.

Treat the report's valid negative comparison as preserved evidence while investigating whether the shared optimizer/quantization regime prevented an informative mechanism test.

## 3. Keep the scientific protocol explicit

Task: binary surface-water segmentation, not distinguishing new inundation from permanent water.

Frozen teacher: Prithvi-EO-2.0 300M task teacher, expected SHA256
3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60

Primary training cohort: restored original N2500. Expected pool manifest SHA256:
1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60
Preserve historical lineage limitations; do not claim absent historical hashes were recovered.

Six bands B2,B3,B4,B8A,B11,B12. Existing normalization:
mean [.1413,.138,.1235,.309,.2045,.1191]
std [.0741,.0737,.0869,.118,.0977,.0766]
Do not normalize cached normalized tensors twice.
Train 224; evaluate 512; preserve resolution, stride, valid support, augmentations, class meanings and label masks.

Preserve the executed v2 base objective:
0.5 focal(gamma=2) + 0.5 T^2 KL(p_teacher || p_student), T=4.
Teacher is frozen/eval on the same augmented input.
FP: AdamW lr5e-5, wd.05, batch16/drop_last, 7800 updates, cosine, no AMP, clip5.
Matched QAT: lr1e-5, wd1e-5, clip1, 1560 updates, QDQ on throughout, BN affine/running state frozen as in v2. Preserve any other actual optimizer/scheduler settings after reading the capsule.
Per arm record common initialization, quantizer state, batch order, optimizer state, selected and completed updates.

Select checkpoints using the existing 89-image S11 validation, pooled two-class mIoU, every 312 QAT updates, strict improvement with earlier ties. Report final-update results as well as selected checkpoints. The validation selection data is reused; its intervals are descriptive.

The legacy quantizer uses those 89 validation INPUTS for range calibration. Reproduce that baseline. New causal diagnostics, surrogate thresholds, loss-strength fitting, alternative learned quantizer parameters and research statistics use TRAINING data only. Do not use validation labels to fit them. If calibration inputs change, include a matched calibration-only control.

Already inspected S11 test, STURM and WF are now explicitly research-exposed. They can be frozen secondary terminal readouts, never described as untouched confirmation. Seek a genuinely new event/site holdout only if authorized data exists, register it before access, and disclose teacher/training geographic exposure. Otherwise qualify generalization claims. Do not invent a holdout.

Maintain S11_TEST512_LEGACY, P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1, P3_WF_NATIVE_ZERO_PAD512_V1 and the same full/known-overlap-excluded supports. Existing exclusions do not restore independence. Do not use these external results to change methods, launch gates or hyperparameters.

## 4. Wave A: inexpensive causal diagnosis before training new methods

Reserve at most 1 GPU-command hour for this wave. Reuse seed 42 checkpoints. Main cells: A3 W4A4, A1 W4A8, with A3 W8A8 as an easier control.

Create deterministic TRAIN probe lists, preferably reusing the existing disjoint event-group feature pools: one for discovery/calibration and the other for verification. Event IDs can revisit sites; group by site as well where available. This split is held out from diagnostic fitting, not from the FP student's training. Include unaugmented and exactly replayed augmented 224 inputs, plus a small 512 geometry probe. Use 32 batches at most for gradient instrumentation; do not make a full expensive new benchmark.

### 4.1 Check the gradient measurement and update semantics

Independently recompute sqrt(sum_parameter sum_element grad^2) in float64 after backward and before clipping; compare against the logger. Verify no accumulation across steps, teacher/adapter parameters are frozen/excluded as intended, gradients are reset after diagnostic autograd calls, loss reductions have the intended valid-pixel denominators, and no duplicate hooks or scaler/unscale errors exist.

Check fake-quant forward/backward formula, clipping derivative, scales, signed ranges, rounding, BN/fusion order and placement. Check step0, a later checkpoint, and a quantization-bypass branch with identical weights and input. Log optimizer parameter groups, LR, actual update/weight norms, moment norms and fraction of nonzero updates.

Collect median/p95/max norms by module and step, not only epoch means. Measure gradient concentration
Gshare_l = ||g_l||_2^2 / sum_j ||g_j||_2^2
before global clipping, plus cosine(task gradient, feature gradient) on a fixed common block.
Avoid conclusions about effective learning from global clipping fractions alone.

### 4.2 Inspect the attention quotient

For each actual normalized linear-attention head/query, expose its numerator N, nonnegative denominator D, epsilon and output Y=N/(D+epsilon). Record matched quantized and quantization-bypassed values, zeros and positive quantiles, ReLU Q/K support, quantizer saturation/zero occupancy, and activation/gradient finiteness.

Do not quantize currently floating attention matmuls just to create the hypothesis. Quantized upstream convolutions can perturb their inputs even if the matmuls remain FP32.

A large reciprocal alone does not prove a large parameter gradient. Trace the combined numerator/denominator derivative through the actual Q/K graph: shared dependencies can cancel terms, and zero ReLU gates can annihilate them. Include exactly zero and small-positive denominators as distinct cases. Establish the proposed causal path with module/parameter gradients and interventions, not denominator histograms alone.

Distinguish:
(a) already small FP denominator;
(b) denominator collapse introduced by quantization;
(c) large upstream gradient for another reason;
(d) bad logging or quantizer derivative implementation.

For a train-fitted per-head threshold tau_l and matched D_F,D_Q, report
C_l = P[D_F > tau_l AND D_Q < D_F/16].
Choose tau_l as the first percentile of strictly positive FP D on the calibration pool, with the actual epsilon as lower floor. If support is too sparse, record that rather than manufacturing a threshold.

Do local counterfactual probes with fixed weights: bypass only input/stem activation QDQ, only Q/K-producing quantizers, only V-path quantizers, and only decoder quantizers, following actual graph connectivity. Change neither labels nor model weights. Compare teacher KL, source diagnostics, D-collapse and gradients. Changing a quantizer may not isolate Q/K when a fused QKV convolution shares its input; label the true scope.

If stage interventions are informative, measure at most the two leading stage pairs:
H(A)=L(all-Q)-L(quantizers in A bypassed).
Pair excess H({l,m})-H({l})-H({m}) measures nonadditive repair under these fixed interventions. It is not a Shapley value, proof of a real-world cause, or an additive bit-allocation guarantee.

### 4.3 Check obvious quantizer and geometry weaknesses

The reported activation quantizer is per-tensor signed symmetric. At a nonnegative ReLU6 site, signed 4-bit [-8,7] uses only eight nonnegative levels. A range 6 site therefore has step about 6/7 rather than 6/15. This is a conventional baseline issue, not a new contribution. Never switch signed normalized input, residual sums or linear tensors to unsigned on the basis of observed samples alone.

Run fixed-checkpoint PTQ comparisons:
- original signed scheme;
- unsigned affine quantization only at graph-proven nonnegative sites;
- W4A4 body with first/stem activation at8 bits;
- W4A4 body with documented attention-sensitive projections at8 bits, only if probes identified them.

Disclose exact mixed-precision exceptions and their parameters/MACs/activation bytes. These probes are not W4A4-everywhere results. Use the same calibration inputs for paired variants.

Log observer coverage, data outliers, dead bins, and 224-vs512 activation/denominator shifts. The current first-convolution A3 W4A4 activation step is about 1.434355 in normalized units: audit whether spectral differences are destroyed at the input. Do not infer this from a water-index proxy alone.

If fixed scales are a dominant issue, implement a faithful LSQ baseline with appropriate signedness and published step-size gradient scaling. Mark per-channel adaptations as adaptations. Do not silently fold it into every new method and attribute its gain to distillation.

### 4.4 Check the old feature statistic without forcing it to succeed

Recompute f=t-s_F, q=s_F-s_Q, e=f+q and
C_e=C_f+C_q+2R
under identical weights, BN state, input and spatial weighting. The report's stored scalar identity is internally consistent. Negative trace(R) is net feature cancellation, not necessarily improved GT performance.

Check 8x8 interpolation/mask sampling, small-water coverage, teacher PCA, frozen adapter conditioning and feature semantics. High retained teacher variance does not prove retained task information. A final-encoder tap can miss stem/decoder damage.

The hard-cell top eigenvalue .307848 exceeded the bootstrap delta .248385 only weakly. Other cells had top eigenvalues below their deltas. Do not remove uncertainty shrinkage or choose a bigger gamma after seeing results just to create a nonidentity metric. If exploring local/class/boundary statistics later, predeclare them and fit/verify on separate TRAIN groups.

## 5. Candidate method A: counterfactual normalization gradients

Implement only if the audit confirms that quantization-induced denominator collapse materially contributes to the large gradients. This is a training-time surrogate; it must preserve the deployed quantized forward graph exactly. It is not an exact gradient of that deployed computation and carries no convergence guarantee.

At every affected head/query:
d = D_Q + epsilon
Y = N_Q / d
Obtain D_F from a no-grad quantization-bypassed pass of the SAME current student weights on the SAME augmented input, with identical frozen BN state and aligned head/query positions.
This is a student counterfactual, not the foundation teacher's differently shaped attention.
Restore all QDQ flags after the reference pass; do not mutate observers or accumulate reference gradients.

Let
c = 1[D_F > tau_l AND D_Q < D_F/16]
d_bar = max(d, D_F/16 + epsilon) if c=1, otherwise d.

All reference values, c, tau_l and d_bar are stop-gradient quantities. At inference, neither D_F nor this mechanism is present.

For an incoming vector gradient G, use the local surrogate VJP
grad_N = G / d_bar
grad_D = -sum_value_channels(G * stopgrad(Y)) / d_bar.

Equivalently, a stop-gradient expression is
Y_sur = stopgrad(Y)
      + ((N_Q-stopgrad(N_Q))
         - stopgrad(Y)*(D_Q-stopgrad(D_Q))) / stopgrad(d_bar).

Forward Y_sur equals Y. Relative to the ordinary quotient VJP, both local derivatives are multiplied by d/d_bar in (0,1]. The numerator derivative is bounded by 1/d_bar; the denominator derivative also depends on Y. This is a local bound, not a bound on the entire model's gradient. A tensor-valued denominator requires the correct reduction/broadcast axes.

Implement a custom autograd function or an equivalent numerically verified expression. Export the original quotient; no training-only reference, autograd wrapper, PCA or adapter should remain in the inference artifact.

Required controls:
1. original QAT;
2. a conventional fixed backward denominator floor tau_l, at the same modules, with exact original forward;
3. this counterfactual rule;
4. a separately labeled forward floor or higher-precision sensitive projection if needed to distinguish forward damage from backward instability.

The first percentile and ratio 16 are preregistered defaults, not evidence of optimality. At most one later ratio 4 sensitivity is allowed after a source-only success gate. If the counterfactual rule loses to the simple floor, report that the extra mechanism is unnecessary. If only the forward repair works, the claimed training-only rescue is unsupported.

Do not confuse a diagnostic decrease in gradient magnitude with a scientific success. Require adequate actual updates and improved source accuracy or teacher-risk behavior under matched compute.

## 6. Candidate method B: distill excess task loss, retaining beneficial changes

This replaces the assumption that every quantization displacement should be minimized. First compute exact task-level diagnostics; only then test the auxiliary loss.

For binary logits, use margins m=z_water-z_background. With temperature T:
a=m_F/T, b=m_Q/T, p_F=sigmoid(a), p_T=sigmoid(m_teacher/T), p_Q=sigmoid(b).
Write ell(p,u)=softplus(u)-p*u.

The exact identity is
Delta = T^2[ell(p_T,b)-ell(p_T,a)]
      = T^2[(p_F-p_T)*(b-a) + KL(Bern(p_F)||Bern(p_Q))].

The KL term is nonnegative; the signed first term can offset it. Delta<0 means quantization improved agreement with the teacher under this loss. It does not prove GT improvement or superiority to the teacher. Compute KL with stable log-sigmoid/log-softmax arithmetic; do not clamp probabilities and then claim exact equality.

Use the same CURRENT student weights with QDQ off/on for this diagnostic. Per image and event, report:
- H_T=E[max(Delta,0)] and B_T=E[max(-Delta,0)];
- E[Delta]=H_T-B_T, sign proportions and quantiles;
- signed interaction term and nonnegative KL term separately;
- class and boundary/interior breakdowns from valid training pseudo masks, with teacher-confidence strata and support counts.
Use equal image mass with fixed valid-pixel reductions; report alternative class-balanced summaries separately.

Training auxiliary:
Use the frozen pre-QAT FP anchor S_0 to define a stable loss budget on the same augmented input:
d_i = T^2[KL(p_T,i || p_Q,i) - KL(p_T,i || p_S0,i)].
Both the teacher and S_0 are no-grad. A moving same-weight FP reference is appropriate for diagnosis but permits a changing loss budget; do not silently substitute it into this auxiliary.

Define the smooth one-sided penalty
psi_k(d)=0, if d<=0;
         d^2/(2k), if 0<d<k;
         d-k/2, if d>=k.
Its derivative lies in[0,1]. Set k to the 75th percentile of positive d on the TRAIN calibration pool, lower bounded by 1e-6; freeze it. If there are no positive d values, record a degenerate auxiliary and skip its duplicate training.

L_new = L_base + lambda * E_valid[psi_k(d)].
Preserve the base focal/KL losses and their reductions. This auxiliary does not penalize beating the frozen FP anchor. It remains a form of task-loss reweighting; the exact identity and the hinge alone are not a novelty claim. Training d includes subsequent learning drift relative to S_0, whereas the same-weight diagnostic Delta isolates quantization's effect at that checkpoint.

Fit lambda on at most 32 replayable TRAIN minibatches after the chosen backward rule is active. Target median auxiliary/task gradient ratio 0.1 on a fixed common block; disclose whole-model ratios too. Reject degenerate ratios rather than allowing arbitrarily large lambda. Freeze lambda. Do not add a broad sweep.

Necessary controls include:
- ordinary KD with the same backward/quantizer repair;
- ordinary feature MSE;
- a teacher-entropy or teacher-KL hard-pixel weighting baseline with the same maximum additional KL multiplier and approximately matched initial gradient contribution;
- the candidate auxiliary.
If only generic hard-example weighting is needed, narrow the claim accordingly.

Foundation-teacher errors, confident errors and reference-anchor weakness are expected failure modes. Retain the existing human/source masks for evaluation. Do not treat teacher agreement, zero teacher water or MNDWI as ground truth.

## 7. Fixed experiments, not an architecture search

Do not launch every optional combination.

Wave B: main seed 42 screen on A3 W4A4, same 1560 updates and common FP anchor.

First compare the original baseline and the conventional quantizer/backward controls warranted by Wave A. Reuse a verified exact v2 baseline if its executable state is identical; run one deterministic replay to check, rather than retraining unchanged arms without cause.

Then use a small factorial to separate the two new effects:
- B0: ordinary KD, ordinary backward;
- BN: ordinary KD + candidate A;
- BR: ordinary KD + candidate B;
- BNR: ordinary KD + A+B.

Keep common quantizer settings across this factorial. If a standard quantizer correction is needed, define it first and run all four on that same corrected baseline; retain the correction-only comparison to v2. Candidate A is omitted with an explicit reason if its diagnostic gate fails.

Include the fixed backward floor control and entropy/hard-pixel weighting control. Limit this screen to ten QAT continuations including any necessary MSE and LSQ baseline; fit their total measured cost before launch.

Retest v2 B2 and, only if its freshly calibrated metric is nonidentity, B3 once under the stable shared recipe. This preserves a chance to discover that the old method was masked by unstable training. These are new conditions, not replacements for the old negative result. Keep v2's uncertainty and amplitude rules. Do not revive its full skipped grid automatically.

Wave C: explanation of the A1/A3 reversal.
Predeclare a 2x2 graph family:
width .50/depth1.00 (existing A1);
width .25/depth .50 (existing A3);
width .50/depth .50 (new crossed graph);
width .25/depth1.00 (new crossed graph).
Use the same legal transition/head rules, activation family, decoder scaling and initialization policy. Freeze resolved graphs before observing their accuracy. Keep full input resolution. Train only the two missing FP graphs first, then perform paired W4A8 PTQ and the small fixed probe suite. This estimates width, depth and their interaction descriptively; seed 42 alone does not establish a scaling law.

Do not prioritize A2 as a depth comparison: its resolved retained blocks match A3. A2 can fill the deployment frontier later, but answers a different question.

Run Wave C only if the fresh runtime estimate fits the cap after the diagnostic/core jobs and deployment reserve. Report deferred nodes as deferred, not failed or pending in a nonexistent worker.

## 8. A high-risk reserve idea if stem quantization is the culprit

Do not schedule this by default. If input/stem bypass accounts for most W4A4 damage, test a cheap function-preserving spectral basis change at fixed checkpoints before any new training.

On the normalized six-channel input z, use a fixed orthogonal 2x2 rotation of the green and SWIR1 channels. One axis is proportional to
sigma_green*z_green - sigma_SWIR1*z_SWIR1.
Choose the perpendicular unit axis for the other coordinate, leaving four bands unchanged. This emphasizes a water-sensitive spectral contrast without asserting that an index threshold supplies correct labels.

Let z'=U z and replace the first convolution's input-channel weights with W'=W U^T at each kernel position. Before quantization, W'z'=Wz exactly. Because U is linear with no offset, zero padding remains zero; do not insert a mean-shift and silently break border equivalence. The physical contrast has an additional known mean offset, which must not be mistaken for zero in normalized coordinates.

Place U before the actual first activation quantizer and recalibrate the transformed model with the matched input list. Count the six-channel transform's actual deployment overhead. Compare identity, this fixed physical rotation, a fixed-seed random orthogonal rotation and training-only PCA at identical precision/granularity. If the actual quantizer is invariant to the transformation or equivalent compensated scaling, record the inert intervention.

The broad concept of quantization-friendly rotations is established. Possible novelty would require task-specific evidence beyond random/PCA rotations, actual export, and a focused literature comparison. This is a bounded exploratory alternative, not another promised contribution.

## 9. Acceptance gates and evaluation

Predeclare hypotheses before new jobs:
H1: low-bit quantization causes denominator collapse absent or substantially rarer in the matched FP branch, and targeted precision interventions reduce the associated gradient pathology on the verification pool.
H2: candidate A improves useful optimization/accuracy beyond a conventional floor without changing inference-time operators.
H3: candidate B improves source results beyond ordinary KD, MSE and generic hard-pixel weighting under the same stable recipe.
H4: the width/depth factorial explains the A1/A3 quantization-sensitivity reversal reproducibly.

Audit/causal evidence may be useful even if all accuracy hypotheses fail.

Use a provisional practical source gate of at least 0.5 percentage points pooled mIoU over the strongest relevant matched control, with no water-IoU deterioration, plus the hypothesized diagnostic change. A mechanism failing this screen can receive one specifically justified implementation repair, not repeated score-driven tuning. Preserve all attempts.

For a source-gated winner, prioritize seed 64 before any expansion; then seeds 91 and 107 if the cap permits. Every seed needs its own FP anchor and matched controls. Freeze graph/method/quantizer/hyperparameters before additional seeds. Reused seed 42 is developmental; report seedwise effects rather than hiding it in a single pooled average. An initially nonsignificant one-seed result is not an equivalence result.

Before a publication claim, require a faithful close baseline such as SQuaT or QFD, with architecture/teacher adaptation described and official code inspected. Candidate A also requires comparison to relevant linear-attention stabilization and teacher-intervention methods. Budget these only for a successful source screen; document pending comparisons honestly. A simplified proxy is not a faithful reproduction.

Report:
- source pooled mIoU/water IoU and per-event paired changes;
- actual selected and completed updates and wall-time matched KD where extra passes materially change cost;
- H_T,B_T, gradient distributions/concentration, denominator collapse and code utilization;
- GT harmful flips H and repairs B for SAME-weight QDQ-off/on predictions, with accuracy_Q-accuracy_F=B-H; do not apply that identity to IoU;
- seed variability separately from paired whole-event bootstrap intervals;
- boundary F1 only with a declared pixel/geospatial tolerance, identical valid-mask erosion and support counts;
- all failures, null identities, unsupported exports and skipped conditions.

Pixel observations are spatially correlated. Use paired event/site resampling, not millions of pixels as independent trials. Small event counts and reused source selection limit inference. No independent dry-reference claim is allowed without verified human ground truth.

Terminal external evaluation runs only after source method/seed/checkpoint freeze. Preserve existing geometry and overlap exclusions; label these secondary, research-exposed benchmarks.

## 10. Deployment is a separate required workstream

The v2 fake-quant artifacts were larger and slower than their native FP counterparts. Do not report hypothetical bits as measured compression. Preserve FP32 and report three separate quantities:
theoretical packed weights; actual serialized inference bytes; measured backend/device performance.

First fix the specific FP16 UpSampleLayer dtype mismatch in a narrowly scoped branch: inspect why interpolation casts inputs to float before a half convolution; use supported FP32 interpolation followed by restoration of the intended incoming dtype, or a supported mixed-precision plan. Validate numerical/accuracy effects as a new export variant. Do not loosen tolerances after seeing mismatches.

Use the existing warmed export validation and identical runtime configuration, with real data probes as well as 224/512/zero probes. Check final selected artifacts, not only zero-input argmax. If the optimized TorchScript plan remains mismatched, preserve the common disabled-optimization measurement and separately record the unresolved optimized backend.

Attempt a supported genuine low-bit deployment only after inspecting the installed backend and target hardware. INT4 convolution support is not implied by INT4 weight-only GEMM support. If supported, serialize/load the real engine and measure accuracy, batch1 latency median/p95, size and memory with explicit mixed-precision maps. If unavailable, report blocked; do not build a new custom kernel framework during this pilot.

Maintain the source-only one-point-tolerance deployment selector. Do not use the unexpectedly good A3 external score to alter this selector retrospectively. For FP16 or packed-engine candidates, rerun their actual source accuracy.

## 11. Mathematical and implementation verification

Before heavy jobs:
- Verify loss-decomposition equality in float64 for random and extreme finite margins, both T=1 and T=4.
- Verify H_T-B_T=mean Delta and the GT flip identity on controlled cases.
- Verify the one-sided loss has zero auxiliary gradient when d<=0, bounded derivative and no gradient into S_0 or the teacher.
- Verify candidate A forward equality for random, tiny-denominator and zero-numerator inputs; grad_N/grad_D against its specified surrogate VJP.
- Do NOT require finite differences of the unchanged forward to match the deliberately modified backward. Test the local surrogate with stop-gradient values held fixed, and ordinary-forward gradients only when the cap is inactive.
- Verify the QDQ-disabled case or D_Q=D_F leaves candidate A inactive and matches ordinary gradients.
- Verify head/query alignment, denominator broadcast axes and reduction over value channels.
- Verify reference passes preserve quantizer flags, observer state, BN buffers and RNG-dependent data order.
- Check feature diagnostics do not accumulate training gradients.
- Check unsigned activation changes only at graph-proven nonnegative tensors.
- Verify any spectral rotation's unquantized convolution identity, including padded borders.
- Run the report collector from a clean frozen capsule before admitting long jobs; include its actual dependencies.
- Smoke-test export load and warmed parity before measuring speed.

Only test concrete mathematical, implementation and reproducibility risks; avoid a large generic test rewrite.

## 12. Bounded scheduling and autonomous completion

Reuse the existing revision_queue.py interface and DAG/receipt format. Discover current active workers, GPU UUID, free memory and repository conventions. Default to one heavy job on the verified RTX4090; do not reuse old two-device assignments automatically or interrupt unrelated jobs.

Set a NEW incremental hard cap of 12 GPU-command hours for this task, or the smaller explicit existing budget. This is a new proposed pilot allowance, not unspent credit silently borrowed from a historical worst-case reservation. Record user/repository budget rules if they override it.

Suggested reservations:
- Wave A diagnosis <=1h;
- core controlled QAT screen <=4h;
- width/depth explanation <=2h;
- source-gated replication/closest baselines <=3h;
- deployment and terminal collection <=2h.

Use measured smoke-job costs with conservative headroom, not optimistic estimates alone. Track actual command seconds plus outstanding maximum reservations across every new wave and recovery. Time out individual jobs; stop admission when the cap cannot be respected. One retry for a diagnosed technical failure; no unbounded recovery or tuning. Reallocate unused reservations explicitly while retaining the total cap.

Freeze a new manifest per wave. Do not edit a running manifest or use a skipped-success receipt as completed scientific work. Gate-dependent jobs need distinct succeeded/skipped/blocked outcomes and reasons.

Implement and verify the jobs, start the detached queue with the existing start command, then independently check its worker PID/state and first substantive receipt or log. Return the actual running/finished/blocked state and exact status/resume commands. A process exit code alone is insufficient evidence of completed science.

If a scientific gate fails, collect and explain the negative result, launch only independently justified diagnostic/deployment work, and stop automatic expansion. If infrastructure is unavailable, provide the ready code/configuration and precise blocker; never claim submitted jobs.

## 13. Required outputs

Keep outputs concise and machine-readable:
1. audit.md distinguishing observed facts, verified bugs, unresolved hypotheses and ruled-out explanations;
2. protocol_v3.yaml with resolved graph/quantizer variants, train groups, hypotheses, controls, gates, seed and budget policy;
3. minimal reusable implementation and focused tests;
4. immutable wave manifests, lineage hashes and complete job receipts;
5. diagnostic JSON and source/terminal CSVs with condition identifiers and actual precision;
6. report_v3.md with failure ledger, results against every relevant control, costs and a claim-evidence table;
7. a small next-report archive including the exact new trainer/loss/quantizer/attention code and tests, dependency versions and receipts, while keeping large weights/data outside git;
8. a launch/status receipt with worker state, reserved/actual GPU-command seconds, and queued/skipped/blocked work.

The novelty assessment must name what is inherited, what differs, which control isolates that difference and what evidence would falsify it. If the useful outcome is a corrected conventional QAT baseline or an informative negative mechanism result, say so plainly.

## Primary-source starting points for the related-work check

Read the actual methods and official implementations before adapting them. This list is a targeted starting point, not an exhaustive novelty search.

- EfficientViT, Cai et al., ICCV2023: https://hanlab.mit.edu/projects/efficientvit
- The Devil in Linear Transformer, Qin et al., EMNLP2022: https://aclanthology.org/2022.emnlp-main.473/
  Already identifies unstable gradients from linear-attention scaling; do not claim that general observation as new.
- Teacher Intervention, Kim et al., EACL2023: https://aclanthology.org/2023.eacl-main.64/
  Uses intact teacher signals to intervene in quantized propagation. Distinguish forward intervention from a student-counterfactual, inference-preserving backward rule.
- Learned Step Size Quantization, Esser et al., ICLR2020: https://arxiv.org/abs/1902.08153
- Quantized Feature Distillation, Zhu et al., AAAI2023: https://arxiv.org/abs/2307.10638
- SQuaT, Lee et al., AISTATS2026: https://proceedings.mlr.press/v300/lee26f.html
  Addresses student/teacher feature-range mismatch; merely quantizing teacher features is not new.
- Mitigating Negative Flips via Margin Preserving Training, official repository: https://github.com/miccunifi/negative_flip_MPT
  Negative-flip reduction and margin preservation also have prior art.
- QuS: Towards High-Performance EfficientViT on FPGA by Quantization and Streamline Co-Design: https://openreview.net/forum?id=B2DRmwStm9
  Relevant EfficientViT-specific prior work. The report-analysis session could retrieve its indexed title but not the full paper because of a verification wall; obtain and read it before detailed priority claims, and verify publication status.

Final instruction: investigate first, implement the smallest discriminating intervention, run matched controls, preserve failures, and let the evidence determine the claim.

