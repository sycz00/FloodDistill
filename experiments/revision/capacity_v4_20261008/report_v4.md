# FloodDistill capacity v4 — interim report (Stages 1–2)

*9 October 2026. Status: **interim**. Stage 1 is complete. Stage 2 is partial: 18 of 24 runs exist, and no run has external-benchmark scores yet. Stages 3–8 have not started. Every number below is S11-validation (89 scenes) or TRAIN-pool data unless stated otherwise. Simulated quantization only (`simulation_only`).*

---

## Part A — Plain-language summary (for non-specialists)

**What the project is doing.** A very large AI model (the "teacher", 300 million parameters) can map surface water in satellite images, but it is too big to run on a small satellite or a cheap device. We train tiny "student" models, from 30,000 to 700,000 parameters (about 1/10,000 to 1/400 of the teacher's size), to copy the teacher. We then try to shrink them further by storing their numbers with fewer bits ("quantization": 4 or 8 bits instead of 32).

**Finding 1: an earlier alarming result was a measurement error.** Earlier experiments showed that training the 4-bit models produced astronomically large internal error signals (up to 10¹¹ times normal). These were not real. The software that imitates low-precision arithmetic was dividing by tiny rounding leftovers, roughly like dividing by 0.000000005 instead of 0. We built a more faithful simulator that does the arithmetic on whole numbers, as real low-precision hardware would. With it, the error signals fall to 1–3× normal, and accuracy is unchanged. *Practical meaning:* low-bit training was never fundamentally broken. The earlier tool was measuring it wrongly.

**Finding 2: almost all the 4-bit damage comes from the model's first layer.** The very first layer, which reads the raw satellite bands, loses most of its information at 4 bits: about two-thirds of its input values round to zero. Giving only this layer 8 bits restores most of the lost accuracy. For one model, 8-bit storage for a single 36-number filter (18 extra bytes) recovers about three-quarters of the damage. A ranking that uses no human labels, only the teacher's opinion, picks the same layer. A simple rule of thumb ("give the few smallest layers 8 bits") does just as well.

**Finding 3: the training recipe may be leaving accuracy on the table.** One quantized student continued training at a 10× higher learning rate and ended about 3 points *more* accurate than its own full-precision parent. That suggests the standard recipe under-trains the tiny models. It is one run on one validation set, so it is a lead to test, not a result.

**Finding 4: model shape matters, and "deeper" is not automatically better.** Among six student shapes, the thin-and-deep one (X2) was clearly the worst, about 3.5 points below the others, consistently across 4 random restarts. Among the remaining shapes, differences of up to about 1 point cannot be told apart from run-to-run noise yet.

**What we cannot say yet.** We have not yet tested any model on the independent flood benchmarks (STURM, WorldFloods, Sen1Floods11-test). We therefore cannot yet say how small a model can be while still matching the teacher on real floods, or whether the small models beat a simple water index (MNDWI). Six training runs were cut off by a time limit while the shared GPUs were busy with other users' work. They need to be re-run.

---

## Part B — Expert report

### B1. Stage status and cost

| Stage | Status | GPU command-s | Notes |
|---|---|---|---|
| Facts F1–F11 | reproduced (fast set) | — | first runner failed on sandbox CUDA; host retry passed; failure preserved |
| 1 measurement repair | **done** | 3,372 (cap 7,200) | 1.1–1.7 complete; `protocol_v4.yaml` frozen 2026-10-08T12:35Z before any Stage 2 result (sha `2f97f32a…`) |
| 2 FP graph family | **partial**: 15/21 new + 3 reused | 18,202 (cap 21,600), incl. 6,010 lost to timeouts | 6 runs timed out (see B5); external eval deferred |
| 3–8 | not started | 0 | Stage 3 needs the Stage 2 anchors A3 s91/s107 |

Total used ≈ 6.0 of ≈ 30 GPU-h. Remaining plan ≈ 24 GPU-h, plus ≈ 1.5 h to retry the six runs.

### B2. Stage 1 results (fixed weights unless stated)

**H1, repaired gradients: PASS.** Under the adopted integer-domain simulator (`integer_domain_v4_exact_native_STE`, sha `7ede34a2…`), the per-step pre-clip norm p50/max (8 TRAIN batches) is reported against the FP frozen-BN norm (≈ 3.9):

| Cell | legacy fake-quant median | integer p50 / max | ratio p50 / max | fwd+bwd floor τ=1e-3 p50 | bwd-only floor p50 |
|---|---|---|---|---|---|
| A3 W8A8 | 7.5 | 3.38 / 5.40 | 0.9× / 1.4× | 4.6 | 4.4 |
| A3 W4A8 | 2.3e6 | 8.14 / 14.6 | 2.1× / 3.7× | 20 | 21 |
| A3 W4A4 | 2.1e10 | 4.54 / 7.77 | 1.2× / 2.0× | 634 | 1,337 |
| A1 W4A8 | 1.2e4 | 10.6 / 20.5 | 2.7× / 5.3× | 13.6 | 12.7 |

The criterion (median ≤ 10×, max ≤ 100×) is met in every cell. Both floors fail at W4A4. An independent float64 recompute agrees with the logger within 1 %, with no accumulation across steps and no duplicate hooks. The unit tests pass: exact-zero cancellation, 8/8 non-binary scale/bias tests, and identical STE masks and gradients. Parity: PTQ |Δ| ≤ 0.008 pp; the largest rescore of a selected v2 QAT checkpoint is 0.138 pp (gate 0.15). **Caveat kept:** argmax agreement on augmented TRAIN batches is 91.8 % at A3 W4A4 (99.8–99.9 % elsewhere). The two simulators therefore define slightly different W4A4 forward functions, so v2 source scores were re-scored and v2 external conclusions are not carried over.

**1.2/1.5 attribution and excess loss (S11-val, pooled mIoU / water-IoU; H = correct→wrong pixel flips, B = wrong→correct).**

| Counterfactual (others stay quantized) | A1 W4A8 | A3 W4A4 |
|---|---|---|
| all quantized | 78.98 / 63.57 (H 2.60 %, B 0.98 %) | 68.99 / 48.10 (H 7.22 %, B 1.65 %) |
| bypass stem input | **83.14 / 70.42** (H 0.85 %) | **76.20 / 58.21** (H 2.82 %) |
| bypass decoder | 79.56 / 64.53 | 71.23 / 51.31 |
| bypass fused QKV (Q, K and V jointly) | 78.44 / 62.75 | 69.88 / 49.27 |
| high-precision reference (all-W8 / W8A8; A3 FP = 82.70) | 82.73 | 82.35 |

The forward damage sits at the stem input. The attention path, which carried the gradient pathology (F2), carries almost no *forward* damage: bypassing it does not help A1. A V-only probe is **blocked**, not fabricated, because the fused QKV conv shares one quantized input. On TRAIN, the teacher excess loss is H_T 0.279 vs B_T 0.107 (A1 W4A8) and 0.548 vs 0.118 (A3 W4A4), so B_T is not negligible. The SB+XL gate for Stage 5 is therefore not closed by degeneracy.

**1.4(b)/U6, label-free mixed precision (A1 W4A8; all-W4 78.98, all-W8 82.73; single-seed PTQ screen).** The TRAIN teacher-excess-KL ranking vs the S11-val label ranking gives top-4 overlap 2/4 and Spearman 0.54. That meets the U6 gate (≥ 2/4). Both rankings put `input_stem…depth_conv` first by a wide margin.

| Allocation | extra weight bytes | mIoU | share of W4 damage recovered |
|---|---|---|---|
| label-free k=1 (stem depthwise) | 18 | 81.86 | 77 % |
| label-free k=2 | 126 | 82.36 | 90 % |
| label-free k=4 | 1,438 | 83.08 | 109 % |
| label-ranked k=4 (optimistic, same images) | 1,682 | 83.01 | 107 % |
| random k=4, 5 draws | 872–7,336 | 78.55–81.34 (median 79.31) | 9 % (median) |
| structural rule (≤ 64 weights or ≤ 8 out-ch, 6 layers) | **358** | **83.08** | 109 % |

**H2, second half: PASS (screen).** **U6 is not falsified**, because it beats random-k by a wide margin. However, the predeclared structural rule matches it with a quarter of the extra bytes. The label-free score therefore adds no measurable value over the structural rule here.

**1.6 optimizer reach (A3 W4A8 QAT continuation from A3 FP s42 = 82.70; 1,560 updates; seed 42).**

| lr | code-flip fraction | selected / final mIoU | grad p50 / max | clipped @1 |
|---|---|---|---|---|
| 1e-5 (v2) | 2.4 % | 82.34 / 82.34 | 4.6 / 137 | 100 % |
| 1e-4 | 12.5 % | 82.79 / 82.74 | 4.9 / 187 | 99.7 % |
| 5e-4 | 38.4 % | **85.91 / 85.67** | 4.4 / 65 | 99.1 % |

Two readings follow. (i) The v2 QAT barely moved the weights, consistent with F4. (ii) At lr 5e-4 the *4-bit-weight* model ends **+3.2 pp above its own FP parent**. The final-update score (85.67) shows this is not checkpoint-selection optimism. The most economical explanation is that the FP recipe (lr 5e-5, 7,800 updates) under-trains scratch students, so the extra high-lr updates continue FP optimisation rather than "repair quantization". This is one seed and is not a claim. It does change the interpretation of Stage 2 and of every FP-vs-quantized gap (see B6).

### B3. Stage 2 results (S11-val selected mIoU; external evaluation deferred)

| Graph | width / depth | params | n seeds | mean ± SD | selected epochs |
|---|---|---|---|---|---|
| A0 | 1.00 / 1.00 | 704,714 | 1 | 84.16 | 36 |
| A1 | 0.50 / 1.00 | 190,870 | 4 | 83.21 ± 0.79 | 50, 30, 28, 42 |
| X1 | 0.50 / 0.50 | 106,274 | 3 | 83.67 ± 1.02 | 44, 38, 32 |
| A2 | 0.35 / (≡ A3 depth, F9) | 62,277 | 4 | 82.81 ± 0.79 | 32, 50, 38, 42 |
| X2 | 0.25 / 1.00 | 55,398 | 4 | **79.56 ± 0.83** | 50, 50, 44, 38 |
| A3 | 0.25 / 0.50 | 30,922 | 2 | 83.16 ± 0.65 | 28, 44 |

Paired-seed differences (Student-t 95 % CI) are resolved only against X2: A1−X2 **+3.65 [+2.09, +5.20]** (n=4); A2−X2 **+3.25 [+2.33, +4.18]** (n=4); X1−X2 +4.50 [+1.28, +7.72] (n=3). A1−A2 is +0.40 [−1.15, +1.95], unresolved. Every A0 and A3 contrast has ≤ 2 pairs, so **Q1 is unanswered**. Q2 and H4 need STURM/WF and are **unanswered**. **Q3 (descriptive):** at width 0.25, doubling depth *costs* about 3.5 pp; at width 0.50, depth makes no resolved difference. That is a width × depth interaction in which the narrow-deep corner fails. Scratch seed SD on S11-val (0.65–1.0 pp) is 3–5× the pretrained F8 figure. The F8-derived MDE therefore under-states the noise for this family. Use the measured value: with 4 paired seeds, the CI half-width is ≈ 1.5 pp.

### B4. Claim–evidence table

| Claim | Evidence | Strength |
|---|---|---|
| v2 W4 gradient explosion was a fake-quant simulation artifact | H1 table; parity ≤ 0.14 pp | **established** (fixed weights; 4 cells × 8 batches) |
| 4-bit forward damage is mainly the stem input | stem bypass recovers 4.2 / 7.2 pp; F5 | established for A1 W4A8 and A3 W4A4 (one checkpoint each) |
| 18 B of 8-bit stem weights recover ~¾ of A1 W4 weight damage, chosen label-free | U6 table | screen (1 seed, PTQ); structural rule equally good |
| FP recipe under-trains scratch students | lr 5e-4 +3.2 pp; X2 late selection | **hypothesis**; to be tested by the Stage 4 LR check |
| narrow-deep (X2) is worse than the other shapes | 4-seed paired CIs exclude 0 | resolved on S11-val only; external transfer untested |
| A1 < A3 externally (H4); A0 vs A3 (Q1); knee; R2 vs A0 (H3) | — | **not yet tested** |

### B5. Failure ledger

* **Observed:** six Stage 2 runs hit the enforced 1,020 s limit (1.2 × the ~850 s estimate) at epoch 37–41/50 while both GPUs were shared with foreign jobs. The same graphs finish in 760–860 s when uncontended (A0 s42: 829 s). Partial checkpoints are not used.
* **Verified bugs, fixed opt-in:** the v2 fake-quant conv divides by float32 rounding residue (the F2/F3 mechanism). The native and composed STE weight-gradient accumulation differed by up to 3.1e-5; this was fixed in `simulator_v4_ste_exact.py` and revalidated. Old code was not modified.
* **Ruled out:** logger error and gradient accumulation (float64 recompute); a real optimisation barrier at W4 (the integer simulator gives normal norms); attention-path forward damage as the main 4-bit loss (QKV bypass gives ≤ +0.9 pp).
* **Unresolved:** the 8 % W4A4 forward disagreement between simulators (first divergence at the LiteMLA `proj` output); the under-training hypothesis; the reason for X2's deficit (capacity vs optimisation).

### B6. Interpretation and consequences for the plan

1. **Stage 3 is now a fair test.** With the gradient artifact gone, the remaining low-bit gap is a forward/representation problem concentrated at the stem. That is exactly what the SB arm targets (stem ≥ 8 bit, unsigned grids at ReLU6 outputs). Stop rule S-a ("SB within 0.5 pp of FP at W4A8") looks likely to fire: stem bypass alone already reaches FP at A1 W4A8. In that case Stage 5 stays closed, and the honest outcome is *"standard tools suffice once the simulator and the stem are handled"*.
2. **The FP anchors may be too weak.** The Stage 1.6 result implies that "FP" in every gap is an under-trained FP. The protocol's own check (A3 at lr 5e-4, seed 42, ≥ 1.0 pp on S11-val ⇒ extend the LR search to A0/A1/A2/A3N) should run **before** Stage 3 compares quantized vs FP. Otherwise the gaps are confounded with training length. This is a recipe hyperparameter chosen on S11-val, which ground rule 4 permits.
3. **Capacity reading so far:** on in-domain validation, 31 k-parameter A3 is within noise of the 191 k A1 and probably of the 705 k A0 (one seed: 1.0 pp). The capacity floor is therefore below 31 k for this task. That supports running the Stage 4 R2/R4 rungs. The failure of X2 warns that "smaller" must mean *wide-enough and shallow*, not thin and deep.
4. **Novelty ledger (no priority sentences: the related-work check has not been run).**

| Candidate | Inherited | What differs | Isolating control | Falsifier |
|---|---|---|---|---|
| integer-domain QDQ simulator | integer-arithmetic QAT (Jacob et al.) | diagnosis of residue-division in LiteMLA under fake-quant | same weights, fake vs integer vs floors | norms explode under integer too — **not observed** |
| label-free teacher-excess sensitivity (U6) | Fisher/Hessian mixed precision (HAWQ-type) | teacher KL, no labels | random-k, label rank, structural rule | ≤ structural rule — **observed: ties at 4× the bytes**, so currently no added value |

### B7. Next actions (none launched)

1. Retry the six timed-out runs once (diagnosed technical failure) with a 1,400 s limit, on idle GPUs only, through a new manifest. Cost ≈ 1.4 GPU-h.
2. Run terminal evaluation (S11-test, STURM, WF; pooled and event-macro; event bootstrap) for all 24 Stage 2 checkpoints. This answers Q1, Q2 and H4.
3. Run the A3 lr 5e-4 FP check (seed 42), then Stage 3 per protocol.

Control: `…/esi/bin/python experiments/revision/capacity_v4_20261008/stage2_fp/revision_queue.py status --manifest experiments/revision/capacity_v4_20261008/stage2_fp/manifest.json` (worker finished; nothing queued). Data: `stage1.csv`, `stage2.csv`, `stage1_note.txt`, `stage2_note.txt`, `results/`.
