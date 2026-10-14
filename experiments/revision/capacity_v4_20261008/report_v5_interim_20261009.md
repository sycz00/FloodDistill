# FloodDistill capacity v5: interim report (low-bit recipe, OOD readout, capacity floor, distillation methods)

*9 Oct 2026, 13:50 CEST. Status: **interim**.*

* **Complete:** the Stage 2 cut-off retry (now 24/24 runs), E2 (the low-bit recipe test), the external readout
  of all 36 Stage 2 and E2 models, and U1 (the distilled index).
* **Running unattended in the v5 queue (99 jobs, ETA ≈ 21:00–22:00):** E3 (capacity floor) and E4
  (CE / Logit-KD / OFA / human-label supervision).
* **Pre-registration:** `protocol_v5_addendum.yaml`, frozen before any v5 result.
* **Evaluator:** reproduces v2's per-image counts exactly (3,550 images, 0 differences).
* **Data:** every number below is a 4-seed mean (seeds 42/64/91/107) unless stated otherwise.
  Quantized results are `simulation_only`.

---

## Part A: Plain-language summary

**1. Did we find a low-bit training recipe that beats the teacher?** No. We found something more useful and more
honest. Yesterday's "the 4-bit model beats its own full-precision version" came from *extra training at a higher
learning rate*, not from 4-bit arithmetic. When we give the full-precision model exactly the same extra training,
it improves just as much. The useful consequences:
* (a) the standard training recipe leaves about 2 points of accuracy unused;
* (b) once both models are trained equally well, storing weights in 4 bits costs **nothing measurable**;
* (c) even a mostly-4-bit model (only the first layer's input kept at 8 bits) loses only about 1 point.

The teacher stays clearly better on the Sen1Floods11 test images (89.5 vs ≈ 87 mIoU). On the two external flood
datasets the small models roughly match it in total pixel counts, but not event by event (point 2).

**2. Do the small models keep their quality on unseen flood datasets?** The answer differs between the two datasets.
* **STURM:** the 31,000-parameter model is on average within about 1 point of the 300-million-parameter teacher.
* **WorldFloods:** the totals look equal. But 82 % of WorldFloods' water pixels come from a single flood event, so
  the total mostly measures that one event. Event by event, every small model is clearly behind the teacher
  (about 5–15 points). The losses come from a few small events where the teacher uses the surrounding image to
  reject things that only *look* like water.

**3. Cut-off experiments.** The six runs stopped by the time limit were re-run and all completed, so the
architecture comparison now has all 24 runs. The thin-and-deep model (X2) is reliably the worst, on every dataset.

**4. How small can the model be?** Still running. A side experiment already shows something important. A formula
with **7 numbers**, fitted to imitate the teacher, matches the 31k model on Sen1Floods11 and STURM, but fails
badly on WorldFloods (−11 points against the classic water index). So Sen1Floods11 and STURM are "easy" for simple
per-pixel rules, and WorldFloods is the dataset that shows whether a model actually learned more than a rule.
The full comparison of training methods (plain labels vs. distillation vs. OFA) at the smallest size is queued
and finishes tonight.

---

## Part B: Expert report

### B1. Q1: Low-bit recipe (E2, A3, 4 seeds, integer-domain simulator, lr 5e-4, 1,560 updates from the Stage 2 FP parent)

| arm | S11-val | S11-test mIoU | STURM mIoU / wIoU | WF mIoU / wIoU | STURM wIoU − teacher (event, 95 %) | WF wIoU − teacher (event, 95 %) |
|---|---|---|---|---|---|---|
| FP parent (Stage 2 LKD) | 82.43 ± 1.88 | 84.80 | 75.83 / 64.80 | 75.27 / 61.30 | −1.73 [−4.78, +1.10] | −9.68 [−15.85, −4.97] |
| **F48**: FP continuation (QDQ bypassed) | 84.88 ± 0.96 | 86.90 | 76.00 / 65.16 | 77.32 / 64.58 | −0.78 [−2.87, +1.28] | −4.92 [−10.02, −1.12] |
| **Q48**: W4A8 QAT | 84.64 ± 1.07 | 86.74 | 76.03 / 65.23 | 77.34 / 64.64 | −0.74 [−2.95, +1.65] | −5.64 [−10.87, −1.70] |
| **Q4b**: W4A4 body, stem input A8 | 83.50 ± 0.53 | 86.04 | 75.84 / 64.99 | 76.87 / 63.98 | −0.59 [−2.76, +1.53] | −8.64 [−14.52, −3.34] |
| teacher (fixed) | — | 89.46 | 75.39 / 64.87 | 77.17 / 64.32 | 0 | 0 |
| MNDWI>0 (fixed) | — | 78.91 | 74.67 / 64.47 | 75.94 / 63.80 | | |

Paired-seed contrasts (Student-t 95 % CI):
* **H6a, Q48 − F48 = −0.25 [−0.81, +0.32] S11-val.** The other benchmarks agree: S11-test −0.16 [−0.85, +0.52],
  STURM wIoU +0.07 [−1.28, +1.42], WF wIoU +0.06 [−2.29, +2.41].
  **H6a is rejected: there is no quantization-specific gain.** F48 − parent is +2.45 [−0.59, +5.50]. That is the
  Stage 1.6 effect, and it is extra optimisation. Seeds 42 and 107 had selected their FP checkpoint early, which is
  consistent with an under-trained FP anchor.
* **H6b, low-bit vs teacher.**
  * In-domain: the teacher leads S11-test by ~2.7 pp.
  * STURM: Q48 is −0.74 pp event mean, one-sided 95 % lower bound −2.59. That is **not non-inferior at
    δ = 2 pp**, but non-inferior at the 3 pp sensitivity margin.
  * WF: event-level, **worse than the teacher** (CI excludes 0).
  * There is no "better than teacher" anywhere.
* **H6c, Q4b − Q48 = −1.14 [−2.34, +0.07]** (criterion "within 1.0 pp": narrowly missed). The 4-bit body is
  viable with a ~1 pp cost and an 8-bit stem exception (disclosed, 36 extra activation-input bits per pixel).

Interpretation. With a faithful simulator and a matched recipe, **W4A8 is free for this model** (≤ 0.8 pp at 95 %).
The remaining in-domain loss is at the 4-bit activation level, at the stem. The headline "low-bit beats FP" was a
confound, and the matched F48 control is what exposed it. Every FP anchor in the programme is probably
under-trained by ~2 pp. E3 tests this directly with an LR grid.

### B2. Q2: OOD quality of the Stage 2 family (4 seeds each; all checkpoints scored, none selected on external data)

| graph (params, MACs/px) | S11-val | S11-test | STURM mIoU / wIoU | WF mIoU / wIoU | STURM wIoU − teacher | WF wIoU − teacher | STURM wIoU − MNDWI |
|---|---|---|---|---|---|---|---|
| A0 (705k, 2,114) | 84.65 ± 0.54 | 85.61 | 74.81 / 63.31 | 75.13 / 61.07 | −1.80 [−4.68, +0.88] | −7.81 [−12.73, −4.30] | −2.22 [−5.63, +0.76] |
| A1 (191k, 609) | 83.21 ± 0.79 | 83.92 | 74.77 / 62.89 | 73.75 / 58.80 | −3.13 [−6.36, −0.17] | −11.47 [−17.99, −6.81] | −3.54 [−7.44, +0.03] |
| X1 (106k, 369) | 84.09 ± 1.18 | 85.69 | 75.75 / 64.66 | 74.64 / 60.30 | −1.57 [−4.35, +1.32] | −9.56 [−16.21, −4.81] | −1.99 [−4.84, +0.68] |
| A2 (62k, 243) | 82.81 ± 0.79 | 85.00 | 75.84 / 64.70 | 74.58 / 60.13 | −2.09 [−4.69, +0.39] | −9.95 [−16.50, −4.97] | −2.50 [−6.21, +0.78] |
| X2 (55k, 246) | 79.56 ± 0.83 | 82.14 | 72.41 / 59.03 | 71.68 / 55.49 | −5.37 [−9.68, −1.69] | −14.52 [−20.45, −9.54] | −5.79 [−10.66, −1.44] |
| A3 (31k, 164) | 82.43 ± 1.88 | 84.80 | 75.83 / 64.80 | 75.27 / 61.30 | −1.73 [−4.78, +1.10] | −9.68 [−15.85, −4.97] | −2.15 [−5.58, +1.01] |
| A3 + F48 continuation (31k) | 84.88 ± 0.96 | **86.90** | **76.00 / 65.16** | **77.32 / 64.58** | −0.78 [−2.87, +1.28] | −4.92 [−10.02, −1.12] | −1.19 [−3.93, +1.29] |

The event columns are bootstrap CIs over 29 (STURM) and 11 (WF) events, with each model seed-averaged within an
event. Findings:

1. **The best efficiency/accuracy point is the smallest graph.** A3 (31k params, 164 MACs/px) is never
   resolvably worse than A0 (705k, 13× the compute) on any external benchmark. After the matched continuation
   (F48/Q48), it is the best model on all three pooled benchmarks. Capacity above 31k buys nothing measurable
   outside the training domain.
2. **STURM: approximately preserved.** The event mean is within 1 pp of the teacher for the continued A3. Pooled
   student STURM mIoU (≈ 76.0) is above the teacher (75.4).
3. **WF: preserved in the pooled count, not per event.** EMSR342 holds **82.1 %** of WF water pixels, so pooled WF
   is essentially one event. Per event, every student is 5–15 pp below the teacher. The gap comes from small events
   where the teacher's context rejects water-like false positives that MNDWI accepts. On EMSR333 the teacher scores
   52.5 water-IoU (precision 80 %), MNDWI 18.9 (precision 20 %), and A3-Q48 25.5 (precision 44 %). On EMSR286 and
   EMSR264 the students already inherit most of that rejection (76.7 vs MNDWI 21.2).
   **This is where distillation is still leaving value on the table, and the most promising target for
   intervention** (ledger rows 5 and 6).
4. **Versus MNDWI** no student is resolvably better on STURM/WF event-level. X2 is resolvably worse on STURM.
   The external ceiling is tight: teacher, MNDWI and students lie within ~1.5 pp pooled.

### B3. Q3: Cut-off experiments and Stage 2 completion

The six runs stopped at the 1,020 s limit (foreign GPU load) were retried once with a 1,400 s limit and fresh run
directories. All six succeeded in 12.7–14.8 min. The partial attempt-1 checkpoints are preserved.

S11-val paired contrasts (4 seeds):
* **A0 − A1 +1.44 [+0.31, +2.58]**
* A1 − X2 +3.65 [+2.09, +5.20]
* X1 − X2 +4.52 [+2.84, +6.21]
* A0 − A3 +2.23 [−0.22, +4.68], unresolved; A3's seed 107 is an outlier at 79.7
* Width × depth interaction +1.99 [−3.11, +7.09], unresolved

**Q2/H4** (A1 < A3 externally, pre-registered: paired-seed CI must exclude 0). The direction holds on every
benchmark, but no CI excludes 0:

| benchmark | A1 − A3 pooled mIoU | 95 % CI |
|---|---|---|
| S11-test | −0.88 | [−2.26, +0.51] |
| STURM | −1.06 | [−2.90, +0.79] |
| WF | −1.52 | [−4.46, +1.42] |

**H4 is not confirmed.** The v2 seed-42 "reversal" is consistent in direction but not resolved by 4 seeds. Externally,
X2 (narrow and deep) is the only clear loser.

### B4. Q4: Capacity floor and training-method comparison (running)

* **E3.** The rungs are R1 linear (14 params, 12 MACs/px), R2 per-pixel MLP (418, 384), R3 tiny conv (1,058, 1,024),
  R4 pure-conv context net (12,230, 417), A3N (A3 without attention, 20,250, 144) and A3 (30,922, 164).
  Each rung gets an LR grid {5e-5, 5e-4, 5e-3} at seed 42 (S11-val selection), then 3 more seeds.
  The floor rule (S11-val only) is: the smallest rung within 1.0 pp of A3. The external knee (event CI vs teacher
  on STURM and WF) is reported separately.
* **E4.** Logit-KD vs OFA-KD vs CE (teacher hard masks) vs SUP (S11 human labels, matched 7,800 updates), 4 seeds
  each, on A0, A3 and the floor rung. H7 is "distillation helps more at small capacity". It is tested as the
  interaction (KD − CE)(small) − (KD − CE)(A0) with paired-seed CIs.
* **What is already known:**
  * U1 (below) shows that a 7-number log-ratio index reaches S11-test 85.9 / STURM 75.7, i.e. A3-level.
  * The first R1 run (14 params, lr 5e-4) is at 86.5 S11-val after 5 epochs.
  * **Expect the S11-val floor rule to select a very small rung, and that rung to collapse on WF.** The
    pre-registered rule will not be changed. If it happens, the honest statement is: *"the capacity floor is
    benchmark-dependent; S11/STURM do not discriminate below 31k; WF event-level does."*
  * A bigger CE-vs-KD gap is a hypothesis, not an expectation. At N = 2,500 the paper's pretrained B0 shows
    CE/LKD/OFA within 0.2–1.3 pp. The capacity-gap literature (TAKD) even predicts *smaller* benefits for very
    small students.

### B5. Q5: Novelty ledger (full version with formulas, controls, falsifiers and prior art: `v5/novelty_ledger.md`)

| # | idea (from an observed failure) | status | result / next step |
|---|---|---|---|
| 1 | Simulation-faithful QAT for ReLU linear attention (fake-quant residue division, metric ρ_res) | tested | strongest candidate. 1e11 → 4 norm; Trio-ViT handles divisors in PTQ only |
| 2 | Log-domain (companded) stem: ratios become differences, uniform A4 grid gives relative error | proposed | cheapest test of the stem bottleneck; control Q4b |
| 3 | U1 distilled, exactly gain-invariant index s = Σ w_b log ρ_b, Σ w_b = 0 | **tested, falsified** | S11-test 85.91 (+7.0 vs MNDWI), STURM 75.69, **WF 64.83 (−11.1 vs MNDWI)**: recall collapse on turbid/large events (EMSR342 recall 46 vs 80 %); fit leaned on G/NIR, near-zero SWIR weight |
| 4 | Flip-aware ("positive-congruent") KD for QAT: weight KL by FP-correctness × margin | proposed (Stage 5) | targets H = 7.2 % harmful flips at W4A4; PCT/ELODI exist only for model updates |
| 5 | Rule-residual student z = a·MNDWI + b + f_θ(x) and disagreement metric V(m) = Acc_{T≠R}(m) − Acc_{T≠R}(R) | proposed | directly targets the WF small-event failures in B2.3 |
| 6 | Uncertainty / capacity-aware KD (w = 4 p_T (1 − p_T), TAKD chain) | proposed; depends on H7 | framed as an empirical study, not a method |
| 7 | Parameter floor ≠ compute floor (A3 164 MACs/px < R2 384 < R3 1,024) | measured | framing point: report both axes |
| 8 | Label-free mixed precision (U6) | tested, neutral | the structural rule ties at ¼ of the bytes, so no claim |

### B6. Claim–evidence status

| claim | evidence | status |
|---|---|---|
| W4A8 QAT is lossless vs a matched FP continuation (A3) | H6a 4 seeds, all benchmarks | **supported** (CI ±0.8 pp S11-val) |
| The "QAT beats FP" effect is a recipe confound | F48 control | **supported** |
| FP anchors are under-trained by ~2 pp | F48 − parent +2.45 (CI incl. 0); E3 LR grid pending | hypothesis, strengthening |
| A 31k student matches a 705k student OOD | 4-seed external, event CIs | supported (no resolvable difference) |
| Students ≈ teacher OOD | STURM event mean −0.7 pp (NI only at 3 pp); WF event −5 to −10 pp | **not supported on WF event level** |
| A distilled 7-parameter index replaces the network | U1 | **falsified on WF** |
| Distillation matters more for tiny students (H7) | E4 | pending |

### B7. Operations

* **Budget.** Stage 1–2 plus retry ≈ 7.3 GPU-h. v5 so far ≈ 1.7 GPU-h. Expected v5 total ≈ 16–18 GPU-h. The
  reservation is 42 h, an upper bound equal to the sum of hard limits. Expected programme total ≈ 25 GPU-h,
  within the ≈ 30 h cap.
* **Deviations and incidents.**
  * One evaluator smoke wrote a file inside the frozen capsule (a cwd change in `perscene_eval`). It was moved out
    and the capsule restored; the scripts now resolve absolute paths.
  * The supervised arm uses persistent loader workers (a speed fix only).
  * R4 widths were reduced to 12,230 params before training, to meet the declared 8–14k.
  * **v5 technical failure, 14:50.**
    * What failed: `fl_A3_lr5e-5_s42`, a reuse-only job. It hit KeyError `parameters`, because Stage 2 receipts
      carry no parameter count in `graph_record`.
    * What it blocked: `sel_lr_A3`, `fl_A3_sel_*`, `sel_floor` and `mf_*` (18 jobs).
    * The fix: `train_v5b.py`, which counts parameters from the graph; that is the only change. It was tested
      on the failing case.
    * The retry: the one permitted retry runs in a follow-up manifest `v5b/manifest.json` (20 jobs, including
      the eval `eval_v5b`). It starts automatically when the v5 worker releases the GPU lock. The v5 manifest
      and its attempt receipts are untouched.
    * Effect: no scientific rule changed. The floor and method results arrive about 2–3 h later than planned,
      ≈ 00:00–01:00.
  * **Early E3 screen** (seed 42, S11-val, LR grid; a screen, so no claim). The recipe LR 5e-5 is the worst LR
    for every rung:
    * R1 (14 params): 65.8 → 86.5 at 5e-4
    * R2 (418 params): 86.0 at 5e-3
    * R4 (12k pure conv): 88.0 at 5e-3
    * A3N: 84.4 at 5e-4
    * A3 at higher LR is still running.

    This supports the under-training hypothesis.
* **Queue control** (from the repo root):
  `…/esi/bin/python experiments/revision/capacity_v4_20261008/v5/revision_queue.py status|pause|resume|stop --manifest experiments/revision/capacity_v4_20261008/v5/manifest.json`
* **Re-analysis:** `…/esi/bin/python experiments/revision/capacity_v4_20261008/code/v5/analyze_v5.py experiments/revision/capacity_v4_20261008`
  writes `v5/analysis/cells.csv` and `summary.json`.
