# FloodDistill capacity v5: final report (low-bit recipe, OOD readout, capacity floor, distillation methods)

*11 Oct 2026. Programme run by Claude on 9–11 Oct; Codex had run out of credits on 8 Oct after Stage 1 and the
Stage 2 launch. Handover letter to Codex: `prompts/codex_handover_v5_20261011.md`. The 9 Oct interim version is
kept as `report_v5_interim_20261009.md`.*

* **Pre-registration:** `protocol_v5_addendum.yaml`, frozen before any v5 result.
* **Evaluator:** reproduces v2 per-image counts exactly (3,550 images, 0 differences).
* **Numbers:** 4-seed means (seeds 42/64/91/107). S11-val is the selection split; S11-test, STURM and WF are
  readouts only.
* **CIs:** paired-seed Student-t 95 % CIs, and event bootstrap (10,000 resamples) for model − teacher.
* **Quantization:** `simulation_only`.
* **Machine-readable:** `v5/analysis/cells.csv` and `summary.json` (`code/v5/analyze_v5.py`).

---

## Part A: Plain-language summary

**What we wanted to know.** A huge AI model (the "teacher", 300 million parameters) maps flood water in satellite
images very well. We want a tiny model that could run on a satellite. Four questions: can a 4-bit model beat the
teacher; do tiny models still work on flood datasets they never saw; how small can the model get; and does
learning *from the teacher* matter more for tiny models than plain training on labels?

**1. A 4-bit model does not beat the teacher.** Earlier it looked as if training in 4-bit made a model better than
its full-precision twin. That was an illusion: the 4-bit model had simply been trained longer at a higher learning
rate. Give the full-precision model the same extra training and both are equally good. The useful news: storing
the weights in 4 bits costs **no measurable accuracy**, and our standard recipe was training the small models with
a learning rate about 100 times too small. Fixing that alone adds about 4 points.

**2. On unseen flood data, the tiny models are close to the teacher, but not everywhere.** On STURM, the best tiny
models are within about 1 point of the teacher on average. On WorldFloods, 82 % of the water pixels come from one
large flood, so the overall number mostly reflects that event. Event by event, all tiny models are behind the
teacher, by about 3 points for the best one and up to 10 for others. The losses come from small floods where the
teacher uses the surrounding scene to reject things that only *look* like water.

**3. How small can it get? It depends on which test you trust.** By our pre-registered rule (Sen1Floods11
validation), a **418-number per-pixel model** is "small enough". It even matches the teacher on the Sen1Floods11
test images. But it is clearly worse on STURM. The best model overall is a **12,000-number plain convolutional
network** (no attention): best on Sen1Floods11, closest to the teacher on both external datasets, and about
60 times smaller than our 700k model. A side test showed how tricky this is: a formula with 7 numbers matches
the 31k model on two datasets and fails badly on the third.

**4. Does distillation matter more for tiny models? Only partly.** Learning from the teacher's soft predictions,
instead of only its hard masks, helps on *unseen* datasets and not on the familiar one:
* about +2.7 points on STURM for the 700k model;
* about +5.4 points on STURM for the 418-number model, so the gap is twice as large there;
* no extra gain on WorldFloods;
* no difference at all for the 31k model.

So the answer depends on the dataset. The OFA variant is never better than plain distillation. Training only on
the 252 human-labelled images helps the tiny per-pixel model on Sen1Floods11 and WorldFloods, but hurts it on
STURM. Each dataset rewards a slightly different labelling style.

---

## Part B: Expert report

### B0. What ran

| block | content | status |
|---|---|---|
| Stage 2 retry | 6 runs that timed out under foreign GPU load; 1,400 s limit, fresh run dirs | 6/6 succeeded |
| E2 | A3 low-bit continuation (lr 5e-4, 1,560 updates): Q48 W4A8, Q4b W4A4 body + stem A8, F48 matched FP control | 12/12 |
| E1 | external readout of every checkpoint (Stage 2, E2, E3, E4) | done (`v5/terminal/`) |
| E3 | rungs R1/R2/R3/R4/A3N/A3: LR grid {5e-5, 5e-4, 5e-3} at seed 42, then 4 seeds at the selected LR | 36 trained + reuses |
| E4 | CE / LKD / OFA / SUP on A0 and A3 (recipe LR 5e-5) and on the floor rung R2 (selected LR 5e-3; OFA undefined) | complete |
| U1 | distilled zero-sum log-linear index (CPU) | falsified |
| Incidents | two reuse-branch crashes, each retried once (`v5b`, `v5c`); capsule pollution reverted | disclosed in B6 |

Not run: Stage 3 (strong conventional baseline beyond E2), Stage 5 (mechanism tests), Stage 6 (deployment),
Stage 7 (temporal audit), and Stage 8 except U1 and U6.

### B1. Q1: low-bit recipe

| A3 | S11-val | S11-test | STURM mIoU / wIoU | WF mIoU / wIoU | STURM wIoU − teacher | WF wIoU − teacher |
|---|---|---|---|---|---|---|
| FP parent (Stage 2, lr 5e-5) | 82.43 ± 1.88 | 84.80 | 75.83 / 64.80 | 75.27 / 61.30 | −1.73 [−4.78, +1.10] | −9.68 [−15.85, −4.97] |
| F48: FP continuation | 84.88 ± 0.96 | 86.90 | 76.00 / 65.16 | 77.32 / 64.58 | −0.78 [−2.87, +1.28] | −4.92 [−10.02, −1.12] |
| Q48: W4A8 QAT | 84.64 ± 1.07 | 86.74 | 76.03 / 65.23 | 77.34 / 64.64 | −0.74 [−2.95, +1.65] | −5.64 [−10.87, −1.70] |
| Q4b: W4A4 body, stem A8 | 83.50 ± 0.53 | 86.04 | 75.84 / 64.99 | 76.87 / 63.98 | −0.59 [−2.76, +1.53] | −8.64 [−14.52, −3.34] |
| teacher / MNDWI>0 (fixed) | — | 89.46 / 78.91 | 75.39 / 64.87 · 74.67 / 64.47 | 77.17 / 64.32 · 75.94 / 63.80 | | |

* **H6a, Q48 − F48 = −0.25 [−0.81, +0.32]** on S11-val. S11-test −0.16 [−0.85, +0.52], STURM water-IoU
  +0.07 [−1.28, +1.42], WF water-IoU +0.06 [−2.29, +2.41].
  **There is no quantization-specific gain; W4A8 is lossless against a matched FP.** The Stage 1.6 effect is
  extra optimisation.
* **H6b.** No arm is better than the teacher anywhere. No arm is non-inferior on WF event-level.
* **H6c, Q4b − Q48 = −1.14 [−2.34, +0.07]:** narrowly misses "within 1.0 pp".

### B2. The recipe learning rate is the largest effect found

At the selected LR, A3 minus A3 at the recipe LR (5e-5) gives:

| benchmark | A3 lr 5e-3 − A3 lr 5e-5 | 95 % CI |
|---|---|---|
| S11-val | **+4.14** | [+1.32, +6.96] |
| S11-test | **+2.53** | [+0.69, +4.37] |
| STURM water-IoU | +0.20 | [−0.67, +1.08] |
| WF water-IoU | +1.81 | [−7.30, +10.93] |

The recipe LR 5e-5 was the worst grid point for **every** rung. The Stage 2 anchors A0, A1, A2, X1 and X2 were
never given the grid, so every comparison against them is confounded by under-training (handover task T1).
The in-domain gain does not translate into a resolved OOD gain for A3.

### B3. Q2/Q4: capacity floor and OOD quality (LKD, each rung at its selected LR)

| rung | params | MACs/px | LR | S11-val | S11-test | STURM mIoU / wIoU | WF mIoU / wIoU | STURM wIoU − T [95 %] | WF wIoU − T [95 %] |
|---|---|---|---|---|---|---|---|---|---|
| R1 linear | 14 | 12 | 5e-4 | 84.30 ± 5.92 | 86.08 | 70.75 / 56.15 | 72.06 / 55.84 | −7.71 [−11.90, −3.75] | −11.17 [−16.63, −6.22] |
| **R2 per-pixel MLP** (floor) | 418 | 384 | 5e-3 | 85.64 ± 0.44 | **90.21** | 74.92 / 63.11 | 77.32 / 64.53 | −3.24 [−7.05, +0.01] | −5.52 [−11.69, −1.31] |
| R3 tiny conv | 1,058 | 1,024 | 5e-3 | 84.53 ± 0.46 | 88.99 | 75.38 / 63.97 | 76.35 / 63.02 | −2.52 [−7.42, +1.57] | −8.58 [−17.34, −1.97] |
| **R4 conv context** | 12,230 | 417 | 5e-3 | **87.94 ± 0.43** | 89.93 | 75.62 / 64.71 | **78.17 / 65.95** | −1.01 [−3.12, +1.16] | **−2.95 [−6.99, −0.09]** |
| A3N (no attention) | 20,250 | 144 | 5e-3 | 85.56 ± 0.86 | 88.26 | **76.36 / 65.67** | 77.47 / 64.82 | **+0.17 [−1.83, +2.44]** | −5.61 [−11.21, −1.71] |
| A3 | 30,922 | 164 | 5e-3 | 86.57 ± 0.46 | 87.32 | 76.04 / 65.00 | 76.46 / 63.11 | −1.22 [−3.57, +1.10] | −6.10 [−10.44, −2.64] |
| A0 (recipe LR, under-trained) | 704,714 | 2,114 | 5e-5 | 84.65 ± 0.54 | 85.61 | 74.81 / 63.31 | 75.13 / 61.07 | −1.80 [−4.68, +0.88] | −7.81 [−12.73, −4.30] |

R1 is unstable: its seeds give 86.5 / 88.1 / 75.5 / 87.1 on S11-val.

Paired contrasts against A3 (all at their selected LR):

| rung − A3 | S11-val | S11-test | STURM water-IoU | WF water-IoU |
|---|---|---|---|---|
| R4 | +1.37 [+0.41, +2.33] | +2.60 [+1.88, +3.32] | −0.29 [−1.86, +1.28] | +2.84 [+1.53, +4.14] |
| A3N | −1.01 [−2.76, +0.75] | +0.94 [+0.13, +1.74] | +0.66 [−0.52, +1.85] | +1.71 [+0.11, +3.30] |
| R2 | −0.93 [−1.82, −0.03] | +2.89 [+1.87, +3.90] | −1.89 [−3.23, −0.55] | +1.42 [+0.58, +2.26] |
| R3 | −2.03 [−3.10, −0.97] | +1.67 | about 0 | about 0 |

Interpretation:

1. **The pre-registered floor rule selects R2** (418 params; S11-val within 1 pp of A3). R2 is the best model on
   S11-test (90.21, above the teacher's pooled 89.46). It is resolvably worse on STURM (−1.89 water-IoU against
   A3) and 5.5 pp below the teacher event-level on WF. **The floor is benchmark-dependent.** The S11 benchmarks
   are largely solvable per pixel. U1 (7 numbers) also reaches 85.9 S11-test. They therefore cannot certify a
   floor; WF event-level and STURM can.
2. **The external knee is undefined.** No rung has a one-sided 95 % lower bound > −δ for both STURM and WF at
   δ = 1, 2 or 3 pp; WF fails everywhere. R4 comes closest (lower bounds −2.77 / −6.28). On STURM alone, A3N is
   non-inferior at 2 pp and R4 at 3 pp.
3. **The best efficiency/accuracy model is R4.** It is a 12k-param pure-conv context net, with no attention and
   no LiteMLA, so the quantization pathology of Stage 1 does not arise. It is the best model on S11-val and
   WF event-level, and resolvably better than A3 on S11 and WF. **A3N beats A3 on S11-test and WF:** attention
   buys nothing at this scale.
4. **Parameters ≠ compute.** R2 (418 params) needs 384 MACs/px, more than A3N (144) or A3 (164).
5. **OOD failure mode.** EMSR342 holds 82.1 % of WF water pixels. Per event, the students lose on small events
   where the teacher's context removes water-like false positives. For example, on EMSR333 the teacher scores
   52.5 water-IoU at 80 % precision, MNDWI 18.9 at 20 %, and A3-Q48 25.5 at 44 %.

### B4. Q4: Training methods (CE = teacher hard masks only; SUP = 252 S11 human-labelled chips, matched updates)

| model | method | S11-val | S11-test | STURM mIoU / wIoU | WF mIoU / wIoU |
|---|---|---|---|---|---|
| A0 (705k) | SUP | 85.20 ± 0.16 | 85.38 | 69.82 / 54.66 | 73.74 / 58.80 |
| | CE | 84.29 ± 0.55 | 85.32 | 73.30 / 60.59 | 73.47 / 58.36 |
| | LKD | 84.65 ± 0.54 | 85.61 | 74.81 / 63.31 | 75.13 / 61.07 |
| | OFA | 84.41 ± 1.03 | 85.76 | 74.77 / 63.43 | 75.56 / 61.79 |
| A3 (31k) | SUP | 79.33 ± 5.31 | 81.72 | 63.41 / 44.11 | 72.52 / 56.94 |
| | CE | 82.37 ± 2.29 | 84.93 | 73.47 / 60.74 | 73.81 / 59.03 |
| | LKD | 82.43 ± 1.88 | 84.80 | 75.83 / 64.80 | 75.27 / 61.30 |
| | OFA | 82.58 ± 1.53 | 85.01 | 75.59 / 64.54 | 75.32 / 61.41 |
| R2 (418, floor; lr 5e-3) | SUP | **90.57 ± 0.07** | **90.60** | 71.87 / 58.41 | **78.63 / 66.74** |
| | CE | 85.67 ± 0.14 | 90.24 | 71.67 / 57.69 | 77.18 / 64.28 |
| | LKD | 85.64 ± 0.44 | 90.21 | **74.92 / 63.11** | 77.32 / 64.53 |
| | OFA | undefined (no stages; recorded as skipped) | | | |

Paired contrasts:

| contrast | S11-val | S11-test | STURM water-IoU | WF water-IoU |
|---|---|---|---|---|
| A0: LKD − CE | +0.37 [−0.31, +1.04] | +0.29 [−0.95, +1.53] | **+2.72 [+0.94, +4.51]** | +2.71 [−1.16, +6.57] |
| A0: OFA − CE | +0.13 | +0.44 | **+2.84 [+1.10, +4.57]** | **+3.43 [+0.87, +5.99]** |
| A0: OFA − LKD | −0.24 [−1.14, +0.66] | +0.15 | +0.11 [−1.38, +1.61] | +0.72 [−1.12, +2.56] |
| A0: LKD − SUP | −0.54 | +0.23 | **+8.65 [+0.46, +16.83]** | +2.27 [−0.17, +4.70] |
| A3: LKD − CE | +0.06 [−1.04, +1.16] | −0.13 | +4.06 [−0.74, +8.86] | +2.27 [−2.24, +6.78] |
| A3: OFA − LKD | +0.16 | +0.22 | −0.26 [−1.59, +1.07] | +0.12 [−1.67, +1.90] |
| H7: (LKD − CE)(A3) − (LKD − CE)(A0) | −0.31 [−1.95, +1.33] | −0.42 | +1.34 [−2.84, +5.52] | −0.44 [−2.88, +2.01] |
| H7: (OFA − CE)(A3) − (OFA − CE)(A0) | +0.09 | −0.35 | +0.96 [−5.36, +7.28] | −1.04 [−5.18, +3.10] |
| R2: LKD − CE | −0.03 [−0.78, +0.72] | −0.03 [−0.58, +0.52] | **+5.42 [+3.49, +7.35]** | +0.25 [−0.91, +1.40] |
| R2: LKD − SUP | **−4.93 [−5.56, −4.31]** | −0.39 [−0.79, −0.00] | **+4.70 [+1.61, +7.79]** | **−2.22 [−3.31, −1.12]** |
| H7: (LKD − CE)(R2) − (LKD − CE)(A0) | −0.39 [−1.62, +0.83] | −0.32 [−1.47, +0.84] | **+2.69 [+1.91, +3.48]** | −2.46 [−6.64, +1.72] |

Interpretation:

* **Soft-target distillation is an out-of-domain effect.** In-domain it adds nothing (S11 ≈ 0). On STURM and WF
  it adds about 2.5–3.5 water-IoU over teacher hard masks, resolved for A0 and in the same direction for A3,
  where seed variance at the recipe LR is large.
* **OFA = LKD** everywhere, consistent with paper Table II.
* **Human labels alone (252 chips)** match or win in-domain but lose OOD. For A3 this is unstable: SUP seed SD is
  5.3 pp.
* **H7 primary (S11-val): not supported** for either small model. The interaction is −0.31 for A3 and −0.39 for R2.
  * **Secondary readouts are mixed.** For the floor model R2, the soft-target benefit on STURM is twice that of A0:
    LKD − CE +5.42 vs +2.72, with the interaction **+2.69 [+1.91, +3.48]**. On WF the interaction goes the other
    way (−2.46, n.s.). For A3, every interaction is about 0.
  * Conclusion: the soft-target benefit appears **out-of-domain only, benchmark-specific, and larger for the
    per-pixel floor model on STURM**. No uniform "distillation matters more for tiny models" holds.
  * Confounds:
    * R2 runs at its LKD-selected LR (5e-3), while A0 runs at 5e-5.
    * The E4 rows for A0/A3 use the under-trained recipe LR.
  * Repeat at matched, selected LRs before publication (handover task T2).
* **Human labels at the floor.** SUP R2 is the best model in-domain (S11-val 90.57, S11-test 90.60) and also on
  WF (event −3.37 vs teacher). It is the worst on STURM (−4.70 vs LKD).
  * A per-pixel model trained on the 252 S11 human labels fits the S11 label protocol best.
  * The teacher's labelling transfers better to STURM's protocol.
  * The ranking of training signals flips by benchmark, which points again to label-protocol differences between
    datasets.

### B5. Novelty ledger (full version: `v5/novelty_ledger.md`)

| # | idea | status | evidence / next |
|---|---|---|---|
| 1 | Simulation-faithful QAT for ReLU linear attention (fake-quant residue division) | tested | norms 2e10 → 4.5 at W4A4; strongest *measurement* contribution; Trio-ViT treats divisors in PTQ only |
| 2 | Log-domain stem (ratios → differences) | proposed | control Q4b (−1.14 pp) |
| 3 | U1 distilled gain-invariant index | **falsified** | S11-test 85.9 / STURM 75.7 / WF 64.8 (−11 vs MNDWI); turbid-water recall failure |
| 4 | Flip-aware QAT distillation | proposed | W4A8 already lossless, so relevant only for W4A4 |
| 5 | Rule-residual student + disagreement metric V(m) | proposed, **recommended next** | targets exactly the WF small-event failures |
| 6 | Capacity-aware / uncertainty-weighted KD | open | H7 holds only on STURM for R2 (+2.69); test only after T2 re-anchoring |
| 7 | Parameter floor ≠ compute floor | measured | framing |
| 8 | Label-free mixed precision (U6) | neutral | the structural rule ties at ¼ of the bytes |
| new | **No-attention conv nets (R4, A3N) ≥ EfficientViT at ≤ 31k params** | measured | publishable as an empirical finding after T1 re-anchoring; removes the LiteMLA quantization problem |

### B6. Claim–evidence table

| claim | status |
|---|---|
| W4A8 lossless vs matched FP (A3) | **supported** |
| "QAT beats FP" was a recipe confound | **supported** |
| Recipe LR under-trains scratch students by ~4 pp in-domain | **supported for A3**; A0/A1 untested |
| Soft-target KD improves OOD over hard pseudo-labels | **supported for A0** (STURM, WF with OFA); direction only for A3 |
| OFA > LKD | **not supported** |
| KD benefit grows as capacity shrinks (H7) | **primary (S11-val) not supported**; secondary STURM supported for R2 (+2.69 [+1.91, +3.48]), not for A3; WF opposite (n.s.) |
| Students ≈ teacher OOD | STURM: approximately (A3N NI at 2 pp); **WF event-level: no** |
| A rung below 31k is "enough" | in-domain yes (R2, R4); **externally undefined** |
| 7-number index replaces the network | **falsified** (WF) |
| No-attention R4 is the best small model | supported vs A3 at matched LR; vs A0 confounded |

### B7. Operations, costs, incidents

* **GPU-h (measured command time).** Stage 1: 0.94. Stage 2: 5.06, of which 1.67 went to the timeouts. Retry: 1.35. v5: 11.82. v5b: 0.67. v5c: 1.47. **Total 21.3 GPU-h** of the ≈ 30 cap. The U1 fit ran on CPU.
* **Incidents.**
  1. A terminal smoke wrote into the frozen capsule (a cwd change in the `perscene_eval` import). It was reverted
     and absolute paths are now enforced.
  2. `fl_A3_lr5e-5_s42` crashed with KeyError `parameters`. Fixed in `train_v5b.py` and rerun in `v5b`.
  3. `fl_R1_sel_s42` crashed because the reuse check required exactly 7,800 updates. Fixed in `train_v5c.py` and
     rerun in `v5c`.
  4. The inherited early stop (patience 20) ended 6 tiny-rung runs at 7,176–7,488 updates (selected epochs 6–8).
     The rule applies to all runs; disclosed.
  5. R4 was resized to 12,230 params before training.
* **Queues** (all finished): `stage2_retry`, `v5`, `v5b`, `v5c`. Status:
  `…/esi/bin/python experiments/revision/capacity_v4_20261008/<queue>/revision_queue.py status --manifest experiments/revision/capacity_v4_20261008/<queue>/manifest.json`
