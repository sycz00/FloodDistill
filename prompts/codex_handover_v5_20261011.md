# Handover to Codex: capacity line v4 → v5 (Claude, 9–11 Oct 2026)

Codex, you ran out of credits on 8 Oct, after Stage 1 and the Stage 2 launch of
`prompts/prompt6_corrected.md`. The user then asked Claude to continue directly. This letter tells you what was
inherited, what was done, what changed relative to your plan, and what you should do next.

Read it with:
* `experiments/revision/capacity_v4_20261008/report_v5.md` (results, for experts and non-experts)
* `protocol_v5_addendum.yaml` (pre-registration)

Every number below was re-derived from receipts by Claude. Treat it as verified-by-Claude and check it yourself
before building on it.

## 1. What you left (inherited and accepted as-is)

* **Facts F1–F11.** The fast set was reproduced after a host-CUDA retry; the failed sandbox receipts are preserved.
* **Stage 1, complete.**
  * The integer-domain simulator with exact native STE was adopted (`code/v3/simulator_v4.py`, sha `7ede34a2…`).
  * All Stage 1 work finished:
    * the parity rescore (≤ 0.138 pp);
    * the TRAIN audit and attribution;
    * the sensitivity maps;
    * the U6 screen;
    * the three optimizer-reach runs.
  * `protocol_v4.yaml` was frozen at 2026-10-08 12:35Z.
  * Claude checked H1 and it passes. The integer simulator gives median pre-clip norms of 0.9–2.7× the FP norm.
    Under fake-quant, the A3 W4A4 median was 2.1e10, about 5e9× the FP norm.
* **Stage 2, launched.** 21 new runs plus 3 reused. Six runs were killed by the 1,020 s limit under foreign GPU load
  (exit 124). No external evaluation was run (`terminal_evaluation=deferred`).

## 2. What Claude did

| when | what | where |
|---|---|---|
| 9 Oct | Interim report for Stages 1–2 | `report_v4.md`, `stage2.csv`, `stage2_note.txt` |
| 9 Oct | **Stage 2 retry:** one retry of the six timeouts, 1,400 s limit, fresh run dirs, 6/6 succeeded | `stage2_retry/` |
| 9 Oct | User directive Q1–Q5, pre-registered before any v5 result | `protocol_v5_addendum.yaml` |
| 9 Oct | New opt-in code, all smoke-tested; the evaluator reproduces v2 per-image counts exactly (3,550 images, 0 differences) | `code/v5/{rungs,train_v5,lowbit_v5,terminal_v5,select_v5,analyze_v5,efficiency_v5}.py` |
| 9 Oct | **E2** low-bit recipe test (12 runs) and **E1** external readout of all checkpoints | `v5/manifest.json` |
| 9 Oct | **E3** capacity floor: R1/R2/R3/R4/A3N/A3, LR grid at seed 42, then 4 seeds | `v5/` |
| 9 Oct | **E4** CE vs Logit-KD vs OFA vs human-label SUP on A0 and A3 | `v5/` |
| 9 Oct | **U1** distilled gain-invariant index (CPU, one shot) | `code/v5u/u1_index.py`, `v5/u1/` |
| 9–11 Oct | Two reuse-branch bugs, each given its one retry in a follow-up manifest | `v5b/`, `v5c/` |
| 11 Oct | Floor selection = **R2**; E4 on R2 (CE/SUP; OFA undefined, recorded as skipped); all 15 v5c jobs succeeded; all 110 checkpoints scored externally | `v5c/` |

## 3. What changed relative to `prompt6_corrected.md`

1. **Order.** At the user's request, Stage 4 core and the method comparison ran *before* Stage 3.
   Stages 3, 5, 6 and 7 have **not** run. Of Stage 8, only U1 (Claude, falsified) and U6 (yours, neutral) ran.
2. **Stage 3 was replaced by E2** for the Q1 question: a W4A8 / 4-bit-body QAT continuation at lr 5e-4, with a
   **matched FP control (F48)**. Still open from Stage 3:
   * MSE/percentile clipping;
   * unsigned grids;
   * LSQ;
   * the ≤ 6-config grid.
3. **The Stage 4 LR rule was extended.** Every rung, including A3 and A3N, got the 3-point LR grid
   {5e-5, 5e-4, 5e-3} at seed 42, selected on S11-val. The Stage 2 graphs A0, A1, A2, X1 and X2 did **not** get
   the grid.
4. **New arms in E4.** CE (focal on teacher hard masks only) and SUP (S11 human labels, 252 chips, matched 7,800
   updates). The teacher is built before the student in every arm, for RNG parity.
5. **R4 was declared at 12,230 params** before any training (the first draft had 20.6k, outside the declared 8–14k).
6. **The supervised arm uses persistent loader workers.** This is a speed fix; the S11-val difference was
   0.01 pp in the smoke test.

## 4. Incidents and defects (all disclosed in `report_v5.md`)

* **The frozen capsule was polluted once.** A terminal smoke wrote into `frozen/297031f5f4ae1df5/v5_smoke/`,
  because the `perscene_eval` import changes the working directory. The file was moved out and the directory
  removed. The v5 scripts now resolve absolute paths. Claude re-verified it on 11 Oct (114/114 files match, no unlisted files). Re-check it as part of T0.
* **`fl_A3_lr5e-5_s42` crashed** (KeyError: Stage 2 receipts have no `graph_record.parameters`).
  Fixed in `train_v5b.py`, and its dependants were rerun in `v5b`.
* **`fl_R1_sel_s42` crashed** (the reuse check demanded exactly 7,800 updates).
  Fixed in `train_v5c.py` (≤ 7,800 accepted, the early stop recorded) and rerun in `v5c`.
* **Inherited early stop.** With patience 20 checks, 6 tiny-rung runs at high LR stopped at 7,176–7,488 updates:
  R1 ×3, R2 ×1, R3 ×2, with selected epochs 6–8. The same rule applies to every run. Decide whether
  protocol_v4's `incomplete_budget_is_failure` was meant to cover this (Claude reads it as covering timeouts only).

## 5. Results you inherit (4 seeds; S11-val = selection split; external = readout only)

* **Q1 (low bit).** Q48 − F48 = **−0.25 [−0.81, +0.32]** S11-val; the external results are also null.
  The Stage 1.6 "QAT beats FP" result was extra optimisation. W4A8 is lossless against a matched FP.
  The 4-bit body with an 8-bit stem costs −1.14 [−2.34, +0.07]. No student beats the teacher (S11-test 89.46).
* **The recipe LR is far too low.** A3 at lr 5e-3 minus A3 at 5e-5 = **+4.14 [+1.32, +6.96]** S11-val
  (+2.53 S11-test). Every rung is worst at 5e-5. **All Stage 2 anchors (A0, A1, A2, X1, X2) are under-trained**,
  so every capacity comparison against them is confounded.
* **Capacity at the selected LR** (LKD):

  | rung | params | MACs/px | S11-val | S11-test mIoU |
  |---|---|---|---|---|
  | R4 | 12k | 417 | 87.94 | 89.93 |
  | A3 | 31k | 164 | 86.57 | 87.32 |
  | R2 | 418 | 384 | 85.64 | 90.21 |
  | A3N | 20k | 144 | 85.56 | 88.26 |
  | R3 | 1k | 1,024 | 84.53 | 88.99 |
  | R1 | 14 | 12 | 84.30 ± 5.9 (unstable) | — |

  * R4 − A3: +1.37 S11-val, +2.60 S11-test, +2.84 WF water-IoU (all CIs exclude 0).
  * A3N − A3: S11-test +0.94, WF water-IoU +1.71 (resolved). **Attention is not needed.**
* **Floor (pre-registered, S11-val):** **R2**. R2 loses on STURM water-IoU against A3 (−1.89 [−3.23, −0.55]).
  The external knee is **undefined at δ = 1/2/3 pp**: no model is event-level non-inferior to the teacher on WF.
  On STURM, A3N is non-inferior at 2 pp and R4 at 3 pp.
* **OOD.** EMSR342 holds **82 % of WF water pixels**; never report WF pooled without event-level CIs. Students trail
  the teacher by 3–10 pp event-level on WF (best: R4, −2.95 [−6.99, −0.09]). The losses come from small events
  where the teacher's context rejects water-like false positives (EMSR333: teacher 52.5, MNDWI 18.9, A3 25.5).
* **Methods (A0 and A3 at recipe LR).**
  * Distillation helps **out-of-domain, not in-domain**: A0 LKD − CE gives STURM water-IoU **+2.72 [+0.94, +4.51]**,
    and A0 OFA − CE gives WF water-IoU **+3.43 [+0.87, +5.99]**. S11 is about 0.
  * OFA − LKD is about 0 everywhere.
  * LKD − SUP on A0: STURM water-IoU **+8.65 [+0.46, +16.83]**.
  * **H7 (gap grows when capacity shrinks): the primary S11-val readout is not supported** (A3 −0.31, R2 −0.39).
    The secondary STURM readout supports it for the floor R2: LKD − CE is **+5.42 [+3.49, +7.35]**, against +2.72 for
    A0, giving an interaction of **+2.69 [+1.91, +3.48]**. WF goes the other way (−2.46, n.s.); A3 is about 0.
  * R2 SUP (human labels) is the best in-domain (S11-val 90.57, S11-test 90.60) and on WF (−2.22 water-IoU vs LKD,
    i.e. better), but worst on STURM (LKD + 4.70). The ranking of training signals flips by benchmark, which points
    to label-protocol differences.
  * OFA on R2 is undefined and recorded as skipped.
* **U1.** The 7-number index reaches A3 level on S11-test (85.9) and STURM (75.7) but collapses on WF (64.8 vs
  MNDWI 75.9). The cause is recall failure on turbid water, from near-zero SWIR weight. Falsified; do not retry.

## 6. Your next tasks (in order; stop at each gate)

**T0. Verify before anything else.**
* Re-run `code/v5/analyze_v5.py experiments/revision/capacity_v4_20261008` and match these numbers to ±0.01 pp:
  * H6a −0.25;
  * A3 LR effect +4.14;
  * R4 − A3 +1.37;
  * the A0 LKD − CE STURM water-IoU +2.72;
  * the R2 LKD − CE STURM water-IoU +5.42.
* Check the capsule identity and the input hashes of `v5/`, `v5b/` and `v5c/`.
* If anything fails to reproduce, stop and report.

Measured cost of everything so far: **21.3 GPU-h** (Stages 1–2 6.0, retry 1.35, v5/v5b/v5c 13.96).

**T1. Re-anchor (~4 GPU-h).** Give A0 and A1 the same 3-point LR grid (seed 42, S11-val selection) and 4 seeds
at the selected LR. Acceptance: report A0, A1, A3, A3N and R4 at their selected LRs. The claims "R4 beats A0"
and "smaller is better" may only be made from this table.

**T2. Repeat E4 on well-trained students (~4 GPU-h).** Run LKD vs CE (and OFA for A3 and A3N) at the selected LR
for A3, R4 and A3N, 4 seeds each. Primary: STURM/WF event water-IoU and paired-seed CIs.
* Claim "distillation improves OOD" only if the CI excludes 0 on at least one external benchmark for at least
  two of the three models.
* H7 stays a pre-registered secondary; never tune to make it appear.

**T3. One novelty test (≤ 3 GPU-h): ledger row 5, the rule-residual student.**
* Model: z = a·MNDWI + b + f_θ(x), with KD weight 1 + γ·1[T≠R].
* Report V(m) = Acc_{T≠R}(m) − Acc_{T≠R}(MNDWI).
* Run it on R4 (and A3N) at the selected LR, 4 seeds, against plain LKD with the same KL mass.
* Falsifier: no {T≠R} gain and no WF event-level gain beyond the MDE.
* Read `v5/novelty_ledger.md` first. Row 2 (log-domain stem vs Q4b) is the alternative if T3 is blocked.

**T4. Stage 3 remainder (≤ 4 GPU-h)**, on the re-anchored FP, focused on the 4-bit body and W4A4 (W4A8 is
already lossless). Then **Stage 6** deployment for R4 / A3N / A3 W4A8.

## 7. Rules (unchanged, and newly important)

* External S11-test/STURM/WF never select an LR, rung, seed, method or gate. `select_v5.py` reads S11-val only;
  keep it that way.
* Never edit a frozen manifest, a frozen code file (`chmod 444`), or the capsule. New behaviour goes in a new
  versioned file and a new manifest. One retry per diagnosed failure.
* WF claims need event-level bootstrap CIs (EMSR342 dominance). Pixels are not replicates.
* Watch out for two pitfalls:
  * `pgrep -f` / `pkill -f` match their own shell when the pattern appears in the command line.
  * The queue's CPU jobs starve while GPU jobs run, so mark trivial jobs `gpu: true`.
* Queue control from the repo root:
  ```
  /home/schmalstieg/miniconda3/envs/esi/bin/python experiments/revision/capacity_v4_20261008/<v5|v5b|v5c>/revision_queue.py status --manifest experiments/revision/capacity_v4_20261008/<v5|v5b|v5c>/manifest.json
  ```
