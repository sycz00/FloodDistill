> **SUPERSEDED on 2026-10-08 by `prompts/prompt6_corrected.md`** (merged with `prompts/prompt6.md`, plus new verified findings and an exploration track). Kept for history; do not execute this file.

# Codex prompt v3 — map the capacity floor; repair the low-bit measurement

Work in the existing FloodDistill checkout on `research`. Read `AGENT.md`, `prompts/codex_compression_prompt_v2.txt` (it still defines the cohort, teacher SHA, recipe, benchmarks, event units and seeds) and `experiments/revision/supervisor_report_20261008.md` (rationale: §3 and §4). **This prompt supersedes v2 Waves 3–5** (matched-spectrum controls, four-seed confirmation, beta sensitivity, M0, QFD/SQuaT baseline). Nothing else in v2 is relaxed: frozen teacher in eval/no-grad, original restored N=2,500 cohort, identical checkpoint rule (89 S11-val scenes, strict improvement, every 2 epochs, patience 20), no use of S11-test/STURM/WF numbers to choose anything.

Order of work is **cheap diagnostics → cheap repairs → expensive runs**. Continue automatically between phases unless a stated gate fails. Each phase ends with one CSV and a note of at most one page (`phase*_note.txt`). No other narrative documents. Do not paste long tables into chat.

## 0. Verified facts you start from

A supervisor audit re-derived these from raw artifacts. Run `bash experiments/revision/supervisor_checks_20261008/run_all.sh` first (CPU, ~40 s, read-only). **If any fact below does not reproduce, stop and report which one; do not proceed on it.**

1. QAT gradients are numerically broken at low bit. Pre-clip global gradient norm: FP warmup 3–5 (8–25 % of steps clipped at 5); QAT W8A8 10³–10⁵, W4A8 10⁶–10⁸, W4A4 10⁹–10¹⁰, with 100 % of steps clipped at 1.0. On one fixed 16-image batch (A3 checkpoints, focal loss): FP BN-train 1.1, FP frozen-BN 3.6, QAT-mode with fake-quant off 2.7, W8A8 4.0, **W4A8 1.3e8, W4A4 1.1e11**.
2. Cause is localised: quantising only the LiteMLA `qkv` conv of stage 2 gives 2.5e6; the median over 34 single-layer cases is 3.0. In the W4A4 forward 7.7–8.9 % of attention denominators are ≤ 1e-6 (A3 attention dim = 4, A1 = 8, A0 = 16). Raising the `LiteMLAQAT` denominator clamp 1e-6 → 1e-2 brings W4A8 to 4.3 (loss 0.1686 → 0.1683) and W4A4 to 525 (→ 36 at 0.1; W4A4 loss 0.2588 → 0.2506, so the forward changes slightly).
3. QAT barely moves weights: lr 1e-5 × 1,560 updates. Integer-code change (FP start → QAT end, fixed scales): **W4 0.4–2.1 %**, W8 28–29 %; max |Δw| ≤ 0.22 quantisation step at W4. The head's last convs move 40–120× less at W4A4 than at W8A8.
4. 4-bit activation quantiser is weak: global absmax scale; signed grid for every conv input; in A3, 15/34 conv inputs are entirely non-negative and at A4 a median 55.7 % (max 88.2 %) of the activation values entering a conv round to zero. 4-bit weight MSE could fall only 22.7 % with MSE-optimal clipping.
5. The v2 feature target is mostly unexplainable by the student: whitened PCA-32 teacher tap; held-out residual energy tr(C_f)/32 = 0.871 (A3) / 0.828 (A1); per-PC residual ≈ 0.2 for PC1, 0.4–0.6 for PCs 2–4, **0.86–1.05 for PCs 5–32**. Interaction share |η| ≤ 0.059, χ₊ ≤ 0.033; R's spectrum is symmetric (±0.3). M = I in 3/4 cells; hard cell ‖M−I‖_F = 0.092.
6. Noise floor (4 seeds, pretrained, N=2,500): seed SD of mIoU median 0.22 / 0.20 / 0.51 pp (S11 / STURM / WF); event-level model-vs-teacher 95 % CI half-width ±0.9 / ±1.3 / ±1.4–1.9 pp. On STURM and WF the teacher, MNDWI>0, EfficientViT and A3 lie within 0.9 / 1.6 pp water-IoU. The one-seed from-scratch ladder is therefore **not** interpretable (A1 is 4.3 pp below A3 on WF; the supplement reports ±3.7 pp seed SD for the non-pretrained backbone on STURM).

## 1. Reclassify, do not rescue

Write `experiments/revision/capacity_v3_20261008/compression_v2_status_note.txt` (≤ 15 lines): "one-seed screening pilot, negative; B3 ≡ B2 by construction in 3/4 cells and ≈ B2 in the fourth; not evidence about the interaction idea; skipped arms were not run by design". Do not delete, edit or re-run any v2 artifact, frozen capsule or skipped node. Do **not** run any interaction-metric, matched-spectrum, beta/gamma or M0 job.

## 2. Phase A — repair and calibrate (CPU / < 2 GPU-h)

**2.1 Gradient-path repair (opt-in variant).** Do not modify `quantization/qat.py` or any code path used by completed runs. In `models/compression_quant.py` add an opt-in LiteMLA-QAT variant (selected by an explicit flag recorded in the run spec) that bounds the normaliser: floor τ ∈ {1e-3, 1e-2, 1e-1} and mask gradients at positions with denominator < τ. Pick the smallest τ that satisfies **all** of:
   (i) on ≥ 8 fixed training batches (fixed seeds), for A3·{W8A8,W4A8,W4A4} and A1·W4A8 checkpoints, median pre-clip norm ≤ 10× the FP frozen-BN norm on the same batches and max ≤ 100×;
   (ii) PTQ S11-validation mIoU changes ≤ 0.1 pp (W8A8, W4A8) and ≤ 0.3 pp (W4A4) versus the unfloored PTQ;
   (iii) FP models are bit-identical (the variant touches the QAT graph only).
   If no τ passes, fall back to keeping LiteMLA `qkv`/`aggreg` convs in float (record it in the precision map) and label that variant. Output: table of norms/mIoU per τ and cell; fraction of exactly-zero denominators per attention block for FP and quantised forwards.

**2.2 Noise floor and pre-registration.** Adopt `supervisor_checks_20261008/receipts/noise_floor.txt` as the noise reference. Before reading any Phase B result, freeze `ladder_prereg.json`: rung list with *measured* parameter counts; seeds; LR protocol; metrics; robustness conditions; disagreement-set definition; one-sided non-inferiority margin δ_NI for external water-IoU (provisional 2.0 pp, or 2× the measured median seed SD of water-IoU if larger; also report the result at 1.0 and 3.0 pp as sensitivity); knee definition (§3.6); analysis-code hash.

## 3. Phase B — capacity floor (primary; cap 16 GPU-h incl. evaluation)

Question: **how small can a foundation-distilled flood segmenter be, and what does it keep that MNDWI does not?** Run on both RTX 4090s via `experiments/revision_queue.py` with a *new* manifest (shared lists—pools, calibration IDs—stored once and referenced by SHA; manifest < 10 MB), one heavy job per device UUID, never touching foreign jobs.

**3.1 Rungs** (all from scratch, ReLU6, identical teacher/cohort/recipe as the v2 `warmup` job: AdamW wd 0.05, cosine, batch 16, 50 epochs/7,800 updates, clip 5, no AMP, 0.5·focal(γ=2)+0.5·T²·KL, T=4, live teacher on the identical augmented input). Add graph IDs; measure and record parameter counts, do not assume them:

| ID | Design | Target params |
|---|---|---|
| P1 | per-pixel linear, 1×1 conv 6→2 | ≈ 14 |
| P2 | per-pixel MLP 6→16→16→2, strictly per-pixel at inference | ≈ 400 |
| C1 | 3×3(6→8)–3×3(8→8)–1×1(8→2), BN+ReLU6 | ≈ 1 k |
| C2 | small pure-conv context net (stride-2 stem, 3 stages, skip decoder, RF ≥ 64 px), no attention | 8–14 k |
| A3N | A3 with every LiteMLA context module replaced by identity (widths/depth unchanged) | ≈ 30 k − attention |
| A3, A2, A1, A0 | existing graphs (A2 not yet trained) | 31 k, 62 k, 191 k, 705 k |

Reference rows, no new training: pretrained EfficientViT-B0 / MobileNetV3-S / LightFormer (existing 4-seed results), teacher, MNDWI>0. Label initialisation (scratch vs pretrained) in every table.

**3.2 Seeds and LR.** Seeds 42, 64, 91, 107 for every rung. Reuse existing seed-42 A0/A1/A3 FP runs only if graph, recipe and config hash match. For P1, P2, C1, C2 choose lr ∈ {5e-5, 5e-4, 5e-3} at seed 42 by S11-validation mIoU of the selected checkpoint, freeze it, then run seeds 64/91/107. Others use the recipe LR, plus one A3 run at lr 5e-4 (seed 42) as a sensitivity check; if it beats the recipe by ≥ 1.0 pp S11-val mIoU, apply the same 2-point search to A0/A1/A2/A3N (cap +8 runs) and record the change. Forecast GPU-hours from the measured ~875 s/run before launching; if over cap, drop the LR extras first — never seeds.

**3.3 Evaluation** (reuse `evaluation/revision_eval.py` / collectors; validation-selected checkpoint only): clean S11-test (legacy 512), STURM (finite six-band support mask), WF (native zero-pad 512): confusion counts per chip/window → pooled and event-macro mIoU, water-IoU, precision, recall; paired-seed deltas; event-bootstrap CIs conditional on checkpoints, kept separate from seed SD. Never pool predictions across seeds.

**3.4 Robustness (reduced, identical for all rungs, seed 42 for every rung; all seeds only for the rung selected as knee — or A3 if the knee is undefined — and for A0):** clean, scalar gain 0.8 and 1.3, haze 0.20, and the opposing visible/infrared per-band tilt at d = 0.30 (the revision's `t1tilt-30`; confirm exact names in the evaluator). MNDWI and the teacher on the same inputs. Measure per-condition time first; keep this part ≤ 3 GPU-h.

**3.5 Disagreement-set analysis (new, cheap).** Cache the teacher's clean binary masks once per benchmark (bit-packed, keyed by input hash). Define cells T×R with T = teacher water, R = MNDWI>0 — defined **without** the reference. For every model add stratified per-chip confusion counts (vs reference) for the four cells, and per-chip agreement counts model↔teacher and model↔MNDWI. Report for {T≠R}: pixel share, reference-water rate, and each model's accuracy; plus pairwise agreement IoU/κ among {reference, teacher, MNDWI, models}. This answers "what does the teacher add beyond the rule, and do tiny students keep it?" and tests whether the external ceiling is label-protocol-limited.

**3.6 Pre-registered readouts.** (a) For each rung: mean ± sample SD over 4 seeds. (b) External **knee** = smallest rung whose per-event water-IoU difference to the teacher (averaged over the 4 seeds within each event; event-bootstrap 95 % CI, seed variation reported separately) has lower CI bound > −δ_NI on STURM *and* WF (one-sided non-inferiority; two-sided equivalence at ±1.5 pp is not attainable with 11–29 events — the pretrained B0 itself is significantly below the teacher on WF). **"Undefined" is a valid outcome**; report it rather than loosening δ_NI. In-domain: report the S11 mIoU gap to the teacher per rung, no margin. (c) Robustness knee likewise on the reduced conditions. (d) Rung-vs-MNDWI statements only when the CI excludes 0. Report every rung and seed; undefined or failed runs stay in the table. No claim from non-significance.

Gate: functional checks on seed 42 of each new graph (finite loss, 512 output geometry, checkpoint reload exact, teacher frozen, per-pixel property of P1/P2 verified by perturbing a neighbouring pixel) before launching its remaining seeds. A failed gate blocks that rung only.

## 4. Phase C — strong low-bit baseline (cap 8 GPU-h; requires task 2.1)

Goal: find out what low-bit gap remains **after** a competent baseline. Cells: A3 and (if A3N's mean FP S11-val mIoU is within 1.0 pp of A3's) A3N × {W8A8, W4A8, W4A4}; seeds 42, 64, 91; scratch FP checkpoints from Phase B.
Strong baseline = the task-2.1 variant + unsigned grids for inputs proven non-negative on calibration images (record the list) + MSE-optimal (or percentile) clipping for activations and weights, calibration images only as in v2 + weights able to move: choose from a predeclared grid of ≤ 6 configs (lr ∈ {1e-4, 5e-4} × epochs {10, 30}, plus learned step sizes (LSQ-style) on at the best of those) **by S11-val on seed 42 at W4A8 only**; freeze across cells and seeds. Log per run: integer-code change fraction, pre-clip gradient-norm ratio vs FP (must be ≤ 10×), head movement, clipped-step fraction. Make the simulated graph match what would be exported (state whether BN is folded before weight quantisation).
Report PTQ-strong and QAT-strong gap to FP (S11-val and terminal protocols as in v2, simulation labelled `simulation_only`).
**Stop rules.** If strong QAT is within 0.5 pp of FP at W4A8 (S11-val), declare these students quantised with standard tools, report a sub-50 k W8A8/W4A8 model and stop Phase C. If ≥ 1.0 pp remains at W8A8 or W4A8, run a **generic** feature term on a task-relevant subspace — un-whitened top-k PCs where held-out student residual < 0.8 (expected k ≈ 4), v2's β rule — versus the strong B1, 3 seeds, with the MDE computed from Phase C seeds first. A residual gap only at W4A4 is reported as a bit-width floor, not a metric problem. No interaction metrics here.

## 5. Phase D — temporal feasibility audit (CPU only, ≤ 2 h, optional)

The pool pairs 11,895 event and 11,825 baseline tiles (baselines 30–395 days earlier, 20 % cloud gate). Report: number of locations with both tiles after the existing pre-filter and in the restored N=2,500 cohort; co-registration check (normalised cross-correlation of B8A and B11 within ±5 px; fraction with |shift| ≤ 1 px); teacher-derived "new water" (event ∧ ¬baseline) pixel statistics and the symmetric "receding water" statistic as a noise estimate; whether WorldFloods flood-only plus JRC permanent water supports a flood-increment reference. One page; no training; end with go/no-go for a later prompt.

## 6. Rules

* New work only under `experiments/revision/capacity_v3_20261008/`; large artifacts under `/data1/schmalstieg/flooddistill/revision_20261005/capacity_v3_20261008/`. Never overwrite existing results; completed runs' hashes must stay valid.
* No S11-test/STURM/WF number may influence LR, margins, rungs, seeds, or what is reported. Validation-only selection. Missing or failed cells stay visible.
* No new architectures, datasets, pretrained inits, budgets or seeds beyond this prompt; no retries to rescue a negative outcome (one recorded retry for a transient infrastructure failure only); no batch/precision/recipe changes to get a run through — block and report.
* No novelty claim in any output. Before any sentence of the form "first / novel", run a short related-work check (onboard/tiny flood segmentation incl. Φsat-1/WorldFloods; learned spectral indices; LSQ/PACT-style QAT; Fisher/Hessian-weighted reconstruction) and record the findings.
* Keep queue controls (status/pause/resume/stop) and the detached worker as in v2; verify liveness and durable state before ending your turn; do not claim anything finished that is only queued.

## 7. Final message (≤ 40 lines)

One table: phase → status (done / running / blocked, with reason) → 3–5 key numbers → which branch of the supervisor report's §4 decision tree applies and why. Then the exact status/pause/resume/stop commands and remaining GPU-hours forecast. No narrative.
