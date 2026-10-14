# Supervisor report — FloodDistill after the compression‑v2 pilot

**Date:** 2026‑10‑08 · **Prompt for Codex:** [`prompts/prompt6_corrected.md`](../../prompts/prompt6_corrected.md) (supersedes the earlier `codex_next_prompt_v3.md`; see Addendum A and [`prompt6_review_20261008.md`](prompt6_review_20261008.md))
**Read:** prompts 1–5, `codex_compression_prompt_v2.txt`, the JSTARS manuscript + supplement (v3 PDFs in `repos/OLD`), `experiments/revision/**` (prompt3–5 packages, compression‑v2 pilot incl. its 8 Oct full report), and the code under `models/compression*.py`, `distillation/compression_metric.py`, `training/compression_loss.py`, `scripts/compression_jobs.py`, `quantization/qat.py`.

**Evidence tags used below.**
**[V]** I re‑derived it from raw artifacts; receipts are in [`supervisor_checks_20261008/receipts/`](supervisor_checks_20261008/receipts/) and re‑run with `bash supervisor_checks_20261008/run_all.sh` (CPU, read‑only, ~40 s).
**[R]** Taken from the student's reports without my own re‑derivation.
**[H]** My judgment or hypothesis — *not tested*.

---

## 0. Bottom line

1. **Two tracks, two different states.** The JSTARS revision (prompts 1–5) is compute‑complete; what is left needs *you* (author history on benchmark use, the real reviewer letter, dry‑scene references, Jetson artifacts). I see no GPU experiment that would change a reviewer‑facing conclusion there **[H]**. The compression‑v2 pilot is a finished one‑seed screen whose headline is "interaction hypothesis unsupported" **[R]**.
2. **The pilot was run carefully and reported honestly — but its null says almost nothing about the idea**, because the experiment could not have produced an effect: the treatment was ≈ the control (M = I in 3 of 4 cells, M ≈ I + 0.09·uuᵀ in the fourth), the feature target is 83–87 % unexplainable by the student, and the QAT it was embedded in barely trains (details §3). **[V]**
3. **Stop** the interaction line: no four‑seed confirmation, no matched‑spectrum controls, no β/γ sweeps, no M0/A2 on this metric. Park it as a documented negative screening result.
4. **The pilot uncovered three real, fixable defects** that matter for *any* low‑bit work — including your deployment story: (i) QAT gradients explode at 4 bit (pre‑clip norm 10⁶–10¹⁰, up to 10¹¹ on a probe batch, vs 3–5 in FP; 100 % of steps clipped) — traced to ~0.1 % of attention positions and shown to be a **float32‑simulation artifact** that integer‑domain arithmetic removes (1.1·10¹¹ → 4.0) without changing eval accuracy (Addendum A); (ii) 4‑bit QAT as configured changes only 0.4–2.1 % of the integer weight codes; (iii) the 4‑bit activation grids are poor — above all the **stem input: 64 % of normalised input values round to zero**, and an 8‑bit stem plus unsigned grids recovers 70 % of the W4A4 PTQ gap with no training (Addendum A). **[V]**
5. **The most valuable fact in your data is not about quantisation.** On STURM and WorldFloods, the 300 M teacher, a 31 k‑parameter student and a one‑line band‑ratio rule (MNDWI > 0) sit within **0.9–1.6 pp water‑IoU** of each other, while seed SD is ~0.1–1.3 pp and event‑level CIs are ±1.3–1.9 pp. **[V]** Those benchmarks cannot rank anything inside that band — which is where *every* OFA/KD/architecture/feature‑loss comparison lives. Research should move to questions the instruments *can* answer: **capacity floor, robustness, and what the teacher adds beyond the rule.**
6. **Recommended path** (details §4, prompt for Codex written): Phase A repairs + noise floor (CPU, < 2 GPU‑h) → Phase B capacity‑floor ladder, 4 seeds, robustness, disagreement‑set analysis (≈ 10–14 GPU‑h) → Phase C a *strong* low‑bit baseline, and only if a gap survives, a task‑relevant distillation term → optional Phase D temporal "flood‑increment" feasibility audit (CPU).
7. **Be realistic about novelty.** KD, OFA, pseudo‑labels, QAT, feature MSE, Mahalanobis/Fisher‑type metrics are all prior art. Defensible novelty here is (i) a rigorous *characterisation* — how small is enough, what the foundation model adds beyond a spectral rule, what breaks in tiny attention students at low bit — and, if it works, (ii) label‑free **flood‑increment** supervision from the pool's pre/post‑event pairs. Neither is guaranteed.
8. **Process:** the diagnostics that decide go/no‑go (interaction share, predictability of the target, optimiser movement, gradient norms, seed noise) came *last or never*; a 64 MB manifest and 99‑node DAG wrapped 1.6 GPU‑hours of science. The new prompt inverts this: cheap kill‑diagnostics and a pre‑computed minimum detectable effect come first.

---

## 1. Roadmap and interventions reviewed

| Step | Intervention | What exists now | Assessment |
|---|---|---|---|
| prompt1 | Local baselines, MobileNetV3 / LightFormer OFA controls, provenance audit, dry‑scene + MNDWI robustness, detached queue | Control matrix + queue **[R]** | Necessary reviewer‑proofing; heavy but sound |
| prompt2 | Audit addendum; S11 MNDWI 65.0 % vs RSE 74.39 % | Gap *not* explained by any tested variant **[R]** | Correctly stopped; do not reopen |
| prompt3 | Clip‑/mask‑matched KD vs OFA, focal‑only, N=100, native WF borders, STURM validity | 60 regenerated‑cohort runs **[R]** | Fixed real confounds (true‑CE vs focal, clipping, KL masks) |
| prompt4 | Restore original cohort (2,500 files rehashed, 127 IDs), 44 runs, 28 checkpoints strict‑load | Rescoring drift ≤ 0.019 pp **[R]** | Strong provenance win |
| prompt5 | Reviewer closure: 13 items, M1–M16 / S1–S8 edits, overlap sensitivity | Drafts; 3 STURM chips + 1 WF map overlap noted **[R]** | Compute‑complete; blocked on *human* inputs |
| compression v2 | New feature‑distillation metric on QAT, finite DAG, 12 GPU‑h cap | 1.58 GPU‑h used; 62 nodes skipped by gate **[R]** | Negative; see §3 |

**What the revision track established (all four seeds, corrected protocols, [R] from `prompt5_closure`):** OFA − matched KD at N = 2,500 is −0.34 / −0.02 / +0.44 pp (S11 / STURM / WF) for EfficientViT, similar‑sized and sign‑inconsistent for MobileNetV3 and LightFormer; only N = 100 shows a modest positive effect. This is a *clean controlled null* and should be written as such. **No further mIoU‑level OFA experiments are warranted.**

**Still open on the revision track — all human‑gated, none needs GPU** (`prompt5_closure/remaining_inputs.txt`): (1) the actual reviewer letter; (2) your answer on whether STURM/WF scores influenced method choices (the manuscript's "benchmark‑guided development" sentence vs your account); (3) optional independent dry‑scene dispositions (11 candidates / 3 events); (4) Jetson artifacts for Table III/IV claims.

---

## 2. What the data actually says

### 2.1 The benchmark ceiling **[V]** (`prompt5_closure/tables/current_full_summary.csv` + pilot report)

Same protocols throughout (S11 legacy 512 · STURM finite‑six‑band · WF native zero‑pad). Values: mIoU / water‑IoU, %.

| Model | S11 test | STURM | WorldFloods |
|---|---|---|---|
| Teacher (300 M) | 89.46 / 81.67 | 75.39 / 64.87 | 77.17 / 64.32 |
| MNDWI > 0 (no parameters) | 78.91 / 65.03 | 74.67 / 64.47 | 75.94 / 63.80 |
| EfficientViT‑B0, pretrained, matched KD (0.7 M, 4 seeds) | 87.82 / 78.82 | 76.05 / 65.36 | 76.69 / 63.51 |
| A0 scratch ReLU6 (705 k, **1 seed**) | 85.48 / 74.81 | 73.69 / 62.06 | 76.68 / 63.51 |
| A1 scratch (191 k, 1 seed) | 83.85 / 71.87 | 74.05 / 61.56 | 73.33 / 58.17 |
| A3 scratch (**31 k**, 1 seed) | 85.15 / 74.26 | 75.71 / 64.65 | 77.59 / 65.11 |

* On the two external sets, teacher / MNDWI / EfficientViT / A3 span **0.9 pp (STURM)** and **1.6 pp (WF)** of water‑IoU. Only the *in‑domain* S11 test separates learned models from the rule (the rule's S11 water‑IoU is 9–17 pp lower).
* Precision/recall tells the real difference: on S11 the rule has 67.5 % precision / 94.6 % recall vs teacher 92.4 / 87.5 — the learned models' advantage is **false‑positive suppression**, paid for with recall (on WF: rule 77.8 / 78.0 vs teacher 91.4 / 68.5).
* A1 (191 k) is 4.3 pp *worse* than A3 (31 k) on WF mIoU and 1.7 pp worse on STURM — implausible as a capacity effect, plausible as seed noise (see 2.3). Do not read the pilot's size ordering as real until replicated.

### 2.2 Which knobs move the needle **[V]** (pp, EfficientViT unless noted)

| Knob | S11 | STURM | WF |
|---|---:|---:|---:|
| Pool size N = 100 → 2,500 (matched KD) | **+9.4** | **+3.1** | **+5.3** |
| Focal‑only → weighted KD (matched, N = 2,500) | +0.75 | +0.96 | +1.22 |
| OFA − KD at N = 2,500 (pooled) | −0.34 | −0.02 | +0.44 |
| Architecture: EViT / MobileNetV3 / LightFormer (KD, N = 2,500) — spread | 0.8 | 0.1 | 0.2 |
| Size 705 k → 31 k (A0 → A3, scratch, **1 seed**) | −0.33 | +2.01 | +0.91 |
| Interaction metric B3 − B2 (hard cell, 1 seed) **[R]** | −1.53 | −0.56 | +0.16 |

The only large levers are **data scale** and (in‑domain) the label/teacher signal. Everything the project has refined since — OFA, KD variants, backbones, feature losses — is at or below the instrument's resolving power.

### 2.3 Noise floor and minimum detectable effect **[V]** (`receipts/noise_floor.txt`)

* Seed SD of pooled mIoU (4 seeds, pretrained models, seven N = 2,500 cells): median **0.22 / 0.20 / 0.51 pp** (S11 / STURM / WF), max 0.4 / 0.7 / 0.8; at N = 100 up to **1.8 / 1.3 / 3.0 pp**. Water‑IoU SD is larger (up to 1.3 pp at N = 2,500).
* Paired‑seed SD of OFA − KD → approximate MDE (α = .05, power .80) with 4 seeds: S11 0.05–2.0 pp, STURM 0.4–1.1 pp, **WF 0.4–2.0 pp**.
* Event‑level 95 % CI half‑width for model − teacher: S11 ±0.9 (10 events), STURM ±1.3 (29), **WF ±1.4–1.9 (11)**.
* The supplement (Table S3 text) already says the only non‑pretrained backbone is the most seed‑unstable (±3.7 pp on STURM). The pilot's ladder is from‑scratch, one seed.
* Free noise sample from the pilot **[R]**: B3 and B2 differ only by a 9 % re‑weighting of one of 32 directions, yet they differ by 0.28 pp in source‑validation mIoU (the gate metric) and by **1.53 pp** in S11‑test mIoU (STURM 0.56, WF 0.16). So the outcome's sensitivity to a negligible change is of the same order as the 0.5 pp effect the gate looked for, and on the held‑out S11 test about 3× larger.

---

## 3. Audit of the compression‑v2 pilot

### 3.1 What is solid (keep it)
Cohort/teacher lineage (2,500 files rehashed; teacher SHA matched), pre‑declared graphs and gates, no outcome‑driven expansion, terminal benchmarks read only after design freeze, event‑clustered bootstrap, honest handling of the failed collector, the export bug hunt (a *warm* TorchScript plan diverged from cold by 7·10⁻⁴ logit — a real find), clear labelling of everything as simulation. Total 5,672 GPU‑s (1.58 h) of a 12 h cap **[R]**.

### 3.2 Why the null is uninformative — five verified reasons

**(a) The treatment was (almost) the control.** Shrinkage returned M = I exactly in A1·W4A8, A3·W4A8, A3·W8A8 (so B3 ≡ B2 by construction); in the hard cell A3·W4A4 it kept one direction: M = I + ≈0.09·uuᵀ, ‖M − I‖_F = 0.092, cond = 1.093 **[R/V]**. The interaction it was built to exploit is small and *negative*: η = 2·tr R/(A+Q) = −0.015 … −0.059, "reinforcing" share χ₊ = 0.005 … 0.033 **[R]**. R's spectrum looks like symmetric noise (A3·W4A4: top eigenvalue +0.31, bottom −0.34; only 46–62 % of positive directions stay positive in the other split‑half — the 50 % you get from noise) **[V: `pc_explained.txt`; R]**. Even with a noise‑free R the largest weight would be ≈ 1.5× on one direction (h = 2γλ/ν with λ ≈ 0.31, ν ≈ 1.27) and ≈ 1.1× on a few others **[H, my arithmetic from the metric code]**. A gate of "+0.5 pp over both controls at one seed" could only be passed by noise.

**(b) The feature target is mostly unlearnable by the student.** The 1024‑d teacher neck was PCA‑reduced to 32 and **whitened** (`/scales`, scales = √λ; `distillation/compression_metric.py:88,122`) **[V]**. Held‑out residual energy per whitened direction: **PC1 0.17–0.23, PCs 2–4 0.4–0.6, PCs 5–32 0.86–1.05** (≈ nothing explained; some > 1). The FP student's adapted tap explains **12.9 % (A3) / 17.2 % (A1)** of the 32‑dim energy (tr C_f/32 = 0.871 / 0.828) **[V]**. 28 of 32 directions are noise to the student, and whitening gives each the same weight as PC1. FP residual A ≈ 27.9 dwarfs quantisation displacement Q = 0.5–12.9. Any metric on this space — K = I or interaction — mostly measures that irreducible residual. The supplement's earlier statement that MSE feature matching "did not improve results" and that logit supervision alone already gives high deep‑stage CKA (supplement S8, representation‑alignment paragraph and Fig. S3) is consistent with this: CKA is variance‑weighted and dominated by the top PCs the student does predict, whereas the whitened energy is dominated by the rest **[H]**.

**(c) The QAT it was embedded in barely trains — three separate defects. [V]**

* *Optimiser reach.* AdamW lr = 1e‑5 × 1,560 updates ⇒ a weight can move at most ≈ 0.016; the 4‑bit step is 0.033–0.037 (median). Measured (FP start → QAT end, fixed scales): **only 0.4–2.1 % of integer weight codes change at W4** vs 28–29 % at W8; max |Δw| is 0.001–0.003 (≤ 0.22 step) at W4 vs 2.6–2.9 steps at W8 (`weight_code_flips.txt`). Because the forward pass sees only rounded weights, QAT at W4 changes the function through ~1 % of the quantised weights (plus the unquantised final conv and biases). In 3 of 9 QAT runs the selected checkpoint is the very first evaluation (312 updates) — extra training did not help.
* *Gradient explosion localised to the attention normaliser.* Pre‑clip gradient norm: FP training **3–5** (8–25 % of steps clipped at 5); QAT W8A8 10³–10⁵, W4A8 10⁶–10⁸, W4A4 **10⁹–10¹⁰**, **100 % of steps clipped at 1** (`grad_norms.txt`, `fp_grad.txt`). Single‑batch probe (same 16 augmented training images, focal loss, A3 checkpoints): FP BN‑train 1.1, FP frozen‑BN 3.6, QAT‑mode with fake‑quant *off* 2.7, W8A8 4.0 — but **W4A8 1.3·10⁸ and W4A4 1.1·10¹¹**. Quantising only the LiteMLA `qkv` conv of stage 2 gives 2.5·10⁶ by itself (stage 3: 6·10⁴); the median over all 34 single‑layer cases is 3.0. In the W4A4 forward, **7.7–8.9 % of attention denominators are ≤ 1e‑6**; A3's attention dim is 4 (A1: 8; A0: 16), so many positions have exactly‑zero ReLU‑kernel normalisers. Raising the clamp from 1e‑6 → 1e‑2 takes W4A8 from 1.3·10⁸ → **4.3** (forward loss 0.1686 → 0.1683) and W4A4 from 1.1·10¹¹ → 525 (→ 36 at 0.1; forward loss 0.2588 → 0.2506, i.e. it *changes* slightly and must be checked on validation) (`grad_bisect.txt`, `grad_clamp.txt`). After global‑norm clipping to 1, every parameter not on the exploding path is scaled by up to 10⁻¹¹, i.e. below Adam's ε [H]; consistent with the head's last two convs moving 40–120× less at W4A4 than at W8A8 (relative change 2.4·10⁻⁴ vs 9.5·10⁻³ and 4.7·10⁻⁵ vs 5.5·10⁻³; `param_motion.txt`). The historical W8A8 deployment (attention dim 16) is not obviously affected.
* *Weak 4‑bit quantiser.* Activation scale = global absmax / qmax over all channels (`quantization/qat.py:412–419`); the grid is signed for every conv input. In A3: **15 of 34 conv inputs are entirely non‑negative** (half the grid unused), and at A4 a **median 55.7 % (max 88.2 %) of the activation values entering a conv round to zero** (8 unaugmented training images, fake‑quant off); p99 of |x| reaches only 0.68 of the clip edge (`quant_headroom.txt`). Weights are less of a problem: MSE‑optimal clipping would cut 4‑bit weight MSE by 22.7 % (+1.1 dB). PTQ damage is also non‑monotone in size (W4A8: A0 −1.6, A1 −4.6, A3 −0.5 pp; W8A8: A1 −0.9) **[R]** — graph‑specific outliers, not a clean capacity story.

⇒ B1/B2/B3 are three slightly perturbed copies of a PTQ model. B2 − B1 = +1.3 pp (source val, W4A4) may be a generic feature‑distillation effect or noise; across the 12 external B2 − B1 comparisons it is positive in 9, mean +0.28 pp, one seed **[R, my arithmetic]** — a hint, not evidence.

**(d) One seed, from scratch, tiny graphs.** See 2.3. Moreover the frontier winner (A1, picked by a 1‑pp *S11‑validation* tolerance) is the worst of the three FP students on WF (−3.3 pp vs A0, −4.3 pp vs A3), while A3 — excluded by that tolerance — is the best on both STURM and WF. The "smallest eligible artifact" therefore depends on which benchmark, and which seed, you trust.

**(e) "Compression" is simulated.** All 4/8‑bit results are fake‑quant floats; exported fake‑quant models run 2.6× *slower* than FP on the 4090 (A3: 3.44 vs 1.33 ms); FP16 export is blocked; the "size frontier" is FP32 bytes **[R]**. No packed low‑bit artifact exists.

### 3.3 Why interaction‑weighting is conceptually shaky **[H]**
R = E[sym(f qᵀ)] is a statistic of the *error* geometry, not of the *task*. When the residual cannot be zeroed, a good K should put the leftover error where it hurts the task least — i.e. weight directions by their influence on the logits (a Gauss‑Newton/Fisher form K ≈ JᵀHJ through the decoder), which `R` never sees. Moreover R is partly endogenous: minimising ‖t − s_q‖² itself drives q against f, so η becomes *more* negative after B2 training (A1·W4A8: −0.065 → −0.147) **[R]**. A frozen‑at‑PTQ R is stale once training starts. If any feature‑level term is revived, the principled variant is task‑aware (Fisher/Jacobian) — but that overlaps known Hessian/Fisher‑weighted reconstruction work in PTQ and should be treated as engineering, not novelty (student must verify in related work).

### 3.4 What the pilot does legitimately show
* A legal 31 k‑param (382 KB FP32) EfficientViT‑style graph trains and — at one seed — lands in the same external band as the 705 k graph (needs replication).
* Feature MSE at the pre‑decoder tap is not harmful (9/12 positive, small).
* The measurement machinery (cluster bootstrap, export validation, collectors) is reusable.

---

## 4. Where to go

**Principle:** ask questions the instruments can answer, with a pre‑computed MDE, ≥ 4 seeds, event units for geography.

### Track 1 — do now: the capacity floor, with robustness and a disagreement lens
**Question.** *How small can a foundation‑distilled flood segmenter be, and what does it retain that a spectral rule lacks?*
**Ladder (all trained with the identical teacher/cohort/recipe, scratch init, 4 seeds):** per‑pixel linear (≈14 params) → per‑pixel MLP (≈400) → tiny 3×3 CNN (≈1 k) → small context CNN (≈10 k) → A3 without attention (≈30 k) → A3 (31 k) → A2 (62 k) → A1 (191 k) → A0 (705 k); reference rows: pretrained EViT/MobileNetV3/LightFormer, teacher, MNDWI.
**Why it is informative.** (i) It places the *knee* of the accuracy curve. If a per‑pixel MLP already matches A3 on externals, the "foundation model → edge" story is *label provenance for a learned spectral classifier*; if it fails by several pp, spatial context is what the teacher transfers. (ii) The A3‑with/without‑attention pair tests whether LiteMLA is needed at 31 k — it is also the component that breaks low‑bit QAT. (iii) Robustness (haze, opposing band tilt) probably needs more capacity than clean accuracy — a testable, non‑obvious hypothesis **[H]**.
**Disagreement‑set analysis (new lens, cheap).** Define sets *without the reference*: T = teacher water, R = MNDWI > 0; evaluate every model on {T≠R} (where the teacher deviates from the rule) via stratified per‑chip confusion counts. That isolates "what the teacher adds beyond the rule" and whether tiny students inherited it, without being drowned by the many pixels where teacher and rule agree. Add pairwise agreement (student↔teacher↔rule↔reference) to test whether the benchmark ceiling is label‑protocol‑limited **[H]**.
**Claim you could defend if it works:** "A ≤ X‑parameter student is non‑inferior to the teacher on STURM and WF (pre‑registered one‑sided margin, event‑level CIs); the rule is matched on clean data but not under haze; the teacher's added value lies in {T≠R} pixels, which students of ≥ Y parameters retain." Note: with 11–29 events a two‑sided equivalence claim at ±1.5 pp is not attainable (even the pretrained B0 is significantly below the teacher on WF), so the knee may legitimately come out *undefined* on WF — that is itself a finding about the benchmark.
**Cost.** ≈ 42 runs × ~15 min (teacher forward dominates; 875 s per 50‑epoch run measured) ≈ 10 GPU‑h + ≈ 3 GPU‑h evaluation; ~5–7 h wall on two 4090s **[estimate]**. *Option:* train several students against one shared teacher pass (teacher cost is ~constant) for ~4× saving — only after verifying equivalence to single‑student runs on one rung.

### Track 2 — after the gradient fix: a *strong* low‑bit baseline, then ask if anything is left
Fix the attention‑normaliser gradient path (opt‑in variant; do not touch `quantization/qat.py`), then make B1 strong: unsigned grids for non‑negative inputs, MSE/percentile clipping, learnable step sizes (LSQ‑style) or a weight LR large enough to move codes, longer schedule, and a simulated graph that matches the exported one (BN folding). Verify the standard techniques in the literature before citing (from memory: LSQ, PACT, BRECQ, HAWQ‑style sensitivity).
**Stop rule.** If strong B1 is within ~0.5 pp of FP at W4A8 (S11‑val) the quantisation of these students is solved with standard tools — report a sub‑50 k INT8/W4A8 model and stop. Only if ≥ 1 pp remains is a *generic* feature term (task‑relevant subspace, **un‑whitened**, top‑k predictable PCs) worth testing, with 3+ seeds and an MDE computed first. K‑metrics come last, if ever.
**Claim you could defend:** "Tiny attention students are numerically unstable under low‑bit QAT (normaliser degeneracy); conv‑only students of equal size quantise to W4A8 without measurable loss using standard QAT." — useful engineering, modest novelty.

### Track 3 — higher novelty, higher risk: temporal flood‑increment distillation (feasibility audit only for now)
Your own limitation section says the student reproduces surface‑water semantics and a flood‑*increment* product needs a reference‑water/temporal component; 96.1 % of teacher‑flagged baselines are reproduced by the student (§IV‑G). The pool already pairs tiles: **11,895 event + 11,825 baseline tiles**, baselines 30–395 days earlier at the same location, stricter 20 % cloud gate **[V: `datasets/download_flood_events.py`, `datasets/README.md`]**, and `datasets/flood_pool_quality.py` already computes an MNDWI "dry→wet new water" proxy. Idea: pseudo‑label *increment* = teacher(event) ∧ ¬teacher(baseline), train a compact student on (event [+ baseline‑mask]) and score it with WorldFloods flood‑only reference **including** JRC permanent water as negatives (a stricter test than the current flood‑only masking) **[H]**.
**Risks:** seasonality over a 30–395 day gap, co‑registration, double pseudo‑label noise, no flood‑increment ground truth beyond WF. **Do only a CPU feasibility audit now** (pair counts after the existing pre‑filter, alignment check, receding‑vs‑new‑water symmetry as a noise estimate); decide on the evidence.

### Track 4 — JSTARS revision
Write the controlled nulls as nulls; adopt the benchmark‑ceiling analysis (2.1) and disagreement lens as the answer to R2.5 ("when do learned models help vs MNDWI": false‑positive suppression in‑domain and under haze; not on clean external mIoU). Deliver only what needs you: the four human inputs in §1. Use Track 2's strong W8A8 baseline if the INT8 loss (−1.1 pp STURM / −1.6 pp WF device‑float → INT8) can be reduced.

### Effort (estimates, not measurements; two RTX 4090s)

| Phase | Content | GPU‑h | Wall | Gate |
|---|---|---:|---|---|
| A | gradient‑path repair, noise floor, pre‑registration | < 2 | < 0.5 day | A's repair must pass its numeric criteria before Phase C |
| B | capacity ladder (≈ 42 runs), robustness, disagreement sets | ≈ 13–16 | 0.5–1 day | functional gate per new graph on seed 42 |
| C | strong low‑bit baseline (3 seeds) → optional generic feature term | ≤ 8 | ~0.5 day | stop rules in the prompt |
| D | temporal feasibility audit | 0 (CPU) | ~2 h | go/no‑go note |

### Decision tree after Track 1
* Knee ≤ ~5 k params and robustness tracks → paper angle "How little is enough"; go to Track 2 with conv‑only graphs and a real edge target.
* Clean accuracy saturates early but robustness needs ≥ 100 k → angle "clean benchmarks hide the capacity cost of robustness"; revisit *task‑relevant* feature distillation for robustness, not quantisation.
* Per‑pixel ≈ teacher ≈ rule on externals and {T≠R} pixels are few → angle "benchmark saturation / spectral sufficiency"; build the harder failure‑set evaluation, Track 3 becomes the main novelty bet.

---

## 5. What must change (summary)

**Must**
1. Reclassify the pilot as a *negative screening result*; do not "rescue" skipped arms.
2. No comparison without a pre‑computed MDE; ≥ 4 seeds for S11/STURM, more for sub‑1 pp WF claims; single‑seed gates may only *kill*, never *select*.
3. Repair the QAT gradient path and build a strong low‑bit baseline before any low‑bit mechanism claim.
4. Keep frontier *selection* validation‑based and pre‑registered (no external number may influence it — the benchmarks are research‑exposed), but **report** every model on all three benchmarks with CIs next to it, and pre‑register a better selector before the next study instead of switching retrospectively. (Corrected: the earlier wording "select on all three benchmarks" contradicted this rule.)

**Should**
5. Move from "mIoU wins" to capacity floor, robustness, and the disagreement lens.
6. Diagnostics‑first prompts: write kill criteria with numbers (η/χ share, predictable‑variance fraction, code‑flip rate, gradient‑norm ratio vs FP) before expensive runs.
7. Shrink artefacts: store shared lists once (hash‑referenced); a 64 MB manifest for ≤ 100 jobs is a smell; ≤ 1 page of text per phase.

**Could**
8. Multi‑student shared‑teacher training; Track 3 after its audit; a human‑plus‑pseudo mixed‑pool run to test whether the in‑domain S11 gap (teacher +0.8–1.7 pp mIoU over the pretrained students) is label‑convention rather than capacity **[H]**.

---

## 6. Feedback on how the student worked

*Good:* rigorous provenance, honest labelling of simulation vs deployment, no peeking at external sets, graceful failure handling, reproducible receipts.
*To change:* (1) decisive diagnostics came late — the interaction share and per‑direction predictability were available after Wave 0/1 and already forecast the null; (2) engineering volume (8 `*_check.py` scripts, 64 MB manifest) vs scientific payload; (3) the "hardest cell" was defined as smallest‑graph × lowest‑bit, i.e. where the *baseline* is broken (−13.7 pp PTQ), so nothing subtle can show; pick hard cells *after* the strong baseline calibrates the landscape; (4) thresholds (0.5 pp, 1 pp tolerance) were set without a measured noise floor; (5) prompts of 25–35 KB create room for contradictions — the next one is deliberately short with numeric acceptance criteria.

---

## 7. Three questions for you (they change what I would do)

1. Is the compression direction a **separate follow‑up paper** or must it feed the JSTARS revision? (I assume separate.)
2. Is **any target hardware** available now (Xavier NX, Orin, an NPU/FPGA board)? Without it, Track 2 ends at "simulated + exported", and the device claims in the revision stay qualified.
3. Do you want Track 3 explored beyond the CPU audit? It is the highest‑novelty option and the only one that addresses the semantics limitation head‑on, but it is the riskiest.

---

## Addendum A — new verified findings after reviewing prompt6 (2026-10-08, later) and corrections to this report

Receipts: `supervisor_checks_20261008/receipts/` (`run_all.sh` fast set ≈ 4 min; `full` adds the slow sweeps). Review and change map: [`prompt6_review_20261008.md`](prompt6_review_20261008.md). Superseding prompt: [`prompts/prompt6_corrected.md`](../../prompts/prompt6_corrected.md) (+ appendix).

1. **The gradient explosion is a simulation artifact [V].** Computing each quantized conv on integer codes and scaling afterwards (exact zeros preserved) gives W4A4 1.14·10¹¹ → 4.0, W4A8 1.29·10⁸ → 4.4, W8A8 3.98 → 3.97, with S11‑val mIoU unchanged (PTQ ≤ 0.03 pp; selected v2 QAT checkpoints ≤ 0.14 pp). The top 0.1 % of attention positions (25/25,088; 12/12,544) hold ≈ 100 % of the squared qkv gradient; all have 0 < D_Q < 1e‑3 (median ≈ 1e‑7, |q| ≈ 5–8e‑9, float32 rounding‑residue scale); 64–83 % of them had a healthy FP denominator. Caveat: on one augmented training batch the two simulators disagree on 7.5 % of argmax pixels at W4A4, so training‑batch parity must be checked.
2. **Exact‑zero denominators already exist in FP [V].** 5–9 % of positions (attention dim 4) in the quantization‑bypassed branch of the same weights, with normal gradients. Quantization‑induced collapse (D_Q < D_F/16) hits 6.5–8.2 % at W4A4, 3–4 % at W4A8, 1.5–1.7 % at W8A8. **Correction to §3.2(c), second bullet:** "many positions have exactly‑zero normalisers" described the quantized forward but wrongly suggested quantization creates them.
3. **Forward damage is activation‑side and mostly the stem [V]** (A3 PTQ, S11‑val mIoU %): W4A4 68.99; W8A4 68.84; stem input at 8 bit 76.23 (+7.24); unsigned 4‑bit grids at 15 non‑negative sites 72.17 (+3.18); both 78.31 (+9.32); W4A8 82.20; FP 82.70; keeping the attention projections at 8 bit adds nothing (75.89). The v2 "hard cell" (W4A4 on the smallest graph) was therefore dominated by a conventional first‑layer and grid‑signedness problem.
4. **A1's W4A8 damage is localised [V].** 8‑bit weights for one 4‑channel stem depthwise conv (36 weights, +18 B): 78.95 → 82.01 (random layer 79.01; all‑W8 82.66); top‑4 82.98 for +1.8 % payload. Optimistic (ranked and scored on the same 89 images); a label‑free/held‑out confirmation is Stage 1.4 of the new prompt.
5. **Per‑pixel pilot, S11‑val only, single fit [V, motivation not evidence].** MNDWI 75.91 mIoU; zero‑sum log‑linear index (6 parameters, exactly gain‑invariant) 81.36; 401‑parameter per‑pixel MLP 85.45 vs A3 82.70, A1 83.60, A0 84.16. This raises the priority of the capacity‑floor study (Track 1) and motivates the creative hypotheses U1–U8 in the new prompt (distilled gain‑invariant index, context‑free sufficiency, invariance‑by‑construction stem, rule‑residual learning, rule–net disagreement as uncertainty, label‑free mixed precision, disagreement‑driven curation, two‑labeler agreement). MACs/pixel of the MLP (≈ 384) exceed A3's (≈ 164): its edge advantage would be memory/streaming, not compute.
6. **Corrections:** (a) Must #4 (above); (b) bottom‑line item 4 (above); (c) the effect of the gradient fix on final QAT accuracy remains **untested**; (d) `prompts/codex_next_prompt_v3.md` is superseded.

## Appendix A — receipts (all CPU, read‑only)

| Script | Shows | Key numbers |
|---|---|---|
| `weight_code_flips.py` | QAT start→end integer‑code changes | W4: 0.4–2.1 %, W8: 28–29 %; max |Δw| ≤ 0.22 step (W4) |
| `grad_norms.py`, `fp_grad.py` | Per‑epoch pre‑clip norms; FP baseline | FP 3–5; QAT 10³–10¹⁰; 100 % clipped |
| `grad_probe.py` | Same weights, one batch, 5 modes | 1.1 / 3.6 / 2.7 / 4.0 / 1.3·10⁸ / 1.1·10¹¹ |
| `grad_bisect.py` | Weights vs activations bits; per‑layer; attention denominators | qkv stage‑2: 2.5·10⁶; 7.7–8.9 % den ≤ 1e‑6 |
| `grad_clamp.py` | Effect of normaliser floor | W4A8 1.3·10⁸ → 4.3; loss unchanged |
| `param_motion.py` | Relative parameter movement | head convs W4A4 4.7·10⁻⁵/2.4·10⁻⁴ vs W8A8 5.5·10⁻³/9.5·10⁻³ |
| `pc_explained.py` | Per‑PC residual; R eigenvalues | PCs 5–32 ≈ 0.86–1.05; R symmetric |
| `quant_headroom.py` | 4‑bit weights vs activations | −22.7 % weight MSE possible; 55.7 % A4 zeros |
| `noise_floor.py` | Seed SD, MDE, event CI widths | see 2.3 |

## Appendix B — not verified by me
Per‑GPU timings and the 12 h cap accounting; the pilot's terminal tables (taken from its report; I only re‑did arithmetic); the claim that FP16 export fails because of `UpSampleLayer`; the unpublished origin of the 74.39 % RSE value; any statement about the literature (LSQ, PACT, BRECQ, HAWQ, Φsat‑1) beyond what your manuscript cites — the student must check prior art before any novelty claim; whether baseline/event tiles in the pool are co‑registered; whether a mixed human+pseudo pool would close the S11 gap; effect of the gradient fix on final QAT accuracy (not run).
