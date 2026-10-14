# Novelty ledger and intervention ideas (draft, 9 Oct 2026)

Every row follows the protocol: observed failure → intervention (with its formula) → the matched control that
isolates it → the metric → the falsifier → the nearest prior art (checked 9 Oct 2026 by web search, not by
full reading). **No row is a claim yet.** Status: *tested* (result in report), *queued* (in the v5 queue),
or *proposed* (not run).

## 1. Simulation-faithful QAT for ReLU linear attention (status: tested in Stage 1; strongest candidate)

* **Failure.** With fake-quant, the LiteMLA output is a ratio num/D. Under 4-bit quantization a few positions have
  0 < D_Q < 1e-3 (median 1e-7, float32 rounding residue). The gradient ∂(num/D)/∂D = −num/D² then reaches 1e8–1e11.
* **Intervention.** Integer-domain QDQ: run the conv on integer codes and rescale afterwards,
  y = s_x s_w · (Σ q_x q_w), so exact cancellations stay exactly 0. The STE masks stay unchanged.
* **Diagnostic metric (new).** Residue-denominator rate
  ρ_res = P(0 < D_Q < ε·|s_q s_k|), with ε = 2⁻²³ (float32 epsilon),
  together with the gradient share G_attn = Σ_{l∈qkv}‖g_l‖² / Σ_l‖g_l‖².
* **Control.** Same weights under three simulators: fake-quant, integer, and a denominator floor τ.
  Forward parity must hold (PTQ ≤ 0.05 pp).
* **Falsifier.** The norms stay above 10× FP under the integer simulator. Outcome: **not falsified**
  (W4A4 1.2×, W4A8 2.1×).
* **Prior art.** Trio-ViT (PTQ for EfficientViT) uses log2 quantization for sensitive linear-attention divisors.
  The flood-segmentation preprint keeps the LiteMLA matmuls in FP16.
  No QAT residue-division mechanism was found → candidate contribution, phrased as a *measurement* finding.

## 2. Log-domain (radiometric companding) stem (status: proposed; cheapest high-value test)

* **Failure.** Stem input at A4: 64 % of normalised reflectance values round to 0 (step 1.43 z-units). The stem
  input causes most of the 4-bit forward damage (bypass +4.2 / +7.2 pp).
* **Intervention.** Feed x̃_b = log(ρ_b + ε_b) − μ_b. Water indices are band *ratios*, and a ratio becomes a
  *difference* in log space: log G − log SWIR. A 1×1 conv computes that exactly, and a uniform 4-bit grid on
  log-reflectance has *relative* error Δρ/ρ ≈ Δ, independent of brightness.
  It also gives invariance: a scalar gain k shifts every band by log k, which zero-sum stem weights
  (Σ_b w_{cb} = 0) cancel exactly.
* **Metric.** Share of stem inputs that round to zero at A4; S11-val damage of a 4-bit stem; out-of-range gain/haze robustness.
* **Control.** Same graph with the z-score stem (T0). The other control is the 8-bit stem exception (Q4b in E2).
* **Falsifier.** The log stem at A4 recovers < 50 % of the stem damage, or clean S11-val is lower by more than the MDE (~1.5 pp).
* **Prior art.** Log-chromaticity and illumination-invariant features are classic in vision. µ-law companding is
  classic in audio. I found no combination with 4-bit stems for multispectral segmentation (search
  was not exhaustive).

## 3. Distilled, exactly gain-invariant spectral index (U1) (status: proposed; CPU-only)

* **Failure / question.** Teacher, MNDWI and the 31 k student lie within ~1.5 pp on STURM/WF. What does the teacher
  know beyond a rule?
* **Intervention.** Fit s(x) = Σ_b w_b log ρ_b + c with Σ_b w_b = 0 to the teacher's soft labels
  (logistic loss on TRAIN only). Gain invariance is then exact: s(kρ) = s(ρ). The model has 5 free weights plus a bias.
* **Metric.** S11-test, STURM and WF water-IoU against MNDWI. Also gain/haze/tilt robustness.
* **Control.** MNDWI>0, a non-zero-sum log-linear fit (no invariance), and R1 (linear on z-scored input, KD-trained, queued).
* **Falsifier.** No CI-excluding S11-test gain over MNDWI, or worse than MNDWI by > 2 pp on STURM/WF.
* **Prior art.** SRWI (symbolic-regression water index, 2026) is learned from WorldCover labels. DeepAqua distils in
  the *opposite* direction (index → network). I found no teacher → closed-form-index distillation (search not exhaustive).

## 4. Flip-aware ("positive-congruent") distillation for QAT (status: proposed for Stage 5)

* **Failure.** QAT damage is concentrated in harmful flips. At A3 W4A4, H = 7.2 % of pixels go from correct to
  wrong, against B = 1.65 % repaired. Accuracy_Q − Accuracy_FP = B − H holds exactly.
* **Intervention.** Weight the KD term by the FP model's correctness and margin:
  L = Σ_i [1 + λ·1(ŷ_FP,i = y_i)·σ(m_FP,i/τ)]·KL(p_FP,i ‖ p_Q,i),
  where m = z_y − z_¬y is the FP margin. An alternative is a margin-preserving hinge:
  Σ_i max(0, m_FP,i − m_Q,i − δ).
* **Metric.** H, B, the water-IoU change, and boundary versus interior H.
* **Control.** The same extra KL budget spread uniformly (SB+HW in protocol §5). The objective (KD to the FP model)
  and the gradient scale are matched.
* **Falsifier.** No reduction of H beyond the uniform control (paired, 3 seeds), or water-IoU worse.
* **Prior art.** Positive-congruent training / focal distillation (Yan et al., CVPR 2021) and ELODI target *model
  updates* in classification. Margin-preserving training (2025) does the same. The compressed-LLM flip study
  (NeurIPS 2024) measures flips but proposes no fix. Applying this to QAT of segmenters appears open (search not exhaustive).

## 5. Learn only what the rule cannot: rule-residual student (U4a) (status: proposed)

* **Failure.** The rule (MNDWI) has high recall and low precision on WF, while learned models show the opposite.
  Most pixels are already decided by the rule.
* **Intervention.** z_water = a·MNDWI(x) + b + f_θ(x) with a and b learnable. The student models only the residual.
  The KD weight is concentrated on the disagreement set {T≠R}:
  w_i = 1 + γ·1(T_i ≠ R_i).
* **Metric (new lens).** Value-over-rule V(m) = Acc_{T≠R}(m) − Acc_{T≠R}(R), the accuracy on pixels where
  teacher and rule disagree, scored against human labels. This makes visible a difference that the pooled IoU hides.
* **Control.** The same student without the prior head, plus random weighting with the same KL mass.
* **Falsifier.** No {T≠R} gain at equal parameters, or the floor rung does not move down.
* **Prior art.** Physics-guided residual learning is common. A rule-residual tiny segmenter distilled from a foundation
  model, scored with a disagreement-set metric, was not found (search not exhaustive).

## 6. Capacity-aware / uncertainty-aware KD at the floor (status: proposed; depends on E4/H7)

* **Failure (expected).** At a 2,500-image cohort, hard pseudo-labels already carry most of the teacher's information.
  CE, LKD and OFA differ by ≤ 1.3 pp even for a pretrained B0 (paper Table II). If H7 is not confirmed, plain KD
  adds nothing at the floor either.
* **Intervention.** Focus KD where soft targets carry information, at the Bernoulli variance of the teacher:
  w_i = 4·p_T,i(1 − p_T,i). Or use a learnability gate w_i = σ((KL_i^{(0)} − KL_i)/τ) (SCKD-style). Or use a
  teacher assistant (A1 → A3 → R-rung, TAKD).
* **Metric.** Boundary F1 at a declared tolerance, and {T≠R} accuracy.
* **Control.** LKD with the same total KL weight.
* **Falsifier.** No paired-seed gain ≥ MDE.
* **Prior art.** TAKD (Mirzadeh et al., AAAI 2020), pixel-wise similarity KD, FAKD, SCKD. All are known. Only the
  *flood/foundation-teacher, sub-30 k-param* setting would be new, so frame this as an empirical study, not a method.

## 7. Parameter floor ≠ compute floor (status: measured; descriptive finding)

* **Observation (verified).** MACs per pixel at 512²: R1 12, A3N 144, **A3 164**, A2 243, R2 384, R4 417,
  R3 1,024, A0 2,114.
  The 31 k-parameter A3 needs *less* compute than the 418-parameter per-pixel MLP, because it computes at stride
  ≥ 2 almost everywhere.
* **Implication.** "Smallest model" must be stated on two axes: parameters/bytes (memory, uplink) and MACs or
  latency (energy). The capacity-floor result will be reported on both.
* **Prior art.** Well known in efficient-architecture work. This is a framing point, not a contribution.

## 8. Label-free sensitivity-guided mixed precision (U6) (status: tested; negative-to-neutral)

* **Observation.** The teacher-excess-KL ranking finds the stem layer (top-4 overlap 2/4 with the label ranking;
  77 % of damage recovered with 18 B). But the simple structural rule "≤ 64 weights or ≤ 8 output channels"
  matched it with 4× fewer extra bytes.
* **Status.** Do not claim. Report it as a negative result: the simple rule suffices.
* **Prior art.** HAWQ, Fisher-weighted mixed precision.
