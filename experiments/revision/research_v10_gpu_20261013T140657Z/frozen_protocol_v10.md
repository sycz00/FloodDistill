# V10 Codex prompt: is small-model distillation worth it, where does it fail, and what can we control? (CPU first)

Prepared 13 October 2026, Europe/Berlin, by Claude (Opus). Revision 3; revisions 1–2 are preserved as prompt_v10_cpu_followup.rev{1,2}_20261013.md.
This prompt runs **in addition to** prompts/prompt_v9_overnight.md. V9 stays as it is; V10 only reads V9 artefacts. The authors need reviewer-ready results within a few days, so the work is tiered: finish Tier A before Tier B, and Tier B before Tier C. This is an experimental instruction, not a claim that any method is novel or successful.

## 1. Mission, priorities and constraints

Work in the existing FloodDistill research branch. Reuse:
- the V7/V9 loaders, graphs, evaluators, label guard, PTQ path and edge code;
- the tested reference implementation in experiments/revision/v10_reference/.

Inspect AGENTS.md and the working tree first. Preserve user changes and every earlier run directory. Write everything new to experiments/revision/research_v10_<UTC>/. Use four seeds (42, 64, 91, 107) for every trained or compared model. Report seed SD and the event-level uncertainty separately.

**Tier A (must; reviewer-critical; little or no new training):**
- **A1 — Is small-model distillation worth it?** Compare tiny and small distilled students against the same and larger models trained only on the 252 human-labelled Sen1Floods11 chips (UNet-ResNet50, EfficientViT-B0, A0, A3, R2, R4), plus the teacher and the spectral rules. Use one protocol, every set and the sealed test.
- **A2 — Shared failures and ceilings.** Which errors do the teacher, every student, the supervised baselines and the rules all share? How much of the remaining error is beyond any six-band optical model?
- **A3 — Size × precision frontier.** Readout from 14 to 705k parameters, at FP32, strict W8A8, W4A8 and W4A4. Split each model's errors into inherited and compression-induced.
- **A4 — Operating-point attribution.** Is V9 arm E3's WorldFloods gain only a more liberal threshold?
- **A5 — Smallest viable model, and whether distillation is what makes it viable.** Train a size × supervision grid:
  - sizes: R2 418, R3 1,058, R4_micro 1,266, R4 12,230 parameters;
  - supervision: S11 human labels only / teacher hard labels / teacher logit KD, with GDACS data for the teacher-label arms.

  Choose the smallest viable model with a source-only rule. Then test the authors' hypothesis: at that size, a model trained only on Sen1Floods11 cannot reach the distilled model's out-of-distribution accuracy.

**OFA is not a V10 subject.** The reviewers are right that OFA adds no consistent gain (OFA ≈ logit KD in every controlled comparison). Report it only as the existing ablation. V10's results should let the paper rest its contribution on A1/A5 (what distillation buys a tiny on-board model, against supervised baselines), on A2 (why all methods saturate) and on A3/B (what compression breaks and how to control it), not on OFA.

**Tier B (the novelty-bearing questions; small CPU training):**
- **B1 — Why do the smallest models fail on dark surfaces and bright water?** Capacity, spatial context, input representation or training data? Hold the parameter count roughly fixed while varying one factor at a time.
- **B2 — Can compression-induced failures be controlled?** Test operating-point refit, calibration ranges, layer protection, and QAT with distillation from the FP parent or from the teacher. Apply them to R4 and to the A5-selected smallest viable model.

**Tier C (only if time remains):** C1 mixed-pixel shoreline targets; C2 decode-time edge rule with stronger indices; C3 strict-integer ONNX Runtime check.

Dropped: cloud-shadow geometry. Record it as future work in negative_results.md.

**Constraints:**
- CPU first. Measured on this host:
  - R4 trains 7,800 updates in about 4 min of compute;
  - UNet-ResNet50 trains at about 1.0 s per batch-16 step at 224 px;
  - inference per 512 × 512 tile: UNet 0.15 s, Prithvi 0.72 s, B0 0.02 s, R4 0.01 s.
- A GPU may be used only for inference of the teacher and UNet on the sealed set, and only if an authorized GPU is idle under V9's Section 9 rules and the shared ledger. Charge it there. No GPU training.
- Use at most 24 of the 48 cores. Keep the combined load with V9 below 40 cores.
- No downloads, no new teacher training, no manuscript edits, no messages to other people, no subagents.
- Out-of-distribution labels are evaluation-only. S11 validation alone selects thresholds, settings, checkpoints and controls.

## 2. Context to verify (treat as [R] until re-derived)

Sources: the V9 runs research_v9_20261012T235412Z and research_v9_recovery_20261013T075000Z, and Claude's CPU probes on exposed sets in experiments/revision/novelty_probe_20261012/ and experiments/revision/v9_early_readout_20261013/.

**V9 facts:**
- E0 (V7 R4_CE), E1, E2, E3 (R4, 4 seeds) and the B0 label-swap pairs are trained.
- Edge setting k = 2, theta = 0.1. E3 used p* = 0.2.
- The teacher decoder outputs H/4 logits, like R4.
- STURM counts permanent water as water; WF excludes it.
- V9's evaluation waits for a GPU because its frozen bundle needs 7.5 GB.

**Early readout on exposed sets** (equal-event water IoU change vs E0, four seeds):

| Arm | S11 test | STURM | WF |
|---|---|---|---|
| E1 | +0.90 | −1.21 | +1.11 |
| E2 | −0.98 | −0.57 | −1.18 |
| E3 | +0.11 | −0.69 | +3.43 |
| D on E0 | +1.09 | +0.13 | +0.89 |

Label swap: human ≈ teacher labels on S11; teacher labels +1.04 on WF.

**Supervised vs distilled, existing evidence** (capacity v5 report B4; legacy protocols; water IoU):

| Model | Distilled − supervised, STURM | Distilled − supervised, WF | In-domain (S11 val) |
|---|---|---|---|
| A0 (705k) | +8.65 [+0.46, +16.83] | +2.27 [−0.17, +4.70] | ≈ equal |
| A3 (31k), supervised SD 5.3 | 64.80 vs 44.11 | 61.30 vs 56.94 | |
| R2 (418) | +4.70 [+1.61, +7.79] | −2.22 [−3.31, −1.12] | −4.93 |

Paper Table I (pooled mIoU; STURM / WF): UNet-ResNet50 CE on the 252 human labels 72.1 / 77.2; B0 CE on human labels 71.7 / 74.4; distilled B0 76.0 / 77.5; teacher 75.4 / 77.5; MNDWI 74.7 / 76.3.

**Shared failures** (teacher + R4 + MNDWI; probe_shared.py):
- Water missed by all three: S11 3.0%, STURM 12.3%, WF 21.4% of water, which is 24% / 74% / 68% of the teacher's misses.
- On STURM/WF these pixels look like land (NIR 0.21–0.25, SWIR1 0.18–0.19). Vegetated reference water is missed by all three 70–95% of the time.
- WF: 22% of the shared misses lie within 50 px of clouds. S11: 56% lie within 2 px of shorelines.
- The shared false water on STURM/WF is spectrally water-like.

**Size ladder** (capacity v5; four-seed checkpoints at /data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008/v5/fl_<G>_sel_s{64,91,107} plus the lr-selected s42 directory):

| Rung | Architecture | Parameters | Receptive field | STURM mIoU |
|---|---|---|---|---|
| R1 | per-pixel linear | 14 | 1 px | 70.8 |
| R2 | per-pixel MLP | 418 | 1 px | 74.9 |
| R3 | 2× 3×3 conv, full resolution | 1,058 | 5 px | 75.4 |
| R4 | context CNN | 12,230 | > 64 px | 75.6 |

**Single-seed s64 stratification** (novelty_probe_20261012/ladder_report.txt):
- Pooled water IoU barely changes along the ladder.
- R2 and R3 call 78–98% of dark land water on S11 and 28–92% on WF, versus 8–9% for R4. R1 misses dark water instead.
- R2/R3 find 48–51% of bright water on S11, versus 64% for R4.

This single-seed look is hypothesis-generating only. Parameter count is confounded with receptive field: R1/R2 see one pixel, R3 sees 5 px. R3 is therefore not a size-matched context model, and in the probe it had the worst dark-surface false-water rate (98% on S11). The size-matched context model is R4_micro (1,266 parameters, receptive field > 64 px; v10_reference.build_tiny).

Risk for the A5 hypothesis: in capacity v5, R2 trained on human labels beat R2 LKD on S11 validation (+4.9) and on WF water IoU (+2.2), and lost only on STURM (−4.7). Distillation is therefore not guaranteed to win at tiny size. A5 must report every sign.

**Indices at threshold 0** (equal-event water IoU, S11 test / STURM / WF; WF dry windows flagged of 216):

| Index | S11 test | STURM | WF | WF dry flagged |
|---|---|---|---|---|
| MNDWI | 68.3 | 39.2 | 40.1 | 140 |
| WI2015 | 73.8 | 40.1 | 43.4 | 128 |
| AWEIsh | 76.8 | 39.5 | 43.1 | 97 |
| R4 | 80.3 | 37.8 | 47.6 | 9 |

**Checkpoint inventory:**
- Human-label B0 CE N252: runs/original_runs/V2/gt_s11_matched/sen_ce_N252_s{42,64,91,107}.
- Supervised A0/A3/R2 (v5): m_sup_A0_*, m_sup_A3_*, mf_sup_*.
- No human-label R4 exists.
- The N252 human-label UNet checkpoints are missing; only a smoke-test s42 exists, which is not usable. The retained sen_ce_unet_N251_* runs are obsolete (N = 252 only) and must never be evaluated or reported.

## 3. Data roles and the sealed set

| Data | Role |
|---|---|
| S11 train (252 human-labelled chips) | teacher fine-tuning (done) and every supervised baseline |
| GDACS N2500 cohort | every distilled student |
| S11 validation | every selection |
| S11 test, STURM, WF test | exposed readouts (descriptive) |
| WorldFloods-v2 train/val (V9 sealed set: 468 scenes, 116 events) | sealed test |

Rules for the sealed test:
- Reuse V9's sealed_windows.json and label guard (common_v9.install_label_guard). Do not re-screen.
- Freeze V10 first. Write frozen_protocol_v10.md with every model, arm, setting, grid, seed and decision rule, then hash it and all V10 code before any V10 training starts. Record whether V9 sealed metrics exist then (research_v9_recovery_*/metrics/sealed).
- If V9 sealed results were opened before the V10 freeze, label every V10 sealed result "second look, non-confirmatory".
- V10 opens sealed labels exactly once, in its final evaluation job, after all checkpoints and settings are frozen. Log every opened label path.

## 4. Tier A

### A1. Distilled versus supervised: the comparison the paper needs

**Rows** (four seeds each, unless the row is not trained):

| Group | Rows |
|---|---|
| Rules, no learning | MNDWI > 0 (primary baseline, as in the paper); WI2015 and AWEIsh (secondary; computed and archived, presentation is the authors' decision) |
| Teacher | Prithvi-EO-2.0 300M, fine-tuned on the 252 human labels |
| Supervised on the 252 human labels | UNet-ResNet50 CE (retrain, see below); EfficientViT-B0 CE (sen_ce_N252); A0 SUP; A3 SUP; R2 SUP (v5); R4 SUP (new, see below); the V9 label-swap human B0 arms (descriptive, different recipe) |
| Distilled on 2,500 GDACS tiles | R4 LKD parents (fl_R4_sel); R4 CE (E0); R2 LKD and R2 CE (v5); A3 LKD; A0 LKD/OFA (v5); EfficientViT-B0 paper OFA (O4 historical_ofa_clip5_allvalid_N2500) and the Table II logit-KD runs if retained and verified |

Flag any arm whose S11 test score is more than 3 pp below its paper/report value as an anomaly, and do not use it in contrasts. The O4 matched_lkd_clip5_valid_N2500 arm read 78.4 pooled S11 mIoU in prompt5's summary, versus 87.5 for the paper arms.

**UNet-ResNet50 SUP:**
- Recover the exact Table I recipe from the retained sen_ce_unet_N251 run configs: same script, only the index differs. Train on the N252 index on CPU, 4 seeds (about 15–20 min each). The N251 checkpoints are never used.
- Acceptance: the four-seed S11 test pooled mIoU lies within 2 SD of Table I (0.862 ± 0.018). Otherwise report the deviation and keep the rows flagged.

**R4 SUP** (the same runs as A5's R4 SUP cell; train once):
- The v5 SUP recipe (matched updates on the 252 human-labelled chips), applied to R4 at its selected LR, 4 seeds on CPU.
- Its init and batch order must match the v5 SUP conventions. Record them.

**Evaluation:**
- One evaluator, current protocols (S11 test legacy, STURM P3 finite support, WF P3 native) and the sealed set.
- Pooled and equal-event water IoU, mIoU, precision, recall, average precision, dry-window alarms (≥0.1/1/5% area) and per-event wins.
- Cost columns: parameters, MACs per pixel, FP32 bytes.

**Paired contrasts** (seed-matched where the architecture matches; otherwise seed-mean):
- LKD − SUP for R2, R4, A0, A3;
- R4-LKD − UNet-SUP and R4-LKD − B0-SUP;
- R4-LKD − MNDWI, WI2015 and AWEIsh.

### A2. Shared failures and ceilings (CPU)

**Inputs:** extend novelty_probe_20261012/probe_shared.py and probe_failure.py to every A1 row and to the sealed set.

**Per set, report:**
- the joint error code of teacher, each model and MNDWI against the reference, per event;
- each model's share of misses and false water shared with the teacher and MNDWI;
- the strata: shoreline distance (≤2, 3–10, >10 px), water half-width ≤ 2 px, vegetated (NDVI > 0.4), dark (visible < 0.06, NIR < 0.08, SWIR1 < 0.06), bright (visible > 0.2), cloud distance (reference cloud band where available, input cloud test elsewhere), and the WF label-source subgroup;
- mean spectra of shared misses and shared false water.

**Ceilings** (reference-based, unattainable): pooled and equal-event water IoU if all shoreline errors, all cloud-margin errors, or all spectrally land-like misses were removed. Label them as limits of six-band optical mapping.

### A3. Size × precision frontier (CPU; existing checkpoints)

**Grid:**
- Rungs R1, R2, R3, R4, A3, A3N, A0 (LKD, four seeds).
- Precisions, all with the adopted simulator (prepare_compression_qat; the same 16 TRAIN calibration IDs and calibration seed 1729 as V9's ptq_manifest.json):
  1. FP32;
  2. strict W8A8 on every layer including stem and head (relevant to R2.6);
  3. W4A8 body / W8A8 stem / FP32 head (the V8 point);
  4. W4A4 body / W8A8 stem / FP32 head.

**Error classes:**
- **inherited:** the teacher is also wrong;
- **compression-induced:** the same-seed FP32 R4 parent is right and the model is wrong. For precision variants of one rung, use that rung's same-seed FP32 model as the reference;
- **other.**

Report each class per stratum, per event and as dry-window alarms.

**Clipping.** For quantized variants, record per layer and per set/event the share of activations clipped at the calibrated range, and its Spearman association with the quantized − FP event difference (descriptive).

**Sets:** every variant runs on the exposed sets. On the sealed set, run only the predeclared subset: {R2, R4} × {FP32, strict W8A8, W4A8} × 4 seeds, plus every B1 arm and every B2 control.

### A4. Operating-point attribution (CPU)

For each seed:
- find tau so that E0's predicted water fraction on S11 validation equals E3's (v10_reference.source_matched_tau on pooled S11-validation margins);
- evaluate E0 at that tau against E3 everywhere.

Apply the same test to E1 and to every B1/B2 arm against its reference. If the matched-tau reference comes within 0.3 pp sealed equal-event water IoU of an arm, attribute that arm's effect to the operating point.

### A5. Smallest viable model and the value of distillation at that size (CPU training: 32 new runs; the 16 CE cells are shared with B1)

**Teacher logit cache.** Run the frozen Prithvi teacher once on CPU over the 2,500 GDACS cohort tiles at 224 px (about 0.1–0.2 s per tile). Store float16 logits (not probabilities) with hashes.

**Offline KD.** Use v10_reference.offline_kd_loss, which reproduces DistillLoss: 0.5·focal + 0.5·T²·KL with T = 4, KL over valid pixels.
- Apply every geometric augmentation jointly to image, mask and cached logits (bilinear for logits, nearest for masks).
- Photometric augmentation (multiplicative noise) changes only the student input; the target stays the clean-image teacher output. Record this as the one difference from live LKD.
- Do not derive logits from the float16 confidence map (it saturates and softens targets).

**Grid.** Four seeds per cell, CPU, identical recipe per supervision type:

| Size | SUP: 252 human-labelled S11 chips (v5 SUP recipe, matched 7,800 updates) | CE: teacher hard labels, GDACS N2500 (R4 CE recipe) | KDoff: cached teacher logits, GDACS N2500 (same recipe + offline KD) |
|---|---|---|---|
| R2 (418, per-pixel) | ✓ | shared with B1 | ✓ |
| R3 (1,058, 5 px receptive field) | ✓ | shared with B1 | ✓ |
| R4_micro (1,266, receptive field > 64 px) | ✓ | shared with B1 | ✓ |
| R4 (12,230) | ✓ | shared with B1 | ✓ |

The CE cells are B1's reference arms; do not train them twice. Existing GPU-trained runs are descriptive references only: v5 R2 SUP/CE/LKD, V7 R4_CE and the live-LKD R4 parents. As a check of the offline approximation, report R4 KDoff against the live-LKD R4 parents on every set.

**Smallest viable model** (predeclared, source-only). Using the KDoff arms, take the smallest size that meets both conditions:
- S11-validation equal-event water IoU within 1.0 pp of R4 KDoff (seed means);
- a teacher-referenced dark-surface false-alarm rate no more than 2× R4 KDoff's, measured on 500 held-out pre-event GDACS pool tiles not in the cohort. Count pixels that pass the input dark test and that the teacher calls land (p < 0.1).

If no smaller size qualifies, R4 is the smallest viable model. Record the decision before any exposed or sealed readout of A5.

**Readout.** Same evaluator and strata as A1, on all sets including the sealed set.

**Contrasts:** KDoff − SUP and CE − SUP at every size; KDoff − CE (soft targets); and the smallest-viable KDoff against UNet-SUP and B0-SUP from A1.

## 5. Tier B

### B1. Tiny-model failure decomposition (CPU training, 32 runs)

**Question:** do the dark-surface and bright-water failures of the smallest models come from too few parameters, from missing spatial context, from an input representation a tiny network cannot turn into illumination-invariant ratios, or from missing training examples?

**Arms.** The R1/R2/R3/R4/R4_micro CE runs double as A5's CE cells. All use the R4 CE recipe on the GDACS N2500 cohort with cached teacher hard labels: focal γ = 2, 7,800 updates, batch 16, AdamW, weight decay 0.05, clipping 5, epoch-stepped cosine, S11-validation selection every 312 updates, the same augmentations. Each arm trains 4 seeds on CPU. Every model comes from v10_reference.build_tiny or rungs.py; parameter counts are verified by tests.

| Arm | Model | Parameters | What it tests |
|---|---|---|---|
| R1 | per-pixel linear | 14 | reference, retrained on CPU |
| R2 | per-pixel MLP | 418 | reference, retrained on CPU |
| R3 | 3×3 conv, 5 px receptive field | 1,058 | reference, retrained on CPU |
| R4 | context CNN | 12,230 | reference, retrained on CPU |
| R2_log | R2 on [bands, standardized log-reflectance] | 514 | representation |
| R2_ctx | R2 on [bands, 9 px and 33 px local band means] | 610 | parameter-free context |
| R4_micro | R4 design at widths (4,4,6,8), decoder 4; receptive field > 64 px | 1,266 | learned context at R3 size |
| R2_hardneg | R2 trained on the A1 cohort of prompt_v9_overnight.md Section 5 (250 pre-event tiles with teacher water < 0.01 and MNDWI water ≥ 0.05, replacing a fixed 250); A2 easy-negative cohort as its control | 418 | training data |

Compute the log-reflectance standardization statistics on the N2500 TRAIN images only, store them as model buffers, and record them.

**Primary metrics:**
- dark-land false-water rate and dark-water recall;
- bright-water recall;
- shoreline recall and false water, thin-water recall;
- dry-window alarms;
- water IoU.

All on S11 validation/test, STURM, WF test and the sealed set.

First check that R2 and R3 trained with CE reproduce the single-seed LKD dark-surface failure. If they don't, report it and treat the decomposition as uninformative.

### B2. Compression controls (CPU)

Each control is selected on S11 validation and aimed at a named failure. Apply them to R4 and to the A5-selected smallest viable model, at the A3 precisions:

| Control | Failure targeted | What it does |
|---|---|---|
| (a) operating point | a margin shift | Source-selected threshold refit (A4 method). |
| (b) calibration | out-of-distribution clipping | Min-max (adopted) vs 99.99th percentile vs per-channel ranges. Calibration set: the 16 adopted tiles vs 64 TRAIN tiles stratified by pool features to include dark, bright and cloudy tiles. Pool features only, never target labels. |
| (c) layer protection | sensitive layers | Rank layer sensitivity on S11 validation by single-layer 4-bit substitution, then keep the two most sensitive layers at 8-bit, against uniform W4A4. |
| (d) QAT with distillation | aggressive points failing the 1 pp source tolerance | Only R4 W4A4 and the A5-selected smallest viable model at W4A8; if that model is R4, use R4 W4A8. Three continuations per point, 4 seeds, ≤1,560 updates, same init/batches/optimizer: (d0) FP continuation control; (d1) QAT with KL to the same-seed FP32 parent; (d2) QAT with offline KD to the cached teacher logits. If every arm worsens, do not rescue QAT by changing its schedule. |

## 6. Tier C (only if time remains; keep the definitions frozen)

### C1. Mixed-pixel shoreline targets

**Definitions** (v10_reference.py):
- band = pixels within k = 2 px of the teacher hard mask's edge;
- fraction f = v10_reference.unmix_fraction(x, teacher_mask, k=2, radius=7, min_px=8, max_residual=1.0); where f is NaN, keep the teacher label;
- p_water from the stored teacher confidence.

**Arms** (R4 CE recipe on CPU, four seeds; targets differ only inside the band; soft_focal_loss everywhere):

| Arm | Target inside the band |
|---|---|
| F0 | teacher hard |
| F1 | physics soft, t = f |
| F2 | teacher soft, t = p_water |
| F3 | physics hard, t = 1[f > 0.5] |
| F4 | operating-point control, t = 1[p_water > 0.2] |

Also report edge-band Brier score and ECE on S11 validation/test.

The source-only feasibility (v10_reference/feasibility_report.txt) found AUROC against the human label inside the band of 0.779 for the teacher, 0.798 for f and 0.819 for WI2015. Expect small effects.

### C2. Decode-time edge rule with stronger indices

**Grid:** index ∈ {MNDWI, WI2015, AWEIsh}, k ∈ {1, 2, 3}, and per-index theta:

| Index | theta values |
|---|---|
| MNDWI | −0.1, 0, 0.1, 0.2 |
| WI2015 | −5, 0, 5, 10 |
| AWEIsh | −0.05, 0, 0.05, 0.1 |

**Selection:** best mean S11-validation equal-event water IoU of the four R4 LKD parents. Freeze the setting, then apply it to every model with v10_reference.decode. Report the CPU time per tile.

### C3. Strict-integer runtime

Export R4 and R2 at strict W8A8 to ONNX QDQ. Compare ONNX Runtime CPU integer kernels with the simulator on S11 validation/test: pixel agreement and per-event IoU difference. CPU latency is workstation latency only.

## 7. Decision rules (declared before training; sealed set only for claims)

Report per-seed differences, the event-bootstrap 95% interval of the seed-mean difference (10,000 resamples, seed 1729) and the across-seed SD separately. The event is the unit. Non-significance is not equivalence.

- **Distillation is worth it at small size.** Requires R4-LKD − R4-SUP ≥ +1.0 pp sealed equal-event water IoU, with lower bound > 0 and ≥ 3 of 4 seeds positive. Report the same contrast for R2, A3 and A0. The existing evidence is mixed for R2 on WF, so report every sign.
- **At the smallest viable size, distillation is what makes the model viable (authors' hypothesis, A5).** Requires all of:
  - for the smallest viable model, KDoff − SUP ≥ +1.0 pp sealed equal-event water IoU, with lower bound > 0 and ≥ 3 of 4 seeds positive;
  - the same sign on at least two of S11 test, STURM and WF test.

  Report every size and every set: the v5 R2 evidence says SUP can win in-domain and on WF. If SUP wins in-domain but loses out of distribution, say exactly that. Soft targets help if KDoff − CE ≥ +0.5 pp (lower bound > 0). Otherwise attribute the gain to the GDACS data, not the soft targets.
- **The tiny distilled model is comparable to a large supervised one.** A non-inferiority claim only: allowed if the event-bootstrap lower bound of R4-LKD − UNet-SUP (and of R4-LKD − B0-SUP) on the sealed set is above −1.5 pp, with the margin declared now. Always report the parameter ratio. If the bound is below −1.5 pp, report it plainly.
- **Learning beats the rule.** Report R4-LKD − MNDWI, WI2015 and AWEIsh on every set: equal-event and pooled water IoU, AP and dry-window alarms. State where the rule wins (expected: STURM equal-event).
- **Failure attribution (B1).** On the sealed set and ≥ 2 exposed sets, with ≥ 3 of 4 seeds consistent, a reduction in R2/R3's dark-land false-water rate (or bright-water miss rate) of ≥ 50% relative is attributed to:
  - context, if R2_ctx or R4_micro achieves it;
  - representation, if R2_log achieves it;
  - data, if R2_hardneg beats both R2 and its easy-negative control;
  - capacity, only if R4 achieves it and no size-matched variant does.

  Several attributions may hold at once.
- **Compression creates and controls specific failures.**
  - A failure mode counts if compression-induced errors concentrate in a stratum at ≥ 2× its share of pixels (sealed set and ≥ 2 exposed sets, ≥ 3 of 4 seeds).
  - A B2 control counts only if it reduces that stratum's compression-induced error by ≥ 50% relative, without lowering equal-event water IoU by more than 0.3 pp or raising dry-window alarms by more than 1 percentage point of windows.
  - If control (a) alone removes most of a quantization loss, call that loss an operating-point shift.
  - QAT rescue: d1 or d2 must beat d0 and the unchanged PTQ point by ≥ 0.5 pp sealed equal-event water IoU (lower bound > 0), and must bring the quantized model within the 1 pp source tolerance of its FP32 parent. If d2 beats d1, report that teacher distillation also helps the quantization recovery, not only the FP training.
  - Strict W8A8 is reported against the 1 pp primary and 2/3 pp secondary source tolerances and is never called lossless.
- **Tier C.** C1 needs F1 to beat both F0 and F2 by ≥ +0.5 pp (lower bound > 0, ≥ 3 of 4 seeds, no STURM loss beyond −0.5 pp). C2 needs ≥ +0.5 pp on the deployed student, with the same conditions.
- **Never:** select, gate, retry or extend anything after reading exposed or sealed outcomes.

## 8. Verification

Before any training:
- run v10_reference/run_all.sh. Its tests check that the soft focal loss equals the existing focal loss on hard labels; that unmixing recovers synthetic fractions; that edge distance and decode match edge_v9; the threshold bisection; the 2 × 2 block test; the tiny-model parameter counts (R2 418, R2_log 514, R2_ctx 610, R4_micro 1,266), with R2 matching rungs.R2's shapes; and that offline_kd_loss equals DistillLoss's focal + KD formula;
- reproduce V9's R4 W4A8 PTQ states for seeds 42/64 on CPU from the same 16 calibration IDs. Compare tensors with research_v9_20261012T235412Z/ptq_states/q_s{42,64}.pt;
- confirm that strict W8A8 quantizes every layer (precision map);
- for the teacher logit cache: on 5 tiles, the cached logits equal a fresh CPU teacher forward (max |difference| < 1e-3); jointly augmented logits equal the teacher output of the augmented image for pure flips and rotations; v10_reference.offline_kd_loss is used unchanged (test_offline_kd_equals_distill_loss);
- check the clipping counters on a synthetic tensor;
- verify sealed-window identity against V9 and that the label guard is active;
- reproduce one existing checkpoint's S11-validation score with the V10 evaluator, to within 0.1 pp;
- do a smoke run of every training arm: 50 updates, finite loss, recorded initial hashes, and identical first-batch images across arms that share a cohort.

After training: checkpoint hashes, selected updates and source-validation metrics per run. Confirm that nothing was selected on a non-S11-validation set. Confirm that no N251 artefact was read.

## 9. Execution and survival

Use one detached CPU worker (setsid/nohup, stdin closed, absolute interpreter /home/schmalstieg/miniconda3/envs/esi/bin/python, PYTHONPATH including /home/schmalstieg/repos/_deps/efficientvit) with:
- a single-worker lock;
- atomic state writes;
- a 30-second heartbeat;
- per-job receipts;
- one retry for a diagnosed transient technical failure;
- status/pause/stop commands that work from a fresh terminal.

It must not touch the V9 worker, its locks or its ledger, except to charge optional GPU inference under V9's Section 9 rules. Run training jobs at most 3 in parallel, at 8 threads each.

Order:
1. freeze, tests, verification;
2. Tier A on the exposed sets, including the UNet and R4 SUP training;
3. B1 and B2 training and exposed readouts;
4. one final sealed evaluation job for every frozen model, variant and arm. Split it over at most 3 CPU processes, or use an idle authorized GPU for the teacher and UNet;
5. Tier C, if time remains. Tier C models are evaluated on the sealed set only if they were frozen before the final sealed job began. Otherwise they are exposed-only.

Rough wall times:

| Step | Wall time |
|---|---|
| Tier A training (UNet, R4 SUP, teacher logit cache, A5 grid of 32 new tiny runs) | ~3 h |
| B1/B2 (B1 adds 4 arms beyond the A5 CE cells; B2 QAT 24 short runs) | ~2 h |
| Exposed readouts | ~2 h |
| Final sealed job on CPU | ~5 h (shorter with a free GPU) |

The wall limit is 20 hours from launch; reserve the final 30 minutes for collection. On a deadline or stop, collect what exists and mark the rest DEFERRED. Publish an interim report_v10.md after Tier A on the exposed sets, so the authors can start writing.

## 10. Deliverables

All in research_v10_*:
- frozen_protocol_v10.md and its hash; the code snapshot; test and verification receipts; the label-access log;
- supervised_vs_distilled.csv, supervised_vs_distilled_by_event.csv and contrasts.csv (with parameters, MACs and bytes);
- shared_failures.csv, shared_spectra.csv and ceilings.csv;
- compression_frontier.csv and activation_clipping.csv;
- operating_point_attribution.csv;
- decomposition_results.csv (B1), compression_controls.csv and qat_pairs.csv (d0/d1/d2);
- size_supervision_grid.csv (A5: size × {SUP, CE, KDoff} × seed × set × stratum), smallest_viable_selection.json, offline_vs_live_kd.csv and teacher_logit_cache_manifest.json;
- optional Tier C files: mixed_pixel_results.csv, decode_rule_results.csv, strict_int8_ort.csv;
- index_baselines.csv;
- negative_results.md;
- report_v10.md in plain language, with an experiment checklist (arm, seeds, status, evidence path) and a section "what this means for the reviewer answers":
  - R1.1/R2.1: contribution or characterization;
  - R1.2/R2.4: alternative models;
  - R2.5: rules;
  - R1.6: false alarms;
  - R2.6: strict INT8.

Separate observation from interpretation. Deferred or unavailable blocks are not negative evidence.

## 11. Prior art (starting points, not a novelty clearance)

- Spectral water indices:
  - Feyisa et al. 2014, Remote Sensing of Environment (AWEI);
  - Fisher, Flood, Danaher 2016, Remote Sensing of Environment 175 (WI2015);
  - Xu 2006, IJRS (MNDWI);
  - Du et al. 2016, Remote Sensing 8, 354 (10 m MNDWI by sharpening);
  - Wang et al. 2018, Remote Sensing 10, 1643 (MuWI).
- Sub-pixel water fraction: Wang et al. 2023, GIScience & Remote Sensing; Water 15(8), 1446, 2023.
- Foundation-model vs supervised segmentation: Marsocci et al., PANGAEA, 2024 (https://arxiv.org/abs/2412.04204).
- Pseudo-label refinement: SegFix (Yuan et al., ECCV 2020).
- Quantization:
  - Nagel et al. 2021, white paper (https://arxiv.org/abs/2106.08295);
  - Nagel et al., ICCV 2019, bias correction (https://arxiv.org/abs/1906.04721);
  - Esser et al., ICLR 2020, LSQ (https://arxiv.org/abs/1902.08153);
  - Kim et al. 2019, QKD (https://arxiv.org/abs/1911.12491).

Claim boundaries:
- A1 tests a practical claim: distilling a geospatial foundation model lets a 12k-parameter model match or beat much larger models trained on the same human labels on unseen floods.
- A5 tests whether, at the smallest viable size, the teacher (through extra unlabelled imagery and soft targets) is what makes the model viable.
- OFA is reported as a neutral ablation, not a contribution.
- B1 and A3 test a mechanism claim: aggressive compression keeps pooled accuracy but breaks specific, nameable pixel types, and a size-matched change of context, representation or data can prevent this.

No first-ever claims and no guaranteed improvements.


## Execution resolution before training

The user approved FP32 teacher-logit caches for all eight geometric views on13October2026. The literal float16 cache failed the0.001 absolute tolerance, and transformed cached teacher outputs failed the fresh-forward check. Every offline target is the teacher output of the matching clean geometric image; photometric jitter applies to the student only. The offline-KD loss function remains byte-identical to v10_reference. The cache uses CPU inference.

UNet N252 follows the retained nsweep launcher and frozen Trainer defaults: ImageNet encoder from local cached weights, true CE, batch16,224px,50epochs,AdamW5e-5,weight_decay0.05,clip0,cosine50,validationevery2epochs,smoothing3,early-stop patience20,no AMP,deterministic. No N251 artifact is opened. No retained per-run N251 configuration sidecars were present, so this provenance limit is disclosed. Tiny SUP uses520epochs×15updates=7800,focal2,clipping5,source validationevery21epochs=315updates as the executable v5 recipe (the historical prose says300). CE/KDoff use50epochs×156updates,AdamW0.005,weight_decay0.05,clipping5,source validationevery312updates,earliest strict-greater mIoU selection. R4_micro inherits R4 LR0.005.

Historical42/64 PTQ calibration ran on GPU. CPU regeneration retains small quantizer-range differences and verifies all master weights and source IDs; no bit-identical calibration claim is made. Every V10 simulator result is labelled CPU simulation, separate from native integer execution and V9 GPU PTQ. Strict W8A8 includes every convolution, stem and head. Fixed simulator convolutions remain FP32 arithmetic.

Shared failure classes overlap: inherited and compression-induced intersections are reported; other excludes both. Spectrally land-like water is the fixed reference-water stratum with NIR>=0.08 and SWIR1>=0.06. Shared failures/oracles do not prove an intrinsic optical identifiability limit.

QAT d1 and d2 use targets from the matching clean geometric image, with student-only photometric jitter in both arms. The FP parent is cheap enough to evaluate per sampled clean view; d2 uses the approved teacher cache. This keeps the target-noise convention controlled while comparing parent versus teacher targets. Both use the unchanged offline_kd_loss. d0 is focal-only FP continuation with QDQ bypassed. All three use the same prepared init, frozen BN, AdamW5e-4,weight_decay1e-5,clipping1,1560updates.

All new CPU arms reseed the data stream after model construction; the human-label augmentation Compose has an explicit seed. This implements the required identical minibatch checks independently of graph initialization, and is a disclosed difference from historical GPU trajectories. Historical GPU arms remain descriptive. Shared-cohort SUP and CE/KD smoke hashes gate their full runs. Optional C models/settings are admitted by remaining wall time only and frozen before final derived-label readout. Missing or incomplete seed groups cannot pass a scientific gate.

## Explicit user-authorized GPU migration

On13October2026 the user commanded GPU execution and declared the CPU restriction outdated, then authorized10additionalGPUhours: cumulativecap40hours. This section supersedes every CPU-only device sentence above; data, graphs, losses, augmentation, seeds, steps, source-only choices and analysis gates remain unchanged. New training and model tests/inference run CUDA FP32 with AMP disabled and TF32 disabled under the existing authorized UUID locks and shared ledger. CPU statistical analysis and the explicitly CPU ORT deployment check remain on their intended backend. The original14October11:07CEST deadline and30minute collection reserve are retained. CPU completed/partial results are retained in the previous directory; all new comparisons use four fresh GPU seeds because exact loader/augmentation checkpoint resumption is unsupported. Verified CPU teacher targets are reused by identity and hashes; missing tiles use GPU eight-view FP32 outputs after five-tile CPU/GPU target parity below0.001 is demonstrated. No sealed outcomes informed this migration or any scientific setting.
