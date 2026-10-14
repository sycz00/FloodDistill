# V10 Codex prompt: shared failure ceiling, compression failure modes, mixed-pixel targets and operating-point attribution (CPU only)

Prepared 13 October 2026, Europe/Berlin, by Claude (Opus) after the V9 training finished and the V9 evaluation was blocked by occupied GPUs.
This prompt runs **in addition to** prompts/prompt_v9_overnight.md. V9 stays as it is; V10 only reads V9 artefacts. This is an experimental instruction, not a claim that any method is novel or successful.

## 1. Mission and constraints

Work in the existing FloodDistill research branch. Reuse the V7/V9 loaders, graphs, evaluators, label guard, edge code and the reference implementation in experiments/revision/v10_reference/. Inspect AGENTS.md and the working tree first. Preserve user changes and every earlier run directory. Write everything new to a timestamped directory experiments/revision/research_v10_<UTC>/.

V10 answers four questions with existing checkpoints and new CPU training of the 12,230-parameter R4 student:

1. **Shared failures (characterization).** Which errors do the Prithvi teacher, the students and the spectral rules share on every set? What share of the remaining error is outside what six Sentinel-2 bands can show?
2. **Operating-point attribution.** Is the V9 control arm E3's WorldFloods gain explained by a more liberal operating point? Test it against E0 with a source-matched threshold.
3. **Mixed-pixel targets (main training question).** At shorelines, does a physically unmixed sub-pixel water fraction, used as a soft target, beat the teacher's hard and soft edge labels? This is a 2 × 2 design: {teacher, physics} × {hard, soft}.
4. **Decode-time rule and baselines.** Does a decode-time edge re-decision with a stronger index (WI2015, AWEIsh) beat the MNDWI version? How do the indices compare with the students when nothing is tuned on out-of-distribution labels?
5. **Compression-induced failure modes (second main question).** Which failure modes are created by aggressive compression rather than inherited from the teacher or the data? Compression here means fewer parameters (14 → 12,230) and fewer bits (FP32 → strict INT8 → 4-bit). Can targeted, source-selected controls remove those failures? Unlike the shared ceiling, these errors depend on architecture and numerics, so they are controllable in principle.

**Constraints:**
- CPU only. No GPU command, so V10 consumes none of the 30-hour GPU cap. Measured: R4 trains 7,800 updates in about 4 min of compute on 8–16 threads; R4 inference takes 0.01 s and EfficientViT-B0 0.02 s per 512 × 512 tile.
- Use at most 24 of the 48 cores at any time, and never more than the V9 worker leaves free. Check `uptime` and the V9 scheduler_status.json before every admission.
- No new data acquisition, no downloads, no new teacher training, no manuscript edits, no messages to other people, no subagents.
- Out-of-distribution labels are evaluation-only. Only S11 validation selects anything: thresholds, k, theta, the index, checkpoints.

## 2. Context to verify (treat as [R] until re-derived)

**Sources:**
- V9 run directories: research_v9_20261012T235412Z and research_v9_recovery_20261013T075000Z.
- Claude's exploratory probes, CPU and read-only on exposed sets: experiments/revision/novelty_probe_20261012/ and experiments/revision/v9_early_readout_20261013/.

**V9 facts:**
- Training finished for E0 (V7 R4_CE anchors), E1, E2, E3 (R4, 4 seeds each) and the B0 label-swap pairs.
- The source-selected edge setting is k = 2, theta = 0.1.
- E3 used p* = 0.2: inside the band, water if teacher p_water > 0.2.
- The teacher decoder outputs H/4 logits, like R4. The earlier "160 m patch" explanation is wrong.
- GDACS tiles are about 11.8 m per pixel after resizing to 224.
- STURM counts permanent water as water; WF excludes it.

**Early readout on exposed sets** (four seeds; equal-event water IoU change versus E0; descriptive only):

| Arm | S11 test | STURM | WF |
|---|---|---|---|
| E1, MNDWI edge labels | +0.90 | −1.21 | +1.11 |
| E2, edge band ignored | −0.98 | −0.57 | −1.18 |
| E3, teacher-confidence control | +0.11 | −0.69 | +3.43 |
| D on E0 (decode only) | +1.09 | +0.13 | +0.89 |

Thin-water recall on S11 test: E0 19.3%, E1 19.7%, E3 30.0%. In the label swap, human labels ≈ teacher labels on S11 (−0.17), and teacher labels are slightly better on WF (+1.04, 4 of 4 seeds).

**Shared failures** (probe_shared.py, teacher + R4 seed 42 + MNDWI>0):
- Water missed by all three: S11 3.0%, STURM 12.3%, WF 21.4% of all water. That is 24% / 74% / 68% of the teacher's misses.
- On STURM and WF those pixels have land-like spectra (NIR 0.21–0.25, SWIR1 0.18–0.19). Vegetated reference water is missed by all three 70–95% of the time.
- On WF, 22% of the shared misses lie within 50 px of clouds.
- On S11, 56% of the shared misses and 74% of the shared false water lie within 2 px of a shoreline.

**Feasibility on S11 validation, the selection set** (v10_reference/feasibility_report.txt). AUROC against the human label inside the teacher edge band:

| Cue | AUROC |
|---|---|
| Teacher probability | 0.779 |
| Unmixed fraction | 0.798 |
| MNDWI | 0.786 |
| WI2015 | 0.819 |
| AWEIsh | 0.815 |

The best hard accuracy is 0.72 at 48% water. Shorelines are intrinsically ambiguous, so expect small effects.

**Indices at their published threshold 0** (indices_report.txt; equal-event water IoU, S11 test / STURM / WF):

| Index | S11 test | STURM | WF | WF dry windows flagged (of 216) |
|---|---|---|---|---|
| MNDWI | 68.3 | 39.2 | 40.1 | 140 |
| AWEIsh | 76.8 | 39.5 | 43.1 | 97 |
| WI2015 | 73.8 | 40.1 | 43.4 | 128 |
| R4 student | 80.3 | 37.8 | 47.6 | 9 |

**Size ladder** (capacity v5, pooled mIoU on STURM):

| Rung | Parameters | STURM mIoU |
|---|---|---|
| R1 | 14 | 70.8 |
| R2 | 418 | 74.9 |
| R3 | 1,058 | 75.4 |
| R4 | 12,230 | 75.6 |
| A3 | 30,922 | 76.0 |
| A0 | 704,714 | 74.8 |

Four-seed checkpoints exist for every rung: /data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008/v5/fl_<G>_sel_s{64,91,107} and the lr-selected s42 directory.

**Exploratory single-seed (s64) stratification** (novelty_probe_20261012/ladder_report.txt; exposed sets):
- Pooled water IoU barely changes along the ladder: on WF, 64.1 (R1), 64.7 (R2), 64.8 (R3), 66.3 (R4).
- The small rungs fail on **dark surfaces** (shadow-like pixels; dark = visible mean < 0.06, NIR < 0.08, SWIR1 < 0.06). The false-water rate on dark land is:

  | Rung | S11 | WF |
  |---|---|---|
  | R2 | 78% | 28% |
  | R3 | 98% | 92% |
  | R4 | 9% | 8% |

  R1 misses dark water instead: recall 0.6% on S11.
- R2/R3 also lose bright (turbid) water on S11: recall 48–51% vs 64% for R4.
- Errors that are new relative to R4 sit mostly at shorelines on S11 (39–56%) and in water interiors on WF (52–61%).

This is a hypothesis-generating single-seed look, not evidence.

**Quantization history:**
- V8 R4 W4A8: mean −0.016 pp, but the three events moved −1.24 / −0.15 / +1.34 pp.
- A3 W4A8 QAT matched its float twin (STURM 76.02 vs 76.00).
- B0 INT8 on Jetson lost 3–4 pp for seed 64 only.
- The adopted PTQ is models' prepare_compression_qat(model, calib, weight_bits, activation_bits, calibration_seed=1729), with 16 TRAIN calibration tiles and the stem overridden to 8-bit (see research_v9_20261012T235412Z/jobs_v9_final02.py and ptq_manifest.json).

## 3. Data roles and the sealed set

| Data | Role |
|---|---|
| S11 train | teacher training (untouched) |
| GDACS N2500 cohort | student training |
| S11 validation | every selection |
| S11 test, STURM, WF test | exposed readouts (descriptive) |
| WorldFloods-v2 train/val (V9 sealed set: 468 scenes, 116 events after exclusions) | sealed test |

Rules for the sealed test:
- Reuse V9's sealed_windows.json and label guard (common_v9.install_label_guard). Do not re-screen or change the set.
- Freeze V10 first. Write frozen_protocol_v10.md with every arm, setting, grid, seed and decision rule, then hash it and all V10 code before any V10 training starts. Record whether V9 sealed metrics exist at that moment (research_v9_recovery_*/metrics/sealed).
- If V9 sealed results were opened before the V10 freeze, label every V10 sealed result "second look, non-confirmatory".
- V10 opens sealed labels exactly once, in its final evaluation job, after all checkpoints and settings are frozen. Log every label path opened.
- Never use sealed or exposed outcomes to select, gate, retry or extend anything.

## 4. Experiments

### 4.1 Shared-failure characterization (CPU, existing models)

**Models:**
- the teacher (Prithvi): CPU inference at 0.7 s per tile. Exposed sets take about 45 min. For the sealed set use 3 processes × 8 threads, about 3 h. If V9 has already produced teacher predictions for a set, reuse them after verifying hashes.
- R4 LKD parents, E0–E3, the B0 paper students and the label-swap B0 pairs;
- MNDWI, WI2015 and AWEIsh at their published threshold 0.

**Per set, report:**
- the 16-way joint confusion of teacher, R4 student (each seed) and MNDWI against the reference, per event;
- the share of each model's misses and false water shared by all three;
- the strata of the shared errors: shoreline distance (≤2, 3–10, >10 px), water half-width ≤ 2 px, vegetated (NDVI > 0.4), dark, bright, cloud distance (reference cloud band where available, input cloud test elsewhere), and same-day Sentinel-2-labelled WF scenes versus other label sources;
- mean spectra of shared misses, shared false water and correctly found water.

Reuse novelty_probe_20261012/probe_shared.py and probe_failure.py, extended to every model and the sealed set.

**Ceiling estimates** (reference-based, unattainable): pooled and equal-event water IoU if all shoreline (≤2 px) errors, all cloud-margin errors, or all spectrally land-like misses were removed. Label these ceilings as limits of what a six-band optical method can reach, not as targets.

### 4.2 Operating-point attribution of E3 (CPU, existing models)

For each seed:
- compute E3's predicted water fraction on S11 validation;
- find tau so that E0's predicted water fraction on S11 validation equals it (v10_reference.source_matched_tau on the pooled S11-validation margins);
- evaluate E0 at that tau against E3 on all sets.

If E0 at the matched tau comes within 0.3 pp equal-event water IoU of E3 on the sealed set, attribute E3's effect to the operating point. Do the same for E1 and for every Section 4.3 arm.

### 4.3 Mixed-pixel targets: 2 × 2 CPU training (main)

**Definitions** (v10_reference.py):
- band = pixels within k = 2 px of the teacher hard mask's edge (the V9 band);
- fraction f = v10_reference.unmix_fraction(x, teacher_mask, k=2, radius=7, min_px=8, max_residual=1.0). This is local two-endmember linear unmixing in standardized space. Water endmember: band-free interior teacher water with MNDWI > 0 within a 15 × 15 window. Land endmember: band-free interior teacher land. Clip to [0, 1]. Where f is NaN (unusable fit), keep the teacher label;
- p_water from the stored teacher confidence (p_water = confidence if mask = 1, else 1 − confidence).

Freeze k, radius, min_px and max_residual now; do not tune them. On S11 validation, 85% of band pixels had a usable fit.

**Arms** (identical R4 CE recipe on CPU; only targets inside the band differ):

| Arm | Target inside the band | Loss |
|---|---|---|
| F0 | teacher hard mask, i.e. an E0 replica trained on CPU | soft_focal_loss with t ∈ {0, 1} ≡ focal loss |
| F1 | physics soft: t = f | soft_focal_loss |
| F2 | teacher soft: t = p_water | soft_focal_loss |
| F3 | physics hard: t = 1[f > 0.5] | soft_focal_loss |
| F4 | operating-point control: t = 1[p_water > 0.2], i.e. an E3 replica on CPU | soft_focal_loss |

Outside the band every arm uses the teacher hard mask.

**Recipe and determinism:**
- Same as V7 R4 CE: 7,800 updates, batch 16, AdamW, lr from the V7 recipe, weight decay 0.05, gradient clipping 5, epoch-stepped cosine, gamma = 2, S11-validation selection every 312 updates, the same augmentations, seeds 42/64/91/107.
- Fix torch threads per run (8). Enable torch.use_deterministic_algorithms(True). Each arm/seed pair must have identical initial model hashes and an identical batch order.
- All arms are trained on CPU, including F0 and F4. They are not paired with the GPU-trained V9 arms; V9 arms are descriptive references only.
- 20 runs. Run at most 3 in parallel (24 cores); about 1–2 h wall time in total.

**Target caching:**
- Precompute targets once per tile (float16 t and a band mask) under research_v10_*/targets/.
- Hash the targets. Verify that outside the band all arms equal the teacher mask, and that F0's targets equal the teacher mask everywhere.
- Report per arm: the mean target in the band; the share of band pixels changed versus the teacher (hard arms) or the mean |t − teacher| (soft arms); water→land and land→water mass.

**Evaluation** (deployed threshold, all sets including sealed):
- equal-event and pooled water IoU, precision and recall;
- shoreline (1–2 px) recall and false water, thin-water recall, interior recall;
- calibration inside the reference shoreline band on S11 validation/test: Brier score and 10-bin ECE of p_water against the human label. Soft targets may change calibration more than IoU;
- dry-window alarms and average precision;
- each arm at its source-matched tau (Section 4.2) and with the decode-time rule of Section 4.4.

### 4.4 Decode-time rule with stronger indices (CPU, existing and new models)

**Grid:** index ∈ {MNDWI, WI2015, AWEIsh}, k ∈ {1, 2, 3}, and per-index theta:

| Index | theta values |
|---|---|
| MNDWI | −0.1, 0, 0.1, 0.2 |
| WI2015 | −5, 0, 5, 10 |
| AWEIsh | −0.05, 0, 0.05, 0.1 |

**Selection:** pick the setting with the best mean equal-event water IoU of the four R4 LKD parents on S11 validation (earliest grid entry on ties). Freeze it, then apply it with v10_reference.decode to every model on every set. Report the CPU time per 512 × 512 tile.

**Baselines:** report MNDWI (primary baseline, as in the paper and the reviewer letter), WI2015 and AWEIsh standalone at threshold 0, with average precision, precision at matched recall, dry-window alarms and per-event wins against the students. How the stronger indices are presented in the manuscript is the authors' decision. Compute and archive them regardless.

### 4.5 Clouds (feasibility only)

Check whether solar zenith/azimuth angles are available locally for the WF-v2, STURM and GDACS scenes (scene metadata or retained product XML). Do not download anything.

If they are available:
- implement a geometric cloud-shadow projection (Fmask-style): input cloud test, cloud heights 500–5,000 m in 500 m steps, the shadow mask as the union of projections;
- test, on S11 validation only, whether excluding projected-shadow pixels from MNDWI/WI2015 water makes the cloud-band re-decision non-harmful;
- report it on exposed and sealed sets as a characterization.

If the angles are not available, mark the block UNAVAILABLE with the reason. Clouds stay a documented failure mode.

### 4.6 Compression-induced failure modes (CPU)

**Definitions.** Errors are classified per pixel against two references:
- **inherited:** the teacher is also wrong;
- **compression-induced:** the FP32 R4 parent of the same seed is right and the compressed model is wrong. For R4 precision variants, the reference is the same-seed FP32 R4;
- **other:** everything else.

Report each class per stratum (shoreline ≤2 px, interior >10 px, thin water, dark, bright, vegetated, cloud distance), per event and as dry-window alarms.

**C1. Frontier readout** (no training; existing checkpoints, 4 seeds):
- Rungs R1, R2, R3, R4, A3, A3N, A0.
- Precisions, all with the adopted simulator, the same 16 TRAIN calibration tiles and calibration seed 1729:
  1. FP32;
  2. strict W8A8 on every layer including stem and head (the attention-free rungs can run fully integer, which is relevant to R2.6);
  3. the V8 point: W4A8 body / W8A8 stem / FP32 head;
  4. W4A4 body / W8A8 stem / FP32 head.
- Run every variant on S11 validation/test, STURM and WF test.
- On the sealed set, run only a predeclared subset: rungs {R2, R4} × {FP32, strict W8A8, W4A8} × 4 seeds, plus each control selected in C2. The full grid on the sealed set is too expensive on CPU.
- For quantized variants, also record:
  - per layer and per set/event, the share of activations clipped at the calibrated range;
  - whether events with more clipping lose more (Spearman, descriptive). This tests whether TRAIN-calibrated ranges saturate on out-of-distribution imagery.

**C2. Targeted controls.** Each is selected on S11 validation only, and each is aimed at a named failure:

| Control | Failure targeted | What it does |
|---|---|---|
| (a) operating point | a margin shift | Refit the decision threshold of each compressed variant on S11 validation (Section 4.5 source-selected tau). Measures how much of the compression loss is only a shift. |
| (b) calibration | out-of-distribution clipping | Min-max (adopted) vs 99.99th percentile vs per-channel activation ranges. Calibration set: the adopted 16 tiles vs 64 TRAIN tiles stratified by pool features to include dark, bright and cloudy tiles. Use pool features only, never target labels. |
| (c) layer protection | sensitive layers | Uniform W4A4 vs the stem and head kept at 8-bit. Rank layer sensitivity on S11 validation by single-layer 4-bit substitution, then protect the top two. |
| (d) QAT with distillation | aggressive points that fail the 1 pp source tolerance | For R4 W4A4 and R2 W4A8 only: CPU QAT with a KL term to the same-seed FP32 parent, up to 1,560 updates. It must have a matched FP continuation control (same init, batches, updates, optimizer). If both worsen, do not rescue QAT by changing its schedule. |
| (e) dark-surface hard negatives | the small-rung dark-surface failure | Retrain R2 (418 parameters; minutes on CPU) on the A1 cohort defined in prompt_v9_overnight.md Section 5 (250 pre-event tiles with teacher water < 0.01 and MNDWI water ≥ 0.05, replacing a fixed 250) vs the A0 cohort and the A2 easy-negative control, 4 seeds each. Do not run this for R4, whose dark-surface rate is already low. |

**C3. Strict-integer check** (optional, CPU). Export R4 and R2 at strict W8A8 to ONNX QDQ. Run ONNX Runtime CPU integer kernels on S11 validation/test and compare with the simulator: pixel agreement and per-event IoU difference. Report CPU latency as workstation latency only, never as edge latency.

## 5. Decision rules (declared before training; sealed set only for claims)

Report per-seed differences, the event-bootstrap 95% interval of the seed-mean difference (10,000 resamples, seed 1729) and the across-seed SD separately. The event is the unit. Non-significance is not equivalence.

- **Mixed-pixel physics helps.** Requires F1 − F0 ≥ +0.5 pp and F1 − F2 ≥ +0.5 pp in sealed equal-event water IoU, each with lower bound > 0 and ≥ 3 of 4 seeds positive, plus:
  - no STURM equal-event loss beyond −0.5 pp;
  - better shoreline accuracy or Brier score on S11 test.
- **Soft versus hard.** F1 versus F3, at the same physics.
- **Physics versus teacher.** F1 versus F2 and F3 versus F0.
- **Operating point.** If F4 or F1 is matched within 0.3 pp by F0 at its source-matched tau, attribute the effect to the operating point.
- **Decode-time rule.** Claimed only if the selected setting improves the sealed equal-event water IoU of the deployed student by ≥ +0.5 pp with lower bound > 0 in ≥ 3 of 4 seeds, without a STURM loss beyond −0.5 pp.
- **Shared-failure ceiling.** A characterization, never a method claim.
- **Compression creates specific failure modes.**
  - Allowed if, for a compressed variant, the compression-induced errors concentrate in one or more strata at ≥2× their share of all pixels, on the sealed set and on at least two exposed sets, with consistent sign across ≥3 of 4 seeds. The dark-surface false water of R2/R3 is the predeclared primary hypothesis.
  - Report the size × precision frontier with these strata regardless of outcome.
- **A control fixes a compression failure.**
  - Requires the control to reduce the compression-induced error in its targeted stratum by ≥50% relative (sealed set, seed mean, event-bootstrap interval excluding 0), without lowering equal-event water IoU by more than 0.3 pp, and without raising dry-window alarms by more than 1 percentage point of windows.
  - Controls (a)–(e) are judged separately; there is no post-hoc combination.
  - If (a) alone removes most of a quantization loss, report that loss as an operating-point shift.
- **Strict INT8.** Report the strict W8A8 R4/R2 versus FP32 difference with the 1 pp primary and 2/3 pp secondary source tolerances. Never call it lossless; non-significance is not equivalence.
- **Null results.** If every arm is within ±0.5 pp of F0 on the sealed set, report that the remaining error is dominated by the shared ceiling (Section 4.1) and stop. Do not add arms, change k/radius/theta, or retry seeds after seeing results.

## 6. Verification

Before any training:
- run experiments/revision/v10_reference/test_v10_reference.py and keep it passing. It checks that the soft focal loss equals the existing focal loss on hard labels; that unmixing recovers synthetic 0.7/0.3 fractions within 0.05; that edge distance and decode match edge_v9; that the threshold bisection is correct; and the 2 × 2 block test;
- re-run v10_reference/feasibility_s11val.py and confirm the reported AUROCs within 0.005 (CPU determinism);
- verify sealed-window identity against V9 (hashes) and that the label guard is active;
- do a smoke run: 50 updates per arm on CPU, checking finite loss, identical initial hashes, identical first-batch images across arms, and targets that differ only inside the band.

For compression:
- reproduce V9's R4 W4A8 PTQ state for seeds 42/64 from the same 16 calibration IDs on CPU and compare tensors with research_v9_20261012T235412Z/ptq_states/q_s{42,64}.pt (scales and integer weights identical; report any float tolerance);
- check that strict W8A8 really quantizes every layer, including the head (precision map);
- verify the clipping counters on a synthetic tensor with known saturation;
- verify that QAT and its FP continuation share initial hashes and first batches.

If a GPU becomes free, a one-batch GPU-vs-CPU parity smoke for the simulator is optional, and not a blocker.

After training: checkpoint hashes, selected updates, and source-validation metrics per run. Confirm that nothing was selected on a non-S11-validation set.

## 7. Execution and survival

Use one detached CPU worker (setsid/nohup, stdin closed, absolute interpreter /home/schmalstieg/miniconda3/envs/esi/bin/python, PYTHONPATH including /home/schmalstieg/repos/_deps/efficientvit for B0) with:
- a single-worker lock;
- atomic state writes;
- a 30-second heartbeat;
- per-job receipts;
- one retry for a diagnosed transient technical failure;
- status/pause/stop commands that work from a fresh terminal.

It must not touch the V9 worker, its locks or its ledger. If the V9 worker becomes active on CPU, reduce V10 parallelism so that the combined load stays below 40 cores.

The wall limit is 16 hours from launch; reserve the final 20 minutes for collection. On a deadline or stop, collect what exists and mark the rest DEFERRED.

Priority order (finish each tier before starting the next):
1. freeze, tests and feasibility;
2. Section 4.1 and C1 on the exposed sets (no training);
3. Section 4.3 training (20 runs) and C2 controls (a)–(c) (no training);
4. C2 (d) and (e) training;
5. the decode-time rule (Section 4.4) and the operating-point attribution (Section 4.2);
6. the single final sealed evaluation job for every frozen model and variant in the predeclared sealed subset;
7. C3 and clouds (Section 4.5) if time remains.

## 8. Deliverables

All in research_v10_*:
- frozen_protocol_v10.md and its hash;
- the code snapshot, target cache manifest and test/feasibility receipts;
- shared_failures.csv (joint confusions and strata, per set/model/event), ceilings.csv and shared_spectra.csv;
- operating_point_attribution.csv;
- mixed_pixel_results.csv, mixed_pixel_by_event.csv, target_statistics.csv and calibration_edge.csv;
- decode_rule_selection.json, decode_rule_results.csv and index_baselines.csv;
- clouds_feasibility.json;
- compression_frontier.csv (rung × precision × seed × set × stratum, with inherited / compression-induced / other errors), activation_clipping.csv, compression_controls.csv, qat_pairs.csv and strict_int8_ort.csv;
- negative_results.md;
- report_v10.md in plain language, with an experiment checklist (arm, seeds, status, evidence path) and a short "what this means for the reviewer answers" section:
  - R1.1/R2.1: specific contribution or characterization;
  - R2.5: rule baselines, false alarms, shared failures;
  - R1.6: dry windows and clouds;
  - R2.2: label swap.

Separate observation from interpretation. Do not count deferred or unavailable blocks as negative evidence.

## 9. Prior art (starting points, not a novelty clearance)

- Feyisa et al. 2014, Remote Sensing of Environment (AWEI).
- Fisher, Flood, Danaher 2016, Remote Sensing of Environment 175 (WI2015).
- Xu 2006, IJRS (MNDWI).
- Du et al. 2016, Remote Sensing 8, 354 (10 m MNDWI by SWIR sharpening).
- Wang et al. 2018, Remote Sensing 10, 1643 (MuWI).
- Sub-pixel water fraction mapping with Sentinel-2:
  - Wang et al. 2023, GIScience & Remote Sensing;
  - Water 15(8), 1446, 2023.
- Pseudo-label refinement: SegFix (Yuan et al., ECCV 2020); Kolesnikov & Lampert, ECCV 2016.
- Fmask cloud-shadow projection (Zhu & Woodcock 2012, Remote Sensing of Environment).
- Quantization:
  - Nagel et al., A white paper on neural network quantization, 2021 (https://arxiv.org/abs/2106.08295): calibration, per-channel ranges, layer sensitivity;
  - Nagel et al., Data-free quantization through weight equalization and bias correction, ICCV 2019 (https://arxiv.org/abs/1906.04721);
  - Esser et al., Learned step size quantization, ICLR 2020 (https://arxiv.org/abs/1902.08153);
  - Kim et al., QKD: quantization-aware knowledge distillation, 2019 (https://arxiv.org/abs/1911.12491).

The compression claim is also narrow: below about 1,000 parameters or at 4 bits, a distilled flood segmenter keeps its pooled accuracy but fails on specific, nameable pixel types (predeclared: dark surfaces). Pooled metrics hide this. A targeted, source-selected control may remove it. This is not a new quantization method.

The candidate claim is narrow: at shorelines, a physically unmixed water fraction may be a better distillation target than the teacher's hard or soft edge. It is tested against both, at matched operating points, on a sealed set. No first-ever claim and no guaranteed improvement.
