# V9 Codex prompt: operating-point control, hard negatives and the remaining reviewer controls

Prepared 12 October 2026, Europe/Berlin. Revised the same evening by Claude (Opus) after a CPU-only probe (Section 2.5). The original draft is preserved as prompts/prompt_v9_overnight.orig_20261012.md.
Purpose: implement and run a bounded overnight follow-up to the audited V8 results, incorporating Opus's reviewer wrap-up and probe supplied on 12 October.
This is an experimental instruction, not a claim that the proposed method is novel or successful.

## 1. Mission and constraints

Work in the existing FloodDistill research branch and reuse its loaders, graphs, evaluators, quantizers, caches, queue, and reporting code. Inspect AGENTS.md and the working tree first. Preserve user changes and immutable historical runs. Avoid a new framework.

Finish high-value reviewer controls and investigate two linked research questions about the precision–recall operating point of the established tiny student:

1. **Operating-point control (main; CPU-first).** On unseen floods the distilled students are precise but miss water, and the source-optimal decision threshold is too conservative there. Calibrate the threshold on an independent, never-used labelled flood archive (WorldFloods-v2 train/val events), in one of three ways:
   - global;
   - risk-controlled;
   - scene-adaptive from label-free scene statistics.

   Does this recover out-of-distribution recall and water IoU on sealed events at a declared dry-tile false-alarm cost? Does it transfer to the exposed benchmarks?
2. **Hard-negative distillation (secondary; GPU).** Re-introduce pre-event GDACS tiles where the teacher sees no water but MNDWI does; these are dry scenes the curation filter removed. Does this move the precision–recall frontier itself, so that recall costs fewer false alarms?

Compression is reduced to the full unchanged-PTQ readout plus an operating-point check (Section 6).

Default to S2-only inputs. Use the existing Prithvi teacher, R4 student and existing MNDWI implementation. An already installed, independently human-supervised S2 UNet is admitted only for the controlled teacher comparison below. No S1 requirement, new foundation-model fine-tuning, NAS, new dataset acquisition, driver changes, paid service, message to another person, or manuscript submission. No subagents. The WorldFloods-v2 train/val splits already on local disk (/data1/schmalstieg/flooddistill/data/WorldFloodsv2) are existing public data, not a new acquisition; Section 4.1 governs their use.

Implement and validate the finite experiment, start the detached worker on the authorized host, and verify it is actually running. Do not stop after writing a plan when the repository/data/device are available. Conversely, do not claim a launch when the current environment only contains reports.

## 2. Ground truth from V8 and historical context

Locate and read the audited V8 report and original run directory. The report identifies:
- /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v8_20261012T123202Z
- analysis_20261012/audit_analysis.py and audited CSVs inside that run.
- Original host environment: /home/schmalstieg/miniconda3/envs/esi/bin/python.

These are observed historical paths; verify rather than assuming they exist on this host. The supplied PDF's SHA256 is:
1d05fef4a19b25dc41f3eddafbf974f0bb0d33a25f50ad0ee180caa35eac62c4

Read reviewer_wrapup_20261012.md, the actual reviewer letter and test.txt if available, earlier revision closure tables, V6/V7 audits, and newer live state. The supplied wrap-up identifies paper/review.docx on the research host and newly recovered raw Jetson logs. V8's statement that its agent could not locate the letter is an input-availability limitation, not proof the letter does not exist. Its 11 figures and verification script are referenced host artifacts, not automatically present merely because the Markdown is attached.

Known facts to reconcile against receipts:
- V8 completed eight jobs and 180 adaptation rows, with no failed jobs. This was a three-event pilot, not 180 independent experiments.
- Target pilot: EMSR264, EMSR286, EMSR333; two support draws and parents 42/64; 92 query tiles. Keep these episodes as diagnostic data.
- R4 has 12,230 parameters, frozen extractor/BN, a 16-feature affine two-class head, and 17 effective margin coordinates.
- Relative to frozen FP: F2_20 ordinary soft-KD +0.079 pp query mIoU; uniform ridge -0.378; confidence/stability-weighted ridge -0.552.
- One-chip supervised ridge -1.771 pp; teacher+label ridge -1.238; supervised SGD20 -0.353. Supervised hyperparameters were inherited from teacher-only tuning, so this was not a strong independent supervised baseline.
- On six support chips, 8,741 of 9,648 examined confident/stable reference-water pixels were incorrectly predicted as land. This is 90.60% conditional water error in that diagnostic subset, not overall teacher error.
- Some support samples had no water. Source-selected lambda=0.1 was at the grid boundary. All teacher-only ridge steps had eta=1.
- Uniform ridge and soft-KD differed in objective, not merely solver. Frozen-feature integrity and solve/folding checks passed; repeating those entire audits is not the next scientific question.
- WF equal-event water IoU: R4 47.95, teacher 50.54, their unattainable oracle 54.00. Teacher/MNDWI oracle: 61.40 equal-event and 73.86 pooled. Do not invent a three-expert oracle from two pairwise tables.
- Quantized parent: W4A8 body, W8A8 stem, FP32 head, fake quantization. Mean query mIoU difference from FP -0.016 pp, but event differences -1.240/-0.148/+1.339 pp. Q3 failed against simpler Q controls.
- Inner fit medians: ridge 0.731 ms, SGD20 6.267 ms. Partial component sums 58.1 versus 63.9 ms omit true cold-start work.
- FP16 artifact 156,695 bytes versus FP32 183,255; almost unchanged source score but slower observed workstation timing. No native INT4 deployment result.
- Corrected OFA curves show some budget-dependent update behavior, not universal speedup or measured time-to-quality. Parser repair is complete.
- V7 C3 continuation failed; even ordinary continuation degraded the parent. Earlier low-bit teacher-excess allocation did not beat a simple structural allocation.
- CE in the current GDACS runs uses teacher hard masks. The fixed teacher inherits 252 labelled source-training scenes.
- Existing external benchmarks are research-exposed. Re-freezing them does not make them untouched.

Retire unchanged F4 and Q3 as candidate recipes. Reuse them only as historical controls. Do not repeat the same +2% stability weighting in search of a favorable seed.

Also retire, before execution, the support-aware protected head adaptation of the original V9 draft: its Sections 4–5, the causal solver replay and the H0–H2 × P0–P2 × D0–D1 matrix. Reasons from receipts:
1. V8's teacher_headroom.csv shows that teacher-induced harm exceeds teacher corrections on STURM and WF for both parents (G < H). Teacher-referenced head updates are therefore bounded by no-update on these targets.
2. The probe (Section 2.5) finds that most out-of-distribution misses are reference water with land-like Sentinel-2 spectra. The correctable part behaves like a one-dimensional threshold shift. Even a perfectly labelled support chip that sets only the threshold gains +2.4 (STURM) / +4.1 (WF) pp on average, with −27 to −36 pp worst cases. A 17-coordinate refit from teacher targets therefore has little to gain.
3. No reviewer asked for test-time adaptation.

Keep V8's adaptation rows as a documented negative result, and record the disagreement-coverage idea as future work in negative_results.md. The teacher/rule disagreement idea is reused at training time in Section 5. Ordinary head KD, teacher/rule complementarity and tiny-model compression remain open topics; this run does not pursue few-shot head adaptation.

### 2.1 Incorporate the wrap-up critically, using CPU evidence first

Keep its [V] re-derived, [R] inherited and [J] judgement distinctions. Verify accessible raw files before adopting a newly recovered number. Produce revision_corrections.csv with claim, original value, proposed correction, protocol/graph/seed, evidence path/hash and unresolved qualification. Do not rewrite the manuscript automatically.

Prioritize the teacher row and dependent comparisons; FP32 mean GPU rail power (reported correction 3.95 to 3.25 W); pre-filter effects by selector; crop accounting (200 chips, 73.6% removed water, 21 dry); OFA training-head count (11,368); teacher-referenced versus true false alarms; and the newly recovered per-seed Jetson chain. Reuse the wrap-up's CPU verification script if present, inspect it before running, and write new outputs rather than modifying earlier evidence.

Resolve these overstatements instead of copying them:
- Four-bit "lossless" is unsupported by the matched -0.25 pp interval or V8 event cancellations; non-significance is not equivalence.
- Better LR supports under-optimization as a contributor to the apparent activation cost; it does not isolate an intrinsic activation-only effect across different training/initialization histories.
- OFA has no demonstrated universal/end-to-end speed advantage. Preserve N100's bounded update evidence, censored thresholds, whole-wrapper timing, and the audited full-budget masking distinction.
- Do not mix legacy WorldFloods teacher values with corrected native-geometry values in one gap calculation.
- Do not use "effect smaller than seed SD" as a significance rule; use the paired design and its uncertainty.
- Do not assume every external protocol removes permanent water; verify the specific full-water/flood-only mask definition.

Use the author's existing STURM clarification: STURM did not determine the choice of OFA or EfficientViT; it was used to investigate additional teacher-derived/OOD criteria, including saliency-related ideas, that gave null results. Mark this as author-reported history, reconcile retained code/comments, and do not extrapolate it to WorldFloods or every selection threshold. Preserve development exposure even for null exploratory tests. Leave only genuinely missing details as author-input items; do not ask again for the clarification already supplied.

### 2.2 Required reviewer control: identical images, different labels

This addresses R2.2 directly. Use the same 252 S11 training images, split, valid support, footprint, augmentations, initialization per seed, optimizer, stopping/budget and evaluation code. Compare human hard masks with frozen Prithvi hard masks. The only difference within each paired contrast is label source.

Primary: four paired seeds of EfficientViT-B0 with true cross-entropy in both arms. Reuse a human-labelled anchor only if its entire recipe/input lineage matches; otherwise train the pair. Do not compare true CE with focal loss or change clipping only for one arm. Apply the same reference-valid mask to both target sources, without learning from validation/test labels. Make new teacher masks immutable; preserve original caches.

Optional extension after the primary pairs: cross label source with the same added soft-KD term, using L=CE(hard target)+0.5*T^2*KL(teacher||student), T=4, for both human-hard and teacher-hard arms. Keep the CE coefficient fixed across presence/absence of KD. This isolates the hard-label source within an objective and provides a controlled soft-target contrast. Existing anchors with another objective are not interchangeable. No extra OFA retraining is needed.

These are labels on images the task-adapted teacher itself saw during source training. The control isolates label-source effects on that source dataset; it does not prove pseudo-label superiority on unseen imagery, remove all original S11/GDACS confounds, or establish label-free learning.

### 2.3 Conditional reviewer control: is the foundation-model teacher necessary?

Inventory the existing UNet-ResNet50 checkpoint suggested by Opus. Verify that it was trained on the 252 human-labelled S11 training images, not Prithvi pseudo-labels or an ensemble trained on those pseudo-labels. Verify six-band handling, preprocessing, output convention, task head and selected checkpoint. If that independent provenance is absent, mark the comparison unavailable and continue; do not spend the budget creating a new teacher.

If available, use this S2-only teacher on the same fixed GDACS N2500 cohort. Do not re-filter/reselect the cohort for each teacher; disclose that the original selection was based on Prithvi, so this is conditional on that cohort. Evaluate teacher/student/rule joint errors and teacher task quality. Then train four EfficientViT-B0 logit-KD student seeds against the existing compatible Prithvi-taught anchors, with identical student recipe and budgets and teacher-specific hard/soft targets. If compatible anchors cannot be established, admit complete matched pairs or defer the comparison.

Admission depends on provenance and cost, not whether the alternative teacher wins its screen. Report poorer, equal or complementary results. This tests the specific Prithvi-versus-supervised-UNet comparison, not the universal necessity of foundation models. Teacher pretraining and adaptation costs remain different and must be disclosed. A fixed teacher plus four student seeds is not four independent teacher trainings.

Scheduling: the provenance inventory and the inference-only teacher-quality readout may run whenever convenient. The four student trainings run only from verified freed budget after the core blocks of Section 3.

### 2.4 Dry-scene evidence without invented verification

Locate the six reference-labelled dry S11 chips and the 11 unreviewed candidate windows described by Opus. A cheap readout of the already verified six chips for compatible parents is useful, with any-water/0.1%/1% area thresholds, but it remains a tiny exposed diagnostic. The reported 43% false-water case belongs to one historical KD checkpoint, not automatically R4 or all KD students.

Run the prepared 11-window evaluation only if independent human verification records actually exist. Otherwise keep it blocked and provide the existing review template/contact-sheet path. Never fill "verified dry" from teacher predictions, MNDWI or the need to finish overnight. Do not call teacher-referenced disagreement a true dry false-alarm rate.

Independent dry-tile evidence already exists and must be reported. A reference-dry window is a valid window with zero flood-water reference pixels; permanent water is ignored by the WF protocol. Such windows exist in WF test (216), STURM (8 chips) and the WorldFloods-v2 events of Section 4.1. Report the windows flagged at ≥0.1/1/5% predicted water area for:
- every student seed;
- the teacher;
- MNDWI>0;
- every operating-point rule.

Describe them as dry land inside flood scenes with EMS/STURM references, not as ordinary unattended imagery.

### 2.5 Exploratory probe after the wrap-up (Claude, 12 October; research-exposed)

**Folder:** experiments/revision/novelty_probe_20261012/
- Scripts: probe_cpu.py, probe_hist.py, probe_analysis.py, probe_oprules.py, probe_fewshot.py, probe_predictability.py, run_all.sh.
- Reports: *_report.txt.
- CPU-only and read-only; about ten minutes.

**Inputs:**
- the four selected R4 LKD parents fl_R4_sel_s{42,64,91,107} (hashes in capacity_v7 reuse_models.json);
- S11 validation/test, STURM (P3 finite support) and WF test (P3 native windows);
- the V7 reference cache (teacher hard masks and MNDWI bits). Every probe MNDWI bit matched the cache.

CPU predictions reproduce V7 GPU per-image counts up to 1–2 near-boundary pixels per image (TF32). The probe did not read WorldFloods-v2 train or val. Verify these statements, and treat the numbers below as [R] until you re-derive the ones you use.

Observations (four-seed means; equal-event water IoU unless stated):
- **Deployed threshold** (water-minus-land margin > 0):

  | Set | Water IoU | Precision | Recall |
  |---|---|---|---|
  | S11 val | 77.27 | | |
  | S11 test | 80.26 | | |
  | STURM | 37.83 | 60.4 | 50.3 |
  | WF test | 47.59 | 84.9 | 51.9 |

- **Event-optimal threshold** chosen with reference labels (unattainable): median 0 on S11, −2.0 on STURM, −1.5 on WF (seed-pooled event optimum). Gains, optimised per seed and then averaged: +1.3/+1.6 (S11 val/test), +7.6 (STURM), +9.3 (WF).
- **Fixed threshold −1:** STURM +1.43±0.38 (20 of 29 events better), WF +3.93±0.30 (10 of 11), S11 test −1.42, S11 val −2.01.
- **Fixed threshold −0.5** (the best non-zero value on S11 val, weakly supported): STURM +0.78, WF +2.31, S11 test +0.20.
- **Cost on WF reference-dry windows** (216 windows with no flood-water reference pixels). Windows with ≥1% predicted water:
  - student 9.0, teacher 5, MNDWI>0 140;
  - student at threshold −1: 23.8.
- **Label-free per-scene thresholds** with S11-val-selected settings:
  - EM prior correction: WF +1.79, STURM −0.27, S11 test −0.56;
  - MNDWI-anchored calibration: WF +1.38, STURM +0.69, S11 test −0.16;
  - Otsu split of the margin histogram: never triggered under its source-selected guard.

  The EM and MNDWI-anchored rules raised WF dry-window alarms to 16.8 and 27.8 (versus 9.0); the fixed −0.5 shift raised them to 15.0.
- **Physics fusion at decode time** (MNDWI union, veto, seeded hysteresis growth; S11-val-selected): at most +0.9 on WF, ≤0 on STURM.
- **One labelled chip setting only the threshold** (dense labels): STURM +2.44 [+0.05, +5.27], WF +4.12 [−0.55, +9.95]. Worst support choice −36 (STURM) / −27 (WF). S11 val −2.8, S11 test −3.3. Using 10–50 labelled clicks per class was worse.
- **Predicting the threshold:** label-free scene statistics (predicted water fraction, MNDWI-positive fraction, near-boundary mass) correlate with the event-optimal threshold at Spearman ≈ +0.45 over the 40 STURM+WF events.
- **Threshold-free comparison:** student average precision exceeds MNDWI on every set (pooled pixels: S11 test 95.3 vs 91.2, STURM 81.2 vs 76.3, WF 88.3 vs 86.0). At MNDWI>0's recall, student precision is 87.2 vs 77.8 on WF and 71.1 vs 71.3 on STURM.
- **What the student misses:** of the reference water the seed-42 student misses, 73% (WF) and 63% (STURM) has MNDWI ≤ 0 and land-like NIR/SWIR reflectance. The misses MNDWI can detect are spectrally close to MNDWI false positives that the student correctly rejects. On same-day Sentinel-2-labelled WF scenes, water-weighted recall is still 72.4 (student) versus 82.6 (MNDWI), so not every miss is a label/time mismatch.

**Exposure:** these rule families were found after reading research-exposed outcomes. Improvements on S11 test, STURM or WF test from them are not confirmatory. Confirmation requires the sealed WorldFloods-v2 events of Section 4.1.

## 3. Budget, schedule and job admission

V8 cumulative spend was 25.969689 GPU-command hours under the existing 30-hour cap, leaving 4.030311 then. Recompute current spend and reservations before doing GPU work.

Incremental cap = min(4.0 hours, max(0, 30 - verified spend - other reservations)).
This prompt does not reset that cumulative cap. Count failed attempts, teacher inference, smokes, adaptation, quantization and evaluation consistently. CPU work is separately timed.

Use two explicit clocks: allow up to eight hours from worker launch to wait for the first GPU command, then allow up to eight hours from that first GPU command (including a smoke) to finish the finite run. The absolute wall limit is sixteen hours from worker launch. Record launch, wait, execution and absolute deadlines in UTC and Europe/Berlin; persist them so a restart never resets them. Waiting after the first GPU command counts against the execution window. Reserve the final ten minutes of the applicable execution/absolute window for collection, and run CPU collection immediately if the initial wait limit expires. Idle polling and CPU-only work consume wall time, not GPU-command hours. The host must remain awake. Do not reopen a finished V8 manifest; create a timestamped V9 directory. If another worker is active, respect its locks and reservations and wait automatically under Section 9.

Suggested admission ceilings if four hours remain:
- Same-image label-source control (Section 2.2, primary pairs): 0.65 h.
- Hard-negative distillation (Section 5; eight R4 trainings plus any GPU-side evaluation): 0.90 h.
- Teacher soft outputs for operating curves (Section 4.4; diagnostic): 0.50 h.
- EfficientViT-B0 inference on WorldFloods-v2 windows (Section 4.2): 0.15 h.
- Full unchanged-PTQ readout, if not done on CPU (Section 6): 0.20 h.
- Protected final evaluation/collection and bounded retry reserve: 0.60 h.
- Conditional on verified freed budget only: provenance-qualified S2 conventional-teacher comparison (1.50 h), label-swap soft-target extension, R4 FP/QAT pairs.

CPU work starts immediately and does not wait for a GPU. It covers:
- revision corrections;
- the WorldFloods-v2 inventory, overlap screen and split freezing;
- R4 inference on every set;
- all operating-point fitting on D;
- dry-tile readouts;
- the PTQ readout of the tiny R4 graph where the simulator runs on CPU;
- collection.

On 12 October both GPUs of this host were fully occupied by another user's jobs. A long WAITING_GPU phase is expected and must not stall CPU work.

GPU priority order:
1. primary label-swap pairs;
2. the hard-negative block;
3. teacher soft outputs;
4. B0 inference;
5. conditional items.

If the alternate teacher is unavailable, its conditional slot may fund the label-swap extension or complete FP/QAT pairs; this fallback is predeclared and cannot depend on external accuracy. Measure smoke throughput before admitting work. V7 receipts show about 6 GPU minutes per 7,800-update R4 CE run on the RTX 4090. Do not fit an optimistic schedule by dropping controls or exceeding the cap.

These are maximums, not targets. Stop when the frozen work is complete. Preserve paired blocks and final collection before adding optional jobs. If the cumulative budget is exhausted, finish CPU analysis, code and a deferred manifest with exact blockers. If capacity is only temporarily reserved by another active worker, wait and reconcile its verified ledger at the next ten-minute check; never cancel its reservations or treat unverified spend as zero.

All conditional branches and criteria must be in the frozen manifest before target scores are read. Source (S11 validation) and development (WorldFloods-v2 D) results may select settings. Sealed (C) and exposed-benchmark outcomes never alter admission, hyperparameters or cohort selection.

## 4. Main question: operating-point control on an untouched flood archive

Working description: archive-calibrated operating points. The student is not retrained; only the decision threshold on its water-minus-land margin changes. On board this costs one stored number (global rule) or a few arithmetic operations per scene (scene-adaptive rule). This is a proposed combination of established calibration and risk-control ideas for an on-board distilled segmenter, not a first-in-literature claim.

### 4.1 Data roles and the untouched archive

Inventory WorldFloods-v2 train (475 scenes, 116 events) and val (16 scenes, 14 events) from dataset_metadata.csv. Before reading any of their labels:
- Search code, manifests, logs and receipts of every earlier revision run and verify that no job opened WorldFloods-v2 train/val label files. Events with any such access are exposed and leave the confirmation set.
- Screen overlap and exclude scenes that intersect, within ±3 days, any GDACS pool tile (all 23,720 pool tiles, not only N2500), any S11 hand-labelled chip (Bolivia included) or any WF test scene. Reuse the prompt5 overlap tooling. If a pool tile has no stored footprint, match by MGRS tile (parsed from the pool scene string, derived from WF-v2 CRS/bounds) and date. Record every exclusion and its reason.
- The event unit is the event code (EMS code, or the UNOSAT/GLOFIMR identifier). Split deterministically by the SHA256 of the event code:
  - D (development/calibration): train events whose hash has an even first byte;
  - C (sealed confirmation): the remaining train events plus every val event.
  No event spans D and C. Write splits_wf2.json with scene lists, counts and hashes, and commit its SHA256 in the frozen manifest.
- C labels are opened exactly once, by the final confirmatory job, after every rule, coefficient and cohort is frozen. Log opened label paths through a small wrapper so the audit can show this.
- Protocol: identical to WF test (P3 native 512-pixel windows, zero padding invalid, permanent water ignored, cloud/invalid pixels excluded). Record each scene's label-source satellite and its label/Sentinel-2 date lag. Report the same-day Sentinel-2-labelled subgroup separately as a secondary result.

| Data | Role |
|---|---|
| S11 train | teacher training (untouched) |
| S11 validation | source selection |
| D | out-of-distribution calibration |
| C | confirmation |
| S11 test, STURM, WF test | exposed transfer readouts |

### 4.2 Models and scores

- **Primary:** the four selected R4 LKD parents of Section 2.5.
- **Secondary:**
  - V7 R4 CE (R4_CE_s{42,64,91,107});
  - the paper's EfficientViT-B0 historical OFA and matched LKD parents (the O4_* checkpoints listed in prompt4_report_20261007/cross_cohort_common_protocol_spec.json). Run B0 on the GPU when admitted. Otherwise run it on CPU for the exposed sets and for a deterministic subsample of D/C scenes declared in the manifest.
- **Index baselines:** MNDWI with the existing definition, and NDWI (green−NIR)/(green+NIR) as a second index. Every index gets the same calibration opportunity as the students (fair comparison).

For every window, model and seed, store the compact histogram from probe_hist.py: margin bin × MNDWI bin × reference class, keeping all imaged pixels for label-free statistics. Reuse that code if it passes your checks. Do not store full-resolution predictions beyond what scoring needs.

### 4.3 Operating-point rules (fitted on D only, frozen before C)

The threshold tau applies to the water-minus-land margin; tau = 0 is the deployed model.
- R0: deployed, tau = 0.
- R1: source-selected global tau from S11 validation, on a grid from −3 to +1 in steps of 0.5 (0 included).
- R2: archive-calibrated global tau_D that maximizes D equal-event water IoU, on the same grid.
- R3: risk-controlled global tau.
  - Per-event losses: L_miss(tau) = 1 − water recall; L_fa(tau) = fraction of the event's reference-dry windows with ≥1% predicted water area.
  - Predeclared levels: alpha_miss ∈ {0.30, 0.40}, alpha_fa = 0.05, delta = 0.10.
  - Method: conformal risk control for each single loss; Learn-then-Test with fixed-sequence testing from conservative to liberal tau for the joint constraint.
  - Report the feasible set, or report that none exists. The guarantee is stated only for exchangeable events (D→C within the archive), never for STURM.
- R4: scene-adaptive threshold, where a scene is one WorldFloods-v2 scene (flood map):
  - tau(scene) = clamp(b0 + b1·logit(f1) + b2·f2 + b3·f3, −4, 2);
  - f1 = predicted water fraction at tau = 0; f2 = fraction of imaged pixels with margin in [−2, 0); f3 = MNDWI-positive fraction;
  - fit by event-weighted ridge regression to the D event-optimal tau, with lambda chosen by leave-one-event-out within D;
  - these three features are frozen now; add none after seeing results.
- R5: the probe's label-free rules (EM prior correction, MNDWI-anchored calibration), with settings frozen from S11 validation.
- Oracle: the best per-event tau chosen with labels (unattainable diagnostic bound).

Fit each rule separately for every model and seed, and record every grid value and coefficient. No rule reads C or exposed-benchmark labels for fitting.

### 4.4 Teacher inheritance and threshold-free comparison

When the GPU is admitted, compute Prithvi teacher soft outputs on S11 validation/test, STURM, WF test and D. Compute them on C only inside the final confirmatory job. Report:
- teacher and student operating curves;
- event-level and pooled average precision;
- precision at matched recall against MNDWI and NDWI;
- for each event, the distance between the teacher's and the student's event-optimal thresholds. This tests whether the conservative operating point is inherited from the teacher.

If no GPU becomes available, use the cached teacher hard masks as single operating points and mark the curves DEFERRED_BUSY.

### 4.5 Outputs

- splits_wf2.json and wf2_exclusions.csv.
- operating_point_rules.csv: rule, parameters, fitted-on set, model, seed.
- op_results_by_event.csv and op_summary.csv, covering D, C, S11 validation/test, STURM and WF test. Report pooled and equal-event water IoU, precision, recall, F1, mIoU, area bias (predicted/reference water area − 1) and dry-window alarms at 0.1/1/5% area.
- risk_control.csv: promised versus achieved risk levels.
- threshold_free_ap.csv.
- missed_water_spectra.csv.

## 5. Secondary question: hard-negative distillation from teacher/rule disagreement

Hypothesis: the curation filter removed dry scenes, including the dark or shadowed surfaces on which MNDWI fires. Replaying such tiles as negatives may suppress MNDWI-type false positives, so a more liberal operating point (Section 4) would cost fewer false alarms. This reuses the teacher/rule disagreement idea for training-time negative mining. Hard-negative mining is established; the specific test is whether disagreement-mined negatives beat easy negatives at a fixed budget.

Cohorts (fixed N = 2,500, same recipe; only membership changes):
- **A0:** the restored original N2500 cohort. Reuse V7 R4_CE_s{42,64,91,107} only if code, initial model hash, batch order and recipe lineage are verified identical; otherwise retrain A0 alongside A1/A2.
- **Removed set:** the same deterministic 250 cohort tiles leave both A1 and A2. They are the last 250 in the stored FPS order; if no order is stored, the 250 with the smallest SHA256 of the filename. Freeze this rule before training.
- **A1 (hard negatives):** the removed set is replaced by 250 tiles chosen by the original FPS selector, in the same feature space, from pool tiles with is_flood_scene = 0, teacher water fraction < 0.01 and MNDWI water fraction ≥ 0.05. The 12 October pool_features.csv lists 671 such tiles.
- **A2 (easy negatives):** the removed set is replaced by FPS-selected pre-event tiles with teacher water fraction < 0.01 and MNDWI water fraction < 0.01.

Training:
- Labels: the cached teacher hard masks, unchanged.
- Recipe: as V7 R4 CE. Focal loss on cached hard labels, 7,800 updates, batch 16, AdamW, weight decay 0.05, gradient clipping 5, epoch-stepped cosine schedule, S11-validation selection every 312 updates. Same seeds, initial weights and augmentations.
- Runs: 4 seeds × (A1, A2) = 8 runs, plus A0 if it is not reusable.
- Exclude candidates that overlap S11 validation/test, STURM, WF test or WorldFloods-v2 C.
- Keep at least 200 unused pre-event pool tiles as a held-out set for teacher-referenced false alarms.

Evaluate every arm exactly as in Section 4: refit R1–R4 on S11 validation and D, read C once, then the exposed sets. Report:
- average precision;
- dry-window alarms;
- recall at a matched alarm rate, and alarm rate at matched recall;
- teacher-referenced alarms on the held-out pre-event tiles.

Run the block even if the effect is null; Section 7 governs the claims.

## 6. Compression branch: full PTQ readout and operating-point check

Reuse R4 FP parents for all available four seeds. Make a full source-validation and full compatible S11-test/STURM/WF readout for the unchanged V8 PTQ precision point; the original three-event cancellation is insufficient. Reuse counts where identities match. The tiny R4 graph may run the adopted fake-quant simulator on CPU, provided parity with the GPU path is shown on a smoke batch. Also refit the R1/R2 thresholds of Section 4 for each quantized parent, on S11 validation and D. Report how much of each quantized-minus-FP event difference disappears after refitting; this tests whether the quantization loss is mainly an operating-point shift.

Report pooled and equal-event water IoU/mIoU, precision/recall, every event, and source loss tolerances 1 pp primary, 2/3 pp secondary relative to each matched FP parent. These are exploratory declared tolerances, not a clinical/operational requirement or proof of equivalence. Do not pool training-seed and event uncertainty.

Build a fixed-graph size frontier from existing compatible R2/R3/R4/A3/A0 anchors. Do not retrain all graphs or assume prior early-stopped runs completed the same updates. Mark incomparable histories.

Conditional on verified freed budget after the core blocks, run a conventional R4 QAT pilot with the established adopted integer-arithmetic fake-quant simulator and its verified STE path:
- Use existing competent QAT/LSQ code if verified for this graph; do not claim faithful LSQ if it is not implemented.
- Keep W4A8 body/W8A8 stem/FP head as the first point. Do not add W4A4 until this baseline is understood.
- Run parents 42/64 with exactly matched floating-point continuation and QAT continuation: same initialization, batches, valid support, duration, optimizer schedule and clipping.
- Use an existing compatible tuned recipe; if absent, admit a small source-only LR pilot rather than copying a known weak recipe. Protect the complete FP/QAT control pairs. This training is conditional on the available compression reservation or freed alternate-teacher budget; the full unmodified PTQ readout comes first.
- Default maximum 1,560 updates per continuation. Preflight timings determine admission; no incomplete unmatched pair should become a positive result.
- If either continuation fails technically, diagnose once and preserve failure receipts. If both scientifically worsen, do not rescue only QAT by changing its schedule.

Do not repeat the V8 quantization-consistency penalty; it did not beat simple controls. No adaptation is applied to quantized parents in V9.

Real INT8/INT4 deployment remains conditional on an already working compatible backend. No driver repair or new hardware dependency tonight. Fake-quant accuracy, theoretical packed weights, serialized engine bytes and measured kernel speed must have separate columns. Count FP heads, scales and metadata.

Repeat FP32/FP16 timings in three fresh process sessions only if affordable. Record flags and dtypes at the actual timing boundary and actual accuracy boundary. Do not compare new latency to V7's different measurement route or call workstation measurements edge results.

## 7. Metrics and decision rules

**Primary outcome for Section 4:** the per-model-seed change in equal-event water IoU on C versus R0. Report the event-bootstrap 95% interval of the seed-mean difference and the across-seed SD as separate quantities.

**Co-primary constraint:** the change in C dry-window alarm rate (≥1% area).

**Secondary outcomes:** recall, precision, F1, mIoU, pooled values, area bias, lower-tail events, the same-day Sentinel-2-label subgroup, average precision and transfer to the exposed sets.

Declared decision rules (these license claims; they are not significance tests or operational guarantees):
- **Archive calibration improves unseen-event flood extent.** Allowed only if R2 or R3 meets all of these on C:
  - equal-event water IoU gain ≥ +1.0 pp with an event-bootstrap lower bound > 0;
  - a positive gain for at least 3 of 4 R4 seeds;
  - an alarm-rate change within alpha_fa, or the increase reported as its explicit price.
  If R1 (source-selected) matches R2, report that the archive was not needed.
- **Scene-adaptivity.** Requires R4 − R2 ≥ +0.5 pp on C with a lower bound > 0. Otherwise attribute any gain to global calibration.
- **Risk control.** Claims are limited to D→C. Report promised versus achieved levels for every alpha. For STURM and WF test, report realized values without a guarantee.
- **No rule improves C.** Report the inherited conservative operating point as a characterization, not as a method.
- **Hard negatives move the frontier.** Requires A1 to beat both A0 and A2 on C average precision (seed means; event-bootstrap lower bound > 0) and on recall at a matched alarm rate. If A2 matches A1, attribute any gain to adding negatives, not to disagreement mining. If A1 lowers recall at tau = 0, report a precision–recall trade.
- **Exposed sets.** S11 test, STURM and WF test results are descriptive transfer evidence. They never select, gate or rescue a rule.
- **Earlier rules still apply:**
  - the event is the transfer unit;
  - do not pool seed and event uncertainty;
  - non-significance is not equivalence;
  - make no "lossless" low-bit claim;
  - qualify mean improvements that hide large event losses.

Stopping interpretations:
- R1 ≈ R2: source calibration suffices; the archive adds nothing.
- R2 ≈ R4: the scene statistics carry no information beyond a global shift.
- R3 is infeasible at the declared alphas: one threshold cannot meet both the recall and the alarm target. Report this as a result.
- A2 ≈ A1: negatives, not disagreement, explain any change.
- Rules improve the exposed sets but not C: treat the exposed-set gains as selection artefacts.

## 8. Minimal verification and reproducibility

Reuse V7/V8 verified code and the probe code, and add targeted checks for actual changes:
- histogram-derived counts at tau = 0 equal direct confusion counts, bit for bit, on a sample of windows from every set;
- MNDWI bits equal the V7 reference-cache bits; CPU/GPU student parity stays within the documented near-boundary tolerance;
- WorldFloods-v2 splits are deterministic, overlap exclusions are applied, and the access log shows no job read C labels before the confirmatory job;
- the conformal risk control and Learn-then-Test implementation reproduces known risk on synthetic events;
- the ridge fit, clamp and leave-one-event-out code for R4 pass on synthetic data;
- hard-negative cohorts: membership hashes and candidate criteria are correct, initial model hashes match the anchors, and first batches are identical except for replaced items;
- for quantized parents, the actual precision map is recorded and FP/QAT starting hashes are identical.

A deterministic tensor check and a real-batch smoke are sufficient where applicable. No repeated full historical audit unless identity mismatches arise. Preserve originals; fixes go in a new frozen snapshot.

Verify the scheduling changes with a fake clock and mocked GPU/process telemetry; do not spend real ten-minute waits or GPU hours on scheduler tests. Cover these meaningful failure cases:
- busy at startup, still busy at 600 seconds, idle at 1,200 seconds: no early launch, then exactly one eligible launch;
- a foreign compute process with near-zero utilization, insufficient memory, an unallocated device, or failed/unknown telemetry: all remain unavailable;
- a device becomes occupied between the initial check and lock acquisition: the under-lock recheck prevents launch;
- simultaneous workers cannot acquire the same device lock or over-reserve the shared budget; a paired block is deferred if the whole block plus collection cannot fit;
- restart while waiting, during a surviving job, and after a completed receipt: preserve deadlines/spend and do not duplicate a job;
- budget, timeout, pause and stop handling affect only owned processes; final collection still runs after failures or a wait deadline with no free GPU.

Finally perform a live read-only availability check and verify the detached worker's state. A healthy WAITING_GPU state is a valid startup outcome on an occupied host. Run the real-batch smoke only after actual GPU admission, and charge it to the budget.

## 9. Autonomous worker and overnight survival

Implement, test and launch a finite host-side worker that checks GPU availability immediately, then every 600 seconds while jobs are waiting, and automatically starts eligible jobs when an authorized GPU becomes idle. This is an execution requirement, not just a suggested shell command. Do not stop after printing the queue or asking the user to check the GPUs. If this environment only contains reports and cannot access the research host, finish the implementation/manifest and state the exact launch blocker; never claim a worker was launched elsewhere.

### 9.1 Freeze the queue and make it independent of the agent session

Reuse the established detached queue, shared budget ledger and GPU UUID locks. Freeze code/config/input hashes and a finite dependency graph before launch. Each node must have a stable job ID, exact argv/environment, inputs, output paths and validation criteria, dependencies, seed, paired-block ID, conservative GPU cost, memory requirement, timeout, retry policy and declared scientific gate. Gates may use the source evidence explicitly permitted above, never target-query accuracy. Implement all queue decisions and collection in executable code; an LLM session, open VSCode window, browser, or ongoing agent credits must not be needed to continue.

Prefer the research host's existing batch scheduler and respect its allocation rules. Otherwise use an already installed scheduler or a detached setsid/nohup worker with stdin disconnected, absolute paths/interpreter, redirected stdout/stderr, a single-worker lock, atomic state writes and a heartbeat. Do not install a system service, change power policy, request new paid resources or extend an allocation automatically.

Persist a small scheduler configuration with these defaults:
- availability_poll_seconds = 600;
- heartbeat_seconds = 30;
- initial_gpu_wait_hours = 8;
- execution_wall_hours = 8;
- absolute_wall_hours = 16;
- final_collection_reserve_minutes = 10;
- max_jobs_per_gpu = 1;
- max_technical_retries_per_job = 1;
- GPU-command cap and protected collection allowance from Section 3.

Use a monotonic clock for in-process polling and elapsed-time accounting, with persisted UTC deadlines for restart reconciliation. Maintain the 600-second polling cadence while waiting; do not add ten minutes after the end of each scan. Schedule from the prior poll's start, without burst catch-up scans after a delay. A fresh admission check after one of our own jobs finishes is allowed. Heartbeats and responsiveness to pause/stop are separate from GPU scans: do not hide the worker in an uninterruptible ten-minute sleep.

### 9.2 Decide whether a GPU is available

Discover eligible devices by stable GPU UUID, using the existing host configuration and allocation. An idle device is eligible only if this project is authorized to use it; on a managed cluster, obtain the normal scheduler allocation before running GPU commands. Do not assume a device index is stable or that another user's idle allocation is free to use.

Use the host's existing NVML/nvidia-smi facilities. An availability scan may admit a device only when all of the following hold:
1. There is no foreign compute process, no active job of this worker on the device, and no incompatible reservation. Near-zero utilization alone is insufficient: a sleeping training process still owns its GPU.
2. Utilization is below 5% in three observations spanning roughly 30 seconds, taken only when the initial observation suggests availability.
3. Free memory covers the job's measured or conservative peak plus max(2 GiB, 20% of that peak). Establish this requirement from existing comparable jobs or a bounded admitted smoke, not an unbounded speculative run. Account for persistent display/graphics use if present.
4. Device health, allocation, process inventory and memory telemetry are readable and consistent. Unknown/N/A telemetry, unresolved MPS ownership, inaccessible process inventory or stale reservations mean wait and log the reason; they do not imply an idle GPU.

After a device appears available, acquire its existing exclusive UUID lock non-blockingly, acquire the shared ledger mutex, reconcile spend/reservations, and recheck device availability before admission. Use one documented lock ordering throughout. Reserve the full remaining paired block and protected collection cost atomically; avoid double-reserving jobs already covered by that block. Release the ledger mutex promptly. If any check fails, release newly acquired locks/reservations and return to waiting. Hold the GPU lock for the running job and release it when the job ends; never hold a GPU lock merely while waiting for a dependency or another user's job. Advisory locks protect cooperating workers only; the host scheduler remains the authority for exclusive allocations.

Run at most one job per GPU. Independent jobs may use multiple authorized idle GPUs only if dependencies, total reservations and host resources permit. Sum elapsed GPU-command hours across concurrent jobs; parallel execution does not make the cumulative cost equal to wall time. Keep paired speed measurements on the same GPU model/device and under comparable conditions. Do not introduce multi-GPU training just to use spare devices.

### 9.3 Run, wait and recover without intervention

On startup, reconcile existing job receipts and process identities, then run CPU-only reconciliation/collection tasks immediately. Scan for available GPUs. If none is available, enter WAITING_GPU and set next_check_at to the next 600-second tick. Keep checking until work becomes admissible, the queue completes, the user stops it, or a Section 3 deadline expires. Re-evaluate verified spend, external reservations, dependencies and the frozen priority order at each admission. Do not launch a block whose conservative remaining runtime, shutdown margin and collection cannot fit the remaining budget/deadline. A bounded smoke can be a separately budgeted admission to refine the estimate.

Waiting for a busy GPU is not an experiment failure. Keep distinct states for READY, WAITING_GPU, WAITING_DEPENDENCY, PAUSED, RUNNING, COMPLETED, TECHNICAL_FAILED, SCIENTIFIC_GATE_FAILED, UNAVAILABLE, DEFERRED_BUDGET, DEFERRED_BUSY and DEFERRED_DEADLINE. A failed scientific gate has no automatic retry. Allow at most one retry after a diagnosed transient technical failure, with both attempts charged and sufficient remaining budget. Do not restart a poor-performing experiment with changed settings.

Monitor owned jobs frequently enough to enforce timeouts and budget limits, independently of the ten-minute idle polling. Reserve a termination grace margin before a hard limit, stop new admissions in time, and checkpoint/terminate only this worker's own verified processes if needed. Identify a process by PID plus start identity and, where available, boot identity; a reused PID is not one of our jobs. If foreign work appears on a shared device after admission, do not kill or change it. Stop further admissions there, checkpoint/pause our own job where supported, or stop only our own job if safe coexistence cannot be established. Record contention and exclude contaminated latency/energy measurements from performance claims.

Resume must reconcile the ledger, live process identities, output hashes and completion receipts before doing anything. Reattach monitoring to surviving owned jobs instead of launching duplicates. Mark genuinely interrupted jobs and resume from a validated checkpoint only under the declared retry policy. Preserve the original caps and deadlines. On partial failure, budget exhaustion, stop or timeout, run final CPU collection and leave a complete manifest of completed and deferred experiments. If no GPU becomes available within the initial wait window, collect the CPU results and mark GPU jobs DEFERRED_BUSY; do not fabricate a scientific result or silently wait forever.

### 9.4 Visibility, controls and launch verification

Write an append-only gpu_availability.jsonl with check time, eligible UUIDs, utilization, free/required memory, process/reservation availability, decision and reason. Maintain scheduler_status.json with heartbeat, last/next GPU check, active/pending jobs, completed receipts, verified/reserved/remaining GPU hours, launch/first-GPU/deadline timestamps and run directory. Never copy credentials or unrelated command lines into these reports.

Provide working status, pause, resume and stop commands for this exact run. Pause prevents new admissions and lets an active job reach a safe checkpoint/completion; stop prevents new admissions, safely ends only owned work and invokes collection. Logs and these controls must work from a fresh terminal after closing VSCode. Report exact commands, not placeholders. Preserve originals and write the launch receipt into the V9 run directory.

Compile/import, run Section 8's targeted checks, launch the detached worker, then verify its identity, fresh heartbeat and persisted state after the launcher exits. If an eligible GPU is already idle, verify the first admitted job or its completion receipt. If GPUs are busy, verify WAITING_GPU, the recorded blocking reason and the next check scheduled within ten minutes; do not force a job onto a busy GPU merely to prove startup. Report the actual PID/service or scheduler job ID, eligible GPU UUIDs, run directory, budgets/deadlines, status and controls. Report a GPU experiment as started only when its own launch receipt exists.

The worker must continue after closing VSCode while the host and any required allocation remain active. It cannot run through host sleep/shutdown, container destruction or allocation termination. This is a finite local polling worker for this V9 queue; stop it after collection when work is complete or a deadline is reached. Do not create an unrelated ChatGPT reminder or an indefinitely recurring experiment schedule.

## 10. Deliverables and reviewer relevance

Write concise research outputs, not an extensive replacement documentation system:
- frozen_protocol.md, manifest/state/ledger and launch receipts;
- scheduler configuration, scheduler_status.json, gpu_availability.jsonl, targeted scheduler-test results and exact status/pause/resume/stop commands;
- evidence_index.json, splits_wf2.json, wf2_exclusions.csv, the label-access log and the hard-negative cohort manifests;
- revision_corrections.csv, reviewer_closure_status.md, same_image_label_swap.csv, alternative_teacher_comparison.csv (with explicit unavailable/deferred states) and dry_window_alarms.csv;
- operating_point_rules.csv, op_results_by_event.csv, op_summary.csv, risk_control.csv, threshold_free_ap.csv and missed_water_spectra.csv;
- hard_negative_results.csv;
- full_precision_frontier.csv with the quantized operating-point refit, any FP/QAT paired results and backend/timing receipts;
- negative_results.md, including the retired head-adaptation rationale;
- report_v9.md and a compact reviewer_response_map.md.

The morning report must state what ran, what was reused, what was skipped and why, with observation versus explanation separated. Include an experiment checklist covering every required and conditional block, its paired controls, admitted/completed seed counts, gate or deferral reason, and evidence path. Report GPU wait duration separately from execution time and summed GPU-command hours. If the worker is still within its declared window, publish a clearly labelled interim status rather than claiming the run is finished. Do not count deferred or unavailable experiments as negative scientific evidence.

Reviewer mapping, grounded in the actual letter:
- R1.1/R2.1: technical novelty requires controlled evidence for a specific contribution; familiar ingredients alone are insufficient. The candidate contribution is archive-calibrated operating-point control for an on-board distilled segmenter (Section 4), plus disagreement-mined hard negatives (Section 5) if they pass Section 7.
- R1.2/R2.4: existing MobileNet/LightFormer KD/OFA comparisons already address the alternative-student gap; use them rather than restarting. Add the B0 and R4 operating-point readouts.
- R2.3: retain the corrected budget-dependent OFA results without inventing wall-clock time-to-quality.
- R2.5: threshold-free average precision, precision at matched recall, dry-window alarms and per-event wins, with MNDWI and NDWI given the same calibration as the students.
- R1.3/R2.2:
  - use the existing author clarification;
  - give the full data-role table (S11 validation as source, D as development, C sealed, exposed benchmarks) and the probe's exposure statement;
  - disclose source labels, unlabelled context, benchmark exposure and geometry/acquisition confounds;
  - add the same-image label-source control.
- R1.4: consolidate the relevant component controls.
- R1.5: distinguish surface-water segmentation, flood-event application, teacher-derived supervision, operating-point calibration on a labelled public archive, and zero-label claims.
- R1.6/deployment: report dry-window alarms with independent references, the operating-point trade-off and the hard-negative arm. Independent ordinary dry imagery remains unresolved unless verified references actually exist. Qualify hardware and mixed precision.

Do not claim this follow-up alone closes all reviewer concerns. Do not alter submitted manuscripts or send anything externally.

## 11. Prior art and claim boundary

Relevant starting points, not an exhaustive novelty clearance:
- Saerens, Latinne, Decaestecker, Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure, Neural Computation 14(1), 2002. EM prior correction.
- Alexandari, Kundaje, Shrikumar, Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation, ICML 2020: https://arxiv.org/abs/1901.06852
- Ovadia et al., Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift, NeurIPS 2019: https://arxiv.org/abs/1906.02530
- Yu, Bates, Ma, Jordan, Robust calibration with multi-domain temperature scaling, NeurIPS 2022: https://arxiv.org/abs/2206.02757. Predicts calibration parameters from domain features learned across many domains; closest relative of R4.
- Angelopoulos, Bates, Fisch, Lei, Schuster, Conformal risk control, ICLR 2024: https://arxiv.org/abs/2208.02814
- Angelopoulos, Bates, Candès, Jordan, Lei, Learn then Test: calibrating predictive algorithms to achieve risk control: https://arxiv.org/abs/2110.01052
- Martinis, Twele, Voigt, Towards operational near real-time flood detection using a split-based automatic thresholding procedure on high resolution TerraSAR-X data, NHESS 9, 2009. Scene-wise thresholding in operational flood mapping.
- Shrivastava, Gupta, Girshick, Training region-based object detectors with online hard example mining, CVPR 2016: https://arxiv.org/abs/1604.03540
- Portalés-Julià, Mateo-García, Purcell, Gómez-Chova, Global flood extent segmentation in optical satellite images, Scientific Reports 13, 20316, 2023 (WorldFloods-v2).
- Context for the retired adaptation draft: EATA, https://arxiv.org/abs/2204.02610; SelectNAdapt, https://arxiv.org/abs/2308.04946.

Our candidate hypothesis is specific:
- a distilled on-board flood segmenter inherits a conservative, source-calibrated operating point, and this dominates its correctable out-of-distribution error;
- a threshold calibrated on an independent labelled archive (global, risk-controlled, or scene-adaptive from label-free statistics) recovers part of that error at a declared false-alarm price;
- disagreement-mined negatives may lower that price.

V9 must test this against source-only calibration, label-free rules, easy negatives and the deployed model. No first-ever claim and no guaranteed improvement.
