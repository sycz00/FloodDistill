# V9 Codex prompt: inherited teacher failures, edge-aware distillation and the remaining reviewer controls

Prepared 12 October 2026, Europe/Berlin. Revised by Claude (Opus) on 12 and 13 October after CPU probes (Section 2.5) and the authors' review of the first revision. The original draft is preserved as prompts/prompt_v9_overnight.orig_20261012.md.
Purpose: implement and run a bounded overnight follow-up to the audited V8 results, incorporating Opus's reviewer wrap-up and probes supplied on 12–13 October.
This is an experimental instruction, not a claim that the proposed method is novel or successful.

## 1. Mission and constraints

Work in the existing FloodDistill research branch and reuse its loaders, graphs, evaluators, quantizers, caches, queue, and reporting code. Inspect AGENTS.md and the working tree first. Preserve user changes and immutable historical runs. Avoid a new framework.

Finish high-value reviewer controls and investigate one main research question, built on a characterization:

1. **Inherited teacher failures (characterization; CPU).** Where does the Prithvi teacher fail on unseen floods, and which failures can be seen in the six input bands? Students trained on teacher labels inherit them. The probe found two failure modes that are visible in the input and one that is not:
   - water at edges and in narrow channels;
   - water next to clouds;
   - flooded vegetation, which is spectrally invisible.
2. **Edge-aware distillation (main; GPU, small).** Can the 12,230-parameter R4 student learn water edges and narrow channels better than its patch-based teacher? R4 predicts at a 4-pixel stride (about 40 m) and upsamples; the teacher works on 16-pixel patches (about 160 m). Two training-time changes are tested against teacher labels and a teacher-confidence control:
   - re-deciding the pseudo-labels in a narrow band around the teacher's edge with the 10 m spectral water signal;
   - leaving that band out of the loss.

   The same physical re-decision is also tested as a zero-parameter decoding step at full 10 m resolution, applied to every model's prediction.

Out-of-distribution labels are never used for training, selection or calibration. WorldFloods-v2 train/val (130 events, never used before) becomes a sealed out-of-distribution test set. It strengthens every out-of-distribution comparison, including teacher versus MNDWI versus a student trained only on the 252 human labels. Compression is reduced to the full unchanged-PTQ readout (Section 6).

Default to S2-only inputs. Use the existing Prithvi teacher, R4 student and existing MNDWI implementation. An already installed, independently human-supervised S2 UNet is admitted only for the controlled teacher comparison below. No S1 requirement, new foundation-model fine-tuning, NAS, new dataset acquisition, driver changes, paid service, message to another person, or manuscript submission. No subagents. The WorldFloods-v2 train/val splits already on local disk (/data1/schmalstieg/flooddistill/data/WorldFloodsv2) are existing public data, not a new acquisition. They are evaluation-only (Section 4.1).

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
2. The probe (Section 2.5) finds that most out-of-distribution misses are reference water with land-like Sentinel-2 spectra. The threshold-correctable part behaves like a one-dimensional shift. Even a perfectly labelled support chip that sets only the threshold gains +2.4 (STURM) / +4.1 (WF) pp on average, with −27 to −36 pp worst cases.
3. No reviewer asked for test-time adaptation.

On 13 October the authors also decided that out-of-distribution labels must not be used for training or calibration, because the paper's claim is out-of-distribution transfer. The first revision's archive-calibrated operating points are therefore withdrawn. Keep V8's adaptation rows as a documented negative result. Record the disagreement-coverage idea and the withdrawn calibration in negative_results.md as future work. This run pursues neither few-shot head adaptation nor out-of-distribution calibration.

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

Independent dry-tile evidence already exists and must be reported. A reference-dry window is a valid window with zero flood-water reference pixels; permanent water is ignored by the WF protocol. Such windows exist in WF test (216), STURM (8 chips) and the sealed WorldFloods-v2 set of Section 4.1. Report the windows flagged at ≥0.1/1/5% predicted water area for:
- every student seed;
- the teacher;
- MNDWI>0.

Describe them as dry land inside flood scenes with EMS/STURM references, not as ordinary unattended imagery.

### 2.5 Exploratory probes after the wrap-up (Claude, 12–13 October; research-exposed)

**Folder:** experiments/revision/novelty_probe_20261012/
- Scripts: probe_cpu.py, probe_hist.py, probe_analysis.py, probe_oprules.py, probe_fewshot.py, probe_predictability.py, probe_failure.py, failure_analysis.py, probe_cloud.py, gpu_vram_check.py, run_all.sh.
- Reports: *_report.txt.
- CPU-only and read-only, except the 15-second VRAM measurement.

**Inputs:**
- the four selected R4 LKD parents fl_R4_sel_s{42,64,91,107} (hashes in capacity_v7 reuse_models.json);
- S11 validation/test, STURM (P3 finite support) and WF test (P3 native windows, with its cloud band);
- the V7 reference cache (teacher hard masks and MNDWI bits). Every probe MNDWI bit matched the cache.

CPU predictions reproduce V7 GPU per-image counts up to 1–2 near-boundary pixels per image (TF32). The probes did not read WorldFloods-v2 train or val. Verify these statements, and treat the numbers below as [R] until you re-derive the ones you use.

Observations (four-seed means; equal-event water IoU unless stated):

- **Where the teacher misses water** (probe_failure.py; reference-based diagnosis):
  - **Edges and narrow channels (S11 test).**
    - 66% of the teacher's missed water lies within 2 px of the reference edge (student: 74%).
    - Recall on edge pixels (1 px): teacher 52.8%, MNDWI 88.5%.
    - Recall on narrow water (half-width ≤ 2 px, about 40 m wide): teacher 20.6%, student 13.2%, MNDWI 83.9%.
    - MNDWI over-extends edges: 61.8% of land pixels 1 px from water are called water, versus 31.4% for the teacher.
  - **Clouds (WF test).** Teacher recall falls with cloud proximity: 45.5% within 5 px of reference cloud, 58.9% at 6–20 px, 67.2% at 21–50 px, 71.7% beyond 50 px. MNDWI keeps 70–80%. About 40% of the teacher's WF misses lie within 50 px of clouds.
  - **Flooded vegetation** (reference water with NDVI > 0.4): recall 0–11% for the teacher and 5–25% for MNDWI on every set. It is invisible to these bands.
  - **Dark and shadowed land:** the teacher is right where MNDWI fails; MNDWI calls 45–95% of dark land water.
- **Edge-band re-decision at decode time.** Pixels within k px of the predicted edge are re-decided by MNDWI > theta; k = 2 and theta = 0.1 were selected on S11 val using the students.
  - S11 test: teacher +1.49 (9 of 10 events better), students +0.92 to +1.52.
  - WF: teacher +0.11, students +0.41 to +1.02.
  - STURM: teacher −0.86, students −0.11 to −0.51.
  - The non-selected theta = 0 gives the students +1.9 to +2.6 on WF and +0.4 to +0.8 on STURM.
- **Cloud-band re-decision fails.** An input-only cloud test reaches precision 0.88 and recall 0.66 against the WF cloud band. Re-deciding pixels within 20–50 px of detected cloud by MNDWI loses 2–6 pp on WF and 1–2 pp on S11, because MNDWI calls cloud shadows water. Clouds stay a documented failure mode without a cheap spectral fix.
- **Comparison with human-label-only students** (legacy protocols, pooled mIoU | water IoU; paper Table I source data and prompt5_closure current_full_summary):

  | Model | S11 test | STURM | WF |
  |---|---|---|---|
  | MNDWI > 0 | 78.9 \| 65.0 | 74.7 \| 64.5 | 76.3 \| 64.3 |
  | Prithvi teacher | 89.5 \| 81.7 | 75.4 \| 64.9 | 77.5 \| 64.8 |
  | B0, CE on 252 human labels | 84.5 \| 73.1 | 71.7 \| 59.7 | 74.4 \| 60.6 |
  | UNet-ResNet50, CE on 252 human labels | 86.2 \| 76.2 | 72.1 \| 61.6 | 77.2 \| 65.0 |
  | B0 distilled on 2,500 GDACS tiles, OFA (paper) | 87.5 \| 78.3 | 76.0 \| 65.4 | 77.5 \| 64.8 |

  Out of distribution, the distilled B0 is 3–4 mIoU above the same B0 trained on the human labels and at teacher level. The teacher is only about 1 mIoU above MNDWI on pooled pixels.
- **Operating point** (the student's decision threshold on its water-minus-land margin; see Section 4.5):
  - Deployed values: S11 val 77.27, S11 test 80.26, STURM 37.83 (precision 60.4, recall 50.3), WF 47.59 (precision 84.9, recall 51.9).
  - Event-optimal threshold chosen with labels (unattainable): median 0 on S11, −2.0 on STURM, −1.5 on WF. Gains, optimised per seed then averaged: +1.3/+1.6 (S11 val/test), +7.6 (STURM), +9.3 (WF).
  - Fixed −0.5 (the best non-zero value on S11 val, weakly supported): STURM +0.78, WF +2.31, S11 test +0.20. WF dry-window alarms rise from 9.0 to 15.0 of 216.
  - Label-free per-scene rules gain at most +1.8 on WF and lose on S11.
  - One labelled chip that sets only the threshold averages +2.4 (STURM) / +4.1 (WF), with worst cases of −36 and −27.
- **Threshold-free comparison:**
  - Student average precision exceeds MNDWI on every set (pooled pixels: S11 test 95.3 vs 91.2, STURM 81.2 vs 76.3, WF 88.3 vs 86.0).
  - At MNDWI>0's recall, student precision is 87.2 vs 77.8 on WF and 71.1 vs 71.3 on STURM.
  - WF reference-dry windows flagged at ≥1% area: student 9.0, teacher 5, MNDWI 140 of 216.
- **Spectral content of misses:** of the reference water the seed-42 student misses, 73% (WF) and 63% (STURM) has MNDWI ≤ 0 and land-like NIR/SWIR reflectance.
- **GPU memory** (gpu_vram_check.json, 13 October, idle GPU index 1): see Section 9.5.

**Exposure:** these rule families and diagnostics were found after reading research-exposed outcomes. Results on S11 test, STURM or WF test are not confirmatory. Confirmation requires the sealed WorldFloods-v2 set of Section 4.1.

## 3. Budget, schedule and job admission

V8 cumulative spend was 25.969689 GPU-command hours under the existing 30-hour cap, leaving 4.030311 then. Recompute current spend and reservations before doing GPU work.

Incremental cap = min(4.0 hours, max(0, 30 - verified spend - other reservations)).
This prompt does not reset that cumulative cap. Count failed attempts, teacher inference, smokes, adaptation, quantization and evaluation consistently. CPU work is separately timed.

Use two explicit clocks: allow up to eight hours from worker launch to wait for the first GPU command, then allow up to eight hours from that first GPU command (including a smoke) to finish the finite run. The absolute wall limit is sixteen hours from worker launch. Record launch, wait, execution and absolute deadlines in UTC and Europe/Berlin; persist them so a restart never resets them. Waiting after the first GPU command counts against the execution window. Reserve the final ten minutes of the applicable execution/absolute window for collection, and run CPU collection immediately if the initial wait limit expires. Idle polling and CPU-only work consume wall time, not GPU-command hours. The host must remain awake. Do not reopen a finished V8 manifest; create a timestamped V9 directory. If another worker is active, respect its locks and reservations and wait automatically under Section 9.

Suggested admission ceilings if four hours remain:
- Same-image label-source control (Section 2.2, primary pairs): 0.65 h.
- Edge-aware distillation (Section 4.3): E1, E2, E3 × 4 seeds with the R4 CE recipe, plus E0 if the anchors are not reusable: 1.30 h.
- Teacher soft outputs on every set, including the sealed test (Section 4.4): 0.50 h.
- EfficientViT-B0 inference on the sealed test (Section 4.4): 0.15 h.
- Full unchanged-PTQ readout, if not done on CPU (Section 6): 0.20 h.
- Protected final evaluation/collection and bounded retry reserve: 0.60 h.
- Conditional on verified freed budget only:
  - B0 version of E1;
  - hard negatives (Section 5);
  - provenance-qualified S2 conventional-teacher comparison (1.50 h);
  - label-swap soft-target extension;
  - R4 FP/QAT pairs.

CPU work starts immediately and does not wait for a GPU. It covers:
- revision corrections;
- the WorldFloods-v2 inventory, overlap screen and sealing;
- R4 inference on every set;
- failure strata;
- edge-band label construction for the cohort;
- dry-window readouts;
- the PTQ readout of the tiny R4 graph where the simulator runs on CPU;
- collection.

GPU priority order:
1. primary label-swap pairs;
2. the edge-aware arms;
3. teacher soft outputs;
4. B0 inference;
5. conditional items.

Section 9.5 gives the authorized devices, the measured memory per job type and the co-location rule. Measure smoke throughput before admitting work. V7 receipts show about 6 GPU minutes per 7,800-update R4 CE run on the RTX 4090. Do not fit an optimistic schedule by dropping controls or exceeding the cap.

These are maximums, not targets. Stop when the frozen work is complete. Preserve paired blocks and final collection before adding optional jobs. If the cumulative budget is exhausted, finish CPU analysis, code and a deferred manifest with exact blockers. If capacity is only temporarily reserved by another active worker, wait and reconcile its verified ledger at the next ten-minute check; never cancel its reservations or treat unverified spend as zero.

All conditional branches and criteria must be in the frozen manifest before target scores are read. Only S11 validation selects settings. Exposed-benchmark and sealed-test outcomes never alter admission, hyperparameters, labels or cohort selection.

## 4. Main question: inherited teacher failures and edge-aware distillation

Working description: edge-aware distillation. The teacher sees 16×16-pixel patches (about 160 m) and blurs water edges and narrow channels. The 12,230-parameter R4 student predicts at a 4-pixel stride (about 40 m; stride-2 stem, finest decoder level H/4) and upsamples, so it is finer than its teacher but not full resolution. It learns the blurred edges from the teacher's labels.

Two ways to fix this are tested:
- **Training-time:** the pseudo-labels in a narrow band around the teacher's edge are re-decided by the 10 m spectral water signal, or are left out of the loss.
- **Decode-time:** the same re-decision is applied at full 10 m resolution as a zero-parameter decoding step after any model.

Verify the teacher's patch size and the EfficientViT-B0 segmentation head's output stride in their configs. This is a proposed combination of pseudo-label refinement and a physical index for foundation-model distillation, not a first-in-literature claim.

### 4.1 Data roles and the sealed out-of-distribution test set

WorldFloods-v2 train (475 scenes, 116 events) and val (16 scenes, 14 events) are evaluation-only. They are never used for training, selection, calibration, gating or stopping. Before reading any of their labels:
- Search code, manifests, logs and receipts of every earlier revision run and verify that no job opened WorldFloods-v2 train/val label files. Report any such events as exposed and keep them out of the sealed results.
- Screen overlap and exclude scenes that intersect, within ±3 days, any GDACS pool tile (all 23,720 pool tiles), any S11 hand-labelled chip (Bolivia included) or any WF test scene. Reuse the prompt5 overlap tooling. If a pool tile has no stored footprint, match by MGRS tile (parsed from the pool scene string, derived from WF-v2 CRS/bounds) and date. Record every exclusion and its reason.
- Write sealed_wf2.json with event codes, scene lists, counts and hashes. Commit its SHA256 in the frozen manifest.
- The labels are opened exactly once, by the final evaluation job, after every arm, setting and checkpoint is frozen. Log opened label paths through a small wrapper so the audit can show this.
- Protocol: identical to WF test (P3 native 512-pixel windows, zero padding invalid, permanent water ignored, clouds excluded from scoring). Record each scene's label-source satellite and label/Sentinel-2 date lag. Report the same-day Sentinel-2-labelled subgroup separately as a secondary result.

| Data | Role |
|---|---|
| S11 train | teacher training (untouched) |
| GDACS N2500 cohort | student training (teacher pseudo-labels and the edge variants below) |
| S11 validation | every selection: checkpoints, band width k, MNDWI threshold theta, any decision threshold |
| S11 test, STURM, WF test | exposed readouts (descriptive) |
| WorldFloods-v2 train/val | sealed out-of-distribution test, read once |

### 4.2 Failure-mode characterization (CPU)

Reuse probe_failure.py and probe_cloud.py, and run them again on the sealed set inside the final evaluation job.

**Predictors to stratify:**
- the teacher;
- the R4 LKD parents and R4 CE anchors;
- the Section 4.3 arms;
- the B0 paper students;
- MNDWI>0;
- the students trained only on human labels (Section 2.2).

**Strata:**
- distance to the reference edge: 1, 2, 3, 4–5, 6–10, 11–20, >20 px;
- water-body half-width: ≤2, 2–5, >5 px;
- distance to the reference cloud band (WF test and WorldFloods-v2), and to the input-only cloud test, on all sets;
- spectral pixel type: dark, bright, vegetated, other;
- WF label-source subgroup.

Report recall and false-water rate per stratum, and each stratum's share of all missed water. This is the evidence for the Discussion section and for R2.5: where learning beats the index, and where the index beats the teacher.

### 4.3 Edge-aware distillation arms (GPU, small)

**Definitions:**
- **Teacher edge band B_k:** the pixels of a GDACS training tile within k px (Euclidean) of the boundary between water and land in the stored teacher hard mask. Compute it on the stored label before augmentation, so spatial augmentations move band and label together.
- **MNDWI:** computed from de-normalized green/SWIR1 with the existing availability rule.
- **p_water:** taken from the stored teacher confidence, which is the max-class probability in [0.5, 1]. p_water = confidence where mask = 1, else 1 − confidence. Verify this on the stored tensors before use.

**Selection of k and theta (S11 validation only):** use the decode-time proxy of probe_failure.py. Re-decide the four R4 parents' predictions within k px of their own predicted edge by MNDWI > theta, with k ∈ {1, 2, 3, 5} and theta ∈ {−0.1, 0, 0.1, 0.2}, and maximize equal-event water IoU. The probe selected k = 2 and theta = 0.1. Re-derive the values and freeze them before any training.

**Arms.** All arms use the identical V7 R4 CE recipe and the same seeds, initial weights, batch order and augmentations. Only the labels inside B_k differ.
- E0, teacher labels: the V7 R4_CE_s{42,64,91,107} anchors if their lineage is verified; otherwise retrain.
- E1, physics edge: inside B_k, label = MNDWI > theta (where MNDWI is unavailable, keep the teacher label).
- E2, ignore edge: inside B_k, label = ignore (−1); interior-only distillation.
- E3, teacher-confidence control: inside B_k, label = p_water > p*. Choose p* on the cohort so that the number of changed labels equals E1's (to the nearest 0.01). This separates "a physical cue" from "more liberal labels at edges".

Record per arm:
- the share of pixels changed or ignored;
- the counts of water→land and land→water flips;
- a hash showing that labels outside B_k equal E0's.

If verified freed budget remains, also run E1 for the paper student: EfficientViT-B0 with the O4 historical_true_ce recipe, 4 seeds, paired with its existing anchors.

**Evaluation.** Use every arm at its deployed threshold (no out-of-distribution calibration) on S11 validation/test, STURM, WF test and the sealed WorldFloods-v2 set. Report:
- equal-event and pooled water IoU, precision, recall and mIoU;
- edge-band accuracy (reference-edge distance 1–2 px);
- thin-water recall;
- dry-window alarms;
- average precision.

**Decode-time arm D.** Apply the decode-time edge re-decision, with the S11-validation-selected k and theta, to every model: E0–E3, the R4 LKD parents, the B0 paper students and the teacher. Report each model with and without it. It needs MNDWI and a distance transform of the predicted edge on board: a few operations per pixel, no parameters. Report its CPU cost per 512 × 512 tile.

### 4.4 Teacher and baselines on the sealed test

When the GPU is admitted, compute Prithvi teacher soft outputs on S11 validation/test, STURM and WF test. Compute them on the sealed set only inside the final evaluation job, together with the EfficientViT-B0 paper students.

Build the out-of-distribution comparison table on all sets, with these rows:
- the teacher;
- MNDWI and NDWI;
- the distilled students (R4 and B0);
- the students trained only on the 252 human labels: the Section 2.2 human-label arms, plus the Table I CE checkpoints if they can be located and verified.

Report pooled and equal-event scores, average precision, precision at matched recall, and per-event wins.

### 4.5 Operating point (analysis only; no out-of-distribution calibration)

**Definition.** The student outputs two logits per pixel, and its margin is the water logit minus the land logit. The deployed decision is margin > tau with tau = 0, which equals water probability > 0.5. For reference:
- tau = −0.5 means p > 0.38;
- tau = −1 means p > 0.27;
- tau = −2 means p > 0.12.

This threshold is unrelated to the MNDWI rule's threshold (0) and to the 5% water-fraction curation filter.

For every model, report on all sets, including the sealed test:
- the event-optimal tau (unattainable);
- the source-selected tau from S11 validation (grid −3 to +1 in steps of 0.5, including 0);
- the probe's label-free rules (EM prior correction, MNDWI-anchored), with settings frozen from S11 validation.

These are characterizations. No threshold is fitted on out-of-distribution labels.

### 4.6 Outputs

- sealed_wf2.json, wf2_exclusions.csv and the label-access log.
- failure_strata.csv.
- edge_arms_results.csv, edge_arms_by_event.csv and edge_label_changes.csv.
- ood_comparison_table.csv and threshold_free_ap.csv.
- operating_point_analysis.csv.
- dry_window_alarms.csv.

## 5. Optional, freed budget only: hard-negative distillation from teacher/rule disagreement

Hypothesis: the curation filter removed dry scenes, including the dark or shadowed surfaces on which MNDWI fires. Replaying such tiles as negatives may suppress MNDWI-type false positives. The probe found that the deployed student already raises few dry-window alarms (9 of 216 on WF test versus 140 for MNDWI), so this block has lower priority than Section 4.

**Cohorts** (fixed N = 2,500, same R4 CE recipe; only membership changes):
- A0: the E0 anchors.
- Removed set: a deterministic 250 tiles — the last 250 in the stored FPS order, or, if no order is stored, the 250 with the smallest filename SHA256.
- A1: the removed set is replaced by 250 FPS-selected pool tiles with is_flood_scene = 0, teacher water fraction < 0.01 and MNDWI water fraction ≥ 0.05 (671 candidates on the 12 October pool table).
- A2: the removed set is replaced by FPS-selected pre-event tiles with teacher water fraction < 0.01 and MNDWI water fraction < 0.01.

Exclude candidates that overlap any evaluation set. Evaluate exactly as in Section 4.3. Claims require A1 to beat both A0 and A2 on sealed-test average precision and on dry-window alarms at equal recall.

## 6. Compression branch: full PTQ readout and operating-point check

Reuse R4 FP parents for all available four seeds. Make a full source-validation and full compatible S11-test/STURM/WF readout for the unchanged V8 PTQ precision point; the original three-event cancellation is insufficient. Reuse counts where identities match. The tiny R4 graph may run the adopted fake-quant simulator on CPU, provided parity with the GPU path is shown on a smoke batch. For each quantized parent, also report its source-selected threshold (Section 4.5) and how much of each quantized-minus-FP event difference disappears with it. This tests whether the quantization loss is mainly an operating-point shift.

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

**Primary outcome (Section 4.3):** the per-seed change in equal-event water IoU versus E0 on the sealed WorldFloods-v2 test. Report the event-bootstrap 95% interval of the seed-mean difference and the across-seed SD as separate quantities.

**Mechanism metrics:**
- edge-band accuracy and thin-water recall, on S11 test (precise human edges) and on the sealed test;
- the dry-window alarm rate, which must not rise by more than 1 percentage point of windows.

**Secondary:** precision, recall, mIoU, pooled values, lower-tail events, average precision, and the exposed sets (S11 test, STURM, WF test).

Declared decision rules (these license claims; they are not significance tests or operational guarantees):
- **Physics-guided edge distillation lets the student beat its teacher at edges.** Requires all of:
  - E1 − E0 ≥ +0.5 pp equal-event water IoU on the sealed test, with lower bound > 0 and a positive difference in ≥3 of 4 seeds;
  - a thin-water recall gain ≥ +10 pp on S11 test;
  - no STURM equal-event loss beyond −0.5 pp.
- **Attribution:**
  - E3 ≈ E1: any gain comes from more liberal edge labels, not from the physical cue.
  - E2 ≈ E1: dropping the teacher's edges is enough (interior-only distillation), and MNDWI adds nothing.
  - E1 ≈ E0: report the inherited edge blur as a characterization only.
- **Decode-time arm D on the deployed student (E0).** Claimed only if it improves sealed-test equal-event water IoU by ≥ +0.5 pp with lower bound > 0, in ≥ 3 of 4 seeds, without a STURM loss beyond −0.5 pp. A trained arm that matches D without MNDWI at inference is preferable for deployment. If E1 + D beats both E1 and E0 + D, report that training-time and decode-time fixes add up.
- **Out-of-distribution comparison table (Section 4.4)** is descriptive. State where the distilled student is better or worse than MNDWI and the human-label-only students, with per-event wins and intervals. Never claim superiority from pooled pixels alone.
- **Exposed sets** are descriptive and never select, gate or rescue anything. S11 validation is the only selection set.
- **Earlier rules still apply:**
  - the event is the transfer unit;
  - do not pool seed and event uncertainty;
  - non-significance is not equivalence;
  - make no "lossless" low-bit claim;
  - qualify mean improvements that hide large event losses.

Stopping interpretations:
- E1 gains on S11 test but not on the sealed test: the edge benefit follows precise human edges and does not survive coarse flood-map references. Report that.
- E2 or E3 beats E1: report the simpler explanation.
- Every arm loses on the sealed test: stop this recipe; keep the characterization.
- Thin-water recall improves while equal-event IoU is flat: report the mechanism gain separately; do not inflate it into an accuracy claim.

## 8. Minimal verification and reproducibility

Reuse V7/V8 verified code and the probe code, and add targeted checks for actual changes:
- confusion counts from the probe code equal the evaluator's counts on a sample of windows from every set;
- MNDWI bits equal the V7 reference-cache bits; CPU/GPU student parity stays within the documented near-boundary tolerance;
- the sealed set is deterministic, overlap exclusions are applied, and the access log shows no job read its labels before the final evaluation job;
- edge-band construction:
  - synthetic masks with known boundaries give the expected band;
  - spatial augmentations move band and label together;
  - E1/E2/E3 labels outside B_k hash-equal E0's;
  - label-change counts are recorded;
- the teacher confidence tensors behave as max-class probabilities (minimum 0.5, consistent with the mask);
- the arms share initial model hashes and first batches, apart from the label tensors;
- for quantized parents, the actual precision map is recorded and FP/QAT starting hashes are identical.

A deterministic tensor check and a real-batch smoke are sufficient where applicable. No repeated full historical audit unless identity mismatches arise. Preserve originals; fixes go in a new frozen snapshot.

Verify the scheduling changes with a fake clock and mocked GPU/process telemetry; do not spend real ten-minute waits or GPU hours on scheduler tests. Cover these meaningful failure cases:
- busy at startup, still busy at 600 seconds, idle at 1,200 seconds: no early launch, then exactly one eligible launch;
- a foreign compute process with near-zero utilization, insufficient memory, an unallocated device, or failed/unknown telemetry: all remain unavailable;
- a device becomes occupied between the initial check and lock acquisition: the under-lock recheck prevents launch;
- simultaneous workers cannot acquire the same device lock or over-reserve the shared budget; a paired block is deferred if the whole block plus collection cannot fit;
- restart while waiting, during a surviving job, and after a completed receipt: preserve deadlines/spend and do not duplicate a job;
- budget, timeout, pause and stop handling affect only owned processes; final collection still runs after failures or a wait deadline with no free GPU;
- one device busy with a foreign job and the other idle: the idle device is used, never the pinned index 0;
- co-location (Section 9.5):
  - a foreign process with stable memory and enough free memory admits only a job of ≤ 1.5 GB, never teacher inference;
  - unstable foreign memory blocks admission;
  - a foreign memory increase or free memory below 1 GiB stops our co-located job and leaves the foreign process untouched.

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
1. There is no foreign compute process (except under the co-location rule of Section 9.5), no active job of this worker on the device, and no incompatible reservation. Near-zero utilization alone is insufficient: a sleeping training process still owns its GPU.
2. Utilization is below 5% in three observations spanning roughly 30 seconds, taken only when the initial observation suggests availability. This is waived only under the co-location rule of Section 9.5, which has its own memory-stability conditions.
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

### 9.5 Authorized devices, measured memory and co-location (added 13 October)

**Authorized devices.** Both RTX 4090s of this host, identified by UUID:
- GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc (index 0);
- GPU-eadba879-32d7-d9cf-1214-f9e16ec4c5cb (index 1).

The V7/V8 queues pinned index 0 (manifest gpu_index/gpu_uuid). V9 must:
- scan both devices;
- keep one UUID lock per device (existing naming /data1/schmalstieg/flooddistill/revision_gpu_<UUID>.lock);
- set CUDA_VISIBLE_DEVICES to the admitted UUID.

At 01:20 CEST on 13 October, index 1 was idle (0%, 11 MiB) and index 0 ran another user's job (20.7 of 24.6 GB). Prefer an idle device whenever one exists.

**Measured peak process memory** (gpu_vram_check.json; includes the CUDA context; random inputs):

| Job | Peak memory |
|---|---|
| R4 training, batch 16 at 224 px | 0.61 GB |
| R4 inference, batch 8 at 512 px | 0.57 GB |
| EfficientViT-B0 training, batch 16 at 224 px | 1.07 GB |
| EfficientViT-B0 inference, batch 8 at 512 px | 0.78 GB |
| Prithvi inference, batch 4 at 512 px | 3.25 GB |
| Prithvi inference, batch 16 at 224 px | 2.87 GB |

Require measured peak + max(2 GiB, 20%) of free memory, and re-measure in the first admitted smoke.

**Co-location with a foreign process.** Authorized by the project owner on 13 October, for small jobs only. A job whose measured peak is ≤ 1.5 GB may run on a device that has a foreign compute process if all of these hold:
1. The compute mode is Default.
2. Free memory ≥ peak + 2 GiB at three consecutive checks spanning ≥ 20 minutes, during which the foreign process memory changed by < 256 MiB.
3. The job caps itself with torch.cuda.set_per_process_memory_fraction at (peak + 0.5 GiB) / total.
4. The worker re-checks every 30 seconds. It stops our job (after a checkpoint where supported) if free memory falls below 1 GiB or the foreign process memory grows.
5. Teacher inference, any job above 1.5 GB, and every timing or energy measurement are never co-located.
6. GPU hours are charged by wall time; contention slows our jobs.

Never kill, signal, renice or otherwise change a foreign process.

**Environment.** EfficientViT jobs need PYTHONPATH to include /home/schmalstieg/repos/_deps/efficientvit. The esi environment has no installed efficientvit package; earlier B0 runs used this checkout. Verify the import in preflight, and fail fast with TECHNICAL_FAILED instead of retrying.

**Ledger.** Charge 0.004 GPU-hours (14.6 s on index 1, 13 October) for Claude's memory measurement.

## 10. Deliverables and reviewer relevance

Write concise research outputs, not an extensive replacement documentation system:
- frozen_protocol.md, manifest/state/ledger and launch receipts;
- scheduler configuration, scheduler_status.json, gpu_availability.jsonl, targeted scheduler-test results and exact status/pause/resume/stop commands;
- evidence_index.json, sealed_wf2.json, wf2_exclusions.csv, the label-access log and the edge-arm label manifests;
- revision_corrections.csv, reviewer_closure_status.md, same_image_label_swap.csv, alternative_teacher_comparison.csv (with explicit unavailable/deferred states) and dry_window_alarms.csv;
- failure_strata.csv, edge_arms_results.csv, edge_arms_by_event.csv and edge_label_changes.csv;
- ood_comparison_table.csv, threshold_free_ap.csv and operating_point_analysis.csv;
- hard_negative_results.csv, if the optional block ran;
- full_precision_frontier.csv with the PTQ operating-point check, any FP/QAT paired results and backend/timing receipts;
- negative_results.md, including the retired head adaptation, the withdrawn out-of-distribution calibration and the failed cloud-band re-decision;
- report_v9.md and a compact reviewer_response_map.md.

The morning report must state what ran, what was reused, what was skipped and why, with observation versus explanation separated. Include an experiment checklist covering every required and conditional block, its paired controls, admitted/completed seed counts, gate or deferral reason, and evidence path. Report GPU wait duration separately from execution time and summed GPU-command hours. If the worker is still within its declared window, publish a clearly labelled interim status rather than claiming the run is finished. Do not count deferred or unavailable experiments as negative scientific evidence.

Reviewer mapping, grounded in the actual letter:
- R1.1/R2.1: technical novelty requires controlled evidence for a specific contribution; familiar ingredients alone are insufficient. The candidate contribution is:
  - the documented, input-visible failure modes that a tiny student inherits from a patch-based foundation teacher;
  - edge-aware distillation, if it passes Section 7. Otherwise the characterization alone is the contribution.
- R1.2/R2.4: existing MobileNet/LightFormer KD/OFA comparisons already address the alternative-student gap; use them rather than restarting. Add the B0 and R4 sealed-test readouts.
- R2.3: retain the corrected budget-dependent OFA results without inventing wall-clock time-to-quality.
- R2.5:
  - the out-of-distribution comparison table on the sealed 130-event set (teacher, MNDWI, NDWI, distilled students, human-label-only students), with average precision, precision at matched recall and dry-window alarms;
  - the failure strata, showing where learning beats the index (dark and shadowed land, false alarms) and where the index beats the teacher (edges, narrow channels, cloud margins).
- R1.3/R2.2:
  - use the existing author clarification;
  - give the data-role table with the sealed set and the probes' exposure statement;
  - disclose source labels, benchmark exposure and geometry/acquisition confounds;
  - add the same-image label-source control.
- R1.4: consolidate the relevant component controls.
- R1.5: distinguish surface-water segmentation, flood-event application, teacher pseudo-labels and the evaluation-only out-of-distribution archive.
- R1.6/deployment: report dry-window alarms with independent references, and clouds as a documented failure mode. Independent ordinary dry imagery remains unresolved unless verified references actually exist. Qualify hardware and mixed precision.

Do not claim this follow-up alone closes all reviewer concerns. Do not alter submitted manuscripts or send anything externally.

## 11. Prior art and claim boundary

Relevant starting points, not an exhaustive novelty clearance:
- Szwarcman et al., Prithvi-EO-2.0: a versatile multi-temporal foundation model for Earth observation applications, 2024: https://arxiv.org/abs/2412.02732. The teacher; verify its spatial patch size in the model config.
- Marsocci et al., PANGAEA: a global and inclusive benchmark for geospatial foundation models, 2024: https://arxiv.org/abs/2412.04204. Geospatial foundation models are not consistently better than a supervised UNet on segmentation.
- Yuan et al., SegFix: model-agnostic boundary refinement for segmentation, ECCV 2020: https://arxiv.org/abs/2007.04269
- He, Sun, Tang, Guided image filtering, IEEE TPAMI 35(6), 2013. Edge-aware refinement guided by the image.
- Kolesnikov, Lampert, Seed, expand and constrain, ECCV 2016: https://arxiv.org/abs/1603.06098. Huang et al., Weakly-supervised semantic segmentation network with deep seeded region growing, CVPR 2018. Both refine pseudo-labels in weakly supervised segmentation.
- Xu, Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery, IJRS 27(14), 2006. MNDWI.
- Portalés-Julià, Mateo-García, Purcell, Gómez-Chova, Global flood extent segmentation in optical satellite images, Scientific Reports 13, 20316, 2023. WorldFloods-v2.
- For the operating-point analysis:
  - Ovadia et al., NeurIPS 2019: https://arxiv.org/abs/1906.02530;
  - Saerens, Latinne, Decaestecker, Neural Computation 2002;
  - Alexandari, Kundaje, Shrikumar, ICML 2020: https://arxiv.org/abs/1901.06852.
- For the optional hard negatives: Shrivastava, Gupta, Girshick, online hard example mining, CVPR 2016: https://arxiv.org/abs/1604.03540.
- Context for the retired adaptation draft: EATA, https://arxiv.org/abs/2204.02610; SelectNAdapt, https://arxiv.org/abs/2308.04946.

Our candidate hypothesis is specific: a tiny student distilled from a patch-tokenized geospatial foundation model inherits the teacher's misses at water edges and in narrow channels. Two things may let the student, or its decoded output, exceed the teacher there:
- physics-guided edge labels, or leaving the teacher's edges out of the loss;
- a zero-parameter physical re-decision at full resolution.

V9 must test this against teacher labels, interior-only and teacher-confidence controls, on a sealed out-of-distribution set. Clouds are a documented teacher failure without a cheap spectral fix in these six bands. No first-ever claim and no guaranteed improvement.
