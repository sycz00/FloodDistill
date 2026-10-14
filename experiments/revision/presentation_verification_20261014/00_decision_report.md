# Presentation verification and Phase B decision report — 14 Oct 2026

Task: `prompts/presentation_verification_claude.md`, applied to the reviewer deck
`MajorRev/FloodDistill_reviewer_presentation.pptx` and its companion `.md`. Phase A audits
are in this folder; this file is the Phase B checkpoint. Nothing was trained, launched or
re-evaluated under a changed protocol. No existing artifact was modified.

Evidence tags: **[V]** re-derived in this session from raw files (script in `checks/`,
output in `outputs/`, rebuild with `run_all.sh`, CPU only); **[R]** taken from an existing
report or CSV without re-derivation; **[H]** my judgement or hypothesis.

## 1. Is everything in the deck correct?

Most numbers are correct. I re-derived about 40 of them. Four statements are wrong, and
five are correct but miss a caveat that changes how they read. Details: `deck_corrections.md`.

**Wrong**

1. **Slide 10 / companion / wrap-up §5.4 and fig03: "ReLU6 was trained from scratch; its cost
   vanishes with a validated learning rate (76.2 vs 74.8)."** [V] The paper's ReLU6 students
   start from the pretrained EfficientViT-B0 weights, with Hardswish swapped for ReLU6 before
   training. Every surviving ReLU6 checkpoint has median cosine 0.99 to the Cityscapes B0
   weights; the scratch v7 A0 control has 0.00 (`outputs/relu6_init_test.txt`). The Table III
   ReLU6 arm is a clean single-factor ablation: same init, LR 5e-5, batch 16, OFA, 50 epochs.
   Its effect is −0.2 mIoU per STURM flood map (p = 0.24) and +0.6 per WF scene (p = 0.07),
   with opposite signs (`outputs/relu6_vs_hs_paired.txt`). The answer to R1.4 is better than
   the slide claims: **activation replacement has no measurable cost**. The 76.2-vs-74.8
   comparison is a learning-rate effect for *scratch* models and does not concern the deployed
   student. The paper's own "ReLU6-from-scratch" wording (§III-D, App. B) is also wrong.
2. **Companion A3, limit 5: "STURM's 28 dry windows: every method flags 22–24, so the
   references miss water."** [V] 20 of the 28 have **zero valid pixels** (only classes 3–5,
   which are ignored). The V9 alarm rule `water ≥ 0.001·n` is true when n = 0. Only 8 STURM
   chips are real dry chips: teacher alarms on 4, MNDWI on 3. V9's 14 Oct recovery already
   corrected the denominator (`research_v9_recovery_20261014T081500Z/dry_denominator_correction_receipt.json`).
   The WF count of 206 is unaffected (0 zero-valid windows).
3. **Slide 14 / companion A3: "S11: 73.5 vs 67.9 %".** [V] MNDWI's S11-test precision is
   **67.5 %**.
4. **Slide 10 / slide 4 status "R1.4 Answered".** [V] The isolated effect of QAT
   (QAT vs PTQ on the same ReLU6 parent) is not measured anywhere. Only "INT8 engine vs float
   export" exists, and that is a different factor. The status should be *Mostly* until P1
   (below) runs.

**Correct, but missing a caveat**

5. **Slide 12 (R1.6):** the deck names only the teacher's 5 % pre-filter. [V] There is an
   earlier dry-scene filter. **Every event-window tile had to contain ≥ 1 % MNDWI > 0 "new"
   water (not JRC-permanent) to be downloaded**, with up to 8 random relocations
   (`datasets/prepare.sh`, `flood_unlabeled_v2/supervisor.sh`). In the pool, all 11,895
   event tiles pass it; 5,225 of 17,120 slots never found a passing location. Only the
   11,825 pre-event baselines are ungated. The paper says only "download and screening".
6. **Slides 12/15, dry windows:** the counts are right (MNDWI 136, teacher 5, paper student
   15/16/8/9 of 206) [V]. But the 206 windows come from **9 events, and 82 % from three**.
   MNDWI alarms on every dry window of EMSR286 and EMSR264, and in 8 of 9 events; the teacher
   in 2 of 9; the student in 5 of 9 (`outputs/wf_dry_by_event.txt`). The replication unit is
   9 events, not 206 windows.
7. **Slides 14/15, "inside the training distribution":** [V] S11 test chips come from the
   **same 10 events** as S11 train, the split the teacher was fine-tuned on. Only Bolivia
   (15 chips) is event-held-out. The 13-point lead over MNDWI is a same-event result.
8. **Slide 13 (R2.2), label swap D:** numbers verified [V]. Additional caveat [R]: the
   teacher drew D's labels on S11 chips **resized 512 → 224 (≈ 22.9 m)**. The teacher was
   fine-tuned on native 10 m crops (official `config.yaml`: RandomCrop 224, no resize).
9. **R1.4 figure, soft targets:** the same scale issue applies to live KD on all S11 arms
   [V, code]. There the teacher's soft targets are computed on 22.9 m resized inputs, against
   about 11.8 m on GDACS arms (`data/sen1floods11.py` train resize; paper App. A). This is an
   implementation inconsistency inside Table I. The B1 crop control partly removes it.

## 2. Decisions for Phase B

| Question | Answer | Basis |
|---|---|---|
| Is the current evaluation protocol defensible? | Defensible as a *flood-increment* target, but internally inconsistent. It ignores permanent water on WF (JRC) and on STURM (classes 3–5), yet keeps STURM class 2 ("river-related features", 5.5 % of scored water, the only water in 41 chips) as flood. Both datasets' own protocols score **all water**. So does the task the labels define (S11 = surface water, R1.5). | `dataset_protocol_audit.md` §3 [V]; WF-v2 Sci. Rep. 2023 [R]; paper App. A [R] |
| Class mappings: keep or change? | **Recommend** (needs your approval): primary = common *surface-water* protocol (WF water = flood + permanent; STURM 1–5 = water; ignore cloud, invalid, no-data). Secondary = current flood-only tables, unchanged, for continuity. Do not silently re-label STURM class 2 without its documented definition. | §3 of the protocol audit |
| Does GDACS preprocessing need correction? | **No correction to pseudo-label generation.** Same L1C product (S2_HARMONIZED), same bands, scaling and normalisation. Differences: 264 → 224 resize (≈ 11.8 m vs native 10 m), bilinear 20 m bands, clip to [0, 1]. These are small and can be measured (P3). The real inconsistency is the teacher scale on **S11** arms (items 8–9), not on GDACS. | `dataset_protocol_audit.md` §2 [V] |
| Is dry-scene training inclusion necessary? | **Not yet.** First measure false alarms on an independent dry set (P5). Only if the student alarms on ordinary dry land should the dry-inclusive training arm (P6) run. | `dry_scene_audit.md` |
| Which experiments are essential? | P1 (QAT vs PTQ), P4 (surface-water re-scoring of existing checkpoints), P5 (independent dry set). P2, P3, P6 and P7 are optional. | `pipeline_ablation_plan.md`, `reviewer_evidence_matrix.md` |

## 3. Proposed experiments, cost, approval

GPU time is measured where a comparable run exists: B0 N = 2,500, 50 epochs ≈ 17 min on one
RTX 4090 (`OLD/esi_project/logs/repro_full`). Both GPUs currently run V10 recoveryD; none of
these jobs should start before V10's queue frees a GPU.

| ID | Experiment | Protocol-dependent? | Cost | Needs approval |
|---|---|---|---|---|
| **P1** | QAT vs PTQ on the four surviving ReLU6 parents (`repro_review/pseudo_ofa_relu6_*`) and their QAT children. Same W8A8 per-tensor simulator, same S11-val calibration batches. Read out on the existing protocol. | No | < 0.5 GPU-h, no training | No (R1.4, protocol-independent) |
| P2 | Activation × LR at pretrained init (Hardswish/ReLU6 × LR {5e-5, 5e-4}, grid seed 42, then 4 seeds at the S11-val choice) | No | ≈ 4–5 GPU-h | Optional. Run only if you want the claim "no ReLU6 cost at any validated LR" |
| P3 | Teacher scale check: teacher on S11 val/test at native 512, at 512 → 224 → 512 and at 264-crop → 224. Quantifies the off-scale teacher behind items 8–9 and the GDACS resize | No | < 0.3 GPU-h | No |
| **P4** | Re-score existing checkpoints (teacher, paper student, R4, MNDWI, NDWI) under the surface-water protocol: WF incl. permanent water; STURM 1–5; equal-event and pooled | **Yes** | ≈ 1 GPU-h of inference | **Yes** |
| **P5** | Independent dry set: about 120 tiles, ≥ 30 locations, stratified (desert, urban, snow, cropland, forest, mountain and cloud shadow). Selected *before* any model runs using JRC "never water" + WorldCover + visual check. Metrics: share of tiles with any / > 0.1 % / > 1 % water, false-water area, event-clustered CIs | **Yes** (new eval set) | GEE download + 1 person-day check; < 0.5 GPU-h | **Yes** |
| P6 | Dry-inclusive training: replace 10 % / 25 % of N = 2,500 with "doubly dry" baselines (teacher < 5 % **and** MNDWI = 0; 1,796 exist), N fixed, 4 seeds | Yes (needs P5 readout) | ≈ 2.5 GPU-h | Yes, and only if P5 shows a problem |
| P7 | Bolivia (official event-held-out S11 split, 15 chips): learned vs MNDWI | No | minutes | No. It is the only *new-event* in-distribution-sensor test available without new labels |

Stopping rules: P1 stops after one pass (it is an attribution, not a search). P2 stops if
the seed-42 grid moves either activation by less than 0.5 S11-val mIoU. P6 does not start
unless P5 gives the paper student > 5 % alarm tiles at the 1 % rule.

## 4. What I need from you

1. Approve or reject the **primary surface-water protocol** (P4). The current flood-only
   tables would stay as supplementary.
2. Approve the **independent dry set** (P5): the selection rule above and the ~1 person-day of
   visual checking.
3. Decide whether STURM class 2 stays "flood" in the supplementary protocol. Its definition
   is not in the dataset files, the Zenodo record or the GitHub README. It needs the Big Earth
   Data paper (Notarangelo et al. 2025, Table of classes), which I could not open (HTTP 403).
4. Approve the wording corrections in `deck_corrections.md`. I have not edited the deck or
   companion.

P1, P3 and P7 are protocol-independent and can be queued once a GPU frees up, with your go-ahead.
