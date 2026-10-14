# V9 prompt assessment and novelty probe

12 October 2026, Claude (Opus).

**Inputs read:**
- `prompts/prompt_v9_overnight.md` (draft by another agent), now preserved as `prompts/prompt_v9_overnight.orig_20261012.md`;
- `paper/kd_quantization_report.docx`;
- the reviewer letter, V7/V8 run folders and the wrap-up.

**New evidence:** a CPU-only probe in `novelty_probe_20261012/` (read-only; ~10 min; `run_all.sh`).

**Output:** the edited prompt, `prompts/prompt_v9_overnight.md`.

## Revision 2 (13 October): this section overrides §1 and §4–§6 below where they conflict

The authors decided that labels from the out-of-distribution datasets must not be used for training or calibration: the paper's claim is transfer to unseen floods. Changes:
- **Archive calibration is withdrawn** (Idea 1 below). WorldFloods-v2 train/val (130 events, never used) becomes a **sealed out-of-distribution test set**, read once at the end.
- **The threshold** discussed below is the student's own decision threshold on its output: water if the water-minus-land score exceeds tau. tau = 0 is the deployed model (probability > 0.5); tau = −1 means probability > 0.27. It is unrelated to MNDWI's 0 and to the 5% curation filter.
- **Is the foundation model better out of distribution?** Paper Table I plus the current audited summary (pooled mIoU, legacy protocols):

  | Model | S11 | STURM | WorldFloods |
  |---|---|---|---|
  | MNDWI | 78.9 | 74.7 | 76.3 |
  | Prithvi teacher | 89.5 | 75.4 | 77.5 |
  | B0 trained only on the 252 human labels | 84.5 | 71.7 | 74.4 |
  | UNet (32.5M) trained only on the 252 human labels | 86.2 | 72.1 | 77.2 |
  | Distilled B0 (paper) | 87.5 | 76.0 | 77.5 |

  Reading the table:
  - The teacher is only about 1 point above MNDWI on pooled pixels out of distribution.
  - The distilled tiny student is 3–4 points above the same student trained on the human labels, and reaches the teacher.
  - The foundation model's value is as a label amplifier for tiny students.
  - Against MNDWI the advantage shows in precision, dry-tile false alarms (9 vs 140 of 216) and ranking quality, not in pooled mIoU.
- **Teacher failure modes visible in the input** (new probe, `probe_failure.py`, `probe_cloud.py`):
  - **Edges and narrow channels.**
    - On S11, 66% of the teacher's missed water lies within 2 px of the true edge.
    - Narrow channels (≤ 2 px half-width) are found 21% of the time by the teacher and 13% by the student, versus 84% by MNDWI.
    - Re-deciding a 2-px band at the predicted edge by MNDWI improves S11 test by +0.9 to +1.5 for every model and WorldFloods by +0.1 to +1.0. STURM is slightly worse.
  - **Clouds.**
    - On WorldFloods, teacher recall drops from 72% far from clouds to 46% within 5 px of a cloud.
    - The cloud fix fails: MNDWI mistakes cloud shadows for water, costing 2–6 points.
    - Clouds are documented as a failure mode with no cheap fix.
  - **Flooded vegetation** is invisible to these bands for every method.
- **New main experiment: edge-aware distillation.** Four R4 cross-entropy arms × 4 seeds, about 0.1 GPU-h per run:
  - E0: teacher labels;
  - E1: edge band re-decided by MNDWI;
  - E2: edge band left out of the loss;
  - E3: a teacher-confidence control.

  The zero-parameter decode-time re-decision (D) is tested on every model. Everything is judged on the sealed set.
- **Hard negatives** become optional, because the student already raises few dry-tile alarms.
- **GPU admission.**
  - V7/V8 pinned GPU 0, which another user occupies; V9 now scans both GPUs (GPU 1 was idle at 01:20).
  - Measured VRAM: R4 0.6 GB, B0 1.1 GB, teacher 3.3 GB.
  - Small jobs (≤ 1.5 GB) may share a busy GPU under strict memory safeguards.
  - B0 jobs need PYTHONPATH set to `/home/schmalstieg/repos/_deps/efficientvit`.
- **Prompt revisions:** revision 1 is preserved as `prompts/prompt_v9_overnight.rev1_20261012.md`.

## 1. Bottom line

- **The draft's core experiment would most likely have wasted the night.** That core was test-time head adaptation from teacher targets, with disagreement-based support selection and protection. V8's own receipts show the teacher makes the student *worse* more often than it corrects it on STURM and WorldFloods (G < H for both parents). Teacher-driven updates are therefore capped at "no change". No reviewer asked for adaptation, and a successful result would not have answered the novelty concern either.
- **mIoU is saturated for three measurable reasons, not for lack of model capacity.**
  - *Invisible water.* Most water the student misses is invisible to Sentinel-2 spectra: 73% of WorldFloods misses and 63% of STURM misses have land-like spectra (flooded vegetation, label/date mismatch).
  - *Operating point.* The student is precise but conservative on unseen floods (WorldFloods precision 85%, recall 52%). The decision threshold that is optimal on Sen1Floods11 is too strict there. Choosing the threshold per event with labels would add +9.3 water-IoU points on WorldFloods and +7.6 on STURM.
  - *Annotation conventions.* The best threshold differs from event to event.
- **Simple physics fusion is not the answer.** Adding MNDWI to the student's output (union, veto or connectivity-based growth) gains ≤ 0.9 points. The water MNDWI finds and the student misses looks spectrally like MNDWI's false alarms.
- **The defensible novelty with gains is operating-point control.** The threshold is calibrated on an independent, never-used labelled flood archive: the WorldFloods-v2 train/val splits, 130 events already on disk. It comes in three forms: global, risk-controlled with guarantees, or scene-adaptive. It is confirmed on a sealed half of that archive. It costs nothing on board (one number) and targets recall, which is the axis with headroom.
- **Second idea, cheap and aimed at precision: hard-negative distillation.** Put back dry pre-event tiles where the teacher sees no water but MNDWI does (shadows, dark surfaces). The aim is to make the recall gain cost fewer false alarms. It uses about 0.9 GPU-h and has an easy-negative control.
- **The R2.5 answer improves, with no training.**
  - Average precision: the student ranks pixels better than MNDWI on every benchmark (WorldFloods 88.3 vs 86.0; STURM 81.2 vs 76.3; S11 95.3 vs 91.2).
  - Dry tiles: on 216 WorldFloods windows whose reference has no flood water, MNDWI marks ≥ 1% water on 140; the student does so on 9.
- **The literature report doesn't change the plan.** It is a generic, mostly LLM-focused survey, and about eight in-text citations point to the wrong reference: e.g. "Compressor [4]" (ref [4] is Cao & Aref), "TinyViT [6]" (NeUQI), "TinTeM [10]" (L4Q), "DistillLens [12]" (an anomaly-detection paper), and [5], [7], [11], [14] are each used for two different works. Don't cite from it without checking.

## 2. Verdict on the original V9 draft, section by section

| Section | What it does | Cost | Verdict |
|---|---|---|---|
| 2.1 Corrections CSV | Re-derive paper errors (teacher row, 3.95→3.25 W, pre-filter claim, crop counts…) | CPU | **Keep** (unchanged) |
| 2.2 Same-image label swap | R2.2 control: same 252 S11 images, human vs teacher labels, four B0 seeds | ≤ 0.65 GPU-h | **Keep** (unchanged); the only real reviewer control |
| 2.3 Second teacher (UNet) | Is the foundation model necessary? | 1.5 GPU-h | **Demote to freed budget only.** Not what R1.2 asks (R1.2 is about lightweight student/distillation methods, already covered). Inference-only inventory and quality readout may still run. |
| 2.4 Dry scenes | Six verified dry chips; 11 unverified windows stay blocked | CPU | **Keep**, and add the 216 + 8 reference-dry windows that already exist |
| 4 Causal solver replay | Why did V8's ridge fail? | part of 0.75 GPU-h | **Retire.** Debugging a failed method; no reviewer value |
| 5 Protected head adaptation (11+ arms × 16 episodes) | Disagreement-selected support, regime protection, Newton vs SGD | 0.75 GPU-h, large engineering | **Retire.** Bounded by "no change" (G < H). Even a perfectly labelled chip that sets only the threshold gains +2.4/+4.1 on average with −27 to −36 worst cases. |
| 6 Compression | Full PTQ readout, size frontier, QAT pairs | 0.5 GPU-h | **Keep the cheap readout** (CPU possible), add a threshold-refit check; QAT only from freed budget |
| 9 Scheduler/worker | GPU polling, locks, budget ledger | — | **Keep verbatim.** Well specified, and needed: both GPUs are busy (another user's jobs) |

## 3. What the probe tested (plain words)

The four 12,230-parameter R4 students were run on CPU over S11 validation and test, STURM and WorldFloods test. For each pixel the probe recorded:
- the student's "water minus land" score;
- the MNDWI value;
- the teacher's cached prediction;
- the reference label.

Every rule below was tuned on **S11 validation only**, then read on the external sets.

| Rule (what it does) | S11 test | STURM | WorldFloods | Dry-window false alarms, WF (of 216) |
|---|---|---|---|---|
| Deployed student (water IoU, equal-event) | 80.26 | 37.83 | 47.59 | 9.0 |
| Teacher (Prithvi, 300M) | 79.06 | 38.84 | 50.54 | 5 |
| MNDWI > 0 | 68.33 | 39.25 | 40.06 | 140 |
| + MNDWI where connected to student water ("growth") | +0.1 | −0.2 | +0.9 | — |
| Lower threshold, −0.5 (picked on S11 val, weakly) | +0.2 | +0.8 | +2.3 | 15.0 |
| Lower threshold, −1 (for comparison) | −1.4 | +1.4 | +3.9 | 23.8 |
| Label-free per-scene threshold (EM prior correction) | −0.6 | −0.3 | +1.8 | 16.8 |
| One labelled chip sets the threshold (average) | −3.3 | +2.4 | +4.1 | — |
| Best threshold per event, chosen with labels (unattainable) | +1.6 | +7.6 | +9.3 | — |

Rows below the first three are changes in water-IoU points versus the deployed student.

Reading the table:
- On data that resembles training (S11), the student's default threshold is already right.
- On unseen floods, a more liberal threshold finds much more water, but how much more varies by event.
- Rules that look only at the image get a small part of this.
- One labelled chip is unreliable: some chips make things far worse.
- Lowering the threshold is not free: dry-tile false alarms rise from 9 to 15–24 of 216 windows.

Label-free scene statistics predict each event's best threshold only moderately (Spearman ≈ 0.45 over 40 events). That is why the edited prompt fits the scene-adaptive rule on ~58 *labelled* archive events and confirms it on a sealed half, rather than trusting a hand-made rule.

The probe has read the exposed benchmarks. Its numbers motivate the plan but cannot confirm it; the sealed WorldFloods-v2 events can.

## 4. The two scheduled ideas

### Idea 1 (main): archive-calibrated operating points

**What.** Keep the student as is and change only its decision threshold. The threshold is set from an independent public archive of labelled floods: WorldFloods-v2 train/val, which no part of this project has used. Three versions:
- **(a) Global.** One threshold that maximises water IoU over the archive's development events.
- **(b) Risk-controlled.** The most conservative threshold that guarantees, for example, ≥ 60–70% recall on future archive-like events, while keeping dry-window false alarms ≤ 5%. It uses conformal risk control; the guarantee holds only within the archive.
- **(c) Scene-adaptive.** A three-feature linear rule predicts each scene's threshold from statistics the satellite can compute itself: predicted water fraction, the share of uncertain pixels, and the MNDWI-positive fraction.

**Protocol.** Events are split by hash into D (development, ~58 events) and C (sealed confirmation, ~58 + 14 events). C labels are opened once, at the end. S11 test, STURM and WorldFloods test are reported only as transfer.

**Why it is promising.** It targets the axis with measured headroom (recall; +7–9 points at the per-event optimum) and costs nothing on board. It also answers three concerns at once:
- R2.5: fair comparison with MNDWI, which gets the same calibration;
- R1.6: explicit false-alarm control;
- R1.3: a clean data-role table with a sealed set.

**Expected outcome (my judgement).**
- Global calibration: about +2–4 points on WorldFloods-like events, paid for with more dry-tile alarms.
- Risk control: may show that a recall target and a 5% alarm limit cannot both be met with one threshold. That is a useful result to report, not a failure.
- Scene-adaptive rule: uncertain; perhaps +0.5–1.5 beyond global, possibly nothing.

**Novelty claim.** Moderate. The ingredients are established (label-shift correction, multi-domain calibration, conformal risk control). The contribution would be twofold:
- the finding that a distilled on-board segmenter inherits a conservative operating point that dominates its correctable error;
- a deployable, sealed-set-validated calibration procedure.

This is honest, specific and controlled, which is what R1.1/R2.1 ask for. It is not a "first ever" claim.

**Cost.** CPU for the R4 students (inference over ~490 scenes); ~0.5 GPU-h for teacher scores (diagnostic); ~0.15 GPU-h for the paper's EfficientViT-B0.

### Idea 2 (secondary): hard-negative distillation

**What.** The data filter removed dry tiles. Replace 250 of the 2,500 training tiles with dry pre-event tiles where the teacher sees no water but MNDWI does: the pool has 671 such tiles. The control replaces the same 250 tiles with ordinary dry tiles. Four seeds each, with the cheap R4 cross-entropy recipe (about 6 GPU minutes per run).

**Why.** Idea 1 trades recall against false alarms along the same curve. Hard negatives could move the curve itself, so that recall costs fewer false alarms. It reuses the draft's teacher-versus-MNDWI disagreement idea where it can help: choosing training data, not adapting at test time.

**Expected outcome.** Uncertain (a low-to-moderate prior), but cheap, and it directly tests R1.6's worry about the filter. If it lowers recall, it is reported as a trade-off.

## 5. Decisions for the authors

1. **Label budget wording.** Using WorldFloods-v2 train/val labels to set one threshold changes the "no target labels" story to "one threshold set on an independent public archive". The paper must say so (R1.5 terminology). If you don't want any external labels in the method, run Idea 1 as an analysis only and keep the deployed threshold.
2. **Which student is "the" student.** The probe and Idea 2 use R4 (12k parameters, attention-free and therefore fully INT8-able, which helps with R2.6). The paper's student is EfficientViT-B0. The edited prompt reads both; decide which the revised paper leads with.
3. **R1.3.** The prompt now records your clarification: STURM did not determine OFA/EfficientViT, and was used for null saliency/OOD criteria. That closes most of R1.3; the remaining wording is a writing task.
4. **Deadline.** J-STARS allows 28 days for major revisions. Check the decision date. The overnight run plus the writing must fit in that window.

## 6. What changed in the prompt (change log)

- **Kept byte-identical:** §2.1, §2.2 and §9 (scheduler); the scheduler-test list in §8; the cap and clock rules in §3.
- **Rewritten:**
  - title and mission (§1);
  - retirement paragraph (§2): head adaptation is retired with reasons;
  - §3 ceilings and priorities: label swap 0.65, hard negatives 0.90, teacher scores 0.50, B0 0.15, PTQ 0.20, reserve 0.60 GPU-h; conditional items from freed budget; CPU work starts at once;
  - §4 and §5: replaced by Idea 1 and Idea 2;
  - §7: new decision rules (C is the only confirmatory set; claims need a +1.0-point gain with lower bound > 0 in ≥ 3 of 4 seeds, plus an alarm-rate check);
  - §8: adapted targeted checks;
  - §10: deliverables and reviewer mapping;
  - §11: prior art.
- **Added:**
  - §2.5: the probe facts and exposure statement;
  - the reference-dry-window readout in §2.4;
  - the scheduling note in §2.3;
  - the threshold-refit check for quantized parents in §6.

## 7. Reproduce

```
experiments/revision/novelty_probe_20261012/run_all.sh
```

Reports: `probe_report.txt` (fusion rules), `oprules_report.txt` (label-free thresholds), `fewshot_report.txt` (one-chip/click calibration), `predictability_report.txt` (threshold predictability, average precision).

Per-rule CSVs: `probe_summary.csv`, `probe_selected.csv`, `probe_events.csv`, `oprules_summary.csv`, `oprules_events.csv`.

The probe never reads the WorldFloods-v2 train/val splits.
