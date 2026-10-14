# FloodDistill · JSTARS major revision — wrap-up of every reviewer concern

**Manuscript:** JSTARS-2026-05386, *Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation*  
**Date of this wrap-up:** 12 October 2026  
**Inputs read:** `paper/main.pdf`, `paper/supplement.pdf`, `paper/review.docx` (the decision letter), all of `experiments/revision/` (prompts 1–5 reviewer programme, compression v2, capacity v4/v5, v7, v8) and the paper's source data.  
**New material made for this wrap-up:** 11 figures and a verification script in [reviewer_wrapup_20261012/](reviewer_wrapup_20261012/). `run_all.sh` rebuilds everything from existing files, CPU only, without changing any input.

---

## How to read this document

The document is written so that someone outside remote sensing and machine learning can follow it. Every concern gets the same structure:

1. **What the reviewer wrote** (verbatim quote).
2. **What it means in plain words.**
3. **The questions we have to answer** (a small logic tree).
4. **What we did** (experiments, described step by step, with numbers and figures).
5. **Verdict:** does our evidence answer the concern?
6. **What to change in the paper** and a **draft answer** for the response letter.
7. **What is still open** and how to close it.

Evidence tags used for important numbers:

| Tag | Meaning |
|---|---|
| **[V]** | Re-derived for this wrap-up from a raw file (log, per-scene counts, pool statistics). |
| **[R]** | Taken from an earlier audited revision table or report, not recomputed today. |
| **[J]** | A judgement or recommendation, not a measurement. |

Verdict labels: **Answered** (our evidence answers the question and survives our own checks) · **Partly** (evidence exists, but a careful reviewer could still point at a gap) · **Open** (needs new data, a measurement, or an author decision).

---

## Contents

0. [Summary on one page](#0-summary-on-one-page)
1. [The study in plain words, and a glossary](#1-the-study-in-plain-words-and-a-glossary)
2. [What the editor and reviewers asked](#2-what-the-editor-and-reviewers-asked)
3. [What has been done since the review](#3-what-has-been-done-since-the-review)
4. [The logic tree used to check each concern](#4-the-logic-tree-used-to-check-each-concern)
5. [Concern by concern](#5-concern-by-concern)
6. [Problems we found that the reviewers did not raise](#6-problems-we-found-that-the-reviewers-did-not-raise)
7. [Everything that is still open](#7-everything-that-is-still-open)
8. [Proposed Discussion section (editor's request)](#8-proposed-discussion-section-editors-request)
9. [Decisions only the authors can make](#9-decisions-only-the-authors-can-make)
10. [Sources and reproduction](#10-sources-and-reproduction)

---

## 0. Summary on one page

**The decision.** Major revision. Both reviewers like the topic, the amount of experiments and the real on-device test. Their main doubts concern novelty, how far the comparisons can be trusted, and whether the small model is practically better than a simple water index.

**Overall state.** Most concerns can be answered with experiments that have already been run, and none of the new evidence overturns a main claim of the paper. Several claims must become narrower: OFA is not a contribution, external benchmarks do not show superiority over MNDWI, and the 8-bit model is mixed INT8/FP16. Re-checking raw files for this wrap-up also turned up numerical errors in the paper that must be fixed (Section 6).

| ID | Concern, in short | Does our data answer it? | Status | What is still missing |
|---|---|---|---|---|
| Editor | Add a Discussion section | Yes, the content exists | Writing | Write it (outline in Section 8) |
| R1.1 / R2.1 | What is new beyond known parts? | Partly: data cannot create novelty | Writing + author decision | A precise list of what is new; drop "OFA (ours)" |
| R1.2 | Only one teacher and one student | Students: yes. Teacher: no | Mostly answered | Optional second-teacher control (~2 GPU-h) |
| R1.3 | Which data trained, selected, tested? Leakage? | Yes, except the development history | **Needs author input** | Which choices were influenced by STURM/WorldFloods scores |
| R1.4 | Ablations are scattered | Yes | Answered | One table + figure; corrected numbers |
| R1.5 | Inconsistent terms | Yes (writing) | Writing | Terminology pass (table in R1.5) |
| R1.6 | Dry scenes removed, false alarms untested | Acknowledgement: yes. Measurement: no | Partly | Independent dry-scene test (optional, valuable) |
| R2.2 | S11 vs GDACS comparison confounded | Partly | Partly | Same-image label-swap control (~0.5 GPU-h) |
| R2.3 | OFA advantage small | Yes, we agree | Answered | Writing |
| R2.4 | OFA not tested on other students | Yes | Answered | Writing |
| R2.5 | When is the model better than MNDWI? | Yes | Answered | Writing |
| R2.6 | Hardware-specific; mixed INT8/FP16 | Acknowledgement: yes. Strict INT8: no | Partly | Optional strict-INT8 and second-device test; fix 3.95 W |

**The five most important next steps** [J]:

1. **Author answer for R1.3:** which method choices were made after looking at STURM or WorldFloods scores? Without it, the leakage answer cannot be completed.
2. **Fix the numerical errors** in Section 6: the teacher row, Table IV power, the supplement's pre-filter claim, the crop count, and the false-alarm wording.
3. **Rewrite the contribution, OFA and MNDWI parts** using the evidence in R1.1, R2.3 and R2.5 (Figures 2, 6, 7, 8).
4. **Write the new Discussion section** (Section 8) and the point-by-point response (draft answers in Section 5).
5. **Optional but strong:** the same-image label-swap control (R2.2), an independent dry-scene test (R1.6) and a second-teacher control (R1.2). Together they need about 3 GPU-hours plus one to two person-days.

---

## 1. The study in plain words, and a glossary

**The problem.** Satellites photograph floods. A computer program can colour every pixel of such an image as "water" or "not water". This is called *segmentation*. The best programs today are huge *foundation models* with hundreds of millions of adjustable numbers (*parameters*), pre-trained on large amounts of satellite imagery. They are too big for small computers in the field or on a satellite.

**The idea of the paper.**

1. Take a large, already trained model, the **teacher**: Prithvi-EO-2.0 with 300 million parameters. It was fine-tuned by IBM/NASA on 252 hand-labelled images from the *Sen1Floods11* dataset.
2. Let the teacher label new satellite images automatically. These come from 23,720 image tiles around floods listed by the disaster service GDACS; nobody draws on them by hand. The teacher's automatic labels are called **pseudo-labels**.
3. Clean and choose a subset of those tiles: a **pre-filter** removes useless tiles, and a diversity rule called **FPS** picks 2,500 of them.
4. Train a **student** 430 times smaller (EfficientViT-B0, 0.7 million parameters) to imitate the teacher. This imitation is called **knowledge distillation**.
5. Shrink the student further to 8-bit numbers (**quantization**) and run it on a small embedded computer (NVIDIA **Jetson Xavier NX**): it needs 5.6 ms per image and a 1.5 MB file.

**The main results as submitted.**

- With the same number of training images (252), teacher-labelled images train the student about as well as human-labelled ones, and better on one external test set.
- With 2,500 teacher-labelled images the student gets close to the teacher.
- A one-line formula, the water index **MNDWI**, is almost as good on the external test sets.

```
 252 hand-labelled images ──► Teacher (300 M) ──labels──► 23,720 GDACS tiles
                                                            │ pre-filter (5–85 % water, ≥0.05 confidence spread)
                                                            ▼
                                                  5,987 candidates ─FPS─► 2,500 training tiles
                                                            │
                         teacher's soft answers (live) ─────┤
                                                            ▼
                                             Student (0.7 M) ──ReLU6 + QAT──► INT8/FP16 engine on Jetson
```

### Glossary

| Term | Plain meaning |
|---|---|
| **Teacher / student** | Big accurate model / small fast model that learns by copying the teacher. |
| **Pseudo-label (teacher label)** | A water map written by the teacher, not by a human. Also called a *hard* label: each pixel is either water or not. |
| **Soft target** | The teacher's confidence per pixel (e.g. "72 % sure it's water"). It carries more information than the hard yes/no. |
| **Hard-label training (CE / focal loss)** | The student learns only from hard yes/no labels. *CE* (cross-entropy) and *focal loss* are two standard formulas for this. |
| **Logit KD (knowledge distillation)** | The student also copies the teacher's soft targets. |
| **OFA ("One-for-All")** | Logit KD plus small extra output branches attached inside the student during training, removed afterwards. |
| **Sen1Floods11 (S11)** | The dataset the teacher was trained on; 252 training, 89 validation, 90 test images. Its "water" includes rivers and lakes, not just flood water. |
| **STURM-Flood, WorldFloods-v2 (WF)** | Two *external* test sets from other floods (29 and 11 flood events), used to see whether the model works outside the data it learned from. |
| **MNDWI** | Modified Normalised Difference Water Index: (green − shortwave-infrared)/(green + shortwave-infrared). Pixels above 0 are called water. No training, no parameters. |
| **IoU** | Intersection over union: overlap between predicted and true water, divided by their combined area. 100 % is perfect. *Water IoU* scores the water class; *mIoU* averages water IoU and land IoU. |
| **Precision / recall** | Precision: of the pixels called water, how many really are (low precision = false alarms). Recall: of the real water, how much is found (low recall = misses). |
| **Point (pp)** | A percentage point, e.g. 75 % → 76 % is +1 point. |
| **Seed** | Training contains randomness. Each model was trained 4 times (seeds 42, 64, 91, 107). A difference smaller than the seed-to-seed spread (usually 0.2–0.8 points) is not reliable. |
| **Pooled vs per event** | *Pooled*: count all pixels of a test set together, so big floods dominate. In WorldFloods one flood (EMSR342) holds 82 % of all water pixels [V]. *Per event*: score each flood separately and average, so every flood counts equally. |
| **Event** | One flood disaster (one emergency-mapping activation). Images from the same event are not independent. |
| **INT8 / FP16 / FP32** | Number formats with 8, 16 and 32 bits. Fewer bits mean a smaller, often faster model, possibly less accurate. |
| **QAT** | Quantization-aware training: fine-tuning a model while it "experiences" 8-bit rounding, so it learns to cope with it. |
| **TensorRT engine** | The model compiled for one specific NVIDIA device. |
| **Validation vs test** | Validation images are used to pick the best training checkpoint (allowed). Test images must only be used for the final score. |

---

## 2. What the editor and reviewers asked

The letter recommends the paper *"for publication subject to satisfactory response to major revisions"*. The full text is in `paper/review.docx`. Earlier revision phases worked from a paraphrase because the letter was not yet in the repository. The paraphrased items match the real letter, except that Reviewer 2's closing deployment remark was unnumbered; it is called **R2.6** here.

**Formal requirements (editor):**

- A clean revised manuscript, plus a version with changes highlighted in colour.
- A point-by-point response letter as a separate file, stating for every comment what was changed and where, and justifying anything not done.
- The corresponding-author information in the author footnote. The submitted PDF already has it; keep it.
- **Return within 28 days** of the decision e-mail. The decision date is not in `review.docx`; check the e-mail.
- **"A 'Discussion Section' is suggested … more likely to get accepted."**

**Reviewer 1 (verbatim, abbreviated only by numbering):**

- **R1.1** "The novelty over existing knowledge distillation and pseudo-labeling frameworks should be explained more clearly. OFA, pseudo-label generation, subset selection, and quantization are established techniques, so the authors should better highlight what is technically new in their overall framework."
- **R1.2** "The model comparison is still somewhat limited. The study mainly relies on Prithvi-EO-2.0 as the teacher and EfficientViT-B0 as the primary student. A clearer discussion or comparison with other recent lightweight remote-sensing segmentation/distillation methods would strengthen the evaluation."
- **R1.3** "The evaluation protocol needs further clarification. Since the Sen1Floods11 validation set is used for checkpoint selection and quantization calibration, while public benchmarks were also involved during method development, the authors should clearly state which data were used for training, model selection, and final evaluation to avoid possible information leakage."
- **R1.4** "The ablation analysis could be presented more clearly. The separate contributions of pseudo-label filtering, FPS-based selection, Logit-KD/OFA, activation replacement, and QAT are distributed across the main paper and supplementary material, making it difficult to identify the contribution of each component."
- **R1.5** "The terminology should be checked carefully for consistency. In particular, the manuscript alternates among 'flood segmentation,' 'water segmentation,' 'teacher-supervised,' and 'pseudo-label' settings, although the authors acknowledge that Sen1Floods11 essentially defines a surface-water segmentation task."
- **R1.6** "The deployment discussion should more clearly acknowledge that the pseudo-label filtering removes fully dry scenes. As a result, false alarms on ordinary water-free imagery are not sufficiently evaluated, which may affect practical unattended flood-monitoring applications."

**Reviewer 2:**

- **R2.1** "The methodological novelty should be clarified, since most components are adaptations or combinations of existing techniques."
- **R2.2** "The comparison between Sen1Floods11 and pseudo-labeled GDACS data is partially confounded by differences in imagery, acquisition geometry, and geographic diversity."
- **R2.3** "The advantage of OFA over standard Logit-KD is relatively limited and disappears at the larger training budget."
- **R2.4** "The student architecture comparison is not fully controlled because OFA is not evaluated with alternative compact backbones."
- **R2.5** "The strong MNDWI baseline raises an important question regarding when the learned model provides meaningful practical advantages over a simple spectral-index approach."
- **R2.6** "Finally, the deployment results are hardware-specific, and the model performs mixed INT8/FP16 rather than strictly INT8 inference."

---

## 3. What has been done since the review

| Phase (folder) | Dates | What was done | Result in one line |
|---|---|---|---|
| Prompts 1–2 (`prompt2_*`) | 5 Oct | Local baselines; architecture controls set up; data-role audit; MNDWI checks; search for dry test scenes | Infrastructure ready; no independent dry scenes in the data |
| Prompt 3 (`prompt3_*`) | 5–6 Oct | 60 training runs on a *regenerated* pool (127 tiles had to be replaced); fixes to the WorldFloods border handling and STURM invalid pixels; 11 dry candidate windows prepared for human review | Matched controls exist; dry review never completed |
| Prompt 4 (`prompt4_report_20261007/`) | 6–7 Oct | Original training data fully restored (2,500 tiles incl. the 127 missing); 28 original checkpoints re-scored; **44 new training runs on the original data** (matched KD/OFA for 3 students, a 100-image budget, a hard-label control); robustness and per-event statistics | Paper's student numbers reproduce within 0.02 points [R]; 9.6 GPU-h |
| Prompt 5 (`prompt5_closure/`) | 7 Oct | Reviewer closure package: data-role table, ablation map, footprint-overlap check with exclusion test, draft text edits M1–M16 / S1–S8, draft response | Drafts exist, but were written without the real letter |
| Compression v2 (`compression_v2_20261007/`) | 7–8 Oct | One-seed pilot of a new feature-distillation idea for low-bit students | Idea not supported |
| Capacity v4/v5 (`capacity_v4_20261008/`) | 8–11 Oct | Fixed a measurement bug in low-bit training; tested 4-bit; **ladder of student sizes from 14 to 705,000 parameters**; hard vs soft vs human labels | 4-bit weights lossless (simulated); a 12k-parameter model competitive; 21 GPU-h |
| v7 (`capacity_v7_20261011T210730Z/`) | 11–12 Oct | Better learning rate for the baselines; teacher-vs-MNDWI disagreement analysis; a new "spectral rule + selective distillation" idea; desktop timing | New idea failed its pre-registered test; useful analyses |
| v8 (`research_v8_20261012T123202Z/`) | 12 Oct | OFA learning curves; teacher "headroom" analysis; one-image adaptation pilot | Adaptation idea did not help; OFA curves useful |
| **This wrap-up** (`reviewer_wrapup_20261012/`) | 12 Oct | 11 new figures; re-verification of the paper's tables against raw logs and counts | Found the errors listed in Section 6 |

Total compute of the follow-up capacity programme: about 26 GPU-hours of a 30-hour allowance [R]. The reviewer programme (prompts 1–5) added about 10 GPU-hours of training plus evaluation [R].

---

## 4. The logic tree used to check each concern

Every reviewer item was sorted by the kind of answer it needs, then checked against the evidence.

```
Decision letter (13 items)
│
├── A. "Explain it better" ............................. answered by writing
│   ├── R1.1 / R2.1  novelty ............................ PARTLY (honest framing; novelty is modest by nature)
│   ├── R1.5         terminology ........................ writing only, not yet done
│   └── Editor       Discussion section ................. outline ready (Section 8)
│
├── B. "Show us" ........................................ needs evidence
│   ├── R1.2         other students / methods / teachers
│   │   ├── other students ............................. ANSWERED (3 backbones × 2 losses × 4 seeds; size ladder)
│   │   ├── other distillation methods ................. ANSWERED with generic methods (6+ variants, none better)
│   │   └── other teachers ............................. OPEN (only Prithvi tested)
│   ├── R2.4         OFA on other students ............... ANSWERED
│   ├── R2.3         OFA advantage small ................. ANSWERED (we agree; matched controls)
│   ├── R1.3         data roles / leakage
│   │   ├── roles of every dataset ...................... ANSWERED (table)
│   │   ├── geographic/temporal overlap ................. ANSWERED (3 chips + 1 map; results unchanged)
│   │   └── were choices tuned on test sets? ............ OPEN → author must answer
│   ├── R1.4         ablations in one place .............. ANSWERED (Fig. 3, 4) + corrections
│   ├── R2.2         S11 vs GDACS confound
│   │   ├── geometry control ............................ done, but itself confounded
│   │   └── same images, swap only the labels ........... OPEN (cheap experiment)
│   └── R2.5         when better than MNDWI .............. ANSWERED (Fig. 6, 7, 8 + labeller control)
│
└── C. "Acknowledge the limit" .......................... writing, optional measurement
    ├── R1.6  dry scenes .................................. acknowledgement ready; measurement OPEN
    └── R2.6  hardware-specific, mixed INT8/FP16 .......... acknowledgement ready; strict INT8 OPEN
```

For each leaf, the test was: *Can we write one paragraph that answers the reviewer's literal question with a number from a controlled comparison, without claiming more than that comparison shows?* If yes: Answered. If the paragraph needs "but we did not test …": Partly. If we cannot write it at all: Open.

---

## 5. Concern by concern

### 5.0 Editor — add a Discussion section

**What it means.** The editor wants a section that steps back from the numbers: what the results mean, how they compare with other approaches, where they stop being valid.

**State.** A draft exists (`prompt5_closure/main_manuscript_proposed_edits.txt`, item M15). It predates the real letter and the analyses of 11–12 October. Section 8 below gives an updated outline that ties every paragraph to a reviewer concern.

**Verdict:** Writing task; content is ready.

---

### 5.1 R1.1 and R2.1 — What is technically new?

**In plain words.** "Each building block (teacher-student training, automatic labels, choosing a subset, 8-bit compression) is already known. What is new about the combination?"

**Questions to answer.**
1. Is any single component new?
2. Is the combination or setting new, and does it produce knowledge that did not exist before?
3. Did we try genuinely new mechanisms, and what happened?

**What the evidence says.**

1. **No single component is new, and the paper should say so openly.** Pseudo-labelling, logit distillation (Hinton 2015), OFA (Hao et al. 2023), farthest-point sampling (k-center coverage) and QAT are all prior art. The only component-level adaptation is OFA for dense per-pixel output with a ViT→CNN pair, which turned out not to matter (R2.3).
2. **The setting and the controlled findings are new** [J]. The closest prior work is InstaGeo, which distils foundation models into smaller versions of the *same* architecture for cloud serving, and Garg et al., who distil across sensors (optical to radar). Neither does:
   - a **430× cross-architecture compression** of a geospatial foundation model, measured **on an embedded device**;
   - a **matched-budget experiment**: the same 252-image annotation budget used directly, or routed through a teacher onto new images (Table I);
   - a **label-free audit of what the student inherits**: on pre-event images, the student copies the teacher's habit of marking pre-existing water (rivers, lakes, seasonal water) on 96 % of the images the teacher flagged, whether or not those images were in training [R];
   - a **systematic map of where the foundation model helps over a spectral index**: precision, haze, difficult floods, and its role as a labeller (Figures 6–8, R2.5).
3. **We tried three new mechanisms during the revision. All failed their pre-registered success tests:**
   - an "interaction" feature-distillation metric for low-bit students (compression v2: +1.0 point over plain KD but −0.3 below plain feature matching on the hard case; needed +0.5 over both);
   - spectral-rule residual plus disagreement-weighted distillation (v7: the selected checkpoint never moved from its starting point; final checkpoints worse than controls);
   - one-solve adaptation of the student's last layer to a new flood (v8: the proposed variants lowered accuracy on average by 0.3–0.8 points; the simple controls moved it by less than 0.1).

   They must not be claimed. Mentioning them in one sentence ("we tested … and none improved over plain logit KD") strengthens credibility [J].

**Verdict: Partly.** No experiment can make the method more original. The answer is an honest re-framing: an application and systems study with unusually careful controls. Some reviewers will still find the novelty modest; that is a judgement we cannot control.

**What to change.**

- Rewrite the contribution bullets (Section I) as:
  1. annotation-budget amplification with a frozen foundation-model teacher, tested at a matched budget and at 10× scale;
  2. a controlled transfer study (student architectures, objectives, data selection) that identifies which parts matter, namely data scale and soft targets, and which do not (OFA, selection rule after filtering);
  3. a measured target-specific deployment (mixed INT8/FP16 on Xavier NX), with its limits;
  4. a label-free audit of inherited teacher semantics.
- Add a small table to Related Work: *component → prior work → what we adapt*.
- Remove "(ours)" from the OFA row of Table II; OFA is a training detail, not a contribution.

**Draft answer.** *"We agree that the individual components are established, and the revised manuscript says so explicitly (new table in Section II). We now state the contribution as an application study: (i) routing a fixed manual annotation budget through a frozen foundation-model teacher onto new imagery, evaluated at a matched budget and at 10× scale; (ii) controlled comparisons that identify which parts of the pipeline matter, now including three compact backbones (R1.2/R2.4); (iii) a measured mixed-precision deployment on embedded hardware; (iv) a label-free audit showing which teacher conventions the student inherits. We no longer present OFA as a contribution (see R2.3)."*

**Still open.** Author decision on whether the size-ladder finding (a 12k-parameter attention-free student is competitive, Figure 9) enters the paper as an additional finding. It would strengthen novelty but widen the scope.

---

### 5.2 R1.2 — Only one teacher and one student

**In plain words.** "You tried one big model and one small model. Compare with other recent small models or distillation methods."

**Questions.** (a) Do other students give different conclusions? (b) Do other distillation methods do better? (c) Would another teacher change the picture?

**(a) Other students: a matched comparison on the original data.**

- **Setup:** three students trained on the same 2,500 teacher-labelled tiles, with the same frozen teacher, the same recipe, and checkpoints chosen on S11 validation only. Four seeds each, with logit KD and with OFA:
  - **EfficientViT-B0**: 0.7 M parameters, the paper's student;
  - **MobileNetV3-Small**: 3.5 M parameters, a classic mobile network;
  - **LightFormer-ResNet18**: 11.4 M parameters; a lightweight remote-sensing segmentation model published in 2026 (Int. J. Appl. Earth Obs. Geoinf.). Its own auxiliary loss was switched off, so that all students are trained the same way.
- **Result (Figure 1):** with logit KD or OFA, the six students differ by at most **1.3 points** on the external sets. STURM spans 75.8–76.1 mIoU and WorldFloods 75.8–77.1 [R]. In-distribution, the larger MobileNet is slightly better (88.6 vs 87.8 mIoU with KD). Bigger is not better outside the training distribution; the paper's UNet-ResNet50 (32.5 M) shows the same.
- **Even smaller students (Figure 9).** The follow-up capacity study trained eight model sizes from 14 to 705,000 parameters, from scratch with a learning rate chosen on S11 validation. **R4**, a 12,230-parameter convolutional network without attention, is 57× smaller than EfficientViT-B0. It reaches 82.5 % water IoU on S11 test (teacher 81.7), 64.7 on STURM (teacher 64.9), and 47.6 per event on WorldFloods, the same as the paper's student (47.6; teacher 50.5) [R].

![Three compact students](reviewer_wrapup_20261012/figures/fig01_architectures.png)

*Figure 1. Matched architecture comparison (data: `figure_data/fig01_architectures.csv`).*

![Size ladder](reviewer_wrapup_20261012/figures/fig09_size_ladder.png)

*Figure 9. Student size from 14 to 705,000 parameters (data: `figure_data/fig09_size_ladder.csv`). The ladder uses a different recipe (scratch-trained ReLU6 graphs, logit KD, learning rate chosen on S11 validation) from the paper's student (blue diamond).*

**(b) Other distillation methods.** Across the revision, the following were tested against plain logit KD:

- hard labels only (CE or focal loss);
- OFA;
- intermediate feature matching (MSE);
- a feature-interaction metric;
- a teacher-assistant chain (an ensemble of UNet-ResNet50 models re-labelling the pool);
- gradient surgery (PCGrad) on the OFA branches;
- disagreement-weighted, entropy-weighted and spectral-rule-residual distillation;
- cached vs live teacher targets.

In full-precision training none beats plain logit KD by more than the seed noise [R]. The one exception is a single-seed 4-bit pilot, where feature matching helped by about 1 point (compression v2). What these did *not* include is a dedicated remote-sensing distillation method from the literature (Reviewer 1 mentions "recent … distillation methods").

**(c) Other teachers.** Only Prithvi-EO-2.0 was used. The v8 run checked whether a second foundation model (TerraMind) could serve as a teacher, but no task-adapted water head was available, so it was deferred [R]. The teacher-assistant UNet ensemble is not an independent teacher, because it was trained on Prithvi's labels.

**Verdict: mostly answered.** Students: answered. Distillation methods: answered with generic methods. Teachers: **open**.

**What to change.** Add the matched three-backbone table and Figure 1 (Section IV-C or the supplement), with parameter counts:

| Student | Inference parameters | Training-only OFA heads |
|---|---|---|
| EfficientViT-B0 | 0.70 M | 11,368 |
| MobileNetV3-Small | 3.55 M | 168,792 |
| LightFormer-ResNet18 | 11.43 M | 176,008 |

State that all were trained with one shared recipe, which is not each backbone's tuned optimum. Optionally add the size ladder as a supplementary figure, labelled as a different (scratch) recipe.

**Draft answer.** *"We added two compact backbones, MobileNetV3-Small and the 2026 remote-sensing model LightFormer-ResNet18, trained on the identical 2,500 teacher-labelled scenes with the same recipe, each with logit KD and OFA, four seeds (new Table X, Fig. Y). Outside the training distribution all six students lie within 1.3 mIoU points of each other; the larger backbones do not transfer better. We also report that intermediate-feature matching, a teacher-assistant chain and several weighted-distillation variants did not improve over logit KD. We acknowledge that a single teacher is a limitation (Section VII) [and, if run: a conventional supervised teacher control shows …]."*

**Still open — recommended control (~2 GPU-h) [J].** Use the existing UNet-ResNet50 trained only on the 252 S11 human labels as an alternative teacher. Let it label the same 2,500 GDACS tiles, train EfficientViT-B0 (hard labels and logit KD, 4 seeds) and compare with the Prithvi-taught student. This tests directly whether a *foundation model* is needed as the teacher, the question the paper currently leaves open (Section III-A "Choice of teacher").

---

### 5.3 R1.3 — Which data were used for what, and could test information have leaked?

**In plain words.** "Say exactly which images were used to train, to choose the best checkpoint, to calibrate the 8-bit model, and to score the final results. Make sure the test images did not influence anything else."

**Questions.**
1. What is the role of every dataset?
2. Do training images overlap with test images in place or time?
3. Were method choices made after looking at test scores?

**1. Data roles (verified from code, configs and run receipts [R]):**

| Role | Data used | Labels used? |
|---|---|---|
| Teacher fine-tuning (done by IBM/NASA, not by us) | 252 S11 training images | human labels |
| Student training | 2,500 GDACS tiles | teacher hard labels + live teacher soft targets |
| S11 control arms (Table I) | 252 S11 training images | human hard labels + live teacher soft targets |
| Input normalisation | S11 training images (band means and spreads) | no |
| Pre-filter and FPS selection | GDACS pool only (teacher outputs, image statistics) | no |
| Checkpoint selection | 89 S11 *validation* images, every 2 epochs, best mIoU | human labels |
| 8-bit calibration and QAT checkpoint choice | S11 validation images | images for calibration; labels for checkpoint choice |
| **Final scores** | S11 test (90 images, 10 events), STURM (2,675 chips, 29 events), WorldFloods (18 scenes, 11 events) | human labels, only for scoring |

In all revision runs, no test image was used for training, checkpoint selection or calibration.

**2. Overlap in place and time.** For each of the 2,500 training tiles, the actual footprint was intersected with every test scene [R]:

- zero overlap with S11 validation, test, or the Bolivia hold-out;
- 3 STURM chips and 1 WorldFloods map overlap training tiles, but **on different dates** (12 days to 2.6 years apart); no image appears in both training and test;
- **exclusion test:** removing those 3 chips and that 1 map from all evaluations and re-scoring every model changes almost nothing.
  - Student OFA on STURM: 76.026 → 76.031.
  - Teacher on STURM: 75.392 → 75.395.
  - WorldFloods: OFA 77.13 → 77.00; teacher 77.17 → 77.03.
  - All OFA-vs-KD differences move by ≤0.02 points [R].

**3. Choices made after looking at test scores.** This cannot be settled from code. The submitted paper itself says (Section VII): *"Method choices were refined using the same public benchmarks reported in the paper."* An old launcher script also cites external scores as the reason for choosing FPS [R]. Only the authors know which choices were made before and which after seeing STURM/WorldFloods numbers. Candidates: filter thresholds, the 2,500 budget, OFA, the 512-pixel evaluation size, the MNDWI threshold.

**Also relevant.** The original training data were restored in full: all 2,500 tiles, including 127 that had gone missing. The paper's 28 original student checkpoints re-score to within 0.02 points of the published numbers [R]. The **teacher row**, however, does not fully reproduce (Section 6, item 1).

**Verdict: Answered for data roles and overlap; open for development history (needs author input).**

**What to change.**

- Add the data-role table to Section IV-A.
- Update Supplement S1 with the footprint check and the exclusion test.
- Replace the vague limitation sentence with a precise one listing which choices were fixed before any external evaluation and which were revised afterwards.
- Note that Table I's exact 252-tile membership could not be re-traced to its run records [R].

**Draft answer.** *"We added a table listing the role of every dataset (Section IV-A). In brief: students are trained on teacher-labelled GDACS tiles only; S11 validation is used for checkpoint selection and 8-bit calibration; S11 test, STURM-Flood and WorldFloods-v2 are used only for the final scores. A footprint-level check found no shared acquisitions between training and test data; three STURM chips and one WorldFloods map overlap training tiles geographically on different dates, and excluding them changes no reported conclusion (Supplement S1). As stated in Section VII, [list of choices] were revised after inspecting external benchmark results; we therefore present these benchmarks as development-exposed and do not claim an untouched hold-out."*

**Still open.** **Author input (blocking):** a factual list of which design choices were influenced by STURM or WorldFloods scores. The longer-term remedy, a genuinely untouched event set, belongs to future work.

---

### 5.4 R1.4 — The contribution of each component is hard to see

**In plain words.** "Show in one place how much each step of the pipeline (filtering, FPS selection, distillation, OFA, activation change, quantization) contributes."

**Questions.** For each component: what was compared, what was held fixed, how big is the effect, and is the comparison fair?

**What we did.** Every comparison that exists was collected into one picture (Figure 3). The selector study was re-computed from its raw per-scene counts (Figure 4). Each row is a separate experiment, so the rows **must not be added up**. Filled dots are fair one-factor comparisons; hollow dots changed other things too.

![Component effects](reviewer_wrapup_20261012/figures/fig03_component_effects.png)

*Figure 3. Effect of each pipeline step (data: `figure_data/fig03_component_effects.csv`).*

| Component | Comparison | Effect on mIoU (S11 / STURM / WF) | Fair? | Reading |
|---|---|---|---|---|
| Teacher labels instead of human labels | 252 GDACS teacher-labelled vs 252 S11 human-labelled, OFA | −0.1 / **+1.7** / −1.0 | No: images, places, tile size differ | Competitive substitute; helps on STURM only |
| Same, S11 cropped to GDACS tile size | geometry-matched control | n/a / +1.7 pooled (+0.6 per flood map) / −0.8 | No: the crop deletes 74 % of S11 water | See R2.2 |
| 10× more teacher-labelled scenes | 252 → 2,500 | **+2.6** / +0.3 / +0.4 | Yes | Scale helps most in-distribution |
| Pre-filter | with vs without, per selector | STURM: FPS +0.1, random +0.3, GLISTER +4.0, confidence ranking **+31** | Yes | A safety net for weak selection rules (Fig. 4) |
| FPS instead of random | both after the filter | STURM +0.2 | Yes | No measurable gain |
| Soft teacher targets (logit KD) | vs hard teacher labels only | +0.7 / **+1.0** / **+1.2** | Mostly: the hard-label weight also halves | The most useful objective change |
| OFA side branches | vs logit KD | −0.3 / 0.0 / +0.4 | Yes | No consistent gain (R2.3) |
| Hardswish → ReLU6 | retrained from scratch | n/a / −0.8 / +0.8 | No: also loses ImageNet initialisation | Explained by a too-small learning rate (below) |
| 8-bit engine on Jetson | vs float export, same device | n/a / −1.1 / −1.6 | Yes | Seed-dependent (R2.6, Fig. 10) |

![Filter and selector](reviewer_wrapup_20261012/figures/fig04_filter_selector.png)

*Figure 4. Pre-filter × selection rule, recomputed from raw per-scene counts [V] (data: `figure_data/fig04_filter_selector.csv`).*

**Two new findings from this re-check.**

1. **The pre-filter matters only for weak selection rules [V].** Without the filter:
   - confidence ranking collapses from 75.6 to 44.4 mIoU on STURM, because it then picks tiles that are almost all land or all water;
   - GLISTER loses 4.0 points;
   - FPS loses only 0.1 and random selection 0.3.

   So the filter and FPS are partly redundant safeguards. The supplement's sentence *"removing it costs up to 0.048 mIoU clean"* matches none of these numbers and must be corrected.
2. **The "activation replacement" cost is a training-recipe effect, not an activation effect [R].** The ReLU6 student is trained from scratch with the learning rate of the pretrained model (5·10⁻⁵).
   - At that rate, a from-scratch EfficientViT-B0 reaches 74.8 mIoU on STURM, the same as Table III's ReLU6 row (74.8).
   - With a learning rate chosen on S11 validation (5·10⁻³), the same from-scratch ReLU6 network reaches **76.2 on STURM, 77.1 on WorldFloods and 87.8 on S11**. That equals or beats the ImageNet-pretrained Hardswish student (76.0 / 77.1 / 87.5) (v7, logit KD, same protocols).

   The deployed model was therefore probably under-trained before QAT [J].

**Verdict: Answered**, with the honest caveat that this is a set of separate comparisons, not an additive breakdown.

**What to change.**

- Replace the scattered ablations with one "component table" in the main paper, using the columns above, plus Figure 3; move details to the supplement.
- Correct the supplement S2 pre-filter sentence.
- Re-describe the ReLU6 step: there is no inherent cost; the drop came from the learning rate.
- Correct the loss naming: the historical "CE" rows are true cross-entropy, while KD/OFA use a focal hard term; historical OFA used gradient clipping and KD did not. The revision's matched controls remove these differences [R].

**Draft answer.** *"We consolidated all component comparisons into one table and figure in the main text (new Table X, Fig. Y). Each row is a separate controlled comparison; rows are not additive. The largest effects come from scaling the teacher-labelled pool (+2.6 mIoU in distribution) and from adding soft teacher targets (+0.7 to +1.2). OFA adds no consistent gain. The pre-filter is essential for confidence- or GLISTER-based selection but changes FPS selection by only 0.1 mIoU. The apparent cost of the ReLU6 activation disappears when the from-scratch model's learning rate is chosen on the validation split."*

**Still open.** Writing only. Optional: retrain the deployed ReLU6 model with the better learning rate and re-run QAT and the Jetson chain. That would raise deployed accuracy, but it changes Table III and is not needed for the reviewers.

---

### 5.5 R1.5 — Inconsistent terminology

**In plain words.** "You switch between 'flood' and 'water', and between 'teacher-supervised' and 'pseudo-label'. Use one term for one thing."

**Why it matters.** Sen1Floods11's labels mark *all surface water*, including permanent rivers and lakes. The model therefore detects water, not new flooding. The pre-event audit confirms the student copies this: it agrees with the teacher on 96 % of pre-event images where the teacher marked water [R]. The external benchmarks remove permanent water before scoring, which makes them "flood-water" scores.

**Proposed terminology** [J]:

| Concept | Use | Avoid |
|---|---|---|
| The task the labels define | **surface-water segmentation** | "flood segmentation" for model outputs |
| The application | flood mapping / flood monitoring | — |
| Scoring after permanent-water removal | **flood-only protocol** (as in WF-fo) | — |
| Human masks of S11 | **human labels** | "ground-truth supervision" for training |
| Teacher's hard masks on GDACS | **teacher pseudo-labels** | mixing "teacher-supervised" and "pseudo" |
| Teacher's live probabilities | **soft teacher targets** | — |
| Students trained on teacher pseudo-labels (± soft targets) | **teacher-labelled configuration** (define once) | "teacher-supervised" without definition |
| Hard-label loss | name the formula: **cross-entropy** or **focal loss** | "CE" for a focal loss |
| Student "false alarm" on pre-event tiles | **student marks ≥ 5 % of the tile as water where the teacher marked < 5 %** (teacher-referenced, not ground truth) | "false alarm", "teacher left empty" (8,883 tiles have < 5 % predicted water; only some have none) |

**Verdict:** Answerable now, by writing. **Open** until the manuscript is edited.

**Draft answer.** *"We adopted one term per concept throughout (surface-water segmentation for the task the labels define; flood mapping only for the application; human labels, teacher pseudo-labels and soft teacher targets for the three supervision signals) and added a short terminology paragraph to Section IV-A."*

---

### 5.6 R1.6 — Dry scenes were filtered out, so false alarms are untested

**In plain words.** "Your filter throws away every image without water, so the model never learns from, and is never tested on, ordinary dry landscapes. In unattended monitoring it might raise false flood alarms."

**Questions.** (a) How much was removed? (b) What do we know about false alarms on dry images? (c) Can we get an independent measurement?

**(a) What the filter removed [V]** (Figure 5, recomputed from the stored statistics of every pool tile):

- **16,186 of 23,720 tiles (68 %)** were dropped because the teacher saw less than 5 % water;
- of these, **5,843** contain no predicted water at all, and **8,883** are pre-event images;
- 711 more were dropped for being over 85 % water and 836 for too uniform a teacher confidence;
- 5,987 candidates remain, and FPS picks 2,500. No dry tile can enter training.

![Pool funnel](reviewer_wrapup_20261012/figures/fig05_pool_funnel.png)

*Figure 5. Where the pre-filter removes images (data: `figure_data/fig05_pool_funnel.csv`).*

**(b) What we know about false alarms.**

1. **Teacher-referenced check (already in the paper):** on the 8,883 held-out pre-event tiles where the teacher sees < 5 % water, the student marks ≥ 5 % water on **1.9 ± 0.8 %** of them (4 seeds) [V: definition checked in `evaluation/relabel_feasibility.py`]. This is agreement with the teacher, not a true false-alarm rate: the tiles are not verified to be dry.
2. **Six human-labelled water-free S11 test chips [V]:** a diagnostic only, with one seed-42 model each, from events that also appear in training and validation:

   | Model | Chips with any false water | Worst chip (share of area) |
   |---|---|---|
   | Teacher | 1 of 6 | 0.2 % |
   | Student, OFA | 2 of 6 | 1.8 % |
   | Student, logit KD | 5 of 6 | **43 %** (one Paraguay chip) |
   | Student, hard labels | 3 of 6 | 4.9 % |
   | MNDWI | 2 of 6 | 11.7 % |

   False alarms on dry images do happen, and they differ a lot between models that score almost identically on the flood benchmarks. Six chips cannot measure reliability.
3. **Prepared but not completed:** 11 candidate dry windows from 3 WorldFloods events, selected without looking at any model output. They need a human to confirm each is dry (template: `prompt3_dry_verification/annotations.template.json`). The evaluation code and job definitions exist; once the confirmations are imported, scoring takes minutes [R].

**Verdict: Partly.** The reviewer asks to *acknowledge* the gap more clearly; that can be done now. A real dry-scene false-alarm rate is **open**.

**What to change.**

- Add a clear statement to Section VI (deployment), Section VII (limitations) and the conclusion:
  - the filter removes dry and nearly dry scenes (68 % of the pool);
  - neither benchmark is designed to measure false alarms on ordinary dry land (both are cut around flood extents);
  - the only measurements are teacher-referenced or very small;
  - unattended operation would need a dry-scene evaluation and probably dry training tiles.
- Fix the wording "teacher left empty" (see R1.5).

**Draft answer.** *"We agree. The revised deployment discussion and limitations state that the pre-filter removes all tiles with less than 5 % predicted water (16,186 of 23,720), so the student never sees dry landscapes, and that both external benchmarks are centred on flood extents. Our only dry-scene evidence is teacher-referenced (1.9 % of 8,883 near-dry pre-event tiles flagged) or very small (six water-free S11 chips, where false-water area varies strongly between models). Unattended monitoring therefore requires a dedicated dry-scene evaluation, which we list as future work [or: which we added in Supplement Sx]."*

**Still open — two options [J].**

- *Quick (about 1 hour of one person, minutes of GPU):* complete the 11-window review and run the prepared jobs. Small: 3 events.
- *Proper (1–2 person-days, under 1 GPU-hour):* assemble about 100 dry tiles from diverse landscapes, selected before running any model, with a human check against an independent reference (e.g., JRC "never water" plus visual inspection). Include deserts, cities, cropland, forest, mountain shadows, snow and cloud shadows. Report the share of tiles with any water, > 0.1 % water and > 1 % water for the teacher, the student and MNDWI. This would turn the limitation into a measured result.

---

### 5.7 R2.2 — The S11-versus-GDACS comparison is confounded

**In plain words.** "The 252 human-labelled S11 images and the 252 teacher-labelled GDACS images differ in more than who labelled them: different images, different sizes, different places. So the comparison does not show what the labels alone are worth."

**Questions.** (a) Which differences exist? (b) Did any control isolate the label source? (c) What can honestly be claimed?

**(a) Differences:**
- **footprint:** S11 chips cover 4.6 × 5.1 km; GDACS tiles 2.6 km;
- **geography:** S11 covers 11 flood events in 11 countries; the 2,500-tile set draws on 1,404 GDACS events;
- **acquisition and cloud handling;**
- **label source** — the one thing we would like to isolate.

**(b) Controls that exist.**

- **Geometry-matched crop** (Figure 11): S11 chips cut to the GDACS footprint, retrained with 4 seeds.
  - *All pixels pooled on STURM:* the teacher-labelled set still leads for all three training objectives (CE +1.0, KD +4.4, OFA +1.7 points).
  - *Averaged per flood map:* the lead shrinks to 0.0 (CE), +2.5 (KD) and +0.6 (OFA).
  - The crop is not clean either: it deletes **74 % of the S11 water pixels** and makes **21 chips completely dry** [R]. The paper wrongly says 86 chips lose more than half their water; it is 200.
- **Overlap exclusion** (R1.3): no change.

![Geometry control](reviewer_wrapup_20261012/figures/fig11_geometry_control.png)

*Figure 11. Matched-budget comparison with the geometry control (data: `figure_data/fig11_geometry_control.csv`).*

**(c) What is missing.** No experiment holds the *images* fixed and swaps only the *labels*.

**Verdict: Partly.** The paper already restricts the claim to "a competitive substitute at a fixed annotation budget, not label superiority". That restricted claim is supported. Separating label source from image content needs one more control.

**What to change.** Keep the restricted claim. Correct the crop numbers (200 chips, 73.6 %, 21 dry). Add Figure 11 and a sentence that the per-flood-map lead depends on geometry for two of three objectives.

**Draft answer.** *"We agree that the matched-budget comparison changes imagery, footprint and geography together with the label source. We therefore claim only that teacher labels on new imagery are a competitive substitute for the manual split at the same budget, not that they are better labels. The geometry-matched control shows that the size of the advantage depends on footprint for two of three objectives. We corrected its description: the crop removes 73.6 % of S11 water pixels and empties 21 chips, so it is itself confounded. [If run: a new control that trains on the identical 252 S11 images with teacher instead of human labels isolates the label source: …]"*

**Still open — recommended control (~0.5 GPU-h) [J]: the label swap.**
1. Let the frozen teacher label the 252 S11 training images.
2. Train EfficientViT-B0 on *these same images* with teacher labels instead of human labels (hard-label, logit KD and OFA; 4 seeds).
3. Compare with the existing human-label runs.

Same images, same size, same places: the only difference is who drew the labels. A second, equally cheap control would test *geographic diversity*: 252 GDACS tiles drawn from few events versus from many events.

---

### 5.8 R2.3 — OFA's advantage over plain distillation is small and vanishes at scale

**In plain words.** "OFA (extra training branches) adds very little over ordinary distillation, and nothing with 2,500 images."

**What we did.**

- **Matched comparisons on the original data (Figure 2a).** OFA and logit KD were trained with identical settings: same gradient clipping, same pixel masks, same checkpoint rule, 4 paired seeds. This removed differences that existed in the original runs [R].

  | Student, budget | S11 | STURM | WorldFloods |
  |---|---|---|---|
  | EfficientViT-B0, 2,500 | −0.34 | −0.02 | +0.44 |
  | MobileNetV3-Small, 2,500 | −0.29 | −0.05 | +0.35 |
  | LightFormer-R18, 2,500 | +0.15 | −0.18 | −0.89 |
  | EfficientViT-B0, 100 | +0.70 | +0.12 | +0.70 |

  The only fairly consistent positive effect is at 100 images (and on WorldFloods per event). Everything else is within about ±1 point and changes sign.
- **Training curves (Figure 2b, c) [R].** OFA does not learn faster. At 2,500 images it first reaches 85 % validation mIoU after 468–2,028 updates, against 468–1,092 for KD (one KD seed never reaches it). OFA costs about **16 % more training time** (mean 1,074 s vs 925 s per run).
- **Scratch-trained capacity ladder:** OFA equals logit KD for both a 705k and a 31k model [R].

![OFA vs KD](reviewer_wrapup_20261012/figures/fig02_ofa_vs_kd.png)

*Figure 2. OFA minus logit KD, and training curves (data: `figure_data/fig02a_ofa_minus_kd.csv`, `fig02bc_training_curves.csv`).*

**Verdict: Answered. We agree with the reviewer.**

**What to change.**
- Present OFA as one of the tested objectives, not as "ours".
- State that final performance comes from data scale and soft targets.
- Keep the gradient-conflict analysis (Supplement S6) only as a description of the small-data regime.
- The deployed model can stay OFA-trained; say so plainly.

**Draft answer.** *"We agree. Under fully matched training (identical clipping, masking and checkpoint selection, four paired seeds) OFA's difference from logit KD at 2,500 scenes is between −0.9 and +0.4 mIoU points across three backbones and three benchmarks, with no consistent sign; OFA does not converge faster and costs about 16 % more training time. A small positive effect remains only at 100 scenes. We removed OFA from the contribution list and state that the final accuracy comes from the larger teacher-labelled pool and the soft targets."*

---

### 5.9 R2.4 — OFA was not tested on other compact backbones

**What we did.** OFA was attached to MobileNetV3-Small and LightFormer-ResNet18: four exits at the same relative depths (stride 4/8/16/32), with 168,792 and 176,008 training-only parameters. Both were trained like EfficientViT-B0, 4 seeds each, and compared with their own logit-KD twin (Figures 1 and 2) [R].

**Result.** No consistent OFA gain on either backbone (−0.9 to +0.35 points).

**Verdict: Answered.** Caveats: one shared recipe, not tuned per backbone; LightFormer's own auxiliary loss was disabled for all its runs; device speed was not measured for the alternative backbones.

**Draft answer.** *"We ported OFA to MobileNetV3-Small and LightFormer-ResNet18 (four exits at strides 4–32, removed at inference) and trained both with logit KD and with OFA under the identical recipe and four seeds. OFA gives no consistent gain on either backbone (Table X, Fig. Y), so the architecture comparison is now fully crossed with the objective."*

---

### 5.10 R2.5 — When is the learned model really better than the simple MNDWI index?

**In plain words.** "A one-line formula is almost as good on the external test sets. In which practical situations is the expensive model actually worth it?"

**Questions.** (a) Within the training distribution? (b) Outside it: pooled versus flood by flood? (c) What kind of mistakes does each make? (d) Under changed atmospheric or sensor conditions? (e) Is the teacher useful as a labeller even if the index scores similarly? (f) Does the index transfer without re-tuning?

**(a) In distribution (S11 test).** The student reaches 78.3 % water IoU, MNDWI 65.0, the teacher 81.7 [R]. The learned model is clearly better: about 13 points.

**(b) External test sets.**

- *Pooled:* differences are small. STURM: student 65.4, MNDWI 64.5. WorldFloods: 64.2 vs 63.8 [R].
- *Flood by flood (Figure 7, new [V]):*
  - **On STURM the learned models beat MNDWI on only about half the floods** (teacher 16 of 29, student 15 of 29).
  - **On WorldFloods the teacher wins 8 of 11** and the student 6 of 11.
  - Wins can be large: EMSR286 teacher 80.9 vs MNDWI 21.2; EMSR333 52.5 vs 18.9; EMSR264 36.0 vs 16.3.
  - So can losses: STURM EMSR562 MNDWI 43.3 vs teacher 11.0.
  - **On WorldFloods' biggest flood, EMSR342, which holds 82 % of all its water pixels, MNDWI beats both learned models (70.5 vs 66.4 and 66.7) [V].** This one flood is why pooled WorldFloods scores look so close.

![Events vs MNDWI](reviewer_wrapup_20261012/figures/fig07_events_vs_mndwi.png)

*Figure 7. Per-event water IoU, learned models vs MNDWI (data: `figure_data/fig07_events_vs_mndwi.csv`).*

**(c) Different kinds of mistakes (Figure 6).**
- The learned models find less water but raise far fewer false alarms. WorldFloods: precision 91.5 % vs 77.8 %, recall 68.3 % vs 78.0 % [R].
- In-distribution, one in three pixels that MNDWI calls water is wrong (precision 67.5 %).
- On the external sets all methods end up with almost the same IoU, but with opposite balances. For a warning system that must not cry wolf, precision matters; for a damage overview that must not miss water, recall matters.

![Precision and recall](reviewer_wrapup_20261012/figures/fig06_precision_recall.png)

*Figure 6. Precision vs recall (data: `figure_data/fig06_precision_recall.csv`).*

**(d) Changed conditions (synthetic stress tests, Figure 8) [R].**
- **Haze:** at the strongest level, the student keeps 74.8 mIoU on WorldFloods while MNDWI drops to 25.7 (STURM: 68.2 vs 41.9).
- **Overall brightness changes:** MNDWI is mathematically immune; the models are almost immune.
- **Visible bands darkened relative to infrared:** MNDWI is clearly more robust on WorldFloods (75.7 vs 54.2 at the strongest setting).
- These are synthetic perturbations, not real atmospheres.

![Perturbations](reviewer_wrapup_20261012/figures/fig08_perturbations.png)

*Figure 8. Synthetic stress tests (data: `figure_data/fig08_perturbations.csv`).*

**(e) The teacher as a labeller (paper source data, Supplement S4) [V].** Replace the teacher's labels by MNDWI labels on the same 2,500 tiles and train with hard labels only:
- STURM drops from 75.4 to 71.5 mIoU, and training becomes unstable (seed spread ±4.7 instead of ±0.2);
- robustness to haze worsens (worst-case drop 12.5 vs 9.9 points);
- on WorldFloods, however, the MNDWI-labelled student is *better* (76.8 vs 75.4);
- when the teacher's soft targets are added, the label source no longer matters on STURM (75.6 vs 75.6).

The teacher's value as a labeller is therefore mostly in its soft, contextual information, and it is dataset-dependent.

**(f) Does the index transfer?** Not reliably [R]:
- the related index NDWI collapses on WorldFloods (water IoU 0.25);
- MNDWI's best threshold differs between datasets (+0.05 on STURM, +0.15 elsewhere);
- tuning the threshold on S11 makes STURM worse (73.8 vs 74.7);
- adaptive per-image thresholds (Otsu) are much worse (61 mIoU).

**Additional evidence [R].**
- Where teacher and MNDWI disagree, the teacher is right on 79 % of such pixels in S11 test, but only 59 % on STURM and 62 % on WorldFloods (v7). Useful information beyond the rule exists, together with real teacher mistakes.
- A 418-parameter per-pixel learned model beats the teacher on S11 test (82.9 water IoU) but is below MNDWI on STURM (63.1 vs 64.5). Spatial context, not just per-pixel colour, is what transfers (Figure 9).

**Verdict: Answered.** The honest answer has two sides.

*The learned model helps*:
1. within the training distribution;
2. when false alarms are costly (much higher precision);
3. under haze;
4. on hard floods where water-like surfaces fool the index;
5. as a labeller for training other models;
6. without per-dataset threshold tuning.

*MNDWI remains competitive*:
1. on clean external scenes overall;
2. in finding more of the water (higher recall);
3. under brightness changes and some spectral shifts;
4. on about half of the individual STURM floods.

**What to change.**
- Add a subsection "When does a learned model help over MNDWI?" with Figures 6–8 and the per-event table.
- Keep the existing careful statement that the external benchmarks show generalisation, not superiority.

**Draft answer.** *"We added a dedicated analysis (Section IV-D, Figs. X–Z). The learned student is 13 water-IoU points better than MNDWI in distribution. On the external sets pooled scores are similar, but the error types differ strongly: the student has much higher precision (91 % vs 78 % on WorldFloods) and lower recall (68 % vs 78 %). Per flood event, the learned models beat MNDWI on about half of the STURM events and on 6–8 of 11 WorldFloods events, with large gains on events where water-like surfaces mislead the index and a loss on the largest WorldFloods event. Under synthetic haze the learned models keep their accuracy while MNDWI collapses; under a relative darkening of the visible bands MNDWI is more robust. As a labeller, replacing the teacher's masks by MNDWI masks costs 3.8 mIoU on STURM. We therefore present the learned model as offering higher precision, robustness to haze and threshold-free transfer, not uniformly higher accuracy."*

**Still open.** Only the general caveats: synthetic perturbations are not real atmospheres, and the benchmarks were seen during development (R1.3). Our S11 MNDWI score (65.0 water IoU, 90 test chips) is also lower than a published 74.4 % (RSE 2025, all 446 chips, different protocol). The revision could not explain the gap from public code and should not compare the two numbers directly [R].

---

### 5.11 R2.6 — Results are hardware-specific; the engine is mixed INT8/FP16, not strictly INT8

**In plain words.** "Your speed and memory numbers belong to one device, and the model is not fully 8-bit."

**What we know.**

- **Why mixed precision.** The attention blocks of EfficientViT (LiteMLA) add up values in the range 200–400. Squeezed into 8 bits per tensor, that leaves only 1–5 usable levels and the attention collapses. TensorRT 8.4 offers no 16-bit integer type, so these 8 matrix multiplications run in FP16 (Supplement S7). On the Xavier NX the FP16 engine is actually faster (5.20 vs 5.57 ms). The mixed engine is smaller (1.5 vs 2.4 MB) and has lower peak power (5.8 vs 7.3 W).
- **New: the device numbers are fully traceable [V].** The four raw Jetson evaluation logs (one per seed, in `OLD/FloodDistill_v2/paper_source_data/.../raw_device/`) reproduce Table III exactly: STURM 75.57 → 74.48, WorldFloods 78.61 → 77.02. The earlier revision audit had listed this chain as missing. The three engine builds with timing logs agree closely (INT8 5.50–5.69 ms, FP16 5.15–5.34 ms), so the timing is repeatable on this device.
- **New: the 8-bit accuracy loss depends on the seed (Figure 10) [V].** Seed 64 loses 3.7 points on STURM and 3.3 on WorldFloods; the others lose 0–1.5. An independent INT8 run with ONNX Runtime gives the same accuracy as the TensorRT engine, so the loss comes from quantization itself, not from the TensorRT compiler. Seed 64's drop is visible on the S11 validation set too (−3.8). A validation check before deployment would have caught it.
- **Error found:** Table IV prints the FP32 mean GPU power as 3.95 W; the log says **3.25 W** [V].
- **Hardware independence.** Parameter count, model file size and operation count do not depend on the device. A second-platform timing exists only for the capacity-ladder models on a desktop RTX 4090, not for the deployed EfficientViT-B0 [R]. Native INT8 compilation failed on the workstation because of a TensorRT/driver mismatch [R].
- **A strictly INT8 path.** Simulated 8-bit weights and activations cost nothing for the convolution parts of a from-scratch EfficientViT-B0 (84.19 vs 84.16 validation mIoU, attention kept in float); 4-bit weights also cost nothing for a smaller model (−0.25 points, 95 % interval −0.81 to +0.32) [R]. The attention-free 12k-parameter R4 network has no attention blocks at all and could in principle run fully in INT8, but that has not been compiled or measured.

![INT8 by seed](reviewer_wrapup_20261012/figures/fig10_int8_by_seed.png)

*Figure 10. Device float vs deployed engine, per training seed (data: `figure_data/fig10_int8_by_seed.csv`).*

**Verdict: Partly.** The acknowledgement and the explanation can be written now. A strict-INT8 measurement and a second-device measurement are **open**.

**What to change.**
- Call the deployed model a "mixed INT8/FP16 TensorRT engine" in the abstract, contributions, Section VI, and Tables III and IV.
- Explain why the attention runs in FP16 and that FP16 is the fastest option on this device.
- Report the per-seed spread of the 8-bit loss and recommend a validation gate.
- Fix 3.95 → 3.25 W; label power as "GPU rail, peak/mean during trtexec".
- Add hardware-independent cost figures (parameters, MACs, activation memory).

**Draft answer.** *"We now describe the deployed model throughout as a mixed INT8/FP16 TensorRT engine and explain why: the LiteMLA attention accumulators span a range that per-tensor INT8 cannot represent, and TensorRT 8.4 offers no INT16 alternative, so eight matrix multiplications remain in FP16. On the Xavier NX the FP16 engine is the fastest option, while the mixed engine is the smallest and has the lowest peak power. Timing is repeatable across three independent engine builds (5.50–5.69 ms). The 8-bit accuracy cost varies by training seed (0 to 3.7 mIoU), and the affected seed is detectable on the validation split before deployment. We also report hardware-independent costs (parameters, operations, activation memory) and state that latency and power are specific to this device. [Optional: a strictly-INT8 engine loses … ; an attention-free student allows full INT8 …]"*

**Still open — optional [J].**
1. Build a strictly INT8 engine on the Jetson (force INT8 for the attention MatMuls) and report its accuracy, so the "collapse" has a number (~1 hour with the device).
2. Time the deployed EfficientViT-B0 on a second platform, e.g. a desktop GPU or an embedded CPU (~1 hour).
3. Optionally compile the attention-free R4 network fully in INT8 as an outlook result.

---

## 6. Problems we found that the reviewers did not raise

These come from re-checking the paper's numbers against raw files. They must be fixed in the revision, because the reviewers may check the released code and data.

| # | Where | Problem | Evidence | Fix |
|---|---|---|---|---|
| 1 | §III-A, Tables I–II, abstract | **Teacher row does not reproduce.** S11 0.897 / 0.822 printed; verified 0.8946 / 0.8167. STURM 0.755 → 0.7539; WorldFloods 0.771 → 0.7748. The printed S11 value has no traceable source | [R] `prompt5_closure/tables/teacher_baseline_reconciliation.csv` | Use the verified teacher values; recompute the student–teacher gaps and their paired tests (S11 gap ≈ 1.7 instead of 2.0 points; WF gap ≈ 1.1 instead of 0.7). The student values themselves reproduce within 0.02 points |
| 2 | Table IV | FP32 mean GPU power printed 3.95 W; log says 3.25 W | [V] `paper_source_data/Table_IV_Jetson_Efficiency/s91_speed.log` | Correct to 3.25 W |
| 3 | Supplement S2 | "Removing [the pre-filter] costs up to 0.048 mIoU clean" matches none of the measured values (0.1 to 31 points, depending on the selector) | [V] Figure 4 | Replace with the per-selector numbers |
| 4 | §IV-C | Crop control: "removes more than half of the positive area from 86 of the 252 chips" | [R] prompt 4/5 audit | 200 chips; 73.6 % of water removed; 21 chips become dry |
| 5 | §IV-G, Appendix A, S5 | "False alarm on 1.9 % of the 8,883 baselines the teacher left empty": these tiles have < 5 % predicted water (only 4,521 pre-event tiles have none), and "false alarm" means the student marks ≥ 5 % water | [V] `evaluation/relabel_feasibility.py`, pool statistics | Precise wording (R1.5 table) |
| 6 | §III-C, Table V | Hard-label "CE" vs focal loss; KD/OFA differ in gradient clipping in the historical runs | [R] prompt 3–4 audit | Describe the executed losses; cite the matched controls |
| 7 | §III-C | OFA heads: "488 parameters"; actually 11,368 training-only parameters (projection, normalisation, activation, classifier) | [R] | Correct |
| 8 | §III-B Eq. 1–2 | Descriptor uses the stored *maximum-class confidence* and its spread, not the water probability | [R] | Correct the definition |
| 9 | Appendix A (STURM) | Paper says "without clipping"; the code clips reflectance to [0, 1]; 10 chips contain invalid pixels (30,672 scored pixels) | [R] | Describe the executed preprocessing; corrected protocol in the supplement |
| 10 | §IV-A (WorldFloods) | 148 of 785 border windows were stretched to 512 × 512 in the legacy evaluator; native-geometry scoring changes OFA−KD from +0.53 to +0.44 | [R] | Disclose; native results in the supplement |
| 11 | §IV-C | "A single AOI carries 57 % of its flood pixels": true per AOI, but per event EMSR342 holds 82 % | [V] | Make the unit explicit |
| 12 | Table III (ReLU6 row) | The −0.8 "activation cost" is a learning-rate effect (R1.4) | [R] v7 | Re-word; optionally retrain |
| 13 | §IV-A statistics | Table I's exact 252-tile membership and run chain could not be re-traced | [R] | Keep with a provenance note |

**Recommendation on which numbers to show [J].** Keep the paper's original student numbers: they re-score within 0.02 points from the original checkpoints. Replace the teacher row by the verified values and recompute the comparisons that use it. Put the revision's new matched controls (architectures, OFA vs KD, 100 images, hard-label control) in a new, clearly separated table. Keep the legacy evaluation protocol in the main text and report the corrected STURM/WorldFloods protocols in the supplement; they change no conclusion. The prompt-5 draft proposed rewriting the abstract with re-trained numbers; that is not necessary and adds risk.

---

## 7. Everything that is still open

**Must do before resubmission** (people time, no GPU):

| # | Item | Concern | Owner | Effort |
|---|---|---|---|---|
| 1 | Factual statement of which choices were made after seeing STURM/WorldFloods scores | R1.3 | Authors | 30 min |
| 2 | Numerical corrections in Section 6 (teacher row and dependent statistics; Table IV power; S2; crop; wording) | — | Authors (+ CPU re-analysis for the teacher gap tests) | 1 day |
| 3 | Rewrite contributions, OFA, MNDWI, deployment and limitation text; terminology pass | R1.1, R1.5, R1.6, R2.1, R2.3, R2.5, R2.6 | Authors | 2–3 days |
| 4 | New tables and figures: data roles, component table, three backbones, MNDWI analysis (Figures 1–8, 10, 11 here) | R1.2–R1.4, R2.2–R2.5 | Authors | 1 day |
| 5 | Discussion section (Section 8) | Editor | Authors | 1 day |
| 6 | Response letter and highlighted manuscript; check the 28-day deadline | Editor | Authors | 1 day |

**Should do** (strengthens the answers, cheap):

| # | Item | Concern | Effort |
|---|---|---|---|
| 7 | Same-image label swap: S11 images with teacher vs human labels, 3 objectives × 4 seeds | R2.2 | ~0.5 GPU-h + 2 h |
| 8 | Dry-scene test: complete the 11-window review (quick) or build a ~100-tile dry set (proper) | R1.6 | 1 h / 1–2 days |
| 9 | Conventional-teacher control: UNet trained on S11 labels as the labeller | R1.2 | ~2 GPU-h + 2 h |

**Could do** (optional):

| # | Item | Concern | Effort |
|---|---|---|---|
| 10 | Strictly-INT8 engine on the Jetson, to put a number on the attention collapse | R2.6 | ~1 h with device |
| 11 | Second-platform timing of the deployed model | R2.6 | ~1 h |
| 12 | Size ladder and R4 in the supplement | R1.2, R1.1 | writing only |
| 13 | Retrain the ReLU6 student with the better learning rate, then QAT and Jetson | R1.4 | ~3 GPU-h + device |

The GPU items 7 and 9 fit in the remaining compute allowance of the capacity programme (about 4 of 30 GPU-hours left) [R].

---

## 8. Proposed Discussion section (editor's request)

An outline, roughly one paragraph per point, each tied to a reviewer concern:

1. **What the teacher buys (R1.1, R2.2).** A fixed manual annotation budget can be extended to new imagery without new labels. At equal size, teacher labels are a competitive substitute (advantage on STURM, geometry-dependent). With 10× more images the student closes most of the in-distribution gap. The gain comes mainly from data scale and soft targets.
2. **What the architecture and objective buy (R1.2, R2.3, R2.4).** Little. Three backbones (0.7–11 M parameters) and two objectives land within about one point externally. OFA is not needed. Even a 12k-parameter attention-free network is competitive, which suggests capacity is not the bottleneck for this task.
3. **What a spectral index still does well (R2.5).** Pooled external accuracy and recall, invariance to brightness, robustness to some spectral shifts. The learned model is preferable when false alarms are costly, under haze, on confusing surfaces, and when no per-dataset threshold tuning is possible. A practical product could combine both (e.g., the index as a recall-oriented second opinion).
4. **What "flood" means here (R1.5).** The labels define surface water. A flood product needs a reference-water layer or a before/after comparison. The student inherits the teacher's convention almost completely.
5. **Deployment reality (R1.6, R2.6).** Mixed precision is a property of this attention design and this compiler. The 8-bit loss varies by seed and should be gated by validation. Dry-scene false alarms are untested because of the pre-filter. Unattended operation needs a dry-scene evaluation.
6. **How far the benchmarks can be trusted (R1.3).** Development exposure. Pooled vs per-event views can disagree (one WorldFloods event holds 82 % of the water). Results are conditional on the evaluation protocol.
7. **Outlook.** Smaller attention-free students that are fully INT8; temporal flood-increment supervision from the pre-/post-event pairs already in the pool; a second teacher; an untouched event set.

---

## 9. Decisions only the authors can make

1. **R1.3 history statement:** what exactly was tuned on STURM/WorldFloods? This determines the wording of the leakage answer.
2. **Which numbers to show:** keep the original student tables plus the verified teacher row (recommended), or switch to the re-trained, corrected-protocol numbers (prompt-5 proposal).
3. **Scope:** include the size ladder and R4 as a supplementary finding (stronger novelty, larger scope), or leave it for a follow-up paper.
4. **Optional experiments:** label swap (R2.2), dry-scene set (R1.6), conventional-teacher control (R1.2), strict-INT8 test (R2.6). Each strengthens one answer; none is strictly required by the letter.
5. **OFA in the deployed model:** keep it (no retraining needed) while describing it as optional, or switch the narrative to logit KD.

---

## 10. Sources and reproduction

**New in this wrap-up** (folder `reviewer_wrapup_20261012/`):

- `make_figures.py`: all 11 figures, read-only on its inputs; `figure_data/*.csv` holds the plotted numbers of every figure.
- `verify_numbers.py` → `verification.json`: 12 checks, re-deriving Table III from the raw Jetson logs, Table IV power, the pre-filter effects from raw per-scene counts, the pool funnel, the false-alarm definition, the dry-chip diagnostic, per-event wins, and EMSR342's weight.
- `run_all.sh`: rebuilds both, CPU only, in under 10 seconds.

**Main evidence used:**

| Topic | File |
|---|---|
| Reviewer letter, paper, supplement | `paper/review.docx`, `paper/main.pdf`, `paper/supplement.pdf` |
| Matched controls (3 backbones, KD/OFA, N=100, focal), precision/recall, protocols | `prompt4_report_20261007/prompt4_report.txt`, `prompt5_closure/tables/current_full_summary.csv`, `OFA_matched_effects.csv` |
| Data roles, overlap exclusion, earlier draft edits and response | `prompt5_closure/data_roles.csv`, `overlap_*`, `main_manuscript_proposed_edits.txt`, `supplement_proposed_edits.txt`, `point_by_point_response_draft.txt` |
| Robustness (haze, tilt, gain) | `prompt4_report_20261007/tables/robustness_all_conditions_metrics_and_drops.csv` |
| Teacher baseline reconciliation | `prompt5_closure/tables/teacher_baseline_reconciliation.csv` |
| Capacity ladder, learning-rate repair, teacher-vs-MNDWI disagreement, desktop timing | `capacity_v4_20261008/report_v5.md`, `capacity_v7_20261011T210730Z/report_v7_codex_audit_20261012.md` |
| OFA learning curves, negative adaptation results | `research_v8_20261012T123202Z/ofa_learning_curves.csv`, `report.md`, `negative_results.md` |
| Paper source data (Table I–IV, selector study, geometry and labeller controls) | `paper_source_data/` |
| Raw Jetson device logs (Table III) | `/home/schmalstieg/repos/OLD/FloodDistill_v2/paper_source_data/Table_III_Deployment_Chain/raw_device/` |
| Pool statistics (funnel) | `/data1/schmalstieg/flooddistill/data/pseudo_gdacs_v2/pool_features.csv`, `diversity_index.json` |

**Earlier drafts to update, not reuse unchanged.** The prompt-5 drafts (M1–M16, S1–S8, response draft) were written before the real letter was available and before the findings in Sections 5–6. Most of their content remains valid. Change in particular:
- the abstract proposal (M1; see Section 6 recommendation);
- the deployment qualification (M13; the device chain is now traced);
- the pre-filter statement (S2);
- the new per-event MNDWI and seed-dependent INT8 findings.
