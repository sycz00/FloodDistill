# R1.4 — pipeline components, existing evidence, missing experiments

R1.4: *"The separate contributions of pseudo-label filtering, FPS-based selection,
Logit-KD/OFA, activation replacement, and QAT are distributed across the main paper and
supplementary material."*

## 1. The pipeline as executed (from code, not from the paper)

| # | Stage | What the code does | Source |
|---|---|---|---|
| 0 | Teacher | Public Prithvi-EO-2.0-300M-TL-Sen1Floods11, frozen. IBM/NASA fine-tuned it on S11 train with RandomCrop 224 at native 10 m, CE, AdamW 5e-5, 50 epochs; evaluation on full 512 chips | HF `config.yaml` (snapshot 91ce9d3) [V] |
| 1 | Acquisition | GDACS Orange/Red, 4 patches/event within 30 km, S2_HARMONIZED L1C, least-cloudy scene. **Event tiles must pass**: patch cloud probability ≤ 40 %, JRC permanent water ≤ 50 %, **MNDWI > 0 new water ≥ 1 %**; up to 8 relocations. Baselines 30–395 d earlier, tile cloud < 20 %, no water gate | `datasets/prepare.sh`, `download_flood_events.py`, `flood_unlabeled_v2/supervisor.sh`, `patch_quality_v2.csv` [V] |
| 2 | Pseudo-labels | Teacher on 264 → 224 bilinear (20 m bands bilinear to 10 m first, clip [0, 1], S11 normalisation); argmax mask + per-pixel confidence; scene kept if mean confidence ≥ 0.80 (23,571 / 23,720) | `data/unlabeled.py`, `data/pseudo_labeled.py` [V] |
| 3 | Pre-filter | teacher water fraction ∈ [0.05, 0.85] and confidence SD ≥ 0.05 → 5,987 | `data/diversity_pseudo.py` [V] |
| 4 | Selection | FPS on a 15-D descriptor → 2,500 (or 252) | same [V] |
| 5 | Objective | hard term (focal for KD/OFA, true CE for historical "CE" rows) + logit KD (T = 4, α = 0.5, live teacher on the augmented 224 input) + OFA branches (ω = 0.3) | `scripts/train.py`, wrap-up §5.4 [R] |
| 6 | Student | EfficientViT-B0 seg, Cityscapes-pretrained, 6-band first conv | `models/student.py` [V] |
| 7 | Activation | Hardswish → ReLU6 swapped **at build time on the pretrained weights**, then the same recipe | `models/student.py`; checkpoints [V] |
| 8 | QAT | W8A8 per-tensor fake-quant, 10 epochs, LR 1e-5, BN frozen, from the ReLU6 checkpoint | paper Table V [R] |
| 9 | Export | TensorRT 8.4, 8 attention MatMuls kept FP16 | wrap-up §5.11 [R] |
| — | Augmentation | dihedral + per-channel multiplicative jitter 0.8–1.2 | `data/sen1floods11.py`, pseudo flags [V] |

## 2. Component evidence matrix

Effects in mIoU points (S11 test / STURM / WF), 4 seeds, unless noted. "Fair" = one factor changed.

| Component | Claimed benefit | Existing comparison | Fair? | Effect | Status |
|---|---|---|---|---|---|
| Acquisition MNDWI new-water gate | finds flooded locations | none | — | unmeasured | **Missing**. Cannot be ablated without a new GEE download; state as a design choice [H] |
| Scene confidence ≥ 0.80 | removes unreliable tiles | none (removes 149 tiles, 0.6 %) | — | negligible by size [V] | Sufficient as a statement |
| Pre-filter | removes uninformative tiles | with/without × 4 selectors (Supp S1 raw) | yes | FPS +0.1, random +0.3, GLISTER +4.0, confidence +31 (STURM) [V] | Sufficient. Interaction measured |
| FPS vs random | diversity | both after the filter | yes | +0.2 STURM [V] | Sufficient (null) |
| Data scale 252 → 2,500 | more teacher labels | nsweep | yes | +2.6 / +0.3 / +0.4 [V] | Sufficient |
| Teacher vs human labels at 252 | label substitution | Table I, B1 crop, V9 label swap | partly | see R2.2 | Mostly; see `reviewer_evidence_matrix.md` |
| Soft targets (logit KD) | dark knowledge | KD vs hard-only at 2,500 | mostly (hard weight halves) | +0.7 / +1.0 / +1.2 [V] | Sufficient, with caveat |
| OFA branches | feature alignment | matched OFA vs KD, 3 backbones | yes | −0.3 / 0.0 / +0.4 [R] | Sufficient (null) |
| Pretrained init | transfer | v7 A0 scratch (LR 5e-3, LKD, 7,800 updates) 87.8 / 76.2 / 77.1 vs prompt4 matched pretrained LKD 87.8 / 76.1 / 76.7 | near-matched: same updates and objective, each at its own LR | ≈ 0 [R] | Usable as a side statement [H] |
| **Activation replacement** | INT8 compatibility | Table III Hardswish vs ReLU6, same pretrained init, recipe, seeds | **yes** | pooled −0.8 STURM / +0.8 WF; per map −0.2 (p = 0.24) / +0.6 (p = 0.07) [V] | **Sufficient; earlier write-ups mislabelled it** |
| **QAT** | recovers INT8 accuracy | none for the deployed B0. "INT8 engine vs float export" (−1.1 / −1.6) measures quantized deployment, not QAT | — | unmeasured | **Missing → P1** |
| Mixed INT8/FP16 export | attention fidelity | qualitative + ORT INT8 equals TRT | — | [R] | R2.6, not R1.4 |

## 3. Activation replacement and learning rate (Task 1.2)

- **Was the LR changed?** Yes, but only for *scratch-initialised* graphs, in capacity v4→v7. The
  grid {5e-5, 5e-4, 5e-3} was searched on seed 42 using S11 validation only; 5e-3 won for A0
  and A1 and was then run with 4 seeds. A0: +2.67 [1.97, 3.37] S11-val mIoU; A1: +3.53 [R:
  `report_v7_codex_audit_20261012.md`]. Optimizer AdamW, wd 0.05, clip 5, cosine, 7,800
  updates, FP32, unchanged.
- **Why?** The paper's LR 5e-5 is a fine-tuning LR for pretrained weights. Applied to random
  init it under-trains [R].
- **Did it help?** For scratch A0: S11 test +2.2, STURM +1.4, WF +1.9 mIoU [V: `results.csv`].
- **Was the activation comparison controlled?** The **paper's** Hardswish-vs-ReLU6 pair is
  controlled: same pretrained init, LR 5e-5, batch 16, OFA, 50 epochs, seeds [V: launcher +
  checkpoints]. The wrap-up/deck comparison "ReLU6 76.2 vs 74.8" contrasts two *scratch* A0
  runs at different LRs, so it is an LR effect. Its second comparison, "≥ Hardswish 76.0 /
  77.1 / 87.5", changes init, objective (LKD vs OFA) and LR together.
- **Is the activation replacement independently costly?** No measurable cost at the paper
  recipe [V]. Not tested: whether a higher LR helps the *pretrained* students, for either
  activation. That would change the absolute student level, not the activation attribution.
  P2 tests it if wanted.

## 4. Smallest defensible experiment set

| ID | Hypothesis | Controlled comparison | Data / models | Stop rule | Cost |
|---|---|---|---|---|---|
| **P1** | QAT recovers accuracy that PTQ loses | Same ReLU6 parent → (a) PTQ W8A8 per-tensor, calibration on the same S11-val batches; (b) its existing QAT child. Same simulator; read out S11-val (selection), then S11-test / STURM / WF on the current protocol | `repro_review/pseudo_ofa_relu6_N2500_s{42,64,91,107}` + `qat_relu6_*` (same lineage, batch 8) | one pass | < 0.5 GPU-h |
| P2 (opt.) | pretrained students are LR-limited too | {Hardswish, ReLU6} × LR {5e-5, 5e-4}, pretrained init, OFA, seed-42 grid on S11 val, then 4 seeds at the selected LR | GDACS N = 2,500 index | stop if the grid moves S11-val < 0.5 | 4–5 GPU-h |
| — | interactions | filter × selector already measured; scale × soft targets (Table I at 252 vs fig03 at 2,500) already measured | — | — | 0 |

Not proposed: one-at-a-time removal of every stage from the full pipeline. The stages are
nested: selection needs the filter, and QAT needs the ReLU6 parent. The existing paired
comparisons plus P1 cover every component the reviewer named. Rows stay non-additive.

## 5. Draft R1.4 answer (after P1)

*"Table X collects one controlled comparison per component (four seeds each; rows are separate
experiments and are not additive). Scaling the teacher-labelled pool tenfold (+2.6 mIoU in
distribution) and soft teacher targets (+0.7 to +1.2) give the largest gains. FPS selection
(+0.2 over random after filtering) and OFA branches (−0.3 to +0.4) give none. The pre-filter
matters only for weak selection rules (FPS +0.1; confidence ranking +31). Replacing Hardswish
by ReLU6 under the same initialisation and recipe has no consistent cost (STURM −0.2 per flood
map, p = 0.24; WorldFloods +0.6 per scene, p = 0.07). QAT versus post-training quantization of
the same model: [P1 result]."*
