# Dataset, preprocessing and protocol audit (Tasks 2 and 4)

Read-only. WorldFloods-v2 train/val (sealed) was not opened.

## 1. How each dataset enters the pipeline

| | Sen1Floods11 (S11) | GDACS pool | WorldFloods-v2 test | STURM-Flood S2 |
|---|---|---|---|---|
| Role | teacher fine-tuning (IBM/NASA); S11-label arms; val = selection/calibration; test = score | student training (teacher labels) | external score | external score |
| Annotation | human, on S2 (`LabelHand`: 1 water, 0 not water, −1 no data) | teacher argmax | EMS/UNOSAT/GLOFIMR maps, manually curated; gt band 1 cloud {0, 1, 2}, band 2 land/water {0, 1, 2} | EMS maps; 0 non-flood, 1 flooded, 2 river-related, 3 open water, 4 reservoirs, 5 lakes, 99 no data |
| Label semantics | all surface water | teacher's surface water | water = flood + permanent (Sci. Rep. 2023) [R] | flood vs permanent classes separated |
| Product | L1C TOA | L1C (`COPERNICUS/S2_HARMONIZED`) [V] | L1C | not L1C-like: blue 1st pct ≈ 0.013, blue < green (paper App. A [R]; median B2 0.094 < B3 0.123 [V]) |
| Bands | B2 B3 B4 B8A B11 B12 | same; 20 m bands bilinear → 10 m [V] | same | B2 B3 B4 **B8** B11 B12 (B8 in the B8A slot) [R] |
| Scaling | DN × 1e-4, **no clip**, NaN → 0 (TerraTorch) [V] | DN × 1e-4, clip [0, 1] [V] | DN × 1e-4, clip [0, 1] [V] | 0.04 + 0.12·v (inverse of 400–1600 min–max), then clip [0, 1] [V] |
| Normalisation | S11 train mean/SD | same | same | same |
| Native size | 512 (≈ 4.6 × 5.1 km) | 264 (2.6 km) | scenes tiled to 512 | 128 (1.28 km) |
| Train size | resize → 224 (**≈ 22.9 m**) | resize → 224 (≈ 11.8 m) | — | — |
| Eval size | 512 native | — | 512 native; partial windows resized (legacy) or zero-padded (revision P3) [V] | 128 → 512 (**2.5 m**) [V] |
| Invalid / ignore | −1 | — | cloud, invalid, all-zero spectrum, **JRC permanent → ignore** | 3–5 and 99 → ignore |
| Splits / overlap | official random-chip split: **test shares all 10 events with train** [V]; Bolivia event-held-out (15 chips) | 23,720 tiles, 3,603 events | 18 maps, 11 events; overlap screens done [R] | 2,675 chips, 46 maps, 29 events; overlap screens done [R] |

Radiometry in reflectance units before normalisation (`outputs/radiometry.json`, band medians
B2 B3 B4 B8A B11 B12) [V]:
- S11 train 0.119 0.116 0.097 0.327 0.204 0.102
- GDACS event 0.107 0.099 0.074 0.296 0.172 0.084
- GDACS baseline 0.100 0.091 0.071 0.265 0.172 0.088
- WF test 0.107 0.102 0.087 0.275 0.175 0.085
- STURM 0.094 0.123 0.113 0.239 0.187 0.123

S11, GDACS and WF share the L1C signature (blue p5 ≈ 0.08). STURM is the radiometric outlier.
The paper discloses this and uses STURM only for paired comparisons [R].

## 2. Teacher preprocessing: fine-tuning vs our use (Task 4.1)

| Use | Input the teacher sees | Matches fine-tuning? |
|---|---|---|
| IBM/NASA fine-tuning | S11 random 224 crops, native 10 m, no clip, S11 normalisation | reference |
| S11 val/test evaluation | 512 native | ✔ |
| **GDACS pseudo-labels** | 264 → 224 (≈ 11.8 m), 20 m bands bilinear, clip [0, 1] | ≈ (−15 % scale; bilinear vs S11's export resampling; clip rarely binds: S11 has ≤ 0.014 % of pixels > 1 [V]) |
| Live KD on GDACS arms | augmented 264 → 224 | ≈ as above |
| **Live KD on S11 arms (Table I KD/OFA)** | S11 512 → 224 (**≈ 22.9 m**) | ✘ 2.3× coarser than training |
| **V9 label swap D** | S11 512 → 224 | ✘ same |
| WF evaluation | 512 native | ✔ |
| STURM evaluation | 128 → 512 (2.5 m), L2A-like, B8 for B8A | ✘ legitimate domain shift + protocol choice. Teacher 0.7619 at native 128 vs 0.7550 at 512 (Supp Table S1) [R] |

**Classification (Task 4.2)** [H]:
- *Implementation inconsistencies* (fixable, small): teacher at 22.9 m on S11 arms (KD targets
  and label swap); bilinear vs S11's 20 m resampling; clip asymmetry; STURM band comment.
  None of these touches the GDACS pseudo-label contract materially. **No harmonisation of GDACS
  to S11 is warranted.** P3 measures the scale effect directly on S11 val/test.
- *Legitimate domain differences*: GDACS geography (1,404 events in N2500), tile footprint,
  acquisition dates and clouds (accepted up to 50–80 %), STURM processing level and B8.
  Measure them; do not remove them.
- *Label-source differences*: teacher pseudo-labels vs human S11 vs EMS-derived WF/STURM
  references. These differ in permanent-water convention (below) and in edge/thin-channel
  behaviour (V9: 66 % of teacher misses within 2 px of an edge on S11 test) [R].

## 3. Evaluation protocols (Task 2.2) — **STOP for approval**

### Class-mapping tables

**WorldFloods-v2** (gt band 2: 1 land, 2 water; band 1: 2 cloud; JRC: 3 permanent, 2 seasonal)

| Pixel | (1) current "flood-only" | (2) original WF-v2 | (3) common surface water |
|---|---|---|---|
| gt water, not JRC-permanent | water | water | water |
| gt water, JRC-permanent | ignore | water | water |
| gt land, JRC-permanent | ignore | land | land |
| gt land | land | land | land |
| cloud / invalid / all-zero spectrum | ignore | ignore | ignore |
| Aggregation | pooled + per scene/event | **mean over flood events** | pooled + per event (report both) |

**STURM-Flood**

| Class | (1) current "paper" | (2) original "published" | (3) common surface water |
|---|---|---|---|
| 0 non-flood | land | land | land |
| 1 flooded | water | water | water |
| 2 river-related features | **water** | water | water |
| 3 open water, 4 reservoirs, 5 lakes | ignore | water | water |
| 99 no data / NaN support | ignore | ignore | ignore |
| Test population | all 2,675 chips | authors' 267-chip test split | all 2,675 (never trained on) |

Facts [V] (`outputs/sturm_classes.txt`): class 2 is in 1,676 chips and makes up 5.5 % of
pixels scored as water under (1). It is the only scored water in 41 chips. 20 chips contain
only classes 3–5, so under (1) they have zero valid pixels. MNDWI > 0 fires on 87.6 % of
class-1, 73.9 % of class-2 and 98.3 % of class-3 pixels.

### Assessment

| | (1) current | (2) original | (3) common surface water |
|---|---|---|---|
| Task evaluated | flood increment, approximated by subtracting reference-water layers | each dataset's own water target | the task S11 labels, the teacher, the students and MNDWI all perform |
| Fair to teacher/student/MNDWI? | all three predict surface water; permanent water is removed from the score, not from the predictions. Fair as an ignore mask, but STURM class 2 (rivers) is kept while WF rivers in JRC are dropped | yes | yes |
| Comparable across datasets? | partly: JRC (30 m, occurrence-based) vs EMS hydrography classes | no: per-dataset aggregation differs | yes, one rule |
| Changes paper claims? | baseline | WF: +0.08–0.09 water IoU for every model, ranking unchanged (paper §IV-A) [R]; STURM unknown | same as (2) for WF; STURM to be measured (P4) |
| Recommendation [H] | **supplementary** (continuity with submitted Tables I–III) | cite for orientation; WF per-event means are already reported | **primary** |

Why (3) as primary [H]: it matches R1.5's point that the target is surface water. It removes
the dataset-specific semantic surgery, including the class-2 inconsistency. And it is closest to
both datasets' own definitions. The flood-only view stays as the supplementary operational
reading, labelled as such.

**Affected by a protocol change:** every external number in Tables I–III, Figs. 2/4/5, the
R2.5 analyses (P/R, AP, per-event wins, haze), and the dry-window definitions. Under (3),
STURM has no dry chips at all, and WF windows whose only water is permanent are no longer dry.
Training, checkpoints and selection are unaffected: re-scoring only, no retraining.
