# Dry-scene audit (Task 3)

R1.6: *"the pseudo-label filtering removes fully dry scenes. As a result, false alarms on
ordinary water-free imagery are not sufficiently evaluated."* The premise is **correct, and
understated**: dry tiles are removed at two stages, not one.

## 1. Definitions used below

- **No water**: no pixel of any water class in the reference (or, for GDACS, in the teacher map).
- **No flood water**: no flood-class pixel, though permanent water may be present (ignored
  under the current protocol).
- **No valid positives**: all pixels ignored. Under the current protocol a STURM chip of only
  lakes is such a case. It is neither dry nor wet and must be excluded from every denominator.

## 2. Where dry scenes are removed or kept

| Stage | Rule | Dry scenes affected | Source |
|---|---|---|---|
| GDACS acquisition, **event tiles** | must have MNDWI > 0 water outside JRC-permanent ≥ 1 % (20 m reduction), permanent ≤ 50 %, cloud probability ≤ 40 %; up to 8 random relocations | Locations without MNDWI water are re-drawn; 5,225 of 17,120 slots never passed; all 11,895 kept event tiles have ≥ 1 % MNDWI new water (median 5.6 %) | `patch_quality_v2.csv`, `supervisor.sh` [V] |
| GDACS acquisition, **baselines** | tile cloud < 20 %, **no water gate** | 11,825 kept, dry ones included | `download_flood_events.py` [V] |
| Pseudo-label generation | `min_flood_frac 0.0`; scene mean confidence ≥ 0.80 | dry tiles kept (149 tiles removed for low confidence) | `prepare.sh` [V] |
| Pre-filter | teacher water ∈ [5 %, 85 %], confidence SD ≥ 0.05 | removes 16,186 tiles with < 5 % teacher water: event 7,303, baseline 8,883; 5,843 with exactly 0 | `outputs/dry_funnel.txt` [V] |
| Training sets | FPS from 5,987 | **0** tiles with < 5 % teacher water in N2500 and N252. N2500 = 1,539 event + **961 pre-event baselines** (38 %) | [V] |
| Checkpoint selection | S11 val, all 89 chips | not filtered; 9 water-free chips with valid pixels | V9 rows [V] |
| S11 test | all 90 chips | not filtered; 6 water-free chips | [V] |
| WF test | windows with ≤ 1 % valid pixels dropped (not water-based) | 206 dry windows of 785 (V9 windows; 216 under legacy tiling), all with valid pixels | [V] |
| STURM | all 2,675 chips | 8 dry chips; 20 chips with no valid pixels (only classes 3–5) | `outputs/sturm_classes.txt` [V] |

Two observations [V]:
- 7,110 of the 7,303 teacher-dry *event* tiles have ≥ 1 % MNDWI water. The pre-filter
  therefore removes mainly tiles where teacher and MNDWI disagree. That bears on Task 5: the
  student is never trained on the scenes where the two water definitions conflict most.
- 1,796 baselines are "doubly dry" (teacher < 5 % and MNDWI water = 0). They are the only
  candidate source of reliable negative supervision in the pool.

## 3. Existing false-alarm evidence and its validity

| Evidence | What it measures | Validity |
|---|---|---|
| 8,883 teacher-dry baselines, student ≥ 5 % on 1.9 ± 0.8 % (paper §IV-G) | agreement with the teacher | teacher-referenced, not false alarms [R] |
| 6 water-free S11 test chips (wrap-up §5.6) | false water on human-verified dry chips | same events as training; n = 6, one seed [R] |
| WF dry windows (V9): alarms ≥ 1 % of valid px: MNDWI 136, teacher 5, paper student 15/16/8/9, R4 10/9/9/8 (of 206) | false water on reference-dry land **inside flood scenes** | counts verified [V]. But **9 events**, 82 % of windows from EMSR347 (78), EMSR286 (51) and EMSR342 (39). Events with ≥ 1 alarm: MNDWI 8/9, teacher 2/9, student 5/9 (any seed) (`outputs/wf_dry_by_event.txt`) |
| MNDWI false water on WF dry windows | is it real dry-land error? | 0.1 % of MNDWI false-water pixels lie within 30 m of JRC permanent water; 13.7 % within 30 m of any ignored pixel. So mostly not boundary ambiguity [V] (`outputs/wf_dry_mndwi.txt`) |
| STURM "dry windows" (V9, 13 Oct) | — | **invalid**: 20 of 28 had zero valid pixels and counted as alarms for every model. 8 real; teacher 4/8, MNDWI 3/8. Corrected in V9 recovery on 14 Oct [V] |

**Answer to "does MNDWI predict much more water in genuinely dry scenes?"** Yes, on the
reference-dry land of WF flood scenes, and robustly: the pixels are not permanent-water or
cloud edges [V]. Limits: one dataset, 9 events, and two events (EMSR286, EMSR264) where MNDWI
fails across every window. That is the same scene-level failure that drives MNDWI's per-event
losses. So it is partly not independent of the per-event IoU evidence [H].

## 4. Proposed false-alarm evaluation (Task 3.2; needs approval: new evaluation set)

- **Selection before any model runs.** About 120 Sentinel-2 L1C tiles, 264 px (GDACS geometry).
  ≥ 30 locations, ≤ 4 tiles per location. Strata: desert/bare, urban, snow/ice, cropland,
  forest, mountain shadow, cloud shadow, dark burnt areas. Dry criterion: JRC GSW maximum extent
  = 0 within the tile + 100 m buffer, and ESA WorldCover not water/wetland. Then a human visual
  check on RGB/SWIR quick-looks, blind to model output.
- **Processing.** The same GDACS loader (`FloodUnlabeledDataset`), so the pre-processing is
  identical to the training tiles.
- **Metrics per method** (teacher, paper student 4 seeds, R4 4 seeds, MNDWI > 0, NDWI,
  WI2015/AWEIsh): false-water pixel rate; share of tiles with any / ≥ 0.1 % / ≥ 1 % / ≥ 5 %
  water; false-water area per tile (km²); stratum breakdown.
- **Uncertainty.** Bootstrap over locations; paired per-tile differences vs MNDWI; seeds
  averaged per tile first.
- **Alarm criterion, fixed in advance:** ≥ 1 % of valid pixels, and a ≥ 0.1 km² connected
  component (both reported).

## 5. Training-selection sensitivity (Task 3.3)

Existing evidence is indirect. Students trained with zero dry tiles raise few alarms on WF dry
windows (teacher level). The small S11-only short runs raise many (72–107 of 206) [R]. So
absence of dry training tiles has not caused visible harm *inside flood scenes* [H]. Whether it
harms on ordinary dry land is exactly what §4 would measure.

Minimal experiment, **only if** §4 shows the paper student alarming on > 5 % of dry tiles at
the 1 % rule: replace 10 % and 25 % of the N = 2,500 set with doubly-dry baselines (fixed N,
same recipe, 4 seeds). Compare dry-set alarms, WF dry-window alarms and external water IoU.
Doubly-dry tiles supply reliable negatives (teacher and MNDWI agree), so the arm does not add
label noise. Cost ≈ 2.5 GPU-h.
