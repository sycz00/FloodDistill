# Learned models vs MNDWI — evidence audit (Task 5)

All numbers are water class, current protocol (revision P3 masks for STURM/WF; S11 test 512),
unless stated.

## 1. Verified existing comparisons

| Level | Teacher | Paper student (B0 OFA N2500, 4 seeds) | MNDWI > 0 | Source | Status |
|---|---|---|---|---|---|
| Pixel, pooled IoU S11 test | 81.7 | 78.3 | 65.0 | fig06 CSV ← `current_full_summary.csv` | [V] |
| Pixel, pooled IoU STURM | 64.9 | 65.4 | 64.5 | same | [V] |
| Pixel, pooled IoU WF | 64.3 | 64.2 | 63.8 | same | [V] |
| Precision / recall WF | 91.4 / 68.5 | 91.5 / 68.3 | 77.8 / 78.0 | same | [V] |
| Precision at MNDWI's recall: S11 / STURM / WF | 82.5 / 71.0 / 86.4 | 73.5 / 71.8 / 85.6 | 67.5 / 71.4 / 78.1 | `threshold_free_ap.csv` (V9) | [V] |
| Pooled AP: S11 / STURM / WF | 96.0 / 83.5 / 87.9 | 94.7 / 82.7 / 88.3 | 90.7 / 77.7 / 88.4 | same | [V] |
| Equal-event AP: S11 / STURM / WF | — | 92.6 / 56.2 / 76.6 | 91.9 / 55.5 / 70.8 | same | [V] |
| Event wins vs MNDWI: STURM / WF | 16/29, 8/11 | 15/29, 6/11 | — | fig07 data | [R] |
| Per-scene macro STURM mIoU | 0.5921 | 0.5909 | 0.5945 | paper Supp S4 | [R] |
| Dry windows, WF (≥ 1 %) | 5/206 (2/9 events) | 8–16/206 (5/9 events) | 136/206 (8/9 events) | V9 + `outputs/wf_dry_by_event.txt` | [V] |
| Haze h = 0.20, WF mIoU | — | 74.8 | 25.7 | fig08 data | [R] |
| Visible darkening, WF mIoU | — | 54.2 | 75.7 | fig08 data | [R] |
| STURM at native 128 px, mIoU | 0.7619 | 0.7464 | 0.7443 | paper Supp Table S1 | [R] |

## 2. Validity issues

1. **Thresholds.** MNDWI > 0 is fixed. That matches the WF-v2 authors' own baseline [R], so it
   is not a strawman. The supplement's "optimal +0.05 STURM / +0.15 elsewhere" is a
   test-label oracle. Its "tuned on the in-distribution split" does not say which split; if it
   was S11 test, that is test-label use. [H] Replace it with a calibration on S11 train+val
   only (permitted data), reported next to the fixed 0 and labelled as such. The learned models'
   thresholds are fixed (argmax). V9's tau grid is selected on S11 val only [R]. The "precision
   at MNDWI recall" analysis moves the student's threshold with test labels: it measures
   **ranking separability, not an attainable operating point** (the deck says so).
2. **"In distribution" = same events.** S11 test shares all 10 events with S11 train, on
   which the teacher was fine-tuned. Bolivia (15 chips) is the official event-held-out split
   and is unused (P7).
3. **Aggregation.** Pooled WF is dominated by EMSR342 (82 % of water pixels), where MNDWI
   wins [R]. STURM per-scene macro favours MNDWI slightly [R]. Event-level CIs exist in v7 for
   R4 only (R4 − MNDWI equal-event: STURM −1.4 [−3.4, +0.6], WF +7.5 [−1.0, +19.9]) [R].
4. **Evaluation scale.** STURM ranking vs MNDWI depends on the 512 upsample. At native 128 the
   teacher leads MNDWI by 1.8 mIoU and the student ties it (+0.2) [R].
5. **Class mapping.** STURM class 2 (rivers, 5.5 % of scored water) is scored as flood; MNDWI
   detects 73.9 % of it [V]. The ranking under the surface-water protocol is unknown (P4).
6. **Training selection.** The pre-filter removed the 7,110 event tiles where MNDWI sees water
   and the teacher does not [V]. The student never learns from the scenes where the two
   definitions conflict.

## 3. Missing measurements

- Event-level paired CIs for the **paper student** vs MNDWI (pooled and equal-event) on
  STURM/WF: CPU from cached per-event cells; protocol-independent.
- Surface-water protocol re-scoring (P4, approval).
- Bolivia held-out event (P7).
- MNDWI threshold calibrated on S11 train+val only (CPU; S11 train/val labels are permitted).
- Dry land outside flood scenes (P5, approval).

## 4. Conclusion as it stands (Task 5.5) [H, from §1–2]

| Question | Answer now |
|---|---|
| Better than MNDWI on the primary task? | In distribution (same events): clearly (+13 water IoU). External, pooled: tie (STURM +0.9, WF +0.4 for the student). |
| Across aggregation levels? | No. STURM per scene: MNDWI ≥ learned. WF per event: learned ahead (student 6/11 events; teacher 8/11), but EMSR342 decides pooled. |
| Under fair threshold calibration? | Ranking quality (AP) is ≥ MNDWI everywhere and higher on STURM/S11. A label-free calibrated threshold for MNDWI has not been tested yet. |
| Fewer dry-scene false alarms? | Yes inside WF flood scenes (5–16 vs 136 windows; 2–5 vs 8 of 9 events). Ordinary dry land untested. |
| More robust under shift? | Under synthetic haze, yes. Under visible-band darkening, no. Under brightness change, MNDWI is invariant by construction. Real atmospheres untested. |
| After compute and annotation confounding? | The learned model needs 252 labels + a 300 M teacher + training for a pooled tie on new floods. Its practical advantages are precision, dry-area false alarms in flood scenes, haze, and in-distribution accuracy. |
| Claims to keep / weaken | Keep: "competitive with MNDWI on unseen floods; higher precision; far fewer false water alarms on reference-dry land within flood scenes; better in distribution". Weaken: any "outperforms MNDWI" on external sets; "in distribution" → "new chips from the training events"; dry-scene claims limited to flood-scene land until P5. |
