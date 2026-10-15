# Surface-water re-evaluation — results (15 Oct 2026)

Run: 15 Oct 10:49–11:06 on one RTX 4090; 113 models + 5 spectral rules (threshold 0 and S11-train+val-calibrated).
Regression: the P3 mappings (STURM paper, WF flood-only) and S11 test reproduce `prompt5_closure/tables/current_full_summary.csv`
in all 33 cells to within 0.0003 water-IoU points. Tables: `tables/` (pooled, per event, paired vs MNDWI, AP, dry windows,
operating points). Calibrated thresholds (`results/calibration.json`): MNDWI +0.148, WI2015 +3.46, AWEIsh +0.051.

Water IoU, pooled | equal-event; paired event difference vs MNDWI>0 with event-bootstrap 95 % CI and Wilcoxon p.

| | S11 test | Bolivia (held-out event) | WF surface | STURM surface (512) |
|---|---|---|---|---|
| Teacher | 81.7 \| 79.1 | 67.6 | 73.9 \| 79.4; +8.2 [−0.6, +20.9] | 67.7 \| 49.9; −0.5 [−3.3, +1.6] |
| Paper student (B0 OFA, 4 seeds) | 78.3 \| 75.8 | 72.4 | 75.6 \| 79.5; +8.3 [−0.3, +20.8] | 68.3 \| 50.3; −0.1 [−2.6, +1.9] |
| R4 LKD (12k) | 82.5 \| 80.3 | 76.6 | 76.7 \| 79.2; +8.0 [−0.1, +19.9] | 67.7 \| 49.2; −1.1 [−2.7, +0.4] |
| MNDWI > 0 | 65.0 \| 68.3 | 55.1 | 74.8 \| 71.2 | 67.4 \| 50.4 |
| MNDWI, S11-calibrated (+0.148) | 77.3 \| 76.4 | 70.1 | 75.5 \| 73.9 | 64.4 \| 42.8 (worse) |
| WI2015 > 0 | 75.6 \| 75.1 | 63.1 | 76.4 \| 73.2 | 68.3 \| 52.0; +1.7 [+0.2, +3.7] vs MNDWI |
| WI2015, S11-calibrated | 81.4 \| 79.5 | 76.6 | 76.6 \| 77.4 | 66.7 \| 45.3 |

Findings
1. Scoring all water (the Sen1Floods11 definition) does not change the conclusions; it raises every method on WF
   (+10–13 points) and STURM (+3), and keeps the ranking. Learned models lead MNDWI per WF event (+8, CIs touch 0,
   n = 11) and tie it on STURM (−0.1 to −1.1 per event, n = 29).
2. A fair, label-free calibration of the rules on S11 train+val helps in distribution and on WF but hurts on STURM:
   threshold transfer is dataset-dependent. The calibrated WI2015 matches the teacher on S11 (81.4 vs 81.7) and the
   12k R4 on Bolivia (76.6 vs 76.6). The stronger rule baseline is WI2015, not MNDWI.
3. Bolivia, the only event-held-out S11 split: learned models beat MNDWI>0 by 12–21 points, but the calibrated
   WI2015 equals the best learned model (R4).
4. Dry windows (WF surface, ≥ 1 % of valid pixels, 200 windows, 9 events): teacher 5, paper student 14, R4 6.5,
   MNDWI 132, calibrated MNDWI 73, calibrated WI2015 32. The false-alarm advantage of the learned models survives
   both the protocol change and threshold calibration of the rules.
5. Matched-budget (N = 252) arms: the S11-human B0 is clearly worse than its teacher-labelled twin on WF surface
   (68.5 vs 72.1 pooled) and raises 47 dry-window alarms vs 7.
