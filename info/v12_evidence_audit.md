# V12 follow-up evidence audit
Prepared 9 October 2026 from the uploaded `v12_followup_2026-10-09.zip`.

This note records what the attachment supports. It is not a new weather-model run, live workstation status check or independent validation. The full revised document explains the methods and research implications.

## 1. Evidence levels
The archive directly contains the J.4/J.5 replay code, frozen configuration, completed replay receipt and score tables. The crop and perturbation narratives refer to successful diagnostic output files that are absent from the uploaded archive. The original repaired four-bit and latent-interpolation results are reported by the follow-up narratives; their original full result packages are not included.

The completed replay receipt `J45/J45_RESULT.json` says:
- status: `COMPLETE_ARCHIVED_REPLAY`;
- J.4 evaluation starts: 6; J.5 starts: 8;
- equal-size perturbed-initial ensemble: `PENDING_J1`;
- no policy selected: `True`;
- GPU seconds: 0; elapsed replay seconds: 4.8399.

This was processor-only reanalysis of existing forecasts, not newly executed live adaptive inference.

## 2. J.4: adaptive two-to-six-query rainfall replay
Sources: `fu_j45.py`, `J45/J4_FRONTIER.csv`, `J45/J4_PER_START.csv`, `J45/J45_FREEZE.json` and `J45/J45_RESULT.json`.

Within a six-hour block, all fine-lead queries use the same pre-update weather history. The magnitude rule forms `3*(p3+p6)` and escalates to all six hourly rainfall fields when the maximum over either regional halo reaches the selected threshold. One block-level decision is shared across the two regions. The 10 mm trigger is a six-hour-estimate threshold, not the daily 25 mm verification threshold. Lead 6 remains the dense weather update.

The relevant six-start frontier is:

| policy        |   calls_per_120h |   added_raw_rmse_mm |   introduced_misses |   recovered_misses |   introduced_false_alarms |
|:--------------|-----------------:|--------------------:|--------------------:|-------------------:|--------------------------:|
| D6            |       120.000000 |            0.000000 |                   0 |                  0 |                         0 |
| T3            |        60.000000 |            0.010604 |                   7 |                  6 |                         8 |
| D2            |        40.000000 |            0.029290 |                  14 |                 22 |                        14 |
| magnitude_10  |        78.000000 |            0.003226 |                   0 |                  9 |                         0 |
| variability_5 |        64.000000 |            0.009499 |                  11 |                 42 |                         6 |

“Added error” is candidate reference RMSE minus dense reference RMSE under the archived daily reducer. It is not candidate-to-dense disagreement RMSE. Mean queries of 78 versus 120 imply a calculated 35% query reduction, not measured latency savings. The 60-query comparator spends less work, so a matched-cost control is still needed.

Introduced and recovered misses are transitions relative to dense. Candidate total misses equal dense total misses plus introduced misses minus recovered misses. Zero introduced misses therefore does not mean zero reference-heavy misses. Ranking only introduced misses can disagree with ranking total misses.

All six non-calibration starts are already exposed development cases. Their magnitude-10 results are:

| date       |   calls_per_120h |   added_raw_rmse_mm |   reference_wet_cell_days |
|:-----------|-----------------:|--------------------:|--------------------------:|
| 2025-01-31 |              116 |           -0.012880 |                      1225 |
| 2025-05-02 |               60 |           -0.007524 |                       266 |
| 2025-09-25 |               88 |            0.011238 |                        82 |
| 2025-01-11 |               64 |            0.013870 |                         0 |
| 2025-03-04 |               76 |           -0.006183 |                        36 |
| 2025-07-15 |               64 |            0.020836 |                        68 |

The `reference_wet_cell_days` field used here counts the reference-heavy support under the implemented event rule. It sums to 1677; 31 January contributes 1225, or 73.05%. These are dependent cell-days in two regions and five daily windows, not independent storms or global event frequency. The error improved on three starts and worsened on three.

## 3. J.5: compression variants as ensemble members
Sources: `fu_j45.py`, `J45/J5_PER_LEAD.csv`, `J45/J45_RESULT.json`.

The following are recalculated equal means of the archived per-start/region/day rows:

| members       |   fair_crps_mm |   dense_mae_mm |    brier |   dense_brier |   spread_skill_ratio |
|:--------------|---------------:|---------------:|---------:|--------------:|---------------------:|
| D+N8+S8+S6    |       0.775127 |       0.825864 | 0.005806 |      0.006003 |             0.132104 |
| D+N8+S8+S6+S4 |       0.730858 |       0.825864 | 0.005709 |      0.006003 |             0.385825 |

D is dense; N8 is native eight-bit floating point; S8/S6/S4 are simulated weight quantizers decoded to 16-bit execution. The simulated members do not supply native low-bit cost savings. The spread/error column is a mean of row-level ratios, not a pooled ratio. The reported ratios show limited spread in this diagnostic, not calibrated operational uncertainty.

Methodological interpretation: the code calculates a finite-ensemble-corrected (“fair”) continuous ranked probability score. Heterogeneous deterministic compression variants are not automatically independent samples from one predictive distribution. Also report the ordinary empirical score and the assumptions behind any correction. Compare with both an equal-member-count and measured-cost-matched perturbation ensemble. That comparison is still pending in the completed replay receipt.

## 4. J.2: physical crop diagnostics
Sources: `README.md`, `RESULTS.md`, and `fu_diag.py`.

The narratives report that changing from patch-aligned to attention-window-aligned crops greatly reduced rainfall disagreement. The smallest aligned crop reportedly covers about 7.1% of the globe and has error 0.79 times the three-query comparator; the larger aligned crop gives 0.83. Both fail the previously fixed maximum of 0.5. Increasing context did not improve this comparison.

Support is six saved blocks from two calibration starts, on the West Coast, not six independent complete forecasts. The inspected code's full-globe-through-crop identity assertion is for fine lead 1 on each block. The reported eager timings for five calls are about 0.44, 1.66 and 6.06 seconds for small crop, large crop and full globe. They are not complete deployed-forecast timings.

The successful `gpu/FU_DIAG/DIAG_RESULT.json`, `J1_SWEEP.json` and `J2_CROPS.json` cited by the narratives are absent from this upload. Preserve these as reported results and recover the underlying receipts in the live repository. The current Central Europe longitude restriction is an implementation-path limitation, not proof that every boundary-consistent regional method is impossible.

## 5. J.1: perturbed initial-state ensemble
Sources: diagnostic narratives and code, `failed_attempts/FAILURES.json`, `wait_and_run_j1.log`, `FU_J1_R2.log`, retry worker/launch records.

The reported perturbation-response rule selected epsilon 0.01 after the earlier tiny perturbations had very limited response. The numerical-rounding explanation is plausible, but the optional full-precision-backbone comparison failed a dtype check and was removed; that particular mechanism was not fully isolated.

The first ensemble attempt stopped when another user's GPU job appeared. The waiter log then records:
`2026-10-09T11:05:09Z GPU 1 idle; launching FU_J1_R2`.

The launched unit is `weather-fu-fu-j1-r2.service`. No final successful retry result or completed final ensemble analysis is attached. Correct status: **restarted; final outcome unavailable in this upload**, not “still waiting,” “complete” or “currently running.”

The old waiter uses two idle observations five minutes apart. That is not itself proof of ten continuous idle minutes. The new execution prompt requires elapsed-time-based admission, unknown-device failure handling and a detached controller including both waiting and analysis.

## 6. Preserve valid negative results
Sources: `README.md` and `RESULTS.md`.

The repaired simulated-four-bit low-rank method reportedly passed zero-factor identities and reduced local layer error slightly (1.3%), yet worsened rollout quality on six starts. Its status supersedes the older construction failure. The original detailed repaired score table is not attached, so exact rollout scores are not reconstructed.

The latent interpolation probe reportedly passed 72/72 decode identity checks, used a refreshed anchor and still failed its quality threshold: 0.81 with anchor versus 1.34 without, using the two-query disagreement denominator and a maximum of 0.5. This denominator differs from the crop study's three-query denominator.

A1 and D4 are not defined in the supplied materials inspected for this revision. The exact `skipU25` bypass mask is also unavailable. The new prompt requires live registry resolution rather than invented expansions.

## 7. Research implications, not completed results
Confirm the live adaptive policy with matching execution and cost controls; do not claim its archived call reduction as latency. Test small recovery on student-generated histories against an equal-size ordinary one-step control. Preserve the repaired low-rank negative rather than repeating the same method as a “bug fix.” Reserve a bounded creative branch for boundary-consistent regional context or sentinel queries that audit sparse-trigger blind spots.

A novel contribution would need a controlled independent/native advantage, not just renamed distillation, low-rank factors or adaptive sampling. Final test and extreme-event membership must be frozen using a full exposure ledger; the attachment does not establish one pristine permanent split for every prior campaign.
