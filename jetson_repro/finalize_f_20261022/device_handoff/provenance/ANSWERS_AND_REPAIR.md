# Answers and verified mixed B0 repair — 22 October 2026

All four current512 B0 QAT deployment graphs now pass the user's amended source-accuracy rule: an absolute drop no greater than 0.02 on each of pooled mIoU, equal-event mIoU and pooled water IoU against the same-seed genuine original float parent. This is workstation CPU/ORT evidence. Native Jetson engine builds, accuracy, precision placement and benchmarks remain pending. Strict QDQ logit equivalence at atol=rtol=.001 remains failed; all 89 unrounded structure checks pass. Accuracy acceptance does not erase that failure.

## Why the earlier B0 QAT failed

The recent short recipe was not the original successful deployment recipe. It quantized the classifier (`skip_last_n=0`), used eight GDACS training images for calibration, folded BN before fitting, and used focal loss with AdamW LR1e-4/WD.05 for 128 updates. Its lower-LR retry retained the other differences. The original project used a floating classifier (`skip_last_n=1`), S11 image calibration, retained/frozen BN, CE and LR1e-5/WD1e-5. The historical deployed B0 also used ReLU6; these current parents use native Hardswish. No activation or parent-weight substitution was made here.

Simply exempting the classifier did not repair source accuracy. The seed42 six-arm diagnostic showed severe errors with scalar activation grids even when the classifier was floating. A power-of-two, channel-balanced input/weight rescaling adapted from the existing SmoothQuant helper preserved the complete 89-image float function within the unchanged .001 tolerance and reduced the seed42 zero-update pooled mIoU loss from 36.36 to 3.11 percentage points under the full-S11 calibration arm. Corrected QAT then recovered the selected graphs. These are recipe comparisons and targeted diagnostics; they do not isolate every changed factor causally.

A separate native-builder omission was identified: the recent mixed builder enabled FP16 but did not enforce the original attention FP32 protection. This cannot explain CPU accuracy failures, but can threaten native attention numerics. The repaired builder requires FP32 for all 8 MatMuls, the normalization path through four Div nodes, and the floating classifier, using obeyed precision constraints and inspector validation. It disables TF32 and rejects missing/stale/failed certificates. Remaining CNN placement is mixed INT8/FP16 and must be reported from the actual engine. No fully integer B0 candidate is asserted.

## Recovered B0 results

All values below are losses in percentage points; a negative loss is an improvement. Evaluation uses all 89 source images and the actual optimized linear-Resize deployment ONNX, with the genuine parent weights as baseline.

| Seed | Selected lineage updates | Pooled mIoU loss | Equal-event mIoU loss | Water IoU loss | User rule |
|---|---:|---:|---:|---:|---|
| 42 | 640 | -0.0348 | 0.1750 | -0.1385 | Pass |
| 64 | 640 | 0.8457 | 0.7733 | 1.5530 | Pass |
| 91 | 1664 | 0.9462 | 1.1570 | 1.6928 | Pass |
| 107 | 384 | -0.0729 | 0.1872 | -0.2685 | Pass |

Seeds 42/64/107 recovered after 640/640/384 updates of corrected CE QAT. Seed 91's first 1280-update fit failed (water loss 2.5820 points), eight declared floating-Conv policies failed, and a separate 1024-update CE continuation failed. Those records remain unchanged. A new bounded run from the 1280-update checkpoint matched the frozen original student parent on the same 512 training images using KL(T=2) plus 0.01 logit-margin MSE. It recovered after 384 additional updates, 1664 in its selected lineage. There was no new foundation-teacher inference, added ground-truth training set or external-score selection. Failed continuation updates are experimental work, not ancestors of the selected checkpoint. Source-based choices and amended thresholds are post hoc, not preregistered.

Every selected graph retains 56 Conv input/weight QDQ sites out of 57 Conv layers, a floating classifier, floating attention and no Conv output QDQ. No additional CNN floating exceptions were selected. Full source float-construction checks passed before fitting. Fresh isolated checkpoint reloads reproduced all 89 Torch confusion records, and fresh optimized ORT runs reproduced all 89 saved deployment records for all four seeds. GPU float parity failed at the frozen tolerance, so all fitting and qualification used the validated two-thread CPU path; the failed GPU diagnostic is retained.

## Tiny and micro QAT fallback degradation

These source losses are modest relative to the user's 2-point rule; native fallback accuracy is still pending. The mixed and pure exported candidates were scored separately against genuine float parents.

| Candidate | Pooled mIoU loss (pp) | Equal-event mIoU loss (pp) | Water IoU loss (pp) | User rule |
|---|---:|---:|---:|---|
| Tiny42 mixed (diagnostic) | 0.3944 | 0.3541 | 0.6668 | Pass |
| Tiny42 pure fallback | 0.4335 | 0.3526 | 0.7466 | Pass |
| Micro107 mixed fallback | 0.8249 | 0.4035 | 1.4404 | Pass |
| Micro107 pure fallback | 0.9228 | 0.4720 | 1.6216 | Pass |

Tiny mixed PTQ already passes the amended rule for all four seeds. Tiny42 QAT is the pure-slot fallback; its mixed QAT graph remains an extra diagnostic. Micro107 uses QAT fallbacks in both quantized slots. All four listed QAT graphs retain strict export-equivalence failures despite passing actual exported-graph accuracy and unrounded structure checks. No external-dataset or native-device recovery is inferred from these source values.

## Files and remaining Jetson work

Use `device_handoff/JETSON_PROMPT.md`, also copied to `FloodDistill/prompts/jetson_prompt.md`. The complete repaired export archive is `MajorRev/revision_package_20261021/Jetson_mixed_repair_20261022.tar.gz`; it includes the exact 353 prepared S11/Bolivia/dry arrays. Raw WF/STURM datasets must already exist on Jetson and are verified by the runner. The original export archive remains intact.

The Git handoff replaces the four failed B0 QAT primary entries with fresh `B0_s<seed>_qat_mixed_repaired` identities. Old failed QAT graphs are retained as historical records, and failed PTQ controls remain scheduled diagnostics. The matrix remains 48 slots: 44 supplied primary candidates plus four unavailable B0 pure-integer slots. There remain 51 declared engine entries, 16 reused tiny engines and 35 new build attempts, with all five datasets, dry subsets and benchmarks required. No four-seed aggregate drops a failed or missing seed.

Use a new native attempt against this updated package. Preserve all logs, graph and policy hashes, raw source counts, build/precision inspectors, timing and steady power evidence. Never add classifier/attention QDQ, silently replace a failed integer build with float, or report native recovery before measurement. Collection keeps source accuracy, strict equivalence, native coverage and precision certification separate.
