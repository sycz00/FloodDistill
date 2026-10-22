#!/usr/bin/env python3
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'device_handoff'
def read(p):return json.loads(Path(p).read_text())
b=read(D/'provenance/B0_MIXED_REPAIR.json');lines=[]
for e in b['seeds']:
 d=e['drops'];lines.append(f"| {e['seed']} | {e['updates']} | {100*d['mIoU']:.4f} | {100*d['equal_event_mIoU']:.4f} | {100*d['water_IoU']:.4f} | Pass |")
fallback=[]
for name,label in [('R4_s42_qat_mixed','Tiny42 mixed (diagnostic)'),('R4_s42_qat_pure','Tiny42 pure fallback'),('R4_micro_s107_qat_mixed','Micro107 mixed fallback'),('R4_micro_s107_qat_pure','Micro107 pure fallback')]:
 g=read(D/'models'/name/'SOURCE_GATE.json');c=g['comparison'];water=g['baseline']['water_IoU']-c['pooled']['water_IoU'];fallback.append(f"| {label} | {-100*c['pooled_delta']:.4f} | {-100*c['equal_event_delta']:.4f} | {100*water:.4f} | Pass |")
text='''# Answers and verified mixed B0 repair — 22 October 2026

All four current512 B0 QAT deployment graphs now pass the user's amended source-accuracy rule: an absolute drop no greater than0.02 on each of pooled mIoU, equal-event mIoU and pooled water IoU against the same-seed genuine original float parent. This is workstation CPU/ORT evidence. Native Jetson engine builds, accuracy, precision placement and benchmarks remain pending. Strict QDQ logit equivalence at atol=rtol=.001 remains failed; all89 unrounded structure checks pass. Accuracy acceptance does not erase that failure.

## Why the earlier B0 QAT failed

The recent short recipe was not the original successful deployment recipe. It quantized the classifier (`skip_last_n=0`), used eight GDACS training images for calibration, folded BN before fitting, and used focal loss with AdamW LR1e-4/WD.05 for128 updates. Its lower-LR retry retained the other differences. The original project used a floating classifier (`skip_last_n=1`), S11 image calibration, retained/frozen BN, CE and LR1e-5/WD1e-5. The historical deployed B0 also used ReLU6; these current parents use native Hardswish. No activation or parent-weight substitution was made here.

Simply exempting the classifier did not repair source accuracy. The seed42 six-arm diagnostic showed severe errors with scalar activation grids even when the classifier was floating. A power-of-two, channel-balanced input/weight rescaling adapted from the existing SmoothQuant helper preserved the complete89-image float function within the unchanged .001 tolerance and reduced the seed42 zero-update pooled mIoU loss from36.36 to3.11 percentage points under the full-S11 calibration arm. Corrected QAT then recovered the selected graphs. These are recipe comparisons and targeted diagnostics; they do not isolate every changed factor causally.

A separate native-builder omission was identified: the recent mixed builder enabled FP16 but did not enforce the original attention FP32 protection. This cannot explain CPU accuracy failures, but can threaten native attention numerics. The repaired builder requires FP32 for all8 MatMuls, the normalization path through four Div nodes, and the floating classifier, using obeyed precision constraints and inspector validation. It disables TF32 and rejects missing/stale/failed certificates. Remaining CNN placement is mixed INT8/FP16 and must be reported from the actual engine. No fully integer B0 candidate is asserted.

## Recovered B0 results

All values below are losses in percentage points; a negative loss is an improvement. Evaluation uses all89 source images and the actual optimized linear-Resize deployment ONNX, with the genuine parent weights as baseline.

| Seed | Selected lineage updates | Pooled mIoU loss | Equal-event mIoU loss | Water IoU loss | User rule |
|---|---:|---:|---:|---:|---|
'''+ '\n'.join(lines)+'''

Seeds42/64/107 recovered after640/640/384 updates of corrected CE QAT. Seed91's first1280-update fit failed (water loss2.5820 points), eight declared floating-Conv policies failed, and a separate1024-update CE continuation failed. Those records remain unchanged. A new bounded run from the1280-update checkpoint matched the frozen original student parent on the same512 training images using KL(T=2) plus0.01 logit-margin MSE. It recovered after384 additional updates,1664 in its selected lineage. There was no new foundation-teacher inference, added ground-truth training set or external-score selection. Failed continuation updates are experimental work, not ancestors of the selected checkpoint. Source-based choices and amended thresholds are post hoc, not preregistered.

Every selected graph retains56 Conv input/weight QDQ sites out of57 Conv layers, a floating classifier, floating attention and no Conv output QDQ. No additional CNN floating exceptions were selected. Full source float-construction checks passed before fitting. Fresh isolated checkpoint reloads reproduced all89 Torch confusion records, and fresh optimized ORT runs reproduced all89 saved deployment records for all four seeds. GPU float parity failed at the frozen tolerance, so all fitting and qualification used the validated two-thread CPU path; the failed GPU diagnostic is retained.

## Tiny and micro QAT fallback degradation

These source losses are modest relative to the user's2-point rule; native fallback accuracy is still pending. The mixed and pure exported candidates were scored separately against genuine float parents.

| Candidate | Pooled mIoU loss (pp) | Equal-event mIoU loss (pp) | Water IoU loss (pp) | User rule |
|---|---:|---:|---:|---|
'''+ '\n'.join(fallback)+'''

Tiny mixed PTQ already passes the amended rule for all four seeds. Tiny42 QAT is the pure-slot fallback; its mixed QAT graph remains an extra diagnostic. Micro107 uses QAT fallbacks in both quantized slots. All four listed QAT graphs retain strict export-equivalence failures despite passing actual exported-graph accuracy and unrounded structure checks. No external-dataset or native-device recovery is inferred from these source values.

## Files and remaining Jetson work

Use `device_handoff/JETSON_PROMPT.md`, also copied to `FloodDistill/prompts/jetson_prompt.md`. The complete repaired export archive is `MajorRev/revision_package_20261021/Jetson_mixed_repair_20261022.tar.gz`; it includes the exact353 prepared S11/Bolivia/dry arrays. Raw WF/STURM datasets must already exist on Jetson and are verified by the runner. The original export archive remains intact.

The Git handoff replaces the four failed B0 QAT primary entries with fresh `B0_s<seed>_qat_mixed_repaired` identities. Old failed QAT graphs are retained as historical records, and failed PTQ controls remain scheduled diagnostics. The matrix remains48 slots:44 supplied primary candidates plus four unavailable B0 pure-integer slots. There remain51 declared engine entries,16 reused tiny engines and35 new build attempts, with all five datasets, dry subsets and benchmarks required. No four-seed aggregate drops a failed or missing seed.

Use a new native attempt against this updated package. Preserve all logs, graph and policy hashes, raw source counts, build/precision inspectors, timing and steady power evidence. Never add classifier/attention QDQ, silently replace a failed integer build with float, or report native recovery before measurement. Collection keeps source accuracy, strict equivalence, native coverage and precision certification separate.
'''
for a,b in [('than0.02','than 0.02'),('all89','all 89'),('complete89','complete 89'),('from36.36','from 36.36'),('to3.11','to 3.11'),('all8','all 8'),('for128','for 128'),('Seeds42','Seeds 42'),('after640','after 640'),('after384','after 384'),('Seed91','Seed 91'),('first1280','first 1280'),('loss2.5820','loss 2.5820'),('separate1024','separate 1024'),('the1280','the 1280'),('same512','same 512'),('updates,1664','updates, 1664'),('plus0.01','plus 0.01'),('retains56','retains 56'),('of57','of 57'),("user's2","user's 2"),('exact353','exact 353'),('remains48','remains 48'),(':44',': 44'),('remain51','remain 51'),(',16',', 16'),('and35','and 35')]:text=text.replace(a,b)
(R/'ANSWERS_AND_REPAIR.md').write_text(text)
(D/'provenance/ANSWERS_AND_REPAIR.md').write_text(text)
