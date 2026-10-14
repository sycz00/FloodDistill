# V8 useful corrections, single-solve adaptation and compression

Collected 2026-10-12T13:01:05.778069+00:00 after completion.

Worker state: finished; active job: None. Results are partial until terminal receipts exist.

Cumulative GPU-command hours: 25.969689; incremental completed/preflight hours: 0.098437. Active elapsed and remaining reservations are in ledger.json.

## Observations

Completed adaptation rows: 180 of 180 predeclared rows.

Mean descriptive query mIoU differences versus the corresponding frozen parent (pp):

- B_supervised: -1.7710

- B_supervised_SGD20: -0.3533

- B_teacher_label: -1.2376

- F0: +0.0000

- F1: -0.2867

- F2_1: +0.0214

- F2_20: +0.0786

- F2_5: +0.0672

- F3: -0.3783

- F4: -0.5517

- Q0: -0.4739

- Q1: -0.8105

- Q2: -0.4473

- Q3: -0.7173

- Qbase: +0.0000

Every event, seed, support draw and arm remains in adaptation_results.csv. Equal-event differences are descriptive; no population CI is estimated from three events.

## Interpretation and next causal test

Useful teacher correction and harmful imitation are computed from same-pixel joint errors in teacher_headroom.csv, event_headroom.csv and joint_confusions.json. G-H is a pixel-accuracy identity; IoU is recomputed from integer confusion counts.

Oracle outputs are unattainable label-aware diagnostics. Stability/confidence is a proposed proxy and can weight confidently wrong teacher predictions.

If uniform ridge fails, inspect frozen-head expressiveness, support/query shift and target bias. If uniform ridge helps without reliability gain, retain the solver and reject this proxy in this setting. If calibration matches the solver, interpret it as an operating-point correction.

Q0–Q3 retain the same fixed quantized backbone and float32 classifier. Q3 can support an interaction only if it exceeds Q0, Q1 and Q2 on matched episodes. This pilot uses fake quantization and establishes neither integer execution nor speedup.

All external benchmarks are research-exposed. Adaptation uses no query labels for fitting or checkpoint selection. The teacher inherits 252 labelled source scenes; zero target-label adaptation is not a label-free pipeline.

One-shot supervised arms use exactly one fully labelled support chip per episode; target_label_pixels and area disclose the annotation budget. A single solve excludes neither teacher queries nor feature extraction; cold and cached cost components are separate.

Feature means/scales and quantizer ranges use TRAIN only. Lambda and SGD learning rate use disjoint S11 validation development pairs before target query scoring. Source labels used for hyperparameter selection are disclosed in choice_s42.json and choice_s64.json.

## Provenance and limitations

The restored original N2500 membership has zero replacements and SHA256 1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60. Historical generation-time byte identity remains unverified.

V7 C3 is retained as a negative scientific result and is not repeated. Historic EfficientViT original-activation OFA curves remain separate from repaired ReLU6 capacity graphs.

FP32/FP16 deployment measurements use separate fresh processes, actual recorded TF32 flags, a common input, 20 warmups and 100 synchronized timings. They are RTX4090 forward timings and exclude input transfer/normalization; edge deployment is unmeasured.

Training checkpoint size is never substituted for deployed artifact size. Low-bit master checkpoints include floating-point tensors and quantizer metadata; no theoretical packed-size claim is made.

See negative_results.md for missing reviewer letter, test.txt, independent dry references, second-teacher inputs and unsupported INT8.

## Reproduction

`/home/schmalstieg/miniconda3/envs/esi/bin/python /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v8_20261012T123202Z/jobs_v8.py recover --out /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v8_20261012T123202Z/receipts/recovered_again.json`

`/home/schmalstieg/miniconda3/envs/esi/bin/python /home/schmalstieg/repos/FloodDistill/experiments/revision/research_v8_20261012T123202Z/collect_v8.py --reason manual`

GPU task argv, input hashes, dependencies and timeouts are frozen in manifest.json; attempts and exact measured durations remain in the ledger.

## Deliverables

evidence_index.json; frozen_protocol.md; manifest.json; ledger.json; status.json; joint_confusions.json; teacher_headroom.csv; event_headroom.csv; ofa_learning_efficiency.csv; ofa_learning_curves.csv; adaptation_results.csv; comparisons.csv; compression_frontier.csv; deployment.csv; reviewer_response_map.md; negative_results.md.
