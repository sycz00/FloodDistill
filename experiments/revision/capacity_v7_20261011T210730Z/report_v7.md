# FloodDistill v7

This report reflects saved artifacts and the actual worker state. Readouts marked not_run have not been established.

Worker: finished; collection reason: completion; prospective C3 gate: failed_scientific_gate.
Training records: {'reused': 20, 'completed': 34, 'not_run': 8}. Historical aliases do not add independent seeds.

The declared questions are whether R4 survives a source-selected, better-trained A0 anchor; whether matched soft supervision helps R4; whether additive MNDWI and disagreement weighting beat both controls; and what supported artifacts cost at runtime.

CE retains focal loss on cached teacher hard pseudo-labels. LKD queries a frozen teacher on augmented images. Continuations use identical selected R4 parents, fresh AdamW, and 1,560 updates; they do not establish improved scratch training.

IoU is the overlap of predicted and reference regions divided by their union. It is not pixel accuracy. Main readouts preserve the registered 512 grid, valid masks, established event IDs, and both full and overlap-excluded domains. Exclusion does not reverse prior research exposure.

Undefined empty-union water IoUs remain missing and are omitted from event means. Pooled pixel counts and equally weighted event scores are separate. Four paired seeds use sample SD and df=3 Student-t 95% intervals; two-seed pilots are descriptive. Event intervals use 10,000 whole-event resamples with seed 1729, conditional on trained checkpoints.

R4_LKD minus A0_LKD_reanchored: +0.62 source mIoU points, n=4; 95% CI [-0.04, 1.28].
R4_LKD minus R4_CE: -0.40 source mIoU points, n=4; 95% CI [-1.23, 0.42].
C3 minus C0: +0.00 source mIoU points, n=2; descriptive only.
C3 minus C4: +0.00 source mIoU points, n=2; descriptive only.
C3 minus C5: +0.00 source mIoU points, n=2; descriptive only.

C3 prospective gate: failed_scientific_gate.
Failed criteria: mean_C3_C0_ge_0.50, mean_C3_C4_ge_0.20, mean_C3_C5_ge_0.20.
C3 exceeded both prespecified control margins only if both corresponding frozen source criteria passed; no external scores determined promotion.

All completed arms are retained, including failures of the novelty hypothesis. Final checkpoints and source scores are retained; final external scores require a separate reservation and remain not_run otherwise.
Mechanism summaries: 4876 completed event/pooled records. Correction, teacher-following harm, rescue, conditioning frequencies, and signed teacher-versus-rule benefit use aggregated integer counts; empty groups stay undefined. Unlabelled TRAIN cannot establish correctness.
Deployment rows distinguish completed native artifacts, unsupported attempts, not_run work, and historical simulation_only. Artifact bytes, synchronized latency, and actual memory are reported only when measured. Parameter count is not substituted for storage, latency, or energy.

Budget state: {"active_elapsed_seconds": 0, "available_unreserved_seconds": 12461.475947980536, "completed_gpu_seconds": 15850.98291406408, "counting": "elapsed GPU command once; includes failures and retries", "cumulative_gpu_seconds": 93136.50980287313, "deadline_at": "2026-10-12T05:44:04.662168+00:00", "historical_gpu_seconds": 76797.98575085367, "incremental_cap_seconds": 28800, "incremental_spent_seconds": 16338.524052019464, "preflight_gpu_seconds": 487.54113795538433, "protected_collection_seconds": 0, "protected_retry_seconds": 0, "reservation_releases": [{"at": "2026-10-12T00:05:44.222425+00:00", "released_to": "declared_optional_priority", "reservation": "unused_technical_retry", "seconds": 2700}], "reserved_gpu_seconds": 0, "updated_at": "2026-10-12T02:08:57.864635+00:00"}

Bundle dispositions:
- A0_grid: completed (no additional reason).
- A0_reanchor: completed (no additional reason).
- R4_CE: completed (no additional reason).
- collection: completed (no additional reason).
- confirmation: failed_scientific_gate (C3 did not meet all prospective criteria).
- diagnostics: completed (no additional reason).
- optional_A1_grid: completed (no additional reason).
- optional_A1_reanchor: completed (no additional reason).
- optional_A3_CE: completed (no additional reason).
- pilot: completed (no additional reason).
- target_refresh: completed (no additional reason).

Completed/reused selected readouts (mean across available seeds; external sets remain research exposed):

| Family | Source mIoU | STURM pooled water IoU | WF pooled water IoU | WF equal-event water IoU | Seeds |
|---|---:|---:|---:|---:|---:|
| A0_LKD_lr0.0005 | 85.84 | 65.53 | 64.51 | 46.24 | 1 |
| A0_LKD_reanchored | 87.32 | 65.34 | 64.13 | 46.74 | 4 |
| A0_LKD_recipe | 84.65 | 63.31 | 61.07 | 42.72 | 4 |
| A1_LKD_lr0.0005 | 86.28 | 65.25 | 62.03 | 46.30 | 1 |
| A1_LKD_reanchored | 86.74 | 65.19 | 63.66 | 45.94 | 4 |
| A1_LKD_recipe | 83.21 | 62.89 | 58.80 | 39.07 | 4 |
| A3N_LKD | 85.56 | 65.67 | 64.82 | 44.93 | 4 |
| A3_CE | 86.81 | 63.74 | 62.92 | 45.83 | 4 |
| A3_LKD | 86.57 | 65.00 | 63.11 | 44.44 | 4 |
| C0 | 87.67 | 64.71 | 65.98 | 47.95 | 2 |
| C1 | 87.67 | 64.71 | 65.98 | 47.95 | 2 |
| C2 | 87.67 | 64.71 | 65.98 | 47.95 | 2 |
| C3 | 87.67 | 64.71 | 65.98 | 47.95 | 2 |
| C4 | 87.67 | 64.71 | 65.98 | 47.95 | 2 |
| C5 | 87.67 | 64.71 | 65.98 | 47.95 | 2 |
| HC | 87.67 | 64.38 | 65.36 | 47.30 | 2 |
| HL | 87.67 | 64.71 | 65.98 | 47.95 | 2 |
| R4_CE | 88.34 | 63.54 | 65.48 | 46.99 | 4 |
| R4_LKD | 87.94 | 64.71 | 65.95 | 47.59 | 4 |

Historical cohort: regenerated approved N2500 with 127 replacement tiles; dataset SHA256 80f9e13c136abe8169b8a5a064e265717b8d241ee879d945fd279ffbbed52711. This is not an exact reproduction of the historical GDACS subset.
Six historical tiny-rung runs stopped under inherited patience; they retain their original identity and fixed-budget exception. No claim is made that the joint external capacity floor or a universal small-model KD law has been found.
Known ingredients include spectral-index integration, selective/disagreement distillation, and learned quantizer step sizes. No first-ever novelty claim is made; failure of a declared gate is not redefined as success.

Reproduce CPU analysis: `/home/schmalstieg/miniconda3/envs/esi/bin/python /home/schmalstieg/repos/FloodDistill/experiments/revision/capacity_v7_20261011T210730Z/collect_v7.py --run-dir /home/schmalstieg/repos/FloodDistill/experiments/revision/capacity_v7_20261011T210730Z --reason status`.
Data tables: results.csv, comparisons.csv, event_results.csv, mechanism_counts.csv, deployment.csv. Missing metric cells are explicit empty CSV values with status.
