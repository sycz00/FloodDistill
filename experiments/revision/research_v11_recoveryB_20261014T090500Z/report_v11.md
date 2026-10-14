# V11 selective repair — final operational report

This bounded add-on tests whether subtracting teacher mismatch to the current FP shadow improves quantization repair beyond uniform, absolute-KL, disagreement, and randomized class-matched supervision. The separate optional R4_micro classifier probe tests image conditioning against static and threshold controls. No outcome is presumed positive.

V10 retains priority for reviewer-critical supervised comparisons and the size/precision frontier. The reviewer letter, paper, supplement and previous-roadmap corrections are indexed in `reviewer_evidence.json` and `history_corrections.md`.

Worker phase: **COLLECTING**. Report refreshed at UTC epoch 1791974374.946.

## Execution evidence

| Job | Status | Reason |
|---|---|---|
| 00_inventory | COMPLETED |  |
| 01_resolve | COMPLETED |  |
| 02_source_gate | COMPLETED |  |
| 03_profile_D_s42 | COMPLETED |  |
| 04_cost_admission | COMPLETED | None |
| 10_train_A_s107 | COMPLETED | None |
| 10_train_A_s42 | COMPLETED | None |
| 10_train_A_s64 | COMPLETED | None |
| 10_train_A_s91 | COMPLETED | None |
| 10_train_D_s107 | COMPLETED | None |
| 10_train_D_s42 | COMPLETED | None |
| 10_train_D_s64 | COMPLETED | None |
| 10_train_D_s91 | COMPLETED | None |
| 10_train_F0_s107 | COMPLETED | None |
| 10_train_F0_s42 | COMPLETED | None |
| 10_train_F0_s64 | COMPLETED | None |
| 10_train_F0_s91 | COMPLETED | None |
| 10_train_L_s107 | COMPLETED | None |
| 10_train_L_s42 | COMPLETED | None |
| 10_train_L_s64 | COMPLETED | None |
| 10_train_L_s91 | COMPLETED | None |
| 10_train_N_s107 | DEFERRED | Per-job/deadline ceiling; protected collection margin |
| 10_train_N_s42 | DEFERRED | Per-job/deadline ceiling; protected collection margin |
| 10_train_N_s64 | DEFERRED | Per-job/deadline ceiling; protected collection margin |
| 10_train_N_s91 | DEFERRED | Per-job/deadline ceiling; protected collection margin |
| 10_train_P_s107 | DEFERRED | Per-job/deadline ceiling; protected collection margin |
| 10_train_P_s42 | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 10_train_P_s64 | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 10_train_P_s91 | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 10_train_U_s107 | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 10_train_U_s42 | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 10_train_U_s64 | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 10_train_U_s91 | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 20_source_selection | DEFERRED | Complete block and protected evaluation cannot fit fixed window |
| 21_optional_decision | COMPLETED | None |
| 22_conditioner | DEFERRED | None |
| 23_final_freeze | COMPLETED | None |
| 30_final_evaluation | DEFERRED | No complete source-frozen model matrix available for final readout |

## Scientific readout

Source gate: `repair/core/source_gate.json`. {"backend": "same CPU float32 backend for both paths; simulator only", "inputs_sha256": "63fadff40dac166395cadbec480caea3f7098555d3aafe26cf3f6ac218f61835", "mean_water_IoU_loss": 0.21614180525711546, "outcome": "ADMITTED", "point": "core", "rule": "mean equal-event water IoU loss>=0.003 OR supported dark/bright error increase>=0.01", "seeds": [42, 64, 91, 107], "source_only": true, "strata": {"bright_water_miss": {"difference": 0.13863113702000215, "passed": true, "reference_pixels": 112188, "sufficient_support": true}, "dark_land_false_water": {"difference": 0.19178525226390686, "passed": true, "reference_pixels": 9276, "sufficient_support": true}}, "water_IoU_gate": true}

External V11 evaluation has not completed. Empty CSVs are schemas, not zero-valued scientific outcomes. Missing dependencies, cost deferrals and interrupted blocks are not negative scientific results.

Optional classifier status: {"action": "Retain solver/tests; do not substitute architecture, incomplete seeds, or label-derived descriptor masks", "backbone_training": false, "cost_inventory": {"C0_coefficients": 0, "C1_coefficients": 5, "C2_coefficients": 17, "C3_C4_coefficients": 65, "dynamic_head_construction_MACs": 60, "excluded_from_MAC_count": "Descriptor pixel reductions, standardization/clipping, native folding, and inference; must be measured end to end before frontier claims", "feature_width": 4, "fp32_coefficient_bytes_C3": 260, "latency_measured": false, "stored_statistics_bytes_fp32": 128, "strict_integer": false}, "cost_limit_reserved_core_hours": 12, "dependency_status": "WAITING_DEPENDENCY", "forms": ["C0", "C1", "C2", "C3", "C4"], "invalid": [], "missing": ["/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v10_recoveryB_20261014T084000Z/scientific_receipts/R4_micro_KDoff_s42.json", "/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v10_recoveryB_20261014T084000Z/scientific_receipts/R4_micro_KDoff_s64.json", "/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v10_recoveryB_20261014T084000Z/scientific_receipts/R4_micro_KDoff_s91.json", "/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v10_recoveryB_20261014T084000Z/scientific_receipts/R4_micro_KDoff_s107.json"], "outcome": "not_executed", "package": "independent_conditioner", "parents": [], "process_cpu_seconds": 0.0022521429999999842, "reason": "Four completed R4_micro KDoff parents and verified V10 teacher cache unavailable at optional decision", "seeds": [42, 64, 91, 107], "solver_implemented": true, "source_probe_executed": false, "status": "DEFERRED", "wall_seconds": 0.002252901904284954}

The optional 1,560-step extension and second precision point are deferred in this core-first launch. Source-selected checkpoints are retained; any secondary external readouts omitted to protect the primary matrix remain explicitly deferred in the final freeze.

## Resource and dependency accounting

CPU-only V11; at most eight allocated threads. Maximum cumulative reserved charge is 64 core-hours, with final evaluation and collection protected before core admission. Old workers are conservatively reserved at configured maxima until terminal. No GPU training or additional GPU reservation is made.

Exact actual/active/reserved costs: `resource_ledger.json`. Process CPU time is recorded separately from elapsed time multiplied by assigned threads.

Current prerequisite ledger: `dependency_evidence_ledger.json`. Frozen V10 cache policy uses float32 logits for eight fresh geometric views, with photometric jitter applied only to student inputs. No partial cache shard or unfinished checkpoint is admitted.

## Interpretation and next step

When dependencies finish, source-only deficit and measured-cost gates determine whether the full paired block can run. D must beat U and A, with L/N/P retained, before subtraction-specific follow-up is warranted. If no complete block fits, retain the exact waiting/deferral cause and continue V10 first. Optional probes cannot consume the protected primary evaluation budget.

All external benchmarks, including reused WorldFloods sealed membership, are treated as non-confirmatory research readouts. Checkpoint/settings freeze precedes the single external phase. Native masks, event grouping and the 512-pixel evaluation geometry are retained. Fake quantization quality does not establish native low-bit speed.

Prior-art boundaries: [QAT-ACS](https://arxiv.org/abs/2306.07215), [GHOST](https://arxiv.org/abs/2301.00131), [Teacher Intervention](https://aclanthology.org/2023.eacl-main.64/), [Dynamic Filter Networks](https://arxiv.org/abs/1605.09673). Disagreement-based selection, teacher-assisted quantization and input-conditioned weights are established; this run makes no first-ever claim.

Fresh-terminal commands and limits are in `CONTROLS.txt`. Detachment is verified separately in `launch_verification.json`; host shutdown/sleep or loss of allocation stops progress.
