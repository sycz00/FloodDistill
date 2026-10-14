# V9 research report — Recovery collection; scientific evaluation incomplete

Collected 2026-10-13T10:26:06.401223+00:00. Run `research_v9_recovery_20261013T075000Z`.

Completed checklist entries: 5. Technical failures: 1. Deferred/unavailable entries: 7.

GPU budget snapshot: {"remaining_unreserved_gpu_hours": 1.6656937373665521, "reserved_gpu_hours": 0.0, "verified_gpu_hours": 28.334306262633447}. Initial GPU wait: 30.21404719352722 seconds. Concurrent GPU-command time is summed; CPU time and waits are separate.

Sealed metadata: 468 retained scenes, 116 events; 23 exclusions; 30458 input-only upper windows. Actual scored windows are determined only in the final reference read. Prior code/log inspection found no retained executed train/val GT read; no historical OS open trace exists.

## Executed evidence

The experiment_checklist.csv records every required and conditional block, controls, state, evidence and deferral reason. same_image_label_swap.csv contains four paired true-CE controls. edge_arms_results.csv and edge_arms_by_event.csv preserve all outcomes. failure_strata.csv reports recall, false water and share of misses. ood_comparison_table.csv and threshold_free_ap.csv distinguish pooled versus equal-event comparisons. operating_point_analysis.csv separates source selection, label-free rules and unattainable event oracles. dry_window_alarms.csv reports independent-reference flood-scene dry land at0.1/1/5% area thresholds. full_precision_frontier.csv keeps simulator accuracy separate from native deployment.

## Declared scientific gates

[
  {
    "arm": "E1",
    "status": "UNAVAILABLE",
    "sealed_delta": null,
    "event_bootstrap_low": null,
    "event_bootstrap_high": null,
    "across_seed_SD": null,
    "positive_seeds": 0,
    "paired_events": 0,
    "S11_thin_recall_delta": 0.004035512510088775,
    "STURM_delta": -0.012151250653289397,
    "sealed_dry_alarm_1percent_delta": null,
    "qualification": "Seed-mean whole-event bootstrap10000 seed1729; seedSD separate; nonsignificance never equivalence"
  },
  {
    "arm": "E2",
    "status": "UNAVAILABLE",
    "sealed_delta": null,
    "event_bootstrap_low": null,
    "event_bootstrap_high": null,
    "across_seed_SD": null,
    "positive_seeds": 0,
    "paired_events": 0,
    "S11_thin_recall_delta": -0.07695525502470522,
    "STURM_delta": -0.005623050720839684,
    "sealed_dry_alarm_1percent_delta": null,
    "qualification": "Seed-mean whole-event bootstrap10000 seed1729; seedSD separate; nonsignificance never equivalence"
  },
  {
    "arm": "E3",
    "status": "UNAVAILABLE",
    "sealed_delta": null,
    "event_bootstrap_low": null,
    "event_bootstrap_high": null,
    "across_seed_SD": null,
    "positive_seeds": 0,
    "paired_events": 0,
    "S11_thin_recall_delta": 0.1073495541250812,
    "STURM_delta": -0.00690744334381066,
    "sealed_dry_alarm_1percent_delta": null,
    "qualification": "Seed-mean whole-event bootstrap10000 seed1729; seedSD separate; nonsignificance never equivalence"
  },
  {
    "arm": "D",
    "status": "UNAVAILABLE",
    "sealed_delta": null,
    "event_bootstrap_low": null,
    "event_bootstrap_high": null,
    "across_seed_SD": null,
    "positive_seeds": 0,
    "paired_events": 0,
    "S11_thin_recall_delta": null,
    "STURM_delta": 0.0013622836881706513,
    "sealed_dry_alarm_1percent_delta": null,
    "qualification": "Seed-mean whole-event bootstrap10000 seed1729; seedSD separate; nonsignificance never equivalence"
  }
]

No attribution claim follows merely because two arms are nonsignificantly different. Positive findings require the complete four-seed paired protocol and the declared mechanism/dry-alarm safeguards.

## What must be done

Authors must apply verified numerical/terminology corrections, reconcile the teacher baseline under one protocol, consolidate component evidence and write the Discussion and point-by-point response. Present OOD transfer at the event level and retain unfavorable outcomes. Independently verify ordinary dry imagery before making unattended false-alarm claims. Establish source training provenance for the conventional UNet before the alternative-teacher control. Native low-bit deployment remains unsupported on this workstation simulator.

## Numerical readout policy

Whole-smoke batch provenance was corrected by reproducing the frozen Trainer seed reset. TF32OFF restored the floating-point CPU/GPU comparison. Unchanged PTQ parent64 had128 CPU/GPU class differences in one real batch; the CPU simulator readout was blocked and all PTQ predictions use the admitted GPU route. Original failed attempts remain charged. No tolerance was relaxed and no training recipe changed.

## Limitations

The teacher saw the252 source images. The cohort was selected using Prithvi. Earlier public benchmarks and rule-family discovery are research-exposed. The sealed set loses scenes/events under conservative overlap exclusions and is not automatically the prompt’s130-event population. Teacher token size differs from decoder output stride. AP uses declared4096-bin ties, EM/anchor inherited50 bins; floating-point CPU/GPU parity is documented by smoke receipts. Seed variability and whole-event uncertainty are distinct.

## Failures and deferrals

[
  {
    "block": "final",
    "job": "sealed_readout_retry01",
    "status": "TECHNICAL_FAILED",
    "seeds": null,
    "paired_controls": null,
    "reason": "telemetry/foreign contention: only owned process stopped",
    "evidence": "/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_recovery_20261013T075000Z/command_receipts/sealed_readout_retry01.json"
  },
  {
    "block": "cpu_final",
    "job": "sealed_analysis_retry01",
    "status": "UNAVAILABLE",
    "seeds": null,
    "paired_controls": null,
    "reason": null,
    "evidence": "/home/schmalstieg/repos/FloodDistill/experiments/revision/research_v9_recovery_20261013T075000Z/command_receipts/sealed_analysis_retry01.json"
  },
  {
    "block": "B0 E1 transfer",
    "status": "UNAVAILABLE",
    "reason": "Original B0 hard-label checkpoints lack full embedded matched input/initialization lineage; eight new N2500 B0 runs exceed protected freed-budget allowance; required normal+sealed inference remains protected; no benchmark scores are used",
    "paired_controls": "four compatible B0 E0/E1 pairs",
    "evidence": "manifest.json"
  },
  {
    "block": "hard negatives A1/A2",
    "status": "DEFERRED_BUDGET",
    "reason": "Eight complete N2500 trainings plus pool-feature/candidate-overlap preparation require3360s beyond core and full inference reservations; required normal+sealed inference remains protected; no benchmark scores are used",
    "paired_controls": "E0 plus four A1/A2 pairs",
    "evidence": "manifest.json"
  },
  {
    "block": "human S2 UNet teacher students",
    "status": "UNAVAILABLE",
    "reason": "Located candidate lacks verified252-human-source training provenance; no new teacher training authorized; required normal+sealed inference remains protected; no benchmark scores are used",
    "paired_controls": "four matched Prithvi/UNet student pairs",
    "evidence": "manifest.json"
  },
  {
    "block": "same-image soft-KD extension",
    "status": "DEFERRED_BUDGET",
    "reason": "Eight full live-teacher CE+KL trainings need a conservative6000s paired allowance beyond protected core/inference; primary true-CE pairs take priority; required normal+sealed inference remains protected; no benchmark scores are used",
    "paired_controls": "four human/teacher-hard pairs with identical0.5*T^2 KL T4",
    "evidence": "manifest.json"
  },
  {
    "block": "R4 FP/QAT continuation",
    "status": "DEFERRED_BUDGET",
    "reason": "No competent verified tuned continuation recipe; bounded source pilot plus complete FP/QAT readouts need a separate1800s reservation; unchanged PTQ has priority; required normal+sealed inference remains protected; no benchmark scores are used",
    "paired_controls": "source-only LR pilot then matched FP/QAT parents42/64",
    "evidence": "manifest.json"
  },
  {
    "block": "11-window independent dry review",
    "status": "UNAVAILABLE",
    "reason": "No independent human verification records; teacher/rule labels do not substitute",
    "evidence": "/home/schmalstieg/repos/FloodDistill/experiments/revision/prompt3_dry_verification/annotations.template.json"
  }
]

# Claim boundary and checked primary references

[Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) is the pretrained backbone reference. Its current paper version is v3 (6 March2026); the local experiment instead pins the actual task-adapted checkpoint bytes. Token patch size is distinct from the inspected downstream decoder output stride.

[PANGAEA](https://arxiv.org/abs/2412.04204) compares geospatial foundation models with supervised segmentation baselines and finds that foundation models do not consistently outperform them. The present experiment tests one fixed task-adapted teacher and does not establish general foundation-model necessity.

[SegFix](https://arxiv.org/abs/2007.04269) already studies model-agnostic boundary refinement. Its learned boundary/interior correspondence differs from this fixed MNDWI re-decision, but boundary refinement itself is established. Pseudo-label refinement, guided image filtering and spectral water indices are established ingredients; this follow-up makes no first-in-literature claim.

[WorldFloods-v2 paper](https://www.nature.com/articles/s41598-023-47595-7) describes a cloud-aware optical flood-mapping framework and dataset. Local native-window/cloud/permanent-water handling is frozen separately and verified against retained evaluator code. Mixed reference satellite/date-lag groups and conservative overlap exclusions limit the interpretation of a teacher-edge hypothesis.

Primary pages checked13October2026. These anchors are context, not exhaustive novelty clearance. Any contribution claim depends on the complete paired controls and declared sealed-test gates in frozen_protocol.md.

## Detailed analysis and scope qualifications

| Set | Family | Models | Equal-event water IoU | Seed SD |
|---|---|---:|---:|---:|
|s11val|Prithvi_fixed_teacher|1|80.70875474410202|not a seed sample|
|s11val|lkd|4|77.271 pp|0.492 pp|
|s11val|E0|4|77.079 pp|0.645 pp|
|s11val|E1|4|77.843 pp|0.713 pp|
|s11val|E2|4|75.752 pp|0.970 pp|
|s11val|E3|4|78.083 pp|0.432 pp|
|s11val|B0_human_labels|4|56.777 pp|6.725 pp|
|s11val|B0_teacher_labels|4|57.216 pp|6.188 pp|
|s11val|paper_ofa|4|74.761 pp|1.024 pp|
|s11val|PTQ|4|74.517 pp|0.897 pp|
|s11val|NDWI|1|55.1822020107399|not a seed sample|
|s11val|MNDWI|1|62.902454925633|not a seed sample|
|s11|Prithvi_fixed_teacher|1|79.06333464790154|not a seed sample|
|s11|lkd|4|80.260 pp|0.394 pp|
|s11|E0|4|79.828 pp|0.327 pp|
|s11|E1|4|80.726 pp|0.432 pp|
|s11|E2|4|78.852 pp|0.477 pp|
|s11|E3|4|79.936 pp|1.179 pp|
|s11|B0_human_labels|4|61.897 pp|3.725 pp|
|s11|B0_teacher_labels|4|61.730 pp|3.707 pp|
|s11|paper_ofa|4|75.783 pp|0.683 pp|
|s11|PTQ|4|78.617 pp|1.422 pp|
|s11|NDWI|1|67.3756182996331|not a seed sample|
|sturm|Prithvi_fixed_teacher|1|38.868659240551295|not a seed sample|
|sturm|lkd|4|37.841 pp|1.064 pp|
|sturm|E0|4|36.368 pp|0.082 pp|
|sturm|E1|4|35.152 pp|1.850 pp|
|sturm|E2|4|35.805 pp|1.339 pp|
|sturm|E3|4|35.677 pp|0.897 pp|
|sturm|B0_human_labels|4|32.106 pp|7.383 pp|
|sturm|B0_teacher_labels|4|32.865 pp|6.678 pp|
|sturm|paper_ofa|4|37.020 pp|0.618 pp|
|sturm|PTQ|4|36.365 pp|1.166 pp|
|sturm|NDWI|1|33.00161530311869|not a seed sample|
|wf|Prithvi_fixed_teacher|1|50.27241130032163|not a seed sample|
|wf|lkd|4|47.125 pp|0.854 pp|
|wf|E0|4|46.530 pp|0.367 pp|
|wf|E1|4|47.703 pp|0.438 pp|
|wf|E2|4|45.321 pp|1.119 pp|
|wf|E3|4|50.170 pp|1.298 pp|
|wf|B0_human_labels|4|30.268 pp|6.039 pp|
|wf|B0_teacher_labels|4|31.345 pp|5.864 pp|
|wf|paper_ofa|4|45.498 pp|0.707 pp|
|wf|PTQ|4|45.438 pp|1.881 pp|
|wf|NDWI|1|29.26465394688828|not a seed sample|
|s11|MNDWI|1|68.32849274052624|not a seed sample|
|sturm|MNDWI|1|39.28966636148362|not a seed sample|
|wf|MNDWI|1|41.01949829987871|not a seed sample|

These family means are descriptive. The primary whole-event interval and mechanism/dry gates are in scientific_gates.json; no claim follows from a mean alone. Every model/event/seed is retained in the CSVs. PTQ event-specific gap removal uses source thresholds only (ptq_operating_point_by_event.csv). model_size_frontier.csv separates FP parameters, serialized model containers, theoretical packed parameters, metadata and unavailable native engine/kernel measurements.

Full sealed matrix cost admission: {"admitted": true, "ceiling_seconds": 7200, "conservative_seconds": 3975.24187758273, "created": "2026-10-13T09:49:25.227800+00:00", "criteria": "All predictor rows together; cost/provenance only; 25% runtime headroom+120s setup; no target scores inspected; never drop slower models or seed controls to fit", "full_matrix_estimated_seconds": 3084.193502066184, "model_records": 17, "normal_measured_seconds": 639.8786384388804, "status": "READY", "target_window_count": 17907}

The original and repaired whole-smoke attempts are preserved and charged. The standalone smoke missed the Trainer seed reset; the actual training adapter already used it. TF32 and PTQ numerical failures were diagnosed before sealed labels. Future GPU inference resets TF32 flags after model construction, since the teacher factory can reset matmul precision. The immutable source teacher masks used cuDNN TF32off with factory matmul precision medium; both paired arms share the exact cached target bytes.

The primary B0 checkpoints retain two legacy metadata descriptions: data_position_state.completed_epochs divides by156 and the scheduler sentence mentions156 updates. For these252-image runs the authoritative receipt/config/training_state/epoch records show15 updates per epoch,50 epochs and750 completed updates. Both arms follow that identical executed recipe; no exact mid-epoch resume is claimed.

GDACS native264-pixel patches are resized224, so the training index/band width operates on about11.79m pixels; the native WF512-window decode operates on10m pixels. Teacher16-pixel tokens are not16-pixel decoder logits. EfficientViT-B0 uses its inspected H/8 segmentation head before full-resolution interpolation. STURM paper masks include both mapped water classes; WF excludes permanent water with JRC3. Spectral strata overlap and the component halfwidth definition does not measure every thin local branch. Ordinary independent dry imagery and the11-window human review remain open.

Authors must incorporate verified corrections, consolidate the evidence under consistent protocols, write the Discussion and response, and qualify any contribution by the sealed gates. If E1 fails, present the inherited-failure characterization and retained negative controls. If simpler E2/E3 explains a gain, retain that explanation; a nonsignificant difference does not establish equivalence. No manuscript was changed or submitted.

## Recovery audit

All20 original training runs were completed and reused. The final teacher readout crashed only in backend precision telemetry, after3639 teacher windows; dependent readouts were unavailable. This fresh queue reruns the full prediction matrix without reusing partial predictions, retains original clocks and cumulative30 GPU-command-hour and incremental4-hour caps. Historical reports and failures remain unchanged.

Scientific evaluation complete: False. Operational collection does not imply complete experimental results.

## Approved recovery window

The user explicitly approved extending execution to18:35 CEST on13October2026. The original launch and first-GPU timestamps remain recorded. This approval overrides the eight-hour execution cutoff only; the original absolute wall limit, cumulative30 GPU-command-hour cap, inherited4-hour V9 incremental cap, scientific settings and ten-minute collection reserve remain enforced. Approval receipt: deadline_extension_approval.json.
