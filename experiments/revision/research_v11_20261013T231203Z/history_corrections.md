# V11 historical corrections and claim limits

This is an additive analysis sidecar. It does not edit V9/V10 protocols, caches, tables, or running inputs. Evidence hashes are in `reviewer_evidence.json`. Reviewer comments were read directly from `/home/schmalstieg/repos/MajorRev/paper/review.docx`; the main paper and supplement were checked for data roles, geometry and deployment qualifications.

## B0 budget row correction

| Variant | N | S11 pooled mIoU | Sample SD | Seeds | CSV line |
|---|---:|---:|---:|---:|---:|
| matched_lkd_clip5_valid | 100 | 0.7840466290 | 0.0165835120 | 4 | 62 |
| matched_lkd_clip5_valid | 2500 | 0.8781847837 | 0.0022176820 | 4 | 92 |

Verified file: `/home/schmalstieg/repos/FloodDistill/experiments/revision/prompt5_closure/tables/current_full_summary.csv`; SHA256 `61829195a517c77f1964e814cc0c7748e084d694c07cd905f1c9fe58fcea29c1`.

The active V10 frozen A1 prose assigns 78.4% to `matched_lkd_clip5_valid_N2500`; the table assigns it to N100. N2500 is 87.81847837%, so this mix-up supplies no basis for excluding the N2500 comparator. This verifies the existing summary row identity, not a fresh re-evaluation of four checkpoints. Any genuine anomaly must be investigated by provenance and retained; no score-based seed exclusion or retraining to restore an expected test score.

## Teacher cache: serialization versus non-equivariance

Historical five-TRAIN-tile receipt: `research_v10_20261013T112818Z/teacher_cache_feasibility.json`. Maximum absolute FP16 serialization difference was 0.00390625 logit units; flip mismatch reached 3.4354996681 and rotation mismatch 2.7732748985. These are different checks. Float16 roundoff needs magnitude/dtype-aware tolerance; approximate teacher rotation/flip equivariance must not be imposed as exact correctness.

A jointly transformed original teacher target is a coherent objective but generally differs from a fresh teacher prediction of the transformed image. Active V10 already has the approved `float32_eight_geometric_views` policy: fresh clean geometric views, with photometric jitter applied to the student input only. V11 reuses the completed verified cache when available. It must not regenerate, rewrite or silently transform that running cache. A manifest must be atomically complete and input/teacher/file hashes matched before cache use. No new teacher inference was run for this sidecar.

Active policy SHA256: `6e2bb22d3c5a463f304d6df88f1e11f241da79fa0ed9e7b8a94a18da6b0c7447`. Historical mismatch receipt SHA256: `427c17fd42d7dc146da7e60ea0d1d36fdc4943c7026f25ad6921310af246fe37`.

## Corrections to interpretation

- Shared misses, shared false water and removal of errors using reference labels are shared-error and reference-conditional repair diagnostics. They do not establish a fundamental information ceiling for every six-band optical model.
- Preserve joint teacher/current-FP/starting-FP/quantized correctness codes. Teacher-inherited error and FP-correct/quantized-wrong error can overlap; additive exclusive categories would lose that overlap.
- Unfavorable scores remain evidence. Investigate identity, data, masks, precision and recipe before interpreting; never delete seeds or select comparators based on exposed test scores.
- Similar scores after threshold matching support an operating-point explanation. They do not prove causal equivalence. Human-S11 versus teacher-GDACS changes several factors; a null soft-KD increment does not show that data alone caused a gain.
- A high absolute score can coexist with a negative paired-difference interval. A small mean penalty does not rule out substantial event-specific harm, and nonsignificance does not prove equivalence.

## Roadmap and reviewer relevance

V8 found complementary correctable errors without reliable transfer in its bounded pilot: reported uniform ridge about -0.378 pp, reliability-weighted ridge about -0.552 pp, and 20-step head KD about +0.079 pp query mIoU. These source-reported context values were read from the audited V8 report, not recomputed here. V8 exposed benchmarks remain exposed; renamed protocols cannot reseal them. Confidence must not be equated with correctness.

V9/V10 supply the primary supervision, architecture, size, operating-point and precision comparisons. V11 tests whether same-weight FP subtraction reallocates teacher loss more usefully than U/A/L/N/P, with explicit damage and speed diagnostics. The independent C0-C4 conditioner is a separate exploratory question. Neither package can promise reviewer closure, establish general novelty, fill the missing independent dry-scene labels, or convert simulated quantization into measured integer-kernel speed.

## Independent conditioner readiness

R4_micro has native head feature width 4 (`build_tiny` uses `ContextConv(..., dec=4)`), hence C3/C4 have 13*(4+1)=65 coefficients, within the 512 cap. Float32 coefficient bytes are 260; feature and descriptor means/scales add 128 bytes. Descriptor reductions, clipping, dynamic head construction and inference remain required measured runtime costs; the current count is not a latency result.

The stored GDACS TRAIN image is a 224 full-field resize, not a random 224 crop. The paper appendix and current input loader agree. The canonical evaluator uses 512 tiles, which have different geographic fields of view across data sources; summary statistics are not asserted identical. The source 224-versus-512 descriptor/output audit has not been run. V9 normal caches expose image, reference mask and cloud fields, but do not provide a distinct deployable input-validity mask. A reference mask must not be repurposed for the image descriptor.

All four completed R4_micro KDoff parents and the final eight-view cache manifest are required. No architecture substitution, partial seed set, backbone retraining or external-score selection is allowed. The optional action records DEFERRED if prerequisites or geometry/mask audit remain unavailable after the main block. Numerical solver tests and implementation are not a trained probe result.
