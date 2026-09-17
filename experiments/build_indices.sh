#!/usr/bin/env bash
# Build the selector indices with the recorded experiment settings.
# Run the feature and consistency pre-passes before R2-Core or weighted GLISTER;
# consistency uses the teacher that generated the labels.
# Filtered arms share f in [0.05,0.85] and conf_std >= 0.05 (5,987 candidates).
# Raw arms use the whole 23,720-tile pool. Each arm selects N=2,500.
# Usage: PSEUDO_DIR=data/pseudo_gdacs GPU=0 bash experiments/build_indices.sh
# Use ARMS=nsweep to build only the per-N FPS indices.

set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"
GPU="${GPU:-0}"
N="${N:-2500}"
PSEUDO_DIR="${PSEUDO_DIR:-./data/pseudo_gdacs}"
TEACHER_CKPT="${TEACHER_CKPT:-hf}"   # the public Prithvi checkpoint; a .pt path overrides it
# GLISTER's outer objective is anchored on a reference student; this is the checkpoint the
# published glister_index.json used, and it is recorded in that index's score_fn string.
GLISTER_REF="${GLISTER_REF:-./runs/ablations_sen1/sen_ofa_N250_s42/best.pt}"
# Per-N FPS indices for the pseudo-vs-GT sweep. Each N needs its own index built at --top_k N:
# PseudoLabeledDataset draws n_samples uniformly at random from the index pool and never
# re-reads the ranking, so a 2,500-entry index with --n_samples 50 is a random 50, not a
# curated one.
NSWEEP_NS=(${NSWEEP_NS:-20 40 50 100 150 200 252 2500 4000})
ARMS="${ARMS:-prepass filtered raw nsweep}"

RAW=(--min_ff 0 --max_ff 1 --min_conf_std 0)

has() { [[ " ${ARMS} " == *" $1 "* ]]; }
step() { echo; echo "== $* =="; }

#  PRE-PASSES
if has prepass; then
    step "pool features -> ${PSEUDO_DIR}/pool_features.csv"
    "${PY}" scripts/curate.py features --pseudo_dir "${PSEUDO_DIR}"

    step "teacher consistency c(x) -> ${PSEUDO_DIR}/consistency.csv"
    "${PY}" scripts/curate.py consistency --pseudo_dir "${PSEUDO_DIR}" \
        --official_ckpt "${TEACHER_CKPT}" --gpu "${GPU}"
fi

#  FILTERED ARMS (the matched-pool comparison)
if has filtered; then
    step "FPS diversity  -> diversity_N${N}_index.json"
    "${PY}" scripts/curate.py diversity --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}"

    step "R2-Core        -> r2core_index.json"
    "${PY}" scripts/curate.py r2core --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}"

    # Decomposition arm: same objective with w == 1. Isolates what the reliability weighting
    # contributes on top of plain facility location (measured answer: nothing detectable).
    step "R2-Core, uniform w (ablation) -> r2core_fl_only_index.json"
    "${PY}" scripts/curate.py r2core --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        --uniform_weights --out_name r2core_fl_only_index.json

    step "GLISTER Alg.2 r=25 -> glister_index.json"
    "${PY}" scripts/curate.py glister --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        --ref "${GLISTER_REF}" --r 25 --lam 1.0 --eta 3e-4 --gpu "${GPU}"

    # Single-factor variants of the GLISTER arm, so any difference is attributable.
    #  _r2  R2-Core's WEIGHTED facility location in the lambda*R(S) slot
    #  _flive lambda*R(S) made active (--lam_scale_mode std --lam_scale_per_round):
    #  scaling lambda by the mean cannot change an argsort, only spread can, so the
    #  plain arm's regularizer moved 1 tile in 2500 and the arm is plain GLISTER,
    #  not F-GLISTER. Do not call the plain arm F-GLISTER in writing.
    #  _ood  bilevel selection against a HAZED outer objective  -  the falsification test for
    #  "val-anchoring is what buys haze robustness"
    step "GLISTER + R2-Core reg weights -> glister_r2_index.json"
    "${PY}" scripts/curate.py glister --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        --ref "${GLISTER_REF}" --r 25 --lam 1.0 --eta 3e-4 --gpu "${GPU}" \
        --reg_weights r2core --lam_scale_mode std --lam_scale_per_round \
        --out_name glister_r2_index.json

    step "GLISTER, regularizer live -> glister_flive_index.json"
    "${PY}" scripts/curate.py glister --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        --ref "${GLISTER_REF}" --r 25 --lam 1.0 --eta 3e-4 --gpu "${GPU}" \
        --lam_scale_mode std --lam_scale_per_round \
        --out_name glister_flive_index.json

    step "GLISTER, hazed outer objective -> glister_ood_index.json"
    "${PY}" scripts/curate.py glister --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        --ref "${GLISTER_REF}" --r 25 --lam 1.0 --eta 3e-4 --gpu "${GPU}" \
        --val_haze 0.10 --val_haze_mode orthogonal \
        --out_name glister_ood_index.json

    step "confidence, matched pool -> confidence_filtered_index.json"
    "${PY}" scripts/curate.py confmatch --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}"

    step "random, matched pool (control) -> random_filtered_index.json"
    "${PY}" scripts/curate.py random --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}"
fi

#  RAW ARMS (no pre-filter)
if has raw; then
    step "FPS diversity, no pre-filter -> diversity_raw_index.json"
    "${PY}" scripts/curate.py diversity --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        "${RAW[@]}" --out_name diversity_raw_index.json

    step "R2-Core, no pre-filter -> r2core_raw_index.json"
    "${PY}" scripts/curate.py r2core --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        "${RAW[@]}" --out_name r2core_raw_index.json

    step "GLISTER, no pre-filter -> glister_raw_index.json"
    "${PY}" scripts/curate.py glister --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        --ref "${GLISTER_REF}" --r 25 --lam 1.0 --eta 3e-4 --gpu "${GPU}" \
        "${RAW[@]}" --out_name glister_raw_index.json

    step "confidence, no pre-filter -> confidence_raw_index.json"
    "${PY}" scripts/curate.py confmatch --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        "${RAW[@]}" --out_name confidence_raw_index.json

    step "random, no pre-filter -> random_raw_index.json"
    "${PY}" scripts/curate.py random --pseudo_dir "${PSEUDO_DIR}" --top_k "${N}" \
        "${RAW[@]}" --out_name random_raw_index.json
fi

#  PER-N FPS INDICES for the pseudo-vs-GT sweep
if has nsweep; then
    for n in "${NSWEEP_NS[@]}"; do
        step "FPS diversity at N=${n} -> diversity_N${n}_index.json"
        "${PY}" scripts/curate.py diversity --pseudo_dir "${PSEUDO_DIR}" --top_k "${n}" \
            --out_name "diversity_N${n}_index.json"
    done
fi

echo
echo "== done. verify with: ${PY} experiments/verify_indices.py --pseudo_dir ${PSEUDO_DIR} =="
