#!/usr/bin/env bash
# Train Hardswish and ReLU6 OFA students on the same N=2,500 GDACS subset.
# ReLU6 is set before training, so weights and BN statistics adapt to it.
# Both variants use the exact-N index, matched pseudo augmentation,
# --no_amp --deterministic, and seeds 42, 64, 91 and 107.
# Usage: bash experiments/repro_n2500.sh
#        VARIANTS=hs SEEDS=42 bash experiments/repro_n2500.sh

set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"
SEEDS=(${SEEDS:-42 64 91 107})
VARIANTS=(${VARIANTS:-hs relu6})
N="${N:-2500}"
IDX="${IDX:-diversity_N${N}_index.json}"
PSEUDO_DIR="${PSEUDO_DIR:-./data/pseudo_gdacs}"
TEACHER_CKPT="${TEACHER_CKPT:-hf}"   # the public Prithvi checkpoint; a .pt path overrides it
DATA_ROOT="${DATA_ROOT:-$HOME/data}"   # holds sen1floods11/, STURM-Flood/, ... (docs/DATA_LOCATION.md)
if [ -d "${DATA_ROOT}/v1.1" ]; then
    echo "[FAIL] DATA_ROOT=${DATA_ROOT} is the Sen1Floods11 directory; set it to its parent."; exit 2
fi
S11_ROOT="${S11_ROOT:-${DATA_ROOT}/sen1floods11}"
OUT_BASE="${OUT_BASE:-runs/repro}"

# Same as TRAIN_COMMON in experiments/nsweep_pseudo_vs_gt.sh.
TRAIN_COMMON="--lr 5e-5 --warmup_epochs 0 --epochs ${EPOCHS:-50} --amp_dtype bf16 --no_amp --deterministic"
AUG="--pseudo_geom_aug --pseudo_mult_noise"

# Keep "relu6" in the run name: evaluation/eval_ablations.py detect_arch() reads the
# architecture from it.
arch_for() { case "$1" in hs) echo efficientvit_b0 ;; relu6) echo efficientvit_b0_relu6 ;; *) echo "" ;; esac; }
# Same seed -> GPU assignment as the reference runs.
gpu_for()  { case "$1" in 42|64) echo 0 ;; *) echo 1 ;; esac; }

# Where an index can live, checked in this order: the pool itself, then the indices
# shipped with the repo (data/, data/index_files_curations/).
IDX_DIR=""
for d in "${PSEUDO_DIR}" data data/index_files_curations; do
    [ -f "${d}/${IDX}" ] && { IDX_DIR="${d}"; break; }
done
if [ -z "${IDX_DIR}" ]; then
    echo "[ABORT] ${IDX} not found in ${PSEUDO_DIR}, data, or data/index_files_curations"
    echo "        build it: ${PY} scripts/curate.py diversity --pseudo_dir ${PSEUDO_DIR} --top_k ${N} --out_name ${IDX}"
    exit 1
fi
# The index must contain exactly N files.
NIDX=$("${PY}" -c "import json;print(len(json.load(open('${IDX_DIR}/${IDX}'))['files']))") || exit 1
if [ "${NIDX}" != "${N}" ]; then
    echo "[ABORT] ${IDX} (in ${IDX_DIR}) holds ${NIDX} files but N=${N}; rebuild with:"
    echo "        ${PY} scripts/curate.py diversity --pseudo_dir ${PSEUDO_DIR} --top_k ${N} --out_name ${IDX}"
    exit 1
fi
# "hf": public Prithvi checkpoint, downloaded on first use.
[ "${TEACHER_CKPT}" = hf ] || [ -f "${TEACHER_CKPT}" ] || {
    echo "[ABORT] teacher ${TEACHER_CKPT} missing (use TEACHER_CKPT=hf for the public one)"
    exit 1
}
echo "== OFA N=${N} | index ${IDX} (${NIDX} files) | variants: ${VARIANTS[*]} | seeds: ${SEEDS[*]} =="

mkdir -p "${OUT_BASE}"
for variant in "${VARIANTS[@]}"; do
    ARCH="$(arch_for "${variant}")"
    [ -n "${ARCH}" ] || { echo "[skip] unknown variant ${variant}"; continue; }
    for seed in "${SEEDS[@]}"; do
        GPU="$(gpu_for "${seed}")"
        RUN="${OUT_BASE}/repro_ofa_${variant}_N${N}_s${seed}"
        if [ -f "${RUN}/best.pt" ]; then echo "[have] ${RUN}"; continue; fi
        echo "[run ] ${RUN}  (arch=${ARCH}, gpu=${GPU})"
        mkdir -p "${RUN}"
        "${PY}" scripts/train.py ofa \
            --official_ckpt "${TEACHER_CKPT}" \
            --student_arch "${ARCH}" \
            --alpha 0.5 --temperature 4.0 \
            --ofa_alpha 0.3 \
            --gamma 1.2 \
            --grad_clip 5.0 \
            --data_root "${S11_ROOT}" \
            --pseudo_dir "${PSEUDO_DIR}" \
            --no_labeled \
            --n_samples "${N}" \
            --curated_pseudo \
            --curated_index_file "${IDX}" \
            ${AUG} \
            ${TRAIN_COMMON} \
            --gpu_ids "${GPU}" \
            --seed "${seed}" \
            --out_dir "${RUN}" \
            --wandb_run_name "repro_ofa_${variant}_N${N}_s${seed}" \
            && echo "[ok  ] ${RUN}" || echo "[FAIL] ${RUN}"
    done
done

echo
echo "== done. next:"
echo "   ${PY} evaluation/eval_ablations.py --runs_base ${OUT_BASE} --runs_reg repro_ofa_hs_N \\"
echo "        --n_filter ${N} --sturm_normalize original --gpu 0"
echo "   ${PY} evaluation/nsweep_collect.py"
