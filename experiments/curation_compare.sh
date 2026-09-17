#!/usr/bin/env bash
# Train the N=2,500 selector comparison with seeds 42, 64, 91 and 107.
# Filtered arms share the same flood-fraction/confidence pre-filter; *_raw arms
# select from the full pool. GLISTER variants change its regularizer or outer
# validation objective. Keep these settings with each index.
# Usage: GPU=0 ARMS="fps confidence" bash experiments/curation_compare.sh

set -u

GPU="${GPU:-0}"
SEEDS=(${SEEDS:-42 64 91 107})
ARMS=(${ARMS:-r2core fps glister confidence})
N="${N:-2500}"
PSEUDO_DIR="${PSEUDO_DIR:-./data/pseudo_gdacs}"
TEACHER_CKPT="${TEACHER_CKPT:-hf}"   # the public Prithvi checkpoint; a .pt path overrides it
DATA_ROOT="${DATA_ROOT:-$HOME/data}"   # holds sen1floods11/, STURM-Flood/, ... (docs/DATA_LOCATION.md)
if [ -d "${DATA_ROOT}/v1.1" ]; then
    echo "[FAIL] DATA_ROOT=${DATA_ROOT} is the Sen1Floods11 directory; set it to its parent."; exit 2
fi
S11_ROOT="${S11_ROOT:-${DATA_ROOT}/sen1floods11}"
OUT_BASE="${OUT_BASE:-runs/curation}"
PY="${PY:-python}"
# RECIPE is empty for the original selector runs (AMP-fp16, non-deterministic). For the GT
# recipe set RECIPE="--lr 5e-5 --warmup_epochs 0 --epochs 50 --amp_dtype bf16 --no_amp --deterministic".
# TAG separates the output directories.
RECIPE="${RECIPE:-}"
TAG="${TAG:-}"

index_for() {
    case "$1" in
        r2core)      echo "r2core_index.json" ;;
        fps)         echo "diversity_N${N}_index.json" ;;
        glister)     echo "glister_index.json" ;;
        confidence)  echo "confidence_filtered_index.json" ;;
        r2core_flonly) echo "r2core_fl_only_index.json" ;;
        glister_r2)   echo "glister_r2_index.json" ;;
        glister_flive) echo "glister_flive_index.json" ;;
        glister_ood)  echo "glister_ood_index.json" ;;
        r2core_raw)     echo "r2core_raw_index.json" ;;
        fps_raw)        echo "diversity_raw_index.json" ;;
        glister_raw)    echo "glister_raw_index.json" ;;
        confidence_raw) echo "confidence_raw_index.json" ;;
        random)         echo "random_filtered_index.json" ;;
        random_raw)     echo "random_raw_index.json" ;;
        *) echo "" ;;
    esac
}

mkdir -p "${OUT_BASE}"
echo "== curation comparison | GPU ${GPU} | N=${N} | arms: ${ARMS[*]} | seeds: ${SEEDS[*]} =="

for arm in "${ARMS[@]}"; do
    IDX="$(index_for "$arm")"
    if [ -z "${IDX}" ]; then echo "[skip] unknown arm ${arm}"; continue; fi
    if [ ! -f "${PSEUDO_DIR}/${IDX}" ]; then
        echo "[skip] ${arm}: ${PSEUDO_DIR}/${IDX} not built yet"; continue
    fi
    # The index must contain exactly N files; otherwise the loader draws a random subset.
    NIDX=$("${PY}" -c "import json,sys; print(len(json.load(open('${PSEUDO_DIR}/${IDX}'))['files']))")
    if [ "${NIDX}" != "${N}" ]; then
        echo "[FAIL] ${arm}: index has ${NIDX} files but N=${N}; rebuild it at --top_k ${N}"
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        RUN="${OUT_BASE}/${arm}${TAG}_ofa_hs_N${N}_s${seed}"
        if [ -f "${RUN}/best.pt" ]; then echo "[have] ${RUN}"; continue; fi
        echo "[run ] ${RUN}"
        mkdir -p "${RUN}"
        "${PY}" scripts/train.py ofa \
            --official_ckpt "${TEACHER_CKPT}" \
            --student_arch efficientvit_b0 \
            --data_root "${S11_ROOT}" \
            --pseudo_dir "${PSEUDO_DIR}" \
            --no_labeled \
            --n_samples "${N}" \
            --curated_pseudo \
            --curated_index_file "${IDX}" \
            --alpha 0.5 --temperature 4.0 \
            --ofa_alpha 0.3 \
            --gamma 1.2 \
            --grad_clip 5.0 \
            ${RECIPE} \
            --gpu_ids "${GPU}" \
            --seed "${seed}" \
            --out_dir "${RUN}" \
            --wandb_run_name "${arm}${TAG}_ofa_hs_N${N}_s${seed}" \
            && echo "[ok  ] ${RUN}" || echo "[FAIL] ${RUN}"
    done
done
echo "== done (GPU ${GPU}) =="
