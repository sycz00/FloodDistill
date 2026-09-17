#!/usr/bin/env bash
# B1 geometry control: centre-crop S11 training chips from 512 to 264 pixels
# before the usual 224 resize. This matches the GDACS 2.64 km ground extent.
# Only --train_center_crop differs from the uncropped N=252 recipe.
# The frozen teacher receives the same cropped tensor as the student.
# Validation and test geometry stay unchanged.
# Usage: GPU=0 ARMS="ofa ce dfkd" bash experiments/b1_crop_train.sh

set -u

GPU="${GPU:-0}"
SEEDS=(${SEEDS:-42 64 91 107})
ARMS=(${ARMS:-ofa ce dfkd})
N="${N:-252}"
CROP="${CROP:-264}"
TEACHER_CKPT="${TEACHER_CKPT:-hf}"   # the public Prithvi checkpoint; a .pt path overrides it
DATA_ROOT="${DATA_ROOT:-$HOME/data}"   # holds sen1floods11/, STURM-Flood/, ... (docs/DATA_LOCATION.md)
if [ -d "${DATA_ROOT}/v1.1" ]; then
    echo "[FAIL] DATA_ROOT=${DATA_ROOT} is the Sen1Floods11 directory; set it to its parent."; exit 2
fi
S11_ROOT="${S11_ROOT:-${DATA_ROOT}/sen1floods11}"
OUT_BASE="${OUT_BASE:-runs/controls_geometry}"
PY="${PY:-python}"

# Identical to the GT arms' recipe (TRAIN_COMMON in nsweep_pseudo_vs_gt.sh).
TRAIN_COMMON="--lr 5e-5 --warmup_epochs 0 --epochs ${EPOCHS:-50} --amp_dtype bf16 --no_amp --deterministic"

# Stop if CUDA is not available (train.py would otherwise run on the CPU).
if ! "${PY}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[FAIL] torch.cuda.is_available() is False -- refusing to train on CPU."; exit 1
fi
echo "[guard] CUDA ok: $("${PY}" -c "import torch;print(torch.cuda.device_count())" 2>/dev/null) device(s)"

# N=252 is the full hand-labelled training split.
SPLIT="${S11_ROOT}/v1.1/splits/flood_handlabeled/flood_train_data.txt"
if [ -f "${SPLIT}" ]; then
    NSPLIT=$(sort -u "${SPLIT}" | grep -c .)
    [ "${NSPLIT}" = "252" ] || { echo "[FAIL] ${SPLIT} holds ${NSPLIT}, expected 252"; exit 1; }
    echo "[guard] split check ok: ${NSPLIT} unique train chips"
fi

mkdir -p "${OUT_BASE}"
echo "== B1 geometry control | crop ${CROP} px | N=${N} | GPU ${GPU} | arms: ${ARMS[*]} | seeds: ${SEEDS[*]} =="

for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN="${OUT_BASE}/sen_${arm}_crop${CROP}_N${N}_s${seed}"
        if [ -f "${RUN}/best.pt" ]; then echo "[have] ${RUN}"; continue; fi
        echo "[run ] ${RUN}"
        mkdir -p "${RUN}"
        COMMON=(--data_root "${S11_ROOT}" --n_labeled_samples "${N}"
                --train_center_crop "${CROP}"
                ${TRAIN_COMMON} --gpu_ids "${GPU}" --seed "${seed}"
                --out_dir "${RUN}" --wandb_run_name "sen_${arm}_crop${CROP}_N${N}_s${seed}")
        case "${arm}" in
          ce)   "${PY}" scripts/train.py finetune --student_arch efficientvit_b0 \
                    --target student --task_loss ce "${COMMON[@]}" ;;
          dfkd) "${PY}" scripts/train.py distill --official_ckpt "${TEACHER_CKPT}" \
                    --student_arch efficientvit_b0 \
                    --alpha 0.5 --temperature 4.0 --feat_alpha 0.0 \
                    --task_loss ce "${COMMON[@]}" ;;
          ofa)  "${PY}" scripts/train.py ofa --official_ckpt "${TEACHER_CKPT}" \
                    --student_arch efficientvit_b0 \
                    --alpha 0.5 --temperature 4.0 --ofa_alpha 0.3 \
                    --gamma 1.2 --grad_clip 5.0 "${COMMON[@]}" ;;
          *) echo "[skip] unknown arm ${arm}"; continue ;;
        esac
        [ -f "${RUN}/best.pt" ] && echo "[ok  ] ${RUN}" || echo "[FAIL] ${RUN}"
    done
done
echo "== done (GPU ${GPU}) =="
