#!/usr/bin/env bash
# Train the five CE / Logit-KD / OFA cells on S11 or GDACS pseudo-labels.
# LABELS=s11 selects human labels; LABELS=pseudo selects the teacher-labelled pool.
# Both use the same seeds, optimizer settings and four training augmentations:
# horizontal/vertical flips, 90-degree rotations and per-band multiplicative noise.
# Spatial transforms are shared across each sample's images and targets.
#
# Build an FPS index at exactly each N. The loader samples uniformly from the
# selected index, so a larger index would change the subset-selection experiment.
#
# LABELS=s11 N_VALUES=252 bash experiments/nsweep_pseudo_vs_gt.sh
# LABELS=pseudo N_VALUES="50 100 150 200 252 2500" bash experiments/nsweep_pseudo_vs_gt.sh
set -u

GPU="${GPU:-0}"
SEEDS=(${SEEDS:-42 64 91 107})
ARMS=(${ARMS:-ofa ce dfkd ce_unet dfkd_unet})
N_VALUES=(${N_VALUES:-50 100 150 200 252})
LABELS="${LABELS:-pseudo}"          # pseudo = GDACS pool, s11 = Sen1Floods11 labels
PSEUDO_DIR="${PSEUDO_DIR:-./data/pseudo_gdacs}"
TEACHER_CKPT="${TEACHER_CKPT:-hf}"   # the public Prithvi checkpoint; a .pt path overrides it
DATA_ROOT="${DATA_ROOT:-$HOME/data}"   # holds sen1floods11/, STURM-Flood/, ... (docs/DATA_LOCATION.md)
if [ -d "${DATA_ROOT}/v1.1" ]; then
    echo "[FAIL] DATA_ROOT=${DATA_ROOT} is the Sen1Floods11 directory; set it to its parent."; exit 2
fi
S11_ROOT="${S11_ROOT:-${DATA_ROOT}/sen1floods11}"
case "${LABELS}" in
  pseudo) OUT_BASE="${OUT_BASE:-runs/nsweep}"; PREFIX="pseudo" ;;
  s11)    OUT_BASE="${OUT_BASE:-runs/ablations_sen1}"; PREFIX="sen" ;;
  *) echo "[FAIL] LABELS must be 'pseudo' or 's11', got '${LABELS}'"; exit 2 ;;
esac
PY="${PY:-python}"
STUDENT_ARCH="${STUDENT_ARCH:-efficientvit_b0}"   # ce/dfkd arms; Table S3 uses mobilenet_v3_small
# IDX_PREFIX=mapped_diversity addresses a regenerated pool (experiments/map_index_to_pool.py).

# Shared recipe of the GT and pseudo arms. experiments/curation_compare.sh uses AMP-fp16 and
# non-deterministic kernels instead, so do not compare those runs directly with these.
TRAIN_COMMON="--lr 5e-5 --warmup_epochs 0 --epochs ${EPOCHS:-50} --amp_dtype bf16 --no_amp --deterministic"
# AUG="" runs the no-augmentation control, isolating the augmentation factor.
AUG="${AUG---pseudo_geom_aug --pseudo_mult_noise}"
TAG="${TAG:-aug}"

# Where an index can live, checked in this order: the pool itself, then the indices
# shipped with the repo (data/, data/index_files_curations/ - built on our own pool,
# see data/index_files_curations/README.md).
find_index() {
    for d in "${PSEUDO_DIR}" data data/index_files_curations; do
        [ -f "${d}/$1" ] && { echo "${d}"; return 0; }
    done
    return 1
}

mkdir -p "${OUT_BASE}"
echo "== pseudo-vs-GT N-sweep | GPU ${GPU} | arms: ${ARMS[*]} | N: ${N_VALUES[*]} | seeds: ${SEEDS[*]} =="

for n in "${N_VALUES[@]}"; do
  if [ "${LABELS}" = pseudo ]; then
    IDX="${IDX_PREFIX:-diversity}_N${n}_index.json"
    IDX_DIR=$(find_index "${IDX}") || {
        echo "[FAIL] N=${n}: ${IDX} not found in ${PSEUDO_DIR}, data, or data/index_files_curations"
        echo "       build it: ${PY} scripts/curate.py diversity --pseudo_dir ${PSEUDO_DIR} --top_k ${n} --out_name ${IDX}"
        exit 1
    }
    # The index must contain exactly N files.
    NIDX=$("${PY}" -c "import json;print(len(json.load(open('${IDX_DIR}/${IDX}'))['files']))")
    if [ "${NIDX}" != "${n}" ]; then
        echo "[FAIL] N=${n}: ${IDX} (in ${IDX_DIR}) holds ${NIDX} files; rebuild at --top_k ${n}"; exit 1
    fi
  fi

    for arm in "${ARMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if [ "${LABELS}" = pseudo ]; then
                NAME="${PREFIX}_${arm}_${TAG}_N${n}_s${seed}"
            else
                NAME="${PREFIX}_${arm}_N${n}_s${seed}"
            fi
            RUN="${OUT_BASE}/${NAME}"
            if [ -f "${RUN}/best.pt" ]; then echo "[have] ${RUN}"; continue; fi
            echo "[run ] ${RUN}"
            mkdir -p "${RUN}"
            if [ "${LABELS}" = pseudo ]; then
                SOURCE=(--pseudo_dir "${PSEUDO_DIR}" --no_labeled --n_samples "${n}"
                        --curated_pseudo --curated_index_file "${IDX}" ${AUG})
            else
                SOURCE=(--n_labeled_samples "${n}")
            fi
            COMMON=(--data_root "${S11_ROOT}" "${SOURCE[@]}"
                    ${TRAIN_COMMON} --gpu_ids "${GPU}" --seed "${seed}"
                    --out_dir "${RUN}" --wandb_run_name "${NAME}")
            case "${arm}" in
              # mirrors sen_ce
              ce)        "${PY}" scripts/train.py finetune --student_arch "${STUDENT_ARCH}" \
                             --target student --task_loss ce "${COMMON[@]}" ;;
              # mirrors sen_ce_unet
              ce_unet)   "${PY}" scripts/train.py finetune --student_arch unet_resnet50 \
                             --target student --task_loss ce "${COMMON[@]}" ;;
              # mirrors sen_dfkd
              dfkd)      "${PY}" scripts/train.py distill --official_ckpt "${TEACHER_CKPT}" \
                             --student_arch "${STUDENT_ARCH}" \
                             --alpha 0.5 --temperature 4.0 --feat_alpha 0.0 \
                             --task_loss ce "${COMMON[@]}" ;;
              # mirrors sen_dfkd_unet
              dfkd_unet) "${PY}" scripts/train.py distill --official_ckpt "${TEACHER_CKPT}" \
                             --student_arch unet_resnet50 \
                             --alpha 0.5 --temperature 4.0 --feat_alpha 0.0 \
                             --task_loss ce "${COMMON[@]}" ;;
              # mirrors sen_ofa (EfficientViT only)
              ofa)       "${PY}" scripts/train.py ofa --official_ckpt "${TEACHER_CKPT}" \
                             --student_arch efficientvit_b0 \
                             --alpha 0.5 --temperature 4.0 --ofa_alpha 0.3 \
                             --gamma 1.2 --grad_clip 5.0 "${COMMON[@]}" ;;
              *) echo "[skip] unknown arm ${arm}"; continue ;;
            esac
            [ -f "${RUN}/best.pt" ] && echo "[ok  ] ${RUN}" || echo "[FAIL] ${RUN}"
        done
    done
done
echo "== done (GPU ${GPU}) =="
