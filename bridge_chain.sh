#!/usr/bin/env bash
# Teacher-assistant control: Prithvi -> UNet ensemble -> EfficientViT.
# Relabel the selected GDACS pool, then distil from the live assistant ensemble.
# Set UNET_RUNS_ROOT to the assistant checkpoints before running.
# Usage: bash bridge_chain.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
SEEDS=(42 64 91 107)
UNET_SEEDS=(42 64 91 107)
DATA_ROOT="${DATA_ROOT:-$HOME/data}"   # holds sen1floods11/, STURM-Flood/, ... (docs/DATA_LOCATION.md)
if [ -d "${DATA_ROOT}/v1.1" ]; then
    echo "[FAIL] DATA_ROOT=${DATA_ROOT} is the Sen1Floods11 directory; set it to its parent."; exit 2
fi
S11_ROOT="${S11_ROOT:-${DATA_ROOT}/sen1floods11}"
UNET_RUNS_ROOT="${UNET_RUNS_ROOT:-./runs/nsweep}"
PSEUDO_SRC="${PSEUDO_SRC:-./data/pseudo_gdacs}"   # Prithvi-labeled GDACS pool
PSEUDO_INDEX="${PSEUDO_INDEX:-diversity_N2500_index.json}"  # index built at exactly --top_k 2500
PSEUDO_UNET="${PSEUDO_UNET:-./data/pseudo_gdacs_unet4}"  # UNet-relabeled pool (stage 1 output)
OUT_BASE="${OUT_BASE:-./runs/bridge}"
STUDENT_ARCH="${STUDENT_ARCH:-efficientvit_b0_relu6}"   # deployment-native; use efficientvit_b0 for the HS ablation

# evaluation/eval_ablations.py detect_arch() reads the student architecture from the run
# name; "unet4" is the teacher ensemble, so the name also contains the student activation.
ARCH_TAG="$(case "${STUDENT_ARCH}" in *_relu6) echo _relu6;; *_lrelu6) echo _lrelu6;; \
                                      *_silu) echo _silu;; *_tanh) echo _tanh;; *) echo "";; esac)"
RUN_NAME_BASE="bridge_ofa_unet4${ARCH_TAG}_N2500"

# Run-name template of the assistants (%s = seed), as nsweep and Docker write it:
#   pseudo_dfkd_unet_aug_N2500_s%s
UNET_RUN_TMPL="${UNET_RUN_TMPL:-pseudo_dfkd_unet_aug_N2500_s%s}"

UNET_CKPTS=()
for s in "${UNET_SEEDS[@]}"; do
    UNET_CKPTS+=("${UNET_RUNS_ROOT}/$(printf "${UNET_RUN_TMPL}" "${s}")/best.pt")
done

# The assistant checkpoints must exist (train them first).
missing=()
for c in "${UNET_CKPTS[@]}"; do [ -f "$c" ] || missing+=("$c"); done
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: ${#missing[@]}/${#UNET_CKPTS[@]} UNet assistant checkpoints are missing," >&2
    echo "       first: ${missing[0]}" >&2
    echo "  The repo ships eval JSONs, not checkpoints. Train the assistants:" >&2
    echo "    ARMS=dfkd_unet N_VALUES=2500 bash experiments/nsweep_pseudo_vs_gt.sh" >&2
    echo "  or point UNET_RUNS_ROOT at a tree that has them." >&2
    exit 1
fi
if [ ! -f "${PSEUDO_SRC}/${PSEUDO_INDEX}" ]; then
    echo "ERROR: ${PSEUDO_SRC}/${PSEUDO_INDEX} not found." >&2
    echo "  Set PSEUDO_SRC to the pool that holds the tiles (the shipped" >&2
    echo "  data/pseudo_gdacs has indices only) and PSEUDO_INDEX to its 2500 index." >&2
    exit 1
fi

#  Stage 1: relabel the 2,500-scene diversity pool with the UNet ensemble
# Images and pool membership stay identical; only the label source changes.
if [ -f "${PSEUDO_UNET}/${PSEUDO_INDEX}" ]; then
    echo "[skip] stage 1 — ${PSEUDO_UNET} already exists"
else
    python scripts/train.py pseudo relabel \
        --pseudo_dir "${PSEUDO_SRC}" \
        --out_dir    "${PSEUDO_UNET}" \
        --index_file "${PSEUDO_INDEX}" \
        --bridge_teacher_arch  unet_resnet50 \
        --bridge_teacher_ckpts "${UNET_CKPTS[@]}" \
        --bridge_ensemble_mode prob \
        --batch_size 32 --num_workers 8 --gpu_ids "${GPU}"
fi

#  Stage 2: EvIT-OFA from the UNet ensemble
# Identical HPs to experiments/repro_n2500.sh; only teacher + label source differ.
for seed in "${SEEDS[@]}"; do
    RUN="${OUT_BASE}/${RUN_NAME_BASE}_s${seed}"
    if [ -f "${RUN}/best.pt" ]; then
        echo "[skip] ${RUN}"
        continue
    fi
    python scripts/train.py ofa \
        --bridge_teacher_arch  unet_resnet50 \
        --bridge_teacher_ckpts "${UNET_CKPTS[@]}" \
        --bridge_ensemble_mode prob \
        --student_arch    "${STUDENT_ARCH}" \
        --data_root       "${S11_ROOT}" \
        --pseudo_dir      "${PSEUDO_UNET}" \
        --no_labeled \
        --n_samples       2500 \
        --curated_pseudo \
        --curated_index_file "${PSEUDO_INDEX}" \
        --alpha 0.5 --temperature 4.0 \
        --ofa_alpha 0.3 \
        --gamma 1.2 \
        --grad_clip 5.0 \
        --lr 5e-5 --warmup_epochs 0 --epochs 50 \
        --gpu_ids "${GPU}" \
        --seed "${seed}" \
        --out_dir "${RUN}" \
        --wandb_run_name "${RUN_NAME_BASE}_s${seed}"
done

echo ""
echo "Done. Evaluate with, e.g.:"
echo "  python scripts/train.py ood-eval --student_arch ${STUDENT_ARCH} \\"
echo "      --model_ckpt ${OUT_BASE}/${RUN_NAME_BASE}_s42/best.pt \\"
echo "      --ood_dataset sturm_s2 --ood_root ${DATA_ROOT}/STURM-Flood/Dataset --data_root ${S11_ROOT}"
