#!/usr/bin/env bash
# Evaluate the matched-budget sweep on S11, STURM and flood-only WorldFloods.
# SIDE=pseudo reads runs/nsweep; SIDE=gt reads runs/ablations_sen1.
# STURM uses original normalization. Pass --arch unet_resnet50 for UNet run
# names that the evaluator cannot identify automatically.
# Usage: GPU=0 SIDE=pseudo ARMS="ofa dfkd ce" bash experiments/nsweep_eval.sh

set -u

GPU="${GPU:-0}"
SIDE="${SIDE:-pseudo}"
N_VALUES=(${N_VALUES:-50 100 150 200 252})
PY="${PY:-python}"

if [ "${SIDE}" = "pseudo" ]; then
    RUNS_BASE="${RUNS_BASE:-runs/nsweep}"; PREFIX="pseudo_"; SUFFIX="${SUFFIX:-_aug_N}"
    ARMS=(${ARMS:-ofa dfkd ce ce_unet dfkd_unet})
else
    RUNS_BASE="${RUNS_BASE:-runs/ablations_sen1}"; PREFIX="sen_"; SUFFIX="${SUFFIX:-_N}"
    ARMS=(${ARMS:-ce dfkd ofa ce_unet dfkd_unet})
fi

for arm in "${ARMS[@]}"; do
    case "${arm}" in
        *unet*) ARCH=(--arch unet_resnet50) ;;
        # EVAL_ARCH mirrors STUDENT_ARCH on the training side: detect_arch() keys off the run
        # name, so any arm not named after its architecture must be told what it is.
        *)      ARCH=(${EVAL_ARCH:+--arch ${EVAL_ARCH}}) ;;
    esac
    REG="${PREFIX}${arm}${SUFFIX}"
    for n in "${N_VALUES[@]}"; do
        ls -d "${RUNS_BASE}/${REG}${n}"_s* >/dev/null 2>&1 || { echo "[skip] ${REG}${n}"; continue; }
        if pgrep -af "scripts/train.py .*${REG}${n}_s" >/dev/null 2>&1; then
            echo "[skip] ${REG}${n}: training still live — refusing to evaluate mid-flight"; continue
        fi
        echo "== eval ${REG}${n} =="
        "${PY}" evaluation/eval_ablations.py \
            --runs_base "${RUNS_BASE}" --runs_reg "${REG}" --n_filter "${n}" \
            --sturm_normalize original --skip_s1s2w --gpu "${GPU}" "${ARCH[@]}" \
            && echo "[ok] ${REG}${n}" || echo "[FAIL] ${REG}${n}"
    done
done
echo "== eval complete (SIDE=${SIDE}, GPU=${GPU}) =="
