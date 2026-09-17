#!/usr/bin/env bash
# Evaluate selector runs on S11, STURM and flood-only WorldFloods.
# Use STURM original normalization; HAZE=1 adds the haze sweep.
# Usage: GPU=0 HAZE=1 bash experiments/curation_eval.sh

set -u

GPU="${GPU:-0}"
N="${N:-2500}"
RUNS_BASE="${RUNS_BASE:-runs/curation}"
ARMS=(${ARMS:-r2core fps glister confidence})
PY="${PY:-python}"
HAZE_VALS=(${HAZE_VALS:-0.02 0.05 0.10 0.15 0.20})

for arm in "${ARMS[@]}"; do
    PAT="${arm}_ofa_hs_N"
    ls -d "${RUNS_BASE}/${PAT}${N}"_s* >/dev/null 2>&1 || { echo "[skip] no runs for ${arm}"; continue; }
    # Do not evaluate an arm while training is running. best.pt is written on every
    # validation improvement, so it exists from the first epoch. It does not show that
    # training has finished. A partial checkpoint can give plausible but wrong results.
    # Require the expected seed count and no live train.py writing into this arm.
    NRUNS=$(ls -d "${RUNS_BASE}/${PAT}${N}"_s*/ 2>/dev/null | wc -l)
    NSEEDS=$(echo "${SEEDS:-42 64 91 107}" | wc -w)
    if pgrep -af "scripts/train.py .*${PAT}${N}" >/dev/null 2>&1; then
        echo "[skip] ${arm}: training still running — refusing to evaluate mid-flight"; continue
    fi
    if [ "${NRUNS}" -lt "${NSEEDS}" ]; then
        echo "[warn] ${arm}: only ${NRUNS}/${NSEEDS} seed dirs present; evaluating those"
    fi
    echo "== eval ${arm} (clean) =="
    "${PY}" evaluation/eval_ablations.py \
        --runs_base "${RUNS_BASE}" --runs_reg "${PAT}" --n_filter "${N}" \
        --sturm_normalize original --gpu "${GPU}" \
        && echo "[ok] ${arm} clean" || echo "[FAIL] ${arm} clean"

    if [ "${HAZE:-0}" = "1" ]; then
        for h in "${HAZE_VALS[@]}"; do
            echo "== eval ${arm} haze h=${h} =="
            "${PY}" evaluation/eval_ablations.py \
                --runs_base "${RUNS_BASE}" --runs_reg "${PAT}" --n_filter "${N}" \
                --sturm_normalize original --only_sturm --gpu "${GPU}" \
                --sturm_haze "${h}" --out_suffix "_HAZE_h${h}" \
                && echo "[ok] ${arm} h${h}" || echo "[FAIL] ${arm} h${h}"
        done
    fi
done
echo "== eval complete =="
echo "Collect with: ${PY} evaluation/curation_collect.py --runs_base ${RUNS_BASE} --n ${N}"
