#!/usr/bin/env bash
# Regenerate the matched-budget robustness test of supplement Sec. S3
# (paper_source_data/Supp_InText_Statistics/S3_harmonisation/B2_robustness_holm.csv):
#   1. train the three N=252 pairs (EfficientViT CE and OFA, UNet Logit-KD) on S11 and pseudo labels
#   2. per-scene STURM cells for clean, gain 0.80/1.20/1.30 and haze 0.05/0.10/0.20
#   3. paired worst-case-drop tests, Holm over the block of six
# 24 trainings and 168 evaluations. Existing best.pt files and cells are reused.
#
# GPU=0 SEEDS="42 64 91 107" bash experiments/robustness_b2.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export GPU="${GPU:-0}"
export SEEDS="${SEEDS:-42 64 91 107}"
PY="${PY:-python}"
ROOT="${PERSCENE_ROOT:-runs}"
ARMS="ce ofa dfkd_unet"
FAMS="ncgt_evit_ce_N252,ncgt_evit_ofa_N252,ncgt_unet_lkd_N252,evit_ce_ps_N252,evit_ofa_ps_N252,unet_lkd_ps_N252"
CONDS="clean,g0.80,g1.20,g1.30,h0.05,h0.10,h0.20"

echo "=== [1/3] train the N=252 pairs (seeds: ${SEEDS}) ==="
ARMS="${ARMS}" LABELS=s11    N_VALUES=252 bash experiments/nsweep_pseudo_vs_gt.sh
ARMS="${ARMS}" LABELS=pseudo N_VALUES=252 bash experiments/nsweep_pseudo_vs_gt.sh

missing=0
for s in ${SEEDS}; do
    for r in runs/ablations_sen1/sen_{ce,ofa,dfkd_unet}_N252_s${s} \
             runs/nsweep/pseudo_{ce,ofa,dfkd_unet}_aug_N252_s${s}; do
        [ -f "${r}/best.pt" ] || { echo "[miss] ${r}/best.pt"; missing=1; }
    done
done
[ "${missing}" = 0 ] || { echo "[FAIL] training incomplete; see the log above"; exit 1; }

echo "=== [2/3] per-scene STURM cells ==="
mkdir -p "${ROOT}/perscene"
"${PY}" evaluation/perscene_eval.py --gpu "${GPU}" --datasets sturm --families "${FAMS}" \
    --seeds "${SEEDS// /,}" --conds "${CONDS}" --out_dir "${ROOT}/perscene"

echo "=== [3/3] paired tests ==="
"${PY}" evaluation/paired_group_tests.py --datasets sturm --cache "${ROOT}/perscene" \
    --out "${ROOT}/paired_group_tests_sturm.csv" --robustness_out "${ROOT}/B2_robustness_holm.csv"
echo "compare with paper_source_data/Supp_InText_Statistics/S3_harmonisation/B2_robustness_holm.csv"
