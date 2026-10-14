#!/usr/bin/env bash
# V10 reference implementation checks (CPU, ~2 min): unit tests, then the S11-validation feasibility check.
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(mktemp -d)}" CUDA_VISIBLE_DEVICES= NO_ALBUMENTATIONS_UPDATE=1 WANDB_MODE=disabled
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
$PY test_v10_reference.py
$PY feasibility_s11val.py > feasibility.log 2>&1   # -> feasibility_report.txt
