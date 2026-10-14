#!/usr/bin/env bash
# Early CPU readout of V9 checkpoints on EXPOSED sets only (S11 val/test, STURM, WF test).
# Never opens WorldFloods-v2 train/val (V9 sealed set). Read-only on the V9 run. ~15 min CPU.
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(mktemp -d)}"
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
CUDA_VISIBLE_DEVICES= $PY early_readout.py > early_readout.log 2>&1          # E0-E3 R4 arms (+ decode-time D)
PYTHONPATH=/home/schmalstieg/repos/_deps/efficientvit CUDA_VISIBLE_DEVICES= $PY labelswap_readout.py > labelswap_readout.log 2>&1   # B0 human vs teacher labels
python3 -I early_analysis.py > /dev/null                                   # -> report.txt, summary.csv
