#!/usr/bin/env bash
# Exploratory CPU-only probes (12-13 Oct 2026). Read-only on every input:
# capacity_v7 R4 checkpoints/spec/reference cache, S11 val/test, STURM, WorldFloods-v2 TEST
# (incl. its cloud band). Never reads WorldFloods-v2 train/val (sealed V9 test set).
# About 15 minutes on 16 CPU threads; no GPU (CUDA_VISIBLE_DEVICES is emptied).
# gpu_vram_check.py is NOT run here: it needs one idle GPU, selected by UUID, e.g.
#   PYTHONPATH=/home/schmalstieg/repos/_deps/efficientvit CUDA_VISIBLE_DEVICES=GPU-<uuid> $PY gpu_vram_check.py
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(mktemp -d)}"
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
CUDA_VISIBLE_DEVICES= $PY probe_cpu.py     > probe_cpu.log 2>&1      # per-image counts of decoding rules
CUDA_VISIBLE_DEVICES= $PY probe_hist.py    > probe_hist.log 2>&1     # margin x MNDWI x class histograms
CUDA_VISIBLE_DEVICES= $PY probe_failure.py > probe_failure.log 2>&1  # failure strata + edge-band re-decision
CUDA_VISIBLE_DEVICES= $PY probe_cloud.py   > probe_cloud.log 2>&1    # input cloud test + cloud-band re-decision
python3 -I probe_analysis.py       > /dev/null   # physics fusion rules        -> probe_report.txt
python3 -I probe_oprules.py        > /dev/null   # label-free thresholds        -> oprules_report.txt
python3 -I probe_fewshot.py        > /dev/null   # one-chip/click calibration   -> fewshot_report.txt
python3 -I probe_predictability.py > /dev/null   # tau* predictability, AP      -> predictability_report.txt
python3 -I failure_analysis.py     > /dev/null   # strata + edge refinement     -> failure_report.txt
