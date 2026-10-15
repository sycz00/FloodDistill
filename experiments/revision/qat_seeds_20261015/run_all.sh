#!/usr/bin/env bash
# Four-seed W8A8 QAT (Jetson integer recipe), CPU only, torch 2 threads (the 14 Oct setting; reproduces the committed
# seed-42 B0 checkpoint byte-for-byte). Outputs: <model>_s<seed>/summary.json; collect.py -> qat_seeds_summary.csv
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
mkdir -p logs
k=0
for m in R4 B0; do for s in 42 64 91 107; do
  [ -f ${m}_s$s/summary.json ] || OMP_NUM_THREADS=2 nice -n 5 taskset -c $((2*k)),$((2*k+1)) $PY run_qat_seeds.py --model $m --seed $s > logs/${m}_s$s.log 2>&1 &
  k=$((k+1))
done; done
wait
$PY collect.py
echo "DONE $(date -Is)"
