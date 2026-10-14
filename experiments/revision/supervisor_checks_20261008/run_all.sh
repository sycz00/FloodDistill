#!/usr/bin/env bash
# Supervisor checks (read-only; no project artifact is modified).
#   bash run_all.sh        fast set  (CPU + a few GPU inference passes on 89 S11-val images, ~8 min)
#   bash run_all.sh full   fast set + slower sweeps (per-layer sensitivity, mixed-precision, distilled-index pilot, QAT-checkpoint parity)
# Outputs: receipts/<script>.txt. GPU use is inference only; do not run while a training job occupies both devices.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
FAST="noise_floor weight_code_flips grad_norms fp_grad param_motion pc_explained grad_probe grad_bisect grad_clamp quant_headroom attn_grad_concentration int_domain_fakequant int_domain_ptq_eval prompt6_probes metric_sanity"
SLOW="int_domain_divergence int_domain_qat_eval a1_layer_sensitivity a1_mixed_precision distilled_index_pilot"
LIST="$FAST"; [ "${1:-}" = "full" ] && LIST="$FAST $SLOW"
mkdir -p "$HERE/receipts"
cd /tmp   # run outside the repo so project modules cannot be shadowed
for s in $LIST; do
  echo "== $s"
  if [ "$s" = "weight_code_flips" ]; then
    $PY "$HERE/$s.py" "$HERE/receipts/$s.json" 2>&1 | grep -v -E "Warning|warn|\[pseudo\]|detach" > "$HERE/receipts/$s.txt"
  elif [ "$s" = "noise_floor" ]; then
    $PY -I "$HERE/$s.py" 2>&1 > "$HERE/receipts/$s.txt"
  else
    $PY "$HERE/$s.py" 2>&1 | grep -v -E "Warning|warn|\[pseudo\]|detach" > "$HERE/receipts/$s.txt"
  fi
done
echo done
