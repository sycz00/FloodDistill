#!/usr/bin/env bash
# Draw the quantitative paper figures from your finished runs. Nothing under runs/ is shipped,
# so run all experiments and evaluations first (figures/README.md).
# Outputs and value CSVs go to assets/sturm_orig/figures (S5: assets/).
# Qualitative figures (Fig. 3, S2) need checkpoints and data; see figures/README.md.
# Usage: bash figures/make_figures.sh
set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"

[ -d runs/perscene ] || { echo "runs/perscene missing — run experiments/perscene_sweeps.sh first"; exit 1; }
ls runs/nsweep/*/eval_results.json >/dev/null 2>&1 || {
    echo "runs/nsweep/*/eval_results.json missing — run evaluation/eval_ablations.py on the N-sweep first"; exit 1; }
echo "== paper figures | $(ls runs/perscene/*.json 2>/dev/null | wc -l) per-scene cells =="

run() { echo; echo "-- $*"; "${PY}" "$@" || echo "[FAIL] $*"; }

run figures/fig_ladder_vs_n.py                                          # Fig. 2
run figures/fig_e3_label_efficiency.py --dataset sturm --arch evit      # Fig. 4
run figures/fig_e3_label_efficiency.py --dataset wf --arch evit         # Fig. 5
run figures/e5_curation_decomp.py                                       # Fig. S1
run figures/fig_cka.py                                                  # Fig. S3
run figures/fig_e3_label_efficiency.py --dataset sturm --arch unet      # Fig. S4
run analysis/plot_grad_conflict_n.py                                    # Fig. S5

echo
echo "== done -> assets/sturm_orig/figures =="
