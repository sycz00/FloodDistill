#!/usr/bin/env bash
# Generate per-scene confusion cells for macro scores and paired comparisons.
# Matched-budget and selector arms use the full gain/haze grid; N-curves use
# clean conditions. Separate protocol controls are in perscene_controls.sh.
# Worst-case drop is taken per seed before averaging.

set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
PY=${PY:-python}
RUN="$PY evaluation/perscene_eval.py --gpu $GPU"
# Where the cells are written. runs/ by default; point PERSCENE_ROOT at a mounted volume in
# the container, where the repo copy is root-owned. The collectors read the same variable.
ROOT=${PERSCENE_ROOT:-runs}
OUT="$ROOT/perscene"

for ds in sturm wf; do
    echo "=== [$ds 1/3] matched budget N=252, both label sources, all conditions ==="
    # The block Table I is built from, and the only one the six-pair robustness test can use.
    $RUN --datasets "$ds" --families n252 --out_dir "$OUT"

    echo "=== [$ds 2/3] curation selectors at N=2,500, all conditions ==="
    $RUN --datasets "$ds" --families curation,matchedcore --out_dir "$OUT"

    echo "=== [$ds 3/3] N-curves, clean only ==="
    $RUN --datasets "$ds" --families ncurve_gt,ncurve_ps,unetcurve,relu6 \
         --conds clean --out_dir "$OUT"
done

echo "=== done: $(ls "$OUT"/*.json 2>/dev/null | wc -l) cells in $OUT ==="
