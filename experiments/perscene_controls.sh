#!/usr/bin/env bash
# Per-scene cells for the Sec. S3/S4 controls. Everything here writes OUTSIDE runs/perscene.
#
# Each block has a separate cache because the cell filename only records
# (dataset, family, seed, condition), not the evaluation protocol. A non-default class
# mapping or eval tensor size cannot share a directory with the published cells:
# a glob over runs/perscene would mix protocols and produce a different table without
# an error. perscene_eval.py refuses to write any of these into
# runs/perscene; evaluation/perscene_stats.py:load_cache() is the reader-side half of the guard.
#
# Which caches may be read together: runs/perscene, perscene_B1, perscene_C1, perscene_C4,
# perscene_T1, perscene_T1_ref are all the paper protocol (512 px, `paper` STURM class map)
# and are globbed together by perscene_stats.py. perscene_operational, perscene_sturmsize/* and
# perscene_indexthr are NOT -- each is its own protocol and its own dict.
#
# Blocks are independent; pass one as $1 to run just that one. ~2,200 cells in total.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
PY=${PY:-python}
RUN="$PY evaluation/perscene_eval.py --gpu $GPU"
BLOCK=${1:-all}
# Where the caches are written. runs/ by default; point PERSCENE_ROOT at a mounted volume in
# the container, where the repo copy is root-owned. The collectors read the same variable.
ROOT=${PERSCENE_ROOT:-runs}

# C1 -- the non-learned spectral-index reference (0 parameters, no checkpoint read).
if [ "$BLOCK" = all ] || [ "$BLOCK" = index ]; then
    echo "=== C1 · MNDWI/NDWI reference · clean ==="
    $RUN --datasets sturm,wf --families index --conds clean --out_dir $ROOT/perscene_C1
fi

# B1 -- the geometry-matched (264 px centre-crop) Sen1Floods11 arm, full perturbation grid.
if [ "$BLOCK" = all ] || [ "$BLOCK" = b1 ]; then
    echo "=== B1 · cropped Sen1Floods11 arms · all conditions ==="
    $RUN --datasets sturm,wf --families b1crop --out_dir $ROOT/perscene_B1
fi

# C4 -- the labeller control: the same 2,500 tiles labelled by MNDWI>0 instead of the teacher.
if [ "$BLOCK" = all ] || [ "$BLOCK" = c4 ]; then
    echo "=== C4 · MNDWI-labelled arms · all conditions + per-band ==="
    $RUN --datasets sturm,wf --families c4mndwi_evit_ce_N2500,c4mndwi_evit_ofa_N2500 \
         --out_dir $ROOT/perscene_C4
    $RUN --datasets sturm,wf --families c4mndwi_evit_ce_N2500,c4mndwi_evit_ofa_N2500 \
         --conds "$($PY -c 'from evaluation.perscene_eval import BAND_CONDS; print(",".join(BAND_CONDS))')" \
         --out_dir $ROOT/perscene_C4
fi

# T1 -- per-band radiometric gain. A NEW perturbation axis, so it gets its own cache even
# though its cell names could not collide with an existing file.
if [ "$BLOCK" = all ] || [ "$BLOCK" = t1 ]; then
    T1_CONDS=$($PY -c 'from evaluation.perscene_eval import BAND_CONDS; print(",".join(k for k in BAND_CONDS if k != "t1unit"))')
    echo "=== T1 · per-band gain · N=252 arms + the deployed N=2,500 arms ==="
    $RUN --datasets sturm,wf --families t1arms,t1arms_n2500 --conds "$T1_CONDS" \
         --out_dir $ROOT/perscene_T1
    echo "=== T1 · reference profiles (teacher + the two rules) over the SAME grid ==="
    $RUN --datasets sturm,wf --families t1ref \
         --conds "clean,g0.70,g0.80,g0.90,g1.20,g1.30,h0.05,h0.10,h0.20,$T1_CONDS" \
         --out_dir $ROOT/perscene_T1_ref
fi

# Operational protocol: permanent standing water is scored as background instead of dropped.
# A deployed product still needs that flood vs water distinction, and a normalised-difference
# water index can't provide it.
if [ "$BLOCK" = all ] || [ "$BLOCK" = operational ]; then
    echo "=== operational class map · clean ==="
    $RUN --datasets sturm,wf --conds clean --operational_protocol \
         --families teacher,index_mndwi_zero,evit_ofa_ps_N2500,evit_ofa_ps_N252,ncgt_evit_ofa_N252,b1crop_evit_ofa_N252 \
         --out_dir $ROOT/perscene_operational
fi

# STURM evaluation scale. 512 (published) is a 4x UPSAMPLE of the native 128 px chips; 128 is
# native, 224 is the training tensor and a non-integer 1.75x resampling.
if [ "$BLOCK" = all ] || [ "$BLOCK" = sturmsize ]; then
    for SZ in 128 224; do
        echo "=== STURM at ${SZ} px · clean ==="
        $RUN --datasets sturm --conds clean --sturm_img_size $SZ \
             --families t1arms,sturmnative --out_dir $ROOT/perscene_sturmsize/$SZ
    done
fi

# Threshold transferability -- the rule's single free parameter, swept on each benchmark
# instead of assumed to transfer.
if [ "$BLOCK" = all ] || [ "$BLOCK" = indexthr ]; then
    echo "=== index threshold sweep · clean ==="
    $RUN --datasets sturm,wf --families indexthr --conds clean --out_dir $ROOT/perscene_indexthr
fi

echo "=== done ==="
for d in $ROOT/perscene_C1 $ROOT/perscene_B1 $ROOT/perscene_C4 $ROOT/perscene_T1 \
         $ROOT/perscene_T1_ref $ROOT/perscene_operational $ROOT/perscene_indexthr \
         $ROOT/perscene_sturmsize/128 $ROOT/perscene_sturmsize/224; do
    [ -d "$d" ] || continue          # `&&` here would trip set -e on the last missing cache
    echo "  $(ls "$d"/*.json 2>/dev/null | wc -l) cells  $d"
done
