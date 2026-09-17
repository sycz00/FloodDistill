#!/usr/bin/env bash
# Run acquisition, training, evaluation or plotting from the repository root.
# Usage: bash pipeline.sh {fetch,train,eval,plot,all}
# DATA_ROOT, PSEUDO_DIR, GPU, SEEDS and N override the defaults.
# Plotting reads the saved evaluation caches.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STAGE="${1:-all}"
GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
PSEUDO_DIR="${PSEUDO_DIR:-./data/pseudo_gdacs}"
SEEDS="${SEEDS:-42 64 91 107}"
N="${N:-252}"
# The paper's teacher is the official ibm-nasa-geospatial Prithvi-EO-2.0 Sen1Floods11
# checkpoint - "hf" downloads it. Verified: it scores S11 test mIoU 0.8946 / flood IoU
# 0.8167, matching runs/teacher_eval_results.json exactly, so there is
# nothing to fine-tune. Point TEACHER at a local .pt only if you want your own teacher.
TEACHER="${TEACHER:-hf}"
PY="${PY:-python}"

banner() { printf '\n========================================\n%s\n========================================\n' "$1"; }
die() { echo "ERROR: $*" >&2; exit 1; }

stage_fetch() {
    banner "FETCH"
    DATA_ROOT="$DATA_ROOT" bash datasets/fetch_all.sh
}

stage_train() {
    banner "TRAIN"
    # 1. Teacher. Everything downstream needs it: it writes the pseudo-labels and it is the
    # live KD target for the distillation arms.
    if [ "$TEACHER" = "hf" ]; then
        echo "[have] teacher: official Prithvi Sen1Floods11 checkpoint (downloaded on first use)"
    elif [ -f "$TEACHER" ]; then
        echo "[have] teacher $TEACHER"
    else
        echo "[run ] fine-tuning Prithvi teacher on Sen1Floods11"
        $PY scripts/train.py finetune --official_ckpt hf --target teacher \
            --data_root "$DATA_ROOT/sen1floods11" --gpu_ids "$GPU" \
            --out_dir "$(dirname "$TEACHER")" || die "teacher fine-tune failed"
    fi

    # 2. Ground-truth arm: the Sen1Floods11 half of the matched-budget comparison.
    # N=252 is the whole training split, so no subsampling flag is passed.
    for seed in $SEEDS; do
        out="runs/pipeline/gt_ofa_N${N}_s${seed}"
        [ -f "$out/best.pt" ] && { echo "[have] $out"; continue; }
        echo "[run ] $out"
        $PY scripts/train.py ofa --official_ckpt "$TEACHER" \
            --student_arch efficientvit_b0 --data_root "$DATA_ROOT/sen1floods11" \
            --alpha 0.5 --temperature 4.0 --ofa_alpha 0.3 --gamma 1.2 --grad_clip 5.0 \
            --lr 5e-5 --warmup_epochs 0 --epochs 50 --amp_dtype bf16 --no_amp --deterministic \
            --gpu_ids "$GPU" --seed "$seed" --out_dir "$out" || die "$out failed"
    done

    # 3. Pseudo-label arm. Needs the GDACS pool, which needs Earth Engine, so this is the one
    # stage that can legitimately be absent on a fresh machine.
    idx="diversity_N${N}_index.json"
    if [ ! -f "$PSEUDO_DIR/$idx" ]; then
        echo
        echo "[skip] pseudo arm: $PSEUDO_DIR/$idx not found."
        echo "       Build the pool first (needs Earth Engine, see docs/EXTERNAL_SETUP.md):"
        echo "         EE_PROJECT=<id> TEACHER_CKPT=$TEACHER bash datasets/prepare.sh"
        echo "         $PY scripts/curate.py diversity --pseudo_dir $PSEUDO_DIR --top_k $N --out_name $idx"
        return 0
    fi
    # The index must hold exactly N: PseudoLabeledDataset draws n_samples uniformly at random
    # from it and never re-reads the ranking, so a bigger index silently degrades to a random
    # draw and the curation factor disappears without an error.
    have=$($PY -c "import json;print(len(json.load(open('$PSEUDO_DIR/$idx'))['files']))")
    [ "$have" = "$N" ] || die "$idx holds $have entries, expected $N -- rebuild at --top_k $N"
    for seed in $SEEDS; do
        out="runs/pipeline/pseudo_ofa_N${N}_s${seed}"
        [ -f "$out/best.pt" ] && { echo "[have] $out"; continue; }
        echo "[run ] $out"
        $PY scripts/train.py ofa --official_ckpt "$TEACHER" \
            --student_arch efficientvit_b0 --data_root "$DATA_ROOT/sen1floods11" \
            --pseudo_dir "$PSEUDO_DIR" --no_labeled --n_samples "$N" \
            --curated_pseudo --curated_index_file "$idx" \
            --pseudo_geom_aug --pseudo_mult_noise \
            --alpha 0.5 --temperature 4.0 --ofa_alpha 0.3 --gamma 1.2 --grad_clip 5.0 \
            --lr 5e-5 --warmup_epochs 0 --epochs 50 --amp_dtype bf16 --no_amp --deterministic \
            --gpu_ids "$GPU" --seed "$seed" --out_dir "$out" || die "$out failed"
    done
}

stage_eval() {
    banner "EVAL"
    [ -d runs/pipeline ] || { echo "[skip] nothing trained yet"; return 0; }
    # --sturm_normalize original is the protocol used throughout the paper (it is also
    # eval_ablations.py's default); the retired percentile protocol is not comparable.
    for reg in gt_ofa_N${N} pseudo_ofa_N${N}; do
        ls -d "runs/pipeline/${reg}"_s* >/dev/null 2>&1 || continue
        echo "-- $reg"
        $PY evaluation/eval_ablations.py --runs_base runs/pipeline --runs_reg "$reg" \
            --n_filter "$N" --sturm_normalize original --skip_s1s2w --gpu "$GPU"
    done
}

stage_plot() {
    banner "PLOT"
    bash figures/make_figures.sh
    echo
    echo "figures written to assets/sturm_orig/figures/"
    echo "each one has a *_values.csv next to it holding every plotted number"
}

case "$STAGE" in
    fetch) stage_fetch ;;
    train) stage_train ;;
    eval)  stage_eval ;;
    plot)  stage_plot ;;
    all)   stage_fetch; stage_train; stage_eval; stage_plot ;;
    *) die "unknown stage '$STAGE' (want: fetch | train | eval | plot | all)" ;;
esac
echo
echo "done: $STAGE"
