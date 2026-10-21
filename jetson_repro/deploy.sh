#!/usr/bin/env bash
# Full on-device pipeline for one QAT export.
#
#   bash deploy.sh [RUN_DIR] [STEPS]
#
#   RUN_DIR  folder with model_qdq_trt_no_out.onnx, model_float.onnx,
#            model_float_trt_base.onnx   (default: runs/deploy_qat/docker_qat)
#   STEPS    comma list out of fix,engine,bench,eval,robust,figs
#            (default: fix,engine,bench,eval,figs)
#
# Env overrides: PYTHON, TRTEXEC, DATA (root holding the three datasets),
#                QUICK=N (only N batches per dataset, for a smoke test)
set -euo pipefail
cd "$(dirname "$0")"

RUN=${1:-runs/deploy_qat/docker_qat}
RUN=${RUN%/}
STEPS=${2:-fix,engine,bench,eval,figs}
PYTHON=${PYTHON:-$HOME/miniforge3/envs/flood/bin/python}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
DATA=${DATA:-$HOME/data}
QUICK=${QUICK:-0}

# TensorRT python bindings come from JetPack, not from conda
case ":${PYTHONPATH:-}:" in
    *:/usr/lib/python3.8/dist-packages:*) ;;
    *) export PYTHONPATH=/usr/lib/python3.8/dist-packages${PYTHONPATH:+:$PYTHONPATH} ;;
esac

NAME=$(basename "$RUN")
OUT=results/$NAME
mkdir -p "$OUT"
DATA_ARGS="--s11_root $DATA/sen1floods11 --sturm_root $DATA/STURM-Flood/Dataset --wf_root $DATA/WorldFloodsv2"

has() { [[ ",$STEPS," == *",$1,"* ]]; }

echo "run=$RUN  steps=$STEPS  quick=$QUICK  out=$OUT"

# 1. prepare the QDQ graph for TRT 8.4 (cubic->linear resize, frozen INT64 pad chain,
#    batch fixed to 1). Writes model_qdq_trt_no_out_static.onnx next to the input.
#    Use the _no_out export: with Conv output QDQ the flood channel collapses in TRT.
if has fix; then
    $PYTHON jetson/fix_qdq_onnx.py "$RUN/model_qdq_trt_no_out.onnx"
fi

# 2. INT8 engine. --fp16 lets TRT fall back to fp16 for layers without QDQ.
if has engine; then
    $TRTEXEC --onnx="$RUN/model_qdq_trt_no_out_static.onnx" \
        --int8 --fp16 \
        --saveEngine="$RUN/model_int8_qat.engine" \
        --workspace=2048 2>&1 | tee "$OUT/trtexec_int8_build.log"
fi

# 3. latency / power. Builds FP32 and FP16 engines from model_float_trt_base.onnx and times
#    all three with trtexec. --fp16_mm2_fp32 keeps the LiteMLA attention MatMuls in FP32,
#    otherwise the FP16 engine overflows. Latency only - see README about FP32/FP16 accuracy.
if has bench; then
    rm -f "$RUN/model_float_fp16.onnx" "$RUN/model_fp16.engine"
    $PYTHON jetson/benchmark_jetson.py --model_dir "$RUN/" \
        --img_size 512 --channels 6 --batch_size 1 \
        --iterations 200 --warmup 20 \
        --measure_power --rebuild --fp16_mm2_fp32 \
        --out "$OUT/benchmark_results.json" 2>&1 | tee "$OUT/benchmark.log"
fi

# eval_jetson.py picks up every *.engine in RUN. Keep only the INT8 one in view so the
# accuracy tables match the ones we report (FP32/FP16 engines stay in $OUT/engines).
park_float_engines() {
    mkdir -p "$OUT/engines"
    for e in model_fp32.engine model_fp16.engine; do
        [ -f "$RUN/$e" ] && mv "$RUN/$e" "$OUT/engines/"
    done
    return 0
}

# 4. report accuracy on S11 test and external surface water, ORT float + ORT INT8 + TRT INT8
if has eval; then
    park_float_engines
    $PYTHON jetson/eval_jetson.py --model_dir "$RUN/" \
        --engine "$RUN/model_int8_qat.engine" \
        --sturm_normalize original --datasets s11,sturm,wf \
        --s11_split test --protocol surface_water \
        --quick "$QUICK" $DATA_ARGS \
        --json_out "$OUT/eval.json" 2>&1 | tee "$OUT/eval.log"
fi

# 4b. same plus gain/haze sweep on the two OOD sets (slow, several hours on the NX)
if has robust; then
    park_float_engines
    $PYTHON jetson/eval_jetson.py --model_dir "$RUN/" \
        --engine "$RUN/model_int8_qat.engine" \
        --sturm_normalize original --datasets s11,sturm,wf \
        --s11_split test --protocol surface_water \
        --robustness --robust_datasets sturm,wf \
        --gains 0.80,1.00,1.20,1.30 --hazes 0.00,0.05,0.10,0.20 \
        --quick "$QUICK" $DATA_ARGS \
        --json_out "$OUT/eval_robustness.json" 2>&1 | tee "$OUT/eval_robustness.log"
fi

# 5. figures for the scenes fixed in manifests/
if has figs; then
    park_float_engines
    $PYTHON jetson/plot_int8_vs_float.py \
        --picks_from manifests/qual_samples_manifest.json \
        --model_dir "$RUN/" --engine "$RUN/model_int8_qat.engine" \
        --eval_img_size 512 --no_agreement --paper_clean --rows_per_figure 4 \
        $DATA_ARGS --out_dir "$OUT/figures"

    for m in haze020:--haze=0.20 haze010:--haze=0.10 gain130:--gain=1.30; do
        tag=${m%%:*}; flag=${m#*:}
        $PYTHON jetson/qualitative_jetson.py --model_dir "$RUN/" \
            --engine "$RUN/model_int8_qat.engine" \
            --manifest "manifests/qual_robust_manifest_$tag.json" \
            "$flag" --img_size 512 --sturm_normalize original \
            $DATA_ARGS --out "$OUT/figures/qual_robust_$tag.png" --dump_npz
    done
fi

echo "done -> $OUT"
