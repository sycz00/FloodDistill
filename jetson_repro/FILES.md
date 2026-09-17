# What is in here and where it came from

Everything below was taken from `~/flood_deploy` on the reference Jetson. The
starting point was `deploy_order.sh` and the run it was used on last
(`runs/quant/deploy_qat/docker_qat`, 2026-09-15). Anything that script did not
touch was left out.

## Needed

| file | from | used by / why |
|---|---|---|
| `deploy.sh` | `deploy_order.sh` | runs all steps; takes the run folder as argument |
| `jetson/fix_qdq_onnx.py` | `scripts/fix_qdq_onnx.py` | step 1, makes the QDQ ONNX loadable by TRT 8.4 |
| `jetson/benchmark_jetson.py` | `scripts/benchmark_jetson.py` | step 3, builds FP32/FP16 engines, trtexec timing, power |
| `jetson/fp16_mm2_layers.py` | `jetson/` | optional, regenerates the FP32-pinned layer list for FP16 |
| `jetson/eval_jetson.py` | `scripts/eval_jetson.py` | step 4, accuracy (ORT + TRT) and robustness sweep |
| `jetson/plot_int8_vs_float.py` | `figures/plot_models.py --jetson` | step 5, Float vs INT8 panels from the manifest |
| `jetson/qualitative_jetson.py` | `jetson/` | step 5, clean/perturbed panels + `.npz` dumps |
| `data/sen1floods11_nodep.py` | `data/` | S11 loader without terratorch |
| `data/ood_datasets.py` | `data/` | STURM-Flood and WorldFloods-v2 loaders, gain/haze |
| `data/channel_adapter.py` | `data/` | STURM 9 bands -> 6 |
| `utils/metrics.py` | `utils/` | confusion matrix / IoU |
| `figures/qual_rgb.py` | `figures/` | true-colour rendering shared with the workstation figures |
| `manifests/*.json` | `runs/qualitative_figures/` | scene picks written on the workstation |
| `runs/deploy_qat/docker_qat/model_qdq_trt_no_out.onnx` | QAT export | INT8 source graph |
| `runs/deploy_qat/docker_qat/model_float.onnx` | QAT export | ORT float reference in eval/figures |
| `runs/deploy_qat/docker_qat/model_float_trt_base.onnx` | QAT export | FP32/FP16 engines for the benchmark |
| `runs/deploy_qat/docker_qat/qat_metrics.json` | QAT export | workstation val/test mIoU for comparison |
| `env.yaml` | `conda env export -n flood` | python env, trimmed to what is imported |

The `jetson/` scripts are the copies from `jetson/jetson_workstation/` (same code as
`scripts/`, only comments/help text differ). Changes made here: default paths point to
`runs/deploy_qat/docker_qat`, and the "engine missing" hint in the benchmark prints the
trtexec command. `plot_int8_vs_float.py` is only the `--jetson` branch of
`figures/plot_models.py`; the rest of that file needs the teacher and PyTorch
checkpoints and cannot run on the device.

## Generated on the device, shipped from the reference run

These come from the run on the reference Xavier NX (JetPack 5.0.2, TensorRT 8.4.1) and are
kept as a record of it:

- `runs/deploy_qat/docker_qat/model_qdq_trt_no_out_static.onnx` (step 1)
- `runs/deploy_qat/docker_qat/model_int8_qat.engine` (step 2). An engine only loads on the
  GPU and TRT version it was built with; on any other setup, rebuild it with step `engine`.
- `results/docker_qat/trtexec_int8_build.log`: build log of the shipped engine. The
  accuracy log (`eval.log`, `eval.json`) and the figures are not shipped; step `eval` and
  step `figs` write them.

`deploy.sh` writes to the same paths, so a new run replaces these files. Copy them
somewhere else first if you want to keep them. `model_fp32.engine` and `model_fp16.engine`
(step 3) are not shipped.

## Left out on purpose

- `best_qat.pt`, `model_qat_stripped.pt`: PyTorch weights, no script on the device loads them
- `runs/quant/deploy_qat/qat_s42`, `runs/quant/{qat_s*,n_qat_*,swap,div_ofa_*}`: older exports
- `train.py`, `quantize.py`, `quantization/`, `cka_analysis.py`, other `data/*` loaders: workstation only
- `scripts/old_scripts`, `scripts/make_fp16_onnx.py` (fp16 ONNX breaks attention in TRT 8.4), `archive/`, `wandb/`
- the `*.log` / `jetson_int8_robustness*.json` files in the old root: outputs of earlier runs
