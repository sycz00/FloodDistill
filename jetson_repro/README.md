# Flood segmentation on Jetson Xavier NX: INT8 deployment

This folder reproduces the on-device results of the QAT EfficientViT-B0 (ReLU6) flood
model: TensorRT INT8 engine, latency/power, accuracy on Sen1Floods11 / STURM-Flood /
WorldFloods-v2, and the qualitative figures. Copy the whole folder to a Jetson, set up
the environment below and run `deploy.sh`.

`FILES.md` lists what each file is and what was left out.

## Reference system

| | |
|---|---|
| board | Jetson Xavier NX Developer Kit (8 GB) |
| JetPack | 5.0.2 (L4T R35.1) |
| CUDA / cuDNN | 11.4 / 8.4.1 |
| TensorRT | 8.4.1.5 (`/usr/src/tensorrt/bin/trtexec`) |
| Python | 3.8 (conda env `flood`) |

Another JetPack version means a different TensorRT. The scripts use the TRT 8.4 Python
API (`execute_async_v2`, binding indices) and the ONNX fix is written for TRT 8.4's
limits, so stay on JetPack 5.0.x if you want the same numbers.

Check what you have:
```bash
cat /etc/nv_tegra_release
dpkg -l | grep -E "tensorrt |libcudnn8 |cuda-toolkit-11"
```

## 1. System packages and power mode

```bash
sudo apt update
sudo apt install -y build-essential gdal-bin libgdal-dev libopenblas-dev libjpeg-dev
```
GDAL is needed to compile rasterio, the CUDA toolkit (part of JetPack) to compile pycuda.

For timing, use max performance and fixed clocks:
```bash
sudo nvpmodel -q          # show current mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

The benchmark reads power from the INA3221 sensor, which is root-only by default:
```bash
sudo chmod a+r /sys/class/hwmon/hwmon4/curr[0-9]_input
sudo chmod a+r /sys/class/hwmon/hwmon4/in[0-9]_input
```
Repeat this after every reboot. Otherwise the benchmark still runs, but power is N/A.

## 2. Miniforge

Miniforge has aarch64 builds and uses conda-forge by default, as needed on
the Jetson (plain Anaconda has limited aarch64 support).

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
source ~/.bashrc
conda config --set auto_activate_base false
```
`deploy.sh` expects the env at `~/miniforge3/envs/flood`. If you install somewhere
else, set `PYTHON=/path/to/envs/flood/bin/python`.

## 3. Python environment

```bash
cd jetson_repro
conda env create -f env.yaml
conda activate flood
```
`env.yaml` pins everything that has a wheel for aarch64/py3.8, including NVIDIA's torch
build for JetPack 5.0.2 (~560 MB download). pycuda and rasterio have to be compiled:

```bash
conda activate flood
export CUDA_ROOT=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
pip install --no-cache-dir pycuda==2022.1

# rasterio against the JetPack GDAL (3.0.4). The rpath matters, see below.
LDSHARED="gcc -pthread -shared -Wl,--disable-new-dtags,-rpath,/usr/lib/aarch64-linux-gnu" \
    pip install --no-cache-dir rasterio==1.2.10
```
Allow a few minutes for both on the NX.

The `LDSHARED` line is needed because conda's python has `envs/flood/lib` in its search path.
Without the rpath, the system GDAL loads the env's `libsqlite3`, which pulls in conda's ICU 78,
and that needs a newer libstdc++ than Ubuntu 20.04 has:
`ImportError: ... libstdc++.so.6: version 'CXXABI_1.3.15' not found (required by .../libicui18n.so.78)`.
With the rpath, GDAL loads the system sqlite/ICU. `--no-cache-dir` prevents pip from
reusing a wheel built without it.

TensorRT is not installed through pip. JetPack puts its Python bindings in the system
dist-packages, so every shell that runs the scripts needs:

```bash
conda deactivate
export PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH
conda activate flood
```
(`deploy.sh` adds the path itself if it is missing, but you need it when calling the
Python scripts by hand.)

Quick check:
```bash
python -c "import torch, tensorrt, pycuda.autoinit, onnx, onnxruntime, rasterio; \
print(torch.__version__, torch.cuda.is_available(), tensorrt.__version__)"
# 1.13.0a0+d0d6b1f2.nv22.10 True 8.4.1.5
```

If you see `libgomp.so.1: cannot allocate memory in static TLS block`, something imported
scikit-learn/opencv/rasterio before torch. The scripts import torch first for that
reason. For your own snippets do the same, or
`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`.

Packages in `~/.local/lib/python3.8/site-packages` win over the ones in the env. If that
folder exists on the target, run with `export PYTHONNOUSERSITE=1` or you may get
different versions than in `env.yaml`.

## 4. Data

Default location is `~/data` (override with `DATA=/some/path bash deploy.sh ...`). Only the
parts below are read:

```
~/data/
├── sen1floods11/v1.1/
│   ├── splits/flood_handlabeled/flood_{train,valid,test}_data.txt
│   └── data/flood_events/HandLabeled/{S2Hand,LabelHand}/        ~1 GB
├── STURM-Flood/Dataset/
│   ├── Sentinel2/{S2,Floodmaps}/                                ~3.2 GB
│   └── Sentinel2_metadata.csv
└── WorldFloodsv2/test/{S2,gt,PERMANENTWATERJRC}/                ~3 GB
```

**Sen1Floods11** (public GCS bucket, needs `gsutil`):
```bash
mkdir -p ~/data/sen1floods11/v1.1/data/flood_events/HandLabeled
cd ~/data/sen1floods11/v1.1
gsutil -m cp -r gs://sen1floods11/v1.1/splits .
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand \
                gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand \
                data/flood_events/HandLabeled/
# the loader wants plain id lists
cd splits/flood_handlabeled
for s in train valid test; do
    cut -d, -f1 flood_${s}_data.csv | tr -d '\r' | sed 's/_S1Hand.tif//' > flood_${s}_data.txt
done
```

**STURM-Flood** (Zenodo, DOI 10.5281/zenodo.12748983, one 4 GB zip):
```bash
mkdir -p ~/data/STURM-Flood && cd ~/data/STURM-Flood
wget https://zenodo.org/records/12748983/files/Dataset.zip
unzip Dataset.zip          # -> Dataset/Sentinel1, Dataset/Sentinel2, *_metadata.csv
```

**WorldFloods-v2** (Hugging Face; the full repo is ~76 GB, only the test split is used):
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download --repo-type dataset isp-uv-es/WorldFloodsv2 \
    --include "test/S2/*" "test/gt/*" "test/PERMANENTWATERJRC/*" \
    --local-dir ~/data/WorldFloodsv2
```

## 5. Pipeline

### 5.0 From the QAT checkpoint to ONNX (workstation, not on the Jetson)

QAT and ONNX export run in the training repo (FloodDistill, `scripts/quantize.py qat`,
see `docs/03_quantization.md` there). The Jetson has neither the model code nor the
training data. For one run the export writes:

```
best_qat.pt                  best QAT epoch (PyTorch)
model_qat_stripped.pt        last epoch, quantizers removed
model_float.onnx             QAT weights, no quantizers, BN folded
model_float_trt_base.onnx    same, simplified + static shapes for TRT
model_qdq_trt_no_out.onnx    per-tensor QDQ on inputs, no Conv output QDQ  <- INT8 source
qat_metrics.json             workstation val/test mIoU
```
The last run is copied to `runs/deploy_qat/docker_qat/` (only the three ONNX files and the
metrics). To deploy another run, copy those same files into a new folder under `runs/`
and pass that folder to `deploy.sh`.

Use `model_qdq_trt_no_out.onnx`, not `model_qdq_trt.onnx`. With a per-tensor QDQ on the
Conv outputs, the scale is set by the background class and the flood channel gets crushed
in TRT (flood IoU goes to ~0).

### Running it
These commands run the scripts together. The individual steps follow below.
```bash
export PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH
conda activate flood
cd jetson_repro

QUICK=3 bash deploy.sh runs/deploy_qat/docker_qat fix,engine,eval   # smoke test, ~15 min
bash deploy.sh runs/deploy_qat/docker_qat                           # full run
nohup bash deploy.sh runs/deploy_qat/docker_qat robust > robust.out & # robustness sweep, long
```
Steps are `fix,engine,bench,eval,robust,figs`; default is all except `robust`. Output goes
to `results/<run name>/`. Engines and the static ONNX are written into the run folder.

### 5.1 Fix the ONNX for TensorRT 8.4 (`fix`)

```bash
python jetson/fix_qdq_onnx.py runs/deploy_qat/docker_qat/model_qdq_trt_no_out.onnx
# -> runs/deploy_qat/docker_qat/model_qdq_trt_no_out_static.onnx
```
TRT 8.4 does not accept the exported graph as is. The script:
- changes cubic Resize to linear (TRT 8.4 has no cubic),
- replaces the INT64 chain in LiteMLA that computes the Pad amounts
  (`Concat -> Reshape[-1,2] -> Slice -> Transpose -> Reshape[-1]`) by the constant it always
  produces, `[0,0,0,0,0,0,1,0]`; TRT treats INT64 tensors as shape tensors and rejects the
  2-D reshape,
- replaces dynamic Reshape shapes by constants and removes the dead nodes,
- makes activation zero points scalar and fixes input/output to `1x6x512x512` / `1x2x512x512`,
- runs the result once in onnxruntime to check the output shape.

### 5.2 Build the INT8 engine (`engine`)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=runs/deploy_qat/docker_qat/model_qdq_trt_no_out_static.onnx \
    --int8 --fp16 \
    --saveEngine=runs/deploy_qat/docker_qat/model_int8_qat.engine \
    --workspace=2048
```
Takes 5-10 minutes. Engines only work on the TRT version and GPU they were built with, so
build them on the device and do not copy `.engine` files between different setups. The
engine in `runs/deploy_qat/docker_qat` is the one from the reference NX; this step overwrites it.

### 5.3 Benchmark (`bench`)

```bash
python jetson/benchmark_jetson.py --model_dir runs/deploy_qat/docker_qat/ \
    --img_size 512 --channels 6 --batch_size 1 --iterations 200 --warmup 20 \
    --measure_power --rebuild --fp16_mm2_fp32
```
Builds FP32 and FP16 engines from `model_float_trt_base.onnx`, then times all three engines
with `trtexec --loadEngine` (batch 1, 200 iterations, 20 ms warm-up) and samples the
`VDD_CPU_GPU_CV` rail during a second run. Writes `benchmark_results.json` and the
per-layer profiles.

`--fp16_mm2_fp32` keeps the 8 LiteMLA attention MatMuls in FP32. In plain FP16 the
un-normalised attention numerator reaches ~6.5e4 and overflows. Do not build the FP16 engine
from an fp16-converted ONNX either; TRT 8.4 miscompiles that.

The FP32/FP16 engines here are for **latency only**. `model_float_trt_base.onnx` from a QAT
run holds QAT weights with the quantizers removed, which scores well below the INT8 graph.
Float accuracy numbers have to come from the pre-QAT checkpoint on the workstation. For the
same reason `deploy.sh` moves the FP32/FP16 engines to `results/<run>/engines/` before the
evaluation, so `eval_jetson.py` only picks up the INT8 engine.

### 5.4 Accuracy (`eval`, `robust`)

```bash
python jetson/eval_jetson.py --model_dir runs/deploy_qat/docker_qat/ \
    --engine runs/deploy_qat/docker_qat/model_int8_qat.engine \
    --sturm_normalize original --datasets s11,sturm,wf \
    --json_out results/docker_qat/eval.json
```
Three models are scored on the same data: `model_float.onnx` and
`model_qdq_trt_no_out.onnx` in onnxruntime (CPU), and the TRT INT8 engine.

- Sen1Floods11: val split, 512x512, S11 band statistics.
- STURM-Flood S2: 9 bands mapped to 6, `normalize=original` (DN inverse from the paper), 512.
- WorldFloods-v2: test split tiled at 512, clouds ignored.
- Labels follow `--protocol` (same for STURM and WF; don't mix protocols in one table):
  - `surface_water` (default since 2026-10-15): every water pixel counts, as in Sen1Floods11.
    WF flood + JRC permanent water = water (836 windows, edge windows zero-padded); STURM
    classes 1-5 = water, pixels without finite input ignored.
  - `flood_p3`: flood only (WF permanent water and STURM classes 3-5 ignored), same
    geometry; the revision P3 tables (785 WF windows).
  - `flood_paper`: flood only with resized WF edge windows and no STURM finite-support mask;
    the submitted paper's tables. `--wf_include_permanent` was removed.

Each dataset is reported micro (one pooled confusion matrix) and macro (mean over scenes
that contain flood). The script also times each model (median of 50 runs) at the end.

`--robustness` adds gain (0.8-1.3) and haze (0-0.2) sweeps on STURM and WorldFloods,
applied in reflectance space before normalisation, and reports worstDrop = clean minus
the lowest score over the sweep. This runs every model many times over both OOD sets;
plan for several hours, and use `nohup`.

`--quick N` stops after N batches per dataset. Good for checking that everything loads,
the numbers are not meaningful.

### 5.5 Figures (`figs`)

Select scenes on the workstation (`figures/plot_models.py`) and save the manifest.
The device renders those scenes with its own models; each pick includes its filename,
and the scripts stop if an index points to another file on this machine.
Different tile sizes or incomplete data can cause this.

```bash
# Float vs TRT INT8, the scenes of the paper figure, 4 rows per sheet
python jetson/plot_int8_vs_float.py \
    --picks_from manifests/qual_samples_manifest.json \
    --model_dir runs/deploy_qat/docker_qat/ \
    --engine runs/deploy_qat/docker_qat/model_int8_qat.engine \
    --eval_img_size 512 --no_agreement --paper_clean --rows_per_figure 4 \
    --out_dir results/docker_qat/figures

# clean vs perturbed rows for the robustness figure (+ .npz for merging on the workstation)
python jetson/qualitative_jetson.py --model_dir runs/deploy_qat/docker_qat \
    --engine runs/deploy_qat/docker_qat/model_int8_qat.engine \
    --manifest manifests/qual_robust_manifest_haze020.json --haze 0.20 \
    --img_size 512 --sturm_normalize original --dump_npz \
    --out results/docker_qat/figures/qual_robust_haze020.png
```
`deploy.sh` does the second call for `haze020`, `haze010` and `gain130`.

To look at other tiles by hand:
```bash
python jetson/qualitative_jetson.py --dataset sturm --list --stats | head -40
python jetson/qualitative_jetson.py --dataset sturm --select 417,EMSR279_03ALFARO_19_19_1_1.tif \
    --engine runs/deploy_qat/docker_qat/model_int8_qat.engine --haze 0.20
```
WorldFloods tiles are named `scene.tif#r<row>_c<col>`; quote them in the shell.
