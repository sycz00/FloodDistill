# Docker

Build from the repository root. The image uses `environment.yml`, TerraTorch,
Prithvi and the EfficientViT source checkout. It supports training, evaluation,
pseudo-labels, QAT and ONNX export. Build TensorRT engines on the Jetson.

```bash
docker build -f docker/Dockerfile -t flooddistill .
```

GPU execution needs the NVIDIA Container Toolkit. Mount the datasets at `/data`,
the pseudo pool at `/pseudo`, outputs at `/out` and the Hugging Face cache at
`/hf`. Use writable mounts for outputs/cache and for pool generation.

## Before you start

The image contains code and environment only, no data.

1. Download the datasets on the host, as in the root README
   ([2. Data](../README.md#2-data)):
   `DATA_ROOT=/path/to/data HF_HOME=/path/to/hf_cache bash datasets/fetch_all.sh`
2. For pseudo-label runs, a pseudo pool with `diversity_N<N>_index.json`
   (`EE_PROJECT=... bash datasets/prepare.sh`, or the `generate` and `curate` services below).
   S11-only runs (`DATASET=s11`), `verify` and `collect` do not need it, but `PSEUDO_ROOT`
   must still point to an existing directory.

EfficientViT-B0 weights are downloaded during `docker build`; no host download is needed.

**Teacher checkpoint.** `--official_ckpt hf` calls `hf_hub_download` in
[models/teacher.py](../models/teacher.py) and fetches
`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` (~1.2 GB) into `HF_CACHE` on first
use. Step 1 pre-fills the cache when run with `HF_HOME` set to the same directory. Without it,
the first teacher run downloads the checkpoint.

## Compose

Set local paths in `docker/.env`, then create the directories **as your user**. If a
mount path is missing, Compose stops with `bind source path does not exist`, so that
Docker does not create the directory as root. A root-owned `OUT_ROOT` causes
`PermissionError: ... '/out/<run>'`. Fix it with
`sudo chown $(id -u):$(id -g) "$OUT_ROOT"`.

```bash
cat > docker/.env <<EOF_CONFIG
DOCKER_UID=$(id -u)
DOCKER_GID=$(id -g)
DATA_ROOT=/path/to/data
PSEUDO_ROOT=/path/to/pseudo_gdacs
OUT_ROOT=/path/to/runs_repro
HF_CACHE=/path/to/hf_cache
EOF_CONFIG
( set -a; . docker/.env; mkdir -p "$PSEUDO_ROOT" "$OUT_ROOT" "$HF_CACHE" )

docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm train
docker compose -f docker/docker-compose.yml run --rm evaluate
docker compose -f docker/docker-compose.yml run --rm train-relu6
docker compose -f docker/docker-compose.yml run --rm export
```

`DATA_ROOT` contains `sen1floods11/`, `STURM-Flood/`, `WorldFloodsv2/` and
`flood_unlabeled/`. The S11 root contains `v1.1/`. Resolve host symlinks before
mounting data. Compose sets shared memory and the host UID/GID for DataLoader
workers and output ownership.

| Service | Purpose |
|---|---|
| `generate`, `curate` | Generate pseudo-labels and FPS indices; writable `/pseudo` |
| `train`, `train-relu6` | Hardswish/ReLU6 student training |
| `evaluate` | S11 checkpoint evaluation |
| `export` | QAT and ONNX export |
| `collect` | Paired statistics from the per-scene cells in `OUT_ROOT/perscene` |
| `verify` | Recompute supplied paper table cells |
| `robustness-b2` | Train, evaluate and test the supplement S3 robustness pairs (GPU) |
| `flooddistill` | Custom entrypoint arguments |

`MODEL`, `OBJECTIVE`, `DATASET`, `N`, `SEED` and `EPOCHS` configure training.
`CKPT` selects the checkpoint for evaluation/export; `export` also takes `N` (QAT pool
index) and `QAT_EPOCHS`. For a quick smoke test, add `EPOCHS=1` / `QAT_EPOCHS=1`. For example:

```bash
DATASET=s11 N=252 OBJECTIVE=ce docker compose -f docker/docker-compose.yml run --rm train
CKPT=/out/ablations_sen1/sen_ce_N252_s42/best.pt docker compose -f docker/docker-compose.yml run --rm evaluate
```

Runs go to `OUT_ROOT` in the layout the shell recipes use for `runs/`: `ablations_sen1/sen_ce_N252_s42`,
`nsweep/pseudo_ofa_aug_N2500_s42`, `repro/repro_ofa_relu6_N2500_s42` and so on. The per-scene sweep,
the collectors and the figure scripts look for them under `runs/`, so mount `OUT_ROOT` there:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -v /path/to/OUT_ROOT:/workspace/FloodDistill/runs flooddistill bash experiments/perscene_sweeps.sh
```

Run this sweep before `collect`: the image ships no results, and the cells must exist first.

The pseudo index defaults to `diversity_N<n>_index.json`. Supply `--index_file`
for a mapped index. Its length must equal N and all referenced tiles must exist.

### Robustness test (supplement S3, GPU)

`robustness-b2` regenerates `B2_robustness_holm.csv`. It trains the three N=252 pairs
(EfficientViT CE and OFA, UNet Logit-KD) on S11 and pseudo labels over four seeds, evaluates
them on STURM under gain and haze, and runs the Holm block. Here `OUT_ROOT` is mounted as
`runs/`. Finished checkpoints and cells are reused, so an interrupted run can be restarted.

```bash
mkdir -p /path/to/runs_b2
OUT_ROOT=/path/to/runs_b2 docker compose -f docker/docker-compose.yml run --rm robustness-b2
diff /path/to/runs_b2/B2_robustness_holm.csv \
  paper_source_data/Supp_InText_Statistics/S3_harmonisation/B2_robustness_holm.csv
```

`PSEUDO_ROOT` must hold `diversity_N252_index.json`; `scripts/curate.py diversity --top_k 252`
writes it into the pool.
For a regenerated pool, set `IDX_PREFIX=mapped_diversity`. `SEEDS=42 EPOCHS=1` is a smoke test.
`GPU` selects the device.

## Entrypoint

A leading option dispatches to `docker/entrypoint.py`; other commands run inside
the conda environment. `--dry_run` prints commands after validating inputs.

| Option | Values |
|---|---|
| `--mode` | `train`, `distill`, `evaluate`, `export`, `all` |
| `--model` | `evit`, `evit_relu6`, `unet` |
| `--objective` | `ce`, `lkd`, `ofa` |
| `--dataset` | `s11`, `pseudo` |
| `--n` | Training budget: 252 matched, 2500 scale-up |
| `--seeds` | Paper seeds: `42 64 91 107` |
| `--ckpt`, `--out` | Input checkpoint and output directory |

`--mode all` runs the five paper architecture/objective cells, skipping existing
`best.pt` files. Trailing arguments after `--` pass through to the training CLI.
The shared training flags follow `experiments/nsweep_pseudo_vs_gt.sh`.

QAT uses the selected pseudo pool and a retrained ReLU6 checkpoint.
`--qat_epochs 0` calibrates and exports without QAT updates. See
[docs/03_quantization.md](../docs/03_quantization.md).

All dataset paths derive from `DATA_ROOT=/data` inside the container, including
STURM and WorldFloods for external evaluation. Point `PERSCENE_ROOT` at the mounted
output directory for generating and reading per-scene control caches. See
[docs/06_controls.md](../docs/06_controls.md) for the corresponding scripts.

Logging defaults to `WANDB_MODE=disabled`. Set it to `online` and supply
`WANDB_API_KEY` to enable W&B.
