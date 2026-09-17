# Datasets

| Dataset | Use | Default directory |
|---|---|---|
| Sen1Floods11 | Official train/validation/test splits; teacher checkpoint and student baselines | `$DATA_ROOT/sen1floods11/` |
| GDACS Sentinel-2 | Unlabelled images for teacher supervision | `$DATA_ROOT/flood_unlabeled/` |
| STURM-Flood, S2 | External evaluation | `$DATA_ROOT/STURM-Flood/Dataset/` |
| WorldFloods-v2 | External evaluation, flood-only | `$DATA_ROOT/WorldFloodsv2/` |

All models receive B2, B3, B4, B8A, B11 and B12. The teacher is the public
Prithvi-EO-2.0-300M-TL checkpoint fine-tuned on Sen1Floods11. It stays frozen here.
Resizing and normalization are described in [docs/01_datasets.md](../docs/01_datasets.md).

## Acquisition

```bash
DATA_ROOT=/path/to/data bash datasets/fetch_all.sh
```

- **Sen1Floods11:** `datasets/prepare_sen1floods11.sh [target]` downloads the hand-labelled
  S2 chips, labels and official splits from the public GCS bucket and writes the split files
  TerraTorch expects. `--data_root` is the directory that contains `v1.1/`.
- **STURM-Flood:** Zenodo record `12748983`. Keep the nested `Dataset/` directory. The loader
  inverts the documented DN scaling and then applies the S11 normalisation.
- **WorldFloods-v2:** Hugging Face dataset `isp-uv-es/WorldFloodsv2`. Only the test split is
  used; it already contains `test/{S2,gt,PERMANENTWATERJRC}`. The permanent-water masks are
  needed for the flood-only protocol.

```bash
hf download isp-uv-es/WorldFloodsv2 --repo-type dataset \
  --include "test/*" --local-dir "$DATA_ROOT/WorldFloodsv2"
```

## GDACS and pseudo-labels

Authenticate Earth Engine and supply a registered project ID:

```bash
earthengine authenticate
EE_PROJECT=YOUR_PROJECT_ID bash datasets/prepare.sh
```

`prepare.sh` downloads raw tiles, generates pseudo-labels and selects the pool.
It accepts `SKIP_DOWNLOAD=1`, `SKIP_PSEUDO=1` and `SKIP_CURATE=1`.

- Download: Orange/Red GDACS flood alerts, Sentinel-2 L1C during each event,
  and pre-event baselines 30–395 days before it. Event imagery uses a 50% cloud
  gate with an 80% fallback; baselines use 20%. Patches cover 264 pixels at
  10 m. Keep the per-patch cloud and water screens in the acquisition recipe.
- Generate: the frozen teacher stores hard masks and per-pixel confidence at
  224×224. The recipe uses `--confidence_threshold 0.80 --min_flood_frac 0.0
  --store_pixel_confidence`. KD/OFA obtain soft targets from the live teacher.
- Select: `scripts/curate.py diversity --top_k N` applies the flood-fraction and
  confidence-variation filter, then FPS on the 15-D descriptor. The output is
  `diversity_N<N>_index.json`; N=2,500 is the main scale-up budget.

Build an index at each training N. `PseudoLabeledDataset` samples uniformly from
that index, so a 2,500-entry index with `--n_samples 50` would select a random
50 from those 2,500. Selector controls are in `experiments/build_indices.sh`.

## Provenance

Pseudo files retain a `source` field; `source_manifest.csv` maps saved tiles to
raw inputs. Record GEE asset IDs and acquisition metadata from the raw tree:

```bash
python -m data.build_flood_unlabeled_manifest \
  --root $DATA_ROOT/flood_unlabeled --out data/flood_unlabeled_manifest.csv
python datasets/redownload_from_manifest.py \
  --manifest data/flood_unlabeled_manifest.csv \
  --save_path $DATA_ROOT/flood_unlabeled_restored --project "$EE_PROJECT"
```

The manifest builder reads local `metadata.json` files. Keep these manifests
with the pool and its selected indices.
