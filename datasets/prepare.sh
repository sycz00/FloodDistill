#!/usr/bin/env bash
# GDACS pool: download, pseudo-label, select.
# Requires Earth Engine authentication and EE_PROJECT for acquisition.
# GDACS_FROM/TO bound event dates. Baselines reach 395 days before each event,
# so the acquisition range extends into December 2015 for the paper query.
# Raw tiles go to $DATA_ROOT/flood_unlabeled (default ~/data); PSEUDO_DIR sets the
# pseudo-label destination. UNLABELED_DIR overrides the raw destination.
# TOP_K defaults to 2500. TEACHER_CKPT defaults to the frozen public checkpoint.
# Usage: EE_PROJECT=YOUR_PROJECT_ID bash datasets/prepare.sh
# Skip stages with SKIP_DOWNLOAD=1, SKIP_PSEUDO=1 or SKIP_CURATE=1.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EE_PROJECT="${EE_PROJECT:-}"
# "hf": public Prithvi Sen1Floods11 checkpoint (the paper's teacher).
TEACHER_CKPT="${TEACHER_CKPT:-hf}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
UNLABELED_DIR="${UNLABELED_DIR:-$DATA_ROOT/flood_unlabeled}"
PSEUDO_DIR="${PSEUDO_DIR:-./data/pseudo_gdacs}"
GDACS_FROM="${GDACS_FROM:-2017-01-01}"
GDACS_TO="${GDACS_TO:-2026-07-31}"
# Orange and Red alerts only; larger events are more likely to show flooding in Sentinel-2.
GDACS_ALERT="${GDACS_ALERT:-orange;red}"
TOP_K="${TOP_K:-2500}"
GPU="${GPU:-0}"

# 1) Download GDACS Sentinel-2 tiles during flood events via Google Earth Engine.
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  : "${EE_PROJECT:?set EE_PROJECT to your Earth Engine project id}"
  echo "[prepare] (1/3) downloading GDACS tiles → $UNLABELED_DIR"
  python datasets/download_flood_events.py \
    --save_path  "$UNLABELED_DIR" \
    --project    "$EE_PROJECT" \
    --fetch_gdacs --gdacs_from "$GDACS_FROM" --gdacs_to "$GDACS_TO" \
    --gdacs_alert "$GDACS_ALERT" --max_gdacs_events 100000 \
    --patches_per_event 4 --add_baseline --num_workers 8 \
    --patch_cloud_prob_max 40 --max_permanent_water_frac 0.5 \
    --min_new_water_frac 0.01 --max_candidates 8
fi

# 2) Teacher pseudo-labels (hard mask, per-pixel confidence). Scenes with mean confidence
#    >= 0.80 are kept (23,571 of 23,720). No soft logits are stored; OFA and Logit-KD use
#    the live teacher.
if [[ "${SKIP_PSEUDO:-0}" != "1" ]]; then
  echo "[prepare] (2/3) generating pseudo-labels → $PSEUDO_DIR"
  python scripts/train.py pseudo generate \
    --official_ckpt "$TEACHER_CKPT" \
    --unlabeled_root "$UNLABELED_DIR" --unlabeled_source flood \
    --out_dir "$PSEUDO_DIR" \
    --confidence_threshold 0.80 --min_flood_frac 0.0 \
    --store_pixel_confidence \
    --batch_size 32 --num_workers 8 --gpu_ids "$GPU"
fi

# 3) FPS selection -> diversity_N<TOP_K>_index.json. Build one index per training N; a larger
#    index with a smaller --n_samples gives a random subset.
if [[ "${SKIP_CURATE:-0}" != "1" ]]; then
  echo "[prepare] (3/3) curating diversity pool (top_k=$TOP_K) → $PSEUDO_DIR/diversity_N${TOP_K}_index.json"
  python scripts/curate.py diversity --pseudo_dir "$PSEUDO_DIR" --top_k "$TOP_K"
  # confidence-ranked baseline (paper comparison); writes curated_index.json
  python scripts/curate.py confidence --pseudo_dir "$PSEUDO_DIR" --top_k "$TOP_K"
fi

echo "[prepare] done. Train with:"
echo "  python scripts/train.py ofa --official_ckpt $TEACHER_CKPT \\"
echo "      --student_arch efficientvit_b0 --data_root $DATA_ROOT/sen1floods11 \\"
echo "      --pseudo_dir $PSEUDO_DIR --no_labeled \\"
echo "      --curated_pseudo --curated_index_file diversity_N${TOP_K}_index.json --n_samples $TOP_K"
