#!/usr/bin/env bash


# Download upstream EfficientViT segmentation weights (B0 by default).
# Run from the repository root: the upstream loader resolves checkpoint paths
# relative to the working directory.
# Usage: EVIT_VARIANTS="b0 b1 b2" bash datasets/download_efficientvit_ckpts.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VARIANTS="${EVIT_VARIANTS:-b0}"
OUT_DIR="assets/checkpoints/efficientvit_seg"
BASE_URL="https://huggingface.co/han-cai/efficientvit-seg/resolve/main"

mkdir -p "$OUT_DIR"

for v in $VARIANTS; do
  case "$v" in
    b0) dataset="cityscapes" ;;
    b1|b2) dataset="ade20k" ;;
    *) echo "[download_efficientvit_ckpts] unknown variant '$v' (expected b0/b1/b2)"; exit 1 ;;
  esac
  fname="efficientvit_seg_${v}_${dataset}.pt"
  dest="$OUT_DIR/$fname"
  if [[ -f "$dest" ]]; then
    echo "[download_efficientvit_ckpts] $fname already present, skipping"
    continue
  fi
  echo "[download_efficientvit_ckpts] downloading $fname ($dataset-pretrained)"
  curl -L -o "$dest" "$BASE_URL/$fname"
done

echo "[download_efficientvit_ckpts] done -> $OUT_DIR"
