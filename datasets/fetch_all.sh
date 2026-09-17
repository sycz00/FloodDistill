#!/usr/bin/env bash
# Download Sen1Floods11, STURM-Flood, the WorldFloods-v2 test split, EfficientViT weights and
# the teacher checkpoint. GDACS needs Earth Engine (datasets/prepare.sh).
# Usage: DATA_ROOT=/mnt/data bash datasets/fetch_all.sh

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-$HOME/data}"
mkdir -p "$DATA_ROOT"
ok=0; skipped=0; failed=0

note() { printf '\n== %s\n' "$*"; }
have() { [ -e "$1" ]; }

# Sen1Floods11 (~1 GB). The helper also writes the split .txt files TerraTorch needs.
note "Sen1Floods11 -> $DATA_ROOT/sen1floods11"
if have "$DATA_ROOT/sen1floods11/v1.1"; then
    echo "   already present, skipping"; skipped=$((skipped+1))
elif bash datasets/prepare_sen1floods11.sh "$DATA_ROOT/sen1floods11"; then
    ok=$((ok+1))
else
    echo "   FAILED (needs gsutil; see docs/EXTERNAL_SETUP.md)"; failed=$((failed+1))
fi

# STURM-Flood (~6 GB). The loaders expect the nested Dataset/ directory.
note "STURM-Flood (Zenodo 12748983) -> $DATA_ROOT/STURM-Flood"
if have "$DATA_ROOT/STURM-Flood/Dataset"; then
    echo "   already present, skipping"; skipped=$((skipped+1))
else
    mkdir -p "$DATA_ROOT/STURM-Flood"
    if wget -q --show-progress -O /tmp/sturm.zip \
        "https://zenodo.org/records/12748983/files/Dataset.zip?download=1" \
       && unzip -q /tmp/sturm.zip -d "$DATA_ROOT/STURM-Flood" && rm -f /tmp/sturm.zip; then
        ok=$((ok+1))
    else
        echo "   FAILED -- download Dataset.zip from https://zenodo.org/records/12748983"
        echo "   and unzip it to $DATA_ROOT/STURM-Flood/"
        failed=$((failed+1))
    fi
fi

# WorldFloods-v2 test split (~3 GB) from Hugging Face. Only test/ is used for evaluation.
note "WorldFloods-v2 test split (huggingface isp-uv-es/WorldFloodsv2) -> $DATA_ROOT/WorldFloodsv2"
if have "$DATA_ROOT/WorldFloodsv2/test/S2"; then
    echo "   already present, skipping"; skipped=$((skipped+1))
elif WF_DIR="$DATA_ROOT/WorldFloodsv2" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("isp-uv-es/WorldFloodsv2", repo_type="dataset", local_dir=os.environ["WF_DIR"],
                  allow_patterns=["test/*", "dataset_metadata.csv", "README.md"])
PY
then
    ok=$((ok+1))
else
    echo "   FAILED -- see https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2"
    failed=$((failed+1))
fi

# EfficientViT weights are loaded relative to the repository root.
note "EfficientViT backbone weights -> assets/checkpoints"
if have "assets/checkpoints/efficientvit_seg/efficientvit_seg_b0_cityscapes.pt"; then
    echo "   already present, skipping"; skipped=$((skipped+1))
elif bash datasets/download_efficientvit_ckpts.sh; then
    ok=$((ok+1))
else
    echo "   FAILED"; failed=$((failed+1))
fi

# Teacher checkpoint (1.2 GB). Training downloads it on first use anyway; this fills the cache.
note "Prithvi-EO-2.0 teacher -> Hugging Face cache"
if python - <<'PY'
try:
    from huggingface_hub import snapshot_download
    snapshot_download("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11")
    print("   cached")
except Exception as e:
    print(f"   skipped: {e}")
    raise SystemExit(1)
PY
then ok=$((ok+1)); else skipped=$((skipped+1)); fi

printf '\n== fetch summary: %d fetched, %d skipped, %d failed\n' "$ok" "$skipped" "$failed"
cat <<'EOF'

Not included: the GDACS pool, which needs a Google Earth Engine project:
  earthengine authenticate
  EE_PROJECT=<id> bash datasets/prepare.sh
Sen1Floods11 training and evaluation work without it. The paper figures do not.
EOF
[ "$failed" -eq 0 ]
