#!/usr/bin/env bash
# Download the hand-labelled S2 chips, labels and official splits using gsutil.
# Convert CSV split pairs to the scene-ID text files required by TerraTorch.
# Usage: bash datasets/prepare_sen1floods11.sh [S11_DIR]
# S11_DIR defaults to $DATA_ROOT/sen1floods11 (DATA_ROOT defaults to ~/data) and contains v1.1/.

set -euo pipefail

S11_DIR="${1:-${DATA_ROOT:-$HOME/data}/sen1floods11}"
BUCKET="gs://sen1floods11/v1.1"
HANDLABELED="$S11_DIR/v1.1/data/flood_events/HandLabeled"
SPLIT_DIR="$S11_DIR/v1.1/splits/flood_handlabeled"

echo "[prepare_sen1floods11] downloading S2Hand + LabelHand + splits -> $S11_DIR"
mkdir -p "$HANDLABELED/S2Hand" "$HANDLABELED/LabelHand" "$SPLIT_DIR"
gsutil -m rsync -r "$BUCKET/data/flood_events/HandLabeled/S2Hand"  "$HANDLABELED/S2Hand"
gsutil -m rsync -r "$BUCKET/data/flood_events/HandLabeled/LabelHand" "$HANDLABELED/LabelHand"
gsutil -m rsync -r "$BUCKET/splits/flood_handlabeled" "$SPLIT_DIR"

echo "[prepare_sen1floods11] converting .csv splits -> .txt (TerraTorch's expected format)"
python3 - "$SPLIT_DIR" << 'EOF'
import csv
import pathlib
import sys

split_dir = pathlib.Path(sys.argv[1])
for csv_path in sorted(split_dir.glob("flood_*_data.csv")):
    txt_path = csv_path.with_suffix(".txt")
    ids = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            stem = row[0].strip().rsplit(".", 1)[0]   # strip .tif
            loc_id = stem.rsplit("_", 1)[0]            # strip _S1Hand/_S2Hand/_LabelHand suffix
            ids.append(loc_id)
    txt_path.write_text("\n".join(ids) + "\n")
    print(f"  {csv_path.name} -> {txt_path.name}: {len(ids)} ids")
EOF

echo "[prepare_sen1floods11] done. data_root for training: $S11_DIR"
