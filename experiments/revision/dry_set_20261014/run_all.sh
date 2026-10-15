#!/usr/bin/env bash
# Independent dry-tile set: per-stratum selection (static maps only, parallel) -> merge -> download (GDACS code) -> render.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
mkdir -p logs
for s in desert_bare urban cropland forest grass_shrub dry_floodplain mountain_shadow snow cloud_shadow; do
  $PY build_dry_set.py select --strata $s --batch 24 --rounds 20 > logs/select_$s.log 2>&1 &
done
wait
$PY build_dry_set.py merge
$PY build_dry_set.py download
$PY build_dry_set.py render
echo "DONE $(date -Is)"
