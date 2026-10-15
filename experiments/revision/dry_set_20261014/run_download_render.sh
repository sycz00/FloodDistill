#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
$PY build_dry_set.py download && $PY build_dry_set.py render
echo "DONE $(date -Is)"
