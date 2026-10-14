#!/usr/bin/env bash
# Rebuild the reviewer wrap-up figures (figures/, figure_data/) and the
# verification record (verification.json). Read-only on every input; CPU only.
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(mktemp -d)}"
python3 -I make_figures.py
python3 verify_numbers.py   # imports make_figures from this folder, so no -I here
