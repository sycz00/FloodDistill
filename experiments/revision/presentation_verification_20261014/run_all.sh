#!/usr/bin/env bash
# Read-only checks behind the presentation verification of 14 Oct 2026. CPU only.
# Never opens WorldFloods-v2 train/val. Writes only to ./outputs.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
R=/home/schmalstieg/repos/FloodDistill
REV=$R/experiments/revision
export OMP_NUM_THREADS=2
PRE=$(ls /data1/schmalstieg/flooddistill/revision_20261005/snapshots/*/assets/checkpoints/efficientvit_seg/efficientvit_seg_b0_cityscapes.pt | head -1)
O=/home/schmalstieg/repos/OLD/esi_project/runs
cd $R
nice -n 10 $PY "$OLDPWD/checks/relu6_init_test.py" "$PRE" \
  $O/repro_review/pseudo_ofa_relu6_N2500_s42/best.pt $O/repro_review/pseudo_ofa_relu6_N2500_s64/best.pt \
  $O/repro_review/qat_relu6_N2500_s42/best.pt /data1/schmalstieg/flooddistill/runs_repro/pseudo_evit_relu6_ofa_N2500_s42/best.pt \
  $O/repro_full/pseudo_ofa_N2500_s42/best.pt $REV/capacity_v7_20261011T210730Z/runs/A0_selected_s64/best.pt \
  2>/dev/null > "$OLDPWD/outputs/relu6_init_test.txt"
nice -n 10 $PY "$OLDPWD/checks/relu6_vs_hs_paired.py" paper_source_data/Table_III_Deployment_Chain/raw_workstation > "$OLDPWD/outputs/relu6_vs_hs_paired.txt"
nice -n 10 $PY "$OLDPWD/checks/dry_funnel.py" > "$OLDPWD/outputs/dry_funnel.txt"
nice -n 10 $PY "$OLDPWD/checks/sturm_classes.py" > "$OLDPWD/outputs/sturm_classes.txt"
nice -n 10 $PY "$OLDPWD/checks/sturm_mndwi_by_class.py" > "$OLDPWD/outputs/sturm_mndwi_by_class.txt"
python3 "$OLDPWD/checks/wf_dry_by_event.py" $REV/research_v9_recovery_20261013T075000Z/dry_window_alarms.csv > "$OLDPWD/outputs/wf_dry_by_event.txt"
nice -n 10 $PY "$OLDPWD/checks/wf_dry_mndwi.py" 2>/dev/null > "$OLDPWD/outputs/wf_dry_mndwi.txt"
[ "${SKIP_RADIOMETRY:-0}" = 1 ] || nice -n 10 $PY "$OLDPWD/checks/radiometry.py" "$OLDPWD/outputs/radiometry.json" 2>/dev/null > "$OLDPWD/outputs/radiometry.txt"
echo done
