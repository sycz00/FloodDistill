# V9 early readout (13 Oct 2026, Claude)

Preliminary, descriptive numbers for the V9 checkpoints while the V9 worker waited for a GPU.
Exposed sets only (S11 val/test, STURM, WF test); the sealed WorldFloods-v2 train/val set is untouched
and the V9 decision rules apply to it alone. Read-only on `research_v9_*`; nothing here feeds the V9 run.

- `report.txt`, `summary.csv`: edge arms E0 (teacher labels, V7 R4 CE anchors), E1 (MNDWI edge band),
  E2 (edge band ignored), E3 (teacher-confidence control), each with and without decode-time D (k=2, theta=0.1).
  Strata: reference-edge distance 1-2 px, thin water (component half-width in (0, 2] px), interior (>5 px).
- `counts_labelswap.json`: same-image label swap (B0 true CE on the 252 S11 images, human vs teacher labels).
- `run_all.sh` reproduces everything on CPU.
