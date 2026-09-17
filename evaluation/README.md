# Evaluation

`eval_ablations.py` evaluates trained checkpoints and stores pooled scores per
run. `perscene_eval.py` stores TP/FP/FN cells by flood map/AOI and event.
Collectors read these records to compute statistics. No model inference there.

```bash
python evaluation/eval_ablations.py --runs_base runs/nsweep \
  --runs_reg pseudo_ --n_filter 2500 --sturm_normalize original --gpu 0
bash experiments/perscene_sweeps.sh
python evaluation/perscene_collect.py
python evaluation/paired_group_tests.py
```

For a new run tree, pass both `--runs_reg` and `--n_filter`. Pass `--arch` when
the run name does not identify the architecture. `--force` recomputes cached
results. Perturbations use separate suffixes so clean results remain separate.

| Benchmark | Pooled result keys | Protocol |
|---|---|---|
| Sen1Floods11 | `s11_{val,test}_*` | Official validation/test splits |
| STURM-Flood | `sturm_*` | Original DN inverse and S11 normalization |
| WorldFloods-v2 | `wf_fo_*` | Clouds and JRC permanent water ignored |

S11 training-loader scores include augmentation. Use validation/test scores
for reporting. The original STURM protocol is the default and writes
`eval_results.json`. Percentile normalization writes
`eval_results_STURM_PERCENTILE.json`. Keep these protocols separate.

Micro IoU pools pixel counts. Macro IoU averages the per-scene scores defined
by each benchmark. Worst-case drop is computed per seed before averaging.
Non-default class maps or evaluation scales need separate cache directories.

| Collector | Results |
|---|---|
| `nsweep_collect.py` | Matched-budget pseudo versus S11 |
| `curation_collect.py`, `curation_master_collect.py` | Selector comparison |
| `stats_scene_level.py` | Paired group statistics and Holm correction |
| `paired_group_tests.py` | Matched N=252 and robustness contrasts |
| `scale_control_collect.py` | Evaluation-scale control |
| `protocol_sweep.py` | Class-map, perturbation and objective comparisons |
| `labeller_control_collect.py` | MNDWI labeller control |
| `verify_paper_source_data.py` | Supplied table cells against raw records |

Run inference after training has finished. The shell evaluation wrappers check
for matching active training processes. For geometry, spectral and scale
controls, see [docs/06_controls.md](../docs/06_controls.md).
