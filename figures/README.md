# Figures

The quantitative figures are drawn from your own runs. Nothing is shipped under `runs/`, so
finish all experiments first: every recipe in README steps 4 and 5 with all four seeds, the
selector runs of `experiments/curation_compare.sh`, and the teacher cells. No recipe writes
the teacher cells. The first line below does that. Run from the repository root:

```bash
python evaluation/perscene_eval.py --datasets sturm,wf --families teacher
bash figures/make_figures.sh
```

The run names and budgets each figure expects are fixed in `FAMILIES`
(`evaluation/perscene_eval.py`) and in `NS` (`fig_ladder_vs_n.py`). They go beyond the README
examples, e.g. N=20 and 40. A missing run leaves a gap in its curve; Fig. S1 stops with an error.

Outputs and value CSVs go to `assets/sturm_orig/figures/`. The submitted values are the
`*_values.csv` files in [paper_source_data/](../paper_source_data/README.md).
`fig_style.py` defines the family aliases, colours and robustness statistics.

| Paper item | Script | Reads |
|---|---|---|
| Fig. 2 | `fig_ladder_vs_n.py` | `runs/nsweep/*/eval_results.json` |
| Fig. 4 | `fig_e3_label_efficiency.py --dataset sturm` | `runs/perscene/sturm__*__clean.json` |
| Fig. 5 | `fig_e3_label_efficiency.py --dataset wf` | `runs/perscene/wf__*__clean.json` |
| Fig. S1 | `e5_curation_decomp.py` | `runs/perscene/sturm__cur_*` |
| Fig. S3 | `fig_cka.py` | `paper_source_data/Supp_Figure_S3_CKA_Alignment/raw/` |
| Fig. S4 | `fig_e3_label_efficiency.py --dataset sturm --arch unet` | `runs/perscene/sturm__*__clean.json` |
| Fig. S5 | `../analysis/plot_grad_conflict_n.py` | `analysis/grad_conflict_n/*.csv` |

Worst-case drop is calculated within each seed before averaging.

## Qualitative figures

These need data (`$DATA_ROOT`, [data paths](../docs/DATA_LOCATION.md)) and model checkpoints.
`qual_rgb.py` supplies the common true-colour conversion.

**Fig. 3** (N=50 vs N=2,500): render both budget panels, then join them.

```bash
python figures/plot_models.py --no_agreement --paper_clean \
  --picks_from paper_source_data/Figure_03_Qualitative_N50_N2500/qual_samples_manifest.json
python figures/stitch/fig_qualN_merge.py
```

**Fig. S2** (haze, h=0.10): clean/perturbed rows. The INT8 column comes from the device;
see section 5.5 in [jetson_repro/README.md](../jetson_repro/README.md).

```bash
python figures/fig_qualitative_batches.py --gpu 0 --axes haze --haze 0.10
```

Without `--picks_from`, `plot_models.py` ranks scenes itself and writes a new manifest.
