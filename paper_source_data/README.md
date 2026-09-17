# Source data for `docs/main.pdf` and `docs/supplement.pdf`

Raw evaluation records and reduced tables, grouped by paper item.
`MANIFEST.csv` maps each copy to its original run path. Value CSVs and `raw/`
records are kept together.

## Experiments

### Training arms (4 seeds each: 42, 64, 91, 107)

| Arm family (run prefix) | Architecture | Objective | Label source | Budgets N reported |
|---|---|---|---|---|
| `ncgt_evit_ce`   / `sen_ce`             | EfficientViT-B0 | CE       | Sen1Floods11 | 252 |
| `ncgt_evit_lkd`  / `sen_dfkd`           | EfficientViT-B0 | Logit-KD | Sen1Floods11 | 252 |
| `ncgt_evit_ofa`  / `sen_ofa`            | EfficientViT-B0 | OFA      | Sen1Floods11 | 20,40,50,60,80,100,150,200,250,252 |
| `ncgt_unet_ce`   / `sen_ce_unet`        | UNet-ResNet50   | CE       | Sen1Floods11 | 252 |
| `ncgt_unet_lkd`  / `sen_dfkd_unet`      | UNet-ResNet50   | Logit-KD | Sen1Floods11 | 20,40,50,60,80,100,150,200,252 |
| `v2_evit_ce_ps`  / `pseudo_ce_aug`      | EfficientViT-B0 | CE       | teacher pseudo | 20,40,50,100,150,200,252,2500 |
| `v2_evit_lkd_ps` / `pseudo_dfkd_aug`    | EfficientViT-B0 | Logit-KD | teacher pseudo | 20,40,50,100,150,200,252,2500 |
| `v2_evit_ofa_ps` / `pseudo_ofa_aug`     | EfficientViT-B0 | OFA      | teacher pseudo | 20,40,50,100,150,200,252,2500,4000 |
| `v2_unet_ce_ps`  / `pseudo_ce_unet_aug` | UNet-ResNet50   | CE       | teacher pseudo | 252, 2500 |
| `v2_unet_lkd_ps` / `pseudo_dfkd_unet_aug` | UNet-ResNet50 | Logit-KD | teacher pseudo | 20,40,50,100,150,200,252,2500,4000 |
| `v2_relu6_ofa_ps_N2500`                 | EfficientViT-B0 (ReLU6) | OFA | teacher pseudo | 2500 |
| `n_qat_*` (QAT + TensorRT export)       | EfficientViT-B0 (ReLU6, W8A8) | OFA | teacher pseudo | 2500 |

**Reference models (single deterministic run, no seeds):** Prithvi-EO-2.0-300M-TL teacher
fine-tuned on Sen1Floods11; MNDWI > 0; NDWI > 0.

### Curation / subset-selection arms (N = 2,500, 4 seeds) — Fig. S1
`v2cur_fps_N2500` (FPS, deployed), `v2cur_confidence_N2500`, `v2cur_glister_N2500`,
`v2cur_r2core_N2500`, plus the unfiltered `*_raw_N2500` counterparts (random over the full
23,720-tile pool) and the `ncgt_evit_ofa_N252` Sen1Floods11 reference.

### Architecture-ablation arms (full curated pool N = 5,000, 4 seeds) — Table S3
`mnv3_ce`, `mnv3_dfkd` (MobileNetV3-Small 3.55 M); `microm_ce`, `microm_dfkd`
(Small CNN 1.64 M, no pretraining); `ce_hs`, `div_dfkd_hs`, `div_ofa_hs` (EfficientViT-B0).

### Diagnostic runs
`conflict_diag_ofa_N{100,300,500}_s{42,64}` (2 seeds) — OFA inter-branch gradient cosine, Fig. S5.
Spatial-CKA probes (`cka_spatial_gt_vs_pseudo_s*`) — Fig. S3.

### Evaluation conditions applied to every arm
Benchmarks: Sen1Floods11 test (in-distribution), STURM-Flood (46 flood maps / 29 events),
WorldFloods-v2 (18 AOIs, flood-only). Aggregations: micro (pixel-pooled) and macro (per-scene).
Perturbation grid: scalar gain `g ∈ {0.70,0.80,0.90,1.00,1.20,1.30}`, haze `h ∈ {0,0.05,0.10,0.20}`,
per-band gain `d ∈ {0.05,0.10,0.20,0.30}` (visible-only and opposed variants).
Evaluation tensors: 128 / 224 / 512 px at fixed 1.28 km ground extent (Table S1).

---

## Source-file map

### Main paper

| Item | What it shows | Values file | Primary raw source |
|---|---|---|---|
| **Fig. 1** | Pipeline overview | *(none — vector diagram)* | `assets/.../pipeline.pdf`, authored in PptxGenJS; no data file exists |
| **Fig. 2** | Objective ladder vs N, 3 benchmarks | `fig_ladder_vs_n_values.csv` | `runs/v2_nsweep/pseudo_{ce,dfkd,ofa}_aug_N*_s*/eval_results_STURM_ORIG.json` |
| **Fig. 3** | Qualitative, N=50 vs N=2,500 | `qual_samples_manifest.json` | `runs/qualitative_figures/v2crit_tc040/` (scene picks + per-panel IoU) |
| **Fig. 4** | Label efficiency, STURM | `fig_e3_sturm_values.csv` | `runs/perscene/sturm__{ncgt_evit_ofa,v2_evit_ofa_ps}_N*__s*__clean.json` |
| **Fig. 5** | Label efficiency, WorldFloods-v2 | `fig_e3_wf_values.csv` | `runs/perscene/wf__{ncgt_evit_ofa,v2_evit_ofa_ps}_N*__s*__clean.json` |
| **Table I** | Matched budget, N = 252 | `Table_I_values.csv` | per-scene caches (STURM, WF) + `runs/{ablations_sen1,v2_nsweep}/*_N252_s*/eval_results_STURM_ORIG.json` (S11 column) |
| **Table II** | Deployment operating point, N = 2,500 | `Table_II_values.csv` | per-scene caches + `runs/v2_nsweep/*_N2500_s*/eval_results_STURM_ORIG.json` |
| **Table III** | Accuracy through the deployment chain | `Table_III_values.csv` | `results/new_jetson/N_both_s{42,64,91,107}` (device rows) + per-scene caches (workstation rows) |
| **Table IV** | Jetson Xavier NX efficiency | `Table_IV_values.csv` | `results/s91_speed.log` (s42/s64 included for cross-check) |
| **Table V** | Training hyperparameters | *(none)* | configuration table, not measured data |

### Supplement

| Item | What it shows | Values file | Primary raw source |
|---|---|---|---|
| **Fig. S1** | Subset-selection robustness, N = 2,500 | `fig_curation_decomp_n2500_values.csv` | `runs/perscene/sturm__v2cur_*__s*__{clean,g*,h*}.json` |
| **Fig. S2** | Haze perturbation, qualitative | `qual_robust_manifest_haze010.json` | scene picks + per-panel IoU |
| **Fig. S3** | Spatial CKA by objective and depth | `fig_cka_values.csv` | `cka_spatial_gt_vs_pseudo_s{42,64,91,107}.json` |
| **Fig. S4** | Architecture control (UNet-LKD) | `fig_e3_sturm_unet_values.csv` | `runs/perscene/sturm__{ncgt_unet_lkd,v2_unet_lkd_ps}_N*__s*__clean.json` |
| **Fig. S5** | OFA inter-branch gradient conflict | `grad_conflict.csv` ×6 | `runs/ablations_diversity/conflict_diag_ofa_N{100,300,500}_s{42,64}/` |
| **Table S1** | Evaluation scale 128/224/512 px | `T3_sturm_native_scale.csv` | `results/revision/` |
| **Table S2** | Disagreement audit on STURM-S2 | `disagreement_audit_v2_N2500.json` | `runs/` (per-seed pixel counts) |
| **Table S3** | Architecture ablation, N = 5,000 | `Table_S3_values.csv` | `runs/ablations_diversity/{7 families}_N5000_s*/eval_results{,_STURM_ORIG}.json` |

`Supp_InText_Statistics/` additionally holds the released CSVs behind the supplement's
in-text numbers: S1 contamination audit, S3 geometry-matched (B1) and per-band-gain (T1)
controls, S4 MNDWI/NDWI threshold sweeps and labeller control, S5 teacher-semantics audit,
and the S7 device memory/latency figures.

---

## Verification

```bash
python evaluation/verify_paper_source_data.py
```

The verifier checks table cells against the supplied raw records. Value CSVs
retain `sd_ddof0` and `sd_ddof1` where provided.

Table III uses the four `N_both_s*` device records (42, 64, 91, 107). The separate
device robustness grids cover seeds 42, 64 and 91.
