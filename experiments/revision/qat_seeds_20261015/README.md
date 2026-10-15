# Four-seed W8A8 QAT — EfficientViT-B0 and R4 (15 Oct 2026)

**Question.** The 14 Oct Jetson integer run (`../jetson_integer_20261014T101741Z/`) quantization-aware trained only
seed 42 of EfficientViT-B0 (R4 only a one-update smoke, because its PTQ passed). Do the results hold for all four
training seeds of both models?

**Recipe (unchanged, same code).** `scripts/jetson_integer_prepare.py`: float ONNX export + parity, BN folding,
`prepare_qat` on all convolutions including the head (W8A8, per-tensor symmetric activations, per-channel weights,
`FixedQATConv2d`), AdamW 1e-4, weight decay 0.05, batch 4, focal γ 2 on the 512 frozen cached GDACS TRAIN tiles,
clip 5, 128 updates, QAT seed 1729, QDQ ONNX export + onnxsim + graph audit + strict parity check, ORT evaluation on
the 89 S11 validation chips, paired whole-event bootstrap vs the float parent. Calibration, validation and QAT-train
arrays are the 14 Oct files (symlinked, read-only); only the parent checkpoint changes per seed. R4 also repeats the
14 Oct selected PTQ variant (`percentile_pc`). CPU only, `torch.set_num_threads(2)` as on 14 Oct.

**Parents.** B0: `/data1/schmalstieg/flooddistill/runs_repro/pseudo_evit_relu6_ofa_N2500_s{seed}` (pretrained-init
ReLU6 OFA, N = 2,500). R4: the selected v5 LKD checkpoints (`fl_R4_lr5e-3_s42`, `fl_R4_sel_s{64,91,107}`), the same
models as V7–V10.

**Results** (`qat_seeds_summary.csv`; source validation, points vs each seed's float parent):

| Model | Seed | Float mIoU | QAT Δ pooled / equal-event | QAT ok | PTQ Δ pooled | PTQ ok |
|---|---|---|---|---|---|---|
| B0 | 42 | 85.95 | +0.75 / +0.42 | ✔ | — | — |
| B0 | 64 | 87.03 | −2.07 / −1.70 | ✘ | — | — |
| B0 | 91 | 85.78 | −1.27 / −0.92 | ✘ | — | — |
| B0 | 107 | 86.24 | −0.69 / −0.87 | ✔ | — | — |
| R4 | 42 | 88.01 | −1.45 / −1.60 | ✘ | +0.16 | ✔ |
| R4 | 64 | 87.32 | −0.86 / −0.95 | ✔ | +0.01 | ✔ |
| R4 | 91 | 88.12 | −1.68 / −1.86 | ✘ | −35.96 | ✘ |
| R4 | 107 | 88.30 | −1.00 / −0.94 | ✔ | −0.44 | ✔ |

**Findings.**
1. The 14 Oct B0 QAT gain (+0.75) does not generalise: 2 of 4 seeds pass, mean −0.82 points; seed 64 loses 2.1
   points, the same seed that lost 3–4 points in the paper's Jetson INT8 chain (Table III).
2. For R4, 128-update QAT is worse than PTQ on three seeds (mean −1.25 vs −0.08 among PTQ-passing seeds); the QAT
   objective (focal on cached hard masks, LR 1e-4) moves the LKD parent. But PTQ collapses on seed 91 (−36 points),
   and QAT rescues it to −1.7. PTQ robustness is seed-dependent.
3. **Numerical fragility.** The QAT trajectory depends on the CPU thread count: with 4 threads the same B0 s42 job
   gives −0.09 instead of +0.75 (byte-identical repeats within a thread setting;
   `repeat_check/`, `sensitivity_threads4/`). Effects of this size are not attributable to the method.
4. Strict logit parity of the QDQ export still fails for all B0 seeds (as on 14 Oct); runtime accuracy is evaluated on
   the exported model, as before. Fully integer execution is not demonstrated.

**Artifacts.** Models and exports: `jetson_repro/qat_seed_models/` (with `MANIFEST.json`). Rerun: `bash run_all.sh`
(about 5 min on 16 cores); `collect.py` rebuilds the CSV.
