# Analysis

CKA and gradient-conflict diagnostics for the supplementary experiments.
Accuracy evaluation is in `evaluation/`.

```bash
python analysis/run_analysis.py cka \
  --official_ckpt hf --student_ckpt <run_dir>/best.pt \
  --data_root $DATA_ROOT/sen1floods11
python analysis/run_analysis.py gradconf \
  --csv <run_dir>/grad_conflict.csv --plot conflict.png
```

CKA compares the teacher's four neck levels with the student's four stages.
Keep the probe mode, sample order and spatial size fixed when comparing runs.
Spatial CKA and global-pool CKA are different probes.

Gradient cosines are recorded during OFA training with `--grad_conflict` and
`--grad_conflict_interval`. The analysis reports mean cosine, conflict rate and
mean cosine on conflicting steps. See supplement Secs. S6 and S8.

```bash
python figures/fig_cka.py                  # Fig. S3
python analysis/plot_grad_conflict_n.py    # Fig. S5
```

The gradient plotter reads `analysis/grad_conflict_n/*.csv` (other runs via `--runs`) and
writes to `assets/`.
