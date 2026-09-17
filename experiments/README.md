# Experiment recipes

Run these scripts from the repository root. `LABELS=s11` and `LABELS=pseudo`
select the two training sources in the matched-budget comparison.

```bash
LABELS=s11 N_VALUES=252 bash experiments/nsweep_pseudo_vs_gt.sh
LABELS=pseudo N_VALUES=252 bash experiments/nsweep_pseudo_vs_gt.sh
LABELS=pseudo N_VALUES=2500 bash experiments/nsweep_pseudo_vs_gt.sh
GPU=0 SEEDS="42 64" ARMS="fps glister" bash experiments/curation_compare.sh
```

Defaults use seeds 42, 64, 91 and 107. `dfkd` in a run name means Logit-KD.

| Script | Purpose |
|---|---|
| `build_indices.sh` | Build exact-N indices and the selector pre-passes |
| `verify_indices.py` | Check index sizes, membership and overlap |
| `curation_compare.sh` | Train the N=2,500 selector comparison |
| `curation_eval.sh` | Evaluate selector arms; `HAZE=1` adds haze |
| `nsweep_pseudo_vs_gt.sh` | Train either label source with the same five architecture/objective cells |
| `nsweep_eval.sh` | Evaluate both sources on S11, STURM and flood-only WorldFloods |
| `perscene_sweeps.sh` | Generate per-scene confusion cells |
| `repro_n2500.sh` | Train Hardswish and ReLU6 OFA students |
| `b1_crop_train.sh` | Match S11 training extent to the GDACS tiles |
| `perscene_controls.sh` | Run spectral-index, geometry, gain, scale and protocol controls |

Build each pseudo index at exactly its training N. The loader samples uniformly
from the index, so selecting a smaller N from a larger index changes the subset
experiment. The verification step checks this before training.

The nsweep recipe uses matching augmentation and optimizer settings for both
label sources. For the selector comparison, compare arms within its own tree:
that recipe uses AMP-fp16.

Evaluation uses STURM `--sturm_normalize original`. Clean results go to
`eval_results.json`; percentile, gain and haze outputs are separate. The eval
wrappers check for a matching active training process before reading `best.pt`.
For control definitions, see [docs/06_controls.md](../docs/06_controls.md).
