"""Descriptive gradient/weight summaries at saved diagnostic steps; CPU reads only."""
import csv
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
rows=[]
for arm in ('C0','C1','C2','C3','C4','C5','HC','HL'):
    for seed in (42,64):
        data=json.loads((ROOT/f'runs/{arm}_s{seed}/loss_diagnostics.json').read_text())
        assert len(data)==32
        rows.append(dict(arm=arm,seed=seed,n_diagnostic_updates=len(data),
            diagnostic_updates='every50updates,startingupdate1',
            median_total_logit_gradient_l2=statistics.median(r['total_logit_gradient_l2'] for r in data),
            mean_weighted_kl_T2_logit_gradient_l2=statistics.mean(r['weighted_kl_T2_logit_gradient_l2'] for r in data),
            mean_unweighted_kl_T2_logit_gradient_l2=statistics.mean(r['unweighted_kl_T2_logit_gradient_l2'] for r in data),
            mean_effective_sample_size_fraction=statistics.mean(r['effective_sample_size']/r['valid_pixels'] for r in data),
            mean_valid_weight=statistics.mean(r['mean_valid_weight'] for r in data),
            uncertainty='descriptive dependent optimization diagnostics, not independent replicates'))
destination=HERE/'loss_diagnostics.csv'
assert not destination.exists()
with destination.open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(str(destination))
