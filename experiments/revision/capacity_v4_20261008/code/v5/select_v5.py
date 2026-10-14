"""Predeclared S11-validation-only selections for the v5 queue (CPU; no external numbers read).

lr    : per rung, the LR whose seed-42 run has the highest selected S11-val mIoU (ties -> smaller LR).
floor : the smallest rung (by parameter count) whose 4-seed S11-val mean mIoU is >= the reference
        (A3 at its selected LR) mean minus MARGIN_PP; if none qualifies, graph = null (floor undefined
        among the declared rungs; downstream floor jobs record a skip).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

MARGIN_PP = 1.0
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['lr', 'floor'])
p.add_argument('--receipts', nargs='+', required=True)
p.add_argument('--out', required=True)
p.add_argument('--reference', default='A3')
a = p.parse_args()
rows = []
for path in a.receipts:
    r = json.loads(Path(path).read_text())
    assert r['status'] == 'complete' and r['outcome'] == 'completed', path
    rows.append({'receipt': str(Path(path).resolve()), 'graph': r['graph'], 'lr': r['lr'], 'seed': r['seed'],
                 'parameters': r['parameters'], 'val_mIoU': 100 * r['source_validation']['best_val_mIoU']})
out = {'status': 'complete', 'outcome': 'completed', 'mode': a.mode, 'rows': rows,
       'data_used': 'S11 validation (89 scenes) only', 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
if a.mode == 'lr':
    graphs = {r['graph'] for r in rows}
    assert len(graphs) == 1 and all(r['seed'] == 42 for r in rows)
    best = max(rows, key=lambda r: (round(r['val_mIoU'], 6), -r['lr']))
    out.update(graph=best['graph'], lr=best['lr'], rule='argmax seed-42 S11-val mIoU; ties -> smaller lr')
else:
    by = {}
    for r in rows:
        by.setdefault(r['graph'], []).append(r)
    summary = {g: {'n': len(v), 'mean': statistics.mean(x['val_mIoU'] for x in v),
                   'sd': statistics.stdev(x['val_mIoU'] for x in v) if len(v) > 1 else None,
                   'parameters': v[0]['parameters'], 'lr': v[0]['lr']} for g, v in by.items()}
    ref = summary[a.reference]['mean']
    eligible = sorted((s['parameters'], g) for g, s in summary.items()
                      if g != a.reference and s['n'] == 4 and s['mean'] >= ref - MARGIN_PP)
    chosen = eligible[0][1] if eligible else None
    out.update(summary=summary, reference=a.reference, reference_mean=ref, margin_pp=MARGIN_PP,
               graph=chosen, lr=summary[chosen]['lr'] if chosen else None,
               rule=f'smallest rung with 4-seed S11-val mean >= {a.reference} mean - {MARGIN_PP} pp')
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('mode', 'graph', 'lr')}))
