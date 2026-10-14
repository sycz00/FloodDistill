#!/usr/bin/env python3
"""Exploratory: failure strata along the size ladder R1 (14) - R2 (418) - R3 (1,058) - R4 (12,230) - A3 (30,922),
seed 64 'sel' checkpoints, exposed sets S11 test and WF test. Compression-induced errors = pixels where R4 is
right and the smaller rung is wrong. CPU, read-only. Writes ladder_report.txt."""
import os, sys, json
from collections import defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import probe_cpu as pc, probe_failure as pf  # noqa
RUNGS = ['R1', 'R2', 'R3', 'R4', 'A3']
CK = '/data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008/v5/fl_{g}_sel_s64/best.pt'

def main():
    import numpy as np, torch
    spec = pc.SPEC; sys.path.insert(0, spec['capsule']); sys.path.insert(0, str(pc.V7))
    torch.set_num_threads(16)
    import evaluate_v7 as ev
    from train_v7 import build_model
    from evaluation import revision_native_border as rb
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    models = {}
    for g in RUNGS:
        m = build_model(g, 'lkd', spec['rungs_dir']); m.load_state_dict(torch.load(CK.format(g=g), map_location='cpu', weights_only=False)['model'], strict=True)
        models[g] = m.eval()
    sets, _, _ = ev.datasets(spec)
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    rep = []
    for name, ds, batch in [s for s in sets if s[0] in ('s11', 'wf')]:
        acc = defaultdict(lambda: np.zeros(4, np.int64)); comp = defaultdict(lambda: np.zeros(2, np.int64))
        for begin in range(0, len(ds), batch):
            ids = list(range(begin, min(begin + batch, len(ds))))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            with torch.no_grad():
                preds = {g: (lambda z: (z[:, 1] - z[:, 0]) > 0)(m(x)).numpy() for g, m in models.items()}
            refl = (x * sd + mu).numpy()
            for l, it in enumerate(items):
                y = it['mask']; y = y.numpy() if torch.is_tensor(y) else np.asarray(y); h, w = y.shape
                d_w, d_l, half = pf.edges_and_width(y); dist = np.where(y == 1, d_w, d_l)
                r = refl[l][:, :h, :w]; vis = r[:3].mean(0); ndvi = (r[3] - r[2]) / np.maximum(r[3] + r[2], 1e-6)
                strata = {'all': np.ones_like(y, bool), 'shore<=2px': dist <= 2, 'interior>10px': (dist > 10) & np.isfinite(dist),
                          'thin<=2px': (y == 1) & (half > 0) & (half <= 2), 'dark': (vis < 0.06) & (r[3] < 0.08) & (r[4] < 0.06),
                          'bright': vis > 0.2, 'vegetated': ndvi > 0.4}
                valid = y != -1
                r4 = preds['R4'][l][:h, :w]; r4ok = r4 == (y == 1)
                for g in RUNGS:
                    p = preds[g][l][:h, :w]
                    for sn, sel in strata.items():
                        v = valid & sel; t = y[v] == 1; q = p[v]
                        acc[(g, sn)] += [np.sum(q & t), np.sum(q & ~t), np.sum(~q & t), np.sum(~q & ~t)]
                        if g != 'R4':
                            wrong = valid & sel & r4ok & (p != (y == 1)); comp[(g, sn)] += [wrong.sum(), (valid & r4ok & (p != (y == 1))).sum()]
        rep.append(f'\n=== {name} (seed 64): water recall / false-water rate per stratum; "new errors" = R4 right, rung wrong ===')
        for g in RUNGS:
            line = []
            for sn in ['all', 'shore<=2px', 'interior>10px', 'thin<=2px', 'dark', 'bright', 'vegetated']:
                tp, fp, fn, tn = acc[(g, sn)]
                rec = tp / max(1, tp + fn); far = fp / max(1, fp + tn)
                line.append(f'{sn} R{100*rec:5.1f}/F{100*far:4.1f}')
            tp, fp, fn, tn = acc[(g, 'all')]
            rep.append(f'{g:3s} water IoU {100*tp/max(1,tp+fp+fn):5.1f} | ' + ' | '.join(line))
        for g in RUNGS:
            if g == 'R4': continue
            tot = comp[(g, 'all')][1]
            rep.append(f'  new errors of {g} vs R4: {tot:,} px; share in ' + ', '.join(f'{sn} {100*comp[(g, sn)][0]/max(1,tot):.0f}%' for sn in ['shore<=2px', 'interior>10px', 'thin<=2px', 'dark', 'bright', 'vegetated']))
    (HERE / 'ladder_report.txt').write_text('\n'.join(rep) + '\n'); print('\n'.join(rep))

if __name__ == '__main__':
    main()
