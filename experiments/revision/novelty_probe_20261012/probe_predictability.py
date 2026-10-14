#!/usr/bin/env python3
"""Exploratory (research-exposed sets): (1) does a label-free scene statistic predict the
event-optimal student threshold?  (2) threshold-free comparison of student and MNDWI:
average precision and precision at matched recall.  Read-only on probe_hist/."""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; D = HERE / 'probe_hist'

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

out = []
feats_all, tau_all, ds_all = [], [], []
for name in ['s11val', 's11', 'sturm', 'wf']:
    z = np.load(D / f'{name}.npz'); H = z['H'].astype(np.int64); edges = z['edges']; seeds = list(z['seeds'])
    meta = json.loads((D / f'{name}_meta.json').read_text())
    b0 = int(np.searchsorted(edges, 0.0)) + 1
    by = defaultdict(list)
    for i, m in enumerate(meta): by[m['event']].append(i)
    rows = []
    for e, idx in by.items():
        Hs = H[idx].sum(axis=0).mean(axis=0)          # seed-averaged (50, 21, 3)
        hc = Hs.sum(axis=1)                            # (50, 3)
        if hc[:, 2].sum() == 0: continue
        iou = [hc[b:, 2].sum() / max(1, hc[b:, 2].sum() + hc[b:, 1].sum() + hc[:b, 2].sum()) for b in range(1, 50)]
        bstar = 1 + int(np.argmax(iou)); tstar = float(edges[bstar - 1])
        free = Hs.sum(axis=2)                          # (50, 21) label-free
        tot = free.sum(); mpos = free[:, 10:20].sum()  # MNDWI > 0 bins 10..19 (bin 20 unavailable)
        predw = free[b0:, :].sum()
        near = free[b0 - 4:b0, :].sum()               # margin in [-2, 0)
        f = dict(disagree=free[:b0, 10:20].sum() / max(1, mpos),   # MNDWI water the student calls land
                 pred_frac=predw / tot, mndwi_frac=mpos / tot, near_frac=near / tot,
                 near_over_pred=near / max(1, predw), mndwi_over_pred=mpos / max(1, predw))
        rows.append((e, tstar, f, iou[bstar - 1] - iou[b0 - 1]))
        feats_all.append(f); tau_all.append(tstar); ds_all.append(name)
    taus = [r[1] for r in rows]
    out.append(f'{name}: {len(rows)} events; event-optimal tau median {st.median(taus):+.2f} (IQR {np.percentile(taus,25):+.2f}..{np.percentile(taus,75):+.2f}); '
               f'oracle gain mean {100*st.mean(r[3] for r in rows):+.2f} pp')
    if len(rows) >= 8:
        for k in rows[0][2]:
            out.append(f'   spearman(tau*, {k:15s}) = {spearman([r[2][k] for r in rows], taus):+.2f}')
ood = [i for i, d in enumerate(ds_all) if d in ('sturm', 'wf')]
out.append(f'OOD pooled (STURM+WF, {len(ood)} events):')
for k in feats_all[0]:
    out.append(f'   spearman(tau*, {k:15s}) = {spearman([feats_all[i][k] for i in ood], [tau_all[i] for i in ood]):+.2f}')

# (2) threshold-free comparison: AP and precision at the student's/MNDWI's recall, pooled pixels, seed-mean
out.append('\nThreshold-free comparison (pooled pixels; student = seed mean of four R4 parents):')
for name in ['s11val', 's11', 'sturm', 'wf']:
    z = np.load(D / f'{name}.npz'); H = z['H'].astype(np.int64)
    S = H.sum(axis=0)                                  # (seeds, 50, 21, 3)
    def curve(counts):                                 # counts: (bins, 3) ordered low->high score
        w = counts[::-1, 2].cumsum(); l = counts[::-1, 1].cumsum()
        prec = w / np.maximum(1, w + l); rec = w / max(1, counts[:, 2].sum())
        ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
        return rec, prec, ap
    aps, curves = [], []
    for k in range(S.shape[0]):
        r, p, ap = curve(S[k].sum(axis=1)); aps.append(ap); curves.append((r, p))
    mcount = S[0].sum(axis=0)[:20]                     # MNDWI bins (available pixels only), class counts
    rm, pm, apm = curve(mcount)
    # precision at matched recall: MNDWI>0 operating point recall
    m_rec = mcount[10:, 2].sum() / mcount[:, 2].sum(); m_prec = mcount[10:, 2].sum() / max(1, mcount[10:, 1:].sum())
    s_prec = st.mean(float(np.interp(m_rec, r, p)) for r, p in curves)
    out.append(f'  {name:6s}: AP student {100*st.mean(aps):5.1f} (±{100*st.stdev(aps):.1f}) vs MNDWI {100*apm:5.1f} | at MNDWI>0 recall {100*m_rec:4.1f}%: '
               f'precision student {100*s_prec:5.1f} vs MNDWI {100*m_prec:5.1f}')
(HERE / 'predictability_report.txt').write_text('\n'.join(out) + '\n'); print('\n'.join(out))
