#!/usr/bin/env python3
"""One-number event calibration: how much of the per-event threshold headroom can one
labelled chip (or a few labelled clicks) recover?  Read-only on probe_hist/.

For every event with at least two eligible images and every eligible support
image j (reference water and land each >= 0.5 % of its valid pixels):
  dense  tau_j = threshold maximizing water IoU on image j (ties -> nearest 0)
  clicks tau_j = threshold maximizing balanced accuracy on k water + k land
                 pixels drawn from image j (k = 10, 20, 50; 20 draws each)
tau_j is clamped to [-4, 2] and applied to the remaining images of the event
(the query). Query water IoU is compared with the deployed tau = 0 on exactly
the same query pixels. Event value = mean over support choices; the event is
the unit. 'oracle' = best single tau for the query itself (unattainable).
This is the 1-parameter analogue of the V8 one-chip head adaptation, which
refit 17 head coordinates and lost 0.35-1.77 pp.
"""
from __future__ import annotations

import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / 'probe_hist'


def main():
    rep = []
    rng = random.Random(1234)
    for name in ['s11val', 's11', 'sturm', 'wf']:
        z = np.load(D / f'{name}.npz'); H = z['H'].astype(np.int64); seeds = [int(s) for s in z['seeds']]; edges = z['edges']
        meta = json.loads((D / f'{name}_meta.json').read_text())
        lo_b = int(np.searchsorted(edges, -4.0)) + 1; hi_b = int(np.searchsorted(edges, 2.0)) + 1; b0 = int(np.searchsorted(edges, 0.0)) + 1
        HC = H.sum(axis=3)                                   # (n, seeds, 50, 3)
        by_ev = defaultdict(list)
        for i, m in enumerate(meta):
            by_ev[m['event']].append(i)
        res = {key: [] for key in ('dense', 'k10', 'k20', 'k50', 'oracle')}
        detail = []
        for e, idx in sorted(by_ev.items()):
            elig = []
            for i in idx:
                hc = HC[i, 0]; valid = hc[:, 1:].sum()
                if valid and hc[:, 2].sum() >= 0.005 * valid and hc[:, 1].sum() >= 0.005 * valid:
                    elig.append(i)
            if len(elig) < 1 or len(idx) < 2:
                continue
            per_key = {key: [] for key in res}
            for k_s in range(len(seeds)):
                for j in elig:
                    query = [i for i in idx if i != j]
                    qc = HC[query, k_s].sum(axis=0)          # (50, 3)
                    if qc[:, 2].sum() == 0:
                        continue
                    iou = lambda c, b: c[b:, 2].sum() / max(1, c[b:, 2].sum() + c[b:, 1].sum() + c[:b, 2].sum())
                    base = iou(qc, b0)
                    sc = HC[j, k_s]
                    cand = list(range(lo_b, hi_b + 1))
                    dense_b = max(cand, key=lambda b: (round(iou(sc, b), 9), -abs(b - b0)))
                    per_key['dense'].append(iou(qc, dense_b) - base)
                    per_key['oracle'].append(max(iou(qc, b) for b in range(1, 50)) - base)
                    wpmf = sc[:, 2] / sc[:, 2].sum(); lpmf = sc[:, 1] / sc[:, 1].sum()
                    for k in (10, 20, 50):
                        gains = []
                        for _ in range(20):
                            wb = np.bincount(np.searchsorted(np.cumsum(wpmf), [rng.random() for _ in range(k)]), minlength=50)[:50]
                            lb = np.bincount(np.searchsorted(np.cumsum(lpmf), [rng.random() for _ in range(k)]), minlength=50)[:50]
                            bb = max(cand, key=lambda b: (wb[b:].sum() + lb[:b].sum(), -abs(b - b0)))
                            gains.append(iou(qc, bb) - base)
                        per_key[f'k{k}'].append(st.mean(gains))
            for key in res:
                if per_key[key]:
                    res[key].append((e, st.mean(per_key[key]), min(per_key[key]), sum(g > 0 for g in per_key[key]) / len(per_key[key])))
        rep.append(f'\n=== {name}: events with a usable support chip: {len(res["dense"])} ===')
        for key, rows in res.items():
            if not rows:
                continue
            means = [r[1] for r in rows]
            boot = sorted(st.mean(rng.choice(means) for _ in means) for _ in range(4000))
            rep.append(f'  {key:6s}: equal-event mean query water-IoU change {100 * st.mean(means):+6.2f} pp '
                       f'[{100 * boot[100]:+.2f}, {100 * boot[3899]:+.2f}]; events improved {sum(m > 0 for m in means)}/{len(means)}; '
                       f'worst support-choice outcome {100 * min(r[2] for r in rows):+.1f} pp; share of support choices that help {st.mean(r[3] for r in rows):.2f}')
        detail_rows = {r[0]: r for r in res['dense']}
        if name in ('sturm', 'wf'):
            rep.append('  per event (dense one-chip): ' + ', '.join(f'{e} {100 * v[1]:+.1f}' for e, v in detail_rows.items()))
    (HERE / 'fewshot_report.txt').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
