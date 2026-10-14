#!/usr/bin/env python3
"""Summarize probe_failure.py: error strata and edge-band refinement (S11-val selection).

Read-only on failure_strata.json and refine_events.json; writes failure_report.txt.
"""
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'failure_strata.json').read_text())
R = json.loads((HERE / 'refine_events.json').read_text())
SEEDS = ['s42', 's64', 's91', 's107']
rep = []


def rate(c, kind):
    tp, fp, fn, tn = c
    if kind == 'recall':
        return tp / (tp + fn) if tp + fn else None
    return fp / (fp + tn) if fp + tn else None


def stud(d, stratum):
    vals = [d[stratum][s] for s in SEEDS if s in d[stratum]]
    return [sum(v[i] for v in vals) for i in range(4)] if vals else None


for ds in ['s11val', 's11', 'sturm', 'wf']:
    d = S[ds]
    rep.append(f'\n=== {ds}: error strata (student = four R4 seeds pooled) ===')
    rep.append(f'{"stratum":18s} {"water px":>11s} | recall T / S / MNDWI        | {"land px":>11s} | false-water rate T / S / MNDWI')
    tot_fn = {k: sum(d[s][k][2] for s in d if s.startswith('edge') and k in d[s]) for k in ('T', 'M')}
    tot_fn['S'] = sum(stud(d, s)[2] for s in d if s.startswith('edge'))
    for s in sorted(d, key=lambda x: (x.split('-')[0], x)):
        sc = stud(d, s); t = d[s].get('T'); m = d[s].get('M')
        wp = (sc[0] + sc[2]) // 4; lp = (sc[1] + sc[3]) // 4
        fmt = lambda x: f'{100 * x:5.1f}' if x is not None else '  n/a'
        rec = ' / '.join(fmt(rate(c, 'recall')) if c else '  n/a' for c in (t, sc, m))
        far = ' / '.join(fmt(rate(c, 'far')) if c else '  n/a' for c in (t, sc, m))
        rep.append(f'{s:18s} {wp:11,d} | {rec:27s} | {lp:11,d} | {far}')
    rep.append('share of all missed water lying within 1-2 px of the true edge: ' + ', '.join(
        f'{k} {100 * sum((stud(d, s) if k == "S" else d[s][k])[2] for s in ("edge1-1", "edge2-2") if s in d) / max(1, tot_fn[k]):.1f}%'
        for k in ('T', 'S', 'M') if (k == 'S' or k in d['edge1-1'])))


def eq(ev, key):
    vals = []
    for e, d in ev.items():
        if key not in d:
            continue
        tp, fp, fn, tn = d[key]
        if tp + fp + fn:
            vals.append(tp / (tp + fp + fn))
    return st.mean(vals) if vals else None


def pooled(ev, key):
    c = [sum(d[key][i] for d in ev.values() if key in d) for i in range(4)]
    tp, fp, fn, tn = c
    return tp / (tp + fp + fn), tp / max(1, tp + fp), tp / max(1, tp + fn)


variants = sorted({k.split('|', 1)[1] for k in R['s11val']['Ghana'] if not k.endswith('|base')})
score = {v: st.mean(eq(R['s11val'], f'{s}|{v}') - eq(R['s11val'], f'{s}|base') for s in SEEDS) for v in variants}
best = max(score, key=score.get)
rep.append('\n=== Edge-band refinement, selection on S11 validation (students only; teacher not cached there) ===')
for v in sorted(score, key=score.get, reverse=True)[:8]:
    rep.append(f'  {v:14s} S11-val equal-event water IoU change {100 * score[v]:+.2f} pp')
rep.append(f'  selected: {best}')
random.seed(3)
for ds in ['s11', 'sturm', 'wf']:
    ev = R[ds]
    rep.append(f'\n--- {ds} ---')
    for who in ['T'] + SEEDS:
        for v in [best, 'k2|local', 'k3|local', 'k1|0.0', 'k2|0.0']:
            b, r = f'{who}|base', f'{who}|{v}'
            de = eq(ev, r) - eq(ev, b)
            pb, pr = pooled(ev, b), pooled(ev, r)
            diffs = []
            for e, d in ev.items():
                if b in d and r in d:
                    x = d[b]; y = d[r]
                    ib = x[0] / max(1, x[0] + x[1] + x[2]); ir = y[0] / max(1, y[0] + y[1] + y[2])
                    if x[0] + x[1] + x[2]:
                        diffs.append(ir - ib)
            up = sum(x > 5e-4 for x in diffs); dn = sum(x < -5e-4 for x in diffs)
            rep.append(f'  {who:5s} {v:10s}: equal-event water IoU {100 * eq(ev, b):5.2f} -> {100 * eq(ev, r):5.2f} ({100 * de:+.2f}); '
                       f'pooled {100 * pb[0]:5.2f} -> {100 * pr[0]:5.2f} (P {100 * pb[1]:4.1f}->{100 * pr[1]:4.1f}, R {100 * pb[2]:4.1f}->{100 * pr[2]:4.1f}); events +{up}/-{dn}')
        if who == 'T':
            rep.append('')
(HERE / 'failure_report.txt').write_text('\n'.join(rep) + '\n')
print('\n'.join(rep))
