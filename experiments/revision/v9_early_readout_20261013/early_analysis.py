#!/usr/bin/env python3
"""Summarize counts.json from early_readout.py (exposed sets only). Writes report.txt and summary.csv."""
import csv
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'counts.json').read_text())
ARMS = ['E0', 'E1', 'E2', 'E3']
SEEDS = ['42', '64', '91', '107']


def iou(c):
    tp, fp, fn, tn = c
    return tp / (tp + fp + fn) if tp + fp + fn else None


def rec(c):
    tp, fp, fn, tn = c
    return tp / (tp + fn) if tp + fn else None


def far(c):
    tp, fp, fn, tn = c
    return fp / (fp + tn) if fp + tn else None


def agg(events, key):
    return [sum(d[key][i] for d in events.values() if key in d) for i in range(4)]


def eq(events, key):
    v = [iou(d[key]) for d in events.values() if key in d and iou(d[key]) is not None]
    return st.mean(v)


rep, rows = [f'Early readout, exposed sets only; decode-time D uses k={C["k"]}, theta={C["theta"]}. Four seeds per arm.'], []
random.seed(11)
for ds, block in C['sets'].items():
    ev, dry = block['events'], block['dry']
    rep.append(f'\n=== {ds} ({len(ev)} events) ===')
    rep.append(f'{"arm":6s} {"var":8s} {"eq water IoU":>14s} {"pooled":>8s} {"P":>6s} {"R":>6s} | {"edge1-2 R":>9s} {"edge FAR":>8s} {"thin R":>7s} {"interior R":>10s} | {"dry alarms":>10s}')
    base = {}
    for arm in ARMS:
        for var in ('deployed', 'D'):
            per_seed = []
            for s in SEEDS:
                mid = f'{arm}_s{s}'; k = f'{mid}|{var}|'
                pc = agg(ev, k + 'all')
                per_seed.append(dict(eq=eq(ev, k + 'all'), pooled=iou(pc), P=pc[0] / max(1, pc[0] + pc[1]), R=rec(pc),
                                     edgeR=rec(agg(ev, k + 'edge1-2')), edgeF=far(agg(ev, k + 'edge1-2')),
                                     thinR=rec(agg(ev, k + 'thin<=2')), intR=rec(agg(ev, k + 'interior>5')),
                                     dry=dry.get(f'{mid}|{var}', [0, 0])))
            m = lambda f: st.mean(x[f] for x in per_seed if x[f] is not None)
            sd = st.stdev([x['eq'] for x in per_seed])
            dr = (sum(x['dry'][1] for x in per_seed) / 4, per_seed[0]['dry'][0])
            line = (f'{arm:6s} {var:8s} {100 * m("eq"):7.2f} ±{100 * sd:4.2f} {100 * m("pooled"):8.2f} {100 * m("P"):6.1f} {100 * m("R"):6.1f} | '
                    f'{100 * m("edgeR"):9.1f} {100 * m("edgeF"):8.1f} {100 * m("thinR"):7.1f} {100 * m("intR"):10.1f} | {dr[0]:5.1f}/{dr[1]}')
            if arm == 'E0':
                base[var] = per_seed
            else:
                d = [per_seed[i]['eq'] - base[var][i]['eq'] for i in range(4)]
                # event-level paired difference of seed means
                evd = []
                for e in ev:
                    a = [iou(ev[e][f'{arm}_s{s}|{var}|all']) for s in SEEDS]; b = [iou(ev[e][f'E0_s{s}|{var}|all']) for s in SEEDS]
                    if None not in a and None not in b:
                        evd.append(st.mean(a) - st.mean(b))
                boot = sorted(st.mean(random.choice(evd) for _ in evd) for _ in range(4000))
                line += (f' | vs E0 {100 * st.mean(d):+.2f}±{100 * st.stdev(d):.2f}, seeds+ {sum(x > 0 for x in d)}/4, '
                         f'events +{sum(x > 5e-4 for x in evd)}/-{sum(x < -5e-4 for x in evd)}, boot [{100 * boot[100]:+.2f},{100 * boot[3899]:+.2f}]')
            rep.append(line)
            rows.append(dict(dataset=ds, arm=arm, variant=var, eq_water_IoU=round(100 * m('eq'), 3), seed_sd=round(100 * sd, 3),
                             pooled_water_IoU=round(100 * m('pooled'), 3), precision=round(100 * m('P'), 2), recall=round(100 * m('R'), 2),
                             edge12_recall=round(100 * m('edgeR'), 2), edge12_false_water=round(100 * m('edgeF'), 2),
                             thin_recall=round(100 * m('thinR'), 2), interior_recall=round(100 * m('intR'), 2), dry_alarms_mean=dr[0], dry_windows=dr[1]))
(HERE / 'report.txt').write_text('\n'.join(rep) + '\n')
with open(HERE / 'summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('\n'.join(rep))
