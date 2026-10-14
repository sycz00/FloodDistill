#!/usr/bin/env python3
"""Score the decoding rules recorded by probe_cpu.py; select settings on S11 validation only.

Read-only on probe_counts/. Writes probe_summary.csv (every rule x setting x
dataset x seed), probe_selected.csv (source-selected settings on the external
sets, paired against the plain student), probe_events.csv (per-event rows) and
probe_report.txt.

Selection rule, frozen before reading external rows: within each rule family,
take the setting with the highest mean (over the four seeds) of equal-event
water IoU on S11 validation (10 events, human labels). External sets (S11
test, STURM, WorldFloods) are research-exposed readouts and never select.
Equal-event means omit events whose water union is empty (undefined IoU).
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = ['42', '64', '91', '107']
SETS = ['s11val', 's11', 'sturm', 'wf']
FAMILIES = {'S@tau': 'S@', 'SuM': 'SuM@', 'SvM': 'SvM@', 'SgM': 'SgM@'}


def load(name, scope):
    rows = json.load(gzip.open(HERE / 'probe_counts' / f'{name}.json.gz', 'rt'))
    if scope == 'overlap_excluded':
        rows = [r for r in rows if not r['omit']]
    ev = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for r in rows:
        for key, c in r['counts'].items():
            acc = ev[r['event']][key]
            for i in range(4):
                acc[i] += c[i]
    return ev


def metrics(c):
    tp, fp, fn, tn = c
    w = tp / (tp + fp + fn) if tp + fp + fn else None
    l = tn / (tn + fp + fn) if tn + fp + fn else None
    d = [x for x in (w, l) if x is not None]
    return dict(water_IoU=w, land_IoU=l, mIoU=sum(d) / len(d) if d else None,
                precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(ev, key):
    per = {e: metrics(v[key]) for e, v in ev.items() if key in v}
    pooled = [0, 0, 0, 0]
    for e, v in ev.items():
        for i in range(4):
            pooled[i] += v[key][i]
    pm = metrics(pooled)
    return dict(eq_water_IoU=mean(p['water_IoU'] for p in per.values()), eq_mIoU=mean(p['mIoU'] for p in per.values()),
                eq_precision=mean(p['precision'] for p in per.values()), eq_recall=mean(p['recall'] for p in per.values()),
                pooled_water_IoU=pm['water_IoU'], pooled_mIoU=pm['mIoU'], pooled_precision=pm['precision'],
                pooled_recall=pm['recall'], events=per)


def keys_for(seed, family):
    prefix = f's{seed}|{FAMILIES[family]}'
    return prefix


def main():
    out = []
    data = {(n, sc): load(n, sc) for n in SETS for sc in ('full', 'overlap_excluded')}
    all_keys = sorted(next(iter(data[('s11', 'full')].values())).keys())
    rule_names = sorted({k.split('|', 1)[1] for k in all_keys if '|' in k})
    # 1. Every rule/setting/dataset/seed
    res = {}
    for (n, sc), ev in data.items():
        keys = sorted({k for v in ev.values() for k in v})
        for k in keys:
            s = summarize(ev, k)
            res[(n, sc, k)] = s
            seed, rule = (k.split('|', 1) if '|' in k else ('', k))
            out.append(dict(dataset=n, scope=sc, seed=seed.lstrip('s'), rule=rule,
                            **{m: (round(100 * v, 4) if v is not None else '') for m, v in s.items() if m != 'events'}))
    with open(HERE / 'probe_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    # 2. Source-only selection on S11 validation (full scope; S11 has no overlap exclusions)
    def family_settings(family):
        return sorted({r for r in rule_names if r.startswith(FAMILIES[family]) and not r.endswith('+v')} |
                      ({r for r in rule_names if r.startswith('SgM@') and r.endswith('+v')} if family == 'SgM' else set()))
    selected = {}
    lines = []
    for family in FAMILIES:
        best = None
        for r in family_settings(family):
            score = st.mean(res[('s11val', 'full', f's{s}|{r}')]['eq_water_IoU'] for s in SEEDS)
            base = st.mean(res[('s11val', 'full', f's{s}|S')]['eq_water_IoU'] for s in SEEDS)
            lines.append(f'  source {family:6s} {r:14s} S11-val eq water IoU {100 * score:6.2f} (student {100 * base:6.2f}, {100 * (score - base):+5.2f})')
            if best is None or score > best[1]:
                best = (r, score)
        selected[family] = best[0]
    overall = max(selected.values(), key=lambda r: st.mean(res[('s11val', 'full', f's{s}|{r}')]['eq_water_IoU'] for s in SEEDS))

    # 3. External readouts of the selected settings, paired against the plain student
    sel_rows, ev_rows = [], []
    rep = ['Source-only selection on S11 validation (mean over 4 seeds of equal-event water IoU):'] + lines
    rep.append(f'Selected per family: {selected}; overall source-best rule: {overall}')
    random.seed(20261012)
    for n in SETS:
        for sc in (('full', 'overlap_excluded') if n in ('sturm', 'wf') else ('full',)):
            rep.append(f'\n=== {n} ({sc}) ===')
            base = {s: res[(n, sc, f's{s}|S')] for s in SEEDS}
            rep.append('  student S      : ' + fmt([base[s] for s in SEEDS]))
            for ref in ('M@0', 'T'):
                if (n, sc, ref) in res:
                    rep.append(f'  {ref:15s}: ' + fmt([res[(n, sc, ref)]]))
            for th in ('TgM@0.0', 'TgM@0.1', 'TgM@0.2'):
                if (n, sc, th) in res:
                    rep.append(f'  {th:15s}: ' + fmt([res[(n, sc, th)]]) + '  (teacher seeds, diagnostic)')
            for family, r in list(selected.items()):
                cur = {s: res[(n, sc, f's{s}|{r}')] for s in SEEDS}
                d = {m: [cur[s][m] - base[s][m] for s in SEEDS] for m in ('eq_water_IoU', 'eq_mIoU', 'eq_precision', 'eq_recall',
                                                                          'pooled_water_IoU', 'pooled_mIoU', 'pooled_precision', 'pooled_recall')}
                events = sorted(base['42']['events'])
                ev_d = []
                for e in events:
                    a = [cur[s]['events'][e]['water_IoU'] for s in SEEDS]
                    b = [base[s]['events'][e]['water_IoU'] for s in SEEDS]
                    if None in a or None in b:
                        continue
                    ev_d.append((e, st.mean(a) - st.mean(b)))
                    ev_rows.append(dict(dataset=n, scope=sc, family=family, rule=r, event=e,
                                        student_water_IoU=round(100 * st.mean(b), 3), rule_water_IoU=round(100 * st.mean(a), 3),
                                        difference_pp=round(100 * (st.mean(a) - st.mean(b)), 3),
                                        student_recall=round(100 * st.mean(base[s]['events'][e]['recall'] or 0 for s in SEEDS), 3),
                                        rule_recall=round(100 * st.mean(cur[s]['events'][e]['recall'] or 0 for s in SEEDS), 3),
                                        student_precision=round(100 * st.mean(base[s]['events'][e]['precision'] or 0 for s in SEEDS), 3),
                                        rule_precision=round(100 * st.mean(cur[s]['events'][e]['precision'] or 0 for s in SEEDS), 3)))
                diffs = [x for _, x in ev_d]
                boot = []
                for _ in range(4000):
                    sample = [random.choice(diffs) for _ in diffs]
                    boot.append(sum(sample) / len(sample))
                boot.sort()
                lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
                wins = sum(x > 0.0005 for x in diffs); losses = sum(x < -0.0005 for x in diffs)
                rep.append(f'  {family:6s} {r:12s}: ' + fmt(list(cur.values())))
                rep.append(f'      paired vs S (pp, mean±SD over seeds): eq water IoU {sd(d["eq_water_IoU"])}, eq mIoU {sd(d["eq_mIoU"])}, '
                           f'eq P {sd(d["eq_precision"])}, eq R {sd(d["eq_recall"])} | pooled water IoU {sd(d["pooled_water_IoU"])}, '
                           f'pooled P {sd(d["pooled_precision"])}, pooled R {sd(d["pooled_recall"])}')
                rep.append(f'      events (seed-mean water IoU): {wins} better, {losses} worse of {len(diffs)}; '
                           f'event-bootstrap 95% interval of the eq-event difference [{100 * lo:+.2f}, {100 * hi:+.2f}] pp')
                sel_rows.append(dict(dataset=n, scope=sc, family=family, rule=r,
                                     **{f'd_{m}_mean_pp': round(100 * st.mean(v), 3) for m, v in d.items()},
                                     **{f'd_{m}_sd_pp': round(100 * st.stdev(v), 3) for m, v in d.items()},
                                     events_better=wins, events_worse=losses, events=len(diffs),
                                     event_boot_lo_pp=round(100 * lo, 3), event_boot_hi_pp=round(100 * hi, 3)))
    for name, rows in (('probe_selected.csv', sel_rows), ('probe_events.csv', ev_rows)):
        with open(HERE / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (HERE / 'probe_report.txt').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


def fmt(items):
    g = lambda m: 100 * st.mean(x[m] for x in items)
    return (f'eq water IoU {g("eq_water_IoU"):6.2f} | eq mIoU {g("eq_mIoU"):6.2f} | eq P {g("eq_precision"):5.1f} R {g("eq_recall"):5.1f}'
            f' | pooled water IoU {g("pooled_water_IoU"):6.2f} P {g("pooled_precision"):5.1f} R {g("pooled_recall"):5.1f}')


def sd(v):
    return f'{100 * st.mean(v):+.2f}±{100 * st.stdev(v):.2f}'


if __name__ == '__main__':
    main()
