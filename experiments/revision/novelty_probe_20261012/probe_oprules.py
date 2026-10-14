#!/usr/bin/env python3
"""Label-free, per-scene operating-point rules for the R4 student, scored from probe_hist/.

Read-only. A rule maps the label-free histogram of one scene (all imaged pixels:
student margin x MNDWI) to one decision threshold tau on the student margin;
prediction is water iff margin >= tau (tau = 0 is the deployed student).
'Scene' = one image for S11 (each chip is its own acquisition) and one flood map
(acquisition/AOI) for STURM and WorldFloods. Scores use reference-valid pixels
only and are aggregated per event (equal-event means) as in the V8 tables.

Rules (all hyperparameters chosen on S11 validation only, then frozen):
  fixed@tau     one global threshold (tau = 0 is the deployed student)
  otsu          Otsu split of the scene's margin histogram, used only when the
                between-class/total variance ratio >= eta; clamped to [lo, 0]
  anchor        threshold maximizing balanced agreement with MNDWI anchors
                (MNDWI >= a as water, MNDWI <= -a as land) within [lo, 0]
  em            Saerens EM prior correction of temperature-calibrated
                probabilities, prior from S11 validation; tau clamped to [lo, hi]
  oracle_event  best tau per event using reference labels (unattainable bound)

Writes oprules_summary.csv, oprules_events.csv, oprules_report.txt.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / 'probe_hist'
SETS = ['s11val', 's11', 'sturm', 'wf']


def load(name):
    z = np.load(D / f'{name}.npz')
    meta = json.loads((D / f'{name}_meta.json').read_text())
    return z['H'].astype(np.int64), [int(s) for s in z['seeds']], z['edges'], meta


EDGES = None


def tau_of(b):
    """bin index b (1..49) -> threshold value; prediction water iff bin >= b."""
    return float(EDGES[b - 1])


def b_of(tau):
    return int(np.searchsorted(EDGES, tau, side='left')) + 1


def scene_groups(name, meta):
    g = defaultdict(list)
    for i, m in enumerate(meta):
        g[m['name'] if name.startswith('s11') else m['map']].append(i)
    return g


def counts_at(hc, b):
    """hc: (50, 3) margin x class counts; returns tp, fp, fn, tn at bin threshold b."""
    tp = hc[b:, 2].sum(); fp = hc[b:, 1].sum(); fn = hc[:b, 2].sum(); tn = hc[:b, 1].sum()
    return np.array([tp, fp, fn, tn], dtype=np.int64)


def otsu(hist, centers):
    w = hist.astype(np.float64); tot = w.sum()
    if tot <= 0:
        return None, 0.0
    mu_t = (w * centers).sum() / tot
    var_t = (w * (centers - mu_t) ** 2).sum() / tot
    best, bb = -1.0, None
    c0 = 0.0; s0 = 0.0
    for b in range(1, len(w)):
        c0 += w[b - 1]; s0 += w[b - 1] * centers[b - 1]
        c1 = tot - c0
        if c0 <= 0 or c1 <= 0:
            continue
        m0 = s0 / c0; m1 = (mu_t * tot - s0) / c1
        vb = c0 * c1 * (m0 - m1) ** 2 / tot ** 2
        if vb > best:
            best, bb = vb, b
    return bb, (best / var_t if var_t > 0 else 0.0)


def rule_tau(rule, params, free, centers, src):
    """free: (50, 21) label-free counts of one scene; returns bin threshold index."""
    b0 = b_of(0.0)
    if rule == 'fixed':
        return b_of(params['tau'])
    marg = free.sum(axis=1)
    lo = b_of(params.get('lo', -4.0))
    if rule == 'otsu':
        bb, eta = otsu(marg, centers)
        if bb is None or eta < params['eta']:
            return b0
        return int(min(max(bb, lo), b0))
    if rule == 'anchor':
        a = params['a']
        mcent = -1 + 0.1 * np.arange(20) + 0.05
        wcols = [k for k in range(20) if mcent[k] >= a]
        lcols = [k for k in range(20) if mcent[k] <= -a]
        aw = free[:, wcols].sum(axis=1); al = free[:, lcols].sum(axis=1)
        if aw.sum() < params['min_anchor'] or al.sum() < params['min_anchor']:
            return b0
        best, bb = -1, b0
        for b in range(lo, b0 + 1):
            tpr = aw[b:].sum() / aw.sum(); tnr = al[:b].sum() / al.sum()
            score = tpr + tnr
            if score > best + 1e-12:
                best, bb = score, b
        return bb
    if rule == 'em':
        T, prior = src['T'], src['prior']
        p = 1 / (1 + np.exp(-centers / T))
        w = marg.astype(np.float64)
        if w.sum() <= 0:
            return b0
        pi = prior
        for _ in range(100):
            a1 = p * pi / prior; a0 = (1 - p) * (1 - pi) / (1 - prior)
            post = a1 / (a1 + a0)
            new = float((w * post).sum() / w.sum())
            new = min(max(new, 1e-4), 1 - 1e-4)
            if abs(new - pi) < 1e-7:
                pi = new; break
            pi = new
        lo_t, hi_t = params.get('lo', -4.0), params.get('hi', 4.0)
        tau = T * (math.log(prior / (1 - prior)) - math.log(pi / (1 - pi)))
        return b_of(min(max(tau, lo_t), hi_t))
    raise ValueError(rule)


def evaluate(name, H, seeds, meta, rule, params, src):
    centers = np.concatenate([[EDGES[0] - 0.25], (EDGES[:-1] + EDGES[1:]) / 2, [EDGES[-1] + 0.25]])
    groups = scene_groups(name, meta)
    ev_counts = {s: defaultdict(lambda: np.zeros(4, dtype=np.int64)) for s in seeds}
    dry = {s: [0, 0] for s in seeds}          # reference-dry images: count, flagged (>=1% predicted water)
    taus = {s: [] for s in seeds}
    for k, s in enumerate(seeds):
        for scene, idx in groups.items():
            free = H[idx, k].sum(axis=(0, 3))             # (50, 21) all imaged pixels
            b = rule_tau(rule, params, free, centers, src.get(s, {}))
            taus[s].append(tau_of(b))
            for i in idx:
                hc = H[i, k].sum(axis=1)                  # (50, 3)
                c = counts_at(hc, b)
                ev_counts[s][meta[i]['event']] += c
                valid = hc[:, 1:].sum()
                if valid > 0 and hc[:, 2].sum() == 0:
                    dry[s][0] += 1
                    dry[s][1] += int(c[1] >= 0.01 * valid)
    return ev_counts, dry, taus


def metr(c):
    tp, fp, fn, tn = [int(x) for x in c]
    w = tp / (tp + fp + fn) if tp + fp + fn else None
    l = tn / (tn + fp + fn) if tn + fp + fn else None
    dd = [x for x in (w, l) if x is not None]
    return dict(w=w, m=sum(dd) / len(dd) if dd else None, p=tp / (tp + fp) if tp + fp else None, r=tp / (tp + fn) if tp + fn else None)


def summ(ev_counts, seeds):
    out = {}
    for s in seeds:
        per = {e: metr(c) for e, c in ev_counts[s].items()}
        pooled = metr(sum(ev_counts[s].values()))
        mean = lambda key: st.mean([v[key] for v in per.values() if v[key] is not None])
        out[s] = dict(eqW=mean('w'), eqM=mean('m'), eqP=mean('p'), eqR=mean('r'), pW=pooled['w'], pP=pooled['p'], pR=pooled['r'], per=per)
    return out


def oracle_event(name, H, seeds, meta):
    ev_counts = {s: {} for s in seeds}
    for k, s in enumerate(seeds):
        by_ev = defaultdict(list)
        for i, m in enumerate(meta):
            by_ev[m['event']].append(i)
        for e, idx in by_ev.items():
            hc = H[idx, k].sum(axis=(0, 2))
            best = None
            for b in range(1, 50):
                c = counts_at(hc, b); w = metr(c)['w']
                if w is not None and (best is None or w > best[0]):
                    best = (w, c)
            ev_counts[s][e] = best[1] if best else counts_at(hc, b_of(0.0))
    return ev_counts


def source_calibration(H, seeds):
    """Temperature and water prior per seed from S11 validation (labels used: source only)."""
    centers = np.concatenate([[EDGES[0] - 0.25], (EDGES[:-1] + EDGES[1:]) / 2, [EDGES[-1] + 0.25]])
    src = {}
    for k, s in enumerate(seeds):
        hc = H[:, k].sum(axis=(0, 2))           # (50, 3)
        land, water = hc[:, 1].astype(float), hc[:, 2].astype(float)
        best = None
        for T in np.arange(0.5, 6.01, 0.25):
            p = 1 / (1 + np.exp(-centers / T)); p = np.clip(p, 1e-6, 1 - 1e-6)
            nll = -(water * np.log(p)).sum() - (land * np.log(1 - p)).sum()
            if best is None or nll < best[0]:
                best = (nll, float(T))
        src[s] = dict(T=best[1], prior=float(water.sum() / (water.sum() + land.sum())))
    return src


def main():
    global EDGES
    data = {}
    for n in SETS:
        H, seeds, edges, meta = load(n)
        EDGES = edges
        data[n] = (H, seeds, meta)
    src = source_calibration(data['s11val'][0], data['s11val'][1])
    rep = [f'Source calibration (S11 validation): {json.dumps(src)}']
    grids = {
        'fixed': [dict(tau=t) for t in (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)],
        'otsu': [dict(eta=e, lo=lo) for e in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9) for lo in (-4.0, -2.0)],
        'anchor': [dict(a=a, lo=lo, min_anchor=m) for a in (0.2, 0.3, 0.4, 0.5) for lo in (-4.0, -2.0) for m in (100, 1000)],
        'em': [dict(lo=lo, hi=hi) for lo in (-4.0, -2.0, -1.0) for hi in (0.0, 2.0, 4.0)],
    }
    rows, ev_rows = [], []
    results = {}
    for rule, grid in grids.items():
        for params in grid:
            for n in SETS:
                H, seeds, meta = data[n]
                evc, dry, taus = evaluate(n, H, seeds, meta, rule, params, src)
                sm = summ(evc, seeds)
                results[(rule, json.dumps(params, sort_keys=True), n)] = (sm, dry, taus)
                for s in seeds:
                    rows.append(dict(rule=rule, params=json.dumps(params, sort_keys=True), dataset=n, seed=s,
                                     **{k: round(100 * v, 4) for k, v in sm[s].items() if k != 'per'},
                                     dry_images=dry[s][0], dry_flagged=dry[s][1],
                                     tau_mean=round(st.mean(taus[s]), 3), tau_min=min(taus[s]), tau_max=max(taus[s])))
    for n in SETS:
        H, seeds, meta = data[n]
        sm = summ(oracle_event(n, H, seeds, meta), seeds)
        results[('oracle_event', '{}', n)] = (sm, None, None)
        for s in seeds:
            rows.append(dict(rule='oracle_event', params='{}', dataset=n, seed=s,
                             **{k: round(100 * v, 4) for k, v in sm[s].items() if k != 'per'}))
    with open(HERE / 'oprules_summary.csv', 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    seeds = data['s11val'][1]
    mean_over = lambda sm, key: st.mean(sm[s][key] for s in seeds)
    base = {n: results[('fixed', json.dumps(dict(tau=0.0), sort_keys=True), n)][0] for n in SETS}
    # source-only selection: highest S11-validation eq water IoU within each family
    chosen = {}
    for rule, grid in grids.items():
        cand = [(mean_over(results[(rule, json.dumps(p, sort_keys=True), 's11val')][0], 'eqW'), json.dumps(p, sort_keys=True)) for p in grid]
        cand.sort(reverse=True)
        chosen[rule] = cand[0][1]
        rep.append(f'{rule:7s} S11-val best {cand[0][1]}: eq water IoU {100 * cand[0][0]:.2f} (deployed {100 * mean_over(base["s11val"], "eqW"):.2f})')
    random.seed(7)
    for n in SETS:
        rep.append(f'\n=== {n} ===')
        b = base[n]
        rep.append(f'  deployed tau=0     : eqW {100 * mean_over(b, "eqW"):6.2f} eqM {100 * mean_over(b, "eqM"):6.2f} P {100 * mean_over(b, "eqP"):5.1f} R {100 * mean_over(b, "eqR"):5.1f} | pooled W {100 * mean_over(b, "pW"):6.2f}')
        ora = results[('oracle_event', '{}', n)][0]
        rep.append(f'  oracle per event   : eqW {100 * mean_over(ora, "eqW"):6.2f} (unattainable; labels choose tau)')
        for rule in list(grids) :
            key = (rule, chosen[rule], n)
            sm, dry, taus = results[key]
            d = [sm[s]['eqW'] - b[s]['eqW'] for s in seeds]
            dm = [sm[s]['eqM'] - b[s]['eqM'] for s in seeds]
            dp = [sm[s]['pW'] - b[s]['pW'] for s in seeds]
            events = sorted(b[seeds[0]]['per'])
            evd = []
            for e in events:
                x = [sm[s]['per'][e]['w'] for s in seeds]; y = [b[s]['per'][e]['w'] for s in seeds]
                if None in x or None in y:
                    continue
                evd.append(st.mean(x) - st.mean(y))
                ev_rows.append(dict(dataset=n, rule=rule, params=chosen[rule], event=e, deployed=round(100 * st.mean(y), 3),
                                    rule_value=round(100 * st.mean(x), 3), diff=round(100 * (st.mean(x) - st.mean(y)), 3)))
            boot = sorted(st.mean(random.choice(evd) for _ in evd) for _ in range(4000))
            dryf = f'dry tiles flagged {sum(dry[s][1] for s in seeds) / 4:.1f}/{dry[seeds[0]][0]} (deployed {sum(results[("fixed", json.dumps(dict(tau=0.0), sort_keys=True), n)][1][s][1] for s in seeds) / 4:.1f})'
            rep.append(f'  {rule:7s} {chosen[rule]:42s}: eqW {100 * mean_over(sm, "eqW"):6.2f} ({100 * st.mean(d):+.2f}±{100 * st.stdev(d):.2f}) '
                       f'eqM {100 * st.mean(dm):+.2f} P {100 * mean_over(sm, "eqP"):5.1f} R {100 * mean_over(sm, "eqR"):5.1f} pooled W {100 * st.mean(dp):+.2f}; '
                       f'events +{sum(x > 5e-4 for x in evd)}/-{sum(x < -5e-4 for x in evd)} of {len(evd)}, boot [{100 * boot[100]:+.2f},{100 * boot[3899]:+.2f}]; '
                       f'tau mean {st.mean(st.mean(taus[s]) for s in seeds):+.2f}; {dryf}')
    with open(HERE / 'oprules_events.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ev_rows[0])); w.writeheader(); w.writerows(ev_rows)
    (HERE / 'oprules_report.txt').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
