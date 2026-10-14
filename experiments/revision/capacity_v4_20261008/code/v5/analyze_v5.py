"""Collect v5 results (CPU): S11-val selection scores, external readouts, paired-seed and event-bootstrap statistics.

Statistics per protocol_v5_addendum.yaml: seed mean +- sample SD; paired-seed Student-t 95% CI on per-seed
differences; event bootstrap (10,000 resamples, seed 1729) of per-event water-IoU differences, model
seed-averaged within event, conditional on fixed checkpoints. External numbers are secondary readouts.
Usage: analyze_v5.py <v4 dir> -> writes <v4>/v5/analysis/*.csv and summary.json, prints markdown.
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

V4 = Path(sys.argv[1]).resolve()
V5 = V4 / 'v5'
OUT = V5 / 'analysis'; OUT.mkdir(exist_ok=True)
SEEDS = [42, 64, 91, 107]
COMP = V4.parent / 'prompt5_closure' / 'sensitivity_complete' / 'confusion_counts.csv'
POOLED = V4.parent / 'prompt5_closure' / 'tables' / 'current_full_pooled_counts.csv'
PROTO = {'s11': 'S11_TEST512_LEGACY', 'sturm': 'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1', 'wf': 'P3_WF_NATIVE_ZERO_PAD512_V1'}
EFF = json.loads((V5 / 'efficiency.json').read_text())['graphs']


def rj(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def miou_w(cell):
    a = np.asarray(cell, dtype=np.float64); u = a.sum(1)
    iou = a[:, 0] / u
    return 100 * iou.mean(), 100 * iou[1]


def ci(d):
    d = [x for x in d if x is not None]
    if len(d) < 2:
        return {'n': len(d), 'mean': d[0] if d else None, 'lo': None, 'hi': None}
    m, s = st.mean(d), st.stdev(d); h = stats.t.ppf(.975, len(d) - 1) * s / len(d) ** .5
    return {'n': len(d), 'mean': m, 'sd': s, 'lo': m - h, 'hi': m + h}


def fmt(c, digits=2):
    if c['mean'] is None:
        return 'n/a'
    if c['lo'] is None:
        return f"{c['mean']:+.{digits}f} (n={c['n']})"
    return f"{c['mean']:+.{digits}f} [{c['lo']:+.{digits}f}, {c['hi']:+.{digits}f}] (n={c['n']})"


# --- fixed comparators (teacher, MNDWI) --------------------------------------------------------------
comp_event, comp_pooled = defaultdict(dict), {}
for r in csv.DictReader(open(COMP)):
    if r['run_id'] in ('R_comparator_teacher', 'R_comparator_index_mndwi_zero') and r['condition'] == 'clean' \
            and r['domain'] == 'full' and r['level'] == 'event' and r['protocol'] == PROTO.get(r['dataset']):
        tp, fp, fn = (float(r[k]) for k in ('TP', 'FP', 'FN'))
        comp_event[(r['run_id'], r['dataset'])][r['unit']] = 100 * tp / (tp + fp + fn) if tp + fp + fn else None
for r in csv.DictReader(open(POOLED)):
    if r['run_id'] in ('R_comparator_teacher', 'R_comparator_index_mndwi_zero') and r['protocol'] == PROTO.get(r['dataset']) \
            and r['domain'] == 'full':
        vals = [float(v) for v in list(r.values())[-4:]]          # TP, FP, FN, TN
        tp, fp, fn, tn = vals
        w = tp / (tp + fp + fn); b = tn / (tn + fp + fn)
        comp_pooled[(r['run_id'], r['dataset'])] = (100 * (w + b) / 2, 100 * w)
NAMES = {'R_comparator_teacher': 'teacher', 'R_comparator_index_mndwi_zero': 'MNDWI>0'}


# --- model readouts ----------------------------------------------------------------------------------
def terminal(rid):
    t = rj(V5 / 'terminal' / f'{rid}.json')
    return t['terminal_results'] if t else None


def readout(rid):
    t = terminal(rid)
    if not t:
        return None
    out = {}
    for b in ('s11', 'sturm', 'wf'):
        m = t[b]['full']['micro']
        out[b] = (100 * m['mIoU'], 100 * m['water_IoU'])
        out[b + '_events'] = {k: miou_w(v)[1] for k, v in t[b]['full']['event'].items()}
    return out


def event_bootstrap(model_events_by_seed, ref_events, n=10000, seed=1729):
    """Per-event water-IoU (model seed-averaged within event) minus reference; bootstrap over events."""
    events = sorted(set(ref_events) & set.intersection(*[set(e) for e in model_events_by_seed]))
    diffs = []
    for e in events:
        vals = [m[e] for m in model_events_by_seed if m.get(e) is not None and not np.isnan(m[e])]
        if vals and ref_events[e] is not None:
            diffs.append(np.mean(vals) - ref_events[e])
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(seed)
    boots = diffs[rng.integers(0, len(diffs), (n, len(diffs)))].mean(1)
    return {'events': len(diffs), 'mean': float(diffs.mean()), 'lo95': float(np.percentile(boots, 2.5)),
            'hi95': float(np.percentile(boots, 97.5)), 'one_sided_lo95': float(np.percentile(boots, 5))}


def val_score(rid, kind='train'):
    r = rj(V5 / 'results' / f'{rid}.json') if kind != 'stage2' else rj(V4 / 'results' / f'{rid}.json')
    if not r or r.get('outcome') != 'completed':
        return None
    if 'validation' in r:                                  # lowbit receipt
        return 100 * r['validation']['selected']['micro']['mIoU']
    return 100 * r['source_validation']['best_val_mIoU']


summary = {'comparators': {NAMES[k[0]] + '_' + k[1]: v for k, v in comp_pooled.items()}}
rows_csv = []


def cell_stats(label, ids, kind='train', graph=None):
    """ids: {seed: run id}. Returns a row with S11-val and external seed stats and bootstrap vs teacher/MNDWI."""
    vals = {s: val_score(i, kind) for s, i in ids.items()}
    reads = {s: readout(i) for s, i in ids.items()}
    row = {'label': label, 'graph': graph, 'n_val': sum(v is not None for v in vals.values())}
    v = [x for x in vals.values() if x is not None]
    row['val_mean'] = st.mean(v) if v else None; row['val_sd'] = st.stdev(v) if len(v) > 1 else None
    have = {s: r for s, r in reads.items() if r}
    row['n_ext'] = len(have)
    for b in ('s11', 'sturm', 'wf'):
        for j, metric in enumerate(('mIoU', 'wIoU')):
            x = [r[b][j] for r in have.values()]
            row[f'{b}_{metric}_mean'] = st.mean(x) if x else None
            row[f'{b}_{metric}_sd'] = st.stdev(x) if len(x) > 1 else None
    for b in ('sturm', 'wf'):
        if have:
            for comp in ('R_comparator_teacher', 'R_comparator_index_mndwi_zero'):
                row[f'{b}_vs_{NAMES[comp]}'] = event_bootstrap([r[b + '_events'] for r in have.values()],
                                                               comp_event[(comp, b)])
    if graph in EFF:
        row.update(params=EFF[graph]['parameters'], macs_px=EFF[graph]['MACs_per_pixel'])
    row['_vals'] = vals; row['_reads'] = {s: {b: r[b] for b in ('s11', 'sturm', 'wf')} for s, r in have.items()}
    rows_csv.append(row)
    return row


def paired(a, b, field='val'):
    """Paired-seed differences between two cell rows (field 'val' or '<bench>_<j>')."""
    d = []
    for s in SEEDS:
        if field == 'val':
            x, y = a['_vals'].get(s), b['_vals'].get(s)
        else:
            bench, j = field.split(':'); j = int(j)
            x = a['_reads'].get(s, {}).get(bench, (None, None))[j] if s in a['_reads'] else None
            y = b['_reads'].get(s, {}).get(bench, (None, None))[j] if s in b['_reads'] else None
        if x is not None and y is not None:
            d.append(x - y)
    return ci(d)


FIELDS = [('val', 'S11-val mIoU'), ('s11:0', 'S11-test mIoU'), ('sturm:1', 'STURM wIoU'), ('wf:1', 'WF wIoU'),
          ('sturm:0', 'STURM mIoU'), ('wf:0', 'WF mIoU')]


def contrast(name, a, b):
    out = {f: paired(a, b, f) for f, _ in FIELDS}
    summary.setdefault('contrasts', {})[name] = out
    return out


def interaction(name, a1, b1, a0, b0):
    """(a1-b1) - (a0-b0) per seed."""
    out = {}
    for f, _ in FIELDS:
        d = []
        for s in SEEDS:
            vals = []
            for r in (a1, b1, a0, b0):
                if f == 'val':
                    vals.append(r['_vals'].get(s))
                else:
                    bench, j = f.split(':')
                    vals.append(r['_reads'].get(s, {}).get(bench, (None, None))[int(j)] if s in r['_reads'] else None)
            if None not in vals:
                d.append((vals[0] - vals[1]) - (vals[2] - vals[3]))
        out[f] = ci(d)
    summary.setdefault('interactions', {})[name] = out
    return out


# --- E1/Stage 2 FP family -----------------------------------------------------------------------------
fam = {g: cell_stats(f'{g} LKD (Stage2)', {s: f'{g}_FP_s{s}' for s in SEEDS}, 'stage2', g)
       for g in ['A0', 'A1', 'A2', 'X1', 'X2', 'A3']}
# --- E2 low-bit ----------------------------------------------------------------------------------------
e2 = {arm: cell_stats(f'A3 {arm}', {s: f'lb_{arm}_A3_s{s}' for s in SEEDS}, 'train', 'A3') for arm in ('Q48', 'Q4b', 'F48')}
if any(e2[a]['n_val'] for a in e2):
    contrast('E2 Q48 - F48 (H6a quantization-specific)', e2['Q48'], e2['F48'])
    contrast('E2 Q48 - FP parent', e2['Q48'], fam['A3'])
    contrast('E2 F48 - FP parent (extra optimisation)', e2['F48'], fam['A3'])
    contrast('E2 Q4b - Q48 (H6c)', e2['Q4b'], e2['Q48'])
# --- E3 capacity floor -------------------------------------------------------------------------------------
floor = {}
for g in ['R1', 'R2', 'R3', 'R4', 'A3N', 'A3']:
    grid = {lr: val_score(f'fl_{g}_lr{lr}_s42') for lr in ('5e-5', '5e-4', '5e-3')}
    sel = rj(V5 / 'selections' / f'lr_{g}.json')
    floor[g] = cell_stats(f'{g} LKD @sel lr', {s: f'fl_{g}_sel_s{s}' for s in SEEDS}, 'train', g)
    floor[g]['lr_grid_s42'] = grid; floor[g]['lr_selected'] = sel['lr'] if sel else None
for g in ['R1', 'R2', 'R3', 'R4', 'A3N']:
    if floor[g]['n_val'] and floor['A3']['n_val']:
        contrast(f'E3 {g} - A3', floor[g], floor['A3'])
summary['floor_selection'] = rj(V5 / 'selections' / 'floor.json')
# --- E4 methods ------------------------------------------------------------------------------------------
methods = {}
for g in ('A0', 'A3'):
    methods[(g, 'lkd')] = fam[g]
    for m in ('ofa', 'ce', 'sup'):
        methods[(g, m)] = cell_stats(f'{g} {m.upper()}', {s: f'm_{m}_{g}_s{s}' for s in SEEDS}, 'train', g)
fsel = summary['floor_selection']
if fsel and fsel.get('graph'):
    F = fsel['graph']
    methods[('F', 'lkd')] = floor[F]
    for m in ('ce', 'sup', 'ofa'):
        methods[('F', m)] = cell_stats(f'floor {F} {m.upper()}', {s: f'mf_{m}_s{s}' for s in SEEDS}, 'train', F)
for g in ('A0', 'A3', 'F'):
    if (g, 'lkd') not in methods:
        continue
    for m, ref in (('lkd', 'ce'), ('ofa', 'ce'), ('ofa', 'lkd'), ('lkd', 'sup')):
        if methods.get((g, m)) and methods.get((g, ref)) and methods[(g, m)]['n_val'] and methods[(g, ref)]['n_val']:
            contrast(f'E4 {g}: {m.upper()} - {ref.upper()}', methods[(g, m)], methods[(g, ref)])
for small in ('A3', 'F'):
    for kd in ('lkd', 'ofa'):
        keys = [(small, kd), (small, 'ce'), ('A0', kd), ('A0', 'ce')]
        if all(methods.get(k) and methods[k]['n_val'] for k in keys):
            interaction(f'H7 [{kd.upper()}-CE]({small}) - [{kd.upper()}-CE](A0)', *[methods[k] for k in keys])

# --- write ----------------------------------------------------------------------------------------------
flat = []
for r in rows_csv:
    f = {k: v for k, v in r.items() if not k.startswith('_') and not isinstance(v, dict)}
    for b in ('sturm', 'wf'):
        for c in ('teacher', 'MNDWI>0'):
            e = r.get(f'{b}_vs_{c}')
            if e:
                f[f'{b}_wIoU_minus_{c}_event_mean'] = e['mean']; f[f'{b}_wIoU_minus_{c}_lo95'] = e['lo95']
                f[f'{b}_wIoU_minus_{c}_hi95'] = e['hi95']; f[f'{b}_wIoU_minus_{c}_onesided_lo95'] = e['one_sided_lo95']
    for k in ('lr_grid_s42',):
        if k in r:
            f[k] = json.dumps(r[k])
    flat.append(f)
keys = sorted({k for f in flat for k in f}, key=lambda k: (k not in ('label', 'graph', 'params', 'macs_px'), k))
with open(OUT / 'cells.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(flat)
summary['cells'] = flat
(OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))


def show(v, d=2):
    return '—' if v is None else f'{v:.{d}f}'


print('| cell | params | MACs/px | n | S11-val | S11-test mIoU | STURM mIoU / wIoU | WF mIoU / wIoU | STURM wIoU−teacher [95%] | WF wIoU−teacher [95%] |')
print('|---|---|---|---|---|---|---|---|---|---|')
for f in flat:
    if not f.get('n_val'):
        continue
    def ev(b):
        lo = f.get(f'{b}_wIoU_minus_teacher_lo95')
        return '—' if lo is None else f"{f[f'{b}_wIoU_minus_teacher_event_mean']:+.2f} [{lo:+.2f}, {f[f'{b}_wIoU_minus_teacher_hi95']:+.2f}]"
    print(f"| {f['label']} | {f.get('params', '')} | {show(f.get('macs_px'), 0)} | {f['n_val']}/{f['n_ext']} | "
          f"{show(f['val_mean'])}±{show(f['val_sd'])} | {show(f.get('s11_mIoU_mean'))} | "
          f"{show(f.get('sturm_mIoU_mean'))} / {show(f.get('sturm_wIoU_mean'))} | {show(f.get('wf_mIoU_mean'))} / {show(f.get('wf_wIoU_mean'))} | {ev('sturm')} | {ev('wf')} |")
for k, (m, w) in sorted(summary['comparators'].items()):
    print(f'| {k} | | | fixed | | {m:.2f} (mIoU) | wIoU {w:.2f} | | | |')
print()
for name, c in summary.get('contrasts', {}).items():
    print(f'- {name}: ' + '; '.join(f'{lab} {fmt(c[f])}' for f, lab in FIELDS))
for name, c in summary.get('interactions', {}).items():
    print(f'- {name}: ' + '; '.join(f'{lab} {fmt(c[f])}' for f, lab in FIELDS))
