"""CPU-only readouts from immutable v5 counts; no selection or training."""
import collections
import csv
import json
from pathlib import Path
import statistics as st

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SEEDS = (42, 64, 91, 107)
PROTO = {'s11': 'S11_TEST512_LEGACY', 'sturm': 'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1',
         'wf': 'P3_WF_NATIVE_ZERO_PAD512_V1'}


def read(p):
    return json.loads(Path(p).read_text())


def metrics(cell):
    c = np.asarray(cell, dtype=np.float64)
    iou = np.divide(c[:, 0], c.sum(1), out=np.full(2, np.nan), where=c.sum(1) > 0)
    return {'mIoU': 100 * np.nanmean(iou), 'water_IoU': 100 * iou[1],
            'water_precision': 100 * c[1, 0] / (c[1, 0] + c[1, 1]) if c[1, 0] + c[1, 1] else None,
            'water_recall': 100 * c[1, 0] / (c[1, 0] + c[1, 2]) if c[1, 0] + c[1, 2] else None}


def ci(values):
    a = np.asarray(values, dtype=float)
    m, sd = float(a.mean()), float(a.std(ddof=1))
    half = float(stats.t.ppf(.975, len(a)-1) * sd / len(a)**.5)
    return {'n': len(a), 'mean': m, 'sd': sd, 'lo': m-half, 'hi': m+half}


def bootstrap(differences):
    a = np.asarray(differences, dtype=float)
    a = a[np.isfinite(a)]
    rng = np.random.default_rng(1729)
    sampled = a[rng.integers(0, len(a), (10000, len(a)))].mean(1)
    return {'n_events': len(a), 'mean': float(a.mean()), 'lo95': float(np.quantile(sampled, .025)),
            'hi95': float(np.quantile(sampled, .975)), 'one_sided_lo95': float(np.quantile(sampled, .05))}


teacher = collections.defaultdict(dict)
for r in csv.DictReader((BASE.parent / 'prompt5_closure/sensitivity_complete/confusion_counts.csv').open()):
    if r['run_id'] == 'R_comparator_teacher' and r['condition'] == 'clean' and r['level'] == 'event' \
            and r['protocol'] == PROTO.get(r['dataset']):
        tp, fp, fn = (float(r[k]) for k in ('TP', 'FP', 'FN'))
        teacher[(r['dataset'], r['domain'])][r['unit']] = 100*tp/(tp+fp+fn) if tp+fp+fn else np.nan

families = {g+' recipe LKD': [f'{g}_FP_s{s}' for s in SEEDS] for g in ('A0','A1','A2','X1','X2','A3')}
families.update({g+' selected LKD': [f'fl_{g}_sel_s{s}' for s in SEEDS] for g in ('R1','R2','R3','R4','A3N','A3')})
families.update({'A3 '+a: [f'lb_{a}_A3_s{s}' for s in SEEDS] for a in ('Q48','Q4b','F48')})
families.update({g+' '+m.upper(): [f'm_{m}_{g}_s{s}' for s in SEEDS]
                 for g in ('A0','A3') for m in ('ce','ofa','sup')})
families.update({'R2 '+m.upper(): [f'mf_{m}_s{s}' for s in SEEDS] for m in ('ce','sup')})
data, flat, finals = {}, [], []
for label, ids in families.items():
    rows = [read(BASE/'v5/terminal'/f'{rid}.json') for rid in ids]
    data[label] = rows
    for bench in PROTO:
        for domain, reference_domain in (('full','full'),('known_overlap_excluded','heldout')):
            results = [r['terminal_results'][bench][domain] for r in rows]
            d = {'family': label, 'benchmark': bench, 'domain': domain,
                 'n_images': results[0].get('n_images', rows[0]['terminal_results'][bench]['n_images']),
                 'n_events': results[0]['n_events']}
            for key in ('mIoU','water_IoU','water_precision','water_recall'):
                vals = [100*r['micro'][key] for r in results]
                d[key+'_mean'], d[key+'_sd'] = st.mean(vals), st.stdev(vals)
            event_maps = [{e:metrics(c)['water_IoU'] for e,c in r['event'].items()} for r in results]
            ref = teacher[(bench,reference_domain)]
            common = sorted(set(ref).intersection(*(set(e) for e in event_maps)))
            model_events = sorted(set.intersection(*(set(e) for e in event_maps)))
            d['event_water_IoU_mean'] = st.mean(st.mean(m[e] for m in event_maps) for e in model_events)
            delta = bootstrap([st.mean(m[e] for m in event_maps)-ref[e] for e in common]) if common else {
                'n_events':0,'mean':None,'lo95':None,'hi95':None,'one_sided_lo95':None}
            d.update({'teacher_event_delta_'+k:v for k,v in delta.items()})
            flat.append(d)
    for seed, rid in zip(SEEDS, ids):
        p = BASE/'v5/results'/f'{rid}.json'
        if not p.exists():
            p = BASE/'results'/f'{rid}.json'
        r = read(p)
        if 'validation' in r:
            selected = 100*r['validation']['selected']['micro']['mIoU']
            final = 100*r['validation']['final']['micro']['mIoU']
        else:
            selected = 100*r['source_validation']['best_val_mIoU']
            last_val = [e for e in r['epochs'] if e.get('phase') == 'val'][-1]
            final = 100*last_val['mIoU']
        finals.append({'family':label,'seed':seed,'selected_val_mIoU':selected,'final_val_mIoU':final,
                       'selected_updates':r['selected_updates'],'completed_updates':r['completed_updates'],
                       'receipt':str(p)})

contrasts = [('R4 selected LKD','A3 selected LKD'),('A3N selected LKD','A3 selected LKD'),
    ('A3 selected LKD','A3 recipe LKD'),('A3 recipe LKD','A1 recipe LKD'),
    ('A0 recipe LKD','A0 CE'),('A0 OFA','A0 recipe LKD'),('A0 OFA','A0 CE'),
    ('A3 recipe LKD','A3 CE'),('R2 selected LKD','R2 CE'),('R2 selected LKD','R2 SUP'),
    ('A3 Q48','A3 F48'),('A3 Q4b','A3 Q48')]
event_contrasts = {}
for lhs,rhs in contrasts:
    out = {}
    for bench in ('s11','sturm','wf'):
        for domain in ('full','known_overlap_excluded'):
            left = [r['terminal_results'][bench][domain] for r in data[lhs]]
            right = [r['terminal_results'][bench][domain] for r in data[rhs]]
            events = sorted(set.intersection(*(set(r['event']) for r in left+right)))
            deltas = [st.mean(metrics(l['event'][e])['water_IoU']-metrics(r['event'][e])['water_IoU']
                              for l,r in zip(left,right)) for e in events]
            out[bench+' '+domain] = {'conditional_event_water_IoU':bootstrap(deltas),
                'paired_seed_pooled_water_IoU':ci([100*(l['micro']['water_IoU']-r['micro']['water_IoU'])
                                                for l,r in zip(left,right)])}
    event_contrasts[lhs+' - '+rhs] = out

event_interactions = {}
for small in ('A3 recipe LKD', 'R2 selected LKD'):
    ce = 'A3 CE' if small.startswith('A3') else 'R2 CE'
    out = {}
    for bench in ('sturm', 'wf'):
        maps = [[r['terminal_results'][bench]['full']['event'] for r in data[label]]
                for label in (small, ce, 'A0 recipe LKD', 'A0 CE')]
        common = sorted(set.intersection(*(set(m) for row in maps for m in row)))
        deltas = [st.mean(metrics(maps[0][s][e])['water_IoU'] - metrics(maps[1][s][e])['water_IoU']
                         - metrics(maps[2][s][e])['water_IoU'] + metrics(maps[3][s][e])['water_IoU']
                         for s in range(4)) for e in common]
        out[bench] = bootstrap(deltas)
    event_interactions[small] = out

original = read(BASE.parent/'compression_v2_20261007/results/terminal_A3_FP_s42.json')
smoke = read(BASE/'v5_smoke/terminal/parity_A3_FP_s42_v2.json')
parity = {}
for bench in PROTO:
    old,new = original['terminal_results'][bench]['per_image_counts'],smoke['terminal_results'][bench]['per_image_counts']
    parity[bench] = {'images':len(new), 'mismatches':sum(old.get(k)!=v for k,v in new.items()),
                     'identical_keys':set(old)==set(new)}

wf = data['R4 selected LKD'][0]['terminal_results']['wf']['full']['event']
shares = {e:np.asarray(c)[1,[0,2]].sum() for e,c in wf.items()}
water_share = float(shares['EMSR342']/sum(shares.values()))
detail = {'event_contrasts':event_contrasts, 'evaluator_parity':parity,
          'event_interactions':event_interactions,
          'WF_EMSR342_reference_water_share':water_share, 'final_source_results':finals}
(OUT/'supplement_summary.json').write_text(json.dumps(detail,indent=2)+'\n')
for filename, rows in [('all_readouts.csv',flat),('selected_and_final_source.csv',finals)]:
    with (OUT/filename).open('w',newline='') as fh:
        writer = csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Evaluator parity:',parity,'WF dominant event water share:',water_share)
for k in event_contrasts:
    print(k, {b: {f:round(v,3) if isinstance(v,float) else v for f,v in event_contrasts[k][b+' full']['conditional_event_water_IoU'].items()}
              for b in ('sturm','wf')})
