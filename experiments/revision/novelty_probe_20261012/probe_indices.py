#!/usr/bin/env python3
"""Literature water indices from the six input bands, scored on EXPOSED sets (exploratory).

Read-only, CPU-only; never reads WorldFloods-v2 train/val. Bands (reflectance):
B=B2, G=B3, R=B4, N=B8A (20 m native), S1=B11 (20 m), S2=B12 (20 m).
  MNDWI   = (G-S1)/(G+S1)                                   Xu 2006
  NDWI8A  = (G-N)/(G+N)                                     McFeeters 1996 with B8A
  AWEInsh = 4(G-S1) - (0.25N + 2.75S2)                       Feyisa et al. 2014
  AWEIsh  = B + 2.5G - 1.5(N+S1) - 0.25S2                    Feyisa et al. 2014 (shadow suppression)
  WI2015  = 1.7204 + 171G + 3R - 70N - 45S1 - 71S2           Fisher et al. 2016
  MNDWIsh = MNDWI with S1 sharpened by the 10 m green detail: S1 * G / lowpass3x3(G)
            (a crude high-pass-modulation stand-in for Du et al. 2016)
Published thresholds are 0 for every index. For each index: standalone water map,
average precision, the edge-band re-decision (k=2) of the four R4 LKD parents, and the
cloud-band re-decision (20 px around input-detected cloud) - all at threshold 0, so
nothing is tuned on exposed labels. Writes indices_report.txt.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_cpu as pc  # noqa: E402
import probe_failure as pf  # noqa: E402
import probe_cloud as pcl  # noqa: E402

NAMES = ['MNDWI', 'NDWI8A', 'AWEInsh', 'AWEIsh', 'WI2015', 'MNDWIsh']


def indices(refl):
    import numpy as np
    from scipy import ndimage
    b, g, r, n, s1, s2 = [x.astype(np.float32) for x in refl]
    eps = 1e-6
    lp = ndimage.uniform_filter(g, 3)
    s1sh = s1 * g / np.maximum(lp, eps)
    out = {
        'MNDWI': (g - s1) / np.maximum(g + s1, eps),
        'NDWI8A': (g - n) / np.maximum(g + n, eps),
        'AWEInsh': 4 * (g - s1) - (0.25 * n + 2.75 * s2),
        'AWEIsh': b + 2.5 * g - 1.5 * (n + s1) - 0.25 * s2,
        'WI2015': 1.7204 + 171 * g + 3 * r - 70 * n - 45 * s1 - 71 * s2,
        'MNDWIsh': (g - s1sh) / np.maximum(g + s1sh, eps),
    }
    valid = np.isfinite(refl).all(axis=0) & (g + s1 > eps)
    return out, valid


def cnt(pred, y):
    import numpy as np
    v = y != -1
    p, w = pred[v], y[v] == 1
    return [int(np.count_nonzero(p & w)), int(np.count_nonzero(p & ~w)), int(np.count_nonzero(~p & w)), int(np.count_nonzero(~p & ~w))]


def work(job):
    import numpy as np
    from scipy import ndimage
    y = job['mask']; h, w = y.shape
    crop = lambda x: x[:h, :w]
    refl = crop(job['refl'].astype(np.float32).transpose(1, 2, 0)).transpose(2, 0, 1)
    idx, ok = indices(refl)
    cl = pcl.cloud_test(refl)
    cdist = ndimage.distance_transform_edt(~cl) if cl.any() else None
    cband = (cdist > 0) & (cdist <= 20) if cdist is not None else None
    out = dict(event=job['event'], counts={}, hist={}, dry={})
    valid = y != -1
    is_dry = bool(valid.any() and not (y == 1).any())
    preds = {f's{s}': crop(mg.astype(np.float32) > 0) for s, mg in job['margins'].items()}
    dists = {k: pf.edge_dist(p) for k, p in preds.items()}
    for name in NAMES:
        v = idx[name]
        water = (v > 0) & ok
        out['counts'][f'{name}|alone'] = cnt(water, y)
        if is_dry:
            out['dry'][name] = bool((water & valid).sum() >= 0.01 * valid.sum())
        # coarse histogram of the index for AP: 200 quantile-free bins over a robust range
        lo, hi = {'MNDWI': (-1, 1), 'NDWI8A': (-1, 1), 'MNDWIsh': (-1, 1), 'AWEInsh': (-3, 3), 'AWEIsh': (-1.5, 1.5), 'WI2015': (-60, 60)}[name]
        bins = np.clip(((v - lo) / (hi - lo) * 200).astype(np.int64), 0, 199)
        sel = valid & ok
        out['hist'][name] = [np.bincount(bins[sel & (y == 0)], minlength=200).tolist(), np.bincount(bins[sel & (y == 1)], minlength=200).tolist()]
        for s, p in preds.items():
            band = dists[s] <= 2
            q = p.copy(); q[band] = water[band]
            out['counts'][f'{name}|edge|{s}'] = cnt(q, y)
            if cband is not None:
                q2 = p.copy(); q2[cband] = water[cband]
            else:
                q2 = p
            out['counts'][f'{name}|cloud|{s}'] = cnt(q2, y)
    for s, p in preds.items():
        out['counts'][f'student|{s}'] = cnt(p, y)
    return out


def main():
    spec = pc.SPEC
    sys.dont_write_bytecode = True
    sys.path.insert(0, spec['capsule']); sys.path.insert(0, str(pc.V7))
    import numpy as np
    import torch
    torch.set_num_threads(12)
    import evaluate_v7 as ev
    from train_v7 import build_model
    from data.channel_adapter import sturm_s2_adapter
    from evaluation import revision_native_border as rb
    from evaluation.revision_eval import NamedDataset
    import scripts.compression_jobs as cj
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    models = {}
    for e in pc.MODELS:
        payload = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)
        mdl = build_model('R4', 'lkd', spec['rungs_dir']); mdl.load_state_dict(payload['model'], strict=True)
        models[e['seed']] = mdl.eval()
    sets, map_of, event_of = ev.datasets(spec)
    val, _ = cj.loaders(spec)
    sets = [('s11val', NamedDataset(val.dataset, [Path(p).name for p in val.dataset.image_files]), 4)] + sets
    adapter = sturm_s2_adapter(target_channels=6)
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    pool = ProcessPoolExecutor(10)
    res = {}
    for name, ds, batch in sets:
        futs = []
        for begin in range(0, len(ds), batch):
            ids = list(range(begin, min(begin + batch, len(ds))))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            if name == 'sturm':
                x = adapter(x)
            with torch.no_grad():
                margins = {s: (lambda z: z[:, 1] - z[:, 0])(m(x)).numpy() for s, m in models.items()}
            refl = (x * sd + mu).numpy()
            for local, (i, it) in enumerate(zip(ids, items)):
                fn = Path(it['filename']).name
                mask = it['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else np.array(mask, copy=True)
                if name == 'sturm':
                    _, invalid = rb.sturm_invalid(ds, i); mask[invalid] = -1; event = event_of[fn]
                elif name == 'wf':
                    event = fn.split('#')[0].split('_')[0]
                else:
                    event = fn.split('_')[0]
                futs.append(pool.submit(work, dict(event=event, mask=mask.astype(np.int8),
                    margins={s: mg[local].astype(np.float16) for s, mg in margins.items()}, refl=refl[local].astype(np.float16))))
        evc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])); hist = {}; dry = defaultdict(lambda: [0, 0])
        for f in futs:
            r = f.result()
            for k, v in r['counts'].items():
                for q in range(4): evc[r['event']][k][q] += v[q]
            for k, (h0, h1) in r['hist'].items():
                a = hist.setdefault(k, [np.zeros(200, np.int64), np.zeros(200, np.int64)]); a[0] += h0; a[1] += h1
            for k, flag in r['dry'].items():
                dry[k][0] += 1; dry[k][1] += int(flag)
        res[name] = (evc, hist, dry)
        print(name, 'done', flush=True)

    def eqw(evc, key):
        v = [d[key][0] / (d[key][0] + d[key][1] + d[key][2]) for d in evc.values() if key in d and d[key][0] + d[key][1] + d[key][2]]
        return st.mean(v)

    def pooled(evc, key):
        tp, fp, fn, tn = [sum(d[key][i] for d in evc.values()) for i in range(4)]
        return tp / (tp + fp + fn), tp / max(1, tp + fp), tp / max(1, tp + fn)

    def ap(h):
        land, water = h[0][::-1].cumsum(), h[1][::-1].cumsum()
        prec = water / np.maximum(1, water + land); rec = water / max(1, h[1].sum())
        return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    seeds = [f's{s}' for s in sorted(models)]
    rep = ['Indices at published threshold 0 (exploratory, research-exposed sets).']
    for name, (evc, hist, dry) in res.items():
        stud = st.mean(eqw(evc, f'student|{s}') for s in seeds)
        rep.append(f'\n=== {name}: R4 student equal-event water IoU {100 * stud:.2f} ===')
        for ix in NAMES:
            a = eqw(evc, f'{ix}|alone'); p = pooled(evc, f'{ix}|alone')
            edge = st.mean(eqw(evc, f'{ix}|edge|{s}') - eqw(evc, f'student|{s}') for s in seeds)
            cloud = st.mean(eqw(evc, f'{ix}|cloud|{s}') - eqw(evc, f'student|{s}') for s in seeds)
            dr = dry.get(ix, [0, 0])
            rep.append(f'  {ix:8s} alone: eqW {100 * a:5.1f} pooledW {100 * p[0]:5.1f} (P {100 * p[1]:4.1f} R {100 * p[2]:4.1f}) AP {100 * ap(hist[ix]):5.1f} '
                       f'dry alarms {dr[1]}/{dr[0]} | student + edge re-decision {100 * edge:+.2f} | student + cloud-band re-decision {100 * cloud:+.2f}')
    (HERE / 'indices_report.txt').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
