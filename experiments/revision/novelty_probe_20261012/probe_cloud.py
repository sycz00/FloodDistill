#!/usr/bin/env python3
"""Cloud-adjacent failure: input-only cloud test and a cloud-band MNDWI re-decision (exploratory).

Read-only, CPU-only. Datasets: S11 validation (selection; its -1 label areas are
mostly cloud) and WF test (exposed; reference cloud band gt[0]==2 for checking the
input cloud test). Cloud test from the six input bands only (no thermal/cirrus):
  vis = mean(B,G,R); whiteness = sum|b - vis|/vis over B,G,R; HOT = B - 0.5*R - 0.08
  cloud = vis > 0.18 & whiteness < 0.7 & HOT > 0 & SWIR2 > 0.03 & SWIR1 > 0.12
Rule: inside a band of d pixels around (not on) input-detected cloud, pixels are
re-decided by MNDWI > theta; elsewhere the prediction is kept.
Writes cloud_report.txt.
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

DS = [5, 20, 50]
THETAS = [0.0, 0.1, 0.2, 0.3]


def cloud_test(refl):
    import numpy as np
    b, g, r, nir, s1, s2 = refl
    vis = (b + g + r) / 3
    white = (np.abs(b - vis) + np.abs(g - vis) + np.abs(r - vis)) / np.maximum(vis, 1e-6)
    hot = b - 0.5 * r - 0.08
    return (vis > 0.18) & (white < 0.7) & (hot > 0) & (s2 > 0.03) & (s1 > 0.12)


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
    m = crop(job['mndwi'].astype(np.float32)); a = crop(job['avail'])
    refl = crop(job['refl'].astype(np.float32).transpose(1, 2, 0)).transpose(2, 0, 1)
    cl = cloud_test(refl)
    out = dict(event=job['event'], counts={}, cloudcheck=None)
    if job.get('refcloud') is not None:
        rc = crop(job['refcloud'])
        out['cloudcheck'] = [int((cl & rc).sum()), int((cl & ~rc).sum()), int((~cl & rc).sum())]
    preds = {f's{s}': crop(mg.astype(np.float32) > 0) for s, mg in job['margins'].items()}
    if job['teacher'] is not None:
        preds['T'] = crop(job['teacher'])
    dist = ndimage.distance_transform_edt(~cl) if cl.any() else None
    for name, p in preds.items():
        out['counts'][f'{name}|base'] = cnt(p, y)
        for d in DS:
            band = (dist is not None) and ((dist > 0) & (dist <= d))
            for th in THETAS:
                q = p.copy()
                if dist is not None:
                    q[band] = ((m > th) & a)[band]
                out['counts'][f'{name}|d{d}|{th}'] = cnt(q, y)
    out['band_px'] = int(((dist > 0) & (dist <= 20) & (y != -1)).sum()) if dist is not None else 0
    return out


def main():
    spec = pc.SPEC
    sys.dont_write_bytecode = True
    sys.path.insert(0, spec['capsule']); sys.path.insert(0, str(pc.V7))
    import hashlib
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import torch
    torch.set_num_threads(16)
    import evaluate_v7 as ev
    from train_v7 import build_model
    from evaluation import revision_native_border as rb
    from evaluation.revision_eval import NamedDataset
    import scripts.compression_jobs as cj
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    models = {}
    for e in pc.MODELS:
        payload = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)
        mdl = build_model('R4', 'lkd', spec['rungs_dir']); mdl.load_state_dict(payload['model'], strict=True)
        models[e['seed']] = mdl.eval()
    sets, _, _ = ev.datasets(spec)
    val, _ = cj.loaders(spec)
    sets = [('s11val', NamedDataset(val.dataset, [Path(p).name for p in val.dataset.image_files]), 4)] + [s for s in sets if s[0] in ('s11', 'wf')]
    refs = json.loads((pc.V7 / 'reference_cache' / 'manifest.json').read_text())
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    pool = ProcessPoolExecutor(12)
    rep = []
    agg = {}
    for name, ds, batch in sets:
        cache = np.load(refs['datasets'][name]['path'], mmap_mode='r') if name != 's11val' else None
        futs = []
        for begin in range(0, len(ds), batch):
            ids = list(range(begin, min(begin + batch, len(ds))))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            with torch.no_grad():
                margins = {s: (lambda z: z[:, 1] - z[:, 0])(mdl(x)).numpy() for s, mdl in models.items()}
            refl = (x * sd + mu)
            g, s1 = refl[:, 1], refl[:, 4]
            avail = (torch.isfinite(g) & torch.isfinite(s1) & (g >= 0) & (s1 >= 0) & (g + s1 > 1e-6))
            mndwi = torch.where(avail, (g - s1) / (g + s1).clamp_min(1e-6), torch.zeros_like(g)).clamp(-1, 1)
            for local, (i, it) in enumerate(zip(ids, items)):
                fn = Path(it['filename']).name
                mask = it['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else np.array(mask, copy=True)
                event = fn.split('#')[0].split('_')[0] if name == 'wf' else fn.split('_')[0]
                teacher = refcloud = None
                if cache is not None:
                    row = refs['datasets'][name]['rows'][i]
                    assert row['filename'] == fn and row['mask_sha256'] == hashlib.sha256(mask.tobytes()).hexdigest()
                    teacher = np.unpackbits(cache[i], axis=1).reshape(3, 512, 512)[0].astype(bool)
                if name == 'wf':
                    ip, lp, win = ds.samples[i]; c, r, w_, h_ = win
                    with rasterio.open(lp) as src:
                        gt0 = src.read(1, window=Window(c, r, w_, h_))
                    refcloud = np.zeros((512, 512), bool); refcloud[:h_, :w_] = gt0 == 2
                futs.append(pool.submit(work, dict(event=event, mask=mask.astype(np.int8),
                    margins={s: mg[local].astype(np.float16) for s, mg in margins.items()},
                    mndwi=mndwi[local].numpy().astype(np.float16), avail=avail[local].numpy(),
                    refl=refl[local].numpy().astype(np.float16), teacher=teacher, refcloud=refcloud)))
        ev_c = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])); cc = [0, 0, 0]; band = 0
        for f in futs:
            r = f.result()
            for k, v in r['counts'].items():
                for q in range(4): ev_c[r['event']][k][q] += v[q]
            if r['cloudcheck']:
                for q in range(3): cc[q] += r['cloudcheck'][q]
            band += r['band_px']
        agg[name] = ev_c
        if name == 'wf':
            rep.append(f'WF input cloud test vs reference cloud band: precision {cc[0] / max(1, cc[0] + cc[1]):.2f}, recall {cc[0] / max(1, cc[0] + cc[2]):.2f}')
        rep.append(f'{name}: valid pixels within 20 px of input-detected cloud: {band:,d}')

    def eqw(ev, key):
        v = [d[key][0] / (d[key][0] + d[key][1] + d[key][2]) for d in ev.values() if key in d and d[key][0] + d[key][1] + d[key][2]]
        return st.mean(v)
    seeds = ['s42', 's64', 's91', 's107']
    variants = sorted({k.split('|', 1)[1] for k in next(iter(agg['s11val'].values())) if not k.endswith('base')})
    sel = max(variants, key=lambda v: st.mean(eqw(agg['s11val'], f'{s}|{v}') - eqw(agg['s11val'], f'{s}|base') for s in seeds))
    for v in sorted(variants, key=lambda v: -st.mean(eqw(agg['s11val'], f'{s}|{v}') - eqw(agg['s11val'], f'{s}|base') for s in seeds))[:5]:
        rep.append(f'  S11-val {v}: {100 * st.mean(eqw(agg["s11val"], f"{s}|{v}") - eqw(agg["s11val"], f"{s}|base") for s in seeds):+.2f}')
    rep.append(f'selected on S11 val: {sel}')
    for name in ('s11', 'wf'):
        for v in [sel, 'd20|0.1', 'd20|0.2', 'd50|0.2']:
            line = []
            for who in ['T'] + seeds:
                line.append(f'{who} {100 * (eqw(agg[name], f"{who}|{v}") - eqw(agg[name], f"{who}|base")):+.2f}')
            rep.append(f'{name} {v:9s}: equal-event water IoU change ' + ', '.join(line))
    (HERE / 'cloud_report.txt').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
