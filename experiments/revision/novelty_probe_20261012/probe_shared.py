#!/usr/bin/env python3
"""Which errors do the teacher, the student and the MNDWI>0 rule SHARE? (exposed sets, CPU, read-only)

Per pixel: teacher (V7 cached hard mask), student (R4 LKD parent seed 42), MNDWI>0 rule,
reference. Joint code = 8*ref + 4*teacher + 2*rule + student. For every dataset it
reports how many of each model's misses / false alarms are shared by all three, and
where the shared errors lie: reference-edge distance, water-body width, spectral pixel
type, cloud distance (WF reference cloud band), plus mean spectra of shared misses.
Never reads WorldFloods-v2 train/val. Writes shared_report.txt and shared_counts.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_cpu as pc  # noqa: E402
import probe_failure as pf  # noqa: E402

STRATA = ['edge<=2px', 'edge3-10px', 'edge>10px', 'thin-water<=2px', 'vegetated(NDVI>0.4)', 'dark', 'bright(vis>0.2)',
          'cloud<=50px', 'no-land-or-no-water-tile']


def work(job):
    import numpy as np
    from scipy import ndimage
    y = job['mask']; h, w = y.shape
    crop = lambda a: a[:h, :w]
    m = crop(job['mndwi'].astype(np.float32)); a = crop(job['avail'])
    refl = crop(job['refl'].astype(np.float32).transpose(1, 2, 0)).transpose(2, 0, 1)
    t = crop(job['teacher']); s = crop(job['margin'].astype(np.float32)) > 0; r = (m > 0) & a
    valid = y != -1
    code = 8 * (y == 1) + 4 * t + 2 * r + s
    d_w, d_l, half = pf.edges_and_width(y)
    dist = np.where(y == 1, d_w, d_l)
    vis = refl[:3].mean(0); nir, sw1 = refl[3], refl[4]
    ndvi = (nir - refl[2]) / np.maximum(nir + refl[2], 1e-6)
    sel = {'edge<=2px': dist <= 2, 'edge3-10px': (dist > 2) & (dist <= 10), 'edge>10px': (dist > 10) & np.isfinite(dist),
           'thin-water<=2px': (y == 1) & (half > 0) & (half <= 2), 'vegetated(NDVI>0.4)': ndvi > 0.4,
           'dark': (vis < 0.06) & (nir < 0.08) & (sw1 < 0.06), 'bright(vis>0.2)': vis > 0.2,
           'no-land-or-no-water-tile': ~np.isfinite(dist)}
    if job.get('cloud') is not None:
        cl = crop(job['cloud'])
        sel['cloud<=50px'] = (ndimage.distance_transform_edt(~cl) <= 50) & ~cl if cl.any() else np.zeros(y.shape, bool)
    out = {'all': np.bincount(code[valid], minlength=16).tolist()}
    for k, v in sel.items():
        out[k] = np.bincount(code[valid & v], minlength=16).tolist()
    # spectra of shared misses (ref water, all three say land) and shared false water
    sm = valid & (code == 8); sf = valid & (code == 7); hit = valid & (code == 15)
    spec = {}
    for name, msk in (('shared_miss', sm), ('shared_false_water', sf), ('all_correct_water', hit)):
        n = int(msk.sum())
        spec[name] = [n] + (refl[:, msk].sum(1).tolist() if n else [0.0] * 6)
    return dict(event=job['event'], counts=out, spectra=spec)


def main():
    spec = pc.SPEC
    sys.dont_write_bytecode = True
    sys.path.insert(0, spec['capsule']); sys.path.insert(0, str(pc.V7))
    import hashlib
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import torch
    torch.set_num_threads(12)
    import evaluate_v7 as ev
    from train_v7 import build_model
    from data.channel_adapter import sturm_s2_adapter
    from evaluation import revision_native_border as rb
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    e42 = [e for e in pc.MODELS if e['seed'] == 42][0]
    mdl = build_model('R4', 'lkd', spec['rungs_dir'])
    mdl.load_state_dict(torch.load(e42['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True); mdl.eval()
    sets, map_of, event_of = ev.datasets(spec)
    refs = json.loads((pc.V7 / 'reference_cache' / 'manifest.json').read_text())
    adapter = sturm_s2_adapter(target_channels=6)
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    pool = ProcessPoolExecutor(10)
    res = {}
    for name, ds, batch in sets:
        cache = np.load(refs['datasets'][name]['path'], mmap_mode='r')
        futs = []
        for begin in range(0, len(ds), batch):
            ids = list(range(begin, min(begin + batch, len(ds))))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            if name == 'sturm':
                x = adapter(x)
            with torch.no_grad():
                z = mdl(x); margin = (z[:, 1] - z[:, 0]).numpy()
            refl = x * sd + mu
            g, s1 = refl[:, 1], refl[:, 4]
            avail = (torch.isfinite(g) & torch.isfinite(s1) & (g >= 0) & (s1 >= 0) & (g + s1 > 1e-6))
            mndwi = torch.where(avail, (g - s1) / (g + s1).clamp_min(1e-6), torch.zeros_like(g)).clamp(-1, 1)
            for local, (i, it) in enumerate(zip(ids, items)):
                fn = Path(it['filename']).name
                mask = it['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else np.array(mask, copy=True)
                cloud = None
                if name == 'sturm':
                    _, invalid = rb.sturm_invalid(ds, i); mask[invalid] = -1; event = event_of[fn]
                elif name == 'wf':
                    event = fn.split('#')[0].split('_')[0]
                    ip, lp, win = ds.samples[i]; c, r_, w_, h_ = win
                    with rasterio.open(lp) as src:
                        gt0 = src.read(1, window=Window(c, r_, w_, h_))
                    cloud = np.zeros((512, 512), bool); cloud[:h_, :w_] = gt0 == 2
                else:
                    event = fn.split('_')[0]
                row = refs['datasets'][name]['rows'][i]
                assert row['filename'] == fn and row['mask_sha256'] == hashlib.sha256(mask.tobytes()).hexdigest()
                teacher = np.unpackbits(cache[i], axis=1).reshape(3, 512, 512)[0].astype(bool)
                futs.append(pool.submit(work, dict(event=event, mask=mask.astype(np.int8), teacher=teacher, cloud=cloud,
                                                   margin=margin[local].astype(np.float16), mndwi=mndwi[local].numpy().astype(np.float16),
                                                   avail=avail[local].numpy(), refl=refl[local].numpy().astype(np.float16))))
        tot = defaultdict(lambda: np.zeros(16, np.int64)); spectra = defaultdict(lambda: np.zeros(7)); per_event = defaultdict(lambda: np.zeros(16, np.int64))
        for f in futs:
            r = f.result()
            for k, v in r['counts'].items():
                tot[k] += np.array(v)
            per_event[r['event']] += np.array(r['counts']['all'])
            for k, v in r['spectra'].items():
                spectra[k] += np.array(v)
        res[name] = dict(strata={k: v.tolist() for k, v in tot.items()}, events={k: v.tolist() for k, v in per_event.items()},
                         spectra={k: v.tolist() for k, v in spectra.items()})
        print(name, 'done', flush=True)
    (HERE / 'shared_counts.json').write_text(json.dumps(res))

    rep = ['Shared errors of teacher (T), R4 student seed 42 (S) and MNDWI>0 (M); exposed sets. Joint code = 8*ref + 4*T + 2*M + S.']
    for name, d in res.items():
        c = np.array(d['strata']['all'])
        water, land = c[8:].sum(), c[:8].sum()
        miss = {k: c[8:][[i for i in range(8) if not (i >> b) & 1]].sum() for k, b in (('T', 2), ('M', 1), ('S', 0))}
        fa = {k: c[:8][[i for i in range(8) if (i >> b) & 1]].sum() for k, b in (('T', 2), ('M', 1), ('S', 0))}
        shared_miss, shared_fa = c[8], c[7]
        rep.append(f'\n=== {name}: reference water {water:,} px, land {land:,} px ===')
        rep.append(f'  water missed by ALL THREE: {shared_miss:,} = {100 * shared_miss / water:.1f}% of all water; '
                   + ', '.join(f'{100 * shared_miss / miss[k]:.0f}% of {k} misses' for k in ('T', 'S', 'M')))
        rep.append(f'  false water by ALL THREE: {shared_fa:,} = {100 * shared_fa / land:.2f}% of land; '
                   + ', '.join(f'{100 * shared_fa / fa[k]:.0f}% of {k} false water' for k in ('T', 'S', 'M')))
        rep.append('  where the shared misses lie (share of all-three-miss pixels; strata overlap):')
        for k in STRATA:
            if k in d['strata']:
                cs = np.array(d['strata'][k]); n_w = cs[8:].sum()
                rep.append(f'    {k:26s} {100 * cs[8] / max(1, shared_miss):5.1f}% of shared misses | in this stratum all three miss {100 * cs[8] / max(1, n_w):5.1f}% of its water'
                           f' | shared false water {100 * cs[7] / max(1, shared_fa):5.1f}%')
        sp = d['spectra']
        for k in ('all_correct_water', 'shared_miss', 'shared_false_water'):
            n, *s = sp[k]
            if n:
                rep.append(f'  mean reflectance {k:20s} (B,G,R,NIR,SWIR1,SWIR2): ' + ' '.join(f'{x / n:.3f}' for x in s))
        ev_share = sorted(((e, np.array(v)[8] / max(1, np.array(v)[8:].sum())) for e, v in d['events'].items() if np.array(v)[8:].sum() > 0), key=lambda x: -x[1])
        rep.append('  events with the largest share of water missed by all three: ' + ', '.join(f'{e} {100 * s:.0f}%' for e, s in ev_share[:6]))
    (HERE / 'shared_report.txt').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
