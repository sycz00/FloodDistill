#!/usr/bin/env python3
"""Where does the teacher fail, and can a 10 m physical cue repair it at the predicted edge?

Read-only, CPU-only; same inputs and checks as probe_cpu.py (S11 val/test, STURM,
WF test; never WorldFloods-v2 train/val). Per pixel it stratifies teacher (cached
V7 hard masks), R4 student (four seeds) and MNDWI>0 errors by:
  - distance to the reference water/land edge (pixels, 10 m) and water-body
    half-width (thin channels), from the reference mask (diagnosis only);
  - distance to cloud (WF test only, from the WF cloud band gt[0] == 2);
  - simple spectral pixel types computed from the input bands.
It also scores 'edge-band refinement': inside a band of k pixels around the
PREDICTED edge, each pixel is re-decided by MNDWI (global theta, or a local
threshold halfway between the mean MNDWI of nearby predicted-water and
predicted-land interiors); outside the band the prediction is kept. The band and
rule use only the input and the prediction, never the reference.
Writes failure_strata.json, refine_events.json, failure_report.txt.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_cpu as pc  # noqa: E402

KS = [1, 2, 3, 5]
THETAS = [-0.1, 0.0, 0.1, 0.2, 'local']
D_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 10**9)]
W_BINS = [('thin<=2px', 0, 2), ('mid2-5px', 2, 5), ('thick>5px', 5, 10**9)]
C_BINS = [('cloud1-5', 0, 5), ('cloud6-20', 5, 20), ('cloud21-50', 20, 50), ('cloud>50', 50, 10**9), ('no-cloud', None, None)]


def edges_and_width(y):
    import numpy as np
    from scipy import ndimage
    water, land = y == 1, y == 0
    d_w = ndimage.distance_transform_edt(~land) if land.any() else np.full(y.shape, np.inf)   # water px -> nearest land
    d_l = ndimage.distance_transform_edt(~water) if water.any() else np.full(y.shape, np.inf)  # land px -> nearest water
    half = np.zeros(y.shape, dtype=np.float32)
    if water.any() and land.any():
        lab, n = ndimage.label(water, structure=np.ones((3, 3), bool))
        mx = ndimage.maximum(d_w, lab, index=np.arange(1, n + 1))
        half = np.concatenate([[0], np.asarray(mx, dtype=np.float32)])[lab]
    return d_w, d_l, half


def edge_dist(p):
    """Distance (px) of every pixel to the opposite PREDICTED class; inf if the prediction is uniform."""
    import numpy as np
    from scipy import ndimage
    if p.all() or (~p).all():
        return np.full(p.shape, np.inf)
    return np.where(p, ndimage.distance_transform_edt(p), ndimage.distance_transform_edt(~p))


def refine(p, m, a, dist, k, theta, win=31):
    import numpy as np
    from scipy import ndimage
    band = dist <= k
    if not band.any():
        return p
    if theta == 'local':
        wi = (p & ~band & a).astype(np.float32); li = (~p & ~band & a).astype(np.float32)
        mm = np.where(a, m, 0).astype(np.float32)
        sw = ndimage.uniform_filter(mm * wi, win); cw = ndimage.uniform_filter(wi, win)
        sl = ndimage.uniform_filter(mm * li, win); cl = ndimage.uniform_filter(li, win)
        ok = (cw > 1e-3) & (cl > 1e-3)
        th = np.where(ok, 0.5 * (sw / np.maximum(cw, 1e-6) + sl / np.maximum(cl, 1e-6)), 0.0)
    else:
        th = theta
    out = p.copy()
    out[band] = ((m > th) & a)[band]
    return out


def cnt(pred, y):
    import numpy as np
    v = y != -1
    p, w = pred[v], y[v] == 1
    return [int(np.count_nonzero(p & w)), int(np.count_nonzero(p & ~w)), int(np.count_nonzero(~p & w)), int(np.count_nonzero(~p & ~w))]


def work(job):
    import numpy as np
    y = job['mask']; h, w = y.shape
    crop = lambda x: x[:h, :w]
    m = crop(job['mndwi'].astype(np.float32)); a = crop(job['avail']); refl = crop(job['refl'].astype(np.float32).transpose(1, 2, 0)).transpose(2, 0, 1)
    preds = {f's{s}': crop(mg.astype(np.float32) > 0) for s, mg in job['margins'].items()}
    if job['teacher'] is not None:
        preds['T'] = crop(job['teacher'])
    preds['M'] = (m > 0) & a
    out = dict(event=job['event'], dataset=job['dataset'], strata={}, refine={})
    valid = y != -1
    # --- strata (reference-based diagnosis)
    d_w, d_l, half = edges_and_width(y)
    vis = refl[:3].mean(axis=0); nir, sw1 = refl[3], refl[4]
    ndvi = (nir - refl[2]) / np.maximum(nir + refl[2], 1e-6)
    strata = {}
    for lo, hi in D_BINS:
        strata[f'edge{lo}-{hi if hi < 10**8 else "inf"}'] = (((y == 1) & (d_w >= lo) & (d_w <= hi)) | ((y == 0) & (d_l >= lo) & (d_l <= hi)))
    strata['edge-none'] = valid & np.isinf(np.where(y == 1, d_w, d_l))
    for name, lo, hi in W_BINS:
        strata['water-' + name] = (y == 1) & (half > lo) & (half <= hi)
    strata['type-dark'] = valid & (vis < 0.06) & (nir < 0.08) & (sw1 < 0.06)
    strata['type-bright'] = valid & (vis > 0.20)
    strata['type-veg'] = valid & (ndvi > 0.4)
    strata['type-other'] = valid & ~strata['type-dark'] & ~strata['type-bright'] & ~strata['type-veg']
    if job.get('cloud') is not None:
        from scipy import ndimage
        cl = crop(job['cloud'])
        if cl.any():
            dc = ndimage.distance_transform_edt(~cl)
            for name, lo, hi in C_BINS[:-1]:
                strata[name] = valid & (dc > lo) & (dc <= hi)
        else:
            strata['no-cloud'] = valid.copy()
    for sname, sel in strata.items():
        if sel.any():
            out['strata'][sname] = {k: cnt(p & sel, np.where(sel, y, -1)) for k, p in preds.items()}
    # --- edge-band refinement (prediction + input only)
    for k_name, p in preds.items():
        if k_name == 'M':
            continue
        out['refine'][f'{k_name}|base'] = cnt(p, y)
        dist = edge_dist(p)
        for k in KS:
            for th in THETAS:
                out['refine'][f'{k_name}|k{k}|{th}'] = cnt(refine(p, m, a, dist, k, th), y)
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
    from data.channel_adapter import sturm_s2_adapter
    from evaluation import revision_native_border as rb
    from evaluation.revision_eval import NamedDataset
    import scripts.compression_jobs as cj
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    models = {}
    for e in pc.MODELS:
        assert pc.sha(e['checkpoint']) == e['sha256']
        payload = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)
        mdl = build_model('R4', 'lkd', spec['rungs_dir']); mdl.load_state_dict(payload['model'], strict=True)
        models[e['seed']] = mdl.eval()
    sets, map_of, event_of = ev.datasets(spec)
    val, _ = cj.loaders(spec)
    sets = [('s11val', NamedDataset(val.dataset, [Path(p).name for p in val.dataset.image_files]), 4)] + sets
    refs = json.loads((pc.V7 / 'reference_cache' / 'manifest.json').read_text())
    adapter = sturm_s2_adapter(target_channels=6)
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    pool = ProcessPoolExecutor(12)
    strata_tot, refine_ev = {}, {}
    for name, ds, batch in sets:
        t0 = time.monotonic()
        cache = np.load(refs['datasets'][name]['path'], mmap_mode='r') if name != 's11val' else None
        futures = []
        N = min(int(os.environ.get('PROBE_LIMIT', len(ds))), len(ds))
        for begin in range(0, N, batch):
            ids = list(range(begin, min(begin + batch, N)))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            if name == 'sturm':
                x = adapter(x)
            with torch.no_grad():
                margins = {s: (lambda z: z[:, 1] - z[:, 0])(mdl(x)).numpy() for s, mdl in models.items()}
            refl = (x * sd + mu)
            g, s1 = refl[:, 1], refl[:, 4]
            avail = (torch.isfinite(g) & torch.isfinite(s1) & (g >= 0) & (s1 >= 0) & (g + s1 > 1e-6))
            mndwi = torch.where(avail, (g - s1) / (g + s1).clamp_min(1e-6), torch.zeros_like(g)).clamp(-1, 1)
            for local, (i, it) in enumerate(zip(ids, items)):
                fn = Path(it['filename']).name
                mask = it['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else np.array(mask, copy=True)
                if name == 'sturm':
                    _, invalid = rb.sturm_invalid(ds, i); mask[invalid] = -1
                    event = event_of[fn]
                elif name == 'wf':
                    event = fn.split('#')[0].split('_')[0]
                else:
                    event = fn.split('_')[0]
                teacher = cloud = None
                if cache is not None:
                    row = refs['datasets'][name]['rows'][i]
                    assert row['filename'] == fn and row['mask_sha256'] == hashlib.sha256(mask.tobytes()).hexdigest()
                    teacher = np.unpackbits(cache[i], axis=1).reshape(3, 512, 512)[0].astype(bool)
                if name == 'wf':
                    ip, lp, win = ds.samples[i]
                    c, r, w_, h_ = win
                    with rasterio.open(lp) as src:
                        gt0 = src.read(1, window=Window(c, r, w_, h_))
                    cloud = np.zeros((512, 512), bool); cloud[:h_, :w_] = gt0 == 2
                futures.append(pool.submit(work, dict(
                    dataset=name, event=event, mask=mask.astype(np.int8),
                    margins={s: mg[local].astype(np.float16) for s, mg in margins.items()},
                    mndwi=mndwi[local].numpy().astype(np.float16), avail=avail[local].numpy(),
                    refl=refl[local].numpy().astype(np.float16), teacher=teacher, cloud=cloud)))
        st_acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
        rf_acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
        for f in futures:
            r = f.result()
            for sname, d in r['strata'].items():
                for k, c4 in d.items():
                    acc = st_acc[sname][k]
                    for q in range(4): acc[q] += c4[q]
            for key, c4 in r['refine'].items():
                acc = rf_acc[r['event']][key]
                for q in range(4): acc[q] += c4[q]
        strata_tot[name] = {s: dict(v) for s, v in st_acc.items()}
        refine_ev[name] = {e: dict(v) for e, v in rf_acc.items()}
        print(name, 'done', round(time.monotonic() - t0, 1), 's', flush=True)
    out = Path(os.environ.get('PROBE_OUT', HERE))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'failure_strata.json').write_text(json.dumps(strata_tot))
    (out / 'refine_events.json').write_text(json.dumps(refine_ev))


if __name__ == '__main__':
    main()
