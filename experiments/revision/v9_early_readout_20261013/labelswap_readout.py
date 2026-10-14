#!/usr/bin/env python3
"""Early CPU readout of the V9 same-image label-swap B0 models (human vs teacher labels, 4 seeds) on EXPOSED sets only.

Read-only on V9 checkpoints and benchmark data; never opens WorldFloods-v2 train/val
(the V9 sealed set). Writes counts.json and report.txt here. Decode-time arm D uses
the V9 source-selected k=2, theta=0.1 (research_v9 source_edge_choice.json).

Per model and event: confusion counts at the deployed threshold and after D, plus
reference-edge strata (distance 1-2 px), thin water (half-width <= 2 px), interior
(> 5 px), and reference-dry windows flagged at >= 1 % predicted water.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
REV = HERE.parent
sys.path.insert(0, str(REV / 'novelty_probe_20261012'))
import probe_cpu as pc  # noqa: E402
import probe_failure as pf  # noqa: E402

V9 = REV / 'research_v9_recovery_20261013T075000Z'
CHOICE = json.loads((V9 / 'source_edge_choice.json').read_text())
K, THETA = int(CHOICE['k']), float(CHOICE['theta'])
MODELS = [m for m in json.loads((V9 / 'final_models_final03.json').read_text())['models'] if m['method'] in ('human', 'teacher')]


def cnt(pred, sel, y):
    import numpy as np
    v = sel & (y != -1)
    p, w = pred[v], y[v] == 1
    return [int(np.count_nonzero(p & w)), int(np.count_nonzero(p & ~w)), int(np.count_nonzero(~p & w)), int(np.count_nonzero(~p & ~w))]


def work(job):
    import numpy as np
    y = job['mask']; h, w = y.shape
    crop = lambda a: a[:h, :w]
    m = crop(job['mndwi'].astype(np.float32)); a = crop(job['avail'])
    d_w, d_l, half = pf.edges_and_width(y)
    dist = np.where(y == 1, d_w, d_l)
    strata = {'all': np.ones(y.shape, bool), 'edge1-2': dist <= 2, 'interior>5': (dist > 5) & np.isfinite(dist),
              'thin<=2': (y == 1) & (half > 0) & (half <= 2)}
    out = dict(event=job['event'], counts={}, dry=None)
    valid = y != -1
    is_dry = bool(valid.any() and not (y == 1).any())
    if is_dry:
        out['dry'] = {}
    for mid, mg in job['margins'].items():
        p = crop(mg.astype(np.float32)) > 0
        d = pf.refine(p, m, a, pf.edge_dist(p), K, THETA)
        for vname, pred in (('deployed', p), ('D', d)):
            for sname, sel in strata.items():
                out['counts'][f'{mid}|{vname}|{sname}'] = cnt(pred, sel, y)
            if is_dry:
                out['dry'][f'{mid}|{vname}'] = bool((pred & valid).sum() >= 0.01 * valid.sum())
    return out


def main():
    spec = pc.SPEC
    sys.dont_write_bytecode = True
    sys.path.insert(0, spec['capsule']); sys.path.insert(0, str(pc.V7))
    import hashlib
    import numpy as np
    import torch
    torch.set_num_threads(8)
    import evaluate_v7 as ev
    sys.path.insert(0, '/home/schmalstieg/repos/_deps/efficientvit')
    from models.student import build_student
    from data.channel_adapter import sturm_s2_adapter
    from evaluation import revision_native_border as rb
    from evaluation.revision_eval import NamedDataset
    import scripts.compression_jobs as cj
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    models = {}
    for e in MODELS:
        payload = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)
        mdl = build_student('efficientvit_b0', pretrained=False, in_channels=6, num_classes=2); mdl.load_state_dict(payload['model'], strict=True)
        models[e['id']] = mdl.eval()
    print('models', sorted(models), 'k', K, 'theta', THETA, flush=True)
    sets, map_of, event_of = ev.datasets(spec)
    val, _ = cj.loaders(spec)
    sets = [('s11val', NamedDataset(val.dataset, [Path(p).name for p in val.dataset.image_files]), 4)] + sets
    adapter = sturm_s2_adapter(target_channels=6)
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    pool = ProcessPoolExecutor(6)
    result = {}
    for name, ds, batch in sets:
        t0 = time.monotonic(); futs = []
        for begin in range(0, len(ds), batch):
            ids = list(range(begin, min(begin + batch, len(ds))))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            if name == 'sturm':
                x = adapter(x)
            with torch.no_grad():
                margins = {mid: (lambda z: z[:, 1] - z[:, 0])(mdl(x)).numpy() for mid, mdl in models.items()}
            refl = x * sd + mu
            g, s1 = refl[:, 1], refl[:, 4]
            avail = (torch.isfinite(g) & torch.isfinite(s1) & (g >= 0) & (s1 >= 0) & (g + s1 > 1e-6))
            mndwi = torch.where(avail, (g - s1) / (g + s1).clamp_min(1e-6), torch.zeros_like(g)).clamp(-1, 1)
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
                    margins={mid: mg[local].astype(np.float16) for mid, mg in margins.items()},
                    mndwi=mndwi[local].numpy().astype(np.float16), avail=avail[local].numpy())))
        ev_c = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])); dry = defaultdict(lambda: [0, 0])
        for f in futs:
            r = f.result()
            for k, v in r['counts'].items():
                for q in range(4): ev_c[r['event']][k][q] += v[q]
            if r['dry'] is not None:
                for k, flag in r['dry'].items():
                    dry[k][0] += 1; dry[k][1] += int(flag)
        result[name] = dict(events={e: dict(v) for e, v in ev_c.items()}, dry=dict(dry))
        print(name, 'done', round(time.monotonic() - t0, 1), 's', flush=True)
    (HERE / 'counts_labelswap.json').write_text(json.dumps(dict(k=K, theta=THETA, models=[m['id'] for m in MODELS], sets=result)))


if __name__ == '__main__':
    main()
