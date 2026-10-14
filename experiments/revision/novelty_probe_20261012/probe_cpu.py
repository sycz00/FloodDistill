#!/usr/bin/env python3
"""CPU-only feasibility probe: can a spectral index recover water the R4 student misses?

Read-only on every input. Runs the four selected R4 parents (capacity v5/v7,
seeds 42/64/91/107) on S11 validation (source), S11 test, STURM and WorldFloods
(research-exposed external sets) and records, per image and seed, the integer
confusion counts of a fixed grid of decoding rules:

  S@tau      student water-minus-land margin > tau      (operating-point control)
  SuM@th     S OR MNDWI > th, pixel-wise                (index override control)
  SvM@th     S AND (MNDWI > th or rule unavailable)      (index veto)
  SgM@th     hysteresis: MNDWI > th pixels are added only when 8-connected to S
  SgM@th+v   SgM@th followed by the SvM@-0.3 veto
  M@0        MNDWI > 0 rule alone; T and TgM@th use the cached teacher masks
             (external sets only, diagnostic: the teacher is not on board)

The MNDWI and rule-availability definition is the one in capacity_v7
evaluate_v7.rule_arrays; the probe re-checks its MNDWI>0 bits against the
cached V7 rule bits. Nothing here selects a setting: probe_analysis.py
selects on S11 validation only.

Run (about 30-40 min on 16 CPU threads):
  CUDA_VISIBLE_DEVICES= /home/schmalstieg/miniconda3/envs/esi/bin/python probe_cpu.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

HERE = Path(__file__).resolve().parent
REV = HERE.parent
V7 = REV / 'capacity_v7_20261011T210730Z'
SPEC = json.loads((V7 / 'training_spec.json').read_text())
MODELS = [m for m in json.loads((V7 / 'reuse_models.json').read_text()) if m['graph'] == 'R4']

TAUS = [-4.0, -2.0, -1.0, 1.0, 2.0]
UNION = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
VETO = [-0.5, -0.3, -0.2, -0.1, 0.0]
GROW = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
STRUCT = None


def sha(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            d.update(b)
    return d.hexdigest()


def counts(pred, mask):
    import numpy as np
    valid = mask != -1
    p, y = pred[valid], mask[valid] == 1
    tp = int(np.count_nonzero(p & y)); fp = int(np.count_nonzero(p & ~y))
    fn = int(np.count_nonzero(~p & y)); tn = int(np.count_nonzero(~p & ~y))
    return [tp, fp, fn, tn]


def grow(seed, cand):
    import numpy as np
    from scipy import ndimage
    region = seed | cand
    lab, n = ndimage.label(region, structure=np.ones((3, 3), dtype=bool))
    keep = np.zeros(n + 1, dtype=bool)
    keep[lab[seed]] = True
    keep[0] = False
    return keep[lab]


def variants(job):
    """All decoding rules for one image; arrays are full 512x512, mask is cropped (h, w)."""
    import numpy as np
    margins, mndwi, avail, teacher, mask = job['margins'], job['mndwi'], job['avail'], job['teacher'], job['mask']
    h, w = mask.shape
    crop = lambda a: a[:h, :w]
    out = {}
    m = mndwi.astype(np.float32)
    out['M@0'] = counts(crop((m > 0) & avail), mask)
    if teacher is not None:
        out['T'] = counts(crop(teacher), mask)
        for th in GROW:
            out[f'TgM@{th}'] = counts(crop(grow(teacher, (m > th) & avail)), mask)
    for seed, margin in margins.items():
        s = margin.astype(np.float32) > 0
        out[f's{seed}|S'] = counts(crop(s), mask)
        for tau in TAUS:
            out[f's{seed}|S@{tau}'] = counts(crop(margin.astype(np.float32) > tau), mask)
        for th in UNION:
            out[f's{seed}|SuM@{th}'] = counts(crop(s | ((m > th) & avail)), mask)
        for th in VETO:
            out[f's{seed}|SvM@{th}'] = counts(crop(s & ((m > th) | ~avail)), mask)
        veto = (m > -0.3) | ~avail
        for th in GROW:
            g = grow(s, (m > th) & avail)
            out[f's{seed}|SgM@{th}'] = counts(crop(g), mask)
            out[f's{seed}|SgM@{th}+v'] = counts(crop(g & veto), mask)
    return dict(name=job['name'], dataset=job['dataset'], event=job['event'], map=job['map'],
                omit=job['omit'], valid=int((mask != -1).sum()), counts=out,
                rule_bits_match=job.get('rule_bits_match'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=None, help='images per dataset (smoke)')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', type=Path, default=HERE / 'probe_counts')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sys.dont_write_bytecode = True
    sys.path.insert(0, SPEC['capsule'])
    sys.path.insert(0, str(V7))
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    assert not torch.cuda.is_available()
    import evaluate_v7 as ev                     # import-safe, CPU-only module
    from train_v7 import build_model
    from data.channel_adapter import sturm_s2_adapter
    from evaluation import revision_native_border as rb
    from evaluation.revision_eval import NamedDataset
    import scripts.compression_jobs as cj

    models = {}
    for e in MODELS:
        assert sha(e['checkpoint']) == e['sha256'], e['id']
        payload = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)
        model = build_model('R4', 'lkd', SPEC['rungs_dir'])
        model.load_state_dict(payload['model'], strict=True)
        models[e['seed']] = model.eval()
    print('models', sorted(models), sum(p.numel() for p in models[42].parameters()), 'parameters', flush=True)

    sets, map_of, event_of = ev.datasets(SPEC)
    val, _ = cj.loaders(SPEC)
    s11val = NamedDataset(val.dataset, [Path(p).name for p in val.dataset.image_files])
    sets = [('s11val', s11val, 4)] + sets
    refs = json.loads((V7 / 'reference_cache' / 'manifest.json').read_text())
    adapter = sturm_s2_adapter(target_channels=6)
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)

    pool = ProcessPoolExecutor(a.workers)
    receipt = dict(started=time.time(), models={str(k): e['sha256'] for k, e in zip(sorted(models), MODELS)},
                   spec=str(V7 / 'training_spec.json'), grid=dict(TAUS=TAUS, UNION=UNION, VETO=VETO, GROW=GROW))
    for name, ds, batch in sets:
        start = time.monotonic()
        n = len(ds) if a.limit is None else min(a.limit, len(ds))
        cache = None
        if name != 's11val':
            item = refs['datasets'][name]
            cache = np.load(item['path'], mmap_mode='r')
        futures = []
        for begin in range(0, n, batch):
            ids = list(range(begin, min(begin + batch, n)))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            if name == 'sturm':
                x = adapter(x)
            with torch.no_grad():
                margins = {seed: (lambda z: (z[:, 1] - z[:, 0]))(mdl(x)).numpy() for seed, mdl in models.items()}
            refl = x * sd + mu
            green, swir, nir = refl[:, 1], refl[:, 4], refl[:, 3]
            avail = torch.isfinite(green) & torch.isfinite(swir) & (green >= 0) & (swir >= 0) & (green + swir > 1e-6)
            mndwi = torch.where(avail, (green - swir) / (green + swir).clamp_min(1e-6), torch.zeros_like(green)).clamp(-1, 1)
            for local, (i, it) in enumerate(zip(ids, items)):
                filename = Path(it['filename']).name
                mask = it['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else np.array(mask, copy=True)
                if name == 'sturm':
                    _, invalid = rb.sturm_invalid(ds, i); mask[invalid] = -1
                    map_id, event, omit = map_of[filename], event_of[filename], filename in SPEC['excluded_sturm_chips']
                elif name == 'wf':
                    map_id = filename.split('#')[0]; event = map_id.split('_')[0]; omit = map_id in SPEC['excluded_wf_maps']
                else:
                    map_id = filename; event = filename.split('_')[0]; omit = False
                teacher, match = None, None
                if cache is not None:
                    row = refs['datasets'][name]['rows'][i]
                    assert row['filename'] == filename, (row['filename'], filename)
                    assert row['mask_sha256'] == hashlib.sha256(mask.tobytes()).hexdigest(), filename
                    bits = np.unpackbits(cache[i], axis=1).reshape(3, 512, 512).astype(bool)
                    teacher = bits[0]
                    mine = ((mndwi[local] > 0) & avail[local]).numpy()
                    match = bool(np.array_equal(mine, bits[1] & bits[2]) and np.array_equal(avail[local].numpy(), bits[2]))
                futures.append(pool.submit(variants, dict(
                    name=filename, dataset=name, event=event, map=map_id, omit=omit, mask=mask,
                    margins={s: m[local].astype(np.float16) for s, m in margins.items()},
                    mndwi=mndwi[local].numpy().astype(np.float16), avail=avail[local].numpy(),
                    teacher=teacher, rule_bits_match=match)))
            if begin % (batch * 50) == 0:
                print(name, begin, '/', n, round(time.monotonic() - start, 1), 's', flush=True)
        rows = [f.result() for f in futures]
        with gzip.open(a.out / f'{name}.json.gz', 'wt') as f:
            json.dump(rows, f)
        mism = sum(1 for r in rows if r['rule_bits_match'] is False)
        receipt[name] = dict(images=len(rows), seconds=time.monotonic() - start, rule_bit_mismatches=mism)
        print(name, 'done', len(rows), 'images', round(time.monotonic() - start, 1), 's; rule-bit mismatches', mism, flush=True)
    receipt['finished'] = time.time()
    (a.out / ('receipt_smoke.json' if a.limit else 'receipt.json')).write_text(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
