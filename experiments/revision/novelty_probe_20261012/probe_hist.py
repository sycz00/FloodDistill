#!/usr/bin/env python3
"""CPU-only second pass: per-image histograms for label-free operating-point rules.

Read-only on every input; same datasets, R4 parents, MNDWI definition and
teacher cache checks as probe_cpu.py. For every image and seed it stores a
3-D count histogram over

  student margin bin  (0.5-wide bins on [-12, 12] plus two overflow bins = 50)
  MNDWI bin           (0.1-wide bins on [-1, 1] = 20, plus 'rule unavailable' = 21)
  pixel class         (0 = not reference-valid, 1 = reference land, 2 = reference water)

Class 0 keeps every image pixel the deployed model would see, so label-free
rules can use all pixels while scores use only reference-valid ones. WF native
zero padding is excluded entirely (it is not imagery). Per dataset it also
stores mean six-band reflectance by teacher/student/MNDWI/reference category
(spectral diagnostic for the 'missed water' pixels).

Run: CUDA_VISIBLE_DEVICES= /home/schmalstieg/miniconda3/envs/esi/bin/python probe_hist.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_cpu as pc  # noqa: E402  (constants and checkpoint list only)

M_EDGES = [-12 + 0.5 * i for i in range(49)]          # 48 inner bins -> indices 1..48, 0 and 49 overflow
CATS = ['Tw', 'Dw', 'Dl', 'Bw', 'Fl', 'Ll']           # see category() below


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--out', type=Path, default=HERE / 'probe_hist')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = pc.SPEC
    sys.dont_write_bytecode = True
    sys.path.insert(0, spec['capsule']); sys.path.insert(0, str(pc.V7))
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    assert not torch.cuda.is_available()
    import evaluate_v7 as ev
    from train_v7 import build_model
    from data.channel_adapter import sturm_s2_adapter
    from evaluation import revision_native_border as rb
    from evaluation.revision_eval import NamedDataset
    import scripts.compression_jobs as cj
    from data.sen1floods11 import NORM_MEAN, NORM_STD

    models = {}
    for e in pc.MODELS:
        assert pc.sha(e['checkpoint']) == e['sha256'], e['id']
        payload = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)
        mdl = build_model('R4', 'lkd', spec['rungs_dir']); mdl.load_state_dict(payload['model'], strict=True)
        models[e['seed']] = mdl.eval()
    seeds = sorted(models)
    sets, map_of, event_of = ev.datasets(spec)
    val, _ = cj.loaders(spec)
    sets = [('s11val', NamedDataset(val.dataset, [Path(p).name for p in val.dataset.image_files]), 4)] + sets
    refs = json.loads((pc.V7 / 'reference_cache' / 'manifest.json').read_text())
    adapter = sturm_s2_adapter(target_channels=6)
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    edges = np.array(M_EDGES, dtype=np.float32)

    for name, ds, batch in sets:
        start = time.monotonic()
        n = len(ds) if a.limit is None else min(a.limit, len(ds))
        cache = np.load(refs['datasets'][name]['path'], mmap_mode='r') if name != 's11val' else None
        H = np.zeros((n, len(seeds), 50, 21, 3), dtype=np.int32)
        meta = []
        spec_sum = {k: np.zeros((len(CATS), 6), dtype=np.float64) for k in ['student42'] + (['teacher'] if cache is not None else [])}
        spec_cnt = {k: np.zeros(len(CATS), dtype=np.int64) for k in spec_sum}
        for begin in range(0, n, batch):
            ids = list(range(begin, min(begin + batch, n)))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            x = torch.stack([it['image'] for it in items]).float()
            if name == 'sturm':
                x = adapter(x)
            with torch.no_grad():
                margins = {s: (lambda z: z[:, 1] - z[:, 0])(models[s](x)).numpy() for s in seeds}
            refl = (x * sd + mu)
            green, swir = refl[:, 1], refl[:, 4]
            avail = (torch.isfinite(green) & torch.isfinite(swir) & (green >= 0) & (swir >= 0) & (green + swir > 1e-6)).numpy()
            mndwi = torch.where(torch.from_numpy(avail), (green - swir) / (green + swir).clamp_min(1e-6), torch.zeros_like(green)).clamp(-1, 1).numpy()
            refl = refl.numpy()
            for local, (i, it) in enumerate(zip(ids, items)):
                filename = Path(it['filename']).name
                mask = it['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else np.array(mask, copy=True)
                if name == 'sturm':
                    _, invalid = rb.sturm_invalid(ds, i); mask[invalid] = -1
                    map_id, event, omit = map_of[filename], event_of[filename], filename in spec['excluded_sturm_chips']
                elif name == 'wf':
                    map_id = filename.split('#')[0]; event = map_id.split('_')[0]; omit = map_id in spec['excluded_wf_maps']
                else:
                    map_id = filename; event = filename.split('_')[0]; omit = False
                h, w = mask.shape
                full = np.full((512, 512), -1, dtype=np.int64); full[:h, :w] = mask
                inside = np.zeros((512, 512), dtype=bool); inside[:h, :w] = True     # excludes WF zero padding
                cls = np.where(full == -1, 0, full + 1)
                mb = np.where(avail[local], np.clip(((mndwi[local] + 1) / 0.1).astype(np.int64), 0, 19), 20)
                teacher = None
                if cache is not None:
                    row = refs['datasets'][name]['rows'][i]
                    assert row['filename'] == filename and row['mask_sha256'] == hashlib.sha256(mask.tobytes()).hexdigest()
                    teacher = np.unpackbits(cache[i], axis=1).reshape(3, 512, 512)[0].astype(bool)
                for k, s in enumerate(seeds):
                    gb = np.searchsorted(edges, margins[s][local], side='right')
                    flat = (gb[inside] * 21 + mb[inside]) * 3 + cls[inside]
                    H[i, k] = np.bincount(flat, minlength=50 * 21 * 3).reshape(50, 21, 3)
                valid = inside & (full != -1)
                m_pos = (mndwi[local] > 0) & avail[local]
                y = full == 1
                for key, pred in (('student42', margins[42][local] > 0), ('teacher', teacher)):
                    if pred is None or key not in spec_sum:
                        continue
                    cats = [pred & y, ~pred & m_pos & y, ~pred & m_pos & ~y, ~pred & ~m_pos & y, pred & ~y, ~pred & ~m_pos & ~y]
                    for c, sel in enumerate(cats):
                        sel = sel & valid
                        cnt = int(sel.sum())
                        if cnt:
                            spec_cnt[key][c] += cnt
                            spec_sum[key][c] += refl[local][:, sel].sum(axis=1)
                meta.append(dict(name=filename, event=event, map=map_id, omit=bool(omit)))
            if begin % (batch * 100) == 0:
                print(name, begin, '/', n, round(time.monotonic() - start, 1), 's', flush=True)
        np.savez_compressed(a.out / f'{name}.npz', H=H, seeds=np.array(seeds), edges=edges)
        (a.out / f'{name}_meta.json').write_text(json.dumps(meta))
        spectra = {k: {CATS[c]: dict(pixels=int(spec_cnt[k][c]),
                                    mean_reflectance=(spec_sum[k][c] / spec_cnt[k][c]).round(5).tolist() if spec_cnt[k][c] else None)
                       for c in range(len(CATS))} for k in spec_sum}
        (a.out / f'{name}_spectra.json').write_text(json.dumps(spectra, indent=1))
        print(name, 'done', n, round(time.monotonic() - start, 1), 's', flush=True)


if __name__ == '__main__':
    main()
