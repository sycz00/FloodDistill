"""U1 (Stage 8 / novelty ledger row 3): distil the teacher into a closed-form, exactly gain-invariant spectral index.

s(x) = sum_b w_b log(rho_b) + c with sum_b w_b = 0  =>  s(k*rho) = s(rho) for any scalar gain k.
Fit: logistic regression on TRAIN teacher hard masks only (N=2,500 cohort, 800 valid pixels/image, seed 1729).
Controls: unconstrained log-linear fit (no invariance) and MNDWI>0 recomputed with the same code
(must reproduce the fixed comparator counts -> checks the reflectance inversion rho = z*std + mean).
Readouts: S11-val (selection split), then S11-test / STURM / WF with the frozen terminal counting loop. CPU only.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

capsule, spec_path, out_path = (str(Path(p).resolve()) for p in sys.argv[1:4])
sys.path.insert(0, capsule)
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import scripts.compression_jobs as cj
from data.sen1floods11 import NORM_MEAN, NORM_STD
from evaluation.revision_eval import scores

torch.set_num_threads(8)
start = time.monotonic()
spec = json.loads(Path(spec_path).read_text()); spec['workers'] = 0
MEAN = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); STD = torch.tensor(NORM_STD).view(1, 6, 1, 1)
FLOOR = 1e-4


def logrho(z):
    return torch.log((z * STD + MEAN).clamp_min(FLOOR))


# ---- fit on TRAIN (teacher hard masks) ----
ds = cj.pseudo(spec, augment=False)
rng = np.random.default_rng(1729)
X, Y = [], []
for i in range(len(ds)):
    item = ds[i]
    f = logrho(item['image'].float().unsqueeze(0))[0].reshape(6, -1).T.numpy()
    m = item['mask'].reshape(-1).numpy()
    valid = np.flatnonzero(m >= 0)
    pick = rng.choice(valid, size=min(800, len(valid)), replace=False)
    X.append(f[pick]); Y.append(m[pick])
X, Y = np.concatenate(X), np.concatenate(Y)
Xc = X - X.mean(1, keepdims=True)                         # zero-sum parametrisation
fits = {}
for name, feats in (('U1_zero_sum', Xc), ('U1b_unconstrained', X)):
    lr = LogisticRegression(C=1e4, max_iter=2000).fit(feats, Y)
    w = lr.coef_[0].astype(np.float64); c = float(lr.intercept_[0])
    if name == 'U1_zero_sum':
        w = w - w.mean()                                   # identical predictions, exact zero sum
    fits[name] = {'w': w.tolist(), 'c': c, 'train_pixels': int(len(Y)), 'train_water_fraction': float(Y.mean())}
print('fits', {k: [round(x, 3) for x in v['w']] + [round(v['c'], 3)] for k, v in fits.items()}, flush=True)


def predict(name, z):
    f = logrho(z)
    if name == 'MNDWI>0':
        g, s = z[:, 1] * STD[0, 1] + MEAN[0, 1], z[:, 4] * STD[0, 4] + MEAN[0, 4]
        return ((g - s) / (g + s + 1e-12) > 0).long()
    w = torch.tensor(fits[name]['w'], dtype=torch.float32).view(1, 6, 1, 1)
    return ((f * w).sum(1) + fits[name]['c'] > 0).long()


NAMES = list(fits) + ['MNDWI>0']
# ---- S11-val ----
from evaluation.revision_native_border import confusion
val, _ = cj.loaders(spec)
valcells = {n: np.zeros((2, 3), dtype=np.int64) for n in NAMES}
for batch in val:
    for n in NAMES:
        p = predict(n, batch['image'].float()).numpy()
        for j in range(len(p)):
            valcells[n] += confusion(p[j], batch['mask'][j].numpy())
results = {n: {'s11_val': scores(valcells[n])} for n in NAMES}
print('val', {n: round(100 * results[n]['s11_val']['mIoU'], 2) for n in NAMES}, flush=True)

# ---- external (copied counting loop of the frozen terminal job) ----
from data.ood_datasets import STURMFloodDataset, WorldFloodsDataset
from data.channel_adapter import sturm_s2_adapter
from evaluation import perscene_eval as pe
from evaluation import revision_native_border as rb
from evaluation.revision_eval import NamedDataset
pe.STURM = spec['roots']['sturm']; map_of, event_of = pe.sturm_maps()
_, test = cj.loaders(spec)
s11 = NamedDataset(test.dataset, [Path(q).name for q in test.dataset.image_files])
sturm = STURMFloodDataset(spec['roots']['sturm'], sensor='S2', img_size=512, normalize='original', class_mapping='paper')
wf = WorldFloodsDataset(spec['roots']['wf'], split='test', img_size=512, tile_size=512, normalize='s11', permanent_water_as='ignore')
adapter = sturm_s2_adapter(target_channels=6)
Z = lambda: np.zeros((2, 3), dtype=np.int64)
with torch.no_grad():
    for bench, dset, bs in (('s11', s11, 4), ('sturm', sturm, 8), ('wf', wf, 4)):
        maps = {n: defaultdict(Z) for n in NAMES}; events = {n: defaultdict(Z) for n in NAMES}
        for begin in range(0, len(dset), bs):
            ids = list(range(begin, min(begin + bs, len(dset))))
            items = [rb.native_item(dset, i) if bench == 'wf' else dset[i] for i in ids]
            inputs = torch.stack([item['image'] for item in items]).float()
            if bench == 'sturm':
                inputs = adapter(inputs)
            preds = {n: predict(n, inputs).numpy() for n in NAMES}
            for k, (i, item) in enumerate(zip(ids, items)):
                filename = Path(item['filename']).name
                mask = item['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else mask.copy()
                if bench == 'sturm':
                    _, invalid = rb.sturm_invalid(dset, i); mask[invalid] = -1
                    map_id, event = map_of[filename], event_of[filename]
                elif bench == 'wf':
                    map_id = filename.split('#')[0]; event = map_id.split('_')[0]
                else:
                    map_id = filename; event = filename.split('_')[0]
                for n in NAMES:
                    pred = preds[n][k][:item['height'], :item['width']] if bench == 'wf' else preds[n][k]
                    cell = rb.confusion(pred, mask); maps[n][map_id] += cell; events[n][event] += cell
        for n in NAMES:
            results[n][bench] = {'micro': scores(sum(maps[n].values())),
                                 'event': {k: v.tolist() for k, v in events[n].items()}, 'n_maps': len(maps[n])}
        print(bench, {n: round(100 * results[n][bench]['micro']['mIoU'], 2) for n in NAMES},
              round(time.monotonic() - start), 's', flush=True)
Path(out_path).write_text(json.dumps({'status': 'complete', 'fits': fits, 'results': results, 'floor': FLOOR,
                                      'elapsed_seconds': time.monotonic() - start,
                                      'data_roles': 'fit TRAIN teacher hard masks; S11-val readout; external secondary'}, indent=1))
