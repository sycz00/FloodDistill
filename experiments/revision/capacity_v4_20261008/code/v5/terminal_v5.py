"""External readout (S11-test, STURM, WF) for many checkpoints in one data pass (opt-in v5).

The per-image scoring loop is copied from the frozen capsule's scripts/compression_jobs.terminal
(same datasets, STURM adapter and invalid mask, WF native crop, overlap exclusions, confusion,
expected counts). Differences: no artifact/latency measurement, several models per batch, and
QAT checkpoints are scored under the adopted integer-domain simulator (or with QDQ bypassed).
External numbers are frozen secondary readouts: nothing here feeds back into training choices.

--models JSON: [{"id", "graph", "kind": "fp"|"qat", "checkpoint", "sha256", "qdq": true|false}]
Writes <out-dir>/<id>.json (skips ids whose output already exists with the same checkpoint hash).
"""
import argparse
import contextlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
for name in ('capsule', 'spec', 'out-dir'):
    p.add_argument('--' + name, required=True)
p.add_argument('--models', help='explicit model list JSON')
p.add_argument('--receipts', nargs='*', default=[], help='training receipts (train_v5, lowbit_v5, Stage 2)')
p.add_argument('--done', help='summary receipt written at the end')
a = p.parse_args()
for _k in ('capsule', 'spec', 'models', 'out_dir', 'done'):
    if getattr(a, _k):
        setattr(a, _k, str(Path(getattr(a, _k)).resolve()))
a.receipts = [str(Path(r).resolve()) for r in a.receipts]  # perscene_eval import changes cwd
sys.path.insert(0, a.capsule); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v3'))
import numpy as np
import torch
import scripts.compression_jobs as cj
from experiments.revision_queue import atomic_json, file_hash, now
from models.compression_quant import load_compression_qat, quantization_mode
from evaluation.revision_eval import scores
import rungs

start = time.monotonic()
spec = json.loads(Path(a.spec).read_text())
out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)


def from_receipt(path):
    r = json.loads(Path(path).read_text())
    if r.get('status') != 'complete' or r.get('outcome') != 'completed':
        return []
    rid = Path(path).stem
    if 'qdq' in r:                                     # lowbit_v5 receipt
        if r['qdq'] == 'on':
            return [dict(id=rid, graph=r['graph'], kind='qat', qdq=True, checkpoint=r['best_path'], sha256=r['best_sha256'])]
        return [dict(id=rid, graph=r['graph'], kind='fp', checkpoint=r['fp_stripped_path'], sha256=r['fp_stripped_sha256'])]
    return [dict(id=rid, graph=r['graph'], kind='fp', checkpoint=r['best_path'], sha256=r['best_sha256'])]


requested = json.loads(Path(a.models).read_text()) if a.models else []
missing = []
for path in a.receipts:
    if Path(path).exists():
        requested += from_receipt(path)
    else:
        missing.append(path)
scored = {}
for f in out_dir.glob('*.json'):
    r = json.loads(f.read_text())
    if r.get('terminal_results') and not r.get('alias_of'):
        scored[r['checkpoint_sha256']] = f
entries, seen, late_alias = [], {}, []
for e in requested:
    target = out_dir / f"{e['id']}.json"
    if target.exists() and json.loads(target.read_text()).get('checkpoint_sha256') == e['sha256']:
        continue
    if e['sha256'] in scored:          # identical checkpoint already scored under another id (reuse receipts)
        src = json.loads(scored[e['sha256']].read_text())
        atomic_json(target, {**src, 'id': e['id'], 'alias_of': src['id']})
        continue
    if e['sha256'] in seen:
        late_alias.append((e, seen[e['sha256']])); continue
    seen[e['sha256']] = e['id']
    if file_hash(e['checkpoint']) != e['sha256']:
        raise RuntimeError(f"checkpoint hash changed: {e['id']}")
    state = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)['model']
    if e['kind'] == 'fp':
        model = rungs.build(e['graph']); model.load_state_dict(state, strict=True)
    else:
        model = load_compression_qat(rungs.build(e['graph']), state)
    entries.append((e, model.cuda().eval()))
print(f'{len(entries)} models to score; {len(missing)} receipts missing', flush=True)


def finish():
    if a.done:
        atomic_json(a.done, {'status': 'complete', 'outcome': 'completed', 'scored_now': [e['id'] for e, _ in entries],
                             'requested': [e['id'] for e in requested], 'missing_receipts': missing, 'at': now()})


if not entries:
    finish(); sys.exit(0)

from data.ood_datasets import STURMFloodDataset, WorldFloodsDataset
from data.channel_adapter import sturm_s2_adapter
from evaluation import perscene_eval as pe
from evaluation import revision_native_border as rb
from evaluation.revision_eval import NamedDataset
from simulator_v4 import simulator

pe.STURM = spec['roots']['sturm']; map_of, event_of = pe.sturm_maps()
_, test = cj.loaders(spec)
s11 = NamedDataset(test.dataset, [Path(q).name for q in test.dataset.image_files])
sturm = STURMFloodDataset(spec['roots']['sturm'], sensor='S2', img_size=512, normalize='original', class_mapping='paper')
wf = WorldFloodsDataset(spec['roots']['wf'], split='test', img_size=512, tile_size=512,
                        normalize='s11', permanent_water_as='ignore')
assert (len(s11), len(sturm), len(wf)) == (90, 2675, 785)
plan = json.loads(Path(spec['overlap_plan']).read_text())
assert spec['excluded_sturm_chips'] == plan['exclude_complete_STURM_chips']
assert spec['excluded_wf_maps'] == plan['exclude_complete_WorldFloods_maps']
excluded_sturm, excluded_wf = set(spec['excluded_sturm_chips']), set(spec['excluded_wf_maps'])
adapter = sturm_s2_adapter(target_channels=6).cuda()
Z = lambda: np.zeros((2, 3), dtype=np.int64)
acc = {e['id']: {} for e, _ in entries}


def forward(e, model, inputs):
    if e['kind'] == 'fp':
        return model(inputs)
    with contextlib.ExitStack() as stack:
        if e['qdq']:
            stack.enter_context(simulator(model, 'integer'))
        stack.enter_context(quantization_mode(model, bool(e['qdq'])))
        return model(inputs)


with torch.no_grad():
    for name, ds, bs in (('s11', s11, 4), ('sturm', sturm, 8), ('wf', wf, 4)):
        st = {e['id']: dict(images={}, maps=defaultdict(Z), events=defaultdict(Z), xmaps=defaultdict(Z),
                            xevents=defaultdict(Z), included=0) for e, _ in entries}
        for begin in range(0, len(ds), bs):
            ids = list(range(begin, min(begin + bs, len(ds))))
            items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
            inputs = torch.stack([item['image'] for item in items]).cuda().float()
            if name == 'sturm':
                inputs = adapter(inputs)
            meta = []
            for i, item in zip(ids, items):
                filename = Path(item['filename']).name
                mask = item['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else mask.copy()
                if name == 'sturm':
                    _, invalid = rb.sturm_invalid(ds, i); mask[invalid] = -1
                    map_id, event = map_of[filename], event_of[filename]; omit = filename in excluded_sturm
                elif name == 'wf':
                    map_id = filename.split('#')[0]; event = map_id.split('_')[0]; omit = map_id in excluded_wf
                else:
                    map_id = filename; event = filename.split('_')[0]; omit = False
                meta.append((filename, mask, map_id, event, omit, item))
            for e, model in entries:
                logits = forward(e, model, inputs); cj.finite_output(model, (inputs,), logits)
                predictions = logits.argmax(1).cpu().numpy()
                s = st[e['id']]
                for (filename, mask, map_id, event, omit, item), pred in zip(meta, predictions):
                    if name == 'wf':
                        pred = pred[:item['height'], :item['width']]
                    cell = rb.confusion(pred, mask); s['images'][filename] = cell.tolist()
                    s['maps'][map_id] += cell; s['events'][event] += cell
                    if not omit:
                        s['xmaps'][map_id] += cell; s['xevents'][event] += cell; s['included'] += 1

        def pack(mm, ee):
            return {'micro': scores(sum(mm.values())), 'map': {k: v.tolist() for k, v in mm.items()},
                    'event': {k: v.tolist() for k, v in ee.items()}, 'n_maps': len(mm), 'n_events': len(ee)}
        expected = {'s11': (90, 10, 90, 90), 'sturm': (46, 29, 2675, 2672), 'wf': (18, 11, 785, 733)}[name]
        for e, _ in entries:
            s = st[e['id']]
            assert (len(s['maps']), len(s['events']), len(s['images']), s['included']) == expected
            r = {'full': pack(s['maps'], s['events']), 'known_overlap_excluded': pack(s['xmaps'], s['xevents']),
                 'per_image_counts': s['images'], 'n_images': len(ds), 'protocol': spec['protocols'][name]}
            r['known_overlap_excluded']['n_images'] = s['included']
            acc[e['id']][name] = r
        print(name, 'done', round(time.monotonic() - start), 's', flush=True)

for e, model in entries:
    atomic_json(out_dir / f"{e['id']}.json", {
        'status': 'complete', 'outcome': 'completed', 'completed_at': now(), 'id': e['id'], 'graph': e['graph'],
        'kind': e['kind'], 'qdq': e.get('qdq'), 'checkpoint_path': e['checkpoint'], 'checkpoint_sha256': e['sha256'],
        'parameters': sum(q.numel() for q in model.parameters()), 'terminal_results': acc[e['id']],
        'simulator': 'integer_domain_v4' if e['kind'] == 'qat' and e.get('qdq') else None,
        'secondary_readout_only': True, 'elapsed_seconds_shared_pass': time.monotonic() - start})
for e, source in late_alias:
    src = json.loads((out_dir / f'{source}.json').read_text())
    atomic_json(out_dir / f"{e['id']}.json", {**src, 'id': e['id'], 'alias_of': source})
finish()
