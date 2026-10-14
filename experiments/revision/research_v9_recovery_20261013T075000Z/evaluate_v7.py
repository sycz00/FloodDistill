"""Frozen 512-pixel terminal readouts and shared teacher/rule mechanism cache.

Importing this module is CPU-only and performs no filesystem mutation. External
readouts never feed a training gate. The scoring loop follows terminal_v5.py;
teacher/rule/student mechanism counts use the exact same reference pixels.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def scores(cell):
    """Fraction units; empty class union is undefined, omitted from mIoU."""
    iou = [row[0] / sum(row) if sum(row) else None for row in cell]
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    defined = [x for x in iou if x is not None]
    return dict(mIoU=sum(defined) / len(defined) if defined else None,
                water_IoU=iou[1], background_IoU=iou[0],
                water_precision=ratio(cell[1][0], cell[1][0] + cell[1][1]),
                water_recall=ratio(cell[1][0], cell[1][0] + cell[1][2]))


def setup(capsule, rungs_dir, backend_policy='inherited_terminal_v5'):
    # Capsule imports have historical cwd side effects, so CLI paths are resolved
    # before setup. Only import libraries; never import the old executable scripts.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(capsule).resolve()))
    sys.path.insert(0, str(Path(rungs_dir).resolve()))
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'mpl_cache'))
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
    import torch
    torch.set_num_threads(2)
    # terminal_v5 FP inference leaves backend flags at inherited process
    # defaults. Disabling cuDNN TF32 changes near-boundary predictions, even
    # though input/checkpoint dtypes remain float32. Do not amend that evaluator.
    if backend_policy=='deployment_tf32_off':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    elif backend_policy!='inherited_terminal_v5':
        raise ValueError('Unknown inference backend policy: '+backend_policy)
    return torch


def backend_record(policy='inherited_terminal_v5'):
    import torch
    return dict(policy=policy,input_dtype='torch.float32',checkpoint_parameter_dtype='torch.float32',
        source_dtype='registered six-band normalized float32',grad_context='torch.no_grad',
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        backend_environment={key:os.environ.get(key) for key in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUBLAS_WORKSPACE_CONFIG')},
        torch_version=torch.__version__,pytorch_cuda_runtime=torch.version.cuda)


def datasets(spec):
    import scripts.compression_jobs as cj
    from data.ood_datasets import STURMFloodDataset, WorldFloodsDataset
    from evaluation import perscene_eval as pe
    from evaluation.revision_eval import NamedDataset
    pe.STURM = spec['roots']['sturm']
    map_of, event_of = pe.sturm_maps()
    _, test = cj.loaders(spec)
    s11 = NamedDataset(test.dataset, [Path(q).name for q in test.dataset.image_files])
    sturm = STURMFloodDataset(spec['roots']['sturm'], sensor='S2', img_size=512,
                             normalize='original', class_mapping='paper')
    wf = WorldFloodsDataset(spec['roots']['wf'], split='test', img_size=512, tile_size=512,
                           normalize='s11', permanent_water_as='ignore')
    assert (len(s11), len(sturm), len(wf)) == (90, 2675, 785)
    plan = read(spec['overlap_plan'])
    assert spec['excluded_sturm_chips'] == plan['exclude_complete_STURM_chips']
    assert spec['excluded_wf_maps'] == plan['exclude_complete_WorldFloods_maps']
    return [('s11', s11, 4), ('sturm', sturm, 8), ('wf', wf, 4)], map_of, event_of


def batch_items(name, dataset, ids, map_of, event_of, spec):
    import numpy as np
    import torch
    from evaluation import revision_native_border as rb
    from data.channel_adapter import sturm_s2_adapter
    items = [rb.native_item(dataset, i) if name == 'wf' else dataset[i] for i in ids]
    inputs = torch.stack([item['image'] for item in items]).cuda().float()
    if name == 'sturm':
        inputs = sturm_s2_adapter(target_channels=6).cuda()(inputs)
    metadata = []
    for i, item in zip(ids, items):
        filename = Path(item['filename']).name
        mask = item['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else mask.copy()
        if name == 'sturm':
            _, invalid = rb.sturm_invalid(dataset, i); mask[invalid] = -1
            map_id, event = map_of[filename], event_of[filename]
            omit = filename in spec['excluded_sturm_chips']
        elif name == 'wf':
            map_id = filename.split('#')[0]; event = map_id.split('_')[0]
            omit = map_id in spec['excluded_wf_maps']
        else:
            map_id = filename; event = filename.split('_')[0]; omit = False
        assert np.isin(mask, [-1, 0, 1]).all()
        metadata.append((filename, mask, map_id, event, omit, item))
    return inputs, metadata


def rule_arrays(inputs, spec):
    import torch
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    mean = spec.get('norm_mean', NORM_MEAN); std = spec.get('norm_std', NORM_STD)
    assert list(mean) == list(NORM_MEAN) and list(std) == list(NORM_STD)
    reflectance = inputs * inputs.new_tensor(std)[None, :, None, None] + inputs.new_tensor(mean)[None, :, None, None]
    green, swir = reflectance[:, 1], reflectance[:, 4]
    epsilon = float(spec.get('rule_epsilon', 1e-6))
    available = torch.isfinite(green) & torch.isfinite(swir) & (green >= 0) & (swir >= 0) & (green + swir > epsilon)
    feature = torch.where(available, (green - swir) / (green + swir).clamp_min(epsilon), 0).clamp(-1, 1)
    return (feature > 0).cpu().numpy(), available.cpu().numpy()


def mechanism(pred, teacher, rule, available, mask):
    """Integer numerators and conditioning denominators on one binary pixel set."""
    import numpy as np
    valid = (mask != -1); eligible = valid & available
    teacher_correct, rule_correct, student_correct = teacher == mask, rule == mask, pred == mask
    correction = eligible & teacher_correct & ~rule_correct
    harm = eligible & ~teacher_correct & rule_correct
    both_wrong = eligible & ~teacher_correct & ~rule_correct
    disagreement = eligible & (teacher != rule)
    n = lambda x: int(np.count_nonzero(x))
    assert n(disagreement)==n(correction)+n(harm), 'Binary disagreement must have exactly one correct comparator'
    return dict(valid_pixels=n(valid), rule_available_pixels=n(eligible),
                rule_unavailable_pixels=n(valid & ~available), disagreements=n(disagreement),
                correction_denominator=n(correction), correction_numerator=n(correction & student_correct),
                harm_denominator=n(harm), harm_numerator=n(harm & ~student_correct),
                rescue_denominator=n(both_wrong), rescue_numerator=n(both_wrong & student_correct),
                both_correct_pixels=n(eligible & teacher_correct & rule_correct),
                signed_benefit_numerator=n(correction) - n(harm))


def sum_mechanisms(records):
    return {key: sum(row[key] for row in records) for key in next(iter(records), {})}


def reference_accumulator():
    import numpy as np
    zero=lambda:np.zeros((2,3),dtype=np.int64)
    return dict(images={},maps=defaultdict(zero),events=defaultdict(zero),
                xmaps=defaultdict(zero),xevents=defaultdict(zero),included=0)


def reference_add(accumulator, metadata, prediction):
    from evaluation import revision_native_border as rb
    filename,mask,map_id,event,omit,item=metadata
    h,w=mask.shape;cell=rb.confusion(prediction[:h,:w],mask)
    accumulator['images'][filename]=cell.tolist()
    accumulator['maps'][map_id]+=cell;accumulator['events'][event]+=cell
    if not omit:
        accumulator['xmaps'][map_id]+=cell;accumulator['xevents'][event]+=cell;accumulator['included']+=1


def reference_pack(accumulator, length, protocol):
    import numpy as np
    def pack(maps,events):
        total=sum(maps.values(),np.zeros((2,3),dtype=np.int64))
        return dict(micro=scores(total.tolist()),pooled_counts=total.tolist(),
            map={key:cell.tolist() for key,cell in maps.items()},event={key:cell.tolist() for key,cell in events.items()},
            n_maps=len(maps),n_events=len(events))
    return dict(full=pack(accumulator['maps'],accumulator['events']),
        known_overlap_excluded={**pack(accumulator['xmaps'],accumulator['xevents']),'n_images':accumulator['included']},
        per_image_counts=accumulator['images'],n_images=length,protocol=protocol)


def entries_from_receipts(paths, final=False, smoke=False):
    entries = []
    for path in paths:
        receipt = read(path)
        if receipt.get('outcome') not in (('completed', 'reused', 'smoke_completed') if smoke else ('completed', 'reused')) or (receipt.get('smoke_only') and not smoke):
            continue
        for kind, prefix in [('selected', 'best')] + ([('final', 'last')] if final else []):
            if not receipt.get(prefix + '_path'):
                continue
            entries.append(dict(id=Path(path).stem + '__' + kind, graph=receipt['graph'],
                method=receipt.get('arm', receipt.get('method', 'lkd')), seed=receipt.get('seed'),
                checkpoint_kind=kind, checkpoint=receipt[prefix + '_path'], sha256=receipt[prefix + '_sha256'],
                training_receipt=str(path), norm_mean=receipt.get('norm_mean'), norm_std=receipt.get('norm_std')))
    return entries


def load_model(entry, rungs_dir):
    import torch
    from train_v7 import build_model
    if file_hash(entry['checkpoint']) != entry['sha256']:
        raise RuntimeError(f"Changed checkpoint bytes: {entry['id']}")
    payload = torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)
    metadata = payload.get('v7_metadata', payload.get('metadata', {}))
    model = build_model(entry['graph'], entry.get('method', 'lkd'), rungs_dir,
                        norm_mean=entry.get('norm_mean') or metadata.get('norm_mean'),
                        norm_std=entry.get('norm_std') or metadata.get('norm_std'))
    model.load_state_dict(payload['model'], strict=True)
    return model.cuda().eval()


def cache_identity(spec, capsule):
    from data.sen1floods11 import NORM_MEAN, NORM_STD
    names = ['data/ood_datasets.py', 'data/sen1floods11.py', 'data/channel_adapter.py',
             'evaluation/revision_native_border.py', 'evaluation/perscene_eval.py']
    return dict(teacher_sha256=spec['teacher_sha256'], overlap_plan_sha256=file_hash(spec['overlap_plan']),
                protocols=spec['protocols'], roots=spec['roots'], norm_mean=NORM_MEAN, norm_std=NORM_STD,
                backend_policy=backend_record(),
                rule_epsilon=spec.get('rule_epsilon', 1e-6), green_index=1, swir1_index=4,
                rule_code_sha256=file_hash(Path(__file__).resolve().parent/'v7_loss.py'),
                source_sha256={n: file_hash(Path(capsule) / n) for n in names})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'capsule', 'rungs-dir', 'spec', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--receipts', nargs='*', default=[], type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--final', action='store_true', help='Separately reserved final external readouts')
    parser.add_argument('--prepare-cache', action='store_true')
    parser.add_argument('--mechanisms', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    args.receipts = [p.resolve() for p in args.receipts] + ([args.receipt] if args.receipt else [])
    start = time.monotonic(); spec = read(args.spec)
    torch = setup(args.capsule, args.rungs_dir)
    import numpy as np
    import scripts.compression_jobs as cj
    from evaluation import revision_native_border as rb
    sets, map_of, event_of = datasets(spec)
    identity = cache_identity(spec, args.capsule)
    cache_dir = args.run_dir / 'reference_cache'; cache_manifest = cache_dir / 'manifest.json'
    out_dir = args.out_dir or args.run_dir / ('smoke_terminal' if args.smoke else 'terminal'); out_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_cache:
        if cache_manifest.exists() and read(cache_manifest).get('identity') == identity and not args.smoke:
            cached=read(cache_manifest)
            if cached.get('status')=='complete' and not cached.get('smoke_only') and all(
                    Path(item['path']).is_file() and file_hash(item['path'])==item['sha256'] for item in cached.get('datasets',{}).values()):
                atomic_json(args.out, dict(status='complete', outcome='reused', at=now(), cache_manifest=str(cache_manifest)))
                return
        cache_dir.mkdir(parents=True, exist_ok=True)
        teacher = cj.teacher(spec)
        refs = dict(identity=identity, smoke_only=args.smoke, datasets={})
        comparator_results={'teacher':{},'rule':{}}
        with torch.no_grad():
            for name, ds, batch in sets:
                length = min(8,len(ds)) if args.smoke else len(ds)
                comparators={key:reference_accumulator() for key in comparator_results}
                cache_path = cache_dir / (name + ('.smoke' if args.smoke else '') + '.npy')
                packed = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.uint8, shape=(length, 3, 512 * 512 // 8))
                rows = []
                for begin in range(0, length, batch):
                    ids = list(range(begin, min(begin + batch, length)))
                    inputs, meta = batch_items(name, ds, ids, map_of, event_of, spec)
                    logits = teacher(inputs); cj.finite_output(teacher, (inputs,), logits)
                    predictions = logits.argmax(1).cpu().numpy(); rules, available = rule_arrays(inputs, spec)
                    cpu_inputs = inputs.cpu().numpy()
                    for local, (index, record) in enumerate(zip(ids, meta)):
                        filename, mask, map_id, event, omit, item = record
                        packed[index] = np.packbits(np.stack([predictions[local], rules[local], available[local]]).reshape(3, -1), axis=1)
                        rows.append(dict(filename=filename, map=map_id, event=event, omit=omit,
                            input_sha256=hashlib.sha256(cpu_inputs[local].tobytes()).hexdigest(),
                            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
                        reference_add(comparators['teacher'],record,predictions[local])
                        reference_add(comparators['rule'],record,rules[local])
                    if begin % 256 == 0: print('reference', name, begin, '/', length, flush=True)
                packed.flush(); del packed
                refs['datasets'][name] = dict(path=str(cache_path), sha256=file_hash(cache_path), rows=rows)
                for key in comparator_results:
                    comparator_results[key][name]=reference_pack(comparators[key],length,spec['protocols'][name])
                    atomic_json(out_dir/f'v7_reference_{key}.json',dict(status='partial',outcome='in_progress',
                        reference_only=True,reference=key,id=f'v7_reference_{key}',
                        checkpoint_sha256=spec['teacher_sha256'] if key=='teacher' else identity['rule_code_sha256'],
                        terminal_results=comparator_results[key],evaluation_identity=identity,smoke_only=args.smoke,
                        rule_unavailable_prediction='land(0)' if key=='rule' else None,
                        rule_unavailable_status='flagged unavailable in mechanisms; predictions retained on registered evaluation valid pixels',
                        completed_datasets=list(comparator_results[key]),secondary_readout_only=True))
        refs.update(status='complete', outcome='completed', completed_at=now(), elapsed_seconds=time.monotonic()-start)
        manifest_path = cache_dir / ('smoke_manifest.json' if args.smoke else 'manifest.json')
        atomic_json(manifest_path, refs)
        for key in comparator_results:
            target=out_dir/f'v7_reference_{key}.json';payload=read(target)
            atomic_json(target,{**payload,'status':'complete','outcome':'completed','completed_at':now()})
        atomic_json(args.out, dict(status='complete', outcome='completed', smoke_only=args.smoke,
            cache_manifest=str(manifest_path), cache_manifest_sha256=file_hash(manifest_path), elapsed_seconds=time.monotonic()-start))
        return
    explicit = read(args.models) if args.models else []
    if isinstance(explicit, dict): explicit = explicit.get('models', [])
    entries = explicit + entries_from_receipts(args.receipts, args.final, args.smoke)
    requested = list(entries); entries = [entry for entry in entries if not (out_dir / (entry['id']+'.json')).exists()
        or read(out_dir / (entry['id']+'.json')).get('status') != 'complete'
        or read(out_dir / (entry['id']+'.json')).get('checkpoint_sha256') != entry['sha256']
        or read(out_dir / (entry['id']+'.json')).get('evaluation_identity') != identity
        or (args.mechanisms and not read(out_dir / (entry['id']+'.json')).get('mechanism_counts'))]
    if not entries:
        atomic_json(args.out, dict(status='complete', outcome='completed', requested=[e['id'] for e in requested], scored_now=[], at=now()))
        return
    refs = None
    if args.mechanisms:
        if args.smoke: cache_manifest=cache_dir/'smoke_manifest.json'
        if not cache_manifest.exists(): raise RuntimeError('Prepare the shared teacher/rule reference cache first')
        refs = read(cache_manifest)
        if refs.get('identity') != identity or bool(refs.get('smoke_only',False)) != args.smoke: raise RuntimeError('Reference cache identity mismatch/incomplete')
        for item in refs['datasets'].values():
            if file_hash(item['path']) != item['sha256']: raise RuntimeError('Changed reference cache bytes')
    models = [(entry, load_model(entry, args.rungs_dir)) for entry in entries]
    terminal = {entry['id']: {} for entry in entries}; mechanisms = {entry['id']: {} for entry in entries}
    zero = lambda: np.zeros((2, 3), dtype=np.int64)
    with torch.no_grad():
        for name, ds, batch in sets:
            length = min(8,len(ds)) if args.smoke else len(ds)
            cache = np.load(refs['datasets'][name]['path'], mmap_mode='r') if refs else None
            acc = {entry['id']: dict(images={}, maps=defaultdict(zero), events=defaultdict(zero),
                xmaps=defaultdict(zero), xevents=defaultdict(zero), included=0, mechanisms={}) for entry in entries}
            for begin in range(0, length, batch):
                ids = list(range(begin, min(begin + batch, length)))
                inputs, meta = batch_items(name, ds, ids, map_of, event_of, spec)
                unpacked = np.unpackbits(cache[ids], axis=2).reshape(len(ids), 3, 512, 512).astype(bool) if refs else None
                if refs:
                    cpu_inputs = inputs.cpu().numpy()
                    for local, (index, record) in enumerate(zip(ids, meta)):
                        cached = refs['datasets'][name]['rows'][index]
                        if (cached['filename'] != record[0] or cached['input_sha256'] != hashlib.sha256(cpu_inputs[local].tobytes()).hexdigest()
                                or cached['mask_sha256'] != hashlib.sha256(record[1].tobytes()).hexdigest()):
                            raise RuntimeError('Mechanism reference is not the exact evaluated image/mask')
                for entry, model in models:
                    logits = model(inputs); cj.finite_output(model, (inputs,), logits)
                    predictions = logits.argmax(1).cpu().numpy(); target = acc[entry['id']]
                    for local, (record, pred) in enumerate(zip(meta, predictions)):
                        filename, mask, map_id, event, omit, item = record
                        h, w = mask.shape; pred = pred[:h, :w]
                        cell = rb.confusion(pred, mask)
                        target['images'][filename] = cell.tolist(); target['maps'][map_id] += cell; target['events'][event] += cell
                        if not omit:
                            target['xmaps'][map_id] += cell; target['xevents'][event] += cell; target['included'] += 1
                        if refs:
                            teacher, rule, available = unpacked[local, :, :h, :w]
                            target['mechanisms'][filename] = dict(event=event, map=map_id, omit=omit,
                                **mechanism(pred, teacher, rule, available, mask))
            def pack(maps, events):
                total = sum(maps.values(), zero())
                return dict(micro=scores(total.tolist()), pooled_counts=total.tolist(),
                    map={key: cell.tolist() for key, cell in maps.items()},
                    event={key: cell.tolist() for key, cell in events.items()}, n_maps=len(maps), n_events=len(events))
            for entry in entries:
                target = acc[entry['id']]
                if not args.smoke:
                    expected = {'s11': (90, 10, 90, 90), 'sturm': (46, 29, 2675, 2672), 'wf': (18, 11, 785, 733)}[name]
                    assert (len(target['maps']), len(target['events']), len(target['images']), target['included']) == expected
                terminal[entry['id']][name] = dict(full=pack(target['maps'], target['events']),
                    known_overlap_excluded={**pack(target['xmaps'], target['xevents']), 'n_images':target['included']},
                    per_image_counts=target['images'], n_images=length, protocol=spec['protocols'][name])
                if refs:
                    mechanisms[entry['id']][name] = target['mechanisms']
                # A timeout after this point must preserve every finished dataset.
                # Partial receipts label command completion separately from the
                # fully covered dataset readouts already present in this file.
                atomic_json(out_dir / (entry['id']+'.json'), dict(status='partial', outcome='in_progress',
                    id=entry['id'],graph=entry['graph'],method=entry.get('method'),seed=entry.get('seed'),
                    checkpoint_kind=entry.get('checkpoint_kind','selected'),checkpoint_path=entry['checkpoint'],
                    checkpoint_sha256=entry['sha256'],terminal_results=terminal[entry['id']],
                    mechanism_counts=mechanisms[entry['id']],mechanism_status='partial' if refs else 'not_run',
                    evaluation_identity=identity,completed_datasets=list(terminal[entry['id']]),
                    secondary_readout_only=True,smoke_only=args.smoke,
                    elapsed_seconds_shared_pass=time.monotonic()-start))
            print(name, 'done', round(time.monotonic()-start, 1), 'seconds', flush=True)
    for entry, model in models:
        atomic_json(out_dir / (entry['id']+'.json'), dict(status='complete', outcome='completed',
            id=entry['id'], graph=entry['graph'], method=entry.get('method'), seed=entry.get('seed'),
            checkpoint_kind=entry.get('checkpoint_kind', 'selected'), checkpoint_path=entry['checkpoint'],
            checkpoint_sha256=entry['sha256'], parameters=sum(p.numel() for p in model.parameters()),
            terminal_results=terminal[entry['id']], mechanism_counts=mechanisms[entry['id']],
            mechanism_status='completed' if refs else 'not_run', secondary_readout_only=True,
            empty_union_convention='undefined water IoU; omit empty union from event mean; omit undefined class from mIoU',
            evaluation_identity=identity, completed_at=now(), elapsed_seconds_shared_pass=time.monotonic()-start,
            smoke_only=args.smoke, final_external_separately_reserved=args.final))
    atomic_json(args.out, dict(status='complete', outcome='completed', scored_now=[e['id'] for e in entries],
        requested=[e['id'] for e in requested], elapsed_seconds=time.monotonic()-start, smoke_only=args.smoke, completed_at=now()))


if __name__ == '__main__':
    main()
