"""V11 selective KD: narrow adapters to the frozen V10 loader, loss and Trainer.

No external evaluator is imported by training. Inputs are admitted only from
completed, hash-checked V10 receipts. All writes are under this V11 run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT.parent / 'research_v10_gpu_20261013T183509Z'
SEEDS = [42, 64, 91, 107]
ARMS = ['D', 'U', 'A', 'L', 'N', 'P', 'F0']
CHECKPOINTS = [0, 50, 200, 600]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def file_identity(path):
    return dict(path=str(path), sha256=sha(path))


def check_identity(identity):
    if sha(identity['path']) != identity['sha256']:
        raise ValueError('Input hash changed: ' + identity['path'])


def setup(threads=4):
    if not 1 <= threads <= 8:
        raise ValueError('V11 permits at most eight CPU threads')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['V10_COMPUTE_DEVICE'] = 'cpu'
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(threads)
    if str(V10 / 'code') not in sys.path:
        sys.path.insert(0, str(V10 / 'code'))
    import common10 as c
    if not hasattr(c, '_v11_setup_original'):
        c._v11_setup_original = c.setup
    # c.load invokes setup without arguments; bind that call to our allocation.
    c.setup = lambda *args, **kwargs: c._v11_setup_original(threads=threads)
    torch, cj, old, v = c.setup()
    torch.set_num_threads(threads)
    if not getattr(torch, '_v11_interop_fixed', False):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        torch._v11_interop_fixed = True
    return torch, c, cj, v


def tensor_hash(tensors):
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        array = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(tuple(array.shape)).encode())
        h.update(array.numpy().tobytes())
    return h.hexdigest()


def pixel_kl(teacher_logits, student_logits, temperature=4.):
    import torch
    from torch.nn import functional as F
    teacher = teacher_logits.detach() / temperature
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher, dim=1), reduction='none').sum(1)


def priority_scores(teacher_logits, fp_logits, quant_logits, temperature=4.):
    """Detached per-pixel scores; callers exclude invalid pixels before sorting."""
    import torch
    with torch.no_grad():
        q = (teacher_logits.detach() / temperature).softmax(1)
        p_q = (quant_logits.detach() / temperature).softmax(1)
        p_f = (fp_logits.detach() / temperature).softmax(1)
        absolute = pixel_kl(teacher_logits, quant_logits, temperature)
        return dict(D=temperature**2 * (absolute - pixel_kl(teacher_logits, fp_logits, temperature)),
                    A=absolute, L=(p_q - q).square().sum(1).sqrt(),
                    N=(p_q - p_f).square().sum(1).sqrt())


def binary_extra_loss(teacher_logits, fp_logits, quant_logits, temperature=4.):
    from torch.nn import functional as F
    q = ((teacher_logits[:, 1] - teacher_logits[:, 0]) / temperature).sigmoid()
    af = (fp_logits[:, 1] - fp_logits[:, 0]) / temperature
    aq = (quant_logits[:, 1] - quant_logits[:, 0]) / temperature
    return temperature**2 * (F.softplus(aq) - F.softplus(af) - q * (aq - af))


@contextmanager
def scoring_state(model):
    """Protect all buffers, mode flags and CPU RNG streams, even on exceptions."""
    import numpy as np
    import torch
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            for tensor, value in buffers:
                # Do not bump versions of unchanged autograd-saved scales.
                if not torch.equal(tensor, value):
                    tensor.copy_(value)
            for module, mode in modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def paired_logits(model, images):
    """One shared live model; both weight and activation QDQ bypassed in FP."""
    import torch
    from models.compression_quant import quantization_mode
    from quantization.qat import QATConv2d, QATMatMul
    supported = (QATConv2d, QATMatMul)
    unsupported = [type(m).__name__ for m in model.modules()
                   if hasattr(m, 'fake_quant_enabled') and not isinstance(m, supported)]
    if unsupported:
        raise TypeError('Unsupported quantizer bypass: ' + str(unsupported))
    import numpy as np
    with scoring_state(model):
        rng = torch.random.get_rng_state()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        with quantization_mode(model, False):
            assert all(not m.fake_quant_enabled for m in model.modules() if isinstance(m, supported))
            fp = model(images).detach()
        torch.random.set_rng_state(rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        with quantization_mode(model, True):
            quant = model(images).detach()
    if not torch.isfinite(fp).all() or not torch.isfinite(quant).all():
        raise FloatingPointError('Nonfinite paired scoring output')
    return fp, quant


def class_counts(selected, teacher_class):
    return [[int((row & (classes == k)).sum()) for k in [0, 1]]
            for row, classes in zip(selected, teacher_class)]


def select_priority(score, valid, teacher_class, *, counts=None, random_seed=None):
    """D: positive top floor(N/4); controls: exact image/class counts from D.

    Stable descending sorting resolves ties by row-major pixel index. P uses a
    private RNG, so selection never changes input ordering or augmentation.
    """
    import torch
    score, valid, teacher_class = score.detach(), valid.bool(), teacher_class.detach()
    if score.shape != valid.shape or teacher_class.shape != valid.shape:
        raise ValueError('Score, support and teacher class geometry differ')
    if not torch.isfinite(score[valid]).all():
        raise FloatingPointError('Nonfinite valid score')
    selected = torch.zeros_like(valid)
    generator = torch.Generator(device='cpu').manual_seed(int(random_seed or 0))
    for i in range(len(valid)):
        values, ok, cls, output = (x[i].reshape(-1) for x in [score, valid, teacher_class, selected])
        groups = [None] if counts is None else [0, 1]
        for group in groups:
            eligible = ok & (values > 1e-8) if group is None else ok & (cls == group)
            indices = eligible.nonzero().flatten()
            k = min(int(ok.sum()) // 4, len(indices)) if group is None else int(counts[i][group])
            if k < 0 or k > len(indices):
                raise ValueError('D trace count is impossible on this valid teacher class')
            if random_seed is None:
                order = torch.argsort(values[indices], descending=True, stable=True)
            else:
                order = torch.randperm(len(indices), generator=generator).to(indices.device)
            output[indices[order[:k]]] = True
    weights = valid.to(score.dtype)
    for i in range(len(valid)):
        n, k = int(valid[i].sum()), int(selected[i].sum())
        if n:
            weights[i] = valid[i] * (1 + 3 * selected[i].to(score.dtype)) / (1 + 3 * k / n)
    return selected.detach(), weights.detach(), class_counts(selected, teacher_class)


def weighted_kd_loss(student_logits, hard_mask, teacher_logits, weights):
    """Use V10's exact uniform path when no pixel was prioritized."""
    from training.losses import focal_loss
    import v10_reference as v
    valid = hard_mask != -1
    if not valid.any() or bool((weights[valid] == 1).all()):
        return v.offline_kd_loss(student_logits, hard_mask, teacher_logits.detach())
    return .5 * focal_loss(student_logits, hard_mask, 2., -1) + .5 * 16 * (
        weights.detach()[valid] * pixel_kl(teacher_logits, student_logits)[valid]).mean()


def resolve_inputs(root=ROOT, point='core'):
    """Freeze only complete V10 dependencies; never manufacture missing parents."""
    root = Path(root)
    target = root / ('repair_inputs.json' if point == 'core' else 'repair_transfer_inputs.json')
    if target.exists():
        return validate_inputs(target)
    v9 = V10.parent / 'research_v9_recovery_20261013T075000Z'
    cache_path, control_path = V10 / 'teacher_logit_cache_manifest.json', V10 / 'compression_control_selection.json'
    cache, controls = read(cache_path), read(control_path)
    assert cache['outcome'] == controls['outcome'] == 'completed'
    assert len(cache['files']) == 2500 and cache['dtype'] == 'float32' and cache['views'] == 8
    assert all(x['maximum_geometric_target_difference'] < .001 for x in cache['fresh_forward_checks'])
    graph, precision = 'R4', 'W4A4'
    dependencies = [file_identity(control_path), file_identity(V10 / 'teacher_cache_policy.json'),
                    file_identity(V10 / 'manifest.json')]
    if point == 'transfer':
        selection_path = V10 / 'smallest_viable_selection.json'
        selection = read(selection_path)
        assert selection.get('source_only', True) and selection.get('graph')
        graph, precision = selection['graph'], 'W4A8'
        dependencies.append(file_identity(selection_path))
    parents = {}
    for seed in SEEDS:
        path = V10 / 'qat_parents' / f'{graph}_{precision}_s{seed}.json'
        receipt = read(path)
        for key in ['entry', 'FP_parent']:
            entry = receipt[key]
            assert entry['graph'] == graph and entry['seed'] == seed
            assert sha(entry['checkpoint']) == entry['sha256']
        assert receipt['entry']['precision'] == precision and receipt['entry']['quantized']
        parents[str(seed)] = dict(receipt=file_identity(path), **receipt)
    spec_path = v9 / 'training_spec.json'
    spec = read(spec_path)
    training = read(V10 / 'manifest.json')['training']
    lrs = {float(training[f'{graph}_{precision}_d2_s{seed}']['lr']) for seed in SEEDS}
    assert len(lrs) == 1
    inputs = dict(schema=1, point=point, graph=graph, precision=precision, seeds=SEEDS,
                  training_spec=file_identity(spec_path), cache_manifest=file_identity(cache_path),
                  source_windows=file_identity(v9 / 'normal_windows.json'),
                  cohort=file_identity(spec['index']), parents=parents, dependency_files=dependencies,
                  lr=lrs.pop(), source_only=True, teacher_sha256=spec['teacher_sha256'],
                  optimizer='frozen Trainer AdamW; weight_decay=1e-5',
                  schedule='cosine T_max=10 epochs, stepped after each full 156-update epoch',
                  augmentation='V10 clean eight-view teacher cache; student-only multiplicative jitter',
                  stream='new V11 deterministic num_workers=0 stream, shared by every arm; no V10 run reuse',
                  backend='CPU float32 fixed-scale fake-quant simulation')
    source_receipt = v9 / 'gpu_inference_normal_final03.json'
    if source_receipt.exists():
        sr = read(source_receipt)
        assert sr['outcome'] == 'completed' and sr['full_set']
        teacher = next(r for r in sr['records'] if r['model'] == 'teacher')
        assert teacher['checkpoint_sha256'] == spec['teacher_sha256']
        inputs['dependency_files'].append(file_identity(source_receipt))
        inputs['source_teacher_predictions'] = []
        for row in read(inputs['source_windows']['path'])['windows']:
            if row['dataset'] != 's11val' or row.get('omit', False):
                continue
            path = Path(teacher['directory']) / (Path(row['path']).stem + '.npy')
            inputs['source_teacher_predictions'].append(dict(path=row['path'], file=str(path), sha256=sha(path)))
    save(target, inputs)
    return validate_inputs(target)


def validate_inputs(path):
    inputs = read(path)
    assert inputs['schema'] == 1 and inputs['seeds'] == SEEDS and inputs['source_only']
    identities = [inputs[k] for k in ['training_spec', 'cache_manifest', 'source_windows', 'cohort']]
    identities += inputs['dependency_files']
    for parent in inputs['parents'].values():
        identities.append(parent['receipt'])
        identities += [dict(path=parent[key]['checkpoint'], sha256=parent[key]['sha256'])
                       for key in ['entry', 'FP_parent']]
    for item in identities:
        check_identity(item)
    return inputs


def input_path(root, point):
    return Path(root) / ('repair_inputs.json' if point == 'core' else 'repair_transfer_inputs.json')


def repair_path(root, point):
    return Path(root) / 'repair' / point


def source_gate(root=ROOT, point='core', threads=4):
    from source11 import evaluate_source, deficit_gate
    torch, c, _, _ = setup(threads)
    inputs = validate_inputs(input_path(root, point))
    folder = repair_path(root, point)
    results = []
    for seed in SEEDS:
        parents = inputs['parents'][str(seed)]
        result = dict(seed=seed)
        for label, key in [('FP', 'FP_parent'), ('Q0', 'entry')]:
            path = folder / f'{label}_s{seed}' / 'source_step0000.json'
            if path.exists():
                metrics = read(path)
                assert metrics['checkpoint_sha256'] == parents[key]['sha256']
            else:
                model = c.load(parents[key])
                metrics = evaluate_source(model, inputs, root, parents[key], seed=seed, arm=label, step=0)
                save(path, metrics)
            result[label] = metrics
        results.append(result)
    receipt = deficit_gate(results)
    receipt.update(inputs_sha256=sha(input_path(root, point)), point=point, results=results)
    save(folder / 'source_gate.json', receipt)
    collect_diagnostics(root)
    return receipt


def _training_loader(spec, seed, cache):
    """V10 dataset and augmentation; private, arm-independent order generator."""
    import torch
    import train10
    from torch.utils.data import DataLoader
    original = train10.loaders(spec, 'KDoff', seed)
    ds = original.dataset
    del original
    table = {Path(x['file']).name: x for x in cache['files']}
    checked = set()

    class Verified(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, index):
            path = ds.files[index]
            if path.name not in checked:
                entry = table[path.name]
                if sha(path) != entry['input_sha256'] or sha(entry['file']) != entry['sha256']:
                    raise ValueError('TRAIN/cache hash changed: ' + path.name)
                checked.add(path.name)
            result = ds[index]
            result['sample_id'] = path.name
            return result

    loader = DataLoader(Verified(), batch_size=16, shuffle=True, num_workers=0,
                        pin_memory=False, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    assert len(loader) == 156
    return loader


def train(root=ROOT, point='core', threads=4, arm='D', seed=42, profile=False):
    """Bound the inherited Trainer at 600 updates without retiming its scheduler."""
    started, started_cpu = time.monotonic(), time.process_time()
    import torch
    from source11 import evaluate_source
    torch, c, cj, v = setup(threads)
    from config import TrainConfig
    from training.trainer import Trainer as FrozenTrainer
    from models.compression_quant import compression_train_mode, precision_map
    from utils.seeding import seed_basic
    assert arm in ARMS and seed in SEEDS
    if profile:
        assert arm == 'D', 'The bounded profile measures D, the most expensive score arm'
    inputs_file = input_path(root, point)
    inputs = validate_inputs(inputs_file)
    folder = repair_path(root, point)
    gate = read(folder / 'source_gate.json')
    assert gate['outcome'] == 'ADMITTED' and gate['inputs_sha256'] == sha(inputs_file)
    endpoint = 50 if profile else 600
    output = folder / (('profile_' if profile else '') + f'{arm}_s{seed}')
    if (output / 'receipt.json').exists():
        receipt = read(output / 'receipt.json')
        assert receipt['outcome'] == 'completed' and receipt['completed_updates'] == endpoint
        for entry in receipt['checkpoints']:
            assert sha(entry['checkpoint']) == entry['sha256']
        return receipt
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Partial attempt retained; refusing duplicate training: ' + str(output))
    reference = folder / f'D_s{seed}'
    d_trace = None
    if arm != 'D':
        receipt = read(reference / 'receipt.json')
        assert receipt['outcome'] == 'completed' and receipt['completed_updates'] == 600
        assert receipt['inputs_sha256'] == sha(inputs_file)
        assert sha(reference / 'D_trace.jsonl') == receipt['D_trace_sha256']
        d_trace = [json.loads(line) for line in (reference / 'D_trace.jsonl').read_text().splitlines()]
        assert len(d_trace) == 600
    output.mkdir(parents=True, exist_ok=True)
    parents = inputs['parents'][str(seed)]
    model = c.load(parents['entry']).requires_grad_(True)
    # F0 shares every parameter and frozen BN with Q0; only QDQ is bypassed.
    compression_train_mode(model, enabled=arm != 'F0')
    initial_parameters = tensor_hash(dict(model.named_parameters()))
    initial_precision = precision_map(model)
    seed_basic(seed)
    loader = _training_loader(read(inputs['training_spec']['path']), seed,
                              read(inputs['cache_manifest']['path']))
    cache_creation = read(inputs['cache_manifest']['path']).get('seconds')
    trace_file = open(output / 'D_trace.jsonl', 'a', buffering=1) if arm == 'D' else None
    batch_trace_file = open(output / 'batch_trace.jsonl', 'a', buffering=1)
    checkpoints, curves = [], []
    timing = dict(score_seconds=0., selection_seconds=0., source_seconds=0., checkpoint_seconds=0.)
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    trainer = None

    class Loss(torch.nn.Module):
        gamma, kd_valid_only = 2., True

        def __init__(self):
            super().__init__()
            # Avoid registering model here, which duplicates optimizer parameters.
            object.__setattr__(self, 'student', model)
            self.batch_identity = None

        def forward(self, z, x, y, precomputed_teacher_logits=None):
            teacher = precomputed_teacher_logits.detach()
            valid = y != -1
            step = trainer._global_step
            if arm in ['U', 'F0']:
                value = v.offline_kd_loss(z, y, teacher)
                return value, dict(offline_loss=float(value.detach()))
            tick = time.monotonic()
            if arm in ['D', 'N']:
                fp, qz = paired_logits(model, x)
                scores = priority_scores(teacher, fp, qz)
                score = scores[arm]
            elif arm == 'A':
                with torch.no_grad():
                    score = pixel_kl(teacher, z.detach())
            elif arm == 'L':
                with torch.no_grad():
                    score = ((teacher / 4).softmax(1) - (z.detach() / 4).softmax(1)).square().sum(1).sqrt()
            else:
                score = torch.zeros_like(y, dtype=z.dtype)
            timing['score_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            cls = teacher.argmax(1)
            counts = None if d_trace is None else d_trace[step]['counts']
            if d_trace is not None:
                assert d_trace[step]['batch_identity'] == self.batch_identity, 'Augmentation/order mismatch with D'
            selected, weights, actual = select_priority(score, valid, cls, counts=counts,
                                                        random_seed=seed * 1000003 + step if arm == 'P' else None)
            if counts is not None:
                assert actual == counts
            if trace_file is not None:
                trace_file.write(json.dumps(dict(update=step + 1, counts=actual,
                                                 batch_identity=self.batch_identity)) + '\n')
            timing['selection_seconds'] += time.monotonic() - tick
            value = weighted_kd_loss(z, y, teacher, weights)
            return value, dict(offline_loss=float(value.detach()), selected_pixels=int(selected.sum()))

    loss = Loss()

    def checkpoint(step, interrupted=False):
        tick = time.monotonic()
        with scoring_state(model):
            compression_train_mode(model, enabled=arm != 'F0')
            model.eval()
            checkpoint_path = output / (f'step{step:04d}.pt' if not interrupted else f'interrupted_step{step:04d}.pt')
            metadata = dict(arm=arm, seed=seed, graph=inputs['graph'], point=point,
                            precision='FP32' if arm == 'F0' else inputs['precision'],
                            updates=step, inputs_sha256=sha(inputs_file),
                            initial_parameter_sha256=initial_parameters,
                            fp_parent_sha256=parents['FP_parent']['sha256'],
                            precision_map=precision_map(model), backend=inputs['backend'])
            # F0 uses the true current FP state with all fake quantizers removed.
            if arm == 'F0':
                from models.compression_quant import strip_compression_qat
                state = strip_compression_qat(model).state_dict()
            else:
                state = model.state_dict()
            temporary = checkpoint_path.with_suffix('.tmp')
            torch.save(dict(model=state, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), v11_metadata=metadata,
                            training_state=dict(global_step=step)), temporary)
            os.replace(temporary, checkpoint_path)
            entry = dict(id=f'v11_{point}_{"profile_" if profile else ""}{arm}_s{seed}_u{step}', graph=inputs['graph'], seed=seed,
                         method=arm, arm=arm, point=point, updates=step,
                         checkpoint=str(checkpoint_path), sha256=sha(checkpoint_path),
                         state_sha256=tensor_hash(state),
                         quantized=arm != 'F0', precision=metadata['precision'],
                         fp_reference=parents['FP_parent']['id'], source_only_selection=True)
            timing['checkpoint_seconds'] += time.monotonic() - tick
            if not interrupted:
                tick = time.monotonic()
                metrics = evaluate_source(model, inputs, root, entry, seed=seed, arm=arm, step=step)
                timing['source_seconds'] += time.monotonic() - tick
                metric_path = output / f'source_step{step:04d}.json'
                save(metric_path, metrics)
                entry['source_metrics'] = str(metric_path)
                checkpoints.append(entry)
                curves.append(dict(arm=arm, seed=seed, point=point, updates=step,
                                   equal_event_water_IoU=metrics['equal_event_water_IoU'],
                                   water_IoU=metrics['pooled']['water_IoU'], mIoU=metrics['pooled']['mIoU'],
                                   cumulative_elapsed_seconds=time.monotonic() - started,
                                   cumulative_process_cpu_seconds=time.process_time() - started_cpu,
                                   allocated_core_hours=(time.monotonic() - started) * threads / 3600,
                                   teacher_cache_generation_seconds=cache_creation,
                                   teacher_cache_cost_allocation='reused V10; generation cost reported separately',
                                   **timing))
                write_csv(output / 'learning_curves.csv', curves)
        compression_train_mode(model, enabled=arm != 'F0')

    class Bounded:
        dataset = loader.dataset

        def __len__(self):
            return min(len(loader), endpoint - (trainer._global_step if trainer else 0))

        def __iter__(self):
            remaining = min(len(loader), endpoint - trainer._global_step)
            iterator = iter(loader)
            for _ in range(remaining):
                if stopping[0]:
                    return
                batch = next(iterator)
                loss.batch_identity = dict(sample_ids=batch['sample_id'],
                                           augmentation_sha256=tensor_hash({k: batch[k] for k in ['image', 'mask', 'soft_logits']}))
                if d_trace is not None:
                    assert d_trace[trainer._global_step]['batch_identity'] == loss.batch_identity, 'Augmentation/order mismatch with D'
                batch_trace_file.write(json.dumps(dict(update=trainer._global_step + 1,
                                                        batch_identity=loss.batch_identity)) + '\n')
                yield batch
                if trainer._global_step in CHECKPOINTS[1:]:
                    checkpoint(trainer._global_step)

    class Adapter(FrozenTrainer):
        def _init_wandb(self):
            return None

    cfg = TrainConfig(epochs=10, lr=inputs['lr'], weight_decay=1e-5, task_loss='focal',
                      use_amp=False, grad_clip=1., early_stop_patience=0,
                      val_smoothing_window=1, val_every_n_epochs=2,
                      defer_terminal_evaluation=True, out_dir=str(output), seed=seed,
                      gpu_ids=[0], deterministic=False)
    trainer = Adapter(model, Bounded(), None, None, cfg, loss_fn=loss)
    assert trainer.device.type == 'cpu' and not trainer.use_amp
    optimizer = trainer._build_optimizer(model, cfg)
    scheduler = trainer._build_scheduler(optimizer)

    def finite_gradients(optimizer, *_):
        gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), 'Nonfinite or missing student gradient'

    optimizer.register_step_pre_hook(finite_gradients)
    try:
        checkpoint(0)
        while trainer._global_step < endpoint and not stopping[0]:
            before = trainer._global_step
            compression_train_mode(model, enabled=arm != 'F0')
            trainer._train_epoch_distill(model, optimizer)
            if trainer._global_step - before == 156:
                scheduler.step()
            if trainer._global_step == before and not stopping[0]:
                raise RuntimeError('No training progress')
        if stopping[0]:
            checkpoint(trainer._global_step, interrupted=True)
    finally:
        if trace_file is not None:
            trace_file.close()
        batch_trace_file.close()
    receipt = dict(outcome='interrupted' if stopping[0] else 'completed', arm=arm, seed=seed,
                   point=point, profile=profile, completed_updates=trainer._global_step,
                   inputs_sha256=sha(inputs_file), initial_parameter_sha256=initial_parameters,
                   initial_precision_map=initial_precision, checkpoints=checkpoints,
                   elapsed_seconds=time.monotonic() - started,
                   process_cpu_seconds=time.process_time() - started_cpu, threads=threads,
                   charged_core_hours=(time.monotonic() - started) * threads / 3600,
                   timing=timing, D_trace_sha256=sha(output / 'D_trace.jsonl') if arm == 'D' else None,
                   batch_trace_sha256=sha(output / 'batch_trace.jsonl'),
                   loss='0.5 focal_gamma2 + 0.5 T^2 mean_valid weighted_KL, T=4',
                   schedule=inputs['schedule'], augmentation=inputs['augmentation'],
                   input_stream=inputs['stream'], parent=parents,
                   training_labels='GDACS teacher hard/soft labels only; source checkpoints use S11 validation')
    save(output / 'receipt.json', receipt)
    if profile:
        # Worst measured D timing protects every arm; source/checkpoint costs are
        # explicitly included. Admission must still reserve the entire matrix.
        receipt['per_update_seconds'] = max(0., receipt['elapsed_seconds'] - timing['source_seconds'] - timing['checkpoint_seconds']) / endpoint
        receipt['per_model_source_tile_seconds'] = max(read(e['source_metrics'])['per_model_source_tile_seconds'] for e in checkpoints)
        receipt['per_source_checkpoint_seconds'] = timing['source_seconds'] / len(checkpoints)
        receipt['max_source_checkpoint_seconds'] = max(read(e['source_metrics']).get('elapsed_seconds', receipt['per_source_checkpoint_seconds']) for e in checkpoints)
        receipt['train_seconds'] = max(0., receipt['elapsed_seconds'] - timing['source_seconds'] - timing['checkpoint_seconds'])
        projected_seconds = 28 * (600 * receipt['per_update_seconds']
                                  + 4 * (timing['source_seconds'] + timing['checkpoint_seconds']) / len(checkpoints))
        receipt['conservative_28_run_core_hours'] = 1.25 * projected_seconds * threads / 3600
        receipt['projection_policy'] = '50-update D training cost x600; four measured source/checkpoint passes per arm; all28arms;25percent margin'
        save(folder / 'profile.json', receipt)
    return receipt


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.csv.tmp')
    with open(temporary, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def select(root=ROOT, point='core'):
    """Common four-seed update per arm; never use target evaluator outcomes."""
    folder = repair_path(root, point)
    selected, primary, source_rows, curves = [], [], [], []
    for arm in ARMS:
        receipts = [read(folder / f'{arm}_s{s}' / 'receipt.json') for s in SEEDS]
        assert all(r['outcome'] == 'completed' and r['completed_updates'] == 600 for r in receipts)
        by_step = {}
        for step in CHECKPOINTS:
            entries = [next(e for e in r['checkpoints'] if e['updates'] == step) for r in receipts]
            by_step[step] = sum(read(e['source_metrics'])['equal_event_water_IoU'] for e in entries) / 4
        chosen = min(CHECKPOINTS, key=lambda s: (-by_step[s], s))
        for receipt in receipts:
            chosen_entry = next(e for e in receipt['checkpoints'] if e['updates'] == chosen)
            selected.append(dict(chosen_entry, endpoint='source_selected', source_selected_update=chosen))
            primary.append(dict(next(e for e in receipt['checkpoints'] if e['updates'] == 600), endpoint='primary_600'))
            source_rows.append(dict(arm=arm, seed=receipt['seed'], selected_update=chosen,
                                    four_seed_mean=by_step[chosen]))
            with open(folder / f'{arm}_s{receipt["seed"]}' / 'learning_curves.csv') as stream:
                curves.extend(csv.DictReader(stream))
    inputs = validate_inputs(input_path(root, point))
    import torch
    baseline = []
    for seed in SEEDS:
        for arm, key in [('Q0', 'entry'), ('FP', 'FP_parent')]:
            entry = inputs['parents'][str(seed)][key]
            state_hash = tensor_hash(torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)['model'])
            baseline.append(dict(entry, id=f'v11_{point}_{arm}_s{seed}_u0', arm=arm,
                                 point=point, step=0, updates=0, endpoint='baseline',
                                 fp_reference=inputs['parents'][str(seed)]['FP_parent']['id'],
                                 quantized=key == 'entry', precision=inputs['precision'] if key == 'entry' else 'FP32',
                                 state_sha256=state_hash,
                                 source_metrics=str(folder / f'{arm}_s{seed}' / 'source_step0000.json')))
    models, seen = [], {}
    for entry in baseline + primary + selected:
        key = (entry['graph'], entry['seed'], entry['quantized'], entry['state_sha256'])
        if key in seen:
            if entry['endpoint'] not in seen[key]['endpoints']:
                seen[key]['endpoints'].append(entry['endpoint'])
            seen[key]['aliases'].append(dict(arm=entry['arm'], step=entry.get('updates', 0),
                                             endpoint=entry['endpoint'], id=entry['id']))
            continue
        model_entry = dict(entry, step=entry.get('updates', 0), endpoints=[entry['endpoint']],
                           aliases=[dict(arm=entry['arm'], step=entry.get('updates', 0), endpoint=entry['endpoint'], id=entry['id'])])
        metrics = read(entry['source_metrics'])
        model_entry['source_matched_tau'] = metrics.get('source_matched_tau')
        model_entry['source_positive_fraction'] = metrics.get('source_positive_fraction')
        models.append(model_entry)
        seen[key] = model_entry
    result = dict(outcome='completed', point=point, primary=primary, source_selected=selected,
                  models=models, arms={arm: next(r['selected_update'] for r in source_rows if r['arm'] == arm) for arm in ARMS},
                  selection=source_rows, source_only=True,
                  primary_endpoint=600, extension='DEFERRED: full extension requires a new protected all-arm budget reservation')
    by_arm = {arm: {e['seed']: read(e['source_metrics']) for e in primary if e['arm'] == arm} for arm in ARMS}
    contrasts = {}
    for other in ['U', 'A', 'L', 'N', 'P']:
        differences = [by_arm['D'][s]['equal_event_water_IoU'] - by_arm[other][s]['equal_event_water_IoU'] for s in SEEDS]
        contrasts['D-' + other] = dict(seed_differences=dict(zip(map(str, SEEDS), differences)),
                                        mean=float(sum(differences) / 4), positive_seeds=sum(d > 0 for d in differences))
    dry_differences = []
    for other in ['U', 'A']:
        for threshold in ['0.001', '0.01', '0.05']:
            values = [(by_arm['D'][s]['dry_window_alarms'][threshold]['rate'], by_arm[other][s]['dry_window_alarms'][threshold]['rate']) for s in SEEDS]
            if all(a is not None and b is not None for a, b in values):
                dry_differences.append(sum(a - b for a, b in values) / 4)
    result['source_screening'] = dict(contrasts=contrasts,
        source_signal=all(contrasts['D-' + a]['mean'] >= .0025 and contrasts['D-' + a]['positive_seeds'] >= 3 for a in ['U', 'A']),
        dry_safety_supported=len(dry_differences) == 6,
        max_dry_alarm_increase=max(dry_differences) if dry_differences else None,
        followup_supported=all(contrasts['D-' + a]['mean'] >= .0025 and contrasts['D-' + a]['positive_seeds'] >= 3 for a in ['U', 'A']) and len(dry_differences) == 6 and max(dry_differences) <= .01)
    time_to_target = []
    for arm in ARMS:
        for seed in SEEDS:
            rows = [r for r in curves if r['arm'] == arm and int(r['seed']) == seed]
            rows.sort(key=lambda r: int(r['updates']))
            parent_score = read(folder / f'FP_s{seed}' / 'source_step0000.json')['equal_event_water_IoU']
            target = parent_score - .01
            reached = next((i for i, row in enumerate(rows) if float(row['equal_event_water_IoU']) >= target), None)
            record = dict(arm=arm, seed=seed, target_water_IoU=target, source_only=True,
                          total_run_seconds=float(rows[-1]['cumulative_elapsed_seconds']),
                          selection_cost_included=True, crossing='NOT_REACHED')
            if reached is not None:
                record.update(crossing='ALREADY_WITHIN_TOLERANCE' if reached == 0 else 'INTERVAL_CENSORED',
                              first_observed_step=int(rows[reached]['updates']),
                              first_observed_seconds=0. if reached == 0 else float(rows[reached]['cumulative_elapsed_seconds']),
                              interval_lower_step=0 if reached == 0 else int(rows[reached - 1]['updates']),
                              interval_lower_seconds=0. if reached == 0 else float(rows[reached - 1]['cumulative_elapsed_seconds']))
            time_to_target.append(record)
    result['time_to_target'] = time_to_target
    save(folder / 'selection.json', result)
    save(Path(root) / 'source_selection.json', result)
    write_csv(Path(root) / 'repair_learning_curves.csv', curves)
    collect_diagnostics(root)
    return result


def collect_diagnostics(root=ROOT):
    root = Path(root)
    for name in ['priority_diagnostics.csv', 'repair_by_event.csv', 'correction_harm.csv']:
        rows = []
        for path in sorted((root / 'source_diagnostics').glob('*/' + name)):
            with open(path) as stream:
                rows.extend(csv.DictReader(stream))
        write_csv(root / name, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--point', choices=['core', 'transfer'], default='core')
    parser.add_argument('action', choices=['resolve-inputs', 'source-gate', 'profile', 'train', 'select'])
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--arm', choices=ARMS, default='D')
    args = parser.parse_args()
    from common11 import source_access_guard
    source_access_guard()
    if args.action == 'resolve-inputs':
        result = resolve_inputs(args.root, args.point)
    elif args.action == 'source-gate':
        result = source_gate(args.root, args.point, args.threads)
    elif args.action in ['profile', 'train']:
        result = train(args.root, args.point, args.threads, args.arm, args.seed, args.action == 'profile')
    else:
        result = select(args.root, args.point)
    print(json.dumps({k: v for k, v in result.items() if k in ['outcome', 'point', 'completed_updates', 'charged_core_hours', 'conservative_28_run_core_hours']}), flush=True)


if __name__ == '__main__':
    main()
