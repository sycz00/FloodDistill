"""Finite compression jobs using the existing Trainer, readers and fixed QAT code."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('WANDB_MODE', 'disabled')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from experiments.revision_queue import atomic_json, read_json, file_hash, now
from models.compression import build_compression_student, compression_graph_record
from models.compression_quant import (prepare_compression_qat, rebind_compression_bits,
    load_compression_qat, quantization_mode, compression_train_mode, precision_map)
from distillation import compression_metric as cm
from training.compression_loss import CompressionLoss, FeatureCapture
from training.trainer import Trainer
from config import TrainConfig
from utils.seeding import seed_basic
from evaluation.revision_eval import scores


class BudgetStop(RuntimeError):
    pass


class InterruptibleLoader:
    def __init__(self, loader, trainer):
        self.loader, self.trainer = loader, trainer

    def __getattr__(self, key):
        return getattr(self.loader, key)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if self.trainer.stop_requested:
                raise BudgetStop('owned command budget requested checkpoint shutdown')
            yield batch


class CompressionTrainer(Trainer):
    """Preserve the existing loop; only enforce QAT modes and bounded shutdown."""
    def __init__(self, *args, quantized=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized, self.stop_requested = quantized, False
        self.current_epoch = 0
        self.epoch_records = []
        self.train_dl = InterruptibleLoader(self.train_dl, self)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop_requested', True))

    def _build_optimizer(self, model, cfg):
        self.optimizer = super()._build_optimizer(model, cfg)
        return self.optimizer

    def _train_epoch_distill(self, model, optimizer):
        if self.quantized:
            # The fixed mechanistic comparison enables QDQ for all ten epochs.
            # Historical one-epoch-off preparation remains a distinct reference.
            compression_train_mode(model, enabled=True)
        output = super()._train_epoch_distill(model, optimizer)
        self.current_epoch += 1
        return output

    def evaluate(self, model, loader):
        handle = model.register_forward_hook(finite_output)
        try:
            if self.quantized:
                with quantization_mode(model, True):
                    return super().evaluate(model, loader)
            return super().evaluate(model, loader)
        finally:
            handle.remove()

    def _log(self, epoch, values, prefix='train'):
        self.epoch_records.append({'epoch': epoch + 1, 'phase': prefix, **values})
        super()._log(epoch, values, prefix)

    def fit(self):
        try:
            return super().fit()
        except BudgetStop:
            self.save_checkpoint(self.model, self.optimizer, self.current_epoch, {},
                'interrupted.pt', selection_state={'interrupted': True,
                 'partial_epoch_not_complete': True, 'completed_updates': self._global_step})
            raise


def save_tensor(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return file_hash(path)


def load_tensor(path, expected=None):
    if expected is not None and file_hash(path) != expected:
        raise RuntimeError(f'upstream artifact hash changed: {path}')
    return torch.load(path, map_location='cpu', weights_only=True)


def teacher(spec):
    from models.teacher import build_teacher
    if file_hash(spec['teacher']) != spec['teacher_sha256']:
        raise RuntimeError('teacher identity changed')
    model = build_teacher(variant='prithvi_eo_v2_300_tl', num_classes=2,
                         pretrained=False, freeze=True, official_ckpt=spec['teacher'])
    return model.to('cuda').eval().requires_grad_(False)


def finite_output(module, inputs, output):
    if (not torch.is_tensor(output) or output.ndim != 4 or output.shape[1] != 2
            or output.shape[0] != inputs[0].shape[0] or output.shape[2:] != inputs[0].shape[2:]
            or not torch.isfinite(output).all()):
        raise RuntimeError('model produced invalid/nonfinite two-class full-resolution logits')


def pseudo(spec, augment=False):
    from data.pseudo_labeled import PseudoLabeledDataset
    return PseudoLabeledDataset(spec['pseudo'], n_samples=2500, seed=spec['seed'],
        use_curated=True, index_file=spec['index'], extra_keys=frozenset(),
        mult_noise=augment, geom_aug=augment)


def loaders(spec):
    from data.sen1floods11 import get_dataloaders
    _, val, test = get_dataloaders(spec['roots']['s11'], batch_size=4,
        num_workers=spec['workers'], img_size=224, eval_img_size=512, drop_last=False)
    assert len(val.dataset) == 89 and len(test.dataset) == 90
    actual = [Path(p).name for p in val.dataset.image_files]
    if actual != spec['quantizer_calibration_ids']:
        raise RuntimeError('allowed S11 validation calibration image IDs/order changed')
    return val, test


def pool_loader(spec, name, batch_size=16):
    ds = pseudo(spec)
    by_name = {p.name: i for i, p in enumerate(ds.files)}
    indices = [by_name[row['file']] for row in spec['feature_pools'][name]]
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False,
                      num_workers=spec['workers'])


def vectors(model, capture, loader, which):
    vv, ww = [], []
    with torch.no_grad():
        for batch in loader:
            model(batch['image'].cuda().float())
            feat = capture.teacher if which == 'teacher' else capture.student
            v, w = cm.probe_features(feat, batch['mask'].cuda())
            vv.append(v.cpu().double()); ww.append(w.cpu())
            capture.clear()
    return torch.cat(vv), torch.cat(ww)


def completed(out, start, **payload):
    atomic_json(out, {'status': 'complete', 'outcome': 'completed', 'completed_at': now(),
        'elapsed_seconds': time.monotonic() - start, **payload})


def skipped(out, reason, **payload):
    atomic_json(out, {'status': 'complete', 'outcome': 'skipped', 'reason': reason, **payload})


def preflight(spec, out):
    """Short real 16-image FP/QAT steps; no source or external accuracy selection."""
    start = time.monotonic(); seed_basic(spec['seed'])
    t = teacher(spec)
    model = build_compression_student('A0', pretrained=False).cuda()
    ds = pseudo(spec, augment=True)
    batch = next(iter(DataLoader(ds, batch_size=16, num_workers=spec['workers'])))
    x, y = batch['image'].cuda(), batch['mask'].cuda()
    loss = CompressionLoss(t, model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.05)
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(7):
        optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); tic = time.monotonic()
        z = model(x); value, _ = loss(z, x, y)
        if not torch.isfinite(value):
            raise RuntimeError('nonfinite real FP preflight loss')
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step(); torch.cuda.synchronize(); elapsed.append(time.monotonic() - tic)
    fp_peak = torch.cuda.max_memory_reserved() // (1024 ** 2)
    del loss, optimizer
    qmodel = prepare_compression_qat(model, [x[:1]], weight_bits=4, activation_bits=8)
    qloss = CompressionLoss(t, qmodel).cuda()
    optimizer = torch.optim.AdamW(qmodel.parameters(), lr=1e-5, weight_decay=1e-5)
    compression_train_mode(qmodel, True)
    qelapsed = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); tic = time.monotonic()
        value, _ = qloss(qmodel(x), x, y)
        if not torch.isfinite(value):
            raise RuntimeError('nonfinite real QAT preflight loss')
        value.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in qmodel.parameters()):
            raise RuntimeError('nonfinite real QAT preflight gradient')
        torch.nn.utils.clip_grad_norm_(qmodel.parameters(), 1.)
        optimizer.step(); torch.cuda.synchronize(); qelapsed.append(time.monotonic() - tic)
    val, _ = loaders(spec)
    vx = next(iter(val))['image'].cuda()
    with torch.no_grad():
        qmodel.eval(); qmodel(vx)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved() // (1024 ** 2)
    completed(out, start, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        fp_step_seconds=float(np.median(elapsed[2:])), qat_step_seconds=float(np.median(qelapsed)),
        fp_peak_reserved_mb=fp_peak, peak_reserved_mb=peak, batch_size=16, eval_batch_size=4,
        main_loss_valid=True, teacher_frozen=all(not p.requires_grad for p in t.parameters()),
        hardware=torch.cuda.get_device_name(), no_accuracy_inspected=True)


def feature_transform(spec, out):
    start = time.monotonic(); seed_basic(spec['seed'])
    t = teacher(spec); stub = build_compression_student('A3', pretrained=False).cuda()
    capture = FeatureCapture(stub, t)
    fv, fw = vectors(t, capture, pool_loader(spec, 'fit'), 'teacher')
    iv, iw = vectors(t, capture, pool_loader(spec, 'interaction'), 'teacher')
    capture.close()
    transform, info = cm.fit_teacher_pca(fv, fw, min_student_dim=spec['smallest_tap_dimension'])
    tensor_path = Path(out).with_suffix('.pt')
    sha = save_tensor(tensor_path, {'transform': transform.state_dict(), 'fit': fv,
                                  'fit_weights': fw, 'interaction': iv, 'interaction_weights': iw})
    completed(out, start, tensor_path=str(tensor_path), tensor_sha256=sha, pca=info,
        pools_sha256=spec['feature_pools_sha256'], fit_n=len(fv), interaction_n=len(iv))


def timing(spec, out):
    """Inputs-only resource timing; discard all temporary weights and scores."""
    start = time.monotonic(); seed_basic(spec['seed']); torch.cuda.reset_peak_memory_stats()
    t = teacher(spec); model = build_compression_student(spec.get('graph', 'A0'), pretrained=False).cuda().eval()
    val, _ = loaders(spec)
    cal = DataLoader(val.dataset, batch_size=1, shuffle=False, num_workers=spec['workers'])
    tic = time.monotonic()
    model = prepare_compression_qat(model, ({'image': b['image'].cuda()} for b in cal))
    calibration_seconds = time.monotonic() - tic
    tic = time.monotonic()
    with torch.no_grad():
        for batch in val:
            z = model(batch['image'].cuda().float()); finite_output(model, (batch['image'],), z)
    torch.cuda.synchronize(); val_seconds = time.monotonic() - tic
    ds = pseudo(spec, augment=True)
    dl = DataLoader(ds, batch_size=16, num_workers=spec['workers'], drop_last=True)
    tic = time.monotonic()
    for _ in dl: pass
    loader_seconds = time.monotonic() - tic
    batch = next(iter(dl)); x, y = batch['image'].cuda().float(), batch['mask'].cuda()
    capture = FeatureCapture(model, t)
    with torch.no_grad(): model(x); t(x)
    dim = capture.teacher.shape[1]; student_dim = capture.student.shape[1]; capture.close()
    # Fixed dummy buffers measure plumbing cost, never a fitted study transform.
    transform = cm.FrozenTeacherTransform(torch.zeros(dim), torch.eye(dim)[:, :32], torch.ones(32))
    adapter = cm.FrozenFeatureAdapter(torch.eye(student_dim)[:, :32], torch.zeros(32))
    loss = CompressionLoss(t, model, transform, adapter, torch.eye(32), beta=.01).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    elapsed = []
    compression_train_mode(model, True)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); tic = time.monotonic()
        value, _ = loss(model(x), x, y); value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize(); elapsed.append(time.monotonic() - tic)
    loss.remove_hooks()
    completed(out, start, calibration_seconds=calibration_seconds, validation_pass_seconds=val_seconds,
        epoch_loader_seconds=loader_seconds, feature_qat_step_seconds=float(np.median(elapsed[1:])),
        diagnostic_step_seconds=elapsed[0], peak_reserved_mb=torch.cuda.max_memory_reserved() // 2**20,
        graph=spec.get('graph', 'A0'),
        batch_size=16,
        no_accuracy_computed=True, purpose='resource timing only; temporary weights/transform discarded')


def warmup(spec, out):
    start = time.monotonic(); seed_basic(spec['seed'])
    t = teacher(spec); model = build_compression_student(spec['graph'], pretrained=False)
    val, test = loaders(spec)
    train = DataLoader(pseudo(spec, augment=True), batch_size=16, shuffle=True,
        num_workers=spec['workers'], pin_memory=True, drop_last=True)
    assert len(train) == 156
    cfg = TrainConfig(epochs=50, lr=5e-5, weight_decay=.05, task_loss='focal',
        use_amp=False, grad_clip=5., early_stop_patience=20, val_smoothing_window=1,
        val_every_n_epochs=2, defer_terminal_evaluation=True,
        out_dir=spec['run_dir'], seed=spec['seed'], gpu_ids=[0])
    loss = CompressionLoss(t, model).cuda()
    trainer = CompressionTrainer(model, train, val, test, cfg, loss_fn=loss)
    result = trainer.fit(); loss.remove_hooks()
    best = Path(spec['run_dir']) / 'best.pt'; last = Path(spec['run_dir']) / 'last.pt'
    bp, lp = load_tensor(best), load_tensor(last)
    completed(out, start, graph=spec['graph'], graph_record=compression_graph_record(model, spec['graph']),
        best_path=str(best), best_sha256=file_hash(best), last_path=str(last), last_sha256=file_hash(last),
        selected_updates=bp['training_state']['global_step'], completed_updates=lp['training_state']['global_step'],
        selected_epoch=bp['epoch'] + 1, completed_epochs=lp['epoch'] + 1,
        source_validation=result, epochs=trainer.epoch_records, terminal_evaluation='deferred')


def transform_and_cache(spec):
    receipt = read_json(spec['transform_receipt'])
    cache = load_tensor(receipt['tensor_path'], receipt['tensor_sha256'])
    tr = cache['transform']
    transform = cm.FrozenTeacherTransform(tr['mean'], tr['basis'], tr['scales'])
    return transform, cache


def validation(model, spec, paired=False, teacher_model=None):
    from evaluation.revision_native_border import confusion
    from evaluation.revision_eval import NamedDataset
    val, _ = loaders(spec)
    val = DataLoader(NamedDataset(val.dataset, [Path(p).name for p in val.dataset.image_files]),
                     batch_size=4, shuffle=False, num_workers=spec['workers'])
    scene, events = {}, defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
    flip = {'valid_pixels': 0, 'harmful_count': 0, 'repaired_count': 0,
            'fp_correct_count': 0, 'quantized_correct_count': 0}
    teacher_disagree, teacher_valid = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in val:
            x = batch['image'].cuda().to(next(model.parameters()).dtype)
            y = batch['mask']
            with quantization_mode(model, True):
                logits = model(x)
            finite_output(model, (x,), logits)
            if paired:
                with quantization_mode(model, False):
                    fp = model(x)
                finite_output(model, (x,), fp)
                stats = cm.accuracy_damage_repair(fp, logits, y.cuda())
                for key in flip:
                    flip[key] += stats[key]
            pred = logits.argmax(1).cpu().numpy()
            if teacher_model is not None:
                reference = teacher_model(x.float()).argmax(1).cpu().numpy()
                support = y.numpy() != -1
                teacher_disagree += int(((pred != reference) & support).sum())
                teacher_valid += int(support.sum())
            for i, name in enumerate(batch['filename']):
                name = Path(name).name; cell = confusion(pred[i], y[i].numpy())
                scene[name] = cell.tolist(); events[name.split('_')[0]] += cell
    result = {'micro': scores(sum(np.asarray(x) for x in scene.values())), 'scene': scene,
        'event': {k: v.tolist() for k, v in events.items()},
        'scene_metrics': {k: scores(v) for k, v in scene.items()},
        'event_metrics': {k: scores(v) for k, v in events.items()}, 'n_images': len(scene)}
    if paired:
        n = flip['valid_pixels']
        # Count identity, not an IoU identity.
        assert flip['quantized_correct_count'] - flip['fp_correct_count'] == flip['repaired_count'] - flip['harmful_count']
        flip.update(H=flip['harmful_count'] / n, B=flip['repaired_count'] / n,
            accuracy_delta=(flip['quantized_correct_count'] - flip['fp_correct_count']) / n,
            identity_verified=True, reference='same post-QAT master weights with quantization bypassed')
        result['damage_repair'] = flip
    if teacher_model is not None:
        result['teacher_disagreement'] = {'pixels': teacher_disagree, 'valid_pixels': teacher_valid,
             'fraction': teacher_disagree / teacher_valid, 'interpretation': 'student versus frozen teacher; separate from GT correctness'}
    return result


def calibrate(spec, out):
    start = time.monotonic(); seed_basic(spec['seed'])
    upstream = read_json(spec['warmup_receipt'])
    payload = load_tensor(upstream['best_path'], upstream['best_sha256'])
    model = build_compression_student(spec['graph'], pretrained=False).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    transform, cache = transform_and_cache(spec)
    fp_val = validation(model, spec)
    val, _ = loaders(spec)
    cal = DataLoader(val.dataset, batch_size=1, shuffle=False, num_workers=spec['workers'])
    q8 = prepare_compression_qat(model, ({'image': b['image'].cuda()} for b in cal),
                                calibration_seed=spec['calibration_seed'])
    # Fit in exactly the same prepared FP-bypass arithmetic/fusion as the
    # quantized branch (the historical LiteMLA wrapper clamps its denominator).
    class Holder: pass
    holder = Holder(); holder.neck = torch.nn.Identity()
    capture = FeatureCapture(q8, holder)
    with quantization_mode(q8, False):
        sv, weights = vectors(q8, capture, pool_loader(spec, 'fit'), 'student')
    capture.close()
    assert torch.equal(weights, cache['fit_weights'])
    adapter, ridge = cm.fit_ridge_adapter(sv, transform(cache['fit']), weights)
    try:
        fp16_validation = {'status': 'complete', **validation(copy.deepcopy(model).half(), spec)}
    except (RuntimeError, TypeError) as error:
        fp16_validation = {'status': 'blocked', 'reason': f'{type(error).__name__}: {error}'}
    results = {}
    for precision, bits in spec['ptq_precisions'].items():
        qm = rebind_compression_bits(q8, *bits)
        results[precision] = validation(qm, spec, paired=True)
        del qm
    artifact = Path(out).with_suffix('.pt')
    sha = save_tensor(artifact, {'model': q8.cpu().state_dict(), 'adapter': adapter.state_dict(),
                               'transform': transform.state_dict()})
    completed(out, start, graph=spec['graph'], common_checkpoint_sha256=upstream['best_sha256'],
        tensor_path=str(artifact), tensor_sha256=sha, ridge=ridge, fp_validation=fp_val,
        fp16_validation=fp16_validation,
        ptq_validation=results, precision_map=precision_map(q8),
        calibration_ids=spec['quantizer_calibration_ids'], calibration_labels_used=False,
        status_label='simulation_only')


def fitted_modules(payload):
    tr, ad = payload['transform'], payload['adapter']
    return (cm.FrozenTeacherTransform(tr['mean'], tr['basis'], tr['scales']),
            cm.FrozenFeatureAdapter(ad['weight'], ad['bias']))


def interaction_moments(model, adapter, transform, cache, spec):
    class Holder: pass
    holder = Holder(); holder.neck = torch.nn.Identity()
    capture = FeatureCapture(model, holder)
    columns = defaultdict(list); offset = 0
    with torch.no_grad():
        for batch in pool_loader(spec, 'interaction'):
            x, y = batch['image'].cuda().float(), batch['mask'].cuda()
            with quantization_mode(model, False):
                model(x); sv, weights = cm.probe_features(capture.student, y); sf = adapter(sv).cpu().double()
            with quantization_mode(model, True):
                model(x); qv, qw = cm.probe_features(capture.student, y); sq = adapter(qv).cpu().double()
            n = len(x); w = weights.cpu(); assert torch.equal(w, qw.cpu())
            assert torch.equal(w, cache['interaction_weights'][offset:offset + n])
            tv = transform(cache['interaction'][offset:offset + n]); offset += n
            moment = cm.per_image_moments(tv - sf, sf - sq, w)
            for key, value in moment.items(): columns[key].append(value)
            capture.clear()
    capture.close()
    return {key: torch.cat(values) for key, values in columns.items()}


def interaction(spec, out):
    start = time.monotonic(); seed_basic(spec['seed'])
    common = read_json(spec['calibration_receipt'])
    payload = load_tensor(common['tensor_path'], common['tensor_sha256'])
    model = load_compression_qat(build_compression_student(spec['graph'], pretrained=False), payload['model']).cuda()
    model = rebind_compression_bits(model, *spec['bits'], inplace=True)
    transform, adapter = fitted_modules(payload)
    _, cache = transform_and_cache(spec)
    per_image = interaction_moments(model, adapter, transform, cache, spec)
    moments = cm.aggregate_moments(per_image)
    groups = [row['group'] for row in spec['feature_pools']['interaction']]
    metric, info = cm.fit_interaction_metric(per_image, groups, gamma=1., bootstrap_resamples=200, seed=1729)
    controls = {'B2': torch.eye(metric.shape[0], dtype=metric.dtype), 'B3': metric,
                **cm.matched_spectrum_controls(metric, moments, seed=1729)}
    scales = {key: cm.initial_loss_scale(value, moments['Ce']) for key, value in controls.items()}
    # beta is measured on four fixed training-only minibatches and the same shared stage.
    t = teacher(spec)
    tr, ad = copy.deepcopy(transform).cuda(), copy.deepcopy(adapter).cuda()
    loss = CompressionLoss(t, model, tr, ad, controls['B2'], beta=0., diagnostic_interval=1).cuda()
    base_norms, feature_norms = [], []
    with quantization_mode(model, True):
        for i, batch in enumerate(pool_loader(spec, 'fit')):
            if i == 4: break
            x, y = batch['image'].cuda().float(), batch['mask'].cuda()
            loss(model(x), x, y)
            diag = loss.diagnostics[-1]
            base_norms.append(diag['task_gradient_rms']); feature_norms.append(diag['feature_gradient_rms'])
            model.zero_grad(set_to_none=True)
    loss.remove_hooks()
    beta, beta_info = cm.beta_from_gradient_norms(base_norms, feature_norms)
    beta_info.update(calibration_image_ids=[row['file'] for row in spec['feature_pools']['fit'][:64]],
        shared_parameter_block='model.backbone.stages.3' if spec['graph'] != 'M0' else 'encoder.extractor.features.12',
        input_geometry='unaugmented training224; same four16-image batches; quantized eval mode; gradients reset')
    final = Path(out).with_suffix('.pt')
    artifact = {**payload, 'model': model.cpu().state_dict(), 'metrics': controls,
                'scales': scales, 'beta': beta, 'per_image_moments': per_image, 'moments': moments}
    sha = save_tensor(final, artifact)
    completed(out, start, graph=spec['graph'], precision=spec['precision'], metric=info,
        tensor_path=str(final), tensor_sha256=sha, beta_rule=beta_info, loss_scales=scales,
        initial_feature_losses={key: float(scales[key] * torch.trace(value @ moments['Ce']) / metric.shape[0])
                                for key, value in controls.items()},
        nonidentity=not torch.equal(metric, torch.eye(metric.shape[0], dtype=metric.dtype)),
        precision_map=precision_map(model), independent_groups=len(set(groups)),
        split_group_stability=cm.split_group_stability(per_image, groups, seed=1729))


def source_gate(spec, out):
    def read_optional(path):
        return read_json(path) if Path(path).exists() else {'status': 'missing', 'outcome': 'unavailable'}
    interaction_receipt = read_optional(spec['interaction_receipt'])
    b1, b2, b3 = [read_optional(p) for p in spec['method_receipts']]
    selected = spec.get('primary_cell', {'graph': 'A3', 'precision': 'W4A4', 'bits': [4, 4],
                                       'interaction_receipt': spec['interaction_receipt']})
    technical_invalid = interaction_receipt.get('status') != 'complete' or any(r.get('status') != 'complete' for r in (b1, b2, b3))
    primary_outcomes = [r.get('outcome') for r in (interaction_receipt, b1, b2, b3)]
    did_fallback = False
    if technical_invalid and spec.get('fallback_cell'):
        did_fallback = True
        selected = spec['fallback_cell']
        interaction_receipt = read_optional(selected['interaction_receipt'])
        b1, b2, b3 = [read_optional(p) for p in spec['fallback_method_receipts']]
    eligible = (interaction_receipt.get('nonidentity', False) and
                all(r['outcome'] == 'completed' for r in (b1, b2, b3)))
    effect = None
    if eligible:
        effect = b3['validation']['micro']['mIoU'] - max(b1['validation']['micro']['mIoU'], b2['validation']['micro']['mIoU'])
    atomic_json(out, {'status': 'complete', 'outcome': 'completed', 'source_only': True,
        'eligible': eligible and effect >= .005, 'improvement_over_both': effect,
        'selected_cell': selected, 'technical_fallback_used': did_fallback,
        'primary_technically_invalid': technical_invalid, 'primary_outcomes': primary_outcomes,
        'predeclared_threshold': .005, 'no_external_scores_read': True,
        'reason': 'nonidentity and >=0.005 source improvement required for matched-spectrum controls'})


def qat(spec, out):
    start = time.monotonic(); seed_basic(spec['seed'])
    if spec.get('expansion_gate'):
        gate = read_json(spec['expansion_gate'])
        if not gate['eligible']:
            skipped(out, 'predeclared source-only expansion gate not met'); return
        spec = {**spec, **gate['selected_cell']}
    receipt = read_json(spec['interaction_receipt'])
    if spec['method'] == 'B3' and not receipt['nonidentity']:
        skipped(out, 'B3 is exact identity; reuse the same B2 intervention', duplicate_of=spec['duplicate_receipt']); return
    payload = load_tensor(receipt['tensor_path'], receipt['tensor_sha256'])
    model = load_compression_qat(build_compression_student(spec['graph'], pretrained=False), payload['model'])
    transform, adapter = fitted_modules(payload)
    if spec['method'] == 'B1':
        loss = CompressionLoss(teacher(spec), model)
    else:
        key = spec.get('feature_metric_key', spec['method'])
        loss = CompressionLoss(teacher(spec), model, transform, adapter, payload['metrics'][key],
                               payload['scales'][key], payload['beta'] * spec.get('beta_multiplier', 1.))
    loss.cuda()
    val, test = loaders(spec)
    train = DataLoader(pseudo(spec, augment=True), batch_size=16, shuffle=True,
        num_workers=spec['workers'], pin_memory=True, drop_last=True)
    # Fixed ten-epoch mechanistic continuation, historical QAT optimizer semantics.
    cfg = TrainConfig(epochs=10, lr=1e-5, weight_decay=1e-5, task_loss='focal', use_amp=False,
        grad_clip=1., early_stop_patience=0, val_smoothing_window=1, val_every_n_epochs=2,
        defer_terminal_evaluation=True, out_dir=spec['run_dir'], seed=spec['seed'], gpu_ids=[0])
    trainer = CompressionTrainer(model, train, val, test, cfg, loss_fn=loss, quantized=True)
    trainer.fit()
    diagnostics = loss.diagnostics; loss.remove_hooks()
    last = load_tensor(Path(spec['run_dir']) / 'last.pt')
    assert last['training_state']['global_step'] == 1560
    best = Path(spec['run_dir']) / 'best.pt'; bp = load_tensor(best)
    model.load_state_dict(bp['model'], strict=True)
    val_result = validation(model.cuda(), spec, paired=True,
                            teacher_model=object.__getattribute__(loss, '_teacher'))
    transform, cache = transform_and_cache(spec)
    _, adapter = fitted_modules(payload)
    final_per_image = interaction_moments(model, adapter, transform, cache, spec)
    final_moments = cm.aggregate_moments(final_per_image)
    completed(out, start, graph=spec['graph'], precision=spec['precision'], method=spec['method'],
        best_path=str(best), best_sha256=file_hash(best), selected_updates=bp['training_state']['global_step'],
        completed_updates=1560, fixed_budget=True, quantization_off_initial_epochs=0,
        common_quant_state_sha256=receipt['tensor_sha256'], validation=val_result,
        final_residual_diagnostics=cm.residual_diagnostics(final_moments),
        final_split_group_stability=cm.split_group_stability(final_per_image,
            [row['group'] for row in spec['feature_pools']['interaction']], seed=1729),
        gradient_diagnostics=diagnostics, epochs=trainer.epoch_records, status_label='simulation_only')


def terminal(spec, out):
    start = time.monotonic()
    if not Path(spec['method_receipt']).exists():
        skipped(out, 'upstream training unavailable; no frozen selected checkpoint'); return
    receipt = read_json(spec['method_receipt'])
    if receipt['outcome'] == 'skipped':
        skipped(out, receipt['reason']); return
    if receipt.get('status') != 'complete':
        skipped(out, 'upstream training failed; partial weights cannot enter terminal evaluation'); return
    spec = {**spec, 'graph': receipt['graph'],
            'precision': receipt.get('precision', 'FP32')}
    payload = load_tensor(receipt['best_path'], receipt['best_sha256'])
    graph = build_compression_student(spec['graph'], pretrained=False)
    if spec['quantized']:
        model = load_compression_qat(graph, payload['model']).cuda().eval()
    else:
        graph.load_state_dict(payload['model'], strict=True); model = graph.cuda().eval()
    # Pin the selected checkpoint before touching any test/OOD image.
    atomic_json(Path(out).with_suffix('.freeze.json'), {'checkpoint_sha256': receipt['best_sha256'],
        'source_design_frozen': True, 'source_gate_sha256': file_hash(spec['design_gate']), 'at': now()})
    from data.ood_datasets import STURMFloodDataset, WorldFloodsDataset
    from data.channel_adapter import sturm_s2_adapter
    from evaluation import perscene_eval as pe
    from evaluation import revision_native_border as rb
    from evaluation.revision_eval import NamedDataset
    pe.STURM = spec['roots']['sturm']; map_of, event_of = pe.sturm_maps()
    _, test = loaders(spec)
    s11 = NamedDataset(test.dataset, [Path(p).name for p in test.dataset.image_files])
    sturm = STURMFloodDataset(spec['roots']['sturm'], sensor='S2', img_size=512,
                             normalize='original', class_mapping='paper')
    wf = WorldFloodsDataset(spec['roots']['wf'], split='test', img_size=512, tile_size=512,
                            normalize='s11', permanent_water_as='ignore')
    assert (len(s11), len(sturm), len(wf)) == (90, 2675, 785)
    plan = read_json(spec['overlap_plan'])
    assert spec['excluded_sturm_chips'] == plan['exclude_complete_STURM_chips']
    assert spec['excluded_wf_maps'] == plan['exclude_complete_WorldFloods_maps']
    excluded_sturm = set(spec['excluded_sturm_chips']); excluded_wf = set(spec['excluded_wf_maps'])
    results = {}
    adapter = sturm_s2_adapter(target_channels=6).cuda()
    with torch.no_grad(), quantization_mode(model, True):
        for name, ds, bs in (('s11', s11, 4), ('sturm', sturm, 8), ('wf', wf, 4)):
            images, maps, events = {}, defaultdict(lambda: np.zeros((2, 3), dtype=np.int64)), defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
            excluded_maps, excluded_events = defaultdict(lambda: np.zeros((2, 3), dtype=np.int64)), defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
            included_images = 0
            for begin in range(0, len(ds), bs):
                ids = list(range(begin, min(begin + bs, len(ds))))
                items = [rb.native_item(ds, i) if name == 'wf' else ds[i] for i in ids]
                inputs = torch.stack([item['image'] for item in items]).cuda().float()
                if name == 'sturm': inputs = adapter(inputs)
                logits = model(inputs); finite_output(model, (inputs,), logits)
                predictions = logits.argmax(1).cpu().numpy()
                for i, item, pred in zip(ids, items, predictions):
                    filename = Path(item['filename']).name
                    mask = item['mask']; mask = mask.numpy().copy() if torch.is_tensor(mask) else mask.copy()
                    if name == 'sturm':
                        _, invalid = rb.sturm_invalid(ds, i); mask[invalid] = -1
                        map_id, event = map_of[filename], event_of[filename]
                        omit = filename in excluded_sturm
                    elif name == 'wf':
                        pred = pred[:item['height'], :item['width']]
                        map_id = filename.split('#')[0]; event = map_id.split('_')[0]; omit = map_id in excluded_wf
                    else:
                        map_id = filename; event = filename.split('_')[0]; omit = False
                    cell = rb.confusion(pred, mask); images[filename] = cell.tolist()
                    maps[map_id] += cell; events[event] += cell
                    if not omit:
                        excluded_maps[map_id] += cell; excluded_events[event] += cell; included_images += 1
            def pack(mm, ee):
                return {'micro': scores(sum(mm.values())), 'map': {k: v.tolist() for k, v in mm.items()},
                        'event': {k: v.tolist() for k, v in ee.items()}, 'n_maps': len(mm), 'n_events': len(ee)}
            results[name] = {'full': pack(maps, events), 'known_overlap_excluded': pack(excluded_maps, excluded_events),
                             'per_image_counts': images, 'n_images': len(ds), 'protocol': spec['protocols'][name]}
            results[name]['known_overlap_excluded']['n_images'] = included_images
            expected = {'s11': (90, 10, 90, 90), 'sturm': (46, 29, 2675, 2672), 'wf': (18, 11, 785, 733)}[name]
            assert (len(maps), len(events), len(images), included_images) == expected
            if name == 'wf': assert len(excluded_maps) == 17
    from evaluation.compression_artifact import measure_artifact
    artifact = measure_artifact(model, Path(spec['artifact_dir']), spec['graph'], spec['quantized'])
    completed(out, start, graph=spec['graph'], precision=spec['precision'],
        method=spec.get('method', 'B0'),
        checkpoint_path=receipt['best_path'], checkpoint_sha256=receipt['best_sha256'],
        terminal_results=results,
        checkpoint_bytes=Path(receipt['best_path']).stat().st_size,
        parameters=sum(p.numel() for p in model.parameters()), precision_map=precision_map(model),
        artifact=artifact, warmups=20, repeats=200,
        batch_size=1, shape=[1, 6, 512, 512],
        backend=artifact.get('runtime', {}).get('backend', 'not measured'),
        device=torch.cuda.get_device_name(), status_label=artifact['status'],
        deployment_claim=False, target_engine='not available; no executable four-bit target kernel claimed',
        overlap_plan_sha256=file_hash(spec['overlap_plan']))


def design_freeze(spec, out):
    records = {str(p): read_json(p) if Path(p).exists() else {'outcome': 'unavailable'} for p in spec['receipts']}
    atomic_json(out, {'status': 'complete', 'outcome': 'completed', 'frozen_at': now(),
        'source_only': True, 'receipt_hashes': {p: file_hash(p) if Path(p).exists() else None for p in records},
        'source_outcomes': {p: r.get('outcome') for p, r in records.items()},
        'terminal_feedback_prohibited': True})


def collect(spec, out):
    from evaluation.compression_collect import write_results
    write_results(spec, out, spec.get('queue_state'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('preflight', 'timing', 'features', 'warmup', 'calibrate', 'interaction', 'qat', 'gate', 'freeze', 'terminal', 'collect'), required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.spec = args.spec.resolve(); args.out = args.out.resolve()
    spec = read_json(args.spec)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    import cv2; cv2.setNumThreads(1)
    if spec.get('conditional_gate'):
        gate = read_json(spec['conditional_gate'])
        if not gate['eligible']:
            skipped(args.out, 'predeclared source-only expansion gate not met')
            return
        if spec.get('select_gate_cell'):
            spec = {**spec, **{k: gate['selected_cell'][k] for k in ('graph', 'precision', 'bits')}}
    if args.action not in ('gate', 'freeze', 'collect') and not torch.cuda.is_available():
        raise RuntimeError('authorized CUDA runtime unavailable; no silent CPU substitution')
    functions = {'preflight': preflight, 'timing': timing, 'features': feature_transform, 'warmup': warmup,
                 'calibrate': calibrate, 'interaction': interaction, 'qat': qat, 'gate': source_gate,
                 'freeze': design_freeze, 'terminal': terminal, 'collect': collect}
    try:
        functions[args.action](spec, args.out)
    except Exception as error:
        atomic_json(args.out, {'status': 'failed', 'outcome': 'budget_exhausted' if isinstance(error, BudgetStop) else 'failed',
                    'error': f'{type(error).__name__}: {error}', 'at': now()})
        raise


if __name__ == '__main__':
    main()
