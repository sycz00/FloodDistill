"""v7 training entry point. Imports are inert; frozen training sources stay untouched."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from v7_loss import RULE_ARMS, NORM_MEAN, NORM_STD, RuleContribution, ContinuationLoss
import common_v9 as c
from data_v9 import dataset


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(graph, method, rungs_dir, norm_mean=None, norm_std=None):
    """Shared strict checkpoint builder; frozen capsule must be on sys.path."""
    source = Path(rungs_dir).resolve() / 'rungs.py'
    module_spec = importlib.util.spec_from_file_location('v7_frozen_rungs', source)
    rungs = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(rungs)
    base = c.build(graph) if graph=='B0' else rungs.build(graph)
    return RuleContribution(base, norm_mean or NORM_MEAN, norm_std or NORM_STD) if method in RULE_ARMS else base


class ObservedLoader:
    """Observe first batches and optionally bound smoke; no random draws."""
    def __init__(self, loader, limit=None):
        self.loader, self.limit = loader, limit
        self.first_minibatch_hashes = []
        self.first_image_hashes=[]

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return min(len(self.loader), self.limit) if self.limit is not None else len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(len(self)):
            batch = next(iterator)
            if len(self.first_minibatch_hashes) < 2:
                self.first_image_hashes.append(state_hash({'image':batch['image']}))
                self.first_minibatch_hashes.append(state_hash(
                    {key: batch[key] for key in ('image', 'mask')}))
            yield batch


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('capsule', 'rungs-dir', 'spec', 'out', 'run-dir', 'graph', 'method'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--lr-from', help='CPU selected LR JSON; graph and lr verified')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--init-checkpoint')
    p.add_argument('--init-sha256')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-updates', type=int, default=3)
    return p


def main():
    a = parser().parse_args()
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(a.capsule).resolve()))
    import scripts.compression_jobs as cj
    from config import TrainConfig
    from utils.seeding import seed_basic
    from training.compression_loss import CompressionLoss
    from utils.metrics import accumulate_metrics, finalize_metrics
    from data.sen1floods11 import NORM_MEAN as frozen_mean, NORM_STD as frozen_std
    assert frozen_mean == NORM_MEAN and frozen_std == NORM_STD, 'Normalization drift'
    continuation = a.method in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL')
    assert a.method in ('ce','E1','E2','E3','human','teacher')
    assert a.graph in ('R4','B0')
    primary=a.graph=='B0'
    assert primary == (a.method in ('human','teacher'))
    c.install_label_guard()
    if a.lr_from:
        choice = json.loads(Path(a.lr_from).read_text())
        assert choice['graph'] == a.graph
        a.lr = float(choice['lr'])
    if continuation:
        assert a.graph == 'R4' and abs(a.lr - 5e-4) < 1e-12
        assert a.init_checkpoint and a.init_sha256, 'Continuation requires verified parent bytes'
    else:
        assert not a.init_checkpoint
    assert a.smoke_updates > 0
    start = time.monotonic()
    spec = json.loads(Path(a.spec).read_text())
    spec.update(seed=a.seed, run_dir=str(Path(a.run_dir).resolve()), graph=a.graph)
    run_dir = Path(a.run_dir).resolve()
    # Resumption of partial DataLoader worker state is deliberately not claimed.
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError('Run directory is nonempty; use a new separately identified attempt')
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_basic(a.seed)
    teacher = c.teacher_cpu(spec)  # Preserve V7 CPU RNG draws; no unused GPU teacher allocation.
    if primary: assert a.lr==5e-5
    model = build_model(a.graph, a.method, a.rungs_dir)
    if continuation:
        parent = cj.load_tensor(a.init_checkpoint, a.init_sha256)
        base = model.base if isinstance(model, RuleContribution) else model
        base.load_state_dict(parent['model'], strict=True)
        parent_seed = parent.get('config', {}).get('seed')
        assert parent_seed == a.seed, (parent_seed, a.seed)
    initial_base_hash = state_hash((model.base if isinstance(model, RuleContribution) else model).state_dict())
    initial_full_hash = state_hash(model.state_dict())
    val, test = cj.loaders(spec)
    raw_train = DataLoader(dataset(spec,a.method,a.seed), batch_size=16, shuffle=True,
                           num_workers=spec['workers'], pin_memory=True, drop_last=True)
    per_epoch=15 if primary else 156
    assert len(raw_train)==per_epoch and len(raw_train.dataset)==(252 if primary else 2500)
    observed_train = ObservedLoader(raw_train, a.smoke_updates if a.smoke else None)
    if a.smoke:
        # One registered-resolution validation minibatch checks the path, never selection science.
        val, test = ObservedLoader(val, 1), ObservedLoader(test, 1)
    scheduled_epochs = 10 if continuation else 50
    cfg = TrainConfig(epochs=scheduled_epochs, lr=a.lr, weight_decay=.05, task_loss='ce' if primary else 'focal',
                      use_amp=False, grad_clip=5., early_stop_patience=0,
                      val_smoothing_window=1, val_every_n_epochs=2,
                      defer_terminal_evaluation=True, out_dir=str(run_dir), seed=a.seed, gpu_ids=[0])
    if continuation:
        loss = ContinuationLoss(teacher, a.method, a.seed)
    elif a.method == 'lkd':
        loss = CompressionLoss(teacher, model).cuda()
    else:
        del teacher
        torch.cuda.empty_cache()
        loss = None
    metadata = {'graph': a.graph, 'method': a.method, 'arm': a.method, 'seed': a.seed, 'lr': a.lr,
                'norm_mean': NORM_MEAN, 'norm_std': NORM_STD,
                'band_order': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2'],
                'parent_path': str(Path(a.init_checkpoint).resolve()) if continuation else None,
                'parent_sha256': a.init_sha256 if continuation else None,
                'initial_base_sha256': initial_base_hash, 'initial_model_sha256': initial_full_hash,
                'rule_contribution': a.method in RULE_ARMS, 'rule_epsilon_reflectance': 1e-6,
                'updates_required':per_epoch*50,
                'scheduler_convention': 'cosine stepped once per 156-update epoch; T_max=10' if continuation
                    else 'inherited cosine stepped once per 156-update epoch; T_max=50',
                'source_validation_frequency_updates':per_epoch*2,
                'selection': 'strict greater source mIoU, smoothing window1; ties retain earliest',
                'start_candidate': continuation, 'batch_norm_policy': 'inherited buffers loaded; all student BN trains',
                'early_stop_patience': 0, 'smoke': a.smoke, 'scientific_run': not a.smoke,
                'teacher_sha256': spec['teacher_sha256'], 'spec_sha256': cj.file_hash(a.spec),
                'workers': spec['workers'], 'index_sha256': cj.file_hash(str(c.ROOT/'source_images/index.json') if primary else spec['index']),
                'augmentation': 'frozen per-band reflectance gain U(.8,1.2) p=.5; shared h/v/rot90 p=.5',
                'frozen_code_sha256': {relative: cj.file_hash(Path(a.capsule) / relative) for relative in
                    ('training/trainer.py', 'training/losses.py', 'training/compression_loss.py',
                     'data/pseudo_labeled.py', 'data/sen1floods11.py', 'scripts/compression_jobs.py')},
                'capsule': str(Path(a.capsule).resolve()),
                'rungs_sha256': cj.file_hash(Path(a.rungs_dir) / 'rungs.py'),
                'v7_code_sha256': {p.name: cj.file_hash(p) for p in
                    (Path(__file__), Path(__file__).with_name('v7_loss.py'))}}

    class V7Trainer(cj.CompressionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_metrics = None
            self.grad_checks = {'updates_checked': 0, 'max_post_clip_gradient_l2': 0.}
            self.timings, self.first_training_logits_sha256 = [], None
            def validate_output(module, inputs, output):
                cj.finite_output(module, inputs, output)
                if module.training and self.first_training_logits_sha256 is None:
                    self.first_training_logits_sha256 = state_hash({'logits': output})
            self.model.register_forward_hook(validate_output)

        def _init_wandb(self):
            return None  # Local bookkeeping; no external service or credentials.

        def _build_optimizer(self, model, config):
            optimizer = super()._build_optimizer(model, config)
            def check_gradients(opt, _args, _kwargs):
                gradients = [p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]
                norm = torch.stack([gradient.detach().norm() for gradient in gradients]).norm()
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient at optimizer step')
                self.grad_checks['updates_checked'] += 1
                self.grad_checks['max_post_clip_gradient_l2'] = max(
                    self.grad_checks['max_post_clip_gradient_l2'], float(norm))
            optimizer.register_step_pre_hook(check_gradients)
            return optimizer

        def _build_scheduler(self, optimizer):
            self.scheduler_ref = super()._build_scheduler(optimizer)
            return self.scheduler_ref

        def _train_epoch_distill(self, model, optimizer):
            tic, before = time.monotonic(), self._global_step
            result = super()._train_epoch_distill(model, optimizer)
            self.timings.append({'phase': 'train', 'seconds': time.monotonic() - tic,
                                 'updates': self._global_step - before})
            return result

        def _train_epoch_supervised(self, model, optimizer):
            tic, before = time.monotonic(), self._global_step
            result = super()._train_epoch_supervised(model, optimizer)
            self.timings.append({'phase': 'train', 'seconds': time.monotonic() - tic,
                                 'updates': self._global_step - before})
            return result

        @torch.no_grad()
        def evaluate(self, model, loader):
            tic = time.monotonic()
            model.eval()
            confusion = None
            total_loss, n_batches = 0., 0
            for batch in loader:
                if self.stop_requested:
                    raise cj.BudgetStop('Controlled stop during validation')
                images = batch['image'].to(self.device, non_blocking=True).float()
                masks = batch['mask'].to(self.device, non_blocking=True).long()
                logits = model(images)
                cj.finite_output(model, (images,), logits)
                confusion = accumulate_metrics(logits.cpu(), masks.cpu(), 2, -1, confusion)
                batch_loss = torch.nn.functional.cross_entropy(logits, masks, ignore_index=-1)
                if torch.isfinite(batch_loss):
                    total_loss += float(batch_loss)
                    n_batches += 1
            metrics = finalize_metrics(confusion)
            tn, fp, fn, tp = [int(i) for i in confusion.reshape(-1)]
            metrics.update(loss=total_loss / max(n_batches, 1),
                water_IoU=metrics['iou_per_class'][1], wIoU=metrics['iou_per_class'][1],
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                confusion_counts={'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}, valid_pixels=tn + fp + fn + tp)
            self.timings.append({'phase': 'source_validation', 'seconds': time.monotonic() - tic,
                                 'batches': len(loader)})
            return metrics

        def save_checkpoint(self, model, optimizer, epoch, metrics, filename='checkpoint.pt',
                            *, scheduler=None, selection_state=None):
            path = super().save_checkpoint(model, optimizer, epoch, metrics, filename,
                                           scheduler=scheduler if scheduler is not None else getattr(self, 'scheduler_ref', None),
                                           selection_state=selection_state)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            payload['v7_metadata'] = metadata
            payload['v7_observations'] = {'first_minibatch_hashes': observed_train.first_minibatch_hashes,
                                          'first_training_logits_sha256': self.first_training_logits_sha256,
                                          'gradient_checks': self.grad_checks, 'timings': self.timings}
            if isinstance(self.loss_fn, ContinuationLoss):
                payload['v7_loss_recovery'] = self.loss_fn.recovery_state()
            payload['data_position_state'] = {'completed_updates': self._global_step,
                'completed_epochs': self._global_step // 156,
                'dataset_jitter_rng': raw_train.dataset._mn_gen.get_state(),
                'dataset_geometry_rng': raw_train.dataset._geom_gen.get_state(),
                'exact_resume': False,
                'limitation': 'DataLoader worker RNG/prefetch state unavailable; retry must be a new attempt'}
            cj.save_tensor(path, payload)
            return path

        def fit(self):
            if not continuation and not a.smoke:
                return super().fit()
            self.model.to(self.device)
            optimizer = self._build_optimizer(self.model, self.cfg)
            scheduler = self._build_scheduler(optimizer)
            best_miou = float('-inf')
            if continuation:
                self.start_metrics = self.evaluate(self.model, self.val_dl)
                best_miou = self.start_metrics['mIoU']
                self._log(-1, self.start_metrics, prefix='start_val')
                self.save_checkpoint(self.model, optimizer, -1, self.start_metrics, 'best.pt',
                                     scheduler=scheduler, selection_state={'start_candidate': True})
            for epoch in range(1 if a.smoke else self.cfg.epochs):
                self.model.train()
                record = (self._train_epoch_distill(self.model, optimizer) if self.loss_fn is not None
                          else self._train_epoch_supervised(self.model, optimizer))
                self._log(epoch, record, prefix='train')
                if scheduler is not None:
                    scheduler.step()
                metrics = {}
                if a.smoke or (epoch + 1) % 2 == 0 or epoch + 1 == self.cfg.epochs:
                    metrics = self.evaluate(self.model, self.val_dl)
                    self._log(epoch, metrics, prefix='val')
                    if metrics['mIoU'] > best_miou:
                        best_miou = metrics['mIoU']
                        self.save_checkpoint(self.model, optimizer, epoch, metrics, 'best.pt', scheduler=scheduler,
                            selection_state={'metric': 'source validation mIoU', 'best_miou': best_miou,
                                             'comparison': 'strict greater; ties retain earlier'})
                self.save_checkpoint(self.model, optimizer, epoch, metrics, 'last.pt', scheduler=scheduler,
                                     selection_state={'best_miou': best_miou})
            return {'best_val_mIoU': best_miou, 'best_val_smoothed_mIoU': best_miou}

    trainer = V7Trainer(model, observed_train, val, test, cfg, loss_fn=loss)
    try:
        result = trainer.fit()
    except BaseException:
        if hasattr(trainer, 'optimizer'):
            trainer.save_checkpoint(trainer.model, trainer.optimizer, trainer.current_epoch - 1, {},
                                    'interrupted.pt', selection_state={'technical_stop': True,
                                                                    'exact_resume': False})
        raise
    finally:
        if isinstance(loss, ContinuationLoss):
            cj.atomic_json(run_dir / 'loss_diagnostics.json', loss.diagnostics)
    best, last = run_dir / 'best.pt', run_dir / 'last.pt'
    bp, lp = cj.load_tensor(best), cj.load_tensor(last)
    updates = int(lp['training_state']['global_step'])
    expected = a.smoke_updates if a.smoke else per_epoch*50
    assert updates == expected, (updates, expected)
    result.update(selected_metrics=bp['metrics'], final_metrics=lp['metrics'], start_metrics=trainer.start_metrics)
    cj.atomic_json(run_dir / 'results.json', result)
    cj.completed(a.out, start, **metadata, outcome='smoke_completed' if a.smoke else 'completed',
        parameters=sum(p.numel() for p in model.parameters()), base_parameters=12230 if a.graph == 'R4' else None,
        training_images=len(raw_train.dataset), updates_per_epoch=per_epoch,
        first_minibatch_hashes=observed_train.first_minibatch_hashes, first_image_hashes=observed_train.first_image_hashes, gradient_checks=trainer.grad_checks,
        first_training_logits_sha256=trainer.first_training_logits_sha256, timings=trainer.timings,
        best_path=str(best), best_sha256=cj.file_hash(best), last_path=str(last), last_sha256=cj.file_hash(last),
        selected_updates=int(bp['training_state']['global_step']), completed_updates=updates,
        selected_epoch=int(bp['epoch']) + 1, completed_epochs=int(lp['epoch']) + 1,
        source_validation=result, epochs=trainer.epoch_records, terminal_evaluation='separate_job',
        smoke_validation_images=4 if a.smoke else None, gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        loss_diagnostics_path=str(run_dir / 'loss_diagnostics.json') if continuation else None)


if __name__ == '__main__':
    c.setup('cuda');main()
