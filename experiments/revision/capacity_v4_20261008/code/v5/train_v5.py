"""FP training for capacity rungs and training-method comparison (opt-in, v5).

Methods (all scratch, ReLU6, FP recipe of protocol_v4 unless --lr overrides):
  lkd : 0.5 focal(teacher hard mask) + 0.5 T^2 KL(teacher||student), T=4, live teacher  [= Stage 2 recipe]
  ofa : lkd + 0.3 * OFA exit-branch losses (gamma 1.2) on EfficientViT stages            [= paper OFA settings]
  ce  : focal on teacher hard masks only (teacher used as an offline labeller; no soft targets)
  sup : focal on Sen1Floods11 human labels (S11 hand-labelled train split); no teacher at all
lkd/ofa/ce use the identical N=2,500 cohort, augmentation, batch and 7,800 updates.
sup uses the 252-chip S11 train split with the same 7,800 updates (checkpoint check every 20 epochs = 300 updates).
For every method the frozen teacher is constructed before the student so that the
student initialisation for a given seed is identical across methods.
"""
import argparse
import json
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
for name in ('capsule', 'spec', 'out', 'run-dir', 'method'):
    p.add_argument('--' + name, required=True)
p.add_argument('--graph')
p.add_argument('--graph-from', help='floor selection JSON; graph null -> skipped receipt')
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--lr-from', help='LR selection JSON written by select_v5.py lr')
p.add_argument('--reuse', nargs='*', default=[], metavar='LR=RECEIPT',
               help='identical-recipe receipts (Stage 2 or v5 grid) reused when the effective lr matches')
p.add_argument('--seed', type=int, required=True)
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
for _k, _v in list(vars(a).items()):
    if isinstance(_v, str) and _k in ('capsule', 'spec', 'out', 'run_dir', 'lr_from', 'graph_from'):
        setattr(a, _k, str(Path(_v).resolve()))
sys.path.insert(0, a.capsule); sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.revision_queue import atomic_json, now
if a.graph_from:
    sel = json.loads(Path(a.graph_from).read_text())
    if sel['graph'] is None:
        atomic_json(a.out, {'status': 'complete', 'outcome': 'skipped', 'reason': 'no rung met the predeclared '
                            'S11-val floor rule', 'selection': a.graph_from, 'completed_at': now()})
        sys.exit(0)
    a.graph, a.lr = sel['graph'], sel['lr']
if a.lr_from:
    sel = json.loads(Path(a.lr_from).read_text())
    assert sel['graph'] == a.graph, (sel['graph'], a.graph)
    a.lr = sel['lr']
assert a.graph
if a.method == 'ofa' and a.graph.startswith('R'):
    atomic_json(a.out, {'status': 'complete', 'outcome': 'skipped', 'graph': a.graph, 'completed_at': now(),
                        'reason': 'OFA exit branches need a staged EfficientViT backbone; undefined for per-pixel/plain-conv rungs'})
    sys.exit(0)
reuse = {float(k): str(Path(v).resolve()) for k, v in (item.split('=', 1) for item in a.reuse)}
match = [v for k, v in reuse.items() if abs(k - a.lr) < 1e-12]
if match and a.method == 'lkd':
    r = json.loads(Path(match[0]).read_text())
    assert r['status'] == 'complete' and r.get('outcome', 'completed') == 'completed', match[0]
    assert r['completed_updates'] == 7800 and r['graph'] == a.graph and r.get('seed', a.seed) == a.seed
    atomic_json(a.out, {'status': 'complete', 'outcome': 'completed', 'reused': match[0],
                        'completed_at': now(), 'graph': a.graph, 'method': 'lkd', 'lr': a.lr, 'seed': a.seed,
                        'parameters': r.get('parameters') or r['graph_record']['parameters'],
                        **{k: r[k] for k in ('best_path', 'best_sha256', 'last_path', 'last_sha256', 'selected_updates',
                                             'completed_updates', 'selected_epoch', 'completed_epochs',
                                             'source_validation', 'epochs')}})
    sys.exit(0)
import torch
from torch.utils.data import DataLoader
import scripts.compression_jobs as cj
from config import TrainConfig
from utils.seeding import seed_basic
from training.compression_loss import CompressionLoss
from experiments.revision_queue import file_hash
import rungs

assert a.method in ('lkd', 'ofa', 'ce', 'sup')
start = time.monotonic()
spec = json.loads(Path(a.spec).read_text())
spec.update(seed=a.seed, run_dir=a.run_dir, graph=a.graph)
seed_basic(a.seed)
teacher = cj.teacher(spec)                      # constructed first for every method (RNG parity)
model = rungs.build(a.graph)
params = sum(q.numel() for q in model.parameters())
val, test = cj.loaders(spec)
if a.method == 'sup':
    from data.sen1floods11 import get_dataloaders
    s11_train, _, _ = get_dataloaders(spec['roots']['s11'], batch_size=16, num_workers=spec['workers'],
                                      img_size=224, eval_img_size=512, drop_last=True)
    # Same dataset/augmentation; persistent workers only avoid re-spawning workers in each of ~520 short epochs.
    train = DataLoader(s11_train.dataset, batch_size=16, shuffle=True, num_workers=6, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    per_epoch = len(train)
    epochs = -(-7800 // per_epoch)
    val_every = max(1, round(312 / per_epoch))
else:
    train = DataLoader(cj.pseudo(spec, augment=True), batch_size=16, shuffle=True,
                       num_workers=spec['workers'], pin_memory=True, drop_last=True)
    assert len(train) == 156
    per_epoch, epochs, val_every = 156, 50, 2
if a.smoke:
    epochs = 2 * val_every
cfg = TrainConfig(epochs=epochs, lr=a.lr, weight_decay=.05, task_loss='focal', use_amp=False,
                  grad_clip=5., early_stop_patience=10 ** 6 if a.method == 'sup' else 20,
                  val_smoothing_window=1, val_every_n_epochs=val_every,
                  defer_terminal_evaluation=True, out_dir=a.run_dir, seed=a.seed, gpu_ids=[0])
loss = None
if a.method == 'lkd':
    loss = CompressionLoss(teacher, model).cuda()
elif a.method == 'ofa':
    from distillation.ofa_kd import OFALoss
    model.cuda()
    loss = OFALoss(teacher=teacher, student=model, num_classes=2, alpha=.5, ofa_alpha=.3,
                   temperature=4., gamma=1.2, ignore_index=-1, kd_valid_only=True)
    loss.register_hooks(); loss.prewarm(train, torch.device('cuda'))
    assert len(loss.exit_branches) > 0, 'OFA needs a staged EfficientViT backbone'
else:
    del teacher; torch.cuda.empty_cache()
trainer = cj.CompressionTrainer(model, train, val, test, cfg, loss_fn=loss)
result = trainer.fit()
if loss is not None:
    loss.remove_hooks()
best, last = Path(a.run_dir) / 'best.pt', Path(a.run_dir) / 'last.pt'
bp = torch.load(best, map_location='cpu', weights_only=False)
lp = torch.load(last, map_location='cpu', weights_only=False)
cj.completed(a.out, start, graph=a.graph, method=a.method, lr=a.lr, seed=a.seed, parameters=params,
             per_pixel=rungs.per_pixel_check(rungs.build(a.graph)) if a.graph in ('R1', 'R2') else False,
             training_images=len(train.dataset), updates_per_epoch=per_epoch,
             best_path=str(best), best_sha256=file_hash(best), last_path=str(last), last_sha256=file_hash(last),
             selected_updates=bp['training_state']['global_step'], completed_updates=lp['training_state']['global_step'],
             selected_epoch=bp['epoch'] + 1, completed_epochs=lp['epoch'] + 1,
             source_validation=result, epochs=trainer.epoch_records, terminal_evaluation='separate_job',
             ofa_branches=len(loss.exit_branches) if a.method == 'ofa' else 0)
