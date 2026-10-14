"""Low-bit recipe test (Q1, opt-in v5): calibrate an FP checkpoint, then continue training.

Arms (identical start state, data order, optimizer, BN freezing, clip, simulator context):
  --qdq on  --bits 4 8            : W4A8 QAT, adopted integer-domain simulator
  --qdq on  --bits 4 4 --stem8    : 4-bit body (W4A4) with the stem input activation at 8 bit
  --qdq off                       : matched FP control - same continuation with quantization bypassed
Recipe = Stage 1.6 lr 5e-4 (or --lr), wd 1e-5, clip 1, 10 epochs = 1,560 updates, S11-val
selection every 2 epochs. Calibration = protocol S11-val images, calibration_seed 1729.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
for name in ('capsule', 'spec', 'fp-receipt', 'out', 'run-dir', 'graph', 'qdq'):
    p.add_argument('--' + name, required=True)
p.add_argument('--bits', type=int, nargs=2, default=(4, 8))
p.add_argument('--stem8', action='store_true')
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
for _k, _v in list(vars(a).items()):
    if isinstance(_v, str) and _k in ('capsule', 'spec', 'out', 'run_dir', 'fp_receipt'):
        setattr(a, _k, str(Path(_v).resolve()))
sys.path.insert(0, a.capsule); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v3'))
import torch
from torch.utils.data import DataLoader
import scripts.compression_jobs as cj
from config import TrainConfig
from utils.seeding import seed_basic
from training.compression_loss import CompressionLoss
from models.compression_quant import (prepare_compression_qat, FixedQATConv2d, _range,
                                      compression_train_mode, quantization_mode, strip_compression_qat,
                                      precision_map)
from experiments.revision_queue import file_hash
from simulator_v4 import simulator
import rungs

assert a.qdq in ('on', 'off')
_adopt = json.loads((Path(__file__).resolve().parents[2] / 'simulator_adoption.json').read_text())
assert file_hash(Path(__file__).resolve().parents[1] / 'v3' / 'simulator_v4.py') == _adopt['source_sha256']
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
start = time.monotonic()
spec = json.loads(Path(a.spec).read_text()); spec.update(seed=a.seed, graph=a.graph, run_dir=a.run_dir)
fp = json.loads(Path(a.fp_receipt).read_text())
assert fp['status'] == 'complete' and fp['completed_updates'] == 7800
seed_basic(a.seed)
fp_model = rungs.build(a.graph)
fp_model.load_state_dict(torch.load(fp['best_path'], map_location='cpu', weights_only=True)['model'], strict=True)
fp_model.cuda().eval()
val, test = cj.loaders(spec)
cal = DataLoader(val.dataset, batch_size=1, shuffle=False, num_workers=spec['workers'])
model = prepare_compression_qat(fp_model, ({'image': b['image'].cuda()} for b in cal),
                                weight_bits=a.bits[0], activation_bits=a.bits[1],
                                calibration_seed=spec['calibration_seed'])
stem = None
if a.stem8:
    stem_name, stem = next((n, m) for n, m in model.named_modules() if isinstance(m, FixedQATConv2d))
    assert 'input_stem' in stem_name, stem_name
    old, new = _range(stem.activation_bits)[1], _range(8)[1]
    stem._act_scale.mul_(old / new); stem._activation_bits.fill_(8)
    stem._activation_bits_value = 8; stem._activation_scale_value = float(stem._act_scale[0])
qdq = a.qdq == 'on'
initial = {n: (m.weight.detach().cpu().clone(), m._w_scale.detach().cpu().clone(), m.weight_bits)
           for n, m in model.named_modules() if isinstance(m, FixedQATConv2d)}


class Trainer(cj.CompressionTrainer):
    """CompressionTrainer with QDQ fixed to the arm's setting for training and selection."""
    def _train_epoch_distill(self, model, optimizer):
        compression_train_mode(model, enabled=qdq)
        output = super(cj.CompressionTrainer, self)._train_epoch_distill(model, optimizer)
        self.current_epoch += 1
        return output

    def evaluate(self, model, loader):
        with quantization_mode(model, qdq):
            return super(cj.CompressionTrainer, self).evaluate(model, loader)


loss = CompressionLoss(cj.teacher(spec), model).cuda()
train = DataLoader(cj.pseudo(spec, augment=True), batch_size=16, shuffle=True,
                   num_workers=spec['workers'], pin_memory=True, drop_last=True)
EPOCHS = 2 if a.smoke else 10
cfg = TrainConfig(epochs=EPOCHS, lr=a.lr, weight_decay=1e-5, task_loss='focal', use_amp=False, grad_clip=1.,
                  early_stop_patience=0, val_smoothing_window=1, val_every_n_epochs=2,
                  defer_terminal_evaluation=True, out_dir=a.run_dir, seed=a.seed, gpu_ids=[0])
trainer = Trainer(model, train, val, test, cfg, loss_fn=loss, quantized=True)
norms, old_clip = [], torch.nn.utils.clip_grad_norm_


def clip(parameters, max_norm, *args, **kwargs):
    ps = list(parameters)
    norms.append(float(torch.sqrt(sum((v.grad.detach().double().square().sum() for v in ps if v.grad is not None),
                                      torch.zeros((), device='cuda', dtype=torch.float64)))))
    return old_clip(ps, max_norm, *args, **kwargs)


try:
    torch.nn.utils.clip_grad_norm_ = clip
    with simulator(model, 'integer'):
        trainer.fit()
finally:
    torch.nn.utils.clip_grad_norm_ = old_clip
    loss.remove_hooks()
best, last = Path(a.run_dir) / 'best.pt', Path(a.run_dir) / 'last.pt'
bp = torch.load(best, map_location='cpu', weights_only=False); lp = torch.load(last, map_location='cpu', weights_only=False)
assert lp['training_state']['global_step'] == 156 * EPOCHS
results, flips = {}, {}
for label, payload in (('selected', bp), ('final', lp)):
    model.load_state_dict(payload['model'], strict=True)
    if qdq:
        with simulator(model, 'integer'):
            results[label] = cj.validation(model.cuda(), spec, paired=True)
    else:
        results[label] = cj.validation(strip_compression_qat(copy.deepcopy(model)).cuda(), spec)
    changed = total = 0
    for n, m in model.named_modules():
        if isinstance(m, FixedQATConv2d):
            w0, s, bits = initial[n]; lo, hi = _range(bits); ss = s.reshape(-1, 1, 1, 1)
            before = (w0 / ss).round().clamp(lo, hi); after = (m.weight.detach().cpu() / ss).round().clamp(lo, hi)
            changed += int((before != after).sum()); total += w0.numel()
    flips[label] = changed / total
model.load_state_dict(bp['model'], strict=True)
fp_export = Path(a.run_dir) / 'best_fp_stripped.pt'
torch.save({'model': strip_compression_qat(copy.deepcopy(model)).state_dict()}, fp_export)
import numpy as np
cj.completed(a.out, start, graph=a.graph, seed=a.seed, qdq=a.qdq, bits=list(a.bits), stem8=a.stem8, lr=a.lr,
             status='complete', status_label='simulation_only' if qdq else 'fp_control',
             fp_parent=a.fp_receipt, fp_parent_sha256=fp['best_sha256'],
             best_path=str(best), best_sha256=file_hash(best), last_path=str(last), last_sha256=file_hash(last),
             fp_stripped_path=str(fp_export), fp_stripped_sha256=file_hash(fp_export),
             selected_updates=bp['training_state']['global_step'], completed_updates=156 * EPOCHS, smoke=a.smoke,
             validation=results, codes_changed_fraction=flips, precision_map=precision_map(model),
             norm_p50=float(np.median(norms)), norm_p95=float(np.percentile(norms, 95)), norm_max=max(norms),
             clipped_fraction=float(np.mean(np.asarray(norms) > 1)), epochs=trainer.epoch_records,
             simulator='integer_domain_v4_exact_native_STE', TF32_enabled=False, no_external_benchmark_access=True)
