"""Focused CPU checks of risks specific to v7; no GPU initialization or jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_v7 import build_model, state_hash
from v7_loss import (ContinuationLoss, NORM_MEAN, NORM_STD, RuleContribution,
                     emphasis_mask, normalized_weights, rule_feature)


class FixedTeacher(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', logits)

    def forward(self, images):
        return self.logits.expand(images.shape[0], -1, -1, -1)


def run(capsule, rungs_dir, out):
    sys.path.insert(0, str(Path(capsule).resolve()))
    from training.losses import focal_loss
    from training.compression_loss import CompressionLoss
    from data.pseudo_labeled import PseudoLabeledDataset
    from data.sen1floods11 import NORM_MEAN as frozen_mean, NORM_STD as frozen_std
    assert frozen_mean == NORM_MEAN and frozen_std == NORM_STD
    torch.set_num_threads(2)
    checks = []

    mu = torch.tensor(NORM_MEAN).reshape(1, 6, 1, 1)
    sigma = torch.tensor(NORM_STD).reshape(1, 6, 1, 1)
    reflectance = torch.full((1, 6, 2, 3), .1)
    reflectance[:, 1] = torch.tensor([[.3, .1, -.1], [.0, .0, .1]])
    reflectance[:, 4] = torch.tensor([[.1, .3, .1], [.0, .0, float('nan')]])
    x = (reflectance - mu) / sigma
    r, available = rule_feature(x)
    assert torch.allclose(r[0, 0, :2], torch.tensor([.5, -.5]), atol=1e-6)
    assert torch.equal(available[0], torch.tensor([[True, True, False], [False, False, False]]))
    assert torch.equal(r[~available], torch.zeros_like(r[~available]))
    checks.append('verified_band_order_denormalization_available_mask')

    base = build_model('R4', 'C0', rungs_dir).eval()
    wrapped = RuleContribution(base).eval()
    inp = torch.randn(2, 6, 32, 32)
    with torch.no_grad():
        old, new = base(inp), wrapped(inp)
    assert torch.equal(old, new)
    assert sum(p.numel() for p in wrapped.parameters()) - sum(p.numel() for p in base.parameters()) == 2
    wrapped.alpha.data.fill_(.2)
    wrapped.beta.data.fill_(-.1)
    r, _ = rule_feature(inp)
    predicted_margin = old[:, 1] - old[:, 0] + 8 * wrapped.alpha.tanh() * r + 8 * wrapped.beta.tanh()
    assert torch.allclose(wrapped(inp)[:, 1] - wrapped(inp)[:, 0], predicted_margin, atol=1e-6)
    reloaded = RuleContribution(build_model('R4', 'C0', rungs_dir)).eval()
    reloaded.load_state_dict(wrapped.state_dict(), strict=True)
    assert torch.equal(wrapped(inp), reloaded(inp))
    checks.append('zero_initial_rule_exact_logits_two_parameters_margin_and_strict_reload')

    t_logits = (4 * torch.tensor([.8, .2]).log()).reshape(1, 2, 1, 1).expand(1, 2, 3, 4)
    teacher = FixedTeacher(t_logits).eval().requires_grad_(False)
    images = torch.zeros(2, 6, 3, 4)
    target = torch.tensor([[[0, 1, -1, 0]] * 3] * 2)
    z = (4 * torch.tensor([.4, .6]).log()).reshape(1, 2, 1, 1).expand(2, 2, 3, 4).clone().requires_grad_()
    loss = ContinuationLoss(teacher, 'C0', 42)
    value, stats = loss(z, images, target)
    expected = 16 * float((torch.tensor([.8, .2]) * (torch.tensor([.8, .2]).log()
                                                                    - torch.tensor([.4, .6]).log())).sum())
    reverse = 16 * float((torch.tensor([.4, .6]) * (torch.tensor([.4, .6]).log()
                                                                    - torch.tensor([.8, .2]).log())).sum())
    assert abs(stats['kd'] - expected) < 1e-5 and abs(stats['kd'] - reverse) > .1
    legacy = CompressionLoss(teacher, nn.Identity())
    inherited, _ = legacy(z, images, target)
    assert torch.allclose(value, inherited, atol=1e-6), (value, inherited)
    value.backward()
    assert torch.equal(z.grad[:, :, :, 2], torch.zeros_like(z.grad[:, :, :, 2]))
    assert all(p.grad is None for p in teacher.parameters())
    empty_z = z.detach().clone().requires_grad_()
    empty_loss, empty_stats = loss(empty_z, images, torch.full_like(target, -1))
    assert float(empty_loss) == 0 and empty_stats['valid_pixels'] == 0
    empty_loss.backward()
    assert torch.equal(empty_z.grad, torch.zeros_like(empty_z))
    checks.append('teacher_to_student_KL_focal_C0_inherited_parity_ignore_and_empty_masks')

    hard_cached = ContinuationLoss(teacher, 'HC', 42)
    hard_live = ContinuationLoss(teacher, 'HL', 42)
    hc, hc_stats = hard_cached(z.detach().clone().requires_grad_(), images, target)
    hl, hl_stats = hard_live(z.detach().clone().requires_grad_(), images, target)
    valid = target != -1
    hc_expected = F.nll_loss(F.log_softmax(z / 4, 1), target, ignore_index=-1) * 16
    hl_expected = -F.log_softmax(z / 4, 1)[:, 0][valid].mean() * 16
    assert abs(hc_stats['kd'] - float(hc_expected)) < 1e-5
    assert abs(hl_stats['kd'] - float(hl_expected)) < 1e-5
    checks.append('HC_HL_onehot_KL_crossentropy_valid_masks')

    conf = torch.tensor([.55, .65, .75, .85, .95, .99]).repeat(8).reshape(1, 6, 8)
    teacher_cls = torch.arange(48).reshape(1, 6, 8) % 2
    p_land = torch.where(teacher_cls == 0, conf, 1 - conf)
    t = torch.stack((p_land.log(), (1 - p_land).log()), 1)
    mismatch = torch.arange(48).reshape(1, 6, 8) % 3 == 0
    rule_label = torch.where(mismatch, 1 - teacher_cls, teacher_cls)
    rr = torch.where(rule_label == 1, .5, -.5)
    support = torch.ones_like(mismatch)
    support[:, 0, 0] = False
    available = torch.ones_like(support)
    available[:, 0, 1] = False
    old_global_state = torch.get_rng_state().clone()
    gen = torch.Generator().manual_seed(42)
    shuffled, d, strata = emphasis_mask(t, rr, available, support, 'C4', gen)
    assert torch.equal(old_global_state, torch.get_rng_state())
    confidence = F.softmax(t, 1).amax(1)
    bins = (confidence >= .7).long() + (confidence >= .9).long()
    for cls in (0, 1):
        for confidence_bin in (0, 1, 2):
            mask = available & support & (teacher_cls == cls) & (bins == confidence_bin)
            assert int(shuffled[mask].sum()) == int(d[mask].sum())
    uncertain, d, _ = emphasis_mask(t, rr, available, support, 'C5', gen)
    entropy = -(F.softmax(t, 1) * F.log_softmax(t, 1)).sum(1)
    for cls in (0, 1):
        mask = available & support & (teacher_cls == cls)
        assert int(uncertain[mask].sum()) == int(d[mask].sum())
        if uncertain[mask].any() and (~uncertain[mask]).any():
            assert entropy[mask & uncertain].min() >= entropy[mask & ~uncertain].max()
    assert not (uncertain & ~(available & support)).any()
    weights = normalized_weights(shuffled, support)
    assert not weights.requires_grad and abs(float(weights[support].mean()) - 1) < 1e-6
    assert torch.equal(weights[~support], torch.zeros_like(weights[~support]))
    assert torch.equal(normalized_weights(d, torch.zeros_like(support)), torch.zeros_like(weights))
    # Stable entropy tie break is increasing flattened pixel index.
    tie_t = torch.zeros(1, 2, 2, 3)
    tie_r = torch.tensor([[[.5, -.5, .5], [-.5, -.5, -.5]]])
    tied, _, _ = emphasis_mask(tie_t, tie_r, torch.ones_like(tie_r).bool(),
                              torch.ones_like(tie_r).bool(), 'C5', gen)
    assert tied.reshape(-1).tolist() == [True, True, False, False, False, False]
    checks.append('detached_mean_one_weights_C4_exact_stratified_counts_C5_entropy_counts_and_ties')

    # Use the ACTUAL frozen dataset, worker seeding, jitter and geometry.
    fixture = Path(tempfile.mkdtemp(prefix='.focused_fixture_', dir=Path(__file__).parent))
    try:
        for i in range(32):
            g = torch.Generator().manual_seed(i + 1700)
            torch.save({'image': torch.rand(6, 32, 32, generator=g),
                        'mask': torch.randint(0, 2, (32, 32), generator=g)}, fixture / f'tile_{i:03d}.pt')
        (fixture / 'index.json').write_text(json.dumps({'files': [f'tile_{i:03d}.pt' for i in range(32)]}))
        signatures = {}
        for arm in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL'):
            torch.manual_seed(42)
            _dummy_teacher = nn.Conv2d(6, 2, 1)  # mandatory teacher-first init for each arm.
            paired_model = build_model('R4', arm, rungs_dir)
            base = paired_model.base if isinstance(paired_model, RuleContribution) else paired_model
            initial = state_hash(base.state_dict())
            ds = PseudoLabeledDataset(str(fixture), n_samples=32, seed=42, use_curated=True,
                index_file='index.json', extra_keys=frozenset(), mult_noise=True, geom_aug=True)
            torch.manual_seed(42)  # inherited Trainer reseeds after capturing initial state.
            dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
            batch = next(iter(dl))
            signature = state_hash({'image': batch['image'], 'mask': batch['mask']})
            signatures[arm] = {'initial_base': initial, 'first_augmented_minibatch': signature}
            before = torch.get_rng_state().clone()
            emphasis_mask(t, rr, available, support, arm, torch.Generator().manual_seed(42))
            assert torch.equal(before, torch.get_rng_state())
        assert len({i['initial_base'] for i in signatures.values()}) == 1
        assert len({i['first_augmented_minibatch'] for i in signatures.values()}) == 1
    finally:
        shutil.rmtree(fixture)
    checks.append('paired_base_initialization_and_actual_first_augmented_worker_minibatches_all_eight_arms')
    result = {'status': 'complete', 'outcome': 'passed', 'device': 'cpu', 'gpu_used': False,
              'checks': checks, 'paired_signatures': signatures,
              'checks_count': len(checks), 'torch_version': torch.__version__}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'outcome': 'passed', 'checks_count': len(checks), 'out': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule', required=True)
    p.add_argument('--rungs-dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run(args.capsule, args.rungs_dir, args.out)
