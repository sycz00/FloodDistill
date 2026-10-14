"""Small, import-safe v7 rule and continuation objectives; class 0 land, 1 water."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

NORM_MEAN = [0.1413, 0.1380, 0.1235, 0.3090, 0.2045, 0.1191]
NORM_STD = [0.0741, 0.0737, 0.0869, 0.1180, 0.0977, 0.0766]
RULE_EPSILON = 1e-6  # sum in unitless reflectance (after raw scale 0.0001)
RULE_ARMS = frozenset(('C1', 'C3', 'C4', 'C5'))


def rule_feature(images, mean=NORM_MEAN, std=NORM_STD, epsilon=RULE_EPSILON):
    """Denormalize verified GREEN index 1 and SWIR_1 index 4 at the exact grid."""
    if images.ndim != 4 or images.shape[1] != 6:
        raise ValueError('Expected six bands BLUE GREEN RED NIR_NARROW SWIR_1 SWIR_2')
    mu = torch.as_tensor(mean, device=images.device, dtype=images.dtype).reshape(1, 6, 1, 1)
    sigma = torch.as_tensor(std, device=images.device, dtype=images.dtype).reshape(1, 6, 1, 1)
    green = images[:, 1] * sigma[:, 1] + mu[:, 1]
    swir = images[:, 4] * sigma[:, 4] + mu[:, 4]
    total = green + swir
    available = (torch.isfinite(green) & torch.isfinite(swir) & (green >= 0)
                 & (swir >= 0) & (total > epsilon))
    safe_sum = torch.where(available, total, torch.ones_like(total))
    r = torch.where(available, ((green - swir) / safe_sum).clamp(-1, 1),
                    torch.zeros_like(green))
    return r, available.detach()


class RuleContribution(nn.Module):
    def __init__(self, base, norm_mean=NORM_MEAN, norm_std=NORM_STD):
        super().__init__()
        self.base = base
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))
        self.register_buffer('norm_mean', torch.tensor(norm_mean).float())
        self.register_buffer('norm_std', torch.tensor(norm_std).float())

    def forward(self, images):
        logits = self.base(images)
        r, _ = rule_feature(images, self.norm_mean, self.norm_std)
        if logits.shape[1] != 2 or logits.shape[-2:] != r.shape[-2:]:
            raise RuntimeError('Rule and two-class base predictions must share the input grid')
        correction = 8 * torch.tanh(self.alpha) * r + 8 * torch.tanh(self.beta)
        return logits + torch.stack((-correction / 2, correction / 2), dim=1)


@torch.no_grad()
def emphasis_mask(teacher_logits, r, available, valid, arm, generator):
    """Counts matched on valid available pixels; controls never consume global RNG."""
    teacher_logits = teacher_logits.detach()
    p = F.softmax(teacher_logits, dim=1)
    hard = p.argmax(1)
    candidates = valid.bool() & available.bool()
    disagreement = ((hard != (r > 0).long()) & candidates).detach()
    stats = {'weight_strata': 0, 'degenerate_strata': 0, 'empty_strata': 0,
             'fully_selected_strata': 0}
    if arm not in ('C4', 'C5'):
        return disagreement, disagreement, stats
    confidence = p.amax(1)
    # bin 0 [0.5,0.7), bin 1 [0.7,0.9), bin 2 [0.9,1].
    confidence_bin = (confidence >= .7).long() + (confidence >= .9).long()
    entropy = -(p * F.log_softmax(teacher_logits, dim=1)).sum(1)
    chosen = torch.zeros_like(disagreement)
    for image in range(hard.shape[0]):
        for cls in (0, 1):
            bins = (0, 1, 2) if arm == 'C4' else (None,)
            for bin_id in bins:
                support = candidates[image] & (hard[image] == cls)
                if bin_id is not None:
                    support = support & (confidence_bin[image] == bin_id)
                indices = support.reshape(-1).nonzero(as_tuple=False).flatten()
                n = int(indices.numel())
                k = int(disagreement[image].reshape(-1)[indices].sum())
                stats['weight_strata'] += 1
                stats['empty_strata'] += int(n == 0)
                stats['fully_selected_strata'] += int(n > 0 and k == n)
                stats['degenerate_strata'] += int(n <= 1 or k == 0 or k == n)
                if not n or not k:
                    continue
                if arm == 'C4':
                    # Permuting the binary mask preserves exact counts, including ties.
                    permutation = torch.randperm(n, generator=generator, device=indices.device)
                    shuffled = disagreement[image].reshape(-1)[indices][permutation]
                    chosen[image].reshape(-1)[indices] = shuffled
                else:
                    # nonzero emits increasing flattened pixel index; stable descending
                    # entropy therefore uses pixel index as the predetermined tie break.
                    order = torch.argsort(entropy[image].reshape(-1)[indices],
                                          descending=True, stable=True)
                    chosen[image].reshape(-1)[indices[order[:k]]] = True
    return chosen.detach(), disagreement, stats


@torch.no_grad()
def normalized_weights(emphasis, valid):
    valid = valid.bool()
    n = int(valid.sum())
    if not n:
        return torch.zeros_like(emphasis, dtype=torch.float32)
    u = 1 + 2 * emphasis.detach().float()
    return torch.where(valid, u / u[valid].mean(), torch.zeros_like(u)).detach()


class ContinuationLoss(nn.Module):
    """Frozen focal settings plus the predeclared live-soft/HC/HL second term."""
    gamma = 2.
    temperature = 4.
    kd_valid_only = True
    ignore_index = -1

    def __init__(self, teacher, arm, seed, norm_mean=NORM_MEAN, norm_std=NORM_STD,
                 diagnostic_interval=50):
        super().__init__()
        object.__setattr__(self, '_teacher', teacher)
        self.arm, self.seed = arm, int(seed)
        self.norm_mean, self.norm_std = list(norm_mean), list(norm_std)
        # CPU initialization is import-safe; actual device is set lazily on forward.
        self.weight_generator = None
        self.calls, self.diagnostics = 0, []
        self.diagnostic_interval = int(diagnostic_interval)

    def forward(self, student_logits, images, masks, precomputed_teacher_logits=None):
        from training.losses import focal_loss
        if precomputed_teacher_logits is not None:
            raise ValueError('v7 requires teacher live on the same augmented input')
        teacher = object.__getattribute__(self, '_teacher')
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise RuntimeError('Teacher must be frozen and in eval mode')
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            teacher_logits = teacher(images.float()).detach().float()
        if (student_logits.shape != teacher_logits.shape or student_logits.shape[1] != 2
                or not torch.isfinite(student_logits).all()
                or not torch.isfinite(teacher_logits).all()):
            raise RuntimeError('Nonfinite or misaligned teacher/student logits')
        if self.weight_generator is None:
            self.weight_generator = torch.Generator(device=images.device).manual_seed(
                (self.seed ^ 0x6A09E667) % ((1 << 63) - 1))
        valid = (masks != self.ignore_index).detach()
        r, available = rule_feature(images, self.norm_mean, self.norm_std)
        emphasis, disagreement, strata = emphasis_mask(
            teacher_logits, r.detach(), available, valid, self.arm, self.weight_generator)
        if self.arm in ('C0', 'C1', 'HC', 'HL'):
            emphasis = torch.zeros_like(emphasis)
        weights = normalized_weights(emphasis, valid)
        log_q = F.log_softmax(student_logits / self.temperature, dim=1)
        if self.arm in ('HC', 'HL'):
            hard_target = masks.clamp_min(0) if self.arm == 'HC' else teacher_logits.argmax(1)
            pixel_kl = -log_q.gather(1, hard_target.unsqueeze(1)).squeeze(1)
        else:
            p = F.softmax(teacher_logits / self.temperature, dim=1).detach()
            pixel_kl = F.kl_div(log_q, p, reduction='none').sum(1)
        n = int(valid.sum())
        zero = student_logits.sum() * 0
        unweighted = pixel_kl[valid].mean() * self.temperature ** 2 if n else zero
        weighted = (weights[valid] * pixel_kl[valid]).mean() * self.temperature ** 2 if n else zero
        focal = focal_loss(student_logits, masks, self.gamma, self.ignore_index)
        value = .5 * focal + .5 * weighted
        if not torch.isfinite(value):
            raise RuntimeError('Nonfinite v7 loss')
        stats = {'focal': float(focal.detach()), 'kd': float(weighted.detach()),
                 'unweighted_kl_T2': float(unweighted.detach()),
                 'weighted_kl_T2': float(weighted.detach()),
                 'valid_pixels': n, 'rule_available_valid_pixels': int((valid & available).sum()),
                 'disagreement_pixels': int(disagreement.sum()),
                 'emphasized_pixels': int(emphasis.sum()),
                 'mean_valid_weight': float(weights[valid].mean()) if n else 0.,
                 'effective_sample_size': float(weights[valid].sum().square()
                    / weights[valid].square().sum()) if n else 0., **strata}
        for name, group in (('agreement', valid & available & ~disagreement),
                            ('disagreement', disagreement), ('unavailable', valid & ~available)):
            stats[name + '_weighted_kl_T2_contribution'] = float(
                (pixel_kl[group] * weights[group]).sum().detach() * self.temperature ** 2 / n) if n else 0.
            stats[name + '_unweighted_kl_T2_contribution'] = float(
                pixel_kl[group].sum().detach() * self.temperature ** 2 / n) if n else 0.
        if torch.is_grad_enabled() and self.calls % self.diagnostic_interval == 0:
            record = {'update': self.calls + 1, **stats}
            for name, term in (('focal', focal), ('weighted_kl_T2', weighted),
                               ('unweighted_kl_T2', unweighted), ('total', value)):
                gradient = torch.autograd.grad(term, student_logits, retain_graph=True,
                                               allow_unused=True)[0]
                record[name + '_logit_gradient_l2'] = float(gradient.norm()) if gradient is not None else 0.
            self.diagnostics.append(record)
        self.calls += 1
        return value, stats

    def recovery_state(self):
        return {'calls': self.calls, 'weight_generator_state': self.weight_generator.get_state()
                if self.weight_generator is not None else None,
                'weight_generator_seed': self.seed ^ 0x6A09E667,
                'weight_generator_device': str(self.weight_generator.device)
                if self.weight_generator is not None else None}

    def remove_hooks(self):
        pass
