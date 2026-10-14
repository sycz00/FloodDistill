"""One-for-All distillation with exit branches on intermediate student stages.

"One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge
Distillation", NeurIPS 2023. The branches predict the frozen teacher's output
distribution and are thrown away at inference. Main objective is the hard-label
term + logit KD + the weighted OFA branch losses.

Other research losses remain for compatibility. Published settings are in
experiments/.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from training.losses import focal_loss, kd_loss

# Band indices for Sentinel-2 6-band input (BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2)
_B_GREEN = 1
_B_NIR   = 3
_B_SWIR1 = 4
_SPEC_EPS = 1e-6


# Exit branch

class OFAExitBranch(nn.Module):
    """Small head from stage features to per-pixel logits.

        Conv2d(C_s, mid, 1x1) + BN + ReLU + Conv2d(mid, num_classes, 1x1), run at feature
        resolution h x w, then bilinear upsampling to the teacher logit size in forward.

        Projecting first or upsampling first (UpsampleFirstExitBranch) is mathematically the
        same for 1x1 convs, since bilinear interpolation and a 1x1 conv are both linear and
        work per position. See discussion.md sec. 24.
    """

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        mid = max(in_ch // 2, num_classes * 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1, bias=True),
        )

    def forward(self, feat: Tensor, target_hw: tuple[int, int]) -> Tensor:
        logits = self.head(feat)
        return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)


class UpsampleFirstExitBranch(nn.Module):
    """Like OFAExitBranch, but mid-channel features are upsampled before the last projection.

        Conv2d(C_s, mid, 1x1) + BN + ReLU, bilinear upsample, then Conv2d(mid, num_classes, 1x1)
        at output resolution H x W. Same structure as the flood head in SPLDExitBranch.

        For 1x1 convs this is mathematically identical to OFAExitBranch, the only real
        difference is the extra FLOPs of running the last conv at H x W. Used as an
        architecture control (sec. 31/32) to check that the order has no effect in practice.
    """

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        mid = max(in_ch // 2, num_classes * 2)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(mid, num_classes, 1, bias=True)

    def forward(self, feat: Tensor, target_hw: tuple[int, int]) -> Tensor:
        up = F.interpolate(self.trunk(feat), size=target_hw, mode="bilinear", align_corners=False)
        return self.head(up)


class EntropyExitBranch(nn.Module):
    """Exit branch with one channel that predicts normalized teacher entropy H(p_T)/log(2), in [0,1].

        SC-OFA uses it on stage0 and stage1. Those early CNN stages see local texture and edges
        but can't decode flood semantics. Teacher entropy is a local signal though (high at
        flood/background transitions, low inside confident regions), so they can learn it.

        Same layout as OFAExitBranch with a single Sigmoid output. Trained with
        _entropy_branch_loss (MSE against _teacher_entropy()).
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        mid = max(in_ch // 2, 8)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat: Tensor, target_hw: tuple[int, int]) -> Tensor:
        pred = self.head(feat)   # (B, 1, h, w)
        return F.interpolate(pred, size=target_hw, mode="bilinear", align_corners=False)


class SpectralExitBranch(nn.Module):
    """Exit branch with K channels that predicts spectral indices (NDWI, MNDWI, NDVI).

        SSES-OFA puts it on early stages, where the RF is too small to explain teacher
        predictions that rely on global context. The indices come straight from the local
        band values, so even a 1px RF can learn them, and stages 0-1 get a target that does
        not depend on the teacher (EntropyExitBranch still does). NDWI = (Green-NIR)/(Green+NIR)
        for example only needs the two band values at each position.

        No output activation: raw values for continuous MSE, logits for BCE in binary mode.
        K = num_proxies. Loss is _sses_branch_loss().
    """

    def __init__(self, in_ch: int, num_proxies: int = 3) -> None:
        super().__init__()
        mid = max(in_ch // 2, num_proxies * 4)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_proxies, 1, bias=True),
        )
        self.num_proxies = num_proxies

    def forward(self, feat: Tensor, target_hw: tuple[int, int]) -> Tensor:
        pred = self.head(feat)   # (B, K, h, w)
        return F.interpolate(pred, size=target_hw, mode="bilinear", align_corners=False)


# OFA loss function

def ofa_loss(
    branch_logits: Tensor,
    teacher_logits: Tensor,
    masks: Tensor,
    gamma: float = 1.2,
    ignore_index: int = -1,
    temperature: float = 4.0,
    pixel_weights: Tensor | None = None,
) -> Tensor:
    """OFA loss (eq. 6) per valid pixel.

        L = -(1 + p_t^T)^gamma * log(p_t^S) - sum_{c != t} p_c^T * log(p_c^S)

        t is the ground-truth class, p^T the teacher distribution at the given temperature and
        p^S the student branch distribution. The second term is computed as
            -sum_c p_c^T log p_c^S + p_t^T log p_t^S = H(p_T, p_S) + p_t^T * log p_t^S

        With gamma=1 this reduces to KD-style cross-entropy on the branch.

        teacher_logits have to be detached by the caller, pixels where masks == ignore_index
        are skipped. gamma >= 1, the paper uses 1.1-1.4. Optional pixel_weights (B, H, W) give
        a weighted mean, normalised to mean 1 over valid pixels so the loss scale stays put.
        Returns the mean over valid pixels, or zero if there are none.
    """
    B, C, H, W = branch_logits.shape

    p_t     = F.softmax(teacher_logits.float() / temperature, dim=1)  # (B, C, H, W)
    log_p_s = F.log_softmax(branch_logits.float(), dim=1)             # (B, C, H, W)

    valid   = (masks != ignore_index)                                  # (B, H, W)
    targets = masks.clamp(0, C - 1)                                    # (B, H, W) safe index
    t_idx   = targets.unsqueeze(1)                                     # (B, 1, H, W)

    # Gather teacher prob and student log-prob at the target class
    p_t_target  = p_t.gather(1, t_idx).squeeze(1)                     # (B, H, W)
    lp_s_target = log_p_s.gather(1, t_idx).squeeze(1)                 # (B, H, W)

    # Term 1: upweighted CE for ground-truth class
    term1 = -(1.0 + p_t_target).pow(gamma) * lp_s_target              # (B, H, W)

    # term 2: soft targets for the other classes
    # -sum_{c != t} p_c^T log p_c^S = H(p_T, p_S) + p_t^T * log p_t^S
    cross_entropy_all = -(p_t * log_p_s).sum(dim=1)                    # (B, H, W)
    term2 = cross_entropy_all + p_t_target * lp_s_target               # (B, H, W)

    per_pixel  = (term1 + term2)                                       # (B, H, W)
    valid_loss = per_pixel[valid]
    if valid_loss.numel() == 0:
        return branch_logits.sum() * 0.0
    if pixel_weights is not None:
        w = pixel_weights[valid]
        w = w / (w.mean() + 1e-8)   # mean 1, keeps the loss scale
        return (valid_loss * w).mean()
    return valid_loss.mean()


def _kd_loss_weighted(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
    pixel_weights: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """KD loss with per-pixel weights (B, H, W), e.g. teacher confidence.

        Same as kd_loss() for uniform weights. The weights are normalised to mean 1 so the
        scale matches the unweighted loss. Logits are (B, C, H, W), temperature T >= 1.
    """
    log_p_s  = F.log_softmax(student_logits.float() / temperature, dim=1)
    p_t      = F.softmax(teacher_logits.float()  / temperature, dim=1)
    kl_pixel = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1)   # (B, H, W)
    if valid_mask is not None:
        kl_pixel, pixel_weights = kl_pixel[valid_mask], pixel_weights[valid_mask]
        if kl_pixel.numel() == 0:
            return student_logits.sum() * 0.0
    w        = pixel_weights / (pixel_weights.mean() + 1e-8)
    return (temperature ** 2) * (kl_pixel * w).mean()


def _sorted_wasserstein_loss(
    branch_logits: Tensor,
    teacher_logits: Tensor,
) -> Tensor:
    """1D Wasserstein-2 distance between branch and teacher flood-probability distributions.

        Sorting both and matching rank to rank gives the exact 1D W2. Position is ignored, the
        branch only has to have the same share of confident / uncertain / boundary pixels as
        the teacher. Spatial correspondence is already handled by the OFA point-wise term.

        For C=2 the class-1 (flood) probability has all the information. T=1 (no softening)
        so the actual confidence histogram is matched.

        The backward goes through the torch.sort() indices, the gradient of the i-th sorted
        value lands on the pixel that had that rank.
    """
    p_branch  = F.softmax(branch_logits.float(), dim=1)[:, 1].reshape(-1)   # (B*H*W,)
    p_teacher = F.softmax(teacher_logits.float(), dim=1)[:, 1].reshape(-1)  # (B*H*W,)
    return (p_branch.sort().values - p_teacher.sort().values.detach()).pow(2).mean()


def _relational_loss(
    branch_logits: Tensor,
    teacher_logits: Tensor,
    n_rel: int = 64,
) -> Tensor:
    """Relational loss on pixel pairs, matches the affinity structure in logit space.

        For each pair of positions (i, j) the branch should have the same flood-probability
        difference as the teacher, A[i,j] = |p_flood[i] - p_flood[j]|. Spatial coherence
        carries over without touching features or assuming a layer mapping. T=1 so the real
        confidence differences are matched and not softened ones.

        If H*W > n_rel a random subset of positions is drawn on every call, which also works
        as some stochastic regularisation over position pairs between batches.
    """
    B, C, H, W = branch_logits.shape
    N = min(H * W, n_rel)

    p_branch  = F.softmax(branch_logits.float(), dim=1)[:, 1].reshape(B, -1)   # (B, H*W)
    p_teacher = F.softmax(teacher_logits.float(), dim=1)[:, 1].reshape(B, -1)  # (B, H*W)

    if H * W > n_rel:
        idx = torch.randperm(H * W, device=branch_logits.device)[:N]
        p_branch  = p_branch[:, idx]
        p_teacher = p_teacher[:, idx]

    # Pairwise absolute difference: (B, N, N)
    A_branch  = (p_branch.unsqueeze(2)  - p_branch.unsqueeze(1)).abs()
    A_teacher = (p_teacher.unsqueeze(2) - p_teacher.unsqueeze(1)).abs()

    return (A_branch - A_teacher.detach()).pow(2).mean()


def _patch_affinity_loss(
    branch_logits: Tensor,
    teacher_logits: Tensor,
    n_patches: int = 14,
) -> Tensor:
    """Patch affinity distillation in logit space, based on C2VKD.

        Adaptive average pooling to n_patches x n_patches cells, then the raw co-occurrence
        affinity M[i,j] = p[i] * p[j], in [0, 1]. Roughly ~0.64 if both patches are clearly
        flood (p~0.8), ~0 if at least one is background (p~0.05), ~0.04 for one flood and
        one uncertain patch. Only uses the softmax maps, so any architecture works.

        No centering + L2 norm as in the original C2VKD. Flood masks are sparse (~5-10% flood
        patches). After centering, background patches sit at p_c ~ -0.04, ~ -0.016 after L2
        norm, and the 90% background-background entries of M end up ~0.0003. MSE is then
        ~1e-5 and adds ~1e-6 to the total loss, far too little gradient. The raw outer
        product gives MSE ~0.01-0.1, same range as OFA and focal.
    """
    B = branch_logits.shape[0]
    p_T = F.softmax(teacher_logits.float(), dim=1)[:, 1]   # (B, H, W)
    p_S = F.softmax(branch_logits.float(), dim=1)[:, 1]    # (B, H, W)

    # pool to the patch grid, works for any H, W
    p_T_p = F.adaptive_avg_pool2d(p_T.unsqueeze(1), (n_patches, n_patches)).squeeze(1)
    p_S_p = F.adaptive_avg_pool2d(p_S.unsqueeze(1), (n_patches, n_patches)).squeeze(1)

    N = n_patches * n_patches
    p_T_flat = p_T_p.reshape(B, N)   # (B, N)
    p_S_flat = p_S_p.reshape(B, N)

    # raw outer product, (B, N, 1) x (B, 1, N) = (B, N, N)
    M_T = torch.bmm(p_T_flat.unsqueeze(2), p_T_flat.unsqueeze(1))
    M_S = torch.bmm(p_S_flat.unsqueeze(2), p_S_flat.unsqueeze(1))

    return F.mse_loss(M_S, M_T.detach())


def _spectral_agreement_mask(
    images: Tensor,
    teacher_logits: Tensor,
    norm_mean: list[float],
    norm_std: list[float],
    confidence_thresh: float = 0.65,
) -> Tensor:
    """Mask of pixels where the teacher prediction can be explained locally.

        A pixel counts if the teacher is confident (p_T > confidence_thresh or
        p_T < 1 - confidence_thresh) and the NDWI sign agrees with its decision
        (NDWI > 0 with p_T > 0.5 for flood, NDWI < 0 with p_T < 0.5 for background).

        Everything else needs global context to explain (ViT self-attention over the whole
        image), which an early CNN branch doesn't have. Zeroing those pixels keeps the branch
        from pushing persistent noise into the backbone.

        images: normalized (B, 6, H, W) S2 input. norm_mean / norm_std: 6 per-band values.
        teacher_logits: (B, 2, H, W), detached by the caller. confidence_thresh defaults
        to 0.65 and is symmetric. Returns a (B, H, W) float32 {0, 1} mask, no gradient.
    """
    with torch.no_grad():
        # Denormalize to reflectance scale (same as compute_spectral_proxies)
        mean = images.new_tensor(norm_mean).view(1, 6, 1, 1)
        std  = images.new_tensor(norm_std ).view(1, 6, 1, 1)
        x    = images.float() * std + mean           # (B, 6, H, W), reflectance in [0, 1]

        ndwi = (x[:, _B_GREEN] - x[:, _B_NIR]) / (
            x[:, _B_GREEN] + x[:, _B_NIR] + _SPEC_EPS
        )  # (B, H, W), in [-1, 1]

        p_t = F.softmax(teacher_logits.float(), dim=1)[:, 1]   # (B, H, W) flood prob

        # teacher confident enough to supervise
        confident = (p_t > confidence_thresh) | (p_t < (1.0 - confidence_thresh))

        # NDWI sign has to agree with the teacher: NDWI > 0 for flood, < 0 for background
        agree = (ndwi > 0) == (p_t > 0.5)

        mask = (confident & agree).float()   # (B, H, W), {0.0, 1.0}

    return mask


def _teacher_entropy(teacher_logits: Tensor) -> Tensor:
    """Binary entropy H(p_T)/log(2), in [0, 1].

        High at flood/background boundaries (p_T ~ 0.5), low in confident regions.
        Dividing by log(2) makes the maximum at p=0.5 exactly 1.
    """
    p   = F.softmax(teacher_logits.float(), dim=1)[:, 1]   # flood prob (B, H, W)
    eps = 1e-8
    h   = -(p * (p + eps).log() + (1.0 - p) * (1.0 - p + eps).log())
    return (h / math.log(2)).clamp(0.0, 1.0)


def _entropy_branch_loss(branch_pred: Tensor, entropy_target: Tensor) -> Tensor:
    """MSE between EntropyExitBranch output (B, 1, H, W) and normalized teacher entropy (B, H, W, detached)."""
    return F.mse_loss(branch_pred.squeeze(1), entropy_target)


def _compute_spectral_targets(
    images: Tensor,
    norm_mean: list[float],
    norm_std: list[float],
    num_proxies: int,
    continuous: bool,
) -> Tensor:
    """Spectral proxy targets from z-scored (B, 6, H, W) Sentinel-2 input.

        Undoes the normalization and computes NDWI, MNDWI, NDVI (num_proxies 1, 2 or 3).
        Returns (B, K, H, W) float32 without gradient: raw index values in [-1,1] for MSE
        if continuous, else binary {0,1} for BCE.
    """
    mean = images.new_tensor(norm_mean).view(1, 6, 1, 1)
    std  = images.new_tensor(norm_std ).view(1, 6, 1, 1)
    x    = images.float() * std + mean      # denormalize to reflectance

    green = x[:, _B_GREEN]
    nir   = x[:, _B_NIR  ]
    swir1 = x[:, _B_SWIR1]
    red   = x[:, 2       ]

    parts: list[Tensor] = []
    if num_proxies >= 1:
        ndwi = (green - nir  ) / (green + nir   + _SPEC_EPS)
        parts.append(ndwi.clamp(-1.0, 1.0) if continuous else (ndwi > 0.0).float())
    if num_proxies >= 2:
        mndwi = (green - swir1) / (green + swir1 + _SPEC_EPS)
        parts.append(mndwi.clamp(-1.0, 1.0) if continuous else (mndwi > 0.0).float())
    if num_proxies >= 3:
        ndvi = (nir - red) / (nir + red + _SPEC_EPS)
        if continuous:
            parts.append(ndvi.clamp(-1.0, 1.0))
        else:
            parts.append((ndvi < 0.2).float())  # low NDVI = potential water/bare

    return torch.stack(parts, dim=1)  # (B, K, H, W)


def _sses_branch_loss(
    branch_pred: Tensor,
    images: Tensor,
    norm_mean: list[float],
    norm_std: list[float],
    num_proxies: int,
    continuous: bool,
) -> Tensor:
    """Loss for the SSES-OFA spectral exit branch (raw SpectralExitBranch output, (B, K, H, W)).

        Scale at init: continuous MSE is around 0.03-0.08 (NDWI/MNDWI/NDVI variance), so
        sses_proxy_alpha ~1.0-3.0 matches the OFA loss (~0.5-1.0). Binary BCE starts around
        0.4-0.6, use sses_proxy_alpha ~0.3 there (same as ofa_alpha).
        continuous=False means BCE on logits. num_proxies is 1-3.
    """
    with torch.no_grad():
        targets = _compute_spectral_targets(images, norm_mean, norm_std, num_proxies, continuous)
    if continuous:
        return F.mse_loss(branch_pred.float(), targets)
    return F.binary_cross_entropy_with_logits(branch_pred.float(), targets)


def _nc_loss(
    branch_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 0.5,
) -> Tensor:
    """Neighborhood consistency loss for early exit branches.

        For horizontally or vertically adjacent pixels (i, j):
            w_ij = exp(-|p_T[i] - p_T[j]| / tau)
            loss = w_ij * (q_S[i] - q_S[j])^2

        When the teacher puts both pixels in the same region (similar p_T) the weight is high.
        At real boundaries it is low, so no smoothness is forced there. Gives the early branches
        the teacher's spatial coherence without any feature alignment. On late stages, which
        already get per-pixel OFA supervision, it just adds noise.

        teacher_logits come in detached. temperature is tau (default 0.5) and sets how sharp
        the weights are.
    """
    p_t = F.softmax(teacher_logits.float(), dim=1)[:, 1].detach()  # (B, H, W)
    q_s = F.softmax(branch_logits.float(), dim=1)[:, 1]             # (B, H, W)

    # Horizontal neighbors (width axis)
    dt_h = (p_t[:, :, 1:] - p_t[:, :, :-1]).abs()          # (B, H, W-1)
    ds_h = (q_s[:, :, 1:] - q_s[:, :, :-1]).pow(2)
    w_h  = (-dt_h / temperature).exp()

    # Vertical neighbors (height axis)
    dt_v = (p_t[:, 1:, :] - p_t[:, :-1, :]).abs()          # (B, H-1, W)
    ds_v = (q_s[:, 1:, :] - q_s[:, :-1, :]).pow(2)
    w_v  = (-dt_v / temperature).exp()

    return (w_h * ds_h).mean() + (w_v * ds_v).mean()


def _coherence_gate(grad: Tensor, rf_radius: int) -> Tensor:
    """GSCF backward hook, softly damps the gradient at spatially incoherent positions.

        Coherence is ||avg_pool(g)|| / ||g||, i.e. how much of the gradient survives averaging
        over the neighbourhood. If the neighbours agree, avg ~ g, ratio ~1 and the gradient is
        kept. If they cancel (e.g. boundary noise from global context) avg goes to 0 and the
        gradient is suppressed. Anything in between gives a smooth value in (0, 1).

        Cosine similarity doesn't work for C=2: the per-pixel gradient is always [a, -a]
        (softmax antisymmetry), so the cosine is just +-1. The magnitude ratio shows how much
        the opposite-class gradients of neighbouring pixels cancel, and that is continuous.

        grad is [B, C, H, W] at full resolution (b_out). rf_radius is the pooling radius in
        pixels, window (2*rf_radius+1)^2. Use RF_stage / 2 at full image resolution, i.e. 9
        for stage 0 (RF=19px) and 21 for stage 1 (RF=43px). Returns the filtered gradient.
    """
    with torch.no_grad():
        k          = 2 * rf_radius + 1
        nbhd       = F.avg_pool2d(grad, k, stride=1, padding=rf_radius)
        g_norm     = grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
        n_norm     = nbhd.norm(dim=1, keepdim=True)
        coherence  = (n_norm / g_norm).clamp(max=1.0)   # [B,1,H,W] in [0,1], soft
    return grad * coherence


# OFALoss: drop-in replacement for SSFDLoss / DistillLoss

class OFALoss(nn.Module):
    """OFA-KD loss with exit branches on intermediate student stages.

        Forward hooks on the student backbone grab per-stage feature maps during the student
        forward. Each stage gets a small OFAExitBranch (features to logits) and an OFA loss
        against the teacher's final logits.

        Used like SSFDLoss:
            loss_fn.register_hooks()
            loss_fn.prewarm(loader, device)   # builds exit branches before optimizer
            loss, components = loss_fn(student_logits, images, masks)
            loss_fn.remove_hooks()

        The branches sit in self.exit_branches (nn.ModuleList), Trainer._build_optimizer()
        picks them up as "loss_fn_extra" with the training LR. Trainer only saves
        model.state_dict(), so the branches never end up in the checkpoint and best.pt
        loads with build_student() as usual.

        teacher is frozen and kept outside nn.Module tracking. student is EfficientViT-B0 or
        a student with an explicitly marked spatial encoder. alpha is the KD weight on the main output (0..1),
        ofa_alpha the weight of the branch losses (paper: 0.3-1.0). gamma >= 1 is the OFA
        modulation exponent, the paper reports 1.1-1.4 as best; gamma=1 is plain KD per
        branch. ignore_index is left out of all losses.
    """

    _EVT_STAGES = ("stage1", "stage2", "stage3", "stage4")

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_classes: int = 2,
        alpha: float = 0.5,
        ofa_alpha: float = 0.3,
        temperature: float = 4.0,
        gamma: float = 1.2,
        ignore_index: int = -1,
        disc_weighted: bool = False,
        uncertainty_weighted: bool = False,
        dist_match_alpha: float = 0.0,
        relational_alpha: float = 0.0,
        upsample_first: bool = False,
        gradcam_weight_ofa: bool = False,
        gradcam_contrast_weight: bool = False,
        sc_ofa: bool = False,
        sc_entropy_stages: int = 2,
        nc_alpha: float = 0.0,
        nc_max_branch: int = 2,
        nc_temperature: float = 0.5,
        patch_affinity_alpha: float = 0.0,
        pa_num_patches: int = 14,
        pa_scales: list[int] | None = None,
        sm_ofa: bool = False,
        sm_stages: int = 2,
        sm_confidence_thresh: float = 0.65,
        norm_mean: list[float] | None = None,
        norm_std: list[float] | None = None,
        use_grad_filter: bool = False,
        grad_filter_stages: list[int] | None = None,
        grad_filter_rf: list[int] | None = None,
        ofa_min_stage: int = 0,
        sses: bool = False,
        sses_late_stage: int = 2,
        sses_proxy_alpha: float = 1.0,
        sses_num_proxies: int = 3,
        sses_proxy_mode: str = "spectral_continuous",
        branch_conflict_mode: bool = False,
        ofa_branch_warmup_epoch: int = 0,
        kd_valid_only: bool = False,
    ) -> None:
        super().__init__()
        if disc_weighted and uncertainty_weighted:
            raise ValueError(
                "disc_weighted and uncertainty_weighted are mutually exclusive: "
                "the former upweights confident pixels, the latter upweights uncertain ones."
            )
        _pw_flags = sum([gradcam_weight_ofa, gradcam_contrast_weight,
                         disc_weighted, uncertainty_weighted])
        if _pw_flags > 1:
            raise ValueError(
                "gradcam_weight_ofa, gradcam_contrast_weight, disc_weighted and "
                "uncertainty_weighted are mutually exclusive pixel-weighting schemes."
            )

        # Store outside nn.Module so optimizer and DataParallel ignore them.
        object.__setattr__(self, "_teacher", teacher)
        object.__setattr__(self, "_student", student)

        self.num_classes             = num_classes
        self.alpha                = alpha
        self.ofa_alpha            = ofa_alpha
        self.temperature          = temperature
        self.gamma                = gamma
        self.ignore_index         = ignore_index
        self.kd_valid_only        = kd_valid_only
        self.disc_weighted        = disc_weighted
        self.uncertainty_weighted = uncertainty_weighted
        self.dist_match_alpha     = dist_match_alpha
        self.relational_alpha     = relational_alpha
        self.upsample_first       = upsample_first
        self.gradcam_weight_ofa      = gradcam_weight_ofa
        self.gradcam_contrast_weight = gradcam_contrast_weight
        self.sc_ofa              = sc_ofa
        self.sc_entropy_stages   = sc_entropy_stages
        self.nc_alpha            = nc_alpha
        self.nc_max_branch       = nc_max_branch
        self.nc_temperature      = nc_temperature
        self.patch_affinity_alpha = patch_affinity_alpha
        self.pa_num_patches       = pa_num_patches
        # HPA: patch size per stage, coarse for early and fine for late branches.
        # None = off, pa_num_patches everywhere.
        self.pa_scales            = list(pa_scales) if pa_scales is not None else None

        # SM-OFA: spectral-agreement masking for early logit branches.
        self.sm_ofa               = sm_ofa
        self.sm_stages            = sm_stages
        self.sm_confidence_thresh = sm_confidence_thresh
        if sm_ofa and (norm_mean is None or norm_std is None):
            from data.sen1floods11 import NORM_MEAN, NORM_STD
            norm_mean = norm_mean or NORM_MEAN
            norm_std  = norm_std  or NORM_STD
        self.norm_mean = list(norm_mean) if norm_mean is not None else None
        self.norm_std  = list(norm_std)  if norm_std  is not None else None

        # GSCF (gradient spatial coherence filtering) for early OFA branches, backward hook on
        # b_out. Default stages 0 and 1 (RF 19px, 43px), rf_radius = RF/2 at full res.
        self.use_grad_filter    = use_grad_filter
        self.grad_filter_stages = list(grad_filter_stages) if grad_filter_stages is not None else [0, 1]
        # rf_radius per stage index: rf_radius[k] applies to grad_filter_stages[k].
        self.grad_filter_rf     = list(grad_filter_rf) if grad_filter_rf is not None else [9, 21]

        # No exit branches below ofa_min_stage, to test whether the small-RF early branches help
        # or hurt backbone training. ofa_min_stage=2 leaves stages 2 and 3 (RF >= 251px).
        self.ofa_min_stage      = ofa_min_stage

        # SSES (stage-specialized exit supervision): stages with abs_i < sses_late_stage get a
        # SpectralExitBranch with spectral index targets, the rest a normal OFAExitBranch.
        # Early stages (RF 19-43px) can't explain the teacher's global-context predictions, so
        # they get a clean local spectral target instead of the noisy teacher-logit gradient.
        # Late stages (RF 251-667px) keep full OFA. MSE targets are ~10x smaller than OFA, so
        # sses_proxy_alpha ~1.0 (continuous) or ~0.3 (binary BCE, same order as ofa_alpha).
        if sses and sses_proxy_mode not in ("spectral", "spectral_continuous"):
            raise ValueError(f"sses_proxy_mode must be 'spectral' or 'spectral_continuous', "
                             f"got {sses_proxy_mode!r}")
        self.sses             = sses
        self.sses_late_stage  = sses_late_stage
        self.sses_proxy_alpha = sses_proxy_alpha
        self.sses_num_proxies = sses_num_proxies
        self.sses_proxy_mode  = sses_proxy_mode
        self.sses_continuous  = (sses_proxy_mode == "spectral_continuous")
        # SSES needs norm_mean/norm_std, they are still None here if sm_ofa was off
        if sses and (self.norm_mean is None or self.norm_std is None):
            from data.sen1floods11 import NORM_MEAN, NORM_STD
            if self.norm_mean is None:
                self.norm_mean = list(NORM_MEAN)
            if self.norm_std is None:
                self.norm_std  = list(NORM_STD)
        # sec. 93: if True, compute_loss_terms() returns {focal, kd, ofa_s0..s3} instead of one
        # "ofa" term, so the early vs late branch cosines can be tracked.
        self.branch_conflict_mode = branch_conflict_mode

        # sec. 94 branch warmup (DBI-OFA): only the late branches (stages >= 2, RF 251/667px)
        # for the first ofa_branch_warmup_epoch epochs, then the early ones too. Avoids the
        # early-branch vs focal conflict while backbone features are still uninformative
        # (N<200). 0 = off, all branches from epoch 0 like normal OFA.
        # Trainer.fit() sets self.current_epoch each epoch.
        self.ofa_branch_warmup_epoch: int = ofa_branch_warmup_epoch
        self.current_epoch: int = 0

        # sec. 96 AGR-OFA (adaptive gradient reliability). _agr_ema[branch] is a beta-EMA of
        # the branch's backbone gradient (same device as the params). _agr_weights[branch] is
        # the weight from the last step, only kept for logging. beta=0.9, so roughly a
        # 10-step window.
        self._agr_ema:     dict[str, "Tensor | None"] = {}
        self._agr_weights: dict[str, float]            = {}

        # diag caches, filled in forward() anyway. _last_branch_losses: unweighted KD loss
        # per branch (ofa_s0..s3). _last_branch_entropy: mean prediction entropy per branch
        # in nats, a rough confidence proxy. Updated every forward, fine to read after a step.
        self._last_branch_losses:  dict[str, float] = {}
        self._last_branch_entropy: dict[str, float] = {}

        # Populated in prewarm(); registered as ModuleList so Trainer optimizer picks them up.
        self.exit_branches: nn.ModuleList = nn.ModuleList()
        self._branches_built: bool = False

        self._stage_feats: list[Tensor] = []
        self._hooks: list = []

    # Hook management

    def register_hooks(self) -> None:
        """Register a forward hook on the student backbone to capture stage features."""
        self.remove_hooks()
        student = object.__getattribute__(self, "_student")

        if hasattr(student, "model") and hasattr(student.model, "backbone"):
            def _s_hook(mod, inp, out):
                self._stage_feats.clear()
                if isinstance(out, dict):
                    self._stage_feats.extend(
                        out[s] for s in self._EVT_STAGES if s in out
                    )
                else:
                    feats = out if isinstance(out, (list, tuple)) else [out]
                    self._stage_feats.extend(feats)
            self._hooks.append(student.model.backbone.register_forward_hook(_s_hook))
            modes = []
            if self.disc_weighted:        modes.append("disc_weighted")
            if self.uncertainty_weighted: modes.append("uncertainty_weighted")
            if self.dist_match_alpha > 0: modes.append(f"dist_match({self.dist_match_alpha})")
            if self.relational_alpha  > 0: modes.append(f"relational({self.relational_alpha})")
            if self.sc_ofa:               modes.append(f"sc_ofa(entropy_stages={self.sc_entropy_stages})")
            if self.nc_alpha > 0:         modes.append(f"nc(alpha={self.nc_alpha},max_branch={self.nc_max_branch},T={self.nc_temperature})")
            if self.patch_affinity_alpha > 0:
                pa_info = f"scales={self.pa_scales}" if self.pa_scales else f"n={self.pa_num_patches}"
                modes.append(f"patch_affinity(alpha={self.patch_affinity_alpha},{pa_info})")
            if self.sm_ofa:
                modes.append(f"sm_ofa(stages={self.sm_stages},conf={self.sm_confidence_thresh})")
            if self.use_grad_filter:
                modes.append(f"gscf(stages={self.grad_filter_stages},rf={self.grad_filter_rf})")
            if self.ofa_min_stage > 0:
                modes.append(f"late_ofa(min_stage={self.ofa_min_stage})")
            if self.sses:
                modes.append(f"sses(late_stage={self.sses_late_stage},"
                             f"K={self.sses_num_proxies},mode={self.sses_proxy_mode},"
                             f"alpha={self.sses_proxy_alpha})")
            mode_str = " ".join(modes) if modes else "standard"
            print(f"[OFALoss] hook: student.model.backbone (EfficientViT)  "
                  f"ofa_alpha={self.ofa_alpha}  gamma={self.gamma}  mode={mode_str}")
        elif (hasattr(student, "encoder") and
              getattr(student.encoder, "_ofa_stage_features_type", None) == "spatial_list"):
            def _encoder_hook(mod, inp, out):
                if not isinstance(out, (list, tuple)) or not out:
                    raise RuntimeError("OFA encoder must return nonempty spatial feature maps.")
                if any(not isinstance(f, Tensor) or f.ndim != 4 for f in out):
                    raise RuntimeError("OFA encoder features must be BCHW tensors.")
                self._stage_feats.clear()
                self._stage_feats.extend(out)
            self._hooks.append(student.encoder.register_forward_hook(_encoder_hook))
            print(f"[OFALoss] hook: student.encoder ({type(student.encoder).__name__})  "
                  f"ofa_alpha={self.ofa_alpha}  gamma={self.gamma}")
        else:
            print("[OFALoss] WARNING: student architecture not recognised; "
                  "backbone hook not registered.  OFA branch losses will be zero.")

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._stage_feats.clear()

    # Branch construction

    def _build_branches(self, stage_feats: list[Tensor], device: torch.device) -> None:
        BranchCls = UpsampleFirstExitBranch if self.upsample_first else OFAExitBranch
        branches = []
        active_feats = stage_feats[self.ofa_min_stage:]
        for i, f in enumerate(active_feats):
            abs_i = i + self.ofa_min_stage
            if self.sses and abs_i < self.sses_late_stage:
                # SSES early stage: spectral indices instead of teacher logits, no teacher
                # needed and learnable with a local RF
                branches.append(SpectralExitBranch(f.shape[1], self.sses_num_proxies).to(device))
            elif self.sc_ofa and abs_i < self.sc_entropy_stages:
                branches.append(EntropyExitBranch(f.shape[1]).to(device))
            else:
                branches.append(BranchCls(f.shape[1], self.num_classes).to(device))
        self.exit_branches = nn.ModuleList(branches)
        self._branches_built = True
        chs   = [f.shape[1] for f in active_feats]
        types = [
            "spectral" if isinstance(b, SpectralExitBranch)
            else "entropy" if isinstance(b, EntropyExitBranch)
            else "logit"
            for b in branches
        ]
        n_par = sum(sum(p.numel() for p in b.parameters()) for b in branches)
        style = "upsample_first" if self.upsample_first else "project_first"
        skipped = f"  skipped stages 0–{self.ofa_min_stage-1}" if self.ofa_min_stage > 0 else ""
        print(f"[OFALoss] exit branches: {len(branches)} stages{skipped}  "
              f"in_ch={chs}  types={types}  total params={n_par:,}  style={style}")

    def prewarm(self, loader, device: torch.device) -> None:
        """One student forward to get the stage channel dims, then build the branches.

                Call after register_hooks() and before creating the Trainer, otherwise the
                exit branches aren't in the optimizer from step 0.
        """
        student = object.__getattribute__(self, "_student")
        student.eval()
        with torch.no_grad():
            batch  = next(iter(loader))
            images = (batch["image"] if isinstance(batch, dict) else batch[0])
            images = images.to(device, dtype=torch.float32)
            student(images)
            if self._stage_feats and not self._branches_built:
                self._build_branches(self._stage_feats, device)
            self._stage_feats.clear()
        print(f"[OFALoss] prewarm done: {len(self.exit_branches)} branches ready for optimizer.")

    # Branch warmup curriculum

    def _effective_min_stage(self) -> int:
        """ofa_min_stage for the current epoch.

                While current_epoch < ofa_branch_warmup_epoch it is forced to 2, so only the late
                stages (RF 251/667px) contribute. After warmup it goes back to self.ofa_min_stage.

                In the sec. 93 data the conflict between focal and the early branches was worst in
                the first 30-50% of training at small N (100-200), while the late branches got
                along with focal from step 1. Adding the early branches later lets the backbone
                settle on the task signal first.
        """
        if self.ofa_branch_warmup_epoch > 0 and self.current_epoch < self.ofa_branch_warmup_epoch:
            return max(self.ofa_min_stage, 2)
        return self.ofa_min_stage

    # Forward

    def forward(
        self,
        student_logits: Tensor,
        images: Tensor,
        masks: Tensor,
        precomputed_teacher_logits: Tensor | None = None,
        precomputed_proxy_target: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Combined OFA-KD loss.

                student(images) must have run already, so self._stage_feats is filled (Trainer
                calls model(images) first and then this forward()).

                Args:
                    student_logits:             (B, C, H, W) main student output.
                    images:                     (B, C, H, W) input images for teacher forward.
                    masks:                      (B, H, W) ground-truth labels.
                    precomputed_teacher_logits: (B, C, H, W) stored teacher logits. If given, used
                                                instead of a live teacher forward on `images`, same
                                                as DistillLoss in training/losses.py. None = live
                                                forward (the old behaviour, and still what pools
                                                without a 'soft_logits' key get).
                    precomputed_proxy_target:   (B, H, W) or (B, 1, H, W) GradCAM map, used as pixel
                                                weights with gradcam_weight_ofa=True. Needs pseudo_*.pt
                                                files with a 'gradcam' key (none of the shipped
                                                selectors write that).

                Returns (total_loss, components) with keys focal, kd, feat_distil (=ofa), at.
        """
        teacher = object.__getattribute__(self, "_teacher")

        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            if precomputed_teacher_logits is not None:
                # Stored teacher logits (pseudo generate/relabel --store_soft_logits). Before
                # 2026-08-26 this argument was accepted and then ignored, so any experiment that
                # changed the student's input space silently broke the teacher targets (`images`
                # is the student's tensor). Also saves a teacher forward on enriched pools, loss
                # is unchanged to ~1e-6 relative.
                teacher_logits = precomputed_teacher_logits.to(images.device).float()
            else:
                teacher_logits = teacher(images.float())

        # pixel weights for the point-wise KD and branch losses:
        #   disc_weighted: w = teacher confidence, favours certain pixels
        #   uncertainty_weighted: w = 1 - confidence, favours uncertain pixels
        #   gradcam_weight_ofa: w = GradCAM_flood in [0,1]
        #   gradcam_contrast_weight: w = |GradCAM_flood - GradCAM_nonflood| in [0,1], favours
        #   class-discriminative pixels (boundaries, ambiguous areas)
        # The GradCAM ones need an external saliency pass. They get the map through
        # precomputed_proxy_target (B,H,W) and normalise it to mean 1 per image.
        pixel_weights: Tensor | None = None
        _use_precomputed = (
            (self.gradcam_weight_ofa or self.gradcam_contrast_weight)
            and precomputed_proxy_target is not None
        )
        if _use_precomputed:
            with torch.no_grad():
                gcam = precomputed_proxy_target.float()
                if gcam.dim() == 4:          # (B, 1, H, W) to (B, H, W)
                    gcam = gcam.squeeze(1)
                # mean 1 per image so the loss scale stays the same
                pixel_weights = gcam / (gcam.mean(dim=(1, 2), keepdim=True) + 1e-8)
        elif self.disc_weighted or self.uncertainty_weighted:
            with torch.no_grad():
                conf = F.softmax(teacher_logits.float(), dim=1).max(dim=1).values
                pixel_weights = conf if self.disc_weighted else (1.0 - conf)
        # Main student output: focal + (optionally weighted) logit KD
        f = focal_loss(student_logits, masks, 2.0, self.ignore_index)
        if pixel_weights is not None:
            kd = _kd_loss_weighted(
                student_logits,
                teacher_logits.to(dtype=student_logits.dtype),
                self.temperature,
                pixel_weights,
                valid_mask=(masks != self.ignore_index) if self.kd_valid_only else None,
            )
        else:
            kd = kd_loss(
                student_logits,
                teacher_logits.to(dtype=student_logits.dtype),
                self.temperature,
                valid_mask=(masks != self.ignore_index) if self.kd_valid_only else None,
            )
        total = (1.0 - self.alpha) * f + self.alpha * kd

        # OFA branch losses + optional distributional / relational / SC / NC
        ofa_val     = student_logits.new_zeros(())
        dist_val    = student_logits.new_zeros(())
        rel_val     = student_logits.new_zeros(())
        entropy_val = student_logits.new_zeros(())
        sses_val    = student_logits.new_zeros(())
        nc_val      = student_logits.new_zeros(())
        pa_val      = student_logits.new_zeros(())

        if self._branches_built and self._stage_feats:
            H, W  = student_logits.shape[-2:]
            t_det = teacher_logits.detach()

            # SC-OFA: entropy target computed once, only when needed
            entropy_target: Tensor | None = None

            # SM-OFA: spectral-agreement mask - computed once, applied to early branches.
            # Shape (B, H, W), no gradient. None when sm_ofa is disabled.
            sm_mask: Tensor | None = None
            if self.sm_ofa:
                sm_mask = _spectral_agreement_mask(
                    images, t_det,
                    self.norm_mean, self.norm_std,      # type: ignore[arg-type]
                    self.sm_confidence_thresh,
                )

            ofa_list     = []
            dist_list    = []
            rel_list     = []
            entropy_list = []
            sses_list    = []
            nc_list      = []
            pa_list      = []

            # Active stage range: respects both ofa_min_stage and branch warmup curriculum.
            # _effective_min_stage() returns max(ofa_min_stage, 2) during the warmup window.
            _eff_min = self._effective_min_stage()
            active_feats   = self._stage_feats[_eff_min:]
            branch_offset  = _eff_min - self.ofa_min_stage
            for i, (feat, branch) in enumerate(zip(active_feats, self.exit_branches[branch_offset:])):
                abs_i = i + _eff_min
                b_out = branch(feat, target_hw=(H, W))

                # GSCF backward hook for early stages, their RF is too small for the
                # teacher's global-context predictions
                if self.use_grad_filter and abs_i in self.grad_filter_stages:
                    _rf = self.grad_filter_rf[self.grad_filter_stages.index(abs_i)]
                    b_out.register_hook(lambda g, r=_rf: _coherence_gate(g, r))

                if isinstance(branch, SpectralExitBranch):
                    # SSES early stage: spectral index targets, no teacher involved and
                    # no RF mismatch
                    sses_list.append(_sses_branch_loss(
                        b_out, images,
                        self.norm_mean, self.norm_std,      # type: ignore[arg-type]
                        self.sses_num_proxies, self.sses_continuous,
                    ))
                elif isinstance(branch, EntropyExitBranch):
                    # SC-OFA early stage: predict teacher entropy (local texture signal)
                    if entropy_target is None:
                        entropy_target = _teacher_entropy(t_det)
                    entropy_list.append(_entropy_branch_loss(b_out, entropy_target))
                else:
                    # SM-OFA: early logit branches use the spectral-agreement mask instead of
                    # pixel_weights, so their OFA gradient only comes from locally grounded
                    # pixels. Late branches keep pixel_weights.
                    if sm_mask is not None and abs_i < self.sm_stages:
                        branch_weights = sm_mask
                    else:
                        branch_weights = pixel_weights

                    _bl = ofa_loss(b_out, t_det, masks,
                                   gamma=self.gamma,
                                   ignore_index=self.ignore_index,
                                   temperature=self.temperature,
                                   pixel_weights=branch_weights)
                    ofa_list.append(_bl)
                    # cache the unweighted branch loss for the diagnostics logger
                    self._last_branch_losses[f"ofa_s{abs_i}"] = _bl.item()

                    with torch.no_grad():
                        _p = F.softmax(b_out.float().detach(), dim=1).clamp(min=1e-8)
                        self._last_branch_entropy[f"ofa_s{abs_i}_H"] = (
                            -(_p * _p.log()).sum(dim=1).mean().item()
                        )
                    # B: sorted 1D Wasserstein - match confidence histogram shape
                    if self.dist_match_alpha > 0.0:
                        dist_list.append(_sorted_wasserstein_loss(b_out, t_det))

                    # C: pixel-pair relational - match inter-pixel affinity structure
                    if self.relational_alpha > 0.0:
                        rel_list.append(_relational_loss(b_out, t_det))

                    # NC: spatial neighborhood consistency for early logit branches
                    if self.nc_alpha > 0.0 and abs_i < self.nc_max_branch:
                        nc_list.append(_nc_loss(b_out, t_det, self.nc_temperature))

                    # PA: patch affinity in logit space (C2VKD-like). With pa_scales (HPA) the
                    # patch size is per stage, pa_num_patches if the index is out of range.
                    if self.patch_affinity_alpha > 0.0:
                        if self.pa_scales is not None and abs_i < len(self.pa_scales):
                            n = self.pa_scales[abs_i]
                        else:
                            n = self.pa_num_patches
                        pa_list.append(_patch_affinity_loss(b_out, t_det, n))

            if ofa_list:
                ofa_val = torch.stack(ofa_list).mean()
                total   = total + self.ofa_alpha * ofa_val

            if entropy_list:
                entropy_val = torch.stack(entropy_list).mean()
                total       = total + self.ofa_alpha * entropy_val

            if sses_list:
                sses_val = torch.stack(sses_list).mean()
                total    = total + self.sses_proxy_alpha * sses_val

            if dist_list:
                dist_val = torch.stack(dist_list).mean()
                total    = total + self.dist_match_alpha * dist_val

            if rel_list:
                rel_val = torch.stack(rel_list).mean()
                total   = total + self.relational_alpha * rel_val

            if nc_list:
                nc_val = torch.stack(nc_list).mean()
                total  = total + self.nc_alpha * nc_val

            if pa_list:
                pa_val = torch.stack(pa_list).mean()
                total  = total + self.patch_affinity_alpha * pa_val

            self._stage_feats.clear()

        components = {
            "focal":       f.item(),
            "kd":          kd.item(),
            "feat_distil": ofa_val.item(),    # reuses Trainer log key for W&B visibility
            "dist_match":  dist_val.item(),
            "relational":  rel_val.item(),
            "sc_entropy":      entropy_val.item(),
            "sses_spectral":   sses_val.item(),
            "nc":              nc_val.item(),
            "patch_affinity":  pa_val.item(),
            "at":              0.0,
            "fird":        0.0,
            "contrastive": 0.0,
        }
        return total, components


    def _grad_conflict_terms(self) -> list[str]:
        """Names of the terms compute_loss_terms() returns.

                Normally ["focal", "kd", "ofa"] (3 terms, 3 pairs). In branch_conflict_mode
                (sec. 93) the per-stage branch names instead, so GradientConflictLogger gets
                the C(4,2)=6 early-vs-late cosines.
        """
        if self.branch_conflict_mode:
            n = 4 - self.ofa_min_stage
            # kd is in here so the backbone_pcgrad step applies it to non-backbone params,
            # and sec. 93 runs also get kd vs branch pairs
            names = ["focal", "kd"] + [f"ofa_s{self.ofa_min_stage + i}" for i in range(n)]
        else:
            names = ["focal", "kd", "ofa"]
        return names

    @property
    def _loss_weights(self) -> dict[str, float]:
        """PCGrad combination weights matching the forward() loss equation."""
        w = {"focal": 1.0 - self.alpha, "kd": self.alpha, "ofa": self.ofa_alpha}
        return w

    def compute_loss_terms(
        self,
        model: nn.Module,
        images: Tensor,
        masks: Tensor,
    ) -> dict[str, Tensor]:
        """Unweighted loss tensors {"focal", "kd", "ofa"} for the gradient conflict analysis.

                Called from Trainer._log_grad_conflict(). Does a fresh student + teacher forward
                so the hook buffers are filled, no optimizer.step(). "ofa" is zero if no stage
                features were captured (hooks inactive).
        """
        teacher = object.__getattribute__(self, "_teacher")

        # Student forward - populates _stage_feats via hook
        with torch.autocast(device_type=images.device.type, enabled=False):
            student_logits = model(images.float())

        # Teacher forward - FP32, no grad
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            teacher_logits = teacher(images.float())

        pixel_weights: Tensor | None = None
        if self.disc_weighted:
            with torch.no_grad():
                pixel_weights = F.softmax(teacher_logits.float(), dim=1).max(dim=1).values

        f  = focal_loss(student_logits, masks, 2.0, self.ignore_index)
        if self.disc_weighted and pixel_weights is not None:
            kd = _kd_loss_weighted(
                student_logits,
                teacher_logits.to(dtype=student_logits.dtype),
                self.temperature,
                pixel_weights,
            )
        else:
            kd = kd_loss(
                student_logits,
                teacher_logits.to(dtype=student_logits.dtype),
                self.temperature,
            )

        ofa_val = student_logits.new_zeros(())
        if self._branches_built and self._stage_feats:
            H, W  = student_logits.shape[-2:]
            t_det = teacher_logits.detach()

            if self.branch_conflict_mode:
                # sec. 93/94: focal + kd + ofa_s{i} per stage (kd again for backbone_pcgrad).
                # active_feats = _stage_feats[ofa_min_stage:] so indexing matches forward().
                active_feats = self._stage_feats[self.ofa_min_stage:]
                result: dict[str, Tensor] = {"focal": f, "kd": kd}
                for i, (feat, branch) in enumerate(zip(active_feats, self.exit_branches)):
                    abs_i = i + self.ofa_min_stage
                    b_logits = branch(feat, target_hw=(H, W))
                    result[f"ofa_s{abs_i}"] = ofa_loss(
                        b_logits, t_det, masks,
                        gamma=self.gamma,
                        ignore_index=self.ignore_index,
                        temperature=self.temperature,
                        pixel_weights=pixel_weights,
                    )
                self._stage_feats.clear()
                return result

            per_branch = []
            for feat, branch in zip(self._stage_feats[self.ofa_min_stage:], self.exit_branches):
                b_logits = branch(feat, target_hw=(H, W))
                per_branch.append(
                    ofa_loss(b_logits, t_det, masks,
                             gamma=self.gamma,
                             ignore_index=self.ignore_index,
                             temperature=self.temperature,
                             pixel_weights=pixel_weights)
                )
            ofa_val = torch.stack(per_branch).mean()
            self._stage_feats.clear()

        terms = {"focal": f, "kd": kd, "ofa": ofa_val}

        return terms

    def get_agr_weight_and_update(
        self,
        name: str,
        g_flat: "Tensor",
        beta: float = 0.9,
    ) -> float:
        """AGR reliability weight for exit branch `name`, also updates its EMA.

                The weight says how well the branch's current backbone gradient lines up with its
                own beta-EMA of past backbone gradients. BB-PCGrad (sec. 94) compared against the
                focal gradient, which is a noisy anchor. Here a branch is only compared with
                itself over time.

                At small N (<=100 scenes) branch gradients flip sign between batches because the
                features are still immature and per-batch predictions are incoherent. A branch
                that keeps reversing gets a weight near 0, one that keeps pointing the same way
                gets close to 1.

                    w_i^t    = max(0, cos(g_i^t, gbar_i^{t-1}))        in [0, 1]
                    gbar_i^t = beta * gbar_i^{t-1} + (1 - beta) * g_i^t

                g_i^t is the current backbone gradient of branch i, gbar the EMA. The weight is
                computed before the EMA update so the current gradient doesn't leak into it.
                Returns 1.0 on the first step (no history yet).
        """
        g = g_flat.detach().float()
        ema = self._agr_ema.get(name)

        if ema is None or ema.norm() < 1e-12:
            # no history yet, trust it fully
            weight = 1.0
        else:
            cos_sim = F.cosine_similarity(g.unsqueeze(0), ema.unsqueeze(0)).item()
            weight = max(0.0, cos_sim)  # negative cosine counts as 0

        # weight is set; now update EMA with the current gradient
        with torch.no_grad():
            if ema is None:
                self._agr_ema[name] = g.clone()
            else:
                self._agr_ema[name] = beta * ema + (1.0 - beta) * g

        self._agr_weights[name] = weight
        return weight

    # per-branch diagnostics

    def get_branch_loss_breakdown(self) -> dict[str, float]:
        """Unweighted KD loss per branch from the last forward.

                Keys ofa_s0..ofa_s3, only for active stages. Cached in forward(), no extra cost.
        """
        return dict(self._last_branch_losses)

    def get_branch_entropy_breakdown(self) -> dict[str, float]:
        """Mean prediction entropy (nats) per branch from the last forward, keys ofa_s0_H..ofa_s3_H.

                High means diffuse predictions. Low means confident, which at small N can still be wrong.
        """
        return dict(self._last_branch_entropy)
