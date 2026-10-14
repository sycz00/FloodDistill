"""V10 reference implementations (Claude, 13 Oct 2026). Pure functions; import-safe; no I/O.

Codex must use these definitions (or byte-identical copies) and keep the unit tests in
test_v10_reference.py passing. Band order: B2 blue, B3 green, B4 red, B8A NIR, B11 SWIR1, B12 SWIR2.
Inputs are the S11-normalized six-band tensors used everywhere in FloodDistill.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

MEAN = np.array([.1413, .138, .1235, .309, .2045, .1191], np.float32)   # identical to edge_v9 / NORM_MEAN
STD = np.array([.0741, .0737, .0869, .118, .0977, .0766], np.float32)


def reflectance(x):
    """S11-normalized (6,H,W) -> reflectance (6,H,W)."""
    return np.asarray(x, np.float32) * STD[:, None, None] + MEAN[:, None, None]


def indices(rho):
    """Published water indices at their published threshold 0 (water if > 0); availability mask."""
    b, g, r, n, s1, s2 = rho
    ok = np.isfinite(rho).all(0) & (g >= 0) & (s1 >= 0) & (g + s1 > 1e-6)
    mndwi = np.where(ok, (g - s1) / np.maximum(g + s1, 1e-6), 0).clip(-1, 1)           # Xu 2006
    aweish = b + 2.5 * g - 1.5 * (n + s1) - 0.25 * s2                                     # Feyisa et al. 2014
    wi2015 = 1.7204 + 171 * g + 3 * r - 70 * n - 45 * s1 - 71 * s2                       # Fisher et al. 2016
    return dict(MNDWI=mndwi, AWEIsh=np.where(ok, aweish, 0), WI2015=np.where(ok, wi2015, 0)), ok


def edge_distance(mask):
    """Euclidean distance of every pixel to the opposite class of a hard mask (identical to edge_v9)."""
    m = np.asarray(mask, bool)
    if m.all() or not m.any():
        return np.full(m.shape, np.inf)
    return np.where(m, ndi.distance_transform_edt(m), ndi.distance_transform_edt(~m))


def unmix_fraction(x, mask, k=2, radius=7, min_px=8, max_residual=1.0):
    """Local two-endmember linear unmixing of the water fraction inside the edge band.

    x: S11-normalized (6,H,W); mask: teacher hard mask (H,W) in {0,1}.
    Band = pixels within k px of the mask edge. Endmembers per pixel = mean standardized
    spectrum of the band-free interior water (mask=1, MNDWI>0) and interior land (mask=0)
    inside a (2*radius+1)^2 window. Fraction f = projection of the pixel onto the segment
    land->water, clipped to [0,1]. Returns (f, used): f is NaN where the fit is unusable
    (fewer than min_px interior pixels of either class, degenerate endmembers, or residual
    norm > max_residual * endmember distance); 'used' marks band pixels with a valid f.
    Standardized space (the network input) keeps NIR/SWIR from dominating the fit.
    """
    z = np.asarray(x, np.float32)
    mask = np.asarray(mask, bool)
    dist = edge_distance(mask)
    band = dist <= k
    f = np.full(mask.shape, np.nan, np.float32)
    if not band.any():
        return f, np.zeros(mask.shape, bool)
    rho = reflectance(z)
    mndwi, ok = indices(rho)[0]['MNDWI'], indices(rho)[1]
    wi = (mask & ~band & ok & (mndwi > 0)).astype(np.float32)
    li = (~mask & ~band & ok).astype(np.float32)
    size = 2 * radius + 1
    cw = ndi.uniform_filter(wi, size, mode='constant') * size * size
    cl = ndi.uniform_filter(li, size, mode='constant') * size * size
    ew = np.stack([ndi.uniform_filter(z[c] * wi, size, mode='constant') * size * size for c in range(6)]) / np.maximum(cw, 1e-6)
    el = np.stack([ndi.uniform_filter(z[c] * li, size, mode='constant') * size * size for c in range(6)]) / np.maximum(cl, 1e-6)
    d = ew - el
    dd = (d * d).sum(0)
    t = ((z - el) * d).sum(0) / np.maximum(dd, 1e-12)
    tc = np.clip(t, 0, 1)
    resid = np.sqrt(((z - (el + tc[None] * d)) ** 2).sum(0))
    usable = band & ok & (cw >= min_px) & (cl >= min_px) & (dd > 1e-4) & (resid <= max_residual * np.sqrt(dd))
    f[usable] = tc[usable]
    return f, usable


def soft_focal_loss(logits, target_water, valid, gamma=2.0):
    """Focal loss with a soft water target t in [0,1]; equals training.losses.focal_loss for t in {0,1}.

    loss_i = -[ t (1-p_w)^g log p_w + (1-t) (1-p_l)^g log p_l ], mean over valid pixels.
    logits: (B,2,H,W) torch; target_water: (B,H,W) float; valid: (B,H,W) bool.
    """
    import torch
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    t = target_water.clamp(0, 1)
    loss = -(t * (1 - p[:, 1]) ** gamma * logp[:, 1] + (1 - t) * (1 - p[:, 0]) ** gamma * logp[:, 0])
    v = loss[valid]
    return v.mean() if v.numel() else logits.sum() * 0.0


def source_matched_tau(margins_ref, target_fraction, lo=-8.0, hi=8.0, iters=60):
    """Threshold tau so that mean(margins_ref > tau) equals target_fraction (bisection on a 1-D array)."""
    m = np.asarray(margins_ref, np.float64).ravel()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (m > mid).mean() > target_fraction:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def decode(pred, index_map, available, k, theta):
    """Decode-time edge re-decision (identical to edge_v9.decode for the MNDWI map)."""
    pred = np.asarray(pred, bool)
    band = (edge_distance(pred) <= k) & available
    out = pred.copy()
    out[band] = (index_map > theta)[band]
    return out


def native_grid_score(band20, block=2):
    """Share of 2x2 blocks with identical values: ~1 for nearest-neighbour 20->10 m upsampling, ~0 otherwise."""
    a = np.asarray(band20)
    h, w = (a.shape[0] // block) * block, (a.shape[1] // block) * block
    b = a[:h, :w].reshape(h // block, block, w // block, block)
    same = (b == b[:, :1, :, :1]).all(axis=(1, 3))
    return float(same.mean())


# ---------------------------------------------------------------------------
# Tiny-model failure decomposition (capacity vs context vs representation vs data).
# Parameter-free input transforms live inside the model, so every evaluator that feeds
# the standard S11-normalized six-band tensor works unchanged.

def _torch():
    import torch
    from torch import nn
    from torch.nn import functional as F
    return torch, nn, F


def log_reflectance(x, eps=0.01, mean=None, std=None):
    """log(reflectance + eps) per band, standardized with TRAIN statistics (mean/std tensors of shape (6,))."""
    torch, _, _ = _torch()
    m = torch.as_tensor(MEAN, dtype=x.dtype, device=x.device).view(1, 6, 1, 1)
    s = torch.as_tensor(STD, dtype=x.dtype, device=x.device).view(1, 6, 1, 1)
    lr = torch.log((x * s + m).clamp_min(0) + eps)
    if mean is not None:
        lr = (lr - mean.view(1, 6, 1, 1)) / std.view(1, 6, 1, 1)
    return lr


def box_context(x, size):
    """Parameter-free local context: mean of each input band over a size x size window (reflect padding)."""
    torch, _, F = _torch()
    p = size // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode='reflect'), size, stride=1)


def build_tiny(kind, log_mean=None, log_std=None):
    """Decomposition models. kind:
    'R2'        per-pixel MLP 6-16-16-2 (418 params; identical to rungs.R2)
    'R2_log'    per-pixel MLP on [x, log-reflectance] (514 params)  -> representation
    'R2_ctx'    per-pixel MLP on [x, box9(x), box33(x)] (610 params) -> parameter-free context
    'R4_micro'  rungs.ContextConv(widths=(4,4,6,8), dec=4) (1,266 params; receptive field > 64 px) -> learned context at R3 size
    """
    torch, nn, F = _torch()

    class PerPixelOn(nn.Module):
        def __init__(self, cin, feats):
            super().__init__()
            self.feats = feats
            self.net = nn.Sequential(nn.Conv2d(cin, 16, 1), nn.ReLU6(inplace=True), nn.Conv2d(16, 16, 1),
                                     nn.ReLU6(inplace=True), nn.Conv2d(16, 2, 1))
            if log_mean is not None:
                self.register_buffer('log_mean', torch.as_tensor(log_mean, dtype=torch.float32))
                self.register_buffer('log_std', torch.as_tensor(log_std, dtype=torch.float32))

        def forward(self, x):
            return self.net(self.feats(self, x))

    if kind == 'R2':
        return PerPixelOn(6, lambda self, x: x)
    if kind == 'R2_log':
        assert log_mean is not None and log_std is not None, 'TRAIN log-reflectance statistics required'
        return PerPixelOn(12, lambda self, x: torch.cat([x, log_reflectance(x, mean=self.log_mean, std=self.log_std)], 1))
    if kind == 'R2_ctx':
        return PerPixelOn(18, lambda self, x: torch.cat([x, box_context(x, 9), box_context(x, 33)], 1))
    if kind == 'R4_micro':
        import importlib.util, json
        from pathlib import Path
        spec = json.loads((Path(__file__).resolve().parents[2] / 'capacity_v7_20261011T210730Z' / 'training_spec.json').read_text())
        s = importlib.util.spec_from_file_location('v10_rungs', Path(spec['rungs_dir']) / 'rungs.py')
        r = importlib.util.module_from_spec(s); s.loader.exec_module(r)
        return r.ContextConv(widths=(4, 4, 6, 8), dec=4)
    raise ValueError(kind)


def offline_kd_loss(student_logits, masks, teacher_logits, alpha=0.5, temperature=4.0, gamma=2.0, kd_valid_only=True):
    """Logit KD with CACHED teacher logits; identical formula to training.losses.DistillLoss:
    (1-alpha)*focal(student, hard) + alpha * T^2 * KL(softmax(t/T) || softmax(s/T)), batchmean over (valid) pixels.
    teacher_logits must be the teacher's float logits (B,2,H,W) for the SAME geometric view as the student input
    (apply flips/rotations/crops to the cached logits jointly with image and mask; bilinear for logits).
    Do NOT rebuild logits from the stored float16 'confidence' map: float16 saturates at 0.99951, capping the
    logit margin at ~7.6 and making targets systematically softer than the live teacher's.
    """
    import torch
    from torch.nn import functional as F
    from training.losses import focal_loss, kd_loss
    f = focal_loss(student_logits, masks, gamma, -1)
    kd = kd_loss(student_logits, teacher_logits, temperature, valid_mask=(masks != -1) if kd_valid_only else None)
    return (1.0 - alpha) * f + alpha * kd
