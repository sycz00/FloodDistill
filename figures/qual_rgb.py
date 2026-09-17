# !/usr/bin/env python3
"""Shared RGB display conversion for six-band Sentinel-2 model tensors.

Invert the fixed S11 normalization on B4/B3/B2 to obtain RGB reflectance.
The default display is min(reflectance/0.40,1), gamma=1, using one scale for
all scenes and conditions. Only for display, model inputs are never touched.
Detect nodata in reflectance space: zero normalized values are not nodata.

Setting fixed=0 enables percentile stretching. Reuse clean or shared limits
for paired perturbation panels; separate per-scene stretches can hide the
perturbation. Per-channel stretching also changes relative band colours.
"""
from __future__ import annotations

import numpy as np
import torch

from data.sen1floods11 import NORM_MEAN, NORM_STD

# (B2, B3, B4, B8A, B11, B12) -> display (R, G, B) = (B4, B3, B2)
RGB_BANDS = (2, 1, 0)




def rgb_reflectance(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """z-scored [C,H,W] -> float32 reflectance [H,W,3] in display order (R, G, B).

        Only the three display bands are inverted, so this works for the raw 9-band STURM
        tensor and for the adapted 6-band one. Both have B2, B3, B4 at indices 0, 1, 2 (STURM
        is [B2,B3,B4,B5,B6,B7,B8A,B11,B12] and its _STURM_S2_NORM_MEAN/STD copy the S11 values
        at exactly those positions), so RGB_BANDS and NORM_MEAN[RGB_BANDS] mean the same in
        both. Inverting the whole tensor would not work: NORM_MEAN has six entries, raw STURM
        has nine bands, and after index 2 they aren't even the same bands.

        No clipping. A hazed or up-gained tile can legitimately go above 1.0 and clipping
        would hide the perturbation in the robustness panel.
    """
    x = torch.as_tensor(np.asarray(img), dtype=torch.float32)
    if x.shape[0] <= max(RGB_BANDS):
        raise ValueError(f"need at least {max(RGB_BANDS) + 1} bands, got {x.shape[0]}")
    return np.stack([(x[i] * NORM_STD[i] + NORM_MEAN[i]).numpy() for i in RGB_BANDS],
                    axis=-1)


def nodata_mask(rgb: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """[H,W] mask, True where the tile has no data. Computed on reflectance (see module docstring).

        `rgb` is the [H,W,3] stack from rgb_reflectance. Loaders that pad write no-data as
        zero reflectance (GDACS scene edges, S1S2Water strips). `tol` instead of == 0 because
        the z-score round trip is float32. All False if nothing is zero, so datasets without
        padding are not affected.
    """
    return np.all(np.abs(rgb) < tol, axis=-1)


# Fixed true-colour display, clip(2.5*reflectance) on B4/B3/B2 (the linear true-colour
# recipe of Sentinel Hub / Copernicus Browser). With one shared scale band ratios survive
# until clipping and a clean tile looks the same in the gain and haze panels. A per-scene
# percentile stretch would move the black point and depend on the paired corruption.
# Gain scales mean and spread, additive haze only raises the mean. Don't change this
# mapping between conditions.
TRUE_COLOUR_CEILING = 0.40


def to_rgb(img: torch.Tensor | np.ndarray,
           ref: list[tuple[float, float]] | None = None,
           p_low: float = 2.0, p_high: float = 98.0,
           gamma: float = 1.0, sat: float = 1.0,
           per_channel: bool = False,
           fixed: float | None = None,
           black_nodata: bool = True):
    """Convert normalized (C,H,W) input to display RGB and return stretch limits.

    Default fixed=0.40 uses one reflectance ceiling. fixed=0 enables percentiles;
    ref reuses given limits for paired images. per_channel changes white balance.
    sat=1 leaves saturation unchanged. Display only.
    """
    rgb = rgb_reflectance(img)
    void = nodata_mask(rgb)

    valid = ~void if (~void).any() else np.ones(void.shape, bool)
    if fixed is not None:
        pcts = [(0.0, float(fixed))] * 3     # zero-referenced, see TRUE_COLOUR_CEILING
    elif ref is not None:
        pcts = list(ref)
    elif per_channel:
        pcts = [tuple(np.percentile(rgb[..., c][valid], [p_low, p_high])) for c in range(3)]
    else:
        lo, hi = np.percentile(rgb[valid], [p_low, p_high])   # all three bands pooled
        pcts = [(float(lo), float(hi))] * 3

    out = np.empty_like(rgb)
    for c in range(3):
        lo, hi = pcts[c]
        out[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    if gamma and gamma != 1.0:
        out = out ** (1.0 / gamma)
    if sat != 1.0:
        lum = out.mean(axis=-1, keepdims=True)
        out = np.clip(lum + sat * (out - lum), 0.0, 1.0)
    if black_nodata:
        out[void] = 0.0
    return out, [(float(a), float(b)) for a, b in pcts]


def shared_stretch(a, b, **kw):
    """Percentiles over both tiles of a clean vs corrupted pair.

        With the clean tile's own stretch an h=0.20 column goes white, with the corrupted
        tile's own stretch the corruption disappears. Taking both keeps the columns readable
        and the shift still shows.
    """
    _, pa = to_rgb(a, **kw)
    _, pb = to_rgb(b, **kw)
    return [(min(pa[c][0], pb[c][0]), max(pa[c][1], pb[c][1])) for c in range(3)]


def input_quality(img_chw: np.ndarray, key: str,
                  mask: np.ndarray | None = None) -> tuple[bool, float, str]:
    """Is the tile worth showing? Returns (keep, contrast, reason_if_dropped).

        The cloud cutoff removes bright cloud sheets but keeps featureless tiles, e.g. flat
        water or bare soil where the visible bands hardly vary. Those render as a flat wash
        and show nothing, which is what this check is for. `contrast` is the p2-p98 spread of
        visible reflectance, the same thing the display stretch divides by, so a low value
        means there's almost nothing to stretch.

        STURM is judged on its own scale: its chips are min-max scaled to rho in [0.04, 0.16]
        by the dataset's DN convention, so a spread that would be flat for S11 is normal there.
    """
    rgb = rgb_reflectance(img_chw)
    void = nodata_mask(rgb)
    # 5%, not 20%. Measured: the GDACS tile this gate used to pass had 17.4% no-data and
    # 96.2% of its positive label lying within 8 px of that no-data edge -- the "flood" it
    # showed was the black border strip, and every model "found" it.
    if void.mean() > 0.05:
        return False, 0.0, f"no-data {100*void.mean():.0f}%"
    if mask is not None and void.any():
        pos = mask == 1
        if pos.any():
            from scipy.ndimage import binary_dilation
            on_edge = float((pos & binary_dilation(void, iterations=8)).sum() / pos.sum())
            if on_edge > 0.30:
                return False, 0.0, f"label on no-data border ({100*on_edge:.0f}% of positives)"
    vis = rgb[~void] if void.any() else rgb
    lo, hi = np.percentile(vis, [2, 98])
    contrast = float(hi - lo)
    floor = 0.030 if key == "STURM-S2" else 0.045
    if contrast < floor:
        return False, contrast, f"flat (p2-p98 spread {contrast:.3f} < {floor})"
    # A tile whose median visible reflectance is very high is a haze/cloud sheet even when
    # the bright-pixel fraction stayed under the cloud cutoff.
    med = float(np.median(vis))
    if med > 0.32:
        return False, contrast, f"washed out (median reflectance {med:.2f})"

    # Haze colour cast. Rayleigh scattering goes with ~lambda^-4, so on a hazy tile blue sits
    # far above red and the tile renders cyan. The stretch shows the data correctly, but it is
    # useless as an illustration. Normal land/water scenes have blue/red around 0.8-1.5.
    med_r = float(np.median(rgb[..., 0][~void])) if void.any() else float(np.median(rgb[..., 0]))
    med_b = float(np.median(rgb[..., 2][~void])) if void.any() else float(np.median(rgb[..., 2]))
    # 1.35, from the measured Sen1Floods11 distribution: the cyan tile this gate used to
    # pass sits at 1.51, the 15th-bluest of 90 scenes. 1.35 removes the bluest quarter.
    if med_r > 1e-6 and med_b / med_r > 1.35:
        return False, contrast, f"haze cast (blue/red median {med_b / med_r:.2f})"

    # Striping / resampling artefacts. Scene-edge tiles and some WorldFloods windows have strong
    # banding in one direction. Real terrain has roughly isotropic gradients, so mean |d/dx|
    # over mean |d/dy| separates the two well.
    lum = rgb.mean(axis=-1)
    gx = float(np.abs(np.diff(lum, axis=1)).mean())
    gy = float(np.abs(np.diff(lum, axis=0)).mean())
    aniso = max(gx, gy) / max(min(gx, gy), 1e-9)
    if aniso > 2.5:
        return False, contrast, f"striping (gradient anisotropy {aniso:.1f})"
    return True, contrast, ""
