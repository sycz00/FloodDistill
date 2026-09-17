#!/usr/bin/env python3
"""Sentinel-2 true-colour conversion shared by all qualitative figures.

There used to be three scripts, each with its own RGB conversion, and two of them were wrong
in different ways. That's where the orange/teal (fig_haze_qualitative), lavender/magenta
(evaluation/plot_models) and half-black (GDACS rows) panels came from. The maths is short,
one copy is enough.

Input tensor
------------
Every loader in this project gives the model a 6-band z-scored tensor

    z_c = (rho_c - mu_c) / sigma_c ,   (B2, B3, B4, B8A, B11, B12)

with the Sen1Floods11 per-band mu/sigma (NORM_MEAN / NORM_STD). That's the case for S11, for
the GDACS pseudo pool (stored already normalised), WorldFloods with normalize="s11", S1S2Water
with normalize="clip", and for the three visible bands of STURM-S2, whose
_STURM_S2_NORM_MEAN/STD copy the S11 values at B2/B3/B4.

Stretch options
---------------
The two old scripts got this wrong in opposite directions.

  (a) joint percentile over the three channels of the z-scored array, what
      fig_haze_qualitative did. Not a colour transform at all: mu and sigma differ per band
      (B2 0.1413+-0.0741 vs B4 0.1235+-0.0869), so the per-band offsets end up in the
      output. Hence the orange/teal panels.

  (b) per-channel percentile, what evaluation/plot_models did. Invariant to any per-channel
      affine map, so at least well defined on z-scores. But it is a per-scene white balance:
      every band is forced to the same output range, which destroys the band ratios that
      carry the colour. On scenes with unusual band statistics (low-contrast farmland, a
      strong haze layer) it invents a colour cast that isn't in the data, e.g. the
      magenta/green S11 panels. `sat < 1` was being used to hide this.

  (c) joint percentile over the three channels of the reflectance array. Reflectance is the
      same physical unit in all bands, so one black point and one white point for all three
      keeps the ratios better than (a) or (b) and still adapts to the scene's dynamic range.

(a) and (c) only differ in the space the percentile is computed in, and that space was the
whole bug. `per_channel=True` selects (b) for the rare dark scene where white balance helps.

None of the three is the default anymore, though. They are all per-scene, which doesn't work
for a figure comparing tiles or clean vs corrupted: the black point shifts the hue from tile
to tile and the limits depend on what else is in the row. The default is now
`fixed=TRUE_COLOUR_CEILING`, the standard Sentinel-2 True Color map (see the comment above
that constant). (c) is kept for single-scene panels where adapting to the scene is wanted.

No-data
-------
evaluation/plot_models also looked for no-data in the wrong space, it tested `all(z == 0)`.
z = 0 means rho = mu, a perfectly normal average-brightness pixel; real no-data (rho = 0) is
at z = -mu/sigma ~ -1.9. So the padding strips were never excluded from the percentiles (they
pulled the 2nd percentile down and washed out the scene) and never painted black. Here the
check happens in reflectance space, where no-data really is zero.

Gamma
-----
Defaults to 1.0 on purpose. The 1/2.2 sRGB transfer is for linear reflectance referenced to
true black. After a per-scene percentile stretch, zero is the scene's 2nd percentile and not
zero reflectance, and the scene range has already been expanded to the display range (for
S11 tile 42 the visible bands span rho 0.062-0.118, and the stretch alone blows that up to
[0,1]). Gamma on top brightens the image a second time, which shows in the numbers:

    mean display value      S11    STURM   WF      (natural imagery sits near 0.35-0.45)
    p2-p98 + gamma 2.2      0.559  0.588   0.540   (washed out, everything too bright)
    p2-p98, no gamma        0.405  0.372   0.306   (correct)
    fixed rho/0.3, gamma    0.544  0.677   0.673   (also too bright)

So never both. Gamma is off for the fixed default too, for a similar reason: the standard S2
true-colour ceiling is defined as a map to the 8-bit display code (ESA's TCI applies no
gamma), so the transfer function is already built in and an sRGB encode would brighten
twice. Measured as scheme D in figures/qual_rgb_smoketest.py.

Use this module rather than writing yet another conversion.
"""
from __future__ import annotations

import importlib.util as _ilu
import numpy as np
import torch

# check for terratorch before importing data.sen1floods11. That module runs
# `import albumentations` (-> OpenCV -> libGL) before the failing `from terratorch...`.
# A try/except cannot undo those imports. On Jetson, loading albumentations before
# tensorrt/pycuda set up CUDA causes a static-TLS conflict
# (see figures/plot_models.py::run_jetson). If terratorch is missing, do not run
# the body of data/sen1floods11.py.
if _ilu.find_spec("terratorch") is not None:
    from data.sen1floods11 import NORM_MEAN, NORM_STD
else:
    # terratorch-free fallback. Values verified identical to
    # data.sen1floods11_nodep.NORM_MEAN/NORM_STD (not imported here for the same reason).
    NORM_MEAN = [0.1413, 0.1380, 0.1235, 0.3090, 0.2045, 0.1191]
    NORM_STD  = [0.0741, 0.0737, 0.0869, 0.1180, 0.0977, 0.0766]

# (B2, B3, B4, B8A, B11, B12) -> display (R, G, B) = (B4, B3, B2)
RGB_BANDS = (2, 1, 0)




def rgb_reflectance(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """z-scored [C,H,W] -> float32 reflectance [H,W,3] in display order (R, G, B).

    Only the three display bands are inverted, so this works for the raw 9-band STURM tensor
    and for the adapted 6-band one. Both have B2, B3, B4 at indices 0, 1, 2 (STURM is
    [B2,B3,B4,B5,B6,B7,B8A,B11,B12] and its _STURM_S2_NORM_MEAN/STD copy the S11 values at
    exactly those positions), so RGB_BANDS and NORM_MEAN[RGB_BANDS] mean the same in both.
    Inverting the whole tensor would not work: NORM_MEAN has six entries, raw STURM has nine
    bands, and after index 2 they aren't even the same bands.

    No clipping. A hazed or up-gained tile can legitimately go above 1.0 and clipping would
    hide the perturbation in the robustness panel.
    """
    x = torch.as_tensor(np.asarray(img), dtype=torch.float32)
    if x.shape[0] <= max(RGB_BANDS):
        raise ValueError(f"need at least {max(RGB_BANDS) + 1} bands, got {x.shape[0]}")
    return np.stack([(x[i] * NORM_STD[i] + NORM_MEAN[i]).numpy() for i in RGB_BANDS],
                    axis=-1)


def nodata_mask(rgb: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """[H,W] mask, True where the tile has no data. Computed on reflectance (see module docstring).

    `rgb` is the [H,W,3] stack from rgb_reflectance. Loaders that pad write no-data as zero
    reflectance (GDACS scene edges, S1S2Water strips). `tol` instead of == 0 because the
    z-score round trip is float32. All False if nothing is zero, so datasets without padding
    are not affected.
    """
    return np.all(np.abs(rgb) < tol, axis=-1)


# Display map: the standard Sentinel-2 True Color recipe,
# display = clip(rho / TRUE_COLOUR_CEILING) = clip(2.5 * rho), applied to B4, B3, B2.
#
# That is exactly the Sentinel Hub / Copernicus Browser "True Color" evalscript,
#     return [2.5 * B04, 2.5 * B03, 2.5 * B02]
# which is about the most reproduced Sentinel-2 RGB recipe. Same family as the True Colour
# Image (TCI) band ESA ships in every S2 product: a zero-referenced linear scale from
# reflectance to the 8-bit display code, saturating at rho = 0.3558 for Level-1C and
# rho = 0.20 for Level-2A. Written out here so it can be reviewed; the evalscript line above
# reproduces it in EO Browser.
#
# Why the per-scene percentile stretch is not the default anymore:
#   1. It has a per-scene black point. Subtracting the same lo from all three bands doesn't
#      keep band ratios, (R-lo)/(G-lo) != R/G, so hue and saturation drift from tile to tile
#      with lo. That was the "some STURM tiles yellowish, some greenish" effect. A plain
#      scale rho/K keeps every ratio, so the hue on the page is the hue in the data.
#   2. Its limits depend on the other images in the figure. shared_stretch uses the union
#      over the clean/corrupted pair, so the same clean tile looks different in a haze panel
#      than in a gain panel. Measured: a WorldFloods tile drops from a 0.30 median to 0.11
#      (badly under-exposed) just because h=0.20 is in the same row.
#   3. There is nothing to cite for "we auto-stretched each image".
#
# K = 0.40 replaces the earlier 0.30. 0.30 was fitted to the pooled median visible reflectance
# (0.111 STURM / 0.111 S11 / 0.101 WF) to land near 0.37 on the display. The selectable tiles
# are brighter than the pool median, though, and 0.30 left no headroom: on four representative
# tiles rho/0.30 showed the clean input at a 0.35-0.45 median (bright and flat) and h=0.20 at a
# 0.97-1.00 median with 30-61% of pixels pure white, so no visible detail. With rho/0.40 the
# same tiles come out at 0.26-0.34 clean (natural, the EO Browser look) and 0.73-0.80 at
# h=0.20 with 0-2% saturated: a strong visible veil with the scene still recognisable below.
# That also rules out the other settings we measured. ESA's own L2A constant (0.20) is too
# bright for the STURM chips, whose stored DN range is already min-max compressed, and an sRGB
# gamma on top of a TCI ceiling brightens twice because the ceiling already gives a display
# code (ESA applies no gamma). figures/qual_rgb_smoketest.py renders all of them side by side.
#
# Gain and haze look different under this map:
#   gain  rho' = g*rho          multiplicative. Mean and spread both scale by g, so it looks
#                               like an exposure/contrast change: bright pixels move much more
#                               than dark ones and the hue doesn't change at all (a scale
#                               cancels in every band ratio).
#   haze  rho' = rho + h*w_i    additive, w ~ (490/lambda), so ~1.00/0.88/0.74 on B2/B3/B4.
#                               The mean goes up but the spread doesn't, so contrast drops:
#                               a flat, blue-shifted veil. That's the physical signature of
#                               atmospheric path radiance (Chavez/Makarau) and the panel
#                               should show it.
# So gain reads as contrast and haze as brightness, and that only renders correctly with a
# fixed stretch, not one re-fitted per tile.
TRUE_COLOUR_CEILING = 0.40


def to_rgb(img: torch.Tensor | np.ndarray,
           ref: list[tuple[float, float]] | None = None,
           p_low: float = 2.0, p_high: float = 98.0,
           gamma: float = 1.0, sat: float = 1.0,
           per_channel: bool = False,
           fixed: float | None = None,
           black_nodata: bool = True):
    """z-scored [C,H,W] -> (rgb [H,W,3] in [0,1], the (lo, hi) stretch used per channel).

    Default is stretch (c) from the module docstring: one black point and one white point for
    all three channels, computed in reflectance space, so band ratios survive and the result is
    true colour. `per_channel=True` gives the white-balancing variant (b).

    `ref` fixes the stretch to limits computed elsewhere. That matters for perturbation panels:
    a per-image stretch re-normalises the corrupted tile and removes exactly the corruption the
    panel is meant to show, so clean and hazed look the same. Pass the clean tile's limits (or
    the union of both) for the corrupted column.

    `sat` < 1 blends towards per-pixel luminance. Only useful with per_channel=True, whose white
    balance can push outlier pixels to neon; 1.0 = off.
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

    With the clean tile's own stretch an h=0.20 column goes white, with the corrupted tile's
    own stretch the corruption disappears. Taking both keeps the columns readable and the
    shift still shows.
    """
    _, pa = to_rgb(a, **kw)
    _, pb = to_rgb(b, **kw)
    return [(min(pa[c][0], pb[c][0]), max(pa[c][1], pb[c][1])) for c in range(3)]


def input_quality(img_chw: np.ndarray, key: str,
                  mask: np.ndarray | None = None) -> tuple[bool, float, str]:
    """Is the tile worth showing? Returns (keep, contrast, reason_if_dropped).

    The cloud cutoff removes bright cloud sheets but keeps featureless tiles, e.g. flat water
    or bare soil where the visible bands hardly vary. Those render as a flat wash and show
    nothing, which is what this check is for. `contrast` is the p2-p98 spread of visible
    reflectance, the same thing the display stretch divides by, so a low value means there's
    almost nothing to stretch.

    STURM is judged on its own scale: its chips are min-max scaled to rho in [0.04, 0.16] by
    the dataset's DN convention, so a spread that would be flat for S11 is normal there.
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
