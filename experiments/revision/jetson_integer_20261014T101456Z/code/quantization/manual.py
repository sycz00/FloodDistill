"""Manual INT8 PTQ: per-channel symmetric weights + per-channel asymmetric activations.

Quantization scheme:
  Weights      per-output-channel symmetric INT8  (scale_c = max|w_c| / 127)
  Activations  per-INPUT-channel asymmetric INT8, calibrated via percentile MSE
               (per-channel matters a lot for EfficientViT, whose activation
               distributions differ between channels: per-tensor loses 11 pp,
               per-channel ~1-2 pp at the cost of a dequant per channel)
  BN           folded exactly into Conv2d before quantization
  Non-linear   Hardswish / ReLU stay in FP32 between quantized ops
  Final head   last Conv2d left float (raw logits are signed)

Calibration: for each input channel, percentile-based range [lo_p, hi_p] with
p searched over {0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0} and MSE-selected.

The model runs in fake-quantized float32 for accuracy evaluation. For real INT8
speedup, use TensorRT / SNPE on the target hardware (Jetson Xavier).
"""

from __future__ import annotations

import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval



# Weight quantization: per-channel symmetric INT8


def _w_quant_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize weight per output channel (symmetric INT8).

        Returns (w_fq, scale) where:
          scale[c] = max(|w[c,...]|) / 127     shape [C_out]
          w_fq     = round(w / scale) * scale  (stored as float32)
    """
    qmax = 127
    C_out = w.shape[0]
    scale = w.view(C_out, -1).abs().max(dim=1).values.clamp(min=1e-8) / qmax
    s = scale.view(C_out, *([1] * (w.dim() - 1)))
    w_fq = (w / s).round().clamp(-qmax, qmax) * s
    return w_fq, scale



# Activation calibration: per-channel asymmetric INT8, percentile MSE


# Percentile values to try for the lower / upper clip boundary (symmetric search)
_PCTS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


def _mse_scale_zp_1d(samples_1d: torch.Tensor) -> Tuple[float, int]:
    """Asymmetric signed INT8 scale and zero_point for a 1-D sample vector.

        Tries the clamp ranges [percentile(p_lo), percentile(100-p_hi)] for all (p_lo, p_hi) in
        _PCTS x _PCTS and keeps the one with the lowest MSE. Signed INT8 range [-128, 127].

        lo <= 0 <= hi is forced so 0 is always representable. Otherwise all-positive channels
        (e.g. ReLU6) give lo_raw > 0 and zp < -128, and in QuantConv2d.forward x=0 would then
        dequantize to lo_raw instead of 0, a systematic bias on every dead (zero-output) neuron.
    """
    qmin, qmax = -128, 127
    t = samples_1d.float()
    n = t.numel()
    if n == 0:
        return 1e-4, 0

    t_sorted = t.sort().values
    best_scale, best_zp, best_mse = None, None, float("inf")

    for p_lo in _PCTS:
        idx_lo = max(0, int(p_lo / 100.0 * n))
        lo_raw = t_sorted[idx_lo].item()
        lo = min(lo_raw, 0.0)   # force lo <= 0 so zp >= -128 and 0 is representable
        for p_hi in _PCTS:
            idx_hi = min(n - 1, n - 1 - int(p_hi / 100.0 * n))
            hi_raw = t_sorted[idx_hi].item()
            hi = max(hi_raw, 0.0)   # force hi >= 0 so zp <= 127

            span = hi - lo
            if span < 1e-8:
                continue
            scale = span / (qmax - qmin)
            zp = int(round(qmin - lo / scale))
            zp = max(qmin, min(qmax, zp))   # clamp for float rounding safety
            t_q = (t / scale + zp).round().clamp(qmin, qmax)
            mse = ((t - (t_q - zp) * scale).pow(2).mean().item())
            if mse < best_mse:
                best_mse, best_scale, best_zp = mse, scale, zp

    if best_scale is None:
        amax = t.abs().max().item()
        best_scale = max(amax / 127.0, 1e-4)
        best_zp = 0

    return best_scale, best_zp



# QuantConv2d: fake-quantized Conv2d layer


class QuantConv2d(nn.Module):
    """Conv2d with per-channel symmetric INT8 weights and per-channel INT8 activations.

        Weights are quantized at construction (stored as fake-quant float32). Activation
        quantization stays off until calibrate() is called.

        Forward fake-quantizes the input per input channel (asymmetric signed INT8, scale[c] and
        zp[c] from the percentile MSE calibration), then runs the float32 conv with the
        fake-quantized weights. No output quantization, the next layer's input quant covers it.

        EfficientViT needs per-channel activation quantization because its post-Hardswish
        activations have very different ranges per channel. Per-tensor costs 11 pp mIoU,
        per-channel ~1-2 pp.
    """

    def __init__(
        self,
        orig: nn.Conv2d,
        w_fq: torch.Tensor,
        w_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.stride   = orig.stride
        self.padding  = orig.padding
        self.dilation = orig.dilation
        self.groups   = orig.groups

        self.weight = nn.Parameter(w_fq.clone(), requires_grad=False)
        self.bias   = (
            nn.Parameter(orig.bias.data.clone(), requires_grad=False)
            if orig.bias is not None else None
        )
        self.register_buffer("w_scale", w_scale.clone())

        # calibrate() sets this; None passes through without activation quantization
        # Per-channel: tensors of shape [C_in]
        self._act_scale: torch.Tensor | None = None   # [C_in]
        self._act_zp:    torch.Tensor | None = None   # [C_in] (int, stored as float)

    def calibrate(self, scale: torch.Tensor, zp: torch.Tensor) -> None:
        """Set per-channel activation quantization parameters.

                scale: [C_in]  float tensor
                zp:    [C_in]  int tensor (stored as float32)
        """
        self._act_scale = scale.float()
        self._act_zp    = zp.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SmoothQuant: divide input by per-channel scale before quantization.
        # smooth_scale was set by apply_smooth_quant(); weights were re-scaled
        # by the same factor so Conv(x, w) = Conv(x/s, w*s) is equivalent.
        if hasattr(self, "smooth_scale"):
            x = x / self.smooth_scale.to(x.device).view(1, -1, 1, 1)
        if self._act_scale is not None:
            # x: [B, C, H, W] - quantize per channel C
            s  = self._act_scale.to(x.device).view(1, -1, 1, 1)
            zp = self._act_zp.to(x.device).view(1, -1, 1, 1)
            x = ((x / s + zp).round().clamp(-128, 127) - zp) * s
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

    def extra_repr(self) -> str:
        if self._act_scale is not None:
            return (
                f"per-ch INT8 act_quant=on  C_in={self._act_scale.numel()}  "
                f"w_scale_range=[{self.w_scale.min():.2e}, {self.w_scale.max():.2e}]"
            )
        return f"act_quant=off  w_scale_range=[{self.w_scale.min():.2e}, {self.w_scale.max():.2e}]"



# BN folding


def fold_bn(model: nn.Module) -> nn.Module:
    """Fold all Conv2d+BatchNorm2d pairs in-place (model must be in eval mode)."""
    for _, module in model.named_modules():
        # EfficientViT ConvNorm: children named 'conv' (Conv2d) + 'norm' (BN2d)
        if (
            hasattr(module, "conv") and hasattr(module, "norm")
            and isinstance(module.conv, nn.Conv2d)
            and isinstance(module.norm, nn.BatchNorm2d)
        ):
            module.conv = fuse_conv_bn_eval(module.conv, module.norm)
            module.norm = nn.Identity()

        # Generic: consecutive Conv2d + BatchNorm2d children in the same parent
        children = list(module.named_children())
        for i in range(len(children) - 1):
            n_a, m_a = children[i]
            n_b, m_b = children[i + 1]
            if isinstance(m_a, nn.Conv2d) and isinstance(m_b, nn.BatchNorm2d):
                setattr(module, n_a, fuse_conv_bn_eval(m_a, m_b))
                setattr(module, n_b, nn.Identity())

    return model



# Conv replacement


def _all_convs(model: nn.Module) -> List[Tuple[nn.Module, str, nn.Conv2d]]:
    """Return [(parent, child_name, Conv2d)] for every Conv2d in the model."""
    result = []
    for _, parent in model.named_modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d):
                result.append((parent, name, child))
    return result


def _replace_convs(
    model: nn.Module, skip_last_n: int = 1
) -> List[QuantConv2d]:
    """Replace Conv2d layers with QuantConv2d (weight-quantized).

        The last `skip_last_n` Conv2d layers are kept as float32 (final head).
        Returns the list of QuantConv2d layers in model-traversal order.
    """
    convs = _all_convs(model)
    n_skip = skip_last_n if skip_last_n >= 0 else 0
    to_quant = convs[: len(convs) - n_skip] if n_skip else convs

    quant_convs: List[QuantConv2d] = []
    for parent, name, conv in to_quant:
        w_fq, w_scale = _w_quant_per_channel(conv.weight.data)
        qconv = QuantConv2d(conv, w_fq, w_scale)
        setattr(parent, name, qconv)
        quant_convs.append(qconv)

    print(
        f"  replaced {len(to_quant)}/{len(convs)} Conv2d with QuantConv2d "
        f"(kept {len(convs) - len(to_quant)} float for final head)"
    )
    return quant_convs



# Activation calibration


_SUBSAMPLE_PER_CH = 500  # max activation samples per channel per batch


def calibrate_activations(
    model: nn.Module,
    quant_convs: List[QuantConv2d],
    cal_batches: list,
) -> None:
    """Calibrate per-channel activation scales for all QuantConv2d layers.

        Forward hooks collect the inputs of every QuantConv2d. For each input channel j,
        _SUBSAMPLE_PER_CH spatial values per batch go into the percentile MSE calibration
        (_mse_scale_zp_1d). Per-channel is what makes EfficientViT work, see QuantConv2d.
    """
    n_layers = len(quant_convs)
    act_ch_samples: List[dict] = [{} for _ in range(n_layers)]

    handles = []
    for i, qconv in enumerate(quant_convs):
        def _hook(m, inp, out, idx=i):
            x = inp[0].detach().cpu().float()  # [B, C, H, W]
            if hasattr(m, "smooth_scale"):
                x = x / m.smooth_scale.cpu().view(1, -1, 1, 1)
            C = x.shape[1]
            for c in range(C):
                xc = x[:, c, :, :].flatten()
                n = min(_SUBSAMPLE_PER_CH, xc.numel())
                perm = torch.randperm(xc.numel())[:n]
                if c not in act_ch_samples[idx]:
                    act_ch_samples[idx][c] = []
                act_ch_samples[idx][c].append(xc[perm])
        handles.append(qconv.register_forward_hook(_hook))

    n_cal = len(cal_batches)
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(cal_batches):
            model(batch["image"])
            if (bi + 1) % 50 == 0 or bi == n_cal - 1:
                print(f"  [{bi+1}/{n_cal}] collecting activation samples …", flush=True)

    for h in handles:
        h.remove()

    for i, qconv in enumerate(quant_convs):
        ch_data = act_ch_samples[i]
        if not ch_data:
            continue
        C = max(ch_data.keys()) + 1
        scales = torch.zeros(C)
        zps    = torch.zeros(C)
        for c in range(C):
            if c not in ch_data:
                scales[c] = 1e-4
                continue
            t = torch.cat(ch_data[c])
            ch_data[c] = []
            s, z = _mse_scale_zp_1d(t)
            scales[c] = s
            zps[c]    = z
        act_ch_samples[i] = {}
        qconv.calibrate(scales, zps)

    print(f"  calibrated {n_layers} layers (per-channel signed INT8)")



# Public API


def quantize_model(
    model: nn.Module,
    cal_batches: list,
    skip_last_n: int = 1,
) -> nn.Module:
    """PTQ: fold BN, quantize weights, calibrate activations.

        `model` is the float32 model in eval mode on CPU, cal_batches a list of
        {'image': Tensor, ...} dicts. skip_last_n Conv2d layers stay float (default 1, the
        final head). Returns a fake-quantized deep copy (float32 ops, simulated INT8 weights
        and activations), `model` itself is not changed.
    """
    m = copy.deepcopy(model).cpu().eval()

    print("[manual_w8a8] step 1/3  fold BN into Conv")
    n_bn = sum(1 for _, x in m.named_modules() if isinstance(x, nn.BatchNorm2d))
    fold_bn(m)
    n_bn_after = sum(1 for _, x in m.named_modules() if isinstance(x, nn.BatchNorm2d))
    print(f"  folded {n_bn - n_bn_after} BN layers ({n_bn_after} BN remaining as Identity)")

    print("[manual_w8a8] step 2/3  replace Conv2d → QuantConv2d (per-channel weight INT8)")
    quant_convs = _replace_convs(m, skip_last_n=skip_last_n)

    print(f"[manual_w8a8] step 3/3  calibrate activation scales ({len(cal_batches)} batches, per-channel MSE)")
    calibrate_activations(m, quant_convs, cal_batches)

    return m


def calibrate_output_scales(
    quant_model: nn.Module,
    cal_batches: list,
    percentile: float = 99.99,
) -> None:
    """Calibrate per-tensor symmetric output scales for each QuantConv2d.

        Required for TRT explicit-quantization mode: TRT needs matched input+output
        Q-DQ pairs around each Conv to form a valid INT8 subgraph.
    """
    qconvs = [m for _, m in quant_model.named_modules() if isinstance(m, QuantConv2d)]
    out_samples: dict = {id(qc): [] for qc in qconvs}

    def _make_hook(qc_id):
        def _hook(module, inp, out):
            v = out.detach().abs().float().flatten()
            n = min(10_000, v.numel())
            out_samples[qc_id].append(v[torch.randperm(v.numel())[:n]])
        return _hook

    handles = [qc.register_forward_hook(_make_hook(id(qc))) for qc in qconvs]
    n_cal = len(cal_batches)
    quant_model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(cal_batches):
            quant_model(batch["image"].float().cpu())
            if (bi + 1) % 50 == 0 or bi == n_cal - 1:
                print(f"  [{bi+1}/{n_cal}] collecting output samples …", flush=True)
    for h in handles:
        h.remove()

    p = percentile / 100.0
    for qc in qconvs:
        samples = out_samples[id(qc)]
        if not samples:
            qc._out_scale = torch.tensor(1e-8 / 127.0, dtype=torch.float32)
            continue
        v = torch.cat(samples).quantile(p).clamp(min=1e-8).item()
        qc._out_scale = torch.tensor(v / 127.0, dtype=torch.float32)

    print(
        f"  output scales calibrated for {len(qconvs)} QuantConv2d layers "
        f"({percentile:.4g}th-percentile)"
    )


def apply_smooth_quant(
    quant_model: nn.Module,
    cal_batches: list,
    alpha: float = 0.5,
) -> None:
    """SmoothQuant: redistribute inter-channel activation variance into weights.

        For each QuantConv2d, computes a per-input-channel scale:
            s_j = max|x_j|^alpha / max|w[:,j]|^(1-alpha)
        then absorbs s into the (fake-quantized) weight by re-scaling and
        re-quantizing.  After calling this, re-run calibrate_activations() so
        activation scales are fitted to the smoothed (x/s) inputs.
    """
    qconvs = [m for _, m in quant_model.named_modules() if isinstance(m, QuantConv2d)]
    if not qconvs:
        print("  apply_smooth_quant: no QuantConv2d layers found, skipping")
        return

    act_max: dict = {id(qc): None for qc in qconvs}

    def _make_hook(qc_id):
        def _hook(module, inp, out):
            x = inp[0].detach().cpu().float()
            ch_max = x.abs().amax(dim=(0, 2, 3))
            prev = act_max[qc_id]
            act_max[qc_id] = torch.maximum(prev, ch_max) if prev is not None else ch_max
        return _hook

    handles = [qc.register_forward_hook(_make_hook(id(qc))) for qc in qconvs]
    quant_model.eval()
    with torch.no_grad():
        for batch in cal_batches:
            quant_model(batch["image"].float().cpu())
    for h in handles:
        h.remove()

    for qc in qconvs:
        x_max = act_max[id(qc)]
        if x_max is None:
            continue

        w = qc.weight.data
        G       = qc.groups
        C_out_G = w.shape[0] // G
        C_in_G  = w.shape[1]
        C_in    = G * C_in_G

        if x_max.shape[0] != C_in:
            continue

        w_r   = w.view(G, C_out_G, C_in_G, *w.shape[2:])
        w_max = w_r.abs().amax(dim=(1, 3, 4)).flatten().clamp(min=1e-8)

        s = (x_max.clamp(min=1e-8) ** alpha) / (w_max ** (1.0 - alpha))
        s = s.clamp(min=1e-5, max=1e5)

        s_r      = s.view(G, 1, C_in_G, *([1] * len(w.shape[2:])))
        w_scaled = (w_r * s_r).view_as(w)
        w_fq_new, w_scale_new = _w_quant_per_channel(w_scaled)
        qc.weight.data.copy_(w_fq_new)
        qc.w_scale.data.copy_(w_scale_new)

        qc.register_buffer("smooth_scale", s.clone())

    n = sum(1 for qc in qconvs if hasattr(qc, "smooth_scale"))
    print(f"  SmoothQuant applied to {n}/{len(qconvs)} QuantConv2d layers (alpha={alpha})")
