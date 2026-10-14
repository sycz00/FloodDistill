"""Quantization-aware training for TRT INT8 deployment.

Called via quantize.py qat. prepare_qat() calibrates activation and weight scales and
inserts QATConv2d layers; train_qat() fine-tunes with the task loss at a low LR, and
strip_qat() turns QATConv2d back into plain float Conv2d.
For ONNX export, export_qat_qdq_onnx() uses the original QAT calibration scales.
Re-calibrating would give mismatched scales.

Activation scale, depending on calibration_type:

  "per_tensor" (default): global running max, scale = amax / 127, zp = 0 (symmetric), using
  fake_quantize_per_tensor_affine with zp=0. The STE gradient is non-zero for anything in the
  representable range, round(x/s) + 0 stays in [-128, 127] as long as |x| <= 127*s. That is
  critical for ReLU6 activations: per-channel asymmetric with zp=-128 kills the gradient for
  values slightly below 0 and QAT collapses (~0.49 mIoU). TensorRT on the Jetson Xavier needs a
  per-tensor activation scale for INT8 inference, so this is the only calibration_type that
  gives a directly deployable model. It needs ReLU6-bounded activations for good precision,
  i.e. efficientvit_b0_relu6 and not efficientvit_b0 (Hardswish).

  "per_channel_symmetric": per input channel, scale_c = abs_max_c / 127, zp_c = 0, using
  fake_quantize_per_channel_affine with all-zero zps. Full 127-step resolution per channel and
  no STE dead zone, and outlier channels don't inflate the scale like with per-tensor. Only for
  research/ablations: TensorRT needs per-tensor activation scales, and such a model can't go on
  the Xavier without collapsing the scales to per-tensor at export, which brings the precision
  loss back.

  "per_channel": per input channel asymmetric INT8, MSE calibrated. Better accuracy at the
  start but can create STE dead zones in ReLU6 networks, only use it if you checked the model
  has mixed-sign activations everywhere. Not deployable on the Xavier either (same reason as
  per_channel_symmetric).

Weight scales are per-channel symmetric (best weight precision). Scales stay fixed after
calibration, the weights adapt and not the grid; learnable scales make ~10 epoch QAT runs
unstable. BN is not folded during QAT, it stays active and is folded at ONNX export. In the
warmup epochs fake-quant is off so the model settles before the quantization noise starts.
LR has to be <= 1e-5, higher values push weights outside the INT8 grid, and weight_decay is
1e-5 because more regularises the weights away from the grid.
"""

from __future__ import annotations

import copy
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Max activation samples collected per channel per batch during calibration.
_SUBSAMPLE_PER_CH = 500

_PCTS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


def _mse_scale_zp_valid(samples_1d: torch.Tensor) -> tuple[float, int]:
    """Per-channel asymmetric INT8 calibration with zero_point kept in [-128, 127].

        Like _mse_scale_zp_1d in manual.py, but the searched range always contains 0
        (lo <= 0 <= hi), so zero_point stays in [-128, 127] as
        torch.fake_quantize_per_channel_affine requires. ReLU channels (lo ~ 0, hi > 0) get
        lo=0 and only hi percentiles are searched, all-negative channels get hi=0 and only lo
        is searched, mixed-sign channels use the normal search since 0 is in range anyway.
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
        lo = min(lo_raw, 0.0)   # force lo <= 0, so zp <= 127

        for p_hi in _PCTS:
            idx_hi = min(n - 1, n - 1 - int(p_hi / 100.0 * n))
            hi_raw = t_sorted[idx_hi].item()
            hi = max(hi_raw, 0.0)   # force hi >= 0, so zp >= -128

            span = hi - lo
            if span < 1e-8:
                continue

            scale = span / (qmax - qmin)
            zp = int(round(qmin - lo / scale))   # guaranteed in [-128, 127]
            zp = max(qmin, min(qmax, zp))         # clamp for float rounding safety

            t_q = (t / scale + zp).round().clamp(qmin, qmax)
            mse = ((t - (t_q - zp) * scale).pow(2).mean().item())
            if mse < best_mse:
                best_mse, best_scale, best_zp = mse, scale, zp

    if best_scale is None:
        amax = t.abs().max().item()
        best_scale = max(amax / 127.0, 1e-4)
        best_zp = 0

    return best_scale, best_zp



# LiteMLA calibration helper
#
# _make_litemla_calib_hook and LiteMLAQAT below follow LiteMLA.forward and
# LiteMLA.relu_linear_att from EfficientViT (mit-han-lab/efficientvit,
# efficientvit/models/nn/ops.py, commit de7d773), Apache-2.0.
# Changed here: range recording for calibration, the two matmuls go through
# QATMatMul, and the normaliser is clamped instead of adding eps.
# See THIRD_PARTY_NOTICES.md.


def _make_litemla_calib_hook(entry: dict, mod: nn.Module) -> callable:
    """Return a calibration replacement for LiteMLA.relu_linear_att.

        Replicates the relu_linear_att computation so that intermediate tensors can
        be observed without calling the @torch.autocast-decorated original.  Stores
        the running abs-max of each matmul operand and output into `entry`:
          mm1_a   : max|v_padded|  -  first input to matmul-1
          mm1_b   : max|trans_k|   -  second input to matmul-1
          mm1_out : max|vk|        -  output of matmul-1 (= first input to matmul-2)
          mm2_b   : max|q|         -  second input to matmul-2
          mm2_out : max|out|       -  raw output of matmul-2 (before division + reshape)
    """
    def _hooked(qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = qkv.size()
        qkv_f = qkv.float() if qkv.dtype == torch.float16 else qkv
        qkv_r = qkv_f.reshape(B, -1, 3 * mod.dim, H * W)
        q = qkv_r[:, :, :mod.dim]
        k = qkv_r[:, :, mod.dim : 2 * mod.dim]
        v = qkv_r[:, :, 2 * mod.dim :]
        q = mod.kernel_func(q)
        k = mod.kernel_func(k)
        trans_k = k.transpose(-1, -2)
        v_pad = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        entry['mm1_a']   = max(entry['mm1_a'],   v_pad.detach().cpu().float().abs().max().item())
        entry['mm1_b']   = max(entry['mm1_b'],   trans_k.detach().cpu().float().abs().max().item())
        vk = torch.matmul(v_pad, trans_k)
        entry['mm1_out'] = max(entry['mm1_out'],  vk.detach().cpu().float().abs().max().item())
        entry['mm2_b']   = max(entry['mm2_b'],    q.detach().cpu().float().abs().max().item())
        out = torch.matmul(vk, q)
        entry['mm2_out'] = max(entry['mm2_out'],  out.detach().cpu().float().abs().max().item())
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + mod.eps)
        out = out.reshape(B, -1, H, W)
        return out
    return _hooked



# Core building blocks


def _set_module(root: nn.Module, dotted_name: str, new_mod: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


class QATConv2d(nn.Conv2d):
    """Conv2d with fixed-scale fake-quant in the forward pass (STE gradients).

        Activations are per-tensor symmetric INT8 by default (per_tensor=True): scale =
        amax/127, zp=0, no STE dead zones with ReLU6. per_tensor=False uses per-input-channel
        asymmetric INT8 with MSE-calibrated scale+zp per channel, better static accuracy but
        gradient dead zones are possible on ReLU6 networks. Weights are per-channel symmetric
        INT8.

        Scales stay fixed after calibration. During fine-tuning the weights adapt to minimise
        the loss on that grid. fake_quant_enabled=False runs a plain float Conv (warmup
        epochs, validation baseline), True the quantized forward with STE gradients (QAT).
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        act_scale: torch.Tensor,   # [C_in] float32 per-channel (or uniform for per_tensor)
        act_zp: torch.Tensor,      # [C_in] int32 (all zeros for per_tensor)
        w_scale: torch.Tensor,     # [C_out] float32 per-channel weight scale
        per_tensor: bool = True,   # True -> per-tensor symmetric (recommended for ReLU6)
    ) -> None:
        super().__init__(
            conv.in_channels, conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        self.weight.data.copy_(conv.weight.data)
        if self.bias is not None:
            self.bias.data.copy_(conv.bias.data)

        self.register_buffer("_act_scale", act_scale.float().clone())
        self.register_buffer("_act_zp",    act_zp.to(torch.int32).clone())
        self.register_buffer("_w_scale",   w_scale.float().clone())
        self.per_tensor: bool = per_tensor
        self.fake_quant_enabled: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fake_quant_enabled:
            if self.per_tensor:
                # Per-tensor symmetric: zp=0, scale = _act_scale[0].
                # STE gradient is always non-zero for |x| <= 127*scale.
                # Avoids the dead zone that collapses ReLU6 QAT to ~0.49 mIoU.
                x = torch.fake_quantize_per_tensor_affine(
                    x, float(self._act_scale[0]), 0, -128, 127
                )
            else:
                # Per-channel asymmetric (MSE-calibrated). Better static
                # accuracy but risks gradient dead zones for ReLU6 networks.
                x = torch.fake_quantize_per_channel_affine(
                    x, self._act_scale, self._act_zp, 1, -128, 127
                )
            w_zp = torch.zeros(self.out_channels, dtype=torch.int32,
                               device=self.weight.device)
            w = torch.fake_quantize_per_channel_affine(
                self.weight, self._w_scale, w_zp, 0, -128, 127
            )
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QATMatMul(nn.Module):
    """torch.matmul with per-tensor symmetric fake-quant on both inputs.

        Used for the two matrix multiplications in LiteMLA's relu_linear_att:
            mm1: vk  = matmul(v_padded, trans_k)   [B, heads, dim+1, dim]
            mm2: out = matmul(vk, q)               [B, heads, dim+1, HW]

        Scales are fixed after calibration, like QATConv2d. Per-tensor symmetric (zp=0)
        because TRT INT8 subgraphs need it. fake_quant_enabled=False is a plain matmul (warmup
        epochs, validation baseline), True the quantized matmul with STE gradients.
    """

    def __init__(
        self,
        a_scale:   float,   # per-tensor input-A activation scale
        b_scale:   float,   # per-tensor input-B activation scale
        out_scale: float,   # per-tensor output scale (for export output Q-DQ)
    ) -> None:
        super().__init__()
        self.register_buffer("_a_scale",   torch.tensor(a_scale,   dtype=torch.float32))
        self.register_buffer("_b_scale",   torch.tensor(b_scale,   dtype=torch.float32))
        self.register_buffer("_out_scale", torch.tensor(out_scale, dtype=torch.float32))
        # Always False: QATMatMul runs as plain float32 matmul during training.
        # Calibration scales are only used by export_qat_qdq_onnx for Q-DQ
        # injection at export time. Enabling fake_quant here distorts the
        # LiteMLA normalizer when S11-calibrated scales (large, 512-px sums) are
        # applied to training data at a different resolution or domain, rounding
        # the normalizer to 0 and causing NaN via division-by-epsilon.
        self.fake_quant_enabled: bool = False

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fake_quant_enabled:
            a = torch.fake_quantize_per_tensor_affine(
                a, float(self._a_scale), 0, -128, 127
            )
            b = torch.fake_quantize_per_tensor_affine(
                b, float(self._b_scale), 0, -128, 127
            )
        return torch.matmul(a, b)


class LiteMLAQAT(nn.Module):
    """LiteMLA wrapper that intercepts the two attention matmuls for QAT.

        Holds a reference to the original LiteMLA (qkv, aggreg, kernel_func, proj)
        and replaces the two torch.matmul calls inside relu_linear_att with
        QATMatMul instances that carry calibrated per-tensor scales.

        Stripped back to the plain LiteMLA by strip_qat() before ONNX export.
        export_qat_qdq_onnx() reads mm1/mm2 scales to inject Q-DQ around the
        MatMul nodes in the float ONNX graph.
    """

    def __init__(
        self,
        litemla: nn.Module,
        mm1: QATMatMul,   # vk = matmul(v_padded, trans_k)
        mm2: QATMatMul,   # out = matmul(vk, q)
    ) -> None:
        super().__init__()
        self.litemla = litemla
        self.mm1 = mm1
        self.mm2 = mm2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate LiteMLA.forward, substituting relu_linear_att with mm1/mm2
        qkv = self.litemla.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.litemla.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.litemla.dim:
            out = self._relu_linear_att_qat(qkv).to(qkv.dtype)
        else:
            out = self.litemla.relu_quadratic_att(qkv)
        return self.litemla.proj(out)

    def _relu_linear_att_qat(self, qkv: torch.Tensor) -> torch.Tensor:
        """relu_linear_att with QATMatMul replacing torch.matmul."""
        B, _, H, W = list(qkv.size())
        qkv_f = qkv.float() if qkv.dtype == torch.float16 else qkv
        qkv_r = qkv_f.reshape(B, -1, 3 * self.litemla.dim, H * W)
        q = qkv_r[:, :, :self.litemla.dim]
        k = qkv_r[:, :, self.litemla.dim : 2 * self.litemla.dim]
        v = qkv_r[:, :, 2 * self.litemla.dim :]
        q = self.litemla.kernel_func(q)
        k = self.litemla.kernel_func(k)
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk  = self.mm1(v, trans_k)
        out = self.mm2(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        # Clamp the normalizer before dividing. If QATConv2d quantization zeros out key channels
        # on OOD data (e.g. pseudo GDACS), vk_norm drops to 0. With clamp(min=1e-6) 0/0 gives 0,
        # while dividing by 1e-15 gives 0 for a zero numerator but 1e17+ for a non-zero one, and
        # the logits can then give a NaN softmax. 1e-6 is safe, real normalizers are much larger,
        # so the clamp never does anything on a normal (non-degenerate) forward.
        out = out[:, :, :-1] / out[:, :, -1:].clamp(min=1e-6)
        out = out.reshape(B, -1, H, W)
        return out



# Preparation


def prepare_qat(
    model: nn.Module,
    cal_batches: list,
    skip_last_n: int = 1,
    calibration_type: str = "per_tensor",
) -> nn.Module:
    """Calibrate scales and replace Conv2d with QATConv2d in a model copy.

    per_tensor uses amax/127 and zp=0, the ReLU6 Jetson path. The two per-channel
    activation modes are research controls; TensorRT deployment requires
    per-tensor activations. Asymmetric per-channel scales can create STE dead
    zones for ReLU6. Weights use max(abs(w))/127 per output channel.

    Calibration batches contain image tensors. skip_last_n leaves the final
    convolutions float. BN is retained during QAT and folded at export."""
    qat_model = copy.deepcopy(model)
    qat_model.eval()
    device = next(qat_model.parameters()).device

    all_convs = [(n, m) for n, m in qat_model.named_modules() if isinstance(m, nn.Conv2d)]
    to_quant  = all_convs[:-skip_last_n] if skip_last_n > 0 else all_convs

    n_layers = len(to_quant)

    # Per-channel activation sample buffers: list of dicts {channel_idx: [Tensor, ...]}
    act_ch_samples: list = [{} for _ in range(n_layers)]

    hooks = []
    for i, (name, _) in enumerate(to_quant):
        def _hook(m, inp, idx=i):
            x = inp[0].detach().cpu().float()  # [B, C, H, W]
            C = x.shape[1]
            for c in range(C):
                xc = x[:, c, :, :].reshape(-1)
                n = min(_SUBSAMPLE_PER_CH, xc.numel())
                perm = torch.randperm(xc.numel())[:n]
                if c not in act_ch_samples[idx]:
                    act_ch_samples[idx][c] = []
                act_ch_samples[idx][c].append(xc[perm])
        hooks.append(
            dict(qat_model.named_modules())[name].register_forward_pre_hook(_hook)
        )

    # LiteMLA matmul calibration: temporarily shadow relu_linear_att on each
    # LiteMLA instance to record abs-max of both matmul operands.
    # Instance attribute lookup takes precedence over the class method, so this
    # overrides relu_linear_att for this instance only without touching the class.
    litemla_calib: dict = {}   # name -> (mod, entry)
    for lm_name, lm_mod in qat_model.named_modules():
        if type(lm_mod).__name__ == 'LiteMLA':
            entry: dict = dict(mm1_a=0.0, mm1_b=0.0, mm1_out=0.0, mm2_b=0.0, mm2_out=0.0)
            litemla_calib[lm_name] = (lm_mod, entry)
            lm_mod.relu_linear_att = _make_litemla_calib_hook(entry, lm_mod)

    with torch.no_grad():
        for batch in cal_batches:
            qat_model(batch["image"].float().to(device))

    for h in hooks:
        h.remove()

    # Restore LiteMLA class methods (delete instance-level shadowing attribute)
    for _, (lm_mod, _) in litemla_calib.items():
        try:
            del lm_mod.relu_linear_att
        except AttributeError:
            pass  # relu_linear_att was never called (H*W <= dim) - no-op

    for i, (name, conv) in enumerate(to_quant):
        C_in = conv.in_channels
        ch_data = act_ch_samples[i]

        if calibration_type == "per_tensor":
            # all activations from all channels go into one symmetric scale.
            # zp=0: 0 is always representable, STE gradient never killed.
            # Only works well when all channels are ReLU6-bounded [0,6]; Hardswish
            # channels with larger range inflate the scale and starve other channels.
            all_samples = [torch.cat(v) for v in ch_data.values() if v]
            amax = torch.cat(all_samples).abs().max().item() if all_samples else 1.0
            scalar_scale = max(amax / 127.0, 1e-4)
            act_scales = torch.full((C_in,), scalar_scale, dtype=torch.float32)
            act_zps    = torch.zeros(C_in, dtype=torch.int32)
            per_tensor_flag = True

        elif calibration_type == "per_channel_symmetric":
            # Per-channel abs-max / 127, zp=0 for every channel.
            # Gives full 127-step resolution per channel AND no STE dead zone.
            # Not deployable on Jetson Xavier (TRT INT8 requires per-tensor act).
            act_scales = torch.zeros(C_in, dtype=torch.float32)
            act_zps    = torch.zeros(C_in, dtype=torch.int32)
            for c in range(C_in):
                if c not in ch_data or not ch_data[c]:
                    act_scales[c] = 1e-4
                else:
                    t = torch.cat(ch_data[c])
                    act_scales[c] = max(t.abs().max().item() / 127.0, 1e-4)
            per_tensor_flag = False  # use per-channel affine path (all zp=0)

        else:  # "per_channel" - asymmetric MSE
            # Per-channel MSE calibration with variable zp.
            # Risks STE dead zones (zp=-128) for ReLU6 channels.
            act_scales = torch.zeros(C_in, dtype=torch.float32)
            act_zps    = torch.zeros(C_in, dtype=torch.int32)
            for c in range(C_in):
                if c not in ch_data or not ch_data[c]:
                    act_scales[c] = 1e-4
                    act_zps[c]    = 0
                else:
                    t = torch.cat(ch_data[c])
                    s, z = _mse_scale_zp_valid(t)
                    act_scales[c] = float(s)
                    act_zps[c]    = int(z)
            per_tensor_flag = False

        w = conv.weight.data.float().view(conv.out_channels, -1)  # [C_out, -1]
        w_scale = w.abs().max(dim=1).values.clamp(min=1e-8) / 127.0
        _set_module(qat_model, name,
                    QATConv2d(conv, act_scales, act_zps, w_scale,
                              per_tensor=per_tensor_flag))

    _act_desc = {
        "per_tensor":            "per-tensor symmetric (deployment-ready)",
        "per_channel_symmetric": "per-channel symmetric (research only)",
        "per_channel":           "per-channel asymmetric (research only)",
    }[calibration_type]
    print(f"[QAT] {len(to_quant)} QATConv2d layers prepared "
          f"({_act_desc} act, skip_last_n={skip_last_n})")

    # largest activation scales point to layers with unbounded inputs
    # (e.g. layers after LiteMLA that bypass ReLU6). A large scale means a coarse INT8 grid for
    # the small-range channels in the same layer.
    layer_scales = []
    for name, mod in qat_model.named_modules():
        if isinstance(mod, QATConv2d):
            layer_scales.append((float(mod._act_scale[0]), name))
    layer_scales.sort(reverse=True)
    print("[QAT] top-10 act scales (layer, scale): high values = unbounded inputs")
    for sc, nm in layer_scales[:10]:
        print(f"  {sc:.4f}  {nm}")

    # Replace LiteMLA -> LiteMLAQAT with calibrated QATMatMul pairs.
    # mm1 covers: vk = matmul(v_padded, trans_k)
    # mm2 covers: out = matmul(vk, q) [mm2's a_scale = mm1's out_scale since vk is shared]
    for lm_name, (lm_mod, entry) in litemla_calib.items():
        a1 = max(entry['mm1_a'],   1e-4) / 127.0
        b1 = max(entry['mm1_b'],   1e-4) / 127.0
        o1 = max(entry['mm1_out'], 1e-4) / 127.0
        b2 = max(entry['mm2_b'],   1e-4) / 127.0
        o2 = max(entry['mm2_out'], 1e-4) / 127.0
        mm1 = QATMatMul(a1, b1, o1)
        mm2 = QATMatMul(o1, b2, o2)   # mm2 input-A scale = mm1 output scale (vk)
        _set_module(qat_model, lm_name, LiteMLAQAT(lm_mod, mm1, mm2))

    n_mla = len(litemla_calib)
    if n_mla:
        print(f"[QAT] {n_mla} LiteMLAQAT wrappers prepared ({n_mla * 2} QATMatMul pairs)")

    return qat_model


def set_fake_quant(model: nn.Module, enabled: bool) -> None:
    """Enable or disable fake-quant in QATConv2d layers.

        QATMatMul layers are intentionally excluded: they always run as plain float
        matmul during training (fake_quant_enabled=False).  Their stored scales are
        only read by export_qat_qdq_onnx to inject Q-DQ nodes at export time.
        Enabling fake_quant on QATMatMul during training causes the LiteMLA
        normalizer (vk_norm) to round to 0 when S11-calibrated scales (computed from
        large 512-px spatial sums) are applied to a different training domain or
        resolution, producing NaN loss via division-by-epsilon on 2499/2500 batches.
    """
    for m in model.modules():
        if isinstance(m, QATConv2d):
            m.fake_quant_enabled = enabled


def strip_qat(model: nn.Module) -> nn.Module:
    """Return a copy with all QATConv2d replaced by plain Conv2d,
        and all LiteMLAQAT wrappers replaced by their original LiteMLA.

        The QAT-trained float weights are preserved.  The result can be saved as a
        normal checkpoint and passed to ptq() for ONNX export.
    """
    m = copy.deepcopy(model)
    # Pass 1: strip QATConv2d first. Paths like foo.litemla.qkv.conv are still
    # valid here because LiteMLAQAT at foo hasn't been removed yet.
    for name, mod in list(m.named_modules()):
        if isinstance(mod, QATConv2d):
            plain = nn.Conv2d(
                mod.in_channels, mod.out_channels,
                kernel_size=mod.kernel_size, stride=mod.stride,
                padding=mod.padding, dilation=mod.dilation,
                groups=mod.groups, bias=mod.bias is not None,
                padding_mode=mod.padding_mode,
            )
            plain.weight.data.copy_(mod.weight.data)
            if mod.bias is not None:
                plain.bias.data.copy_(mod.bias.data)
            _set_module(m, name, plain)
    # Pass 2: unwrap LiteMLAQAT. QATConv2d inside litemla are already plain,
    # so the restored LiteMLA is a clean float module.
    for name, mod in list(m.named_modules()):
        if isinstance(mod, LiteMLAQAT):
            _set_module(m, name, mod.litemla)
    return m



# Training loop


def train_qat(
    model: nn.Module,
    train_dl,
    loss_fn: Callable,       # loss_fn(model, batch) -> Tensor
    evaluate_fn: Callable,   # evaluate_fn(model) -> {'mIoU': float, ...}
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 1,
    grad_clip: float = 1.0,
    scheduler: str = "cosine",
    val_every_n_epochs: int = 1,
    out_dir: str = "runs/qat",
) -> nn.Module:
    """Fine-tune with fixed fake-quant scales, selecting by validation mIoU.

    loss_fn(model, batch) returns a scalar; evaluate_fn(model) returns metrics
    including mIoU. Warmup disables fake quantization. A low learning rate keeps
    weights within the calibrated grid. The input model retains the best QAT
    weights for export; return its stripped float copy and save best_qat.pt."""
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)

    # Freeze BN gamma/beta so folded_bias can't flip sign. BN running stats are already frozen
    # during QAT (m.eval() in the training loop), but gamma and beta have requires_grad=True
    # and the optimizer would update them. If gamma grows, eff_scale = gamma/sqrt(var) grows
    # and folded_bias = -eff_scale*running_mean + beta can flip sign or grow. That gives large
    # negative residual offsets that cascade (ReLU6 doesn't clip them), the activation scales
    # blow up and TRT INT8 collapses.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                           nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)

    # Low weight decay -- don't regularise weights away from the INT8 grid.
    # Default 1e-5; anything > 1e-4 risks pushing weights outside the INT8 grid.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler == "cosine" else None
    )

    best_miou  = 0.0
    best_state = None
    ckpt_path  = os.path.join(out_dir, "best_qat.pt")

    for epoch in range(1, epochs + 1):
        in_warmup = epoch <= warmup_epochs
        set_fake_quant(model, not in_warmup)
        tag = "warmup" if in_warmup else "QAT"

        model.train()
        # BN stays in eval mode so the activation statistics match calibration, which ran with
        # model.eval() (running_mean/var). In train mode BN would use per-batch statistics that
        # differ from the running stats, which is really bad with batch_size=1 (very noisy),
        # and running_mean/var would drift and change eval behaviour between epochs. Normal
        # practice for fixed-scale QAT.
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.BatchNorm3d, nn.SyncBatchNorm)):
                m.eval()

        total_loss  = 0.0
        valid_steps = 0
        skipped     = 0
        for batch in train_dl:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            loss = loss_fn(model, batch)
            if not torch.isfinite(loss):
                # Sen1Floods11 has fully-masked images (all pixels = ignore_index).
                # F.cross_entropy(reduction='mean') returns 0/0 = NaN for those.
                # Calling backward on NaN corrupts all weights -- skip instead.
                skipped += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss  += loss.item()
            valid_steps += 1

        if sched is not None:
            sched.step()

        avg_loss = total_loss / max(valid_steps, 1)
        if skipped:
            print(f"[QAT] epoch {epoch:3d}  [{tag}]  skipped {skipped} all-masked batch(es)")

        if (epoch % val_every_n_epochs == 0) or (epoch == epochs):
            # fake-quant ON for INT8-simulated accuracy
            set_fake_quant(model, True)
            metrics = evaluate_fn(model)
            miou = metrics.get("mIoU", 0.0)
            print(f"[QAT] epoch {epoch:3d}/{epochs}  [{tag}]  "
                  f"loss={avg_loss:.4f}  val_mIoU={miou:.4f}")

            if miou > best_miou:
                best_miou  = miou
                best_state = copy.deepcopy(model.state_dict())
                stripped   = strip_qat(model)
                torch.save({"student": stripped.state_dict()}, ckpt_path)
                print(f"[QAT]   ^ best  mIoU={miou:.4f}  -> {ckpt_path}")
        else:
            print(f"[QAT] epoch {epoch:3d}/{epochs}  [{tag}]  loss={avg_loss:.4f}")

    # Restore best QATConv2d weights into model (needed for export_qat_qdq_onnx)
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[QAT] done  best_val_mIoU={best_miou:.4f}  checkpoint -> {ckpt_path}")
    return strip_qat(model)



# ONNX simplification helper


def _simplify_onnx(src: str, dst: str, input_shapes: dict | None = None) -> bool:
    """Run onnx-simplifier on src -> dst. Returns True on success.

        Required for jp50/TRT 8.4: LiteMLA computes dynamic padding as
        (key_dim - H*W % key_dim) % key_dim.  Even with static H/W, PyTorch's
        ONNX exporter emits a Shape->Gather->Mul->Mod chain feeding a Reshape with
        shape [-1, 2].  TRT 8.4 rejects any IShuffleLayer on a shape tensor with
        >1 dim.  onnxsim constant-folds 512*512 % 8 = 0, eliminating these nodes.
    """
    try:
        import onnxsim
        import onnx as _onnx
    except ImportError:
        print("[QAT export] WARNING: onnx-simplifier not installed -- "
              "model_qdq_trt.onnx may fail in TRT 8.4 due to LiteMLA shape ops.\n"
              "  Fix: pip install onnx-simplifier")
        return False

    model = _onnx.load(src)
    kwargs = {}
    if input_shapes:
        kwargs["input_shapes"] = input_shapes
    simplified, ok = onnxsim.simplify(model, **kwargs)
    if not ok:
        print("[QAT export] WARNING: onnxsim reported simplification failed; "
              "using unsimplified static ONNX")
        return False

    # TRT 8.4 does not support bicubic interpolation -- patch to linear.
    # EfficientViT's SegHead uses bicubic upsample; bilinear is nearly identical.
    cubic_count = 0
    for node in simplified.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "mode" and attr.s == b"cubic":
                attr.s = b"linear"
                cubic_count += 1
    if cubic_count:
        print(f"  onnxsim: patched {cubic_count} Resize node(s) cubic->linear "
              "(TRT 8.4 does not support bicubic)")

    _onnx.save(simplified, dst)
    before = os.path.getsize(src) // 1024
    after  = os.path.getsize(dst) // 1024
    print(f"  onnxsim: {src} ({before} KB) -> {dst} ({after} KB)  [nodes folded]")
    return True



# ONNX export preserving QAT calibration scales


def _pt_scale_from_ch(act_s_ch: "np.ndarray", act_zp_ch: "np.ndarray") -> float:
    """Convert per-channel asymmetric (scale, zp) to a per-tensor symmetric scale.

        Computes the symmetric range that covers the asymmetric range of every
        channel, then derives scale = max_abs / 127.  Used for TRT output Q-DQ
        (TRT 8.4 requires per-tensor symmetric activation quantization).
    """
    import numpy as np
    lo   = (-128.0 - act_zp_ch.astype(np.float64)) * act_s_ch.astype(np.float64)
    hi   = ( 127.0 - act_zp_ch.astype(np.float64)) * act_s_ch.astype(np.float64)
    amax = float(np.maximum(np.abs(lo), np.abs(hi)).max())
    return max(amax / 127.0, 1e-8)


def _calibrate_output_scales(
    folded_model: nn.Module,
    qat_named: list,
    cal_batches: list,
) -> list[float]:
    """max|output| of each QAT Conv in the BN-folded model.

        Forward hooks go on the BN-folded Conv layers (same order as qat_named) and
        cal_batches run through the model. The per-layer amax is the output Q-DQ scale for
        model_qdq_trt.onnx.

        That's the right scale when the post-activation range is much smaller than the
        pre-activation range, e.g. tanh (pre-tanh values typically in [-5, 5], post-tanh in
        (-1, 1)). The old approach used the next layer's input scale instead, which clips
        pre-activation values to the post-activation range and destroys the dynamic range for
        compressive activations like tanh.
    """
    folded_mods = {n: m for n, m in folded_model.named_modules()
                   if isinstance(m, nn.Conv2d)}
    amaxes = [0.0] * len(qat_named)
    hooks  = []
    for idx, (name, _) in enumerate(qat_named):
        if name not in folded_mods:
            continue
        def _hook(m, inp, out, i=idx):
            amaxes[i] = max(amaxes[i],
                            out.detach().cpu().float().abs().max().item())
        hooks.append(folded_mods[name].register_forward_hook(_hook))

    folded_model.eval()
    with torch.no_grad():
        for batch in cal_batches:
            folded_model(batch["image"].float().cpu())

    for h in hooks:
        h.remove()
    return amaxes


def export_qat_qdq_onnx(
    qat_model: nn.Module,
    dummy_input: torch.Tensor,
    out_dir: str,
    skip_last_n: int = 1,
    opset: int = 13,
    trt_compat: str = "jp50",
    cal_batches: list | None = None,
    output_qdq_scale: str = "calibrated",
    matmul_qdq: bool = False,
) -> tuple[str, str]:
    """Export the QAT model to ONNX with the fixed QAT calibration scales.

        No re-calibration: PTQ on the stripped QAT checkpoint would give different scales
        than the ones used in training and break the INT8 grid the weights adapted to. So the
        original _act_scale / _act_zp buffers are read from QATConv2d. Activation scales are
        converted from asymmetric to per-tensor symmetric for TRT 8.4 (no per-channel axis on
        QuantizeLinear). Weight scales are recomputed from the BN-folded weights, since BN
        folding changes the effective weights and the pre-fold _w_scale isn't valid anymore.

        trt_compat:
            "jp50" (default), JetPack 5.0 / TRT 8.4.x: float32 weights + QL + DQL bracket
                (avoids the trtexec segfault), fully static export + onnxsim (folds the LiteMLA
                shape ops), bicubic Resize patched to linear (TRT 8.4 has no cubic).
            "modern", TRT 8.5+ / JetPack 5.1+: INT8 initializer + DQL (compact, ~4x smaller
                ONNX weights), dynamic batch axis, no onnxsim step.

        cal_batches are used to measure the actual pre-activation output distribution of each
        Conv and are required for output_qdq_scale="calibrated". Use the same batches as for
        QAT preparation (training resolution, CPU tensors).

        output_qdq_scale:
            "calibrated" (default): output Q-DQ scale = max|Conv_output| / 127, measured by
                running cal_batches through the BN-folded model. Correct for any activation,
                tanh included (pre-activation range [-5, 5] is much larger than post-tanh
                [-1, 1]).
            "from_next_input": per-tensor equivalent of the next layer's input scale. Only
                safe for ReLU6 (pre- and post-activation ranges are similar), kept for backward
                compatibility. Don't use it with tanh or SiLU: it clips pre-activation values to
                the post-activation range, the model's dynamic range collapses and flood-class
                IoU goes to 0.

        Exported files:
            model_float.onnx            BN-folded float (ORT float accuracy baseline)
            model_float_trt_base.onnx   onnxsim-simplified float (benchmark_jetson.py
                                        builds FP32/FP16 engines from this)
            model_qdq_trt_no_out.onnx   INT8 QDQ for the Jetson: per-tensor activation QDQ,
                                        no Conv output QDQ. Output QDQ with a per-tensor scale
                                        crushes the minority-class (flood) channels, and
                                        TRT 8.4 doesn't need it.

        Returns qdq_no_out_path (model_qdq_trt_no_out.onnx).
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from quantization.onnx_qdq import export_float_onnx
    from quantization.manual import fold_bn

    os.makedirs(out_dir, exist_ok=True)

    qat_named = [(n, m) for n, m in qat_model.named_modules()
                 if isinstance(m, QATConv2d)]
    n_quant = len(qat_named)
    print(f"[QAT export] {n_quant} QATConv2d layers  (skip_last_n={skip_last_n})")

    # Collect per-tensor matmul scales from LiteMLAQAT wrappers.
    # Order: for each LiteMLAQAT in named_modules() DFS order, mm1 before mm2.
    # This matches the topological order of MatMul nodes in the ONNX graph.
    qat_litemlas = [(n, m) for n, m in qat_model.named_modules()
                    if isinstance(m, LiteMLAQAT)]
    # MatMul Q-DQ is opt-in (--matmul_qdq flag).
    # By default all attention MatMul stays FP16 - this matches the validated
    # deployment model (relu6_ofa_N5000_s91_qat_v3, 0.8713 ORT mIoU).
    # With --matmul_qdq, PTQ ORT calibration is run after float ONNX export and
    # Q-DQ is injected on mm1 inputs only (mm2 stays FP16, normalizer-safe).
    matmul_scales: list = []
    n_matmul = 0
    if qat_litemlas:
        if matmul_qdq:
            for _, qla in qat_litemlas:
                matmul_scales.append({
                    "a_scale": float(qla.mm1._a_scale),
                    "b_scale": float(qla.mm1._b_scale),
                })
                matmul_scales.append(None)   # mm2: skip, leave in FP16
            n_matmul = len(matmul_scales)
            print(f"[QAT export] {len(qat_litemlas)} LiteMLAQAT wrappers — "
                  f"mm1 MatMul will be INT8, mm2 stays FP16 (--matmul_qdq)")
        else:
            print(f"[QAT export] {len(qat_litemlas)} LiteMLAQAT wrappers — "
                  f"MatMul stays FP16 (add --matmul_qdq to enable INT8 mm1)")

    stripped = strip_qat(qat_model)
    stripped.eval().cpu()
    fold_bn(stripped)
    folded_by_name = {n: m for n, m in stripped.named_modules()
                      if isinstance(m, nn.Conv2d)}

    scales = []
    for name, qconv in qat_named:
        # After strip_qat pass 2, LiteMLAQAT at "...main" is replaced by its
        # .litemla attribute, removing the ".litemla." segment from all child paths.
        # qat_named names still contain ".litemla." from the pre-strip model.
        stripped_name = name.replace(".litemla.", ".")
        fconv = folded_by_name.get(stripped_name) or folded_by_name[name]

        # act scales from the QATConv2d buffers.
        # per_tensor=True: _act_scale is uniform [s,s,...,s] and _act_zp all zeros, so the
        # per-channel-to-symmetric conversion below gives act_s_sym_ch = 128*s/127 for every
        # channel, i.e. effectively per-tensor.
        act_s_ch  = qconv._act_scale.cpu().float().numpy()          # [C_in] float32
        act_zp_ch = qconv._act_zp.cpu().to(torch.float32).numpy()   # [C_in] int values as float

        # pre-fold w_scale is stale; recompute on BN-folded weights
        w = fconv.weight.data.float().view(fconv.out_channels, -1)
        w_scale = w.abs().max(dim=1).values.clamp(min=1e-8) / 127.0

        scales.append({
            "act_scale_ch": act_s_ch,
            "act_zp_ch":    act_zp_ch,
            "w_scale":      w_scale.cpu().numpy().astype(np.float32),
            "weight":       fconv.weight.data.cpu().numpy().astype(np.float32),
        })

    # Output Q-DQ scale (used only for model_qdq_trt.onnx).
    # Must represent the actual pre-activation Conv output range so that
    # QuantizeLinear does not clip values before the activation function runs.
    if output_qdq_scale == "calibrated" and cal_batches:
        output_amaxes = _calibrate_output_scales(stripped, qat_named, cal_batches)
        for i, amax in enumerate(scales):
            scales[i]["out_scale"] = max(output_amaxes[i] / 127.0, 1e-8)
        print(f"[QAT export] output_qdq_scale=calibrated  "
              f"(amax range: {min(output_amaxes):.3f}–{max(output_amaxes):.3f})")
    else:
        # Legacy: next layer's input scale as a proxy. Only safe for ReLU6 (pre- and
        # post-activation ranges match), it clips pre-tanh/pre-SiLU values, so don't use it
        # with compressive activations.
        for i in range(len(scales)):
            nxt = i + 1 if i + 1 < len(scales) else i
            scales[i]["out_scale"] = _pt_scale_from_ch(
                scales[nxt]["act_scale_ch"], scales[nxt]["act_zp_ch"]
            )
        if output_qdq_scale == "calibrated":
            print("[QAT export] WARNING: output_qdq_scale=calibrated but cal_batches=None; "
                  "falling back to from_next_input.  Pass cal_batches to fix.")
        else:
            print("[QAT export] output_qdq_scale=from_next_input  (legacy ReLU6 mode)")

    dummy = dummy_input[:1].float().cpu()
    float_path = os.path.join(out_dir, "model_float.onnx")
    export_float_onnx(stripped, dummy, float_path, opset, static=False)

    if trt_compat == "jp50":
        _float_static_tmp = os.path.join(out_dir, "_float_static_tmp.onnx")
        float_trt_path    = os.path.join(out_dir, "model_float_trt_base.onnx")
        export_float_onnx(stripped, dummy, _float_static_tmp, opset, static=True)
        _simplified = _simplify_onnx(_float_static_tmp, float_trt_path,
                                      input_shapes={"image": list(dummy.shape)})
        try:
            os.remove(_float_static_tmp)
        except OSError:
            pass
    else:
        float_trt_path = float_path
        _simplified    = True

    print(f"[QAT export] trt_compat={trt_compat!r}")

    # PTQ-style MatMul calibration: measure activation ranges on the exported float ONNX and
    # build a tensor name to scale dict for _inject_matmul_qdq. Only with matmul_qdq=True,
    # otherwise all MatMul stays FP16 (matmul_scales is empty).
    mm_scales_ort: dict | None = None
    if matmul_qdq and cal_batches:
        from quantization.onnx_qdq import _calibrate_matmul_scales as _cal_mm
        _mm_src = (float_trt_path
                   if (trt_compat == "jp50"
                       and os.path.exists(float_trt_path)
                       and os.path.getsize(float_trt_path) > 0)
                   else float_path)
        print(f"[QAT export] matmul_qdq: calibrating MatMul scales on {os.path.basename(_mm_src)!r}")
        mm_scales_ort = _cal_mm(_mm_src, cal_batches)
    elif matmul_qdq:
        print("[QAT export] WARNING: matmul_qdq=True but cal_batches=None; "
              "MatMul Q-DQ will not be injected (pass cal_batches to fix)")

    def _inject(out_path: str, with_output_qdq: bool, trt_base: bool = False) -> str:
        # ORT path (with_output_qdq=False, trt_base=False): per-channel axis=1 act Q-DQ, no
        # output Q-DQ, from float_path.
        # TRT path (with_output_qdq=True, trt_base=False): per-tensor act Q-DQ with output
        # Q-DQ, from float_trt_path.
        # TRT-no-out path (with_output_qdq=False, trt_base=True): per-tensor act Q-DQ, no
        # output Q-DQ, from float_trt_path. This is the right one for TRT 8.4 / Jetson Xavier.
        # Per-tensor output QDQ crushes the minority-class (flood) channels because the scale
        # is dominated by the majority class (background). TRT 8.4 doesn't need output QDQ,
        # it infers the Conv output precision from the next layer's input QDQ.
        per_tensor_act = with_output_qdq or trt_base
        _trt_base_exists = os.path.exists(float_trt_path) and os.path.getsize(float_trt_path) > 0
        src = float_trt_path if ((with_output_qdq or trt_base) and _trt_base_exists) else float_path
        model_onnx = onnx.load(src)
        graph = model_onnx.graph
        conv_nodes = [nd for nd in graph.node if nd.op_type == "Conv"]
        n_total = len(conv_nodes)
        float_init_map = {init.name: init for init in graph.initializer}

        all_new_inits:    list = []
        stale_init_names: set  = set()
        output_qdq_nodes: list = []
        output_remap:     dict = {}
        new_nodes:        list = []
        quant_idx   = 0
        matmul_idx  = 0

        for nd in graph.node:
            if nd.op_type == "MatMul" and matmul_idx < n_matmul and not matmul_qdq:
                ms = matmul_scales[matmul_idx]
                matmul_idx += 1
                if ms is None:
                    # mm2: skip. Its first input is vk (mm1 output, the linear-attention
                    # accumulator whose last channel is the normaliser). An INT8 scale of
                    # ~200-400 squashes the normaliser to 1-2 distinct values and attention
                    # collapses, so mm2 stays FP16.
                    new_nodes.append(nd)
                    continue

                tag = f"_m{matmul_idx - 1}"

                # Input A Q-DQ (per-tensor symmetric, no axis attribute for TRT 8.4)
                a_s  = np.array(ms["a_scale"],  dtype=np.float32)
                a_zp = np.array(0, dtype=np.int8)
                asn, azn = f"mm_a_scale{tag}", f"mm_a_zp{tag}"
                all_new_inits += [numpy_helper.from_array(a_s, asn),
                                  numpy_helper.from_array(a_zp, azn)]
                orig_a   = nd.input[0]
                ql_a_out = f"ql_mm_a{tag}"
                dq_a_out = f"dq_mm_a{tag}"
                new_nodes += [
                    helper.make_node("QuantizeLinear",   [orig_a,    asn, azn], [ql_a_out]),
                    helper.make_node("DequantizeLinear", [ql_a_out,  asn, azn], [dq_a_out]),
                ]
                nd.input[0] = dq_a_out

                # Input B Q-DQ (per-tensor symmetric, no axis attribute for TRT 8.4)
                b_s  = np.array(ms["b_scale"],  dtype=np.float32)
                b_zp = np.array(0, dtype=np.int8)
                bsn, bzn = f"mm_b_scale{tag}", f"mm_b_zp{tag}"
                all_new_inits += [numpy_helper.from_array(b_s, bsn),
                                  numpy_helper.from_array(b_zp, bzn)]
                orig_b   = nd.input[1]
                ql_b_out = f"ql_mm_b{tag}"
                dq_b_out = f"dq_mm_b{tag}"
                new_nodes += [
                    helper.make_node("QuantizeLinear",   [orig_b,    bsn, bzn], [ql_b_out]),
                    helper.make_node("DequantizeLinear", [ql_b_out,  bsn, bzn], [dq_b_out]),
                ]
                nd.input[1] = dq_b_out
                # No output Q-DQ on MatMul: unlike Conv, TRT does not require a
                # closed INT8 subgraph boundary here, and the mm1 output (vk) must
                # remain FP16 so mm2's normaliser channel is not step-quantised.

            elif nd.op_type == "Conv" and quant_idx < n_quant:
                s   = scales[quant_idx]
                tag = f"_c{quant_idx}"

                # Convert asymmetric per-channel (scale, zp) -> symmetric for ONNX
                # Formula: symmetric_scale_c = max(|lo_c|, |hi_c|) / 127
                act_s_raw  = s["act_scale_ch"].astype(np.float64)
                act_zp_raw = s["act_zp_ch"].astype(np.float64)
                lo   = (-128.0 - act_zp_raw) * act_s_raw
                hi   = ( 127.0 - act_zp_raw) * act_s_raw
                amax = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1e-8)
                act_s_sym_ch = (amax / 127.0).astype(np.float32)  # [C_in] symmetric

                if per_tensor_act:
                    # TRT: collapse to per-tensor max
                    act_s  = np.array(act_s_sym_ch.max(), dtype=np.float32)
                    act_zp = np.array(0, dtype=np.int8)
                    ql_kwargs  = {}
                    dql_kwargs = {}
                else:
                    # ORT: per-channel axis=1 symmetric
                    act_s  = act_s_sym_ch                               # [C_in]
                    act_zp = np.zeros(act_s.shape, dtype=np.int8)        # [C_in]
                    ql_kwargs  = {"axis": 1}
                    dql_kwargs = {"axis": 1}

                asc_n, azp_n = f"act_scale{tag}", f"act_zp{tag}"
                all_new_inits += [numpy_helper.from_array(act_s, asc_n),
                                  numpy_helper.from_array(act_zp, azp_n)]
                orig_x   = nd.input[0]
                ql_x_out = f"ql_x{tag}"
                dq_x_out = f"dq_x{tag}"
                new_nodes += [
                    helper.make_node("QuantizeLinear",
                                     [orig_x, asc_n, azp_n], [ql_x_out], **ql_kwargs),
                    helper.make_node("DequantizeLinear",
                                     [ql_x_out, asc_n, azp_n], [dq_x_out], **dql_kwargs),
                ]
                nd.input[0] = dq_x_out

                w_name_orig = nd.input[1]
                w_s   = s["w_scale"]
                w_zp  = np.zeros(w_s.shape, dtype=np.int8)
                wsc_n = f"w_scale{tag}"
                wzp_n = f"w_zp{tag}"
                dq_w_out = f"dq_w{tag}"

                if trt_compat == "jp50":
                    w_f  = s["weight"].astype(np.float32)
                    wf_n = f"w_float{tag}"
                    ql_w_out = f"ql_w{tag}"
                    all_new_inits += [
                        numpy_helper.from_array(w_f,  wf_n),
                        numpy_helper.from_array(w_s,  wsc_n),
                        numpy_helper.from_array(w_zp, wzp_n),
                    ]
                    new_nodes += [
                        helper.make_node("QuantizeLinear",
                                         [wf_n, wsc_n, wzp_n], [ql_w_out], axis=0),
                        helper.make_node("DequantizeLinear",
                                         [ql_w_out, wsc_n, wzp_n], [dq_w_out], axis=0),
                    ]
                else:
                    w_int8 = np.clip(
                        np.round(s["weight"].astype(np.float32)
                                 / w_s[:, None, None, None]),
                        -128, 127,
                    ).astype(np.int8)
                    wi_n = f"w_int8{tag}"
                    all_new_inits += [
                        numpy_helper.from_array(w_int8, wi_n),
                        numpy_helper.from_array(w_s,    wsc_n),
                        numpy_helper.from_array(w_zp,   wzp_n),
                    ]
                    new_nodes += [
                        helper.make_node("DequantizeLinear",
                                         [wi_n, wsc_n, wzp_n], [dq_w_out], axis=0),
                    ]

                nd.input[1] = dq_w_out
                if w_name_orig in float_init_map:
                    stale_init_names.add(w_name_orig)

                if with_output_qdq:
                    out_s  = np.array(s["out_scale"], dtype=np.float32)
                    out_zp = np.array(0, dtype=np.int8)
                    old_out  = nd.output[0]
                    raw_out  = f"conv_raw{tag}"
                    ql_out   = f"ql_out{tag}"
                    dql_out  = f"dql_out{tag}"
                    nd.output[0] = raw_out
                    output_remap[old_out] = dql_out
                    osc_n, ozp_n = f"out_scale{tag}", f"out_zp{tag}"
                    all_new_inits += [numpy_helper.from_array(out_s, osc_n),
                                      numpy_helper.from_array(out_zp, ozp_n)]
                    output_qdq_nodes += [
                        helper.make_node("QuantizeLinear",
                                         [raw_out, osc_n, ozp_n], [ql_out]),
                        helper.make_node("DequantizeLinear",
                                         [ql_out, osc_n, ozp_n], [dql_out]),
                    ]

                quant_idx += 1
            new_nodes.append(nd)

        for nd in new_nodes + output_qdq_nodes:
            for j, inp in enumerate(nd.input):
                if inp in output_remap:
                    nd.input[j] = output_remap[inp]

        del graph.node[:]
        graph.node.extend(new_nodes)
        graph.node.extend(output_qdq_nodes)

        surviving = [it for it in graph.initializer if it.name not in stale_init_names]
        del graph.initializer[:]
        graph.initializer.extend(surviving)
        graph.initializer.extend(all_new_inits)

        # PTQ MatMul Q-DQ (TRT path only, activated by matmul_qdq=True).
        # Uses tensor-name matching from ORT calibration; mutually exclusive with
        # the QAT-buffer injection above (which is gated by `not matmul_qdq`).
        n_mm_ptq = 0
        if with_output_qdq and mm_scales_ort:
            from quantization.onnx_qdq import _inject_matmul_qdq
            mm_inits = _inject_matmul_qdq(graph, mm_scales_ort)
            graph.initializer.extend(mm_inits)
            n_mm_ptq = len(mm_inits) // 2

        onnx.save(model_onnx, out_path)
        size_kb = os.path.getsize(out_path) // 1024
        n_mm_quantised = sum(1 for ms in matmul_scales if ms is not None) if matmul_scales else 0
        if matmul_qdq and n_mm_ptq:
            mm_info = f"  PTQ MatMul QDQ: {n_mm_ptq} input tensors"
        elif n_mm_quantised and not matmul_qdq:
            mm_info = f"  {n_mm_quantised} mm1 MatMul quantised ({n_matmul - n_mm_quantised} mm2 FP16)"
        else:
            mm_info = ""
        print(f"  {out_path}  ({size_kb} KB)  {quant_idx}/{n_total} Conv quantised{mm_info}")
        return out_path

    qdq_no_out_path = os.path.join(out_dir, "model_qdq_trt_no_out.onnx")
    # Per-tensor activation QDQ (TRT 8.4 compatible), no Conv output QDQ. Conv output QDQ with
    # a per-tensor scale makes the minority-class channels collapse (flood_IoU drops to 0)
    # because background dominates the scale. TRT 8.4 infers the Conv output precision from
    # the next layer's input QDQ, so output QDQ isn't needed.
    _inject(qdq_no_out_path, with_output_qdq=False, trt_base=True)

    print(f"[QAT export] ONNX files saved to {out_dir}/")
    return qdq_no_out_path
