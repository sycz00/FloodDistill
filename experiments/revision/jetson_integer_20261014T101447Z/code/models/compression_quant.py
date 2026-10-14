"""Fixed-scale low-bit controls built on the project's historical QAT code.

These are fake-quant simulations: the master weights, convolution arithmetic,
BN, bias, final logits and LiteMLA MatMuls remain float32. Four-bit support here
does not assert an executable four-bit TensorRT kernel.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from quantization.qat import (
    QATConv2d, QATMatMul, LiteMLAQAT, prepare_qat, strip_qat, _set_module,
)

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _range(bits: int) -> tuple[int, int]:
    if bits not in (2, 4, 8):
        raise ValueError("Integer precision must be 2, 4 or 8 bits")
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


class FixedQATConv2d(QATConv2d):
    """Historical per-tensor input/per-output-channel weight STE at fixed bits."""

    def __init__(self, conv, act_scale, act_zp, w_scale, *, weight_bits=8,
                 activation_bits=8):
        _range(weight_bits)
        if activation_bits not in (4, 8, 16):
            raise ValueError("Activation precision must be 4, 8 or 16 bits")
        super().__init__(conv, act_scale, act_zp, w_scale, per_tensor=True)
        self.register_buffer("_weight_bits", torch.tensor(weight_bits, dtype=torch.int64))
        self.register_buffer("_activation_bits", torch.tensor(activation_bits, dtype=torch.int64))
        self.register_buffer("_compression_enabled", torch.tensor(True))
        self._weight_bits_value = int(weight_bits)
        self._activation_bits_value = int(activation_bits)
        self._quant_enabled_value = True
        self._activation_scale_value = float(self._act_scale[0])
        self.to(device=conv.weight.device)

    @property
    def fake_quant_enabled(self):
        if "_compression_enabled" in getattr(self, "_buffers", {}):
            return self._quant_enabled_value
        return getattr(self, "_initial_enabled", True)

    @fake_quant_enabled.setter
    def fake_quant_enabled(self, enabled):
        if "_compression_enabled" in getattr(self, "_buffers", {}):
            self._compression_enabled.fill_(bool(enabled))
            self._quant_enabled_value = bool(enabled)
        else:
            self._initial_enabled = bool(enabled)

    @property
    def weight_bits(self):
        return self._weight_bits_value

    @property
    def activation_bits(self):
        return self._activation_bits_value

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._weight_bits_value = int(self._weight_bits.item())
        self._activation_bits_value = int(self._activation_bits.item())
        self._quant_enabled_value = bool(self._compression_enabled.item())
        self._activation_scale_value = float(self._act_scale[0])

    def forward(self, x):
        if self.fake_quant_enabled:
            if self.activation_bits == 16:
                # Explicit weight-only diagnostic with IEEE fp16 input rounding,
                # followed by float32 convolution; not a deployed fp16 operator.
                x = x.to(torch.float16).to(x.dtype)
            else:
                amin, amax = _range(self.activation_bits)
                x = torch.fake_quantize_per_tensor_affine(
                    x, self._activation_scale_value, 0, amin, amax)
            wmin, wmax = _range(self.weight_bits)
            w = torch.fake_quantize_per_channel_affine(
                self.weight, self._w_scale,
                torch.zeros(self.out_channels, dtype=torch.int32, device=self.weight.device),
                0, wmin, wmax)
        else:
            w = self.weight
        return self._conv_forward(x, w, self.bias)


def freeze_batch_norm(model: nn.Module) -> None:
    """Freeze running statistics AND affine parameters, including affine=False BN."""
    for module in model.modules():
        if isinstance(module, _BN):
            module.eval()
            for param in module.parameters(recurse=False):
                param.requires_grad_(False)


def compression_train_mode(model: nn.Module, enabled: bool = True) -> None:
    """Call after each training-mode transition; ordinary model.train unfreezes BN."""
    model.train()
    freeze_batch_norm(model)
    for module in model.modules():
        if isinstance(module, FixedQATConv2d):
            module.fake_quant_enabled = enabled


@contextmanager
def quantization_mode(model: nn.Module, enabled: bool, *, measurement: bool = True):
    """Share master weights and restore every mode/quant/observer flag on exit.

    Measurement disables dropout and observer updates. It does not implicitly
    disable autograd: callers collecting statistics should also use no_grad.
    No observers exist in this fixed-scale family after preparation.
    """
    modules = list(model.modules())
    modes = [(m, m.training) for m in modules]
    flags = [(m, m.fake_quant_enabled) for m in modules
             if isinstance(m, (QATConv2d, QATMatMul))]
    observers = [(m, m.observer_enabled.detach().clone()) for m in modules
                 if isinstance(getattr(m, "observer_enabled", None), torch.Tensor)]
    try:
        if measurement:
            model.eval()
            for m, _ in observers:
                m.observer_enabled.zero_()
        for m, _ in flags:
            # MatMuls are always float in the existing simulation family.
            m.fake_quant_enabled = bool(enabled) if isinstance(m, QATConv2d) else False
        yield model
    finally:
        for m, flag in flags:
            m.fake_quant_enabled = flag
        for m, flag in observers:
            m.observer_enabled.copy_(flag)
        # Direct assignment preserves deliberately mixed parent/child modes.
        for m, mode in modes:
            m.training = mode


def rebind_compression_bits(model: nn.Module, weight_bits: int, activation_bits: int,
                            *, inplace: bool = False) -> nn.Module:
    """Re-bin identical observed ranges without recalibration or weight changes."""
    _, wmax = _range(weight_bits)
    if activation_bits not in (4, 8, 16):
        raise ValueError("Activation precision must be 4, 8 or 16 bits")
    target = model if inplace else copy.deepcopy(model)
    for m in target.modules():
        if isinstance(m, FixedQATConv2d):
            _, old_wmax = _range(m.weight_bits)
            m._w_scale.mul_(old_wmax / wmax)
            old_amax = _range(m.activation_bits)[1] if m.activation_bits != 16 else 127
            new_amax = _range(activation_bits)[1] if activation_bits != 16 else 127
            m._act_scale.mul_(old_amax / new_amax)
            m._weight_bits.fill_(weight_bits)
            m._activation_bits.fill_(activation_bits)
            m._weight_bits_value = int(weight_bits)
            m._activation_bits_value = int(activation_bits)
            m._activation_scale_value = float(m._act_scale[0])
    return target


def prepare_compression_qat(model: nn.Module, cal_batches: Iterable, *,
                            weight_bits: int = 8, activation_bits: int = 8,
                            calibration_seed: int = 1729) -> nn.Module:
    """Use historical S11-image calibration, isolated RNG, and a float logits Conv.

    Supply the fixed allowed S11 validation IMAGE batches (no labels are read).
    Calibration IDs/geometry belong to the experiment manifest. Historical
    calibration uses at most 500 random input values/channel/batch, symmetric
    per-tensor inputs and per-output-channel weights. All scales stay fixed.
    """
    device = next(model.parameters()).device
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("The matched QAT family requires float32 master parameters")
    if any(isinstance(m, QATConv2d) for m in model.modules()):
        raise ValueError("Expected a common unquantized FP checkpoint")
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if not convs or convs[-1][1].out_channels != 2:
        raise ValueError("Last registered convolution must be the two-class float logits head")
    images = [{"image": b["image"] if isinstance(b, Mapping) else b}
              for b in cal_batches]
    if not images:
        raise ValueError("Calibration image pool is empty")
    # Existing preparation consumes only CPU random numbers. Avoid changing the
    # caller's data/model RNG while reproducing the historical sampler.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(calibration_seed)
        prepared = prepare_qat(model, images, skip_last_n=1, calibration_type="per_tensor")
        for name, m in list(prepared.named_modules()):
            if isinstance(m, QATConv2d):
                replacement = FixedQATConv2d(m, m._act_scale, m._act_zp, m._w_scale)
                _set_module(prepared, name, replacement)
    prepared.to(device)
    rebind_compression_bits(prepared, weight_bits, activation_bits, inplace=True)
    freeze_batch_norm(prepared)
    prepared.eval()
    return prepared


def load_compression_qat(model: nn.Module, state: Mapping, *, inplace: bool = False) -> nn.Module:
    """Reconstruct all quantizers from a complete state_dict, without recalibration."""
    target = model if inplace else copy.deepcopy(model)
    if any(isinstance(m, QATConv2d) for m in target.modules()):
        raise ValueError("Reload expects an unquantized graph")
    # Existing LiteMLAQAT adds '.litemla' to child paths; recreate wrappers first.
    for name, mod in list(target.named_modules()):
        key = name + ".mm1._a_scale"
        if type(mod).__name__ == "LiteMLA" and key in state:
            def mm(which):
                stem = name + "." + which
                return QATMatMul(float(state[stem + "._a_scale"]),
                                 float(state[stem + "._b_scale"]),
                                 float(state[stem + "._out_scale"]))
            _set_module(target, name, LiteMLAQAT(mod, mm("mm1"), mm("mm2")))
    with torch.random.fork_rng(devices=[]):
        for name, mod in list(target.named_modules()):
            stem = name + "."
            if isinstance(mod, nn.Conv2d) and stem + "_weight_bits" in state:
                _set_module(target, name, FixedQATConv2d(
                    mod, state[stem + "_act_scale"], state[stem + "_act_zp"],
                    state[stem + "_w_scale"], weight_bits=int(state[stem + "_weight_bits"]),
                    activation_bits=int(state[stem + "_activation_bits"])))
    target.load_state_dict(state, strict=True)
    freeze_batch_norm(target)
    return target


def strip_compression_qat(model: nn.Module) -> nn.Module:
    """Remove quantizers and wrappers; these QAT-updated weights are NOT B0."""
    return strip_qat(model)


def precision_map(model: nn.Module) -> dict:
    """Machine-readable actual simulation precisions, grids, and FP exceptions."""
    operators = []
    has_quantizers = any(isinstance(m, FixedQATConv2d) for m in model.modules())
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    last_conv = conv_names[-1] if conv_names else None
    for name, m in model.named_modules():
        if isinstance(m, FixedQATConv2d):
            amin, amax = _range(m.activation_bits) if m.activation_bits != 16 else (None, None)
            wmin, wmax = _range(m.weight_bits)
            operators.append(dict(
                name=name, op="Conv2d", arithmetic="float32", weight_master="float32",
                weight_bits=m.weight_bits, activation_bits=m.activation_bits,
                weight_qmin=wmin, weight_qmax=wmax, activation_qmin=amin, activation_qmax=amax,
                signed=True, zero_points={"weight": 0, "activation": 0},
                weight_granularity="per_output_channel_axis0", activation_granularity="per_tensor",
                weight_scales=m._w_scale.detach().cpu().tolist(),
                activation_scale=float(m._act_scale[0]), scales_learned=False,
                rounding="nearest_even", fake_quant_enabled=m.fake_quant_enabled,
                bias="float32" if m.bias is not None else "absent",
                activation_16_semantics="float16_roundtrip_then_float32_conv" if m.activation_bits == 16 else None))
        elif isinstance(m, nn.Conv2d):
            operators.append(dict(name=name, op="Conv2d", arithmetic="float32",
                                  weight_bits=32, activation_bits=32,
                                  exception=("unquantized_final_logits" if has_quantizers and name == last_conv
                                             else "unquantized_float_graph_operator"),
                                  bias="float32" if m.bias is not None else "absent"))
        elif isinstance(m, QATMatMul):
            operators.append(dict(name=name, op="MatMul", arithmetic="float32",
                                  fake_quant_enabled=False, exception="historical_float_attention_simulation"))
        elif isinstance(m, _BN):
            operators.append(dict(name=name, op=type(m).__name__, arithmetic="float32",
                                  running_frozen=not m.training,
                                  affine_frozen=all(not p.requires_grad for p in m.parameters(recurse=False))))
    return dict(status="simulation_only" if has_quantizers else "floating_reference",
                backend="PyTorch_native_fake_quant_STE" if has_quantizers else "PyTorch_float32",
                arithmetic="float32", device=str(next(model.parameters()).device),
                calibration=("historical_per_tensor_symmetric_500_samples_per_input_channel_per_batch"
                             if has_quantizers else "not_applicable_unquantized_graph"),
                observers_after_preparation=False, scales_fixed=True if has_quantizers else None,
                other_ops="float32 including activation, interpolation, residuals, attention normalization",
                operators=operators)
