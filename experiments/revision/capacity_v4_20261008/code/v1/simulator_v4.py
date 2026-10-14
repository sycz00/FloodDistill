"""Opt-in integer-code convolution and denominator controls; no legacy edits."""
from contextlib import contextmanager
import types
import torch
from torch.nn import functional as F
from models.compression_quant import FixedQATConv2d, _range
from quantization.qat import LiteMLAQAT


def integer_forward(self, x):
    if not self.fake_quant_enabled:
        return self._conv_forward(x, self.weight, self.bias)
    if self.activation_bits not in (4, 8):
        raise ValueError('v4 integer simulator supports A4/A8 only')
    sa = self._activation_scale_value
    amin, amax = _range(self.activation_bits)
    wmin, wmax = _range(self.weight_bits)
    xs = x / sa
    xr = xs.detach().round()
    codes_x = xr.clamp(amin, amax) + (xs - xs.detach()) * ((xr >= amin) & (xr <= amax))
    sw = self._w_scale.reshape(-1, 1, 1, 1)
    ws = self.weight / sw
    wr = ws.detach().round()
    codes_w = wr.clamp(wmin, wmax) + (ws - ws.detach()) * ((wr >= wmin) & (wr <= wmax))
    fanin = self.weight[0].numel()
    bound = fanin * max(abs(amin), abs(amax)) * max(abs(wmin), abs(wmax))
    # CUDA TF32 is disabled by the caller; switch to double whenever even the
    # conservative sum-of-absolute-products bound exceeds exact FP32 integers.
    dtype = torch.float64 if bound >= 2**24 else torch.float32
    out = F.conv2d(codes_x.to(dtype), codes_w.to(dtype), None,
                   self.stride, self.padding, self.dilation, self.groups)
    out = out * (sa * self._w_scale.reshape(1, -1, 1, 1).to(dtype))
    out = out.to(x.dtype)
    if self.bias is not None:
        out = out + self.bias.reshape(1, -1, 1, 1)
    return out


def floor_forward(tau, backward_only):
    def forward(self, qkv):
        b, _, h, w = qkv.shape
        dim = self.litemla.dim
        z = qkv.reshape(b, -1, 3*dim, h*w)
        q = self.litemla.kernel_func(z[:, :, :dim])
        k = self.litemla.kernel_func(z[:, :, dim:2*dim])
        v = F.pad(z[:, :, 2*dim:], (0, 0, 0, 1), value=1)
        prod = self.mm2(self.mm1(v, k.transpose(-1, -2)), q)
        numerator, denominator = prod[:, :, :-1], prod[:, :, -1:]
        original = numerator / denominator.clamp(min=1e-6)
        floored = numerator / denominator.clamp(min=tau)
        value = original.detach() + (floored - floored.detach()) if backward_only else floored
        return value.reshape(b, -1, h, w)
    return forward


@contextmanager
def simulator(model, kind='integer', tau=None):
    saved=[]
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        for module in model.modules():
            if kind == 'integer' and isinstance(module, FixedQATConv2d):
                saved.append((module, 'forward' in module.__dict__, module.forward))
                module.forward = types.MethodType(integer_forward, module)
            elif kind in ('floor', 'backward_floor') and isinstance(module, LiteMLAQAT):
                key='_relu_linear_att_qat'
                saved.append((module, key in module.__dict__, module._relu_linear_att_qat))
                module._relu_linear_att_qat = types.MethodType(floor_forward(tau, kind=='backward_floor'),module)
        yield model
    finally:
        for module, local, fn in saved:
            key = 'forward' if isinstance(module, FixedQATConv2d) else '_relu_linear_att_qat'
            if local:setattr(module,key,fn)
            else:delattr(module,key)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
