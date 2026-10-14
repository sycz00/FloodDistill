"""Is the QAT gradient explosion a float32 rounding artifact of fake-quant simulation?

Hypothesis: with quantized (integer x scale) inputs and weights, pre-activations that are exactly 0 in integer
arithmetic come out as ~1e-9 rounding residue in float32, ReLU passes the positive residue, the attention
denominator becomes ~1e-7 and the 1/D gradient amplification appears.
Test: integer-domain fake-quant -- conv(x_int, w_int) (exact for integers < 2^24), THEN multiply by the scales
(= INT32 accumulate + requantize, i.e. the deployed arithmetic). Same STE masks. Compare logits and gradients.
"""
import sys, json, types, torch
import torch.nn.functional as F
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, compression_train_mode, quantization_mode, FixedQATConv2d, _range
from data.pseudo_labeled import PseudoLabeledDataset
from training.losses import focal_loss
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=True, geom_aug=True)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:16].tolist()
b = [ds[i] for i in idx]
x = torch.stack([t["image"] for t in b]).float(); y = torch.stack([t["mask"] for t in b]).long()

orig_forward = FixedQATConv2d.forward
def int_forward(self, inp):
    if not self.fake_quant_enabled:
        return self._conv_forward(inp, self.weight, self.bias)
    sa = float(self._act_scale[0]); amin, amax = _range(self.activation_bits); wmin, wmax = _range(self.weight_bits)
    xs = inp / sa; xr = torch.round(xs.detach())
    xm = ((xr >= amin) & (xr <= amax)).to(inp.dtype)                       # STE pass-through mask (as fake_quantize)
    x_ste = torch.clamp(xr, amin, amax) + (xs - xs.detach()) * xm
    sw = self._w_scale.view(-1, 1, 1, 1); ws = self.weight / sw; wr = torch.round(ws.detach())
    wm = ((wr >= wmin) & (wr <= wmax)).to(self.weight.dtype)
    w_ste = torch.clamp(wr, wmin, wmax) + (ws - ws.detach()) * wm
    out = F.conv2d(x_ste, w_ste, None, self.stride, self.padding, self.dilation, self.groups)   # exact on integers
    out = out * (sa * self._w_scale.view(1, -1, 1, 1))
    if self.bias is not None: out = out + self.bias.view(1, -1, 1, 1)
    return out

def gradnorm(m):
    m.zero_grad(set_to_none=True)
    out = m(x); loss = focal_loss(out, y); loss.backward()
    return float(sum(p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None) ** .5), float(loss), out.detach()

for cell, wb in (("A3_W4A4_B1_s42", "W4A4"), ("A3_W4A8_B1_s42", "W4A8"), ("A3_W8A8_B1_s42", "W8A8")):
    ck = torch.load(f"{D}/{cell}/best.pt", map_location="cpu", weights_only=False)["model"]
    m = load_compression_qat(build_compression_student("A3", pretrained=False), ck)
    compression_train_mode(m, enabled=True)
    FixedQATConv2d.forward = orig_forward
    n0, l0, o0 = gradnorm(m)
    FixedQATConv2d.forward = int_forward
    n1, l1, o1 = gradnorm(m)
    FixedQATConv2d.forward = orig_forward
    with torch.no_grad(): 
        agree = float((o0.argmax(1) == o1.argmax(1)).float().mean())
    print(f"{wb} ({cell}): float32 fake-quant  grad_norm={n0:.4g} loss={l0:.4f}   |   integer-domain  grad_norm={n1:.4g} loss={l1:.4f}   "
          f"max|logit diff|={float((o0-o1).abs().max()):.3g}  argmax agreement={100*agree:.3f}%")
