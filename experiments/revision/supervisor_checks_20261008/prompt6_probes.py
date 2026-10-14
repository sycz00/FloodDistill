"""Supervisor probes for the claims in prompt6.md (read-only; GPU used for S11-val inference only).

Part 1  attention-denominator collapse: quantization bypassed vs enabled, identical weights/inputs.
Part 2  fixed-checkpoint PTQ decomposition of the A3 W4A4 damage on the 89 S11-validation images:
        weights vs activations, stem activation precision, unsigned grids at non-negative sites.
Nothing is trained; no project file is modified.
"""
import sys, json, types, copy, torch
import torch.nn.functional as F
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import (load_compression_qat, compression_train_mode, quantization_mode,
                                       FixedQATConv2d, _range)
from quantization import qat as Q
from data.pseudo_labeled import PseudoLabeledDataset
import scripts.compression_jobs as cj

R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))

# ------------------------------------------------------------------ Part 1
def load_payload(cell):
    return torch.load(R + f"results/{cell}_interaction.pt", map_location="cpu", weights_only=True)

def build(cell):
    p = load_payload(cell)
    return load_compression_qat(build_compression_student("A3", pretrained=False), p["model"])

ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=True, geom_aug=True)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:16].tolist()
x = torch.stack([ds[i]["image"] for i in idx]).float()

stats = {}
orig = Q.LiteMLAQAT._relu_linear_att_qat
def make_patch(tag):
    def patched(self, qkv):
        B, _, H, W = list(qkv.size()); dim = self.litemla.dim
        qkv_r = qkv.reshape(B, -1, 3 * dim, H * W)
        q = self.litemla.kernel_func(qkv_r[:, :, :dim]); k = self.litemla.kernel_func(qkv_r[:, :, dim:2 * dim])
        v = F.pad(qkv_r[:, :, 2 * dim:], (0, 0, 0, 1), value=1)
        out = self.mm2(self.mm1(v, k.transpose(-1, -2)), q)
        stats.setdefault(tag, []).append(out[:, :, -1:].detach().clone())   # denominator D
        return orig(self, qkv)
    return patched

def collect(model, enabled, tag):
    stats.pop(tag, None)
    Q.LiteMLAQAT._relu_linear_att_qat = make_patch(tag)
    with torch.no_grad(), quantization_mode(model, enabled):
        model(x)
    Q.LiteMLAQAT._relu_linear_att_qat = orig

print("PART 1 - attention denominators D, same weights/inputs, quantization OFF (FP bypass) vs ON")
for cell, label in (("A3_W4A4", "W4A4 PTQ start"), ("A3_W4A8", "W4A8 PTQ start"), ("A3_W8A8", "W8A8 PTQ start")):
    m = build(cell); m.eval()
    collect(m, False, "F"); collect(m, True, "Q")
    print(f"\n  {label} (A3, attention dim 4; 2 attention blocks)")
    for b, (DF, DQ) in enumerate(zip(stats["F"], stats["Q"])):
        pos = DF[DF > 0]
        tau = max(float(torch.quantile(pos.flatten()[:2_000_000], 0.01)), 1e-6) if pos.numel() else float("nan")
        coll = float(((DF > tau) & (DQ < DF / 16)).float().mean())
        print(f"    block {b}: FP  frac(D==0)={float((DF==0).float().mean()):.4f} frac(D<1e-6)={float((DF<1e-6).float().mean()):.4f} frac(D<1e-3)={float((DF<1e-3).float().mean()):.4f}"
              f" | Q  frac(D==0)={float((DQ==0).float().mean()):.4f} frac(D<1e-6)={float((DQ<1e-6).float().mean()):.4f} frac(D<1e-3)={float((DQ<1e-3).float().mean()):.4f}"
              f" | tau(1%ile of FP D>0)={tau:.3g}  C=P[D_F>tau & D_Q<D_F/16]={coll:.4f}")

# ------------------------------------------------------------------ Part 2
def set_bits(m, wb, ab):
    _, old_w = _range(m.weight_bits); _, new_w = _range(wb)
    m._w_scale.mul_(old_w / new_w)
    old_a = _range(m.activation_bits)[1]; new_a = _range(ab)[1]
    m._act_scale.mul_(old_a / new_a)
    m._weight_bits.fill_(wb); m._activation_bits.fill_(ab)
    m._weight_bits_value = wb; m._activation_bits_value = ab
    m._activation_scale_value = float(m._act_scale[0])

def make_unsigned(m):
    # same calibrated absmax; grid [0, 2^b-1] instead of the signed [-2^(b-1), 2^(b-1)-1]
    s_signed = float(m._act_scale[0]); qpos_s = _range(m.activation_bits)[1]
    bits = m.activation_bits; qmax_u = 2 ** bits - 1; s_u = s_signed * qpos_s / qmax_u
    def fwd(self, inp):
        if self.fake_quant_enabled:
            inp = torch.fake_quantize_per_tensor_affine(inp, s_u, 0, 0, qmax_u)
            wmin, wmax = _range(self.weight_bits)
            w = torch.fake_quantize_per_channel_affine(self.weight, self._w_scale,
                    torch.zeros(self.out_channels, dtype=torch.int32, device=self.weight.device), 0, wmin, wmax)
        else:
            w = self.weight
        return self._conv_forward(inp, w, self.bias)
    m.forward = types.MethodType(fwd, m)

def nonneg_sites(model):
    # empirical proxy on the 89 S11-val INPUT images (labels unused): conv inputs with min >= 0
    val, _ = cj.loaders(spec)
    mins = {n: float("inf") for n, mod in model.named_modules() if isinstance(mod, FixedQATConv2d)}
    hooks = []
    for n, mod in model.named_modules():
        if isinstance(mod, FixedQATConv2d):
            hooks.append(mod.register_forward_pre_hook(lambda m_, inp, n=n: mins.__setitem__(n, min(mins[n], float(inp[0].min())))))
    with torch.no_grad(), quantization_mode(model, False):
        for b in val:
            model(b["image"].cuda().float())
    for h in hooks: h.remove()
    return [n for n, v in mins.items() if v >= -1e-6]

def miou(model):
    r = cj.validation(model.cuda(), spec)["micro"]
    return r

def fresh(cell):
    return build(cell).cuda()

names = lambda m: [n for n, mod in m.named_modules() if isinstance(mod, FixedQATConv2d)]
first = "model.backbone.input_stem.op_list.0.conv"

print("\nPART 2 - fixed-checkpoint PTQ on the 89 S11-validation images (pooled two-class mIoU, %)")
def row(label, m):
    r = miou(m); mi = r.get("mIoU", r.get("miou")); wi = r.get("water_IoU", r.get("water_iou"))
    print(f"  {label:70s} mIoU={100*mi:6.3f}" + (f"  waterIoU={100*wi:6.3f}" if wi is not None else ""))
    return mi

m = fresh("A3_W4A4"); mods = dict(m.named_modules())
nn_sites = nonneg_sites(m)
print(f"  [info] empirically non-negative conv inputs on the 89 calibration images: {len(nn_sites)}/{len(names(m))}; first conv included: {first in nn_sites}")
row("W4A4 as calibrated (reproduces the pilot's 68.990)", m)

m = fresh("A3_W4A4"); mods = dict(m.named_modules()); set_bits(mods[first], 4, 8)
row("W4A4, first-conv (stem) INPUT activation at 8 bit", m)

m = fresh("A3_W4A4"); mods = dict(m.named_modules())
for n in names(m): set_bits(mods[n], 4, 8)
row("W4A8 (all activations 8 bit; reproduces 82.200)", m)

m = fresh("A3_W4A4"); mods = dict(m.named_modules())
for n in names(m): set_bits(mods[n], 8, 4)
row("W8A4 (weights 8 bit, activations 4 bit)", m)

m = fresh("A3_W4A4"); mods = dict(m.named_modules())
attn = [n for n in names(m) if "litemla.qkv" in n or "litemla.aggreg" in n]
set_bits(mods[first], 4, 8)
for n in attn: set_bits(mods[n], 4, 8)
row(f"W4A4, stem + {len(attn)} LiteMLA qkv/aggreg inputs at 8 bit", m)

m = fresh("A3_W4A4"); mods = dict(m.named_modules())
for n in nn_sites:
    if n != first: make_unsigned(mods[n])
row(f"W4A4, UNSIGNED 4-bit grid at {len([n for n in nn_sites if n != first])} empirically non-negative sites", m)

m = fresh("A3_W4A4"); mods = dict(m.named_modules())
set_bits(mods[first], 4, 8)
for n in nn_sites:
    if n != first: make_unsigned(mods[n])
row("W4A4, stem 8 bit + unsigned grids at non-negative sites", m)

# input-stem resolution
m = fresh("A3_W4A4"); mods = dict(m.named_modules()); s4 = float(mods[first]._act_scale[0])
val, _ = cj.loaders(spec); z = torch.cat([b["image"] for b in list(val)[:8]]).float()
print(f"\n  [info] stem activation step at A4 = {s4:.4f} z-units -> {100*float((z.abs() < s4/2).float().mean()):.1f}% of normalized input values round to 0 "
      f"(A8 step would be {s4*7/127:.4f} -> {100*float((z.abs() < s4*7/127/2).float().mean()):.2f}%)")
