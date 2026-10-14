"""Where do float32 fake-quant and integer-domain fake-quant first diverge? (A3 W4A4 B1 best, CPU)"""
import sys, json, torch
import torch.nn.functional as F
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, compression_train_mode, FixedQATConv2d, _range
from data.pseudo_labeled import PseudoLabeledDataset
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=True, geom_aug=True)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[:16].tolist()
x = torch.stack([ds[i]["image"] for i in idx]).float()
ck = torch.load(f"{D}/A3_W4A4_B1_s42/best.pt", map_location="cpu", weights_only=False)["model"]
m = load_compression_qat(build_compression_student("A3", pretrained=False), ck); compression_train_mode(m, enabled=True); m.eval()
orig_forward = FixedQATConv2d.forward

def make_int(use_recip):
    def int_forward(self, inp):
        sa = float(self._act_scale[0]); amin, amax = _range(self.activation_bits); wmin, wmax = _range(self.weight_bits)
        if use_recip:   # exactly what torch.fake_quantize_per_tensor_affine does: x * (1/scale), float32
            inv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(sa, dtype=torch.float32)
            xs = inp * inv
        else:
            xs = inp / sa
        x_int = torch.clamp(torch.round(xs), amin, amax)
        sw = self._w_scale.view(-1, 1, 1, 1)
        if use_recip:
            invw = torch.tensor(1.0, dtype=torch.float32) / sw
            ws = self.weight * invw
        else:
            ws = self.weight / sw
        w_int = torch.clamp(torch.round(ws), wmin, wmax)
        out = F.conv2d(x_int, w_int, None, self.stride, self.padding, self.dilation, self.groups) * (sa * self._w_scale.view(1, -1, 1, 1))
        return out + self.bias.view(1, -1, 1, 1) if self.bias is not None else out
    return int_forward

def run(fwd):
    FixedQATConv2d.forward = fwd; outs = {}; hs = []
    for n, mod in m.named_modules():
        if isinstance(mod, FixedQATConv2d):
            hs.append(mod.register_forward_hook(lambda mm, i, o, n=n: outs.__setitem__(n, o.detach().clone())))
    with torch.no_grad(): logit = m(x)
    for h in hs: h.remove()
    FixedQATConv2d.forward = orig_forward
    return outs, logit

o0, l0 = run(orig_forward)
for label, fwd in (("int-domain, x/s", make_int(False)), ("int-domain, x*(1/s)  [mimics torch op]", make_int(True))):
    o1, l1 = run(fwd)
    print(f"\n{label}: final logits max|diff|={float((l0-l1).abs().max()):.3g}, argmax agreement={100*float((l0.argmax(1)==l1.argmax(1)).float().mean()):.3f}%")
    first = None
    for n in o0:
        d = float((o0[n] - o1[n]).abs().max()); rel = d / (float(o0[n].abs().max()) + 1e-12)
        if first is None and rel > 1e-5: first = (n, d, rel)
    print("   first conv whose OUTPUT differs by >1e-5 relative:", first if first else "none")
    shown = 0
    for n in o0:
        d = float((o0[n] - o1[n]).abs().max()); rel = d / (float(o0[n].abs().max()) + 1e-12)
        if shown < 6: print(f"     {n[-58:]:58s} max|diff|={d:.2e} rel={rel:.2e}"); shown += 1
