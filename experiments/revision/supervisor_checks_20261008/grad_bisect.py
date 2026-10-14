import sys, json, copy, torch
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import (load_compression_qat, compression_train_mode, FixedQATConv2d,
                                       rebind_compression_bits)
from quantization import qat as Q
from data.pseudo_labeled import PseudoLabeledDataset
from training.losses import focal_loss
import torch.nn.functional as F
spec = json.load(open("/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/pilot_final/specs/A3_W4A4_B1_s42.json"))
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=True, geom_aug=True)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:16].tolist()
b = [ds[i] for i in idx]
x = torch.stack([t["image"] for t in b]).float(); y = torch.stack([t["mask"] for t in b]).long()

def gradnorm(m):
    m.zero_grad(set_to_none=True)
    out = m(x)
    loss = focal_loss(out, y)
    loss.backward()
    return float(sum(p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None) ** .5), float(loss)

ck = torch.load(f"{D}/A3_W4A4_B1_s42/best.pt", map_location="cpu", weights_only=False)
m = load_compression_qat(build_compression_student("A3", pretrained=False), ck["model"])
compression_train_mode(m, enabled=True)
qmods = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, FixedQATConv2d)]
print("n quantized convs:", len(qmods))

# (a) weight bits vs activation bits
for wb, ab in [(4, 4), (4, 8), (8, 4), (8, 8)]:
    mm = rebind_compression_bits(m, wb, ab, inplace=False)
    compression_train_mode(mm, enabled=True)
    n, l = gradnorm(mm)
    print(f"W{wb}A{ab} (same observed ranges, rebinned): grad_norm={n:.4g} loss={l:.4f}")

# (b) only ONE layer's quantizer ON at W4A4
print("\nsingle-layer fake-quant ON (all others OFF), W4A4:")
res = []
for n, mod in qmods:
    for _, o in qmods: o.fake_quant_enabled = False
    mod.fake_quant_enabled = True
    gn, l = gradnorm(m)
    res.append((gn, n, mod.weight.shape))
for gn, n, s in sorted(res, reverse=True)[:8]:
    print(f"  {gn:12.4g}  {tuple(s)}  {n}")
print("  ... median over layers:", sorted(r[0] for r in res)[len(res)//2])

# (c) LiteMLA denominators under W4A4 (all quantizers ON)
for _, o in qmods: o.fake_quant_enabled = True
stats = []
orig = Q.LiteMLAQAT._relu_linear_att_qat
def patched(self, qkv):
    B, _, H, W = list(qkv.size())
    qkv_r = qkv.reshape(B, -1, 3 * self.litemla.dim, H * W)
    dim = self.litemla.dim
    q = self.litemla.kernel_func(qkv_r[:, :, :dim]); k = self.litemla.kernel_func(qkv_r[:, :, dim:2*dim])
    v = F.pad(qkv_r[:, :, 2*dim:], (0, 0, 0, 1), value=1)
    vk = self.mm1(v, k.transpose(-1, -2)); out = self.mm2(vk, q)
    den = out[:, :, -1:]
    stats.append(dict(den_min=float(den.min()), den_p1=float(torch.quantile(den.flatten(), .01)),
                      frac_lt_1e6=float((den < 1e-6).float().mean()), frac_lt_1e3=float((den < 1e-3).float().mean()),
                      q_zero_frac=float((q == 0).float().mean()), k_zero_frac=float((k == 0).float().mean())))
    return orig(self, qkv)
Q.LiteMLAQAT._relu_linear_att_qat = patched
compression_train_mode(m, enabled=True)
gn, l = gradnorm(m)
print("\nLiteMLA denominators in W4A4 forward (per attention block):")
for s in stats: print("  ", {k: float(f"{v:.4g}") for k, v in s.items()})
Q.LiteMLAQAT._relu_linear_att_qat = orig
