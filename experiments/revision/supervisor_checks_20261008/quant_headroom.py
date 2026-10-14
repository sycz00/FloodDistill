import sys, json, torch
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, compression_train_mode, FixedQATConv2d
from data.pseudo_labeled import PseudoLabeledDataset
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
spec = json.load(open("/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/pilot_final/specs/A3_W4A4_B1_s42.json"))

# ---- (1) weights: absmax vs MSE-optimal clipping at 4 bits, per output channel (A3 FP weights)
fp = torch.load(f"{D}/A3_FP_s42/best.pt", map_location="cpu", weights_only=False)["model"]
tot_abs = tot_opt = tot_energy = 0.0
for k, w in fp.items():
    if not k.endswith("conv.weight") or w.dim() != 4: continue
    w = w.float().flatten(1)               # [out, rest]
    amax = w.abs().amax(1, keepdim=True).clamp_min(1e-8)
    best = None
    for c in torch.linspace(0.3, 1.0, 36):
        s = c * amax / 7
        q = torch.clamp(torch.round(w / s), -8, 7) * s
        mse = ((q - w) ** 2).sum(1)
        best = mse if best is None else torch.minimum(best, mse)
    s0 = amax / 7
    q0 = torch.clamp(torch.round(w / s0), -8, 7) * s0
    tot_abs += float(((q0 - w) ** 2).sum()); tot_opt += float(best.sum()); tot_energy += float((w ** 2).sum())
print(f"(1) 4-bit WEIGHT quantization, A3 FP weights: SQNR absmax = {10*torch.log10(torch.tensor(tot_energy/tot_abs)):.2f} dB, "
      f"MSE-optimal per-channel clip = {10*torch.log10(torch.tensor(tot_energy/tot_opt)):.2f} dB  "
      f"(MSE reduction {100*(1-tot_opt/tot_abs):.1f}%)")

# ---- (2) activations: signedness and effective resolution at conv inputs (W4A4 model, real batch)
ck = torch.load(f"{D}/A3_W4A4_B1_s42/best.pt", map_location="cpu", weights_only=False)
m = load_compression_qat(build_compression_student("A3", pretrained=False), ck["model"])
compression_train_mode(m, enabled=False)
m.eval()
ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=False, geom_aug=False)
x = torch.stack([ds[i]["image"] for i in range(8)]).float()
rows = []
hooks = []
def mk(name, mod):
    def h(module, inp):
        a = inp[0].detach()
        s4 = float(mod._act_scale[0]) * (127 / 7)       # A4 step after rebinned scale (act_scale already rebinned in this ckpt)
        s_act = float(mod._act_scale[0])
        qmax = 7 if int(mod._activation_bits) == 4 else 127
        rows.append(dict(name=name, min=float(a.min()), max=float(a.max()), step=s_act,
                         nonneg=bool((a >= 0).all()), frac_round_to_zero=float((a.abs() < s_act / 2).float().mean()),
                         frac_clipped=float((a.abs() > qmax * s_act).float().mean()),
                         p99_over_qmaxstep=float(torch.quantile(a.abs().flatten()[:2_000_000], 0.99) / (qmax * s_act))))
    return h
for n, mod in m.named_modules():
    if isinstance(mod, FixedQATConv2d):
        hooks.append(mod.register_forward_pre_hook(mk(n, mod)))
with torch.no_grad(): m(x)
for h in hooks: h.remove()
nn_ = sum(r["nonneg"] for r in rows)
print(f"\n(2) activation inputs at the {len(rows)} quantized convs (A4 grid = signed [-8,7], absmax scale):")
print(f"    inputs that are entirely non-negative (post-ReLU6, so the negative half of the grid is unused): {nn_}/{len(rows)}")
import statistics as st
print(f"    median fraction of input values that round to 0 at A4: {100*st.median(r['frac_round_to_zero'] for r in rows):.1f}%   "
      f"max over layers: {100*max(r['frac_round_to_zero'] for r in rows):.1f}%")
print(f"    median p99(|x|)/(7*step) (1.0 = p99 reaches the clip edge): {st.median(r['p99_over_qmaxstep'] for r in rows):.3f}")
worst = sorted(rows, key=lambda r: -r["frac_round_to_zero"])[:5]
for r in worst: print(f"    zero-rounding {100*r['frac_round_to_zero']:5.1f}%  p99/edge {r['p99_over_qmaxstep']:.3f}  min {r['min']:+.2f} max {r['max']:.2f}  {r['name']}")
