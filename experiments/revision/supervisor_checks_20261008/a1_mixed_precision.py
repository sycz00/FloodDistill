"""Does giving the few most sensitive layers 8-bit WEIGHTS recover A1's W4A8 damage? (fixed weights, S11-val, no training)

Ranking comes from the 'damage-one' sweep (a1_layer_sensitivity.py; labels of the 89 S11-val images are used, as for any
validation-based selection). Control: k RANDOM layers (seeded) at 8 bit. Extra weight bytes are reported.
"""
import sys, json, random, torch
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, FixedQATConv2d, _range
import scripts.compression_jobs as cj
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
def set_bits(m, wb, ab):
    _, old_w = _range(m.weight_bits); _, new_w = _range(wb); m._w_scale.mul_(old_w / new_w)
    old_a = _range(m.activation_bits)[1]; new_a = _range(ab)[1]; m._act_scale.mul_(old_a / new_a)
    m._weight_bits.fill_(wb); m._activation_bits.fill_(ab); m._weight_bits_value = wb; m._activation_bits_value = ab
    m._activation_scale_value = float(m._act_scale[0])
def build():
    p = torch.load(R + "results/A1_W4A8_interaction.pt", map_location="cpu", weights_only=True)
    return load_compression_qat(build_compression_student("A1", pretrained=False), p["model"]).cuda()
def miou(m):
    r = cj.validation(m, spec)["micro"]; return 100 * r.get("mIoU", r.get("miou"))
def run(upgrade):
    m = build(); mods = dict(m.named_modules())
    for n in upgrade: set_bits(mods[n], 8, 8)
    return miou(m), sum(mods[n].weight.numel() for n in upgrade)
# damage-one ranking from receipts/a1_layer_sensitivity.txt (top 8) -- re-derived here for safety
m0 = build(); mods0 = dict(m0.named_modules()); names = [n for n, mod in mods0.items() if isinstance(mod, FixedQATConv2d)]
total_w = sum(mods0[n].weight.numel() for n in names)
for n in names: set_bits(mods0[n], 8, 8)
base8 = miou(m0); rank = []
for n in names:
    set_bits(mods0[n], 4, 8); rank.append((base8 - miou(m0), n)); set_bits(mods0[n], 8, 8)
rank.sort(reverse=True); order = [n for _, n in rank]
print(f"A1: W4A8 all layers = 78.952 (pilot); W8A8 all layers = {base8:.3f}; quantized-conv weights = {total_w}")
print(f"{'k layers at 8-bit weights':28s} {'mIoU (ranked top-k)':>20s} {'mIoU (random-k, mean of 3)':>28s} {'extra weight bytes vs all-W4':>30s}")
rng = random.Random(0)
for k in (0, 1, 2, 4, 8):
    top, w = run(order[:k]) if k else (miou(build()), 0)
    rnd = []
    if k:
        for s in range(3): rnd.append(run(rng.sample(names, k))[0])
    extra = 0.5 * w   # 4 -> 8 bit weights: +0.5 byte per weight
    print(f"{k:28d} {top:20.3f} {(sum(rnd)/len(rnd) if rnd else float('nan')):28.3f} {extra:14.0f} B ({100*extra/(0.5*total_w):.1f}% of the 4-bit weight payload)")
print("top-8 layers:", [n[-48:] for n in order[:8]])
