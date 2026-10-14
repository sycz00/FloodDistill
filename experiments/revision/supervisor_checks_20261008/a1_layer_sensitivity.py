"""Why does A1 lose 4.6 pp at W4A8 PTQ while A3 loses 0.5? Per-layer 'damage-one' sweep on S11-val (89 images).

Base = all layers W8A8 (A1 PTQ W8A8 = 82.667). For each quantized conv i: only layer i's WEIGHTS are re-binned to 4 bit
(activations stay 8 bit). A large drop localises the damage. Fixed weights, no training.
"""
import sys, json, torch
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

def build(graph):
    p = torch.load(R + f"results/{graph}_W4A8_interaction.pt", map_location="cpu", weights_only=True)
    return load_compression_qat(build_compression_student(graph, pretrained=False), p["model"]).cuda()

def miou(m):
    r = cj.validation(m, spec)["micro"]; return 100 * r.get("mIoU", r.get("miou"))

for graph in ("A1",):
    m = build(graph); names = [n for n, mod in m.named_modules() if isinstance(mod, FixedQATConv2d)]
    mods = dict(m.named_modules())
    w4a8 = miou(m)
    for n in names: set_bits(mods[n], 8, 8)
    w8a8 = miou(m)
    print(f"{graph}: W4A8 (all layers)={w4a8:.3f}   W8A8 (all layers)={w8a8:.3f}   => weight-4-bit damage total = {w8a8-w4a8:.3f} pp")
    rows = []
    for n in names:
        set_bits(mods[n], 4, 8); rows.append((w8a8 - miou(m), n, tuple(mods[n].weight.shape))); set_bits(mods[n], 8, 8)
    rows.sort(reverse=True)
    tot = sum(max(r[0], 0) for r in rows)
    print(f"  single-layer 4-bit-weight damage (pp, others W8A8); sum of positive parts = {tot:.2f}")
    for d, n, s in rows[:8]: print(f"    {d:6.3f}  {s}  {n}")
    print("    ... median over layers:", sorted(r[0] for r in rows)[len(rows)//2], " n_layers:", len(rows))
