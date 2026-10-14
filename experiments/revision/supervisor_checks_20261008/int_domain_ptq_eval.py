"""Does the float32-simulation artifact change PTQ accuracy? S11-val (89 images), A3 PTQ start states, fixed weights."""
import sys, json, torch
import torch.nn.functional as F
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, FixedQATConv2d, _range
import scripts.compression_jobs as cj
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
orig_forward = FixedQATConv2d.forward
def int_forward(self, inp):
    if not self.fake_quant_enabled: return self._conv_forward(inp, self.weight, self.bias)
    sa = float(self._act_scale[0]); amin, amax = _range(self.activation_bits); wmin, wmax = _range(self.weight_bits)
    x_int = torch.clamp(torch.round(inp / sa), amin, amax)
    w_int = torch.clamp(torch.round(self.weight / self._w_scale.view(-1, 1, 1, 1)), wmin, wmax)
    out = F.conv2d(x_int, w_int, None, self.stride, self.padding, self.dilation, self.groups) * (sa * self._w_scale.view(1, -1, 1, 1))
    return out + self.bias.view(1, -1, 1, 1) if self.bias is not None else out
print("S11-validation pooled two-class mIoU (%), PTQ start states of the pilot (no training)")
print(f"{'cell':8s} {'float32 fake-quant (pilot)':>28s} {'integer-domain (exact zeros)':>30s}")
for cell, label in (("A3_W4A4", "W4A4"), ("A3_W4A8", "W4A8"), ("A3_W8A8", "W8A8"), ("A1_W4A8", "A1 W4A8")):
    p = torch.load(R + f"results/{cell}_interaction.pt", map_location="cpu", weights_only=True)
    graph = cell.split("_")[0]
    res = []
    for fwd in (orig_forward, int_forward):
        FixedQATConv2d.forward = fwd
        m = load_compression_qat(build_compression_student(graph, pretrained=False), p["model"]).cuda()
        r = cj.validation(m, spec)["micro"]; res.append((100 * r.get("mIoU", r.get("miou")), 100 * r.get("water_IoU", r.get("water_iou", float("nan")))))
    FixedQATConv2d.forward = orig_forward
    print(f"{label:8s} {res[0][0]:20.3f} (wIoU {res[0][1]:5.2f}) {res[1][0]:20.3f} (wIoU {res[1][1]:5.2f})")
