"""Do the pilot's QAT-trained checkpoints (selected `best.pt`) survive integer-domain arithmetic? S11-val, 89 images."""
import sys, json, torch
import torch.nn.functional as F
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, FixedQATConv2d, _range
import scripts.compression_jobs as cj
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
orig_forward = FixedQATConv2d.forward
def int_forward(self, inp):
    if not self.fake_quant_enabled: return self._conv_forward(inp, self.weight, self.bias)
    sa = float(self._act_scale[0]); amin, amax = _range(self.activation_bits); wmin, wmax = _range(self.weight_bits)
    x_int = torch.clamp(torch.round(inp / sa), amin, amax)
    w_int = torch.clamp(torch.round(self.weight / self._w_scale.view(-1, 1, 1, 1)), wmin, wmax)
    out = F.conv2d(x_int, w_int, None, self.stride, self.padding, self.dilation, self.groups) * (sa * self._w_scale.view(1, -1, 1, 1))
    return out + self.bias.view(1, -1, 1, 1) if self.bias is not None else out
print("S11-val pooled mIoU (%): selected QAT checkpoints under the pilot's float32 fake-quant vs integer-domain arithmetic")
print(f"{'checkpoint':18s} {'float32 (pilot)':>16s} {'integer-domain':>16s} {'delta':>8s}")
for cell in ("A3_W8A8_B1_s42","A3_W8A8_B2_s42","A3_W4A8_B1_s42","A3_W4A8_B2_s42","A3_W4A4_B1_s42","A3_W4A4_B2_s42","A3_W4A4_B3_s42","A1_W4A8_B1_s42","A1_W4A8_B2_s42"):
    ck = torch.load(f"{D}/{cell}/best.pt", map_location="cpu", weights_only=False)["model"]
    graph = cell.split("_")[0]; vals = []
    for fwd in (orig_forward, int_forward):
        FixedQATConv2d.forward = fwd
        m = load_compression_qat(build_compression_student(graph, pretrained=False), ck).cuda()
        r = cj.validation(m, spec)["micro"]; vals.append(100 * r.get("mIoU", r.get("miou")))
    FixedQATConv2d.forward = orig_forward
    print(f"{cell:18s} {vals[0]:16.3f} {vals[1]:16.3f} {vals[1]-vals[0]:+8.3f}")
