"""Guard against a metric bug in distilled_index_pilot.py: score the pilot's A3/A0 FP32 checkpoints with the SAME confusion/mIoU
code used for the per-pixel models and compare with the recorded source-validation numbers (A3 82.699, A0 84.155)."""
import sys, json, numpy as np, torch
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
from models.compression import build_compression_student
import scripts.compression_jobs as cj
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
val, _ = cj.loaders(spec)
for g, rec in (("A3", 82.699), ("A0", 84.155)):
    m = build_compression_student(g, pretrained=False)
    m.load_state_dict(torch.load(f"{D}/{g}_FP_s42/best.pt", map_location="cpu", weights_only=False)["model"], strict=True); m = m.cuda().eval()
    c = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for b in val:
            p = m(b["image"].cuda().float()).argmax(1).cpu(); ref = b["mask"].long(); ok = ref >= 0
            for a in (0, 1):
                for q in (0, 1): c[a, q] += int(((ref == a) & (p == q) & ok).sum())
    tp, fp, fn, tn = c[1, 1], c[0, 1], c[1, 0], c[0, 0]
    print(f"{g}: my confusion-based pooled mIoU = {100*(tp/(tp+fp+fn)+tn/(tn+fp+fn))/2:.3f}   recorded source-validation mIoU = {rec}")
