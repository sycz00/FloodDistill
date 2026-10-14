"""Was the paper's ReLU6 student initialised from pretrained EfficientViT-seg-B0 (Cityscapes) weights or from scratch?
Cosine similarity between trained checkpoint tensors and the pretrained tensors (same name, same shape).
Controls: Hardswish student (known pretrained init), v7 A0 (declared random init)."""
import sys, glob, torch, numpy as np
PRE = sys.argv[1]
ckpts = sys.argv[2:]
def sd(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "student", "model_state_dict", "student_state_dict"):
        if isinstance(o, dict) and k in o and isinstance(o[k], dict):
            o = o[k]; break
    return {k: v for k, v in o.items() if torch.is_tensor(v)}
pre = sd(PRE)
def norm(k):
    for p in ("model.model.", "student.model.", "module.model.", "model.", "student."):
        if k.startswith(p): k = k[len(p):]
    return k
pre_n = {norm(k): v for k, v in pre.items()}
for c in ckpts:
    s = {norm(k): v for k, v in sd(c).items()}
    cos = []
    for k, v in s.items():
        if k in pre_n and pre_n[k].shape == v.shape and v.ndim == 4 and v.numel() > 64:
            a, b = v.flatten().double(), pre_n[k].flatten().double()
            cos.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
    cos = np.array(cos)
    print(f"{c}\n  matched conv tensors={len(cos)}  median cos={np.median(cos) if len(cos) else float('nan'):.3f}  "
          f"min={cos.min() if len(cos) else float('nan'):.3f}  frac>0.5={np.mean(cos>0.5) if len(cos) else float('nan'):.2f}")
