"""Read-only CPU probe: where do QAT gradients explode? (single batch, focal hard loss only)."""
import sys, json, copy, torch
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, compression_train_mode, freeze_batch_norm, quantization_mode
from data.pseudo_labeled import PseudoLabeledDataset
from training.losses import focal_loss
import inspect
spec = json.load(open("/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/pilot_final/specs/A3_W4A4_B1_s42.json"))
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"

ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=True, geom_aug=True)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:16].tolist()
batch = [ds[i] for i in idx]
x = torch.stack([b["image"] for b in batch]).float()
y = torch.stack([b["mask"] for b in batch]).long()
print("batch", tuple(x.shape), tuple(y.shape), "ignore frac", float((y < 0).float().mean()), "water frac", float((y == 1).float().mean()))
print("focal_loss signature:", inspect.signature(focal_loss))

def lossfn(logits, y):
    lg = logits
    if lg.shape[-2:] != y.shape[-2:]:
        lg = torch.nn.functional.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return focal_loss(lg, y)

def probe(model, label, topk=8):
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = lossfn(out, y)
    loss.backward()
    named = [(n, p.grad.detach().norm().item(), p.numel()) for n, p in model.named_parameters() if p.grad is not None]
    tot = sum(g * g for _, g, _ in named) ** 0.5
    named.sort(key=lambda t: -t[1])
    print(f"\n--- {label}: loss={loss.item():.4f}  total_grad_norm={tot:.4g}  n_params_with_grad={len(named)}")
    for n, gn, k in named[:topk]:
        print(f"   {gn:12.4g}  ({k:5d})  {n}")
    return tot

def load_qat(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = load_compression_qat(build_compression_student("A3", pretrained=False), ck["model"])
    return m

# 1) the QAT model at W4A4 (B1 best checkpoint) in the exact QAT training mode
m = load_qat(f"{D}/A3_W4A4_B1_s42/best.pt")
compression_train_mode(m, enabled=True)
probe(m, "A3 W4A4 B1 best: QAT mode (BN frozen, fake-quant ON)")

# 2) same weights, fake-quant OFF, BN frozen
compression_train_mode(m, enabled=False)
probe(m, "A3 W4A4 B1 best: fake-quant OFF, BN frozen")

# 3) W8A8 checkpoint in QAT mode
m8 = load_qat(f"{D}/A3_W8A8_B1_s42/best.pt")
compression_train_mode(m8, enabled=True)
probe(m8, "A3 W8A8 B1 best: QAT mode (BN frozen, fake-quant ON)")

# 4) FP warmup checkpoint: train-mode BN (as during FP training) and eval-mode (frozen) BN
ck = torch.load(f"{D}/A3_FP_s42/best.pt", map_location="cpu", weights_only=False)
fp = build_compression_student("A3", pretrained=False)
fp.load_state_dict(ck["model"], strict=True)
fp.train()
probe(fp, "A3 FP: BN in TRAIN mode (as FP training)")
fp.eval()
for p in fp.parameters(): p.requires_grad_(True)
probe(fp, "A3 FP: BN in EVAL mode (frozen running stats, no quantization)")
