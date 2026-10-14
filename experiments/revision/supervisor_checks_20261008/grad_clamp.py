import sys, json, torch
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
import torch.nn.functional as F
from models.compression import build_compression_student, compression_graph_spec
from models.compression_quant import load_compression_qat, compression_train_mode
from quantization import qat as Q
from data.pseudo_labeled import PseudoLabeledDataset
from training.losses import focal_loss
spec = json.load(open("/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/pilot_final/specs/A3_W4A4_B1_s42.json"))
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=True, geom_aug=True)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:16].tolist()
b = [ds[i] for i in idx]
x = torch.stack([t["image"] for t in b]).float(); y = torch.stack([t["mask"] for t in b]).long()

orig = Q.LiteMLAQAT._relu_linear_att_qat
def make(clamp):
    def patched(self, qkv):
        B, _, H, W = list(qkv.size())
        dim = self.litemla.dim
        qkv_r = qkv.reshape(B, -1, 3 * dim, H * W)
        q = self.litemla.kernel_func(qkv_r[:, :, :dim]); k = self.litemla.kernel_func(qkv_r[:, :, dim:2*dim])
        v = F.pad(qkv_r[:, :, 2*dim:], (0, 0, 0, 1), value=1)
        out = self.mm2(self.mm1(v, k.transpose(-1, -2)), q)
        out = out[:, :, :-1] / out[:, :, -1:].clamp(min=clamp)
        return out.reshape(B, -1, H, W)
    return patched

def run(ckpt, label):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = load_compression_qat(build_compression_student("A3", pretrained=False), ck["model"])
    compression_train_mode(m, enabled=True)
    print(f"\n{label}")
    for c in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        Q.LiteMLAQAT._relu_linear_att_qat = make(c)
        m.zero_grad(set_to_none=True)
        out = m(x); loss = focal_loss(out, y); loss.backward()
        gn = float(sum(p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None) ** .5)
        print(f"  clamp(min={c:g}): loss={float(loss):.4f}  grad_norm={gn:.4g}")
    Q.LiteMLAQAT._relu_linear_att_qat = orig

run(f"{D}/A3_W4A4_B1_s42/best.pt", "A3 W4A4 (B1 best)")
run(f"{D}/A3_W4A8_B1_s42/best.pt", "A3 W4A8 (B1 best)")
run(f"{D}/A3_W8A8_B1_s42/best.pt", "A3 W8A8 (B1 best)")
print("\nattention dim per graph:", {gid: compression_graph_spec(gid)["attention_dim"] for gid in ("A0","A1","A2","A3")})
