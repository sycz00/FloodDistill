"""Where does the exploding QAT gradient sit? (A3 W4A4 B1 best, one 16-image batch, focal loss, CPU)

For every (image, head, position) of the LiteMLA qkv tensor: gradient magnitude vs the quantized and
bypassed (FP) attention denominators D_Q, D_F. Tests prompt6's cases (a) small already in FP,
(b) collapse introduced by quantization, (c) large for another reason.
"""
import sys, json, torch
import torch.nn.functional as F
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(4)
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, compression_train_mode, quantization_mode
from quantization import qat as Q
from data.pseudo_labeled import PseudoLabeledDataset
from training.losses import focal_loss
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=True, geom_aug=True)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:16].tolist()
b = [ds[i] for i in idx]
x = torch.stack([t["image"] for t in b]).float(); y = torch.stack([t["mask"] for t in b]).long()

ck = torch.load(f"{D}/A3_W4A4_B1_s42/best.pt", map_location="cpu", weights_only=False)["model"]
m = load_compression_qat(build_compression_student("A3", pretrained=False), ck)

orig = Q.LiteMLAQAT._relu_linear_att_qat
rec = {}
def make(tag, keep_grad):
    def patched(self, qkv):
        if keep_grad and qkv.requires_grad: qkv.retain_grad()
        B, _, H, W = list(qkv.size()); dim = self.litemla.dim
        qkv_r = qkv.reshape(B, -1, 3 * dim, H * W)
        qq = self.litemla.kernel_func(qkv_r[:, :, :dim]); kk = self.litemla.kernel_func(qkv_r[:, :, dim:2 * dim])
        v = F.pad(qkv_r[:, :, 2 * dim:], (0, 0, 0, 1), value=1)
        out = self.mm2(self.mm1(v, kk.transpose(-1, -2)), qq)
        rec.setdefault(tag, []).append(dict(qkv=qkv, D=out[:, :, -1, :].detach().clone(), qmag=qq.detach().norm(dim=2)))
        return orig(self, qkv)
    return patched

# bypassed (FP) reference denominators, no grad
rec.clear(); Q.LiteMLAQAT._relu_linear_att_qat = make("F", False)
compression_train_mode(m, enabled=True)
with torch.no_grad(), quantization_mode(m, False): m(x)
DF = [r["D"] for r in rec["F"]]

def run(enabled, tag):
    rec.pop(tag, None)
    Q.LiteMLAQAT._relu_linear_att_qat = make(tag, True)
    compression_train_mode(m, enabled=enabled); m.zero_grad(set_to_none=True)
    loss = focal_loss(m(x), y); loss.backward()
    out = []
    for blk, r in enumerate(rec[tag]):
        gr = r["qkv"].grad                                # [B, C, H, W]
        B, C, H, W = gr.shape; heads = r["D"].shape[1]
        gh = gr.reshape(B, heads, -1, H * W).norm(dim=2)  # [B, heads, HW] per-position gradient magnitude
        out.append((gh, r["D"], DF[blk], r["qmag"]))
    return float(loss), out

for enabled, tag in ((True, "Q"), (False, "OFF")):
    loss, out = run(enabled, tag)
    print(f"\n=== {'W4A4 fake-quant ON' if enabled else 'same weights, quantization OFF (FP bypass)'}: loss={loss:.4f}")
    for blk, (gh, Dq, Df, qm) in enumerate(out):
        flat = gh.flatten(); n = flat.numel()
        sq = (flat ** 2); tot = float(sq.sum())
        k = max(1, int(0.001 * n)); top = torch.topk(flat, k).indices
        share = float(sq[top].sum() / tot)
        Dq_f, Df_f, qm_f = Dq.flatten(), Df.flatten(), qm.flatten()
        def frac(cond): return float(cond[top].float().mean())
        print(f"  attention block {blk}: positions={n}  total grad^2 at qkv={tot:.3g}  top-0.1% positions ({k}) hold {100*share:.1f}% of grad^2")
        print(f"     among top-0.1%: D_Q==0: {100*frac(Dq_f==0):.0f}% | 0<D_Q<1e-3: {100*frac((Dq_f>0)&(Dq_f<1e-3)):.0f}% | 1e-3<=D_Q<1e-1: {100*frac((Dq_f>=1e-3)&(Dq_f<1e-1)):.0f}% | D_Q>=1e-1: {100*frac(Dq_f>=1e-1):.0f}%")
        print(f"     among top-0.1%: D_F<1e-3 (already small in FP): {100*frac(Df_f<1e-3):.0f}% | D_F>=1e-1 but D_Q<D_F/16 (collapse by quantization): {100*frac((Df_f>=1e-1)&(Dq_f<Df_f/16)):.0f}% | median D_Q={float(Dq_f[top].median()):.3g}, median D_F={float(Df_f[top].median()):.3g}, median |q|={float(qm_f[top].median()):.3g}")
        allm = lambda c: 100 * float(c.float().mean())
        print(f"     baseline over ALL positions: D_Q==0 {allm(Dq_f==0):.1f}% | 0<D_Q<1e-3 {allm((Dq_f>0)&(Dq_f<1e-3)):.2f}% | D_F<1e-3 {allm(Df_f<1e-3):.1f}%")
Q.LiteMLAQAT._relu_linear_att_qat = orig
