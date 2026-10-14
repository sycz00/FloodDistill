"""Validation-only pilot: can the teacher be distilled into a closed-form, gain-invariant spectral index?

Fit on TRAIN pool pixels (restored original N=2,500 cohort, teacher hard masks). Score on the 89 S11-VALIDATION chips only
(S11-test / STURM / WorldFloods are NOT touched). Models (all per-pixel, no spatial context):
  MNDWI>0        parameter-free rule (project's own definition)
  ZSUM-LOG       score = sum_b w_b * log(rho_b),  sum_b w_b = 0   (5 free weights + bias): exactly gain-invariant
  FREE-LOG       same without the zero-sum constraint (6 weights + bias)
  LIN-Z          linear on z-scored bands (6 weights + bias)
  MLP-Z          per-pixel MLP 6-16-16-1 on z-scored bands (~400 params)
Also reports how many decisions change under a common reflectance gain (x1.3, x0.8).
"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = ["/home/schmalstieg/repos/FloodDistill", "/home/schmalstieg/repos/_deps/efficientvit"]
torch.set_num_threads(6); torch.manual_seed(0); rng = np.random.default_rng(0)
from data.pseudo_labeled import PseudoLabeledDataset
from data.sen1floods11 import NORM_MEAN, NORM_STD
import scripts.compression_jobs as cj
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/"
spec = json.load(open(R + "pilot_final/specs/A3_W4A4_B1_s42.json"))
MU = torch.tensor(NORM_MEAN).view(1, 6, 1, 1); SD = torch.tensor(NORM_STD).view(1, 6, 1, 1)
EPS = 1e-3     # reflectance floor for log (documented; breaks exact invariance only for rho ~ EPS)

# ---- training pixels: teacher hard masks on the restored original pool, no augmentation
ds = PseudoLabeledDataset(spec["pseudo"], n_samples=2500, seed=42, use_curated=True, index_file=spec["index"],
                          extra_keys=frozenset(), mult_noise=False, geom_aug=False)
Xs, Ys = [], []
for i in range(len(ds)):
    it = ds[i]; img = it["image"].float().unsqueeze(0); m = it["mask"].long()
    rho = (img * SD + MU)[0].reshape(6, -1).T                  # [P, 6] reflectance
    lab = m.reshape(-1); ok = (lab >= 0).nonzero().squeeze(1)
    if ok.numel() == 0: continue
    sel = ok[torch.from_numpy(rng.choice(ok.numel(), size=min(150, ok.numel()), replace=False))]
    Xs.append(rho[sel]); Ys.append(lab[sel].float())
rho_tr = torch.cat(Xs); y_tr = torch.cat(Ys)
print(f"train pixels: {rho_tr.shape[0]} from {len(Xs)} tiles; teacher-water fraction {float(y_tr.mean()):.3f}")

def feat_log(rho, c): return torch.log(rho.clamp_min(EPS)) - c
def feat_z(rho): return (rho - MU.view(1, 6)) / SD.view(1, 6)
c_glob = float(torch.log(rho_tr.clamp_min(EPS)).mean())

def fit_logreg(X, y, zero_sum, l2=1e-4, iters=300):
    d = X.shape[1]; u = torch.zeros(d, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([u, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); w = u - u.mean() if zero_sum else u
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    w = (u - u.mean() if zero_sum else u).detach(); return w, b.detach()

w_zs, b_zs = fit_logreg(feat_log(rho_tr, c_glob), y_tr, True)
w_fl, b_fl = fit_logreg(feat_log(rho_tr, c_glob), y_tr, False)
w_lz, b_lz = fit_logreg(feat_z(rho_tr), y_tr, False)
mlp = nn.Sequential(nn.Linear(6, 16), nn.ReLU6(), nn.Linear(16, 16), nn.ReLU6(), nn.Linear(16, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=3e-3); Xz = feat_z(rho_tr)
for ep in range(6):
    perm = torch.randperm(Xz.shape[0])
    for k in range(0, Xz.shape[0], 4096):
        ix = perm[k:k + 4096]; opt.zero_grad()
        F.binary_cross_entropy_with_logits(mlp(Xz[ix]).squeeze(1), y_tr[ix]).backward(); opt.step()
names = ["B2", "B3", "B4", "B8A", "B11", "B12"]
print("ZSUM-LOG weights (sum = %.1e):" % float(w_zs.sum()), {n: round(float(v), 3) for n, v in zip(names, w_zs)}, "bias", round(float(b_zs), 3))
print("FREE-LOG weights:", {n: round(float(v), 3) for n, v in zip(names, w_fl)}, "bias", round(float(b_fl), 3))

def predict(kind, rho):                                         # rho [P,6] -> bool water
    if kind == "MNDWI>0": g, s = rho[:, 1], rho[:, 4]; return ((g - s) / (g + s).clamp_min(1e-6)) > 0
    if kind == "ZSUM-LOG": return (feat_log(rho, c_glob) @ w_zs + b_zs) > 0
    if kind == "FREE-LOG": return (feat_log(rho, c_glob) @ w_fl + b_fl) > 0
    if kind == "LIN-Z": return (feat_z(rho) @ w_lz + b_lz) > 0
    if kind == "MLP-Z": return mlp(feat_z(rho)).squeeze(1) > 0
kinds = ["MNDWI>0", "ZSUM-LOG", "FREE-LOG", "LIN-Z", "MLP-Z"]
nparam = {"MNDWI>0": 0, "ZSUM-LOG": 6, "FREE-LOG": 7, "LIN-Z": 7, "MLP-Z": sum(p.numel() for p in mlp.parameters())}

# ---- S11 VALIDATION only
val, _ = cj.loaders(spec)
cm = {k: np.zeros((2, 2), dtype=np.int64) for k in kinds}      # [ref, pred]
chg = {(k, g): [0, 0] for k in kinds for g in (1.3, 0.8)}
with torch.no_grad():
    for b in val:
        img = b["image"].float(); ref = b["mask"].long()
        rho = (img * SD + MU).permute(0, 2, 3, 1).reshape(-1, 6); lab = ref.reshape(-1); ok = lab >= 0
        for k in kinds:
            p = predict(k, rho)
            for a in (0, 1):
                for c in (0, 1): cm[k][a, c] += int(((lab == a) & (p.long() == c) & ok).sum())
            for g in (1.3, 0.8):
                p2 = predict(k, rho * g); chg[(k, g)][0] += int(((p != p2) & ok).sum()); chg[(k, g)][1] += int(ok.sum())
print("\nS11-VALIDATION (89 chips), pooled; per-pixel models trained on teacher pseudo-labels of the TRAIN pool")
print(f"{'model':10s} {'params':>7s} {'mIoU':>7s} {'waterIoU':>9s} {'prec':>7s} {'recall':>7s} | decisions changed by gain x1.3 / x0.8")
for k in kinds:
    c = cm[k]; tp = c[1, 1]; fp = c[0, 1]; fn = c[1, 0]; tn = c[0, 0]
    iw = tp / (tp + fp + fn); ib = tn / (tn + fp + fn)
    print(f"{k:10s} {nparam[k]:7d} {100*(iw+ib)/2:7.2f} {100*iw:9.2f} {100*tp/(tp+fp):7.2f} {100*tp/(tp+fn):7.2f} | "
          f"{100*chg[(k,1.3)][0]/chg[(k,1.3)][1]:.4f}% / {100*chg[(k,0.8)][0]/chg[(k,0.8)][1]:.4f}%")
