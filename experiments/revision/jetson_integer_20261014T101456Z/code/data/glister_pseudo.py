"""Offline GLISTER subset selection, last-layer approximation.

Killamsetty et al., AAAI 2021, arXiv:2012.10630, Algorithm 2. Candidates are
scored by the raw inner product of training and validation loss gradients (the
two log-likelihood signs cancel). The head is updated between the r greedy
rounds, the cached penultimate features are not. --lam weights the
facility-location regularizer.

Selection runs once here, unlike GLISTER-ONLINE. Default r=25; report the value
you used. Optional variants: --reg_weights r2core changes the coverage
weights, --val_haze shifts the validation objective with the same
reflectance-space haze as in evaluation. Selector recipes are in
experiments/build_indices.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_FF, MAX_FF, MIN_CONF_STD = 0.05, 0.85, 0.05
_PHI = ["flood_frac", "mean_conf", "conf_std",
        *[f"band_mean_{i}" for i in range(6)], *[f"band_std_{i}" for i in range(6)]]


def head_module(model: nn.Module, num_classes: int = 2) -> nn.Conv2d:
    """Last-layer classifier: the deepest module with out_dim == num_classes."""
    cand = None
    for _, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is not None and w.ndim >= 2 and w.shape[0] == num_classes:
            cand = m
    if cand is None:
        raise ValueError(f"no head with out_channels={num_classes}")
    return cand


class _Tiles(Dataset):
    def __init__(self, base: Path, files: list[str]):
        self.base, self.files = base, files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        d = torch.load(self.base / self.files[i], map_location="cpu", weights_only=True)
        return {"image": d["image"], "mask": d["mask"].long(), "idx": i}


@torch.no_grad()
def cache_features(model, head, loader, dev, tag: str):
    """Penultimate features and labels for all items; they do not change between rounds."""
    feats, labels = [], []
    grab = {}
    h = head.register_forward_hook(lambda m, i, o: grab.__setitem__("f", i[0].detach()))
    n = 0
    for b in loader:
        x = b["image"].to(dev, dtype=torch.float32)
        model(x)
        feats.append(grab["f"].half().cpu())
        labels.append(b["mask"].to(torch.int8).cpu())
        n += x.shape[0]
        if n % 2000 < x.shape[0]:
            print(f"  [{tag}] cached {n}", flush=True)
    h.remove()
    return torch.cat(feats), torch.cat(labels)


def per_sample_head_grads(W: torch.Tensor, feats: torch.Tensor, labels: torch.Tensor,
                          dev, batch: int = 256) -> torch.Tensor:
    """Per-sample gradient of the loss w.r.t. the head weight, shape (N, 2*C).

    For a 1x1-conv head, grad_W L_i = sum_px delta_i[:,px] (x) f_i[:,px] with
    delta = dL_i/dlogits. One backward pass on the summed loss gives all deltas; the loss
    uses full label resolution.
    """
    out = []
    # Gradients w.r.t. the head weight only (last-layer approximation), on a local copy.
    Wg = W.detach().clone().requires_grad_(True)
    for i in range(0, len(feats), batch):
        f = feats[i:i + batch].to(dev, dtype=torch.float32)
        y = labels[i:i + batch].to(dev, dtype=torch.long)
        logits = F.conv2d(f, Wg)                                 # (B,2,h,w)
        lg = logits
        if lg.shape[-2:] != y.shape[-2:]:
            lg = F.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
        per = F.cross_entropy(lg, y, ignore_index=-1, reduction="none")   # (B,H,W)
        valid = (y != -1).float()
        li = (per * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
        g_logits = torch.autograd.grad(li.sum(), logits)[0]      # (B,2,h,w)
        g = torch.einsum("bchw,bfhw->bcf", g_logits, f).reshape(len(f), -1)
        out.append(g.detach().cpu())
        del f, y, logits, g_logits
    return torch.cat(out)


def facility_marginal(sim: np.ndarray, best: np.ndarray, e: np.ndarray,
                      w: np.ndarray | None = None) -> np.ndarray:
    """R(e|S) for all candidates e.

    Unweighted (F-GLISTER):    sum_x [max(best_x, sim(x,e)) - best_x]
    Weighted (--reg_weights r2core): sum_x [max(best_x, w_x*sim(x,e)) - best_x]

    with best_x = w_x * max_{s in S} sim(x,s); w == 1 gives the unweighted form.
    """
    s = sim[:, e] if w is None else sim[:, e] * w[:, None]
    return np.maximum(s, best[:, None]).sum(0) - best.sum()


def run(pseudo_dir: str, ref: str, arch: str, top_k: int, r_rounds: int, lam: float,
        eta: float, gpu: int, out_name: str, batch_size: int, num_workers: int,
        limit: int, reg_weights: str = "uniform", val_haze: float = 0.0,
        consistency_name: str = "consistency.csv",
        lam_scale_mode: str = "mean", val_haze_mode: str = "absolute",
        lam_scale_per_round: bool = False,
        min_ff: float = MIN_FF, max_ff: float = MAX_FF,
        min_conf_std: float = MIN_CONF_STD) -> None:
    from data.sen1floods11 import get_dataloaders
    from models.student import build_student

    t0 = time.time()
    base = Path(pseudo_dir)
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # same candidate pre-filter as all the other selectors
    feats_csv = base / "pool_features.csv"
    if not feats_csv.exists():
        raise FileNotFoundError(f"{feats_csv} missing — run data.pool_features first")
    rows = list(csv.DictReader(open(feats_csv)))
    def _f(rw, k, d=0.0):
        v = rw.get(k, "")
        return float(v) if v not in ("", None) else d
    # (A) reliability() reads c_mean, so the consistency join has to run before `cand`
    # is built. Same join as in r2core_pseudo.select.
    if reg_weights == "r2core":
        cpath = base / consistency_name
        if not cpath.exists():
            raise FileNotFoundError(f"{cpath} missing — --reg_weights r2core needs c(x); "
                                    f"run: python -m data.teacher_consistency")
        cmap = {r["file"]: r["c_mean"] for r in csv.DictReader(open(cpath))}
        for rw in rows:
            rw["c_mean"] = cmap.get(rw["file"], "")
        print(f"[glister] c(x) joined for {sum(1 for r in rows if r['c_mean'])} tiles")

    cand = [rw for rw in rows
            if min_ff <= _f(rw, "flood_frac", -1) <= max_ff
            and _f(rw, "conf_std", -1) >= min_conf_std]
    if limit:
        cand = cand[:limit]
    files = [rw["file"] for rw in cand]
    print(f"[glister] {len(rows)} tiles -> {len(cand)} candidates | k={top_k} r={r_rounds} "
          f"lambda={lam} eta={eta}")
    if len(cand) < top_k:
        raise SystemExit(f"only {len(cand)} candidates for top_k={top_k}")

    # reference model theta^(0)
    model = build_student(arch, in_channels=6, num_classes=2, pretrained=False).to(dev).eval()
    sd = torch.load(ref, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd.get("student", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False is needed because the reference may carry distillation-only heads, but a
    # ref that loads almost nothing ranks the pool off random init and looks like it worked.
    if len(missing) > len(model.state_dict()) // 10:
        raise SystemExit(f"[glister] {ref} matched only "
                         f"{len(model.state_dict()) - len(missing)}/{len(model.state_dict())} "
                         f"parameters - wrong architecture or checkpoint container?")
    head = head_module(model, 2)
    W0 = head.weight.detach().clone()
    print(f"[glister] ref={ref} | {len(model.state_dict()) - len(missing)} keys loaded "
          f"({len(unexpected)} unexpected) | head weight {tuple(W0.shape)}")

    # cache pool features (pseudo labels) and validation features (clean GT)
    pool_dl = DataLoader(_Tiles(base, files), batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    pf, pl = cache_features(model, head, pool_dl, dev, "pool")
    print(f"[glister] pool features {tuple(pf.shape)} ({pf.numel()*2/1e9:.2f} GB fp16)")

    from utils.paths import dataset_root
    _, va, _ = get_dataloaders(str(dataset_root("S11_ROOT")), batch_size=8,
                               num_workers=4, img_size=224, eval_img_size=224)
    vitems = [(b["image"], b["mask"]) for b in va]
    vx = torch.cat([x for x, _ in vitems])
    vy = torch.cat([y for _, y in vitems])

    # (B) Haze on the validation objective. In normalised space, rho + h*w_b becomes
    # x + h*w_b/sigma_b; labels stay the same.
    vf_clean = None
    if val_haze > 0.0:
        from data.ood_datasets import _HAZE_W_6
        from data.sen1floods11 import NORM_STD
        d = (val_haze * torch.as_tensor(_HAZE_W_6, dtype=torch.float32)
             / torch.as_tensor(NORM_STD, dtype=torch.float32))
        if vx.shape[1] != d.numel():
            raise ValueError(f"val has {vx.shape[1]} channels, haze weights have {d.numel()}")
        if val_haze_mode in ("excess", "orthogonal"):
            # Also cache clean validation features: g_V(h) - g_V(0) is the gradient of the
            # robustness gap. g_V(h) alone is close to g_V(0) (cos 0.86 at h=0.1) and changes
            # only 4% of the top 2,500.
            vf_clean = cache_features(
                model, head,
                [{"image": vx[i:i+8], "mask": vy[i:i+8]} for i in range(0, len(vx), 8)],
                dev, "val_clean")
        vx = vx + d[None, :, None, None]
        print(f"[glister] (B) val hazed h={val_haze} mode={val_haze_mode} | per-band shift "
              f"in normalized units: " + " ".join(f"{v:.3f}" for v in d.tolist()))
    vf, vl = cache_features(
        model, head,
        [{"image": vx[i:i+8], "mask": vy[i:i+8]} for i in range(0, len(vx), 8)],
        dev, "val")
    print(f"[glister] val features {tuple(vf.shape)} over {len(vf)} clean GT tiles")

    # facility-location regularizer R over the same phi as the other selectors
    X = np.array([[_f(rw, k, 0.0) for k in _PHI] for rw in cand], np.float32)
    lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    X = np.clip((X - lo) / np.where(hi - lo > 1e-8, hi - lo, 1.0), 0, 1)
    d2 = ((X*X).sum(1)[:, None] + (X*X).sum(1)[None, :] - 2*X @ X.T).clip(min=0)
    sim = np.exp(-d2 / (2.0 * np.median(d2[d2 > 0]))).astype(np.float32)
    best = np.zeros(len(cand), np.float32)

    # (A) coverage weights w(x). Imported from r2core_pseudo so this arm and the
    # R2-Core arm use the same weight definition.
    w = None
    if reg_weights == "r2core":
        from data.r2core_pseudo import reliability, informativeness, img_quality
        w = (reliability(cand) * informativeness(cand) * img_quality(cand)).astype(np.float32)
        w = np.clip(w / max(w.mean(), 1e-8), 0.02, 20.0)     # mean-1, tails bounded
        print(f"[glister] (A) r2core weights: mean {w.mean():.3f} "
              f"p5 {np.percentile(w,5):.3f} p95 {np.percentile(w,95):.3f}")

    # Algorithm 2
    W = W0.clone()
    remaining = np.ones(len(cand), bool)
    selected: list[int] = []
    per_round = max(1, int(round(top_k / max(r_rounds, 1))))
    lam_scale = None

    for t in range(r_rounds):
        if len(selected) >= top_k:
            break
        # grad LL_V at the CURRENT theta (line 5 uses theta^(t), which line 9 moved)
        gv = per_sample_head_grads(W, vf, vl, dev).mean(0)                 # (2C,)
        if vf_clean is not None:
            gc = per_sample_head_grads(W, vf_clean[0], vf_clean[1], dev).mean(0)
            if val_haze_mode == "orthogonal":
                # remove the clean direction completely, so only tiles that help with the
                # haze-specific error get rewarded and not the shared part
                gv = gv - (torch.dot(gv, gc) / gc.dot(gc).clamp_min(1e-30)) * gc
            else:
                gv = gv - gc
        gpool = per_sample_head_grads(W, pf, pl, dev)                      # (N, 2C)
        score = (eta * (gpool @ gv)).numpy()                               # RAW inner product

        if lam > 0:
            reg = facility_marginal(sim, best, np.arange(len(cand)), w)
            # Scale lambda once at round 0 so both terms contribute about equally (as in the paper).
            if lam_scale is None or lam_scale_per_round:
                if lam_scale_mode == "std":
                    # For a ranking only the spread matters (cv: regulariser 0.24, score 2.81),
                    # so 'std' matches the spread.
                    s_sd = float(score.std()) or 1.0
                    r_sd = float(reg.std()) or 1.0
                else:
                    s_sd = np.abs(score).mean() or 1.0     # legacy: what glister_index.json used
                    r_sd = np.abs(reg).mean() or 1.0
                lam_scale = s_sd / r_sd
                print(f"  [diag] mode={lam_scale_mode} | "
                      f"score mean {score.mean():+.4e} std {score.std():.4e} "
                      f"(cv {abs(score.std()/(score.mean() or 1e-30)):.3f}) | "
                      f"reg mean {reg.mean():+.4e} std {reg.std():.4e} "
                      f"(cv {abs(reg.std()/(reg.mean() or 1e-30)):.3f}) | "
                      f"lam_scale {lam_scale:.4e} -> reg contributes std "
                      f"{lam*lam_scale*reg.std():.4e} vs score std {score.std():.4e}")
            score = score + lam * lam_scale * reg

        take = min(per_round, top_k - len(selected))
        order = np.argsort(-np.where(remaining, score, -np.inf))[:take]
        selected.extend(order.tolist())
        remaining[order] = False
        for e in order:
            best = np.maximum(best, sim[:, e] if w is None else w * sim[:, e])

        # line 9 - step theta along the selected batch's LL gradient (= -loss gradient)
        step = gpool[order].mean(0).to(dev).reshape(W.shape)
        W = W - eta * step
        print(f"  [round {t+1}/{r_rounds}] +{take} (total {len(selected)}) | "
              f"score mean {score[order].mean():+.4e} | |W-W0| "
              f"{(W-W0).norm().item():.4e} | {time.time()-t0:.0f}s", flush=True)

    sel = [cand[i] for i in selected]
    print(f"[glister] selected {len(sel)} in {time.time()-t0:.0f}s")

    ff = np.array([_f(rw, "flood_frac") for rw in cand])
    print(f"  flood_frac  candidates {ff.mean():.3f} -> selected {ff[selected].mean():.3f} "
          f"(the r=1/cosine variant collapsed this toward 0 — the prevalence artefact)")

    index = {
        "n_total": len(rows), "n_candidates": len(cand), "n_curated": len(sel),
        "min_ff": min_ff, "max_ff": max_ff, "min_conf_std": min_conf_std,
        "score_fn": (
            f"GLISTER GreedyTaylorApprox (Killamsetty et al. AAAI'21, Alg. 2), "
            f"last-layer approximation, RAW inner product, r={r_rounds}, "
            f"lambda={lam} (facility-location regularizer = F-GLISTER, "
            f"weights={reg_weights}), eta={eta}, val_haze={val_haze}, "
            f"ref={ref}; same pre-filter as diversity_index"
        ),
        "reg_weights": reg_weights,
        "val_haze": val_haze,
        "lam_scale_mode": lam_scale_mode,
        "val_haze_mode": val_haze_mode,
        "lam_scale_per_round": lam_scale_per_round,
        "files": [rw["file"] for rw in sel],
        "scores": list(range(len(sel), 0, -1)),   # selection order (round-major)
    }
    out = base / out_name
    json.dump(index, open(out, "w"), indent=2)
    print(f"[glister] wrote {out}")
    for other in (f"diversity_N{len(sel)}_index.json", "r2core_index.json",
                  "glister_index.json"):
        if other == out_name:
            continue
        p = base / other
        if p.exists():
            o = set(json.load(open(p))["files"])
            ov = len(o & set(index["files"]))
            print(f"  overlap with {other}: {ov}/{len(sel)} ({ov/len(sel):.1%})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pseudo_dir", default="data/pseudo_gdacs")
    p.add_argument("--ref", default="runs/ablations_sen1/sen_ofa_N250_s42/best.pt",
                   help="reference student theta^(0) (GT-trained, never saw the pool)")
    p.add_argument("--arch", default="efficientvit_b0")
    p.add_argument("--top_k", type=int, default=2500)
    p.add_argument("--r", type=int, default=25,
                   help="greedy rounds (Alg. 2 'r'); k/r elements per round")
    p.add_argument("--lam", type=float, default=1.0,
                   help="lambda for the facility-location regularizer; 0 = plain GLISTER")
    p.add_argument("--eta", type=float, default=3e-4, help="learning rate eta in Alg. 2")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_name", default="glister_index.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N candidates")
    p.add_argument("--reg_weights", choices=["uniform", "r2core"], default="uniform",
                   help="(A) Weights of the facility-location term: 'uniform' (F-GLISTER) or "
                        "'r2core' (needs consistency.csv)")
    p.add_argument("--val_haze", type=float, default=0.0,
                   help="(B) haze strength h applied to the OUTER objective's validation "
                        "set (reflectance units, Makarau 2014). 0 = clean V = plain GLISTER.")
    p.add_argument("--min_ff", type=float, default=MIN_FF,
                   help="pre-filter floor on flood_frac. 0.0 disables it, letting Alg. 2 "
                        "rank the FULL pool including negatives.")
    p.add_argument("--max_ff", type=float, default=MAX_FF)
    p.add_argument("--min_conf_std", type=float, default=MIN_CONF_STD)
    p.add_argument("--consistency_name", default="consistency.csv")
    p.add_argument("--val_haze_mode", choices=["absolute", "excess", "orthogonal"], default="absolute",
                   help="(B) 'absolute': hazed validation loss; 'excess': g_V(h)-g_V(0), the "
                        "gradient of the robustness gap")
    p.add_argument("--lam_scale_per_round", action="store_true",
                   help="(A) Re-fit the lambda scale in every round (default: round 0 only)")
    p.add_argument("--lam_scale_mode", choices=["mean", "std"], default="mean",
                   help="Lambda scaling: 'mean' (default, used for glister_index.json) or 'std'")
    a = p.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    run(a.pseudo_dir, a.ref, a.arch, a.top_k, a.r, a.lam, a.eta, a.gpu,
        a.out_name, a.batch_size, a.num_workers, a.limit,
        reg_weights=a.reg_weights, val_haze=a.val_haze,
        consistency_name=a.consistency_name, lam_scale_mode=a.lam_scale_mode,
        val_haze_mode=a.val_haze_mode, lam_scale_per_round=a.lam_scale_per_round,
        min_ff=a.min_ff, max_ff=a.max_ff, min_conf_std=a.min_conf_std)


if __name__ == "__main__":
    main()
