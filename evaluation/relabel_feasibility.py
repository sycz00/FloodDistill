# !/usr/bin/env python3
"""Audit student predictions on paired event and pre-event GDACS acquisitions.

Measure inheritance of teacher water predictions and compare spectral
corroboration: rho_water=IoU(mask,MNDWI water), rho_new=IoU(mask,event water
absent from the baseline). Pre-event water may be permanent surface water;
it is not automatically a teacher error and cannot be the later event's flood.
Neither MNDWI nor acquisition timing supplies independent flood ground truth.

Requires the pseudo pool, provenance and a student checkpoint. --train_index
separates selected from held-out tiles. See supplement Sec. S5.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from torch.utils.data import DataLoader, Dataset
from data.pool_features import _mndwi_water, _iou          # identical definitions
from models.student import build_student

FF_FLOOD = 0.05          # the same flood-fraction threshold the pre-filter uses


class _Tiles(Dataset):
    def __init__(self, base: Path, files: list[str]):
        self.base, self.files = base, files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.base / self.files[i], map_location="cpu", weights_only=True)
        return {"image": d["image"], "mask": d["mask"].to(torch.uint8), "idx": i}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/curation/fps_r1_ofa_hs_N2500_s42/best.pt")
    ap.add_argument("--arch", default="efficientvit_b0")
    ap.add_argument("--pseudo_dir", default=os.environ.get("PSEUDO_DIR", "data/pseudo_gdacs"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    # Splits every audited number by whether the tile is in the training subset. The
    # pre-filter lets pre-event tiles into the candidate pool the N=2,500 subset is drawn
    # from, so some unknown share of the flagged baselines may be training tiles. Then the
    # 97.4% would partly be a fit statistic and not evidence of retention.
    # Default None gives exactly the old behaviour.
    ap.add_argument("--train_index", default=None,
                    help="Path to a training index json (e.g. "
                         "data/pseudo_gdacs/diversity_N2500_index.json). When given, TEST 1 "
                         "is reported for in-training and held-out baselines separately.")
    ap.add_argument("--out", default=None,
                    help="Per-tile CSV path. Default <pseudo_dir>/student_relabel_stats.csv.")
    a = ap.parse_args()

    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    base = Path(a.pseudo_dir)
    rows = list(csv.DictReader(open(base / "pool_features.csv")))
    if a.limit:
        rows = rows[:a.limit]

    def f(r, k, d=np.nan):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else d

    files = [r["file"] for r in rows]
    is_flood = np.array([r.get("is_flood_scene") == "1" for r in rows])
    t_ff = np.array([f(r, "flood_frac", -1) for r in rows])
    t_rw = np.array([f(r, "rho_water", np.nan) for r in rows])
    t_rn = np.array([f(r, "rho_new", np.nan) for r in rows])
    loc = np.array([r.get("location", "") for r in rows])

    model = build_student(a.arch, in_channels=6, num_classes=2, pretrained=False).to(dev).eval()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    print(f"[relabel] student = {a.ckpt}")

    dl = DataLoader(_Tiles(base, files), batch_size=a.batch_size, shuffle=False,
                    num_workers=a.num_workers, pin_memory=True)
    s_ff = np.zeros(len(files)); s_rw = np.zeros(len(files))
    s_masks: dict[int, np.ndarray] = {}
    waters: dict[int, np.ndarray] = {}
    agree = np.zeros(len(files))
    with torch.no_grad():
        done = 0
        for b in dl:
            x = b["image"].to(dev, dtype=torch.float32)
            pred = model(x)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pm = pred.argmax(1).to(torch.uint8).cpu().numpy().astype(bool)
            tm = b["mask"].numpy().astype(bool)
            for j, gi in enumerate(b["idx"].tolist()):
                w = _mndwi_water(b["image"][j])
                s_ff[gi] = pm[j].mean()
                s_rw[gi] = _iou(pm[j], w)
                agree[gi] = _iou(pm[j], tm[j])
                s_masks[gi] = np.packbits(pm[j]); waters[gi] = np.packbits(w)
            done += x.shape[0]
            if done % 4000 < x.shape[0]:
                print(f"  inferred {done}/{len(files)}", flush=True)

    shp = (int(np.sqrt(np.unpackbits(s_masks[0]).size)),) * 2
    npx = shp[0] * shp[1]

    def unp(d, i):
        return np.unpackbits(d[i], count=npx).reshape(shp).astype(bool)

    # rho_new for the student, exactly as pool_features defines it
    s_rn = np.full(len(files), np.nan)
    by_loc: dict[str, list[int]] = {}
    for i, l in enumerate(loc):
        by_loc.setdefault(l, []).append(i)
    for l, idxs in by_loc.items():
        fl = [i for i in idxs if is_flood[i]]
        bl = [i for i in idxs if not is_flood[i]]
        for i in bl:
            s_rn[i] = _iou(unp(s_masks, i), np.zeros(shp, bool))
        if not fl or not bl:
            continue
        perm = unp(waters, bl[0])
        for i in fl:
            s_rn[i] = _iou(unp(s_masks, i), unp(waters, i) & ~perm)

    # TEST 1
    hall = (~is_flood) & (t_ff >= FF_FLOOD)        # teacher hallucinated a flood
    ok_base = (~is_flood) & (t_ff < FF_FLOOD)      # teacher correctly said dry
    print("\n" + "=" * 78)
    print("TEST 1 — does the student inherit the teacher's SYSTEMATIC error?")
    print("=" * 78)
    print(f"  pool {len(files)} | flood-date {is_flood.sum()} | pre-event baselines {(~is_flood).sum()}")
    print(f"  teacher-hallucinated baselines (ff>={FF_FLOOD} on a pre-event scene): {hall.sum()}")
    if hall.sum():
        print(f"    teacher mean ff on them : {t_ff[hall].mean():.4f}")
        print(f"    STUDENT mean ff on them : {s_ff[hall].mean():.4f}")
        print(f"    student ALSO calls flooded (ff>={FF_FLOOD}): "
              f"{(s_ff[hall] >= FF_FLOOD).mean():.1%}  ({int((s_ff[hall]>=FF_FLOOD).sum())}/{int(hall.sum())})")
        print(f"    student mask-vs-teacher-mask IoU on them: {agree[hall].mean():.4f}")
    print(f"  CONTROL — teacher-correct dry baselines: {ok_base.sum()}")
    if ok_base.sum():
        print(f"    student calls flooded (false alarm): "
              f"{(s_ff[ok_base] >= FF_FLOOD).mean():.1%}  (student mean ff {s_ff[ok_base].mean():.4f})")

    # A4: the same two rows, split by training-set membership
    in_train = np.zeros(len(files), bool)
    if a.train_index:
        tr = set(json.load(open(a.train_index))["files"])
        in_train = np.array([fn in tr for fn in files])
        print("\n" + "-" * 78)
        print(f"A4 — split by membership in {a.train_index} ({int(in_train.sum())} tiles)")
        print("-" * 78)
        print(f"  {'subset':22s} {'n':>7s} {'agree%':>8s} {'maskIoU':>9s} {'stu mean ff':>12s}")
        for nm, m in [("flagged, IN training", hall & in_train),
                      ("flagged, HELD OUT",    hall & ~in_train),
                      ("flagged, all",         hall),
                      ("teacher-empty, IN tr", ok_base & in_train),
                      ("teacher-empty, HELD OUT", ok_base & ~in_train),
                      ("teacher-empty, all",  ok_base)]:
            if not m.sum():
                print(f"  {nm:22s} {0:7d}        --        --            --"); continue
            agree_rate = (s_ff[m] >= FF_FLOOD).mean()
            print(f"  {nm:22s} {int(m.sum()):7d} {100*agree_rate:8.2f} "
                  f"{agree[m].mean():9.4f} {s_ff[m].mean():12.4f}")
        print(f"  whole-pool student-vs-teacher mask IoU: {agree.mean():.4f}")

    # TEST 2
    print("\n" + "=" * 78)
    print("TEST 2 — agreement with the teacher-INDEPENDENT physical reference")
    print("=" * 78)
    for nm, m in [("all tiles", np.ones(len(files), bool)),
                  ("flood-date only", is_flood),
                  ("baselines only", ~is_flood)]:
        rw_t, rw_s = t_rw[m], s_rw[m]
        v = m & ~np.isnan(t_rn) & ~np.isnan(s_rn)
        print(f"  {nm:<18} n={m.sum():6d}   "
              f"rho_water  teacher {np.nanmean(rw_t):.4f} -> student {np.nanmean(rw_s):.4f} "
              f"({np.nanmean(rw_s)-np.nanmean(rw_t):+.4f})")
        if v.sum():
            print(f"  {'':<18} n={v.sum():6d}   "
                  f"rho_new    teacher {np.nanmean(t_rn[v]):.4f} -> student {np.nanmean(s_rn[v]):.4f} "
                  f"({np.nanmean(s_rn[v])-np.nanmean(t_rn[v]):+.4f})")
    print(f"\n  overall student-vs-teacher mask IoU: {agree.mean():.4f} "
          f"(1.0 = pure imitation, no new information)")

    out = Path(a.out) if a.out else base / "student_relabel_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "is_flood_scene", "in_train", "teacher_ff", "student_ff",
                    "teacher_rho_water", "student_rho_water",
                    "teacher_rho_new", "student_rho_new", "student_teacher_iou"])
        for i, fn in enumerate(files):
            w.writerow([fn, int(is_flood[i]), int(in_train[i]),
                        f"{t_ff[i]:.6f}", f"{s_ff[i]:.6f}",
                        f"{t_rw[i]:.6f}", f"{s_rw[i]:.6f}",
                        "" if np.isnan(t_rn[i]) else f"{t_rn[i]:.6f}",
                        "" if np.isnan(s_rn[i]) else f"{s_rn[i]:.6f}",
                        f"{agree[i]:.6f}"])
    print(f"\n[relabel] per-tile stats -> {out}")


if __name__ == "__main__":
    main()
