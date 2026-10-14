#!/usr/bin/env python3
"""Source-only feasibility for V10 (S11 validation = the selection set; CPU; read-only).

Inside the teacher edge band (k=2), compare candidate edge targets against the human label:
teacher water probability, unmixed water fraction f, MNDWI, WI2015, AWEIsh (AUROC, Brier for the
probability-like targets, hard accuracy at 0.5 / published 0). Also reports the 2x2 block
score of each band (20 m bands copied by nearest neighbour score ~1) on S11 val and GDACS tiles.
Writes feasibility_report.txt.
"""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v10_reference as v  # noqa: E402

REV = HERE.parent.parent
SPEC = json.loads((REV / 'capacity_v7_20261011T210730Z' / 'training_spec.json').read_text())


def auroc(score, label):
    import numpy as np
    s, y = np.asarray(score, np.float64), np.asarray(label, bool)
    if y.all() or (~y).all():
        return float('nan')
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import numpy as np
    import torch
    sys.path.insert(0, SPEC['capsule'])
    torch.set_num_threads(16)
    import scripts.compression_jobs as cj
    from models.teacher import build_teacher
    teacher = build_teacher(variant='prithvi_eo_v2_300_tl', num_classes=2, pretrained=False, freeze=True,
                            official_ckpt=SPEC['teacher']).eval()
    val, _ = cj.loaders(SPEC)
    ds = val.dataset
    rows = {k: [] for k in ('label', 'p_teacher', 'f', 'MNDWI', 'WI2015', 'AWEIsh')}
    usable_share = []
    grid = {b: [] for b in range(6)}
    for i in range(len(ds)):
        it = ds[i]
        x = it['image'].float().numpy(); y = it['mask'].numpy() if torch.is_tensor(it['mask']) else np.asarray(it['mask'])
        with torch.no_grad():
            z = teacher(torch.from_numpy(x)[None])
        p = torch.softmax(z, 1)[0, 1].numpy(); tm = p > 0.5
        f, used = v.unmix_fraction(x, tm, k=2)
        band = v.edge_distance(tm) <= 2
        rho = v.reflectance(x); idx, ok = v.indices(rho)
        sel = band & (y != -1) & used
        usable_share.append((int((band & (y != -1) & used).sum()), int((band & (y != -1)).sum())))
        rows['label'].append(y[sel] == 1); rows['p_teacher'].append(p[sel]); rows['f'].append(f[sel])
        for k in ('MNDWI', 'WI2015', 'AWEIsh'):
            rows[k].append(idx[k][sel])
        for b in range(6):
            grid[b].append(v.native_grid_score(rho[b]))
    R = {k: np.concatenate(vv) for k, vv in rows.items()}
    y = R['label']
    rep = [f'S11 validation, teacher edge band k=2, usable unmixing pixels {sum(a for a, _ in usable_share):,} of {sum(b for _, b in usable_share):,}; '
           f'water share in band {y.mean():.3f}']
    for k in ('p_teacher', 'f', 'MNDWI', 'WI2015', 'AWEIsh'):
        s = R[k]; thr = 0.5 if k in ('p_teacher', 'f') else 0.0
        acc = ((s > thr) == y).mean()
        extra = f', Brier {np.mean((s - y) ** 2):.4f}' if k in ('p_teacher', 'f') else ''
        rep.append(f'  {k:10s} AUROC {auroc(s, y):.4f}, accuracy at {thr:g}: {acc:.4f}{extra}')
    names = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
    rep.append('2x2 block-identity score on S11 val (1 = nearest-neighbour 20 m copy): ' + ', '.join(f'{names[b]} {np.mean(grid[b]):.3f}' for b in range(6)))
    # GDACS training tiles (first 50 of the restored cohort)
    files = sorted(Path(SPEC['pseudo']).glob('pseudo_*.pt'))[:50]
    gg = {b: [] for b in range(6)}
    for fp in files:
        a = torch.load(fp, map_location='cpu', weights_only=True)
        rho = v.reflectance(a['image'].numpy())
        for b in range(6):
            gg[b].append(v.native_grid_score(rho[b]))
    rep.append('2x2 block-identity score on 50 GDACS tiles: ' + ', '.join(f'{names[b]} {np.mean(gg[b]):.3f}' for b in range(6)))
    (HERE / 'feasibility_report.txt').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
