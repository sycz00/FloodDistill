import json, glob, os
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/results"
names = ["A1_W4A8_B1_s42","A1_W4A8_B2_s42","A3_W4A8_B1_s42","A3_W4A8_B2_s42","A3_W8A8_B1_s42","A3_W8A8_B2_s42",
         "A3_W4A4_B1_s42","A3_W4A4_B2_s42","A3_W4A4_B3_s42"]
for n in names:
    d = json.load(open(f"{R}/{n}.json"))
    eps = d["epochs"]
    tr = [e for e in eps if e.get("phase") == "train"]
    va = [e for e in eps if e.get("phase") != "train"]
    print(f"== {n}  selected={d.get('selected_updates')} completed={d.get('completed_updates')} n_epoch_rows={len(eps)} (train {len(tr)}, other {len(va)})")
    print("  train: epoch  pre_clip_grad_norm_mean   clipped_frac   loss_focal  loss_kd  loss_feature")
    for e in tr:
        print(f"   {e['epoch']:>2d}  {e.get('pre_clip_grad_norm_mean', float('nan')):>14.4g}  {e.get('clipped_step_fraction', float('nan')):>6.3f}   {e.get('loss_focal', float('nan')):.4f}  {e.get('loss_kd', float('nan')):.4f}  {e.get('loss_feature', 0):.4f}")
    if va:
        print("  non-train rows keys:", list(va[0].keys())[:14])
        for e in va:
            print("   ", {k: (round(v,5) if isinstance(v,float) else v) for k, v in e.items() if k in ("epoch","phase","updates","val_mIoU","mIoU","miou","water_iou")})
