import json
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/results"
for n in ["A0_FP_s42","A1_FP_s42","A3_FP_s42"]:
    d = json.load(open(f"{R}/{n}.json"))
    print("==", n, "keys:", list(d.keys())[:25])
    eps = d.get("epochs", [])
    tr = [e for e in eps if e.get("phase") == "train"]
    print("  n train rows", len(tr))
    for e in tr[:3] + tr[-3:]:
        print("   ", e["epoch"], {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in e.items() if k in ("pre_clip_grad_norm_mean","clipped_step_fraction","loss_total","loss_focal","loss_kd")})
    import statistics as st
    g = [e.get("pre_clip_grad_norm_mean") for e in tr if e.get("pre_clip_grad_norm_mean") is not None]
    c = [e.get("clipped_step_fraction") for e in tr if e.get("clipped_step_fraction") is not None]
    if g: print("   pre_clip_grad_norm_mean: median", float(f"{st.median(g):.4g}"), "max", float(f"{max(g):.4g}"), "| clipped fraction mean", float(f"{st.mean(c):.3f}"))
