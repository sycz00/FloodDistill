import torch
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"
fp = torch.load(f"{D}/A3_FP_s42/best.pt", map_location="cpu", weights_only=False)["model"]
def rel(cell):
    q = torch.load(f"{D}/{cell}/best.pt", map_location="cpu", weights_only=False)["model"]
    rows = []
    for k, w in q.items():
        if k.endswith("weight") or k.endswith("bias"):
            if k in fp and fp[k].shape == w.shape and fp[k].is_floating_point():
                d = (w.float() - fp[k].float()).norm().item(); n = fp[k].float().norm().item()
                rows.append((k, d / max(n, 1e-12), w.numel()))
    return rows
for cell in ("A3_W8A8_B1_s42", "A3_W4A8_B1_s42", "A3_W4A4_B1_s42"):
    rows = rel(cell)
    head = [r for r in rows if "head." in r[0]]
    body = [r for r in rows if "backbone" in r[0] and r[0].endswith("weight") and r[2] > 20 and "norm" not in r[0]]
    import statistics as st
    print(f"\n{cell}")
    print("  backbone conv weights: median relative change ||dw||/||w|| = %.2e (n=%d)" % (st.median(r[1] for r in body), len(body)))
    for k, r, n in sorted(head, key=lambda t: t[0])[-6:]:
        print(f"  head   {r:.2e}  ({n:5d})  {k}")
