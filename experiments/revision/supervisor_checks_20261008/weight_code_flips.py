"""Read-only CPU check: how far can QAT move the latent weights relative to the quantization step?

For each conv weight: step = fixed per-output-channel _w_scale stored in the QAT checkpoint.
code(w) = clamp(round(w/step), qmin, qmax). Compare the QAT start (FP warmup checkpoint, the
documented common initial weights) with the QAT end (best.pt / last.pt).
"""
import sys, json, torch
D = "/data1/schmalstieg/flooddistill/revision_20261005/compression_v2_20261007"

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck["model"], ck

def analyse(fp_path, qat_path, wbits):
    fp, _ = load(fp_path)
    q, ck = load(qat_path)
    qmin, qmax = -(2 ** (wbits - 1)), 2 ** (wbits - 1) - 1
    tot = changed = 0
    rows = []
    dw_over_step = []
    chan_steps = []
    missing = 0
    for k, w in q.items():
        if not k.endswith(".weight"):
            continue
        sk = k[:-len("weight")] + "_w_scale"
        if sk not in q:
            continue
        if k not in fp:
            missing += 1
            continue
        w0 = fp[k].float()
        w1 = w.float()
        step = q[sk].float().view(-1, *([1] * (w1.dim() - 1)))
        c0 = torch.clamp(torch.round(w0 / step), qmin, qmax)
        c1 = torch.clamp(torch.round(w1 / step), qmin, qmax)
        n = w1.numel()
        ch = int((c0 != c1).sum())
        tot += n; changed += ch
        d = (w1 - w0).abs()
        dw_over_step.append((d / step).flatten())
        chan_steps.append(step.flatten())
        rows.append((k, n, ch / n, float(d.max()), float(step.min()), float(step.median()), float(step.max())))
    dos = torch.cat(dw_over_step)
    st = torch.cat(chan_steps)
    out = dict(
        wbits=wbits, total_weights=tot, changed_codes=changed, frac_changed=changed / tot,
        mean_abs_dw_over_step=float(dos.mean()), p99_abs_dw_over_step=float(torch.quantile(dos, 0.99)),
        max_abs_dw_over_step=float(dos.max()),
        median_step=float(st.median()), min_step=float(st.min()),
        max_abs_dw=max(r[3] for r in rows),
        n_tensors=len(rows), missing_in_fp=missing)
    return out, rows

cells = [
    ("A3_W8A8_B1_s42", 8), ("A3_W4A8_B1_s42", 4), ("A3_W4A8_B2_s42", 4),
    ("A3_W4A4_B1_s42", 4), ("A3_W4A4_B2_s42", 4), ("A3_W4A4_B3_s42", 4),
    ("A1_W4A8_B1_s42", 4), ("A1_W4A8_B2_s42", 4),
]
res = {}
for name, wb in cells:
    g = name.split("_")[0]
    fp = f"{D}/{g}_FP_s42/best.pt"
    for which in ("best", "last"):
        qp = f"{D}/{name}/{which}.pt"
        try:
            out, rows = analyse(fp, qp, wb)
            res[f"{name}:{which}"] = out
            print(f"{name:16s} {which:4s} W{wb} tensors={out['n_tensors']:3d} codes_changed={out['frac_changed']:.3%}  "
                  f"mean|dw|/step={out['mean_abs_dw_over_step']:.4f}  p99={out['p99_abs_dw_over_step']:.3f}  "
                  f"max={out['max_abs_dw_over_step']:.2f}  max|dw|={out['max_abs_dw']:.5f}  median_step={out['median_step']:.4f}")
        except Exception as e:
            print(name, which, "ERR", repr(e)[:300])
json.dump(res, open(sys.argv[1], "w"), indent=1)
