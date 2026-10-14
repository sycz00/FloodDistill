import torch
R = "/home/schmalstieg/repos/FloodDistill/experiments/revision/compression_v2_20261007/results"
for cell in ("A1_W4A8", "A3_W4A8", "A3_W4A4"):
    d = torch.load(f"{R}/{cell}_interaction.pt", map_location="cpu", weights_only=True)
    Cf = d["moments"]["Cf"]; Cq = d["moments"]["Cq"]; Rm = d["moments"]["R"]
    diag = torch.diagonal(Cf)
    # residual energy per PC (teacher PC has unit variance on the fit pool => total energy 32)
    print(f"\n{cell}: tr(Cf)={float(Cf.trace()):.2f} of 32 -> mean explained fraction {1-float(Cf.trace())/32:.3f}")
    print("  residual energy per standardized PC (1.0 = nothing explained), PCs 1..32:")
    print("  ", " ".join(f"{float(v):.2f}" for v in diag))
    print("  PCs 1-4 mean residual: %.3f | PCs 5-16: %.3f | PCs 17-32: %.3f" % (diag[:4].mean(), diag[4:16].mean(), diag[16:].mean()))
    dq = torch.diagonal(Cq)
    print("  quantization displacement energy per PC (Cq diag), PCs 1..8:", " ".join(f"{float(v):.2f}" for v in dq[:8]), "| mean(17-32): %.2f" % dq[16:].mean())
    ev = torch.linalg.eigvalsh(Rm.float())
    print("  R eigenvalues: top5 =", [round(float(v),3) for v in ev[-5:].flip(0)], " bottom5 =", [round(float(v),3) for v in ev[:5]])
