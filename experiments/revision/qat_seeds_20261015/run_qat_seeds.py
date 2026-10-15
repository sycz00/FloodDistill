#!/usr/bin/env python3
"""Four-seed W8A8 QAT for EfficientViT-B0 (ReLU6 OFA, N=2,500) and the 12,230-parameter R4 LKD student.

Extends experiments/revision/jetson_integer_20261014T101741Z, which ran QAT for seed 42 of B0 only
(R4: one-update smoke). Same recipe and the same code: scripts/jetson_integer_prepare.py
(float export + ORT source evaluation, `qat`: BN folded, all convolutions incl. the head W8A8,
per-tensor symmetric activations, per-channel weights, AdamW 1e-4, wd 0.05, batch 4, focal gamma 2,
128 updates on the frozen 512 cached GDACS TRAIN tiles, QAT seed 1729, ONNX export + parity +
graph audit + source-validation evaluation). The frozen calibration / validation / QAT-train arrays
of the 14 Oct run are reused read-only via symlinks, so seeds differ only in the parent checkpoint.
For R4 the selected PTQ variant of that run (percentile, per-channel) is repeated per seed as the
QAT-vs-PTQ comparison. CPU only, like the original run.

Usage: run_qat_seeds.py --model {B0,R4} --seed {42,64,91,107}
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path("/home/schmalstieg/repos/FloodDistill")
HERE = Path(__file__).resolve().parent
SRC = REPO / "experiments/revision/jetson_integer_20261014T101741Z"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC / "code"))                      # rungs.py (R4 architecture) as used on 14 Oct
sys.path.insert(0, "/home/schmalstieg/repos/_deps/efficientvit")
import importlib.util
spec = importlib.util.spec_from_file_location("jip", REPO / "scripts/jetson_integer_prepare.py")
jip = importlib.util.module_from_spec(spec); sys.modules["jip"] = jip; spec.loader.exec_module(jip)

PARENTS = {
    "B0": {s: f"/data1/schmalstieg/flooddistill/runs_repro/pseudo_evit_relu6_ofa_N2500_s{s}/best.pt" for s in (42, 64, 91, 107)},
    "R4": {42: "/data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008/v5/fl_R4_lr5e-3_s42/best.pt",
           **{s: f"/data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008/v5/fl_R4_sel_s{s}/best.pt"
              for s in (64, 91, 107)}},
}


def build(name):
    if name == "R4":
        from rungs import ContextConv
        return ContextConv()
    from models.student import build_student
    return build_student("efficientvit_b0_relu6", pretrained=False, in_channels=6, num_classes=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["B0", "R4"], required=True)
    ap.add_argument("--seed", type=int, choices=[42, 64, 91, 107], required=True)
    a = ap.parse_args()
    import numpy as np, torch, onnx
    from onnxsim import simplify
    from quantization.onnx_qdq import export_float_onnx
    from quantization.jetson_integer import ptq, graph_audit, compare_metrics, session
    torch.set_num_threads(2); torch.set_num_interop_threads(1)   # the 14 Oct setting; QAT outcome depends on it
    name, seed = a.model, a.seed
    root = HERE / f"{name}_s{seed}"
    (root / "bundle" / name).mkdir(parents=True, exist_ok=True)
    (root / "receipts").mkdir(exist_ok=True)
    for rel in ("qat_train", "bundle/calibration", "bundle/calibration_s11", "bundle/source_val"):
        link = root / rel
        if not link.exists():
            link.symlink_to(SRC / rel)
    # portable float checkpoint of this seed's parent
    parent = PARENTS[name][seed]
    raw = torch.load(parent, map_location="cpu", weights_only=False)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    model = build(name); model.load_state_dict(state, strict=True); model.eval()
    portable = root / "bundle" / name / "checkpoint_float.pt"
    torch.save({"model": model.state_dict(), "source": parent, "source_sha256": jip.file_hash(parent)}, portable)
    protocol = json.loads((SRC / "protocol.json").read_text())
    protocol["models"] = {name: dict(protocol["models"][name], checkpoint=parent, sha256=jip.file_hash(parent),
                                     portable_checkpoint=str(portable), portable_sha256=jip.file_hash(portable))}
    protocol["seed_extension"] = dict(parent_seed=seed, created=time.time(), source_run=str(SRC),
                                      rule="identical recipe/assets; only the parent checkpoint changes")
    jip.atomic_json(root / "protocol.json", protocol)
    if seed == 42:   # the 14 Oct portable checkpoint must be the same model
        old = torch.load(SRC / "bundle" / name / "checkpoint_float.pt", map_location="cpu", weights_only=True)["model"]
        assert all(torch.equal(old[k], v) for k, v in model.state_dict().items()), "seed-42 parent differs from 14 Oct"
    model = jip.load_model(name, protocol)
    params = sum(p.numel() for p in model.parameters())
    # float export + source evaluation (same steps as jip.experiment)
    target = root / "bundle" / name
    path = target / "model_float.onnx"
    export_float_onnx(model, torch.zeros(1, 6, 512, 512), str(path), opset=13, static=True)
    m, ok = simplify(onnx.load(path)); assert ok; m.ir_version = min(m.ir_version, 10); onnx.save(m, path)
    calibration = np.stack([np.load(p, allow_pickle=False) for p in sorted((root / "bundle/calibration").glob("*.npy"))])
    with torch.no_grad():
        expected = model(torch.from_numpy(calibration[:1])).numpy()
    actual = session(path).run(None, {"image": calibration[:1]})[0]
    assert np.allclose(expected, actual, atol=1e-3, rtol=1e-3), "float export parity failure"
    baseline = jip.evaluate(path, root / "bundle/source_val", target / "float_evaluation.json")
    record = dict(model=name, seed=seed, parent=parent, parameters=params, float=dict(
        mIoU=baseline["mIoU"], water_IoU=baseline["water_IoU"],
        equal_event_mIoU=compare_metrics(baseline["rows"], baseline["rows"])["equal_event_mIoU"]))
    if name == "R4":   # the 14 Oct selected PTQ candidate, repeated per seed
        cfg = next(c for c in protocol["ptq_variants"] if c["id"] == "percentile_pc")
        out = target / "percentile_pc.onnx"
        audit = ptq(path, out, calibration, **{k: v for k, v in cfg.items() if k not in ("id", "calibration")})
        ev = jip.evaluate(out, root / "bundle/source_val", target / "percentile_pc_evaluation.json")
        cmp = compare_metrics(ev["rows"], baseline["rows"])
        record["ptq_percentile_pc"] = dict(mIoU=ev["mIoU"], water_IoU=ev["water_IoU"], pooled_delta=cmp["pooled_delta"],
                                           equal_event_delta=cmp["equal_event_delta"],
                                           acceptable=cmp["pooled_delta"] >= -.01 and cmp["equal_event_delta"] >= -.01
                                           and ev["water_IoU"] >= baseline["water_IoU"] - .02,
                                           jp50_qdq_compatible=audit.get("jp50_qdq_compatible"))
    start = time.monotonic()
    q = jip.qat(root, name, protocol, train=True)
    ev, cmp = q.get("evaluation", {}), q.get("comparison", {})
    record["qat"] = dict(status=q["status"], updates=q["updates"], elapsed_seconds=time.monotonic() - start,
                         mIoU=ev.get("mIoU"), water_IoU=ev.get("water_IoU"), pooled_delta=cmp.get("pooled_delta"),
                         equal_event_delta=cmp.get("equal_event_delta"), pooled_ci95=cmp.get("pooled_delta_whole_event_CI95"), equal_event_ci95=cmp.get("equal_event_delta_whole_event_CI95"),
                         acceptable=q.get("acceptable_runtime_accuracy"),
                         strict_logit_parity=q["parity"].get("strict_logit_parity_pass"),
                         checkpoint=str(root / "bundle" / name / "qat128" / "checkpoint_qat.pt"),
                         onnx=str(root / "bundle" / name / "qat128" / "model_qdq.onnx"))
    jip.atomic_json(root / "summary.json", record)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
