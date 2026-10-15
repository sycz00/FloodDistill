"""(1) Re-evaluate the committed 14 Oct B0 QAT ONNX with today's evaluator; (2) repeat today's B0 s42 QAT twice."""
import os, sys, json, shutil, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.argv = ["x"]
sys.path.insert(0, str(HERE.parent))
import run_qat_seeds as r
jip = r.jip
from quantization.jetson_integer import compare_metrics
base = json.load(open(HERE.parent / "B0_s42/bundle/B0/float_evaluation.json"))
ev = jip.evaluate("/home/schmalstieg/repos/FloodDistill/jetson_repro/integer_models/evit_b0_int8_qat.onnx",
                  HERE.parent / "B0_s42/bundle/source_val", HERE / "committed_eval.json")
print("committed 14 Oct ONNX today: mIoU", ev["mIoU"], "delta", compare_metrics(ev["rows"], base["rows"])["pooled_delta"])
protocol = json.load(open(HERE.parent / "B0_s42/protocol.json"))
for k in (1, 2):
    root = HERE / f"rep{k}"
    if root.exists(): shutil.rmtree(root)
    shutil.copytree(HERE.parent / "B0_s42", root, symlinks=True, ignore=shutil.ignore_patterns("qat128"))
    q = jip.qat(root, "B0", protocol, train=True)
    print(f"repeat {k}: mIoU", q["evaluation"]["mIoU"], "delta", q["comparison"]["pooled_delta"], "ckpt", q["checkpoint_sha256"][:12])
