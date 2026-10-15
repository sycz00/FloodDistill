"""Repeat B0 s42 QAT with torch.set_num_threads(2), interop 1 (the 14 Oct setting)."""
import os, sys, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.argv = ["x"]; sys.path.insert(0, str(HERE.parent))
import torch
import run_qat_seeds as r
torch.set_num_threads(2); torch.set_num_interop_threads(1)
protocol = json.load(open(HERE.parent / "B0_s42/protocol.json"))
root = HERE / "threads2"
if root.exists(): shutil.rmtree(root)
shutil.copytree(HERE.parent / "B0_s42", root, symlinks=True, ignore=shutil.ignore_patterns("qat128"))
q = r.jip.qat(root, "B0", protocol, train=True)
print("threads2: mIoU", q["evaluation"]["mIoU"], "delta", q["comparison"]["pooled_delta"], "ckpt", q["checkpoint_sha256"][:12])
