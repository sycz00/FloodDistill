#!/usr/bin/env python
"""Build container commands for a model, objective and label source.

Checks input paths and dispatches to scripts/train.py or scripts/quantize.py.
Training recipes follow experiments/nsweep_pseudo_vs_gt.sh and repro_n2500.sh.
Use --dry_run to inspect commands before running them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.paths import dataset_root  # noqa: E402

# --model -> --student_arch
MODELS = {
    "evit":       "efficientvit_b0",
    "evit_relu6": "efficientvit_b0_relu6",
    "unet":       "unet_resnet50",
}

# Table I has five architecture/objective cells. UNet-OFA is rejected: OFALoss needs the
# EfficientViT stage names and would run as plain Logit-KD on a UNet.
OBJECTIVES = ("ce", "lkd", "ofa")

# The cells "--mode all" runs: paper Table I (both label sources) and Table II (pseudo).
CELLS = [
    ("evit", "ce"), ("evit", "lkd"), ("evit", "ofa"),
    ("unet", "ce"), ("unet", "lkd"),
]

# evit_relu6 is the deployment model (paper Sec. III-D), trained from scratch because
# TensorRT 8.4 per-tensor INT8 does not work with Hardswish. It is not in Table I, so
# "all" does not train it.
TRAINABLE = CELLS + [("evit_relu6", o) for o in OBJECTIVES]

# The shared recipe: experiments/nsweep_pseudo_vs_gt.sh TRAIN_NOGPU
COMMON = ["--lr", "5e-5", "--warmup_epochs", "0", "--amp_dtype", "bf16",
          "--no_amp", "--deterministic"]

# Arm names used by the shell recipes. dfkd means Logit-KD.
ARM = {"ce": "ce", "lkd": "dfkd", "ofa": "ofa"}


def run_dir(out, dataset, model, objective, n, seed):
    """Same layout as runs/ in experiments/*.sh, so perscene_eval.py, nsweep_collect.py
    and the figure scripts find container runs under the same names."""
    arm = ARM[objective]
    if model == "evit_relu6":
        # Keep "relu6" in the name; eval_ablations.detect_arch reads the architecture from it.
        if dataset == "pseudo":
            return os.path.join(out, "repro", f"repro_{arm}_relu6_N{n}_s{seed}")
        return os.path.join(out, "ablations_sen1", f"sen_{arm}_relu6_N{n}_s{seed}")
    if model == "unet":
        arm += "_unet"
    if dataset == "pseudo":
        return os.path.join(out, "nsweep", f"pseudo_{arm}_aug_N{n}_s{seed}")
    return os.path.join(out, "ablations_sen1", f"sen_{arm}_N{n}_s{seed}")


def _ensure_out(out):
    try:
        os.makedirs(out, exist_ok=True)
    except PermissionError:
        die(f"cannot create --out {out}. Mount a writable directory there, "
            f"e.g. -v $PWD/runs:{out}")


def die(msg):
    print(f"\n[flooddistill] {msg}\n", file=sys.stderr)
    sys.exit(2)


def check_s11(data_root):
    p = Path(data_root)
    split = p / "v1.1" / "splits" / "flood_handlabeled" / "flood_train_data.txt"
    if not split.exists():
        die(f"Sen1Floods11 not found under {p}.\n"
            f"  expected: {split}\n"
            f"  --data_root must be the directory CONTAINING v1.1/, not v1.1 itself\n"
            f"  (terratorch appends v1.1 internally).\n"
            f"  Mount it with:  -v /host/sen1floods11:/data/sen1floods11")


def check_pseudo(pseudo_dir, index_file, n, cell):
    """Check that the pseudo tiles and the exact-N index exist before training."""
    import json

    p = Path(pseudo_dir)
    if not p.is_dir():
        die(f"pseudo-label pool not found: {p}\n"
            f"  required by: {cell}\n"
            f"  Build it (needs the teacher + the unlabeled GDACS tiles, see\n"
            f"  docs/01_datasets.md and docs/EXTERNAL_SETUP.md), then mount it at\n"
            f"  {p} with  -v /host/pseudo_gdacs:{p}")

    idx = p / index_file
    if not idx.exists():
        have = sorted(f.name for f in p.glob("*_index.json"))
        die(f"curation index not found: {idx}\n"
            f"  required by: {cell}\n"
            f"  indices present in this pool: {have or 'none'}\n"
            f"  Build one with:  python scripts/curate.py diversity "
            f"--pseudo_dir {p} --top_k {n} --out_name {index_file}")

    files = json.load(open(idx))["files"]
    if len(files) != n:
        die(f"index {index_file} holds {len(files)} entries, expected {n}.\n"
            f"  PseudoLabeledDataset draws n_samples uniformly at random from the index\n"
            f"  and never re-reads the ranking, so a larger index silently degrades to a\n"
            f"  random draw and the curation factor disappears without an error.\n"
            f"  Rebuild at --top_k {n}.")

    missing = [f for f in files if not (p / f).exists()]
    if missing:
        die(f"{len(missing)} of {len(files)} pseudo-label tiles named by {index_file} are\n"
            f"  missing from {p} (first: {missing[0]}).\n"
            f"  required by: {cell}\n"
            f"  Neither the tiles nor the indices are shipped. Generate the tiles with:\n"
            f"    python scripts/train.py pseudo generate --official_ckpt hf \\\n"
            f"        --unlabeled_root <gdacs tiles> --unlabeled_source flood \\\n"
            f"        --out_dir {p} --confidence_threshold 0.80 --store_pixel_confidence")


def train_argv(a, model, objective, out_dir):
    arch = MODELS[model]
    argv = [sys.executable, "scripts/train.py"]
    base = ["--data_root", a.data_root, *COMMON,
            "--epochs", str(a.epochs), "--gpu_ids", *[str(g) for g in a.gpu_ids],
            "--seed", str(a.seed), "--out_dir", out_dir,
            "--wandb_run_name", Path(out_dir).name]
    if a.batch_size:
        base += ["--batch_size", str(a.batch_size)]
    if a.num_workers is not None:
        base += ["--num_workers", str(a.num_workers)]

    if a.dataset == "pseudo":
        base += ["--pseudo_dir", a.pseudo_dir, "--no_labeled",
                 "--n_samples", str(a.n),
                 "--curated_pseudo", "--curated_index_file", a.index_file,
                 "--pseudo_geom_aug", "--pseudo_mult_noise"]
    elif a.n < 252:
        # 252 is the whole split; passing the flag at 252 would drop a seed-dependent image
        base += ["--n_labeled_samples", str(a.n)]

    if objective == "ce":
        argv += ["finetune", "--student_arch", arch, "--target", "student",
                 "--task_loss", "ce"]
    elif objective == "lkd":
        argv += ["distill", "--official_ckpt", a.teacher, "--student_arch", arch,
                 "--alpha", "0.5", "--temperature", "4.0", "--feat_alpha", "0.0",
                 "--task_loss", "ce"]
    else:
        argv += ["ofa", "--official_ckpt", a.teacher, "--student_arch", arch,
                 "--alpha", "0.5", "--temperature", "4.0", "--ofa_alpha", "0.3",
                 "--gamma", "1.2", "--grad_clip", "5.0"]
    return argv + base + a.extra


def eval_argv(a, model, out_dir):
    argv = [sys.executable, "scripts/train.py", "eval",
            "--student_arch", MODELS[model], "--target", "student",
            "--model_ckpt", a.ckpt, "--data_root", a.data_root,
            "--gpu_ids", *[str(g) for g in a.gpu_ids], "--out_dir", out_dir]
    if a.batch_size:
        argv += ["--batch_size", str(a.batch_size)]
    if a.num_workers is not None:
        argv += ["--num_workers", str(a.num_workers)]
    return argv + a.extra


def export_argv(a, model, out_dir):
    argv = [sys.executable, "scripts/quantize.py", "qat",
            "--model_ckpt", a.ckpt, "--student_arch", MODELS[model],
            "--data_root", a.data_root, "--no_bn_reset",
            "--qat_epochs", str(a.qat_epochs), "--trt_compat", a.trt_compat,
            "--calibration_batches", str(a.calibration_batches),
            "--gpu_ids", *[str(g) for g in a.gpu_ids],
            "--seed", str(a.seed), "--out_dir", out_dir]
    if a.batch_size:
        argv += ["--batch_size", str(a.batch_size)]
    if a.num_workers is not None:
        argv += ["--num_workers", str(a.num_workers)]
    if a.pseudo_dir and os.path.isdir(a.pseudo_dir) and a.qat_epochs > 0:
        argv += ["--pseudo_dir", a.pseudo_dir, "--curated_pseudo",
                 "--curated_index_file", a.index_file, "--n_samples", str(a.n)]
    return argv + a.extra


def run(argv, dry):
    print("\n$ " + " ".join(argv), flush=True)
    if dry:
        return 0
    return subprocess.call(argv, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(
        prog="flooddistill",
        description="Run a FloodDistill experiment cell inside the container.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
supported cells (paper Table I / II):

  model  objective  S11  pseudo
  evit   ce          y     y
  evit   lkd         y     y
  evit   ofa         y     y
  unet   ce          y     y
  unet   lkd         y     y
  unet   ofa         -     -     OFA branch heads are EfficientViT-specific

examples:

  # one cell, written to /out/nsweep/pseudo_ofa_aug_N2500_s42
  --mode train --model evit --objective ofa --dataset pseudo --n 2500 --out /out

  # every supported cell at the matched budget, 1 seed
  --mode all --dataset s11 --n 252

  # evaluate a checkpoint on S11 val + test
  --mode evaluate --model evit --ckpt /out/nsweep/pseudo_ofa_aug_N2500_s42/best.pt --out /out/eval

  # trained checkpoint -> QAT -> ONNX QDQ
  --mode export --model evit_relu6 --ckpt /out/repro/repro_ofa_relu6_N2500_s42/best.pt --out /out/qat
""")
    p.add_argument("--mode", required=True,
                   choices=["train", "distill", "evaluate", "export", "all"])
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--objective", choices=OBJECTIVES)
    p.add_argument("--dataset", default="s11", choices=["s11", "pseudo"])
    p.add_argument("--ckpt", help="Checkpoint to evaluate or export (best.pt).")
    p.add_argument("--out", default="/out", help="Output directory (mount it).")
    p.add_argument("--data_root", default=str(dataset_root("S11_ROOT")),
                   help="Directory containing v1.1/ (not v1.1 itself).")
    p.add_argument("--pseudo_dir", default=os.environ.get("PSEUDO_DIR",
                                                          "/data/pseudo_gdacs"))
    p.add_argument("--index_file", default=None,
                   help="Curation index inside --pseudo_dir. Default: diversity_N<n>_index.json")
    p.add_argument("--teacher", default="hf",
                   help="'hf' downloads the official Prithvi Sen1Floods11 checkpoint "
                        "(this is the paper's teacher), or pass a local .pt path.")
    p.add_argument("--n", type=int, default=252,
                   help="Training budget: 252 = paper Table I, 2500 = Table II.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Train each cell once per seed (paper uses 42 64 91 107). "
                        "Overrides --seed for train/distill/all.")
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--qat_epochs", type=int, default=10,
                   help="0 = calibrate and export only, no QAT fine-tuning.")
    p.add_argument("--calibration_batches", type=int, default=200)
    p.add_argument("--trt_compat", default="jp50", choices=["jp50", "modern"])
    p.add_argument("--dry_run", action="store_true", help="Print the commands, run nothing.")
    p.add_argument("extra", nargs="*", default=[],
                   help="Extra flags passed through to the underlying CLI.")
    a = p.parse_args()

    if a.index_file is None:
        a.index_file = f"diversity_N{a.n}_index.json"

    # Validate arguments before creating --out, so errors are not hidden by a permission error.
    if a.mode in ("train", "distill", "all"):
        check_s11(a.data_root)

    if a.mode == "all":
        _ensure_out(a.out)
        cells = CELLS
        if a.model:
            cells = [c for c in cells if c[0] == a.model]
        if a.objective:
            cells = [c for c in cells if c[1] == a.objective]
        if not cells:
            die(f"no supported cell matches --model {a.model} --objective {a.objective}")
        if a.dataset == "pseudo":
            check_pseudo(a.pseudo_dir, a.index_file, a.n, "every pseudo-label cell")
        seeds = a.seeds or [a.seed]
        print(f"[flooddistill] all: {len(cells)} cells x {len(seeds)} seed(s) "
              f"{seeds}, {a.dataset}, N={a.n}, {a.epochs} epochs")
        failed = []
        for seed in seeds:
            a.seed = seed
            for model, obj in cells:
                out = run_dir(a.out, a.dataset, model, obj, a.n, seed)
                if os.path.exists(os.path.join(out, "best.pt")):
                    print(f"[have] {out}")
                    continue
                if run(train_argv(a, model, obj, out), a.dry_run) != 0:
                    failed.append(f"{model}/{obj}/s{seed}")
        if failed:
            die(f"failed cells: {', '.join(failed)}")
        print(f"\n[flooddistill] all done -> {a.out}")
        return

    if a.mode in ("train", "distill"):
        if not a.model or not a.objective:
            die("--mode train needs --model and --objective")
        if (a.model, a.objective) not in TRAINABLE:
            if a.model == "unet" and a.objective == "ofa":
                die("UNet-OFA is not a supported cell. The paper trains UNet under CE and\n"
                    "  Logit-KD only, and OFALoss registers its exit branches on the\n"
                    "  EfficientViT backbone: a UNet run gets zero branches and silently\n"
                    "  trains as plain Logit-KD. Use --model evit for OFA, or\n"
                    "  --objective lkd for UNet.")
            die(f"unsupported cell: {a.model} x {a.objective}. Supported: "
                + ", ".join(f"{m}/{o}" for m, o in TRAINABLE))
        if a.mode == "distill" and a.objective == "ce":
            die("--mode distill is for lkd/ofa; CE involves no teacher. "
                "Use --mode train --objective ce.")
        if a.dataset == "pseudo":
            check_pseudo(a.pseudo_dir, a.index_file, a.n,
                         f"{a.model}/{a.objective} on the pseudo-label pool")
        _ensure_out(a.out)
        rc = 0
        for seed in (a.seeds or [a.seed]):
            a.seed = seed
            out = run_dir(a.out, a.dataset, a.model, a.objective, a.n, seed)
            if os.path.exists(os.path.join(out, "best.pt")):
                print(f"[have] {out}")
                continue
            rc = run(train_argv(a, a.model, a.objective, out), a.dry_run) or rc
        sys.exit(rc)

    if a.mode == "evaluate":
        if not a.model:
            die("--mode evaluate needs --model (the checkpoint does not record its architecture)")
        if not a.ckpt:
            die("--mode evaluate needs --ckpt <path to best.pt>")
        if not os.path.isfile(a.ckpt):
            die(f"checkpoint not found: {a.ckpt}\n  Mount it, e.g. -v /host/runs:/out")
        check_s11(a.data_root)
        _ensure_out(a.out)
        sys.exit(run(eval_argv(a, a.model, a.out), a.dry_run))

    if a.mode == "export":
        if not a.model:
            die("--mode export needs --model (the checkpoint does not record its architecture)")
        if not a.ckpt:
            die("--mode export needs --ckpt <path to best.pt>")
        if not os.path.isfile(a.ckpt):
            die(f"checkpoint not found: {a.ckpt}\n  Mount it, e.g. -v /host/runs:/out")
        if a.model == "unet":
            die("the INT8 export path targets the EfficientViT student "
                "(--model evit_relu6). UNet is a capacity reference, not a deployment target.")
        if a.model == "evit":
            print("[flooddistill] warning: this checkpoint is Hardswish. TRT 8.4 per-tensor\n"
                  "  INT8 does not hold accuracy on Hardswish (docs/03_quantization.md) -\n"
                  "  the deployable path trains efficientvit_b0_relu6 from scratch. Exporting\n"
                  "  anyway; expect the INT8 numbers to be unrepresentative.")
        # QAT fine-tunes on the pool the checkpoint was trained on; only check when used.
        if a.qat_epochs > 0 and os.path.isdir(a.pseudo_dir):
            check_pseudo(a.pseudo_dir, a.index_file, a.n, "QAT fine-tuning")
        check_s11(a.data_root)
        _ensure_out(a.out)
        rc = run(export_argv(a, a.model, a.out), a.dry_run)
        if rc == 0 and not a.dry_run:
            rc = verify_onnx(a.out, MODELS[a.model])
        sys.exit(rc)


def verify_onnx(out_dir, arch):
    """Run onnx.checker on all exports and compare the float ONNX with PyTorch."""
    import numpy as np
    import onnx

    found = sorted(Path(out_dir).glob("*.onnx"))
    if not found:
        print("[flooddistill] ERROR: export produced no .onnx file", file=sys.stderr)
        return 1
    for f in found:
        m = onnx.load(str(f))
        onnx.checker.check_model(m)
        q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
        print(f"[flooddistill] {f.name:<30} ok  {f.stat().st_size/1024:7.0f} KB  "
              f"QuantizeLinear={q}")

    float_onnx = Path(out_dir) / "model_float.onnx"
    stripped = Path(out_dir) / "model_qat_stripped.pt"
    if not (float_onnx.exists() and stripped.exists()):
        return 0
    try:
        import onnxruntime as ort
        import torch
        from models.student import build_student
    except ImportError as e:
        print(f"[flooddistill] skipping numeric check: {e}")
        return 0

    sd = torch.load(stripped, map_location="cpu", weights_only=True)
    sd = sd.get("student", sd.get("model", sd))
    m = build_student(arch=arch, in_channels=6, num_classes=2, pretrained=False)
    m.load_state_dict(sd, strict=True)
    m.eval()

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 6, 512, 512, generator=g)
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(str(float_onnx), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    d = float(np.abs(ref - got).max())
    ok = d < 1e-3
    print(f"[flooddistill] float ONNX vs checkpoint: max|diff|={d:.2e}  "
          f"{'ok' if ok else 'MISMATCH'}")
    if not ok:
        print("[flooddistill] ERROR: the exported graph does not match the checkpoint",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    main()
