"""QAT, ONNX export and deployment commands.

qat adapts a trained ReLU6 student and exports model_qdq_trt_no_out.onnx.
eval-trt uses local TensorRT bindings; deploy copies exports to a device.
Compile the paper's TensorRT 8.4 engine on the Jetson using jetson/deploy.sh.
best_qat.pt retains QAT state; model_qat_stripped.pt and model_float*.onnx
contain QAT-updated weights without quantizers, not the pre-QAT baseline.
Use docs/03_quantization.md for the reproduction command and calibration rules.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared arg-building helpers

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--student_arch", default="efficientvit_b0",
                   choices=["vit_tiny", "efficientvit_b0", "efficientvit_b0_relu6",
                             "efficientvit_b0_lrelu6", "efficientvit_b0_silu", "efficientvit_b0_tanh",
                             "efficientvit_b1", "efficientvit_b2",
                             "tiny_vit_5m", "tiny_vit_11m", "tiny_vit_21m",
                             "mobilenet_v3_small", "unet_resnet50", "unet_resnet101"])
    p.add_argument("--num_classes",  type=int, default=2)
    from utils.paths import dataset_root
    p.add_argument("--data_root",    default=str(dataset_root("S11_ROOT")),
                   help="Sen1Floods11 root. Default: $DATA_ROOT/sen1floods11.")
    p.add_argument("--batch_size",   type=int, default=1,
                   help="Batch size for calibration/QAT (default 1 for CPU inference).")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--img_size",     type=int, default=224)
    p.add_argument("--eval_img_size",type=int, default=512)
    p.add_argument("--out_dir",      default="./runs/quant")
    p.add_argument("--gpu_ids",      type=int, nargs="+", default=[0])
    p.add_argument("--seed",         type=int, default=None)
    # OOD evaluation datasets (optional; evaluated after PTQ/QAT/TRT when provided)
    p.add_argument("--sturm_root", default=None,
                   help="STURM-Flood dataset root. If set, the model is evaluated on "
                        "STURM-S2 (OOD) after quantization.")
    p.add_argument("--sturm_normalize", default="percentile",
                   choices=["percentile", "clip", "original"],
                   help="STURM normalisation: 'percentile' (default) or 'original' (paper "
                        "protocol, 0.04+0.12*v)")
    p.add_argument("--s1s2w_root", default=None,
                   help="S1S2-Water dataset root. If set, the model is evaluated on "
                        "S1S2W (OOD) after quantization.")


def _build_student(args, num_classes, pretrained=False):
    from models.student import build_student
    return build_student(
        arch        = args.student_arch,
        in_channels = 6,
        num_classes = num_classes,
        pretrained  = pretrained,
    )


def _load_checkpoint(model, ckpt_path):
    import torch
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # trainer.py saves {"model": ...}, strip_qat saves {"student": ...}, bare dicts also accepted
    state = ckpt.get("model", ckpt.get("student", ckpt))
    model.load_state_dict(state, strict=True)
    return model


def _get_dataloaders(args):
    from data.sen1floods11 import get_dataloaders
    return get_dataloaders(
        data_root     = args.data_root,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        img_size      = args.img_size,
        eval_img_size = args.eval_img_size,
    )


def _eval_float_ood(model, args, device, label_prefix=""):
    """Evaluate a float model on STURM and S1S2W if their roots are given ('sturm_mIoU', 's1s2w_mIoU')."""
    import torch
    from utils.metrics import accumulate_metrics, finalize_metrics

    ood_results = {}
    ignore_index = -1

    if getattr(args, "sturm_root", None):
        from data.ood_datasets import get_sturm_flood_dataloader
        _sturm_norm = getattr(args, "sturm_normalize", "percentile")
        print(f"[quantize{label_prefix}] evaluating on STURM-S2 (OOD, normalize={_sturm_norm}) …")
        sturm_dl = get_sturm_flood_dataloader(
            root        = args.sturm_root,
            sensor      = "S2",
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            img_size    = args.eval_img_size,
            normalize   = _sturm_norm,
        )
        from data.channel_adapter import sturm_s2_adapter as _sturm_s2_adapter
        _sadapt = _sturm_s2_adapter(target_channels=6).to(device)
        model.eval()
        conf = None
        with torch.no_grad():
            for batch in sturm_dl:
                images = _sadapt(batch["image"].to(device).float())
                logits = model(images)
                conf = accumulate_metrics(logits.cpu(), batch["mask"].long(),
                                         args.num_classes, ignore_index, conf)
        m = finalize_metrics(conf) if conf is not None else {}
        ood_results["sturm_mIoU"] = m.get("mIoU", 0.0)
        print(f"  STURM-S2 mIoU = {ood_results['sturm_mIoU']:.4f}")

    if getattr(args, "s1s2w_root", None):
        from data.ood_datasets import get_s1s2water_dataloader
        print(f"[quantize{label_prefix}] evaluating on S1S2-Water (OOD) …")
        s1s2w_dl = get_s1s2water_dataloader(
            root        = args.s1s2w_root,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            img_size    = args.eval_img_size,
        )
        model.eval()
        conf = None
        with torch.no_grad():
            for batch in s1s2w_dl:
                logits = model(batch["image"].to(device).float())
                conf = accumulate_metrics(logits.cpu(), batch["mask"].long(),
                                         args.num_classes, ignore_index, conf)
        m = finalize_metrics(conf) if conf is not None else {}
        ood_results["s1s2w_mIoU"] = m.get("mIoU", 0.0)
        print(f"  S1S2-Water mIoU = {ood_results['s1s2w_mIoU']:.4f}")

    return ood_results


def _eval_trt_ood(engine_path, args, device_id):
    """Evaluate a TRT INT8 engine on STURM and S1S2W if roots are provided."""
    from quantization.trt_eval import eval_trt

    ood_results = {}

    if getattr(args, "sturm_root", None):
        from data.ood_datasets import get_sturm_flood_dataloader
        _sturm_norm = getattr(args, "sturm_normalize", "percentile")
        print(f"[quantize eval-trt] evaluating on STURM-S2 (OOD, normalize={_sturm_norm}) …")
        sturm_dl = get_sturm_flood_dataloader(
            root        = args.sturm_root,
            sensor      = "S2",
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            img_size    = args.eval_img_size,
            normalize   = _sturm_norm,
        )
        from data.channel_adapter import sturm_s2_adapter as _sturm_s2_adapter
        _sadapt = _sturm_s2_adapter(target_channels=6)

        class _AdaptedDL:
            def __init__(self, dl, adapter):
                self.dl = dl
                self.adapter = adapter
            def __iter__(self):
                for batch in self.dl:
                    batch = dict(batch)
                    batch["image"] = self.adapter(batch["image"].float())
                    yield batch
            def __len__(self):
                return len(self.dl)

        m = eval_trt(engine_path, _AdaptedDL(sturm_dl, _sadapt),
                     num_classes=args.num_classes, device_id=device_id)
        ood_results["sturm_mIoU"] = m.get("mIoU", 0.0)
        print(f"  STURM-S2 mIoU = {ood_results['sturm_mIoU']:.4f}")

    if getattr(args, "s1s2w_root", None):
        from data.ood_datasets import get_s1s2water_dataloader
        print("[quantize eval-trt] evaluating on S1S2-Water (OOD) …")
        s1s2w_dl = get_s1s2water_dataloader(
            root        = args.s1s2w_root,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            img_size    = args.eval_img_size,
        )
        m = eval_trt(engine_path, s1s2w_dl,
                     num_classes=args.num_classes, device_id=device_id)
        ood_results["s1s2w_mIoU"] = m.get("mIoU", 0.0)
        print(f"  S1S2-Water mIoU = {ood_results['s1s2w_mIoU']:.4f}")

    return ood_results

# qat subcommand

def cmd_qat(args):
    import torch
    import torch.nn.functional as F
    from config import QuantConfig
    from quantization.ptq import qat

    if args.data_root is None:
        print("[quantize qat] error: --data_root is required", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # --pseudo_dataset sets curated_pseudo and curated_index_file and overrides both flags.
    pseudo_dataset = getattr(args, "pseudo_dataset", None)
    if pseudo_dataset == "diversity":
        # One index per budget (scripts/curate.py, rule 2), so the shortcut has to name the
        # budget too: PseudoLabeledDataset samples uniformly from whatever index it is handed.
        args.curated_pseudo = True
        args.n_samples = args.n_samples or 2500
        args.curated_index_file = f"diversity_N{args.n_samples}_index.json"
    elif pseudo_dataset == "curated":
        args.curated_pseudo = True
        args.curated_index_file = "curated_index.json"
    elif pseudo_dataset == "random":
        args.curated_pseudo = False

    model = _build_student(args, args.num_classes)
    model = _load_checkpoint(model, args.model_ckpt)
    print(f"[quantize qat] loaded checkpoint from {args.model_ckpt!r}")

    train_dl, val_dl, test_dl = _get_dataloaders(args)
    s11_train_dl = train_dl          # kept for the BN re-estimation pass below

    # With --pseudo_dir, QAT trains on the same pseudo-labelled pool as the float model.
    # Validation and test always use S11.
    if getattr(args, "pseudo_dir", None):
        import torch as _torch
        from torch.utils.data import DataLoader as _DataLoader
        from data.pseudo_labeled import PseudoLabeledDataset
        pseudo_ds = PseudoLabeledDataset(
            args.pseudo_dir,
            n_samples   = getattr(args, "n_samples", None),
            seed        = getattr(args, "seed", None),
            use_curated = getattr(args, "curated_pseudo", False),
            index_file  = getattr(args, "curated_index_file", "curated_index.json"),
        )
        train_dl = _DataLoader(
            pseudo_ds,
            batch_size  = train_dl.batch_size,
            shuffle     = True,
            num_workers = train_dl.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
        print(f"[quantize qat] QAT training on pseudo-labels: {len(pseudo_ds)} scenes "
              f"(index={getattr(args,'curated_index_file','random')})")

    # BN statistics are only re-estimated with --bn_reset, i.e. when a Hardswish checkpoint is
    # loaded into a ReLU6 model. The deployed ReLU6 student is trained from scratch (Sec. III-D)
    # and keeps its statistics; resetting them costs about 0.25 mIoU.
    import torch as _torch
    device_pre = _torch.device(f"cuda:{args.gpu_ids[0]}"
                               if args.gpu_ids and _torch.cuda.is_available() else "cpu")
    needs_bn_reset = args.bn_reset and "relu6" in args.student_arch
    if args.bn_reset and "relu6" not in args.student_arch:
        print("[quantize qat] --bn_reset ignored: it only applies to a ReLU6 student arch.")
    if needs_bn_reset:
        # Re-estimate on Sen1Floods11, never on the pseudo pool: the pool's activation
        # statistics are the ones that inflate the per-tensor INT8 scales (docs/03).
        print(f"[quantize qat] re-estimating BN running stats for ReLU6 on Sen1Floods11 "
              f"({args.calibration_batches} calibration batches) ...")
        model.train()
        model.to(device_pre)
        with _torch.no_grad():
            for i, batch in enumerate(s11_train_dl):
                if i >= args.calibration_batches:
                    break
                model(batch["image"].to(device_pre).float())
        model.eval().cpu()
        print("[quantize qat] BN recalibration done.")

    cfg = QuantConfig(
        mode                = "qat",
        backend             = args.backend,
        calibration_batches = args.calibration_batches,
        qat_epochs          = args.qat_epochs,
        qat_lr              = args.qat_lr,
        weight_decay        = args.weight_decay,
        grad_clip           = args.grad_clip,
        scheduler           = "cosine",
        val_every_n_epochs  = 1,   # validate every epoch for 10-epoch QAT
        trt_compat          = args.trt_compat,
        output_qdq_scale    = args.output_qdq_scale,
        matmul_qdq          = getattr(args, "matmul_qdq", False),
        out_dir             = args.out_dir,
        seed                = args.seed,
    )

    ignore_index = -1

    # Optional teacher for KD during QAT (kd_alpha about 0.1 makes training more stable).
    teacher = None
    if getattr(args, "kd_alpha", 0.0) > 0.0 and getattr(args, "teacher_ckpt", None):
        from models.teacher import build_teacher, load_official_ckpt
        teacher = build_teacher(
            variant     = "prithvi_eo_v2_300_tl",
            num_classes = args.num_classes,
        )
        teacher = load_official_ckpt(teacher, args.teacher_ckpt)
        teacher.eval()
        print(f"[quantize qat] KD enabled  kd_alpha={args.kd_alpha}  "
              f"teacher from {args.teacher_ckpt!r}")

    # CE for QAT; focal loss concentrates the updates on few hard pixels.
    task_loss_fn = lambda logits, masks: F.cross_entropy(
        logits, masks, ignore_index=ignore_index
    )

    quant_model, results = qat(
        model             = model,
        train_dl          = train_dl,
        val_dl            = val_dl,
        test_dl           = test_dl,
        cfg               = cfg,
        task_loss_fn      = task_loss_fn,
        num_classes       = args.num_classes,
        ignore_index      = ignore_index,
        teacher           = teacher,
        kd_alpha          = getattr(args, "kd_alpha", 0.0),
        kd_temperature    = 4.0,
        calibration_type  = args.calibration_type,
    )

    print("\n[quantize qat] results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # OOD evaluation of the stripped float model returned by qat() (QAT weights, no quantizers).
    if getattr(args, "sturm_root", None) or getattr(args, "s1s2w_root", None):
        import torch as _t
        device_ood = _t.device(f"cuda:{args.gpu_ids[0]}"
                               if args.gpu_ids and _t.cuda.is_available() else "cpu")
        quant_model = quant_model.to(device_ood)
        ood = _eval_float_ood(quant_model, args, device_ood, label_prefix=" qat")
        results.update(ood)
        if ood:
            print("\n[quantize qat] OOD summary:")
            for k, v in ood.items():
                print(f"  {k}: {v:.4f}")

    import json as _json
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "qat_metrics.json")
    with open(metrics_path, "w") as _f:
        _json.dump({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in results.items()}, _f, indent=2)
    print(f"[quantize qat] metrics saved → {metrics_path}")

# eval-trt subcommand

def cmd_eval_trt(args):
    from quantization.trt_eval import build_engine, eval_trt

    if args.data_root is None:
        print("[quantize eval-trt] error: --data_root is required", file=sys.stderr)
        sys.exit(1)

    if not args.engine_path:
        # Derive engine path from ONNX path if not given
        args.engine_path = os.path.splitext(args.onnx_path)[0] + ".trt"

    if not os.path.exists(args.engine_path) or args.rebuild:
        build_engine(
            onnx_path    = args.onnx_path,
            engine_path  = args.engine_path,
            fp16_fallback = True,
            workspace_gb = args.workspace_gb,
        )
    else:
        print(f"[eval-trt] reusing engine {args.engine_path!r}  "
              f"(pass --rebuild to force re-compile)")

    _, val_dl, test_dl = _get_dataloaders(args)

    device_id = args.gpu_ids[0] if args.gpu_ids else 0

    print("[eval-trt] evaluating on val set ...")
    val_m = eval_trt(args.engine_path, val_dl,
                     num_classes=args.num_classes, device_id=device_id)
    print(f"  val  mIoU = {val_m.get('mIoU', 0.0):.4f}")

    if "lat_ms_mean" in val_m:
        print(f"  INT8 TRT latency: {val_m['lat_ms_mean']:.1f} ms/batch  "
              f"p99={val_m['lat_ms_p99']:.1f} ms  "
              f"FPS={val_m['fps']:.1f}  "
              f"engine={val_m.get('engine_size_mb', 0.0):.1f} MB")

    print("[eval-trt] evaluating on test set ...")
    test_m = eval_trt(args.engine_path, test_dl,
                      num_classes=args.num_classes, device_id=device_id)
    print(f"  test mIoU = {test_m.get('mIoU', 0.0):.4f}")

    if "iou_per_class" in test_m:
        for i, iou in enumerate(test_m["iou_per_class"]):
            print(f"    class {i}: IoU = {iou:.4f}")

    # OOD evaluation on STURM and S1S2W (TRT engine - INT8, actual deployment numbers)
    ood_trt = _eval_trt_ood(args.engine_path, args, device_id)

    import json as _json
    trt_all = {
        "val_mIoU":        val_m.get("mIoU", 0.0),
        "test_mIoU":       test_m.get("mIoU", 0.0),
        "lat_ms_mean":     val_m.get("lat_ms_mean"),
        "lat_ms_p99":      val_m.get("lat_ms_p99"),
        "fps":             val_m.get("fps"),
        "engine_size_mb":  val_m.get("engine_size_mb"),
    }
    trt_all.update(ood_trt)
    out_dir = os.path.dirname(os.path.abspath(args.engine_path))
    metrics_path = os.path.join(out_dir, "trt_metrics.json")
    with open(metrics_path, "w") as _f:
        _json.dump({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in trt_all.items() if v is not None}, _f, indent=2)
    print(f"[eval-trt] metrics saved → {metrics_path}")

# deploy subcommand

def cmd_deploy(args):
    import subprocess

    if not os.path.exists(args.onnx_path):
        print(f"[quantize deploy] error: {args.onnx_path!r} does not exist",
              file=sys.stderr)
        sys.exit(1)

    cmd = ["rsync", "-avz", "--progress", args.onnx_path,
           f"{args.jetson_host}:{args.jetson_dest}/"]

    print(f"[quantize deploy] {'(dry run) ' if args.dry_run else ''}rsync: {' '.join(cmd)}")
    if not args.dry_run:
        result = subprocess.run(cmd, check=True)
        print(f"[quantize deploy] done — exit code {result.returncode}")
    else:
        print("[quantize deploy] dry run complete — no files transferred")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QAT and Jetson export pipeline (ESI_project)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # qat
    qat_p = sub.add_parser("qat",
                           help="Quantization-aware training (custom QATConv2d INT8)")
    qat_p.add_argument("--model_ckpt", required=True,
                       help="Trained float32 checkpoint from train.py")
    _add_shared_args(qat_p)
    qat_p.add_argument("--backend", default="qnnpack",
                       choices=["qnnpack", "fbgemm"])
    qat_p.add_argument("--calibration_batches", type=int, default=200)
    qat_p.add_argument("--qat_epochs",   type=int,   default=10)
    qat_p.add_argument("--qat_lr",       type=float, default=1e-5)
    qat_p.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay for QAT AdamW. Keep at 1e-5 — higher values "
                            "regularise weights away from the INT8 grid.")
    qat_p.add_argument("--grad_clip",    type=float, default=1.0)
    qat_p.add_argument("--trt_compat", default="jp50",
                       choices=["jp50", "modern"])
    qat_p.add_argument("--calibration_type", default="per_tensor",
                       choices=["per_tensor", "per_channel_symmetric", "per_channel"],
                       help="per_tensor (default, used for TensorRT/Jetson; amax/127, zp=0, "
                            "needs ReLU6), per_channel_symmetric or per_channel (both not "
                            "supported by the TensorRT path)")
    qat_p.add_argument("--output_qdq_scale", default="calibrated",
                       choices=["calibrated", "from_next_input"],
                       help="Output Q-DQ scale for model_qdq_trt.onnx: 'calibrated' (default, "
                            "max|Conv output| on calibration data) or 'from_next_input' (older, "
                            "ReLU6 only)")
    qat_p.add_argument("--teacher_ckpt", default=None,
                       help="Optional teacher checkpoint for KD loss during QAT.")
    qat_p.add_argument("--kd_alpha", type=float, default=0.0,
                       help="KD loss weight (0 = CE only). "
                            "Use 0.1 to match climatecompression default.")
    # Pseudo-label data for QAT training (keeps model in GDACS distribution)
    qat_p.add_argument("--pseudo_dir", default=None,
                       help="Pseudo-labelled .pt files for QAT training; validation and test use S11")
    qat_p.add_argument("--curated_pseudo", action="store_true", default=False,
                       help="Use --curated_index_file for deterministic scene selection "
                            "within the pseudo-label dir.")
    qat_p.add_argument("--curated_index_file", default="diversity_N2500_index.json",
                       help="Index file when --curated_pseudo is set. It must be the index the "
                            "checkpoint was trained on, built at exactly --n_samples entries.")
    qat_p.add_argument("--n_samples", type=int, default=None,
                       help="Cap pseudo dataset to N scenes (default: all 2500 in index).")
    qat_p.add_argument("--bn_reset", action="store_true", default=False,
                       help="Re-estimate BN statistics on Sen1Floods11 before QAT. Only for a "
                            "Hardswish checkpoint loaded into ReLU6; not for the deployed model.")
    qat_p.add_argument("--no_bn_reset", action="store_true", default=False,
                       help="Accepted and ignored: BN re-estimation is off unless --bn_reset "
                            "is passed. Kept so existing commands keep working.")
    qat_p.add_argument(
        "--pseudo_dataset",
        default=None,
        choices=["diversity", "curated", "random"],
        help="Pseudo pool selection: 'diversity' (diversity_N<n_samples>_index.json, "
             "n_samples default 2500), 'curated' (curated_index.json) or 'random'. "
             "Overrides --curated_pseudo/--curated_index_file. Use the same selection "
             "and --seed as in training.",
    )
    qat_p.add_argument("--matmul_qdq", action="store_true", default=False,
                       help="Add INT8 QDQ to LiteMLA MatMul/Einsum inputs with ORT calibration "
                            "at export (TensorRT path). Off by default.")

    # deploy
    dep_p = sub.add_parser("deploy",
                           help="Copy ONNX model to Jetson Xavier via rsync")
    dep_p.add_argument("--onnx_path",   required=True, help="Local ONNX file path")
    dep_p.add_argument("--jetson_host", required=True, help="SSH target (user@host)")
    dep_p.add_argument("--jetson_dest", required=True,
                       help="Remote destination directory on Jetson")
    dep_p.add_argument("--dry_run", action="store_true",
                       help="Print rsync command without executing")

    # eval-trt
    trt_p = sub.add_parser(
        "eval-trt",
        help="Build a TensorRT INT8 engine from model_qdq_trt_no_out.onnx and evaluate it "
             "locally (needs tensorrt and pycuda)."
    )
    trt_p.add_argument("--onnx_path", required=True,
                       help="Path to model_qdq_trt.onnx (per-tensor QDQ, TRT-compatible)")
    trt_p.add_argument("--engine_path", default=None,
                       help="Where to save/load the compiled .trt engine. "
                            "Defaults to <onnx_path>.trt")
    trt_p.add_argument("--rebuild", action="store_true",
                       help="Force engine re-compilation even if .trt file already exists")
    trt_p.add_argument("--workspace_gb", type=float, default=4.0,
                       help="TRT tactic-search workspace in GB (default 4.0)")
    _add_shared_args(trt_p)

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    dispatch = {
        "qat":      cmd_qat,
        "deploy":   cmd_deploy,
        "eval-trt": cmd_eval_trt,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
