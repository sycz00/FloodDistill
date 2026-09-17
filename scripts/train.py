"""Training, pseudo-label generation and evaluation commands.

finetune: CE/focal supervision; distill: Logit-KD; ofa: intermediate-exit KD.
pseudo generate writes masks/confidence; pseudo relabel preserves images and
membership while changing the teacher. eval, ood-eval and viz inspect models.

The paper uses the public Sen1Floods11 Prithvi checkpoint (--official_ckpt hf)
without further teacher fine-tuning. Published recipes and seeds are under
experiments/. Stored hard masks train the task loss; the reported KD/OFA arms
query the live frozen teacher for soft targets.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as `python scripts/train.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# W&B stays off unless WANDB_MODE is set, e.g. WANDB_MODE=online. Set it here, before
# any teacher import can start a run.
os.environ.setdefault("WANDB_MODE", "disabled")

# Shared flag builders

def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--teacher_variant", default="prithvi_eo_v2_300_tl",
                   choices=["prithvi_eo_v2_300", "prithvi_eo_v2_300_tl",
                             "prithvi_eo_v2_600", "prithvi_eo_v2_600_tl"])
    p.add_argument("--official_ckpt", default=None,
                   help="Path to official Lightning .pt checkpoint, or 'hf' to download.")
    p.add_argument("--teacher_ckpt", default=None,
                   help="Path to a previously fine-tuned teacher checkpoint.")
    p.add_argument("--student_arch", default="efficientvit_b0",
                   choices=["vit_tiny", "efficientvit_b0", "efficientvit_b0_relu6",
                             "efficientvit_b0_lrelu6", "efficientvit_b0_silu", "efficientvit_b0_tanh",
                             "efficientvit_b1", "efficientvit_b2",
                             "tiny_vit_5m", "tiny_vit_11m", "tiny_vit_21m",
                             "mobilenet_v3_small", "unet_resnet50", "unet_resnet101",
                             "micro_s", "micro_m", "lcnet_050", "lcnet_050_relu6", "lcnet_075"])
    p.add_argument("--pretrained_tag", default="dist_in22k_ft_in1k",
                   choices=["dist_in22k_ft_in1k", "dist_in22k", "in1k"])
    p.add_argument("--num_classes", type=int, default=2)
    # Bridge chain (docs/02_distillation.md): with --bridge_teacher_ckpts the teacher is a frozen
    # ensemble of student checkpoints (e.g. four UNet Logit-KD seeds); --official_ckpt is ignored.
    p.add_argument("--bridge_teacher_ckpts", nargs="+", default=None,
                   help="Checkpoint path(s) of student-arch model(s) to use as the "
                        "teacher (TA-KD chain). Several paths = seed ensemble.")
    p.add_argument("--bridge_teacher_arch", default="unet_resnet50",
                   help="Architecture of the bridge teacher checkpoints.")
    p.add_argument("--bridge_ensemble_mode", default="prob", choices=["prob", "logit"],
                   help="Ensemble combination: 'prob' = log(mean softmax) [default], "
                        "'logit' = mean logits.")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Skip pretrained weights download (smoke tests).")
    # Augmentation for pseudo-labels as for Sen1Floods11 (data/sen1floods11.py): flips, rot90
    # and MultiplicativeNoise(0.80-1.20). Off by default; experiments/nsweep_pseudo_vs_gt.sh
    # sets both flags.
    p.add_argument("--pseudo_mult_noise", action="store_true",
                   help="Per-channel multiplicative reflectance jitter k~U(0.80,1.20), p=0.5, "
                        "for pseudo-label tiles (as for Sen1Floods11)")
    p.add_argument("--pseudo_geom_aug", action="store_true",
                   help="Flips and rot90 (p=0.5 each) for pseudo-label tiles, applied to all "
                        "spatial keys of a sample")


def _add_data_args(p: argparse.ArgumentParser, data_root_required=False) -> None:
    if data_root_required:
        p.add_argument("--data_root", required=True)
    else:
        from utils.paths import dataset_root
        p.add_argument("--data_root", default=str(dataset_root("S11_ROOT")),
                       help="Directory CONTAINING v1.1/ (terratorch appends v1.1 itself). "
                            "Default: $DATA_ROOT/sen1floods11.")
    p.add_argument("--batch_size",    type=int, default=None,
                   help="Default: 8 for vit_tiny, 16 otherwise.")
    p.add_argument("--num_workers",   type=int, default=8)
    p.add_argument("--img_size",      type=int, default=None,
                   help="Training crop size. Default: 512 for vit_tiny, 224 otherwise.")
    p.add_argument("--eval_img_size", type=int, default=None,
                   help="Val/test resize. Default: 224 for TinyViT (window constraint), "
                        "512 otherwise.")
    p.add_argument("--train_center_crop", type=int, default=None,
                   help="Centre-crop Sen1Floods11 training chips to this many pixels before "
                        "resizing (264 matches the GDACS tiles; geometry control). Training only.")


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs",              type=int,   default=50)
    p.add_argument("--lr",                  type=float, default=5e-5)
    p.add_argument("--weight_decay",        type=float, default=0.05)
    p.add_argument("--task_loss",           default="ce", choices=["focal", "ce"])
    p.add_argument("--warmup_epochs",       type=int,   default=None,
                   help="Default: 5 for vit_tiny, 0 otherwise.")
    p.add_argument("--early_stop_patience", type=int,   default=20)
    p.add_argument("--val_smoothing_window", type=int,  default=1,
                   help="Rolling-mean window of val mIoU for checkpoint selection (1 = best epoch)")
    p.add_argument("--eval_all_splits", action="store_true",
                   help="Evaluate the best checkpoint on the S11 train, val and test splits")
    p.add_argument("--use_amp",  action="store_true",   default=True)
    p.add_argument("--no_amp",   dest="use_amp",        action="store_false")
    p.add_argument("--amp_dtype", default="fp16",       choices=["fp16", "bf16"])
    p.add_argument("--grad_clip", type=float,           default=0.0)


def _add_infra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out_dir",        default="./runs")
    p.add_argument("--gpu_ids",        type=int, nargs="+", default=[0])
    p.add_argument("--seed",           type=int, default=0,
                   help="-1 disables seeding.")
    p.add_argument("--deterministic",  action="store_true")
    p.add_argument("--wandb_project",  default="EfficientSatelliteImagery")
    p.add_argument("--wandb_run_name", default=None)


# Architecture defaults

def _resolve_arch_defaults(args) -> argparse.Namespace:
    """Apply architecture-specific defaults for img_size, batch_size, warmup."""
    is_vit      = getattr(args, "student_arch", "") == "vit_tiny"
    is_tiny_vit = getattr(args, "student_arch", "").startswith("tiny_vit_")

    if getattr(args, "batch_size", None) is None:
        args.batch_size = 8 if is_vit else 16
    if getattr(args, "img_size", None) is None:
        args.img_size = 512 if is_vit else 224
    if getattr(args, "warmup_epochs", None) is None:
        args.warmup_epochs = 5 if is_vit else 0

    if getattr(args, "eval_img_size", None) is None:
        # TinyViT: window_size=14/7 tiles exactly at 224px - stay there.
        # vit_tiny / all others: eval at 512px (native Sen1Floods11 resolution).
        args.eval_img_size = 224 if is_tiny_vit else 512

    if is_vit and getattr(args, "img_size", None) == 512:
        print(f"[train] vit_tiny: img_size=512  batch_size={args.batch_size}  "
              f"warmup_epochs={args.warmup_epochs}")
    return args


# Shared model / dataloader builders

def _get_dataloaders(args, batch_size=None):
    from data.sen1floods11 import get_dataloaders
    return get_dataloaders(
        data_root          = args.data_root,
        batch_size         = batch_size or args.batch_size,
        num_workers        = args.num_workers,
        img_size           = args.img_size,
        eval_img_size      = args.eval_img_size,
        n_labeled_samples  = getattr(args, "n_labeled_samples", None),
        seed               = getattr(args, "seed", None),
        train_center_crop_px = getattr(args, "train_center_crop", None),
    )


def _build_teacher(args, freeze=False):
    import torch
    from models.teacher import build_teacher
    # Bridge / TA-chain: use (an ensemble of) student-arch checkpoints as teacher.
    if getattr(args, "bridge_teacher_ckpts", None):
        from models.bridge_teacher import build_bridge_teacher
        return build_bridge_teacher(
            arch          = args.bridge_teacher_arch,
            ckpts         = args.bridge_teacher_ckpts,
            num_classes   = args.num_classes,
            in_channels   = 6,
            ensemble_mode = args.bridge_ensemble_mode,
        )  # always frozen + eval
    model = build_teacher(
        variant       = args.teacher_variant,
        num_classes   = args.num_classes,
        pretrained    = not (args.teacher_ckpt or args.official_ckpt),
        official_ckpt = args.official_ckpt,
    )
    if args.teacher_ckpt:
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"[train] teacher loaded: {args.teacher_ckpt}")
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    return model


def _build_student(args):
    from models.student import build_student
    return build_student(
        arch           = args.student_arch,
        in_channels    = 6,
        num_classes    = args.num_classes,
        pretrained     = not args.no_pretrained,
        pretrained_tag = args.pretrained_tag,
    )


def _make_train_cfg(args):
    from config import TrainConfig
    seed = None if getattr(args, "seed", 0) == -1 else getattr(args, "seed", 0)
    return TrainConfig(
        epochs              = args.epochs,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        task_loss           = args.task_loss,
        gamma               = getattr(args, "gamma", 2.0),
        ignore_index        = -1,
        alpha               = getattr(args, "alpha", 0.5),
        temperature         = getattr(args, "temperature", 4.0),
        at_beta             = getattr(args, "at_beta", 0.0),
        feat_alpha           = getattr(args, "feat_alpha", 0.0),
        concept_distil_alpha = getattr(args, "concept_distil_alpha", 0.0),
        use_amp              = args.use_amp,
        amp_dtype            = args.amp_dtype,
        grad_clip            = getattr(args, "grad_clip", 0.0),
        warmup_epochs        = args.warmup_epochs,
        early_stop_patience  = args.early_stop_patience,
        val_smoothing_window = getattr(args, "val_smoothing_window", 1),
        eval_all_splits      = getattr(args, "eval_all_splits", False),
        class_names          = ["background", "flood"],
        out_dir             = args.out_dir,
        gpu_ids             = args.gpu_ids,
        seed                = seed,
        deterministic       = getattr(args, "deterministic", False),
        wandb_project       = args.wandb_project,
        wandb_run_name      = args.wandb_run_name,
    )

# Subcommand handlers

def cmd_finetune(args):
    from torch.utils.data import DataLoader
    from training.trainer import Trainer
    from training.losses import DistillLoss

    args = _resolve_arch_defaults(args)

    if args.target == "teacher":
        model = _build_teacher(args, freeze=False)
        if args.freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            print(f"[finetune] backbone frozen")
    else:
        model = _build_student(args)

    train_dl, val_dl, test_dl = _get_dataloaders(args)
    train_cfg = _make_train_cfg(args)

    # Pseudo-only supervised baseline: replace train_dl with pseudo dataset.
    # No teacher required - pseudo masks are hard argmax labels from a prior run.
    if getattr(args, "pseudo_dir", None):
        if not args.no_labeled:
            raise ValueError(
                "finetune --pseudo_dir requires --no_labeled "
                "(mixed labeled+pseudo supervised is not supported; use distill instead)"
            )
        from data.pseudo_labeled import PseudoLabeledDataset
        pseudo_ds = PseudoLabeledDataset(
            args.pseudo_dir,
            n_samples            = args.n_samples,
            pixel_conf_threshold = args.pixel_conf_threshold,
            seed                 = getattr(args, "seed", None),
            use_curated          = getattr(args, "curated_pseudo", False),
            index_file           = getattr(args, "curated_index_file", "curated_index.json"),
            mult_noise           = getattr(args, "pseudo_mult_noise", False),
            geom_aug             = getattr(args, "pseudo_geom_aug", False),
        )
        print(f"[finetune] pseudo-only supervised: {len(pseudo_ds)} scenes, no teacher")
        train_dl = DataLoader(
            pseudo_ds,
            batch_size  = train_dl.batch_size,
            shuffle     = True,
            num_workers = train_dl.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )

    if getattr(args, "pretrained_backbone", None):
        import torch as _torch
        pb_path = args.pretrained_backbone
        ckpt    = _torch.load(pb_path, map_location="cpu", weights_only=False)
        state   = ckpt.get("model", ckpt)
        bb_keys = {k: v for k, v in state.items() if "model.backbone" in k}
        missing, unexpected = model.load_state_dict(bb_keys, strict=False)
        print(f"[finetune] Spectral pretrain backbone loaded from {pb_path}: "
              f"{len(bb_keys)} backbone keys  {len(missing)} missing  {len(unexpected)} unexpected")

    loss_fn = None
    if train_cfg.feat_alpha > 0 and args.target == "student":
        if not args.official_ckpt and not args.teacher_ckpt:
            raise ValueError("--feat_alpha > 0 requires --official_ckpt or --teacher_ckpt")
        print(f"[finetune] joint feat distil: feat_alpha={train_cfg.feat_alpha}")
        teacher = _build_teacher(args, freeze=True)
        loss_fn = DistillLoss(
            teacher           = teacher,
            student           = model,
            alpha             = train_cfg.alpha,
            temperature       = train_cfg.temperature,
            at_beta           = train_cfg.at_beta,
            feat_distil_alpha = train_cfg.feat_alpha,
            ignore_index      = train_cfg.ignore_index,
            gamma             = train_cfg.gamma,
        )
        loss_fn.register_hooks()

    results = Trainer(model, train_dl, val_dl, test_dl, train_cfg,
                      loss_fn=loss_fn, num_classes=args.num_classes,
                      class_names=train_cfg.class_names).fit()
    if loss_fn is not None:
        loss_fn.remove_hooks()

    print("\n[finetune] results:")
    _print_results(results)



def cmd_distill(args):
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
    from training.trainer import Trainer
    from training.losses import DistillLoss

    args = _resolve_arch_defaults(args)

    if (not args.official_ckpt and not args.teacher_ckpt
            and not getattr(args, "bridge_teacher_ckpts", None)):
        raise ValueError("distill requires --official_ckpt, --teacher_ckpt, "
                         "or --bridge_teacher_ckpts")

    teacher = _build_teacher(args, freeze=True)
    student = _build_student(args)

    if getattr(args, "pretrained_backbone", None):
        import torch as _torch
        pb_path = args.pretrained_backbone
        ckpt    = _torch.load(pb_path, map_location="cpu", weights_only=False)
        state   = ckpt.get("model", ckpt)
        bb_keys = {k: v for k, v in state.items() if "model.backbone" in k}
        missing, unexpected = student.load_state_dict(bb_keys, strict=False)
        print(f"[distill] Spectral pretrain backbone loaded from {pb_path}: "
              f"{len(bb_keys)} backbone keys  {len(missing)} missing  {len(unexpected)} unexpected")

    train_dl, val_dl, test_dl = _get_dataloaders(args)
    train_cfg = _make_train_cfg(args)

    # Optionally augment training set with pseudo-labeled data (DFKD)
    if args.pseudo_dir:
        from data.pseudo_labeled import PseudoLabeledDataset

        if args.no_labeled:
            pseudo_ds = PseudoLabeledDataset(
                args.pseudo_dir,
                n_samples             = args.n_samples,
                pixel_conf_threshold  = args.pixel_conf_threshold,
                seed                  = getattr(args, "seed", None),
                use_curated           = getattr(args, "curated_pseudo", False),
                index_file            = getattr(args, "curated_index_file", "curated_index.json"),
                mult_noise            = getattr(args, "pseudo_mult_noise", False),
                geom_aug              = getattr(args, "pseudo_geom_aug", False),
            )
            n_pseudo = len(pseudo_ds)
            print(f"[distill] no_labeled=True — training on {n_pseudo} pseudo scenes only")
            combined_dl = DataLoader(
                pseudo_ds,
                batch_size  = train_dl.batch_size,
                shuffle     = True,
                num_workers = train_dl.num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
        else:
            pseudo_ds = PseudoLabeledDataset(
                args.pseudo_dir,
                n_samples             = args.n_samples,
                pixel_conf_threshold  = args.pixel_conf_threshold,
                seed                  = getattr(args, "seed", None),
                use_curated           = getattr(args, "curated_pseudo", False),
                index_file            = getattr(args, "curated_index_file", "curated_index.json"),
                mult_noise            = getattr(args, "pseudo_mult_noise", False),
                geom_aug              = getattr(args, "pseudo_geom_aug", False),
                extra_keys            = frozenset(),
            )
            n_pseudo = len(pseudo_ds)
            labeled_ds = train_dl.dataset
            n_lab      = len(labeled_ds)
            weights    = [1.0] * n_lab + [0.5] * n_pseudo
            sampler    = WeightedRandomSampler(weights, num_samples=n_lab * 2, replacement=True)
            combined_dl = DataLoader(
                ConcatDataset([labeled_ds, pseudo_ds]),
                batch_size  = train_dl.batch_size,
                sampler     = sampler,
                num_workers = train_dl.num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
            print(f"[distill] labeled={n_lab}  pseudo={n_pseudo}  "
                  f"steps/epoch≈{n_lab*2 // train_dl.batch_size}")
        train_dl = combined_dl

    loss_fn = DistillLoss(
        teacher           = teacher,
        student           = student,
        alpha             = train_cfg.alpha,
        temperature       = train_cfg.temperature,
        at_beta           = train_cfg.at_beta,
        feat_distil_alpha = train_cfg.feat_alpha,
        ignore_index      = train_cfg.ignore_index,
        gamma             = train_cfg.gamma,
    )
    loss_fn.register_hooks()

    results = Trainer(student, train_dl, val_dl, test_dl, train_cfg,
                      loss_fn=loss_fn, num_classes=args.num_classes,
                      class_names=train_cfg.class_names).fit()
    loss_fn.remove_hooks()

    print("\n[distill] results:")
    _print_results(results)



def cmd_pseudo_generate(args):
    import torch
    from data.pseudo_labeled import generate_pseudo_labels

    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    teacher = _build_teacher(args, freeze=True).to(device)

    n = generate_pseudo_labels(
        teacher               = teacher,
        unlabeled_root        = args.unlabeled_root,
        out_dir               = args.out_dir,
        unlabeled_source      = args.unlabeled_source,
        confidence_threshold  = args.confidence_threshold,
        min_flood_frac        = args.min_flood_frac,
        img_size              = args.img_size or 224,
        batch_size            = args.batch_size or 32,
        num_workers           = args.num_workers,
        device                = device,
        store_pixel_confidence = args.store_pixel_confidence,
        store_soft_logits     = args.store_soft_logits,
    )
    print(f"[pseudo generate] saved {n} pseudo-labeled scenes → {args.out_dir}")



def cmd_pseudo_relabel(args):
    """Relabel an existing pseudo pool with another teacher; images and pool membership stay the same.

    With --bridge_teacher_ckpts the labels come from the UNet ensemble, otherwise from Prithvi.
    """
    import torch
    from data.pseudo_labeled import relabel_pseudo_labels

    device  = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    teacher = _build_teacher(args, freeze=True).to(device)

    index_file = None if args.index_file in ("", "none") else args.index_file
    n = relabel_pseudo_labels(
        teacher                = teacher,
        pseudo_dir             = args.pseudo_dir,
        out_dir                = args.out_dir,
        index_file             = index_file,
        batch_size             = args.batch_size or 32,
        num_workers            = args.num_workers,
        device                 = device,
        store_pixel_confidence = args.store_pixel_confidence,
        store_soft_logits      = args.store_soft_logits,
    )
    print(f"[pseudo relabel] wrote {n} relabeled scenes → {args.out_dir}")



def cmd_ofa(args):
    """OFA-KD: One-for-All logit-space distillation via intermediate exit branches."""
    import torch
    from torch.utils.data import DataLoader
    from training.trainer import Trainer
    from distillation.ofa_kd import OFALoss
    from utils.seeding import seed_basic

    args = _resolve_arch_defaults(args)

    seed = None if args.seed == -1 else args.seed
    if seed is not None:
        seed_basic(seed)

    device  = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    teacher = _build_teacher(args, freeze=True)
    student = _build_student(args)

    train_dl, val_dl, test_dl = _get_dataloaders(args)
    train_cfg = _make_train_cfg(args)

    # Optionally replace / augment training set with pseudo-labeled data
    if args.pseudo_dir:
        from data.pseudo_labeled import PseudoLabeledDataset

        if args.no_labeled:
            pseudo_ds = PseudoLabeledDataset(
                args.pseudo_dir,
                n_samples   = args.n_samples,
                seed        = getattr(args, "seed", None),
                use_curated = getattr(args, "curated_pseudo", False),
                index_file  = getattr(args, "curated_index_file", "curated_index.json"),
                mult_noise  = getattr(args, "pseudo_mult_noise", False),
                geom_aug    = getattr(args, "pseudo_geom_aug", False),
            )
            n_pseudo = len(pseudo_ds)
            print(f"[ofa] no_labeled=True — training on {n_pseudo} pseudo scenes only")
            train_dl = DataLoader(
                pseudo_ds,
                batch_size  = train_dl.batch_size,
                shuffle     = True,
                num_workers = train_dl.num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
        else:
            from torch.utils.data import ConcatDataset
            from torch.utils.data import WeightedRandomSampler
            # No optional keys in the pseudo samples, so they collate with the labelled samples.
            pseudo_ds = PseudoLabeledDataset(
                args.pseudo_dir,
                n_samples   = args.n_samples,
                seed        = getattr(args, "seed", None),
                use_curated = getattr(args, "curated_pseudo", False),
                index_file  = getattr(args, "curated_index_file", "curated_index.json"),
                mult_noise  = getattr(args, "pseudo_mult_noise", False),
                geom_aug    = getattr(args, "pseudo_geom_aug", False),
                extra_keys  = frozenset(),
            )
            n_pseudo = len(pseudo_ds)
            labeled_ds = train_dl.dataset
            n_lab      = len(labeled_ds)
            weights    = [1.0] * n_lab + [0.5] * n_pseudo
            sampler    = WeightedRandomSampler(weights, num_samples=n_lab * 2, replacement=True)
            train_dl   = DataLoader(
                ConcatDataset([labeled_ds, pseudo_ds]),
                batch_size  = train_dl.batch_size,
                sampler     = sampler,
                num_workers = train_dl.num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
            print(f"[ofa] labeled={n_lab}  pseudo={n_pseudo}")

    teacher.to(device)
    student.to(device)

    loss_fn = OFALoss(
        teacher              = teacher,
        student              = student,
        num_classes          = args.num_classes,
        alpha                = args.alpha,
        ofa_alpha            = args.ofa_alpha,
        temperature          = args.temperature,
        gamma                = args.gamma,
        ignore_index         = -1,
        disc_weighted        = args.disc_weighted,
        uncertainty_weighted = args.uncertainty_weighted,
        dist_match_alpha     = args.dist_match_alpha,
        relational_alpha     = args.relational_alpha,
        upsample_first       = args.upsample_first,
        gradcam_weight_ofa      = getattr(args, "gradcam_weight_ofa", False),
        gradcam_contrast_weight = getattr(args, "gradcam_contrast_weight_ofa", False),
        sc_ofa              = getattr(args, "sc_ofa", False),
        sc_entropy_stages   = getattr(args, "sc_entropy_stages", 2),
        nc_alpha            = getattr(args, "nc_alpha", 0.0),
        nc_max_branch       = getattr(args, "nc_max_branch", 2),
        nc_temperature      = getattr(args, "nc_temperature", 0.5),
        patch_affinity_alpha = getattr(args, "patch_affinity_alpha", 0.0),
        pa_num_patches       = getattr(args, "pa_num_patches", 14),
        pa_scales            = getattr(args, "pa_scales", None),
        sm_ofa               = getattr(args, "sm_ofa", False),
        sm_stages            = getattr(args, "sm_stages", 2),
        sm_confidence_thresh = getattr(args, "sm_confidence_thresh", 0.65),
        use_grad_filter      = getattr(args, "grad_filter", False),
        grad_filter_stages   = getattr(args, "grad_filter_stages", None),
        grad_filter_rf       = getattr(args, "grad_filter_rf", None),
        ofa_min_stage        = getattr(args, "ofa_min_stage", 0),
        sses               = getattr(args, "sses", False),
        sses_late_stage    = getattr(args, "sses_late_stage", 2),
        sses_proxy_alpha   = getattr(args, "sses_proxy_alpha", 1.0),
        sses_num_proxies   = getattr(args, "sses_num_proxies", 3),
        sses_proxy_mode    = getattr(args, "sses_proxy_mode", "spectral_continuous"),
        branch_conflict_mode     = getattr(args, "ofa_branch_conflict",     False),
        ofa_branch_warmup_epoch  = getattr(args, "ofa_branch_warmup_epoch", 0),
    )

    # Load spectral pre-trained backbone weights if provided.
    if getattr(args, "pretrained_backbone", None):
        import torch as _torch
        pb_path = args.pretrained_backbone
        ckpt    = _torch.load(pb_path, map_location="cpu", weights_only=False)
        state   = ckpt.get("model", ckpt)
        bb_keys = {k: v for k, v in state.items() if "model.backbone" in k}
        missing, unexpected = student.load_state_dict(bb_keys, strict=False)
        n_head  = len([k for k in state if "model.backbone" not in k])
        print(f"[ofa] Spectral pretrain backbone loaded from {pb_path}: "
              f"{len(bb_keys)} backbone keys  {n_head} head keys skipped  "
              f"{len(missing)} missing  {len(unexpected)} unexpected")

    loss_fn.register_hooks()
    loss_fn.prewarm(train_dl, device)

    results = Trainer(
        student, train_dl, val_dl, test_dl, train_cfg,
        loss_fn                = loss_fn,
        num_classes            = args.num_classes,
        class_names            = train_cfg.class_names,
        grad_conflict          = args.grad_conflict,
        grad_conflict_interval = args.grad_conflict_interval,
        pcgrad                 = args.pcgrad,
        ofa_backbone_pcgrad    = getattr(args, "ofa_backbone_pcgrad", False),
        ofa_agr                = getattr(args, "ofa_agr",             False),
        ofa_fagr               = getattr(args, "ofa_fagr",            False),
        ofa_diagnose           = getattr(args, "ofa_diagnose",        False),
        ofa_diagnose_interval  = getattr(args, "ofa_diagnose_interval", 50),
        sam                    = getattr(args, "sam", False),
        sam_rho                = getattr(args, "sam_rho", 0.05),
    ).fit()
    loss_fn.remove_hooks()

    print("\n[ofa] Training complete.")
    _print_results(results)



def _ofa_build_train_dl(args, train_dl):
    """Optionally replace or augment train_dl with pseudo-labeled data."""
    if not getattr(args, "pseudo_dir", None):
        return train_dl
    from data.pseudo_labeled import PseudoLabeledDataset
    from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
    no_labeled = getattr(args, "no_labeled", False)
    pseudo_ds = PseudoLabeledDataset(
        args.pseudo_dir,
        n_samples   = args.n_samples,
        seed        = getattr(args, "seed", None),
        use_curated = getattr(args, "curated_pseudo", False),
        index_file  = getattr(args, "curated_index_file", "curated_index.json"),
        mult_noise  = getattr(args, "pseudo_mult_noise", False),
        geom_aug    = getattr(args, "pseudo_geom_aug", False),
        # Strip optional keys when mixing with labeled data to avoid key-mismatch
        # crashes in PyTorch's default_collate (labeled items have no optional keys).
        extra_keys  = None if no_labeled else frozenset(),
    )
    n_pseudo  = len(pseudo_ds)
    if no_labeled:
        print(f"[ofa_shared] no_labeled=True — training on {n_pseudo} pseudo scenes only")
        return DataLoader(
            pseudo_ds, batch_size=train_dl.batch_size, shuffle=True,
            num_workers=train_dl.num_workers, pin_memory=True, drop_last=True,
        )
    labeled_ds = train_dl.dataset
    n_lab      = len(labeled_ds)
    weights    = [1.0] * n_lab + [0.5] * n_pseudo
    sampler    = WeightedRandomSampler(weights, num_samples=n_lab * 2, replacement=True)
    print(f"[ofa_shared] labeled={n_lab}  pseudo={n_pseudo}")
    return DataLoader(
        ConcatDataset([labeled_ds, pseudo_ds]), batch_size=train_dl.batch_size,
        sampler=sampler, num_workers=train_dl.num_workers, pin_memory=True, drop_last=True,
    )



def cmd_eval(args):
    import torch
    from training.trainer import Trainer

    args = _resolve_arch_defaults(args)

    if args.target == "teacher":
        model = _build_teacher(args)
    else:
        model = _build_student(args)
        if args.model_ckpt:
            ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("model", ckpt), strict=True)
            print(f"[eval] loaded checkpoint: {args.model_ckpt}")

    _, val_dl, test_dl = _get_dataloaders(args)
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    train_cfg = _make_train_cfg(args)
    trainer   = Trainer(model, val_dl, val_dl, test_dl, train_cfg,
                        num_classes=args.num_classes,
                        class_names=train_cfg.class_names)
    val_m  = trainer.evaluate(model, val_dl)
    test_m = trainer.evaluate(model, test_dl)

    results = {**{f"val_{k}": v for k, v in val_m.items()},
               **{f"test_{k}": v for k, v in test_m.items()}}

    print(f"\n[eval] val_mIoU={val_m['mIoU']:.4f}  test_mIoU={test_m['mIoU']:.4f}")
    _print_results(results)

    import json
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_name = ("teacher_official.pt" if args.official_ckpt else
                 f"{args.target}_eval.pt")
    ckpt_path = os.path.join(args.out_dir, ckpt_name)
    torch.save({"model": model.state_dict(), "results": results}, ckpt_path)
    with open(ckpt_path.replace(".pt", "_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] checkpoint → {ckpt_path}")



def cmd_ood_eval(args):
    import torch
    from evaluation.ood_eval import OODEval

    args = _resolve_arch_defaults(args)

    if not args.ood_root:
        raise ValueError("ood-eval requires --ood_root")

    if args.target == "teacher":
        model = _build_teacher(args)
    else:
        model = _build_student(args)
        if args.model_ckpt:
            ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("model", ckpt), strict=True)

    results = OODEval(
        model        = model,
        model_name   = args.target,
        dataset_name = args.ood_dataset,
        ood_root     = args.ood_root,
        in_channels  = 6,
        num_classes  = args.num_classes,
        batch_size   = args.batch_size or 8,
        num_workers  = args.num_workers,
        img_size     = args.eval_img_size or 224,
        out_dir      = args.ood_out_dir,
        gpu_ids      = args.gpu_ids,
    ).run()

    print("\n[ood-eval] results:")
    _print_results(results)



def cmd_viz(args):
    import torch
    from models.student import build_student
    from utils.plotting import plot_all_splits

    args = _resolve_arch_defaults(args)

    device  = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    student = build_student(
        arch        = args.student_arch,
        in_channels = 6,
        num_classes = args.num_classes,
        pretrained  = False,
        ckpt_path   = args.model_ckpt or os.path.join(args.out_dir, "best.pt"),
    ).to(device)
    student.eval()

    train_dl, val_dl, test_dl = _get_dataloaders(args, batch_size=1)
    results = {"saved_figures": plot_all_splits(
        student, train_dl, val_dl, test_dl,
        device=device, out_dir=args.viz_dir, n_samples=args.n_samples,
    )}
    print(f"\n[viz] {results}")


# Utilities


def _print_results(results: dict) -> None:
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# Parser construction


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Flood segmentation: training, pseudo-labels and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = root.add_subparsers(dest="subcommand", required=True)

    # finetune
    ft = sub.add_parser("finetune", help="Supervised fine-tuning of teacher or student")
    _add_model_args(ft)
    _add_data_args(ft)
    _add_train_args(ft)
    _add_infra_args(ft)
    ft.add_argument("--target", default="student", choices=["student", "teacher"])
    ft.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze teacher backbone; train only decoder+head.")
    ft.add_argument("--feat_alpha", type=float, default=0.0,
                    help="Feature distillation weight (requires teacher for reference).")
    # Pseudo-only supervised baseline (no teacher, CE loss on hard pseudo-labels)
    ft.add_argument("--pseudo_dir", default=None,
                    help="Directory of pseudo-labeled .pt files. Requires --no_labeled.")
    ft.add_argument("--no_labeled", action="store_true",
                    help="Train on pseudo-labels only (no Sen1Floods11 ground truth).")
    ft.add_argument("--n_samples", type=int, default=None,
                    help="Cap pseudo dataset to N randomly selected scenes.")
    ft.add_argument("--n_labeled_samples", type=int, default=None,
                    help="Subsample N images from the labeled Sen1Floods11 train split "
                         "(252 total). For labeled N-ablation (CE-only / DFKD baselines).")
    ft.add_argument("--pixel_conf_threshold", type=float, default=0.0,
                    help="Mask pseudo pixels below this per-pixel confidence (0=no masking).")
    ft.add_argument("--curated_pseudo", action="store_true", default=False,
                    help="Use curated_index.json for deterministic top-N patch selection.")
    ft.add_argument("--curated_index_file", default="curated_index.json",
                    help="Index file for --curated_pseudo (default curated_index.json; FPS: "
                         "diversity_N<n>_index.json)")
    ft.add_argument("--pretrained_backbone", default=None,
                    help="Path to a spectral pre-training checkpoint (from pretrain_spectral). "
                         "Loads model.backbone.* weights before fine-tuning; head is re-initialised.")


    # distill
    ds = sub.add_parser("distill", help="Knowledge distillation (with optional DFKD)")
    _add_model_args(ds)
    _add_data_args(ds)
    _add_train_args(ds)
    _add_infra_args(ds)
    ds.add_argument("--alpha",       type=float, default=0.5)
    ds.add_argument("--temperature", type=float, default=4.0)
    ds.add_argument("--at_beta",     type=float, default=0.0)
    ds.add_argument("--feat_alpha",           type=float, default=0.0)
    ds.add_argument("--gamma",       type=float, default=2.0)
    # DFKD flags
    ds.add_argument("--pseudo_dir",  default=None,
                    help="Directory of pseudo-labeled .pt files (DFKD).")
    ds.add_argument("--no_labeled",  action="store_true",
                    help="Group F: train on pseudo-labels only, no Sen1Floods11.")
    ds.add_argument("--n_samples",   type=int, default=None,
                    help="Cap pseudo dataset to N randomly selected scenes.")
    ds.add_argument("--n_labeled_samples", type=int, default=None,
                    help="Subsample N images from the labeled Sen1Floods11 train split "
                         "(252 total). For labeled N-ablation (DFKD baseline).")
    ds.add_argument("--pixel_conf_threshold", type=float, default=0.0,
                    help="Mask pseudo-labeled pixels below this per-pixel confidence.")
    ds.add_argument("--curated_pseudo", action="store_true", default=False,
                    help="Use curated_index.json (top-N by 4·p·(1-p)·mean_conf) "
                         "instead of random sampling from pseudo_dir.")
    ds.add_argument("--curated_index_file", default="curated_index.json",
                    help="Index file when --curated_pseudo is set (default: curated_index.json).")
    ds.add_argument("--pretrained_backbone", default=None,
                    help="Path to a spectral pre-training checkpoint (from pretrain_spectral). "
                         "Loads model.backbone.* weights before distillation; head is re-initialised.")


    # pseudo
    ps = sub.add_parser("pseudo", help="Pseudo-label generation subcommands")
    ps_sub = ps.add_subparsers(dest="pseudo_command", required=True)

    gen = ps_sub.add_parser("generate", help="Generate pseudo-labels from teacher")
    _add_model_args(gen)
    _add_infra_args(gen)
    gen.add_argument("--unlabeled_root", required=True,
                     help="Root of unlabeled GeoTIFF directory.")
    gen.add_argument("--unlabeled_source", default="flood", choices=["flood"],
                     help="The GDACS flood-event pool (B8A). Kept as a flag so the "
                          "published commands keep working; there is no other pool.")
    gen.add_argument("--confidence_threshold", type=float, default=0.90)
    gen.add_argument("--min_flood_frac",       type=float, default=0.0)
    gen.add_argument("--store_pixel_confidence", action="store_true")
    gen.add_argument("--store_soft_logits",      action="store_true")
    gen.add_argument("--img_size",    type=int, default=224)
    gen.add_argument("--batch_size",  type=int, default=32)
    gen.add_argument("--num_workers", type=int, default=8)

    rel = ps_sub.add_parser(
        "relabel",
        help="Re-label an existing pseudo pool with a different teacher "
             "(bridge / TA-chain; keeps images + pool membership identical)")
    _add_model_args(rel)
    _add_infra_args(rel)
    rel.add_argument("--pseudo_dir", required=True,
                     help="Source pool of pseudo_*.pt files (e.g. ./data/pseudo_gdacs).")
    rel.add_argument("--index_file", default="diversity_N2500_index.json",
                     help="Relabel only the files in this index (copied to --out_dir). "
                          "'none' = relabel every *.pt in --pseudo_dir.")
    rel.add_argument("--store_pixel_confidence", action="store_true", default=True)
    rel.add_argument("--no_pixel_confidence", dest="store_pixel_confidence",
                     action="store_false")
    rel.add_argument("--store_soft_logits", action="store_true")
    rel.add_argument("--batch_size",  type=int, default=32)
    rel.add_argument("--num_workers", type=int, default=4)

    # ofa (primary objective; core flags only)
    of = sub.add_parser("ofa", help="OFA-KD: exit-branch logit distillation (NeurIPS 2023)")
    _add_model_args(of)
    _add_data_args(of)
    _add_train_args(of)
    _add_infra_args(of)
    of.add_argument("--alpha",       type=float, default=0.5,
                    help="Logit-KD weight in the main-head loss.")
    of.add_argument("--temperature", type=float, default=4.0, help="KD temperature.")
    of.add_argument("--ofa_alpha",   type=float, default=0.3,
                    help="Weight on the summed OFA branch losses.")
    of.add_argument("--gamma",       type=float, default=1.2, help="Focal gamma for branch CE.")
    of.add_argument("--upsample_first", action="store_true",
                    help="Bilinear-upsample branch features before the 1x1 head (dense OFA).")
    # Pseudo-label data (DFKD-style); curated/diversity selection
    of.add_argument("--pseudo_dir",  default=None,
                    help="Directory of pseudo-labeled .pt files.")
    of.add_argument("--no_labeled",  action="store_true",
                    help="Train on pseudo-labels only (no Sen1Floods11 GT).")
    of.add_argument("--n_samples",   type=int, default=None,
                    help="Cap pseudo dataset to N scenes (5000 resolves to the full 2500 diversity pool).")
    of.add_argument("--n_labeled_samples", type=int, default=None,
                    help="Use N of the 252 Sen1Floods11 training images")
    of.add_argument("--curated_pseudo", action="store_true", default=False,
                    help="Use an index file for deterministic scene selection.")
    of.add_argument("--curated_index_file", default="curated_index.json",
                    help="Index file when --curated_pseudo is set "
                         "(use diversity_N<n>_index.json for the FPS-curated diversity pool).")
    of.add_argument("--pretrained_backbone", default=None,
                    help="Optional spectral pre-training checkpoint (loads model.backbone.* weights).")
    # Gradient-conflict logging / PCGrad (analysis toggles; see analysis/grad_conflict.py)
    of.add_argument("--grad_conflict",          action="store_true",
                    help="Log inter-branch gradient cosine similarity to grad_conflict.csv.")
    of.add_argument("--grad_conflict_interval", type=int, default=50)
    of.add_argument("--pcgrad",                 action="store_true",
                    help="Apply PCGrad gradient surgery across OFA branch losses.")
    # The remaining OFALoss toggles default OFF; set_defaults keeps cmd_ofa's call happy.
    of.set_defaults(disc_weighted=False, uncertainty_weighted=False,
                    dist_match_alpha=0.0, relational_alpha=0.0)


    ev = sub.add_parser("eval", help="Evaluate a checkpoint on val + test sets")
    _add_model_args(ev)
    _add_data_args(ev)
    _add_infra_args(ev)
    ev.add_argument("--model_ckpt", default=None, help="Student checkpoint path.")
    ev.add_argument("--target", default="student", choices=["student", "teacher"])
    ev.add_argument("--epochs",              type=int,   default=1)
    ev.add_argument("--lr",                  type=float, default=5e-5)
    ev.add_argument("--weight_decay",        type=float, default=0.05)
    ev.add_argument("--task_loss",           default="ce", choices=["focal", "ce"])
    ev.add_argument("--warmup_epochs",       type=int,   default=0)
    ev.add_argument("--early_stop_patience", type=int,   default=0)
    ev.add_argument("--use_amp",  action="store_true",   default=False)
    ev.add_argument("--no_amp",   dest="use_amp",        action="store_false")
    ev.add_argument("--amp_dtype", default="fp16")
    ev.add_argument("--grad_clip", type=float,           default=0.0)

    # ood-eval
    oe = sub.add_parser("ood-eval", help="Evaluate on an OOD flood dataset")
    _add_model_args(oe)
    _add_data_args(oe)
    _add_infra_args(oe)
    oe.add_argument("--model_ckpt", default=None)
    oe.add_argument("--target", default="student", choices=["student", "teacher"])
    oe.add_argument("--ood_dataset", default="floodplanet",
                    choices=["floodplanet", "worldfloods", "sturm_s2", "sturm_s1",
                             "deepflood", "urban_sar", "sen1floods11"])
    oe.add_argument("--ood_root",    required=True)
    oe.add_argument("--ood_out_dir", default="./runs/ood")

    # viz
    vz = sub.add_parser("viz", help="Visualise predictions on Sen1Floods11 splits")
    _add_model_args(vz)
    _add_data_args(vz)
    _add_infra_args(vz)
    vz.add_argument("--model_ckpt", default=None)
    vz.add_argument("--n_samples",  type=int, default=8)
    vz.add_argument("--viz_dir",    default="./runs/viz")

    return root

# Entry point

def main():
    parser = build_parser()
    args   = parser.parse_args()

    from utils.seeding import seed_basic
    seed = None if getattr(args, "seed", 0) == -1 else getattr(args, "seed", 0)
    if seed is not None:
        seed_basic(seed)

    dispatch = {
        "finetune": cmd_finetune,
        "distill":  cmd_distill,
        "ofa":      cmd_ofa,
        "eval":     cmd_eval,
        "ood-eval": cmd_ood_eval,
        "viz":      cmd_viz,
    }

    if args.subcommand == "pseudo":
        if args.pseudo_command == "generate":
            cmd_pseudo_generate(args)
        elif args.pseudo_command == "relabel":
            cmd_pseudo_relabel(args)
        else:
            parser.error(f"Unknown pseudo subcommand: {args.pseudo_command}")
    elif args.subcommand in dispatch:
        dispatch[args.subcommand](args)
    else:
        parser.error(f"Unknown subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
