"""CKA between teacher neck levels and student stages, either on a spatial grid
(the paper uses 8x8) or globally pooled. Pooled and spatial numbers are not the
same probe, don't mix them.

Use the same probe batches, in the same order, for teacher and student features.
Save the probe split, size, number of batches and seed with every result.

Kornblith et al., ICML 2019, arXiv:1905.00414.
See python analysis/cka.py --help for checkpoint and probe options.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# project root on sys.path, needed for the lazy models/data imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


# CKA core

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA for paired feature matrices X=(N,p), Y=(N,q), on CPU in float32."""
    X = X.float() - X.float().mean(0)
    Y = Y.float() - Y.float().mean(0)
    # HSIC numerator: ||X^T Y||_F^2; denominators: ||X^T X||_F, ||Y^T Y||_F
    xy = (X.T @ Y).norm(p="fro").square()
    xx = (X.T @ X).norm(p="fro")
    yy = (Y.T @ Y).norm(p="fro")
    denom = xx * yy
    if denom < 1e-12:
        return 0.0
    return (xy / denom).item()


# Feature extraction helpers

def _teacher_neck_features(teacher: torch.nn.Module,
                            images: torch.Tensor) -> list[torch.Tensor]:
    """Prithvi backbone + neck, returns the 4 neck maps (B, D, h, w)."""
    with torch.no_grad():
        # float32 even with AMP, to avoid precision noise in CKA
        token_lists = teacher.backbone.forward_features(images.float())
        return teacher.neck(token_lists)   # list[Tensor(B, D, h, w)]


def _student_stage_features(student: torch.nn.Module,
                             images: torch.Tensor,
                             n: int = 4) -> list[torch.Tensor]:
    """Last n stage feature maps of the student backbone (without the segmentation head).

    EfficientViT: "stageN" keys of student.model.backbone. ViT-Tiny: four evenly spaced
    blocks. TinyViT: features_only output. UNet and MobileNet are not supported.
    """
    with torch.no_grad():
        # EfficientViT
        if hasattr(student, "model") and hasattr(student.model, "backbone"):
            out = student.model.backbone(images)
            if isinstance(out, dict):
                # Numeric stage keys only; "stage_final" is the same tensor as stage4.
                stages = [v for k, v in out.items() if re.match(r"^stage\d+$", k)]
            else:
                stages = list(out)
            if not stages:
                raise RuntimeError(
                    "EfficientViT backbone returned no 'stageN' keys. "
                    f"Got keys: {list(out.keys()) if isinstance(out, dict) else type(out)}"
                )
            return stages[-n:]

        # TinyViT (features_only)
        if getattr(student, "_stage_features_type", None) == "spatial_list":
            out = student.backbone(images)
            return list(out)[-n:]

        # ViT-Tiny (hook-based)
        if hasattr(student, "backbone") and not hasattr(student, "model"):
            backbone = student.backbone
            n_blocks = len(backbone.blocks)
            # Tap 4 equally-spaced blocks (floor division to stay in range)
            step = max(1, n_blocks // 4)
            tap_indices = [step - 1, 2 * step - 1, 3 * step - 1, n_blocks - 1]
            tap_indices = sorted(set(max(0, min(i, n_blocks - 1)) for i in tap_indices))

            captured: list[torch.Tensor] = []

            def _hook(module, inp, out):
                # out: (B, N+1, D) - strip CLS, reshape to spatial map
                B, NP1, D = out.shape
                N = NP1 - 1
                h = int(N ** 0.5)
                captured.append(out[:, 1:].permute(0, 2, 1).reshape(B, D, h, h).detach())

            hooks = [backbone.blocks[i].register_forward_hook(_hook) for i in tap_indices]
            try:
                backbone.forward_features(images)
            finally:
                for h in hooks:
                    h.remove()

            return captured[-n:]

    raise RuntimeError(
        "_student_stage_features: unsupported architecture. "
        "Supported: EfficientViT (student.model.backbone), "
        "TinyViT (_stage_features_type='spatial_list'), "
        "ViT-Tiny (student.backbone without student.model). "
        f"Got: {type(student)}"
    )


# Feature collection over dataloader

def _collect_features(
    extract_fns,
    dataloader,
    device: torch.device,
    n_batches: int,
    spatial_size: int | None = None,
) -> list[list[torch.Tensor]]:
    """Reduce feature maps to (samples, C) and accumulate over batches.

    All extract_fns see the same batch in the same pass, so teacher and student features
    stay paired. spatial_size=None pools each map to one vector per image; an integer
    resamples to a spatial_size x spatial_size grid with one sample per location (paper: 8).
    """
    def reduce(feats):
        if spatial_size is None:
            return [F.adaptive_avg_pool2d(f.float(), 1).flatten(1).cpu() for f in feats]
        # (B, C, g, g) -> (B*g*g, C): each grid cell is a sample
        return [F.adaptive_avg_pool2d(f.float(), spatial_size)
                 .permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu() for f in feats]

    accumulated: list[list[list[torch.Tensor]]] = [[] for _ in extract_fns]
    n_seen = 0
    for batch in dataloader:
        if n_seen >= n_batches:
            break
        images = batch["image"].to(device)
        for k, fn in enumerate(extract_fns):
            pooled = reduce(fn(images))
            if not accumulated[k]:
                accumulated[k] = [[] for _ in pooled]
            if len(pooled) != len(accumulated[k]):
                raise RuntimeError(
                    f"Feature level count changed mid-run: expected "
                    f"{len(accumulated[k])}, got {len(pooled)}"
                )
            for lvl, p in zip(accumulated[k], pooled):
                lvl.append(p)
        n_seen += 1

    if not accumulated[0] or not accumulated[0][0]:
        raise RuntimeError("No features collected — dataloader returned no batches.")
    if n_seen < n_batches:
        # few samples vs. 1024 teacher channels, CKA comes out too high
        print(f"[cka] WARNING: probe exhausted after {n_seen} batches, {n_batches} requested "
              f"({accumulated[0][0][0].shape[0] * n_seen} samples). Absolute CKA is biased "
              f"upward at small sample counts.")

    return [[torch.cat(lvl, dim=0) for lvl in per_fn] for per_fn in accumulated]


# Main CKA computation

def run_cka_analysis(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader,
    device: torch.device,
    n_batches: int = 50,
    n_student_stages: int = 4,
    spatial_size: int | None = None,
) -> list[list[float]]:
    """Return matrix[i][j] = CKA(teacher neck level i, student stage j).

    Interpret values within a fixed probe protocol; there is no universal transfer threshold.
    """
    student.eval()
    teacher.eval()

    print(f"[cka] collecting teacher and student features over up to {n_batches} batches …")
    t_feats, s_feats = _collect_features(
        [lambda imgs: _teacher_neck_features(teacher, imgs),
         lambda imgs: _student_stage_features(student, imgs, n=n_student_stages)],
        dataloader, device, n_batches, spatial_size,
    )

    n_t, n_s = len(t_feats), len(s_feats)
    n_samples = t_feats[0].shape[0]
    print(f"[cka] computing {n_t}×{n_s} matrix over {n_samples} samples …")

    matrix = [
        [linear_cka(t_feats[i], s_feats[j]) for j in range(n_s)]
        for i in range(n_t)
    ]

    _print_matrix(matrix, n_t, n_s)
    return matrix


def _print_matrix(matrix: list[list[float]], n_t: int, n_s: int) -> None:
    col_w = 12
    header = " " * 16 + "".join(f"stud_s{j}".center(col_w) for j in range(n_s))
    print()
    print("CKA similarity  (teacher neck level × student backbone stage)")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, row in enumerate(matrix):
        vals = "".join(f"{v:.4f}".center(col_w) for v in row)
        print(f"  teach_lvl{i}      {vals}")
    print()
    diag_n = min(n_t, n_s)
    diag   = [matrix[i][i] for i in range(diag_n)]
    print(f"  Diagonal CKA:     {[f'{v:.4f}' for v in diag]}")
    print(f"  Mean diag CKA:    {sum(diag)/len(diag):.4f}")
    max_per_t = [max(row) for row in matrix]
    print(f"  Max per teach lvl:{[f'{v:.4f}' for v in max_per_t]}")
    print()


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CKA analysis: Prithvi teacher neck levels × student backbone stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Teacher
    p.add_argument("--official_ckpt", default=None,
                   help="Path to teacher fine-tuned checkpoint (best.pt).")
    p.add_argument("--teacher_ckpt", default=None,
                   help="Alias for --official_ckpt (legacy).")
    p.add_argument("--teacher_variant", default="prithvi_eo_v2_300_tl",
                   choices=["prithvi_eo_v2_300", "prithvi_eo_v2_300_tl",
                             "prithvi_eo_v2_600", "prithvi_eo_v2_600_tl"])

    # Student
    p.add_argument("--student_arch", default="efficientvit_b0",
                   choices=["efficientvit_b0", "efficientvit_b0_relu6",
                             "efficientvit_b1", "efficientvit_b2",
                             "vit_tiny", "tiny_vit_5m", "tiny_vit_11m", "tiny_vit_21m",
                             "unet_resnet50", "unet_resnet101",
                             "mobilenet_v3_small"])
    p.add_argument("--student_ckpt", default=None,
                   help="Path to trained student checkpoint (best.pt). "
                        "Omit to run with random-init student (sanity check).")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Do not load ImageNet pretrained weights for student backbone.")
    p.add_argument("--pretrained_tag", default="dist_in22k_ft_in1k")
    p.add_argument("--num_classes", type=int, default=2)

    # Data
    from utils.paths import dataset_root
    p.add_argument("--data_root", default=str(dataset_root("S11_ROOT")),
                   help="Sen1Floods11 data root, default $DATA_ROOT/sen1floods11 (used when --pseudo_dir is not given "
                        "or for val/test sets).")
    p.add_argument("--pseudo_dir", default=None,
                   help="Pseudo-label directory. If given, probe features on this "
                        "dataset instead of the Sen1Floods11 val split.")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Max pseudo samples to use as probe data.")
    p.add_argument("--use_curated", action="store_true",
                   help="Load files from a curated index (diversity_N<n>_index.json etc.).")
    p.add_argument("--index_file", default="diversity_N2500_index.json",
                   help="Curated index filename inside --pseudo_dir.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--probe_split", default="val", choices=["train", "val", "test"],
                   help="Sen1Floods11 probe split: val (default, 89 scenes, at most 12 batches "
                        "of 8) or train (252). Report it with the result.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--eval_img_size", type=int, default=512)

    # CKA settings
    p.add_argument("--n_batches", type=int, default=50,
                   help="Number of batches used to estimate the CKA matrix.")
    p.add_argument("--n_student_stages", type=int, default=4,
                   help="Number of student backbone stages to tap.")

    # Misc
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    p.add_argument("--out_json", default=None,
                   help="Optional path to save the CKA matrix as JSON.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cka_mode", default="spatial", choices=["spatial", "pooled"],
                   help="spatial (default, paper): each location of a --spatial_size grid is "
                        "a sample; pooled: one average vector per image")
    p.add_argument("--spatial_size", type=int, default=8)

    # Batch comparison mode
    p.add_argument(
        "--compare_ckpts",
        nargs="*",
        default=None,
        metavar="LABEL:PATH",
        help="Several student checkpoints as 'label:path' pairs, e.g. "
             "'random: ce:runs/<run>/best.pt ofa:runs/<run>/best.pt' ('random:' = untrained). "
             "Replaces --student_ckpt; UNet is skipped.",
    )

    return p.parse_args()


def _verify_student_support(
    student: torch.nn.Module,
    device: torch.device,
    raise_on_fail: bool = False,
) -> bool:
    """Check feature extraction before reading the full probe; optionally raise on failure."""
    dummy = torch.zeros(1, 6, 64, 64, device=device)
    try:
        feats = _student_stage_features(student, dummy)
        print(f"[cka] student stage dims: {[tuple(f.shape[1:]) for f in feats]}")
        return True
    except RuntimeError as e:
        msg = f"[cka] student architecture not supported for CKA.\n  {e}"
        if raise_on_fail:
            raise RuntimeError(msg) from e
        print(f"[cka] ERROR: {msg}")
        sys.exit(1)


def _probe_record(args):
    """Probe settings to save with the result, absolute CKA depends on them."""
    return {"source": "pseudo" if args.pseudo_dir else f"sen1floods11:{args.probe_split}",
            "batch_size": args.batch_size, "img_size": args.img_size,
            "eval_img_size": args.eval_img_size, "n_batches_requested": args.n_batches}


def _build_dataloader(args, device):
    """Build probe dataloader: pseudo data if --pseudo_dir, else val split."""
    if args.pseudo_dir:
        from data.pseudo_labeled import PseudoLabeledDataset
        from torch.utils.data import DataLoader

        ds = PseudoLabeledDataset(
            pseudo_dir  = args.pseudo_dir,
            n_samples   = args.n_samples,
            use_curated = args.use_curated,
            index_file  = args.index_file,
        )
        print(f"[cka] probe data: pseudo dataset ({len(ds)} samples)")
        return DataLoader(
            ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = True,
        )
    else:
        from data.sen1floods11 import get_dataloaders

        dls = get_dataloaders(
            data_root      = args.data_root,
            batch_size     = args.batch_size,
            num_workers    = args.num_workers,
            img_size       = args.img_size,
            eval_img_size  = args.eval_img_size,
        )
        dl = dict(zip(("train", "val", "test"), dls))[args.probe_split]
        print(f"[cka] probe data: Sen1Floods11 {args.probe_split} split "
              f"({len(dl.dataset)} scenes, {len(dl)} batches)")
        return dl


def _run_batch_cka(
    args,
    teacher: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> None:
    """Run CKA for each 'label:path' in --compare_ckpts and print a table.

    An empty path means untrained; the untrained baseline is added unless "random:" is given.
    """
    from models.student import build_student

    # Parse "label:path" pairs
    entries: list[tuple[str, str | None]] = []
    for token in (args.compare_ckpts or []):
        if ":" not in token:
            raise ValueError(
                f"--compare_ckpts entries must be 'label:path' (got {token!r}). "
                "Use 'random:' for untrained baseline."
            )
        label, path = token.split(":", 1)
        entries.append((label, path.strip() or None))

    # add the untrained baseline unless it was passed already
    has_random = any(p is None for _, p in entries)
    if not has_random:
        entries = [("random", None)] + entries

    print(f"\n[cka] batch comparison: {len(entries)} students")
    print(f"[cka] teacher: {args.teacher_variant}")
    print()

    all_results: list[dict] = []

    for label, ckpt_path in entries:
        print(f"{'─'*60}")
        print(f"[cka] ── {label}  (ckpt={ckpt_path or 'random-init'})")

        student = build_student(
            arch           = args.student_arch,
            in_channels    = 6,
            num_classes    = args.num_classes,
            pretrained     = not args.no_pretrained,
            pretrained_tag = args.pretrained_tag,
        )

        if ckpt_path:
            raw   = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = raw.get("student", raw.get("model", raw.get("state_dict", raw)))
            missing, unexpected = student.load_state_dict(state, strict=False)
            if missing:
                print(f"  missing keys ({len(missing)}): {missing[:3]} …")
            print(f"  loaded: {ckpt_path}")
        else:
            print(f"  using random-init student (ImageNet pretrain only)")

        student = student.to(device).eval()
        for p in student.parameters():
            p.requires_grad_(False)

        # Skip if architecture is not CKA-compatible (e.g. UNet)
        dummy = torch.zeros(1, 6, 64, 64, device=device)
        try:
            test_feats = _student_stage_features(student, dummy)
            print(f"  stage dims: {[tuple(f.shape[1:]) for f in test_feats]}")
        except RuntimeError as e:
            print(f"  [SKIP] architecture not CKA-compatible: {e}")
            all_results.append({"label": label, "skipped": True,
                                 "reason": str(e)})
            continue

        matrix = run_cka_analysis(
            student          = student,
            teacher          = teacher,
            dataloader       = dataloader,
            device           = device,
            n_batches        = args.n_batches,
            n_student_stages = args.n_student_stages,
            spatial_size     = args.spatial_size if args.cka_mode == "spatial" else None,
        )
        diag_n = min(len(matrix), len(matrix[0]) if matrix else 0)
        diag   = [matrix[i][i] for i in range(diag_n)]
        mean_d = sum(diag) / len(diag) if diag else 0.0
        all_results.append({
            "label":        label,
            "ckpt":         ckpt_path,
            "cka_matrix":   matrix,
            "diagonal":     diag,
            "mean_diagonal": mean_d,
            "skipped":      False,
        })

    # Summary table
    print(f"\n{'═'*60}")
    print("CKA BATCH COMPARISON SUMMARY")
    print(f"{'─'*60}")
    col_w = 10
    n_stages = args.n_student_stages
    header = f"  {'label':<20}" + "".join(f"diag_s{i}".center(col_w)
                                           for i in range(n_stages)) + "  mean_diag"
    print(header)
    print(f"  {'─'*len(header)}")
    for r in all_results:
        if r.get("skipped"):
            print(f"  {r['label']:<20}  [skipped — UNet or unsupported arch]")
            continue
        diag_str = "".join(f"{v:.4f}".center(col_w) for v in r["diagonal"])
        print(f"  {r['label']:<20}{diag_str}  {r['mean_diagonal']:.4f}")
    print()

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "teacher_variant":  args.teacher_variant,
                "student_arch":     args.student_arch,
                "n_batches":        args.n_batches,
                "cka_mode":         args.cka_mode,
                "spatial_size":     args.spatial_size if args.cka_mode == "spatial" else None,
                "probe":            _probe_record(args),
                "comparisons":      all_results,
            }, f, indent=2)
        print(f"[cka] batch results saved to {args.out_json}")


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[cka] device: {device}")

    # Teacher
    from models.teacher import build_teacher

    ckpt_path = args.official_ckpt or args.teacher_ckpt
    teacher = build_teacher(
        variant     = args.teacher_variant,
        num_classes = args.num_classes,
        pretrained  = ckpt_path is None,
        official_ckpt = ckpt_path,
    )
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[cka] teacher ready ({args.teacher_variant})")

    dataloader = _build_dataloader(args, device)

    # Batch comparison mode
    if args.compare_ckpts is not None:
        _run_batch_cka(args, teacher, dataloader, device)
        return

    # Single-checkpoint mode
    from models.student import build_student

    student = build_student(
        arch           = args.student_arch,
        in_channels    = 6,
        num_classes    = args.num_classes,
        pretrained     = not args.no_pretrained,
        pretrained_tag = args.pretrained_tag,
    )

    if args.student_ckpt:
        raw = torch.load(args.student_ckpt, map_location="cpu", weights_only=True)
        state = raw.get("student", raw.get("model", raw.get("state_dict", raw)))
        missing, unexpected = student.load_state_dict(state, strict=False)
        if missing:
            print(f"[cka] student missing keys ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"[cka] student unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
        print(f"[cka] student loaded: {args.student_ckpt}")
    else:
        print("[cka] WARNING: no --student_ckpt supplied — using random-init student")

    student = student.to(device)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)

    _verify_student_support(student, device)

    matrix = run_cka_analysis(
        student          = student,
        teacher          = teacher,
        dataloader       = dataloader,
        device           = device,
        n_batches        = args.n_batches,
        n_student_stages = args.n_student_stages,
        spatial_size     = args.spatial_size if args.cka_mode == "spatial" else None,
    )

    if args.out_json:
        payload = {
            "student_arch":     args.student_arch,
            "student_ckpt":     args.student_ckpt,
            "teacher_variant":  args.teacher_variant,
            "teacher_ckpt":     ckpt_path,
            "n_batches":        args.n_batches,
            "probe":            _probe_record(args),
            "cka_matrix":       matrix,
            "diagonal":         [matrix[i][i] for i in range(min(len(matrix), len(matrix[0])))],
            "mean_diagonal":    sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0])))) /
                                min(len(matrix), len(matrix[0])),
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[cka] results saved to {args.out_json}")


if __name__ == "__main__":
    main()
