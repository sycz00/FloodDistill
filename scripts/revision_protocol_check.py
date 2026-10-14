"""Small concrete protocol checks; uses the pinned real teacher on CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WANDB_MODE", "disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arch", default="efficientvit_b0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--pseudo_pool")
    ap.add_argument("--pseudo_index")
    args = ap.parse_args()
    import torch
    from distillation.ofa_kd import OFALoss
    from models.student import build_student
    from models.teacher import build_teacher
    from data.sen1floods11 import NORM_MEAN, NORM_STD, get_dataloaders
    from evaluation.perscene_eval import SpectralIndexPredictor
    from evaluation.revision_eval import atomic_json, sha256
    from utils.metrics import accumulate_metrics, finalize_metrics

    start = time.monotonic()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Declared CUDA unavailable; refusing substitution")
        torch.cuda.reset_peak_memory_stats(device)
    teacher = build_teacher(pretrained=False, freeze=True, official_ckpt=args.teacher).to(device)
    student = build_student(args.arch, pretrained=True).to(device)
    criterion = OFALoss(teacher=teacher, student=student, num_classes=2,
                        alpha=0.5, ofa_alpha=0.3, temperature=4.0, gamma=1.2)
    criterion.register_hooks()
    criterion.prewarm([{"image": torch.zeros(args.batch_size, 6, 224, 224)}], device)
    student.train()
    if args.pseudo_pool:
        from data.pseudo_labeled import PseudoLabeledDataset
        dataset = PseudoLabeledDataset(args.pseudo_pool, n_samples=2500,
            seed=42, use_curated=True, index_file=args.pseudo_index,
            mult_noise=True, geom_aug=True)
        batch = next(iter(torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)))
        images, masks = batch["image"].to(device), batch["mask"].to(device)
    else:
        images = torch.randn(args.batch_size, 6, 224, 224, device=device)
        masks = torch.randint(0, 2, (args.batch_size, 224, 224), device=device)
        masks[:, :2] = -1
    logits = student(images)
    loss, terms = criterion(logits, images, masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
               for branch in criterion.exit_branches)
    assert logits.shape == (args.batch_size, 2, 224, 224)
    optimizer = torch.optim.AdamW(list(student.parameters()) +
        list(criterion.exit_branches.parameters()), lr=5e-5, weight_decay=0.05)
    torch.nn.utils.clip_grad_norm_(list(student.parameters()) +
        list(criterion.exit_branches.parameters()), 5.0)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in student.parameters())
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        torch.save({"model": student.state_dict(), "loss": criterion.state_dict()}, path)
        loaded = torch.load(path, weights_only=True)
        student.load_state_dict(loaded["model"], strict=True)
        criterion.load_state_dict(loaded["loss"], strict=True)
    criterion.remove_hooks()
    student.eval()
    with torch.no_grad():
        evaluation = student(torch.zeros(1, 6, 512, 512, device=device))
        assert evaluation.shape == (1, 2, 512, 512)
    mu = torch.tensor(NORM_MEAN).view(1, 6, 1, 1)
    sd = torch.tensor(NORM_STD).view(1, 6, 1, 1)
    rho = torch.rand(2, 6, 32, 32)
    rule = SpectralIndexPredictor("mndwi_zero")
    baseline = rule((rho - mu) / sd).argmax(1)
    changes = {}
    for gain in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        pred = rule((rho * gain - mu) / sd).argmax(1)
        changes[str(gain)] = int((pred != baseline).sum())
    assert not any(changes.values())
    expected = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]])
    reference = torch.tensor([[[0, 1], [1, -1]]])
    confusion = accumulate_metrics(expected, reference, 2)
    assert torch.equal(confusion, torch.tensor([[1, 0], [0, 2]]))
    metrics = finalize_metrics(confusion)
    assert abs(metrics["mIoU"] - 1.0) < 2e-7
    assert all(abs(value - 1.0) < 2e-7 for value in metrics["iou_per_class"])
    train, val, test = get_dataloaders("/home/schmalstieg/data/sen1floods11", batch_size=2, num_workers=0)
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (252, 89, 90)
    atomic_json(args.out, {"status": "verified", "real_teacher_sha256": sha256(args.teacher),
        "device": str(device), "architecture": args.arch,
        "scientific_batch_size": args.batch_size,
        "input_source": "augmented frozen GDACS N2500" if args.pseudo_pool else "synthetic",
        "optimizer_update_finite": True,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "teacher_frozen_eval_no_grad": True,
        "six_band_output_shape": list(logits.shape), "loss_finite": True,
        "loss": float(loss.detach()), "terms": {k: float(v) for k, v in terms.items()},
        "student_and_exit_gradients": True, "checkpoint_main_and_aux_reload": True,
        "scalar_mndwi_changed_pixels": changes, "metric_ignore_hand_check": metrics,
        "s11_split_counts": [252, 89, 90], "elapsed_seconds": time.monotonic() - start,
        "scope": "Concrete CPU correctness checks; archived-evaluation and training-reproduction gates remain separate."})
    print("Real-teacher protocol checks passed", flush=True)


if __name__ == "__main__":
    main()
