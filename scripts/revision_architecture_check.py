"""CPU checks for the bounded revision students; no dataset or GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from distillation.ofa_kd import OFALoss, ofa_loss
from models.student import build_student


def check(arch, pretrained):
    torch.manual_seed(42)
    student = build_student(arch, pretrained=pretrained)
    teacher = nn.Conv2d(6, 2, 1).eval().requires_grad_(False)
    teacher_state = {key: value.clone() for key, value in teacher.state_dict().items()}
    images = torch.randn(2, 6, 224, 224)
    masks = torch.randint(0, 2, (2, 224, 224))
    masks[:, :4, :4] = -1
    criterion = OFALoss(teacher, student)
    criterion.register_hooks()
    criterion.prewarm([{"image": images}], torch.device("cpu"))
    assert len(criterion.exit_branches) == 4
    student.train()
    logits = student(images)
    assert logits.shape == (2, 2, 224, 224)
    features = list(criterion._stage_feats)
    channels = [feature.shape[1] for feature in features]
    strides = [224 // feature.shape[-1] for feature in features]
    assert strides == [4, 8, 16, 32]
    loss, components = criterion(logits, images, masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert not teacher.training and all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(value, teacher_state[key]) for key, value in teacher.state_dict().items())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in criterion.exit_branches.parameters())

    # Isolate exits from the main loss: every feature and the encoder stem must
    # receive a nonzero finite gradient through the existing OFA objective.
    student.zero_grad(set_to_none=True)
    criterion.zero_grad(set_to_none=True)
    student(images)
    features = list(criterion._stage_feats)
    for feature in features:
        feature.retain_grad()
    with torch.no_grad():
        teacher_logits = teacher(images)
    branch_loss = torch.stack([
        ofa_loss(branch(feature, (224, 224)), teacher_logits, masks)
        for branch, feature in zip(criterion.exit_branches, features)
    ]).mean()
    branch_loss.backward()
    stage_gradient_l1 = [feature.grad.abs().sum().item() for feature in features]
    assert all(value > 0 for value in stage_gradient_l1)
    assert all(torch.isfinite(feature.grad).all() for feature in features)
    encoder = student.encoder if hasattr(student, "encoder") else student.model.backbone
    stem_gradient = next(encoder.parameters()).grad
    assert stem_gradient is not None and torch.isfinite(stem_gradient).all() and stem_gradient.abs().sum() > 0
    native_aux = getattr(student, "native_aux_parameters", 0)
    if native_aux:
        assert not student.decoder.training
        assert student.decoder.first_conv[1].training
        assert all(p.grad is None for p in student.decoder.AuxHead.parameters())

    student.eval()
    with torch.no_grad():
        before = student(images)
        evaluation_shape = list(student(torch.randn(1, 6, 512, 512)).shape)
    assert evaluation_shape == [1, 2, 512, 512]
    criterion.remove_hooks()
    with torch.no_grad():
        after = student(images)
    assert torch.equal(before, after)
    assert all("exit_branch" not in key for key in student.state_dict())

    # Keep the existing main checkpoint interface and separately validate the
    # auxiliary state that the trainer includes in a resumable checkpoint.
    with tempfile.TemporaryDirectory(prefix="flooddistill-architecture-") as temp:
        checkpoint = Path(temp) / "check.pt"
        torch.save({"student": student.state_dict(), "loss_fn": criterion.state_dict()}, checkpoint)
        restored = build_student(arch, pretrained=False, ckpt_path=str(checkpoint)).eval()
        with torch.no_grad():
            assert torch.equal(after, restored(images))
        restored_criterion = OFALoss(teacher, restored)
        restored_criterion.register_hooks()
        restored_criterion.prewarm([{"image": images}], torch.device("cpu"))
        restored_criterion.load_state_dict(torch.load(checkpoint, weights_only=True)["loss_fn"])
        assert all(torch.equal(value, restored_criterion.state_dict()[key])
                   for key, value in criterion.state_dict().items())
        restored_criterion.remove_hooks()

    adapted_total = sum(p.numel() for p in student.parameters())
    return {
        "status": "passed", "device": "cpu", "seed": 42,
        "teacher": "frozen synthetic Conv2d only; actual Prithvi checked separately",
        "pretrained_loaded": pretrained,
        "pretrained_cfg": getattr(encoder, "pretrained_cfg", None),
        "channels": channels, "strides": strides, "branch_count": 4,
        "adapted_total_parameters": adapted_total,
        "main_inference_parameters": adapted_total - native_aux,
        "native_auxiliary_unused_parameters": native_aux,
        "ofa_auxiliary_parameters": sum(p.numel() for p in criterion.exit_branches.parameters()),
        "branch_loss_normalization": "mean valid pixels per branch, arithmetic mean across 4 branches",
        "finite_combined_loss": loss.item(), "finite_branch_loss": branch_loss.item(),
        "components": components, "branch_stage_gradient_l1": stage_gradient_l1,
        "checks": ["224 train output", "512 eval output", "finite main and branch loss",
                   "exit-only gradients to all stages and encoder stem", "frozen eval teacher",
                   "main inference identical after hook removal", "main checkpoint reload",
                   "auxiliary checkpoint reload"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arches", nargs="+", default=["mobilenet_v3_small", "lightformer_resnet18"])
    parser.add_argument("--out", type=Path, default=Path("experiments/revision/architecture_cpu_check.json"))
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {"torch": torch.__version__, "pretrained_requested": args.pretrained, "architectures": {}}
    failed = False
    for arch in args.arches:
        started = time.monotonic()
        try:
            results["architectures"][arch] = check(arch, args.pretrained)
        except Exception as error:
            failed = True
            results["architectures"][arch] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        results["architectures"][arch]["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps({arch: results["architectures"][arch]}, indent=2))
    checkpoint_files = list((Path(torch.hub.get_dir()) / "checkpoints").glob("mobilenet_v3_small-*.pth"))
    checkpoint_files += list((Path.home() / ".cache/huggingface/hub/models--timm--resnet18.a1_in1k/snapshots").glob("*/*"))
    results["pretraining_cached_files"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        for path in checkpoint_files if path.is_file()
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(args.out)
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
