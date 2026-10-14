"""CPU Trainer gate: live teacher, removable fixed feature term, source-only fit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import TrainConfig
from distillation.compression_metric import (
    FrozenFeatureAdapter, FrozenTeacherTransform, feature_loss, probe_features,
)
from training.compression_loss import CompressionLoss
from training.losses import focal_loss, kd_loss
from training.trainer import Trainer


class NeckList(nn.Module):
    def forward(self, x):
        return [x * .5, x]


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(6, 4, 1), nn.ReLU6())
        self.neck = NeckList()
        self.decoder = nn.Conv2d(4, 2, 1)
        self.calls = []
        self.eval()
        self.requires_grad_(False)

    def forward(self, x):
        self.calls.append({"input": x.detach().clone(), "grad_enabled": torch.is_grad_enabled(),
                           "training": self.training})
        return self.decoder(self.neck(self.encoder(x))[-1])


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(6, 4, 1), nn.ReLU6())
        self.decoder = nn.Conv2d(4, 2, 1)
        with torch.no_grad():
            self.decoder.weight.zero_()
            self.decoder.bias.copy_(torch.tensor([1., -1.]))
        self.training_inputs = []

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            self.training_inputs.append(x.detach().clone())
        return self.decoder(self.encoder(x))


class AugmentedSourceData(Dataset):
    def __init__(self):
        self.images = torch.randn(4, 6, 16, 16)
        self.masks = torch.zeros(4, 16, 16, dtype=torch.long)
        self.masks[:, :, 8:] = 1
        self.masks[:, :2] = -1
        self.gains = .8 + .4 * torch.rand(4, 6, 1, 1)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        # This gate deliberately requires the teacher to see gain/rotation too.
        return {"image": torch.rot90(self.images[index], index, (-2, -1)) * self.gains[index],
                "mask": torch.rot90(self.masks[index], index, (-2, -1))}


class ForbiddenTerminalLoader:
    def __iter__(self):
        raise AssertionError("source-only training attempted terminal test evaluation")

    def __len__(self):
        raise AssertionError("source-only training inspected terminal test data")


class LocalTrainer(Trainer):
    def _init_wandb(self):
        return None

    def _build_optimizer(self, model, cfg):
        self.checked_optimizer = super()._build_optimizer(model, cfg)
        return self.checked_optimizer


def check() -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    teacher, student = ToyTeacher(), ToyStudent()
    transform = FrozenTeacherTransform(torch.zeros(4), torch.eye(4), torch.ones(4))
    adapter = FrozenFeatureAdapter(torch.eye(4), torch.zeros(4))
    metric = torch.eye(4)
    criterion = CompressionLoss(teacher, student, transform, adapter, metric,
                                beta=.2, diagnostic_interval=1)
    assert not list(criterion.parameters()), "fixed training feature modules acquired trainable parameters"
    frozen_buffers = {k: v.detach().clone() for k, v in criterion.state_dict().items()}
    teacher_state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    loader = DataLoader(AugmentedSourceData(), batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    student.eval()
    logits = student(batch["image"])
    value, components = criterion(logits, batch["image"], batch["mask"])
    assert torch.isfinite(value)
    assert criterion.capture.student is None and criterion.capture.teacher is None
    with torch.no_grad():
        t_logits = teacher(batch["image"])
        sv, w = probe_features(student.encoder(batch["image"]), batch["mask"])
        tv, _ = probe_features(teacher.encoder(batch["image"]), batch["mask"])
        expected = (.5 * focal_loss(logits, batch["mask"], gamma=2.)
                    + .5 * kd_loss(logits, t_logits, temperature=4., valid_mask=batch["mask"] != -1)
                    + .2 * feature_loss(transform(tv), adapter(sv), w, metric))
    assert torch.allclose(value, expected, atol=1e-7, rtol=1e-6)
    criterion.capture.clear()
    value.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in student.encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(torch.equal(teacher_state[k], v) for k, v in teacher.state_dict().items())
    assert all(torch.equal(frozen_buffers[k], v) for k, v in criterion.state_dict().items())
    assert criterion.diagnostics and criterion.diagnostics[0]["feature_gradient_rms"] > 0
    # No cached teacher path is allowed under the fixed live-augmentation recipe.
    rejected = False
    try:
        criterion(student(batch["image"]), batch["image"], batch["mask"],
                  precomputed_teacher_logits=t_logits)
    except (ValueError, RuntimeError):
        rejected = True
    assert rejected, "cached teacher logits must be explicitly rejected"
    assert criterion.capture.student is None and criterion.capture.teacher is None
    teacher.calls.clear()
    student.training_inputs.clear()

    with tempfile.TemporaryDirectory(prefix="flooddistill_compression_integration_") as directory:
        cfg = TrainConfig(epochs=2, lr=1e-4, weight_decay=.05, task_loss="focal", use_amp=False,
                          grad_clip=5., scheduler="cosine", early_stop_patience=0,
                          val_every_n_epochs=1, val_smoothing_window=1, seed=42,
                          out_dir=directory, defer_terminal_evaluation=True)
        with patch("torch.cuda.is_available", return_value=False):
            trainer = LocalTrainer(student, loader, loader, ForbiddenTerminalLoader(), cfg, criterion)
            results = trainer.fit()
        assert trainer.device.type == "cpu" and not any(k.startswith("test_") for k in results)
        assert len(student.training_inputs) == len(teacher.calls) == 4
        assert all(torch.equal(x, call["input"]) for x, call in zip(student.training_inputs, teacher.calls))
        assert all(not c["grad_enabled"] and not c["training"] for c in teacher.calls)
        optimizer_ids = {id(p) for group in trainer.checked_optimizer.param_groups for p in group["params"]}
        assert optimizer_ids == {id(p) for p in student.parameters()}
        assert optimizer_ids.isdisjoint({id(p) for p in teacher.parameters()})
        best = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(Path(directory) / "last.pt", map_location="cpu", weights_only=True)
        assert set(best["model"]) == set(student.state_dict())
        assert all("adapter" not in k and "transform" not in k and "teacher" not in k
                   for k in best["model"])
        assert last["training_state"]["global_step"] == 4
        assert best["loss_policy"]["main_kl_mask"] == "valid_only"
        assert best["rng_state"] and last["optimizer"] and last["scheduler"]
        fresh = ToyStudent().eval()
        fresh.load_state_dict(best["model"], strict=True)
        student.eval()
        with torch.no_grad():
            assert torch.equal(fresh(batch["image"]), student(batch["image"]))
        assert all(torch.equal(teacher_state[k], v) for k, v in teacher.state_dict().items())
        assert all(torch.equal(frozen_buffers[k], v) for k, v in criterion.state_dict().items())
    criterion.remove_hooks()
    assert criterion.capture.student is None and criterion.capture.teacher is None
    assert not student.encoder._forward_hooks and not teacher.neck._forward_hooks
    return {"status": "passed", "device": "cpu", "checks": [
        "existing_trainer_source_only_without_terminal_access", "live_identical_augmented_teacher_input",
        "frozen_eval_no_grad_teacher", "exact_existing_valid_focal_plus_two_class_kd",
        "fixed_adapter_transform_not_in_optimizer_or_inference_model", "finite_feature_encoder_gradients",
        "capture_cleared_after_step_and_cached_target_rejection", "task_feature_gradient_diagnostics",
        "strict_inference_checkpoint_reload", "training_rng_optimizer_scheduler_saved", "training_hooks_removed"],
        "completed_updates": 4, "best_source_miou": results["best_val_mIoU"],
        "terminal_evaluation": "forbidden_loader_never_accessed",
        "resume_scope": "state serialization/reload here; exact QAT optimizer resume in compression_quant_check"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "--output", dest="out", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    result = check()
    result["elapsed_seconds"] = time.monotonic() - start
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
