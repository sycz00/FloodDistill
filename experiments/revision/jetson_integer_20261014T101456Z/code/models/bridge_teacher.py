"""Frozen student-checkpoint ensembles for teacher-assistant controls.

forward(x) returns logits shaped (B,C,H,W), as expected by DistillLoss and
OFALoss. In prob mode return log(mean softmax(logits)); logit mode averages
logits. A single member is passed through. This implements a control, not a
claim that the assistant is a better teacher. See docs/02_distillation.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.student import build_student

_EPS = 1e-8


class EnsembleTeacher(nn.Module):
    """Frozen ensemble of segmentation models acting as a single teacher."""

    def __init__(self, models: list[nn.Module], ensemble_mode: str = "prob") -> None:
        super().__init__()
        if ensemble_mode not in ("prob", "logit"):
            raise ValueError(f"ensemble_mode must be 'prob' or 'logit', got {ensemble_mode!r}")
        self.models = nn.ModuleList(models)
        self.ensemble_mode = ensemble_mode
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # noqa: D102 - teacher stays in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.models) == 1:
            return self.models[0](x)
        if self.ensemble_mode == "logit":
            out = None
            for m in self.models:
                logits = m(x)
                out = logits if out is None else out + logits
            return out / len(self.models)
        # "prob": mean softmax, returned in log-space so it quacks like logits.
        prob_sum = None
        for m in self.models:
            p = F.softmax(m(x), dim=1)
            prob_sum = p if prob_sum is None else prob_sum + p
        return torch.log(prob_sum / len(self.models) + _EPS)


def build_bridge_teacher(
    arch: str,
    ckpts: list[str],
    num_classes: int = 2,
    in_channels: int = 6,
    ensemble_mode: str = "prob",
) -> EnsembleTeacher:
    """Build a frozen (ensemble) teacher from student-architecture checkpoints.

        arch is a models/student.py name, e.g. "unet_resnet50". ckpts are one or more best.pt
        paths, several of them make a seed ensemble. ensemble_mode is "prob" (mean softmax,
        default) or "logit" (mean logits).
    """
    if not ckpts:
        raise ValueError("build_bridge_teacher needs at least one checkpoint path")
    members = [
        build_student(
            arch        = arch,
            in_channels = in_channels,
            num_classes = num_classes,
            pretrained  = False,
            ckpt_path   = str(c),
        )
        for c in ckpts
    ]
    teacher = EnsembleTeacher(members, ensemble_mode=ensemble_mode)
    n_par = sum(p.numel() for p in teacher.parameters())
    print(f"[bridge] EnsembleTeacher: {len(members)}× {arch}  "
          f"mode={ensemble_mode}  params={n_par:,} (frozen)")
    return teacher
