"""One removable, fixed feature term around the existing two-class KD loss."""
from __future__ import annotations

import torch
from training.losses import DistillLoss
from distillation.compression_metric import probe_features, feature_loss, gradient_rms


def last_shared_block(student):
    if hasattr(student, 'model'):
        return student.model.backbone.stages[-1]
    if hasattr(student.encoder, 'extractor'):
        return student.encoder.extractor.get_submodule('features.12')
    return student.encoder


class FeatureCapture:
    def __init__(self, student, teacher):
        self.student = self.teacher = None
        if hasattr(student, 'model'):
            tap = student.model.backbone.stages[-1]
        else:
            tap = student.encoder
        self.handles = [tap.register_forward_hook(self._student),
                        teacher.neck.register_forward_hook(self._teacher)]

    def _student(self, module, inputs, output):
        self.student = output[-1] if isinstance(output, (list, tuple)) else output

    def _teacher(self, module, inputs, output):
        self.teacher = output[-1]

    def clear(self):
        self.student = self.teacher = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.clear()


class CompressionLoss(DistillLoss):
    def __init__(self, teacher, student, transform=None, adapter=None, metric=None,
                 loss_scale=1., beta=0., diagnostic_interval=50):
        super().__init__(teacher, student, alpha=.5, temperature=4., gamma=2.,
                         kd_valid_only=True)
        teacher.eval().requires_grad_(False)
        if diagnostic_interval <= 0:
            raise ValueError('diagnostic interval must be positive')
        self.transform, self.adapter = transform, adapter
        self.beta, self.loss_scale = float(beta), float(loss_scale)
        if metric is not None:
            self.register_buffer('metric', metric.detach().float())
        self.capture = FeatureCapture(student, teacher) if adapter is not None else None
        self.diagnostic_interval, self.calls = diagnostic_interval, 0
        self.diagnostics = []

    def forward(self, student_logits, images, masks, precomputed_teacher_logits=None):
        try:
            if precomputed_teacher_logits is not None:
                raise ValueError('compression pilot requires live teacher targets on the same input')
            teacher = object.__getattribute__(self, '_teacher')
            if teacher.training or any(p.requires_grad for p in teacher.parameters()):
                raise RuntimeError('compression teacher must remain frozen and in eval mode')
            return self._forward(student_logits, images, masks)
        finally:
            if self.capture is not None:
                self.capture.clear()

    def _forward(self, student_logits, images, masks, precomputed_teacher_logits=None):
        # DistillLoss queries the frozen teacher on these exact augmented inputs.
        if student_logits.ndim != 4 or student_logits.shape[1] != 2 or not torch.isfinite(student_logits).all():
            raise RuntimeError('compression loss requires finite two-class spatial logits')
        if self.capture is not None and precomputed_teacher_logits is not None:
            raise ValueError('feature matching requires the live teacher on the same input')
        base, components = super().forward(student_logits, images, masks,
                    precomputed_teacher_logits=precomputed_teacher_logits)
        feature = base.new_zeros(())
        if self.capture is not None:
            if self.capture.student is None or self.capture.teacher is None:
                raise RuntimeError('declared student/teacher feature hooks did not fire')
            sv, weights = probe_features(self.capture.student, masks)
            tv, teacher_weights = probe_features(self.capture.teacher.detach(), masks)
            if not torch.equal(weights, teacher_weights):
                raise RuntimeError('teacher/student feature support differs')
            feature = feature_loss(self.transform(tv), self.adapter(sv), weights,
                                   self.metric, self.loss_scale)
            if torch.is_grad_enabled() and self.calls % self.diagnostic_interval == 0:
                student = object.__getattribute__(self, '_student')
                block = last_shared_block(student)
                parameters = tuple(block.parameters())
                feature_norm = gradient_rms(feature, parameters)
                self.diagnostics.append({'step': self.calls, 'task_loss': float(base.detach()),
                    'feature_loss': float(feature.detach()),
                    'task_gradient_rms': gradient_rms(base, parameters),
                    'feature_gradient_rms': feature_norm,
                    'weighted_feature_gradient_rms': self.beta * feature_norm})
            self.capture.clear()
        self.calls += 1
        components.update(feature=float(feature.detach()), feature_beta=self.beta)
        total = base + self.beta * feature
        if not torch.isfinite(total):
            raise RuntimeError('nonfinite compression objective')
        return total, components

    def remove_hooks(self):
        if self.capture:
            self.capture.close()
