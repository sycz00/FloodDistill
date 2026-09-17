"""Segmentation and distillation losses.

The main loss combines the task term and T^2*KL(p_teacher || p_student).
For dense predictions, KD reshapes (B,C,H,W) to (B*H*W,C) before batchmean.
Optional attention/feature losses remain for controls; published flags are
in experiments/. Learned feature projectors belong to the loss module and
must be included in the optimizer. Keep teacher outputs detached.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



# Multi-view teacher helpers (Temporal Privileged Distillation)


def _rand_aug_teacher(x: Tensor) -> Tensor:
    """Radiometric-only augmentation for the teacher views.

        No spatial augmentations (flip, rotate) here. Teacher logits/features can only be averaged
        over views with the same spatial layout. If view 1 is
        flipped, the flood pixels sit at different (x,y) positions, the averaged logit is
        ~0.5 everywhere and the student target collapses.

        So only spectral/radiometric perturbations are averaged: a per-channel multiplicative
        scale U(0.85, 1.15), standing in for sensor inter-calibration, atmospheric variability
        and sun angle across acquisition dates. The student should learn features that cope
        with these spectral shifts while the spatial semantics stay the same.
    """
    scale = x.new_empty(x.shape[0], x.shape[1], 1, 1).uniform_(0.85, 1.15)
    return (x * scale).clamp(0.0, 1.0)


def _multiview_teacher_forward(
    teacher: nn.Module,
    images: Tensor,
    n_views: int,
    feats_buf: list,
) -> Tensor:
    """Run the teacher n_views times (view 0 = original) and average logits + features.

        Afterwards feats_buf holds the per-level averaged features, exactly as after a single
        teacher forward, so the rest of the loss doesn't change. feats_buf is the list the
        teacher.neck forward hook fills (self._teacher_feats in DistillLoss / SSFDLoss).
        n_views=1 means only the original, no averaging. Returns (B, num_classes, H, W) logits
        averaged over all views.
    """
    all_logits: list[Tensor] = []
    all_feats:  list[list[Tensor]] = []

    for i in range(n_views):
        feats_buf.clear()
        aug = images if i == 0 else _rand_aug_teacher(images)
        logits = teacher(aug)
        all_logits.append(logits)
        all_feats.append(list(feats_buf))   # snapshot after hook fires

    avg_logits = torch.stack(all_logits).mean(0)

    # Replace feats_buf content with per-level averages
    feats_buf.clear()
    if all_feats and all_feats[0]:
        feats_buf.extend(
            torch.stack([view[lvl] for view in all_feats]).mean(0)
            for lvl in range(len(all_feats[0]))
        )

    return avg_logits



# Standalone loss functions


def ce_loss(logits: Tensor, target: Tensor, ignore_index: int = -1) -> Tensor:
    """Cross-entropy on (B,C,H,W) logits and (B,H,W) labels, excluding ignore_index."""
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def focal_loss(
    logits: Tensor,
    target: Tensor,
    gamma: float = 2.0,
    ignore_index: int = -1,
) -> Tensor:
    """Mean -(1-p_t)^gamma*log(p_t) over labels other than ignore_index (Lin et al., 2017)."""
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    valid   = target != ignore_index
    p_t     = torch.exp(-ce)                    # probability of the correct class
    loss    = (1.0 - p_t) ** gamma * ce
    return loss[valid].mean()                   # mean over valid pixels only


def dkd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    ignore_index: int = -1,
) -> Tensor:
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022).

        Splits KL(student || teacher) into two separate terms:

          TCKD = KL( [p_s[y], 1-p_s[y]] || [p_t[y], 1-p_t[y]] )
                 binary KL over (target class, rest), i.e. whether the student knows which
                 class to predict.

          NCKD = (1-p_t[y]) * KL( p_t[not y]/(1-p_t[y]) || p_s[not y]/(1-p_s[y]) )
                 weighted KL over the renormalised non-target probabilities, the "dark
                 knowledge" (confusion among the wrong classes). Computed as
                 sum_{j != y} p_t[j] * log((p_t[j]/(1-p_t_y)) / (p_s[j]/(1-p_s_y))), which is
                 the same expression without renormalising explicitly.

        L_DKD = T^2 * (alpha * TCKD + beta * NCKD)

        alpha = beta = 1 gives exactly standard KD (TCKD + NCKD = KL per pixel). beta > alpha
        weights dark knowledge more than target-class alignment.

        For C=2 segmentation NCKD is identically 0 (only one non-target class, so the
        renormalised distribution is a point mass and KL = 0), and DKD is just alpha * standard
        KD. It only makes a difference for C >= 3.

        targets are (B, H, W) integer labels (pseudo or ground truth), temperature T >= 1,
        alpha weights TCKD and beta NCKD, pixels labelled ignore_index are excluded.
    """
    B, C, H, W = student_logits.shape

    # flatten to (N, C) and keep the valid pixels
    s = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
    t = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)
    y = targets.reshape(-1).long()

    valid = y != ignore_index
    s, t, y = s[valid], t[valid], y[valid]
    M = s.shape[0]
    if M == 0:
        return student_logits.new_zeros(())

    p_s = F.softmax(s / temperature, dim=1)   # (M, C)
    p_t = F.softmax(t / temperature, dim=1)   # (M, C)
    eps = 1e-8

    # TCKD: binary KL over (target class, rest)
    p_s_y = p_s.gather(1, y.unsqueeze(1)).squeeze(1)   # (M,)
    p_t_y = p_t.gather(1, y.unsqueeze(1)).squeeze(1)   # (M,)

    tckd = (
          p_t_y        * (p_t_y.clamp(eps).log()        - p_s_y.clamp(eps).log())
        + (1 - p_t_y)  * ((1 - p_t_y).clamp(eps).log() - (1 - p_s_y).clamp(eps).log())
    ).mean()

    # NCKD: (1-p_t[y]) * KL over the renormalised non-target probs, computed element-wise as
    # sum_{j != y} p_t[j] * log(p_t[j]/(1-p_t_y) / (p_s[j]/(1-p_s_y))).
    # The target position is zeroed by the nontarget mask, log_ratio there is finite (no NaN).
    # For C=2 the only non-target class j=1-y has p_t[j]=1-p_t_y, so
    # log_ratio = log(1 / 1) = 0 and NCKD = 0 exactly.
    if C > 2:
        nontarget = torch.ones(M, C, dtype=torch.bool, device=s.device) \
                        .scatter_(1, y.unsqueeze(1), False)              # (M, C)

        log_ratio = (
            (p_t / (1 - p_t_y).clamp(eps).unsqueeze(1)).clamp(eps).log()
            - (p_s / (1 - p_s_y).clamp(eps).unsqueeze(1)).clamp(eps).log()
        )                                                                 # (M, C)

        nckd = (p_t * log_ratio * nontarget.float()).sum(dim=1).mean()
    else:
        nckd = student_logits.new_zeros(())

    return (temperature ** 2) * (alpha * tckd + beta * nckd)


def kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    """T^2*KL(p_teacher || p_student), using F.kl_div(log p_student, p_teacher).

    Dense (B,C,H,W) logits are flattened to (B*H*W,C), so batchmean divides
    by pixel count. Also accepts (B,C). This helper has no ignore mask.
    """
    if student_logits.dim() == 4:
        B, C, H, W = student_logits.shape
        s = F.log_softmax(student_logits / temperature, dim=1) \
              .permute(0, 2, 3, 1).reshape(B * H * W, C)
        t = F.softmax(teacher_logits / temperature, dim=1) \
              .permute(0, 2, 3, 1).reshape(B * H * W, C)
    else:
        s = F.log_softmax(student_logits / temperature, dim=1)
        t = F.softmax(teacher_logits   / temperature, dim=1)

    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def _activation_attention(feat: Tensor, target_h: int, target_w: int) -> Tensor:
    """Sum-squared activation map, resized and L2-normalised per sample.

        A(F) = sum_c F_c^2 gives (B, H, W), which is resized, flattened and L2-normalised,
        returning (B, target_h * target_w). Scale-invariant, so it compares across channel
        counts and spatial sizes. Teacher features should be detached before calling this.
    """
    attn = feat.pow(2).sum(dim=1)               # (B, H, W)
    if attn.shape[-2:] != (target_h, target_w):
        attn = F.interpolate(
            attn.unsqueeze(1), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
    flat = attn.reshape(attn.shape[0], -1)      # (B, H_t * W_t)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return flat / norm


def at_loss(teacher_feats: list[Tensor], student_feats: list[Tensor]) -> Tensor:
    """Attention Transfer loss (Zagoruyko & Komodakis, ICLR 2017).

        L_AT = (1/L) * sum_l MSE( A_norm_s^l, A_norm_t^l )

        Levels are paired from the deepest down (teacher[-1] with student[-1], and so on), and
        all maps are resized to the teacher's resolution first. Takes lists of
        (B, C_t, H_t, W_t) teacher and (B, C_s, H_s, W_s) student features and returns the
        scalar loss averaged over the paired levels.
    """
    target_h, target_w = teacher_feats[-1].shape[-2:]
    n_pairs = min(len(teacher_feats), len(student_feats))
    total   = teacher_feats[-1].new_zeros(())

    for t_f, s_f in zip(teacher_feats[-n_pairs:], student_feats[-n_pairs:]):
        at_t = _activation_attention(t_f.detach(), target_h, target_w)
        at_s = _activation_attention(s_f,          target_h, target_w)
        total = total + F.mse_loss(at_s, at_t)

    return total / max(n_pairs, 1)



# DistillLoss - self-contained distillation loss module


def feat_distil_loss(
    teacher_feats: list[Tensor],
    student_feats: list[Tensor],
    projectors: "nn.ModuleList",
) -> Tensor:
    """MSE between projected student and detached teacher global-average features.

    Spatial inputs are (B,C,H,W); student token inputs (B,N,C) average over N.
    Each learned Linear maps C_student to C_teacher. Average losses over levels.
    """
    n_pairs = min(len(teacher_feats), len(student_feats), len(projectors))
    total   = teacher_feats[-1].new_zeros(())

    for t_f, s_f, proj in zip(
        teacher_feats[-n_pairs:], student_feats[-n_pairs:], projectors[-n_pairs:]
    ):
        # teacher: global average pool, (B, C_t, H, W) to (B, C_t)
        if t_f.dim() == 4:
            t_gap = t_f.detach().mean(dim=[-2, -1])       # (B, C_t)
        else:
            t_gap = t_f.detach().mean(dim=1)               # (B, N, C) to (B, C)

        # student: (B, C_s, H, W) or (B, N, C_s) to (B, C_s)
        if s_f.dim() == 4:
            s_gap = s_f.mean(dim=[-2, -1])                 # (B, C_s)
        else:
            s_gap = s_f.mean(dim=1)                        # (B, N, C) to (B, C)

        s_proj = proj(s_gap)                               # (B, C_t)
        total  = total + F.mse_loss(s_proj, t_gap)

    return total / max(n_pairs, 1)


class DistillLoss(nn.Module):
    """Combine task loss, teacher-to-student KL and optional attention/feature losses.

    The frozen teacher is kept outside module parameter tracking. Register hooks
    before training and remove them afterwards. Feature projectors are learned
    loss-module parameters and must be present in the optimizer.
    """

    # EfficientViT backbone stage keys, shallow to deep
    _EVT_STAGES = ("stage1", "stage2", "stage3", "stage4")

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        alpha: float = 0.5,
        temperature: float = 4.0,
        at_beta: float = 0.0,
        feat_distil_alpha: float = 0.0,
        ignore_index: int = -1,
        gamma: float = 2.0,
        n_teacher_views: int = 1,
        use_dkd: bool = False,
        dkd_alpha: float = 1.0,
        dkd_beta: float = 1.0,
    ) -> None:
        super().__init__()
        # object.__setattr__ bypasses nn.Module.__setattr__, so teacher and student don't end
        # up in self._modules. The optimizer and DataParallel then don't pick up the teacher
        # parameters and .parameters() doesn't recurse into them.
        object.__setattr__(self, "_teacher", teacher)
        object.__setattr__(self, "_student", student)

        self.alpha             = alpha
        self.temperature       = temperature
        self.at_beta           = at_beta
        self.feat_distil_alpha = feat_distil_alpha
        self.ignore_index      = ignore_index
        self.gamma             = gamma
        self.n_teacher_views   = max(1, n_teacher_views)
        self.use_dkd           = use_dkd
        self.dkd_alpha         = dkd_alpha
        self.dkd_beta          = dkd_beta
        self._at_enabled       = at_beta > 0.0
        self._feat_enabled     = feat_distil_alpha > 0.0

        # Feature buffers filled by forward hooks
        self._teacher_feats: list[Tensor] = []
        self._student_feats: list[Tensor] = []

        # Learnable feat_distil projectors, one nn.Linear(C_s, C_t) per feature level, built
        # lazily on the first forward. They are a real nn.ModuleList so they show up in
        # self.parameters(); build the optimizer from loss_fn.parameters() or
        # loss_fn.feat_projectors.parameters() so they're included.
        self.feat_projectors: nn.ModuleList = nn.ModuleList()
        self._feat_projectors_built: bool = False

        # Hook handles (stored for clean removal)
        self._hooks: list = []

    # Hook management

    def register_hooks(self) -> None:
        """Register feature capture hooks on teacher.neck and the student backbone.

                Used by AT (at_beta > 0) and feature distillation (feat_distil_alpha > 0). The
                student attachment point is found from named attributes rather than isinstance
                checks, so a new architecture works if it follows the same naming:

                    EfficientViT _FullResWrapper     .model.backbone  (dict of stage tensors)
                    MobileNetV3  LightweightSegModel .encoder         (list of tensors)
                    ViT-Tiny     ViTTinySegModel     .backbone        (timm ViT, tokens)
                    Prithvi student TeacherModel     .neck            (list of tensors)

                Hooks are disabled with a warning if at_beta == 0 and feat_distil_alpha == 0,
                with DataParallel (hooks don't fire on replicas), if the teacher has no .neck,
                or if the student architecture isn't recognised.
        """
        if not self._at_enabled and not self._feat_enabled:
            return

        teacher = object.__getattribute__(self, "_teacher")
        student = object.__getattribute__(self, "_student")

        # DataParallel check - hooks only fire on the original module, not replicas
        if isinstance(teacher, nn.DataParallel) or isinstance(student, nn.DataParallel):
            print("[DistillLoss] AT/feat_distil disabled: DataParallel is not compatible "
                  "with hooks. Use a single GPU (gpu_ids=[0]) to enable these losses.")
            self._at_enabled   = False
            self._feat_enabled = False
            return

        if not hasattr(teacher, "neck"):
            print("[DistillLoss] AT/feat_distil disabled: teacher has no .neck attribute.")
            self._at_enabled   = False
            self._feat_enabled = False
            return

        # Teacher hook: .neck returns list[Tensor(B, D, H_t, W_t)]
        def _teacher_hook(module, inp, out):
            self._teacher_feats.clear()
            self._teacher_feats.extend(out)

        self._hooks.append(teacher.neck.register_forward_hook(_teacher_hook))

        # Student hook: resolve by attribute inspection
        # Each hook populates self._student_feats as a list of spatial tensors
        # (B, C, H, W) or token sequences (B, N, C) - both are accepted by
        # feat_distil_loss and at_loss (AT reshapes/resizes internally).
        if hasattr(student, "model") and hasattr(student.model, "backbone"):
            # EfficientViT: _FullResWrapper, .model.backbone returns a dict
            def _student_hook_evt(module, inp, out):
                self._student_feats.clear()
                if isinstance(out, dict):
                    self._student_feats.extend(
                        out[s] for s in self._EVT_STAGES if s in out
                    )
                else:
                    feats = out if isinstance(out, (list, tuple)) else [out]
                    self._student_feats.extend(feats)

            self._hooks.append(
                student.model.backbone.register_forward_hook(_student_hook_evt)
            )
            print(f"[DistillLoss] hooks registered: teacher=.neck  student=.model.backbone  "
                  f"levels=4  at_beta={self.at_beta}  feat_distil_alpha={self.feat_distil_alpha}")

        elif hasattr(student, "backbone") and not hasattr(student, "model"):
            # ViT-Tiny: ViTTinySegModel.backbone is the timm ViT model. forward() calls
            # backbone.forward_features() directly, which bypasses backbone.__call__(), so a
            # hook on the backbone itself would never fire. The hook goes on backbone.norm (the
            # last LayerNorm inside forward_features) instead, which does go through __call__
            # and returns (B, N+1, D) tokens (CLS + patches). feat_distil_loss mean-pools both
            # spatial (B, C, H, W) and token (B, N, C) inputs, so no reshape here.
            hook_module = getattr(student.backbone, "norm", student.backbone)
            hook_label  = ".backbone.norm" if hasattr(student.backbone, "norm") else ".backbone"

            def _student_hook_vit(module, inp, out):
                self._student_feats.clear()
                feats = out if isinstance(out, (list, tuple)) else [out]
                self._student_feats.extend(feats)

            self._hooks.append(
                hook_module.register_forward_hook(_student_hook_vit)
            )
            print(f"[DistillLoss] hooks registered: teacher=.neck  student={hook_label} (ViT-Tiny)  "
                  f"levels=1  at_beta={self.at_beta}  feat_distil_alpha={self.feat_distil_alpha}")

        elif hasattr(student, "encoder"):
            # MobileNetV3: LightweightSegModel, .encoder returns list[Tensor]
            def _student_hook_mnv3(module, inp, out):
                self._student_feats.clear()
                feats = out if isinstance(out, (list, tuple)) else [out]
                self._student_feats.extend(feats)

            self._hooks.append(
                student.encoder.register_forward_hook(_student_hook_mnv3)
            )
            print(f"[DistillLoss] hooks registered: teacher=.neck  student=.encoder  "
                  f"levels=4  at_beta={self.at_beta}  feat_distil_alpha={self.feat_distil_alpha}")

        elif hasattr(student, "neck"):
            # Prithvi student: TeacherModel, .neck returns list[Tensor]
            def _student_hook_prithvi(module, inp, out):
                self._student_feats.clear()
                feats = out if isinstance(out, (list, tuple)) else [out]
                self._student_feats.extend(feats)

            self._hooks.append(
                student.neck.register_forward_hook(_student_hook_prithvi)
            )
            print(f"[DistillLoss] hooks registered: teacher=.neck  student=.neck  "
                  f"levels=4  at_beta={self.at_beta}  feat_distil_alpha={self.feat_distil_alpha}")

        else:
            print("[DistillLoss] AT/feat_distil disabled: student architecture not recognised "
                  "(no .model.backbone, .backbone, .encoder, or .neck attribute).")
            self._at_enabled   = False
            self._feat_enabled = False

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # Forward

    def forward(
        self,
        student_logits: Tensor,
        images: Tensor,
        masks: Tensor,
        precomputed_teacher_logits: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Combined distillation loss.

                Args:
                    student_logits:             (B, C, H, W), already computed by the Trainer
                    images:                     (B, C, H, W), needed for the teacher forward pass
                    masks:                      (B, H, W) ground-truth integer labels
                    precomputed_teacher_logits: optional (B, C, H, W) teacher logits stored during
                                                pseudo-label generation (Fix 4). If given, the
                                                teacher forward is skipped completely, which is
                                                faster and consistent with the argmax
                                                pseudo-labels stored at generation time. Only
                                                valid for all-pseudo batches (--no_labeled).

                Returns (total_loss, components) with keys "focal", "kd", "at".
        """
        teacher = object.__getattribute__(self, "_teacher")

        # Fix 4: with precomputed_teacher_logits (stored at generation time) the live teacher
        # forward for the KD term is skipped. Saves recomputation in --no_labeled mode, where
        # the teacher would process images it already saw during generate. The teacher feature
        # hooks don't fire in this path, so AT / feat_distil (which need live teacher
        # inference) are skipped below.
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            if precomputed_teacher_logits is not None:
                teacher_logits = precomputed_teacher_logits.to(images.device).float()
                # no teacher forward, so the hook buffers stay empty and AT/feat_distil are skipped
            elif self.n_teacher_views > 1:
                teacher_logits = _multiview_teacher_forward(
                    teacher, images.float(), self.n_teacher_views, self._teacher_feats
                )
            else:
                teacher_logits = teacher(images.float())

        f = focal_loss(student_logits, masks, self.gamma, self.ignore_index)

        t_logits = teacher_logits.to(dtype=student_logits.dtype)
        if self.use_dkd:
            kd = dkd_loss(
                student_logits, t_logits, masks,
                temperature  = self.temperature,
                alpha        = self.dkd_alpha,
                beta         = self.dkd_beta,
                ignore_index = self.ignore_index,
            )
        else:
            kd = kd_loss(student_logits, t_logits, self.temperature)

        total = (1.0 - self.alpha) * f + self.alpha * kd

        at_val   = student_logits.new_zeros(())
        feat_val = student_logits.new_zeros(())

        if (self._at_enabled or self._feat_enabled) \
                and self._teacher_feats and self._student_feats:

            if self._at_enabled:
                # AT: compares spatial attention maps (sum-squared, L2-normalised)
                at_val = at_loss(self._teacher_feats, self._student_feats)
                total  = total + self.at_beta * at_val

            if self._feat_enabled:
                # Feature distillation: projectors are built lazily on the first call, once the
                # hook buffers give the feature dims. One nn.Linear(C_s, C_t) per paired level.
                if not self._feat_projectors_built:
                    self._build_feat_projectors(self._teacher_feats, self._student_feats)

                feat_val = feat_distil_loss(
                    self._teacher_feats, self._student_feats, self.feat_projectors
                )
                total = total + self.feat_distil_alpha * feat_val

            # clear used buffers; stale features would silently give wrong gradients
            # on the next batch.
            self._teacher_feats.clear()
            self._student_feats.clear()

        components = {
            "focal":      f.item(),
            "kd":         kd.item(),
            "at":         at_val.item(),
            "feat_distil": feat_val.item(),
        }
        return total, components

    # Projector construction

    def _build_feat_projectors(
        self,
        teacher_feats: list[Tensor],
        student_feats: list[Tensor],
    ) -> None:
        """One nn.Linear(C_s, C_t) per feature level, built lazily on the first forward.

                Only called once, after that the projector shapes are fixed for the rest of
                training. They go into self.feat_projectors (an nn.ModuleList, part of
                DistillLoss.parameters()), so an optimizer built from loss_fn.parameters() or
                loss_fn.feat_projectors.parameters() includes them.
        """
        n_pairs = min(len(teacher_feats), len(student_feats))
        projs   = []
        for t_f, s_f in zip(teacher_feats[-n_pairs:], student_feats[-n_pairs:]):
            # Infer C_t (teacher) and C_s (student) from feature tensors.
            # Spatial (B, C, H, W): channel is dim 1.
            # Token (B, N, C): channel is dim 2.
            c_t = t_f.shape[1] if t_f.dim() == 4 else t_f.shape[2]
            c_s = s_f.shape[1] if s_f.dim() == 4 else s_f.shape[2]
            proj = nn.Linear(c_s, c_t, bias=False).to(t_f.device)
            # Initialise as random orthogonal - better than default kaiming for
            # projectors that start from an aligned space (CKA 0.86 after BD).
            nn.init.orthogonal_(proj.weight)
            projs.append(proj)

        # Replace the empty ModuleList with the built projectors
        self.feat_projectors = nn.ModuleList(projs)
        self._feat_projectors_built = True
        dims_s = [
            (f.shape[1] if f.dim() == 4 else f.shape[2]) for f in student_feats[-n_pairs:]
        ]
        dims_t = [
            (f.shape[1] if f.dim() == 4 else f.shape[2]) for f in teacher_feats[-n_pairs:]
        ]
        print(f"[DistillLoss] feat_distil projectors built: "
              f"student_dims={dims_s} → teacher_dims={dims_t}  "
              f"feat_distil_alpha={self.feat_distil_alpha}")
