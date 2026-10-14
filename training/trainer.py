"""Training loop for task supervision and distillation.

loss_fn=None uses the configured task loss; DistillLoss and OFALoss provide
teacher supervision. The loop manages AMP, scheduling, validation and saved
state. QAT has its own loop in quantization/qat.py.
Checkpoints contain epoch, metrics, model and optimizer state; student names
and state-dict keys are kept stable for evaluation and deployment.
"""

from __future__ import annotations

import inspect
import hashlib
import os
import random
import tempfile
import time
from collections import deque
from dataclasses import asdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from config import TrainConfig
from training.losses import ce_loss, focal_loss, DistillLoss
from utils.metrics import accumulate_metrics, finalize_metrics
from utils.seeding import seed_basic, seed_full


def _discover_term_names(loss_fn) -> list[str]:
    """Infer which loss term names loss_fn.compute_loss_terms() may return."""
    # Loss functions can declare their own term list explicitly.
    if hasattr(loss_fn, "_grad_conflict_terms"):
        return list(loss_fn._grad_conflict_terms)
    names = ["focal", "kd"]
    if getattr(loss_fn, "_feat_enabled", False):
        names.append("feat_mse")
        if getattr(loss_fn, "fird_alpha", 0.0) > 0.0:
            names.append("fird")
    if getattr(loss_fn, "_at_enabled", False):
        names.append("at")
    return names


class Trainer:
    """Training loop with task loss alone or an attached distillation loss. cfg controls the run settings."""

    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        cfg: TrainConfig,
        loss_fn: DistillLoss | None = None,
        num_classes: int = 2,
        class_names: list[str] | None = None,
        pseudo_dl_ssfd: DataLoader | None = None,
        pseudo_ssfd_weight: float = 1.0,
        grad_conflict: bool = False,
        grad_conflict_interval: int = 50,
        pcgrad: bool = False,
        ofa_backbone_pcgrad: bool = False,
        ofa_agr: bool = False,
        ofa_fagr: bool = False,
        ofa_diagnose: bool = False,
        ofa_diagnose_interval: int = 50,
        dynamic_importance: bool = False,
        importance_reestimate_interval: int = 10,
        analysis_loader: DataLoader | None = None,
        concept_loss_fn=None,
        concept_distil_alpha: float = 0.0,
        sam: bool = False,
        sam_rho: float = 0.05,
    ) -> None:
        self.model              = model
        self.train_dl           = train_dl
        self.val_dl             = val_dl
        self.test_dl            = test_dl
        self.cfg                = cfg
        self.loss_fn            = loss_fn
        self.num_classes        = num_classes
        self.class_names        = class_names   # e.g. ["background", "flood"]
        self.pseudo_dl_ssfd     = pseudo_dl_ssfd
        self.pseudo_ssfd_weight = pseudo_ssfd_weight
        self.grad_conflict                  = grad_conflict
        self.grad_conflict_interval         = grad_conflict_interval
        self.pcgrad                         = pcgrad
        self.ofa_backbone_pcgrad            = ofa_backbone_pcgrad
        self.ofa_agr                        = ofa_agr
        self.ofa_fagr                       = ofa_fagr
        self.ofa_diagnose                   = ofa_diagnose
        self.ofa_diagnose_interval          = ofa_diagnose_interval
        self.dynamic_importance             = dynamic_importance
        self.importance_reestimate_interval = importance_reestimate_interval
        self.analysis_loader                = analysis_loader
        self._global_step                   = 0
        self.concept_loss_fn                = concept_loss_fn
        self.concept_distil_alpha           = concept_distil_alpha
        self.sam                            = sam
        self.sam_rho                        = sam_rho
        # Capture actual main tensors before optimization. Historical runs without
        # this field retain unknown initialization identity.
        initial = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            initial.update(name.encode())
            initial.update(str(tensor.dtype).encode())
            initial.update(str(tuple(tensor.shape)).encode())
            initial.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.initial_model_sha256 = initial.hexdigest()
        self.loss_policy = {
            "main_hard": "focal" if loss_fn is not None else cfg.task_loss,
            "main_focal_gamma": (2.0 if hasattr(loss_fn, "exit_branches") else
                                 getattr(loss_fn, "gamma", cfg.gamma)),
            "main_kl_mask": ("valid_only" if getattr(loss_fn, "kd_valid_only", False)
                             else "legacy_all_pixels") if loss_fn is not None else None,
            "hard_and_ofa_branch_mask": "target != ignore_index",
            "all_ignored_hard_or_valid_kl": (
                "graph-connected zero loss; optimizer step retained" if loss_fn is not None or cfg.task_loss == "focal"
                else "legacy CE all-ignored reduction remains undefined"),
            "grad_clip_norm": cfg.grad_clip,
            "grad_clip_scope": "main model parameters; inherited norm2 after AMP unscale",
        }

        if cfg.seed is not None:
            if getattr(cfg, "deterministic", False):
                seed_full(cfg.seed)
            else:
                seed_basic(cfg.seed)

        os.makedirs(cfg.out_dir, exist_ok=True)

        # Device
        self.device = torch.device(
            f"cuda:{cfg.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
        )

        # AMP - only when CUDA is available
        _cuda_available = self.device.type == "cuda"
        self.use_amp    = cfg.use_amp and _cuda_available
        self.amp_dtype  = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
        self.scaler     = GradScaler(enabled=self.use_amp)

        # W&B (optional)
        self.wandb_run = self._init_wandb()

        # Gradient conflict logger (optional)
        self.grad_logger = None
        if grad_conflict and loss_fn is not None and hasattr(loss_fn, "compute_loss_terms"):
            from utils.grad_conflict import GradientConflictLogger
            term_names = _discover_term_names(loss_fn)
            self.grad_logger = GradientConflictLogger(term_names)
            print(f"[Trainer] grad_conflict enabled: terms={term_names}  "
                  f"interval={grad_conflict_interval} steps")

        if pcgrad and loss_fn is not None:
            if not hasattr(loss_fn, "compute_loss_terms"):
                print("[Trainer] pcgrad=True but loss_fn has no compute_loss_terms(); "
                      "PCGrad disabled.")
                self.pcgrad = False
            elif self.use_amp:
                print("[Trainer] pcgrad=True: AMP bypassed (PCGrad runs FP32 per-term "
                      "autograd; GradScaler inactive for pcgrad steps).")

        if ofa_backbone_pcgrad and loss_fn is not None:
            if not hasattr(loss_fn, "branch_conflict_mode"):
                print("[Trainer] ofa_backbone_pcgrad=True but loss_fn is not OFALoss; disabled.")
                self.ofa_backbone_pcgrad = False
            elif self.use_amp:
                print("[Trainer] ofa_backbone_pcgrad=True: AMP bypassed (runs FP32 per-stage "
                      "autograd; overhead ≈5 backward passes per tracking step).")
            else:
                print("[Trainer] ofa_backbone_pcgrad=True: focal-anchored backbone gradient "
                      "surgery active — early OFA branch backbone grads projected onto focal's "
                      "direction when conflicting; non-backbone params receive standard grads.")

        if ofa_agr and loss_fn is not None:
            if not hasattr(loss_fn, "get_agr_weight_and_update"):
                print("[Trainer] ofa_agr=True but loss_fn has no AGR support (not OFALoss); "
                      "disabled.")
                self.ofa_agr = False
            elif ofa_backbone_pcgrad:
                print("[Trainer] ofa_agr=True and ofa_backbone_pcgrad=True: AGR-OFA takes "
                      "priority; bb_pcgrad is ignored.")
                self.ofa_backbone_pcgrad = False
            else:
                print("[Trainer] ofa_agr=True: §96 AGR-OFA active (ABANDONED — use "
                      "--ofa_fagr for the corrected FA-AGR variant; §96 measures "
                      "temporal self-consistency, which does not detect inter-branch conflict).")

        if ofa_fagr and loss_fn is not None:
            if not hasattr(loss_fn, "branch_conflict_mode"):
                print("[Trainer] ofa_fagr=True but loss_fn is not OFALoss; disabled.")
                self.ofa_fagr = False
            elif ofa_agr:
                print("[Trainer] ofa_fagr=True and ofa_agr=True: FA-AGR (§97) takes "
                      "priority; ofa_agr (§96) is disabled.")
                self.ofa_agr = False
            elif ofa_backbone_pcgrad:
                print("[Trainer] ofa_fagr=True and ofa_backbone_pcgrad=True: FA-AGR (§97) "
                      "takes priority; bb_pcgrad is disabled.")
                self.ofa_backbone_pcgrad = False
            else:
                print("[Trainer] ofa_fagr=True: §97 Focal-Anchored AGR active — each OFA "
                      "branch backbone gradient is scaled by intra-batch cosine agreement "
                      "with normalize(g_focal_bb + g_kd_bb); focal/kd grads unmodified.")

        if ofa_diagnose and loss_fn is not None:
            has_breakdown = hasattr(loss_fn, "get_branch_loss_breakdown")
            has_gnorm     = hasattr(loss_fn, "branch_conflict_mode")
            if not has_breakdown:
                print("[Trainer] ofa_diagnose=True but loss_fn has no branch diagnostics "
                      "(not OFALoss); per-branch loss/entropy logging disabled.")
                self.ofa_diagnose = False
            else:
                extras = []
                if has_gnorm:
                    extras.append(f"backbone gnorm+conflict every {ofa_diagnose_interval} steps")
                print(f"[Trainer] ofa_diagnose=True: per-branch loss + entropy every step"
                      + (f"; {extras[0]}" if extras else "") + ".")

    # Public API

    def fit(self) -> dict:
        """Train and return best-validation and final test metrics."""
        cfg = self.cfg
        t0  = time.perf_counter()

        model = self.model.to(self.device)
        if len(cfg.gpu_ids) > 1 and self.device.type == "cuda":
            model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
            print(f"[Trainer] DataParallel on GPUs {cfg.gpu_ids}")

        # Teacher is stored outside nn.Module tracking, so .to(device) above
        # doesn't reach it - move it explicitly.
        if self.loss_fn is not None:
            teacher = object.__getattribute__(self.loss_fn, "_teacher")
            object.__setattr__(self.loss_fn, "_teacher", teacher.to(self.device))

        optimizer = self._build_optimizer(model, cfg)
        if self.sam:
            from utils.sam import SAM
            self._sam = SAM(optimizer, rho=self.sam_rho)
        scheduler = self._build_scheduler(optimizer)

        window             = max(1, getattr(cfg, "val_smoothing_window", 3))
        miou_history       = deque(maxlen=window)
        best_smoothed_miou = 0.0
        best_miou          = 0.0   # actual mIoU at the saved checkpoint epoch
        es_counter         = 0
        best_ckpt_path     = os.path.join(cfg.out_dir, "best.pt")

        is_distill = self.loss_fn is not None
        mode_tag   = "Distill" if is_distill else "FineTune"
        print(f"[Trainer] mode={mode_tag}  device={self.device}  "
              f"epochs={cfg.epochs}  amp={self.use_amp}({cfg.amp_dtype})")

        for epoch in range(cfg.epochs):
            self._update_feat_alpha(epoch)
            # Propagate current epoch to loss_fn for branch warmup curriculum (DBI-OFA).
            if self.loss_fn is not None and hasattr(self.loss_fn, "current_epoch"):
                self.loss_fn.current_epoch = epoch
            model.train()

            if is_distill:
                log_dict = self._train_epoch_distill(model, optimizer)
            else:
                log_dict = self._train_epoch_supervised(model, optimizer)

            self._log(epoch, log_dict, prefix="train")

            if scheduler is not None:
                scheduler.step()

            # Validation
            run_val = (epoch + 1) % cfg.val_every_n_epochs == 0 or epoch == cfg.epochs - 1
            epoch_metrics = {}
            improved = False
            should_stop = False
            if run_val:
                val_metrics = self.evaluate(model, self.val_dl)
                epoch_metrics = val_metrics
                self._log(epoch, val_metrics, prefix="val")

                miou_history.append(val_metrics["mIoU"])
                smoothed_miou = sum(miou_history) / len(miou_history)
                improved = smoothed_miou > best_smoothed_miou
                if improved:
                    best_smoothed_miou = smoothed_miou
                    best_miou = val_metrics["mIoU"]
                    iou_per_class = val_metrics.get("iou_per_class", [])
                    flood_iou = iou_per_class[1] if len(iou_per_class) > 1 else float("nan")
                    print(f"  [{mode_tag}] new best  smoothed_mIoU={best_smoothed_miou:.4f}"
                          f"  flood_IoU={flood_iou:.4f}"
                          f"  (window={len(miou_history)}/{window})")

                # Early stopping: aligned with checkpoint criterion (smoothed mIoU).
                # Monitoring val loss would diverge from mIoU on imbalanced data - the
                # background-dominated loss can plateau while flood IoU still improves.
                if cfg.early_stop_patience > 0:
                    if improved:
                        es_counter = 0
                    else:
                        es_counter += 1
                        if es_counter >= cfg.early_stop_patience:
                            print(f"  [{mode_tag}] early stop at epoch {epoch+1} "
                                  f"(smoothed_mIoU no improvement for "
                                  f"{cfg.early_stop_patience} checks)")
                            should_stop = True

            selection_state = {
                "window": window,
                "miou_history": list(miou_history),
                "best_smoothed_miou": best_smoothed_miou,
                "best_miou": best_miou,
                "es_counter": es_counter,
                "validated_this_epoch": run_val,
                "early_stop_reached": should_stop,
                "metric": "S11 validation smoothed mIoU",
                "comparison": "strict greater than; ties retain earlier checkpoint",
            }
            if improved:
                self.save_checkpoint(
                    model, optimizer, epoch, epoch_metrics, "best.pt",
                    scheduler=scheduler, selection_state=selection_state,
                )
            self.save_checkpoint(
                model, optimizer, epoch, epoch_metrics, "last.pt",
                scheduler=scheduler, selection_state=selection_state,
            )
            if should_stop:
                break

        # Evaluation on best checkpoint: test (always) + all splits (optional)
        print(f"\n[Trainer] loading best validation checkpoint …")
        self.load_checkpoint(model, best_ckpt_path)
        test_metrics = {}
        if not getattr(cfg, "defer_terminal_evaluation", False):
            test_metrics = self.evaluate(model, self.test_dl)
            self._log(cfg.epochs, test_metrics, prefix="test")

        extra_results: dict = {}
        if getattr(cfg, "eval_all_splits", False):
            print("[Trainer] eval_all_splits: evaluating on train and val splits …")
            train_eval = self.evaluate(model, self.train_dl)
            self._log(cfg.epochs, train_eval, prefix="s11_train")
            val_best = self.evaluate(model, self.val_dl)
            self._log(cfg.epochs, val_best, prefix="s11_val")
            extra_results = {
                **{f"s11_train_{k}": v for k, v in train_eval.items()},
                **{f"s11_val_{k}":   v for k, v in val_best.items()},
            }

        elapsed = self._elapsed(t0)
        print(f"\n[Trainer] done in {elapsed}  best_val_mIoU={best_miou:.4f}  "
              f"terminal_evaluation={'deferred' if not test_metrics else test_metrics['mIoU']}")

        if self.grad_logger is not None:
            self.grad_logger.summary()
            csv_path = self.grad_logger.save_csv(cfg.out_dir)
            print(f"[GradConflict] saved CSV → {csv_path}")
            png_path = self.grad_logger.plot(cfg.out_dir)
            if png_path:
                print(f"[GradConflict] saved plot → {png_path}")

        if self.wandb_run is not None:
            self.wandb_run.finish()

        results = {
            "best_val_mIoU":          best_miou,
            "best_val_smoothed_mIoU": best_smoothed_miou,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **extra_results,
        }

        # save next to best.pt; stdout alone would mean re-running eval
        # on every checkpoint to recover the test numbers.
        import json
        with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> dict:
        """Evaluate model on a DataLoader, accumulate confusion matrix.

                Returns:
                    dict with keys: mIoU, iou_per_class, pixel_acc, f1, f1_per_class, loss
        """
        model.eval()
        conf_matrix = None
        total_loss  = 0.0
        n_batches   = 0

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True).float()
            masks  = batch["mask"].to(self.device, non_blocking=True).long()

            logits = model(images)

            conf_matrix = accumulate_metrics(
                logits.cpu(), masks.cpu(),
                self.num_classes, self.cfg.ignore_index, conf_matrix,
            )
            batch_loss = F.cross_entropy(logits, masks, ignore_index=self.cfg.ignore_index).item()
            if not (batch_loss != batch_loss):   # skip NaN (all-ignore batch)
                total_loss += batch_loss
                n_batches  += 1

        metrics = finalize_metrics(conf_matrix) if conf_matrix is not None else {}
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    # Checkpoint helpers

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: dict,
        filename: str = "checkpoint.pt",
        *,
        scheduler=None,
        selection_state: dict | None = None,
    ) -> str:
        """Atomically preserve inference weights and completed-epoch training state.

        Existing load_checkpoint remains an inference/model+optimizer loader.
        DataLoader workers, samplers and augmentation state are not restored;
        this payload does not claim exact unattended training resumption.
        """
        path = os.path.join(self.cfg.out_dir, filename)
        numpy_state = np.random.get_state()
        payload = {
            "epoch":   epoch,
            "metrics": metrics,
            "model":   model.module.state_dict()
                       if isinstance(model, nn.DataParallel)
                       else model.state_dict(),
            "checkpoint_schema_version": 2,
            "config": asdict(self.cfg),
            "loss_fn": self.loss_fn.state_dict() if self.loss_fn is not None else None,
            "loss_policy": dict(self.loss_policy),
            "initial_model_sha256": self.initial_model_sha256,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "training_state": {
                "next_epoch": epoch + 1,
                "global_step": self._global_step,
                "selection": dict(selection_state or {}),
            },
            "rng_state": {
                "python": random.getstate(),
                # ndarray pickles require unsafe globals under weights_only;
                # a list keeps the original uint32 state losslessly loadable.
                "numpy": {
                    "bit_generator": numpy_state[0],
                    "keys": numpy_state[1].tolist(),
                    "position": int(numpy_state[2]),
                    "has_gauss": int(numpy_state[3]),
                    "cached_gaussian": float(numpy_state[4]),
                },
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            },
            "resume_support": {
                "exact_automatic_resume": False,
                "limitation": "DataLoader worker RNG, sampler/generator and augmentation state are not restored; no resume CLI or automatic retry.",
                "completed_epoch_state_saved": True,
            },
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        directory = os.path.dirname(path) or "."
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(filename)}.", suffix=".tmp", dir=directory,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            if hasattr(os, "O_DIRECTORY"):
                directory_descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return path

    def load_checkpoint(
        self,
        model: nn.Module,
        path: str,
        optimizer: torch.optim.Optimizer | None = None,
        strict: bool = True,
    ) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state   = payload.get("model", payload)
        target  = model.module if isinstance(model, nn.DataParallel) else model
        missing = target.load_state_dict(state, strict=strict)
        if missing.missing_keys:
            print(f"[load_checkpoint] missing keys: {missing.missing_keys}")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    # Private training epoch methods

    def _train_epoch_supervised(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """One finetune epoch: forward, task loss, backward, step."""
        cfg        = self.cfg
        total_loss = 0.0
        n          = 0
        clip_norm_sum, clipped_steps = 0.0, 0

        for batch in self.train_dl:
            images = batch["image"].to(self.device, non_blocking=True).float()
            masks  = batch["mask"].to(self.device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.use_amp):
                output = model(images)
                # TeacherModel returns (logits, aux_logits) during training when
                # aux_head is present (standard UperNet deep supervision).
                if isinstance(output, tuple):
                    logits, aux_logits = output
                    if cfg.task_loss == "focal":
                        loss = focal_loss(logits, masks, cfg.gamma, cfg.ignore_index) \
                             + 0.4 * focal_loss(aux_logits, masks, cfg.gamma, cfg.ignore_index)
                    else:
                        loss = ce_loss(logits, masks, cfg.ignore_index) \
                             + 0.4 * ce_loss(aux_logits, masks, cfg.ignore_index)
                else:
                    logits = output
                    if cfg.task_loss == "focal":
                        loss = focal_loss(logits, masks, cfg.gamma, cfg.ignore_index)
                    else:
                        loss = ce_loss(logits, masks, cfg.ignore_index)

            self.scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                self.scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
                clip_norm_sum += grad_norm
                clipped_steps += int(grad_norm > cfg.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()

            self._global_step += 1
            total_loss += loss.item()
            n += 1

        result = {"loss": total_loss / max(n, 1)}
        if cfg.grad_clip > 0:
            result.update(pre_clip_grad_norm_mean=clip_norm_sum / max(n, 1),
                          clipped_step_fraction=clipped_steps / max(n, 1))
        return result

    def _train_epoch_distill(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """One distillation epoch: student forward, DistillLoss, backward, step.

                With self.pcgrad=True the per-term gradients are computed separately and the
                conflicting components are projected out (PCGrad, Yu et al. NeurIPS 2020). That
                path runs in FP32 without AMP and skips the GradScaler.

                If self.pseudo_dl_ssfd is set and loss_fn has forward_feat_only(), each labeled
                step is followed by an SSFD-only forward on a pseudo batch (standard path only,
                doesn't work with PCGrad).
        """
        cfg          = self.cfg
        loss_fn      = self.loss_fn
        clip_norm_sum, clipped_steps, clip_steps = 0.0, 0, 0
        total_loss        = 0.0
        component_totals: dict[str, float] = {}   # dynamic: accumulates all keys from components
        total_pseudo_feat = 0.0
        total_concept     = 0.0
        n                 = 0

        has_concept = (
            self.concept_loss_fn is not None and self.concept_distil_alpha > 0.0
        )
        use_pcgrad        = self.pcgrad             and hasattr(loss_fn, "compute_loss_terms")
        use_bb_pcgrad     = self.ofa_backbone_pcgrad and hasattr(loss_fn, "branch_conflict_mode")
        use_agr           = self.ofa_agr            and hasattr(loss_fn, "get_agr_weight_and_update")
        use_fagr          = self.ofa_fagr           and hasattr(loss_fn, "branch_conflict_mode")

        # Secondary SSFD-only pass on pseudo batches is incompatible with PCGrad
        # (PCGrad runs its own forward; interleaving would double-count gradients).
        has_feat_only = (
            self.pseudo_dl_ssfd is not None
            and hasattr(loss_fn, "forward_feat_only")
            and not use_pcgrad
        )
        pseudo_iter = iter(self.pseudo_dl_ssfd) if has_feat_only else None

        for batch in self.train_dl:
            images = batch["image"].to(self.device, non_blocking=True).float()
            masks  = batch["mask"].to(self.device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            c_loss_item = 0.0

            # Get the pre-computed targets before the pcgrad/standard split, both paths need
            # them. PCGrad reads them via loss_fn._cached_proxy_target, the standard path uses
            # them directly.
            precomputed_teacher_logits = None
            if "soft_logits" in batch:
                precomputed_teacher_logits = batch["soft_logits"].to(
                    self.device, non_blocking=True
                ).float()

            precomputed_proxy_target = None
            proxy_mode = getattr(loss_fn, "proxy_mode", None)
            _needs_gradcam = (
                ("gradcam" in batch) and (
                    proxy_mode == "teacher_gradcam"
                    or getattr(loss_fn, "gradcam_weight_ofa", False)
                )
            )
            _needs_contrast = (
                "gradcam_contrast" in batch
                and getattr(loss_fn, "gradcam_contrast_weight", False)
            )
            _needs_sens = "sens_map" in batch and proxy_mode == "band_attribution"
            if _needs_gradcam:
                precomputed_proxy_target = batch["gradcam"].to(
                    self.device, non_blocking=True
                ).float()
            elif _needs_contrast:
                precomputed_proxy_target = batch["gradcam_contrast"].to(
                    self.device, non_blocking=True
                ).float()   # (B, H, W) class-discrepancy contrast map
            elif _needs_sens:
                precomputed_proxy_target = batch["sens_map"].to(
                    self.device, non_blocking=True
                ).float()   # (B, 6, H, W)

            # Cache on loss_fn so compute_loss_terms() (called by PCGrad and
            # grad_conflict) can access the precomputed target without signature
            # changes to those internal methods.
            if hasattr(loss_fn, "_cached_proxy_target"):
                loss_fn._cached_proxy_target = precomputed_proxy_target

            if use_fagr:
                # FA-AGR path (sec. 97): intra-batch focal+KD anchored weighting
                loss_item, components = self._fagr_ofa_step(model, optimizer, images, masks)

            elif use_agr:
                # AGR-OFA path (sec. 96, abandoned): EMA temporal self-consistency
                loss_item, components = self._agr_ofa_step(model, optimizer, images, masks)

            elif use_bb_pcgrad:
                # Backbone PCGrad path: focal-anchored surgery on backbone only
                loss_item, components = self._ofa_backbone_pcgrad_step(model, optimizer, images, masks)

            elif use_pcgrad:
                # PCGrad path: FP32, no AMP, writes .grad and calls step
                loss_item, components = self._pcgrad_step(model, optimizer, images, masks)

            elif self.sam:
                # SAM path: two forward/backward passes (utils/sam.py).
                # Closure recomputes the full student+teacher forward and loss;
                # each call re-populates the OFA hook buffers, so loss_fn works
                # unchanged. Assumes no_amp (these runs use --no_amp).
                def _closure():
                    student_logits = model(images)
                    _fp = inspect.signature(loss_fn.forward).parameters
                    _kw = {}
                    if "precomputed_proxy_target" in _fp:
                        _kw["precomputed_proxy_target"] = precomputed_proxy_target
                    l, c = loss_fn(student_logits, images, masks,
                                   precomputed_teacher_logits=precomputed_teacher_logits, **_kw)
                    return l, c

                optimizer.zero_grad(set_to_none=True)
                loss, components = _closure()
                loss.backward()                                   # grad at w
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                self._sam.first_step()                            # w to w + rho*g/||g||
                optimizer.zero_grad(set_to_none=True)
                loss2, _ = _closure()
                loss2.backward()                                  # grad at w'
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                self._sam.second_step()                           # restore w, base step
                loss_item = loss.item()

            else:
                # Standard AMP path

                c_loss_item = 0.0
                with torch.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_amp):
                    student_logits = model(images)
                    # Teacher forward inside loss_fn also fires ConceptDistilLoss hooks
                    _fwd_params = inspect.signature(loss_fn.forward).parameters
                    loss, components = loss_fn(
                        student_logits, images, masks,
                        precomputed_teacher_logits=precomputed_teacher_logits,
                        **({"precomputed_proxy_target": precomputed_proxy_target}
                           if "precomputed_proxy_target" in _fwd_params else {}),
                    )
                    if has_concept:
                        # Teacher neck feats already populated above - skips 2nd forward
                        c_loss, _ = self.concept_loss_fn(images)
                        loss = loss + self.concept_distil_alpha * c_loss
                        c_loss_item = c_loss.item()
                        total_concept += c_loss_item

                self.scaler.scale(loss).backward()

                # Secondary step: SSFD-only on pseudo batch
                if has_feat_only:
                    try:
                        pseudo_batch = next(pseudo_iter)
                    except StopIteration:
                        pseudo_iter  = iter(self.pseudo_dl_ssfd)
                        pseudo_batch = next(pseudo_iter)

                    pseudo_images = pseudo_batch["image"].to(
                        self.device, non_blocking=True
                    ).float()

                    with torch.autocast(device_type=self.device.type,
                                        dtype=self.amp_dtype, enabled=self.use_amp):
                        _ = model(pseudo_images)
                        pseudo_loss, pseudo_feat_val = loss_fn.forward_feat_only(
                            pseudo_images
                        )

                    pseudo_loss = pseudo_loss * self.pseudo_ssfd_weight
                    self.scaler.scale(pseudo_loss).backward()
                    total_pseudo_feat += pseudo_feat_val

                if cfg.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
                    clip_norm_sum += grad_norm
                    clipped_steps += int(grad_norm > cfg.grad_clip)
                    clip_steps += 1
                self.scaler.step(optimizer)
                self.scaler.update()

                loss_item = loss.item()

            # Step-level tracking (both paths)
            self._global_step += 1
            if (self.grad_logger is not None
                    and self._global_step % self.grad_conflict_interval == 0):
                self._log_grad_conflict(model, images, masks, self._global_step)

            # diag: per-branch loss and entropy, already cached in forward so no extra cost
            if self.ofa_diagnose and loss_fn is not None:
                if hasattr(loss_fn, "get_branch_loss_breakdown"):
                    for k, v in loss_fn.get_branch_loss_breakdown().items():
                        components[f"branch_{k}"] = v
                        component_totals[f"branch_{k}"] = component_totals.get(f"branch_{k}", 0.0) + v
                if hasattr(loss_fn, "get_branch_entropy_breakdown"):
                    for k, v in loss_fn.get_branch_entropy_breakdown().items():
                        components[f"branch_{k}"] = v
                        component_totals[f"branch_{k}"] = component_totals.get(f"branch_{k}", 0.0) + v

                # Periodic backbone gradient norm + conflict angle diagnostic.
                if (hasattr(loss_fn, "branch_conflict_mode")
                        and self._global_step % self.ofa_diagnose_interval == 0):
                    diag = self._ofa_gnorm_diagnostic(model, optimizer, images, masks)
                    if diag and self.wandb_run is not None:
                        self.wandb_run.log(
                            {f"diag/{k}": v for k, v in diag.items()},
                            step=self._global_step,
                        )

            # Per-step wandb: all component magnitudes for HP tuning and instability detection.
            # Logs every key returned by loss_fn.forward() - new loss functions
            # (GradMatch-D "grad", SPL-D "proxy", etc.) appear automatically.
            if self.wandb_run is not None:
                step_log = {"step/loss_total": loss_item, "step/concept": c_loss_item}
                step_log.update({f"step/{k}": v for k, v in components.items()})
                self.wandb_run.log(step_log, step=self._global_step)

            total_loss += loss_item
            for k, v in components.items():
                component_totals[k] = component_totals.get(k, 0.0) + v
            n += 1

        nb = max(n, 1)
        log_dict: dict[str, float] = {"loss_total": total_loss / nb}
        log_dict.update({f"loss_{k}": v / nb for k, v in component_totals.items()})
        if clip_steps:
            log_dict.update(pre_clip_grad_norm_mean=clip_norm_sum / clip_steps,
                            clipped_step_fraction=clipped_steps / clip_steps)
        if has_feat_only:
            log_dict["loss_pseudo_feat"] = total_pseudo_feat / nb
        if has_concept:
            log_dict["loss_concept"] = total_concept / nb
        return log_dict

    # PCGrad

    def _get_loss_weights(self, term_names: list[str]) -> dict[str, float]:
        """Per-term combination weights from loss_fn, for PCGrad scaling.

                If loss_fn declares a _loss_weights dict/property (term name to combination
                weight) that is used, so this file doesn't have to know every loss function and
                new ones just work. Older loss functions without _loss_weights go through a
                heuristic fallback.
        """
        loss_fn = self.loss_fn

        # Preferred: loss_fn knows its own weights
        if hasattr(loss_fn, "_loss_weights"):
            declared = dict(loss_fn._loss_weights)
            # Fill any terms not declared (shouldn't happen if _loss_weights is complete)
            return {name: declared.get(name, 1.0) for name in term_names}

        # Fallback: heuristic extraction for legacy loss functions
        alpha   = getattr(loss_fn, "alpha", 0.5)
        weights: dict[str, float] = {}
        for name in term_names:
            if name == "focal":
                weights[name] = 1.0 - alpha
            elif name == "kd":
                weights[name] = alpha
            elif name == "feat_mse":
                w = getattr(loss_fn, "selective_feat_alpha", None)
                weights[name] = w if w is not None else getattr(loss_fn, "feat_distil_alpha", 0.0)
            elif name == "fird":
                weights[name] = getattr(loss_fn, "fird_alpha", 0.0)
            elif name == "at":
                weights[name] = getattr(loss_fn, "at_beta", 0.0)
            else:
                weights[name] = 1.0
        return weights

    def _pcgrad_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[float, dict]:
        """One PCGrad gradient surgery step (Yu et al., NeurIPS 2020).

                For each pair of loss terms (i, j) whose flat gradient vectors conflict (dot
                product < 0), the component of g_i that opposes g_j is removed. The corrected
                gradients are then weighted and summed into p.grad.

                compute_loss_terms() gives the individual FP32 losses, autograd.grad() one flat
                gradient per term, then the surgery g_i = g_i - (g_i.g_j / ||g_j||^2) g_j where
                g_i.g_j < 0, p.grad = sum_i w_i * g_i, grad clip and optimizer.step().

                No GradScaler, everything runs in FP32. Costs about N extra backward passes
                (N = number of active loss terms, usually 2-4).
        """
        cfg       = self.cfg
        loss_fn   = self.loss_fn
        unwrapped = model.module if isinstance(model, nn.DataParallel) else model

        # Use optimizer param groups - not just student model - so that loss_fn's
        # trainable params (neck adapter, student-side projectors) also receive
        # gradient writes. Otherwise they stay frozen at prewarm initialization,
        # making feat_mse match student features to a never-updated projection.
        params = [p for group in optimizer.param_groups
                  for p in group["params"] if p.requires_grad]
        _empty    = {"focal": 0.0, "kd": 0.0, "at": 0.0, "feat_distil": 0.0, "fird": 0.0}

        if not params:
            return 0.0, _empty

        # Fresh FP32 forward - fills hook buffers, returns individual Tensors
        terms = loss_fn.compute_loss_terms(unwrapped, images, masks)
        if not terms:
            return 0.0, _empty

        term_names = list(terms.keys())
        weights    = self._get_loss_weights(term_names)

        # Per-term gradients as flat GPU vectors
        # None entries (param not in the term's graph) become zeros so all vectors match.
        term_grads: list[torch.Tensor] = []
        term_items = list(terms.items())

        for i, (_, term) in enumerate(term_items):
            retain = i < len(term_items) - 1
            raw_g  = torch.autograd.grad(
                term, params,
                retain_graph = retain,
                create_graph = False,
                allow_unused = True,
            )
            flat = torch.cat([
                g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
                for g, p in zip(raw_g, params)
            ])
            term_grads.append(flat)

        # PCGrad surgery on the flat gradient vectors: for each ordered pair (i, j), i != j,
        # with g_i . g_j < 0, subtract the projection of g_i onto g_j so the conflicting part is
        # gone. Always projected against the original (unsurgered) g_j, so the order of the
        # pairs doesn't matter (same convention as Yu et al.).
        n_terms  = len(term_grads)
        modified = [g.clone() for g in term_grads]

        for i in range(n_terms):
            for j in range(n_terms):
                if i == j:
                    continue
                dot = torch.dot(modified[i], term_grads[j])
                if dot < 0:
                    g_j_sq      = (term_grads[j] * term_grads[j]).sum().clamp_min(1e-8)
                    modified[i] = modified[i] - (dot / g_j_sq) * term_grads[j]

        # weighted sum, written back to p.grad
        total_flat = sum(
            weights[name] * modified[i]
            for i, name in enumerate(term_names)
        )

        sizes  = [p.numel() for p in params]
        offset = 0
        for p, size in zip(params, sizes):
            p.grad  = total_flat[offset : offset + size].reshape(p.shape)
            offset += size

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()

        combined_loss = sum(weights[n] * t.item() for n, t in terms.items())
        # "feat_mse" is the SSFD key; OFA/SPL-D use "ofa" and "proxy".
        # Fall back through all aliases so any loss class logs correctly.
        _feat_val = (
            terms["feat_mse"].item() if "feat_mse" in terms else
            terms["ofa"].item()      if "ofa"      in terms else 0.0
        )
        components = {
            "focal":       terms["focal"].item()  if "focal"  in terms else 0.0,
            "kd":          terms["kd"].item()     if "kd"     in terms else 0.0,
            "at":          0.0,
            "feat_distil": _feat_val,
            "proxy":       terms["proxy"].item()  if "proxy"  in terms else 0.0,
            "fird":        terms["fird"].item()   if "fird"   in terms else 0.0,
        }
        return combined_loss, components

    # OFA Backbone PCGrad

    def _ofa_backbone_pcgrad_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[float, dict]:
        """Focal-anchored backbone gradient surgery for OFA-KD (sec. 94).

                PCGrad-style projection, but only on the shared backbone parameters: if a
                per-stage OFA branch's backbone gradient conflicts with focal's (dot product < 0),
                the conflicting component is projected out. Non-backbone parameters (exit branch
                heads, decoder head) get the normal weighted-sum gradient without surgery.

                sec. 93 shows the focal vs early-branch backbone conflict is the main problem at
                N=100-200. Full PCGrad projects the whole gradient, branch-head params included;
                this only operates where the conflict actually hurts, in the shared backbone.
                kd isn't touched, it usually cooperates with focal.

                compute_loss_terms(branch_conflict_mode=True) gives {focal, kd, ofa_s0..s3}.
                autograd.grad per term on all params is split into backbone / non-backbone, each
                ofa_si backbone gradient with a negative dot product against focal gets projected,
                p.grad = weighted surgical backbone gradient + weighted non-backbone sum, then
                grad clip and optimizer.step().

                Costs N+1 autograd.grad calls (N ~ 5-6 terms), same order as _pcgrad_step.
        """
        cfg        = self.cfg
        loss_fn    = self.loss_fn
        unwrapped  = model.module if isinstance(model, nn.DataParallel) else model

        all_params = [p for group in optimizer.param_groups
                      for p in group["params"] if p.requires_grad]
        _empty = {"focal": 0.0, "kd": 0.0, "at": 0.0, "feat_distil": 0.0, "fird": 0.0}
        if not all_params:
            return 0.0, _empty

        # Identify backbone parameter indices
        bb_module = None
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "backbone"):
            bb_module = unwrapped.model.backbone   # EfficientViT-B0 path
        if bb_module is None:
            # Fallback: plain .backbone attribute (other student architectures)
            bb_module = getattr(unwrapped, "backbone", None)
        if bb_module is None:
            # No backbone found - use standard PCGrad without a warning
            return self._pcgrad_step(model, optimizer, images, masks)

        bb_ids     = {id(p) for p in bb_module.parameters() if p.requires_grad}
        bb_idx     = [i for i, p in enumerate(all_params) if id(p) in bb_ids]
        other_idx  = [i for i, p in enumerate(all_params) if id(p) not in bb_ids]
        bb_params  = [all_params[i] for i in bb_idx]

        # Fresh forward in branch_conflict_mode
        old_mode = loss_fn.branch_conflict_mode
        loss_fn.branch_conflict_mode = True
        terms = loss_fn.compute_loss_terms(unwrapped, images, masks)
        loss_fn.branch_conflict_mode = old_mode

        if not terms or len(terms) < 2:
            return 0.0, _empty

        weights    = self._get_loss_weights(list(terms.keys()))
        term_items = list(terms.items())
        n_terms    = len(term_items)

        # Per-term gradients for ALL params; split into bb / other
        bb_flats:    list[torch.Tensor] = []
        other_flats: list[torch.Tensor] = []

        for i, (_, term) in enumerate(term_items):
            retain = i < n_terms - 1
            raw = torch.autograd.grad(
                term, all_params,
                retain_graph=retain, create_graph=False, allow_unused=True,
            )
            def _flat(indices):
                return torch.cat([
                    raw[j].flatten() if raw[j] is not None
                    else torch.zeros(all_params[j].numel(), device=all_params[j].device)
                    for j in indices
                ]) if indices else torch.zeros(0, device=images.device)
            bb_flats.append(_flat(bb_idx))
            other_flats.append(_flat(other_idx))

        # Focal-anchored PCGrad surgery on backbone
        focal_i  = next((i for i, (k, _) in enumerate(term_items) if k == "focal"), 0)
        focal_bb = bb_flats[focal_i]

        modified_bb = [g.clone() for g in bb_flats]
        for i, (name, _) in enumerate(term_items):
            if name in ("focal", "kd"):
                continue   # anchor and KD are never projected
            dot = torch.dot(modified_bb[i], focal_bb)
            if dot < 0:
                focal_sq    = (focal_bb * focal_bb).sum().clamp_min(1e-8)
                modified_bb[i] = modified_bb[i] - (dot / focal_sq) * focal_bb

        # weighted sum, written to .grad
        total_bb = sum(weights.get(n, 1.0) * modified_bb[i]
                       for i, (n, _) in enumerate(term_items))
        total_other = sum(weights.get(n, 1.0) * other_flats[i]
                          for i, (n, _) in enumerate(term_items))

        offset = 0
        for j, p in zip(bb_idx, bb_params):
            sz = p.numel()
            p.grad = total_bb[offset:offset + sz].reshape(p.shape)
            offset += sz

        if other_idx:
            offset = 0
            for j in other_idx:
                p  = all_params[j]
                sz = p.numel()
                p.grad = total_other[offset:offset + sz].reshape(p.shape)
                offset += sz

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        optimizer.step()

        combined_loss = sum(weights.get(n, 1.0) * t.item() for n, t in terms.items())
        ofa_val = sum(t.item() for n, t in terms.items() if n.startswith("ofa_"))
        components = {
            "focal":       terms["focal"].item() if "focal" in terms else 0.0,
            "kd":          terms["kd"].item()    if "kd"    in terms else 0.0,
            "at":          0.0,
            "feat_distil": ofa_val,
            "fird":        0.0,
        }
        return combined_loss, components

    def _agr_ofa_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[float, dict]:
        """Adaptive Gradient Reliability weighting for OFA-KD (sec. 96).

                Each exit branch's contribution to the shared backbone gradient is scaled by its
                consistency between steps. A branch whose backbone gradient lines up with its own
                recent EMA (beta=0.9) gets full weight, one that flips direction gets a weight
                equal to the cosine similarity (ReLU-clipped to [0, 1]).

                Why: sec. 94 (BB-PCGrad) projected the OFA branch gradients against focal, but at
                N<=100 the focal gradient itself is noisy, and projecting against a noisy anchor
                removes useful signal together with the harmful part. AGR-OFA needs no anchor. It
                checks whether a branch is consistent with itself over time instead of whether it
                agrees with focal right now. A self-consistent gradient has found a direction worth
                following; one that keeps reversing is noise (the branch is undecided and adds net
                zero over several steps), and the proportional weight suppresses it.

                Weight at step t:
                    g_i^t        backbone gradient of branch i at step t
                    gbar_i^{t-1} beta-EMA of branch i's backbone gradients up to step t-1
                    w_i^t        = max(0, cos(g_i^t, gbar_i^{t-1}))   in [0, 1]

                Backbone update:
                    delta theta_bb = g_focal + g_kd + sum_i w_i^t * g_i^t

                Non-backbone parameters (exit branch heads, decoder head) get the normal weighted
                sum of all gradients. They are branch-specific, so there's no cross-contamination
                to regularise.

                The EMA is updated in loss_fn.get_agr_weight_and_update() after the weight is
                computed, so the weight only reflects the history before this step.

                Same overhead as _ofa_backbone_pcgrad_step (N+1 autograd.grad calls, FP32).
        """
        cfg       = self.cfg
        loss_fn   = self.loss_fn
        unwrapped = model.module if isinstance(model, nn.DataParallel) else model

        all_params = [p for group in optimizer.param_groups
                      for p in group["params"] if p.requires_grad]
        _empty = {"focal": 0.0, "kd": 0.0, "at": 0.0, "feat_distil": 0.0, "fird": 0.0}
        if not all_params:
            return 0.0, _empty

        # Identify backbone parameter indices
        bb_module = None
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "backbone"):
            bb_module = unwrapped.model.backbone
        if bb_module is None:
            bb_module = getattr(unwrapped, "backbone", None)
        if bb_module is None:
            return self._pcgrad_step(model, optimizer, images, masks)

        bb_ids    = {id(p) for p in bb_module.parameters() if p.requires_grad}
        bb_idx    = [i for i, p in enumerate(all_params) if id(p) in bb_ids]
        other_idx = [i for i, p in enumerate(all_params) if id(p) not in bb_ids]
        bb_params = [all_params[i] for i in bb_idx]

        # Fresh forward in branch_conflict_mode
        old_mode = loss_fn.branch_conflict_mode
        loss_fn.branch_conflict_mode = True
        terms = loss_fn.compute_loss_terms(unwrapped, images, masks)
        loss_fn.branch_conflict_mode = old_mode

        if not terms or len(terms) < 2:
            return 0.0, _empty

        weights    = self._get_loss_weights(list(terms.keys()))
        term_items = list(terms.items())
        n_terms    = len(term_items)

        # Per-term backbone and non-backbone gradients
        bb_flats:    list[torch.Tensor] = []
        other_flats: list[torch.Tensor] = []

        for i, (_, term) in enumerate(term_items):
            retain = i < n_terms - 1
            raw = torch.autograd.grad(
                term, all_params,
                retain_graph=retain, create_graph=False, allow_unused=True,
            )
            def _flat(indices):
                return torch.cat([
                    raw[j].flatten() if raw[j] is not None
                    else torch.zeros(all_params[j].numel(), device=all_params[j].device)
                    for j in indices
                ]) if indices else torch.zeros(0, device=images.device)
            bb_flats.append(_flat(bb_idx))
            other_flats.append(_flat(other_idx))

        # AGR: reliability weights + EMA update. focal and kd backbone grads always get full
        # weight, no regularisation needed (sec. 93: kd and focal usually cooperate, the conflict
        # is branch-specific). The EMA update happens in get_agr_weight_and_update, after the
        # weight is read.
        agr_weights: list[float] = []
        for i, (name, _) in enumerate(term_items):
            if name in ("focal", "kd"):
                agr_weights.append(1.0)
            else:
                w = loss_fn.get_agr_weight_and_update(name, bb_flats[i])
                agr_weights.append(w)

        # Weighted backbone gradient
        total_bb = sum(
            weights.get(name, 1.0) * agr_weights[i] * bb_flats[i]
            for i, (name, _) in enumerate(term_items)
        )
        # Non-backbone params: standard weighted sum (no AGR needed - branch-specific)
        total_other = sum(
            weights.get(name, 1.0) * other_flats[i]
            for i, (name, _) in enumerate(term_items)
        )

        # Write to .grad and step
        offset = 0
        for j, p in zip(bb_idx, bb_params):
            sz = p.numel()
            p.grad = total_bb[offset:offset + sz].reshape(p.shape)
            offset += sz

        if other_idx:
            offset = 0
            for j in other_idx:
                p  = all_params[j]
                sz = p.numel()
                p.grad = total_other[offset:offset + sz].reshape(p.shape)
                offset += sz

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        optimizer.step()

        combined_loss = sum(weights.get(n, 1.0) * t.item() for n, t in terms.items())
        ofa_val = sum(t.item() for n, t in terms.items() if n.startswith("ofa_"))
        components = {
            "focal":       terms["focal"].item() if "focal" in terms else 0.0,
            "kd":          terms["kd"].item()    if "kd"    in terms else 0.0,
            "at":          0.0,
            "feat_distil": ofa_val,
            "fird":        0.0,
        }
        return combined_loss, components

    def _fagr_ofa_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[float, dict]:
        """Focal-anchored AGR for OFA-KD (sec. 97).

                Fixes a design flaw of sec. 96, which measured each branch's temporal
                self-consistency (EMA cosine). That doesn't detect conflict between branches: a
                branch that consistently conflicts with focal is self-consistent and gets a high
                EMA weight.

                sec. 97 compares each branch, within the batch, with the combined focal+KD
                backbone gradient instead, which is the actual conflict signal:

                    anchor   = normalize(g_focal_bb + g_kd_bb)
                    w_i      = max(0, cos(g_i_bb, anchor))   in [0, 1]

                focal and kd get weight 1.0 (trusted anchors, not regularised). Branches that agree
                with the main objectives get a high weight, conflicting ones are scaled down.

                focal+KD as the anchor (sec. 94 BB-PCGrad used focal only) is a more stable
                reference at small N: teacher KD comes from a 300M-param pretrained model and is
                less noisy than the sparse labeled focal signal. No temporal state needed, the
                weights are computed entirely within the batch.
        """
        cfg       = self.cfg
        loss_fn   = self.loss_fn
        unwrapped = model.module if isinstance(model, nn.DataParallel) else model

        all_params = [p for group in optimizer.param_groups
                      for p in group["params"] if p.requires_grad]
        _empty = {"focal": 0.0, "kd": 0.0, "at": 0.0, "feat_distil": 0.0, "fird": 0.0}
        if not all_params:
            return 0.0, _empty

        # Identify backbone parameter indices
        bb_module = None
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "backbone"):
            bb_module = unwrapped.model.backbone
        if bb_module is None:
            bb_module = getattr(unwrapped, "backbone", None)
        if bb_module is None:
            return self._pcgrad_step(model, optimizer, images, masks)

        bb_ids    = {id(p) for p in bb_module.parameters() if p.requires_grad}
        bb_idx    = [i for i, p in enumerate(all_params) if id(p) in bb_ids]
        other_idx = [i for i, p in enumerate(all_params) if id(p) not in bb_ids]
        bb_params = [all_params[i] for i in bb_idx]

        # Fresh forward in branch_conflict_mode
        old_mode = loss_fn.branch_conflict_mode
        loss_fn.branch_conflict_mode = True
        terms = loss_fn.compute_loss_terms(unwrapped, images, masks)
        loss_fn.branch_conflict_mode = old_mode

        if not terms or len(terms) < 2:
            return 0.0, _empty

        weights    = self._get_loss_weights(list(terms.keys()))
        term_items = list(terms.items())
        n_terms    = len(term_items)

        # Per-term backbone and non-backbone gradients
        bb_flats:    list[torch.Tensor] = []
        other_flats: list[torch.Tensor] = []

        for i, (_, term) in enumerate(term_items):
            retain = i < n_terms - 1
            raw = torch.autograd.grad(
                term, all_params,
                retain_graph=retain, create_graph=False, allow_unused=True,
            )
            def _flat(indices):
                return torch.cat([
                    raw[j].flatten() if raw[j] is not None
                    else torch.zeros(all_params[j].numel(), device=all_params[j].device)
                    for j in indices
                ]) if indices else torch.zeros(0, device=images.device)
            bb_flats.append(_flat(bb_idx))
            other_flats.append(_flat(other_idx))

        # FA-AGR: build anchor and compute per-branch weights
        # anchor = normalize(g_focal_bb + g_kd_bb)
        # focal and kd are the primary objectives; their combined backbone
        # gradient is the reference direction that all branches are judged against.
        anchor = torch.zeros_like(bb_flats[0])
        for i, (name, _) in enumerate(term_items):
            if name in ("focal", "kd"):
                anchor = anchor + bb_flats[i]
        anchor_norm = anchor.norm()

        fagr_weights: list[float] = []
        for i, (name, _) in enumerate(term_items):
            if name in ("focal", "kd"):
                fagr_weights.append(1.0)
            elif anchor_norm < 1e-12:
                # no anchor signal (rare), keep the branch at full weight
                fagr_weights.append(1.0)
            else:
                g_norm = bb_flats[i].norm()
                if g_norm < 1e-12:
                    fagr_weights.append(0.0)
                else:
                    anchor_unit = anchor / anchor_norm
                    cos_sim = torch.dot(bb_flats[i] / g_norm, anchor_unit).item()
                    fagr_weights.append(max(0.0, cos_sim))

        # Weighted backbone gradient
        total_bb = sum(
            weights.get(name, 1.0) * fagr_weights[i] * bb_flats[i]
            for i, (name, _) in enumerate(term_items)
        )
        # Non-backbone params: standard weighted sum (no FA-AGR needed)
        total_other = sum(
            weights.get(name, 1.0) * other_flats[i]
            for i, (name, _) in enumerate(term_items)
        )

        # Write to .grad and step
        offset = 0
        for j, p in zip(bb_idx, bb_params):
            sz = p.numel()
            p.grad = total_bb[offset:offset + sz].reshape(p.shape)
            offset += sz

        if other_idx:
            offset = 0
            for j in other_idx:
                p  = all_params[j]
                sz = p.numel()
                p.grad = total_other[offset:offset + sz].reshape(p.shape)
                offset += sz

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        optimizer.step()

        combined_loss = sum(weights.get(n, 1.0) * t.item() for n, t in terms.items())
        ofa_val = sum(t.item() for n, t in terms.items() if n.startswith("ofa_"))
        components = {
            "focal":       terms["focal"].item() if "focal" in terms else 0.0,
            "kd":          terms["kd"].item()    if "kd"    in terms else 0.0,
            "at":          0.0,
            "feat_distil": ofa_val,
            "fird":        0.0,
        }
        return combined_loss, components

    # OFA diagnostic: backbone gradient norms + conflict angles

    def _ofa_gnorm_diagnostic(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict[str, float]:
        """Compute per-term backbone gradient norms and conflict angles vs focal.

                Runs a fresh forward + per-term autograd.grad (same as _fagr_ofa_step) but
                ONLY to measure; no gradients are written to parameters, optimizer is untouched.

                Returns dict with keys:
                    bb_gnorm_{focal,kd,ofa_s0..s3}   -  L2 norm of each term's backbone gradient
                    bb_cos_{ofa_s0..s3}_focal         -  cosine similarity of branch vs focal bb grad
                    bb_gnorm_ratio_{ofa_s0..s3}       -  branch gnorm / focal gnorm (>1 = branch dominates)

                All values are floats. Returns {} if backbone cannot be identified or
                loss_fn lacks branch_conflict_mode.
        """
        loss_fn   = self.loss_fn
        unwrapped = model.module if isinstance(model, nn.DataParallel) else model

        if loss_fn is None or not hasattr(loss_fn, "branch_conflict_mode"):
            return {}

        optimizer_params = [p for group in optimizer.param_groups
                            for p in group["params"] if p.requires_grad]
        if not optimizer_params:
            return {}

        bb_module = None
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "backbone"):
            bb_module = unwrapped.model.backbone
        if bb_module is None:
            bb_module = getattr(unwrapped, "backbone", None)
        if bb_module is None:
            return {}

        bb_ids  = {id(p) for p in bb_module.parameters() if p.requires_grad}
        bb_idx  = [i for i, p in enumerate(optimizer_params) if id(p) in bb_ids]
        if not bb_idx:
            return {}

        old_mode = loss_fn.branch_conflict_mode
        loss_fn.branch_conflict_mode = True
        try:
            terms = loss_fn.compute_loss_terms(unwrapped, images, masks)
        except Exception:
            loss_fn.branch_conflict_mode = old_mode
            return {}
        loss_fn.branch_conflict_mode = old_mode

        if not terms:
            return {}

        term_items = list(terms.items())
        n_terms    = len(term_items)
        bb_flats: list[torch.Tensor] = []

        try:
            for i, (_, term) in enumerate(term_items):
                retain = i < n_terms - 1
                raw = torch.autograd.grad(
                    term, optimizer_params,
                    retain_graph=retain, create_graph=False, allow_unused=True,
                )
                flat = torch.cat([
                    raw[j].detach().flatten() if raw[j] is not None
                    else torch.zeros(optimizer_params[j].numel(),
                                     device=optimizer_params[j].device)
                    for j in bb_idx
                ])
                bb_flats.append(flat)
        except Exception:
            return {}

        result: dict[str, float] = {}
        focal_flat: torch.Tensor | None = None
        focal_norm = 0.0

        for i, (name, _) in enumerate(term_items):
            norm = bb_flats[i].norm().item()
            result[f"bb_gnorm_{name}"] = norm
            if name == "focal":
                focal_flat = bb_flats[i]
                focal_norm = norm

        if focal_flat is not None and focal_norm > 1e-12:
            focal_unit = focal_flat / focal_norm
            for i, (name, _) in enumerate(term_items):
                if name in ("focal", "kd"):
                    continue
                g_norm = bb_flats[i].norm().item()
                if g_norm > 1e-12:
                    cos = torch.dot(bb_flats[i] / g_norm, focal_unit).item()
                    result[f"bb_cos_{name}_focal"] = cos
                    result[f"bb_gnorm_ratio_{name}"] = g_norm / focal_norm

        return result

    # Gradient conflict logging

    def _log_grad_conflict(
        self,
        model: nn.Module,
        images: torch.Tensor,
        masks: torch.Tensor,
        step: int,
    ) -> None:
        """Compute per-term loss gradients and record pairwise cosine similarities.

                Runs a fresh student+teacher forward (FP32, no autocast) via
                loss_fn.compute_loss_terms(), then uses torch.autograd.grad to get the
                gradient vector each term would induce on the student's parameters.
                No optimizer.step(), this is only for analysis.

                retain_graph=True is set for all but the last term so the shared
                computation graph (through student_logits) survives multiple grad calls.
        """
        loss_fn = self.loss_fn
        if loss_fn is None or not hasattr(loss_fn, "compute_loss_terms"):
            return

        unwrapped = model.module if isinstance(model, nn.DataParallel) else model
        params    = [p for p in unwrapped.parameters() if p.requires_grad]
        if not params:
            return

        try:
            terms = loss_fn.compute_loss_terms(unwrapped, images, masks)
        except Exception as e:
            print(f"[GradConflict] compute_loss_terms failed at step {step}: {e}")
            return

        if len(terms) < 2:
            return

        named_grads: dict[str, torch.Tensor] = {}
        term_items  = list(terms.items())

        for i, (name, term) in enumerate(term_items):
            retain = i < len(term_items) - 1   # keep graph alive for subsequent calls
            try:
                raw_grads = torch.autograd.grad(
                    term, params,
                    retain_graph = retain,
                    create_graph = False,
                    allow_unused = True,
                )
                # None (param not in this term's graph) becomes zeros so every flat vector has
                # length sum(p.numel()). feat_mse for example doesn't flow through the decoder
                # head, so the head params would be missing and cosine_similarity would get
                # mismatched sizes.
                flat_parts = [
                    g.detach().cpu().flatten() if g is not None
                    else torch.zeros(p.numel(), dtype=torch.float32)
                    for g, p in zip(raw_grads, params)
                ]
                named_grads[name] = torch.cat(flat_parts)
            except RuntimeError as e:
                print(f"[GradConflict] grad failed for '{name}' at step {step}: {e}")
                continue

        if len(named_grads) >= 2:
            self.grad_logger.log(step, named_grads)

    # Scheduler & helpers

    def _build_optimizer(
        self,
        model: nn.Module,
        cfg,
    ) -> torch.optim.Optimizer:
        # Student backbone + head
        param_groups = [{"params": list(model.parameters()), "lr": cfg.lr,
                         "name": "student"}]

        if self.loss_fn is not None:
            # feat_projectors at 0.1x LR, otherwise the projectors absorb the mismatch and the
            # backbone stops aligning (projector shortcut).
            proj_param_ids: set[int] = set()
            if hasattr(self.loss_fn, "feat_projectors"):
                for p in self.loss_fn.feat_projectors.parameters():
                    proj_param_ids.add(id(p))
                proj_params = [p for p in self.loss_fn.feat_projectors.parameters()
                               if p.requires_grad]
                if proj_params:
                    param_groups.append(
                        {"params": proj_params, "lr": cfg.lr * 0.1,
                         "name": "feat_projectors"}
                    )

            # All other trainable loss_fn params (neck_adapter, windowed_neck, ...)
            # at full LR. Without this they receive gradients but are never updated.
            extra = [p for p in self.loss_fn.parameters()
                     if p.requires_grad and id(p) not in proj_param_ids]
            if extra:
                param_groups.append(
                    {"params": extra, "lr": cfg.lr, "name": "loss_fn_extra"}
                )
                extra_count = sum(p.numel() for p in extra)
                print(f"[Trainer] loss_fn extra trainable params: {extra_count:,} "
                      f"(neck_adapter / windowed_neck)")

        # ConceptDistilLoss probes + adapters at 1e-3 (fast convergence for linear probes)
        if self.concept_loss_fn is not None:
            concept_params = [p for p in self.concept_loss_fn.parameters() if p.requires_grad]
            if concept_params:
                param_groups.append(
                    {"params": concept_params, "lr": 1e-3, "name": "concept_probes"}
                )
                print(f"[Trainer] concept probes/adapters: "
                      f"{sum(p.numel() for p in concept_params):,} params  lr=1e-3")

        # If only one group with no special LR, pass a plain list for AdamW
        if len(param_groups) == 1:
            return torch.optim.AdamW(
                param_groups[0]["params"], lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        return torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ):
        cfg = self.cfg
        warmup = getattr(cfg, "warmup_epochs", 0)

        # Build the main scheduler (runs after warmup, or alone if warmup=0)
        main_epochs = max(cfg.epochs - warmup, 1)
        if cfg.scheduler == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_epochs
            )
        elif cfg.scheduler == "step":
            main_sched = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma
            )
        else:
            main_sched = None

        if warmup <= 0:
            return main_sched

        # Linear warmup: LR goes from 0 to cfg.lr over `warmup` epochs, then main_sched takes
        # over for the remaining epochs. SequentialLR handles the switch.
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,   # near-zero start (avoids exactly 0 to prevent /0 in some opts)
            end_factor=1.0,
            total_iters=warmup,
        )
        if main_sched is None:
            return warmup_sched

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup],
        )

    def _init_wandb(self):
        cfg = self.cfg
        try:
            import os
            # Off by default. WANDB_MODE=online turns remote logging on.
            os.environ.setdefault("WANDB_MODE", "disabled")
            import wandb
            # wandb.finish() does nothing if no run is active. It clears a run that
            # HuggingFace / Prithvi imports may have started before Trainer construction,
            # otherwise wandb.init() silently reuses that run and its project (typically
            # "FoundationModelExperiments").
            wandb.finish()
            # Set env var as a hard override: local ~/.config/wandb/settings or a
            # WANDB_PROJECT env var already in the environment can silently override
            # the project= kwarg on some wandb versions.
            
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
            return wandb.init(
                project = cfg.wandb_project,
                name    = cfg.wandb_run_name,
                config  = cfg.__dict__,
            )
        except Exception as e:
            print(f"[Trainer] wandb init failed: {e}")
            return None

    def _log(self, step: int, metrics: dict, prefix: str = "") -> None:
        tag   = f"[{prefix}] " if prefix else ""
        parts = [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
            if not isinstance(v, list)
        ]
        # Expand per-class lists (iou_per_class, f1_per_class) for print + W&B.
        # Keys: IoU_background / IoU_flood (or IoU_class0 / IoU_class1 for >2 classes).
        expanded: dict[str, float] = {}
        for list_key in ("iou_per_class", "f1_per_class"):
            if list_key in metrics:
                short = "IoU" if list_key == "iou_per_class" else "f1"
                for i, v in enumerate(metrics[list_key]):
                    if isinstance(v, float) and not (v != v):   # skip NaN
                        name = (self.class_names[i]
                                if self.class_names and i < len(self.class_names)
                                else f"class{i}")
                        expanded[f"{short}_{name}"] = v
                        parts.append(f"{short}_{name}={v:.4f}")
        print(f"  {tag}step={step}  " + "  ".join(parts))

        if self.wandb_run is not None:
            log_dict = {
                f"{prefix}/{k}" if prefix else k: v
                for k, v in metrics.items()
                if not isinstance(v, list)
            }
            for k, v in expanded.items():
                log_dict[f"{prefix}/{k}" if prefix else k] = v
            # use self._global_step for epoch-level logs too. Step-level logs also use
            # _global_step and may already be past the epoch index; using epoch
            # here would move the wandb step counter backwards.
            self.wandb_run.log(log_dict, step=self._global_step)

    # feat_alpha curriculum scheduler

    def _compute_feat_alpha(self, epoch: int) -> float:
        """Scheduled feat_alpha for this epoch.

                "warmup": linear ramp from 0 to target over feat_alpha_warmup_epochs, then hold.
                "decay": hold at target, then ramp linearly down to 0 over the last
                feat_alpha_decay_epochs. "warmup_decay": both, the two factors multiply.
                "constant": returns target unchanged.
        """
        cfg      = self.cfg
        target   = cfg.feat_alpha
        schedule = getattr(cfg, "feat_alpha_schedule", "constant")
        if schedule == "constant" or target == 0.0:
            return target

        warmup_e    = max(0, getattr(cfg, "feat_alpha_warmup_epochs", 0))
        decay_e     = max(0, getattr(cfg, "feat_alpha_decay_epochs", 0))
        total_e     = cfg.epochs

        warmup_factor = min(1.0, epoch / warmup_e) if warmup_e > 0 else 1.0

        decay_start  = max(0, total_e - decay_e)
        if decay_e > 0 and epoch >= decay_start:
            decay_factor = max(0.0, 1.0 - (epoch - decay_start) / decay_e)
        else:
            decay_factor = 1.0

        return target * warmup_factor * decay_factor

    def _update_feat_alpha(self, epoch: int) -> None:
        """Apply feat_alpha schedule by mutating the live loss_fn attribute."""
        if self.loss_fn is None:
            return
        schedule = getattr(self.cfg, "feat_alpha_schedule", "constant")
        if schedule == "constant":
            return
        alpha = self._compute_feat_alpha(epoch)
        # SSFDLoss uses selective_feat_alpha; DistillLoss uses feat_distil_alpha.
        for attr in ("selective_feat_alpha", "feat_distil_alpha"):
            if hasattr(self.loss_fn, attr):
                setattr(self.loss_fn, attr, alpha)
                break
        log_interval = max(1, self.cfg.epochs // 10)
        if epoch % log_interval == 0 or epoch < 3:
            print(f"[Trainer] feat_alpha schedule={schedule}  "
                  f"epoch={epoch}/{self.cfg.epochs}  feat_alpha={alpha:.4f}")

    @staticmethod
    def _elapsed(t0: float) -> str:
        s = time.perf_counter() - t0
        return f"{s/60:.1f}min" if s > 90 else f"{s:.1f}s"
