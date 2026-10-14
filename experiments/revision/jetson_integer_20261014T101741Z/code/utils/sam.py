"""Sharpness-Aware Minimization around an existing optimizer.

Reference: Foret et al., ICLR 2021. first_step ascends by rho*g/||g||;
second_step restores the weights and applies the base optimizer using the
perturbed gradient. Existing parameter groups and learning rates are retained.
This is an optional optimization control, not the main training recipe.
"""

from __future__ import annotations

import torch


class SAM:
    """Sharpness-Aware Minimization around a pre-constructed base optimizer."""

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05) -> None:
        self.base = base_optimizer
        self.rho = rho
        self._e_w: dict[torch.nn.Parameter, torch.Tensor] = {}

    @property
    def param_groups(self):
        return self.base.param_groups

    def _grad_norm(self) -> torch.Tensor:
        return torch.norm(torch.stack([
            p.grad.norm(2)
            for group in self.base.param_groups
            for p in group["params"] if p.grad is not None
        ]), 2)

    @torch.no_grad()
    def first_step(self) -> None:
        """Move to w + rho*g/||g|| (the local worst case) and remember the offset."""
        scale = self.rho / (self._grad_norm() + 1e-12)
        self._e_w.clear()
        for group in self.base.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self._e_w[p] = e_w

    @torch.no_grad()
    def second_step(self) -> None:
        """Restore w, then take the base-optimizer step with the gradient at w'."""
        for group in self.base.param_groups:
            for p in group["params"]:
                if p in self._e_w:
                    p.sub_(self._e_w[p])
        self.base.step()
        self._e_w.clear()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, sd):
        self.base.load_state_dict(sd)
