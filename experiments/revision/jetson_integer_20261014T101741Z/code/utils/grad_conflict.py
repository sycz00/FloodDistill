"""Gradient conflict analysis for multi-term distillation losses.

Records pairwise cosine similarities between the gradient vectors each loss
term induces on the student parameters.  Negative similarity = conflict.

Usage:
    logger = GradientConflictLogger(["focal", "kd", "feat_mse"])
    logger.log(step, {"focal": flat_grad_a, "kd": flat_grad_b, "feat_mse": flat_grad_c})
    logger.save_csv(out_dir)   # grad_conflict.csv
    logger.plot(out_dir)       # grad_conflict.png
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from itertools import combinations

import torch.nn.functional as F
from torch import Tensor


class GradientConflictLogger:
    """Tracks pairwise cosine similarities between per-term loss gradients.

        N active loss terms give C(N,2) unique pairs, and each log() call records one sample
        at a global step. Reading the plot: cosine > 0 means the terms pull the parameters the
        same way, < 0 means conflict, ~0 means orthogonal (neither helps nor hurts).
    """

    def __init__(self, term_names: list[str]) -> None:
        self.term_names    = list(term_names)
        self._steps:   list[int]              = []
        self._history: dict[str, list[float]] = defaultdict(list)
        # |cos| when conflicting, 0 otherwise - tracks PCGrad suppression fraction
        self._suppressed: dict[str, list[float]] = defaultdict(list)
        # Per-term gradient L2 norms (GradNorm-style relative magnitudes):
        # a term can have a small loss but a dominant gradient or vice versa -
        # cosine alone cannot distinguish "harmless because orthogonal" from
        # "harmless because negligible".
        self._norms: dict[str, list[float]] = defaultdict(list)

    @property
    def pair_keys(self) -> list[str]:
        return [f"{a}_vs_{b}" for a, b in combinations(self.term_names, 2)]

    def log(self, step: int, named_grads: dict[str, Tensor]) -> None:
        """Compute and record pairwise cosine similarities.

                Args:
                    step:        Global training step number.
                    named_grads: {term_name: flat gradient vector (1-D CPU float Tensor)}.
                                 Keys absent from self.term_names are silently ignored.
        """
        active = [n for n in self.term_names if n in named_grads]
        if len(active) < 2:
            return

        self._steps.append(step)

        for name in active:
            self._norms[name].append(named_grads[name].float().norm().item())

        for a, b in combinations(active, 2):
            key = f"{a}_vs_{b}"
            g1  = named_grads[a].float()
            g2  = named_grads[b].float()
            cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
            self._history[key].append(cos)
            # Fraction of g1 that PCGrad would project away: |cos| when conflicting.
            # When cos < 0, the projection of g1 onto g2 has magnitude |cos|*|g1|,
            # so suppressed_frac = |cos|. Zero when cooperative (nothing removed).
            self._suppressed[key].append(abs(cos) if cos < 0 else 0.0)

    def summary(self) -> str:
        """Build a formatted summary, print it and return it.

                Per pair: conflict_rate (fraction of steps with cosine < 0), mean_cos,
                neg_mean_cos (mean cosine on conflicting steps only), pcgrad_suppressed (expected
                fraction of gradient magnitude PCGrad would remove per step, i.e. mean |cos| on
                conflicting steps weighted by conflict_rate; 0.40 means PCGrad would take away
                ~40% of that term's gradient magnitude on average) and skip_suppressed (fraction of
                steps a "skip when negative" policy throws away, equal to conflict_rate since all
                of it is removed on each conflicting step). An interpretation guide is printed at
                the bottom.
        """
        pair_keys = [k for k in self.pair_keys if k in self._history]
        if not pair_keys:
            return ""

        lines = [
            "",
            "┌─ Gradient Conflict Summary " + "─" * 62,
            f"│  {len(self._steps)} logged steps",
            "│",
            f"│  {'Pair':<28} {'neg%':>5}  {'mean':>6}  {'neg_mean':>8}  "
            f"{'PCGrad_sup%':>11}  {'skip_sup%':>9}",
            "│  " + "─" * 72,
        ]
        for key in pair_keys:
            vals = self._history[key]
            sup  = self._suppressed[key]
            n    = len(vals)
            neg  = [v for v in vals if v < 0]
            mean_cos     = sum(vals) / n
            neg_rate     = len(neg) / n
            neg_mean     = sum(neg) / len(neg) if neg else 0.0
            # PCGrad: removes |cos| fraction on conflicting steps
            pcgrad_sup   = sum(sup) / n          # expected fraction per step
            # skip-when-negative: removes 100% on conflicting steps
            skip_sup     = neg_rate

            flag = "  ← CONFLICT" if neg_rate > 0.4 else ""
            lines.append(
                f"│  {key:<28} {neg_rate:>4.0%}  {mean_cos:>+6.3f}  "
                f"{neg_mean:>+8.3f}  {pcgrad_sup:>10.1%}  {skip_sup:>8.1%}{flag}"
            )

        # Per-term gradient norms (unweighted). ratio_vs_focal shows each term's
        # raw gradient magnitude relative to the task anchor - multiply by the
        # loss weight to get the effective pull on parameters.
        norm_terms = [n for n in self.term_names if n in self._norms]
        if norm_terms:
            lines += [
                "│",
                f"│  {'Term':<28} {'mean ‖g‖':>10}  {'ratio_vs_focal':>14}",
                "│  " + "─" * 56,
            ]
            focal_mean = (sum(self._norms["focal"]) / len(self._norms["focal"])
                          if self._norms.get("focal") else None)
            for name in norm_terms:
                vals = self._norms[name]
                mean_n = sum(vals) / len(vals)
                ratio = (f"{mean_n / focal_mean:>13.2f}x"
                         if focal_mean else f"{'—':>14}")
                lines.append(f"│  {name:<28} {mean_n:>10.3e}  {ratio}")

        lines += [
            "│",
            "│  PCGrad_sup%  = avg fraction of gradient magnitude PCGrad removes/step",
            "│  skip_sup%    = fraction of steps where skip-when-negative drops the term",
            "│  ratio_vs_focal = unweighted ‖g_term‖/‖g_focal‖; ≪1 → term is a no-op,",
            "│                   ≫1 → term dominates parameter updates at equal weight.",
            "│  Rule of thumb: PCGrad_sup% > 20% or skip_sup% > 40% → consider",
            "│                 lowering feat_alpha or dropping feature loss entirely.",
            "└" + "─" * 79,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def save_csv(self, out_dir: str) -> str:
        """Write step + pairwise cosine similarities to grad_conflict.csv."""
        os.makedirs(out_dir, exist_ok=True)
        path      = os.path.join(out_dir, "grad_conflict.csv")
        pair_keys = [k for k in self.pair_keys if k in self._history]
        norm_keys = [n for n in self.term_names if n in self._norms]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + pair_keys
                            + [f"norm_{n}" for n in norm_keys])
            for i, step in enumerate(self._steps):
                row = [step] + [
                    self._history[k][i] if i < len(self._history[k]) else ""
                    for k in pair_keys
                ] + [
                    self._norms[n][i] if i < len(self._norms[n]) else ""
                    for n in norm_keys
                ]
                writer.writerow(row)
        return path

    def plot(self, out_dir: str) -> str | None:
        """Save a figure with one subplot per pair (scatter + smoothed MA line).

                Returns the path to the saved PNG, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[GradientConflictLogger] matplotlib not available; skipping plot.")
            return None

        pair_keys = [k for k in self.pair_keys if k in self._history]
        if not pair_keys or not self._steps:
            return None

        n_pairs = len(pair_keys)
        ncols   = min(n_pairs, 3)
        nrows   = (n_pairs + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        steps = np.array(self._steps)

        for idx, key in enumerate(pair_keys):
            row_i, col_i = divmod(idx, ncols)
            ax           = axes[row_i][col_i]
            vals         = np.array(self._history[key])

            ax.scatter(steps, vals, s=6, alpha=0.35, color="steelblue", label="raw")

            if len(vals) >= 5:
                window = min(20, max(3, len(vals) // 5))
                kernel = np.ones(window) / window
                smooth = np.convolve(vals, kernel, mode="valid")
                pad    = len(vals) - len(smooth)
                ax.plot(steps[pad:], smooth,
                        color="firebrick", linewidth=1.5, label=f"MA({window})")

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            # Shade the conflict region (below zero)
            ax.fill_between(
                steps, np.minimum(vals, 0.0), 0.0,
                alpha=0.12, color="crimson", label="conflict (< 0)",
            )

            term_a, term_b = key.split("_vs_")
            ax.set_title(f"{term_a}  vs  {term_b}", fontsize=10, fontweight="bold")
            ax.set_xlabel("global step", fontsize=8)
            ax.set_ylabel("cosine similarity", fontsize=8)
            ax.set_ylim(-1.12, 1.12)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=7)

        for idx in range(len(pair_keys), nrows * ncols):
            row_i, col_i = divmod(idx, ncols)
            axes[row_i][col_i].set_visible(False)

        plt.suptitle(
            "Gradient Conflict Between Loss Terms\n"
            "(positive = cooperative,  negative = conflict)",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "grad_conflict.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Second figure: per-term gradient norms over training (log scale).
        norm_keys = [n for n in self.term_names if n in self._norms]
        if norm_keys:
            fig2, ax = plt.subplots(figsize=(7, 4.5))
            for name in norm_keys:
                vals = np.array(self._norms[name])
                ax.plot(steps[:len(vals)], vals, linewidth=1.2, label=name)
            ax.set_yscale("log")
            ax.set_xlabel("global step", fontsize=9)
            ax.set_ylabel("‖grad‖₂ (unweighted)", fontsize=9)
            ax.set_title("Per-term gradient norms\n"
                         "(multiply by loss weight for effective pull)",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            fig2.tight_layout()
            fig2.savefig(os.path.join(out_dir, "grad_norms.png"),
                         dpi=120, bbox_inches="tight")
            plt.close(fig2)
        return path
