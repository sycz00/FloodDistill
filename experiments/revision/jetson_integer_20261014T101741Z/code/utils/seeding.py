"""Seed Python, NumPy and PyTorch RNGs.

seed_basic also enables cuDNN determinism and disables benchmarking.
seed_full requests deterministic algorithms with warn_only=True and sets the
cuBLAS workspace option. Warnings still allow nondeterministic operations;
neither mode guarantees identical results across hardware or library versions.
Keep the seed and --deterministic setting of the run being reproduced.
"""

from __future__ import annotations

import os
import random


def seed_basic(seed: int | None) -> None:
    """Seed RNGs and configure cuDNN determinism. None leaves the state unchanged."""
    if seed is None:
        return

    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic cuDNN algorithms, no convolution auto-tuner at startup.
    # First epoch is a bit slower; same speed afterwards. This changes cuDNN's
    # internal choices, not the top-level CUDA op.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"[seed] seed={seed}  (cudnn.deterministic=True, use_deterministic_algorithms=False)")


def seed_full(seed: int | None) -> None:
    """seed_basic() + torch.use_deterministic_algorithms(warn_only=True).

        Goes for bit-exact reproducibility, possibly with slower kernel fallbacks. Some CUDA ops
        have no deterministic implementation and only emit a warning. seed=None does nothing.
    """
    if seed is None:
        return

    import torch

    seed_basic(seed)

    # Required by some deterministic CUDA kernels (e.g. cublasGemmEx)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # warn_only=True: emits UserWarning instead of RuntimeError for ops that
    # have no deterministic CUDA kernel. Change to False for strict auditing.
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[seed] seed={seed}  (full determinism: use_deterministic_algorithms=True, warn_only=True)")


# Old name, kept so existing scripts still work. Same as seed_basic (the old version already
# had warn_only set).
seed_everything = seed_basic
