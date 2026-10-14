"""Model profiling: FLOPs, parameter memory and latency.

FLOPs are counted with forward hooks on nn.Conv2d and nn.Linear, the dominant ops for both
ViT (Linear projections, MLP) and CNN (Conv2d) models. FLOPs = 2 x MACs (one multiply + one
accumulate per MAC).

Profile float models only. In quantized GraphModules (PT2E),
nn.Conv2d / nn.Linear are replaced with int8 ops. These are not nn.Module subclasses,
so the hooks undercount them or miss them entirely. Call measure_flops() on the float model before quantization.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def measure_flops(model: nn.Module, example_input: torch.Tensor) -> int:
    """FLOPs of a single forward pass, counted with forward hooks.

        Counts nn.Conv2d (depthwise with groups > 1 and strided included) and nn.Linear
        (transformer Q/K/V/O projections and MLP layers, 2-D (B, in) or 3-D (B, seq, in)
        inputs). BatchNorm, LayerNorm, ReLU, interpolate and softmax aren't counted, they are
        negligible next to Conv/Linear in typical seg models.

        model must be in eval mode on the same device as example_input, a single (B, C, H, W)
        batch. Returns total FLOPs as int, divide by 1e9 for GFLOPs:

            flops = measure_flops(model, torch.randn(1, 6, 224, 224, device=device))
            print(f"{flops / 1e9:.2f} GFLOPs")
    """
    flops = [0]
    hooks = []

    def _conv_hook(module: nn.Conv2d, inp, out) -> None:
        x = inp[0]
        N = x.shape[0]
        C_out, H_out, W_out = out.shape[1], out.shape[2], out.shape[3]
        # MACs per output element = (C_in / groups) * K_h * K_w
        kernel_macs = (
            (module.in_channels // module.groups)
            * module.kernel_size[0]
            * module.kernel_size[1]
        )
        flops[0] += 2 * N * C_out * H_out * W_out * kernel_macs

    def _linear_hook(module: nn.Linear, inp, out) -> None:
        x = inp[0]
        # x is (B, in) for flat layers or (B, seq, in) for transformer projections
        batch_elements = x.numel() // x.shape[-1]
        flops[0] += 2 * batch_elements * module.in_features * module.out_features

    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(_conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_linear_hook))

    with torch.no_grad():
        model(example_input)

    for h in hooks:
        h.remove()

    return flops[0]


def measure_param_memory_mb(model: nn.Module) -> float:
    """Parameter memory in MB, respecting actual dtype (int8=1B, fp16=2B, fp32=4B).

        Also right for bitsandbytes Linear8bitLt after .cuda(), the weight tensor is
        stored as int8 there so element_size() == 1.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024


@torch.no_grad()
def measure_latency_gpu(
    model: nn.Module,
    loader: DataLoader,
    n_runs: int,
    device: torch.device,
) -> float:
    """Mean per-sample GPU latency in ms, measured with CUDA Events.

        CUDA Events take timestamps on the GPU timeline, so CPU-GPU sync overhead and OS
        scheduler jitter don't affect them. The first 5 batches are warm-up and get discarded
        (fills the CUDA caches and, for bnb_int8, finishes the activation-scale init). n_runs
        counts those 5 too, so the mean is over n_runs - 5 batches. model in eval mode on
        `device`, loader yields dicts with "image".
    """
    model.eval()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    times: list[float] = []

    for i, batch in enumerate(loader):
        if i >= n_runs:
            break
        images = batch["image"].to(device).float()
        if i < 5:
            model(images)
            torch.cuda.synchronize(device)
            continue
        start_evt.record()
        model(images)
        end_evt.record()
        torch.cuda.synchronize(device)
        times.append(start_evt.elapsed_time(end_evt))  # ms already

    return float(torch.tensor(times).mean()) if times else 0.0


@torch.no_grad()
def measure_latency_cpu(
    model: nn.Module,
    loader: DataLoader,
    n_runs: int,
) -> float:
    """Mean per-sample CPU latency in milliseconds using time.perf_counter.

        First 5 batches are discarded as warm-up.
    """
    model.eval()
    times: list[float] = []

    for i, batch in enumerate(loader):
        if i >= n_runs:
            break
        images = batch["image"].cpu().float()
        if i < 5:
            model(images)
            continue
        t0 = time.perf_counter()
        model(images)
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(torch.tensor(times).mean()) if times else 0.0
