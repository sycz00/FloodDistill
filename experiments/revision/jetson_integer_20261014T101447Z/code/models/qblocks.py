"""Convolutional blocks and small backbones for quantization experiments.

QConvBnReLU6, depthwise-separable blocks and MBConv blocks feed the micro
students. MicroBackbone returns four spatial feature maps. Quantization
reports are operator heuristics; they do not verify an engine's accuracy,
latency or DLA coverage. The paper's deployment is measured on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn



# Quantization metadata


@dataclass
class QuantMeta:
    """Operator-level quantization hints; these are not measured deployment results."""
    tier: str
    trt_int8: bool
    dla_ok: bool
    notes: str = ""



# Easy-tier building blocks (all DLA-compatible)


class QConvBnReLU6(nn.Sequential):
    """Conv2d, BatchNorm2d and ReLU6. Padding defaults to kernel//2."""
    quant_meta = QuantMeta("easy", trt_int8=True, dla_ok=True)

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        padding: int | None = None,
    ) -> None:
        pad = kernel // 2 if padding is None else padding
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class QDWSepBlock(nn.Sequential):
    """Depthwise 3x3 then pointwise 1x1, each with BN/ReLU6. Stride applies to the depthwise convolution."""
    quant_meta = QuantMeta("easy", trt_int8=True, dla_ok=True)

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            QConvBnReLU6(in_ch, in_ch,  3, stride=stride, groups=in_ch),  # DW
            QConvBnReLU6(in_ch, out_ch, 1),                                # PW
        )


class QMBConvBlock(nn.Module):
    """Expand, depthwise convolution and linear projection with a residual when shape permits."""
    quant_meta = QuantMeta(
        "easy", trt_int8=True, dla_ok=True,
        notes="residual add branch requires symmetric INT8 — handled by TRT automatically",
    )

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: int = 4,
    ) -> None:
        super().__init__()
        mid = in_ch * expand_ratio
        self.conv = nn.Sequential(
            QConvBnReLU6(in_ch, mid,  1),                              # expand
            QConvBnReLU6(mid,   mid,  3, stride=stride, groups=mid),   # DW
            nn.Conv2d(mid, out_ch, 1, bias=False),                     # project
            nn.BatchNorm2d(out_ch),                                    # no ReLU6 here
        )
        self.use_res: bool = (stride == 1 and in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out + x if self.use_res else out



# MicroBackbone - configurable pure-Q-easy backbone


_BLOCK_MAP: dict[str, type] = {
    "dw": QDWSepBlock,
    "mb": QMBConvBlock,
}

MICRO_CONFIGS: dict[str, dict] = {
    "micro_s": dict(
        stem_ch=16,
        stages=[
            # (out_channels, n_blocks, block_type)
            (32,  1, "dw"),   # H/4, 32ch
            (64,  2, "dw"),   # H/8, 64ch
            (96,  2, "dw"),   # H/16, 96ch
            (128, 2, "dw"),   # H/32, 128ch
        ],
        decoder_channels=32,
    ),
    "micro_m": dict(
        stem_ch=24,
        stages=[
            (48,  2, "mb"),   # H/4, 48ch
            (96,  3, "mb"),   # H/8, 96ch
            (128, 3, "mb"),   # H/16, 128ch
            (160, 2, "mb"),   # H/32, 160ch
        ],
        decoder_channels=64,
    ),
}


class MicroBackbone(nn.Module):
    """Configurable pure-Q-easy backbone for dense prediction.

        The stem is QConvBnReLU6(in_channels, stem_ch, 3x3, stride=2), giving H/2. Each stage
        is n_blocks of QDWSepBlock or QMBConvBlock, the first with stride=2 and the rest with
        stride=1, so the skip features come out at H/4, H/8, H/16 and H/32.

        Returns the 4 spatial feature maps [s4, s8, s16, s32] (finest first), which is what
        UNetDecoder in models/student.py expects. _stage_features_type = "spatial_list" for
        the DistillLoss hooks, same convention as TinyViTSegModel.
    """

    _stage_features_type = "spatial_list"  # hook path in training/losses.py

    def __init__(self, in_channels: int, config: dict) -> None:
        super().__init__()
        stem_ch = config["stem_ch"]
        stages  = config["stages"]   # list of (out_ch, n_blocks, block_type)

        self.stem = QConvBnReLU6(in_channels, stem_ch, 3, stride=2)

        self._stages: nn.ModuleList = nn.ModuleList()
        in_ch = stem_ch
        self.out_channels: list[int] = []

        for out_ch, n_blocks, block_type in stages:
            block_cls = _BLOCK_MAP[block_type]
            blocks: list[nn.Module] = []
            for i in range(n_blocks):
                stride = 2 if i == 0 else 1
                blocks.append(block_cls(in_ch, out_ch, stride=stride))
                in_ch = out_ch
            self._stages.append(nn.Sequential(*blocks))
            self.out_channels.append(out_ch)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        _ = self.stem(x)     # H/2 - used internally but not a skip target
        feats: list[torch.Tensor] = []
        h = _
        for stage in self._stages:
            h = stage(h)
            feats.append(h)  # [H/4, H/8, H/16, H/32]
        return feats



# score_quant - quantization tier reporter for any nn.Module


# Ops that make static INT8 impractical (FP16 required)
_HARD_OPS: tuple[type, ...] = (
    nn.MultiheadAttention,
    nn.LayerNorm,
    nn.GELU,
    nn.Softmax,
)

# Ops that work with per-tensor INT8 calibration but are NOT DLA-compatible
_MEDIUM_OPS: tuple[type, ...] = (
    nn.Hardswish,
    nn.SiLU,
    nn.Sigmoid,
    nn.Hardsigmoid,
    nn.GroupNorm,
)


def score_quant(model: nn.Module) -> dict:
    """Quantization tier report for a model.

        Scans all named submodules for ops that prevent static INT8 quantization or DLA
        deployment, prints a one-line summary and returns a dict with
            "tier"           : "easy" | "medium" | "hard"
            "trt_int8"       : bool, TRT INT8 static likely works
            "dla_compatible" : bool, all ops in the Jetson Xavier DLA op set
            "issues"         : list[(name, op_type_str, tier_str)] for non-easy ops
            "n_hard"         : int, number of hard ops
            "n_medium"       : int, number of medium ops
    """
    issues: list[tuple[str, str, str]] = []

    for name, mod in model.named_modules():
        if isinstance(mod, _HARD_OPS):
            issues.append((name, type(mod).__name__, "hard"))
        elif isinstance(mod, _MEDIUM_OPS):
            issues.append((name, type(mod).__name__, "medium"))

    n_hard   = sum(1 for *_, t in issues if t == "hard")
    n_medium = sum(1 for *_, t in issues if t == "medium")

    if n_hard > 0:
        tier = "hard"
    elif n_medium > 0:
        tier = "medium"
    else:
        tier = "easy"

    trt_int8       = tier in ("easy", "medium")
    dla_compatible = tier == "easy"

    return {
        "tier":           tier,
        "trt_int8":       trt_int8,
        "dla_compatible": dla_compatible,
        "issues":         issues,
        "n_hard":         n_hard,
        "n_medium":       n_medium,
    }


def print_quant_summary(arch: str, model: nn.Module) -> None:
    """Print a one-line quantization summary. Called by build_student() after each build."""
    report = score_quant(model)
    tier   = report["tier"]

    if tier == "easy":
        detail = "DLA-compatible"
    elif tier == "medium":
        ops = {op for _, op, _ in report["issues"]}
        hint = f"{', '.join(sorted(ops))} ×{report['n_medium']}"
        relu6_hint = f" — use {arch}_relu6 for DLA" if "relu6" not in arch else ""
        detail = f"{hint}{relu6_hint}"
    else:
        ops = {op for _, op, _ in report["issues"]}
        hint = f"{', '.join(sorted(ops))} ×{report['n_hard']}"
        detail = f"{hint} — FP16 recommended"

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[student] {arch:<26}  params={n_params:.2f}M  "
        f"quant={tier:<6}  ({detail})"
    )
