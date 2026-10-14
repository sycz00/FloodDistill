"""Fixed dense ReLU6 EfficientViT-B0 graphs for the compression pilot.

These declarations use the existing EfficientViT backbone/head and full-resolution
wrapper. Channels and repeat retention are settled by graph constraints, without
accuracy-based search. No pretrained weights exist for the scaled graphs.
"""

from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn

from models.student import _FullResWrapper, build_student


# Ratios are declared together so later intermediate graphs cannot be selected
# after inspecting accuracy. Eligible repeats exclude every stride transition.
COMPRESSION_LADDER = {
    "A0": (1.00, 1.00),
    "A1": (0.50, 1.00),
    "A2": (0.35, 0.75),
    "A3": (0.25, 0.50),
}
_B0_WIDTHS = (8, 16, 32, 64, 128)
_ELIGIBLE_REPEATS = (1, 1, 1, 2, 2)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def compression_graph_spec(graph_id: str) -> dict:
    """Return a fresh JSON-compatible declaration of the complete legal graph."""
    graph_id = graph_id.upper()
    if graph_id == "M0":
        return mobilenet_confirmation_declaration()
    if graph_id not in COMPRESSION_LADDER:
        raise ValueError(f"Unknown compression graph {graph_id!r}; use {[*COMPRESSION_LADDER, 'M0']}")
    width, depth = COMPRESSION_LADDER[graph_id]
    dim = max(4, _round_half_up(16 * width))
    channels = [max(4, _round_half_up(c * width)) for c in _B0_WIDTHS]
    # LiteMLA defaults to heads=in_channels//dim. Full head groups prevent an
    # implicit smaller qkv bottleneck introduced solely by integer truncation.
    for index in (3, 4):
        channels[index] = max(dim, math.ceil(channels[index] / dim) * dim)
    retained = [math.floor(depth * n) for n in _ELIGIBLE_REPEATS]
    # EfficientViT's depth semantics differ across stages: early-stage d includes
    # the mandatory transition; attention-stage d excludes that transition.
    depths = [retained[0], 1 + retained[1], 1 + retained[2], retained[3], retained[4]]
    return {
        "graph_id": graph_id,
        "family": "efficientvit_b0",
        "width_multiplier": width,
        "repeatable_depth_multiplier": depth,
        "initialization": "random_from_scratch_no_pretrained_transfer",
        "activation": "relu6",
        "attention_kernel_activation": "relu",
        "channel_rounding": "half_up_integer_min4; attention_stages_ceil_to_head_dim",
        "width_list": channels,
        "depth_list": depths,
        "attention_dim": dim,
        "attention_heads": [channels[3] // dim, channels[4] // dim],
        "repeat_retention_rule": "prefix_floor_multiplier_times_eligible_count",
        "eligible_repeat_counts": list(_ELIGIBLE_REPEATS),
        "retained_original_block_indices": [[0] + list(range(1, n + 1)) for n in retained],
        "mandatory_original_block_indices": [[0]] * 5,
        "stage_output_strides": [2, 4, 8, 16, 32],
        "head": {
            "input_features": ["stage4", "stage3", "stage2"],
            "input_strides": [32, 16, 8],
            "stride": 8,
            "width": max(8, _round_half_up(32 * width)),
            "depth": 1,
            "expand_ratio": 4,
            "final_expand": 4,
        },
        "in_channels": 6,
        "num_classes": 2,
        "output_geometry": "input_H_W_bilinear_align_corners_false",
        "student_feature_tap": {
            "module": "model.backbone.stages.3",
            "semantic_output": "stage4",
            "output_kind": "tensor",
            "channels": channels[4],
            "stride": 32,
        },
        "teacher_feature_tap": {
            "module": "neck",
            "output_index": -1,
            "semantic_output": "last_selected_transformer_feature_before_decoder",
        },
    }


def build_compression_student(
    graph_id: str = "A1", *, in_channels: int = 6, num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    """Construct actual smaller tensors, using the existing segmentation modules."""
    if pretrained:
        raise ValueError("Compression graphs use a common random initialization policy; pretrained=True is unsupported")
    if in_channels != 6 or num_classes != 2:
        raise ValueError("The fixed compression protocol requires six input bands and two output classes")
    if graph_id.upper() == "M0":
        return build_student("mobilenet_v3_small", pretrained=False,
                             in_channels=in_channels, num_classes=num_classes,
                             head_dropout=.1, decoder_channels=128)
    from efficientvit.models.efficientvit.backbone import EfficientViTBackbone
    from efficientvit.models.efficientvit.seg import EfficientViTSeg, SegHead

    spec = compression_graph_spec(graph_id)
    backbone = EfficientViTBackbone(
        width_list=spec["width_list"], depth_list=spec["depth_list"],
        in_channels=in_channels, dim=spec["attention_dim"], expand_ratio=4,
        norm="bn2d", act_func="relu6",
    )
    head_spec = spec["head"]
    head = SegHead(
        fid_list=head_spec["input_features"],
        in_channel_list=[spec["width_list"][i] for i in (4, 3, 2)],
        stride_list=head_spec["input_strides"], head_stride=head_spec["stride"],
        head_width=head_spec["width"], head_depth=head_spec["depth"],
        expand_ratio=head_spec["expand_ratio"], middle_op="mbconv",
        final_expand=head_spec["final_expand"], n_classes=num_classes,
        norm="bn2d", act_func="relu6",
    )
    return _FullResWrapper(EfficientViTSeg(backbone, head))


def compression_graph_record(model: nn.Module, graph_id: str) -> dict:
    """Record real tensors/operator attributes and a weight-independent graph hash."""
    tensors = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
               for name, value in model.state_dict().items()]
    modules = []
    for name, module in model.named_modules():
        row = {"name": name, "type": f"{type(module).__module__}.{type(module).__qualname__}"}
        if isinstance(module, nn.Conv2d):
            row.update(in_channels=module.in_channels, out_channels=module.out_channels,
                       kernel_size=list(module.kernel_size), stride=list(module.stride),
                       padding=list(module.padding), dilation=list(module.dilation),
                       groups=module.groups, bias=module.bias is not None)
        elif isinstance(module, nn.BatchNorm2d):
            row.update(num_features=module.num_features, eps=module.eps,
                       momentum=module.momentum, affine=module.affine,
                       track_running_stats=module.track_running_stats)
        # The custom modules expose interpolation/attention behavior in their
        # repr; plain module repr includes child text, so capture leaves only.
        if not list(module.children()):
            row["operator_repr"] = repr(module)
        modules.append(row)
    structure = {"state_tensors": tensors, "operators": modules}
    encoded = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
    record = compression_graph_spec(graph_id)
    record.update(
        graph_sha256=hashlib.sha256(encoded).hexdigest(),
        main_parameters=sum(p.numel() for p in model.parameters()),
        auxiliary_training_parameters=0,
        state_tensor_elements=sum(v.numel() for v in model.state_dict().values()),
        state_tensor_bytes=sum(v.numel() * v.element_size() for v in model.state_dict().values()),
        structure=structure,
    )
    return record


def compression_feature_tap(model: nn.Module) -> nn.Module:
    """Tap the last encoder output; MobileNet's list hook selects output[-1]."""
    if hasattr(model, "encoder"):
        # The integrated MobileNet encoder returns all four scales; consumers
        # select output[-1], the 576-channel output immediately before decoder.
        return model.encoder
    return model.model.backbone.stages[3]


def mobilenet_confirmation_declaration() -> dict:
    """Use the integrated complete second family; no second architecture search."""
    return {
        "graph_id": "M0",
        "family": "mobilenet_v3_small",
        "architecture": "mobilenet_v3_small",
        "scaling": "existing_complete_graph_no_new_width_or_depth_sweep",
        "width_multiplier": 1.,
        "repeatable_depth_multiplier": 1.,
        "width_list": [16, 24, 48, 576],
        "stage_output_strides": [4, 8, 16, 32],
        "head": {"type": "existing_UNetDecoder", "decoder_channels": 128,
                 "input_channels": [16, 24, 48, 576], "retained_decoder_levels": 3,
                 "head_dropout": .1, "num_classes": 2},
        "in_channels": 6,
        "num_classes": 2,
        "output_geometry": "input_H_W_bilinear_align_corners_false",
        "student_feature_tap": {"module": "encoder", "output_index": -1,
                                "channels": 576, "stride": 32},
        "teacher_feature_tap": {"module": "neck", "output_index": -1},
        "initialization": "random_from_scratch_no_pretrained_transfer",
        "activation": "native_MobileNetV3_Hardswish_and_ReLU; existing_UNetDecoder_ReLU6",
        "activation_conversion": False,
        "relu6_family_claim": False,
    }
