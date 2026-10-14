"""CPU graph legality, dense tensor reduction and checkpoint checks for A0-A3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from models.compression import (
    COMPRESSION_LADDER, build_compression_student, compression_feature_tap,
    compression_graph_record, compression_graph_spec,
    mobilenet_confirmation_declaration,
)
from models.student import build_student


def check_graph(graph_id: str) -> dict:
    torch.manual_seed(42)
    model = build_compression_student(graph_id).eval()
    record = compression_graph_record(model, graph_id)
    spec = compression_graph_spec(graph_id)
    assert all(not isinstance(module, nn.Hardswish) for module in model.modules())
    assert any(isinstance(module, nn.ReLU6) for module in model.modules())
    assert all(not hasattr(module, "weight_mask") for module in model.modules())
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            assert module.in_channels % module.groups == 0
            assert module.out_channels % module.groups == 0
            assert module.weight.shape == (
                module.out_channels, module.in_channels // module.groups,
                *module.kernel_size,
            )
        if type(module).__name__ == "LiteMLA":
            heads = module.qkv.conv.out_channels // (3 * module.dim)
            assert heads >= 1
            assert module.qkv.conv.in_channels % module.dim == 0
            assert module.qkv.conv.out_channels == 3 * heads * module.dim
            assert module.proj.conv.in_channels == 2 * heads * module.dim
            assert module.aggreg[0][1].groups == 3 * heads
    backbone = model.model.backbone
    assert len(backbone.stages) == 4
    assert len(backbone.input_stem.op_list) == 1 + spec["depth_list"][0]
    assert [len(stage.op_list) for stage in backbone.stages] == [
        spec["depth_list"][1], spec["depth_list"][2],
        1 + spec["depth_list"][3], 1 + spec["depth_list"][4],
    ]
    # SegHead must consume all three original levels despite dense width changes.
    assert model.model.head.input_keys == ["stage4", "stage3", "stage2"]
    stage_shapes = {}
    tap = []

    def capture_stages(module, inputs, output):
        stage_shapes.update({key: list(output[key].shape) for key in
                             ("stage0", "stage1", "stage2", "stage3", "stage4")})

    stage_hook = backbone.register_forward_hook(capture_stages)
    tap_hook = compression_feature_tap(model).register_forward_hook(
        lambda module, inputs, output: tap.append(output))
    shapes = {}
    with torch.no_grad():
        for size in (224, 512):
            image = torch.randn(1, 6, size, size)
            output = model(image)
            assert list(output.shape) == [1, 2, size, size]
            assert torch.isfinite(output).all()
            for index, stride in enumerate(spec["stage_output_strides"]):
                assert stage_shapes[f"stage{index}"] == [
                    1, spec["width_list"][index], size // stride, size // stride]
            assert list(tap[-1].shape) == stage_shapes["stage4"]
            shapes[str(size)] = {"logits": list(output.shape),
                                 "encoder_stages": dict(stage_shapes)}
    stage_hook.remove()
    tap_hook.remove()
    image = torch.randn(1, 6, 224, 224)
    with torch.no_grad():
        original_logits = model(image)
    with tempfile.TemporaryDirectory(prefix="flooddistill-compression-graph-") as directory:
        checkpoint = Path(directory) / "student.pt"
        torch.save({"student": model.state_dict(), "graph_id": graph_id}, checkpoint)
        torch.manual_seed(107)
        restored = build_compression_student(graph_id).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored.load_state_dict(state["student"], strict=True)
        assert compression_graph_record(restored, graph_id)["graph_sha256"] == record["graph_sha256"]
        with torch.no_grad():
            assert torch.equal(original_logits, restored(image))
            assert list(restored(torch.randn(1, 6, 512, 512)).shape) == [1, 2, 512, 512]
    # A0 keeps exactly the existing B0 ReLU6 checkpoint tensor interface.
    if graph_id == "A0":
        existing = build_student("efficientvit_b0_relu6", pretrained=False).eval()
        existing.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.equal(original_logits, existing(image))
    if graph_id == "A3":
        model.train()
        images = torch.randn(2, 6, 224, 224)
        masks = torch.randint(0, 2, (2, 224, 224))
        logits = model(images)
        loss = F.cross_entropy(logits, masks)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        record["finite_gradient_loss"] = float(loss.detach())
    record.pop("structure")
    return {"status": "passed", "graph": record, "shapes": shapes,
            "checks": ["all convolution/attention group divisions",
                       "mandatory stride transitions and three decoder levels",
                       "six-band 224/512 input and full-resolution two-class logits",
                       "last retained encoder hook before decoder",
                       "strict checkpoint reload and bitwise logits",
                       "actual dense tensors without pruning masks",
                       "ReLU6 activation family and retained ReLU attention kernel"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    result = {"status": "passed", "device": "cpu", "torch": torch.__version__,
              "seed": 42, "graphs": {},
              "second_family_declaration": mobilenet_confirmation_declaration()}
    for graph_id in COMPRESSION_LADDER:
        result["graphs"][graph_id] = check_graph(graph_id)
    records = [result["graphs"][graph_id]["graph"] for graph_id in COMPRESSION_LADDER]
    counts = [record["main_parameters"] for record in records]
    assert all(left > right for left, right in zip(counts, counts[1:])), counts
    hashes = [record["graph_sha256"] for record in records]
    assert len(set(hashes)) == len(hashes), "Fixed recipes produced duplicate actual graphs"
    result["dense_parameter_counts"] = dict(zip(COMPRESSION_LADDER, counts))
    result["distinct_graph_count"] = len(set(hashes))
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out.resolve()),
                      "dense_parameter_counts": result["dense_parameter_counts"],
                      "distinct_graph_count": result["distinct_graph_count"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
