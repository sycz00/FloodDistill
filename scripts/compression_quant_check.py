"""CPU checks for fixed-bit STE, state isolation, BN freeze and strict resume."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from models.compression_quant import (
    FixedQATConv2d, prepare_compression_qat, rebind_compression_bits,
    quantization_mode, compression_train_mode, load_compression_qat,
    strip_compression_qat, precision_map,
)
from quantization.qat import prepare_qat


class ToySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(6, 8, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(8), nn.ReLU6())
        self.mix = nn.Sequential(nn.Conv2d(8, 8, 1), nn.ReLU6(), nn.Dropout2d(.1))
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.head(self.mix(self.stem(x)))


def check():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = ToySeg().eval()
    image = torch.randn(2, 6, 16, 16)
    batches = [{"image": image}, {"image": image * .8}]
    before_rng = torch.get_rng_state().clone()
    q8 = prepare_compression_qat(model, batches, calibration_seed=77)
    assert torch.equal(before_rng, torch.get_rng_state()), "Calibration changed caller RNG"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(77)
        legacy = prepare_qat(model, batches, skip_last_n=1, calibration_type="per_tensor")
    assert torch.equal(q8(image), legacy(image)), "W8 changed the historical quantizer"
    with quantization_mode(q8, False), torch.no_grad():
        assert torch.equal(q8(image), model(image)), "FP bypass changed the common graph"
    q4 = rebind_compression_bits(q8, 4, 8)
    q44 = rebind_compression_bits(q8, 4, 4)
    for original, low, stress in zip(q8.modules(), q4.modules(), q44.modules()):
        if isinstance(original, FixedQATConv2d):
            assert torch.equal(original.weight, low.weight)
            assert torch.allclose(low._w_scale, original._w_scale * (127 / 7))
            assert torch.equal(low._act_scale, original._act_scale)
            assert torch.allclose(stress._act_scale, original._act_scale * (127 / 7))
            assert not original._act_scale.requires_grad and not original._w_scale.requires_grad
    assert type(q4.head) is nn.Conv2d, "Final two-class logits must stay floating"
    compression_train_mode(q4)
    q4.mix[0].fake_quant_enabled = False
    modes = [m.training for m in q4.modules()]
    flags = [m.fake_quant_enabled for m in q4.modules() if isinstance(m, FixedQATConv2d)]
    class ObserverStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.int))
    q4.observer_stub = ObserverStub()
    try:
        with quantization_mode(q4, True):
            assert all(not m.training for m in q4.modules())
            assert int(q4.observer_stub.observer_enabled) == 0
            with quantization_mode(q4, False):
                assert not any(m.fake_quant_enabled for m in q4.modules()
                               if isinstance(m, FixedQATConv2d))
            assert all(m.fake_quant_enabled for m in q4.modules()
                       if isinstance(m, FixedQATConv2d))
            raise RuntimeError("intentional isolation check")
    except RuntimeError as exc:
        assert str(exc) == "intentional isolation check"
    assert modes == [m.training for m in list(q4.modules())[:-1]], "Mixed modes not restored"
    assert flags == [m.fake_quant_enabled for m in q4.modules() if isinstance(m, FixedQATConv2d)]
    assert int(q4.observer_stub.observer_enabled) == 1
    del q4.observer_stub

    # Check the defined STE surrogate, not the derivative of true rounding.
    conv = nn.Conv2d(1, 1, 1, bias=False)
    conv.weight.data.fill_(.5)
    ste = FixedQATConv2d(conv, torch.tensor([.25]), torch.tensor([0]),
                        torch.tensor([.1]), weight_bits=4, activation_bits=4)
    x = torch.tensor([[[[-3., -.5, 0., .5, 3.]]]], requires_grad=True)
    ste(x).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([[[[0., .5, .5, .5, 0.]]]]))
    assert torch.isfinite(ste.weight.grad).all() and ste.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(ste.parameters(), lr=.02)
    xfit = torch.tensor([[[[.25, .5, 1.]]]])
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = (ste(xfit) - .2 * xfit).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], "Defined-STE smoke loss did not decrease"

    compression_train_mode(q4)
    bn = q4.stem[1]
    running = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
    opt = torch.optim.AdamW([p for p in q4.parameters() if p.requires_grad], lr=1e-5)
    opt.zero_grad()
    loss = q4(image).square().mean()
    loss.backward()
    assert bn.weight.grad is None and bn.bias.grad is None
    assert all(torch.equal(a, b) for a, b in zip(running,
               (bn.running_mean, bn.running_var, bn.num_batches_tracked)))
    opt.step()
    # Serialize complete quantizer state and resume a matched stochastic step.
    buf = io.BytesIO()
    torch.save({"model": q4.state_dict(), "optimizer": opt.state_dict()}, buf)
    buf.seek(0)
    saved = torch.load(buf, map_location="cpu", weights_only=True)
    resumed = load_compression_qat(model, saved["model"])
    opt_r = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-5)
    opt_r.load_state_dict(saved["optimizer"])
    rng = torch.get_rng_state().clone()
    for net, optim in ((q4, opt), (resumed, opt_r)):
        torch.set_rng_state(rng)
        compression_train_mode(net)
        optim.zero_grad()
        net(image).square().mean().backward()
        optim.step()
    for name, value in q4.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), f"Resume diverged at {name}"
    other = copy.deepcopy(q4)
    other.stem[0].weight.data.add_(.01)
    assert not torch.equal(other.stem[0].weight, q4.stem[0].weight), "Methods shared mutable weights"
    stripped = strip_compression_qat(q4).eval()
    with quantization_mode(q4, False), torch.no_grad():
        assert torch.equal(q4(image), stripped(image))
    assert not any(isinstance(m, FixedQATConv2d) for m in stripped.modules())
    mapping = precision_map(q4)
    assert mapping["status"] == "simulation_only"
    assert all(op.get("arithmetic") == "float32" for op in mapping["operators"])
    result = dict(status="passed", device="cpu", checks=[
        "historical_w8_bitwise_equivalence", "rng_isolation", "shared_master_fp_bypass",
        "fixed_ranges_rebind", "floating_final_logits", "nested_exception_mode_restore",
        "observer_isolation", "defined_ste_finite_gradient", "decreasing_loss_smoke",
        "frozen_bn_running_and_affine", "strict_quantizer_checkpoint_reload",
        "stochastic_optimizer_resume", "independent_continuation_weights",
        "quantizers_removed_from_float_export", "honest_precision_map"],
        loss_smoke={"initial": losses[0], "final": losses[-1]})
    result["actual_graphs"] = check_actual_graphs()
    result["checks"].extend(["scaled_graph_litemla_strict_reload",
                            "historical_relu6_reference_strict_graph_compatibility"])
    return result


def check_actual_graphs():
    from models.compression import build_compression_student
    rows = []
    image = torch.randn(1, 6, 224, 224)
    for graph in ("A1", "A3"):
        model = build_compression_student(graph).eval()
        q = prepare_compression_qat(model, [image], weight_bits=4, activation_bits=8,
                                    calibration_seed=77)
        with torch.no_grad():
            fp = model(image)
            with quantization_mode(q, False):
                bypass = q(image)
            y = q(image)
            resumed = load_compression_qat(model, q.state_dict()).eval()
            y_resumed = resumed(image)
            stress = rebind_compression_bits(q, 4, 4)
            y_stress = stress(image)
        assert torch.equal(y, y_resumed), f"{graph} strict quantizer reload changed predictions"
        assert torch.allclose(fp, bypass, rtol=1e-6, atol=1e-6)
        assert y.shape == (1, 2, 224, 224)
        assert torch.isfinite(y).all() and torch.isfinite(y_stress).all()
        compression_train_mode(q)
        q(image).square().mean().backward()
        gradients = [p.grad for p in q.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        rows.append({"graph": graph, "input_shape": [1, 6, 224, 224],
                     "output_shape": list(y.shape), "w4a8_finite": True,
                     "w4a4_finite": True, "ste_gradients_finite": True,
                     "fp_bypass_max_difference": float((fp - bypass).abs().max()),
                     "strict_reload_max_difference": float((y - y_resumed).abs().max()),
                     "litemla_wrappers": sum(type(m).__name__ == "LiteMLAQAT" for m in q.modules())})
    checkpoint = Path("/data1/schmalstieg/flooddistill/runs_repro/pseudo_evit_relu6_ofa_N2500_s42/best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError("Expected historical ReLU6 provenance reference is unavailable")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    expected = "d2a6c75ae464ebf5198b9bd4ac4bb0e9937c5676fd0e8a73b09697081d8b078b"
    assert digest.hexdigest() == expected, "Historical ReLU6 reference changed"
    historical = torch.load(checkpoint, map_location="cpu", weights_only=True)
    a0 = build_compression_student("A0").eval()
    a0.load_state_dict(historical["model"], strict=True)
    with torch.no_grad():
        output = a0(image)
    assert output.shape == (1, 2, 224, 224) and torch.isfinite(output).all()
    return {"quantized_graph_checks": rows, "historical_reference": {
        "path": str(checkpoint), "sha256": expected, "epoch": int(historical["epoch"]),
        "a0_strict_graph_compatibility": True,
        "use": "historical_unmatched_reference_only",
        "limitation": "checkpoint lacks training config and initialization lineage; filename does not establish matched Logit-KD recipe"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
