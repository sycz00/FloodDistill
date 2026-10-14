"""CPU numerical gates for the compression interaction prototype."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from distillation.compression_metric import (
    FrozenFeatureAdapter, FrozenTeacherTransform, accuracy_damage_repair,
    aggregate_moments, balanced_position_weights, beta_from_gradient_norms,
    bounded_interaction_metric, cluster_bootstrap_radius, feature_loss,
    fit_interaction_metric, fit_ridge_adapter, fit_teacher_pca, gradient_rms,
    initial_loss_scale, matched_spectrum_controls, per_image_moments,
    probe_features, residual_diagnostics, split_group_stability,
)


def close(actual, expected, atol=1e-10, rtol=1e-10):
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    if actual.dtype != expected.dtype:
        actual, expected = actual.double(), expected.double()
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"numerical mismatch: {actual} vs {expected}")


def run_checks() -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dtype = torch.float64
    checks = []

    labels = torch.tensor([[0, 0, 0, 1], [1, 1, 1, 1], [-1, -1, -1, -1]])
    weights = balanced_position_weights(labels)
    close(weights[0], torch.tensor([1/6, 1/6, 1/6, .5], dtype=dtype))
    close(weights[1], torch.full((4,), .25, dtype=dtype))
    close(weights[2], torch.zeros(4, dtype=dtype))
    features = torch.arange(2 * 3 * 16 * 16, dtype=dtype).reshape(2, 3, 16, 16)
    targets = torch.zeros(2, 16, 16, dtype=torch.long)
    targets[:, :, 8:] = 1
    vectors, probe_weights = probe_features(features, targets)
    expected = torch.nn.functional.interpolate(features, size=(8, 8), mode="bilinear", align_corners=False)
    close(vectors, expected.permute(0, 2, 3, 1).reshape(2, 64, 3))
    close(probe_weights.sum(1), torch.ones(2, dtype=dtype))
    assert vectors.shape == (2, 64, 3)
    center_support = torch.zeros_like(targets, dtype=torch.bool)
    center_support[:, 1::2, 1::2] = True
    _, center_weights = probe_features(features, targets, center_support)
    close(center_weights.sum(1), torch.ones(2, dtype=dtype))
    checks.append("equal_class_then_image_weights_and_fixed_grid")

    labels = torch.randint(0, 2, (40, 64))
    weights = balanced_position_weights(labels)
    latent = torch.randn(40, 64, 3, dtype=dtype)
    teacher_raw = torch.cat([latent, latent[:, :, :1] + 2*latent[:, :, 1:2],
                             torch.ones(40, 64, 1, dtype=dtype)], -1)
    transform, pca_info = fit_teacher_pca(teacher_raw, weights, min_student_dim=3)
    standardized = transform(teacher_raw)
    close(torch.einsum("bp,bpc->c", weights, standardized) / 40, torch.zeros(3, dtype=dtype))
    close(torch.einsum("bp,bpc,bpd->cd", weights, standardized, standardized) / 40,
          torch.eye(3, dtype=dtype))
    assert pca_info["rank"] == 3 and pca_info["numerical_rank"] == 3
    assert list(transform.parameters()) == []
    student = torch.randn(40, 64, 5, dtype=dtype)
    true_weight = torch.randn(5, 3, dtype=dtype)
    true_bias = torch.randn(3, dtype=dtype)
    teacher = student @ true_weight + true_bias
    adapter, adapter_info = fit_ridge_adapter(student, teacher, weights, relative_ridge=0.)
    close(adapter.weight, true_weight)
    close(adapter.bias, true_bias)
    assert list(adapter.parameters()) == [] and adapter_info["bias_penalized"] is False
    degenerate, info = fit_ridge_adapter(torch.ones_like(student), teacher, weights)
    assert info["degenerate_student_gram"] and torch.count_nonzero(degenerate.weight) == 0
    close(degenerate.bias, torch.einsum("bp,bpc->c", weights, teacher) / 40)
    checks.append("weighted_pca_whitening_and_frozen_ridge_unpenalized_bias")

    f = torch.randn(40, 64, 8, dtype=dtype)
    q = torch.randn_like(f) * torch.linspace(.1, 1., 8, dtype=dtype)
    per_image = per_image_moments(f, q, weights)
    moments = aggregate_moments(per_image)
    close(per_image["Ce"], per_image["Cf"] + per_image["Cq"] + 2*per_image["R"])
    close(moments["Ce"], moments["C"] + 2*moments["R"])
    alternating = torch.tensor([[[1.], [-1.]]], dtype=dtype)
    pos_weights = torch.tensor([[.5, .5]], dtype=dtype)
    alternating_moments = aggregate_moments(per_image_moments(alternating, alternating, pos_weights))
    close(alternating_moments["Cf"], torch.ones(1, 1, dtype=dtype))
    assert float(alternating.mean()) == 0
    checks.append("uncentered_per_position_moments_not_spatial_mean_outer_product")

    interaction = torch.diag(torch.tensor([2., .4, .05, -.2], dtype=dtype))
    independent = torch.eye(4, dtype=dtype) * 4
    metric, metric_info = bounded_interaction_metric(interaction, independent, delta=.1)
    close(metric, metric.T)
    eigenvalues = torch.linalg.eigvalsh(metric)
    assert (eigenvalues > 0).all() and float(eigenvalues[-1]/eigenvalues[0]) <= 4 + 1e-12
    close(torch.trace(metric), torch.tensor(4., dtype=dtype))
    assert metric_info["retained_directions"] == 2
    capped, _ = bounded_interaction_metric(torch.diag(torch.tensor([1e8, 0., 0., 0.], dtype=dtype)),
                                           torch.eye(4, dtype=dtype))
    close(torch.linalg.cond(capped), torch.tensor(4., dtype=dtype))
    identity = torch.eye(4, dtype=dtype)
    for r, c, delta, reliable in ((torch.zeros_like(identity), identity, 0., True),
                                   (-identity, identity, 0., True),
                                   (identity, identity, 2., True),
                                   (identity, torch.zeros_like(identity), 0., True),
                                   (interaction, independent, 0., False),
                                   (identity * .2, identity, 0., True)):
        result, _ = bounded_interaction_metric(r, c, delta=delta, reliable_support=reliable)
        assert torch.equal(result, identity)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4, dtype=dtype))
    rotated, _ = bounded_interaction_metric(rotation @ interaction @ rotation.T,
                                            rotation @ independent @ rotation.T, delta=.1)
    close(rotated, rotation @ metric @ rotation.T)
    errors = []
    for scale in (1., 1e-2, 1e-4, 1e-6, 0.):
        result, _ = bounded_interaction_metric(interaction * scale, independent)
        errors.append(float(torch.linalg.matrix_norm(result - identity)))
    assert all(a >= b for a, b in zip(errors, errors[1:])) and errors[-1] == 0
    zero_q = aggregate_moments(per_image_moments(f[:, :, :4], torch.zeros_like(f[:, :, :4]), weights))
    zero_metric, _ = bounded_interaction_metric(zero_q["R"], zero_q["C"])
    assert torch.equal(zero_metric, identity)
    checks.append("metric_spd_trace_condition_exact_identity_continuity_and_equivariance")

    # Unequal group sizes must preserve image mass when whole groups resample.
    group_r = torch.tensor([1., 1., 10.], dtype=dtype).reshape(3, 1, 1)
    delta, boot_info = cluster_bootstrap_radius(group_r, ["a", "a", "b"], resamples=200, seed=9)
    draws = torch.randint(2, (200, 2), generator=torch.Generator().manual_seed(9))
    totals, counts = torch.tensor([2., 10.], dtype=dtype), torch.tensor([2., 1.], dtype=dtype)
    bootstrap_expected = totals[draws].sum(1)/counts[draws].sum(1)
    expected_delta = torch.quantile((bootstrap_expected - 4.).abs(), .95)
    close(torch.tensor(delta, dtype=dtype), expected_delta)
    assert not boot_info["reliable_support"]
    _, missing_info = cluster_bootstrap_radius(group_r, ["a", None, "b"])
    assert not missing_info["reliable_support"] and not missing_info["group_metadata_complete"]
    stable_f = torch.ones(40, 2, 4, dtype=dtype)
    stable_q = torch.tensor([1., .1, 0., 0.], dtype=dtype).expand_as(stable_f).clone()
    stable_per_image = per_image_moments(stable_f, stable_q, torch.full((40, 2), .5, dtype=dtype))
    reliable_metric, reliable_info = fit_interaction_metric(stable_per_image, [f"event{i}" for i in range(40)])
    assert reliable_info["uncertainty"]["reliable_support"] and not reliable_info["identity"]
    fallback_metric, fallback_info = fit_interaction_metric(stable_per_image, ["one_event"]*40)
    assert torch.equal(fallback_metric, identity) and fallback_info["identity"]
    rank32_f = torch.ones(63, 1, 32, dtype=dtype)
    rank32_per_image = per_image_moments(rank32_f, rank32_f, torch.ones(63, 1, dtype=dtype))
    _, rank32_info = fit_interaction_metric(rank32_per_image, [f"event{i}" for i in range(63)])
    assert rank32_info["identity"] and rank32_info["uncertainty"]["required_group_count"] == 64
    checks.append("cluster_bootstrap_group_totals_support_floor_and_missing_metadata_fallback")

    constant_r = torch.diag(torch.tensor([2., .5, -.25, -1.], dtype=dtype))
    split_images = {"R": constant_r.expand(80, 4, 4).clone(),
                    "usable_images": torch.ones(80, dtype=torch.bool)}
    split_ids = [f"event{i}" for i in range(80)]
    rng_before = torch.get_rng_state().clone()
    split = split_group_stability(split_images, split_ids)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert split["support_floor_met"] and split["half_group_counts"] == [40, 40]
    assert not (set(split["split_group_ids"][0]) & set(split["split_group_ids"][1]))
    close(split["r_operator_gap"], 0.)
    close(split["positive_projector_normalized_overlap"], 1.)
    close(split["a_positive_directions_in_b"]["other_half_rayleigh_values"], [2., .5])
    assert not split["used_for_selection"]
    assert split == split_group_stability(split_images, split_ids)
    rotated_images = {"R": (rotation @ constant_r @ rotation.T).expand(80, 4, 4).clone(),
                      "usable_images": split_images["usable_images"]}
    rotated_split = split_group_stability(rotated_images, split_ids)
    close(rotated_split["r_operator_gap"], split["r_operator_gap"])
    close(rotated_split["positive_projector_normalized_overlap"], split["positive_projector_normalized_overlap"])
    uneven_split = {"R": torch.stack([constant_r, constant_r, -constant_r]),
                    "usable_images": torch.ones(3, dtype=torch.bool)}
    uneven = split_group_stability(uneven_split, ["one", "one", "two"])
    assert not uneven["support_floor_met"] and sorted(uneven["half_image_counts"]) == [1, 2]
    close(uneven["r_operator_gap"], 4.)
    close(uneven["positive_projector_normalized_overlap"], 0.)
    assert all(value <= 0 for value in uneven["a_positive_directions_in_b"]["other_half_rayleigh_values"])
    unavailable = split_group_stability(uneven_split, ["one", None, "two"])
    assert unavailable["status"] == "unavailable" and not unavailable["support_floor_met"]
    checks.append("fixed_whole_group_split_raw_rayleigh_projector_stability_without_selection")

    training_teacher, training_sq = f[:, :, :4] + q[:, :, :4], torch.zeros_like(f[:, :, :4])
    training_moments = aggregate_moments(per_image_moments(f[:, :, :4], q[:, :, :4], weights))
    controls = matched_spectrum_controls(metric, training_moments, seed=42)
    base_loss = feature_loss(training_teacher, training_sq, weights, identity)
    for name, candidate in {"interaction": metric, **controls}.items():
        c = initial_loss_scale(candidate, training_moments["Ce"])
        close(feature_loss(training_teacher, training_sq, weights, candidate, c), base_loss)
        close(torch.linalg.eigvalsh(candidate), torch.linalg.eigvalsh(metric))
    assert initial_loss_scale(identity, training_moments["Ce"]) == 1.
    duplicate_controls = matched_spectrum_controls(identity, training_moments)
    assert all(torch.equal(k, identity) for k in duplicate_controls.values())
    controls_again = matched_spectrum_controls(metric, training_moments, seed=42)
    assert all(torch.equal(controls[name], controls_again[name]) for name in controls)
    checks.append("same_spectrum_controls_deterministic_orientation_and_exact_initial_loss_matching")

    for errors in ((f, q), (f, f), (f, -f), (f, torch.zeros_like(f))):
        diagnostics = residual_diagnostics(aggregate_moments(per_image_moments(*errors, weights)))
        eta, rho = diagnostics["eta"], diagnostics["rho"]
        assert abs(eta) <= 1 + 1e-12
        assert abs(eta) <= 2*(rho*(1-rho))**.5 + 1e-12
        assert 0 <= diagnostics["total_error_amplification"] <= 2 + 1e-12
        close(diagnostics["total_error_amplification"], 1 + eta)
        close(eta, diagnostics["chi_plus"] - diagnostics["chi_minus"])
        assert diagnostics["chi_plus"] + diagnostics["chi_minus"] <= 1 + 1e-12
    cancelling_f = torch.tensor([[[1., 0.], [0., 1.]]], dtype=dtype)
    cancelling_q = torch.tensor([[[1., 0.], [0., -1.]]], dtype=dtype)
    cancelling = residual_diagnostics(aggregate_moments(per_image_moments(cancelling_f, cancelling_q, pos_weights)))
    close(cancelling["eta"], 0.)
    assert cancelling["chi_plus"] > 0 and cancelling["chi_minus"] > 0
    checks.append("eta_rho_chi_bounds_and_opposing_direction_cancellation")

    target = torch.tensor([[[0, 0, 1], [1, -1, 1]]])
    fp_prediction = torch.tensor([[[0, 1, 1], [0, 0, 1]]])
    q_prediction = torch.tensor([[[1, 0, 1], [1, 1, 1]]])
    fp_logits = torch.nn.functional.one_hot(fp_prediction, 2).permute(0, 3, 1, 2).double()
    q_logits = torch.nn.functional.one_hot(q_prediction, 2).permute(0, 3, 1, 2).double()
    flip = accuracy_damage_repair(fp_logits, q_logits, target)
    assert flip["harmful_count"] == 1 and flip["repaired_count"] == 2 and flip["valid_pixels"] == 5
    assert flip["fp_correct_count"] == 3 and flip["quantized_correct_count"] == 4
    close(flip["accuracy_difference"], flip["B"] - flip["H"])
    checks.append("valid_two_class_accuracy_damage_repair_integer_identity")

    teacher_grad = torch.randn(2, 4, 4, dtype=dtype, requires_grad=True)
    student_grad = torch.randn(2, 4, 4, dtype=dtype, requires_grad=True)
    grad_weights = balanced_position_weights(torch.tensor([[0, 0, 1, -1], [1, 1, 1, 1]]))
    loss = feature_loss(teacher_grad, student_grad, grad_weights, metric)
    loss.backward()
    assert teacher_grad.grad is None and torch.isfinite(student_grad.grad).all()
    close(student_grad.grad[0, 3], torch.zeros(4, dtype=dtype))
    grad_student = student_grad.detach().clone().requires_grad_()
    base = grad_student.square().mean()
    feature = feature_loss(teacher_grad, grad_student, grad_weights, identity)
    base_norm = gradient_rms(base, [grad_student])
    feature_norm = gradient_rms(feature, [grad_student])
    beta, beta_info = beta_from_gradient_norms([base_norm]*4, [feature_norm]*4)
    close(beta, .1*base_norm/(feature_norm + 1e-12))
    assert grad_student.grad is None and beta_info["norm_aggregation"].startswith("arithmetic_mean")
    try:
        beta_from_gradient_norms([1.], [0.])
        raise AssertionError("degenerate feature gradient must stop the cell")
    except ValueError:
        pass
    checks.append("frozen_teacher_valid_feature_gradients_and_predeclared_beta_rule")

    # A short real optimization and save/resume must preserve a matched run.
    model = torch.nn.Linear(5, 3, dtype=dtype)
    train_x = student[:2, :8].detach()
    train_y = teacher[:2, :8].detach()
    train_weights = balanced_position_weights(labels[:2, :8])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.5)
    def step(current, opt):
        opt.zero_grad(set_to_none=True)
        value = feature_loss(train_y, current(train_x), train_weights, torch.eye(3, dtype=dtype))
        value.backward()
        opt.step()
        return float(value.detach())
    first = step(model, optimizer)
    for _ in range(10):
        step(model, optimizer)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "adapter": adapter.state_dict(), "transform": transform.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    resumed = torch.nn.Linear(5, 3, dtype=dtype)
    resumed.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=.1, momentum=.5)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    for _ in range(5):
        close(step(model, optimizer), step(resumed, resumed_optimizer))
    for original, restored in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(original, restored)
    last = step(model, optimizer)
    assert last < first
    assert any(not torch.equal(a, b) for a, b in zip(model.parameters(), resumed.parameters()))
    restored_adapter = FrozenFeatureAdapter(torch.empty_like(adapter.weight), torch.empty_like(adapter.bias))
    restored_adapter.load_state_dict(saved["adapter"])
    close(restored_adapter(train_x), adapter(train_x))
    restored_transform = FrozenTeacherTransform(torch.empty_like(transform.mean), torch.empty_like(transform.basis),
                                                torch.empty_like(transform.scales))
    restored_transform.load_state_dict(saved["transform"])
    close(restored_transform(teacher_raw), transform(teacher_raw))
    checks.append("loss_decreasing_optimization_checkpoint_resume_and_independent_parameter_state")

    return {"status": "passed", "device": "cpu", "checks": checks, "check_count": len(checks),
            "feature_probe": "8x8_bilinear_align_corners_false_nearest_exact_center_support",
            "bootstrap": "200_whole_source_groups_equal_image_expectation",
            "precision": "float64_statistics", "optimization_loss_initial": first,
            "optimization_loss_final": last}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "--output", dest="out", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    result = run_checks()
    result["elapsed_seconds"] = time.monotonic() - start
    serialized = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(serialized)
        temporary.replace(args.out)
    print(serialized, end="")


if __name__ == "__main__":
    main()
