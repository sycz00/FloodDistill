"""Training-only feature geometry for the fixed compression pilot.

All fitted statistics use uncentered, per-position moments and equal image mass.
The bootstrap unit is a supplied source group, never an image or pixel invented
as an independent group. Fitted modules contain buffers only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NUMERICAL_FLOOR = 1e-12  # squared standardized-feature units


def balanced_position_weights(labels: Tensor, valid: Tensor | None = None) -> Tensor:
    """Equal available class mass, then equal image mass (labels are [B,P])."""
    if labels.ndim != 2:
        raise ValueError("labels must be [images, positions]")
    allowed = (labels == 0) | (labels == 1)
    if valid is not None:
        if valid.shape != labels.shape:
            raise ValueError("valid mask must match labels")
        allowed &= valid.bool()
    counts = torch.stack([((labels == c) & allowed).sum(1) for c in (0, 1)], 1)
    strata = (counts > 0).sum(1)
    weights = torch.zeros_like(labels, dtype=torch.float64)
    for c in (0, 1):
        denom = (counts[:, c] * strata).clamp_min(1).to(weights.dtype)
        weights += ((labels == c) & allowed).to(weights.dtype) / denom[:, None]
    return weights


def probe_features(features: Tensor, targets: Tensor, valid: Tensor | None = None,
                   grid_size: int = 8) -> tuple[Tensor, Tensor]:
    """Centered 8x8 grid: bilinear features and nearest-exact class/valid support."""
    if features.ndim != 4 or targets.ndim != 3 or features.shape[0] != targets.shape[0]:
        raise ValueError("features must be BCHW and targets BHW with equal batch")
    if grid_size != 8:
        raise ValueError("the initial pilot fixes an 8x8 feature probe")
    sampled = F.interpolate(features, size=(8, 8), mode="bilinear", align_corners=False)
    labels = F.interpolate(targets[:, None].to(torch.float64), size=(8, 8),
                           mode="nearest-exact")[:, 0].long().flatten(1)
    support = None
    if valid is not None:
        if valid.shape != targets.shape:
            raise ValueError("valid mask must match target geometry")
        support = F.interpolate(valid[:, None].to(torch.float64), size=(8, 8),
                                mode="nearest-exact")[:, 0].bool().flatten(1)
    vectors = sampled.permute(0, 2, 3, 1).reshape(sampled.shape[0], 64, sampled.shape[1])
    return vectors, balanced_position_weights(labels, support)


def _fit_inputs(vectors: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if vectors.ndim != 3 or weights.shape != vectors.shape[:2]:
        raise ValueError("vectors/weights must be [B,P,C]/[B,P]")
    x = vectors.detach().to(device="cpu", dtype=torch.float64)
    w = weights.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    sums = w.sum(1)
    usable = sums > 0
    if not usable.any():
        raise ValueError("no valid feature positions")
    if not torch.allclose(sums[usable], torch.ones_like(sums[usable]), atol=1e-7, rtol=1e-7):
        raise ValueError("position weights must give each usable image unit mass")
    if not torch.isfinite(x[w > 0]).all():
        raise ValueError("nonfinite features on valid probe support")
    x = torch.where((w > 0)[..., None], x, torch.zeros_like(x))
    return x, w, usable


class FrozenTeacherTransform(nn.Module):
    def __init__(self, mean: Tensor, basis: Tensor, scales: Tensor):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("basis", basis.detach().clone())
        self.register_buffer("scales", scales.detach().clone())

    def forward(self, features: Tensor) -> Tensor:
        mean = self.mean.to(device=features.device, dtype=features.dtype)
        basis = self.basis.to(device=features.device, dtype=features.dtype)
        scales = self.scales.to(device=features.device, dtype=features.dtype)
        return ((features - mean) @ basis) / scales


class FrozenFeatureAdapter(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("bias", bias.detach().clone())

    def forward(self, features: Tensor) -> Tensor:
        return (features @ self.weight.to(device=features.device, dtype=features.dtype)
                + self.bias.to(device=features.device, dtype=features.dtype))


def fit_teacher_pca(vectors: Tensor, weights: Tensor, max_rank: int = 32,
                    min_student_dim: int | None = None,
                    relative_variance_floor: float = 1e-10
                    ) -> tuple[FrozenTeacherTransform, dict[str, Any]]:
    x, w, usable = _fit_inputs(vectors, weights)
    image_count = int(usable.sum())
    mean = torch.einsum("bp,bpc->c", w, x) / image_count
    centered = x - mean
    covariance = torch.einsum("bp,bpc,bpd->cd", w, centered, centered) / image_count
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues, eigenvectors = eigenvalues.flip(0), eigenvectors.flip(1)
    largest = max(float(eigenvalues[0]), 0.0)
    floor = max(NUMERICAL_FLOOR, relative_variance_floor * largest)
    numerical_rank = int((eigenvalues > floor).sum())
    smallest_dim = x.shape[-1] if min_student_dim is None else int(min_student_dim)
    rank = min(int(max_rank), x.shape[-1], smallest_dim, numerical_rank)
    if rank < 1:
        raise ValueError("teacher feature covariance has no usable numerical rank")
    values = eigenvalues[:rank]
    transform = FrozenTeacherTransform(mean, eigenvectors[:, :rank], values.clamp_min(floor).sqrt())
    total_variance = float(eigenvalues.clamp_min(0).sum())
    return transform, {
        "rank": rank, "teacher_dimension": x.shape[-1], "smallest_student_dimension": smallest_dim,
        "numerical_rank": numerical_rank, "usable_images": image_count,
        "variance_floor": floor, "retained_eigenvalues": values.tolist(),
        "retained_variance_fraction": float(values.sum()) / total_variance,
        "position_reduction": "equal_available_class_then_equal_image",
        "feature_grid": [8, 8], "interpolation": "bilinear_align_corners_false",
        "support_interpolation": "nearest_exact_same_cell_centers",
    }


def fit_ridge_adapter(student_vectors: Tensor, teacher_vectors: Tensor, weights: Tensor,
                      relative_ridge: float = 1e-3
                      ) -> tuple[FrozenFeatureAdapter, dict[str, Any]]:
    if relative_ridge < 0:
        raise ValueError("relative ridge must be nonnegative")
    x, w, usable = _fit_inputs(student_vectors, weights)
    y, _, _ = _fit_inputs(teacher_vectors, weights)
    if x.shape[:2] != y.shape[:2]:
        raise ValueError("student and teacher must share positions and images")
    image_count = int(usable.sum())
    x_mean = torch.einsum("bp,bpc->c", w, x) / image_count
    y_mean = torch.einsum("bp,bpc->c", w, y) / image_count
    xc, yc = x - x_mean, y - y_mean
    gram = torch.einsum("bp,bpc,bpd->cd", w, xc, xc) / image_count
    gram = (gram + gram.T) / 2
    cross = torch.einsum("bp,bpc,bpd->cd", w, xc, yc) / image_count
    energy = float(torch.trace(gram))
    ridge = relative_ridge * energy / x.shape[-1]
    degenerate = energy <= NUMERICAL_FLOOR
    if degenerate:
        coefficient = torch.zeros(x.shape[-1], y.shape[-1], dtype=torch.float64)
    elif ridge > 0:
        coefficient = torch.linalg.solve(gram + ridge * torch.eye(x.shape[-1], dtype=gram.dtype), cross)
    else:
        coefficient = torch.linalg.pinv(gram, hermitian=True) @ cross
    bias = y_mean - x_mean @ coefficient
    adapter = FrozenFeatureAdapter(coefficient, bias)
    fit_error = ((y - adapter(x)).square().sum(-1) * w).sum() / image_count / y.shape[-1]
    return adapter, {
        "ridge": ridge, "relative_ridge": relative_ridge, "bias_penalized": False,
        "degenerate_student_gram": degenerate, "usable_images": image_count,
        "student_dimension": x.shape[-1], "output_dimension": y.shape[-1],
        "fit_feature_mse": float(fit_error), "frozen": True,
    }


def per_image_moments(f: Tensor, q: Tensor, weights: Tensor) -> dict[str, Tensor]:
    if f.shape != q.shape:
        raise ValueError("f and q must share shape")
    f, w, usable = _fit_inputs(f, weights)
    q, _, _ = _fit_inputs(q, weights)
    outer = lambda a, b: torch.einsum("bp,bpc,bpd->bcd", w, a, b)
    cf, cq = outer(f, f), outer(q, q)
    fq = outer(f, q)
    r = (fq + fq.transpose(-1, -2)) / 2
    e = f + q
    return {"Cf": cf, "Cq": cq, "R": r, "C": cf + cq,
            "Ce": outer(e, e), "usable_images": usable}


def aggregate_moments(per_image: dict[str, Tensor]) -> dict[str, Tensor]:
    usable = per_image["usable_images"].bool()
    if not usable.any():
        raise ValueError("no usable per-image moments")
    return {name: per_image[name][usable].to(torch.float64).mean(0)
            for name in ("Cf", "Cq", "R", "C", "Ce")}


def cluster_bootstrap_radius(per_image_r: Tensor, group_ids: Sequence[Any],
                             usable_images: Tensor | None = None, resamples: int = 200,
                             seed: int = 42) -> tuple[float | None, dict[str, Any]]:
    if per_image_r.ndim != 3 or per_image_r.shape[-1] != per_image_r.shape[-2]:
        raise ValueError("per-image R must be [images,r,r]")
    if len(group_ids) != per_image_r.shape[0]:
        raise ValueError("one actual source group ID is required per image")
    if resamples < 1:
        raise ValueError("bootstrap requires at least one resample")
    usable = (torch.ones(per_image_r.shape[0], dtype=torch.bool) if usable_images is None
              else usable_images.detach().cpu().bool())
    r = per_image_r.detach().cpu().to(torch.float64)
    members: dict[str, list[int]] = defaultdict(list)
    missing = False
    for i, group in enumerate(group_ids):
        if not bool(usable[i]):
            continue
        if group is None or not str(group).strip():
            missing = True
        else:
            members[str(group)].append(i)
    group_count = len(members)
    floor = max(32, 2 * r.shape[-1])
    info: dict[str, Any] = {"group_count": group_count, "required_group_count": floor,
                            "group_metadata_complete": not missing,
                            "reliable_support": not missing and group_count >= floor,
                            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                            "uncertainty_unit": "supplied_source_event_site_group"}
    if missing or group_count == 0:
        info["reason"] = "missing_source_group_metadata"
        return None, info
    if not usable.any():
        raise ValueError("no usable images for bootstrap")
    ordered_groups = sorted(members)
    totals = torch.stack([r[members[g]].sum(0) for g in ordered_groups])
    counts = torch.tensor([len(members[g]) for g in ordered_groups], dtype=torch.float64)
    estimate = totals.sum(0) / counts.sum()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randint(group_count, (resamples, group_count), generator=generator)
    boot = totals[samples].sum(1) / counts[samples].sum(1)[:, None, None]
    difference = boot - estimate
    difference = (difference + difference.transpose(-1, -2)) / 2
    norms = torch.linalg.eigvalsh(difference).abs().amax(-1)
    delta = float(torch.quantile(norms, .95, interpolation="linear"))
    info["delta"] = delta
    half_a, half_b = ordered_groups[::2], ordered_groups[1::2]
    if half_a and half_b:
        a_idx = [i for g in half_a for i in members[g]]
        b_idx = [i for g in half_b for i in members[g]]
        split_difference = r[a_idx].mean(0) - r[b_idx].mean(0)
        info["split_group_r_operator_difference"] = float(torch.linalg.eigvalsh(split_difference).abs().max())
        info["split_group_counts"] = [len(half_a), len(half_b)]
    if group_count < floor:
        info["reason"] = "insufficient_independent_source_groups"
    return delta, info


def split_group_stability(per_image: dict[str, Tensor], group_ids: Sequence[Any],
                           seed: int = 1729) -> dict[str, Any]:
    """Descriptive raw-R stability on fixed disjoint source-group halves.

    This diagnostic does not fit a training metric, select a model, or establish
    a confidence guarantee. Each half retains equal mass per available image.
    """
    matrices = per_image["R"].detach().cpu().double()
    usable = per_image["usable_images"].detach().cpu().bool()
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("per-image R must be [images,r,r]")
    if len(group_ids) != len(matrices) or usable.shape != (len(matrices),):
        raise ValueError("one source group and validity flag is required per image")
    if not torch.isfinite(matrices[usable]).all():
        raise ValueError("nonfinite usable interaction moments")
    rank = matrices.shape[-1]
    floor = max(32, 2 * rank)
    members: dict[str, list[int]] = defaultdict(list)
    missing = False
    for i, group in enumerate(group_ids):
        if not bool(usable[i]):
            continue
        if group is None or not str(group).strip():
            missing = True
        else:
            members[str(group)].append(i)
    result: dict[str, Any] = {
        "status": "descriptive", "seed": seed, "group_count": len(members),
        "required_groups_per_half": floor, "group_metadata_complete": not missing,
        "moment_definition": "uncentered_per_position_equal_class_then_equal_image",
        "interpretation": "raw interaction direction stability only; no confidence guarantee or selection rule",
        "used_for_selection": False,
    }
    if missing or len(members) < 2:
        result.update(status="unavailable", support_floor_met=False,
                      reason="missing_source_group_metadata" if missing else "fewer_than_two_source_groups")
        return result
    sorted_groups = sorted(members)
    order = torch.randperm(len(sorted_groups), generator=torch.Generator().manual_seed(seed)).tolist()
    groups = [[sorted_groups[i] for i in order[::2]], [sorted_groups[i] for i in order[1::2]]]
    indices = [[i for group in half for i in members[group]] for half in groups]
    estimates = [matrices[half].mean(0) for half in indices]
    estimates = [(estimate + estimate.T) / 2 for estimate in estimates]
    eigenpairs = [torch.linalg.eigh(estimate) for estimate in estimates]
    eigenvalues = [pair[0].flip(0) for pair in eigenpairs]
    eigenvectors = [pair[1].flip(1) for pair in eigenpairs]
    positive_bases = [v[:, value > 0] for value, v in zip(eigenvalues, eigenvectors)]
    cross_rayleigh = []
    for half, other in ((0, 1), (1, 0)):
        basis = positive_bases[half]
        rayleigh = torch.einsum("ci,cd,di->i", basis, estimates[other], basis)
        cross_rayleigh.append({
            "positive_direction_count": basis.shape[1],
            "own_positive_eigenvalues": eigenvalues[half][eigenvalues[half] > 0].tolist(),
            "other_half_rayleigh_values": rayleigh.tolist(),
            "positive_in_other_half_fraction": float((rayleigh > 0).double().mean()) if len(rayleigh) else None,
        })
    basis_a, basis_b = positive_bases
    count_a, count_b = basis_a.shape[1], basis_b.shape[1]
    if count_a and count_b:
        singular = torch.linalg.svdvals(basis_a.T @ basis_b)
        projector_overlap = float(singular.square().sum())
        normalized_overlap = projector_overlap / (count_a * count_b) ** .5
        principal_cosines_squared = singular.square().tolist()
    else:
        projector_overlap = normalized_overlap = None
        principal_cosines_squared = []
    gap = float(torch.linalg.eigvalsh(estimates[0] - estimates[1]).abs().max())
    result.update({
        "half_group_counts": [len(half) for half in groups],
        "half_image_counts": [len(half) for half in indices],
        "support_floor_met": all(len(half) >= floor for half in groups),
        "split_group_ids": groups,
        "r_operator_gap": gap,
        "half_r_eigenvalues": [value.tolist() for value in eigenvalues],
        "a_positive_directions_in_b": cross_rayleigh[0],
        "b_positive_directions_in_a": cross_rayleigh[1],
        "positive_projector_trace_overlap": projector_overlap,
        "positive_projector_normalized_overlap": normalized_overlap,
        "projector_normalization": "trace(Pa Pb)/sqrt(rank(Pa)*rank(Pb))",
        "positive_principal_cosines_squared": principal_cosines_squared,
    })
    if not result["support_floor_met"]:
        result["reason"] = "insufficient_groups_per_half; computed descriptive quantities only"
    return result


def bounded_interaction_metric(r_matrix: Tensor, c_matrix: Tensor, delta: float = 0.,
                                gamma: float = 1., reliable_support: bool = True
                                ) -> tuple[Tensor, dict[str, Any]]:
    r_matrix = r_matrix.detach().cpu().to(torch.float64)
    c_matrix = c_matrix.detach().cpu().to(torch.float64)
    if (r_matrix.ndim != 2 or r_matrix.shape[0] != r_matrix.shape[1]
            or c_matrix.shape != r_matrix.shape):
        raise ValueError("R and C must be equal square matrices")
    if not torch.isfinite(r_matrix).all() or not torch.isfinite(c_matrix).all():
        raise ValueError("moment matrices must be finite")
    if delta < 0 or gamma < 0:
        raise ValueError("noise radius and gamma must be nonnegative")
    rank = r_matrix.shape[0]
    identity = torch.eye(rank, dtype=torch.float64)
    sym_r = (r_matrix + r_matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(sym_r)
    eigenvalues, eigenvectors = eigenvalues.flip(0), eigenvectors.flip(1)
    positive = (eigenvalues - delta).clamp_min(0)
    energy = float(torch.trace(c_matrix))
    info: dict[str, Any] = {"rank": rank, "gamma": gamma, "delta": delta,
                           "numerical_floor": NUMERICAL_FLOOR,
                           "raw_r_eigenvalues": eigenvalues.tolist(),
                           "shrunk_positive_eigenvalues": positive.tolist(),
                           "retained_directions": int((positive > 0).sum()),
                           "reliable_support": reliable_support}
    reason = None
    if not reliable_support:
        reason = "insufficient_reliable_group_support"
    elif energy <= NUMERICAL_FLOOR:
        reason = "negligible_total_residual_energy"
    elif not (positive > 0).any() or gamma == 0:
        reason = "no_retained_positive_interaction"
    if reason is not None:
        metric = identity
    else:
        nu = energy / rank
        h = (2 * gamma * positive / (nu + NUMERICAL_FLOOR)).clamp_max(3.)
        d = (1 + h) / (1 + h).mean()
        # Isotropic positive interaction has no directional intervention.
        if float(d.max() - d.min()) <= 1e-12:
            metric, reason = identity, "isotropic_interaction"
        else:
            metric = (eigenvectors * d[None]) @ eigenvectors.T
            metric = (metric + metric.T) / 2
    eig_m = torch.linalg.eigvalsh(metric)
    info.update({"identity": bool(torch.equal(metric, identity)), "identity_reason": reason,
                 "metric_eigenvalues": eig_m.flip(0).tolist(),
                 "metric_trace": float(torch.trace(metric)),
                 "metric_condition_number": float(eig_m[-1] / eig_m[0]),
                 "metric_distance_from_identity": float(torch.linalg.matrix_norm(metric - identity))})
    return metric, info


def fit_interaction_metric(per_image: dict[str, Tensor], group_ids: Sequence[Any],
                           gamma: float = 1., bootstrap_resamples: int = 200,
                           seed: int = 42) -> tuple[Tensor, dict[str, Any]]:
    moments = aggregate_moments(per_image)
    delta, uncertainty = cluster_bootstrap_radius(per_image["R"], group_ids,
                                                  per_image["usable_images"],
                                                  bootstrap_resamples, seed)
    metric, info = bounded_interaction_metric(moments["R"], moments["C"],
                                               0. if delta is None else delta, gamma,
                                               uncertainty["reliable_support"])
    info["uncertainty"] = uncertainty
    info["initial_loss_scale"] = initial_loss_scale(metric, moments["Ce"])
    info["diagnostics"] = residual_diagnostics(moments)
    return metric, info


def initial_loss_scale(metric: Tensor, ce: Tensor) -> float:
    metric, ce = metric.detach().cpu().double(), ce.detach().cpu().double()
    if metric.shape != ce.shape or metric.ndim != 2:
        raise ValueError("metric and Ce must be equal square matrices")
    energy = float(torch.trace(ce))
    if energy <= NUMERICAL_FLOOR:
        return 1.
    denominator = float(torch.trace(metric @ ce))
    if denominator <= 0 or not torch.isfinite(torch.tensor(denominator)):
        raise ValueError("nonpositive/nonfinite metric-weighted residual energy")
    if torch.equal(metric, torch.eye(metric.shape[0], dtype=metric.dtype)):
        return 1.
    return energy / denominator


def matched_spectrum_controls(metric: Tensor, moments: dict[str, Tensor],
                               seed: int = 42) -> dict[str, Tensor]:
    metric = metric.detach().cpu().double()
    rank = metric.shape[0]
    identity = torch.eye(rank, dtype=metric.dtype)
    if torch.equal(metric, identity):
        return {name: identity.clone() for name in ("random", "quant", "independent", "total")}
    spectrum = torch.linalg.eigvalsh(metric).flip(0)
    random = torch.randn(rank, rank, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    basis, triangular = torch.linalg.qr(random)
    signs = torch.sign(torch.diag(triangular))
    basis = basis * torch.where(signs == 0, torch.ones_like(signs), signs)[None]
    controls = {"random": (basis * spectrum[None]) @ basis.T}
    for name, covariance in (("quant", moments["Cq"]), ("independent", moments["C"]),
                             ("total", moments["Ce"])):
        covariance = covariance.detach().cpu().double()
        _, basis = torch.linalg.eigh((covariance + covariance.T) / 2)
        basis = basis.flip(1)
        controls[name] = (basis * spectrum[None]) @ basis.T
    return {name: (value + value.T) / 2 for name, value in controls.items()}


def feature_loss(teacher: Tensor, student_quantized: Tensor, weights: Tensor,
                 metric: Tensor, loss_scale: float = 1.) -> Tensor:
    if teacher.shape != student_quantized.shape or teacher.ndim != 3:
        raise ValueError("teacher/student features must be equal [B,P,r]")
    if weights.shape != teacher.shape[:2] or metric.shape != (teacher.shape[-1],) * 2:
        raise ValueError("weights/metric must match feature dimensions")
    residual = teacher.detach() - student_quantized
    weights = weights.to(device=residual.device, dtype=residual.dtype)
    usable = weights.sum(1) > 0
    residual = torch.where((weights > 0)[..., None], residual, torch.zeros_like(residual))
    if not usable.any():
        return residual.sum() * 0
    metric = metric.to(device=residual.device, dtype=residual.dtype)
    position_losses = torch.einsum("bpc,cd,bpd->bp", residual, metric, residual)
    return loss_scale * (position_losses * weights).sum(1)[usable].mean() / residual.shape[-1]


def gradient_rms(loss: Tensor, parameters: Iterable[Tensor], retain_graph: bool = True) -> float:
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        raise ValueError("gradient diagnostic requires a trainable shared block")
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    squared, count = 0., 0
    for parameter, gradient in zip(parameters, gradients):
        count += parameter.numel()
        if gradient is not None:
            squared += float(gradient.detach().double().square().sum())
    return (squared / count) ** .5


def beta_from_gradient_norms(base_norms: Sequence[float], identity_feature_norms: Sequence[float],
                             fraction: float = .1, floor: float = 1e-12) -> tuple[float, dict[str, Any]]:
    if not base_norms or len(base_norms) != len(identity_feature_norms):
        raise ValueError("use identical nonempty calibration minibatches for both gradient norms")
    values = torch.tensor([*base_norms, *identity_feature_norms], dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("gradient norms must be finite and nonnegative")
    base = sum(base_norms) / len(base_norms)
    feature = sum(identity_feature_norms) / len(identity_feature_norms)
    if feature <= floor:
        raise ValueError("degenerate identity-feature gradient; stop this configuration")
    beta = fraction * base / (feature + floor)
    return beta, {"beta": beta, "gradient_fraction": fraction, "gradient_floor": floor,
                  "base_gradient_rms": list(base_norms), "feature_gradient_rms": list(identity_feature_norms),
                  "norm_aggregation": "arithmetic_mean_of_per_minibatch_parameter_rms"}


def residual_diagnostics(moments: dict[str, Tensor]) -> dict[str, Any]:
    a = float(torch.trace(moments["Cf"]))
    q = float(torch.trace(moments["Cq"]))
    r = (moments["R"] + moments["R"].T) / 2
    x = float(torch.trace(r))
    total = a + q
    result: dict[str, Any] = {"A": a, "Q": q, "X": x,
                              "Ce_trace": float(torch.trace(moments["Ce"]))}
    if total <= NUMERICAL_FLOOR:
        result.update({"meaningful": False, "eta": None, "rho": None,
                       "total_error_amplification": None, "chi_plus": None, "chi_minus": None})
    else:
        eig = torch.linalg.eigvalsh(r)
        result.update({"meaningful": True, "eta": 2 * x / total, "rho": q / total,
                       "total_error_amplification": result["Ce_trace"] / total,
                       "chi_plus": 2 * float(eig.clamp_min(0).sum()) / total,
                       "chi_minus": 2 * float((-eig).clamp_min(0).sum()) / total})
    return result


def accuracy_damage_repair(fp_logits: Tensor, quantized_logits: Tensor, targets: Tensor,
                           valid: Tensor | None = None) -> dict[str, Any]:
    if fp_logits.shape != quantized_logits.shape or fp_logits.ndim != 4 or fp_logits.shape[1] != 2:
        raise ValueError("paired logits must be equal BCHW two-class tensors")
    if targets.shape != fp_logits.shape[:1] + fp_logits.shape[2:]:
        raise ValueError("targets must match paired logits geometry")
    support = (targets == 0) | (targets == 1)
    if valid is not None:
        if valid.shape != targets.shape:
            raise ValueError("valid support must match targets")
        support &= valid.bool()
    count = int(support.sum())
    if not count:
        raise ValueError("no valid GT support for damage/repair diagnostic")
    fp_correct = fp_logits.argmax(1) == targets
    q_correct = quantized_logits.argmax(1) == targets
    harmful = int((fp_correct & ~q_correct & support).sum())
    repaired = int((~fp_correct & q_correct & support).sum())
    fp_count = int((fp_correct & support).sum())
    q_count = int((q_correct & support).sum())
    if q_count - fp_count != repaired - harmful:
        raise AssertionError("integer damage/repair identity failed")
    return {"valid_pixels": count, "harmful_count": harmful, "repaired_count": repaired,
            "fp_correct_count": fp_count, "quantized_correct_count": q_count,
            "H": harmful / count, "B": repaired / count,
            "accuracy_fp": fp_count / count, "accuracy_q": q_count / count,
            "accuracy_difference": (q_count - fp_count) / count,
            "identity_residual_count": q_count - fp_count - repaired + harmful}
