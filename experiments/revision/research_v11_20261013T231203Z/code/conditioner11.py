"""Bounded V11 C0--C4 convex readout utilities. No backbone training or I/O on import.

The source-array interface deliberately does not open dataset labels. A caller must
provide the verified, input-mask-only geometry audit before admitting extraction.
This module alone is not a completed conditioner experiment.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

SEEDS = (42, 64, 91, 107)
FORMS = ("C0", "C1", "C2", "C3", "C4")
LAMBDAS = (0.001, 0.1, 10.0)
TEMPERATURE = 4.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def sigmoid(x):
    x = np.asarray(x, np.float64)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def objective_gradient_hessian(theta, design, margin0, q, weights, regularization,
                               temperature=TEMPERATURE, *, hessian=True):
    """BCE differs from teacher KL only by a coefficient-independent entropy.

    All rows must already be sampled using input validity alone. We reject invalid
    rows rather than silently filter rows differently between controls.
    """
    theta, design, margin0, q, weights = [np.asarray(x, np.float64) for x in
                                        (theta, design, margin0, q, weights)]
    if design.ndim != 2 or theta.shape != (design.shape[1],):
        raise ValueError("Coefficient/design shape mismatch")
    n = len(design)
    if n == 0 or any(x.shape != (n,) for x in (margin0, q, weights)):
        raise ValueError("A nonempty common row set is required")
    if not all(np.isfinite(x).all() for x in (theta, design, margin0, q, weights)):
        raise ValueError("Nonfinite source row")
    if not (np.all((q >= 0) & (q <= 1)) and np.all(weights > 0)
            and np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-10)):
        raise ValueError("Require soft probabilities and positive normalized weights")
    if regularization <= 0 or temperature <= 0:
        raise ValueError("Positive regularization and temperature required")
    a = (margin0 + design @ theta) / temperature
    p = sigmoid(a)
    value = temperature**2 * np.dot(weights, np.logaddexp(0, a) - q * a)
    value += regularization / 2 * np.dot(theta, theta)
    gradient = temperature * (design.T @ (weights * (p - q))) + regularization * theta
    H = None
    if hessian:
        H = design.T @ (design * (weights * p * (1 - p))[:, None])
        H += regularization * np.eye(len(theta))
    return float(value), gradient, H


def fit_newton(design, margin0, q, weights, regularization, *, max_iterations=10,
               gradient_tolerance=1e-7):
    """At most ten damped Newton steps, with Armijo backtracking and diagnostics."""
    if not 1 <= max_iterations <= 10:
        raise ValueError("The predeclared cap is ten Newton iterations")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    theta = np.zeros(np.asarray(design).shape[1], np.float64)
    history = []
    converged = False
    reason = "iteration_limit"
    for iteration in range(max_iterations):
        value, gradient, H = objective_gradient_hessian(theta, design, margin0, q, weights, regularization)
        norm = float(np.linalg.norm(gradient))
        if norm <= gradient_tolerance:
            converged, reason = True, "gradient_tolerance"
            break
        step = np.linalg.solve(H, gradient)
        directional = float(np.dot(gradient, step))
        if directional <= 0 or not np.isfinite(directional):
            reason = "invalid_newton_direction"
            break
        scale = 1.0
        for backtracks in range(31):
            proposal = theta - scale * step
            new_value, _, _ = objective_gradient_hessian(proposal, design, margin0, q, weights,
                                                        regularization, hessian=False)
            if new_value <= value - 1e-4 * scale * directional:
                theta = proposal
                history.append(dict(iteration=iteration + 1, objective_before=value,
                                    objective_after=new_value, gradient_norm=norm,
                                    step_scale=scale, backtracks=backtracks))
                break
            scale *= 0.5
        else:
            reason = "backtracking_failed"
            break
    final_value, final_gradient, _ = objective_gradient_hessian(theta, design, margin0, q,
                                                                weights, regularization, hessian=False)
    norm = float(np.linalg.norm(final_gradient))
    if norm <= gradient_tolerance:
        converged, reason = True, "gradient_tolerance"
    return theta, dict(converged=converged, reason=reason, iterations=len(history),
                      gradient_norm=norm, objective=final_value, history=history,
                      wall_seconds=time.monotonic() - start_wall,
                      process_cpu_seconds=time.process_time() - start_cpu)


def hash_selected_tiles(tile_ids, limit=512):
    if not 1 <= limit <= 512 or len(set(tile_ids)) != len(tile_ids):
        raise ValueError("Require unique TRAIN tile IDs and at most 512 tiles")
    return sorted(tile_ids, key=lambda x: (hashlib.sha256(x.encode()).hexdigest(), x))[:limit]


def sample_coordinates(tile_id, input_valid, limit=256):
    """Hash sampling uses only tile ID, coordinates and a deployable input mask."""
    mask = np.asarray(input_valid)
    if mask.dtype != bool or mask.ndim != 2 or not 1 <= limit <= 256:
        raise ValueError("Require a two-dimensional input-valid bool mask")
    coordinates = np.argwhere(mask)
    keys = [(hashlib.sha256(f"{tile_id}:{y}:{x}".encode()).digest(), int(y), int(x))
            for y, x in coordinates]
    return np.array([(y, x) for _, y, x in sorted(keys)[:limit]], dtype=np.int64).reshape(-1, 2)


def equal_tile_weights(tile_ids):
    _, inverse, counts = np.unique(tile_ids, return_inverse=True, return_counts=True)
    if not len(counts):
        raise ValueError("Empty fit subset")
    return 1.0 / (len(counts) * counts[inverse])


def input_descriptor(image, input_valid):
    """Six band means and standard deviations; accepts no reference-label mask.

    The mask is explicitly supplied by the deployable loader. The caller records
    its provenance; a reference cloud/water mask is never an acceptable argument.
    """
    image, valid = np.asarray(image, np.float64), np.asarray(input_valid)
    if image.ndim != 3 or image.shape[0] != 6 or valid.dtype != bool or valid.shape != image.shape[1:]:
        raise ValueError("Expected six bands and their input validity mask")
    if not valid.any() or not np.isfinite(image[:, valid]).all():
        raise ValueError("No valid finite descriptor support")
    values = image[:, valid]
    return np.concatenate([values.mean(1), values.std(1, ddof=0)])


def train_statistics(features, descriptors, row_tile_ids, descriptor_tile_ids, *, split):
    """TRAIN-only estimates; feature rows and descriptor tiles receive equal tile weight."""
    if split != "GDACS_TRAIN":
        raise ValueError("Statistics may be fit only on GDACS TRAIN")
    features, descriptors = np.asarray(features, np.float64), np.asarray(descriptors, np.float64)
    if (features.ndim != 2 or descriptors.shape != (len(descriptor_tile_ids), 12)
            or len(set(descriptor_tile_ids)) != len(descriptor_tile_ids)
            or set(row_tile_ids) != set(descriptor_tile_ids)
            or len(row_tile_ids) != len(features)):
        raise ValueError("Misaligned source rows/descriptors")
    if not np.isfinite(features).all() or not np.isfinite(descriptors).all():
        raise ValueError("Nonfinite source statistics")
    weights = equal_tile_weights(row_tile_ids)
    mu = weights @ features
    scale = np.sqrt(weights @ ((features - mu)**2))
    # A constant feature gets scale one, preserving the exact constant correction.
    scale = np.where(scale > 1e-8, scale, 1.0)
    zmu, zscale = descriptors.mean(0), descriptors.std(0)
    zscale = np.where(zscale > 1e-8, zscale, 1.0)
    return dict(feature_mean=mu, feature_scale=scale, descriptor_mean=zmu,
                descriptor_scale=zscale, split=split)


def standardize(features, descriptors, statistics):
    if statistics["split"] != "GDACS_TRAIN":
        raise ValueError("Target-fitted statistics are prohibited")
    h = (features - statistics["feature_mean"]) / statistics["feature_scale"]
    phi = np.concatenate([h, np.ones((len(h), 1))], axis=1)
    z = np.clip((descriptors - statistics["descriptor_mean"]) / statistics["descriptor_scale"], -3, 3)
    return phi, z


def shuffled_descriptors(z_tiles, tile_ids):
    """A fixed hash-ordered cyclic derangement. C4 inference uses actual descriptors."""
    if len(tile_ids) < 2 or len(set(tile_ids)) != len(tile_ids) or len(z_tiles) != len(tile_ids):
        raise ValueError("C4 requires at least two distinct aligned TRAIN tiles")
    order = np.asarray(sorted(range(len(tile_ids)), key=lambda i: hashlib.sha256(tile_ids[i].encode()).digest()))
    permutation = np.empty_like(order)
    permutation[order] = np.roll(order, 1)
    assert np.all(permutation != np.arange(len(order)))
    return np.asarray(z_tiles)[permutation], permutation


def design_matrix(phi, z, form):
    phi, z = np.asarray(phi, np.float64), np.asarray(z, np.float64)
    if phi.ndim != 2 or z.shape != (len(phi), 12) or not np.all(phi[:, -1] == 1):
        raise ValueError("Require standardized features plus bias and twelve descriptors")
    if 13 * phi.shape[1] > 512:
        raise ValueError("Conditioner exceeds the frozen coefficient cap")
    if form == "C0":
        return np.zeros((len(phi), 0))
    if form == "C1":
        return phi
    if form == "C2":
        return np.concatenate([phi, z], axis=1)
    if form in ("C3", "C4"):
        return np.concatenate([phi, (phi[:, :, None] * z[:, None, :]).reshape(len(phi), -1)], axis=1)
    raise ValueError(f"Unknown predeclared form: {form}")


def coefficients(theta, feature_width, form):
    theta = np.asarray(theta, np.float64)
    p = feature_width + 1
    sizes = dict(C0=0, C1=p, C2=p + 12, C3=p * 13, C4=p * 13)
    if form not in sizes or theta.shape != (sizes[form],):
        raise ValueError("Unexpected coefficient count")
    a, H = np.zeros(p), np.zeros((p, 12))
    if form != "C0":
        a[:] = theta[:p]
    if form == "C2":
        H[-1] = theta[p:]
    elif form in ("C3", "C4"):
        H[:] = theta[p:].reshape(p, 12)
    return a, H


def native_correction(theta, descriptor, feature_mean, feature_scale, form):
    """Per-tile coefficients in the original head feature basis, not a global head."""
    mu, scale = np.asarray(feature_mean), np.asarray(feature_scale)
    if mu.ndim != 1 or mu.shape != scale.shape or np.any(scale <= 0):
        raise ValueError("Invalid frozen feature scales")
    a, H = coefficients(theta, len(mu), form)
    c = a + H @ np.asarray(descriptor)
    dw = c[:-1] / scale
    db = c[-1] - np.dot(dw, mu)
    return dw, float(db)


def fold_two_logit_head(weight, bias, delta_weight, delta_bias):
    weight, bias = np.array(weight, copy=True), np.array(bias, copy=True)
    if weight.shape[0] != 2 or bias.shape != (2,):
        raise ValueError("Expected land row 0 and water row 1")
    correction = np.asarray(delta_weight).reshape(weight.shape[1:])
    weight[1] += correction / 2
    weight[0] -= correction / 2
    bias[1] += delta_bias / 2
    bias[0] -= delta_bias / 2
    return weight, bias


def source_select_regularization(scores):
    """Input: lambda -> seed -> equal-event water IoU. Ties prefer stronger lambda."""
    if set(scores) != set(LAMBDAS) or any(set(by_seed) != set(SEEDS) for by_seed in scores.values()):
        raise ValueError("All regularizations and all four seeds are mandatory")
    values = {lam: float(np.mean([scores[lam][seed] for seed in SEEDS])) for lam in LAMBDAS}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Nonfinite source selection metric")
    return max(LAMBDAS, key=lambda lam: (values[lam], lam)), values


def dependency_status(v10_root, geometry_receipt=None):
    """Lightweight terminal-receipt gate; never opens incomplete checkpoint/cache files."""
    root = Path(v10_root)
    missing, invalid, parents = [], [], []
    for seed in SEEDS:
        path = root / "scientific_receipts" / f"R4_micro_KDoff_s{seed}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        receipt = json.loads(path.read_text())
        if not (receipt.get("status") == "complete" and receipt.get("outcome") == "completed"
                and receipt.get("graph") == "R4_micro" and receipt.get("method") == "KDoff"
                and receipt.get("seed") == seed and receipt.get("completed_updates") == 7800
                and receipt.get("source_only_selection") is True and receipt.get("smoke") is False
                and receipt.get("parameters") == 1266
                and receipt.get("cache_policy", {}).get("mode") == "float32_eight_geometric_views"):
            invalid.append(dict(path=str(path), reason="parent identity/recipe mismatch"))
            continue
        checkpoint = Path(receipt["best_path"])
        if not checkpoint.is_file() or sha256(checkpoint) != receipt["best_sha256"]:
            invalid.append(dict(path=str(path), reason="completed parent hash mismatch"))
            continue
        parents.append(dict(seed=seed, checkpoint=str(checkpoint), sha256=receipt["best_sha256"],
                            receipt=str(path), receipt_sha256=sha256(path)))
    cache = root / "teacher_logit_cache_manifest.json"
    if not cache.exists():
        missing.append(str(cache))
    else:
        manifest = json.loads(cache.read_text())
        if not (manifest.get("outcome") == "completed" and manifest.get("dtype") == "float32"
                and manifest.get("views") == 8 and len(manifest.get("files", [])) == 2500):
            invalid.append(dict(path=str(cache), reason="cache identity incomplete"))
    if missing:
        return dict(status="WAITING_DEPENDENCY", missing=missing, invalid=invalid, parents=parents)
    if invalid:
        return dict(status="DEFERRED", reason="Invalid source dependency", invalid=invalid)
    if not geometry_receipt:
        return dict(status="DEFERRED", reason="Source 224-versus-512 descriptor/output audit and deployable input-mask provenance not yet established", parents=parents)
    required = {"source_only": True, "input_masks_only": True, "same_field_of_view": True,
                "descriptor_output_audit_passed": True, "no_pipeline_replacement": True}
    if any(geometry_receipt.get(k) != v for k, v in required.items()):
        return dict(status="DEFERRED", reason="Descriptor geometry/source-only gate failed", parents=parents)
    return dict(status="READY", parents=parents, cache_manifest=str(cache), cache_sha256=sha256(cache))


def cost_inventory(feature_width=4):
    p = feature_width + 1
    return dict(feature_width=feature_width, C0_coefficients=0, C1_coefficients=p,
                C2_coefficients=p + 12, C3_C4_coefficients=13 * p,
                fp32_coefficient_bytes_C3=4 * 13 * p,
                stored_statistics_bytes_fp32=4 * (2 * feature_width + 24),
                dynamic_head_construction_MACs=12 * p,
                excluded_from_MAC_count="Descriptor pixel reductions, standardization/clipping, native folding, and inference; must be measured end to end before frontier claims",
                strict_integer=False, latency_measured=False)


def fit_source_block(source_rows, source_evaluator, *, provenance):
    """Complete source-array probe, without extracting pixels or opening target labels.

    Each seed's rows has features, margin0, teacher_q, tile_ids,
    descriptors_by_tile, descriptor_tile_ids, and coordinate_hash. The evaluator
    callback returns equal_event_water_IoU from S11 validation plus its counts.
    Extraction/validation and evaluation still need their separate budget gate.
    """
    import common11
    common11.source_access_guard()
    if set(source_rows) != set(SEEDS):
        raise ValueError("Require all four frozen R4_micro KDoff parents")
    if (provenance.get("fit_split") != "GDACS_TRAIN"
            or provenance.get("selection_split") != "S11_VALIDATION"
            or provenance.get("input_masks_only") is not True
            or provenance.get("geometry_audited") is not True
            or provenance.get("estimated_reserved_core_hours", 13) > 12):
        raise ValueError("Require audited source-only inputs and the complete 12-core-hour budget gate")
    if len({source_rows[s]["coordinate_hash"] for s in SEEDS}) != 1:
        raise ValueError("All seeds and forms must share sampled input coordinates")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    fits, evaluations, selections, statistics = {}, {}, {}, {}
    for seed in SEEDS:
        rows = source_rows[seed]
        tile_ids = rows["tile_ids"]
        descriptor_ids = list(rows["descriptor_tile_ids"])
        if len(descriptor_ids) > 512 or any(n > 256 for n in np.unique(tile_ids, return_counts=True)[1]):
            raise ValueError("Source sampling cap exceeded")
        stats = train_statistics(rows["features"], rows["descriptors_by_tile"], tile_ids,
                                 descriptor_ids, split="GDACS_TRAIN")
        statistics[seed] = stats
        phi, z_tiles = standardize(rows["features"], rows["descriptors_by_tile"], stats)
        index = {name: i for i, name in enumerate(descriptor_ids)}
        row_index = np.array([index[name] for name in tile_ids])
        shuffled, permutation = shuffled_descriptors(z_tiles, descriptor_ids)
        weights = equal_tile_weights(tile_ids)
        fits[seed], evaluations[seed] = {}, {}
        evaluations[seed]["C0"] = source_evaluator(seed, "C0", np.zeros(0), stats)
        for form in FORMS[1:]:
            # Only C4 fit uses the shuffled descriptor. The evaluator receives
            # actual input descriptors and must never shuffle inference context.
            z = shuffled[row_index] if form == "C4" else z_tiles[row_index]
            design = design_matrix(phi, z, form)
            fits[seed][form], evaluations[seed][form] = {}, {}
            for lam in LAMBDAS:
                theta, diagnostic = fit_newton(design, rows["margin0"], rows["teacher_q"], weights, lam)
                fits[seed][form][lam] = dict(theta=theta, solver=diagnostic,
                                             descriptor_permutation=permutation if form == "C4" else None)
                evaluations[seed][form][lam] = source_evaluator(seed, form, theta, stats)
    for form in FORMS[1:]:
        scores = {lam: {seed: evaluations[seed][form][lam]["equal_event_water_IoU"]
                        for seed in SEEDS} for lam in LAMBDAS}
        lam, mean_scores = source_select_regularization(scores)
        selections[form] = dict(regularization=lam, four_seed_source_means=mean_scores)
    return dict(fits=fits, statistics=statistics, source_evaluations=evaluations,
                selections=selections, provenance=provenance,
                source_only=True, external_evaluation_performed=False,
                wall_seconds=time.monotonic() - start_wall,
                process_cpu_seconds=time.process_time() - start_cpu,
                C4_interpretation="Equal-parameter association control: shuffled TRAIN descriptors; actual descriptors at inference. Not a full static-architecture benchmark.")


def action(output, v10_root):
    """Finite optional-block admission: never leave final evaluation waiting."""
    import common11
    common11.source_access_guard()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    result = dependency_status(v10_root)
    prior = result["status"]
    # The optional decision occurs after the core decision. Unavailable optional
    # dependencies are deferred, not a new waiting stage or a failed experiment.
    result["dependency_status"] = prior
    result["status"] = "DEFERRED"
    if prior == "WAITING_DEPENDENCY":
        result["reason"] = "Four completed R4_micro KDoff parents and verified V10 teacher cache unavailable at optional decision"
    result.update(outcome="not_executed", package="independent_conditioner", forms=list(FORMS),
                  seeds=list(SEEDS), solver_implemented=True, source_probe_executed=False,
                  backbone_training=False, cost_inventory=cost_inventory(),
                  wall_seconds=time.monotonic() - start_wall,
                  process_cpu_seconds=time.process_time() - start_cpu,
                  cost_limit_reserved_core_hours=12,
                  action="Retain solver/tests; do not substitute architecture, incomplete seeds, or label-derived descriptor masks")
    common11.save(Path(output), result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["dependencies", "action"], nargs="?", default="dependencies")
    parser.add_argument("--v10-root", default=str(Path(__file__).resolve().parents[2] / "research_v10_gpu_20261013T183509Z"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "conditioner_receipt.json"))
    args = parser.parse_args()
    result = action(args.output, args.v10_root) if args.command == "action" else dependency_status(args.v10_root)
    print(json.dumps(result, indent=2))
