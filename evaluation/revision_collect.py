"""Collect an explicit frozen evaluator cohort, retaining failures and null cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METRICS = ("mIoU", "water_IoU", "background_IoU", "water_precision", "water_recall", "water_F1")


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def axis(condition):
    if condition == "clean":
        return "clean", 0.0
    if re.fullmatch(r"g[0-9.]+", condition):
        return "gain", float(condition[1:])
    if re.fullmatch(r"h[0-9.]+", condition):
        return "haze", float(condition[1:])
    match = re.fullmatch(r"t1(vis|tilt)([+-])(\d+)", condition)
    if match:
        return "perband_" + match[1], int(match[3]) / 100 * (1 if match[2] == "+" else -1)
    return condition, None


def collect_matched(spec_path, out):
    """Export only the predeclared pairs; each metric comes from one evaluator.

    Missing/failed rows remain visible. Seeds are never pooled as an ensemble.
    This supplements the existing count collector rather than changing its
    archived cohort rules.
    """
    import numpy as np
    from evaluation.revision_eval import scores
    from evaluation.stats_scene_level import boot_ci
    spec = json.loads(spec_path.read_text())
    if len({r['run_id'] for r in spec['runs']}) != len(spec['runs']):
        raise ValueError('Repeated run identities in comparison specification')
    for contrast in spec['comparisons']:
        if len({p['seed'] for p in contrast['pairs']}) != len(contrast['pairs']):
            raise ValueError('Repeated seed in a paired comparison')
    runs, rows, counts, event_rows, requirements = {}, [], [], [], []
    for item in spec["runs"]:
        data, training = {}, {}
        for key, dest in (("evaluation", data), ("training_receipt", training)):
            if item.get(key) and Path(item[key]).is_file():
                dest.update(json.loads(Path(item[key]).read_text()))
        meta = {k: item.get(k) for k in ("run_id", "architecture", "variant", "budget", "seed")}
        checked = training.get("checkpoint_checks", {})
        meta.update(selected_epoch=checked.get("best", {}).get("epochs_completed"),
                    optimizer_steps=checked.get("last", {}).get("optimizer_steps"),
                    initial_model_sha256=checked.get("best", {}).get("initial_model_sha256"),
                    checkpoint_sha256=data.get("identity", {}).get("checkpoint_sha256"),
                    source=item.get("evaluation"))
        cells = dict(data.get("cells", {}))
        for key in item.get('canonical_expected_cells', []):
            cells.setdefault(key, {})
        for extra, dataset in (("native", "wf"), ("sturm_valid", "sturm")):
            if not item.get(extra):
                continue
            corrected = json.loads(Path(item[extra]).read_text()) if Path(item[extra]).is_file() else {}
            requirements.append({"run_id": item["run_id"], "artifact": item[extra],
                                 "status": corrected.get("status", "missing")})
            from evaluation.revision_native_border import WF_PROTOCOL, STURM_PROTOCOL
            protocol = WF_PROTOCOL if dataset == "wf" else STURM_PROTOCOL
            expected = item.get("sturm_conditions", ["clean"]) if dataset == "sturm" else ["clean"]
            for condition in expected:
                name = protocol + (":" + condition if dataset == "sturm" else "")
                cells.setdefault(dataset + "__" + name, {"status": "missing", "source": item[extra]})
            if corrected.get("status") == "complete":
                corrections = corrected.get("cells", {"clean": corrected})
                for condition, correction in corrections.items():
                    result = correction["native"]
                    protocol = corrected["protocol"] + (":" + condition if dataset == "sturm" else "")
                    cells[dataset + "__" + protocol] = {
                        "micro": result["micro"], "scene": result["map"], "event": result["event"],
                        "scene_metrics": {k: scores(v) for k, v in result["map"].items()},
                        "event_metrics": {k: scores(v) for k, v in result["event"].items()},
                        "source": item[extra], "status": "complete"}
        for dataset in ("s11", "sturm", "wf"):
            cells.setdefault(dataset + "__clean", {})
        for key, cell in cells.items():
            dataset, condition = key.split("__", 1)
            source = cell.get("source", meta["source"])
            status = cell.get("status", "complete" if cell and data.get("status") == "complete"
                              else "available_partial_run" if cell else "missing")
            for metric in METRICS:
                rows.append({**meta, "source": source, "dataset": dataset, "condition": condition,
                             "metric": metric, "value": number(cell.get("micro", {}).get(metric)), "status": status})
            for level in ("scene", "event"):
                for group, confusion in cell.get(level, {}).items():
                    a = np.asarray(confusion)
                    counts.append({**meta, "source": source, "dataset": dataset, "condition": condition,
                                   "level": level, "unit": group, "TP": int(a[1, 0]), "FP": int(a[1, 1]),
                                   "FN": int(a[1, 2]), "TN": int(a[0, 0]),
                                   "valid_pixels": int(a[0, 0] + a[1].sum()), **scores(a)})
            if cell.get("scene"):
                a = sum(np.asarray(v) for v in cell["scene"].values())
                counts.append({**meta, "source": source, "dataset": dataset, "condition": condition,
                               "level": "pooled", "unit": "all", "TP": int(a[1, 0]), "FP": int(a[1, 1]),
                               "FN": int(a[1, 2]), "TN": int(a[0, 0]),
                               "valid_pixels": int(a[0, 0] + a[1].sum()), **scores(a)})
        runs[item["run_id"]] = {"item": item, "cells": cells, "initial_model_sha256": meta['initial_model_sha256']}
    paired = []
    initialization_checks = []
    for contrast in spec["comparisons"]:
        for pair in contrast["pairs"]:
            left, right = runs[pair["left"]], runs[pair["right"]]
            eligible = True
            if contrast.get('require_matching_initialization'):
                ah, bh = left['initial_model_sha256'], right['initial_model_sha256']
                eligible = bool(ah and bh and ah == bh)
                initialization_checks.append({'contrast': contrast['name'], 'seed': pair['seed'],
                    'left': ah, 'right': bh, 'status': 'verified' if eligible else 'missing-or-different'})
            keys = contrast.get('cell_keys', sorted(set(left["cells"]) | set(right["cells"])))
            for key in keys:
                dataset, condition = key.split("__", 1)
                clean_variant = (condition == "clean" or condition.startswith("P3_WF_NATIVE_") or
                                 (condition.startswith("P3_STURM_") and condition.endswith(":clean")))
                if contrast["architecture"] != "efficientvit_b0" and not clean_variant:
                    continue  # No new secondary-architecture robustness sweep.
                a, b = left["cells"].get(key, {}), right["cells"].get(key, {})
                meta = {"contrast": contrast["name"], "architecture": contrast["architecture"],
                        "budget": contrast["budget"], "seed": pair["seed"], "dataset": dataset,
                        "condition": condition, "left_run": pair["left"], "right_run": pair["right"]}
                for metric in METRICS:
                    av, bv = number(a.get("micro", {}).get(metric)), number(b.get("micro", {}).get(metric))
                    paired.append({**meta, "metric": metric, "left": av, "right": bv,
                                   "difference": av - bv if eligible and av is not None and bv is not None else None,
                                   "status": 'complete' if eligible and av is not None and bv is not None else 'missing-or-ineligible'})
                    for event in sorted(set(a.get("event_metrics", {})) & set(b.get("event_metrics", {}))):
                        av = number(a["event_metrics"][event].get(metric))
                        bv = number(b["event_metrics"][event].get(metric))
                        event_rows.append({**meta, "event": event, "metric": metric,
                                           "difference": av - bv if eligible and av is not None and bv is not None else None})
    def summaries(source, keys, value):
        grouped = {}
        for row in source:
            grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
        result = []
        for group, records in sorted(grouped.items()):
            values = [r[value] for r in records if r[value] is not None]
            expected = 1 if group[0] in ("teacher", "index:mndwi_zero") else 4
            result.append({**dict(zip(keys, group)), "n_available_seeds": len(values), "expected_seeds": expected,
                           "complete_declared_seed_row": len(values) == expected,
                           "mean": float(np.mean(values)) if len(values) == expected else None,
                           "sample_SD_ddof1": float(np.std(values, ddof=1)) if len(values) == expected and expected > 1 else None})
        return result
    seed_summary = summaries(rows, ["architecture", "variant", "budget", "dataset", "condition", "metric"], "value")
    pair_summary = summaries(paired, ["contrast", "architecture", "budget", "dataset", "condition", "metric"], "difference")
    event_summary = []
    groups = {}
    for row in event_rows:
        key = tuple(row[k] for k in ("contrast", "architecture", "budget", "dataset", "condition", "metric"))
        groups.setdefault(key, {}).setdefault(row["event"], []).append(row["difference"])
    for key, events in sorted(groups.items()):
        # The event resampling averages each seed's metric differences; it never
        # averages predictions or treats four seeds as geographic replicates.
        values = [float(np.mean(v)) for v in events.values() if len(v) == 4 and all(x is not None for x in v)]
        low, high = boot_ci(np.asarray(values)) if len(values) >= 5 else (None, None)
        event_summary.append({**dict(zip(("contrast", "architecture", "budget", "dataset", "condition", "metric"), key)),
                              "n_events_with_four_seed_pairs": len(values),
                              "mean_event_metric_difference": float(np.mean(values)) if values else None,
                              "ci95_low": low, "ci95_high": high,
                              "uncertainty": "Event resampling conditional on these four checkpoints; separate from sample seed SD"})
    tables = {"matched_by_seed.csv": rows, "confusion_counts.csv": counts,
              "paired_by_seed.csv": paired, "matched_seed_summary.csv": seed_summary,
              "paired_seed_summary.csv": pair_summary, "paired_event_differences.csv": event_rows,
              "paired_event_summary.csv": event_summary}
    for filename, table in tables.items():
        if table:
            write_csv(out / filename, table, list(table[0]))
    (out / "matched_provenance.json").write_text(json.dumps({
        "status": "complete" if all(r["status"] == "complete" for r in rows) and
                  all(r["status"] == "complete" for r in requirements) and
                  all(r['status'] == 'complete' for r in paired) else "partial",
        "comparison_spec": str(spec_path), "comparison_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "seed_SD": "sample SD, ddof=1; one seed has undefined SD",
        "archived_SD": "Historical aggregate scripts used population SD where np.std defaulted to ddof=0; retained as archived, not a performance difference",
        "focal_comparison": "Full-weight focal versus 0.5 focal + 0.5 KL; this tests the weighted recipe and changes the hard coefficient",
        "mask_and_protocol": "Count rows from their own checkpoint/evaluator only; legacy/native/corrected scores remain distinct",
        "scope": spec.get("scope"), "required_corrections": requirements,
        "initialization_checks": initialization_checks,
        "tables": list(tables)}, indent=2, allow_nan=False) + "\n")


def collect_original_reproduction(spec_path, out):
    """Keep printed references, old scalar aggregates and new count domains apart."""
    import numpy as np
    spec = json.loads(spec_path.read_text())
    references = list(csv.DictReader(Path(spec['printed_table']).open()))
    small = list(csv.DictReader(Path(spec['printed_N100']).open()))
    rows, groups = [], {}
    prefixes = {'s11': 's11_test', 'sturm': 'sturm', 'wf': 'wf_fo'}
    labels = {'ce': 'CE', 'dfkd': 'Logit-KD', 'ofa': 'OFA (ours)',
              'ce_unet': 'CE', 'dfkd_unet': 'Logit-KD'}
    for item in spec['runs']:
        old = json.loads(Path(item['archive_metrics']).read_text()) if Path(item['archive_metrics']).exists() else {}
        new = json.loads(Path(item['evaluation']).read_text()) if Path(item['evaluation']).exists() else {}
        for dataset, prefix in prefixes.items():
            for metric, suffix in [('mIoU', 'mIoU'), ('water_IoU', 'flood_IoU')]:
                archived = number(old.get(prefix + '_' + suffix))
                cell = new.get('cells', {}).get(dataset + '__clean', {})
                rerun = number(cell.get('micro', {}).get(metric))
                key = (item['architecture'], item['declared_objective'], item['budget'], dataset, metric)
                groups.setdefault(key, []).append((archived, rerun))
                rows.append({'original_id': item['original_id'], 'architecture': item['architecture'],
                             'declared_objective': item['declared_objective'], 'budget': item['budget'],
                             'seed': item['seed'], 'dataset': dataset, 'metric': metric,
                             'original_scalar': archived, 'new_legacy_count_domain': rerun,
                             'new_minus_original_scalar': rerun - archived if archived is not None and rerun is not None else None,
                             'checkpoint_sha256': item['checkpoint_sha256'],
                             'original_scalar_source': item['archive_metrics'], 'new_source': item['evaluation'],
                             'status': 'complete' if rerun is not None and new.get('status') == 'complete' else 'missing',
                             'historical_identity': item['historical_objective_and_seed_identity']})
    aggregate = []
    benchmark = {'s11': 'S11', 'sturm': 'STURM', 'wf': 'WF-fo'}
    for key, records in groups.items():
        arch, objective, budget, dataset, metric = key
        old = [a for a, _ in records if a is not None]
        new = [b for _, b in records if b is not None]
        printed, sd, printed_source = None, None, None
        if metric == 'mIoU' and budget == 2500:
            model = ('EfficientViT-B0' if arch == 'efficientvit_b0' else 'UNet-ResNet50') + ', ' + labels[objective]
            match = next((r for r in references if r['model'] == model and r['benchmark'] == benchmark[dataset]), None)
            if match:
                printed, sd = float(match['mean']), float(match['sd_ddof1'])
                printed_source = spec['printed_table']
        elif metric == 'mIoU' and budget == 100:
            objective_label = 'OFA' if objective == 'ofa' else labels[objective]
            match = next((r for r in small if int(r['N']) == 100 and r['objective'] == objective_label
                          and r['benchmark'] == prefix_for_small(dataset)), None)
            if match:
                printed, sd = float(match['mean_micro_mIoU']), float(match['sd'])
                printed_source = spec['printed_N100']
        complete = len(records) == len(old) == len(new) == 4
        old_mean = float(np.mean(old)) if len(old) == 4 else None
        new_mean = float(np.mean(new)) if len(new) == 4 else None
        aggregate.append({'architecture': arch, 'declared_objective': objective, 'budget': budget,
                          'dataset': dataset, 'metric': metric, 'expected_seeds': 4,
                          'original_available_seeds': len(old), 'new_available_seeds': len(new),
                          'complete_declared_seed_row': complete, 'printed_mean': printed, 'printed_reported_SD': sd,
                          'printed_source': printed_source, 'recovered_original_mean': old_mean,
                          'recovered_original_sample_SD': float(np.std(old, ddof=1)) if len(old) == 4 else None,
                          'new_legacy_mean': new_mean, 'new_legacy_sample_SD': float(np.std(new, ddof=1)) if len(new) == 4 else None,
                          'new_minus_recovered_original': new_mean - old_mean if complete else None,
                          'new_minus_printed': new_mean - printed if complete and printed is not None else None,
                          'rounding_agreement_is_not_bitwise_equivalence': True})
    write_csv(out / 'original_score_reproduction_by_seed.csv', rows, list(rows[0]))
    write_csv(out / 'original_score_reproduction_summary.csv', aggregate, list(aggregate[0]))
    sources = {spec_path: hashlib.sha256(spec_path.read_bytes()).hexdigest()}
    for path in [spec['printed_table'], spec['printed_N100'], *[r['archive_metrics'] for r in spec['runs']]]:
        p = Path(path)
        if p.exists():
            sources[p] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'original_score_reproduction_provenance.json').write_text(json.dumps({
        'status': 'complete' if all(r['status'] == 'complete' for r in rows) else 'partial',
        'sources': {str(k): v for k, v in sources.items()},
        'definition': 'Printed paper-source references, saved original scalar aggregates and strict-loaded original-checkpoint new legacy count metrics are separate columns. Corrected protocol metrics live in matched_by_seed/count tables.',
        'limits': ['No archived checkpoint-to-prediction/mask/source hash chain; associated scalar references cannot establish bitwise old-environment equivalence.',
                   'Original scalar precision/recall are not fabricated from IoU; new precision/recall remain in their own fresh count-domain rows.',
                   'Copied checkpoints are evaluation references, not current-machine training reproductions.']}, indent=2, allow_nan=False) + '\n')


def prefix_for_small(dataset):
    return {'s11': 's11_test_mIoU', 'sturm': 'sturm_mIoU', 'wf': 'wf_fo_mIoU'}[dataset]


def collect_dry(paths, out):
    rows, summaries, references, missing = [], [], set(), []
    for path in paths:
        if not path.is_file():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text())
        if data.get('status') != 'complete':
            missing.append(str(path))
            continue
        spec = data['spec']
        identity = data['identity']
        references.add(identity['reference_manifest_sha256'])
        meta = {'run_id': spec['run_id'], 'architecture': spec['arch'], 'family': spec['family'],
                'seed': spec['seed'], 'checkpoint_sha256': identity['checkpoint_sha256'],
                'reference_manifest_sha256': identity['reference_manifest_sha256'], 'source': str(path)}
        rows.extend({**meta, **row} for row in data['scenes'])
        summaries.append({**meta, 'n_chips': data['n_chips'], 'n_event_groups': data['n_event_groups'],
                          'false_positive_pixel_rate': data['false_positive_pixel_rate'], **data['scene_alarm_frequency']})
    if len(references) > 1:
        raise RuntimeError('Dry outputs use different frozen reference sets')
    if rows:
        write_csv(out / 'dry_by_chip.csv', rows, list(rows[0]))
    if summaries:
        write_csv(out / 'dry_by_checkpoint.csv', summaries, list(summaries[0]))
    (out / 'dry_provenance.json').write_text(json.dumps({'status': 'partial' if missing else 'complete',
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
        'missing': missing, 'reference_hashes': sorted(references),
        'interpretation': 'Fixed reviewed dry chips; four student seeds are distinct checkpoints, teacher/MNDWI are fixed comparators. Within-benchmark few-event evidence does not establish fully dry acquisitions or operational landscape reliability.'}, indent=2) + '\n')


def queue_handoff(manifest_path, reviewer_path, status_path, result_dir=None):
    """Refresh the existing operational ledger without claiming pending results."""
    from collections import Counter
    from datetime import datetime, timezone
    from experiments import revision_queue as q
    manifest = q.load_manifest(manifest_path)
    state_path = manifest_path.with_name("manifest.state.json")
    state = q.read_json(state_path)
    reviewer = q.read_json(reviewer_path)
    cost_path = manifest_path.parent / "cost_before_launch.json"
    cost = q.read_json(cost_path)
    estimates = {r["job"]: r for r in cost["jobs"]}
    rows, remaining = [], {"original": 0.0, "additional": 0.0}
    completed = {"original": 0.0, "additional": 0.0}
    for job in manifest["jobs"]:
        record = state["jobs"][job["id"]]
        attempts = record["attempts"]
        seconds = 0.0
        if attempts and attempts[-1].get("started_at"):
            attempt = attempts[-1]
            end = datetime.fromisoformat(attempt["finished_at"]) if attempt.get("finished_at") else datetime.now(timezone.utc)
            seconds = (end - datetime.fromisoformat(attempt["started_at"])).total_seconds()
        estimate = estimates.get(job["id"], {})
        category = estimate.get("category", "original" if not job["id"].startswith("P3_") else "additional")
        if job.get("gpu") and record["status"] == "succeeded":
            completed[category] += seconds / 3600
        if record["status"] in {"pending", "running", "launching"} and estimate:
            remaining[category] += max(0, estimate["estimated_GPU_seconds"] - seconds) / 3600
        row = {"job": job["id"], "kind": job["kind"], "status": record["status"],
               "category": category, "gpu": job.get("gpu", False), "reason": record.get("reason"),
               "block_kind": record.get("block_kind"), "attempts": len(attempts),
               "elapsed_seconds": seconds, "identity_sha256": record["identity_sha256"],
               "deps": json.dumps(job["deps"]), "argv": json.dumps(job.get("argv")),
               "recovered_validation_only": bool(record.get("validation_recovery"))}
        rows.append(row)
    write_csv(manifest_path.parent / "job_matrix.csv", rows, list(rows[0]))
    controls = {action: [str(Path(sys.executable).resolve()), str(manifest_path.parent / "revision_queue.py"),
                         action, "--manifest", str(manifest_path)] for action in ["status", "pause", "resume", "stop"]}
    scope = manifest["scientific_scope"]
    training_ids = scope.get("training_job_ids")
    computation_ids = scope.get("computation_job_ids")
    train_records = [r for j, r in state["jobs"].items()
                     if j in training_ids] if training_ids is not None else [r for j, r in state["jobs"].items() if j.startswith("P3_train_")]
    computations = [state["jobs"][j] for j in computation_ids] if computation_ids is not None else [r for j, r in state["jobs"].items() if j.startswith(("P3_train_", "P3_eval_", "P3_wf_native_", "P3_sturm_valid_", "R_wf_native_", "R_sturm_valid_"))]
    complete = all(r["status"] == "succeeded" for r in computations)
    handoff = {"captured_at": q.now(), "status": "evidence-ready" if complete else "needs-experiment",
               "manifest": str(manifest_path), "state": str(state_path),
               "manifest_sha256": q.digest(manifest), "worker": state.get("worker"),
               "worker_alive": q.alive(state.get("worker")), "worker_status": state["worker_status"],
               "heartbeat_at": state.get("heartbeat_at"), "paused": state["paused"],
               "stop_requested": state["stop_requested"], "job_states": dict(Counter(r["status"] for r in state["jobs"].values())),
               "new_training_states": dict(Counter(r["status"] for r in train_records)),
               "finite_matrix": manifest["scientific_scope"], "job_matrix": str(manifest_path.parent / "job_matrix.csv"),
               "controls": controls, "remaining_GPU_hours": remaining,
               "completed_measured_GPU_hours": completed,
               "ideal_remaining_two_GPU_wall_hours": sum(remaining.values()) / 2,
               "time_estimate_limits": cost["uncertainty"], "prelaunch_cost": str(cost_path),
               "reviewer_classification": {"path": str(reviewer_path), "sha256": q.file_hash(reviewer_path),
                                           "categories": reviewer["category_counts"], "concern_status_counts": reviewer["concern_status_counts"],
                                           "audit_completion_is_not_concern_resolution": True},
               "result_tables": str(scope.get("primary_result_dir") or result_dir or manifest_path.parents[1] / "analysis_prompt3"),
               "pending_author_history": reviewer["author_history"], "dry_reference": reviewer["dry_reference"],
               "deployment": reviewer["deployment"], "missing_sources": reviewer["missing_sources"],
               "blocked_jobs": [{"job": r["job"], "reason": r["reason"]} for r in rows if r["status"] in {"blocked", "failed"}],
               "completion_class": ("computation-complete with explicit external-input blockers" if complete and any(r["status"] in {"blocked", "failed"} for r in rows)
                                    else "complete for the declared runnable scope" if complete else "running/partial"),
               "persistence": "Existing detached worker with inherited GPU/attempt locks, atomic state and PID/start-ticks/boot identity. Prior parent-process sentinel reused. Jobs do not run during host sleep/power-off; no reboot survival claim.",
               "next_phase": "Stop adding experiments when this finite queue drains; use completed controls and explicit source/reference/device limits for manuscript and reviewer-response work."}
    q.atomic_json(status_path, handoff)
    return handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=[], type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--comparison_spec", type=Path)
    parser.add_argument("--queue_manifest", type=Path)
    parser.add_argument("--reviewer_status", type=Path)
    parser.add_argument("--status_out", type=Path)
    parser.add_argument("--status_only", action="store_true")
    parser.add_argument("--matched_only", action="store_true")
    parser.add_argument("--original_reference_spec", type=Path)
    parser.add_argument("--dry_only", action="store_true")
    args = parser.parse_args()
    if args.dry_only:
        if not args.inputs or not args.out_dir:
            parser.error("--dry_only requires --inputs and --out_dir")
        out = args.out_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        collect_dry([p.resolve() for p in args.inputs], out)
        if args.queue_manifest:
            queue_handoff(args.queue_manifest.resolve(), args.reviewer_status.resolve(), args.status_out.resolve(), out)
        return
    if args.status_only:
        handoff = queue_handoff(args.queue_manifest.resolve(), args.reviewer_status.resolve(), args.status_out.resolve(), args.out_dir)
        print(json.dumps({"status": handoff["status"], "job_states": handoff["job_states"],
                          "remaining_GPU_hours": handoff["remaining_GPU_hours"]}))
        return
    if args.matched_only:
        if not args.comparison_spec or not args.out_dir:
            parser.error("--matched_only requires --comparison_spec and --out_dir")
        out = args.out_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        collect_matched(args.comparison_spec.resolve(), out)
        if args.original_reference_spec:
            collect_original_reproduction(args.original_reference_spec.resolve(), out)
        if args.queue_manifest:
            queue_handoff(args.queue_manifest.resolve(), args.reviewer_status.resolve(), args.status_out.resolve(), out)
        return
    if not args.inputs or not args.out_dir:
        parser.error("--inputs and --out_dir are required for metric collection")
    paths = [p.resolve() for p in args.inputs]
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate input paths would duplicate checkpoint weighting")
    import numpy as np
    from evaluation.stats_scene_level import boot_ci
    from figures.fig_style import STYLE, DS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, groups, paired, sources, failures = [], [], [], [], []
    for path in paths:
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError) as error:
            failures.append({"source": str(path), "status": "unreadable", "error": str(error)})
            for metric in METRICS:
                rows.append({"source": str(path), "family": path.stem, "cohort": "unknown",
                             "cohort_key": "unknown", "dataset": "unknown", "condition": "unknown",
                             "axis": "unknown", "metric": metric, "status": "unreadable",
                             "run_status": "unknown", "value": None, "declared_cells_missing_or_null": 1,
                             "error": str(error)})
            continue
        spec = data.get("spec", {})
        family = spec.get("family", spec.get("run_id", path.stem))
        cohort = str(spec.get("cohort", spec.get("provenance", "unspecified cohort")))
        # Identical family names do not authorize pooling distinct recipe identities.
        fp = spec.get("fingerprints", {})
        recipe = {k: fp[k] for k in ("protocol", "recipe", "environment", "data", "code") if k in fp}
        cohort_key = hashlib.sha256(json.dumps([cohort, recipe], sort_keys=True).encode()).hexdigest()[:10]
        meta = {"source": str(path), "family": family, "cohort": cohort,
                "cohort_key": cohort_key, "seed": spec.get("seed", "unspecified"),
                "checkpoint_sha256": data.get("identity", {}).get("checkpoint_sha256")}
        sources.append({**meta, "sha256": hashlib.sha256(raw).hexdigest(),
                        "status": data.get("status", "unknown"), "spec": spec,
                        "identity": data.get("identity", {}), "error": data.get("error")})
        cells = data.get("cells", {})
        keys = list(cells)
        for ds in spec.get("datasets", []):
            for cond in spec.get("conditions", ["clean"]):
                key = ds + "__" + cond
                if key not in keys:
                    keys.append(key)
        source_rows = []
        for key in keys:
            cell = cells.get(key, {})
            ds, cond = key.split("__", 1)
            clean = cells.get(ds + "__clean", {})
            status = cell.get("status", "available" if key in cells else "missing")
            skipped = ((ds == "s11" and cond != "clean") or
                       (ds != "sturm" and cond == "g0.70") or
                       (ds != "sturm" and cond == "g0.90" and
                        not spec.get("complete_reported_scalar_grid", False)))
            if key not in cells and skipped:
                status = "protocol_not_applicable"
            ax, severity = axis(cond)
            for metric in dict.fromkeys([*METRICS, *cell.get("micro", {})]):
                value = number(cell.get("micro", {}).get(metric))
                base = number(clean.get("micro", {}).get(metric))
                drop = base - value if base is not None and value is not None else None
                row = {**meta, "dataset": ds, "condition": cond, "axis": ax,
                       "severity": severity, "metric": metric, "status": status,
                       "run_status": data.get("status", "unknown"), "value": value,
                       "clean": base, "absolute_drop": drop,
                       "relative_drop": drop / base if drop is not None and base else None,
                       "n_tiles": cell.get("n_tiles"), "n_scenes": len(cell.get("scene", {})),
                       "n_events": len(cell.get("event", {})), "error": cell.get("error", data.get("error"))}
                source_rows.append(row)
                if metric in {"mIoU", "water_IoU"}:
                    current_events = cell.get("event_metrics", {})
                    clean_events = clean.get("event_metrics", {})
                    common = sorted(set(current_events) & set(clean_events))
                    delta = [number(clean_events[g].get(metric)) - number(current_events[g].get(metric))
                             for g in common if number(clean_events[g].get(metric)) is not None and number(current_events[g].get(metric)) is not None]
                    lo, hi = boot_ci(np.asarray(delta)) if len(delta) >= 5 else (None, None)
                    paired.append({**meta, "dataset": ds, "condition": cond, "metric": metric,
                                   "n_events": len(delta), "mean_event_drop": float(np.mean(delta)) if delta else None,
                                   "ci95_low": lo, "ci95_high": hi,
                                   "uncertainty": "paired event macro drop conditional on this checkpoint; no training-seed inference"})
            for level in ("scene", "event"):
                for group_id, scores in cell.get(level + "_metrics", {}).items():
                    for metric, value in scores.items():
                        groups.append({**meta, "dataset": ds, "condition": cond, "level": level,
                                       "group_id": group_id, "metric": metric, "value": number(value)})
        for row in source_rows:
            same = [r for r in source_rows if r["dataset"] == row["dataset"] and r["metric"] == row["metric"]]
            for name in ("absolute_drop", "relative_drop"):
                values = [r[name] for r in same if r[name] is not None and r["status"] != "protocol_not_applicable"]
                row["worst_" + name] = max(values) if values else None
            row["declared_cells_missing_or_null"] = sum(r["value"] is None and r["status"] != "protocol_not_applicable" for r in same)
        rows.extend(source_rows)

    base_columns = ["source", "family", "cohort", "cohort_key", "seed", "checkpoint_sha256"]
    write_csv(out / "metrics.csv", rows, base_columns + ["dataset", "condition", "axis", "severity", "metric", "status", "run_status", "value", "clean", "absolute_drop", "relative_drop", "worst_absolute_drop", "worst_relative_drop", "declared_cells_missing_or_null", "n_tiles", "n_scenes", "n_events", "error"])
    write_csv(out / "group_metrics.csv", groups, base_columns + ["dataset", "condition", "level", "group_id", "metric", "value"])
    write_csv(out / "paired_event_drop.csv", paired, base_columns + ["dataset", "condition", "metric", "n_events", "mean_event_drop", "ci95_low", "ci95_high", "uncertainty"])
    plots = []
    for ds in sorted({r["dataset"] for r in rows}):
        for ax in ("gain", "haze", "perband_vis", "perband_tilt"):
            relevant = [r for r in rows if r["dataset"] == ds and r["metric"] == "mIoU" and r["axis"] in {ax, "clean"}]
            if not any(r["axis"] == ax and r["value"] is not None for r in relevant):
                continue
            fig, panels = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
            for metric, panel in zip(("mIoU", "water_IoU"), panels):
                all_rows = [r for r in rows if r["dataset"] == ds and r["metric"] == metric and r["axis"] in {ax, "clean"}]
                for family, ck in sorted({(r["family"], r["cohort_key"]) for r in all_rows}):
                    cells = [r for r in all_rows if r["family"] == family and r["cohort_key"] == ck]
                    x_values = sorted({(1.0 if ax == "gain" else 0.0) if r["axis"] == "clean" else r["severity"] for r in cells if r["severity"] is not None})
                    means, stds, counts = [], [], []
                    for x in x_values:
                        # g1.00/t1unit and clean describe one anchor; do not double-weight.
                        by_source = {}
                        for r in cells:
                            rx = (1.0 if ax == "gain" else 0.0) if r["axis"] == "clean" else r["severity"]
                            if rx == x and r["value"] is not None:
                                by_source[r["source"]] = r["value"]
                        values = list(by_source.values())
                        means.append(float(np.mean(values)) if values else np.nan)
                        stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else np.nan)
                        counts.append(len(values))
                    style_family = re.sub(r"_N\d+$", "", family)
                    label, color, linestyle, marker = STYLE.get(style_family, (family, None, "-", "o"))
                    label += " [" + cells[0]["cohort"] + "; checkpoints=" + str(max(counts, default=0)) + "]"
                    line, = panel.plot(x_values, means, label=label, color=color, linestyle=linestyle, marker=marker)
                    panel.fill_between(x_values, np.asarray(means) - stds, np.asarray(means) + stds, color=line.get_color(), alpha=0.12)
                panel.set(title=metric, xlabel={"gain": "Common scalar gain", "haze": "Additive haze", "perband_vis": "Signed visible-band gain delta", "perband_tilt": "Signed opposing visible/infrared gain delta"}[ax], ylabel="Score")
                panel.grid(alpha=0.2)
            panels[1].legend(fontsize=6)
            fig.suptitle(DS.get(ds, ds) + " · fixed evaluation inputs; shading = checkpoint SD")
            for extension in ("png", "pdf"):
                name = f"{ds}_{ax}.{extension}"
                fig.savefig(out / name, dpi=160)
                plots.append(name)
            plt.close(fig)
    provenance = {"status": "complete", "sources": sources, "failures": failures, "n_metric_rows": len(rows),
                  "plots": plots, "source_retention": "Original JSON files retain all scene/event confusion cells; paths and SHA256 recorded above",
                  "interpretation": "Explicit cohort only. Curves are available-checkpoint means, with checkpoint SD; unequal counts and missing cells are retained. Archived recipes/environments are references, not matched local training controls. Event CIs condition on each checkpoint; no equivalence or training-procedure superiority claim."}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    if args.comparison_spec:
        collect_matched(args.comparison_spec.resolve(), out)
    if args.original_reference_spec:
        collect_original_reproduction(args.original_reference_spec.resolve(), out)
    if args.queue_manifest:
        queue_handoff(args.queue_manifest.resolve(), args.reviewer_status.resolve(), args.status_out.resolve(), out)
    print(json.dumps({"status": "complete", "out_dir": str(out), "n_metric_rows": len(rows), "failures": len(failures), "plots": len(plots)}))


if __name__ == "__main__":
    main()
