"""Terminal-only compression bookkeeping, paired event analysis and audit tables."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re

import numpy as np

from experiments.revision_queue import atomic_json, file_hash, read_json
from evaluation.revision_eval import scores

METRICS = ('mIoU', 'water_IoU', 'background_IoU', 'water_precision', 'water_recall', 'water_F1')
SOURCE_DEPLOYMENT_TOLERANCE = .01


def _optional(path):
    try:
        return read_json(path) if Path(path).is_file() else {}
    except (ValueError, OSError) as error:
        return {'status': 'unreadable', 'reason': str(error)}


def _state(value):
    return value if isinstance(value, dict) else _optional(value) if value else {}


def _verify_file(path=None, sha=None, size=None):
    """Validate reported files, retaining missing hints as unavailable evidence."""
    if not any(value is not None for value in (path, sha, size)):
        return {'status': 'not_recorded', 'reason': 'No path or identity hints in receipt'}
    if not path or not sha:
        return {'status': 'unavailable', 'reason': 'Path and SHA256 are both required to verify the file'}
    path = Path(path).resolve()
    if not path.is_file():
        return {'status': 'unavailable', 'path': str(path), 'reason': 'Reported file is missing'}
    try:
        actual_size = path.stat().st_size; actual_sha = file_hash(path)
    except OSError as error:
        return {'status': 'unavailable', 'path': str(path), 'reason': str(error)}
    valid = actual_sha == sha and (size is None or actual_size == size)
    return {'status': 'verified' if valid else 'invalid', 'path': str(path),
            'actual_sha256': actual_sha, 'actual_bytes': actual_size,
            'recorded_sha256': sha, 'recorded_bytes': size,
            'reason': None if valid else 'SHA256 or byte size differs from completed receipt'}


def _deployment_summary(terminal, anchors, source_scores, frontier):
    candidates = []
    for record, receipt in terminal:
        for precision, measurement, simulation in (
            (record['precision'], receipt.get('artifact', {}), receipt.get('status_label') == 'simulation_only'),
            ('FP16', receipt.get('artifact', {}).get('float16_reference', {}), False)):
            if not measurement:
                continue
            exported = measurement.get('artifact', {}); runtime = measurement.get('runtime', {})
            integrity = measurement.get('collector_integrity', {'status': 'unavailable'})
            anchor = anchors.get(record['seed'])
            score = source_scores.get((receipt.get('checkpoint_sha256'), precision))
            candidate = {key: record[key] for key in ('job_id', 'graph', 'seed', 'method')}
            candidate.update(precision=precision, source_validation_mIoU=score,
                artifact_path=exported.get('path'), inference_artifact_bytes=integrity.get('actual_bytes'),
                artifact_integrity=integrity['status'], runtime_status=runtime.get('status'),
                artifact_scope='fake_quant_simulation_bytes_no_packed_compression' if simulation else 'native_float_inference_artifact',
                packed_compression=False, deployment_claim=False)
            if record['graph'] not in ('A0', 'A1', 'A2', 'A3'):
                reason = 'Different family/activation; no matching ReLU6 A0 anchor'
            elif not anchor:
                reason = 'Matching same-seed completed and verified A0 FP anchor unavailable'
            elif record['checkpoint_integrity']['status'] != 'verified' or integrity['status'] != 'verified':
                reason = 'Checkpoint or exported artifact identity not verified'
            elif runtime.get('status') != 'measured' or exported.get('status') != 'exported_prediction_validated':
                reason = 'No prediction-validated measured inference artifact'
            elif score is None:
                reason = 'Source validation for this exact checkpoint and precision unavailable'
            elif score < anchor['source_validation_mIoU'] - SOURCE_DEPLOYMENT_TOLERANCE:
                reason = 'Outside predeclared source-validation tolerance'
            else:
                reason = None
            candidate.update(eligible=reason is None, reason=reason,
                anchor_source_validation_mIoU=anchor['source_validation_mIoU'] if anchor else None)
            candidates.append(candidate)
    eligible = [row for row in candidates if row['eligible']]
    order = lambda row: (row['inference_artifact_bytes'], row['job_id'], row['precision'])
    smallest = lambda rows: min(rows, key=order) if rows else None
    return {'status': 'selected' if eligible else 'unavailable', 'tolerance': SOURCE_DEPLOYMENT_TOLERANCE,
            'selection_rule': 'source-validation mIoU >= matching same-seed ReLU6 A0 FP anchor minus 0.01; minimize verified measured inference artifact bytes',
            'selection_data': 'S11 source-validation only; no terminal test/OOD accuracy used',
            'anchors': list(anchors.values()), 'candidates': candidates,
            'smallest_measured_inference_artifact': smallest(eligible),
            'smallest_native_FP_artifact': smallest([r for r in eligible if r['artifact_scope'] == 'native_float_inference_artifact']),
            'smallest_fakequant_simulation_artifact': smallest([r for r in eligible if r['artifact_scope'].startswith('fake_quant')]),
            'complete_frontier': frontier, 'packed_lowbit_deployment_established': False}


def _support(cell):
    raw = np.asarray(cell)
    if raw.shape != (2, 3) or not np.isfinite(raw).all() or (raw < 0).any() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError('Expected nonnegative integer [class, TP/FP/FN] counts')
    a = raw.astype(np.int64)
    return tuple((a[:, 0] + a[:, 2]).tolist())


def _paired(left, right):
    """Pair all event IDs and GT support; bootstrap events, never pixels."""
    if set(left) != set(right):
        return {'status': 'support_mismatch', 'reason': 'different event IDs'}
    for key in left:
        if _support(left[key]) != _support(right[key]):
            return {'status': 'support_mismatch', 'reason': f'different ground truth support: {key}'}
    differences = []
    for key in sorted(left):
        l, r = scores(left[key])['mIoU'], scores(right[key])['mIoU']
        if l is not None and r is not None:
            differences.append(l - r)
    if not differences:
        return {'status': 'insufficient', 'n_events': 0}
    # Reuse the repository's event/scene paired bootstrap with isolated RNG.
    cwd = Path.cwd()
    try:
        from evaluation import stats_scene_level as bootstrap
    finally:
        os.chdir(cwd)
    previous_rng = bootstrap.RNG
    try:
        bootstrap.RNG = np.random.default_rng(1729)
        low, high = bootstrap.boot_ci(np.asarray(differences, dtype=np.float64))
    finally:
        bootstrap.RNG = previous_rng
    return {'status': 'complete', 'n_events': len(differences),
            'event_macro_mIoU_delta': float(np.mean(differences)),
            'event_macro_mIoU_delta_ci95': [float(low), float(high)],
            'bootstrap_unit': 'paired event; fixed selected checkpoints',
            'bootstrap_resamples': int(bootstrap.NBOOT), 'bootstrap_seed': 1729}


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _compare_readouts(readouts, domains):
    indexed = defaultdict(dict)
    for record, receipt in readouts:
        indexed[(record['graph'], record['precision'], record['seed'])][record['method']] = (record, receipt)
    comparisons = []
    for (graph, precision, seed), methods in sorted(indexed.items()):
        pairs = [('B3', control) for control in ('B1', 'B2', 'random', 'quant', 'independent', 'total')]
        pairs += [('B3_beta' + suffix, 'B2_beta' + suffix) for suffix in ('0p5', '2')]
        for proposed, control in pairs:
            if proposed not in methods or control not in methods:
                continue
            lp, rp = methods[proposed][1], methods[control][1]
            for dataset in sorted(set(lp['terminal_results']) & set(rp['terminal_results'])):
                for domain in domains:
                    l = lp['terminal_results'][dataset][domain]; r = rp['terminal_results'][dataset][domain]
                    same_maps = set(l['map']) == set(r['map']) and all(_support(l['map'][key]) == _support(r['map'][key]) for key in l['map'])
                    paired = _paired(l['event'], r['event']) if same_maps else {'status': 'support_mismatch', 'reason': 'different map IDs/ground truth support'}
                    comparison = {'graph': graph, 'precision': precision, 'seed': seed,
                                  'proposed': proposed, 'control': control, 'dataset': dataset, 'domain': domain,
                                  **paired}
                    if paired['status'] == 'complete':
                        for metric in METRICS:
                            a, b = l['micro'].get(metric), r['micro'].get(metric)
                            comparison[metric + '_pooled_delta'] = a - b if a is not None and b is not None else None
                        comparison['direction'] = ('positive' if comparison['mIoU_pooled_delta'] > 0 else
                            'negative' if comparison['mIoU_pooled_delta'] < 0 else 'zero')
                    comparisons.append(comparison)
    return comparisons


def write_results(spec, out, parent_state=None):
    """Write JSON plus sibling metrics, outcomes, comparisons CSV and report TXT."""
    out = Path(out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    state = _state(parent_state or spec.get('queue_state'))
    manifest = _optional(state.get('manifest_path', ''))
    by_output = {}
    for name, job in state.get('jobs', {}).items():
        for output in job.get('outputs', []):
            by_output[str(Path(output['path']).resolve())] = (name, job)
    records, measurements, interactions, terminal, source_metrics, source_diagnostics = [], [], [], [], [], []
    anchors, source_scores = {}, {}
    source_readouts = []
    costs = defaultdict(float)
    for item in dict.fromkeys(spec['receipts']):
        path = Path(item).resolve()
        name, job = by_output.get(str(path), (path.stem, {}))
        argv = job.get('argv') or []
        config = _optional(argv[argv.index('--spec') + 1]) if '--spec' in argv else {}
        action = argv[argv.index('--action') + 1] if '--action' in argv else ('terminal' if name.startswith('terminal_') else 'unknown')
        receipt = _optional(path)
        # Export-only recoveries never overwrite the frozen terminal accuracy
        # receipt. Bind replacement measurements to both its hash and checkpoint.
        recovery_path = spec.get('artifact_overrides', {}).get(str(path))
        original_artifact_status = receipt.get('artifact', {}).get('status')
        if recovery_path:
            recovery = read_json(recovery_path)
            if (recovery.get('status') != 'complete' or recovery.get('outcome') != 'completed'
                    or recovery.get('terminal_receipt_sha256') != file_hash(path)
                    or recovery.get('checkpoint_sha256') != receipt.get('checkpoint_sha256')
                    or recovery.get('new_training') is not False
                    or recovery.get('new_benchmark_evaluation') is not False):
                raise ValueError(f'Export recovery does not match frozen terminal receipt: {path}')
            receipt = {**receipt, 'artifact': recovery['artifact'],
                       'status_label': recovery['artifact']['status']}
        outcome = receipt.get('outcome') if receipt.get('status') == 'complete' else None
        outcome = outcome or job.get('status', 'unavailable')
        # A queue failure invalidates partial scientific outputs, even if present.
        if job.get('status') in ('failed', 'blocked', 'interrupted'):
            outcome = job['status']
        seed_match = re.search(r'_s(42|64|91|107)(?:_|$)', name)
        seed = config.get('seed', int(seed_match.group(1)) if seed_match else 42)
        graph = receipt.get('graph', config.get('graph'))
        precision = receipt.get('precision', config.get('precision', 'FP32'))
        method = receipt.get('method', config.get('method', 'B0' if action in ('terminal', 'warmup') else None))
        elapsed = 0.
        for attempt in job.get('attempts', []):
            attempt_receipt = _optional(attempt.get('receipt_path', ''))
            elapsed += float(attempt_receipt.get('command_elapsed_seconds', 0.))
        if not elapsed:
            elapsed = float(receipt.get('elapsed_seconds', 0.))
        costs[action] += elapsed
        record = {'job_id': name, 'action': action, 'outcome': outcome, 'graph': graph,
                  'precision': precision, 'seed': seed, 'method': method,
                  'elapsed_seconds': elapsed, 'receipt': str(path),
                  'receipt_sha256': file_hash(path) if path.is_file() else None,
                  'reason': receipt.get('reason') or job.get('reason'),
                  'queue_status': job.get('status', 'unknown')}
        checkpoint_path = receipt.get('checkpoint_path') or receipt.get('best_path')
        checkpoint_sha = receipt.get('checkpoint_sha256') or receipt.get('best_sha256')
        if 'terminal_results' in receipt and not checkpoint_path and config.get('method_receipt'):
            checkpoint_path = _optional(config['method_receipt']).get('best_path')
        record['checkpoint_integrity'] = _verify_file(checkpoint_path, checkpoint_sha, receipt.get('checkpoint_bytes'))
        if outcome == 'completed' and record['checkpoint_integrity']['status'] in ('invalid', 'unavailable'):
            record['outcome'] = outcome = record['checkpoint_integrity']['status']
            record['reason'] = record['checkpoint_integrity']['reason']
        records.append(record)
        if recovery_path:
            record.update(artifact_recovery_receipt=str(recovery_path),
                          artifact_recovery_sha256=file_hash(recovery_path),
                          original_artifact_status=original_artifact_status)
        if outcome != 'completed':
            continue
        if action == 'interaction' or 'initial_feature_losses' in receipt:
            interactions.append({**record, **{key: receipt.get(key) for key in
                ('metric', 'nonidentity', 'beta_rule', 'loss_scales', 'initial_feature_losses',
                 'independent_groups', 'split_group_stability')}})
        variants = {}
        warmup_score = receipt.get('source_validation', {}).get('best_val_mIoU')
        if warmup_score is not None and checkpoint_sha and record['checkpoint_integrity']['status'] == 'verified':
            source_scores[(checkpoint_sha, 'FP32')] = float(warmup_score)
            if graph == 'A0' and action == 'warmup':
                anchors[seed] = {'job_id': name, 'seed': seed, 'graph': graph,
                    'checkpoint_sha256': checkpoint_sha, 'source_validation_mIoU': float(warmup_score)}
        if isinstance(receipt.get('validation'), dict):
            variants[precision] = receipt['validation']
        if isinstance(receipt.get('fp_validation'), dict):
            variants['FP32'] = receipt['fp_validation']
        if isinstance(receipt.get('fp16_validation'), dict):
            variants['FP16'] = receipt['fp16_validation']
        variants.update(receipt.get('ptq_validation', {}))
        for variant, values in variants.items():
            if 'micro' not in values:
                continue
            flips = values.get('damage_repair', {})
            source_sha = checkpoint_sha or receipt.get('common_checkpoint_sha256')
            if source_sha and values['micro'].get('mIoU') is not None:
                source_scores[(source_sha, variant)] = float(values['micro']['mIoU'])
            source_metrics.append({**{key: record[key] for key in ('job_id', 'graph', 'seed', 'method')},
                'precision': variant, 'action': action, 'dataset': 'source_validation',
                **{metric: values['micro'].get(metric) for metric in METRICS},
                **{key: flips.get(key) for key in ('H', 'B', 'accuracy_delta', 'valid_pixels', 'harmful_count', 'repaired_count', 'identity_verified')},
                'teacher_disagreement_fraction': values.get('teacher_disagreement', {}).get('fraction')})
        if action == 'qat':
            source_diagnostics.append({**record, **{key: receipt.get(key) for key in
                ('selected_updates', 'completed_updates', 'common_quant_state_sha256', 'final_residual_diagnostics',
                 'final_split_group_stability', 'gradient_diagnostics', 'epochs')}})
            validation = receipt.get('validation', {})
            if all(key in validation for key in ('micro', 'event', 'scene')):
                source_readouts.append((record, {'terminal_results': {'source_validation': {'source': {
                    'micro': validation['micro'], 'event': validation['event'], 'map': validation['scene']}}}}))
        if 'terminal_results' not in receipt:
            continue
        terminal.append((record, receipt))
        artifact = receipt.get('artifact', {})
        runtime = artifact.get('runtime', {})
        exported = artifact.get('artifact', {})
        for measurement in (artifact, artifact.get('float16_reference', {})):
            if measurement:
                exported_file = measurement.get('artifact', {})
                measurement['collector_integrity'] = _verify_file(exported_file.get('path'), exported_file.get('sha256'), exported_file.get('serialized_bytes'))
        record['artifact_integrity'] = artifact.get('collector_integrity', {'status': 'unavailable'})
        verified_export = record['artifact_integrity']['status'] == 'verified'
        for dataset, result in receipt['terminal_results'].items():
            for domain in ('full', 'known_overlap_excluded'):
                values = result[domain]
                measurements.append({**{key: record[key] for key in ('job_id', 'graph', 'precision', 'seed', 'method')},
                    'dataset': dataset, 'domain': domain, 'protocol': result['protocol'],
                    'n_images': values.get('n_images', result['n_images']),
                    'n_maps': values['n_maps'], 'n_events': values['n_events'],
                    **{metric: values['micro'].get(metric) for metric in METRICS},
                    'parameters': receipt.get('parameters'), 'training_checkpoint_bytes': receipt.get('checkpoint_bytes'),
                    'inference_artifact_bytes': record['artifact_integrity'].get('actual_bytes') if verified_export else None,
                    'recorded_inference_artifact_bytes': exported.get('serialized_bytes'),
                    'artifact_status': artifact.get('status') if verified_export else record['artifact_integrity']['status'],
                    'runtime_status': runtime.get('status') if verified_export else record['artifact_integrity']['status'],
                    'checkpoint_integrity': record['checkpoint_integrity']['status'], 'artifact_integrity': record['artifact_integrity']['status'],
                    'latency_ms_median': runtime.get('latency_ms_median'),
                    'latency_ms_p95': runtime.get('latency_ms_p95'),
                    'total_macs': artifact.get('macs', {}).get('total_macs'),
                    'convolution_macs': artifact.get('macs', {}).get('convolution_macs'),
                    'attention_matmul_macs': artifact.get('macs', {}).get('attention_matmul_macs'),
                    'runtime_process_peak_reserved_bytes': runtime.get('process_peak_reserved_bytes'),
                    'runtime_process_peak_allocated_bytes': runtime.get('process_peak_allocated_bytes'),
                    'runtime_additional_peak_allocated_bytes': runtime.get('additional_peak_allocated_bytes'),
                    'fp16_reference_status': artifact.get('float16_reference', {}).get('status'),
                    'quantization_scope': receipt.get('status_label'), 'checkpoint_sha256': receipt.get('checkpoint_sha256')})
    comparisons = _compare_readouts(terminal, ('full', 'known_overlap_excluded'))
    source_comparisons = _compare_readouts(source_readouts, ('source',))
    summaries = []
    grouped = defaultdict(list)
    for comparison in comparisons:
        if comparison['status'] == 'complete':
            grouped[tuple(comparison[key] for key in ('graph', 'precision', 'proposed', 'control', 'dataset', 'domain'))].append(comparison)
    for key, entries in sorted(grouped.items()):
        differences = [r['mIoU_pooled_delta'] for r in entries]
        summaries.append({**dict(zip(('graph', 'precision', 'proposed', 'control', 'dataset', 'domain'), key)),
                          'completed_seeds': sorted(r['seed'] for r in entries), 'n_seeds': len(entries),
                          'mean_pooled_mIoU_delta': float(np.mean(differences)),
                          'sample_sd_pooled_mIoU_delta': float(np.std(differences, ddof=1)) if len(entries) > 1 else None,
                          'scope': 'descriptive seed variation; event CI does not establish seed robustness'})
    csv_paths = {label: out.with_name(out.stem + '_' + label + '.csv') for label in ('metrics', 'source_metrics', 'source_comparisons', 'outcomes', 'comparisons')}
    for label, rows in (('metrics', measurements), ('source_metrics', source_metrics), ('source_comparisons', source_comparisons), ('outcomes', records), ('comparisons', comparisons)):
        # CI lists are encoded as JSON in the flat CSV.
        _write_csv(csv_paths[label], [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows])
    counts = dict(Counter(record['outcome'] for record in records))
    deployment = _deployment_summary(terminal, anchors, source_scores, spec.get('frontier', {}))
    report = out.with_suffix('.txt')
    lines = [f'Compression v2 terminal collection: {len(records)} recorded jobs; outcomes {counts}.',
             f'{len(terminal)} completed terminal checkpoints; {len(measurements)} dataset/domain rows; {len(comparisons)} paired comparisons.',
             'Metrics use stored valid-pixel confusion counts and frozen protocols. Full and known-overlap-excluded domains remain separate.',
             'Paired 95% intervals resample events, conditional on fixed selected checkpoints; seed summaries are descriptive.',
             'Four-bit and eight-bit results are floating-point fake-quantization simulations; artifact/runtime measurements are reported independently of logical bitwidth.',
             'Unavailable, failed, blocked, skipped and exact-identity deduplicated arms remain explicit; no missing result is treated as a negative accuracy result.']
    winner = deployment['smallest_measured_inference_artifact']
    lines.append('Predeclared deployment summary uses only source-validation mIoU within 0.01 below the same-seed ReLU6 A0 FP anchor; measured artifact bytes are verified against their SHA256 receipts.')
    lines.append(f"Smallest eligible measured inference artifact: {winner['job_id']} {winner['precision']}, {winner['inference_artifact_bytes']} bytes, {winner['artifact_scope']}." if winner else 'Smallest eligible measured inference artifact: unavailable; missing anchors, source scores, integrity or measured exports remain explicit.')
    for entry in summaries:
        if entry['dataset'] == 'sturm' and entry['domain'] == 'known_overlap_excluded':
            lines.append(f"{entry['graph']} {entry['precision']} {entry['proposed']} minus {entry['control']}: mean pooled mIoU delta {entry['mean_pooled_mIoU_delta']:+.6f}; completed seeds {entry['completed_seeds']}.")
    pending = ['Faithful published QFD/SQuaT baseline remains pending; no reproduction or novelty claim.',
               'Time/compute-matched ordinary KD remains required before interpreting positive screening beyond the equal-update B1 control.',
               'Second-family seeds 64/91/107 and matched ReLU6 OFA continuation are deferred by the fixed cap.',
               'The author disclosed prior unsuccessful teacher-derived STURM criteria; external benchmarks are terminal readouts after source design freeze, not historically untouched data.']
    lines += pending
    report.write_text('\n'.join(lines) + '\n')
    result = {'status': 'complete', 'outcome': 'completed', 'outcome_counts': counts,
              'records': records, 'metrics': measurements, 'comparisons': comparisons, 'seed_summaries': summaries,
              'interaction_diagnostics': interactions, 'frontier': spec.get('frontier', {}),
              'source_metrics': source_metrics, 'source_post_qat_diagnostics': source_diagnostics,
              'source_comparisons': source_comparisons,
              'confirmation': spec.get('confirmation', {}), 'cost_seconds_by_action': dict(costs),
              'deployment_summary': deployment,
              'reservation': {key: manifest.get(key) for key in ('reserved_main_gpu_seconds', 'reserved_preflight_gpu_seconds', 'pilot_cap_seconds')},
              'report': str(report), 'csv': {k: str(v) for k, v in csv_paths.items()},
              'limitations': pending, 'new_gpu_work': False, 'external_feedback_used_for_scheduling': False,
              'analysis_scope': 'terminal receipt bookkeeping; missing runs retained; descriptive seed statistics and conditional paired event bootstrap'}
    atomic_json(out, result)
    return result
