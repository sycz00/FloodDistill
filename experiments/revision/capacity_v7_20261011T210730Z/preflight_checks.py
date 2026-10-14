"""Independent CPU-only admission checks over completed v7 preflight artifacts."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite_numbers(value, where='root'):
    if isinstance(value, float):
        assert math.isfinite(value), f'Nonfinite value at {where}'
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_numbers(item, where + '.' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_numbers(item, where + '.' + str(index))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--expected-total', type=float,
                   help='Independent declared preflight total; if supplied must match exactly')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    root = a.run_dir.resolve()
    output = a.out or root / 'preflight_checks.json'
    if output.exists():
        raise RuntimeError('Checks artifact already exists; preserve it and choose a new --out')
    checks, errors, input_hashes = {}, [], {}

    def load(relative):
        path = root / relative
        if not path.is_file():
            raise AssertionError(f'Pending/missing required artifact: {path}')
        input_hashes[str(path)] = sha(path)
        return read(path)

    def gate(name, fn):
        try:
            details = fn()
            checks[name] = {'passed': True, 'details': details}
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            errors.append(name + ': ' + message)
            checks[name] = {'passed': False, 'error': message}

    def focused():
        report = load('checks/focused_cpu.json')
        assert report['outcome'] == 'passed' and report['gpu_used'] is False
        assert report['checks_count'] >= 6 and len(report['paired_signatures']) == 8
        signatures = report['paired_signatures'].values()
        for key in ('initial_base', 'first_augmented_minibatch'):
            assert len({item[key] for item in signatures}) == 1
        return {'checks': report['checks'], 'cpu_only': True}

    def scheduler():
        current = sha(root / 'run_overnight.py')
        queue, recovery = load('queue_checks.json'), load('recovery_checks.json')
        assert queue['passed'] and recovery['passed']
        assert recovery['gpu_calls'] is False and recovery['gpu_spend_seconds'] == 0
        assert recovery['scheduler_sha256'] == current, 'Recovery checks used stale scheduler bytes'
        mock_manifest = Path(queue['mock_root']) / 'manifest_initial.json'
        assert mock_manifest.is_file(), 'Queue sentinel manifest is unavailable'
        mock = read(mock_manifest)
        assert mock['frozen_inputs'][str(root / 'run_overnight.py')] == current, 'Queue checks used stale scheduler bytes'
        required = ('second_sentinel_after_launcher_exit', 'worker_session_is_independent',
                    'dependency_block', 'full_bundle_budget_refusal', 'promotion_pass_and_fail',
                    'cpu_only_spend_zero', 'diagnosed_one_retry', 'controlled_stop_reaps_owned_child')
        assert all(queue.get(key) is True for key in required)
        assert recovery['guardian_death_recovery']['passed']
        assert recovery['persisted_attempt2_argv_recovery']['passed']
        return {'scheduler_sha256': current, 'queue_mock_manifest_sha256': sha(mock_manifest),
                'detachment_guarantee_tested': queue['actually_tested']}

    def smoke(relative, graph, method, updates):
        report = load(relative)
        assert report['status'] == 'complete' and report['outcome'] == 'smoke_completed'
        assert report['scientific_run'] is False and report['smoke'] is True
        assert (report['graph'], report['method'], report['seed']) == (graph, method, 42)
        assert report['completed_updates'] == report['gradient_checks']['updates_checked'] == updates
        assert math.isfinite(report['gradient_checks']['max_post_clip_gradient_l2'])
        assert report['early_stop_patience'] == 0 and report['workers'] == 2
        assert len(report['first_minibatch_hashes']) == 2 and report['first_training_logits_sha256']
        finite_numbers(report['epochs'])
        finite_numbers(report['source_validation'])
        for kind in ('best', 'last'):
            assert sha(report[kind + '_path']) == report[kind + '_sha256'], f'{relative} {kind} hash changed'
        return report

    def training():
        reports = {}
        paths = [('A0', 'lkd', 'smoke_A0_lkd_retry1_timing', 156),
                 ('R4', 'ce', 'smoke_R4_ce_timing', 156), ('A1', 'lkd', 'smoke_A1_lkd', 156)]
        paths += [('R4', arm, 'smoke_R4_' + ('C0_final' if arm == 'C0' else arm), 4)
                  for arm in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL')]
        for graph, method, name, updates in paths:
            reports[graph + ':' + method] = smoke('smokes/' + name + '.json', graph, method, updates)
        arms = [reports['R4:' + arm] for arm in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL')]
        for key in ('initial_base_sha256', 'first_training_logits_sha256', 'parent_sha256'):
            assert len({report[key] for report in arms}) == 1, f'Continuation parity failed: {key}'
        assert len({tuple(report['first_minibatch_hashes']) for report in arms}) == 1
        assert all(report['start_candidate'] is True for report in arms)
        assert all(report['batch_norm_policy'] == arms[0]['batch_norm_policy'] for report in arms)
        main_reuse = load('reuse_index.json')['records']
        a1_reuse = load('optional_A1_reuse_index.json')
        assert a1_reuse['status'] == 'complete' and all(record['verified'] for record in a1_reuse['records'])
        for graph, records in (('A0', main_reuse), ('A1', a1_reuse['records'])):
            historical = next(record for record in records if record['graph'] == graph and record['seed'] == 42)
            expected = historical['metadata']['best']['initial_model_sha256']
            assert reports[graph + ':lkd']['initial_base_sha256'] == expected
        return {'required_paths_checked': [name for _, _, name, _ in paths],
                'all_eight_continuation_arms_paired': True,
                'continuation_initial_base_sha256': arms[0]['initial_base_sha256'],
                'continuation_first_logits_sha256': arms[0]['first_training_logits_sha256'],
                'continuation_first_minibatch_hashes': arms[0]['first_minibatch_hashes'],
                'A0_initialization_sha256': reports['A0:lkd']['initial_base_sha256'],
                'A1_initialization_sha256': reports['A1:lkd']['initial_base_sha256']}

    def timing():
        specifications = [('A0_lkd', 'smoke_A0_lkd_retry1_timing', 7800, 25, 1400),
                          ('R4_ce', 'smoke_R4_ce_timing', 7800, 25, 700),
                          ('R4_C4', 'smoke_R4_C4_timing', 1560, 6, 450),
                          ('R4_C5', 'smoke_R4_C5_timing', 1560, 6, 450),
                          ('A1_lkd', 'smoke_A1_lkd', 7800, 25, 1400)]
        projection = {}
        for label, name, updates, evaluations, hard_timeout in specifications:
            report = load('smokes/' + name + '.json')
            assert report['completed_updates'] == 156
            trains = [row for row in report['timings'] if row['phase'] == 'train']
            vals = [row for row in report['timings'] if row['phase'] == 'source_validation']
            assert sum(row['updates'] for row in trains) == 156
            seconds_per_update = sum(row['seconds'] for row in trains) / 156
            seconds_per_val_batch = max(row['seconds'] / row['batches'] for row in vals)
            # 89 source tiles at batch4 ->23 batches. Multiplying the measured
            # first batch (which includes worker startup) by23 is conservative.
            training_seconds = updates * seconds_per_update
            validation_seconds = evaluations * 23 * seconds_per_val_batch
            conservative = 1.10 * (training_seconds + validation_seconds) + 60
            assert conservative < hard_timeout, f'{label} projection {conservative} exceeds {hard_timeout}'
            projection[label] = dict(seconds_per_update=seconds_per_update,
                measured_validation_batch_seconds=seconds_per_val_batch,
                projected_training_seconds=training_seconds, projected_validation_seconds=validation_seconds,
                relative_margin=1.10, startup_checkpoint_reserve_seconds=60,
                conservative_full_job_seconds=conservative, declared_hard_timeout_seconds=hard_timeout)
        return projection

    def ledger():
        attempts = load('preflight_ledger.json')
        assert attempts and all(row['gpu'] is True for row in attempts)
        identities = []
        for row in attempts:
            assert row.get('finished_at') and row.get('exit_code') is not None, 'Pending GPU preflight attempt'
            assert math.isfinite(row['command_elapsed_seconds']) and row['command_elapsed_seconds'] >= 0
            identities.append(tuple(row['command'].get(k) for k in ('boot_id', 'pid', 'start_ticks')))
            attempt_path = root / 'preflight_attempts' / (row['job']['id'] + '.json')
            disk = read(attempt_path)
            assert disk['command_elapsed_seconds'] == row['command_elapsed_seconds']
            assert disk['exit_code'] == row['exit_code']
        assert len(identities) == len(set(identities)), 'Preflight command counted more than once'
        total = sum(row['command_elapsed_seconds'] for row in attempts)
        if a.expected_total is not None:
            assert total == a.expected_total, f'Preflight total {total} differs from declared {a.expected_total}'
        failed = [row for row in attempts if row['exit_code'] != 0]
        old_failure = next(row for row in failed if row['job']['id'] == 'smoke_A0_lkd')
        assert old_failure['command_elapsed_seconds'] == 13.06687134806998
        assert total == sum(row['command_elapsed_seconds'] for row in attempts if row['exit_code'] == 0) + sum(
            row['command_elapsed_seconds'] for row in failed) or math.isclose(total,
                sum(row['command_elapsed_seconds'] for row in attempts if row['exit_code'] == 0)
                + sum(row['command_elapsed_seconds'] for row in failed), abs_tol=1e-10)
        frozen = root / 'manifest_initial.json'
        if frozen.is_file():
            assert read(frozen)['preflight_gpu_seconds'] == total, 'Frozen declared preflight spend mismatch'
        return dict(attempts=len(attempts), gpu_command_elapsed_seconds=total,
                    declared_preflight_gpu_seconds=total, independent_expected_total=a.expected_total,
                    failed_attempts=len(failed), failed_gpu_seconds=sum(row['command_elapsed_seconds'] for row in failed),
                    initial_failure_seconds_counted=old_failure['command_elapsed_seconds'],
                    failed_and_successful_commands_counted_once=True)

    def fresh_launch_budget():
        proof = load('launch_budget_check.json')
        original = load('historical_budget.json')
        raw = read(proof['raw_ledger'])
        assert proof['passed'] is True and proof['verified'] is True and not proof['unresolved']
        assert raw['verified'] is True and not raw['unresolved']
        assert sha(proof['raw_ledger']) == proof['raw_ledger_sha256']
        assert sha(root / 'historical_budget.json') == proof['original_ledger_sha256']
        total = sum(row['command_elapsed_seconds'] for row in load('preflight_ledger.json'))
        assert proof['preflight_gpu_seconds'] == total
        assert proof['historical_gpu_seconds'] == original['cumulative_gpu_seconds'] == raw['cumulative_gpu_seconds']
        assert proof['cumulative_now_seconds'] == proof['historical_gpu_seconds'] + total
        remaining_programme_seconds = max(0., 30 * 3600 - proof['cumulative_now_seconds'])
        assert proof['incremental_cap_seconds'] == min(8 * 3600, remaining_programme_seconds)
        preview = load('manifest_preview.json')
        assert preview['protocol']['budget']['incremental_cap_seconds'] == proof['incremental_cap_seconds']
        assert preview['protocol']['budget']['historical_gpu_seconds'] == proof['historical_gpu_seconds']
        return {'checked_at': proof['checked_at'], 'historical_gpu_seconds': proof['historical_gpu_seconds'],
                'preflight_gpu_seconds': total, 'cumulative_now_seconds': proof['cumulative_now_seconds'],
                'incremental_cap_seconds': proof['incremental_cap_seconds'],
                'remaining_programme_seconds_at_launch': remaining_programme_seconds,
                'independent_raw_ledger_sha256': proof['raw_ledger_sha256']}

    def final_inference():
        attempts = load('preflight_ledger.json')
        selected = {}
        # These final checks cover the last frozen source and multi-model cache path.
        for jid in ('evaluator_multifinal', 'reference_multifinal', 'mechanism_multifinal', 'deployment_multifinal'):
            matches = [row for row in attempts if row['job']['id'] == jid]
            assert len(matches) == 1, f'Pending/missing final GPU check: {jid}'
            row = matches[0]
            assert row['exit_code'] == 0 and row.get('finished_at'), f'Final GPU check did not pass: {jid}'
            for path, expected in row['frozen_inputs'].items():
                if Path(path).name in ('evaluate_v7.py', 'deploy_v7.py', 'train_v7.py', 'v7_loss.py'):
                    assert sha(path) == expected, f'{jid} used stale bytes: {path}'
            out = next(item['path'] for item in row['job']['outputs'] if item.get('type') == 'json')
            report = read(out)
            assert report['status'] == 'complete' and report['outcome'] == 'completed'
            assert report['smoke_only'] is True
            selected[jid] = {'receipt_path': out, 'receipt_sha256': sha(out), 'report': report, 'job': row['job']}
        return selected

    def evaluation():
        finals = final_inference()
        reuse = load('reuse_index.json')['records']
        historical_record = next(row for row in reuse if row['graph'] == 'R4' and row['seed'] == 42)
        historical = read(historical_record['terminal_receipt_path'])
        current = load('smoke_terminal_parity/R4_anchor_smoke.json')
        assert current['checkpoint_sha256'] == historical['checkpoint_sha256'] == historical_record['best_sha256']
        total, per_dataset = 0, {}
        for dataset in ('s11', 'sturm', 'wf'):
            counts = current['terminal_results'][dataset]['per_image_counts']
            assert len(counts) == 8
            for filename, value in counts.items():
                assert value == historical['terminal_results'][dataset]['per_image_counts'][filename], filename
            total += len(counts)
            per_dataset[dataset] = len(counts)
        assert total == 24
        backend = current['evaluation_identity']['backend_policy']
        assert backend['policy'] == 'inherited_terminal_v5' and backend['grad_context'] == 'torch.no_grad'
        assert backend['input_dtype'] == backend['checkpoint_parameter_dtype'] == 'torch.float32'
        assert backend['cudnn_allow_tf32'] is True and backend['cuda_matmul_allow_tf32'] is True
        assert backend['float32_matmul_precision'] == 'medium'
        reference = finals['reference_multifinal']['report']
        manifest = read(reference['cache_manifest'])
        assert sha(reference['cache_manifest']) == reference['cache_manifest_sha256']
        assert manifest['identity'] == current['evaluation_identity']
        for dataset, payload in manifest['datasets'].items():
            assert len(payload['rows']) == 8 and sha(payload['path']) == payload['sha256']
            assert all(row['input_sha256'] and row['mask_sha256'] for row in payload['rows'])
        mechanism_report = finals['mechanism_multifinal']['report']
        mechanism_job = finals['mechanism_multifinal']['job']
        directory = Path(mechanism_job['argv'][mechanism_job['argv'].index('--out-dir') + 1])
        assert set(mechanism_report['scored_now']) == set(mechanism_report['requested'])
        plain_report = finals['evaluator_multifinal']['report']
        plain_job = finals['evaluator_multifinal']['job']
        plain_directory = Path(plain_job['argv'][plain_job['argv'].index('--out-dir') + 1])
        assert set(plain_report['scored_now']) == set(plain_report['requested'])
        assert set(plain_report['requested']) == set(mechanism_report['requested'])
        assert len(plain_report['requested']) == 20
        mechanism_images, historical_images, tested_arms = 0, 0, set()
        for mid in mechanism_report['requested']:
            report = read(directory / (mid + '.json'))
            plain = read(plain_directory / (mid + '.json'))
            assert report['mechanism_status'] == 'completed'
            assert report['evaluation_identity'] == manifest['identity']
            assert plain['evaluation_identity'] == manifest['identity']
            assert report['checkpoint_sha256'] == plain['checkpoint_sha256']
            assert sha(report['checkpoint_path']) == report['checkpoint_sha256']
            assert report['graph'] == plain['graph'] and report['method'] == plain['method']
            if report['method'] in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL'):
                tested_arms.add(report['method'])
                expected_parameters = 12232 if report['method'] in ('C1', 'C3', 'C4', 'C5') else 12230
                assert report['parameters'] == expected_parameters
            else:
                anchor = next(row for row in reuse if row['best_sha256'] == report['checkpoint_sha256'])
                old = read(anchor['terminal_receipt_path'])
            for dataset, rows in report['mechanism_counts'].items():
                assert len(rows) == 8
                assert report['terminal_results'][dataset]['per_image_counts'] == plain['terminal_results'][dataset]['per_image_counts']
                assert set(rows) == {row['filename'] for row in manifest['datasets'][dataset]['rows']}
                for filename, row in rows.items():
                    denominator_keys = ('correction_denominator', 'harm_denominator', 'rescue_denominator')
                    numerator_keys = ('correction_numerator', 'harm_numerator', 'rescue_numerator')
                    for num, den in zip(numerator_keys, denominator_keys):
                        assert isinstance(row[num], int) and 0 <= row[num] <= row[den]
                    assert row['disagreements'] == row['correction_denominator'] + row['harm_denominator']
                    assert row['signed_benefit_numerator'] == row['correction_denominator'] - row['harm_denominator']
                    assert row['rule_available_pixels'] == row['both_correct_pixels'] + sum(row[key] for key in denominator_keys)
                    assert row['valid_pixels'] == row['rule_available_pixels'] + row['rule_unavailable_pixels']
                    counts = report['terminal_results'][dataset]['per_image_counts'][filename]
                    assert row['valid_pixels'] == sum(counts[cls][0] + counts[cls][2] for cls in (0, 1))
                    if report['method'] not in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL'):
                        assert counts == old['terminal_results'][dataset]['per_image_counts'][filename]
                        historical_images += 1
                mechanism_images += len(rows)
        assert tested_arms == {'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL'}
        assert historical_images == 288 and mechanism_images == 480
        return dict(exact_historical_count_parity=True, images_checked=total, per_dataset=per_dataset,
                    backend_policy=backend, mechanism_model_count=len(mechanism_report['requested']),
                    mechanism_image_model_pairs=mechanism_images,
                    historical_anchor_image_model_pairs=historical_images,
                    new_rule_and_control_arms_tested=sorted(tested_arms), plain_and_mechanism_counts_identical=True,
                    final_gpu_checks={key: {k: v for k, v in value.items() if k not in ('report', 'job')}
                                      for key, value in finals.items()})

    def deployment():
        final = final_inference()['deployment_multifinal']['report']
        assert final['smoke_only'] is True and final['calibration']['dataset'] == 'TRAIN'
        assert final['unsupported_does_not_block_training'] is True
        statuses, actual_backends = [], []
        for row in final['rows']:
            assert row['smoke_only'] is True
            assert row['status'] in ('completed', 'unsupported', 'not_run')
            if row['status'] == 'completed':
                assert row['engine'] == 'torchscript_cuda' or row['engine'] == 'tensorrt_native_ptq'
                assert row['teacher_loaded'] is False
                assert row['warmup'] == 2 and row['timed_iterations'] == 3
                assert row['batch_size'] == 1 and row['input_shape'] == '1x6x512x512'
                assert row['artifact_bytes'] == Path(row['artifact_path']).stat().st_size
                assert row['artifact_sha256'] == sha(row['artifact_path'])
                assert row['eager_artifact_prediction_disagreement_pixels'] == 0
                finite_numbers({key: row[key] for key in ('latency_p50_ms', 'latency_p95_ms', 'mIoU', 'water_IoU')})
                assert row['operator_fallback'] and row['timing_boundary']
                backend = row['backend_policy']
                expected_dtype = {'FP32': 'torch.float32', 'FP16': 'torch.float16'}.get(row['precision'])
                if expected_dtype:
                    assert backend['input_dtype'] == backend['checkpoint_parameter_dtype'] == expected_dtype
                assert backend['grad_context'] == 'torch.inference_mode'
                actual_backends.append({'id': row['id'], 'precision': row['precision'], 'backend_policy': backend})
            elif row['status'] == 'unsupported':
                assert row.get('reason'), 'Unsupported attempt needs a concrete failure reason'
                assert row.get('latency_p50_ms') is None and row.get('mIoU') is None
            statuses.append({'id': row['id'], 'precision': row['precision'], 'status': row['status'],
                             'reason': row.get('reason')})
        assert any(row['precision'] == 'FP32' and row['status'] == 'completed' for row in final['rows'])
        return {'statuses': statuses, 'recorded_actual_backends': actual_backends,
                'smoke_benchmark_is_not_final_20_warmup_100_iteration_measurement': True,
                'policy_names_are_labels_actual_TF32_flags_are_recorded_independently': True}

    for name, fn in (('focused_cpu', focused), ('scheduler_current_hash', scheduler),
                     ('GPU_training_smokes_and_paired_rng', training), ('conservative_timing', timing),
                     ('historical_evaluator_and_mechanism_parity', evaluation),
                     ('deployment_truthful_status', deployment), ('all_preflight_spend_counted', ledger),
                     ('fresh_launch_budget_proof', fresh_launch_budget)):
        gate(name, fn)
    payload = dict(status='complete' if not errors else 'blocked', passed=not errors,
                   checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   checker_sha256=sha(__file__), gpu_calls=False, checks=checks,
                   input_sha256=input_hashes, errors=errors)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, output)
    print(json.dumps({'passed': payload['passed'], 'out': str(output), 'errors': errors}))
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
