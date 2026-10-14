"""CPU-only synthetic check of final compression outcomes and paired statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.compression_collect import _paired, _verify_file, write_results
from evaluation.revision_eval import scores
from experiments.revision_queue import atomic_json, file_hash


def check():
    # Exercise the actual capsule outside the checkout: imports must not rely
    # on the caller's cwd/PYTHONPATH to find the plotting dependency.
    from experiments.compression_prepare import snapshot
    with tempfile.TemporaryDirectory(prefix='compression-frozen-collect-') as folder:
        capsule, _ = snapshot(Path(folder))
        isolated = (
            f'import sys; sys.path.insert(0, {str(capsule)!r}); '
            'import site; sys.path.append(site.getusersitepackages()); '
            f'sys.path.append({str(Path(__file__).resolve().parents[2] / "_deps/flooddistill_revision_python")!r}); '
            'from evaluation.compression_collect import _paired; '
            'import figures.fig_style as style; '
            f'assert style.__file__.startswith({str(capsule)!r}); '
            'c=[[75,25,25],[75,25,25]]; '
            'assert _paired({"E1":c},{"E1":c})["event_macro_mIoU_delta_ci95"]==[0.,0.]'
        )
        completed = subprocess.run([sys.executable, '-I', '-c', isolated], cwd=folder,
                                   capture_output=True, text=True, timeout=60)
        if completed.returncode:
            raise AssertionError(f'Frozen collector import failed:\n{completed.stderr}')
    with tempfile.TemporaryDirectory(prefix='compression-collect-check-') as folder:
        base = Path(folder); receipts = []; jobs = {}
        common = [[75, 25, 25], [75, 25, 25]]
        better = [[85, 15, 15], [85, 15, 15]]
        for seed in (42, 64):
            for method in ('B2', 'B3'):
                name = f'terminal_A3_{method}_s{seed}'
                path = base / (name + '.json'); spec = base / (name + '_spec.json')
                atomic_json(spec, {'seed': seed, 'graph': 'A3', 'precision': 'W4A4', 'method': method})
                cell = better if method == 'B3' else common
                events = {'E1': cell, 'E2': common}
                pooled = [[sum(events[k][i][j] for k in events) for j in range(3)] for i in range(2)]
                pack = {'event': events, 'map': events, 'micro': scores(pooled), 'n_maps': 2, 'n_events': 2}
                result = {ds: {'full': pack, 'known_overlap_excluded': pack, 'n_images': 2,
                              'protocol': 'synthetic_counts_only'} for ds in ('s11', 'sturm', 'wf')}
                checkpoint = base / (name + '.pt'); checkpoint.write_bytes(b'c' * 999999)
                artifact = base / (name + '.ts'); artifact.write_bytes(b'a' * 123)
                atomic_json(path, {'status': 'complete', 'outcome': 'completed', 'graph': 'A3', 'method': method,
                    'terminal_results': result, 'status_label': 'simulation_only', 'checkpoint_bytes': 999999,
                    'checkpoint_path': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
                    'artifact': {'status': 'simulation_only', 'artifact': {'serialized_bytes': 123,
                        'status': 'exported_prediction_validated', 'path': str(artifact), 'sha256': file_hash(artifact)},
                                 'runtime': {'status': 'measured', 'latency_ms_median': 4., 'latency_ms_p95': 5.}}})
                receipts.append(str(path)); jobs[name] = {'status': 'completed', 'outputs': [{'path': str(path)}],
                    'argv': ['--action', 'terminal', '--spec', str(spec)], 'attempts': []}
        failed = base / 'failed.json'; skipped = base / 'skipped.json'
        jobs['failed'] = {'status': 'failed', 'reason': 'cap', 'outputs': [{'path': str(failed)}], 'attempts': []}
        atomic_json(skipped, {'status': 'complete', 'outcome': 'skipped', 'reason': 'exact identity'})
        receipts += [str(failed), str(skipped)]
        spec = {'receipts': receipts, 'frontier': {'A2': {'status': 'deferred'}}, 'confirmation': {'seeds': [42, 64, 91, 107]}}
        outcome = write_results(spec, base / 'report.json', {'jobs': jobs})
        assert outcome['outcome_counts'] == {'completed': 4, 'failed': 1, 'skipped': 1}
        assert len(outcome['metrics']) == 24 and len(outcome['comparisons']) == 12
        assert len(outcome['seed_summaries']) == 6
        assert all(r['completed_seeds'] == [42, 64] for r in outcome['seed_summaries'])
        assert all(r['n_events'] == 2 and r['direction'] == 'positive' for r in outcome['comparisons'])
        assert all(r['inference_artifact_bytes'] == 123 and r['training_checkpoint_bytes'] == 999999 for r in outcome['metrics'])
        assert outcome['frontier']['A2']['status'] == 'deferred'
        assert all(Path(p).is_file() for p in [outcome['report'], *outcome['csv'].values()])
        assert _paired({'E1': common}, {'E2': common})['status'] == 'support_mismatch'
        assert _paired({'E1': common}, {'E1': [[75, 25, 24], [75, 25, 25]]})['status'] == 'support_mismatch'
        assert _paired({'E1': common}, {'E1': common})['event_macro_mIoU_delta_ci95'] == [0., 0.]
        again = write_results(spec, base / 'again.json', {'jobs': jobs})
        assert outcome['comparisons'] == again['comparisons']
        assert outcome['deployment_summary']['status'] == 'unavailable', 'Missing A0 anchor must remain unavailable'
        assert _verify_file(checkpoint, file_hash(checkpoint), 1)['status'] == 'invalid'
        recorded_sha = file_hash(artifact); artifact.write_bytes(b'altered')
        assert _verify_file(artifact, recorded_sha, 123)['status'] == 'invalid'
        artifact.unlink(); assert _verify_file(artifact, recorded_sha, 123)['status'] == 'unavailable'
        # Matching source anchor and scores precede the source-only selection.
        frontier = {graph: {'status': 'predeclared'} for graph in ('A0', 'A1', 'A2', 'A3', 'M0')}
        def add(name, graph, method, score, size, *, action, simulation=False, source=None):
            cp = base / (name + '.pt'); cp.write_bytes(name.encode())
            ap = base / (name + '.ts'); ap.write_bytes(b'x' * size)
            sp = base / (name + '_spec.json'); atomic_json(sp, {'seed': 42, 'graph': graph,
                'method': method, 'precision': 'W4A4' if simulation else 'FP32'})
            record = {'status': 'complete', 'outcome': 'completed', 'graph': graph, 'method': method,
                'best_path': str(cp), 'best_sha256': file_hash(cp),
                'source_validation': {'best_val_mIoU': score}}
            if action == 'qat':
                record.update(validation={'micro': {**scores(common), 'mIoU': score}, 'event': {'E1': common}, 'scene': {'E1': common}})
            if action == 'terminal':
                record.update(checkpoint_path=str(source), checkpoint_sha256=file_hash(source), checkpoint_bytes=source.stat().st_size,
                    terminal_results=result, status_label='simulation_only' if simulation else 'FP32_measured',
                    artifact={'status': 'simulation_only' if simulation else 'deployed_measured',
                        'artifact': {'status': 'exported_prediction_validated', 'path': str(ap), 'sha256': file_hash(ap), 'serialized_bytes': size},
                        'runtime': {'status': 'measured'}})
                record.pop('best_path'); record.pop('best_sha256'); record.pop('source_validation')
            rp = base / (name + '.json'); atomic_json(rp, record)
            receipts.append(str(rp)); jobs[name] = {'status': 'completed', 'outputs': [{'path': str(rp)}],
                'argv': ['--action', action, '--spec', str(sp)], 'attempts': []}
            return cp, ap
        anchor, _ = add('anchor_s42', 'A0', 'B0', .80, 300, action='warmup')
        fp, _ = add('fp_s42', 'A3', 'B0', .795, 200, action='warmup')
        small, _ = add('small_s42', 'A3', 'B3', .785, 20, action='qat', simulation=True)
        valid, _ = add('valid_s42', 'A3', 'B2', .795, 100, action='qat', simulation=True)
        mobile, _ = add('mobile_s42', 'M0', 'B0', .90, 1, action='warmup')
        add('terminal_anchor_s42', 'A0', 'B0', .80, 300, action='terminal', source=anchor)
        add('terminal_fp_s42', 'A3', 'B0', .795, 200, action='terminal', source=fp)
        add('terminal_small_s42', 'A3', 'B3', .785, 20, action='terminal', simulation=True, source=small)
        add('terminal_valid_s42', 'A3', 'B2', .795, 100, action='terminal', simulation=True, source=valid)
        add('terminal_mobile_s42', 'M0', 'B0', .90, 1, action='terminal', source=mobile)
        spec['frontier'] = frontier
        selected = write_results(spec, base / 'selected.json', {'jobs': jobs})
        deployment = selected['deployment_summary']
        assert deployment['smallest_measured_inference_artifact']['job_id'] == 'terminal_valid_s42'
        assert deployment['smallest_native_FP_artifact']['job_id'] == 'terminal_fp_s42'
        assert deployment['smallest_fakequant_simulation_artifact']['artifact_scope'] == 'fake_quant_simulation_bytes_no_packed_compression'
        assert len(deployment['complete_frontier']) == 5 and deployment['tolerance'] == .01
        assert selected['source_comparisons'] and all(row['dataset'] == 'source_validation' for row in selected['source_comparisons'])
        # Corruption never retains a green deployment result.
        valid_export = base / 'terminal_valid_s42.ts'; valid_export.write_bytes(b'corruption')
        corrupted = write_results(spec, base / 'corrupted.json', {'jobs': jobs})
        assert corrupted['deployment_summary']['smallest_measured_inference_artifact']['job_id'] == 'terminal_fp_s42'
        corrupt_rows = [row for row in corrupted['metrics'] if row['job_id'] == 'terminal_valid_s42']
        assert corrupt_rows and all(row['artifact_integrity'] == 'invalid' and row['runtime_status'] == 'invalid' for row in corrupt_rows)
        terminal_path = base / 'terminal_valid_s42.json'
        original_sha = file_hash(terminal_path)
        repair_export = base / 'repaired.ts'; repair_export.write_bytes(b'r' * 90)
        repair = base / 'repair.json'
        frozen = json.loads(terminal_path.read_text())
        repair_data = {'status': 'complete', 'outcome': 'completed',
            'terminal_receipt_sha256': original_sha,
            'checkpoint_sha256': frozen['checkpoint_sha256'],
            'new_training': False, 'new_benchmark_evaluation': False,
            'artifact': {'status': 'simulation_only', 'artifact': {
                'status': 'exported_prediction_validated', 'path': str(repair_export),
                'sha256': file_hash(repair_export), 'serialized_bytes': 90},
                'runtime': {'status': 'measured'}}}
        atomic_json(repair, repair_data)
        spec['artifact_overrides'] = {str(terminal_path): str(repair)}
        recovered = write_results(spec, base / 'recovered.json', {'jobs': jobs})
        assert recovered['deployment_summary']['smallest_measured_inference_artifact']['inference_artifact_bytes'] == 90
        assert file_hash(terminal_path) == original_sha, 'Frozen terminal receipt changed'
        repair_data['checkpoint_sha256'] = '0' * 64; atomic_json(repair, repair_data)
        try:
            write_results(spec, base / 'wrong_checkpoint.json', {'jobs': jobs})
            raise AssertionError('Unmatched export recovery accepted')
        except ValueError:
            pass
    root = Path(__file__).resolve().parents[1]
    return {'status': 'passed', 'device': 'cpu', 'gpu_jobs_executed': 0,
            'accuracy_scope': 'fabricated integer counts only; no benchmark readout',
            'checks': ['isolated_frozen_collector_dependency_import',
                       'explicit_failure_skip_and_deferred_frontier', 'paired_event_gt_support_required',
                       'deterministic_existing_event_bootstrap', 'seed_descriptive_scope',
                       'checkpoint_vs_artifact_bytes', 'full_vs_overlap_excluded_rows', 'machine_csv_and_text_report',
                       'source_only_same_seed_A0_tolerance_artifact_selection', 'different_family_excluded',
                       'missing_anchor_explicit', 'altered_missing_SHA_size_never_green', 'separate_source_event_comparisons',
                       'export_only_recovery_binds_frozen_receipt_and_checkpoint'],
            'source_sha256': {str(path): file_hash(path) for path in
                (root / 'evaluation/compression_collect.py', Path(__file__).resolve())}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = check(); atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2))
