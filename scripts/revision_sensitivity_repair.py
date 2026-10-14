"""Supply a missing pinned audit asset; rerun only pre-inference failures."""
import copy
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import revision_queue as q


def prepare():
    root = Path('/data1/schmalstieg/flooddistill/revision_20261005')
    pointer_path = root / 'queue/active_queue.json'
    pointer = q.read_json(pointer_path)
    old_path = Path(pointer['active_manifest'])
    prior = q.read_json(pointer['active_state'])
    assert prior['worker_status'] == 'finished' and not q.alive(prior.get('worker'))
    assert not any(r['status'] in {'running', 'launching'} for r in prior['jobs'].values())
    old = q.load_manifest(old_path)
    raw = q.read_json(old_path)
    old_snapshot = Path(raw['env']['PYTHONPATH'].split(':')[0])
    failed = [j for j in old['jobs'] if prior['jobs'][j['id']]['status'] == 'failed']
    assert len(failed) == 104
    missing = 'experiments/revision/prompt2_reference_manifest.json'
    for job in failed:
        attempt = prior['jobs'][job['id']]['attempts'][-1]
        log = Path(attempt['log_path']).read_text()
        assert 'FileNotFoundError' in log and missing in log
        assert job['argv'][job['argv'].index('--mode') + 1] in {'worldfloods', 'sturm_valid'}
        assert all(not Path(output['path']).exists() for output in job['outputs'])
    files = q.read_json(old_snapshot / 'source_identity.json')['files']
    sources = {rel: old_snapshot / rel for rel in files
               if '__pycache__' not in Path(rel).parts and Path(rel).suffix not in {'.pyc', '.pyo'}}
    for rel in [missing, 'experiments/revision/prompt2_worldfloods_windows.json',
                'experiments/revision/prompt3_evaluation_plan.json',
                'evaluation/revision_collect.py', 'scripts/revision_sensitivity_repair.py']:
        sources[rel] = REPO / rel
    hashes = {rel: q.file_hash(path) for rel, path in sorted(sources.items())}
    code = q.digest(hashes)
    snapshot = root / 'snapshots' / code[:16]
    snapshot.mkdir(exist_ok=False)
    for rel, path in sources.items():
        dest = snapshot / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest); dest.chmod(0o444)
    q.atomic_json(snapshot / 'source_identity.json', {'code_sha256': code, 'files': hashes})
    dest_dir = old_path.parent.with_name('queue_prompt3_sensitivity_repair')
    dest_dir.mkdir(exist_ok=False)
    wrapper = dest_dir / 'revision_queue.py'
    shutil.copyfile(snapshot / 'experiments/revision_queue.py', wrapper); wrapper.chmod(0o444)
    pins = {rel: {'path': str(snapshot / rel), 'sha256': checksum} for rel, checksum in hashes.items()}
    failed_ids = {j['id'] for j in failed}
    jobs = copy.deepcopy(raw['jobs'])
    first = ['R_wf_native_teacher', 'R_sturm_valid_teacher']
    for job in jobs:
        if job['id'] not in failed_ids:
            continue
        job['cwd'] = str(snapshot)
        job['argv'] = [arg.replace(str(old_snapshot), str(snapshot)) for arg in job['argv']]
        job['env']['PYTHONPATH'] = job['env']['PYTHONPATH'].replace(str(old_snapshot), str(snapshot))
        job['env']['PYTHONDONTWRITEBYTECODE'] = '1'
        inputs = {item['path']: item for item in job['inputs'] if not Path(item['path']).is_relative_to(old_snapshot)}
        inputs.update({pin['path']: pin for pin in pins.values()})
        job['inputs'] = list(inputs.values())
        job['fingerprints'] = {**job.get('fingerprints', {}), 'code': code,
                               'evaluation_repair': 'Missing audit asset supplied; evaluator/input/mask/checkpoint definitions unchanged'}
        if job.get('gpu') and job['id'] not in first:
            job['deps'] = list(dict.fromkeys(job['deps'] + first))
        job['pre_inference_packaging_repair'] = {
            'previous_manifest': str(old_path), 'previous_identity_sha256': prior['jobs'][job['id']]['identity_sha256'],
            'previous_receipt': prior['jobs'][job['id']]['attempts'][-1]['receipt_path'],
            'reason': 'Audit reference manifest missing from snapshot; verified failure before model load or prediction. Distinct corrected identity; no retraining or method change.',
            'added_asset': pins[missing]}
    collector = copy.deepcopy(next(j for j in raw['jobs'] if j['id'] == 'P3_collect'))
    collector['id'] = 'P3_collect_completed_sensitivity'
    collector['cwd'] = str(snapshot)
    collector['argv'] = [arg.replace(str(old_snapshot), str(snapshot)) for arg in collector['argv']]
    new_path = dest_dir / 'manifest.json'
    argv = collector['argv']
    argv[argv.index('--queue_manifest') + 1] = str(new_path)
    out_dir = old_path.parents[1] / 'analysis_prompt3_complete'
    argv[argv.index('--out_dir') + 1] = str(out_dir)
    collector['outputs'] = [{**out, 'path': out['path'].replace('/analysis_prompt3/', '/analysis_prompt3_complete/')}
                            for out in collector['outputs']]
    collector['env']['PYTHONPATH'] = collector['env']['PYTHONPATH'].replace(str(old_snapshot), str(snapshot))
    collector['env']['PYTHONDONTWRITEBYTECODE'] = '1'
    collector['inputs'] = [item for item in collector['inputs'] if not Path(item['path']).is_relative_to(old_snapshot)] + list(pins.values())
    collector['fingerprints'] = {**collector.get('fingerprints', {}), 'code': code}
    collector['deps'] = sorted(failed_ids)
    collector['run_after_terminal_deps'] = True
    jobs.append(collector)
    priority = {name: i for i, name in enumerate(first)}
    jobs.sort(key=lambda j: (0 if j['id'] not in failed_ids else 1 if j['id'] in first else 2, priority.get(j['id'], 2), j['id']))
    manifest = {**raw, 'jobs': jobs, 'manifest_revision': 'prompt3_missing_audit_asset_sensitivity_repair',
                'previous_queue': str(old_path), 'source_repair_snapshot': str(snapshot),
                'repair_scope': {'failed_eval_only_jobs': 104, 'new_training_runs': 0,
                                 'asset': pins[missing], 'original_failures_preserved': True}}
    q.atomic_json(new_path, manifest); new_path.chmod(0o444)
    loaded = q.load_manifest(new_path)
    state_path = new_path.with_name('manifest.state.json')
    q.initialize(new_path, state_path, loaded)
    with q.edit_state(state_path) as state:
        for job in loaded['jobs']:
            name = job['id']
            if name in failed_ids:
                state['jobs'][name]['superseded_failed_identity'] = prior['jobs'][name]
            elif name != collector['id']:
                assert q.digest(job) == prior['jobs'][name]['identity_sha256'], name
                state['jobs'][name] = prior['jobs'][name]
        state.update(paused=True, migration={'at': q.now(), 'source_manifest': str(old_path),
                     'source_state_sha256': q.digest(prior), 'corrected_pre_inference_failures': sorted(failed_ids),
                     'all_completed_job_identities_preserved': True, 'new_training_runs': 0})
    q.atomic_json(dest_dir / 'migration.json', state['migration'])
    cost_rows = []
    for job in failed:
        spec = q.read_json(job['argv'][job['argv'].index('--spec') + 1])
        seconds = (25 + 2 * len(spec.get('conditions', ['clean']))) if '--mode' in job['argv'] and job['argv'][job['argv'].index('--mode') + 1] == 'sturm_valid' else 60
        if job.get('gpu'):
            cost_rows.append({'job': job['id'], 'category': 'additional', 'estimated_GPU_seconds': seconds,
                             'basis': 'Affected-input inference only plus audited reference/checkpoint verification; approximate until first successful GPU readout'})
    hours = sum(row['estimated_GPU_seconds'] for row in cost_rows) / 3600
    q.atomic_json(dest_dir / 'cost_before_launch.json', {'captured_at': q.now(), 'original_remaining_GPU_hours': 0,
                  'additional_remaining_GPU_hours': hours, 'total_remaining_GPU_hours': hours,
                  'ideal_two_GPU_wall_hours': hours / 2, 'jobs': cost_rows,
                  'uncertainty': 'Inference-only repair estimate; reference hashing and serialized CPU checks add wall time; use measured first successful durations to refresh.'})
    pointer.update(active_manifest=str(new_path), active_state=str(state_path), active_wrapper=str(wrapper),
                   previous_manifest=str(old_path), migration_receipt=str(dest_dir / 'migration.json'))
    q.atomic_json(pointer_path, pointer)
    q.atomic_json(REPO / 'experiments/revision/prompt3_dry_verification/queue_resume.json',
                  {'argv': [sys.executable, str(wrapper), 'resume', '--manifest', str(new_path)]})
    print({'manifest': str(new_path), 'snapshot': str(snapshot), 'eval_only_jobs': 104,
           'estimated_GPU_hours': hours, 'completed_tables': str(out_dir)})


if __name__ == '__main__':
    prepare()
