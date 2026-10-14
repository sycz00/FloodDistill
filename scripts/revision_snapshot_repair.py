"""Correct runtime-bytecode pinning before any prompt3 scientific run starts."""
import copy
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import revision_queue as q


def repair():
    root = Path('/data1/schmalstieg/flooddistill/revision_20261005')
    pointer_path = root / 'queue/active_queue.json'
    pointer = q.read_json(pointer_path)
    old_path = Path(pointer['active_manifest'])
    old_dir = old_path.parent
    prior = q.read_json(old_path.with_name('manifest.state.json'))
    assert prior['stop_requested'] and not q.alive(prior.get('worker'))
    assert not any(r['status'] in {'running', 'launching'} for r in prior['jobs'].values())
    raw = q.read_json(old_path)
    source_old = Path(raw['env']['PYTHONPATH'].split(':')[0])
    old_code = raw['fingerprints']['code']
    files = q.read_json(source_old / 'source_identity.json')['files']
    selected = {rel: source_old / rel for rel in files
                if '__pycache__' not in Path(rel).parts and Path(rel).suffix not in {'.pyc', '.pyo'}}
    selected['experiments/revision_prepare_matched.py'] = REPO / 'experiments/revision_prepare_matched.py'
    selected['scripts/revision_snapshot_repair.py'] = Path(__file__).resolve()
    hashes = {rel: q.file_hash(p) for rel, p in sorted(selected.items())}
    code = q.digest(hashes)
    source_new = root / 'snapshots' / code[:16]
    source_new.mkdir(exist_ok=False)
    for rel, p in selected.items():
        dest = source_new / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest); dest.chmod(0o444)
    q.atomic_json(source_new / 'source_identity.json', {'code_sha256': code, 'files': hashes})
    new_dir = old_dir.with_name('queue_prompt3_source_only')
    new_dir.mkdir(exist_ok=False)
    (new_dir / 'specs').mkdir()
    wrapper = new_dir / 'revision_queue.py'
    shutil.copyfile(source_new / 'experiments/revision_queue.py', wrapper); wrapper.chmod(0o444)
    old_environment = old_dir / 'environment.json'
    environment = q.read_json(old_environment)
    environment.update(source_sha256=code, source_packaging='Source inputs only; runtime bytecode excluded; PYTHONDONTWRITEBYTECODE=1 on new jobs')
    new_environment = new_dir / 'environment.json'
    q.atomic_json(new_environment, environment); new_environment.chmod(0o444)
    replacements = [(str(source_old), str(source_new)), (str(old_dir), str(new_dir)),
                    (old_code, code), (q.file_hash(old_environment), q.file_hash(new_environment))]
    def replace(value):
        if isinstance(value, str):
            for a, b in replacements:
                value = value.replace(a, b)
            return value
        if isinstance(value, list): return [replace(v) for v in value]
        if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
        return value
    for path in (old_dir / 'specs').glob('*.json'):
        dest = new_dir / 'specs' / path.name
        q.atomic_json(dest, replace(q.read_json(path))); dest.chmod(0o444)
    jobs, changed = [], []
    cache = {}
    def pin(path):
        if path not in cache: cache[path] = q.file_hash(path)
        return {'path': path, 'sha256': cache[path]}
    for original in raw['jobs']:
        if original['cwd'] != str(source_old):
            jobs.append(copy.deepcopy(original)); continue
        record = prior['jobs'][original['id']]
        # The sole failed receipt was rejected before creating its command.
        assert not original.get('gpu') or not record['attempts'], original['id']
        assert not record['attempts'] or (original['id'] == 'P3_verify_N100' and
               record['attempts'][-1]['exit_code'] == 126 and not record['attempts'][-1].get('command'))
        job = replace(original)
        job['env']['PYTHONDONTWRITEBYTECODE'] = '1'
        job['inputs'] = [pin(item['path']) for item in job.get('inputs', [])
                         if '__pycache__' not in Path(item['path']).parts and Path(item['path']).suffix not in {'.pyc', '.pyo'}]
        job['source_packaging_replacement'] = {'previous_identity_sha256': record['identity_sha256'],
             'reason': 'Mutable interpreter caches were mistakenly pinned; source-only snapshot correction before any scientific command, without changing recipe/data/seeds/budgets',
             'previous_manifest': str(old_path), 'original_attempt_receipts': [a['receipt_path'] for a in record['attempts']]}
        jobs.append(job); changed.append(original['id'])
    manifest = replace({k: v for k, v in raw.items() if k != 'jobs'})
    manifest.update(jobs=jobs, previous_queue=str(old_path), manifest_revision='prompt3_source_only_snapshot_repair')
    manifest_path = new_dir / 'manifest.json'
    q.atomic_json(manifest_path, manifest); manifest_path.chmod(0o444)
    loaded = q.load_manifest(manifest_path)
    state_path = new_dir / 'manifest.state.json'
    q.initialize(manifest_path, state_path, loaded)
    with q.edit_state(state_path) as state:
        for job in loaded['jobs']:
            name = job['id']
            if name in changed:
                state['jobs'][name]['retired_previous_identity'] = copy.deepcopy(prior['jobs'][name])
            else:
                assert q.digest(job) == prior['jobs'][name]['identity_sha256'], name
                state['jobs'][name] = copy.deepcopy(prior['jobs'][name])
        state.update(paused=True, migration={'at': q.now(), 'previous_manifest': str(old_path),
             'previous_state_sha256': q.digest(prior), 'scientific_new_runs_started_before_repair': 0,
             'source_packaging_replacements': changed, 'legacy_job_identities_preserved': True,
             'failed_precommand_receipt_preserved': prior['jobs']['P3_verify_N100']['attempts'][-1]['receipt_path']})
    for name in ['cost_before_launch.json', 'preparation.json']:
        data = replace(q.read_json(old_dir / name)); data['source_packaging_repair'] = state['migration']
        q.atomic_json(new_dir / name, data)
    q.atomic_json(new_dir / 'migration.json', state['migration'])
    pointer.update(active_manifest=str(manifest_path), active_state=str(state_path), active_wrapper=str(wrapper),
                   previous_manifest=str(old_path), migration_receipt=str(new_dir / 'migration.json'))
    q.atomic_json(pointer_path, pointer)
    reference_dir = REPO / 'experiments/revision/prompt3_dry_verification'
    q.atomic_json(reference_dir / 'queue_resume.json', {'argv': [sys.executable, str(wrapper), 'resume', '--manifest', str(manifest_path)]})
    print(json.dumps({'manifest': str(manifest_path), 'snapshot': str(source_new), 'changed_unlaunched_jobs': len(changed),
                      'new_training_cells': 24, 'scientific_runs_started_before_repair': 0}))


if __name__ == '__main__':
    repair()
