"""Read-only completion gates for the finite V11 DAG."""
from pathlib import Path
import json
import common11 as c

def receipt(path):
    value = c.read(path)
    return value if value and value.get('outcome') == 'completed' else None

def v10_ready(kind):
    missing = []
    required = ['teacher_logit_cache_manifest.json', 'receipts/01_teacher_cache.json']
    if kind in ('repair', 'transfer'):
        required += ['compression_control_selection.json', 'receipts/40_compression_controls.json']
        if kind == 'transfer': required += ['smallest_viable_selection.json']
    for name in required:
        value=receipt(c.V10 / name)
        if not value:
            missing.append(name)
        elif name.startswith('receipts/'):
            stage=c.V10/'stage_results'/Path(name).name
            if not value.get('result_sha256') or not stage.is_file() or c.sha(stage)!=value['result_sha256']:
                missing.append(name+' (stage receipt hash verification)')
    if kind == 'conditioner':
        for seed in c.SEEDS:
            name = f'scientific_receipts/R4_micro_KDoff_s{seed}.json'
            if not receipt(c.V10 / name): missing.append(name)
    if kind == 'repair':
        for seed in c.SEEDS:
            name = f'qat_parents/R4_W4A4_s{seed}.json'
            if not (c.V10 / name).is_file(): missing.append(name)
    # No checkpoint or cache shard is opened until its writer publishes completion.
    return missing

def check(root, job, state):
    root = Path(root)
    kind = job.get('dependency_kind')
    if kind in ('repair', 'transfer', 'conditioner'):
        missing = v10_ready(kind)
        if missing:
            terminal = c.read(c.V10 / 'state.json', {}).get('finished', False)
            return dict(status='UNAVAILABLE' if terminal else 'WAITING_DEPENDENCY',
                        reason='V10 atomically completed inputs required: ' + ', '.join(missing))
    if job.get('source_gate'):
        gate = c.read(root / 'repair/core/source_gate.json', {})
        admitted = gate.get('admitted', gate.get('outcome') == 'ADMITTED')
        if not admitted:
            return dict(status='DEFERRED', reason='Source point has no admitted material quantization deficit; retain diagnostic')
    if job.get('profile_gate'):
        profile = c.read(root / 'profile_admission.json', {})
        if not profile.get('admitted'):
            return dict(status='DEFERRED', reason=profile.get('reason', 'Complete matched block plus final evaluation has not passed measured cost admission'))
    if job.get('selected_models'):
        selection = c.read(root / 'final_evaluation_freeze.json', {})
        if not selection.get('models'):
            return dict(status='DEFERRED', reason='No complete source-frozen model matrix available for final readout')
    for item in job.get('external_dependencies', []):
        path = Path(item['path'])
        if not path.exists():return dict(status='WAITING_DEPENDENCY', reason='Missing ' + str(path))
        if item.get('sha256') and c.sha(path) != item['sha256']:
            return dict(status='UNAVAILABLE', reason='Frozen dependency hash changed: ' + str(path))
    return dict(status='READY', reason='Declared dependencies satisfied')
