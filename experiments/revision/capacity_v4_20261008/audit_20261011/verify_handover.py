"""Read-only provenance audit; preserve derived tables before the mandated T0 rerun."""
import collections
import datetime
import hashlib
import importlib.util
import json
import pathlib
import shutil
import statistics
import subprocess
import sys

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent
CACHE = {}


def sha(path):
    path = pathlib.Path(path)
    if str(path) not in CACHE:
        h = hashlib.sha256()
        with path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        CACHE[str(path)] = h.hexdigest()
    return CACHE[str(path)]


def check(records):
    results = []
    for r in records:
        p = pathlib.Path(r['path'])
        actual = sha(p) if p.is_file() else None
        results.append({**r, 'actual_sha256': actual, 'match': actual == r['sha256']})
    return results


audit = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
capsule = BASE / 'frozen/297031f5f4ae1df5'
identity = json.loads((capsule / 'source_identity.json').read_text())
audit['capsule'] = check([{'path': str(capsule / k), 'sha256': v} for k, v in identity['files'].items()])
audit['capsule_unlisted_files'] = sorted(str(p.relative_to(capsule)) for p in capsule.rglob('*')
    if p.is_file() and p.name != 'source_identity.json' and '__pycache__' not in p.parts
    and str(p.relative_to(capsule)) not in identity['files'])
audit['queues'] = {}
for directory in sorted(BASE.iterdir()):
    mp, sp = directory / 'manifest.json', directory / 'manifest.state.json'
    if not mp.exists() or not sp.exists():
        continue
    m, s = json.loads(mp.read_text()), json.loads(sp.read_text())
    module_spec = importlib.util.spec_from_file_location('audit_queue_' + directory.name, directory / 'revision_queue.py')
    queue = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(queue)
    loaded = queue.load_manifest(str(mp))
    canonical_sha = queue.digest(loaded)
    inputs = {(r['path'], r['sha256']) for j in m['jobs'] for r in j.get('inputs', [])}
    verified = check([{'path': p, 'sha256': h} for p, h in sorted(inputs)])
    gpu_ids = {j['id'] for j in m['jobs'] if j.get('gpu')}
    elapsed = sum(a.get('command_elapsed_seconds', 0) for name, j in s['jobs'].items()
                  for a in j.get('attempts', []) if name in gpu_ids)
    audit['queues'][directory.name] = {
        'manifest_sha256': sha(mp), 'state_manifest_sha256': s.get('manifest_sha256'),
        'canonical_loaded_manifest_sha256': canonical_sha,
        'manifest_match': canonical_sha == s.get('manifest_sha256'),
        'worker_status': s.get('worker_status'),
        'counts': dict(collections.Counter(j['status'] for j in s['jobs'].values())),
        'gpu_command_seconds': elapsed, 'poll_seconds': m.get('poll_seconds'),
        'inputs': verified,
    }
audit['shared_input_identity'] = check(json.loads((BASE / 'shared_input_identity.json').read_text())['files'])
audit['artifact_checks'] = []
audit['early_stops'] = []
audit['skipped_receipts'] = []
for p in sorted(list((BASE / 'results').glob('*_FP_s*.json')) + list((BASE / 'v5/results').glob('*.json'))):
    r = json.loads(p.read_text())
    if r.get('outcome') == 'skipped':
        audit['skipped_receipts'].append({'receipt': str(p), 'reason': r.get('reason')})
    if r.get('completed_updates', 7800) < 7800 and 'source_validation' in r:
        audit['early_stops'].append({'receipt': str(p), 'graph': r.get('graph'), 'seed': r.get('seed'),
            'updates': r.get('completed_updates'), 'selected_updates': r.get('selected_updates'),
            'best_path': r.get('best_path'), 'lr': r.get('lr')})
    for prefix in ('best', 'last', 'fp_stripped'):
        if r.get(prefix + '_path') and r.get(prefix + '_sha256'):
            c = check([{'path': r[prefix + '_path'], 'sha256': r[prefix + '_sha256']}])[0]
            c['receipt'] = str(p)
            audit['artifact_checks'].append(c)
terminals = list((BASE / 'v5/terminal').glob('*.json'))
audit['terminal_checkpoint_checks'] = []
for p in sorted(terminals):
    r = json.loads(p.read_text())
    if r.get('checkpoint_path') and r.get('checkpoint_sha256'):
        audit['terminal_checkpoint_checks'] += check([{'path': r['checkpoint_path'], 'sha256': r['checkpoint_sha256']}])
audit['terminal_receipt_count'] = len(terminals)
before = OUT / 'analysis_before_t0'
before.mkdir(exist_ok=True)
for p in (BASE / 'v5/analysis').iterdir():
    if p.is_file() and not (before / p.name).exists():
        shutil.copy2(p, before / p.name)
audit['analysis_before'] = {p.name: sha(p) for p in (BASE / 'v5/analysis').iterdir() if p.is_file()}
provenance_ok = (all(r['match'] for r in audit['capsule']) and not audit['capsule_unlisted_files']
    and all(r['match'] for r in audit['shared_input_identity'])
    and all(q['manifest_match'] and all(r['match'] for r in q['inputs']) for q in audit['queues'].values())
    and all(r['match'] for r in audit['artifact_checks'] + audit['terminal_checkpoint_checks']))
audit['provenance_pass'] = provenance_ok
if provenance_ok:
    result = subprocess.run([sys.executable, str(BASE / 'code/v5/analyze_v5.py'), str(BASE)],
                            text=True, capture_output=True, cwd=str(BASE.parents[2]))
    (OUT / 'analyze_v5_rerun.txt').write_text(result.stdout)
    (OUT / 'analyze_v5_rerun.stderr.txt').write_text(result.stderr)
    audit['analyzer_exit_code'] = result.returncode
    if result.returncode == 0:
        summary = json.loads((BASE / 'v5/analysis/summary.json').read_text())
        expected = [('H6a', 'E2 Q48 - F48 (H6a quantization-specific)', 'val', -.25),
            ('A3 LR effect', 'E3 A3 selected LR - Stage2 recipe LR', 'val', 4.14),
            ('R4 - A3', 'E3 R4 - A3', 'val', 1.37),
            ('A0 LKD - CE STURM', 'E4 A0: LKD - CE', 'sturm:1', 2.72),
            ('R2 LKD - CE STURM', 'E4 F: LKD - CE', 'sturm:1', 5.42)]
        # The frozen analyzer omits this reported contrast; derive it from matched seed receipts.
        from scipy import stats
        lr_deltas = []
        for seed in (42, 64, 91, 107):
            chosen = json.loads((BASE / f'v5/results/fl_A3_sel_s{seed}.json').read_text())
            parent = json.loads((BASE / f'results/A3_FP_s{seed}.json').read_text())
            lr_deltas.append(100 * (chosen['source_validation']['best_val_mIoU']
                                    - parent['source_validation']['best_val_mIoU']))
        avg, sd = statistics.mean(lr_deltas), statistics.stdev(lr_deltas)
        half = stats.t.ppf(.975, 3) * sd / 2
        summary['contrasts'][expected[1][1]] = {'val': {'n': 4, 'mean': avg, 'sd': sd,
                                                       'lo': avg - half, 'hi': avg + half}}
        audit['lr_effect_seed_deltas_pp'] = lr_deltas
        audit['numeric_checks'] = []
        for label, key, field, target in expected:
            value = summary['contrasts'][key][field]
            audit['numeric_checks'].append({'label': label, 'contrast': key, 'field': field,
                'expected_pp': target, **value, 'pass': abs(value['mean'] - target) <= .01})
        audit['t0_pass'] = all(c['pass'] for c in audit['numeric_checks'])
    else:
        audit['t0_pass'] = False
else:
    audit['t0_pass'] = False
(OUT / 't0_verification.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k not in ('capsule', 'shared_input_identity', 'artifact_checks',
    'terminal_checkpoint_checks', 'queues')}, indent=2))
for name, q in audit['queues'].items():
    print(name, q['worker_status'], q['counts'], 'GPU-h', round(q['gpu_command_seconds']/3600, 4),
          'input mismatches', sum(not r['match'] for r in q['inputs']))
if not audit['t0_pass']:
    sys.exit(1)
