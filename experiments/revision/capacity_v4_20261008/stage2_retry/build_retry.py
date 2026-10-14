"""Stage 2 retry (one permitted retry, ground rule 8) for the six runs killed by the 1020 s limit.

Diagnosis: wall-clock slowdown under foreign GPU load (killed at val epoch 37-41/50; uncontended runs 759-862 s).
Changes vs stage2_fp: max_seconds 1020 -> 1400 and a fresh run_dir suffix `_retry1` so the partial
attempt-1 checkpoints stay untouched. Graph, seed, recipe, code, capsule and inputs are identical.
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
SRC = json.loads((V4 / 'stage2_fp/manifest.json').read_text())
FAILED = ['A0_FP_s64', 'A0_FP_s91', 'A0_FP_s107', 'A3_FP_s91', 'A3_FP_s107', 'X1_FP_s42']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


(HERE / 'specs').mkdir(exist_ok=True)
jobs = []
for job in SRC['jobs']:
    if job['id'] not in FAILED:
        continue
    old_spec = V4 / 'stage2_fp/specs' / f"{job['id']}.json"
    spec = json.loads(old_spec.read_text())
    spec['run_dir'] = spec['run_dir'] + '_retry1'
    new_spec = HERE / 'specs' / f"{job['id']}.json"
    new_spec.write_text(json.dumps(spec, indent=1, sort_keys=True))
    new_spec.chmod(0o444)
    new = copy.deepcopy(job)
    new['argv'] = [str(new_spec) if a == str(old_spec) else a for a in job['argv']]
    new['inputs'] = [{'path': str(new_spec), 'sha256': sha(new_spec)} if i['path'] == str(old_spec) else i
                     for i in job['inputs']]
    new['max_seconds'] = 1400
    new['retry_of'] = {'attempt1_receipt': str(V4 / 'stage2_fp/attempts' / f"{job['id']}.attempt1.json"),
                       'reason': 'exit 124 wall-clock limit under foreign GPU load; no code fault'}
    jobs.append(new)
assert len(jobs) == len(FAILED)

manifest = {k: v for k, v in SRC.items() if k not in ('jobs', 'reused', 'scope', 'stage', 'forecast_basis',
                                                       'scientific_state', 'remaining_stage1_seconds')}
manifest.update(
    stage='2 retry (one permitted retry of six timed-out FP runs)',
    scope='A0 s64/s91/s107, A3 s91/s107, X1 s42; identical recipe; fresh run dirs',
    forecast_basis='uncontended 759-862 s; 1400 s hard limit',
    gpu_budget_seconds=6 * 1400,
    parent_manifest_sha256=sha(V4 / 'stage2_fp/manifest.json'),
    jobs=jobs,
)
out = HERE / 'manifest.json'
out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
out.chmod(0o444)
print(out, sha(out), len(jobs))
