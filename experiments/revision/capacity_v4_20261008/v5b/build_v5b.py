"""Follow-up manifest v5b: the one permitted retry of a diagnosed technical failure in v5.

fl_A3_lr5e-5_s42 (a reuse-only job) crashed with KeyError: Stage 2 receipts carry no parameter count in
graph_record. Its dependants (sel_lr_A3, fl_A3_sel_*, sel_floor, mf_*) were blocked. train_v5b.py differs from
train_v5.py only by counting parameters from the graph. Jobs, arguments, rules and outputs are otherwise identical
to v5. The v5 manifest is untouched. v5b GPU jobs cannot start before the v5 worker releases the shared GPU lock,
so all v5 inputs (grid and other rung expansions) exist when v5b runs.
"""
import hashlib
import json
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
HERE = V4 / 'v5b'
CODE = V4 / 'code' / 'v5'
CAPSULE = V4 / 'frozen' / '297031f5f4ae1df5'
DATA = Path('/data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008/v5')
PY = '/home/schmalstieg/miniconda3/envs/esi/bin/python'
SPEC = V4 / 'stage2_fp' / 'specs' / 'A3_FP_s64.json'
RES, TERM, SEL = V4 / 'v5' / 'results', V4 / 'v5' / 'terminal', V4 / 'v5' / 'selections'
SEEDS = [42, 64, 91, 107]
LRS = {'5e-5': 5e-5, '5e-4': 5e-4, '5e-3': 5e-3}
RUNGS = ['R1', 'R2', 'R3', 'R4', 'A3N', 'A3']
v5 = json.loads((V4 / 'v5' / 'manifest.json').read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stage2(g, s):
    return V4 / 'results' / f'{g}_FP_s{s}.json'


COMMON = [CODE / 'rungs.py', SPEC, V4 / 'protocol_v4.yaml', V4 / 'protocol_v5_addendum.yaml']
jobs = []


def job(jid, script, args, *, deps=(), max_s=1400, free=20000, extra=(), after_terminal=False, out=None):
    out = out or RES / f'{jid}.json'
    base = [] if script == 'select_v5.py' else ['--capsule', str(CAPSULE), '--spec', str(SPEC)]
    j = {'id': jid, 'kind': 'command', 'gpu': True, 'cwd': str(CAPSULE), 'deps': list(deps),
         'argv': [PY, str(CODE / script)] + base + [str(x) for x in args],
         'inputs': [{'path': str(p), 'sha256': sha(p)} for p in [CODE / script, *COMMON, *extra]],
         'max_seconds': max_s, 'shutdown_grace_seconds': 20, 'required_free_mb': free,
         'outputs': [{'path': str(out), 'type': 'json', 'required_keys': ['status'], 'equals': {'status': 'complete'}}]}
    if after_terminal:
        j['run_after_terminal_deps'] = True
    jobs.append(j)
    return jid


def train(jid, graph, method, seed, *, lr=None, lr_from=None, graph_from=None, reuse=(), deps=()):
    args = ['--out', RES / f'{jid}.json', '--run-dir', DATA / jid, '--method', method, '--seed', seed]
    args += ['--graph', graph] if graph else []
    args += ['--lr', lr] if lr is not None else []
    args += ['--lr-from', lr_from] if lr_from else []
    args += ['--graph-from', graph_from] if graph_from else []
    args += (['--reuse'] + [f'{k}={v}' for k, v in reuse]) if reuse else []
    return job(jid, 'train_v5b.py', args, deps=deps, max_s=3000 if method == 'sup' else 1400)


grid_a3 = train('fl_A3_lr5e-5_s42', 'A3', 'lkd', 42, lr=5e-5, reuse=[(5e-5, stage2('A3', 42))])
sel_a3 = job('sel_lr_A3', 'select_v5.py', ['lr', '--out', SEL / 'lr_A3.json', '--receipts',
                                           *[RES / f'fl_A3_lr{t}_s42.json' for t in LRS]],
             deps=[grid_a3], max_s=120, free=0, out=SEL / 'lr_A3.json')
exp_a3 = []
for s in SEEDS:
    reuse = [(lr, RES / f'fl_A3_lr{t}_s42.json') for t, lr in LRS.items()] if s == 42 else [(5e-5, stage2('A3', s))]
    exp_a3.append(train(f'fl_A3_sel_s{s}', 'A3', 'lkd', s, lr_from=SEL / 'lr_A3.json', reuse=reuse, deps=[sel_a3]))
floor_sel = SEL / 'floor.json'
job('sel_floor', 'select_v5.py', ['floor', '--reference', 'A3', '--out', floor_sel, '--receipts',
                                  *[RES / f'fl_{g}_sel_s{s}.json' for g in RUNGS for s in SEEDS]],
    deps=exp_a3, max_s=120, free=0, out=floor_sel)
mf = [train(f'mf_{m}_s{s}', None, m, s, graph_from=floor_sel, deps=['sel_floor'])
      for m in ('ce', 'sup', 'ofa') for s in SEEDS]
v5_train = [j['id'] for j in v5['jobs'] if j['argv'][1].endswith('train_v5.py') or 'lowbit' in j['argv'][1]]
job('eval_v5b', 'terminal_v5.py', ['--out-dir', TERM, '--done', RES / 'eval_v5b.json', '--receipts',
                                   *[V4 / 'results' / f'{g}_FP_s{s}.json' for g in ['A0', 'A1', 'A2', 'A3', 'X1', 'X2'] for s in SEEDS],
                                   *[RES / f'{j}.json' for j in sorted(set(v5_train) | {j['id'] for j in jobs if 'train_v5b' in j['argv'][1]})]],
    deps=[grid_a3, *exp_a3, *mf], after_terminal=True, max_s=5400, free=12000, out=RES / 'eval_v5b.json',
    extra=[V4 / 'code' / 'v3' / 'simulator_v4.py'])

manifest = {k: v for k, v in v5.items() if k not in ('jobs', 'stage', 'gpu_budget_seconds', 'budget_note',
                                                      'protocol', 'protocol_sha256')}
manifest.update(stage='v5b: one retry of the diagnosed v5 reuse failure and its blocked dependants',
                parent_manifest_sha256=sha(V4 / 'v5' / 'manifest.json'),
                protocol=str(V4 / 'protocol_v5_addendum.yaml'), protocol_sha256=sha(V4 / 'protocol_v5_addendum.yaml'),
                retry_of={'job': 'fl_A3_lr5e-5_s42', 'reason': "KeyError 'parameters' in reuse branch (code bug, no science changed)",
                          'attempt_receipt': str(V4 / 'v5' / 'attempts' / 'fl_A3_lr5e-5_s42.attempt1.json')},
                gpu_budget_seconds=sum(j['max_seconds'] for j in jobs), jobs=jobs)
out = HERE / 'manifest.json'
out.write_text(json.dumps(manifest, indent=1)); out.chmod(0o444)
print(out, sha(out), len(jobs), 'jobs; reserved h', round(manifest['gpu_budget_seconds'] / 3600, 1))
