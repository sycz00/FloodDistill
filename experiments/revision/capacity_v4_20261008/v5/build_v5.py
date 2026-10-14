"""Build the immutable v5 queue manifest (protocol_v5_addendum.yaml). Order = launch priority."""
import hashlib
import json
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
HERE = V4 / 'v5'
CODE = V4 / 'code' / 'v5'
CAPSULE = V4 / 'frozen' / '297031f5f4ae1df5'
DATA = Path('/data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008/v5')
PY = '/home/schmalstieg/miniconda3/envs/esi/bin/python'
SPEC = V4 / 'stage2_fp' / 'specs' / 'A3_FP_s64.json'      # base spec; seed/graph/run_dir overridden per job
RES, TERM, SEL = HERE / 'results', HERE / 'terminal', HERE / 'selections'
SEEDS = [42, 64, 91, 107]
LRS = {'5e-5': 5e-5, '5e-4': 5e-4, '5e-3': 5e-3}
RUNGS = ['R1', 'R2', 'R3', 'R4', 'A3N', 'A3']
src = json.loads((V4 / 'stage2_fp' / 'manifest.json').read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stage2(g, s):
    return V4 / 'results' / f'{g}_FP_s{s}.json'


COMMON_INPUTS = [CODE / 'rungs.py', SPEC, V4 / 'protocol_v4.yaml', V4 / 'protocol_v5_addendum.yaml']
jobs = []


def job(jid, script, args, *, deps=(), max_s=1400, free=20000, equals=None, extra_inputs=(), after_terminal=False,
        out=None):
    out = out or RES / f'{jid}.json'
    base = [] if script == 'select_v5.py' else ['--capsule', str(CAPSULE), '--spec', str(SPEC)]
    j = {'id': jid, 'kind': 'command', 'gpu': True, 'cwd': str(CAPSULE), 'deps': list(deps),
         'argv': [PY, str(CODE / script)] + base + [str(x) for x in args],
         'inputs': [{'path': str(p), 'sha256': sha(p)} for p in [CODE / script, *COMMON_INPUTS, *extra_inputs]],
         'max_seconds': max_s, 'shutdown_grace_seconds': 20, 'required_free_mb': free,
         'outputs': [{'path': str(out), 'type': 'json', 'required_keys': ['status'],
                      'equals': equals or {'status': 'complete'}}]}
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
    return job(jid, 'train_v5.py', args, deps=deps, max_s=3000 if method == 'sup' else 1400)


# E2 - low-bit recipe (Q1): A3, 4 seeds x {Q48, Q4b, F48}
e2 = []
for s in SEEDS:
    for arm, extra in (('Q48', ['--qdq', 'on', '--bits', 4, 8]), ('Q4b', ['--qdq', 'on', '--bits', 4, 4, '--stem8']),
                       ('F48', ['--qdq', 'off', '--bits', 4, 8])):
        jid = f'lb_{arm}_A3_s{s}'
        e2.append(job(jid, 'lowbit_v5.py', ['--fp-receipt', stage2('A3', s), '--out', RES / f'{jid}.json',
                                             '--run-dir', DATA / jid, '--graph', 'A3', '--seed', s, '--lr', 5e-4] + extra,
                      max_s=1200, equals={'status': 'complete', 'outcome': 'completed'},
                      extra_inputs=[V4 / 'code' / 'v3' / 'simulator_v4.py', V4 / 'simulator_adoption.json']))

# E1a - external readout: all Stage 2 FP checkpoints + E2
stage2_receipts = [stage2(g, s) for g in ['A0', 'A1', 'A2', 'A3', 'X1', 'X2'] for s in SEEDS]
job('eval_stage2_E2', 'terminal_v5.py', ['--out-dir', TERM, '--done', RES / 'eval_stage2_E2.json', '--receipts',
                                          *stage2_receipts, *[RES / f'{j}.json' for j in e2]],
    deps=e2, after_terminal=True, max_s=3600, free=12000, out=RES / 'eval_stage2_E2.json',
    extra_inputs=[V4 / 'code' / 'v3' / 'simulator_v4.py'])

# E3 - capacity floor LR grid at seed 42 (A3 5e-5 reused from Stage 2)
grid = {}
for g in RUNGS:
    for tag, lr in LRS.items():
        jid = f'fl_{g}_lr{tag}_s42'
        reuse = [(5e-5, stage2('A3', 42))] if g == 'A3' else []
        grid[(g, tag)] = train(jid, g, 'lkd', 42, lr=lr, reuse=reuse)

# E4 - training-method comparison on A3 (LKD = Stage 2 runs)
e4_a3 = [train(f'm_{m}_A3_s{s}', 'A3', m, s, lr=5e-5) for m in ('ofa', 'ce', 'sup') for s in SEEDS]

# E3 - LR selection, then seed expansion at the selected LR (seed 42 reuses its grid run)
sel = {}
for g in RUNGS:
    out = SEL / f'lr_{g}.json'
    sel[g] = job(f'sel_lr_{g}', 'select_v5.py', ['lr', '--out', out, '--receipts',
                                                 *[RES / f'{grid[(g, t)]}.json' for t in LRS]],
                 deps=[grid[(g, t)] for t in LRS], max_s=120, free=0, out=out)
expansion = []
for g in RUNGS:
    for s in SEEDS:
        reuse = [(lr, RES / f'{grid[(g, t)]}.json') for t, lr in LRS.items()] if s == 42 else (
            [(5e-5, stage2('A3', s))] if g == 'A3' else [])
        expansion.append(train(f'fl_{g}_sel_s{s}', g, 'lkd', s, lr_from=SEL / f'lr_{g}.json', reuse=reuse,
                               deps=[sel[g]]))
floor_sel = SEL / 'floor.json'
job('sel_floor', 'select_v5.py', ['floor', '--reference', 'A3', '--out', floor_sel, '--receipts',
                                  *[RES / f'{j}.json' for j in expansion]],
    deps=expansion, max_s=120, free=0, out=floor_sel)

# E4 - method comparison at the floor rung F (LKD = fl_F_sel_*), then at A0 (LKD = Stage 2)
e4_floor = [train(f'mf_{m}_s{s}', None, m, s, graph_from=floor_sel, deps=['sel_floor'])
            for m in ('ce', 'sup', 'ofa') for s in SEEDS]
e4_a0 = [train(f'm_{m}_A0_s{s}', 'A0', m, s, lr=5e-5) for m in ('ce', 'sup', 'ofa') for s in SEEDS]

# E1b - final external readout of every v5 training checkpoint
v5_train = [j['id'] for j in jobs if j['argv'][1].endswith('train_v5.py')]
job('eval_final', 'terminal_v5.py', ['--out-dir', TERM, '--done', RES / 'eval_final.json', '--receipts',
                                      *stage2_receipts, *[RES / f'{j}.json' for j in e2 + v5_train]],
    deps=v5_train + ['eval_stage2_E2'], after_terminal=True, max_s=5400, free=12000, out=RES / 'eval_final.json',
    extra_inputs=[V4 / 'code' / 'v3' / 'simulator_v4.py'])

manifest = {k: v for k, v in src.items() if k in ('env', 'gpu_devices', 'gpu_lock', 'headroom_mb', 'host_limits',
                                                   'nice', 'poll_seconds', 'programme_target_seconds', 'schema_version')}
manifest.update(stage='v5 (user directive 2026-10-09): E2 low-bit recipe, E1 external readout, E3 capacity floor, '
                      'E4 training-method comparison',
                protocol=str(V4 / 'protocol_v5_addendum.yaml'), protocol_sha256=sha(V4 / 'protocol_v5_addendum.yaml'),
                gpu_budget_seconds=sum(j['max_seconds'] for j in jobs),
                budget_note='reservation = sum of hard limits (upper bound); expected actual ~20 GPU-h',
                jobs=jobs)
for d in (RES, TERM, SEL):
    d.mkdir(parents=True, exist_ok=True)
out = HERE / 'manifest.json'
out.write_text(json.dumps(manifest, indent=1)); out.chmod(0o444)
print(out, sha(out), len(jobs), 'jobs; reserved h', round(manifest['gpu_budget_seconds'] / 3600, 1),
      'size KB', out.stat().st_size // 1024)
