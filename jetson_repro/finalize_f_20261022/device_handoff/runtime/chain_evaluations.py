#!/usr/bin/env python3
"""Wait for old workers, import compatible completed work, then finish a new attempt."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_remaining import sha, save
from validate_results import validate_rows, validate_dry, benchmark_receipts_valid

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('s11val', 's11test', 'bolivia', 'wf', 'sturm512')


def read(path):
    return json.loads(Path(path).read_text())


def same_recipe(old, new):
    keys = ('onnx_sha256', 'io_sha256', 'mode', 'fp32_islands_sha256')
    return all(old.get(k) == new.get(k) for k in keys) and old.get('island_constraint_mode', 'obey') == new.get('island_constraint_mode', 'obey')


def child_alive(attempt):
    path = attempt / 'HEARTBEAT.json'
    if not path.exists():
        return False
    old = read(path)
    pid = old.get('child_pid')
    if not pid:
        return False
    cmdline = Path('/proc') / str(pid) / 'cmdline'
    try:
        actual = cmdline.read_bytes().rstrip(b'\0').decode().split('\0')
    except FileNotFoundError:
        return False
    return actual == old.get('argv')


def acquire_predecessors(attempts, heartbeat, stop):
    """Hold source package/attempt locks so an old resume cannot overlap us."""
    handles = []
    paths = sorted({p for a in attempts for p in (a.parents[1] / '.controller.lock', a / 'controller.lock')})
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError('Missing predecessor lock; inspect its worker manually: ' + str(path))
        handle = path.open('r')
        handles.append(handle)
        while True:
            if stop.exists():
                raise InterruptedError('Chain stopped before launch')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                save(heartbeat, dict(state='WAITING_PREDECESSOR_LOCK', lock=str(path), chain_pid=os.getpid(), updated=time.time()))
                time.sleep(5)
    while any(child_alive(a) for a in attempts):
        if stop.exists():
            raise InterruptedError('Chain stopped before launch')
        save(heartbeat, dict(state='WAITING_ORPHANED_PREDECESSOR_CHILD', chain_pid=os.getpid(), updated=time.time()))
        time.sleep(5)
    return handles


def valid_build(root, model, dest):
    if model.get('reuse'):
        spec = model['reuse']
        if read(dest / 'REUSE.json') != spec:
            raise ValueError('Reused engine specification changed')
        build = read(root / spec['build_receipt'])
        engine = root / spec['engine']
    else:
        build = read(dest / 'build.json')
        engine = dest / 'model.engine'
    if build.get('status') != 'built' or build.get('model_sha256') != model['onnx_sha256'] or build.get('mode') != model['mode'] or build.get('engine_sha256') != sha(engine):
        raise ValueError('Predecessor engine/model/build identity failed')
    if model['precision'] == 'pure' and not build.get('precision_audit', {}).get('full_integer_tensor_path'):
        raise ValueError('Predecessor integer certificate failed')
    if model.get('fp32_islands'):
        from precision_islands import validate_certificate
        validate_certificate(build, model, root, read(Path(str(engine) + '.layers.json')))
    if model.get('integer_repair_certificate'):
        from repair_policy import integer_certificate
        integer_certificate(build, read(Path(str(engine) + '.layers.json')), model)
    return build


def valid_complete(root, model, dest, state):
    build = valid_build(root, model, dest)
    complete = read(dest / 'evaluation/COMPLETE.json')
    if complete['engine_sha256'] != build['engine_sha256']:
        raise ValueError('Predecessor evaluation belongs to another engine')
    validate_dry(root, read(dest / 'evaluation/dry159.json')['rows'])
    if not model.get('reuse'):
        if complete.get('io_sha256') != model.get('io_sha256'):
            raise ValueError('Predecessor IO changed')
        for ds in DATASETS:
            validate_rows(root, ds, read(dest / 'evaluation' / (ds + '.json'))['rows'])
        timing = read(dest / 'evaluation/timing.json')
        profile = read(dest / 'benchmark/bench.json')
        power = read(dest / 'steady_power/power_steady.json')
        if any(r.get('engine_sha256') != build['engine_sha256'] for r in (timing, profile)):
            raise ValueError('Predecessor benchmark engine identity failed')
        if power.get('engine_sha256') is not None:
            if power['engine_sha256'] != build['engine_sha256']:
                raise ValueError('Predecessor power engine identity failed')
        elif Path(power.get('engine', '')).resolve() != (dest / 'model.engine').resolve():
            raise ValueError('Legacy power receipt must name the verified predecessor engine')
        if not benchmark_receipts_valid(profile, power, state):
            raise ValueError('Incomplete predecessor benchmark')
        if model['precision'] == 'fp32':
            from evaluate_native import summarize
            actual = summarize(read(dest / 'evaluation/s11val.json')['rows'])['pooled']['miou']
            expected = read(root / model['source_reference'])['mIoU']
            if abs(actual - expected) > .005:
                raise ValueError('Predecessor FP32 wiring failed')
    return True


def import_previous(root, attempt, predecessors):
    """Copy results, never change old attempts or fabricate missing measurements."""
    out = root / 'results' / attempt
    if out.exists():
        raise FileExistsError('Choose a fresh chain attempt; use --resume only after a controlled stop')
    out.mkdir(parents=True)
    current = read(root / 'RUN_MANIFEST.json')['engines']
    previous = []
    for source in predecessors:
        oldroot = source.parents[1]
        manifest = {m['id']: m for m in read(oldroot / 'RUN_MANIFEST.json')['engines']}
        states = {s['id']: s for s in read(source / 'STATUS.json')} if (source / 'STATUS.json').exists() else {}
        previous.append((source, manifest, states))
    plan, states = [], []
    for model in current:
        name = model['id']
        decision = dict(id=name, action='RUN_CURRENT_RECIPE')
        for source, oldmodels, oldstates in previous:
            old = oldmodels.get(name)
            if old is None or not same_recipe(old, model):
                continue
            state = oldstates.get(name)
            if not state:
                continue
            src = source / name
            if state['status'].startswith('FAILED'):
                log = src / 'evaluation.log'
                text = log.read_text(errors='replace') if log.is_file() else ''
                if state['status'] == 'FAILED_EVALUATION' and ("AssertionError: ('input', 'sturm512'" in text or 'Unrecognised input bytes sturm512' in text):
                    # This is a confirmed input-admission defect, not a completed accuracy failure.
                    decision.update(action='RUN_WITH_FIXED_STURM_PROFILE', predecessor=str(source))
                    break
                # Same-recipe scientific failures stay terminal, never auto-retried.
                if src.is_dir():
                    shutil.copytree(src, out / name, symlinks=True)
                states.append(dict(state, inherited_from=str(source)))
                decision.update(action='RETAIN_SAME_RECIPE_FAILURE', predecessor=str(source))
                break
            try:
                valid_complete(root, model, src, state)
            except (OSError, KeyError, ValueError, AssertionError) as error:
                decision.setdefault('incompatible_receipts', []).append(dict(predecessor=str(source), reason=str(error)))
                continue
            shutil.copytree(src, out / name, symlinks=True)
            states.append(dict(state, inherited_from=str(source)))
            decision.update(action='REUSE_VERIFIED_COMPLETE', predecessor=str(source), evidence='Copied receipts retain their original timestamps/input identities/session conditions')
            break
        plan.append(decision)
    save(out / 'STATUS.json', states)
    save(out / 'CHAIN_IMPORT.json', dict(predecessors=[str(p) for p in predecessors], plan=plan,
        note='No old result was overwritten. Changed B0 precision policies and new micro IDs are new recipes. Reused measurements retain their original hardware session and recorded input profiles.'))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predecessor', type=Path, action='append', required=True, help='Repeat for an earlier micro repair attempt if one exists')
    p.add_argument('--attempt', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--sturm-root', type=Path, required=True)
    p.add_argument('--wf-root', type=Path, required=True)
    a = p.parse_args()
    if Path(a.attempt).name != a.attempt or a.attempt in ('.', '..'):
        p.error('attempt must be one directory name')
    predecessors = [s.resolve() for s in a.predecessor]
    if any(s.parents[1] == ROOT for s in predecessors):
        p.error('Use a fresh separately unpacked handoff; never update an active predecessor in place')
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha(ROOT / name) != digest:
            raise ValueError('Current package hash mismatch ' + name)
    logs = ROOT / 'results_logs'
    logs.mkdir(exist_ok=True)
    chain_lock = (logs / (a.attempt + '.chain.lock')).open('a')
    fcntl.flock(chain_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    heartbeat = logs / (a.attempt + '.chain.json')
    handles = acquire_predecessors(predecessors, heartbeat, logs / (a.attempt + '.chain.STOP'))
    try:
        if not a.resume:
            import_previous(ROOT, a.attempt, predecessors)
        else:
            prior = read(ROOT / 'results' / a.attempt / 'CHAIN_IMPORT.json')
            if prior['predecessors'] != [str(s) for s in predecessors]:
                raise ValueError('Resume predecessor identity changed')
        save(heartbeat, dict(state='RUNNING_CURRENT_FOLLOWUP', chain_pid=os.getpid(), updated=time.time()))
        code = subprocess.call([sys.executable, str(ROOT / 'runtime/run_remaining.py'), '--attempt', a.attempt,
            '--resume', '--sturm-root', str(a.sturm_root), '--wf-root', str(a.wf_root)], stdin=subprocess.DEVNULL)
        save(heartbeat, dict(state='FOLLOWUP_EXITED', returncode=code, updated=time.time()))
        return code
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
