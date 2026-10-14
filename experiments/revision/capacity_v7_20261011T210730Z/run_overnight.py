#!/usr/bin/env python3
"""Bounded v7 orchestration; audited queue helpers, fixed prospective decisions."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import audited_queue as q

ROOT = Path(__file__).resolve().parent
TERMINAL = {'completed', 'reused', 'technical_failure', 'blocked', 'deferred_budget',
            'failed_scientific_gate', 'not_run', 'unsupported'}


def read(path, default=None):
    return q.read_json(path) if Path(path).exists() else default


def stamp_seconds(value):
    return datetime.fromisoformat(value).timestamp()


def decision(root, kind, **values):
    with open(root / 'decision_log.jsonl', 'a', buffering=1) as stream:
        stream.write(json.dumps({'at': q.now(), 'kind': kind, **values}, allow_nan=False) + '\n')


def control(root):
    return read(root / 'control.json', {'paused': False, 'stop': False})


def ledger(root, state, manifest):
    attempts = [a for rec in state['jobs'].values() for a in rec.get('attempts', [])]
    spent = sum(a.get('command_elapsed_seconds', 0) for a in attempts if a.get('gpu') and a.get('finished_at'))
    active = sum(time.time() - stamp_seconds(a['command_started_at']) for a in attempts
                 if a.get('gpu') and a.get('command_started_at') and not a.get('finished_at'))
    reserved = sum(b['reserved_seconds'] for b in state['bundles'].values() if b['status'] == 'admitted')
    bootstrap = manifest.get('preflight_gpu_seconds', 0)
    collection_protected = manifest['collection_reserve_seconds'] if state['bundles'].get('collection', {}).get('status') == 'pending' else 0
    retry_protected = 0 if state.get('retry_reserve_released') else manifest.get('retry_reserve_seconds',0)
    result = {'updated_at': q.now(), 'historical_gpu_seconds': manifest['historical_gpu_seconds'],
              'incremental_cap_seconds': manifest['incremental_cap_seconds'],
              'preflight_gpu_seconds': bootstrap, 'completed_gpu_seconds': spent,
              'active_elapsed_seconds': max(0, active), 'reserved_gpu_seconds': reserved,
              'protected_collection_seconds': collection_protected, 'protected_retry_seconds': retry_protected,
              'reservation_releases': state.get('reservation_releases', []),
              'incremental_spent_seconds': bootstrap + spent,
              'cumulative_gpu_seconds': manifest['historical_gpu_seconds'] + bootstrap + spent,
              'available_unreserved_seconds': manifest['incremental_cap_seconds'] - bootstrap - spent - reserved,
              'deadline_at': state.get('deadline_at'), 'counting': 'elapsed GPU command once; includes failures and retries'}
    q.atomic_json(root / 'budget_ledger.json', result)
    return result


def save(root, state, manifest):
    state['heartbeat_at'] = q.now()
    state['control'] = control(root)
    q.atomic_json(root / 'state.json', state)
    q.atomic_json(root / 'heartbeat.json', {'at': state['heartbeat_at'], 'worker': state.get('worker'),
                                         'active_job': state.get('active_job'), 'status': state['worker_status']})
    ledger(root, state, manifest)


def terminate_owned(command):
    if not q.alive(command):
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(command['pid'], signal.SIGTERM)


def reap_owned_group(command, grace=5):
    """Recover a dead guardian without permitting its known session to spend untracked."""
    if not command:
        return
    current = q.process_identity(command['pid'])
    if current is not None and current != command:
        return  # PID reused; original session cannot share this new leader.
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != command['boot_id']:
        return
    def members():
        found=[]
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                stat=path.read_text();fields=stat[stat.rfind(')')+2:].split()
                if fields[0] != 'Z' and int(fields[2]) == command['pid'] and int(fields[3]) == command['pid']:
                    found.append(int(path.parent.name))
            except (OSError,ValueError,IndexError):
                continue
        return found
    if members():
        with contextlib.suppress(ProcessLookupError): os.killpg(command['pid'],signal.SIGTERM)
        end=time.monotonic()+grace
        while members() and time.monotonic()<end: time.sleep(.1)
        if members():
            with contextlib.suppress(ProcessLookupError): os.killpg(command['pid'],signal.SIGKILL)
        end=time.monotonic()+2
        while members() and time.monotonic()<end: time.sleep(.1)
        if members():raise RuntimeError('owned process group could not be reaped')


def attempt_runner(root, attempt_path, lock_fds):
    """Independent timeout guardian retains GPU locks if its worker exits."""
    a = read(attempt_path)
    job = a['job']
    a['runner'] = q.process_identity(os.getpid())
    q.atomic_json(attempt_path, a)
    cmd = None
    signalled = [False]
    signal.signal(signal.SIGTERM, lambda *_: signalled.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: signalled.__setitem__(0, True))
    began = None
    try:
        for path, expected in a['frozen_inputs'].items():
            if q.file_hash(path) != expected:
                raise ValueError('frozen input changed: ' + path)
        env = dict(os.environ)
        env.update(q.THREAD_ENV)
        env.update(job.get('env', {}))
        env.update(CUDA_VISIBLE_DEVICES=a['gpu_uuid'] if job['gpu'] else '', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
        os.nice(10)
        with open(a['log_path'], 'ab', buffering=0) as log:
            began = time.monotonic()
            cmd = subprocess.Popen(job['argv'], cwd=job['cwd'], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=tuple(lock_fds))
            a.update(command=q.process_identity(cmd.pid), command_started_at=q.now())
            q.atomic_json(attempt_path, a)
            hard_end = min(began + job['max_seconds'], began + max(0, a['deadline_epoch'] - time.time()))
            grace = min(20, max(1, job['max_seconds'] / 10))
            cause = None
            while cmd.poll() is None:
                if signalled[0] or control(root).get('stop'):
                    cause = 'controlled_stop'
                    break
                if time.monotonic() >= hard_end - grace:
                    cause = 'hard_timeout_or_deadline'
                    break
                time.sleep(.5)
            if cause:
                terminate_owned(a['command'])
                try:
                    cmd.wait(timeout=max(.1, min(grace, hard_end - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(cmd.pid, signal.SIGKILL)
                cmd.wait()
                a.update(exit_code=124 if cause == 'hard_timeout_or_deadline' else 130, error=cause)
            else:
                a['exit_code'] = cmd.wait()
                # Ensure owned DataLoader children cannot outlive a terminal command.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(cmd.pid, signal.SIGKILL)
            a['command_elapsed_seconds'] = time.monotonic() - began
            if a['exit_code'] == 0:
                valid, evidence = q.validate_outputs({'outputs': job['outputs']})
                a['output_validation'] = evidence
                if not valid:
                    a.update(exit_code=125, error=str(evidence))
    except BaseException as error:
        a.update(exit_code=126, error=f'{type(error).__name__}: {error}')
        if cmd is not None:
            terminate_owned(a.get('command'))
            with contextlib.suppress(ProcessLookupError):
                os.killpg(cmd.pid, signal.SIGKILL)
            cmd.wait()
        a['command_elapsed_seconds'] = time.monotonic() - began if began else 0
    a['finished_at'] = q.now()
    q.atomic_json(attempt_path, a)


def selected_score(receipt, metric='mIoU'):
    r = read(receipt)
    sv = r['source_validation']
    if metric == 'mIoU':
        return float(sv.get('best_val_mIoU', sv.get('best_val_miou')))
    selected = sv.get('selected_metrics', {})
    value = selected.get(metric, selected.get('water_iou', selected.get('water_IoU', selected.get('IoU_water'))))
    if value is None and 'iou_per_class' in selected:
        value = selected['iou_per_class'][1]
    return float(value)


def promotion_gate(receipts):
    """Only C3 can be promoted; source-only, exact declared thresholds in pp."""
    diffs = {}
    for comparator in ('C0', 'C4', 'C5'):
        diffs[comparator] = [100 * (selected_score(receipts[('C3', s)]) - selected_score(receipts[(comparator, s)]))
                             for s in (42, 64)]
    water = [100 * (selected_score(receipts[('C3', s)], 'wIoU') - selected_score(receipts[('C0', s)], 'wIoU'))
             for s in (42, 64)]
    criteria = {'mean_C3_C0_ge_0.50': sum(diffs['C0']) / 2 >= .50,
                'each_C3_C0_nonnegative': min(diffs['C0']) >= 0,
                'mean_C3_C4_ge_0.20': sum(diffs['C4']) / 2 >= .20,
                'mean_C3_C5_ge_0.20': sum(diffs['C5']) / 2 >= .20,
                'mean_water_C3_C0_ge_minus0.50': sum(water) / 2 >= -.50}
    return {'passed': all(criteria.values()), 'criteria': criteria, 'differences_pp': diffs,
            'water_differences_pp': water, 'data_role': 'source_validation_only', 'pilot_seeds': [42, 64]}


def initialize(root, manifest):
    old = read(root / 'state.json')
    if old:
        if old['manifest_sha256'] != q.digest(manifest):
            raise ValueError('existing state belongs to a different immutable manifest')
        return old
    state = {'created_at': q.now(), 'manifest_sha256': q.digest(manifest), 'worker_status': 'initialized',
             'manifest_file_sha256': q.file_hash(root / 'manifest_initial.json'),
             'worker': None, 'active_job': None, 'jobs': {}, 'bundles': {}}
    for bundle in manifest['bundles']:
        state['bundles'][bundle['id']] = {'status': 'pending', 'reserved_seconds': 0, 'reason': None}
        for job in bundle.get('jobs', []):
            state['jobs'][job['id']] = {'job': job, 'status': 'pending', 'attempts': [], 'reason': None}
    q.atomic_json(root / 'control.json', {'paused': False, 'stop': False})
    save(root, state, manifest)
    return state


def admission(root, state, manifest, bundle, jobs):
    cost = sum(j['max_seconds'] for j in jobs if j['gpu']) + bundle.get('future_reserve_seconds', 0)
    b = ledger(root, state, manifest)
    protect = manifest['collection_reserve_seconds'] if bundle['id'] != 'collection' else 0
    retry = (0 if state.get('retry_reserve_released') or bundle['id'] in {'collection', 'optional_A3_CE'}
             else manifest.get('retry_reserve_seconds', 0))
    remaining_wall = stamp_seconds(state['deadline_at']) - time.time()
    fits = (cost + protect + retry <= b['available_unreserved_seconds'] and
            sum(j['max_seconds'] for j in jobs) + bundle.get('future_reserve_seconds', 0) + protect + 120 <= remaining_wall)
    if not fits:
        reason = (f'full bundle reservation {cost}s plus collection {protect}s and retry {retry}s '
                  f'exceeds available budget {b["available_unreserved_seconds"]:.1f}s or wall {remaining_wall:.1f}s')
        state['bundles'][bundle['id']].update(status='deferred_budget', reason=reason)
        for j in jobs:
            state['jobs'][j['id']] = {'job': j, 'status': 'deferred_budget', 'reason': reason, 'attempts': []}
        decision(root, 'deferred_budget', bundle=bundle['id'], reason=reason)
        return False
    child = {'bundle': bundle['id'], 'jobs': jobs, 'created_at': q.now(), 'source_manifest_sha256': q.digest(manifest)}
    path = root / 'child_manifests' / (bundle['id'] + '.json')
    if path.exists() and read(path)['jobs'] != jobs:
        raise ValueError('immutable child manifest differs: ' + str(path))
    if not path.exists():
        q.atomic_json(path, child)
    state['bundles'][bundle['id']].update(status='admitted', reserved_seconds=cost, manifest=str(path),
                                        manifest_sha256=q.file_hash(path), reason=None)
    if bundle['id'] == 'optional_A3_CE' and not state.get('retry_reserve_released'):
        state['retry_reserve_released'] = True
        state.setdefault('reservation_releases', []).append({'at':q.now(),'reservation':'unused_technical_retry',
            'seconds':manifest.get('retry_reserve_seconds',0),'released_to':'declared_optional_priority'})
        decision(root,'reservation_released',reservation='unused_technical_retry',seconds=manifest.get('retry_reserve_seconds',0))
    for job in jobs:
        rec = state['jobs'].setdefault(job['id'], {'job': job, 'status': 'pending', 'attempts': [], 'reason': None})
        if rec['status'] == 'pending' and not rec['attempts']:
            rec['job'] = job
    decision(root, 'admitted', bundle=bundle['id'], reserved_seconds=cost, child_manifest_sha256=q.file_hash(path))
    save(root, state, manifest)
    return True


def run_job(root, state, manifest, job):
    if q.file_hash(root / 'manifest_initial.json') != state['manifest_file_sha256']:
        raise ValueError('initial manifest changed while worker active')
    record = state['jobs'][job['id']]
    if record.get('next_attempt_number', 1) > 1:
        job = record['job']  # Recovery retains the separate retry directory/receipt.
    if record['status'] in TERMINAL:
        return
    deps = [state['jobs'].get(d, {}).get('status') for d in job.get('deps', [])]
    if any(s not in {'completed', 'reused'} for s in deps):
        record.update(status='blocked', reason='failed or unavailable dependency')
        return
    number = record.get('next_attempt_number', record['attempts'][-1]['number'] if record['attempts'] else 1)
    active_path = root / 'attempts' / (job['id'] + f'.attempt{number}.json')
    existing = read(active_path)
    if existing and not existing.get('finished_at'):
        if not q.alive(existing.get('runner')):
            reap_owned_group(existing.get('command'))
            existing.update(finished_at=q.now(), exit_code=126, error='guardian ended without terminal receipt',
                            command_elapsed_seconds=max(0, time.time() - stamp_seconds(existing.get('command_started_at', existing['created_at']))))
            q.atomic_json(active_path, existing)
    fds = []
    while not existing:
        if control(root).get('stop'):
            record.update(status='not_run', reason='controlled stop before launch')
            return
        protect = manifest['collection_reserve_seconds'] if job['bundle'] != 'collection' else 0
        if time.time() + job['max_seconds'] + protect + 120 > stamp_seconds(state['deadline_at']):
            record.update(status='deferred_budget', reason='wall deadline admission refusal')
            return
        if control(root).get('paused'):
            state['worker_status'] = 'paused'; save(root, state, manifest); time.sleep(2); continue
        allowed, reason = q.host_available(manifest, job)
        if allowed and job['gpu']:
            for path in (manifest['gpu_lock'], manifest['gpu_uuid_lock']):
                fd = q.lock_fd(path)
                if fd is None:
                    allowed, reason = False, 'shared GPU lock held'
                    break
                fds.append(fd)
            if allowed:
                try:
                    allowed, reason = q.gpu_available(manifest, {**job, 'env': {'CUDA_VISIBLE_DEVICES': manifest['gpu_uuid']}})
                except Exception as e:
                    allowed, reason = False, str(e)
        if not allowed:
            for fd in fds: os.close(fd)
            fds = []
            record['reason'] = 'resource_wait: ' + reason
            state['worker_status'] = 'resource_wait'; save(root, state, manifest); time.sleep(5); continue
        break
    if not existing:
        attempt = {'number': number, 'created_at': q.now(), 'job': job, 'gpu': job['gpu'],
                   'gpu_uuid': manifest['gpu_uuid'], 'deadline_epoch': stamp_seconds(state['deadline_at']),
                   'log_path': str(root / 'logs' / (job['id'] + f'.attempt{number}.log')),
                   'frozen_inputs': manifest['frozen_inputs']}
        Path(attempt['log_path']).parent.mkdir(exist_ok=True)
        q.atomic_json(active_path, attempt)
        record.update(status='running', reason=None)
        record['attempts'].append(attempt)
        state.update(active_job=job['id'], worker_status='running')
        save(root, state, manifest)
        try:
            guardian = subprocess.Popen([sys.executable, str(ROOT / 'run_overnight.py'), '_attempt',
                                         '--run-dir', str(root), '--attempt', str(active_path),
                                         '--fds', ','.join(map(str, fds))], stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        start_new_session=True, pass_fds=tuple(fds))
            attempt['runner'] = q.process_identity(guardian.pid)
            # Guardian owns the attempt file from here; worker state records initial identity.
            record['attempts'][-1] = attempt
        finally:
            for fd in fds: os.close(fd)
    while True:
        a = read(active_path)
        record['attempts'][-1] = a
        if a.get('finished_at'):
            break
        if not q.alive(a.get('runner')) and time.time() - stamp_seconds(a['created_at']) > 20:
            reap_owned_group(a.get('command'))
            a.update(finished_at=q.now(), exit_code=126, error='guardian died',
                     command_elapsed_seconds=max(0, time.time() - stamp_seconds(a.get('command_started_at', a['created_at']))))
            q.atomic_json(active_path, a)
            continue
        save(root, state, manifest); time.sleep(2)
    record.update(status='completed' if a['exit_code'] == 0 else 'technical_failure', reason=a.get('error'))
    if a['exit_code'] != 0 and not a.get('error'):
        a['error'] = f'command exited {a["exit_code"]}; see {a["log_path"]}'
        record['reason'] = a['error']
        q.atomic_json(active_path,a)
    if job['gpu']:
        state['bundles'][job['bundle']]['reserved_seconds'] -= job['max_seconds']
    state['active_job'] = None
    if record['status'] == 'technical_failure':
        failures = read(root / 'failure_ledger.json', [])
        failures.append({'job': job['id'], 'attempt': number, 'at': q.now(), 'reason': a.get('error'),
                         'exit_code': a['exit_code'], 'elapsed_seconds': a.get('command_elapsed_seconds', 0),
                         'retry': 'not_attempted; no diagnosed transient condition'})
        q.atomic_json(root / 'failure_ledger.json', failures)
    decision(root, 'job_terminal', job=job['id'], status=record['status'])
    save(root, state, manifest)
    retry = job.get('retry', {})
    if number == 1 and a['exit_code'] in retry.get('exit_codes', []) and retry.get('diagnosis'):
        b = ledger(root, state, manifest)
        protect = manifest['collection_reserve_seconds'] if job['bundle'] != 'collection' else 0
        cost = job['max_seconds'] if job['gpu'] else 0
        if (cost + protect <= b['available_unreserved_seconds'] and
                time.time() + job['max_seconds'] + protect + 120 <= stamp_seconds(state['deadline_at'])):
            retried = json.loads(json.dumps(job))
            if '--run-dir' in retried['argv'] and job['role'] == 'train':
                pos = retried['argv'].index('--run-dir') + 1
                retried['argv'][pos] += '_retry1'
            if '--out' in retried['argv']:
                pos = retried['argv'].index('--out') + 1
                old = Path(retried['argv'][pos]); new = old.with_name(old.stem + '_retry1' + old.suffix)
                retried['argv'][pos] = str(new); retried['receipt'] = str(new)
                for out in retried['outputs']:
                    if out['path'] == str(old): out['path'] = str(new)
            child = root / 'child_manifests' / (job['id'] + '_retry1.json')
            q.atomic_json(child, {'job':retried, 'diagnosis':retry['diagnosis'], 'parent_attempt':str(active_path)})
            record.update(job=retried,status='pending',next_attempt_number=2)
            state['bundles'][job['bundle']]['reserved_seconds'] += cost
            decision(root,'diagnosed_technical_retry',job=job['id'],diagnosis=retry['diagnosis'],reserved_seconds=cost)
            save(root,state,manifest)
            run_job(root,state,manifest,retried)


def materialize(root, state, manifest, bundle):
    jobs = json.loads(json.dumps(bundle.get('jobs', [])))
    condition = bundle.get('condition')
    if condition in {'A0_selected_lr','A1_selected_lr'}:
        graph = condition[:2]
        candidates = [(5e-5, manifest['reuse_'+graph+'_seed42'])]
        for lr, jid in ((5e-4, graph+'_lr5e-4_s42'), (5e-3, graph+'_lr5e-3_s42')):
            if state['jobs'].get(jid, {}).get('status') != 'completed':
                raise ValueError(graph+' LR grid incomplete')
            candidates.append((lr, state['jobs'][jid]['job']['receipt']))
        lr, receipt = max(candidates, key=lambda x: (selected_score(x[1]), -x[0]))
        selection = {'graph': graph, 'lr': lr, 'receipt': receipt, 'tie_rule': 'smaller_lr', 'data_role': 'source_validation'}
        q.atomic_json(root / 'selections' / (graph+'_lr.json'), selection)
        decision(root, graph+'_lr_selected', **selection)
        if lr == 5e-5:
            state['bundles'][bundle['id']].update(status='reused', reason='original LR won; all verified old seeds reused')
            for j in jobs: state['jobs'][j['id']].update(status='reused', reason='verified historical original LR')
            return []
        for j in jobs:
            j['argv'][j['argv'].index('--lr') + 1] = str(lr)
            j['lr'] = lr
    elif condition == 'C3_promotion':
        pilot = [r for r in state['jobs'].values() if r['job'].get('bundle') == 'pilot']
        if len(pilot) != 12 or any(r['status'] != 'completed' for r in pilot):
            raise ValueError('all six pilot arms and both seeds must complete')
        receipts = {(r['job']['method'], r['job']['seed']): r['job']['receipt'] for r in pilot}
        gate = promotion_gate(receipts)
        gate['criteria']['focused_loss_label_alignment'] = read(root / 'checks/focused_cpu.json', {}).get('outcome') == 'passed'
        gate['criteria']['real_smoke_checks'] = read(root / 'preflight_checks.json', {}).get('passed') is True
        integrity = []
        for rec in pilot:
            payload = read(rec['job']['receipt'])
            finite = payload.get('gradient_checks', {}).get('updates_checked') == 1560
            for epoch in payload.get('epochs', []):
                for key, value in epoch.items():
                    if ('loss' in key or 'grad_norm' in key) and isinstance(value, (float, int)):
                        finite = finite and math.isfinite(value)
            integrity.append(finite)
        gate['criteria']['all_1560_updates_finite_loss_and_gradients'] = all(integrity)
        for seed in (42, 64):
            hashes = [read(receipts[(arm, seed)]).get('initial_base_sha256') for arm in ('C0','C1','C2','C3','C4','C5')]
            batch = [tuple(read(receipts[(arm, seed)]).get('first_minibatch_hashes', [])) for arm in ('C0','C1','C2','C3','C4','C5')]
            gate['criteria']['initialization_parity_s' + str(seed)] = bool(hashes[0] and len(set(hashes)) == 1)
            gate['criteria']['minibatch_parity_s' + str(seed)] = bool(batch[0] and len(set(batch)) == 1)
            logits = [read(receipts[(arm, seed)]).get('first_training_logits_sha256') for arm in ('C0','C1','C2','C3','C4','C5')]
            gate['criteria']['initial_logits_parity_s' + str(seed)] = bool(logits[0] and len(set(logits)) == 1)
        b = ledger(root,state,manifest)
        gate['criteria']['budget_integrity'] = b['incremental_spent_seconds'] <= manifest['incremental_cap_seconds']
        gate['passed'] = all(gate['criteria'].values())
        q.atomic_json(root / 'selections' / 'promotion.json', gate)
        decision(root, 'C3_promotion_gate', **gate)
        if not gate['passed']:
            state['bundles'][bundle['id']].update(status='failed_scientific_gate', reason='C3 did not meet all prospective criteria')
            for j in jobs: state['jobs'][j['id']].update(status='not_run', reason='failed prospective promotion gate')
            save(root,state,manifest)
            collect(root,'gate')
            return []
    elif condition == 'target_refresh':
        diag = read(root / 'diagnostics.json')
        rate = diag['train']['cached_live_augmented_disagreement_rate']
        if rate < .01:
            state['bundles'][bundle['id']].update(status='not_run', reason='TRAIN mismatch below prospective 1% trigger')
            for j in jobs: state['jobs'][j['id']].update(status='not_run', reason='TRAIN mismatch below trigger')
            return []
    elif condition == 'completed_terminal':
        receipts = [r['job']['receipt'] for r in state['jobs'].values()
                    if r['job'].get('role') == 'train' and r['status'] == 'completed']
        for j in jobs:
            if j['role'] == 'eval':
                j['argv'] += receipts
    return jobs


def collect(root, reason):
    script = root / 'collect_v7.py'
    if script.exists():
        with open(root / 'collector.log', 'ab') as log:
            result = subprocess.run([sys.executable, str(script), '--run-dir', str(root), '--reason', reason],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **q.THREAD_ENV, 'CUDA_VISIBLE_DEVICES': '', 'MPLBACKEND':'Agg'}, timeout=120)
        if result.returncode:
            decision(root, 'collector_failure', exit_code=result.returncode)


def worker(root, manifest):
    fd = q.lock_fd(root / 'worker.lock')
    if fd is None:
        raise RuntimeError('matching v7 worker already active')
    experiment_fd = q.lock_fd(manifest.get('experiment_lock', root / 'experiment.lock'))
    if experiment_fd is None:
        os.close(fd)
        raise RuntimeError('matching scientific experiment identity already active')
    state = initialize(root, manifest)
    if state.get('worker_status') in {'finished', 'stopped', 'budget_closed'}:
        raise RuntimeError('terminal worker; never silently reset an old run')
    state['worker'] = q.process_identity(os.getpid())
    state['gpu_uuid'] = manifest['gpu_uuid']
    state['session_id'] = os.getsid(0)
    state.setdefault('launched_at', q.now())
    state.setdefault('deadline_at', datetime.fromtimestamp(time.time() + manifest['wall_seconds'], timezone.utc).isoformat())
    signal.signal(signal.SIGTERM, lambda *_: q.atomic_json(root / 'control.json', {**control(root), 'stop': True}))
    signal.signal(signal.SIGINT, lambda *_: q.atomic_json(root / 'control.json', {**control(root), 'stop': True}))
    try:
        for bundle in manifest['bundles']:
            bs = state['bundles'][bundle['id']]
            if bs['status'] in TERMINAL:
                continue
            if control(root).get('stop') or time.time() >= stamp_seconds(state['deadline_at']):
                break
            try:
                jobs = materialize(root, state, manifest, bundle)
                if not jobs:
                    save(root, state, manifest); continue
                if bs['status'] != 'admitted' and not admission(root, state, manifest, bundle, jobs):
                    save(root, state, manifest); collect(root,'budget'); continue
                if bs['status'] == 'admitted':
                    jobs = read(bs['manifest'])['jobs']
                    if q.file_hash(bs['manifest']) != bs['manifest_sha256']:
                        raise ValueError('child manifest changed after submission')
                for job in jobs:
                    run_job(root, state, manifest, job)
                bs.update(status='completed' if all(state['jobs'][j['id']]['status'] in {'completed','reused'} for j in jobs)
                          else 'technical_failure', reserved_seconds=0)
            except Exception as error:
                bs.update(status='blocked', reason=f'{type(error).__name__}: {error}', reserved_seconds=0)
                for j in bundle.get('jobs', []):
                    if state['jobs'][j['id']]['status'] == 'pending':
                        state['jobs'][j['id']].update(status='blocked', reason=bs['reason'])
                decision(root, 'bundle_blocked', bundle=bundle['id'], reason=bs['reason'])
                save(root,state,manifest)
                collect(root,'gate')
            save(root, state, manifest)
        stopped = control(root).get('stop')
        state['worker_status'] = 'stopped' if stopped else 'budget_closed' if time.time() >= stamp_seconds(state['deadline_at']) else 'finished'
        for r in state['jobs'].values():
            if r['status'] == 'pending':
                r.update(status='not_run' if stopped else 'deferred_budget', reason='worker terminal before admission')
        for b in state['bundles'].values():
            if b['status'] == 'pending': b.update(status='not_run' if stopped else 'deferred_budget', reason='worker terminal before admission')
        state['finished_at'] = q.now(); save(root, state, manifest)
        collect(root, 'stop' if stopped else 'completion')
    finally:
        os.close(experiment_fd)
        os.close(fd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['start','run','status','pause','resume','stop','_attempt'])
    p.add_argument('--run-dir', default=str(ROOT)); p.add_argument('--attempt'); p.add_argument('--fds', default='')
    a = p.parse_args(); root = Path(a.run_dir).resolve()
    if a.command == '_attempt':
        attempt_runner(root, a.attempt, [int(i) for i in a.fds.split(',') if i]); return
    if a.command in {'pause','resume','stop'}:
        with q.edit_state(root / 'control.json') as c:
            if a.command == 'pause': c['paused'] = True
            elif a.command == 'resume': c['paused'] = False
            else: c['stop'] = True
        print(json.dumps(control(root))); return
    if a.command == 'status':
        state = read(root / 'state.json', {}); counts = {}
        for rec in state.get('jobs', {}).values(): counts[rec['status']] = counts.get(rec['status'], 0) + 1
        print(json.dumps({**{k:state.get(k) for k in ['worker_status','worker','session_id','heartbeat_at','active_job','deadline_at']},
                          'worker_alive':q.alive(state.get('worker')), 'control':control(root), 'counts':counts,
                          'gpu_uuid':read(root / 'manifest_initial.json', {}).get('gpu_uuid'),
                          'bundles':state.get('bundles'), 'budget':read(root / 'budget_ledger.json'),
                          'report':str(root / 'report_v7.md')}, indent=2)); return
    manifest = read(root / 'manifest_initial.json')
    if a.command == 'start':
        old = read(root / 'state.json', {})
        if q.alive(old.get('worker')):
            print(json.dumps({'attached_existing_worker':old['worker'], 'status':old.get('worker_status')})); return
        for path, expected in manifest['frozen_inputs'].items():
            if q.file_hash(path) != expected: raise ValueError('launch frozen hash mismatch: ' + path)
        initialize(root, manifest)
        with open(root / 'worker.log', 'ab', buffering=0) as log:
            proc = subprocess.Popen([sys.executable, str(ROOT / 'run_overnight.py'), 'run', '--run-dir', str(root)],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, **q.THREAD_ENV, 'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
        print(json.dumps({'worker_pid':proc.pid, 'run_dir':str(root)}))
    else:
        worker(root, manifest)


if __name__ == '__main__':
    main()
