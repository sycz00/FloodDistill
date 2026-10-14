"""Bounded V11 CPU adapter to the V10 guardian / audited_queue worker.

The imported audited queue owns locks, atomic JSON, PID/start/boot identity and
output validation. V11 adds finite-block CPU accounting and deadline admission.
No GPU command, old-worker mutation or scientific choice belongs in this file.
"""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / 'research_v9_recovery_20261013T075000Z'))
import audited_queue as q

TERMINAL = {'COMPLETED', 'TECHNICAL_FAILED', 'UNAVAILABLE', 'DEFERRED',
            'SOURCE_GATE_NOT_MET', 'NO_MATERIAL_SOURCE_DEFICIT'}
ACTIVE = {'STARTING', 'RUNNING', 'STOPPING'}
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
CHECK_SECONDS = 600
HEARTBEAT_SECONDS = 30
GRACE_SECONDS = 30


def read(path, default=None):
    try:
        return q.read_json(path)
    except FileNotFoundError:
        return default


def env(threads, additions=None):
    result = dict(os.environ, **(additions or {}))
    result.update({key: str(threads) for key in THREAD_KEYS})
    result.update(CUDA_VISIBLE_DEVICES='', V10_COMPUTE_DEVICE='cpu',
                  V11_COMPUTE_DEVICE='cpu', V11_THREADS=str(threads),
                  OMP_THREAD_LIMIT=str(threads), TORCH_NUM_THREADS=str(threads),
                  TORCH_NUM_INTEROP_THREADS='1', V11_DATALOADER_WORKERS='0',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                  NO_ALBUMENTATIONS_UPDATE='1', PYTHONDONTWRITEBYTECODE='1')
    return result


def freeze_check(root, expected_manifest=None):
    if expected_manifest and q.file_hash(root / 'manifest.json') != expected_manifest:
        raise ValueError('Immutable V11 manifest changed')
    frozen = read(root / 'code_freeze.json')
    if not frozen or not frozen.get('files'):
        raise ValueError('Missing code_freeze.json')
    for name, expected in frozen['files'].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if q.file_hash(path) != expected:
            raise ValueError('Frozen input modified: ' + str(path))


def validate_manifest(manifest):
    jobs = manifest['jobs']
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError('V11 requires a nonempty finite jobs dictionary')
    blocks = manifest['blocks']
    pending = set(jobs)
    cleared = set()
    while pending:
        ready = {name for name in pending if set(jobs[name].get('deps', [])) <= cleared}
        if not ready:
            raise ValueError('Invalid or cyclic DAG dependency')
        cleared |= ready
        pending -= ready
    for name, job in jobs.items():
        if not name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ' for x in name) or ' ' in name:
            raise ValueError('Unsafe job name')
        threads, seconds = job['threads'], job['max_seconds']
        if not isinstance(threads, int) or not 1 <= threads <= 8 or not math.isfinite(seconds) or seconds <= GRACE_SECONDS:
            raise ValueError('Job needs 1..8 threads and a finite >30-second ceiling')
        if job.get('gpu') or job.get('device', 'cpu') != 'cpu':
            raise ValueError('V11 adapter does not admit GPU work')
        if not job.get('argv') or not Path(job['argv'][0]).is_absolute():
            raise ValueError('Absolute executable required')
        if not job.get('outputs') or any(not Path(x['path']).is_absolute() for x in job['outputs']):
            raise ValueError('Absolute expected output paths required')
        if job['block'] not in blocks or name not in blocks[job['block']]['job_ids']:
            raise ValueError('Every job must belong to its declared whole block')
        retry = job.get('retry', {})
        if retry and (not retry.get('reason') or not retry.get('exit_codes')):
            raise ValueError('Retry requires diagnosed transient reason and exit codes')
    for name, block in blocks.items():
        reserve = block['reserve_core_hours'] * 3600
        if not math.isfinite(reserve) or reserve < 0:
            raise ValueError('Invalid block reserve')
        required = sum(jobs[n]['threads'] * jobs[n]['max_seconds'] for n in block['job_ids'])
        if required > reserve + 1e-6:
            raise ValueError('Whole block command ceilings exceed reservation: ' + name)
    if manifest.get('cap_core_hours', 64) > 64:
        raise ValueError('V11 cap may not exceed 64 core-hours')


def fresh_state(manifest, at):
    initial = manifest.get('initial_charges', [])
    charges = {row['id']: dict(row) for row in initial}
    state = dict(schema_version=11, launch=at, prerequisite_deadline=at + 8 * 3600,
                 absolute_deadline=at + 16 * 3600, first_compute_at=None,
                 execution_deadline=None, jobs={n: {'status': 'PENDING', 'attempts': 0} for n in manifest['jobs']},
                 blocks={}, active={}, charges=charges, finished=False,
                 phase='STARTING', next_check_at=at, last_check_at=None)
    for name, block in manifest['blocks'].items():
        if block.get('protected'):
            state['blocks'][name] = dict(remaining_core_seconds=block['reserve_core_hours'] * 3600,
                                         admitted_at=at, protected=True)
    if account(state, at)['total_committed_core_seconds'] > manifest.get('cap_core_hours', 64) * 3600:
        raise ValueError('Initial charges and protected reservations exceed V11 cap')
    return state


def cutoff(state):
    if state['first_compute_at'] is None:
        return min(state['prerequisite_deadline'], state['absolute_deadline'] - 1800)
    return min(state['execution_deadline'], state['absolute_deadline']) - 1800


def account(state, at):
    charged = sum(row['reserved_core_seconds'] for row in state['charges'].values())
    active = list(state['active'].values())
    if state.get('collection_active'):
        active.append(state['collection_active'])
    accrued = sum(max(0, at - row['started_at']) * row['threads'] for row in active)
    active_remaining = sum(max(0, row['max_seconds'] - (at - row['started_at'])) * row['threads']
                           for row in active)
    planned = sum(row['remaining_core_seconds'] for row in state['blocks'].values())
    return dict(completed_charged_core_seconds=charged, active_accrued_core_seconds=accrued,
                active_reserved_remaining_core_seconds=active_remaining,
                planned_reserved_remaining_core_seconds=planned,
                total_committed_core_seconds=charged + accrued + active_remaining + planned,
                measured_process_cpu_seconds=sum(row.get('measured_process_cpu_seconds', 0) or 0 for row in state['charges'].values()),
                hardware_occupied_threads=sum(row['threads'] for row in active))


def reserve_block(manifest, state, name, at):
    if name in state['blocks']:
        return True
    block = manifest['blocks'][name]
    reserve = block['reserve_core_hours'] * 3600
    if account(state, at)['total_committed_core_seconds'] + reserve > manifest.get('cap_core_hours', 64) * 3600 + 1e-6:
        return False
    state['blocks'][name] = dict(remaining_core_seconds=reserve, admitted_at=at,
                                 protected=bool(block.get('protected')))
    return True


def allocate(manifest, state, name, at):
    """One locked transaction converts planned cost to active cost exactly once."""
    if name in state['active'] or state['jobs'][name]['status'] != 'PENDING':
        return None
    job = manifest['jobs'][name]
    if not reserve_block(manifest, state, job['block'], at):
        return None
    needed = job['threads'] * job['max_seconds']
    block = state['blocks'][job['block']]
    if block['remaining_core_seconds'] + 1e-6 < needed:
        return None
    token = uuid.uuid4().hex
    block['remaining_core_seconds'] -= needed
    active = dict(token=token, threads=job['threads'], max_seconds=job['max_seconds'],
                  started_at=at, block=job['block'])
    state['active'][name] = active
    row = state['jobs'][name]
    row.update(status='STARTING', token=token, started_at=at, attempts=row['attempts'] + 1)
    if job.get('compute_heavy', True) and state['first_compute_at'] is None:
        state['first_compute_at'] = at
        state['execution_deadline'] = min(at + 8 * 3600, state['absolute_deadline'])
    return token


def charge(state, name, receipt, at):
    active = state['active'].pop(name, None)
    if active is None:
        return
    seconds = max(0, receipt.get('seconds', at - active['started_at']))
    state['charges'][active['token']] = dict(job=name, seconds=seconds,
        threads=active['threads'], reserved_core_seconds=seconds * active['threads'],
        measured_process_cpu_seconds=receipt.get('measured_process_cpu_seconds'),
        outcome=receipt['outcome'], receipt=receipt.get('receipt'), reason=receipt.get('reason'))


def release_finished_blocks(manifest, state):
    for name, row in state['blocks'].items():
        block = manifest['blocks'][name]
        if not block.get('protected') and all(state['jobs'][n]['status'] in TERMINAL for n in block['job_ids']):
            row['remaining_core_seconds'] = 0


def external_allocation(manifest):
    """Old workers do not share this ledger: reserve their maximum potential use."""
    reserved, evidence = 0, []
    for spec in manifest.get('external_workers', []):
        data = read(Path(spec['state_path']), {})
        worker_alive = q.alive(data.get('worker'))
        rows = list(data.get('jobs', {}).values())
        statuses = [row.get('status') if isinstance(row, dict) else row for row in rows]
        terminal = bool(data.get('finished') and not worker_alive and not data.get('active') and
                        all(status in TERMINAL | {'succeeded', 'failed', 'blocked'} for status in statuses))
        # A dead controller with unfinished jobs may be resumed at any moment.
        amount = 0 if terminal else int(spec['max_threads'])
        reserved += amount
        evidence.append(dict(name=spec['name'], threads_reserved=amount,
                             terminal_receipt=terminal, worker_alive=worker_alive,
                             state_path=spec['state_path']))
    return reserved, evidence


def telemetry(manifest):
    try:
        available = len(os.sched_getaffinity(0))
        mem = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            mem[key] = int(value.split()[0]) * 1024
        load = os.getloadavg()[0]
        other, evidence = external_allocation(manifest)
        return dict(known=True, host_threads=available,
                    safe_threads=min(int(manifest.get('safe_host_threads', 40)), max(1, available - 8)),
                    free_ram_gib=mem['MemAvailable'] / 2**30, load1=load,
                    external_reserved_threads=other, external_workers=evidence)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(known=False, reason=repr(error))


def availability(row, job, active_threads):
    if not row.get('known'):
        return False, 'Unknown CPU/RAM/worker telemetry'
    threads = job['threads']
    if active_threads + threads > 8:
        return False, 'V11 thread cap'
    if row['external_reserved_threads'] + active_threads + threads > row['safe_threads']:
        return False, 'Configured maximum old-worker allocations leave insufficient CPU capacity'
    if row['load1'] + threads > row['safe_threads']:
        return False, 'CPU load leaves insufficient conservative headroom'
    if row['free_ram_gib'] < job.get('ram_gib', 4) + 8:
        return False, 'Free RAM below job reservation plus 8 GiB margin'
    return True, 'Conservative maximum old-worker allocations and RAM admit job'


def dependencies(root, manifest, job, state):
    rows = [state['jobs'][name]['status'] for name in job.get('deps', [])]
    if any(status not in TERMINAL for status in rows):
        return dict(status='WAITING_DEPENDENCY', reason='Local DAG predecessors incomplete')
    if not job.get('soft_dependencies') and any(status != 'COMPLETED' for status in rows):
        return dict(status='UNAVAILABLE', reason='Required predecessor did not complete')
    if manifest.get('dependency_module'):
        try:
            sys.path.insert(0, str(root / 'code'))
            return importlib.import_module(manifest['dependency_module']).check(root, job, state)
        except Exception as error:
            return dict(status='WAITING_DEPENDENCY', reason='Dependency verification unavailable: ' + repr(error))
    return dict(status='READY', reason='No external prerequisites')


def verify_inputs(result):
    # Only the dependency checker may authorize completed artifact paths here.
    for item in result.get('inputs', []):
        if q.file_hash(item['path']) != item['sha256']:
            return False
    return True


def owned_stop(identity, sig=signal.SIGTERM):
    if q.alive(identity):
        try:
            if os.getpgid(identity['pid']) == identity['pid']:
                os.killpg(identity['pid'], sig)
            else:
                os.kill(identity['pid'], sig)
        except ProcessLookupError:
            pass


def receipt_path(root, name, token):
    return root / 'receipts' / (name + '.' + token + '.json')


def guard(root, name, token):
    """V10 guardian pattern: independently enforce ceiling if controller dies."""
    manifest = read(root / 'manifest.json')
    state = read(root / 'state.json')
    row = state['jobs'][name]
    if row.get('token') != token or row['status'] not in ACTIVE:
        return
    fd = q.lock_fd(root / 'running' / (name + '.' + token + '.lock'))
    if fd is None:
        return
    freeze_check(root, state['manifest_sha256'])
    job = manifest['jobs'][name]
    started = row['started_at']
    reason = None
    child = None
    def requested(*_):
        nonlocal reason
        reason = 'Owned V11 stop requested'
    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    guardian = q.process_identity(os.getpid())
    running = dict(token=token, guardian=guardian, child=None, started_at=started)
    q.atomic_json(root / 'running' / (name + '.json'), running)
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    before_self = resource.getrusage(resource.RUSAGE_SELF)
    result = None
    try:
        os.sched_setaffinity(0, set(row['cpus']))
        with open(root / 'logs' / (name + '.' + token + '.log'), 'a') as output:
            child = subprocess.Popen(job['argv'], cwd=manifest['cwd'],
                env=env(job['threads'], {**manifest.get('env', {}), **job.get('env', {})}),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            running['child'] = q.process_identity(child.pid)
            q.atomic_json(root / 'running' / (name + '.json'), running)
            limit = min(started + job['max_seconds'], cutoff(state))
            while child.poll() is None:
                control = read(root / 'control.json', {})
                if control.get('stop'):
                    reason = 'Owned V11 stop requested'
                if reason or time.time() >= limit - GRACE_SECONDS:
                    reason = reason or 'Per-job/deadline ceiling; protected collection margin'
                    owned_stop(running['child'])
                    try:
                        child.wait(timeout=GRACE_SECONDS - 2)
                    except subprocess.TimeoutExpired:
                        owned_stop(running['child'], signal.SIGKILL)
                        child.wait(timeout=2)
                    break
                time.sleep(1)
        result = read(root / 'stage_results' / (name + '.json'), {})
        valid, evidence = q.validate_outputs(job)
        raw = str(result.get('outcome', '')).upper()
        status = raw if raw in TERMINAL else 'COMPLETED' if raw in {'COMPLETED', 'SUCCESS', 'SUCCEEDED'} else 'TECHNICAL_FAILED'
        if child.returncode or not valid:
            status = 'TECHNICAL_FAILED'
        if reason:
            status = 'DEFERRED'
        receipt = dict(outcome=status, result_outcome=result.get('outcome'),
                       returncode=child.returncode, outputs=evidence, reason=reason,
                       result_sha256=q.file_hash(root / 'stage_results' / (name + '.json')) if result else None)
    except BaseException as error:
        if child and child.poll() is None:
            owned_stop(running['child'])
            try:
                child.wait(timeout=GRACE_SECONDS - 2)
            except subprocess.TimeoutExpired:
                owned_stop(running['child'], signal.SIGKILL)
                child.wait(timeout=2)
        receipt = dict(outcome='TECHNICAL_FAILED', reason=repr(error), returncode=child.returncode if child else None)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    after_self = resource.getrusage(resource.RUSAGE_SELF)
    cpu = after.ru_utime + after.ru_stime - usage_start.ru_utime - usage_start.ru_stime
    cpu += after_self.ru_utime + after_self.ru_stime - before_self.ru_utime - before_self.ru_stime
    receipt.update(token=token, job=name, started_at=started, ended_at=time.time(),
                   seconds=time.time() - started, threads=job['threads'],
                   measured_process_cpu_seconds=cpu, guardian=guardian,
                   child=running['child'], manifest_sha256=state['manifest_sha256'])
    q.atomic_json(receipt_path(root, name, token), receipt)
    os.close(fd)


def reconcile(root, manifest, state, at):
    for name, active in list(state['active'].items()):
        row = state['jobs'][name]
        path = receipt_path(root, name, active['token'])
        receipt = read(path)
        running = read(root / 'running' / (name + '.json'), {})
        if running.get('token') != active['token']:
            running = {}
        identity = running.get('guardian') or row.get('guardian')
        if q.alive(identity):
            row.update(status='RUNNING', guardian=identity)
            continue
        if receipt and receipt.get('token') == active['token']:
            receipt['receipt'] = str(path)
            charge(state, name, receipt, at)
            row.update(status=receipt['outcome'], receipt=str(path), reason=receipt.get('reason'))
            retry = manifest['jobs'][name].get('retry', {})
            if (row['status'] == 'TECHNICAL_FAILED' and row['attempts'] == 1 and
                    receipt.get('returncode') in retry.get('exit_codes', [])):
                needed = active['threads'] * active['max_seconds']
                if account(state, at)['total_committed_core_seconds'] + needed <= manifest.get('cap_core_hours', 64) * 3600:
                    state['blocks'][active['block']]['remaining_core_seconds'] += needed
                    row.update(status='PENDING', retry_reason=retry['reason'])
            state['next_check_at'] = at
        elif at - active['started_at'] >= 60:
            # Never relaunch a lost attempt. Stop only a start/boot-verified owned
            # child; conservatively charge elapsed time and retain its artifacts.
            if q.alive(running.get('child')):
                row.setdefault('orphan_stop_started', at)
                owned_stop(running['child'], signal.SIGKILL if at - row['orphan_stop_started'] >= GRACE_SECONDS - 2 else signal.SIGTERM)
                row.update(status='STOPPING', reason='Lost guardian: waiting for verified owned child to exit')
                continue
            receipt = dict(token=active['token'], job=name, outcome='TECHNICAL_FAILED',
                reason='Guardian lost; owned child signalled; no duplicate restart',
                seconds=max(at - active['started_at'], active['max_seconds']),
                measured_process_cpu_seconds=None, receipt=str(path))
            q.atomic_json(path, receipt)
            charge(state, name, receipt, at)
            row.update(status='TECHNICAL_FAILED', reason=receipt['reason'], receipt=str(path))
            state['next_check_at'] = at
    release_finished_blocks(manifest, state)


def publish(root, manifest, state, at):
    state['heartbeat'] = at
    q.atomic_json(root / 'state.json', state)
    ledger = account(state, at)
    ledger.update(at=at, cap_core_hours=manifest.get('cap_core_hours', 64),
                  charges=state['charges'], active=state['active'], blocks=state['blocks'],
                  accounting_rule='completed elapsed x allocation + active accrued + active unused ceiling + planned block remainder')
    q.atomic_json(root / 'resource_ledger.json', ledger)
    q.atomic_json(root / 'scheduler_status.json', dict(
        worker=state.get('worker'), phase=state['phase'], heartbeat=at,
        launch=state['launch'], first_compute_at=state['first_compute_at'],
        prerequisite_deadline=state['prerequisite_deadline'], execution_deadline=state['execution_deadline'],
        absolute_deadline=state['absolute_deadline'], collection_cutoff=cutoff(state),
        last_check_at=state.get('last_check_at'), next_check_at=state.get('next_check_at'),
        reason=state.get('reason'), finished=state['finished'],
        counts={label: sum(row['status'] == label for row in state['jobs'].values())
                for label in sorted({row['status'] for row in state['jobs'].values()})},
        running=list(state['active']), resource=ledger))


def collect(root, manifest, state, reason):
    """Collection is restartable; only mark finished after its receipt is durable."""
    state.update(phase='COLLECTING', reason=reason)
    for row in state['jobs'].values():
        if row['status'] not in TERMINAL:
            row.update(status='DEFERRED', reason=reason)
    for name, block in state['blocks'].items():
        if name != manifest.get('collection_block', 'collection'):
            block['remaining_core_seconds'] = 0
    started = time.time()
    collection_key = manifest.get('collection_block', 'collection')
    available = state['blocks'].get(collection_key, {}).get('remaining_core_seconds', 0)
    threads = int(manifest.get('collection_threads', 1))
    max_seconds = min(1500, available / threads, max(0, state['absolute_deadline'] - started - 10))
    if max_seconds > GRACE_SECONDS and manifest.get('collect_argv'):
        state['blocks'][collection_key]['remaining_core_seconds'] -= max_seconds * threads
        state['collection_active'] = dict(started_at=started, threads=threads,
                                         max_seconds=max_seconds, identity=None)
    publish(root, manifest, state, started)
    status, returncode = 'UNAVAILABLE', None
    child = None
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    if manifest.get('collect_argv') and max_seconds > GRACE_SECONDS:
        with open(root / 'logs' / 'final_collection.log', 'a') as output:
            child = subprocess.Popen(manifest['collect_argv'], cwd=manifest['cwd'],
                env=env(threads, manifest.get('env')), stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            identity = q.process_identity(child.pid)
            state['collection_identity'] = identity
            state['collection_active']['identity'] = identity
            publish(root, manifest, state, time.time())
            while child.poll() is None and time.time() - started < max_seconds - GRACE_SECONDS:
                publish(root, manifest, state, time.time())
                time.sleep(1)
            if child.poll() is None:
                owned_stop(identity)
                try:
                    child.wait(timeout=GRACE_SECONDS - 2)
                except subprocess.TimeoutExpired:
                    owned_stop(identity, signal.SIGKILL)
                    child.wait(timeout=2)
            returncode = child.returncode
            status = 'COMPLETED' if returncode == 0 else 'TECHNICAL_FAILED'
    elapsed = time.time() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    state.pop('collection_active', None)
    state['charges']['collection-' + str(started)] = dict(job='collection', seconds=elapsed,
        threads=threads, reserved_core_seconds=elapsed * threads, outcome=status,
        measured_process_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime)
    if collection_key in state['blocks']:
        state['blocks'][collection_key]['remaining_core_seconds'] = 0
    state.update(phase='FINISHED', finished=True, finished_at=time.time(),
                 collection_returncode=returncode, collection_status=status)
    publish(root, manifest, state, time.time())


def tick(root, manifest, state, at, launch=True, telemetry_fn=telemetry, dependency_fn=dependencies):
    """One V10-style reconciliation/admission pass; injectable clock inputs for tests."""
    reconcile(root, manifest, state, at)
    control = read(root / 'control.json', {})
    exhausted = account(state, at)['total_committed_core_seconds'] > manifest.get('cap_core_hours', 64) * 3600 + 1e-6
    stopping = state.get('stop_requested') or control.get('stop') or at >= cutoff(state) or exhausted
    if stopping:
        state.update(phase='STOPPING', reason='Budget exhaustion' if exhausted else 'Deadline collection cutoff' if at >= cutoff(state) else 'User stop')
        for name in state['active']:
            owned_stop(state['jobs'][name].get('guardian'))
        return 'COLLECT' if not state['active'] else None
    if all(row['status'] in TERMINAL for row in state['jobs'].values()):
        state['reason'] = 'Bounded queue exhausted'
        return 'COLLECT'
    if control.get('pause'):
        state['phase'] = 'PAUSED'
        return None
    if at < state['next_check_at']:
        if state['active']:
            state['phase'] = 'RUNNING'
        return None
    state.update(last_check_at=at, next_check_at=at + CHECK_SECONDS)
    telemetry_before = telemetry_fn(manifest)
    waits = []
    for name, job in manifest['jobs'].items():
        row = state['jobs'][name]
        if row['status'] != 'PENDING':
            continue
        dependency = dependency_fn(root, manifest, job, state)
        status = dependency.get('status', 'WAITING_DEPENDENCY')
        if status in TERMINAL:
            row.update(status=status, reason=dependency.get('reason'))
            continue
        if status != 'READY':
            row['waiting_reason'] = dependency.get('reason') or 'Dependency checker has not admitted this job'
            waits.append(('WAITING_DEPENDENCY', row['waiting_reason']))
            continue
        active_threads = account(state, at)['hardware_occupied_threads']
        ok, reason = availability(telemetry_before, job, active_threads)
        if not ok:
            row['waiting_reason'] = reason
            waits.append(('WAITING_RESOURCES', reason))
            continue
        block = manifest['blocks'][job['block']]
        # Serial worker protects complete block wall time and any remaining final
        # evaluation commands. Internal per-step decisions may only defer work.
        block_seconds = sum(manifest['jobs'][n]['max_seconds'] for n in block['job_ids']
                            if state['jobs'][n]['status'] == 'PENDING')
        protected_seconds = sum(manifest['jobs'][n]['max_seconds'] for b in manifest['blocks'].values()
                                if b.get('protected') and b is not block for n in b['job_ids']
                                if state['jobs'][n]['status'] == 'PENDING')
        compute_cutoff = min(at + 8 * 3600, state['absolute_deadline']) - 1800 if state['first_compute_at'] is None else cutoff(state)
        if job.get('compute_heavy', True) and at + block_seconds + protected_seconds > compute_cutoff:
            for n in block['job_ids']:
                if state['jobs'][n]['status'] == 'PENDING':
                    state['jobs'][n].update(status='DEFERRED', reason='Complete block and protected evaluation cannot fit fixed window')
            continue
        # Atomic CPU admission uses the existing q.lock_fd primitive. Historical
        # workers do not cooperate; their configured maxima are re-reserved above.
        fd = q.lock_fd(root / 'cpu_admission.lock')
        if fd is None:
            waits.append(('WAITING_RESOURCES', 'CPU admission lock busy'))
            continue
        try:
            fresh = telemetry_fn(manifest)
            ok, reason = availability(fresh, job, active_threads)
            checked = dependency_fn(root, manifest, job, state)
            if not ok or checked.get('status') != 'READY' or not verify_inputs(checked):
                waits.append(('WAITING_RESOURCES' if not ok else 'WAITING_DEPENDENCY', reason if not ok else 'Dependency changed at admission recheck'))
                continue
            if state['active']:
                # One job avoids oversubscribed 8-thread libraries or heavy IO.
                waits.append(('WAITING_RESOURCES', 'An owned V11 job holds the CPU slot'))
                continue
            if launch:
                freeze_check(root, state['manifest_sha256'])
            token = allocate(manifest, state, name, at)
            if token is None:
                row.update(status='DEFERRED', reason='Complete block exceeds cumulative 64-core-hour allowance')
                continue
            cpus = sorted(os.sched_getaffinity(0))[-job['threads']:]
            row.update(cpus=cpus, input_receipts=checked.get('inputs', []))
            row.pop('waiting_reason', None)
            q.atomic_json(root / 'state.json', state)
            if launch:
                with open(root / 'logs' / 'guardians.log', 'a') as output:
                    child = subprocess.Popen([manifest['python'], str(Path(__file__).resolve()), 'guard',
                            '--root', str(root), '--job', name, '--token', token],
                            cwd=manifest['cwd'], env=env(1), stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                row['guardian'] = q.process_identity(child.pid)
            state['next_check_at'] = at + CHECK_SECONDS
            break
        finally:
            os.close(fd)
    state['phase'] = 'RUNNING' if state['active'] else waits[0][0] if waits else 'WAITING_DEPENDENCY'
    state['reason'] = None if state['active'] else '; '.join(dict.fromkeys(reason for _, reason in waits))
    with open(root / 'availability.jsonl', 'a') as output:
        output.write(json.dumps(dict(at=at, next_check_at=state['next_check_at'], telemetry=telemetry_before,
                                     phase=state['phase'], reason=state['reason'])) + '\n')
    release_finished_blocks(manifest, state)
    return None


def run(root):
    fd = q.lock_fd(root / 'worker.lock')
    if fd is None:
        raise RuntimeError('An owned V11 worker is already active')
    manifest = read(root / 'manifest.json')
    validate_manifest(manifest)
    for name in ('logs', 'running', 'receipts'):
        (root / name).mkdir(exist_ok=True)
    state = read(root / 'state.json')
    if state and state.get('finished'):
        return
    freeze_check(root, state.get('manifest_sha256') if state else None)
    if state is None:
        state = fresh_state(manifest, time.time())
        state['manifest_sha256'] = q.file_hash(root / 'manifest.json')
    state['worker'] = q.process_identity(os.getpid())
    # Reattach to an independently surviving collector instead of duplicating it.
    # Its exit code is unavailable to a new parent; retain that limitation.
    if state.get('collection_active'):
        collection = state['collection_active']
        while q.alive(collection.get('identity')):
            limit = min(collection['started_at'] + collection['max_seconds'], state['absolute_deadline'] - 10)
            if time.time() >= limit - GRACE_SECONDS:
                owned_stop(collection['identity'], signal.SIGKILL if time.time() >= limit - 2 else signal.SIGTERM)
            publish(root, manifest, state, time.time())
            time.sleep(2)
        elapsed = max(0, time.time() - collection['started_at'])
        state['charges']['collection-recovered-' + str(collection['started_at'])] = dict(
            job='collection', seconds=elapsed, threads=collection['threads'],
            reserved_core_seconds=elapsed * collection['threads'], outcome='UNKNOWN_AFTER_RESTART',
            measured_process_cpu_seconds=None)
        state.pop('collection_active')
        for block in state['blocks'].values():
            block['remaining_core_seconds'] = 0
        state.update(phase='FINISHED', finished=True, finished_at=time.time(),
                     collection_status='UNKNOWN_AFTER_RESTART', collection_returncode=None)
        publish(root, manifest, state, time.time())
        return
    state['next_check_at'] = time.time()
    def stop_requested(*_):
        state['stop_requested'] = True
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    publish(root, manifest, state, time.time())
    while True:
        result = tick(root, manifest, state, time.time())
        publish(root, manifest, state, time.time())
        if result == 'COLLECT':
            collect(root, manifest, state, state.get('reason', 'Queue exhausted'))
            return
        # Five-second controller polling; heavyweight availability checks remain
        # 600 seconds apart. Guardians independently enforce owned-job ceilings.
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'guard', 'status', 'pause', 'resume', 'stop'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--job')
    parser.add_argument('--token')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'run':
        run(root)
    elif args.action == 'guard':
        guard(root, args.job, args.token)
    elif args.action == 'status':
        state = read(root / 'state.json', {})
        print(json.dumps(dict(read(root / 'scheduler_status.json', state),
                              worker_alive=q.alive(state.get('worker'))), indent=2))
    else:
        control = read(root / 'control.json', {})
        if args.action == 'stop':
            control['stop'] = True
        elif args.action == 'pause':
            control['pause'] = True
        else:
            control['pause'] = False
        control['updated_at'] = time.time()
        q.atomic_json(root / 'control.json', control)


if __name__ == '__main__':
    main()
