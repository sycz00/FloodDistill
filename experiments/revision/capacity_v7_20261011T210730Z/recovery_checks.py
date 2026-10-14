#!/usr/bin/env python3
"""Focused CPU-only process recovery checks, isolated in temporary run directories."""
import contextlib
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import run_overnight as r


def wait_until(condition, timeout=25):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = condition()
        if value:
            return value
        time.sleep(.1)
    raise AssertionError('Timed out waiting for recovery fixture')


def manifest(root, job):
    return dict(bundles=[{'id': 'recovery', 'jobs': [job]}], historical_gpu_seconds=0,
                incremental_cap_seconds=100, preflight_gpu_seconds=0,
                collection_reserve_seconds=0, retry_reserve_seconds=0, wall_seconds=900,
                frozen_inputs={str(r.ROOT/'run_overnight.py'): r.q.file_hash(r.ROOT/'run_overnight.py')},
                gpu_uuid='CPU_RECOVERY_NO_GPU', gpu_lock=str(root/'gpu.lock'),
                gpu_uuid_lock=str(root/'gpu_uuid.lock'), experiment_lock=str(root/'experiment.lock'))


def start(root):
    launched = subprocess.run([sys.executable, str(r.ROOT/'run_overnight.py'), 'start', '--run-dir', str(root)],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=10)
    return json.loads(launched.stdout)


def terminate_identity(identity):
    if r.q.alive(identity):
        os.kill(identity['pid'], signal.SIGKILL)


def guardian_recovery():
    root = Path(tempfile.mkdtemp(prefix='flooddistill_v7_guardian_recovery_', dir='/tmp'))
    child_file, count_file, out = root/'child.json', root/'launch_count.txt', root/'terminal.json'
    code = ('import subprocess,time,json,pathlib; '
            'c=pathlib.Path(' + repr(str(count_file)) + '); '
            'c.write_text(str(int(c.read_text())+1) if c.exists() else "1"); '
            'p=subprocess.Popen([' + repr(sys.executable) + ',"-c","import time;time.sleep(90)"]); '
            'pathlib.Path(' + repr(str(child_file)) + ').write_text(json.dumps({"pid":p.pid})); '
            'time.sleep(90); pathlib.Path(' + repr(str(out)) + ').write_text("{}")')
    job = dict(id='owned', bundle='recovery', role='mock', gpu=False, cwd=str(root),
               argv=[sys.executable, '-c', code], max_seconds=120, deps=[], receipt=str(out),
               outputs=[dict(path=str(out), type='json')])
    m = manifest(root, job)
    r.q.atomic_json(root/'manifest_initial.json', m)
    launched = start(root)
    worker = guardian = command = child = None
    try:
        wait_until(lambda: child_file.exists())
        def identities_ready():
            state = r.read(root/'state.json', {})
            attempts = state.get('jobs', {}).get('owned', {}).get('attempts', [])
            if not attempts:
                return None
            a = r.read(root/'attempts/owned.attempt1.json', {})
            return (state, a) if a.get('command') and a.get('runner') else None
        state, attempt = wait_until(identities_ready)
        worker, guardian, command = state['worker'], attempt['runner'], attempt['command']
        child = r.q.process_identity(r.read(child_file)['pid'])
        assert all(r.q.alive(i) for i in (worker, guardian, command, child))
        # All four identities were created by this fixture. Kill only its worker
        # and guardian; leave the independent command group alive for recovery.
        terminate_identity(guardian)
        terminate_identity(worker)
        wait_until(lambda: not r.q.alive(worker) and not r.q.alive(guardian))
        assert r.q.alive(command) and r.q.alive(child)
        initial_digest = state['manifest_sha256']
        relaunched = start(root)
        state = wait_until(lambda: (lambda s: s if s.get('worker_status') == 'finished' else None)(
            r.read(root/'state.json', {})))
        recovered = r.read(root/'attempts/owned.attempt1.json')
        assert not r.q.alive(command) and not r.q.alive(child)
        assert recovered['exit_code'] == 126 and recovered['finished_at']
        assert recovered['command_elapsed_seconds'] > 0
        assert state['jobs']['owned']['status'] == 'technical_failure'
        assert len(state['jobs']['owned']['attempts']) == 1
        assert count_file.read_text() == '1', 'Recovery must not launch a duplicate command'
        assert state['manifest_sha256'] == initial_digest == r.q.digest(m)
        assert state['jobs']['owned']['attempts'][0]['command_elapsed_seconds'] == recovered['command_elapsed_seconds']
        assert r.read(root/'budget_ledger.json')['incremental_spent_seconds'] == 0
        return dict(passed=True, root=str(root), initial_worker_pid=launched['worker_pid'],
                    recovered_worker_pid=relaunched['worker_pid'], command_pid=command['pid'],
                    child_pid=child['pid'], launch_count=1, owned_group_reaped=True,
                    receipt_path=str(root/'attempts/owned.attempt1.json'),
                    recorded_elapsed_seconds=recovered['command_elapsed_seconds'],
                    terminal_state=state['jobs']['owned']['status'], normalized_manifest_preserved=True)
    finally:
        for identity in (guardian, worker):
            terminate_identity(identity)
        if command:
            with contextlib.suppress(Exception):
                r.reap_owned_group(command, grace=.1)


def persisted_retry_recovery():
    root = Path(tempfile.mkdtemp(prefix='flooddistill_v7_retry_recovery_', dir='/tmp'))
    old_dir, fresh_dir = root/'original_run', root/'original_run_retry1'
    old_dir.mkdir()
    (old_dir/'existing_scientific_attempt.txt').write_text('preserve')
    old_out, fresh_out = root/'receipt.json', root/'receipt_retry1.json'
    fixture = root/'retry_fixture.py'
    fixture.write_text('import argparse,json,pathlib\n'
        'p=argparse.ArgumentParser();p.add_argument("--run-dir");p.add_argument("--out");a=p.parse_args()\n'
        'd=pathlib.Path(a.run_dir);assert d.name.endswith("_retry1") and not d.exists(), "original argv launched"\n'
        'd.mkdir();(d/"launch.json").write_text(json.dumps(vars(a)))\n'
        'pathlib.Path(a.out).write_text(json.dumps({"status":"complete","run_dir":a.run_dir}))\n')
    original = dict(id='retry_train', bundle='recovery', role='train', gpu=False, cwd=str(root),
                    argv=[sys.executable, str(fixture), '--run-dir', str(old_dir), '--out', str(old_out)],
                    max_seconds=10, deps=[], receipt=str(old_out),
                    outputs=[dict(path=str(old_out), type='json', equals={'status':'complete'})])
    retried = json.loads(json.dumps(original))
    retried['argv'][retried['argv'].index('--run-dir')+1] = str(fresh_dir)
    retried['argv'][retried['argv'].index('--out')+1] = str(fresh_out)
    retried.update(receipt=str(fresh_out))
    retried['outputs'][0]['path'] = str(fresh_out)
    m = manifest(root, original)
    r.q.atomic_json(root/'manifest_initial.json', m)
    for dirname in ('attempts','logs','child_manifests'):
        (root/dirname).mkdir()
    state = r.initialize(root, m)
    first = dict(number=1, gpu=False, created_at=r.q.now(), finished_at=r.q.now(),
                 exit_code=75, command_elapsed_seconds=.01, job=original, error='fixture transient condition')
    r.q.atomic_json(root/'attempts/retry_train.attempt1.json', first)
    child = root/'child_manifests/recovery.json'
    r.q.atomic_json(child, dict(bundle='recovery', jobs=[original], source_manifest_sha256=r.q.digest(m)))
    state['bundles']['recovery'].update(status='admitted', reserved_seconds=0,
                                       manifest=str(child), manifest_sha256=r.q.file_hash(child))
    state['jobs']['retry_train'].update(job=retried, status='pending', attempts=[first], next_attempt_number=2)
    state.update(worker_status='running', worker=None, deadline_at=(
        datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=900)).isoformat())
    r.save(root, state, m)
    normalized_digest = state['manifest_sha256']
    start(root)
    completed = wait_until(lambda: (lambda s: s if s.get('worker_status') == 'finished' else None)(
        r.read(root/'state.json', {})))
    record = completed['jobs']['retry_train']
    second = r.read(root/'attempts/retry_train.attempt2.json')
    assert record['status'] == 'completed', record
    assert len(record['attempts']) == 2 and record['attempts'][0]['number'] == 1
    assert record['attempts'][1]['number'] == 2
    assert second['job']['argv'] == retried['argv']
    assert fresh_out.exists() and not old_out.exists()
    assert r.read(fresh_out)['run_dir'] == str(fresh_dir)
    assert (old_dir/'existing_scientific_attempt.txt').read_text() == 'preserve'
    assert completed['manifest_sha256'] == normalized_digest == r.q.digest(m)
    assert r.read(root/'budget_ledger.json')['incremental_spent_seconds'] == 0
    return dict(passed=True, root=str(root), normalized_manifest_preserved=True,
                attempts_preserved=2, launched_argv=second['job']['argv'], fresh_run_directory=str(fresh_dir),
                fresh_receipt=str(fresh_out), original_paths_preserved=True, duplicate_first_attempt=False)


def main():
    guardian = guardian_recovery()
    retry = persisted_retry_recovery()
    result = dict(passed=True, gpu_calls=False, gpu_spend_seconds=0,
                  guardian_death_recovery=guardian, persisted_attempt2_argv_recovery=retry,
                  scheduler_sha256=r.q.file_hash(r.ROOT/'run_overnight.py'),
                  tested='Actual CPU command and child group; worker and guardian killed and worker restarted. Retry2 persisted fixture uses original immutable child manifest.')
    r.q.atomic_json(r.ROOT/'recovery_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
