"""Detach the frozen V10 recovery worker; never reset an existing deadline."""
import os
from pathlib import Path
import subprocess
import time
import worker_gpu_fast10 as worker


def main():
    c = worker.c
    worker.freeze_check()
    state = c.read(c.ROOT / 'state.json', {})
    if state.get('finished'):
        raise SystemExit('This finite recovery is already finished; no deadline reset')
    if worker.q.alive(state.get('worker')):
        raise SystemExit('Recovery worker already alive')
    for name in ['logs', 'running', 'receipts', 'stage_results', 'GPU_admissions']:
        (c.ROOT / name).mkdir(exist_ok=True)
    command = [c.PYTHON, str(c.ROOT / 'code/worker_gpu_fast10.py'), 'run']
    environment = worker.env()
    environment.update({key: '1' for key in worker.THREADS})
    with open(c.ROOT / 'logs/worker.log', 'a') as output:
        child = subprocess.Popen(command, cwd=c.REPO, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(at=time.time(), launcher=worker.q.process_identity(os.getpid()),
                   worker=worker.q.process_identity(child.pid), command=command,
                   manifest_sha256=c.sha(c.ROOT / 'manifest.json'),
                   shared_ledger=str(worker.gq.LEDGER),
                   cumulative_gpu_cap_seconds=c.read(c.ROOT / 'GPU_budget_policy.json')['cumulative_cap_seconds'],
                   deadline_policy='Fresh authorized 12-hour recovery; subsequent restarts retain first launch deadline',
                   CPU_policy='Three disjoint eight-core slots; GPU main two threads plus at most six historical loader workers; CPU command eight threads')
    c.save(c.ROOT / 'detached_launch.json', receipt)
    print('Detached V10 recovery worker', child.pid, flush=True)


if __name__ == '__main__':
    main()
