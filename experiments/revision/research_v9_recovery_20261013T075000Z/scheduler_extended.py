"""User-approved execution-window extension; scientific/GPU limits are unchanged."""
from pathlib import Path
import scheduler_recovery as recovery
import common_v9 as c

scheduler = recovery.scheduler
scheduler.__file__ = __file__
original_read, original_sha = c.read, c.sha

def read(path, default=None):
    if Path(path) == c.ROOT / 'manifest.json':
        path = c.ROOT / 'manifest_extended.json'
    return original_read(path, default)

def sha(path):
    if Path(path) == c.ROOT / 'manifest.json':
        path = c.ROOT / 'manifest_extended.json'
    return original_sha(path)

c.read, c.sha = read, sha
approval = c.read(c.ROOT / 'deadline_extension_approval.json')
assert approval['user_answer'] == 'Extend recovery to 18:35 CEST'
assert approval['cumulative_gpu_cap_seconds'] == 108000

def runtime_deadline(state):
    return min(state['absolute_deadline'], approval['approved_execution_deadline_unix'])

scheduler.runtime_deadline = runtime_deadline
original_status = scheduler.status

def status(state, manifest):
    original_status(state, manifest)
    result = c.read(c.ROOT / 'scheduler_status.json')
    result['original_execution_deadline'] = result['execution_deadline']
    result['execution_deadline'] = scheduler.stamp(runtime_deadline(state))
    result['deadline_extension_approval'] = str(c.ROOT / 'deadline_extension_approval.json')
    result['recovery_wait_seconds'] = max(0, __import__('time').time() - approval['applied_at_unix'])
    c.save(c.ROOT / 'scheduler_status.json', result)

scheduler.status = status
original_run = scheduler.subprocess.run

def run(command, *args, **kwargs):
    if isinstance(command, list) and len(command) > 1 and command[1] == str(c.ROOT / 'collect_v9.py'):
        command = list(command)
        command[1] = str(c.ROOT / 'collect_extended.py')
    return original_run(command, *args, **kwargs)

scheduler.subprocess.run = run

if __name__ == '__main__':
    scheduler.main()
