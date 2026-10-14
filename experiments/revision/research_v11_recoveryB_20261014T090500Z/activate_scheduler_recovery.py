"""Replace only the owned controller; retain all scientific inputs and job state."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'code'))
import worker11 as old

overlay = root / 'scheduler_runtime_recovery'
assert (overlay / 'worker11.py').is_file()
manifest = old.read(root / 'manifest.json')
state = old.read(root / 'state.json')
old.freeze_check(root, state['manifest_sha256'])
identity = state['worker']
assert old.q.alive(identity), 'Only replace the verified live owned controller'
command = Path(f'/proc/{identity["pid"]}/cmdline').read_bytes().split(b'\0')
assert str(root / 'code/worker11.py').encode() in command and b'run' in command
assert str(root).encode() in command
before = json.loads(json.dumps(state))
os.kill(identity['pid'], signal.SIGKILL)
time.sleep(.5)
# The controller has no training child group ownership; guardians run independently.
state = old.read(root / 'state.json')
assert state['jobs'] == before['jobs'], 'Unexpected job change during controller handoff'
state['charges']['scheduler_cpu_measurement_recovery'] = dict(
    reserved_core_seconds=180, measured_process_cpu_seconds=None,
    reason='Conservative allowance for CPU telemetry diagnosis, overlay tests and controller handoff')
state.setdefault('controller_recoveries', []).append(dict(
    at=time.time(), previous=identity, replacement_code=str(overlay / 'worker11.py'),
    replacement_sha256=old.q.file_hash(overlay / 'worker11.py'),
    reason='Use sampled CPU utilization with existing allocation limits; localized runnable threads inflated load average'))
old.q.atomic_json(root / 'state.json', state)
with (root / 'logs/worker.log').open('a') as log:
    process = subprocess.Popen([manifest['python'], str(overlay / 'worker11.py'),
        'run', '--root', str(root)], cwd=manifest['cwd'], env=old.env(1, manifest.get('env')),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(outcome='launched', at=time.time(), worker=old.q.process_identity(process.pid),
    previous=identity, launcher_pid=os.getpid(), unchanged_jobs=True,
    unchanged_manifest_sha256=state['manifest_sha256'],
    unchanged_execution_deadline=state['execution_deadline'],
    controller_sha256=old.q.file_hash(overlay / 'worker11.py'))
old.q.atomic_json(overlay / 'launch_receipt.json', receipt)
for path in [root / 'CONTROLS.txt', root.parent / 'recovery_controls_20261014.txt']:
    path.write_text(path.read_text().replace(str(root / 'code/worker11.py'), str(overlay / 'worker11.py')))
print(json.dumps(receipt))
