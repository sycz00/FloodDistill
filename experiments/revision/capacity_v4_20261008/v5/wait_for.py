"""Block until v5 queue jobs reach a terminal state; print a compact status. Usage: wait_for.py <state> [job ids | any]"""
import json
import sys
import time

TERMINAL = {'succeeded', 'failed', 'blocked', 'skipped'}
state_path, targets = sys.argv[1], sys.argv[2:]
known = {j for j, r in json.load(open(state_path))['jobs'].items() if r['status'] in {'failed', 'blocked'}}
while True:
    jobs = json.load(open(state_path))['jobs']
    if targets == ['any']:
        done = [j for j, r in jobs.items() if r['status'] in TERMINAL]
        if done:
            break
    elif all(jobs[t]['status'] in TERMINAL for t in targets):
        break
    if any(r['status'] in {'failed', 'blocked'} and j not in known for j, r in jobs.items()):
        break
    time.sleep(30)
counts = {}
for j, r in jobs.items():
    counts.setdefault(r['status'], []).append(j)
for k, v in counts.items():
    print(k, len(v), v if k != 'pending' else '')
for j, r in jobs.items():
    if r['status'] in {'failed', 'blocked'}:
        print('PROBLEM', j, r.get('reason'))
