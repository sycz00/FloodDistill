"""Lightweight reconciliation. Never opens incomplete cache/checkpoint files."""
import argparse
from collections import Counter
from pathlib import Path
import time
import common11 as c
from dependency11 import v10_ready

def main():
    started = time.time()
    queues = []
    for root in sorted(c.ROOT.parent.glob('research_v*')):
        if root == c.ROOT or not root.is_dir(): continue
        path = root / 'state.json'
        state = c.read(path)
        if not isinstance(state, dict): continue
        jobs = state.get('jobs', {})
        if not isinstance(jobs, dict): jobs = {}
        queues.append(dict(run=str(root), state_path=str(path), state_sha256=c.sha(path),
                           phase=state.get('phase'), finished=state.get('finished', False),
                           worker_alive=c.q.alive(state.get('worker')),
                           worker=state.get('worker'),
                           counts=dict(Counter(v.get('status', 'unknown') for v in jobs.values() if isinstance(v, dict)))))
    manifest = c.read(c.V10 / 'manifest.json')
    state = c.read(c.V10 / 'state.json', {})
    identities = []
    for name, job in manifest['jobs'].items():
        recipe = manifest.get('training', {}).get(name.removeprefix('20_train_').removeprefix('42_train_'))
        evidence = dict(run=str(c.V10), job=name, declaration=job, training=recipe,
                        frozen_protocol_sha256=c.sha(c.V10/'frozen_protocol_v10.md'),
                        code_freeze_sha256=c.sha(c.V10/'code_freeze.json'))
        identities.append(dict(**evidence, fingerprint=c.fingerprint(evidence),
                               status=state.get('jobs',{}).get(name,{}).get('status','PENDING'),
                               receipt=str(c.V10/'receipts'/(name+'.json'))))
    value = dict(outcome='completed', at=time.time(), queues=queues,
                 V10_jobs=identities, V11_dependencies={kind:v10_ready(kind) for kind in ['repair','transfer','conditioner']},
                 reuse='No V10 continuation assumed compatible: d0 uses a different objective, d2 lacks a matching 600-step receipt. All 28 core continuations reserved if source/cost gates pass.',
                 sealed='V9/V10 membership reused; non-confirmatory research exposure. Scores not read for design or admission.',
                 seconds=time.time()-started)
    c.save(c.ROOT/'dependency_evidence_ledger.json',value)
    print('Inventory complete:', len(queues), 'historical queues;', len(identities), 'V10 jobs;', value['V11_dependencies'], flush=True)

if __name__ == '__main__': main()
