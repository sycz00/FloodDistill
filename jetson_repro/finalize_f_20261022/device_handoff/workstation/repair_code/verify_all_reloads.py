#!/usr/bin/env python3
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(seed):
    env=dict(os.environ,MPLCONFIGDIR='/tmp/b0_reload_mpl')
    with (ROOT/f'logs/verify_reload_s{seed}.log').open('x') as stream:
        subprocess.run([sys.executable,str(ROOT/'code/verify_reload_v2.py'),'--seed',str(seed)],env=env,check=True,stdout=stream,stderr=subprocess.STDOUT)
    data=json.loads((ROOT/f'provenance/RELOAD_VERIFICATION_s{seed}.json').read_text());assert data['status']=='PASS';return data['rows'][0]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,[42,64,91,107]))
(ROOT/'provenance/RELOAD_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',process_isolation='One frozen runtime per seed; original multi-seed verifier stopped on PyTorch interop-thread reinitialization after seed42. No scientific results changed.',rows=rows),indent=2)+'\n')
print('PASS_ALL_FOUR_RELOAD_AND_ORT_REPRODUCTION',flush=True)
