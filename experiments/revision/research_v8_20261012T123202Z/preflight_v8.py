"""Bounded real smoke using the inherited timeout guardian and GPU locks."""
import os
from pathlib import Path
import subprocess
import sys
import time
import run_overnight as r

ROOT=Path(__file__).resolve().parent


def main():
    old=r.read(ROOT/'preflight_ledger.json',[])
    cap=r.read(ROOT/'budget_reconciliation.json')['incremental_cap_seconds']
    spent=sum(a.get('command_elapsed_seconds',0) for a in old)
    if spent+300>cap:raise RuntimeError('No remaining budget for preflight')
    job=dict(id='real_smoke',bundle='preflight',role='smoke',gpu=True,max_seconds=300,cwd=str(ROOT.parents[2]),
        argv=[sys.executable,str(ROOT/'jobs_v8.py'),'smoke','--out',str(ROOT/'smoke_receipt.json')],
        outputs=[dict(path=str(ROOT/'smoke_receipt.json'),type='json',equals={'outcome':'completed'})],
        env={'HF_HUB_OFFLINE':'1','WANDB_MODE':'disabled','NO_ALBUMENTATIONS_UPDATE':'1',
            'MPLCONFIGDIR':str(ROOT/'mpl_cache'),'PYTHONPATH':'/home/schmalstieg/repos/_deps/efficientvit:/home/schmalstieg/repos/_deps/flooddistill_revision_python'})
    manifest=dict(gpu_index=0,gpu_uuid='GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc',headroom_mb=3072)
    fds=[]
    try:
        for path in ['/data1/schmalstieg/flooddistill/revision_gpu.lock',
                     '/data1/schmalstieg/flooddistill/revision_gpu_GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc.lock']:
            fd=r.q.lock_fd(path)
            if fd is None:raise RuntimeError('GPU lock held')
            fds.append(fd)
        allowed,why=r.q.gpu_available(manifest,{**job,'env':{'CUDA_VISIBLE_DEVICES':manifest['gpu_uuid']}})
        if not allowed:raise RuntimeError(why)
        attempt=ROOT/f'preflight_attempt{len(old)+1}.json'
        frozen={str(p):r.q.file_hash(p) for p in ROOT.glob('*.py')}
        frozen[str(ROOT/'config.json')]=r.q.file_hash(ROOT/'config.json')
        r.q.atomic_json(attempt,dict(number=len(old)+1,created_at=r.q.now(),job=job,gpu=True,
            gpu_uuid=manifest['gpu_uuid'],deadline_epoch=time.time()+300,log_path=str(ROOT/f'preflight_attempt{len(old)+1}.log'),frozen_inputs=frozen))
        proc=subprocess.Popen([sys.executable,str(ROOT/'run_overnight.py'),'_attempt','--run-dir',str(ROOT),
            '--attempt',str(attempt),'--fds',','.join(map(str,fds))],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,pass_fds=tuple(fds))
        proc.wait(timeout=330)
        result=r.read(attempt);old.append(result);r.q.atomic_json(ROOT/'preflight_ledger.json',old)
        print({'exit_code':result.get('exit_code'),'elapsed_seconds':result.get('command_elapsed_seconds'),'log':result['log_path']},flush=True)
        if result.get('exit_code')!=0:raise RuntimeError(result.get('error','smoke command failed'))
    finally:
        for fd in fds:os.close(fd)


if __name__=='__main__':main()
