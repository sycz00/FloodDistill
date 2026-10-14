"""Execute only declared GPU smokes under the same independent timeout guardian."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import audited_queue as q


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=q.read_json(a.plan);root=Path(plan['run_dir']);code=root/'run_overnight.py'
    ledger=root/'preflight_ledger.json';attempts=q.read_json(ledger) if ledger.exists() else []
    env={**q.THREAD_ENV,'CUDA_VISIBLE_DEVICES':plan['gpu_uuid'],'PYTHONUNBUFFERED':'1',
         'PYTHONDONTWRITEBYTECODE':'1','NO_ALBUMENTATIONS_UPDATE':'1','WANDB_MODE':'disabled',
         'MPLCONFIGDIR':str(root/'mpl_cache'),'HF_HUB_OFFLINE':'1',
         'PYTHONPATH':'/home/schmalstieg/repos/_deps/efficientvit:/home/schmalstieg/repos/_deps/flooddistill_revision_python'}
    for job in plan['jobs']:
        path=root/'preflight_attempts'/(job['id']+'.json')
        if path.exists():
            previous=q.read_json(path)
            if previous.get('exit_code')==0: continue
            raise RuntimeError('preserve failed preflight; create new identified job after diagnosis: '+str(path))
        spent=sum(r.get('command_elapsed_seconds',0) for r in attempts)
        if spent+job['max_seconds']>plan['preflight_cap_seconds']:
            raise RuntimeError('preflight reservation exhausted')
        fds=[]
        try:
            for lock in (plan['gpu_lock'],plan['gpu_uuid_lock']):
                fd=q.lock_fd(lock)
                if fd is None:raise RuntimeError('shared GPU lock is occupied; defer')
                fds.append(fd)
            allowed,reason=q.gpu_available(plan,{**job,'env':{'CUDA_VISIBLE_DEVICES':plan['gpu_uuid']}})
            if not allowed:raise RuntimeError(reason)
            log=root/'logs'/('preflight_'+job['id']+'.log');log.parent.mkdir(exist_ok=True)
            attempt={'number':1,'created_at':q.now(),'job':{**job,'gpu':True,'env':env},'gpu':True,
                     'gpu_uuid':plan['gpu_uuid'],'deadline_epoch':time.time()+job['max_seconds'],
                     'log_path':str(log),'frozen_inputs':{str(code):q.file_hash(code),
                     **{str(root/name):q.file_hash(root/name) for name in job.get('code_files',[])}}}
            q.atomic_json(path,attempt)
            guardian=subprocess.Popen([sys.executable,str(code),'_attempt','--run-dir',str(root),
                '--attempt',str(path),'--fds',','.join(map(str,fds))],stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,pass_fds=tuple(fds),env={**os.environ,**env})
            print('preflight_started',job['id'],guardian.pid,flush=True)
            guardian.wait(timeout=job['max_seconds']+30)
            result=q.read_json(path);attempts.append(result);q.atomic_json(ledger,attempts)
            print('preflight_finished',job['id'],result.get('exit_code'),result.get('command_elapsed_seconds'),flush=True)
            if result.get('exit_code')!=0:raise RuntimeError('preflight failed: '+job['id']+' '+str(result.get('error')))
        finally:
            for fd in fds:os.close(fd)


if __name__=='__main__':main()
