"""Focused queue checks with tiny CPU commands and no scientific artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import run_overnight as r


def main():
    root = Path(tempfile.mkdtemp(prefix='flooddistill_v8_queue_', dir='/tmp'))
    py = str(Path(sys.executable).resolve())
    def job(name, bundle, good=True, deps=()):
        out = root / (name + '.json')
        code = ('import json,time,pathlib;time.sleep(.3);pathlib.Path(' + repr(str(out)) +
                ').write_text(json.dumps({"ok":True}))') if good else 'raise SystemExit(9)'
        return {'id':name,'bundle':bundle,'role':'mock','gpu':False,'cwd':str(root),
                'argv':[py,'-c',code],'max_seconds':5,'deps':list(deps),
                'outputs':[{'path':str(out),'type':'json','equals':{'ok':True}}]}
    bundles = [{'id':'sentinel','jobs':[job('first','sentinel'),job('second','sentinel',deps=['first'])]},
               {'id':'failed','jobs':[job('failure','failed',False),job('dependent','failed',deps=['failure'])]},
               {'id':'refused','future_reserve_seconds':500,'jobs':[job('refused','refused')]}]
    retry=job('transient','retry');counter=root/'retry_count.txt';out=root/'transient.json'
    retry['argv']=[py,'-c','import pathlib,json; p=pathlib.Path('+repr(str(counter))+'); n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); '+
        'pathlib.Path('+repr(str(out))+').write_text(json.dumps({"ok":True})) if n else None; raise SystemExit(0 if n else 75)']
    retry['retry']={'exit_codes':[75],'diagnosis':'mock job explicitly signals one transient I/O condition'}
    bundles.insert(2,{'id':'retry','jobs':[retry]})
    manifest={'bundles':bundles,'historical_gpu_seconds':0,'incremental_cap_seconds':100,
              'preflight_gpu_seconds':0,'collection_reserve_seconds':0,'retry_reserve_seconds':0,
              'wall_seconds':1800,'frozen_inputs':{str(r.ROOT/'run_overnight.py'):r.q.file_hash(r.ROOT/'run_overnight.py')},
              'gpu_uuid':'MOCK_NO_GPU','gpu_lock':str(root/'gpu.lock'),'gpu_uuid_lock':str(root/'gpu_uuid.lock')}
    r.q.atomic_json(root/'manifest_initial.json',manifest)
    # The starter exits before either sentinel's guardian can finish.
    launcher = subprocess.run([py,str(r.ROOT/'run_overnight.py'),'start','--run-dir',str(root)],
                              capture_output=True,text=True,check=True)
    launch_time=time.time()
    for _ in range(40):
        state=r.read(root/'state.json',{})
        if state.get('worker_status')=='finished': break
        time.sleep(.5)
    assert state['worker_status']=='finished',state
    assert state['jobs']['first']['status']=='completed'
    assert state['jobs']['second']['status']=='completed'
    assert state['jobs']['dependent']['status']=='blocked'
    assert state['jobs']['refused']['status']=='deferred_budget'
    assert state['jobs']['transient']['status']=='completed'
    assert len(state['jobs']['transient']['attempts'])==2
    assert int(counter.read_text())==2
    assert r.stamp_seconds(state['jobs']['second']['attempts'][0]['command_started_at']) > launch_time
    assert state['session_id']==state['worker']['pid']
    assert r.read(root/'budget_ledger.json')['incremental_spent_seconds']==0
    # Promotion uses source scores only and exact C3 thresholds; passing/failing fixtures.
    receipts={}
    for arm in ['C0','C1','C2','C3','C4','C5']:
        for seed in [42,64]:
            path=root/f'{arm}_{seed}.json'
            score=.806 if arm=='C3' else .80
            r.q.atomic_json(path,{'source_validation':{'best_val_mIoU':score,'selected_metrics':{'wIoU':.70}}})
            receipts[(arm,seed)]=str(path)
    assert r.promotion_gate(receipts)['passed']
    path=Path(receipts[('C5',42)]); payload=r.read(path);payload['source_validation']['best_val_mIoU']=.809
    r.q.atomic_json(path,payload)
    assert not r.promotion_gate(receipts)['passed']
    # Stop a owned long mock and its child; ensure guardian records all elapsed.
    stop_root=Path(tempfile.mkdtemp(prefix='flooddistill_v8_stop_',dir='/tmp'))
    longjob=job('long','stop');pid_file=stop_root/'owned_child.json'
    longjob['argv']=[py,'-c','import subprocess,time,json,pathlib; p=subprocess.Popen(['+repr(py)+',"-c","import time;time.sleep(30)"]); '+
        'pathlib.Path('+repr(str(pid_file))+').write_text(json.dumps({"pid":p.pid}));time.sleep(30)']
    longjob['max_seconds']=40
    longmanifest={**manifest,'bundles':[{'id':'stop','jobs':[longjob]}]}
    r.q.atomic_json(stop_root/'manifest_initial.json',longmanifest)
    subprocess.run([py,str(r.ROOT/'run_overnight.py'),'start','--run-dir',str(stop_root)],capture_output=True,check=True)
    for _ in range(40):
        if pid_file.exists():break
        time.sleep(.1)
    assert pid_file.exists()
    child_identity=r.q.process_identity(r.read(pid_file)['pid'])
    subprocess.run([py,str(r.ROOT/'run_overnight.py'),'stop','--run-dir',str(stop_root)],capture_output=True,check=True)
    for _ in range(40):
        stopped=r.read(stop_root/'state.json',{})
        if stopped.get('worker_status')=='stopped':break
        time.sleep(.25)
    assert stopped['worker_status']=='stopped'
    assert not r.q.alive(child_identity)
    assert stopped['jobs']['long']['attempts'][0]['finished_at']
    report={'passed':True,'mock_root':str(root),'detached_starter_stdout':launcher.stdout.strip(),
            'second_sentinel_after_launcher_exit':True,'worker_session_is_independent':True,
            'dependency_block':True,'full_bundle_budget_refusal':True,'promotion_pass_and_fail':True,
            'cpu_only_spend_zero':True,'diagnosed_one_retry':True,'controlled_stop_reaps_owned_child':True,
            'actually_tested':'starter-process exit; not editor close, logout, reboot or sleep'}
    r.q.atomic_json(r.ROOT/'queue_checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
