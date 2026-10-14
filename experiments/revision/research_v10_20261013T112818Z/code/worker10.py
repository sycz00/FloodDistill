"""Three-slot CPU wrapper using the existing audited queue's identity and atomic IO."""
import argparse,os,signal,subprocess,sys,time,fcntl,shutil
from pathlib import Path
import common10 as c
sys.path.insert(0,str(c.V9))
import audited_queue as q
TERMINAL={'COMPLETED','TECHNICAL_FAILED','UNAVAILABLE','DEFERRED','SOURCE_GATE_NOT_MET'}
THREADS={k:'8' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}
def env():
    return dict(os.environ,**THREADS,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',NO_ALBUMENTATIONS_UPDATE='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/flooddistill_mpl',PYTHONPATH='/home/schmalstieg/repos/_deps/efficientvit:'+str(c.ROOT/'code'))
def freeze_check():
    for name,expected in c.read(c.ROOT/'code_freeze.json')['files'].items():assert c.sha(c.ROOT/name)==expected,('Frozen input modified',name)
def stop(identity,sig=signal.SIGTERM):
    if q.alive(identity):
        try:os.killpg(identity['pid'],sig)
        except ProcessLookupError:pass
def reconcile_gpu_reservation():
    # Reconcile only this queue's reservation after an owned GPU guardian is lost.
    import scheduler_v9 as old
    key=str(c.ROOT)+':UNet_GPU_final';path=c.ROOT/'gpu_lost_guardian_receipt.json'
    with old.Ledger() as ledger:
        reservation=ledger.data['reservations'].get(key)
        if reservation:
            c.save(path,dict(outcome='technical_failed',seconds=1800,charge='Conservative full reserved ceiling after lost owned guardian',CPU_fallback=True))
            ledger.data['charges'][str(path)]=dict(seconds=1800,receipt=str(path),sha256=c.sha(path));ledger.data['reservations'].pop(key)
def guard(name,slot):
    freeze_check();manifest=c.read(c.ROOT/'manifest.json');job=manifest['jobs'][name];state=c.read(c.ROOT/'state.json');started=time.time()
    os.sched_setaffinity(0,set(state['CPU_slots'][slot]));child=None;reason=None
    def request_stop(*_):
        nonlocal reason
        reason='Owned worker stop requested'
    signal.signal(signal.SIGTERM,request_stop)
    with open(c.ROOT/'logs'/(name+'.log'),'a') as output:
        child=subprocess.Popen([c.PYTHON,str(c.ROOT/'code/action10.py'),name],cwd=c.REPO,env=env(),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        identity=q.process_identity(child.pid);c.save(c.ROOT/'running'/(name+'.json'),dict(guardian=q.process_identity(os.getpid()),child=identity,started_at=started,slot=slot))
        limit=min(started+job['max_seconds'],state['deadline']-1800)
        while child.poll() is None:
            if reason or time.time()>=limit:
                reason=reason or 'Per-job ceiling or collection cutoff';stop(identity)
                try:child.wait(timeout=40)
                except subprocess.TimeoutExpired:stop(identity,signal.SIGKILL);child.wait()
                break
            time.sleep(2)
    result=c.read(c.ROOT/'stage_results'/(name+'.json'));success=child.returncode==0 and bool(result) and result.get('outcome')=='completed'
    c.save(c.ROOT/'receipts'/(name+'.json'),dict(outcome='completed' if success else 'deferred' if reason else 'technical_failed',job=name,returncode=child.returncode,reason=reason,seconds=time.time()-started,result_sha256=c.sha(c.ROOT/'stage_results'/(name+'.json')) if result else None,log=str(c.ROOT/'logs'/(name+'.log')),guardian=q.process_identity(os.getpid()),child=identity))
def condition(job,state):
    if job.get('optional') and time.time()+job['max_seconds']+6*3600>=state['deadline']-1800:return 'DEFERRED','Tier C does not fit the fixed final-evaluation reserve'
    if job.get('qat'):
        selection=c.read(c.ROOT/'smallest_viable_selection.json',{});graph=selection.get('graph') if job['graph']=='SELECTED' else job['graph']
        if not graph:return 'UNAVAILABLE','Smallest viable source group unavailable'
        rows=c.read(c.ROOT/'compression_control_selection.json',{}).get('qat_admission',[])
        admission=next((x for x in rows if x['graph']==graph and x['precision']==job['precision']),None)
        if not admission:return 'UNAVAILABLE','Required prepared source QAT parent unavailable'
        if not admission['admitted']:return 'SOURCE_GATE_NOT_MET','Unchanged source PTQ point is within the 1pp source gate'
    return None,None
def collect(state,reason):
    for name,row in state['jobs'].items():
        if row['status'] not in TERMINAL:row.update(status='DEFERRED',reason=reason)
    state.update(finished=True,finished_at=time.time(),phase='COLLECTING');c.save(c.ROOT/'state.json',state)
    with open(c.ROOT/'logs/final_collection.log','a') as output:
        result=subprocess.run([c.PYTHON,str(c.ROOT/'code/collect10.py')],env=env(),cwd=c.REPO,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=1500)
    state.update(phase='FINISHED',collection_returncode=result.returncode);c.save(c.ROOT/'state.json',state)
def run():
    fd=q.lock_fd(c.ROOT/'worker.lock');assert fd is not None,'An owned worker is already active'
    freeze_check();manifest=c.read(c.ROOT/'manifest.json');jobs=manifest['jobs'];state=c.read(c.ROOT/'state.json')
    if state and state.get('finished'):return
    if state is None:
        affinity=sorted(os.sched_getaffinity(0));cores=[];seen=set()
        for cpu in affinity:
            p=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology';key=( (p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
            if key not in seen:cores.append(cpu);seen.add(key)
        assert len(cores)>=24,'24 physical CPU cores are required'
        launch=time.time();state=dict(launch=launch,deadline=launch+72000,collection_cutoff=launch+70200,CPU_slots=[cores[i:i+8] for i in range(0,24,8)],jobs={name:dict(status='PENDING') for name in jobs},phase='RUNNING',finished=False)
    state['worker']=q.process_identity(os.getpid());c.save(c.ROOT/'state.json',state);requested=False
    def terminate(*_):
        nonlocal requested
        requested=True
    signal.signal(signal.SIGTERM,terminate)
    while True:
        now=time.time();control=c.read(c.ROOT/'control.json',{});active={}
        for name,row in state['jobs'].items():
            if row['status']!='RUNNING':continue
            receipt=c.read(c.ROOT/'receipts'/(name+'.json'));ident=row.get('guardian')
            if receipt and not q.alive(ident):
                row.update(status={'completed':'COMPLETED','deferred':'DEFERRED','technical_failed':'TECHNICAL_FAILED'}[receipt['outcome']],receipt=str(c.ROOT/'receipts'/(name+'.json')),reason=receipt.get('reason'));continue
            if not q.alive(ident):
                running=c.read(c.ROOT/'running'/(name+'.json'),{});stop(running.get('child'))
                if jobs[name]['action']=='gpu':
                    time.sleep(3);reconcile_gpu_reservation()
                row.update(status='TECHNICAL_FAILED',reason='Guardian lost; owned child stopped; retained outputs; no duplicate restart');continue
            active[name]=row
        stopping=requested or control.get('stop') or now>=state['deadline']-1800
        if stopping:
            for name,row in active.items():stop(row['guardian'])
            if not active:
                collect(state,'Fixed deadline collection cutoff' if now>=state['deadline']-1800 else 'User stop');return
        elif not control.get('pause'):
            slots={x['slot'] for x in active.values()}
            for name,job in jobs.items():
                row=state['jobs'][name]
                if row['status']!='PENDING':continue
                dependencies=[state['jobs'][d]['status'] for d in job.get('deps',[])]
                if not all(x in TERMINAL for x in dependencies):continue
                if not job.get('soft_dependencies') and any(x!='COMPLETED' for x in dependencies):row.update(status='UNAVAILABLE',reason='Required verification/source dependency did not complete');continue
                status,reason=condition(job,state)
                if status:row.update(status=status,reason=reason);continue
                reserve=5*3600 if job.get('tier') in ['A','B','C'] else 0
                if now+job['max_seconds']+reserve>=state['deadline']-1800:row.update(status='DEFERRED',reason='Does not fit fixed deadline and final-readout reserve');continue
                if len(slots)>=3:continue
                if shutil.disk_usage('/data1/schmalstieg/flooddistill').free<32*2**30:row.update(status='UNAVAILABLE',reason='Data disk below 32GiB headroom');continue
                slot=next(i for i in range(3) if i not in slots)
                with open(c.ROOT/'logs/guardians.log','a') as output:
                    child=subprocess.Popen([c.PYTHON,str(Path(__file__).resolve()),'guard','--job',name,'--slot',str(slot)],cwd=c.REPO,env=env(),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                row.update(status='RUNNING',guardian=q.process_identity(child.pid),slot=slot,started_at=now);slots.add(slot);c.save(c.ROOT/'state.json',state)
        state.update(heartbeat=time.time(),phase='STOPPING' if stopping else 'PAUSED' if control.get('pause') else 'RUNNING');c.save(c.ROOT/'state.json',state)
        c.save(c.ROOT/'scheduler_status.json',dict(heartbeat=state['heartbeat'],worker=state['worker'],phase=state['phase'],deadline=state['deadline'],remaining_seconds=state['deadline']-time.time(),counts={label:sum(x['status']==label for x in state['jobs'].values()) for label in sorted({x['status'] for x in state['jobs'].values()})},running=[name for name,x in state['jobs'].items() if x['status']=='RUNNING']))
        if all(row['status'] in TERMINAL for row in state['jobs'].values()):collect(state,'Bounded queue exhausted');return
        time.sleep(15)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','guard','status','pause','resume','stop']);parser.add_argument('--job');parser.add_argument('--slot',type=int);args=parser.parse_args()
    if args.action=='run':run()
    elif args.action=='guard':guard(args.job,args.slot)
    elif args.action=='status':
        state=c.read(c.ROOT/'state.json',{});print(__import__('json').dumps(c.read(c.ROOT/'scheduler_status.json',state),indent=2));print('worker_alive',q.alive(state.get('worker')))
    else:
        control=c.read(c.ROOT/'control.json',{});control['pause']=args.action=='pause';control['stop']=args.action=='stop';c.save(c.ROOT/'control.json',control)
if __name__=='__main__':main()
