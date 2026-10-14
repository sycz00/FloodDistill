"""GPU-aware three-slot host wrapper using the existing audited queue's identity and atomic IO."""
import argparse,os,signal,subprocess,sys,time,fcntl,shutil,math
from pathlib import Path
import common10 as c
if os.environ.get("V10_WORKER_ROOT"):
    c.ROOT=Path(os.environ["V10_WORKER_ROOT"]).resolve()
sys.path.insert(0,str(c.V9))
import audited_queue as q
import scheduler_v9 as gq
import common_v9 as old
TERMINAL={'COMPLETED','TECHNICAL_FAILED','UNAVAILABLE','DEFERRED','SOURCE_GATE_NOT_MET'}
THREADS={k:'8' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}
def env():
    return dict(os.environ,**THREADS,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',NO_ALBUMENTATIONS_UPDATE='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/flooddistill_mpl',PYTHONPATH='/home/schmalstieg/repos/_deps/efficientvit:'+str(c.ROOT/'code'))
def protected_key():return str(c.ROOT)+':protected_final'
def cohort_key(job):return str(c.ROOT)+':cohort:'+job['cohort_group']
def cohort_coverage(ledger,job):
    return ledger.data['reservations'].get(cohort_key(job),{}).get('remaining_seconds',0) if job.get('cohort_group') else 0
def protected_coverage(ledger,job):
    if job.get('action')=='evaluate' and job.get('mode')=='sealed':
        return min(job['max_seconds'],ledger.data['reservations'].get(protected_key(),{}).get('remaining_seconds',0))
    return 0
def budget_fits(ledger,job,cap):
    if job.get('cohort_group'):
        add=max(0,job.get('cohort_remaining_seconds',job['max_seconds'])-cohort_coverage(ledger,job))
    else:add=job['max_seconds']-protected_coverage(ledger,job)
    return ledger.spend()+ledger.reserved()+add<=cap
def reserve_job(ledger,name,job):
    covered=protected_coverage(ledger,job)
    if covered:ledger.data['reservations'][protected_key()]['remaining_seconds']-=covered
    cohort=None
    if job.get('cohort_group'):
        cohort=cohort_key(job)
        required=job.get('cohort_remaining_seconds',job['max_seconds'])
        entry=ledger.data['reservations'].setdefault(cohort,dict(run=str(c.ROOT),block='cohort:'+job['cohort_group'],remaining_seconds=required,created=q.now()))
        entry['remaining_seconds']=max(entry['remaining_seconds'],required)
        assert entry['remaining_seconds']>=job['max_seconds']
        entry['remaining_seconds']-=job['max_seconds']
    ledger.data['reservations'][str(c.ROOT)+':'+name]=dict(run=str(c.ROOT),block=name,remaining_seconds=job['max_seconds'],protected_coverage=covered,cohort_reservation=cohort,created=q.now())
def undo_reservation(ledger,name):
    row=ledger.data['reservations'].pop(str(c.ROOT)+':'+name,None)
    if row and row.get('protected_coverage'):
        ledger.data['reservations'][protected_key()]['remaining_seconds']+=row['protected_coverage']
    if row and row.get('cohort_reservation'):
        ledger.data['reservations'][row['cohort_reservation']]['remaining_seconds']+=row['remaining_seconds']

def pending_provider(job):
    spec=job.get('reuse_pending_receipt')
    if not spec:return None
    receipt=c.read(Path(spec['path']))
    if not receipt:return None
    if receipt.get('outcome')!='completed':return dict(status='UNAVAILABLE',reason='Existing owned provider failed; no duplicate scientific retry')
    stage=Path(spec['stage_result'])
    assert c.sha(stage)==receipt['result_sha256'],'Completed provider stage hash mismatch'
    path=Path(spec['scientific_receipt']);scientific=c.read(path)
    assert scientific.get('outcome')=='completed' and scientific.get('status')=='complete'
    for stem in ['best','last']:
        assert c.sha(scientific[stem+'_path'])==scientific[stem+'_sha256'],'Completed provider checkpoint hash mismatch'
    destination=Path(spec['publish_scientific_receipt'])
    assert destination.resolve().is_relative_to(c.ROOT.resolve()),'Provider publication outside successor root'
    c.save(destination,scientific)
    return dict(status='COMPLETED',receipt=str(spec['path']),reason='Reused matching prior active provider after atomic completion and checkpoint hash verification')

def inherited_active(jobs):
    active=[]
    for provider in sorted({job['inherited_active_provider'] for job in jobs.values() if job.get('inherited_active_provider')}):
        root=Path(provider);state=c.read(root/'state.json',{})
        assert state,'Inherited provider state is unavailable'
        assert state.get('finished') or c.read(root/'control.json',{}).get('pause'),'Inherited provider must remain paused while successor owns admissions'
        for name,row in state.get('jobs',{}).items():
            if row.get('status')=='RUNNING' and q.alive(row.get('guardian')):
                active.append(dict(row,provider=provider,job=name))
    return active

def initialize_protection(ledger,manifest,cap):
    scheduler=manifest.get('scheduler',{});protected=scheduler.get('protected_gpu_seconds',0)
    cohort_provider=scheduler.get('cohort_reservation_provider')
    if cohort_provider:
        for key in list(ledger.data['reservations']):
            if key.startswith(cohort_provider+':cohort:'):
                destination=str(c.ROOT)+key[len(cohort_provider):]
                assert destination not in ledger.data['reservations'],'Duplicate successor cohort reservation'
                row=ledger.data['reservations'].pop(key)
                ledger.data['reservations'][destination]=dict(row,run=str(c.ROOT),transferred_from=key)
    if not protected:return
    provider=scheduler.get('protected_reservation_provider')
    existing=ledger.data['reservations'].get(protected_key(),{}).get('remaining_seconds',0)
    if provider and provider!=protected_key() and not existing:
        entry=ledger.data['reservations'].get(provider)
        if entry:
            assert entry['remaining_seconds']==protected,'Prior protected reserve differs from frozen successor requirement'
            ledger.data['reservations'].pop(provider)
            ledger.data['reservations'][protected_key()]=dict(entry,run=str(c.ROOT),transferred_from=provider)
            c.save(c.ROOT/'protected_reservation_transfer.json',dict(at=q.now(),provider=provider,destination=protected_key(),seconds=protected,charged_GPU_seconds=0))
            existing=protected
    assert ledger.spend()+ledger.reserved()+max(0,protected-existing)<=cap,'Protected final readout does not fit existing cumulative GPU cap'
    if not existing:ledger.data['reservations'][protected_key()]=dict(run=str(c.ROOT),block='protected_final',remaining_seconds=protected,created=q.now())

def cohort_remaining(job,jobs,state,manifest):
    names=manifest.get('scheduler',{}).get('cohorts',{}).get(job.get('cohort_group'),[])
    return sum(jobs[n]['max_seconds'] for n in names if state['jobs'][n]['status']=='PENDING' and not jobs[n].get('reuse_pending_receipt'))

def defer_cohort(name,job,jobs,state,manifest,reason):
    for member in manifest.get('scheduler',{}).get('cohorts',{}).get(job.get('cohort_group'),[name]):
        if state['jobs'][member]['status']=='PENDING' and not jobs[member].get('reuse_pending_receipt'):
            state['jobs'][member].update(status='DEFERRED',reason=reason)

def readout_profile(path,verify_current=True):
    result=c.read(path)
    assert result and result.get('outcome')=='completed' and result.get('source_only') is True,'Source-only readout profile incomplete'
    if verify_current:
        assert result['frozen_model_manifest_sha256']==c.sha(c.ROOT/'readout_models.json'),'Profile model inventory changed'
        assert result['frozen_settings_sha256']==c.sha(c.ROOT/'operating_points.json'),'Profile operating points changed'
    seconds=result['slot_seconds']
    assert len(seconds)==3 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>30 for x in seconds),'Invalid full-readout slot ceilings'
    return result,[int(math.ceil(x)) for x in seconds]

def verify_profile_receipt(name,path):
    stage=c.ROOT/'stage_results'/(name+'.json');receipt=c.read(c.ROOT/'receipts'/(name+'.json'),{})
    assert receipt.get('outcome')=='completed' and receipt.get('result_sha256')==c.sha(stage),'Source profile guardian/stage receipt mismatch'
    assert c.read(stage).get('profile_sha256')==c.sha(path),'Source profile changed after completed stage'

def expand_readout_protection(ledger,required,cap):
    existing=ledger.data['reservations'].get(protected_key(),{}).get('remaining_seconds',0)
    extra=max(0,required-existing)
    if ledger.spend()+ledger.reserved()+extra>cap:return False
    if extra:
        row=ledger.data['reservations'].setdefault(protected_key(),dict(run=str(c.ROOT),block='protected_final',remaining_seconds=0,created=q.now()))
        row['remaining_seconds']+=extra;row['source_profile_reservation']=True
    return True

def apply_early_readout_gate(state,jobs,manifest):
    name=manifest.get('scheduler',{}).get('early_readout_profile_job')
    if not name or state.get('early_readout_cost_gate') or state['jobs'][name]['status']!='COMPLETED':return
    path=c.ROOT/jobs[name]['profile_output'];verify_profile_receipt(name,path);profile,seconds=readout_profile(path)
    cap=c.read(c.ROOT/'GPU_budget_policy.json')['cumulative_cap_seconds']
    with gq.Ledger() as ledger:fits=expand_readout_protection(ledger,sum(seconds),cap)
    state['early_readout_cost_gate']=dict(status='RESERVED' if fits else 'DEFERRED',seconds=seconds,total_seconds=sum(seconds),profile_sha256=c.sha(path),source_only=True,reason='Measured full Tier A readout protected before extra training' if fits else 'Measured Tier A readout exceeds remaining cumulative allowance; extras prohibited')
    c.save(c.ROOT/'early_readout_cost_gate.json',state['early_readout_cost_gate'])

def final_readout_job(name,job,state,jobs,manifest):
    path=c.ROOT/job['dynamic_readout_profile']
    verify_profile_receipt(manifest['scheduler']['final_readout_profile_job'],path)
    profile,seconds=readout_profile(path)
    frozen=c.read(c.ROOT/'final_evaluation_freeze.json',{})
    assert frozen.get('model_manifest_sha256')==profile['frozen_model_manifest_sha256'] and frozen.get('source_settings_sha256')==profile['frozen_settings_sha256'],'Final profile differs from exact final scientific freeze'
    names=manifest['scheduler']['final_readout_jobs']
    remaining=sum(seconds[jobs[n]['slot']] for n in names if state['jobs'][n]['status']=='PENDING')
    active_remaining=sum(max(0,state['jobs'][n]['started_at']+state['jobs'][n]['effective_max_seconds']-time.time()) for n in names if state['jobs'][n]['status']=='RUNNING')
    if time.time()+remaining+active_remaining>=state['deadline']-1800:
        return None,'Measured whole final readout does not fit inherited collection cutoff'
    cap=c.read(c.ROOT/'GPU_budget_policy.json')['cumulative_cap_seconds']
    with gq.Ledger() as ledger:
        if not expand_readout_protection(ledger,remaining,cap):return None,'Measured whole final readout does not fit cumulative GPU cap'
    return dict(job,max_seconds=seconds[job['slot']],readout_profile_sha256=c.sha(path)),None
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
def availability(row,job,history,now):
    # Recovery uses the existing V9 three-observation idle rule and UUID locks.
    # No foreign-process co-location is admitted in this launch.
    if row.get('processes'):return False,'foreign compute process; recovery requires exclusive idle UUID',False
    return gq.availability(row,job,history,now)

def reusable(job):
    spec=job.get('reuse_receipt')
    if not spec:return None
    path=Path(spec['path'] if isinstance(spec,dict) else spec)
    if isinstance(spec,dict) and spec.get('sha256') and c.sha(path)!=spec['sha256']:
        raise ValueError('Reused completion receipt hash changed: '+str(path))
    receipt=c.read(path)
    if receipt.get('outcome')!='completed':raise ValueError('Reused receipt is not completed')
    for item in job.get('reuse_inputs',[]):
        if c.sha(item['path'])!=item['sha256']:raise ValueError('Reused output hash changed: '+item['path'])
    return str(path)

def cpu_available():
    try:
        mem={line.split(':')[0]:int(line.split(':')[1].split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        return os.getloadavg()[0]<40 and mem['MemAvailable']>24*2**30
    except (OSError,ValueError,KeyError):return False

def guard(name,slot,gpu=None,lock_fd=None):
    freeze_check();manifest=c.read(c.ROOT/'manifest.json');job=manifest['jobs'][name];state=c.read(c.ROOT/'state.json');started=time.time()
    if job.get('dynamic_readout_profile'):
        row=state['jobs'][name];path=c.ROOT/job['dynamic_readout_profile'];_,seconds=readout_profile(path)
        assert row['effective_max_seconds']==seconds[job['slot']] and row['readout_profile_sha256']==c.sha(path),'Dynamic ceiling differs from validated source profile'
        job=dict(job,max_seconds=row['effective_max_seconds'])
    os.sched_setaffinity(0,set(state['CPU_slots'][slot]));child=None;reason=None
    def request_stop(*_):
        nonlocal reason
        reason='Owned worker stop requested'
    signal.signal(signal.SIGTERM,request_stop)
    command_env=env()
    if gpu:
        command_env.update({key:'2' for key in THREADS})
        command_env.update(CUDA_VISIBLE_DEVICES=gpu,V10_COMPUTE_DEVICE='cuda',V10_WORKER_CPU_THREADS='2',V10_PEAK_MIB=str(job.get('peak_mib',12288)))
    else:command_env.update(V10_COMPUTE_DEVICE='cpu',V10_WORKER_CPU_THREADS='8')
    with open(c.ROOT/'logs'/(name+'.log'),'a') as output:
        child=subprocess.Popen(job.get('command') or [c.PYTHON,str(c.ROOT/'code/action10.py'),name],cwd=c.REPO,env=command_env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        identity=q.process_identity(child.pid);baseline={p['pid']:p['memory'] for p in c.read(c.ROOT/'GPU_admissions'/(name+'.json'),{}).get('allowed_foreign',[])};c.save(c.ROOT/'running'/(name+'.json'),dict(guardian=q.process_identity(os.getpid()),child=identity,started_at=started,slot=slot,gpu_uuid=gpu))
        limit=min(started+job['max_seconds'],state['deadline']-1800);lastcheck=0
        while child.poll() is None:
            if gpu and time.time()-lastcheck>=30:
                telemetry=gq.telemetry().get(gpu,{});foreign=[]
                for process in telemetry.get('processes',[]):
                    try:owned=os.getpgid(process['pid'])==identity['pid'] and q.alive(identity)
                    except ProcessLookupError:owned=False
                    if not owned:foreign.append(process['pid'])
                unexpected=[pid for pid in foreign if pid not in baseline];growth=any(p['memory']>baseline.get(p['pid'],p['memory'])+256 for p in telemetry.get('processes',[]) if p['pid'] in baseline)
                if not telemetry.get('known') or unexpected or growth or telemetry.get('free',0)<1024:reason='Unknown telemetry or changed foreign GPU contention; only owned job stopped'
                lastcheck=time.time()
            if c.read(c.ROOT/'control.json',{}).get('stop'):reason='Owned worker stop requested'
            if reason or time.time()>=limit-30:
                reason=reason or 'Per-job ceiling or collection cutoff';stop(identity)
                try:child.wait(timeout=25)
                except subprocess.TimeoutExpired:stop(identity,signal.SIGKILL);child.wait()
                break
            time.sleep(2)
    result=c.read(c.ROOT/'stage_results'/(name+'.json'));success=child.returncode==0 and bool(result) and result.get('outcome')=='completed'
    receipt_path=c.ROOT/'receipts'/(name+'.json')
    c.save(receipt_path,dict(outcome='completed' if success else 'deferred' if reason else 'technical_failed',job=name,returncode=child.returncode,reason=reason,seconds=time.time()-started,gpu_uuid=gpu,GPU_command_seconds=time.time()-started if gpu else 0,result_sha256=c.sha(c.ROOT/'stage_results'/(name+'.json')) if result else None,log=str(c.ROOT/'logs'/(name+'.log')),guardian=q.process_identity(os.getpid()),child=identity))
    if gpu:
        with gq.Ledger() as ledger:
            ledger.data['charges'][str(receipt_path)]=dict(seconds=c.read(receipt_path)['GPU_command_seconds'],receipt=str(receipt_path),sha256=c.sha(receipt_path));ledger.data['reservations'].pop(str(c.ROOT)+':'+name,None)
        os.close(lock_fd)
def condition(job,state):
    if job.get('optional') and time.time()+job['max_seconds']+6*3600>=state['deadline']-1800:return 'DEFERRED','Tier C does not fit the fixed final-evaluation reserve'
    if job.get('qat'):
        selection=c.read(c.ROOT/'smallest_viable_selection.json',{});graph=selection.get('graph') if job['graph']=='SELECTED' else job['graph']
        if not graph:return 'UNAVAILABLE','Smallest viable source group unavailable'
        rows=c.read(c.ROOT/'compression_control_selection.json',{}).get('qat_admission',[])
        if not rows:rows=c.read(c.ROOT/'core_qat_parent_preparation.json',{}).get('qat_admission',[])
        admission=next((x for x in rows if x['graph']==graph and x['precision']==job['precision']),None)
        if not admission:return 'UNAVAILABLE','Required prepared source QAT parent unavailable'
        if not admission['admitted']:return 'SOURCE_GATE_NOT_MET','Unchanged source PTQ point is within the 1pp source gate'
    return None,None
def collect(state,reason):
    for name,row in state['jobs'].items():
        if row['status'] not in TERMINAL:row.update(status='DEFERRED',reason=reason)
    with gq.Ledger() as ledger:
        for key in list(ledger.data['reservations']):
            if key.startswith(str(c.ROOT)+':'):ledger.data['reservations'].pop(key)
    state.update(finished=True,finished_at=time.time(),phase='COLLECTING');c.save(c.ROOT/'state.json',state)
    with open(c.ROOT/'logs/final_collection.log','a') as output:
        result=subprocess.run(c.read(c.ROOT/'manifest.json').get('collect_command') or [c.PYTHON,str(c.ROOT/'code/collect10.py')],env=env(),cwd=c.REPO,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=1500)
    state.update(phase='FINISHED',collection_returncode=result.returncode,finished_at=time.time(),heartbeat=time.time());c.save(c.ROOT/'state.json',state)
    c.save(c.ROOT/'scheduler_status.json',dict(heartbeat=state['heartbeat'],worker=state['worker'],phase='FINISHED',finished=True,deadline=state['deadline'],counts={label:sum(x['status']==label for x in state['jobs'].values()) for label in sorted({x['status'] for x in state['jobs'].values()})},running=[]))
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
        scheduler=manifest.get('scheduler',{});launch=scheduler.get('inherited_launch',time.time());hours=scheduler.get('execution_hours',12);assert 0<hours<=12
        inherited_deadline=scheduler.get('inherited_deadline',launch+hours*3600);assert inherited_deadline<=launch+12*3600
        state=dict(launch=launch,recovery_started=time.time(),deadline=inherited_deadline,collection_cutoff=inherited_deadline-1800,manifest_sha256=c.sha(c.ROOT/'manifest.json'),CPU_slots=[cores[i:i+8] for i in range(0,24,8)],jobs={name:dict(status='PENDING') for name in jobs},phase='RUNNING',finished=False)
        for name,job in jobs.items():
            reused=reusable(job)
            if reused:state['jobs'][name].update(status='COMPLETED',receipt=reused,reason='Reused matching immutable receipt; no new command or GPU charge')
        inherited_active(jobs)
        with gq.Ledger() as ledger:
            initialize_protection(ledger,manifest,c.read(c.ROOT/'GPU_budget_policy.json')['cumulative_cap_seconds'])
    assert state.get('manifest_sha256')==c.sha(c.ROOT/'manifest.json'),'Recovery manifest changed after launch'
    state['worker']=q.process_identity(os.getpid());c.save(c.ROOT/'state.json',state);requested=False;history={u:[] for u in old.GPUS};nextcheck=0
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
                row.update(status={'completed':'COMPLETED','deferred':'DEFERRED','technical_failed':'TECHNICAL_FAILED'}[receipt['outcome']],receipt=str(c.ROOT/'receipts'/(name+'.json')),reason=receipt.get('reason'))
                if row.get('gpu_uuid') and not c.read(c.ROOT/'GPU_admissions'/(name+'.json'),{}).get('allowed_foreign'):history[row['gpu_uuid']]=[];nextcheck=0
                continue
            if not q.alive(ident):
                running=c.read(c.ROOT/'running'/(name+'.json'),{});stop(running.get('child'))
                if row.get('gpu_uuid'):
                    time.sleep(3)
                    key=str(c.ROOT)+':'+name
                    with gq.Ledger() as ledger:
                        if key in ledger.data['reservations']:
                            ceiling=row.get('effective_max_seconds',jobs[name]['max_seconds']);path=c.ROOT/'lost_guardian_receipts'/(name+'.json');c.save(path,dict(outcome='technical_failed',seconds=ceiling,charge='Full reserved GPU ceiling after lost guardian'))
                            ledger.data['charges'][str(path)]=dict(seconds=ceiling,receipt=str(path),sha256=c.sha(path));ledger.data['reservations'].pop(key)
                row.update(status='TECHNICAL_FAILED',reason='Guardian lost; owned child stopped; retained outputs; no duplicate restart');continue
            active[name]=row
        apply_early_readout_gate(state,jobs,manifest)
        stopping=requested or control.get('stop') or now>=state['deadline']-1800
        if stopping:
            for name,row in active.items():stop(row['guardian'])
            if not active:
                collect(state,'Fixed deadline collection cutoff' if now>=state['deadline']-1800 else 'User stop');return
        elif not control.get('pause'):
            inherited=inherited_active(jobs)
            GPU_busy={row['gpu_uuid'] for row in list(active.values())+inherited if row.get('gpu_uuid')}
            max_gpu_jobs=manifest.get('scheduler',{}).get('max_gpu_jobs',2);assert 1<=max_gpu_jobs<=2
            if len(GPU_busy)<max_gpu_jobs and now>=nextcheck:
                for uuid,telemetry in gq.telemetry().items():
                    telemetry['at']=now;history[uuid].append(telemetry);history[uuid]=history[uuid][-3:]
                nextcheck=now+(15 if any(len(h)<3 for h in history.values()) else 600)
            slots={x['slot'] for x in list(active.values())+inherited}
            for name,job in sorted(jobs.items(),key=lambda item:(item[1].get('priority',100),item[0])):
                row=state['jobs'][name]
                if row['status']!='PENDING':continue
                if job.get('reuse_pending_receipt'):
                    try:provider=pending_provider(job)
                    except (AssertionError,ValueError,KeyError,OSError) as error:
                        provider=dict(status='TECHNICAL_FAILED',reason='Completed provider validation failed: '+str(error))
                    if provider:row.update(provider);nextcheck=0
                    else:row['waiting_reason']='Waiting for existing owned provider completion; duplicate launch prohibited'
                    continue
                dependencies=[state['jobs'][d]['status'] for d in job.get('deps',[])]
                if not all(x in TERMINAL for x in dependencies):continue
                if not job.get('soft_dependencies') and any(x!='COMPLETED' for x in dependencies):row.update(status='UNAVAILABLE',reason='Required verification/source dependency did not complete');continue
                if job.get('requires_early_readout_reserve'):
                    gate=state.get('early_readout_cost_gate')
                    if not gate:
                        profile_name=manifest['scheduler']['early_readout_profile_job']
                        if state['jobs'][profile_name]['status'] in TERMINAL:
                            row.update(status='DEFERRED',reason='Required early source readout profile did not complete')
                        else:row['waiting_reason']='Waiting for measured Tier A readout reservation before extras'
                        continue
                    if gate['status']!='RESERVED':row.update(status='DEFERRED',reason=gate['reason']);continue
                if job.get('dynamic_readout_profile'):
                    try:resolved,reason=final_readout_job(name,job,state,jobs,manifest)
                    except (AssertionError,KeyError,ValueError,OSError) as error:
                        resolved=None;reason='Final readout source profile validation failed: '+str(error)
                    if resolved is None:
                        row['waiting_reason']=reason
                        if not GPU_busy:
                            for member in manifest['scheduler']['final_readout_jobs']:
                                if state['jobs'][member]['status']=='PENDING':state['jobs'][member].update(status='DEFERRED',reason=reason)
                        continue
                    job=resolved
                status,reason=condition(job,state)
                if status:row.update(status=status,reason=reason);continue
                reserve=5*3600 if job.get('tier') in ['A','B','C'] else 0
                remaining=cohort_remaining(job,jobs,state,manifest) if job.get('cohort_group') else job['max_seconds']
                if now+remaining+reserve>=state['deadline']-1800:
                    defer_cohort(name,job,jobs,state,manifest,'Complete remaining seed cohort does not fit fixed deadline and final-readout reserve');continue
                if len(slots)>=3 or not cpu_available():continue
                if shutil.disk_usage('/data1/schmalstieg/flooddistill').free<32*2**30:row.update(status='UNAVAILABLE',reason='Data disk below 32GiB headroom');continue
                slot=next(i for i in range(3) if i not in slots)
                gpu=None;GPU_fd=None
                if job.get('device')=='cuda':
                    if len(GPU_busy)>=max_gpu_jobs:continue
                    if job.get('cohort_group'):
                        job=dict(job,cohort_remaining_seconds=remaining)
                    cap=c.read(c.ROOT/'GPU_budget_policy.json')['cumulative_cap_seconds']
                    budget_ok=False
                    with gq.Ledger() as ledger:budget_ok=budget_fits(ledger,job,cap)
                    if not budget_ok:
                        row['waiting_reason']='Shared GPU cap does not admit this conservative command ceiling'
                        if not GPU_busy:
                            defer_cohort(name,job,jobs,state,manifest,'Complete remaining seed cohort exceeds cumulative GPU allowance with final readout protected')
                        continue
                    for uuid in old.GPUS:
                        if uuid in GPU_busy:continue
                        telemetry=gq.telemetry().get(uuid,{})
                        admissible,reason,_=availability(telemetry,dict(peak_mib=job.get('peak_mib',12288),teacher=job.get('teacher',False)),history[uuid],now)
                        with open(c.ROOT/'gpu_availability.jsonl','a') as stream:stream.write(__import__('json').dumps(dict(at=now,job=name,uuid=uuid,admissible=admissible,reason=reason))+'\n')
                        if not admissible:continue
                        GPU_fd=q.lock_fd('/data1/schmalstieg/flooddistill/revision_gpu_'+uuid+'.lock')
                        if GPU_fd is None:continue
                        with gq.Ledger() as ledger:
                            if not availability(gq.telemetry().get(uuid,{}),dict(peak_mib=job.get('peak_mib',12288),teacher=job.get('teacher',False)),history[uuid],time.time())[0] or not budget_fits(ledger,job,cap):
                                os.close(GPU_fd);GPU_fd=None;continue
                            reserve_job(ledger,name,job)
                        gpu=uuid;c.save(c.ROOT/'GPU_admissions'/(name+'.json'),dict(uuid=uuid,at=time.time(),allowed_foreign=telemetry.get('processes',[]),peak_mib=job.get('peak_mib',12288),rule='V9 three-observation exclusive-idle qualification; independent UUID lock; additive cumulative ledger'));break
                    if gpu is None:
                        row['waiting_reason']='Waiting for idle authorized UUID, stable telemetry and lock';continue
                command=[c.PYTHON,str(Path(__file__).resolve()),'guard','--job',name,'--slot',str(slot)]
                if gpu:command += ['--gpu',gpu,'--lock-fd',str(GPU_fd)]
                # Publish dynamic admission before the guardian can read it.
                if job.get('dynamic_readout_profile'):
                    row.update(effective_max_seconds=job['max_seconds'],readout_profile_sha256=job['readout_profile_sha256']);c.save(c.ROOT/'state.json',state)
                with open(c.ROOT/'logs/guardians.log','a') as output:
                    try:child=subprocess.Popen(command,cwd=c.REPO,env=env(),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(GPU_fd,) if gpu else ())
                    except Exception:
                        if gpu:
                            with gq.Ledger() as ledger:undo_reservation(ledger,name)
                            os.close(GPU_fd)
                        raise
                if gpu:os.close(GPU_fd);GPU_busy.add(gpu)
                row.pop('waiting_reason',None);row.update(status='RUNNING',guardian=q.process_identity(child.pid),slot=slot,started_at=now,gpu_uuid=gpu);slots.add(slot);c.save(c.ROOT/'state.json',state)
        state.update(heartbeat=time.time(),phase='STOPPING' if stopping else 'PAUSED' if control.get('pause') else 'RUNNING' if any(x['status']=='RUNNING' for x in state['jobs'].values()) else 'WAITING_GPU');c.save(c.ROOT/'state.json',state)
        with gq.Ledger() as ledger:
            for group,names in manifest.get('scheduler',{}).get('cohorts',{}).items():
                if not any(state['jobs'][n]['status']=='PENDING' for n in names):
                    ledger.data['reservations'].pop(str(c.ROOT)+':cohort:'+group,None)
            GPU_spent=ledger.spend();GPU_reserved=ledger.reserved()
        c.save(c.ROOT/'GPU_budget_status.json',dict(at=time.time(),spent_GPU_hours=GPU_spent/3600,reserved_GPU_hours=GPU_reserved/3600,remaining_GPU_hours=(c.read(c.ROOT/'GPU_budget_policy.json')['cumulative_cap_seconds']-GPU_spent-GPU_reserved)/3600))
        c.save(c.ROOT/'scheduler_status.json',dict(heartbeat=state['heartbeat'],worker=state['worker'],phase=state['phase'],deadline=state['deadline'],remaining_seconds=state['deadline']-time.time(),next_GPU_stability_sample_at=nextcheck,counts={label:sum(x['status']==label for x in state['jobs'].values()) for label in sorted({x['status'] for x in state['jobs'].values()})},running=[name for name,x in state['jobs'].items() if x['status']=='RUNNING']))
        if all(row['status'] in TERMINAL for row in state['jobs'].values()):collect(state,'Bounded queue exhausted');return
        time.sleep(5)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','guard','status','pause','resume','stop']);parser.add_argument('--job');parser.add_argument('--slot',type=int);parser.add_argument('--gpu');parser.add_argument('--lock-fd',type=int);args=parser.parse_args()
    if args.action=='run':run()
    elif args.action=='guard':guard(args.job,args.slot,args.gpu,args.lock_fd)
    elif args.action=='status':
        state=c.read(c.ROOT/'state.json',{});print(__import__('json').dumps(c.read(c.ROOT/'scheduler_status.json',state),indent=2));print('worker_alive',q.alive(state.get('worker')))
    else:
        control=c.read(c.ROOT/'control.json',{});control['pause']=args.action=='pause';control['stop']=args.action=='stop';c.save(c.ROOT/'control.json',control)
if __name__=='__main__':main()
