"""Finite detached UUID-locked worker. GPU lock -> ledger mutex is the sole lock order."""
import argparse,csv,fcntl,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import common_v9 as c

LEDGER=c.ROOT.parent/'.v9_shared_gpu_ledger.json'
MUTEX=LEDGER.with_suffix('.lock')
CONTROL=c.ROOT/'control.json'
STATE=c.ROOT/'state.json'
TERMINAL={'COMPLETED','TECHNICAL_FAILED','UNAVAILABLE','DEFERRED_BUDGET','DEFERRED_DEADLINE','DEFERRED_BUSY','STOPPED'}

def stamp(t):
    d=datetime.fromtimestamp(t,timezone.utc)
    return dict(UTC=d.isoformat(),Europe_Berlin=d.astimezone(ZoneInfo('Europe/Berlin')).isoformat())

def identity(pid):
    try:return dict(pid=int(pid),start_ticks=Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError,IndexError):return None

def alive(ident):
    return bool(ident) and identity(ident['pid'])==ident

def own_stop(ident):
    if alive(ident):os.killpg(ident['pid'],signal.SIGTERM)

def telemetry():
    """Read identities and memory only; never record foreign command lines."""
    query=['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.free,memory.total,compute_mode','--format=csv,noheader,nounits']
    try:
        raw=subprocess.run(query,capture_output=True,text=True,timeout=12,check=True).stdout
        procs=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12,check=True).stdout
        rows={}
        for line in raw.splitlines():
            v=[x.strip() for x in line.split(',')]
            rows[v[0]]=dict(uuid=v[0],util=float(v[1]),free=float(v[2]),total=float(v[3]),mode=v[4],processes=[],known=True,authorized=v[0] in c.GPUS)
        for line in procs.splitlines():
            if not line.strip():continue
            u,p,m=[x.strip() for x in line.split(',')];rows[u]['processes'].append(dict(pid=int(p),memory=float(m)))
        return rows
    except (OSError,subprocess.SubprocessError,ValueError,KeyError):return {u:dict(uuid=u,known=False,authorized=True,reason='unknown or unreadable telemetry') for u in c.GPUS}

def availability(row,job,history,now):
    if not row.get('known') or not row.get('authorized'):return False,'unknown telemetry or unallocated GPU',False
    if row['mode']!='Default':return False,'non-Default compute mode; MPS/exclusive ownership not admitted',False
    peak=job['peak_mib'];required=peak+max(2048,.2*peak)
    if row['free']<required:return False,'insufficient memory',False
    if not row['processes']:
        if row['util']>=5:return False,'utilization >=5%',False
        eligible=[h for h in history if not h['processes'] and h['util']<5 and h['free']>=required]
        if len(eligible)>=3 and now-eligible[-3]['at']>=29:return True,'idle three observations',False
        return False,'idle qualification pending',False
    if peak>1536 or job.get('measurement') or job.get('teacher'):return False,'foreign compute; co-location forbidden for this job',False
    eligible=[h for h in history if h['processes'] and h['free']>=required and h.get('mode')=='Default']
    if len(eligible)<3 or now-eligible[-3]['at']<1200:return False,'foreign compute; twenty-minute stability pending',False
    last=eligible[-3:];pids=[set(p['pid'] for p in h['processes']) for h in last]
    sums=[sum(p['memory'] for p in h['processes']) for h in last]
    if pids[0]!=pids[1] or pids[1]!=pids[2] or max(sums)-min(sums)>=256:return False,'foreign memory unstable',False
    return True,'authorized stable-memory co-location',True

class Ledger:
    def __enter__(self):
        self.fd=open(MUTEX,'a+');fcntl.flock(self.fd,fcntl.LOCK_EX)
        self.data=c.read(LEDGER,dict(version=1,historical_seconds=93490.88168258697,charges={},reservations={}))
        # A uniquely named imported receipt is charged once, using exact 14.6s.
        self.data['charges'].setdefault('claude_memory_probe_20261013',dict(seconds=14.6,receipt=str(c.PROBE/'gpu_vram_check.json'),sha256=c.sha(c.PROBE/'gpu_vram_check.json')))
        # Reconcile all V9 command receipts, including failed attempts and peers.
        for p in c.ROOT.parent.glob('research_v9_*/command_receipts/*.json'):
            r=c.read(p)
            self.data['charges'].setdefault(str(p),dict(seconds=r['seconds'],receipt=str(p),sha256=c.sha(p)))
        return self
    def spend(self):return self.data['historical_seconds']+sum(x['seconds'] for x in self.data['charges'].values())
    def reserved(self):return sum(x['remaining_seconds'] for x in self.data['reservations'].values())
    def __exit__(self,*args):
        c.save(LEDGER,self.data);fcntl.flock(self.fd,fcntl.LOCK_UN);self.fd.close()

def reserve_block(ledger,run,block,seconds,protected,cap=108000):
    key=run+':'+block;protectedkey=run+':final'
    add=max(0,seconds-ledger.data['reservations'][key]['remaining_seconds']) if key in ledger.data['reservations'] else seconds
    guard=0 if protectedkey in ledger.data['reservations'] else protected
    owncharges=sum(v['seconds'] for k,v in ledger.data['charges'].items() if k.startswith(run+'/'))
    ownreserve=sum(v['remaining_seconds'] for k,v in ledger.data['reservations'].items() if k.startswith(run+':'))
    if ledger.spend()+ledger.reserved()+add+guard>cap or owncharges+ownreserve+add+guard>14400:return False
    if add:ledger.data['reservations'][key]=dict(run=run,block=block,remaining_seconds=seconds,created=c.q.now())
    if guard:ledger.data['reservations'][protectedkey]=dict(run=run,block='final',remaining_seconds=protected,created=c.q.now())
    return True

def run_guardian(args):
    job=c.read(c.ROOT/'manifest.json')['jobs'][args.job];receipt=c.ROOT/'command_receipts'/f'{args.job}.json';launch=c.read(c.ROOT/'job_launches'/f'{args.job}.json')
    for _ in range(100):
        if launch.get('guardian'):break
        time.sleep(.05);launch=c.read(c.ROOT/'job_launches'/f'{args.job}.json')
    cpu=job.get('device')=='cpu'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if cpu else args.gpu,V9_COLOCATION='1' if launch['colocation'] else '0',V9_PEAK_MIB=str(job['peak_mib']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH='/home/schmalstieg/repos/_deps/efficientvit',MPLCONFIGDIR=str(c.ROOT/'mpl_cache'))
    begin=time.monotonic();wall=time.time();status='TECHNICAL_FAILED';reason=None;child=None
    try:
        with open(c.ROOT/'logs'/f'{args.job}.log','w',buffering=1) as log:
            child=subprocess.Popen(job['command'],cwd=c.REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ident=identity(child.pid);launch.update(child=ident,guardian=identity(os.getpid()),started=c.q.now());c.save(c.ROOT/'job_launches'/f'{args.job}.json',launch)
            nextcheck=begin;foreign0=sum(x['memory'] for x in launch['admitted_telemetry']['processes'])
            while child.poll() is None:
                ctl=c.read(CONTROL,{'action':'resume'})
                deadline=min(launch['hard_deadline'],wall+job['max_seconds'])
                if ctl['action']=='stop' or time.time()>=deadline-15:
                    reason='owned stop' if ctl['action']=='stop' else 'command budget or wall timeout';own_stop(ident);break
                if not cpu and time.monotonic()>=nextcheck:
                    row=telemetry().get(args.gpu,{});foreign=[x for x in row.get('processes',[]) if x['pid']!=child.pid]
                    foreign_mem=sum(x['memory'] for x in foreign)
                    unsafe=not row.get('known') or (launch['colocation'] and (row['free']<1024 or foreign_mem>foreign0)) or (not launch['colocation'] and bool(foreign))
                    if unsafe:reason='telemetry/foreign contention: only owned process stopped';own_stop(ident);break
                    nextcheck=time.monotonic()+30
                time.sleep(1)
            if reason and child.poll() is None:
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if alive(ident):os.killpg(child.pid,signal.SIGKILL)
            code=child.wait(timeout=5)
            if code==0 and reason is None:
                if all(Path(p).exists() for p in job['outputs']):status='COMPLETED'
                else:reason='exit zero but required completion artifact absent'
            elif reason is None:reason=f'child exit {code}'
    except BaseException as exc:reason=repr(exc);code=None
    finally:
        elapsed=time.monotonic()-begin;seconds=0 if cpu else elapsed
        result=dict(job=args.job,block=job['block'],status=status,reason=reason,exit_code=code,seconds=seconds,cpu_seconds=elapsed if cpu else None,gpu_uuid=None if cpu else args.gpu,started=stamp(wall),finished=c.q.now(),contaminated=bool(reason and 'contention' in reason),outputs={p:c.sha(p) for p in job['outputs'] if Path(p).is_file()})
        c.save(receipt,result)
        with Ledger() as ledger:
            ledger.data['charges'][str(receipt)]=dict(seconds=seconds,receipt=str(receipt),sha256=c.sha(receipt))
            key=str(c.ROOT)+':'+job['block']
            if not cpu and key in ledger.data['reservations']:ledger.data['reservations'][key]['remaining_seconds']=max(0,ledger.data['reservations'][key]['remaining_seconds']-job['max_seconds'])
        os.close(args.lock_fd)

def runtime_deadline(state):
    return min(state['absolute_deadline'],state['first_gpu']+28800 if state.get('first_gpu') else state['wait_deadline'])

def status(state,manifest):
    with Ledger() as l:
        b=dict(verified_gpu_hours=l.spend()/3600,reserved_gpu_hours=l.reserved()/3600,remaining_unreserved_gpu_hours=max(0,(108000-l.spend()-l.reserved())/3600))
    now=time.time();out=dict(state=state['phase'],heartbeat=stamp(now),pid=identity(os.getpid()),run_directory=str(c.ROOT),authorized_gpu_uuids=c.GPUS,
        last_gpu_check=stamp(state['last_scan']) if state.get('last_scan') else None,next_gpu_check=stamp(state['next_scan']),
        launch=stamp(state['launch']),first_gpu_command=stamp(state['first_gpu']) if state.get('first_gpu') else None,wait_deadline=stamp(state['wait_deadline']),execution_deadline=stamp(state['first_gpu']+28800) if state.get('first_gpu') else None,absolute_deadline=stamp(state['absolute_deadline']),
        budget=b,active=state['active'],jobs=state['jobs'],completed_receipts=[str(c.ROOT/'command_receipts'/f'{k}.json') for k,v in state['jobs'].items() if v=='COMPLETED'],wait_seconds=(state.get('first_gpu') or now)-state['launch'])
    c.save(c.ROOT/'scheduler_status.json',out);c.save(STATE,state)

def collect(state):
    with Ledger() as l:
        for key in list(l.data['reservations']):
            if key.startswith(str(c.ROOT)+':'):del l.data['reservations'][key]
    state['phase']='COLLECTING';status(state,c.read(c.ROOT/'manifest.json'))
    with open(c.ROOT/'logs/collection.log','w') as f:
        subprocess.run([c.PYTHON,str(c.ROOT/'collect_v9.py')],cwd=c.REPO,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT,timeout=550)
    state['phase']='COMPLETED' if all(v=='COMPLETED' for v in state['jobs'].values()) else 'FINISHED_WITH_DEFERRALS'
    state['finished']=time.time()
    with Ledger() as l:
        for key in list(l.data['reservations']):
            if key.startswith(str(c.ROOT)+':'):del l.data['reservations'][key]
    status(state,c.read(c.ROOT/'manifest.json'))

def run():
    manifest=c.read(c.ROOT/'manifest.json');config=c.read(c.ROOT/'scheduler_config.json');state=c.read(STATE)
    for path,digest in manifest.get('frozen_files',{}).items():
        if c.sha(path)!=digest:raise RuntimeError('Frozen input changed before launch/resume: '+path)
    # A singleton worker lock is unrelated to either GPU lock.
    singleton=open(c.ROOT/'worker.lock','a+')
    try:fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Worker already running')
    if state and state.get('finished'):raise SystemExit('Finite run already collected; deadlines are not reset')
    if not state:
        now=time.time();state=dict(launch=now,wait_deadline=now+28800,absolute_deadline=now+57600,first_gpu=None,next_scan=now,last_scan=None,phase='READY',active={},jobs={k:'READY' for k in manifest['jobs']})
    c.save(STATE,state)
    for name in manifest['jobs']:
        receipt=c.read(c.ROOT/'command_receipts'/f'{name}.json')
        launch=c.read(c.ROOT/'job_launches'/f'{name}.json',{})
        if receipt:
            for path,h in receipt.get('outputs',{}).items():
                if not Path(path).exists() or c.sha(path)!=h:raise RuntimeError('Completed output hash changed: '+path)
            state['jobs'][name]=receipt['status'];state['active'].pop(name,None)
        elif alive(launch.get('guardian')) or alive(launch.get('child')):
            state['active'][name]=dict(gpu_uuid=launch['gpu_uuid'],guardian=launch.get('guardian'));state['jobs'][name]='RUNNING'
    c.save(c.ROOT/'launch_receipt.json',dict(worker=identity(os.getpid()),launch=stamp(state['launch']),manifest_sha256=c.sha(c.ROOT/'manifest.json'),command='scheduler_v9.py run',detached=True))
    histories={u:[] for u in c.GPUS};nextmono=time.monotonic()+max(0,state['next_scan']-time.time());lastheartbeat=0
    while True:
        now=time.time();changed=False
        for name in list(state['active']):
            rp=c.ROOT/'command_receipts'/f'{name}.json';launch=c.read(c.ROOT/'job_launches'/f'{name}.json',{})
            if rp.exists():
                r=c.read(rp);state['jobs'][name]=r['status'];del state['active'][name];changed=True
            elif not alive(launch.get('guardian')) and not alive(launch.get('child')):
                # Never guess zero elapsed spend after an interrupted guardian.
                job=manifest['jobs'][name];c.save(rp,dict(job=name,block=job['block'],status='TECHNICAL_FAILED',reason='guardian lost; conservatively charged entire reserved GPU command ceiling',seconds=0 if job.get('device')=='cpu' else job['max_seconds'],gpu_uuid=launch.get('gpu_uuid'),outputs={}))
                state['jobs'][name]='TECHNICAL_FAILED';del state['active'][name];changed=True
        ctl=c.read(CONTROL,{'action':'resume'})['action']
        if ctl=='stop' or now>=runtime_deadline(state)-600:
            c.save(CONTROL,dict(action='stop',at=c.q.now()))
            if state['active']:state['phase']='STOPPING'
            else:
                reason='STOPPED' if ctl=='stop' else ('DEFERRED_BUSY' if not state['first_gpu'] else 'DEFERRED_DEADLINE')
                for name,v in state['jobs'].items():
                    if v not in TERMINAL:state['jobs'][name]=reason
                collect(state);return
        elif ctl=='pause':state['phase']='PAUSED'
        else:
            pending=[]
            for name,job in manifest['jobs'].items():
                if state['jobs'][name] in TERMINAL or name in state['active']:continue
                if job.get('conditional_branch'):
                    admission=c.read(c.ROOT/'conditional_admission.json')
                    if admission:
                        branch=next(x for x in admission['branches'] if x['name']==job['conditional_branch'])
                        if branch['status']!='READY':state['jobs'][name]=branch['status'];continue
                deps=[state['jobs'][d] for d in job.get('dependencies',[])]
                if job.get('dependency_policy')=='settled':
                    if not all(d in TERMINAL for d in deps):state['jobs'][name]='WAITING_DEPENDENCY';continue
                    pending.append((name,job));state['jobs'][name]='READY';continue
                if any(d in TERMINAL and d!='COMPLETED' for d in deps):state['jobs'][name]='UNAVAILABLE';continue
                if not all(d=='COMPLETED' for d in deps):state['jobs'][name]='WAITING_DEPENDENCY';continue
                pending.append((name,job));state['jobs'][name]='READY'
            if not pending and not state['active']:
                if all(v in TERMINAL for v in state['jobs'].values()):collect(state);return
            # CPU work starts independently and never sets the first-GPU clock.
            for name,job in list(pending):
                if job.get('device')!='cpu':continue
                if sum(v['gpu_uuid']=='CPU' for v in state['active'].values())>=1:continue
                if now+job['max_seconds']+615>=runtime_deadline(state):state['jobs'][name]='DEFERRED_DEADLINE';pending.remove((name,job));continue
                fd=os.open(c.ROOT/f'{name}.cpu.lock',os.O_CREAT|os.O_RDWR,0o664);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                launch=dict(job=name,gpu_uuid='CPU',colocation=False,admitted_telemetry={'processes':[]},hard_deadline=runtime_deadline(state)-600,guardian=None,child=None,at=c.q.now())
                c.save(c.ROOT/'job_launches'/f'{name}.json',launch)
                p=subprocess.Popen([c.PYTHON,str(Path(__file__)),'guardian','--job',name,'--gpu','CPU','--lock-fd',str(fd)],cwd=c.REPO,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,pass_fds=(fd,))
                launch['guardian']=identity(p.pid);c.save(c.ROOT/'job_launches'/f'{name}.json',launch);os.close(fd)
                state['active'][name]=dict(gpu_uuid='CPU',guardian=launch['guardian']);state['jobs'][name]='RUNNING';pending.remove((name,job))
            pending=[(name,job) for name,job in pending if job.get('device')!='cpu']
            if pending and (time.monotonic()>=nextmono or changed):
                scanstart=time.monotonic();state['last_scan']=time.time();rows=telemetry()
                busy={x['gpu_uuid'] for x in state['active'].values()}
                for u in c.GPUS:
                    row=rows.get(u,dict(uuid=u,known=False));row['at']=time.time();histories[u].append(row);histories[u]=histories[u][-10:]
                # Qualify apparently idle devices only; active heartbeats continue.
                potential=[u for u in c.GPUS if u not in busy and rows.get(u,{}).get('known') and not rows[u]['processes'] and rows[u]['util']<5]
                for delay in [15,15]:
                    if not potential:break
                    end=time.monotonic()+delay
                    while time.monotonic()<end:
                        if c.read(CONTROL,{'action':'resume'})['action']!='resume':potential=[];break
                        if time.monotonic()-lastheartbeat>=config['heartbeat_seconds']:status(state,manifest);lastheartbeat=time.monotonic()
                        time.sleep(1)
                    fresh=telemetry()
                    for u in potential:
                        if u in fresh:fresh[u]['at']=time.time();histories[u].append(fresh[u]);rows[u]=fresh[u]
                eligible_order=sorted(c.GPUS,key=lambda u:bool(rows.get(u,{}).get('processes')))
                for name,job in pending:
                    if sum(v['gpu_uuid']!='CPU' for v in state['active'].values())>=2:break
                    if now+job['max_seconds']+615>=runtime_deadline(state):state['jobs'][name]='DEFERRED_DEADLINE';continue
                    for u in eligible_order:
                        if u in busy:continue
                        row=rows.get(u,{});ok,why,coloc=availability(row,job,histories[u],time.time())
                        with open(c.ROOT/'gpu_availability.jsonl','a') as f:f.write(json.dumps(dict(at=c.q.now(),gpu_uuid=u,job=name,decision=ok,reason=why,telemetry=row,required_mib=job['peak_mib']+max(2048,.2*job['peak_mib'])))+'\n')
                        if not ok:continue
                        fd=os.open('/data1/schmalstieg/flooddistill/revision_gpu_'+u+'.lock',os.O_CREAT|os.O_RDWR,0o664)
                        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:os.close(fd);continue
                        admitted=False
                        try:
                            with Ledger() as l:
                                fresh=telemetry().get(u,{});recheck,_,recoloc=availability(fresh,job,histories[u],time.time())
                                if not recheck or recoloc!=coloc:continue
                                remaining=sum(x['max_seconds'] for k,x in manifest['jobs'].items() if x['block']==job['block'] and state['jobs'][k] not in TERMINAL and k not in state['active'])
                                if not reserve_block(l,str(c.ROOT),job['block'],remaining,manifest['protected_final_seconds']):
                                    # Existing foreign reservations can expire; wait rather than cancel.
                                    if any(v['remaining_seconds']>0 and (v['run']!=str(c.ROOT) or (v['block'] not in [job['block'],'final'])) for v in l.data['reservations'].values()):state['jobs'][name]='WAITING_GPU';continue
                                    for k,x in manifest['jobs'].items():
                                        if x['block']==job['block'] and state['jobs'][k] not in TERMINAL and k not in state['active']:state['jobs'][k]='DEFERRED_BUDGET'
                                    continue
                                if state['first_gpu'] is None:state['first_gpu']=time.time();c.save(STATE,state)
                                launch=dict(job=name,gpu_uuid=u,colocation=coloc,admitted_telemetry=fresh,hard_deadline=min(runtime_deadline(state)-600,time.time()+job['max_seconds']),guardian=None,child=None,at=c.q.now())
                                c.save(c.ROOT/'job_launches'/f'{name}.json',launch)
                                p=subprocess.Popen([c.PYTHON,str(Path(__file__)),'guardian','--job',name,'--gpu',u,'--lock-fd',str(fd)],cwd=c.REPO,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,pass_fds=(fd,))
                                launch['guardian']=identity(p.pid);c.save(c.ROOT/'job_launches'/f'{name}.json',launch)
                                state['active'][name]=dict(gpu_uuid=u,guardian=launch['guardian']);state['jobs'][name]='RUNNING';busy.add(u);admitted=True
                        finally:os.close(fd)
                        if admitted:break
                # Poll from previous scan start; skip missed ticks without bursts.
                nextmono=scanstart+config['gpu_poll_interval_seconds']
                if nextmono<=time.monotonic():nextmono=time.monotonic()+config['gpu_poll_interval_seconds']
                state['next_scan']=time.time()+max(0,nextmono-time.monotonic())
            state['phase']='RUNNING' if state['active'] else ('WAITING_GPU' if pending else 'WAITING_DEPENDENCY')
        if time.monotonic()-lastheartbeat>=config['heartbeat_seconds'] or changed:status(state,manifest);lastheartbeat=time.monotonic()
        time.sleep(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','guardian','status','pause','resume','stop']);p.add_argument('--job');p.add_argument('--gpu');p.add_argument('--lock-fd',type=int);args=p.parse_args()
    if args.action=='guardian':run_guardian(args)
    elif args.action=='run':run()
    elif args.action=='status':print(json.dumps(c.read(c.ROOT/'scheduler_status.json',{'state':'not launched'}),indent=2))
    else:
        c.save(CONTROL,dict(action=args.action,at=c.q.now()));print(args.action+' requested; existing deadlines and caps preserved')
        if args.action=='resume':
            st=c.read(c.ROOT/'scheduler_status.json',{})
            if not alive(st.get('pid')):subprocess.Popen([c.PYTHON,str(Path(__file__)),'run'],cwd=c.REPO,stdout=open(c.ROOT/'logs/worker.log','a'),stderr=subprocess.STDOUT,start_new_session=True)

if __name__=='__main__':main()
