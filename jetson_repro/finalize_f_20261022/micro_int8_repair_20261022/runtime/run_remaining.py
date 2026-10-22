#!/usr/bin/env python3
"""Execute the finite declared Jetson matrix, with immutable inputs and durable resume."""
import argparse,hashlib,json,os,subprocess,sys,time,platform,fcntl
from pathlib import Path
from validate_results import benchmark_receipts_valid
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def command(argv,log,out):
    log=Path(log);i=0
    while log.exists():i+=1;log=log.with_name(Path(log).stem.split('.resume')[0]+'.resume'+str(i)+'.log')
    with log.open('x') as stream:
        proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            save(out/'HEARTBEAT.json',dict(controller_pid=os.getpid(),child_pid=proc.pid,argv=argv,log=str(log.relative_to(out)),updated=time.time()));time.sleep(2)
        return proc.returncode
def archive_partial(path):
    path=Path(path)
    if path.exists():path.rename(path.with_name(path.name+'.interrupted.'+str(time.time_ns())))
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--sturm-root',type=Path,default=Path.home()/'data/STURM-Flood/Dataset');p.add_argument('--wf-root',type=Path,default=Path.home()/'data/WorldFloodsv2');a=p.parse_args()
    if a.attempt in ('.','..') or Path(a.attempt).name!=a.attempt:p.error('attempt must be one new directory name')
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        wanted,name=line.split(maxsplit=1)
        if sha(ROOT/name)!=wanted:raise RuntimeError('Package hash mismatch '+name)
    package_lock=(ROOT/'.controller.lock').open('a');fcntl.flock(package_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    import tensorrt as trt
    import pycuda.driver as cuda
    if not trt.__version__.startswith('8.4.'):raise RuntimeError('Target requires TensorRT8.4')
    if not a.sturm_root.is_dir() or not a.wf_root.is_dir():raise FileNotFoundError('Supply the existing complete STURM and WorldFloods test roots')
    out=ROOT/'results'/a.attempt;out.mkdir(parents=True,exist_ok=a.resume)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(package_sha256=sha(ROOT/'SHA256SUMS'),asset_manifest_sha256=sha(ROOT/'ASSET_SHA256SUMS'),sturm_root=str(a.sturm_root.resolve()),wf_root=str(a.wf_root.resolve()))
    if (out/'IDENTITY.json').exists():assert json.loads((out/'IDENTITY.json').read_text())==identity
    else:save(out/'IDENTITY.json',identity)
    if a.resume and (out/'HEARTBEAT.json').exists():
        previous=json.loads((out/'HEARTBEAT.json').read_text());pid=previous.get('child_pid');proc=Path('/proc')/str(pid)/'cmdline'
        if pid and proc.exists():
            argv=proc.read_bytes().rstrip(b'\0').decode().split('\0')
            if argv==previous.get('argv'):raise RuntimeError('The previously owned child is still running; wait for it to finish before resume')
    if not (out/'environment.json').exists():
        env=dict(python=sys.version,platform=platform.platform(),tensorrt=trt.__version__,torch=torch.__version__,started=time.time(),clock_changes=False,power_mode_changes=False)
        for name,argv in [('nvpmodel',['nvpmodel','-q']),('clocks',['jetson_clocks','--show']),('l4t',['cat','/etc/nv_tegra_release'])]:
            try:
                r=subprocess.run(argv,capture_output=True,text=True);env[name]=dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
            except OSError as error:env[name]=dict(error=str(error))
        save(out/'environment.json',env)
    manifest=json.loads((ROOT/'RUN_MANIFEST.json').read_text());states=json.loads((out/'STATUS.json').read_text()) if (out/'STATUS.json').exists() else []
    states={s['id']:s for s in states}
    def record(state):states[state['id']]=state;save(out/'STATUS.json',list(states.values()))
    if not (out/'PREFLIGHT_STATUS.json').exists():
        rc=command([sys.executable,str(ROOT/'runtime/preflight.py'),'--sturm-root',str(a.sturm_root),'--wf-root',str(a.wf_root),'--out',str(out/'PREFLIGHT.json')],out/'preflight.log',out)
        save(out/'PREFLIGHT_STATUS.json',dict(returncode=rc,finished=time.time()))
    if json.loads((out/'PREFLIGHT_STATUS.json').read_text())['returncode']!=0:
        raise RuntimeError('Preflight failed; no model work admitted. See preflight.log and preserve the failed attempt')
    for model in manifest['engines']:
        name=model['id'];dest=out/name
        if name in states and states[name]['status']!='STARTED':continue # Preserve completed failures; no automatic scientific retries.
        while (out/'PAUSE').exists() and not (out/'STOP').exists():save(out/'HEARTBEAT.json',dict(controller_pid=os.getpid(),state='PAUSED',updated=time.time()));time.sleep(2)
        if (out/'STOP').exists():save(out/'STOPPED.json',dict(at=time.time(),next_model=name));return 2
        dest.mkdir(exist_ok=True);state=dict(id=name,architecture=model['architecture'],seed=model['seed'],precision=model['precision'],method=model['method'],status='STARTED',source_admission=model.get('source_admission'));record(state)
        if model.get('reuse'):
            engine=ROOT/model['reuse']['engine'];assert sha(engine)==model['reuse']['engine_sha256']
            reused_build=json.loads((ROOT/model['reuse']['build_receipt']).read_text());assert reused_build['engine_sha256']==model['reuse']['engine_sha256']
            if model['precision']=='pure' and not reused_build['precision_audit']['full_integer_tensor_path']:
                state.update(status='FAILED_INTEGER_TENSOR_CERTIFICATE');record(state);continue
            state['reuse']=model['reuse'];save(dest/'REUSE.json',model['reuse']);onlydry=True
        else:
            engine=dest/'model.engine';build=dest/'build.json'
            if build.exists():
                b=json.loads(build.read_text());code=0 if b.get('status')=='built' and engine.exists() and b.get('engine_sha256')==sha(engine) else 1
            else:
                archive_partial(engine)
                cmd=[sys.executable,str(ROOT/'runtime/integer_tools.py'),'build','--model',str(ROOT/model['onnx']),'--engine',str(engine),'--receipt',str(build),'--mode',model['mode']]
                if model.get('fp32_islands'):cmd+=['--fp32-islands',str(ROOT/model['fp32_islands'])]
                code=command(cmd,dest/'build.log',out)
            if code or not build.exists() or json.loads(build.read_text()).get('status')!='built':
                state.update(status='FAILED_BUILD',returncode=code);record(state);continue
            if model['precision']=='pure':
                from repair_policy import integer_certificate
                try:integer_certificate(json.loads(build.read_text()),json.loads(Path(str(engine)+'.layers.json').read_text()),model)
                except (OSError,KeyError,ValueError) as error:
                    state.update(status='FAILED_INTEGER_TENSOR_CERTIFICATE',reason=str(error));record(state);continue
            if model.get('fp32_islands'):
                from precision_islands import validate_certificate
                try:
                    validate_certificate(json.loads(build.read_text()), model, ROOT, json.loads(Path(str(engine)+'.layers.json').read_text()))
                except (OSError, KeyError, ValueError) as error:
                    state.update(status='FAILED_FP32_ISLAND_CERTIFICATE',reason=str(error));record(state);continue
            onlydry=False
        cmd=[sys.executable,str(ROOT/'runtime/evaluate_native.py'),'--engine',str(engine),'--out',str(dest/'evaluation'),'--sturm-root',str(a.sturm_root),'--wf-root',str(a.wf_root),'--resume']
        if model.get('io'):cmd+=['--io',str(ROOT/model['io'])]
        if onlydry:cmd+=['--only-dry']
        elif model['precision']=='fp32':cmd+=['--source-reference',str(ROOT/model['source_reference'])]
        elif model['precision']=='pure':
            baseline=out/(model['architecture']+'_s'+str(model['seed'])+'_fp32')/'evaluation/s11val.json'
            if not baseline.is_file() or not (baseline.parent/'SOURCE_WIRING.json').is_file() or not json.loads((baseline.parent/'SOURCE_WIRING.json').read_text())['passed']:
                state.update(status='FAILED_FP32_BASELINE');record(state);continue
            cmd+=['--quant-source-reference',str(baseline)]
        code=command(cmd,dest/'evaluation.log',out)
        if code:state.update(status='FAILED_EVALUATION',returncode=code);record(state);continue
        if not onlydry:
            for script,folder,receipt,label in [('bench_trtexec.py','benchmark','bench.json','profile'),('bench_power_steady.py','steady_power','power_steady.json','steady')]:
                if (dest/folder/receipt).exists():rc=0
                else:
                    archive_partial(dest/folder)
                    rc=command([sys.executable,str(ROOT/'runtime'/script),'--engine',str(engine),'--out',str(dest/folder)],dest/(folder+'.log'),out)
                state[label+'_returncode']=rc
            try:
                profile=json.loads((dest/'benchmark/bench.json').read_text());steady=json.loads((dest/'steady_power/power_steady.json').read_text())
                benchmark_ok=benchmark_receipts_valid(profile,steady,state)
            except (OSError,KeyError,ValueError):benchmark_ok=False
            if not benchmark_ok:
                state.update(status='FAILED_BENCHMARK');record(state);continue
        state['status']='MEASURED_PENDING_ACCEPTANCE';record(state)
    save(out/'ATTEMPTS_RESOLVED.json',dict(status='ALL_DECLARED_ATTEMPTS_RESOLVED',states=list(states.values()),unsupported_slots=manifest['unsupported_slots'],finished=time.time(),native_acceptance='Collector computes same-seed FP32 differences; failed seeds retained'))
    return subprocess.run([sys.executable,str(ROOT/'runtime/collect_native.py'),'--attempt',a.attempt]).returncode
if __name__=='__main__':raise SystemExit(main())
