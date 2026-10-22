#!/usr/bin/env python3
"""Prepare a separate four-seed R4 QAT series using the existing 128-update recipe.

Preparation is read-only except for its new output folder. --execute is explicit:
finalize_f prohibits new component experiments, so this task prepares but does
not run the series. Existing PTQ results and their frozen failures remain intact.
"""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
REPO=Path(__file__).resolve().parents[2]
F=REPO/'experiments/revision/controlled512_v2_20261016T213414Z/revision_finalize_20261019T141008Z'
sys.path.insert(0,str(F/'code'))
import experiment_quant as q
x=q.x

def prepare(out):
    q.verify(x.read(F/'deployment/quant_author_tools/PROTOCOL.json'))
    parent=x.read(F/'deployment/quant_author_tools/PARENTS.json')
    entries=[e for e in parent['entries'] if e['architecture']=='R4']
    assert sorted(e['seed'] for e in entries)==[42,64,91,107]
    plan=dict(status='PREPARED_NOT_RUN',authority=x.pin(REPO.parent/'MajorRev/prompts/finalize_f.md'),
        inherited_protocol=x.pin(F/'deployment/quant_author_tools/PROTOCOL.json'),entries=entries,
        code=[x.pin(Path(__file__)),x.pin(F/'code/experiment_quant.py'),x.pin(F/'code/experiment_quant_author_tools.py')],
        comparison='All four QAT repeats vs their own float parent and retained PTQ; never report a hybrid PTQ/QAT average as a matched QAT study',
        updates_per_seed=128,total_updates=512,training_status='NOT_RUN',
        selection='final update only, existing source gate; no extra LR/objective/seed search',
        frozen_accuracy_tolerance=x.read(F/'deployment/quant_author_tools/PROTOCOL.json')['accuracy_tolerance'],
        export_gate='All89 float semantic/optimized parity at atol=rtol=.001 before training; QAT full-source parity required before device admission',
        evaluation='All four seeds, source val then S11 test/Bolivia/STURM512/WF test and dry sets; no 97-event panel reread or tuning')
    out.mkdir(parents=True,exist_ok=True);x.write(out/'PLAN.json',plan,True);return plan

def one(out,seed):
    plan=x.read(out/'PLAN.json')
    for p in plan['code']+[plan['inherited_protocol']]:assert x.sha(p['path'])==p['sha256']
    protocol=x.read(plan['inherited_protocol']['path']);q.verify(protocol)
    entry=next(e for e in plan['entries'] if e['seed']==seed);cfg=x.read(entry['config']);assert x.sha(entry['config'])==entry['config_sha256']
    target=out/f's{seed}';target.mkdir(exist_ok=False)
    torch,model,ji,ds,inv,ev=q.runtime(cfg)
    import onnx
    from onnxsim import simplify
    from quantization.onnx_qdq import export_float_onnx
    floatpath=target/'float.onnx';export_float_onnx(model,torch.zeros(1,6,512,512),str(floatpath),opset=13,static=True)
    g,ok=simplify(onnx.load(floatpath));assert ok;g.ir_version=min(g.ir_version,10);onnx.save(g,floatpath)
    semantic,parent,sp=q.metric_rows(ji.session(floatpath,disable_optimization=True),ds,inv,ev,model)
    optimized,_,op=q.metric_rows(ji.session(floatpath),ds,inv,ev,model)
    passed=all(r['strict_logit_parity_pass'] for r in sp+op)
    x.write(target/'FLOAT_PARITY.json',dict(passed=passed,semantic=sp,optimized=op,float=x.pin(floatpath)),True)
    if not passed:raise RuntimeError('Float full-source parity failed; QAT blocked')
    result=q.qat(model,protocol,target,q.streaming_samples(protocol),torch,ji,ds,inv,ev,parent)
    admitted=bool(result['acceptable'] and result['strict_logit_parity_all89'] and result['audit']['jp50_qdq_compatible'])
    x.write(target/'DEVICE_ADMISSION.json',dict(status='ADMITTED_PENDING_DEVICE' if admitted else 'REJECTED',seed=seed,qat=x.pin(target/'QAT_RECEIPT.json'),accepted=admitted,no_external_selection=True),True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--seed',type=int,choices=[42,64,91,107]);a=p.parse_args();a.output=a.output.resolve()
    if a.seed is not None:
        if not a.execute:p.error('--seed requires --execute')
        return one(a.output,a.seed)
    plan=prepare(a.output)
    if not a.execute:print('Four-seed QAT prepared; 0 updates run');return
    states=[]
    for seed in (42,64,91,107):
        log=a.output/f's{seed}.log'
        with log.open('x') as stream:
            code=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),'--execute','--seed',str(seed)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=REPO).returncode
        receipt=a.output/f's{seed}/DEVICE_ADMISSION.json'
        states.append(dict(seed=seed,returncode=code,admission=x.read(receipt) if receipt.exists() else dict(status='FAILED_EXECUTION',log=str(log))))
        x.write(a.output/'RUN_STATUS.json',dict(status='RUNNING',seeds=states))
    x.write(a.output/'RUN_STATUS.json',dict(status='ALL_ATTEMPTS_RESOLVED',seeds=states))
    if any(s['returncode'] for s in states):raise SystemExit(1)

if __name__=='__main__':main()
