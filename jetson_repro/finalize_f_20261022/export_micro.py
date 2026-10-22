#!/usr/bin/env python3
"""Export the four existing R4_micro parents with full frozen-source parity.

No training, no external-score selection, no modification of historical packages.
Requires the workstation esi environment and the existing frozen experiment tree.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
REPO=Path(__file__).resolve().parents[2]
F=REPO/'experiments/revision/controlled512_v2_20261016T213414Z/revision_finalize_20261019T141008Z'

def one(seed,out):
    sys.path.insert(0,str(F/'code'))
    import experiment_quant as q
    x=q.x
    import numpy as np, onnx, onnxruntime as ort
    from onnxsim import simplify
    package=x.verify_package(F/'tiny/ladder/package.json')
    entry=next(e for e in package['entries'] if e['architecture']=='R4_micro' and e['seed']==seed)
    cfg=x.read(entry['config']);assert x.sha(entry['config'])==entry['config_sha256']
    out.mkdir(parents=True,exist_ok=False)
    torch,model,ji,ds,inv,ev=q.runtime(cfg)
    from quantization.onnx_qdq import export_float_onnx
    model.eval();assert sum(p.numel() for p in model.parameters())==1266
    path=out/f'R4_micro_s{seed}_float.onnx'
    export_float_onnx(model,torch.zeros(1,6,512,512),str(path),opset=13,static=True)
    graph,ok=simplify(onnx.load(path));assert ok
    graph.ir_version=min(graph.ir_version,10);onnx.checker.check_model(graph);onnx.save(graph,path)
    semantic,parent,sp=q.metric_rows(ji.session(path,disable_optimization=True),ds,inv,ev,model)
    optimized,_,op=q.metric_rows(ji.session(path),ds,inv,ev,model)
    passed=all(p['strict_logit_parity_pass'] for p in sp+op)
    receipt=dict(status='PASSED_FULL_SOURCE_PARITY' if passed else 'FAILED_FULL_SOURCE_PARITY',
        architecture='R4_micro',seed=seed,parameters=1266,checkpoint=x.pin(entry['checkpoint']),
        training_receipt=x.pin(entry['training_receipt']),config=x.pin(entry['config']),
        float_onnx=x.pin(path),n_images=89,atol=.001,rtol=.001,semantic_parity=sp,optimized_parity=op,
        float_evaluation=q.evaluate_rows(optimized,ji),source_only=True,native_device_execution=False)
    x.write(out/'EXPORT_RECEIPT.json',receipt,True)
    if not passed:raise RuntimeError('Float construction parity failed; no downstream conversion admitted')
    print(f'R4_micro seed {seed}: full source parity passed',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[42,64,91,107]);a=p.parse_args()
    if a.seed is not None:return one(a.seed,a.output/f's{a.seed}')
    a.output.mkdir(parents=True,exist_ok=False)
    for s in (42,64,91,107):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),'--seed',str(s)],check=True,stdin=subprocess.DEVNULL,cwd=REPO)
    # All four remain in the manifest; a failed seed stops the preparation.
    import hashlib
    artifacts=[dict(path=str(p.relative_to(a.output)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(a.output.rglob('*')) if p.is_file()]
    (a.output/'MANIFEST.json').write_text(json.dumps(dict(status='FOUR_FLOAT_EXPORTS_FULL_SOURCE_PARITY_PASSED',seeds=[42,64,91,107],artifacts=artifacts,training_launched=False,device_status='NOT_RUN',ptq_status='NOT_RUN'),indent=2)+'\n')

if __name__=='__main__':main()
