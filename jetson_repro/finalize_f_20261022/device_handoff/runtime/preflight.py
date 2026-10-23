#!/usr/bin/env python3
"""Verify prepared arrays, complete raw inventories and the preserved target runtime."""
import argparse,json,sys,time,gc
from pathlib import Path
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_inputs import verify,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--sturm-root',type=Path,required=True);p.add_argument('--wf-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cpu-only',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve prior preflight receipt')
    started=time.time();prepared=verify()
    import torch
    native={}
    if not a.cpu_only:
        import tensorrt as trt
        import pycuda.driver,pycuda.autoinit
        import onnx
        if not trt.__version__.startswith('8.4.'):raise RuntimeError('Require preserved TensorRT8.4 runtime')
        runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING));manifest=json.loads((ROOT/'RUN_MANIFEST.json').read_text());checked=[]
        for model in manifest['engines']:
            if not model.get('reuse'):continue
            spec=model['reuse'];engine=ROOT/spec['engine']
            if sha(engine)!=spec['engine_sha256']:raise ValueError('Reused engine hash mismatch')
            cert=json.loads((ROOT/spec['build_receipt']).read_text())
            if cert['engine_sha256']!=spec['engine_sha256']:raise ValueError('Reused build receipt mismatch')
            if model['precision']=='pure' and not cert['precision_audit']['full_integer_tensor_path']:raise ValueError('Reused pure engine lacks integer tensor certificate')
            native_engine=runtime.deserialize_cuda_engine(engine.read_bytes())
            if native_engine is None:raise RuntimeError('Existing engine cannot deserialize on this target: '+model['id'])
            checked.append(model['id']);del native_engine;gc.collect()
        native=dict(tensorrt=trt.__version__,torch=torch.__version__,reused_engines_deserialized=checked)
    import dataset_io
    counts={};profiles={}
    for ds in ('s11val','s11test','bolivia','wf','sturm512'):
        counts[ds]=0
        for _,_,rec in dataset_io.images(ds,a.sturm_root,a.wf_root):
            counts[ds]+=1;tag=rec.get('input_profile','canonical_workstation');profiles[tag]=profiles.get(tag,0)+1
        print(ds,counts[ds],'exact input/mask hashes verified',flush=True)
    counts['dry159']=sum(1 for _ in dataset_io.dry())
    result=dict(status='CPU_INPUTS_VERIFIED_NATIVE_PENDING' if a.cpu_only else 'PASS_NATIVE_PREREQUISITES',prepared=prepared,full_input_inventory=counts,input_profiles=profiles,native=native,elapsed_seconds=time.time()-started,heldback97='NOT_READ')
    a.out.parent.mkdir(parents=True,exist_ok=True);temp=a.out.with_name(a.out.name+'.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.out);print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
