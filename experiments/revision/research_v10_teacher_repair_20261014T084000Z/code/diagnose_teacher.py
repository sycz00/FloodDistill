"""Backend-only diagnostic on fixed GDACS TRAIN tiles; no target label access."""
import os,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'research_v10_recovery_20261014T081500Z'
sys.path.insert(0,str(PARENT/'code'))
import common10 as c

def stats(a,b):
    import torch
    d=(a-b).abs();pa=torch.softmax(a/4,1);pb=torch.softmax(b/4,1);pd=(pa-pb).abs()
    return dict(max_abs=float(d.max()),rms=float(d.square().mean().sqrt()),max_logit_magnitude=float(torch.maximum(a.abs(),b.abs()).max()),relative_L2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)),T4_probability_max_abs=float(pd.max()),T4_probability_rms=float(pd.square().mean().sqrt()),argmax_disagreement_fraction=float((a.argmax(1)!=b.argmax(1)).float().mean()),all_finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))

def main():
    t=time.monotonic();torch,cj,old,_=c.setup();spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=42),augment=False)
    cpu=old.teacher_cpu(spec);gpu=old.teacher_cpu(spec).to(c.device());x=ds[0]['image'].float();images=torch.stack([(x.flip(-1) if flip else x).rot90(k,(-2,-1)) for flip in [False,True] for k in range(4)]);outputs={};timings={}
    def run(name,fn):
        started=time.monotonic();outputs[name]=fn().cpu();torch.cuda.synchronize();timings[name]=time.monotonic()-started;print(name,timings[name],flush=True)
    with torch.inference_mode():
        run('GPU_batch8_default',lambda:gpu(images.to(c.device())))
        run('CPU2_batch8_default',lambda:cpu(images))
        torch.set_num_threads(8);run('CPU8_batch8_default',lambda:cpu(images));torch.set_num_threads(2)
        run('GPU_batch1_default',lambda:torch.cat([gpu(v[None].to(c.device())) for v in images]))
        run('CPU2_batch1_default',lambda:torch.cat([cpu(v[None]) for v in images]))
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        run('GPU_batch8_mathSDPA',lambda:gpu(images.to(c.device())))
        run('CPU2_batch8_mathSDPA',lambda:cpu(images))
        oldcache=Path(c.read(PARENT/'device_override.json')['cache_reuse_source'])/ds.files[0].name
        if oldcache.exists():outputs['retained_CPU_cache']=torch.load(oldcache,map_location='cpu',weights_only=True)['logits']
    reference=outputs['CPU2_batch8_default'];comparisons={name:stats(value,reference) for name,value in outputs.items()}
    pairs={a+'__vs__'+b:stats(outputs[a],outputs[b]) for a,b in [('GPU_batch8_default','GPU_batch1_default'),('GPU_batch8_mathSDPA','CPU2_batch8_mathSDPA'),('CPU2_batch1_default','GPU_batch1_default')]}
    if 'retained_CPU_cache' in outputs:pairs['retained_CPU_cache__vs__GPU_batch8_default']=stats(outputs['retained_CPU_cache'],outputs['GPU_batch8_default'])
    result=dict(outcome='completed',tile=str(ds.files[0]),input_sha256=c.sha(ds.files[0]),teacher_sha256=spec['teacher_sha256'],comparisons_against_CPU2_batch8= comparisons,pairs=pairs,timing_seconds=timings,seconds=time.monotonic()-t,torch_version=torch.__version__,GPU=torch.cuda.get_device_name(0),TF32=False,source_only=True,scope='Numerical diagnosis only; no tolerance relaxation or cache admission')
    c.save(ROOT/'teacher_backend_diagnostic.json',result);c.save(ROOT/'stage_results'/'00_diagnose_teacher.json',result)
if __name__=='__main__':main()
