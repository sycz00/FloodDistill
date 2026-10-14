"""Homogeneous GPU FP32 eight-view cache with exact canonical-forward checks."""
import time,os
from pathlib import Path
import common10 as c
from teacher_preflight10 import geometric_views

def valid_cached(saved,input_sha,teacher_sha,policy_sha,torch):
    return (saved.get('source_sha256')==input_sha and saved.get('teacher_sha256')==teacher_sha
            and saved.get('policy_sha256')==policy_sha and saved.get('forward_batch_size')==8
            and saved.get('device')=='cuda' and saved.get('TF32') is False
            and saved['logits'].shape==(8,2,224,224) and saved['logits'].dtype==torch.float32
            and bool(torch.isfinite(saved['logits']).all()))

def main():
    torch,cj,old,_=c.setup();assert c.device().type=='cuda';approval=c.read(c.ROOT/'teacher_cache_policy.json');assert approval['mode']=='float32_eight_geometric_views' and approval['reuse_CPU_targets'] is False
    preflight=c.read(c.ROOT/'GPU_teacher_preflight.json');assert preflight['outcome']=='completed' and preflight['policy_sha256']==c.sha(c.ROOT/'teacher_cache_policy.json')
    spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=42),augment=False);teacher=old.teacher_cpu(spec).to(c.device());out=c.DATA/'teacher_logits';out.mkdir(parents=True,exist_ok=True);rows=[];tic=time.monotonic();reused=0;policy_sha=c.sha(c.ROOT/'teacher_cache_policy.json')
    with torch.inference_mode():
        for i,path in enumerate(ds.files):
            input_sha=c.sha(path);target=out/path.name;existing=target.exists()
            if existing:
                saved=torch.load(target,map_location='cpu',weights_only=True)
                assert valid_cached(saved,input_sha,spec['teacher_sha256'],policy_sha,torch),'Existing GPU cache does not match frozen backend/input policy; retained for diagnosis'
                reused+=1
            else:
                logits=teacher(geometric_views(ds[i]['image'].float()).to(c.device())).float().cpu()
                assert logits.shape==(8,2,224,224) and torch.isfinite(logits).all()
                temporary=target.with_name(target.name+'.tmp');torch.save(dict(logits=logits,source_sha256=input_sha,teacher_sha256=spec['teacher_sha256'],policy_sha256=policy_sha,views='index=4*horizontal_flip+rotation',forward_batch_size=8,device='cuda',TF32=False),temporary);os.replace(temporary,target)
            rows.append(dict(file=str(target),sha256=c.sha(target),input_sha256=input_sha,backend='GPU FP32 TF32off canonicalbatch8',retained_matching_GPU_file=existing))
            if i%100==0:print('canonical GPU teacher cache',i,'/',len(ds),'matching GPU files retained',reused,flush=True)
    checks=[]
    with torch.inference_mode():
        for i in range(5):
            saved=torch.load(out/ds.files[i].name,map_location='cpu',weights_only=True)['logits'];fresh=teacher(geometric_views(ds[i]['image'].float()).to(c.device())).float().cpu();difference=float((saved-fresh).abs().max());assert difference<.001,(i,difference)
            checks.append(dict(file=ds.files[i].name,maximum_geometric_target_difference=difference,bit_identical=bool(torch.equal(saved,fresh)),canonical_forward_batch_size=8,backend='GPU FP32 TF32off; every geometric view recomputed'))
    c.save(c.ROOT/'teacher_logit_cache_manifest.json',dict(outcome='completed',files=rows,seconds=time.monotonic()-tic,dtype='float32',views=8,canonical_forward_batch_size=8,CPU_targets_reused=0,GPU_targets=len(ds),same_backend_GPU_targets_reused=reused,fresh_forward_checks=checks,policy_sha256=policy_sha,preflight_sha256=c.sha(c.ROOT/'GPU_teacher_preflight.json'),homogeneous_backend=True,CPU_GPU_bit_identity_claimed=False,difference_from_live_KD='Photometric student-only jitter; exact clean geometric GPU teacher targets; fixed batch8 numerical convention'))
if __name__=='__main__':main()
