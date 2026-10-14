"""GPU continuation of exact eight-view cache, retaining verified CPU targets."""
import time,os,shutil
from pathlib import Path
import common10 as c

def main():
    torch,cj,old,v=c.setup();assert c.device().type=='cuda';approval=c.read(c.ROOT/'teacher_cache_policy.json',{});assert approval['mode']=='float32_eight_geometric_views'
    spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=42),augment=False);teacher=old.teacher_cpu(spec).to(c.device());out=c.DATA/'teacher_logits';out.mkdir(parents=True,exist_ok=True);rows=[];tic=time.monotonic();reused=0;source=Path(c.read(c.ROOT/'device_override.json')['cache_reuse_source'])
    with torch.no_grad():
        for i,path in enumerate(ds.files):
            input_sha=c.sha(path);target=out/path.name;oldpath=source/path.name;use=False;existing=False
            candidate=target if target.exists() else oldpath
            if candidate.exists():
                try:
                    saved=torch.load(candidate,map_location='cpu',weights_only=True)
                    use=saved['source_sha256']==input_sha and saved['teacher_sha256']==spec['teacher_sha256'] and saved['logits'].shape==(8,2,224,224) and saved['logits'].dtype==torch.float32 and bool(torch.isfinite(saved['logits']).all())
                except Exception:use=False
                existing=candidate==target
                if existing:assert use,'Existing recovery cache file failed hash/shape validation; retained for diagnosis'
            if use:
                if not existing:os.link(oldpath,target)
                reused+=1
            else:
                x=ds[i]['image'].float();views=[(x.flip(-1) if flip else x).rot90(k,(-2,-1)) for flip in [False,True] for k in range(4)];logits=teacher(torch.stack(views).to(c.device())).float().cpu();assert logits.shape==(8,2,224,224) and torch.isfinite(logits).all()
                temporary=target.with_name(target.name+'.tmp');torch.save(dict(logits=logits,source_sha256=input_sha,teacher_sha256=spec['teacher_sha256'],views='index=4*horizontal_flip+rotation',device='cuda',TF32=False),temporary);os.replace(temporary,target)
            rows.append(dict(file=str(target),sha256=c.sha(target),input_sha256=input_sha,backend='retained verified complete recovery file' if existing else 'retained CPU' if use else 'GPU FP32 TF32off'))
            if i%100==0:print('teacher GPU cache',i,'/',len(ds),'CPU cache reused',reused,flush=True)
    checks=[]
    with torch.no_grad():
        for i in range(5):
            x=ds[i]['image'].float();saved=torch.load(out/ds.files[i].name,weights_only=True)['logits'];errors=[]
            for flip in [False,True]:
                for k in range(4):
                    image=(x.flip(-1) if flip else x).rot90(k,(-2,-1));fresh=teacher(image[None].to(c.device()))[0].cpu();errors.append(float((saved[4*int(flip)+k]-fresh).abs().max()))
            assert max(errors)<.001,(i,errors);checks.append(dict(file=ds.files[i].name,maximum_geometric_target_difference=max(errors)))
    c.save(c.ROOT/'teacher_logit_cache_manifest.json',dict(outcome='completed',files=rows,seconds=time.monotonic()-tic,dtype='float32',views=8,CPU_targets_reused=reused,GPU_targets=len(ds)-reused,fresh_forward_checks=checks,policy_sha256=c.sha(c.ROOT/'teacher_cache_policy.json'),difference_from_live_KD='Photometric student-only jitter; fresh geometric views; backend parity verified'))
if __name__=='__main__':main()
