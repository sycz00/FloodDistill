"""Exact geometric teacher targets; proposal activates only with explicit cache approval."""
import time
from pathlib import Path
import common10 as c
def main():
    torch,cj,old,v=c.setup();approval=c.read(c.ROOT/'teacher_cache_policy.json',{})
    assert approval.get('mode')=='float32_eight_geometric_views','Offline cache correction is awaiting user choice'
    spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=42),augment=False);teacher=old.teacher_cpu(spec);out=c.DATA/'teacher_logits';out.mkdir(parents=True,exist_ok=True);rows=[];tic=time.monotonic()
    with torch.no_grad():
        for i,path in enumerate(ds.files):
            x=ds[i]['image'].float();views=[]
            for flip in [False,True]:
                for rotation in range(4):
                    z=x.flip(-1) if flip else x
                    views.append(z.rot90(rotation,(-2,-1)))
            logits=teacher(torch.stack(views)).float().cpu()
            assert logits.shape[0]==8 and torch.isfinite(logits).all()
            target=out/path.name;torch.save(dict(logits=logits,source_sha256=c.sha(path),teacher_sha256=spec['teacher_sha256'],views='index=4*horizontal_flip+rotation; target is fresh teacher on that clean geometric image'),target)
            rows.append(dict(file=str(target),sha256=c.sha(target),input_sha256=c.sha(path)))
            if i%100==0:print('teacher CPU cache',i,'/',len(ds),flush=True)
    checks=[]
    with torch.no_grad():
        for i in range(5):
            x=ds[i]['image'].float();saved=torch.load(out/ds.files[i].name,weights_only=True)['logits'];errors=[]
            for flip in [False,True]:
                for rotation in range(4):
                    view=(x.flip(-1) if flip else x).rot90(rotation,(-2,-1));fresh=teacher(view[None])[0];errors.append(float((saved[4*int(flip)+rotation]-fresh).abs().max()))
            assert max(errors)<1e-3,(i,errors)
            checks.append(dict(file=ds.files[i].name,maximum_geometric_target_difference=max(errors)))
    c.save(c.ROOT/'teacher_logit_cache_manifest.json',dict(outcome='completed',files=rows,seconds=time.monotonic()-tic,dtype='float32',views=8,fresh_forward_checks=checks,policy_sha256=c.sha(c.ROOT/'teacher_cache_policy.json'),difference_from_live_KD='Photometric student-only jitter; geometric targets use exact fresh teacher views'))
if __name__=='__main__':main()
