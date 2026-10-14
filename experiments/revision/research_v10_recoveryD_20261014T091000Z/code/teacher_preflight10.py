"""Validate homogeneous canonical GPU targets; disclose CPU backend differences."""
import time
import common10 as c

def numeric_summary(a,b):
    import torch
    d=(a-b).abs();pd=(torch.softmax(a/4,1)-torch.softmax(b/4,1)).abs()
    return dict(logit_max_abs=float(d.max()),logit_RMS=float(d.square().mean().sqrt()),relative_L2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)),T4_probability_max_abs=float(pd.max()),T4_probability_RMS=float(pd.square().mean().sqrt()),argmax_disagreement_fraction=float((a.argmax(1)!=b.argmax(1)).float().mean()))

def geometric_views(x):
    import torch
    return torch.stack([(x.flip(-1) if flip else x).rot90(k,(-2,-1)) for flip in [False,True] for k in range(4)])

def main():
    tic=time.monotonic();torch,cj,old,_=c.setup();assert c.device().type=='cuda'
    policy=c.read(c.ROOT/'teacher_cache_policy.json');assert policy['canonical_forward_batch_size']==8 and policy['reuse_CPU_targets'] is False
    spec=c.read(c.V9/'training_spec.json');cpu=old.teacher_cpu(spec);gpu=old.teacher_cpu(spec).to(c.device());ds=cj.pseudo(dict(spec,seed=42),augment=False);checks=[]
    with torch.inference_mode():
        for i in range(5):
            images=geometric_views(ds[i]['image'].float());a=gpu(images.to(c.device())).float().cpu();b=gpu(images.to(c.device())).float().cpu();reference=cpu(images)
            assert a.shape==(8,2,224,224) and a.dtype==torch.float32 and torch.isfinite(a).all()
            difference=float((a-b).abs().max());assert difference<.001,(i,difference)
            checks.append(dict(tile=ds.files[i].name,input_sha256=c.sha(ds.files[i]),canonical_repeat_logit_max_difference=difference,canonical_repeat_bit_identical=bool(torch.equal(a,b)),CPU_GPU_numeric_disclosure=numeric_summary(a,reference),views=8))
    c.save(c.ROOT/'GPU_teacher_preflight.json',dict(outcome='completed',canonical_forward_batch_size=8,canonical_backend='GPU FP32 TF32off default SDPA',checks=checks,CPU_GPU_logit_bit_identity_claimed=False,CPU_GPU_difference_used_as_tolerance=False,strict_cache_fresh_forward_tolerance=.001,seconds=time.monotonic()-tic,FP32=True,TF32=False,raw_sealed_labels_opened=False,policy_sha256=c.sha(c.ROOT/'teacher_cache_policy.json')))
if __name__=='__main__':main()
