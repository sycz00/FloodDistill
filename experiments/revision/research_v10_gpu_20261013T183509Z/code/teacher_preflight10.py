"""Teacher CPU/CUDA parity for all eight views, no labels read."""
import time
import common10 as c

def main():
    tic=time.monotonic();torch,cj,old,v=c.setup();assert c.device().type=='cuda'
    spec=c.read(c.V9/'training_spec.json');teacher_cpu=old.teacher_cpu(spec);teacher_gpu=old.teacher_cpu(spec).to(c.device());ds=cj.pseudo(dict(spec,seed=42),augment=False);checks=[]
    with torch.no_grad():
        for i in range(5):
            x=ds[i]['image'].float();images=torch.stack([(x.flip(-1) if flip else x).rot90(k,(-2,-1)) for flip in [False,True] for k in range(4)]);GPU=teacher_gpu(images.to(c.device())).cpu();CPU=teacher_cpu(images);delta=float((GPU-CPU).abs().max());assert delta<.001,(i,delta)
            checks.append(dict(tile=ds.files[i].name,CPU_GPU_logit_max_difference=delta,views=8))
    c.save(c.ROOT/'GPU_teacher_preflight.json',dict(outcome='completed',CPU_GPU_teacher_checks=checks,seconds=time.monotonic()-tic,FP32=True,TF32=False,raw_sealed_labels_opened=False))
