"""Distinct component validation; whole-smoke attempts are not repeated."""
import time,numpy as np
import common_v9 as c
from policy_final02 import activate
activate()
torch,cj,ev=c.setup('cuda');c.install_label_guard();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
spec=c.read(c.ROOT/'training_spec.json');val,_=cj.loaders(spec);batch=next(iter(val));x=batch['image'];records=[]
parents=[p for p in c.read(c.V7/'reuse_models.json') if p['graph']=='R4']
for p in parents:
    for quant in [False,True]:
        if quant:
            path=c.ROOT/'ptq_states'/f"q_s{p['seed']}.pt";entry=dict(id=f"PTQ_s{p['seed']}",graph='R4',quantized=True,checkpoint=str(path),sha256=c.sha(path))
        else:entry=p
        model=c.load_model(entry,'cpu')
        with torch.no_grad():
            a=model(x).numpy();model=model.cuda();b=model(x.cuda()).cpu().numpy()
        dif=[int(np.count_nonzero((z[1]>z[0])!=(w[1]>w[0]))) for z,w in zip(a,b)]
        print(entry['id'],'FP32_TF32_OFF differences',dif,flush=True)
        assert max(dif)<=2,(entry['id'],dif)
        records.append(dict(model=entry['id'],seed=p['seed'],quantized=quant,different_pixels=dif,max_logit_error=float(np.max(np.abs(a-b)))))
        del model;torch.cuda.empty_cache()
throughput={}
for graph,bs in [('R4',4),('B0',4),('teacher',4)]:
    model=c.teacher_cpu(spec).cuda() if graph=='teacher' else c.build(graph).cuda();model.eval();xx=x[:bs].cuda();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(xx);torch.cuda.synchronize();tic=time.perf_counter()
        for _ in range(3):model(xx)
        torch.cuda.synchronize();seconds=(time.perf_counter()-tic)/3
    throughput[graph]=dict(batch_size=bs,seconds_per_batch=seconds,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    del model;torch.cuda.empty_cache()
c.save(c.ROOT/'numerical_gate_final02.json',dict(outcome='completed',components=records,throughput=throughput,backend=ev.backend_record('deployment_tf32_off'),
    diagnosis='First smoke omitted Trainer seed reset; verified CPU first-batch hashes now exact. Second whole smoke passed those checks but TF32 permitted16 boundary flips in one image. New readout policy disables TF32 at every GPU evaluation boundary.',
    whole_smoke_retries_exhausted=True,training_recipe_unchanged=True,sealed_GT_unopened=True))
