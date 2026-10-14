"""Concrete correction risks checked before any training admission."""
import ast,time,torch
import common10 as c

def main():
 tic=time.monotonic();torch,cj,old,v=c.setup(8)
 for p in (c.ROOT/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 from quant10 import prepare,PerChannel,load
 from models.compression_quant import FixedQATConv2d
 from quantization.qat import _set_module
 torch.manual_seed(1729);model=c.build('R2').eval();x=torch.randn(2,6,32,32);batches=[dict(image=x)]
 q=prepare(model,batches,'W4A4');stem=next(m for m in q.modules() if isinstance(m,FixedQATConv2d));assert stem.activation_bits==8 and stem.weight_bits==8
 assert float(stem._act_scale[0])==stem._activation_scale_value
 for name,m in list(q.named_modules()):
  if isinstance(m,FixedQATConv2d):_set_module(q,name,PerChannel(m,torch.full((m.in_channels,),float(m._act_scale[0]))))
 reloaded=load(c.build('R2'),q.state_dict()).eval()
 with torch.no_grad():a=q(x);b=reloaded(x)
 assert torch.equal(a,b),'Per-channel checkpoint reload differs'
 teacher=old.teacher_cpu(c.read(c.V9/'training_spec.json'));ds=cj.pseudo(dict(c.read(c.V9/'training_spec.json'),seed=42),augment=False);checks=[]
 with torch.no_grad():
  for i in range(5):
   x=ds[i]['image'].float();views=[(x.flip(-1) if flip else x).rot90(k,(-2,-1)) for flip in [False,True] for k in range(4)];batch=teacher(torch.stack(views)).float();errors=[float((z-teacher(t[None])[0]).abs().max()) for z,t in zip(batch,views)];assert max(errors)<.001
   checks.append(dict(tile=ds.files[i].name,max_difference=max(errors),dtype='float32',geometric_views=8))
 m=c.read(c.ROOT/'manifest.json');jobs=m['jobs']
 assert len(m['training'])==116 and all(d in jobs for j in jobs.values() for d in j['deps'])
 remaining=set(jobs);ordered=[]
 while remaining:
  ready={name for name in remaining if set(jobs[name]['deps'])<=set(ordered)};assert ready,'Dependency cycle';ordered.extend(sorted(ready));remaining-=ready
 import worker10 as worker
 cutoff=time.time()+5000;status,_=worker.condition(dict(optional=True,max_seconds=100),dict(deadline=cutoff));assert status=='DEFERRED'
 c.save(c.ROOT/'preflight_corrections.json',dict(outcome='completed',seconds=time.monotonic()-tic,PerChannel_reload_bit_identical=True,stem_A8_scale_consistent=True,FP32_geometric_checks=checks,DAG_jobs=len(jobs),DAG_acyclic=True,finite_queue=True,source_only_time_admission_test=True,raw_sealed_GT_access=False))
if __name__=='__main__':main()
