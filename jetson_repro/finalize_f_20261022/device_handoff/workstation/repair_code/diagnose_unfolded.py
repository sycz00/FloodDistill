#!/usr/bin/env python3
"""Finite source-only calibration/head diagnostic. No optimizer updates or external reads."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[2];REPO=WORK/'FloodDistill';OLD=ROOT.parent/'8_jetson_export_20261022'
F=REPO/'experiments/revision/controlled512_v2_20261016T213414Z/revision_finalize_20261019T141008Z'
sys.path.insert(0,str(F/'code'));import experiment_quant as q
x=q.x

def rows(model,items,torch):
 import numpy as np
 out=[]
 with torch.no_grad():
  for image,mask,rec in items:
   z=model(torch.from_numpy(image)).numpy();assert np.isfinite(z).all();pred=z[0].argmax(0);valid=(mask>=0)&(mask<2)
   c=np.bincount((2*mask[valid]+pred[valid]).astype('int64'),minlength=4).reshape(2,2)
   out.append(dict(file=rec['name'],event=rec['event'],confusion=c.tolist()))
 return out

def score(model,items,torch,ji,parent):
 out=rows(model,items,torch);a=q.evaluate_rows(out,ji);b=q.evaluate_rows(parent,ji)
 drops={k:b[k]-a[k] for k in ('mIoU','equal_event_mIoU','water_IoU')}
 return dict(evaluation=a,baseline=b,drops=drops,accuracy_pass=all(v<=.02 for v in drops.values()))

def make(model,batches,torch):
 from quantization.qat import prepare_qat,QATConv2d,LiteMLAQAT,_set_module
 from models.compression_quant import FixedQATConv2d
 torch.manual_seed(1729);target=prepare_qat(copy.deepcopy(model).eval(),batches,skip_last_n=0,calibration_type='per_tensor')
 for name,m in list(target.named_modules()):
  if isinstance(m,LiteMLAQAT):_set_module(target,name,m.litemla)
 for name,m in list(target.named_modules()):
  if isinstance(m,QATConv2d):_set_module(target,name,FixedQATConv2d(m,m._act_scale,m._act_zp,m._w_scale,weight_bits=8,activation_bits=8))
 return target.eval()

def main(seed):
 import numpy as np
 out=ROOT/f'diagnostic_unfolded_s{seed}';out.mkdir(exist_ok=False);start=time.time();plan=x.read(OLD/'workstation/PROTOCOL.json');e=next(e for e in plan['entries'] if e['seed']==seed)
 for pin in (e['config'],e['checkpoint']):assert x.sha(pin['path'])==pin['sha256']
 torch,model,ji,ds,inv,ev=q.runtime(x.read(e['config']['path']));items=list(q.source_items(ds,inv,ev));assert len(items)==89
 from quantization.manual import fold_bn
 from quantization.qat import _set_module
 from models.compression_quant import FixedQATConv2d
 prepared_base=copy.deepcopy(model).eval();parent=x.read(OLD/f'workstation/B0_s{seed}/FLOAT_EVALUATION.json')['rows'];original=rows(model,items,torch);assert original==parent
 checks=[]
 with torch.no_grad():
  for image,mask,rec in items:
   a=model(torch.from_numpy(image)).numpy();b=prepared_base(torch.from_numpy(image)).numpy();checks.append(dict(file=rec['name'],passed=bool(np.allclose(a,b,atol=.001,rtol=.001)),max_abs=float(np.max(abs(a-b)))))
 x.write(out/'FLOAT_CONSTRUCTION_PARITY.json',dict(passed=all(c['passed'] for c in checks),images=89,atol=.001,rtol=.001,rows=checks),True)
 if not all(c['passed'] for c in checks):x.write(out/'RESULT.json',dict(status='BLOCKED_CONSTRUCTION_PARITY',seed=seed));return
 inherited=x.read(plan['inherited_protocol']['path']);load=q.streaming_samples(inherited)
 source_order=sorted(items,key=lambda z:hashlib.sha256(z[2]['name'].encode()).hexdigest())
 policies=[('gdacs8',[dict(image=load(rec)[0][None]) for rec in inherited['qat_train'][:8]]),('s11_8',[dict(image=torch.from_numpy(z[0])) for z in source_order[:8]]),('s11_89',[dict(image=torch.from_numpy(z[0])) for z in source_order])]
 convs=[(n,m) for n,m in prepared_base.named_modules() if isinstance(m,torch.nn.Conv2d)];last,head=convs[-1];assert head.out_channels==2
 results=[]
 for name,cal in policies:
  quant=make(prepared_base,cal,torch);scales={n:float(m._act_scale[0]) for n,m in quant.named_modules() if isinstance(m,FixedQATConv2d)}
  x.write(out/(name+'_scales.json'),scales,True)
  for keep_float in (False,True):
   target=copy.deepcopy(quant)
   if keep_float:_set_module(target,last,copy.deepcopy(head))
   value=score(target,items,torch,ji,parent);value.update(calibration=name,calibration_images=len(cal),head_float=keep_float,attention_matmul_float=True,quantized_convs=sum(isinstance(m,FixedQATConv2d) for m in target.modules()),head_name=last)
   results.append(value);print(seed,name,'float_head',keep_float,value['drops'],value['accuracy_pass'],flush=True)
   x.write(out/'PROGRESS.json',dict(seed=seed,completed_arms=len(results),planned=6,elapsed=time.time()-start))
  del quant
 x.write(out/'RESULT.json',dict(status='SOURCE_DIAGNOSTICS_COMPLETE',seed=seed,arms=results,elapsed_seconds=time.time()-start,optimizer_updates=0,external_datasets_read=False),True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.seed)
