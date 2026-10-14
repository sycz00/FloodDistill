"""Predeclared three-LR, equal-update optimizer-reach screen on A3 W4A8."""
import argparse
import json
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
for key in ('capsule','spec','out','run-dir','adoption'):p.add_argument('--'+key,required=True)
p.add_argument('--lr',type=float,required=True)
a=p.parse_args();sys.path.insert(0,a.capsule)
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat,FixedQATConv2d,_range
from training.compression_loss import CompressionLoss
from config import TrainConfig
from utils.seeding import seed_basic
from experiments.revision_queue import atomic_json,file_hash
import scripts.compression_jobs as cj
from simulator_v4 import simulator

torch.set_num_threads(2);seed_basic(42)
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
assert a.lr in (1e-5,1e-4,5e-4)
adopt=json.loads(Path(a.adoption).read_text());assert adopt['adopted_simulator']=='integer_domain_v4'
start=time.monotonic();spec=json.loads(Path(a.spec).read_text());spec['workers']=0
receipt=json.loads(Path(spec['interaction_receipt']).read_text())
assert file_hash(receipt['tensor_path'])==receipt['tensor_sha256']
payload=torch.load(receipt['tensor_path'],map_location='cpu',weights_only=True)
m=load_compression_qat(build_compression_student('A3',pretrained=False),payload['model'])
initial={name:(mod.weight.detach().cpu().clone(),mod._w_scale.detach().cpu().clone(),mod.weight_bits) for name,mod in m.named_modules() if isinstance(mod,FixedQATConv2d)}
loss=CompressionLoss(cj.teacher(spec),m).cuda()
val,test=cj.loaders(spec)
train=DataLoader(cj.pseudo(spec,augment=True),batch_size=16,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
assert len(train)==156
cfg=TrainConfig(epochs=10,lr=a.lr,weight_decay=1e-5,task_loss='focal',use_amp=False,
    grad_clip=1.,early_stop_patience=0,val_smoothing_window=1,val_every_n_epochs=2,
    defer_terminal_evaluation=True,out_dir=a.run_dir,seed=42,gpu_ids=[0])
trainer=cj.CompressionTrainer(m,train,val,test,cfg,loss_fn=loss,quantized=True)
norms=[];old_clip=torch.nn.utils.clip_grad_norm_
def clip(parameters,max_norm,*args,**kwargs):
    ps=list(parameters)
    norm64=float(torch.sqrt(sum((v.grad.detach().double().square().sum() for v in ps if v.grad is not None),torch.zeros((),device='cuda',dtype=torch.float64))))
    logged=old_clip(ps,max_norm,*args,**kwargs)
    assert abs(float(logged)-norm64)/max(norm64,1e-12)<=.01
    norms.append(norm64)
    return logged
try:
    torch.nn.utils.clip_grad_norm_=clip
    with simulator(m,'integer'):trainer.fit()
finally:
    torch.nn.utils.clip_grad_norm_=old_clip
    loss.remove_hooks()
best=Path(a.run_dir)/'best.pt';last=Path(a.run_dir)/'last.pt'
bp=torch.load(best,map_location='cpu',weights_only=False);lp=torch.load(last,map_location='cpu',weights_only=False)
assert lp['training_state']['global_step']==1560
results={};movement={}
for label,payload in [('selected',bp),('final',lp)]:
    m.load_state_dict(payload['model'],strict=True)
    with simulator(m,'integer'):results[label]=cj.validation(m.cuda(),spec,paired=True)
    moves=[];changed=0;total=0
    for name,mod in m.named_modules():
        if not isinstance(mod,FixedQATConv2d):continue
        old,scale,bits=initial[name];current=mod.weight.detach().cpu();assert torch.equal(scale,mod._w_scale.cpu())
        qmin,qmax=_range(bits);ss=scale.reshape(-1,1,1,1)
        before=(old/ss).round().clamp(qmin,qmax);after=(current/ss).round().clamp(qmin,qmax)
        n=int((before!=after).sum());changed+=n;total+=old.numel()
        moves.append({'layer':name,'weights':old.numel(),'codes_changed_fraction':n/old.numel(),
            'mean_abs_update_in_steps':float(((current-old).abs()/ss).mean()),
            'max_abs_update_in_steps':float(((current-old).abs()/ss).max()),
            'update_L2':float((current-old).double().norm()),'is_decoder':'.head.' in name})
    movement[label]={'codes_changed_fraction':changed/total,'weights':total,'layers':moves}
atomic_json(a.out,{'status':'complete','outcome':'completed','stage':'1.6','lr':a.lr,
    'graph':'A3','precision':'W4A8','seed':42,'status_label':'simulation_only',
    'completed_updates':1560,'selected_updates':bp['training_state']['global_step'],
    'best_path':str(best),'best_sha256':file_hash(best),'last_path':str(last),'last_sha256':file_hash(last),
    'validation':results,'movement':movement,'preclip_norms':norms,
    'norm_p50':float(np.median(norms)),'norm_p95':float(np.percentile(norms,95)),'norm_max':max(norms),
    'clipped_fraction':float(np.mean(np.asarray(norms)>1)),'epochs':trainer.epoch_records,
    'common_quant_state_sha256':receipt['tensor_sha256'],'simulator_sha256':adopt['source_sha256'],
    'TF32_enabled':False,'no_external_benchmark_access':True,
    'elapsed_seconds':time.monotonic()-start,'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20})
