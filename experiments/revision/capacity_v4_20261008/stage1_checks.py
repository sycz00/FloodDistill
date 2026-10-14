"""Stage 1 fixed-weight checks; opt-in, finite, original receipts untouched."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--capsule',required=True)
p.add_argument('--spec',required=True)
p.add_argument('--cell',required=True)
p.add_argument('--out',required=True)
p.add_argument('--batches',type=int,default=8)
p.add_argument('--unit-only',action='store_true')
p.add_argument('--identity-manifest')
a=p.parse_args()
sys.path.insert(0,a.capsule)
import numpy as np
import torch
from torch import nn
from models.compression import build_compression_student
from models.compression_quant import FixedQATConv2d, load_compression_qat, compression_train_mode, quantization_mode
from training.losses import focal_loss
from experiments.revision_queue import atomic_json,file_hash
import scripts.compression_jobs as cj
from simulator_v4 import simulator

torch.set_num_threads(2)
torch.manual_seed(1729)
np.random.seed(1729)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
start=time.monotonic()
if a.identity_manifest:
    identities=json.loads(Path(a.identity_manifest).read_text())
    for item in identities['files']+identities['script_sources']:
        if file_hash(item['path'])!=item['sha256']:
            raise RuntimeError('Immutable input bytes changed: '+item['path'])

def layer_tests():
    results=[]
    for bits in (4,8):
        conv=nn.Conv2d(2,3,3,padding=1)
        with torch.no_grad():conv.bias.zero_()
        m=FixedQATConv2d(conv,torch.tensor([.25]),torch.tensor([0],dtype=torch.int32),torch.full((3,),.125),weight_bits=bits,activation_bits=bits)
        x=torch.randn(2,2,7,7,requires_grad=True)
        old=m(x)
        gx,gw=torch.autograd.grad(old.sum(),(x,m.weight))
        with simulator(m):
            new=m(x)
            ix,iw=torch.autograd.grad(new.sum(),(x,m.weight))
        assert torch.equal(old,new)  # binary-exact scales, random inputs away from ties
        assert torch.equal(gx,ix) and torch.equal(gw,iw)
        results.append({'bits':bits,'forward_equal':True,'STE_masks_and_gradients_equal':True})
    conv=nn.Conv2d(2,1,1,bias=False)
    with torch.no_grad():conv.weight.copy_(torch.tensor([[[[.125]],[[-.125]]]]))
    m=FixedQATConv2d(conv,torch.tensor([.25]),torch.tensor([0],dtype=torch.int32),torch.tensor([.125]),weight_bits=4,activation_bits=4)
    with simulator(m):y=m(torch.ones(1,2,3,3)*.25)
    assert torch.equal(y,torch.zeros_like(y))
    return {'single_layer_checks':results,'exact_cancellation_zero':True}

unit=layer_tests()
if a.unit_only:
    atomic_json(a.out,{'status':'complete','outcome':'completed','unit':unit,'elapsed_seconds':time.monotonic()-start})
    raise SystemExit

spec=json.loads(Path(a.spec).read_text())
spec['workers']=0
base=Path(spec['interaction_receipt']).parent
graph=a.cell.split('_')[0]
payload_path=base/(a.cell+'_interaction.pt')
payload=torch.load(payload_path,map_location='cpu',weights_only=True)
model=load_compression_qat(build_compression_student(graph,pretrained=False),payload['model']).cuda()
fp_receipt=json.loads((base/f'{graph}_FP_s42.json').read_text())
assert file_hash(fp_receipt['best_path'])==fp_receipt['best_sha256']
fp=build_compression_student(graph,pretrained=False).cuda()
fp.load_state_dict(torch.load(fp_receipt['best_path'],map_location='cpu',weights_only=False)['model'],strict=True)
fp.eval()
ds=cj.pseudo(spec,augment=True)
indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(1729))[:16*a.batches].tolist()
assert len(indices)==16*a.batches
rows=[]
original_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def norm64(m):
    return float(torch.sqrt(sum((p.grad.detach().double().square().sum() for p in m.parameters() if p.grad is not None),torch.zeros((),device='cuda',dtype=torch.float64))))

def run(m,x,y,kind,tau=None):
    m.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in m.parameters())
    with simulator(m,kind,tau):
        out=m(x);loss=focal_loss(out,y);loss.backward()
    assert torch.isfinite(out).all() and torch.isfinite(loss)
    n=norm64(m)
    logger=float(torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf'),error_if_nonfinite=True))
    assert abs(logger-n)/max(n,1e-12)<=.01
    shares={name:float(param.grad.double().square().sum())/max(n*n,1e-30) for name,param in m.named_parameters() if param.grad is not None}
    m.zero_grad(set_to_none=True)
    return n,float(loss),out.detach(),shares

for batch in range(a.batches):
    # Reuse these exact augmented tensors for every probe: no RNG-dependent arm differences.
    samples=[ds[i] for i in indices[batch*16:(batch+1)*16]]
    x=torch.stack([s['image'] for s in samples]).float().cuda()
    y=torch.stack([s['mask'] for s in samples]).long().cuda()
    fp_norm,_,_,_=run(fp,x,y,'legacy')
    compression_train_mode(model,enabled=True)
    old,oldloss,oldout,shares=run(model,x,y,'legacy')
    fixed,fixedloss,fixedout,fixedshares=run(model,x,y,'integer')
    rows.append({'batch':batch,'kind':'integer','fp_frozen_norm':fp_norm,'legacy_norm':old,'norm64':fixed,
        'ratio_to_FP':fixed/max(fp_norm,1e-12),'legacy_loss':oldloss,'loss':fixedloss,
        'argmax_agreement':float((oldout.argmax(1)==fixedout.argmax(1)).float().mean()),
        'valid_pixels':int((y>=0).sum()),'module_gradient_shares':fixedshares,
        'legacy_module_gradient_shares':shares})
    for tau in (.001,.01,.1):
        for kind in ('floor','backward_floor'):
            n,l,o,_=run(model,x,y,kind,tau)
            if kind=='backward_floor':assert torch.equal(o,oldout)
            rows.append({'batch':batch,'kind':kind,'tau':tau,'norm64':n,'fp_frozen_norm':fp_norm,
                'ratio_to_FP':n/max(fp_norm,1e-12),'loss':l,
                'argmax_agreement':float((o.argmax(1)==oldout.argmax(1)).float().mean()),
                'exact_legacy_forward':torch.equal(o,oldout)})
    print(json.dumps({'cell':a.cell,'batch':batch,'legacy_norm':old,'integer_norm':fixed,'fp_norm':fp_norm}),flush=True)

assert all(torch.equal(original_state[k],v.detach().cpu()) for k,v in model.state_dict().items())
parity={}
for kind in ('legacy','integer'):
    with simulator(model,kind):parity[kind]=cj.validation(model,spec,paired=True)
diff=abs(parity['integer']['micro']['mIoU']-parity['legacy']['micro']['mIoU'])
integers=[r for r in rows if r['kind']=='integer']
ratio=np.asarray([r['ratio_to_FP'] for r in integers])
gate=bool(np.median(ratio)<=10 and max(ratio)<=100 and diff<=.0005)
summary={kind:{'p50':float(np.median([r['norm64'] for r in rows if r['kind']==kind])),
                'p95':float(np.percentile([r['norm64'] for r in rows if r['kind']==kind],95)),
                'max':max(r['norm64'] for r in rows if r['kind']==kind)} for kind in ('integer','floor','backward_floor')}
atomic_json(a.out,{'status':'complete','outcome':'completed','stage':'1.1/1.3 fixed-weight first gate',
    'cell':a.cell,'unit':unit,'rows':rows,'summary':summary,'validation':parity,
    'PTQ_parity_absolute_mIoU_difference':diff,'integer_norm_and_PTQ_gate':gate,
    'augmented_argmax_agreement_mean':float(np.mean([r['argmax_agreement'] for r in integers])),
    'norm_logger_agreement_within_one_percent':True,'no_gradient_accumulation':True,
    'fixed_state_preserved':True,'loss_scope':'focal-only attribution probe; full live-teacher KD audit remains pending',
    'no_training':True,'no_external_benchmark_access':True,'TF32_enabled':False,
    'payload_sha256':file_hash(payload_path),'elapsed_seconds':time.monotonic()-start,
    'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20})
flat=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]
keys=list(dict.fromkeys(k for r in flat for k in r))
with Path(a.out).with_suffix('.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
