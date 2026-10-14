"""Full task-gradient audit and selected-checkpoint source rescore for Stage 1."""
import argparse
import json
from pathlib import Path
import sys
import time
import types
from contextlib import contextmanager

p=argparse.ArgumentParser()
for name in ('capsule','identity-manifest','spec','mode','out'):p.add_argument('--'+name,required=True)
p.add_argument('--cell',default='A1_W4A8')
a=p.parse_args();sys.path.insert(0,a.capsule)
import numpy as np
import torch
from torch.nn import functional as F
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat,compression_train_mode,quantization_mode,FixedQATConv2d
from quantization.qat import LiteMLAQAT
from training.losses import focal_loss,kd_loss
from experiments.revision_queue import atomic_json,file_hash
import scripts.compression_jobs as cj
from simulator_v4 import simulator

torch.set_num_threads(2);torch.manual_seed(1729);np.random.seed(1729)
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.benchmark=False
start=time.monotonic();spec=json.loads(Path(a.spec).read_text());spec['workers']=0
for item in json.loads(Path(a.identity_manifest).read_text())['files']:
    assert file_hash(item['path'])==item['sha256'],item['path']
base=Path(spec['interaction_receipt']).parent

def load(graph,state):
    return load_compression_qat(build_compression_student(graph,pretrained=False),state).cuda()

def excess(Fmargin,Qmargin,Tmargin,T=4):
    f,q,t=Fmargin.double()/T,Qmargin.double()/T,Tmargin.double()/T
    pf,pt=f.sigmoid(),t.sigmoid()
    signed=(pf-pt)*(q-f)*T*T
    kl=(pf*(F.logsigmoid(f)-F.logsigmoid(q))+(1-pf)*(F.logsigmoid(-f)-F.logsigmoid(-q)))*T*T
    delta=(F.softplus(q)-pt*q-F.softplus(f)+pt*f)*T*T
    assert torch.allclose(delta,signed+kl,rtol=1e-9,atol=1e-9)
    return delta,signed,kl

if a.mode=='parity':
    rows=[]
    for path in sorted(base.glob('*_B[123]_s42.json')):
        receipt=json.loads(path.read_text())
        if receipt.get('outcome')!='completed' or not receipt.get('best_path'):continue
        assert file_hash(receipt['best_path'])==receipt['best_sha256']
        m=load(receipt['graph'],torch.load(receipt['best_path'],map_location='cpu',weights_only=False)['model'])
        results={}
        for kind in ('legacy','integer'):
            with simulator(m,kind):results[kind]=cj.validation(m,spec,paired=True)
        delta=results['integer']['micro']['mIoU']-results['legacy']['micro']['mIoU']
        rows.append({'job_id':path.stem,'checkpoint_sha256':receipt['best_sha256'],
            'delta_mIoU':delta,'parity_within_0_15pp':abs(delta)<=.0015,'validation':results})
        print(path.stem,delta,flush=True)
        del m;torch.cuda.empty_cache()
    assert len(rows)==9
    atomic_json(a.out,{'status':'complete','outcome':'completed','stage':'1.3 selected v2 source rescore',
        'rows':rows,'all_selected_parity_pass':all(r['parity_within_0_15pp'] for r in rows),
        'no_training':True,'no_external_benchmark_access':True,'elapsed_seconds':time.monotonic()-start,
        'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20})
    raise SystemExit

for T in (1,4):
    margins=torch.cat((torch.randn(128,dtype=torch.float64)*10,torch.tensor([-1e4,1e4],dtype=torch.float64)))
    excess(margins,margins.flip(0),margins.roll(1),T)
payload=base/(a.cell+'_interaction.pt');graph=a.cell.split('_')[0]
m=load(graph,torch.load(payload,map_location='cpu',weights_only=True)['model'])
compression_train_mode(m,True)
teacher=cj.teacher(spec)
assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
ds=cj.pseudo(spec,augment=True)
indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(1729))[:128].tolist()
scopes=['all_quant','bypass_stem','bypass_QKV_joint','bypass_decoder']
rows=[];excess_rows=[]
def denominator_capture(module,store):
    original=module._relu_linear_att_qat
    def wrapped(self,qkv):
        with torch.no_grad():
            b,_,h,w=qkv.shape;dim=self.litemla.dim;z=qkv.detach().reshape(b,-1,3*dim,h*w)
            q=self.litemla.kernel_func(z[:,:,:dim]);k=self.litemla.kernel_func(z[:,:,dim:2*dim]);v=F.pad(z[:,:,2*dim:],(0,0,0,1),value=1)
            store.append(self.mm2(self.mm1(v,k.transpose(-1,-2)),q)[:,:,-1:].cpu())
        return original(qkv)
    return types.MethodType(wrapped,module)

for batch in range(8):
    samples=[ds[i] for i in indices[batch*16:(batch+1)*16]]
    x=torch.stack([s['image'] for s in samples]).float().cuda()
    y=torch.stack([s['mask'] for s in samples]).long().cuda();valid=y>=0
    with torch.no_grad():target=teacher(x).detach()
    assert target.shape== (16,2,224,224)
    den_fp=[];den_q=[]
    modules=[mod for mod in m.modules() if isinstance(mod,LiteMLAQAT)]
    originals=[mod._relu_linear_att_qat for mod in modules]
    for mod in modules:mod._relu_linear_att_qat=denominator_capture(mod,den_fp)
    with torch.no_grad(),quantization_mode(m,False):fp=m(x).detach()
    for mod,fn in zip(modules,originals):delattr(mod,'_relu_linear_att_qat')
    for scope in scopes:
        m.zero_grad(set_to_none=True);assert all(param.grad is None for param in m.parameters())
        compression_train_mode(m,True)
        disabled=[]
        for name,mod in m.named_modules():
            if not isinstance(mod,FixedQATConv2d):continue
            take=(scope=='bypass_stem' and 'input_stem' in name) or (scope=='bypass_QKV_joint' and ('.qkv.' in name or '.aggreg.' in name)) or (scope=='bypass_decoder' and '.head.' in name)
            if take:disabled.append(name);mod.fake_quant_enabled=False
        current=[]
        for mod in modules:mod._relu_linear_att_qat=denominator_capture(mod,current)
        with simulator(m,'integer'):
            out=m(x);hard=focal_loss(out,y);kl=kd_loss(out,target,4,valid);loss=.5*hard+.5*kl
            assert torch.isfinite(loss)
            loss.backward()
        for mod in modules:delattr(mod,'_relu_linear_att_qat')
        sumsq={name:float(param.grad.detach().double().square().sum()) for name,param in m.named_parameters() if param.grad is not None}
        norm=sum(sumsq.values())**.5
        logger=float(torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf'),error_if_nonfinite=True))
        assert abs(logger-norm)/max(norm,1e-12)<=.01
        assert all(param.grad is None for param in teacher.parameters())
        collapse=[]
        for df,dq in zip(den_fp,current):
            collapse.append({'already_small_FP':float((df<1e-3).float().mean()),
                'quant_collapse_from_healthy_FP':float(((df>=.1)&(dq<df/16)).float().mean()),
                'tiny_positive_D_Q':float(((dq>0)&(dq<1e-3)).float().mean())})
        rows.append({'batch':batch,'scope':scope,'norm64':norm,'logger_norm':logger,
            'task_loss':float(loss.detach()),'teacher_KL':float(kl.detach()),'valid_pixels':int(valid.sum()),
            'disabled_quantizers':disabled,'denominator_attribution':collapse,
            'parameter_gradient_share':{name:value/max(norm*norm,1e-30) for name,value in sumsq.items()}})
        if scope=='all_quant':
            delta,signed,klterm=excess(fp[:,1]-fp[:,0],out.detach()[:,1]-out.detach()[:,0],target[:,1]-target[:,0])
            for i in range(16):
                yy=y[i];v=valid[i]
                # Existing pseudo mask; one-pixel class-transition boundary, explicitly diagnostic.
                edge=torch.zeros_like(v);edge[1:]|=(yy[1:]!=yy[:-1])&(v[1:]&v[:-1]);edge[:,1:]|=(yy[:,1:]!=yy[:,:-1])&(v[:,1:]&v[:,:-1])
                confidence=target[i].softmax(0).amax(0)
                for label,mask in [('valid',v),('background',v&(yy==0)),('water',v&(yy==1)),('boundary_1px',v&edge),('interior',v&~edge),('teacher_confidence_ge_0_9',v&(confidence>=.9)),('teacher_confidence_lt_0_9',v&(confidence<.9))]:
                    if not mask.any():continue
                    z=delta[i][mask];s=signed[i][mask];k=klterm[i][mask]
                    ht=float(z.clamp(min=0).mean());bt=float((-z).clamp(min=0).mean())
                    assert abs(ht-bt-float(z.mean()))<=1e-9
                    excess_rows.append({'image_index':indices[batch*16+i],'stratum':label,'support':int(mask.sum()),
                        'H_T':ht,'B_T':bt,'mean_delta':float(z.mean()),'signed_mean':float(s.mean()),
                        'KL_mean':float(k.mean()),'positive_fraction':float((z>0).double().mean()),
                        'q05':float(torch.quantile(z,.05)),'q50':float(z.median()),'q95':float(torch.quantile(z,.95))})
        m.zero_grad(set_to_none=True)
    print(json.dumps({'cell':a.cell,'full_KD_batch':batch}),flush=True)
teacher.eval();compression_train_mode(m,True)
source={}
for scope in scopes:
    compression_train_mode(m,True)
    for name,mod in m.named_modules():
        if isinstance(mod,FixedQATConv2d):
            mod.fake_quant_enabled=not ((scope=='bypass_stem' and 'input_stem' in name) or (scope=='bypass_QKV_joint' and ('.qkv.' in name or '.aggreg.' in name)) or (scope=='bypass_decoder' and '.head.' in name))
    desired={mod:mod.fake_quant_enabled for mod in m.modules() if isinstance(mod,FixedQATConv2d)}
    original_context=cj.quantization_mode
    @contextmanager
    def partial_mode(model,enabled,**kwargs):
        with original_context(model,enabled,**kwargs):
            if enabled:
                for mod,flag in desired.items():mod.fake_quant_enabled=flag
            yield model
    try:
        cj.quantization_mode=partial_mode
        with simulator(m,'integer'):source[scope]=cj.validation(m,spec,paired=True)
    finally:
        cj.quantization_mode=original_context
atomic_json(a.out,{'status':'complete','outcome':'completed','stage':'1.1/1.2/1.5 TRAIN audit',
    'cell':a.cell,'rows':rows,'excess_rows':excess_rows,'source_counterfactuals':source,
    'V_only':{'status':'blocked','reason':'Fused QKV shares a quantized input; joint Q/K/V scope disclosed; no V-only result fabricated'},
    'loss_scope':'same frozen live teacher on identical augmented TRAIN tensors; .5 focal + .5 T^2 KL, T4',
    'teacher_frozen_no_grad':True,'global_norm_independent_F64':True,'no_gradient_accumulation':True,
    'diagnostic_identity_random_and_extreme_margins':True,'no_training':True,
    'no_external_benchmark_access':True,'elapsed_seconds':time.monotonic()-start,
    'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20})
