"""Finite V8 tasks. All adaptation choices precede target query scoring."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime
import numpy as np
import audited_queue as q
import head_solver as hs

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())
def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temp,path)


def setup():
    c=read(ROOT/'config.json')
    import evaluate_v7 as ev
    torch=ev.setup(c['capsule'],c['rungs_dir'])
    import scripts.compression_jobs as cj
    return c,ev,torch,cj


def model_for(c,parent,torch,cj):
    from train_v7 import build_model
    model=build_model('R4','lkd',c['rungs_dir'])
    model.load_state_dict(cj.load_tensor(parent['checkpoint'],parent['sha256'])['model'],strict=True)
    assert model.cls.in_channels==16 and model.cls.out_channels==2 and model.cls.kernel_size==(1,1)
    assert sum(p.numel() for p in model.parameters())==12230
    return model.cuda().eval().requires_grad_(False)


def features(model,x,torch):
    captured=[]
    handle=model.cls.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    try:
        with torch.no_grad():logits=model(x)
    finally:handle.remove()
    f=torch.nn.functional.interpolate(captured[0].float(),size=x.shape[-2:],mode='bilinear',align_corners=False)
    assert torch.isfinite(f).all() and torch.isfinite(logits).all()
    return f,logits[:,1].float()-logits[:,0].float()


def snapshot_buffers(model):
    return {k:v.detach().cpu().clone() for k,v in model.named_buffers()}


def assert_buffers(model,buffers,torch):
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for k,v in model.named_buffers():assert torch.equal(v.detach().cpu(),buffers[k]),k


def get_target(c,ev,ds,episode,row):
    x,meta=ev.batch_items('wf',ds,[row['index']],{}, {},c['spec'])
    assert meta[0][0]==row['filename']
    if 'input_sha256' in row:
        assert hashlib.sha256(x[0].cpu().numpy().tobytes()).hexdigest()==row['input_sha256']
        assert hashlib.sha256(meta[0][1].tobytes()).hexdigest()==row['mask_sha256']
    return x,meta[0]


def cache_task(smoke=False):
    c,ev,torch,cj=setup();t0=time.perf_counter()
    train=cj.pseudo(c['spec'],False);val,_=cj.loaders(c['spec'])
    datasets,_,_=ev.datasets(c['spec']);wf=next(ds for name,ds,b in datasets if name=='wf')
    directory=ROOT/('smoke_cache' if smoke else 'support_cache');directory.mkdir(exist_ok=True)
    teacher=cj.teacher(c['spec']);teacher_buffers=snapshot_buffers(teacher)
    mean=torch.tensor(c['spec'].get('norm_mean',[.1413,.138,.1235,.309,.2045,.1191]),device='cuda')[None,:,None,None]
    std=torch.tensor(c['spec'].get('norm_std',[.0741,.0737,.0869,.118,.0977,.0766]),device='cuda')[None,:,None,None]
    teacher_data={};diagnostics=[]
    episodes=c['episodes'][:1] if smoke else c['episodes']
    def teacher_cache(key,x,mask):
        began=time.perf_counter()
        # Common gain in declared physical reflectance [0,1], only on valid support pixels.
        valid=torch.as_tensor(mask!=-1,device='cuda')[None,None]
        reflectance=x*std+mean
        gained=(reflectance*1.02).clamp(0,1)
        xp=torch.where(valid,(gained-mean)/std,x)
        with torch.no_grad():
            z=teacher(x);zp=teacher(xp)
            p=torch.softmax(z,1)[0,1].cpu().numpy();pp=torch.softmax(zp,1)[0,1].cpu().numpy()
            p4=torch.softmax(z/4,1)[0,1].cpu().numpy()
        torch.cuda.synchronize()
        seconds=time.perf_counter()-began
        teacher_data[key]=dict(p=p,pp=pp,p4=p4,seconds=seconds)
        confident=np.maximum(p,1-p)>=.9;stable=np.abs(p-pp)<=.05;correct=(p>.5)==mask
        for cls in (0,1):
            eligible=(mask==cls)&confident&stable
            diagnostics.append(dict(support=key,reference_class=cls,confident_stable=int(eligible.sum()),
                confident_stable_correct=int((eligible&correct).sum()),confident_stable_wrong=int((eligible&~correct).sum()),
                offline_reference_diagnosis=True))
    for ep in episodes:
        for draw,row in enumerate(ep['supports'][:1] if smoke else ep['supports']):
            x,meta=get_target(c,ev,wf,ep,row);teacher_cache(f"{ep['event']}_{draw}",x,meta[1])
    dev=c['source_development'][:1] if smoke else c['source_development']
    for di,ep in enumerate(dev):
        item=val.dataset[ep['support_index']];x=item['image'][None].cuda().float()
        teacher_cache(f'dev_{di}',x,item['mask'].numpy())
    assert_buffers(teacher,teacher_buffers,torch);del teacher;torch.cuda.empty_cache()
    manifest=dict(parents={},teacher_stability_diagnostics=diagnostics,teacher_sha256=c['spec']['teacher_sha256'],
        gain=1.02,reflectance_bounds=[0,1],confident_probability=.9,stability_difference=.05,smoke_only=smoke)
    for parent in c['parents'][:1] if smoke else c['parents']:
        seed=parent['seed'];model=model_for(c,parent,torch,cj);buffers=snapshot_buffers(model)
        fsum=np.zeros(16);fsquare=np.zeros(16);n=0;norm_start=time.perf_counter()
        calibration=[]
        for i in c['train_feature_indices'][:2] if smoke else c['train_feature_indices']:
            item=train[i];xcpu=item['image'][None].float();x=xcpu.cuda()
            f,_=features(model,x,torch)
            # Statistics use valid support, without inspecting class values.
            ids=hs.sample_indices(item['mask'].numpy()!=-1,1729)
            z=f[0].permute(1,2,0).reshape(-1,16)[ids].cpu().numpy().astype(np.float64)
            fsum+=z.sum(0);fsquare+=(z*z).sum(0);n+=len(z)
            if i in c['calibration_indices']:calibration.append({'image':xcpu})
        mu=fsum/n;scale=np.maximum(np.sqrt(np.maximum(fsquare/n-mu*mu,0)),c['feature_scale_floor'])
        torch.cuda.synchronize();norm_seconds=time.perf_counter()-norm_start
        from models.compression_quant import prepare_compression_qat,precision_map
        qt=time.perf_counter()
        quant=prepare_compression_qat(model,calibration,weight_bits=4,activation_bits=8,calibration_seed=1729)
        # Competent declared PTQ stem exception. Preserve the same observed real range.
        stem=quant.stem[0]
        stem._w_scale.mul_(7/127);stem._weight_bits.fill_(8);stem._weight_bits_value=8
        quant.eval().requires_grad_(False);qbuffers=snapshot_buffers(quant)
        assert type(quant.cls)==torch.nn.Conv2d and quant.cls.weight.dtype==torch.float32
        torch.save(dict(model=quant.state_dict(),precision=precision_map(quant),parent_sha256=parent['sha256']),directory/f'q_s{seed}.pt')
        torch.cuda.synchronize();quant_seconds=time.perf_counter()-qt
        records=[];real_checks=[]
        sources=[]
        for ep in episodes:
            for draw,row in enumerate(ep['supports'][:1] if smoke else ep['supports']):
                x,meta=get_target(c,ev,wf,ep,row)
                sources.append((f"{ep['event']}_{draw}",x,meta[1],dict(event=ep['event'],draw=draw,filename=row['filename'],role='target_support')))
        for di,ep in enumerate(dev):
            for role in ['support','query']:
                item=val.dataset[ep[role+'_index']]
                sources.append((f'dev_{di}'+('' if role=='support' else '_query'),item['image'][None].cuda().float(),
                    item['mask'].numpy(),dict(event=ep['event'],draw=di,filename=ep[role],role='source_dev_'+role)))
        for key,x,mask,meta in sources:
            began=time.perf_counter();ff,mf=features(model,x,torch);torch.cuda.synchronize();fp_seconds=time.perf_counter()-began
            began=time.perf_counter();fq,mq=features(quant,x,torch);torch.cuda.synchronize();q_seconds=time.perf_counter()-began
            valid=mask!=-1;ids=hs.sample_indices(valid,seed+meta['draw']*1009)
            ffp=ff[0].permute(1,2,0).reshape(-1,16).cpu().numpy()
            fqp=fq[0].permute(1,2,0).reshape(-1,16).cpu().numpy()
            mf=mf[0].cpu().numpy().ravel();mq=mq[0].cpu().numpy().ravel()
            source_query=meta['role']=='source_dev_query'
            take=np.arange(len(mf)) if source_query else ids
            td=teacher_data.get(key,{})
            payload=dict(f=ffp[take],q=fqp[take],mf=mf[take],mq=mq[take],mask=mask.ravel()[take],
                valid=valid.ravel()[take],indices=ids,shape=np.asarray(mask.shape),
                p=td.get('p',np.zeros(mask.shape)).ravel()[take],pp=td.get('pp',np.zeros(mask.shape)).ravel()[take],
                p4=td.get('p4',np.zeros(mask.shape)).ravel()[take])
            path=directory/f'{key}_s{seed}.npz';np.savez_compressed(path,**payload)
            rec={**meta,'key':key,'seed':seed,'path':str(path),'sha256':q.file_hash(path),
                'feature_seconds':fp_seconds,'quant_feature_seconds':q_seconds,'teacher_seconds':td.get('seconds',0),
                'teacher_queries':0 if source_query else 2,'valid_label_pixels':int(valid.sum()),
                'label_area_m2':int(valid.sum())*100 if meta['role']=='target_support' else None}
            records.append(rec)
            if not real_checks:
                phi=hs.design(payload['f'],mu,scale);delta,info=hs.solve(phi,hs.target(payload['p'])-payload['mf'])
                raw=hs.unstandardize(delta,mu,scale)
                # Fold in actual classifier, then compare against the unfused margin at full grid.
                w=model.cls.weight.clone();bias=model.cls.bias.clone()
                with torch.no_grad():
                    dv=torch.as_tensor(raw[:-1],device='cuda',dtype=w.dtype).view(16,1,1)
                    model.cls.weight[1].add_(dv/2);model.cls.weight[0].sub_(dv/2)
                    model.cls.bias[1].add_(float(raw[-1])/2);model.cls.bias[0].sub_(float(raw[-1])/2)
                    folded=model(x)[0];expected=torch.as_tensor(mf.reshape(mask.shape),device='cuda')
                    expected+=torch.einsum('c,chw->hw',torch.as_tensor(raw[:-1],device='cuda',dtype=ff.dtype),ff[0])+float(raw[-1])
                    error=float(((folded[1]-folded[0])-expected).abs().max())
                    model.cls.weight.copy_(w);model.cls.bias.copy_(bias)
                assert error<.001,error
                real_checks.append(dict(folded_max_error=error,objective_gradient_max=info['objective_gradient_max'],
                    finite_losses_gradients=hs.sgd(phi,payload['mf'],payload['p4'],1,.01)[1]['finite_gradients']))
                pq=hs.design(payload['q'],mu,scale)
                dq,qinfo=hs.solve(pq,hs.target(payload['p'])-payload['mq'],hs.reliability(payload['p'],payload['pp']),pq-phi,.01,1)
                rawq=hs.unstandardize(dq,mu,scale);qw=quant.cls.weight.clone();qb=quant.cls.bias.clone()
                with torch.no_grad():
                    dv=torch.as_tensor(rawq[:-1],device='cuda',dtype=qw.dtype).view(16,1,1)
                    quant.cls.weight[1].add_(dv/2);quant.cls.weight[0].sub_(dv/2)
                    quant.cls.bias[1].add_(float(rawq[-1])/2);quant.cls.bias[0].sub_(float(rawq[-1])/2)
                    actual=quant(x)[0];expected=torch.as_tensor(mq.reshape(mask.shape),device='cuda')
                    expected+=torch.einsum('c,chw->hw',torch.as_tensor(rawq[:-1],device='cuda',dtype=fq.dtype),fq[0])+float(rawq[-1])
                    qerror=float(((actual[1]-actual[0])-expected).abs().max())
                    quant.cls.weight.copy_(qw);quant.cls.bias.copy_(qb)
                assert qerror<.001 and qinfo['objective_gradient_max']<1e-8
                real_checks[0].update(quant_folded_max_error=qerror,quant_objective_gradient_max=qinfo['objective_gradient_max'])
            del ff,fq,x
        assert_buffers(model,buffers,torch);assert_buffers(quant,qbuffers,torch)
        # Quantizer checkpoint reload establishes fixed-scales parity.
        from models.compression_quant import load_compression_qat
        reloaded=load_compression_qat(model,torch.load(directory/f'q_s{seed}.pt',weights_only=True)['model']).cuda().eval()
        item=train[c['train_feature_indices'][0]]['image'][None].cuda().float()
        with torch.no_grad():assert torch.equal(quant(item),reloaded(item))
        manifest['parents'][str(seed)]=dict(records=records,mean=mu.tolist(),scale=scale.tolist(),
            statistic_samples=n,normalization_seconds=norm_seconds,quant_calibration_seconds=quant_seconds,
            q_checkpoint=str(directory/f'q_s{seed}.pt'),q_sha256=q.file_hash(directory/f'q_s{seed}.pt'),
            parent_sha256=parent['sha256'],real_checks=real_checks,precision_map=precision_map(quant),
            frozen_buffers_unchanged=True)
        del model,quant,reloaded;torch.cuda.empty_cache()
    manifest.update(outcome='completed',elapsed_seconds=time.perf_counter()-t0,backend=ev.backend_record())
    q.atomic_json(directory/'manifest.json',manifest)
    return manifest


def fit_arms(z,mu,scale,lam,lr,quant=False,labelled=True):
    phi=hs.design(z['q' if quant else 'f'],mu,scale);pf=hs.design(z['f'],mu,scale)
    base=z['mq' if quant else 'mf'];t=hs.target(z['p']);w=hs.reliability(z['p'],z['pp'])
    results={'Qbase' if quant else 'F0':(np.zeros(17),dict(solve_seconds=0,trainable_parameters=0,labels=False))}
    def add(name,delta,info,labels=False):
        results[name]=(hs.unstandardize(delta,mu,scale),{**info,'labels':labels,'trainable_parameters':17})
    if quant:
        for arm,use_w,gamma in [('Q0',False,0),('Q1',True,0),('Q2',False,1),('Q3',True,1)]:
            d,info=hs.solve(phi,t-base,w if use_w else None,phi-pf,lam,gamma)
            add(arm,d,info)
    else:
        design=np.column_stack([base,np.ones(len(base))]);d,info=hs.solve(design,t-base,lam=lam)
        # F1 transforms the original margin; store its two coefficients separately.
        results['F1']=(d,{**info,'labels':False,'trainable_parameters':2,'calibration':True})
        for steps in [1,5,20]:
            d,info=hs.sgd(phi,base,z['p4'],steps,lr,4,lam);add(f'F2_{steps}',d,info)
        for arm,weights in [('F3',None),('F4',w)]:
            d,info=hs.solve(phi,t-base,weights,lam=lam);add(arm,d,info)
        if labelled:
            ty=np.where(z['mask']==1,np.log(19),-np.log(19))
            for arm,tar in [('B_supervised',ty),('B_teacher_label',.5*ty+.5*t)]:
                d,info=hs.solve(phi,tar-base,lam=lam);add(arm,d,info,True)
            d,info=hs.sgd(phi,base,np.where(z['mask']==1,.95,.05),20,lr,1,lam)
            add('B_supervised_SGD20',d,info,True)
    return results


def margins(z,arm,update,quant=False):
    base=z['mq' if quant else 'mf']
    return base+z['q' if quant else 'f']@update[:-1]+update[-1] if arm!='F1' else base*(1+update[0])+update[1]


def source_select(c,cache,seed):
    entry=cache['parents'][str(seed)];mu=np.asarray(entry['mean']);scale=np.asarray(entry['scale'])
    recs={r['key']:r for r in entry['records']}
    trials=[]
    for lam in c['lambda_grid']:
        for lr in c['lr_grid']:
            values=[];sgdvalues=[];supportlabels=querylabels=0
            for di,ep in enumerate(c['source_development']):
                z=np.load(recs[f'dev_{di}']['path']);query=np.load(recs[f'dev_{di}_query']['path'])
                arms=fit_arms(z,mu,scale,lam,lr,labelled=False)
                for arm in ['F3','F4']:
                    values.append(score_prediction(margins(query,arm,arms[arm][0])>0,query['mask'])['mIoU'])
                sgdvalues.append(score_prediction(margins(query,'F2_20',arms['F2_20'][0])>0,query['mask'])['mIoU'])
                supportlabels+=recs[f'dev_{di}']['valid_label_pixels'];querylabels+=recs[f'dev_{di}_query']['valid_label_pixels']
            trials.append(dict(lambda_=lam,lr=lr,ridge_mean=float(np.mean(values)) if values else None,
                sgd_mean=float(np.mean(sgdvalues)) if sgdvalues else None,source_query_label_pixels=querylabels,
                fitting_labels=0,source_support_reference_pixels_offline_diagnosis=supportlabels))
    if c['source_development']:
        lam=max(trials,key=lambda t:(t['ridge_mean'],-t['lambda_']))['lambda_']
        lr=max([t for t in trials if t['lambda_']==lam],key=lambda t:(t['sgd_mean'],-t['lr']))['lr']
    else:lam,lr=.01,.01
    return dict(seed=seed,lambda_=lam,lr=lr,trials=trials,selection='source-only development query mIoU; same lambda for controls',
        no_target_query_labels=True)


def counts(pred,mask):
    valid=mask!=-1;pred=np.asarray(pred,dtype=bool)
    tp=int(((mask==1)&pred&valid).sum());fp=int(((mask==0)&pred&valid).sum());fn=int(((mask==1)&~pred&valid).sum());tn=int(((mask==0)&~pred&valid).sum())
    return np.array([[tn,fn,fp],[tp,fp,fn]],dtype=np.int64)


def score_prediction(pred,mask):
    import evaluate_v7 as ev
    return ev.scores(counts(pred,mask).tolist())


def adapt_task(quant=False):
    c,ev,torch,cj=setup();cache=read(ROOT/'support_cache/manifest.json')
    datasets,_,_=ev.datasets(c['spec']);wf=next(ds for name,ds,b in datasets if name=='wf')
    resultrows=[];detail=[];artifactdir=ROOT/'adapted_checkpoints';artifactdir.mkdir(exist_ok=True)
    for parent in c['parents']:
        seed=parent['seed'];entry=cache['parents'][str(seed)];mu=np.asarray(entry['mean']);scale=np.asarray(entry['scale'])
        for rec in entry['records']:assert q.file_hash(rec['path'])==rec['sha256']
        choice=read(ROOT/f'choice_s{seed}.json') if quant else source_select(c,cache,seed)
        if not quant:q.atomic_json(ROOT/f'choice_s{seed}.json',choice)
        model=model_for(c,parent,torch,cj)
        if quant:
            from models.compression_quant import load_compression_qat
            assert q.file_hash(entry['q_checkpoint'])==entry['q_sha256']
            model=load_compression_qat(model,cj.load_tensor(entry['q_checkpoint'])['model']).cuda().eval().requires_grad_(False)
        buffers=snapshot_buffers(model)
        for ep in c['episodes']:
            allarms=[]
            for draw in c['support_draws']:
                rec=next(r for r in entry['records'] if r['key']==f"{ep['event']}_{draw}")
                began=time.perf_counter()
                with np.load(rec['path']) as archive:z={k:archive[k] for k in archive.files}
                cache_read_seconds=time.perf_counter()-began
                arms=fit_arms(z,mu,scale,choice['lambda_'],choice['lr'],quant)
                for arm,(update,info) in arms.items():
                    path=artifactdir/f"{ep['event']}_s{seed}_d{draw}_{arm}.pt"
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if arm=='F1':
                        w=(state['cls.weight'][1]-state['cls.weight'][0])*float(update[0])
                        b=(state['cls.bias'][1]-state['cls.bias'][0])*float(update[0])+float(update[1])
                    else:w=torch.as_tensor(update[:-1],dtype=state['cls.weight'].dtype).view(16,1,1);b=float(update[-1])
                    state['cls.weight'][1]+=w/2;state['cls.weight'][0]-=w/2
                    state['cls.bias'][1]+=b/2;state['cls.bias'][0]-=b/2
                    torch.save(dict(model=state,base_sha256=parent['sha256'],arm=arm,seed=seed,event=ep['event'],draw=draw,
                        precision='simulation_only_W4A8_stem8_headFP32' if quant else 'FP32',support=rec['filename'],
                        query_labels_used=False,fit=info),path)
                    allarms.append(dict(arm=arm,draw=draw,update=update,info=info,support=rec,path=str(path),cache_read_seconds=cache_read_seconds,
                        artifact_sha256=q.file_hash(path),total=np.zeros((2,3),np.int64),query_seconds=0))
            for row in ep['queries']:
                began=time.perf_counter();x,meta=get_target(c,ev,wf,ep,row);f,m=features(model,x,torch)
                torch.cuda.synchronize();feature_seconds=time.perf_counter()-began
                f=f[0].permute(1,2,0).reshape(-1,16);m=m[0].reshape(-1);h,w=meta[1].shape
                for armrec in allarms:
                    update=torch.as_tensor(armrec['update'],device='cuda',dtype=f.dtype)
                    adjusted=m*(1+update[0])+update[1] if armrec['arm']=='F1' else m+f@update[:-1]+update[-1]
                    pred=(adjusted>0).reshape(512,512)[:h,:w].cpu().numpy()
                    armrec['total']+=counts(pred,meta[1]);armrec['query_seconds']+=feature_seconds
                del f,m,x
            for r in allarms:
                rec=r['support'];info=r['info'];labelled=info['labels']
                extraction=0 if r['arm'] in ['F0','Qbase'] else rec['quant_feature_seconds'] if quant else rec['feature_seconds']
                extra=rec['feature_seconds'] if quant and r['arm']!='Qbase' else 0
                teacher_queries=rec['teacher_queries'] if r['arm'] not in ['F0','Qbase','B_supervised','B_supervised_SGD20'] else 0
                teacher_seconds=rec['teacher_seconds'] if teacher_queries else 0
                cached=0 if r['arm'] in ['F0','Qbase'] else info['solve_seconds']+r['cache_read_seconds'];cold=cached+extraction+extra+teacher_seconds
                resultrows.append(dict(dataset='wf',event=ep['event'],map=ep['map'],seed=seed,support_draw=r['draw'],
                    arm=r['arm'],status='completed',precision='simulation_only' if quant else 'FP32',
                    parent_sha256=parent['sha256'],support_id=rec['filename'],query_chips=len(ep['queries']),
                    teacher_queries=teacher_queries,target_label_pixels=rec['valid_label_pixels'] if labelled else 0,
                    target_label_area_m2=rec['label_area_m2'] if labelled else 0,trainable_parameters=info['trainable_parameters'],
                    feature_extraction_seconds=extraction,extra_FP_feature_seconds=extra,teacher_seconds=teacher_seconds,
                    solve_optimization_seconds=info['solve_seconds'],cache_read_seconds=r['cache_read_seconds'],cached_end_to_end_seconds=cached,cold_end_to_end_seconds=cold,
                    shared_peak_gpu_allocator_bytes=int(torch.cuda.max_memory_allocated()),memory_scope='shared command process including feature caches, not isolated per arm',
                    adaptation_timing_scope='sum of measured components; shared cache loading; excludes artifact serialization; no cold wall-clock speed claim',
                    shared_normalization_seconds=entry['normalization_seconds'],shared_quant_calibration_seconds=entry['quant_calibration_seconds'] if quant else 0,
                    query_feature_seconds=r['query_seconds'],artifact_path=r['path'],artifact_bytes=Path(r['path']).stat().st_size,
                    artifact_sha256=r['artifact_sha256'],lambda_=choice['lambda_'],lr=choice['lr'],
                    eta=info.get('eta'),objective_gradient_max=info.get('objective_gradient_max'),
                    query_labels_used_for_fit=False,**ev.scores(r['total'].tolist())))
                detail.append(dict(event=ep['event'],seed=seed,draw=r['draw'],arm=r['arm'],counts=r['total'].tolist(),fit=info,
                    adaptation_cost='teacher and FP/Q passes charged once in command ledger; shared per-method cold cost estimates from timed components'))
            print('adapted',seed,ep['event'],'quant',quant,flush=True)
        assert_buffers(model,buffers,torch);del model;torch.cuda.empty_cache()
    kind='q' if quant else 'fp'
    write_csv(ROOT/f'adaptation_{kind}.csv',resultrows)
    q.atomic_json(ROOT/f'adaptation_{kind}_details.json',detail)
    return dict(outcome='completed',rows=len(resultrows),query_events=[e['event'] for e in c['episodes']],
        precision_status='simulation_only' if quant else 'FP32',frozen_buffers_unchanged=True,
        backend=ev.backend_record(),all_arms_retained=True)


def joint_counts(teacher,student,mask):
    result={}
    for cls in (0,1):
        valid=mask==cls;t=teacher==mask;s=student==mask
        result[str(cls)]=[int((valid&t&s).sum()),int((valid&t&~s).sum()),
            int((valid&~t&s).sum()),int((valid&~t&~s).sum())]
    return result


def headroom_task():
    c,ev,torch,cj=setup();refs=read(c['reference_manifest'])
    assert refs['identity']==ev.cache_identity(c['spec'],c['capsule'])
    models=[(p,model_for(c,p,torch,cj)) for p in c['parents']]
    dsets,mapof,eventof=ev.datasets(c['spec']);val,_=cj.loaders(c['spec'])
    from evaluation.revision_eval import NamedDataset
    dsets=[('source_validation',NamedDataset(val.dataset,[Path(p).name for p in val.dataset.image_files]),4)]+dsets
    allrecords=[]
    for name,ds,batch in dsets:
        teacher=cj.teacher(c['spec']) if name=='source_validation' else None
        mem=None
        if teacher is None:
            record=refs['datasets'][name];assert q.file_hash(record['path'])==record['sha256'];mem=np.load(record['path'],mmap_mode='r')
        accum={}
        for begin in range(0,len(ds),batch):
            ids=list(range(begin,min(begin+batch,len(ds))))
            if name=='source_validation':
                items=[ds[i] for i in ids];x=torch.stack([v['image'] for v in items]).cuda().float()
                meta=[(Path(v['filename']).name,v['mask'].numpy(),Path(v['filename']).name,Path(v['filename']).name.split('_')[0],False,v) for v in items]
                with torch.no_grad():t=teacher(x).argmax(1).cpu().numpy()
                rule,available=ev.rule_arrays(x,c['spec'])
            else:
                x,meta=ev.batch_items(name,ds,ids,mapof,eventof,c['spec'])
                unpack=np.unpackbits(mem[ids],axis=2).reshape(len(ids),3,512,512).astype(bool)
                t,rule,available=unpack[:,0],unpack[:,1],unpack[:,2]
                for local,(i,v) in enumerate(zip(ids,meta)):
                    old=refs['datasets'][name]['rows'][i]
                    assert v[0]==old['filename'] and hashlib.sha256(x[local].cpu().numpy().tobytes()).hexdigest()==old['input_sha256']
                    assert hashlib.sha256(v[1].tobytes()).hexdigest()==old['mask_sha256']
            for parent,model in models:
                with torch.no_grad():s=model(x).argmax(1).cpu().numpy()
                for local,v in enumerate(meta):
                    filename,mask,mapid,event,omit,_=v;h,w=mask.shape
                    tt=t[local,:h,:w];ss=s[local,:h,:w];rr=rule[local,:h,:w]
                    for comparator,expert in [('T|S',ss),('T|MNDWI',rr)]:
                        for scope in ['full']+([] if omit else ['overlap_excluded']):
                            key=(parent['seed'],event,comparator,scope)
                            a=accum.setdefault(key,dict(joint=np.zeros((2,4),np.int64),teacher=np.zeros((2,3),np.int64),
                                student=np.zeros((2,3),np.int64),oracle=np.zeros((2,3),np.int64),chips=0,rule_unavailable=0))
                            joint=joint_counts(tt,expert,mask);a['joint']+=np.asarray([joint['0'],joint['1']])
                            a['teacher']+=counts(tt,mask);a['student']+=counts(expert,mask)
                            oracle=np.where((tt==mask)|(expert==mask),mask,tt);a['oracle']+=counts(oracle,mask);a['chips']+=1
                            a['rule_unavailable']+=int(((mask!=-1)&~available[local,:h,:w]).sum())
            if begin%256==0:print('headroom',name,begin,len(ds),flush=True)
        for (seed,event,comp,scope),a in accum.items():
            allrecords.append(dict(dataset=name,seed=seed,event=event,comparator=comp,scope=scope,
                joint_by_reference_class=a['joint'].tolist(),teacher_counts=a['teacher'].tolist(),student_counts=a['student'].tolist(),
                unattainable_oracle_counts=a['oracle'].tolist(),chips=a['chips'],rule_unavailable_pixels=a['rule_unavailable']))
        if teacher is not None:del teacher;torch.cuda.empty_cache()
    q.atomic_json(ROOT/'joint_confusions.json',dict(records=allrecords,order=['both_correct','teacher_only_correct','student_only_correct','both_wrong'],oracle='unattainable label-aware correct-expert choice'))
    events=[];summary=[]
    def pack(records,event):
        j=sum((np.asarray(r['joint_by_reference_class']) for r in records),np.zeros((2,4),np.int64))
        n=int(j.sum());g=int(j[:,1].sum());h=int(j[:,2].sum())
        row=dict(dataset=records[0]['dataset'],seed=records[0]['seed'],scope=records[0]['scope'],comparator=records[0]['comparator'],
            event=event,valid_pixels=n,G=g/n if n else None,H=h/n if n else None,G_minus_H=(g-h)/n if n else None,
            both_wrong_pixels=int(j[:,3].sum()),water_teacher_corrections=int(j[1,1]),water_harm=int(j[1,2]),
            water_support=int(j[1].sum()),land_support=int(j[0].sum()),oracle='unattainable')
        tcorrect=sum(np.asarray(r['teacher_counts'])[:,0].sum() for r in records)
        scorrect=sum(np.asarray(r['student_counts'])[:,0].sum() for r in records)
        assert int(tcorrect-scorrect)==g-h
        row['accuracy_identity_verified']=True
        row['rule_unavailable_pixels']=sum(r['rule_unavailable_pixels'] for r in records)
        for kind,key in [('teacher','teacher_counts'),('student','student_counts'),('oracle','unattainable_oracle_counts')]:
            cell=sum((np.asarray(r[key]) for r in records),np.zeros((2,3),np.int64))
            row.update({kind+'_'+k:v for k,v in ev.scores(cell.tolist()).items()})
        return row
    for r in allrecords:events.append(pack([r],r['event']))
    for key in sorted({(r['dataset'],r['seed'],r['scope'],r['comparator']) for r in allrecords}):
        rs=[r for r in allrecords if (r['dataset'],r['seed'],r['scope'],r['comparator'])==key]
        row=pack(rs,'pooled_pixels');row['aggregation']='pooled_pixel_counts';summary.append(row)
        per=[pack([r],r['event']) for r in rs]
        row={**row,'event':'equal_event_mean','aggregation':'equal_event','valid_pixels':sum(p['valid_pixels'] for p in per),'events':len(per)}
        for k in list(row):
            if k in ['G','H','G_minus_H'] or any(k.startswith(p+'_') for p in ['teacher','student','oracle']):
                vals=[p.get(k) for p in per if p.get(k) is not None]
                if vals:row[k]=float(np.mean(vals))
        summary.append(row)
    write_csv(ROOT/'teacher_headroom.csv',summary);write_csv(ROOT/'event_headroom.csv',events)
    return dict(outcome='completed',datasets=[d[0] for d in dsets],seeds=[42,64],records=len(allrecords),same_pixel_joint_errors=True)


def recover_task():
    c=read(ROOT/'config.json');base=Path('/data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006')
    curves=[];crossings=[];receipts=[]
    for mp in sorted(base.glob('queue*/manifest.json')):
        manifest=read(mp);sp=mp.with_suffix('.state.json')
        if not sp.exists():continue
        state=read(sp)
        for j in manifest.get('jobs',[]):
            jid=j['id'];argv=j.get('argv',j.get('command',[]))
            if not isinstance(argv,list):continue
            if not jid.startswith('O4_train_'):continue
            effective=argv;specpath=None
            if any('revision_cohort_check.py' in arg for arg in argv) and 'train' in argv and '--spec' in argv:
                specpath=Path(argv[argv.index('--spec')+1]);effective=read(specpath)['argv']
            if not any('scripts/train.py' in arg for arg in effective):continue
            if not ('matched_' in jid or 'historical_ofa_clip5_allvalid_' in jid) or not any(s in jid for s in ['_lkd_','_ofa_']):continue
            rec=state['jobs'].get(jid,{})
            if rec.get('status')!='succeeded':continue
            log=mp.parent/'logs'/f'{jid}.attempt1.log'
            n=100 if '_N100_' in jid else 2500;u=n//16
            body=log.read_text(errors='replace') if log.exists() else ''
            # Record only actual validation observations, never interpolate thresholds.
            observations=[]
            for line in body.splitlines():
                match=re.search(r'\[val\].*?ep(?:och)?[= ]+(\d+).*?mIoU[=: ]+([0-9.]+)',line,re.I)
                if not match:match=re.search(r'\[val\s+(?:epoch\s*)?(\d+)\].*?mIoU[=: ]+([0-9.]+)',line,re.I)
                if match:observations.append((int(match[1]),float(match[2])))
                else:
                    match=re.search(r'\[val\]\s+step=(\d+)\s+mIoU=([0-9.]+)',line)
                    if match:observations.append((int(match[1])+1,float(match[2])))
            seen=set();observations=[x for x in observations if not (x[0] in seen or seen.add(x[0]))]
            verified_path=next((Path(o['path']) for o in j.get('outputs',[]) if o.get('type')=='json' and Path(o['path']).exists()),None)
            verified=read(verified_path) if verified_path else {}
            checks=verified.get('checkpoint_checks',{})
            selected_epoch=checks.get('best',{}).get('epochs_completed');final_epoch=checks.get('last',{}).get('epochs_completed')
            for epoch,miou in observations:curves.append(dict(run_id=jid,epoch=epoch,optimizer_updates=epoch*u,
                distinct_scenes=n,source_validation_mIoU=miou,measured_elapsed_seconds=None,
                timing_status='per-validation timestamps unavailable',graph_identity='historical_original_activation',
                selected_checkpoint=epoch==selected_epoch,final_checkpoint=epoch==final_epoch,logged_score_decimals=4,
                log_path=str(log),log_sha256=q.file_hash(log)))
            attempts=rec.get('attempts',[])
            duration=0
            for a in attempts:
                if a.get('finished_at'):
                    receipt=read(a['receipt_path']) if a.get('receipt_path') and Path(a['receipt_path']).exists() else a
                    elapsed=receipt.get('command_elapsed_seconds')
                    if elapsed is None and receipt.get('started_at'):
                        elapsed=(datetime.fromisoformat(receipt['finished_at'])-datetime.fromisoformat(receipt['started_at'])).total_seconds()
                    duration+=elapsed or 0
            receipts.append(dict(run_id=jid,argv=argv,effective_training_argv=effective,inner_spec=str(specpath) if specpath else None,
                training_command=True,total_measured_command_seconds=duration,timing_boundary='whole training wrapper including terminal evaluation and checkpoint checks; isolated training time unavailable',
                selected_epoch=selected_epoch,final_epoch=final_epoch,checkpoint_checks=checks,
                verified_receipt_path=str(verified_path) if verified_path else None,
                verified_receipt_sha256=q.file_hash(verified_path) if verified_path else None,
                includes_teacher_forward_and_OFA_heads=True,observations=len(observations),graph_identity='historical_original_activation'))
            for threshold in [.70,.75,.80,.85]:
                ix=next((i for i,x in enumerate(observations) if x[1]>=threshold),None)
                first=observations[ix] if ix is not None else None;previous=observations[ix-1][0]*u if ix is not None and ix>0 else 0
                crossings.append(dict(run_id=jid,threshold_mIoU=threshold,distinct_scenes=n,
                    first_observed_update=first[0]*u if first else None,previous_observed_update=previous if first else None,
                    first_observed_epoch=first[0] if first else None,censored=first is None,
                    status='observed' if first else 'unreached' if observations else 'missing_curve',
                    crossing_time_seconds=None,time_status='missing validation timestamps',total_measured_training_command_seconds=duration or None,
                    graph_identity='historical_original_activation',complete_curve_path='ofa_learning_curves.csv'))
    write_csv(ROOT/'ofa_learning_curves.csv',curves,['run_id','epoch','optimizer_updates','distinct_scenes','source_validation_mIoU','measured_elapsed_seconds','timing_status','graph_identity','selected_checkpoint','final_checkpoint','logged_score_decimals','log_path','log_sha256'])
    write_csv(ROOT/'ofa_learning_efficiency.csv',crossings,['run_id','threshold_mIoU','distinct_scenes','first_observed_update','previous_observed_update','first_observed_epoch','censored','status','crossing_time_seconds','time_status','total_measured_training_command_seconds','graph_identity','complete_curve_path'])
    q.atomic_json(ROOT/'ofa_training_receipts.json',receipts)
    # Conventional capacity anchor frontier reuses identified source/external rows.
    index=read(Path(c['v7'])/'reuse_index.json')['records'];frontier=[]
    for r in index:
        p=Path(r['best_path'])
        frontier.append(dict(graph=r['graph'],seed=r['seed'],family=r.get('family'),status='reused',checkpoint=str(p),
            checkpoint_sha256=r['best_sha256'],checkpoint_bytes=p.stat().st_size,
            final_checkpoint=r.get('last_path'),final_checkpoint_sha256=r.get('last_sha256'),final_source_mIoU=r.get('final_source_metrics',{}).get('mIoU'),
            deployed_artifact_bytes=None,latency_ms=None,precision='FP32',selection='source_validation',
            completed_updates=r.get('completed_updates'),historical_early_stop=r.get('completed_updates')!=7800,
            source_mIoU=r.get('selected_source_metrics',{}).get('mIoU'),
            limitation='Training checkpoint bytes include nondeployment state; never use as deployed bytes'))
    # Include smaller fixed rungs from their own preserved training receipts.
    for p in sorted((ROOT.parent/'capacity_v4_20261008/v5/results').glob('fl_R[23]*.json')):
        r=read(p)
        if r.get('outcome') not in ['completed','reused']:continue
        frontier.append(dict(graph=r.get('graph'),seed=r.get('seed'),family='tiny_rung',status='reused',
            checkpoint=r.get('best_path'),checkpoint_sha256=r.get('best_sha256'),precision='FP32',
            source_mIoU=r.get('source_validation',{}).get('best_val_mIoU'),completed_updates=r.get('completed_updates'),
            parameters=r.get('parameters'),
            historical_early_stop=r.get('completed_updates')!=7800,latency_ms=None,deployed_artifact_bytes=None))
    for p in sorted((Path(c['v7'])/'receipts').glob('A*.json')):
        r=read(p)
        if r.get('method')!='lkd' or r.get('lr')!=.005 or r.get('graph') not in ['A0','A1']:continue
        frontier.append(dict(graph=r['graph'],seed=r['seed'],family='repaired_LKD',status='reused',
            checkpoint=r['best_path'],checkpoint_sha256=r['best_sha256'],precision='FP32',parameters=r['parameters'],
            completed_updates=r['completed_updates'],historical_early_stop=False,
            source_mIoU=r['source_validation']['best_val_mIoU'],deployed_artifact_bytes=None,latency_ms=None))
        frontier[-1].update(final_checkpoint=r['last_path'],final_checkpoint_sha256=r['last_sha256'],
            final_source_mIoU=r['source_validation'].get('final_metrics',{}).get('mIoU'))
    # Derive parameter counts from the actual adapted graphs, with source receipt checks.
    sys.path.insert(0,c['capsule']);sys.path.insert(0,c['rungs_dir'])
    import rungs
    parameters={}
    for graph in sorted({r['graph'] for r in frontier if r.get('graph')}):
        model=rungs.build(graph);parameters[graph]=sum(p.numel() for p in model.parameters());del model
    for row in frontier:
        actual=parameters.get(row.get('graph'))
        if row.get('parameters') is not None:assert row['parameters']==actual
        row['parameters']=actual
    write_csv(ROOT/'compression_anchors.csv',frontier)
    return dict(outcome='completed',real_training_commands=len(receipts),curve_observations=len(curves),
        timing='No inferred per-update or validation crossing wall-clock time',teacher_screen='blocked_missing_inputs')


def benchmark_task(half=False):
    c,ev,torch,cj=setup();parent=c['parents'][0];model=model_for(c,parent,torch,cj)
    dtype=torch.float16 if half else torch.float32
    # Record strict settings after model import (which can change global flags).
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model=model.to(dtype).eval();train=cj.pseudo(c['spec'],False)
    x=train[c['train_feature_indices'][0]]['image'][None].cuda().float()
    x=torch.nn.functional.interpolate(x,size=(512,512),mode='bilinear',align_corners=False).to(dtype)
    artifact=ROOT/f'R4_{"FP16" if half else "FP32"}.torchscript.pt'
    with torch.inference_mode():
        traced=torch.jit.trace(model,x,check_trace=True);torch.jit.save(traced,str(artifact))
        del traced,model;torch.cuda.empty_cache()
        model=torch.jit.load(str(artifact),map_location='cuda').eval()
        for _ in range(20):model(x)
        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(100):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();y=model(x);b.record();torch.cuda.synchronize();times.append(a.elapsed_time(b));del y
        peak=torch.cuda.max_memory_allocated();val,_=cj.loaders(c['spec']);total=np.zeros((2,3),np.int64)
        for batch in val:
            z=model(batch['image'].cuda().to(dtype)).argmax(1).cpu().numpy()
            for pred,mask in zip(z,batch['mask'].numpy()):total+=counts(pred,mask)
    row=dict(outcome='completed',graph='R4',seed=42,precision='FP16' if half else 'FP32',engine='torchscript_cuda',
        parameters=12230,artifact_path=str(artifact),artifact_sha256=q.file_hash(artifact),deployed_artifact_bytes=artifact.stat().st_size,
        p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),warmups=20,repetitions=100,
        fresh_process=True,teacher_loaded=False,baseline_allocator_bytes=baseline,peak_allocator_bytes=peak,
        incremental_allocator_bytes=peak-baseline,input_shape=[1,6,512,512],timing='synchronized forward only',
        backend=ev.backend_record('deployment_tf32_off'),hardware=torch.cuda.get_device_name(0),edge_latency='unmeasured',
        source_scores=ev.scores(total.tolist()),source_counts=total.tolist())
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--run-dir',default=str(ROOT));p.add_argument('--out',required=True)
    a=p.parse_args();start=time.perf_counter()
    if a.task=='recover':result=recover_task()
    elif a.task=='headroom':result=headroom_task()
    elif a.task in ['cache','smoke']:result=cache_task(a.task=='smoke')
    elif a.task=='adapt_fp':result=adapt_task(False)
    elif a.task=='adapt_q':result=adapt_task(True)
    elif a.task in ['bench_fp32','bench_fp16']:result=benchmark_task(a.task=='bench_fp16')
    elif a.task=='collect':
        import collect_v8
        collect_v8.collect('scheduled');result={'outcome':'completed'}
    else:raise ValueError(a.task)
    result.update(command_task=a.task,completed_at=q.now(),elapsed_seconds=time.perf_counter()-start)
    q.atomic_json(a.out,result)


if __name__=='__main__':main()
