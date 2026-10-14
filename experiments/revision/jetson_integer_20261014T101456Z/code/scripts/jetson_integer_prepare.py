#!/usr/bin/env python3
"""Bounded workstation experiments and portable Jetson integer-inference bundle."""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.revision_queue import atomic_json, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def initialize():
    root = REPO/'experiments/revision'/('jetson_integer_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir()
    code = root/'code'
    for name in ['models', 'quantization', 'utils', 'data', 'training', 'distillation']:
        shutil.copytree(REPO/name, code/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['scripts', 'experiments']:
        (code/name).mkdir()
    for name in ['scripts/jetson_integer_prepare.py', 'experiments/revision_queue.py', 'config.py']:
        shutil.copy2(REPO/name, code/name)
    shutil.copy2(REPO/'jetson_repro/integer_tools.py', code/'integer_tools.py')
    shutil.copy2(REPO/'jetson_repro/data/sen1floods11_nodep.py', code/'s11_nodep.py')
    rungs = REPO/'experiments/revision/capacity_v4_20261008/code/v5/rungs.py'
    shutil.copy2(rungs, code/'rungs.py')
    deps = REPO.parent/'_deps/efficientvit'
    shutil.copytree(deps/'efficientvit', code/'efficientvit', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(deps/'LICENSE', code/'EFFICIENTVIT_LICENSE')
    spec = read(REPO/'experiments/revision/capacity_v7_20261011T210730Z/training_spec.json')
    reuse = read(REPO/'experiments/revision/capacity_v7_20261011T210730Z/reuse_index.json')
    parent = next(r for r in reuse['records'] if r['graph']=='R4' and r['seed']==42 and r['family']=='R4_LKD_selected')
    models = {
        'R4': dict(checkpoint=parent['best_path'], sha256=parent['best_sha256'],
                   motivation='Quantize every convolution including the output head of the 12,230-parameter LKD student.'),
        'B0': dict(checkpoint='/data1/schmalstieg/flooddistill/runs_repro/pseudo_evit_relu6_ofa_N2500_s42/best.pt',
                   motivation='ReLU6 EfficientViT-B0 OFA; BN folding and all-convolution QDQ, then residual/attention QDQ.')}
    (root/'bundle/tools').mkdir(parents=True)
    shutil.copy2(code/'integer_tools.py', root/'bundle/tools/integer_tools.py')
    for name, row in models.items():
        actual = file_hash(row['checkpoint'])
        if row.get('sha256') and actual != row['sha256']:
            raise RuntimeError('Parent checkpoint hash changed')
        row['sha256'] = actual
        target = root/'bundle'/name/'checkpoint_float.pt'
        target.parent.mkdir()
        # Export weights only. Never amend the historical optimizer/receipt.
        import torch
        payload = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
        torch.save({'model': payload['model'], 'parent_sha256': actual}, target)
        row['portable_checkpoint'] = str(target)
        row['portable_sha256'] = file_hash(target)
    protocol = dict(schema_version=1, created=datetime.now(timezone.utc).isoformat(), root=str(root),
        repository=str(REPO), branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        dirty_state=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
        prompt_sha256=file_hash(REPO/'prompts/jetson_prompt.md'), models=models,
        data_root=spec['roots']['s11'], pseudo=spec['pseudo'], pseudo_index=spec['index'],
        pseudo_index_sha256=file_hash(spec['index']), band_order=['B2','B3','B4','B8A','B11','B12'],
        normalization_mean=[.1413,.138,.1235,.309,.2045,.1191],
        normalization_std=[.0741,.0737,.0869,.118,.0977,.0766],
        input_shape=[1,6,512,512], ignore_index=-1, output='two-class logits, argmax after original bilinear interpolation',
        calibration=dict(dataset='GDACS original N2500 TRAIN; S11 TRAIN alternative',
            count=32, seed=1729, order='smallest SHA256 filename',
            geometry='GDACS native normalized224 bilinear to512; S11 native512 unchanged', labels_used=False),
        evaluation='All 89 S11 validation scenes; source selection only; external datasets untouched',
        ptq_variants=[dict(id='minmax_pc',method='MinMax',per_channel=True,symmetric=True,operators=['Conv','MatMul','Add']),
                      dict(id='percentile_pc',method='Percentile',per_channel=True,symmetric=True,operators=['Conv','MatMul','Add']),
                      dict(id='minmax_pt',method='MinMax',per_channel=False,symmetric=True,operators=['Conv','MatMul','Add']),
                      dict(id='s11_minmax_pc',method='MinMax',per_channel=True,symmetric=True,operators=['Conv','MatMul','Add'],calibration='s11'),
                      dict(id='asymmetric_ort',method='MinMax',per_channel=True,symmetric=False,operators=['Conv','MatMul','Add'])],
        # Selection is prospective: no external accuracy or latency selects recipes.
        acceptable_degradation=dict(pooled_mIoU_max_drop=.01, equal_event_mIoU_max_drop=.01,
                                   water_IoU_max_drop=.02, units='fraction; engineering criterion, not equivalence test'),
        qat=dict(trigger='No symmetric deployment candidate meets all three source accuracy tolerances',
            max_updates=128, batch_size=4, lr=.0001, optimizer='AdamW', weight_decay=.05,
            seed=1729, bn='folded before calibration/training', gradients='clip5',
            objective='focal gamma2 on GDACS cached teacher hard masks; separate QAT adaptation, no live KD/OFA retraining',
            quantizers='all-conv including head, symmetric per-tensor activations, per-channel weights; float residual/Resize/attention remain disclosed',
            checkpoint_rule='Final update only; parent retained; no target labels',
            smoke_updates=1, smoke_is_not_accuracy_evidence=True),
        int4_gate='Only after native INT8 execution and minor source accuracy degradation, including integer layer audit; ORT/QDQ alone does not open gate',
        target=dict(python='3.8.20',torch='1.13.0a0+d0d6b1f2.nv22.10',jetpack='5.0.2',cuda='11.4',
                    tensorrt='8.4.1.5',onnx='1.17.0',onnxruntime='1.19.2'),
        versions={n:importlib.metadata.version(n) for n in ['torch','onnx','onnxruntime','tensorrt','numpy']},
        efficientvit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=deps,text=True).strip(),
        code_hashes={str(p.relative_to(code)):file_hash(p) for p in code.rglob('*') if p.is_file()},
        latency=dict(backend='workstation ORT CPU only',threads=2,warmup=20,iterations=100,batch=1,
                     excludes='disk/preprocessing', jetson='pending hardware; no workstation engine transfer'))
    atomic_json(root/'protocol.json',protocol)
    shutil.copy2(root/'protocol.json',root/'bundle/protocol.json')
    jobs=[]
    for action, name, deps in [('assets','assets',[]),('model','R4',['assets']),('model','B0',['assets']),('collect','collect',['R4','B0'])]:
        argv=[sys.executable,str(code/'scripts/jetson_integer_prepare.py'),action,'--root',str(root)]
        if action=='model':argv += ['--model',name]
        jobs.append(dict(id=name,argv=argv,cwd=str(code),deps=deps,max_seconds=7200,
                         inputs=[dict(path=str(root/'protocol.json'),sha256=file_hash(root/'protocol.json'))],
                         outputs=[dict(path=str(root/'receipts'/(name+'.json')),type='json',nonempty=True)]))
    atomic_json(root/'queue_manifest.json',dict(schema_version=1,jobs=jobs,
        gpu_lock=str(root/'unused_CPU_only.lock'),env={'PYTHONDONTWRITEBYTECODE':'1','NO_ALBUMENTATIONS_UPDATE':'1','CUDA_VISIBLE_DEVICES':''}))
    print(root)


def load_model(name, protocol):
    import torch
    if name=='R4':
        from rungs import ContextConv
        model=ContextConv()
    else:
        from models.student import build_student
        model=build_student('efficientvit_b0_relu6',pretrained=False,in_channels=6,num_classes=2)
    row=protocol['models'][name]
    if file_hash(row['portable_checkpoint'])!=row['portable_sha256']:
        raise RuntimeError('Portable checkpoint changed')
    model.load_state_dict(torch.load(row['portable_checkpoint'],map_location='cpu',weights_only=True)['model'],strict=True)
    return model.eval()


def assets(root, protocol):
    import torch
    from s11_nodep import Sen1Floods11Dataset
    from quantization.jetson_integer import session
    calib=root/'bundle/calibration'; calib.mkdir(parents=True,exist_ok=True)
    index=read(protocol['pseudo_index'])
    if file_hash(protocol['pseudo_index'])!=protocol['pseudo_index_sha256']:
        raise RuntimeError('Cohort identity changed')
    ids=sorted(index['files'],key=lambda n:hashlib.sha256(n.encode()).hexdigest())[:protocol['calibration']['count']]
    entries=[]
    for name in ids:
        path=Path(protocol['pseudo'])/name
        item=torch.load(path,map_location='cpu',weights_only=True)
        x=torch.nn.functional.interpolate(item['image'][None].float(),size=(512,512),mode='bilinear',align_corners=False)[0].numpy()
        np = __import__('numpy')
        np.save(calib/(Path(name).stem+'.npy'),x)
        entries.append(dict(file=name,source_sha256=file_hash(path),asset_sha256=file_hash(calib/(Path(name).stem+'.npy'))))
    s11cal=root/'bundle/calibration_s11';s11cal.mkdir(exist_ok=True)
    train=Sen1Floods11Dataset(protocol['data_root'],split='train')
    tids=sorted(range(len(train)),key=lambda i:hashlib.sha256(Path(train.image_files[i]).name.encode()).hexdigest())[:len(ids)]
    sentries=[]
    for i in tids:
        item=train[i];name=item['filename'];path=s11cal/(Path(name).stem+'.npy')
        np.save(path,item['image'].numpy())
        sentries.append(dict(file=name,source_sha256=file_hash(train.image_files[i]),asset_sha256=file_hash(path)))
    val=Sen1Floods11Dataset(protocol['data_root'],split='val')
    assert len(val)==89 and len(train)==252
    # Verify this lightweight Jetson loader against the established TerraTorch
    # workstation evaluator before adopting portable arrays.
    from data.sen1floods11 import get_dataloaders
    _, reference, _ = get_dataloaders(protocol['data_root'],batch_size=1,num_workers=0,img_size=224,eval_img_size=512,drop_last=False)
    assert [Path(p).name for p in reference.dataset.image_files]==[Path(p).name for p in val.image_files]
    parity=[]
    for i in [0,44,88]:
        a,b=val[i],reference.dataset[i]
        assert torch.equal(a['mask'],b['mask']) and torch.allclose(a['image'],b['image'],atol=1e-6,rtol=1e-6)
        parity.append(dict(index=i,max_abs=float((a['image']-b['image']).abs().max()),mask_equal=True))
    target=root/'bundle/source_val';target.mkdir(exist_ok=True)
    ventries=[]
    for i in range(len(val)):
        item=val[i];name=item['filename'];path=target/(Path(name).stem+'.npz')
        np.savez_compressed(path,image=item['image'].numpy(),mask=item['mask'].numpy(),event=name.split('_')[0])
        ventries.append(dict(file=name,asset_sha256=file_hash(path),image_sha256=file_hash(val.image_files[i]),label_sha256=file_hash(val.label_files[i])))
    # Preserve the original cached training images/masks for optional QAT. No
    # external labels or teacher-empty selection.
    train_assets=root/'qat_train';train_assets.mkdir(exist_ok=True)
    qids=sorted(index['files'],key=lambda n:hashlib.sha256(('qat1729'+n).encode()).hexdigest())[:512]
    qentries=[]
    for name in qids:
        path=Path(protocol['pseudo'])/name
        a=torch.load(path,map_location='cpu',weights_only=True)
        np.savez_compressed(train_assets/(Path(name).stem+'.npz'),image=a['image'].float().numpy(),mask=a['mask'].numpy())
        qentries.append(dict(file=name,source_sha256=file_hash(path)))
    result=dict(status='complete',calibration=entries,calibration_s11=sentries,validation=ventries,
                qat_train=qentries,lightweight_loader_parity=parity,labels_for_calibration=False,
                train_split=252,source_validation=89,external_evaluation=False)
    atomic_json(root/'receipts/assets.json',result)
    atomic_json(root/'bundle/assets_manifest.json',result)


def evaluate(path, assets_dir, output):
    import numpy as np
    from quantization.jetson_integer import session,scores
    sess=session(path,Path(output).with_suffix('.optimized.onnx'))
    rows=[];conf=np.zeros((2,2),np.int64)
    for asset in sorted(Path(assets_dir).glob('*.npz')):
        with np.load(asset,allow_pickle=False) as a:
            x=a['image'][None].astype(np.float32);mask=a['mask'];event=str(a['event'])
        z=sess.run(None,{'image':x})[0]
        if z.shape!=(1,2,512,512) or not np.isfinite(z).all():raise RuntimeError('Invalid logits: '+asset.name)
        pred=z[0].argmax(0);valid=(mask>=0)&(mask<2)
        c=np.bincount((mask[valid]*2+pred[valid]).ravel(),minlength=4).reshape(2,2)
        conf+=c;rows.append(dict(file=asset.name,event=event,confusion=c.tolist()))
    for _ in range(20):sess.run(None,{'image':x})
    times=[]
    for _ in range(100):
        start=time.perf_counter();sess.run(None,{'image':x});times.append(1000*(time.perf_counter()-start))
    import onnx
    from collections import Counter
    optimized=onnx.load(Path(output).with_suffix('.optimized.onnx'))
    result=dict(status='evaluated',rows=rows,**scores(conf),latency_cpu_ms_mean=float(np.mean(times)),
        latency_cpu_ms_p50=float(np.median(times)),latency_cpu_ms_p95=float(np.quantile(times,.95)),
        warmup=20,iterations=100,threads=2,peak_process_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        memory_note='Process high-water RSS includes export/calibration and session; not isolated model memory',
        optimized_operators=dict(Counter(n.op_type for n in optimized.graph.node)),model_bytes=Path(path).stat().st_size,
        backend='ORT CPUExecutionProvider; kernel coverage audited separately; not Jetson latency')
    atomic_json(output,result)
    return result


def qat(root,name,protocol,train):
    import torch
    import numpy as np
    from quantization.manual import fold_bn
    from quantization.qat import prepare_qat,QATConv2d,LiteMLAQAT,_set_module
    from models.compression_quant import FixedQATConv2d,load_compression_qat
    from quantization.jetson_integer import graph_audit,session
    torch.manual_seed(1729)
    model=fold_bn(load_model(name,protocol))
    training=sorted((root/'qat_train').glob('*.npz'))
    batches=[]
    for p in training[:8]:
        with np.load(p,allow_pickle=False) as a:batches.append({'image':torch.from_numpy(a['image'].copy())[None]})
    q=prepare_qat(model,batches,skip_last_n=0,calibration_type='per_tensor')
    # The historical LiteMLA path simulates only attention mm1 at inference.
    # Keep it explicit as floating here so export/reload cannot silently switch
    # the attention quantizers on/off. Attention PTQ is a separate graph arm.
    for n,m in list(q.named_modules()):
        if isinstance(m,LiteMLAQAT):_set_module(q,n,m.litemla)
    for n,m in list(q.named_modules()):
        if isinstance(m,QATConv2d):_set_module(q,n,FixedQATConv2d(m,m._act_scale,m._act_zp,m._w_scale,weight_bits=8,activation_bits=8))
    q.eval() # BN already folded; no Dropout. Parameters still train normally.
    optimizer=torch.optim.AdamW(q.parameters(),lr=protocol['qat']['lr'],weight_decay=.05)
    updates=128 if train else 1; losses=[]; start=time.monotonic()
    for step in range(updates):
        samples=[]
        for p in training[step*4:step*4+4]:
            with np.load(p,allow_pickle=False) as a:samples.append((torch.from_numpy(a['image'].copy()),torch.from_numpy(a['mask'].copy()).long()))
        x=torch.stack([s[0] for s in samples]);mask=torch.stack([s[1] for s in samples]);valid=(mask>=0)&(mask<2)
        z=q(x);ce=torch.nn.functional.cross_entropy(z,mask,ignore_index=-1,reduction='none')
        loss=(((1-torch.exp(-ce[valid]))**2)*ce[valid]).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite QAT loss')
        optimizer.zero_grad();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in q.parameters())
        torch.nn.utils.clip_grad_norm_(q.parameters(),5);optimizer.step();losses.append(float(loss))
        if step%16==0:print(name,'QAT',step,'/',updates,'loss',losses[-1],flush=True)
    target=root/'bundle'/name/('qat128' if train else 'qat_smoke');target.mkdir(exist_ok=True)
    checkpoint=target/'checkpoint_qat.pt';torch.save({'model':q.state_dict(),'recipe':protocol['qat'],'updates':updates},checkpoint)
    restored=load_compression_qat(model,q.state_dict()).eval()
    with torch.no_grad():
        expected=q(batches[0]['image']);actual=restored(batches[0]['image'])
    assert torch.equal(expected,actual),'QAT reload altered quantizer state'
    path=target/'model_qdq.onnx'
    torch.onnx.export(q,torch.zeros(1,6,512,512),str(path),opset_version=13,input_names=['image'],output_names=['logits'],dynamo=False)
    import onnx
    m=onnx.load(path);m.ir_version=min(m.ir_version,10)
    # Folding constants and shape arithmetic is required by TRT8.4.
    from onnxsim import simplify
    m,ok=simplify(m);assert ok
    onnx.save(m,path);audit=graph_audit(path)
    test=__import__('numpy').load(next((root/'bundle/calibration').glob('*.npy')))[None]
    with torch.no_grad():expected=q(torch.from_numpy(test)).numpy()
    actual=session(path).run(None,{'image':test})[0]
    error=float(np.max(np.abs(expected-actual)))
    if not np.allclose(expected,actual,atol=1e-3,rtol=1e-3):raise RuntimeError('QAT ONNX parity failed: '+str(error))
    result=dict(status='complete',updates=updates,smoke_only=not train,losses=losses,elapsed_seconds=time.monotonic()-start,
        reload_exact=True,onnx_max_abs=error,audit=audit,checkpoint_sha256=file_hash(checkpoint),
        limitation='Only Conv rounding is trained; residual/Resize/attention stay floating. Strict device certification required.')
    if train:result['evaluation']=evaluate(path,root/'bundle/source_val',target/'evaluation.json')
    atomic_json(target/'receipt.json',result)
    return result


def experiment(root,name,protocol):
    import torch
    import numpy as np
    from quantization.onnx_qdq import export_float_onnx
    from quantization.jetson_integer import ptq,graph_audit,compare_metrics,session
    from onnxsim import simplify
    import onnx
    model=load_model(name,protocol)
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==12230 if name=='R4' else parameters>600000
    target=root/'bundle'/name
    path=target/'model_float.onnx'
    export_float_onnx(model,torch.zeros(1,6,512,512),str(path),opset=13,static=True)
    m,ok=simplify(onnx.load(path));assert ok;m.ir_version=min(m.ir_version,10);onnx.save(m,path)
    calibration=np.stack([np.load(p,allow_pickle=False) for p in sorted((root/'bundle/calibration').glob('*.npy'))])
    alternative=np.stack([np.load(p,allow_pickle=False) for p in sorted((root/'bundle/calibration_s11').glob('*.npy'))])
    with torch.no_grad():expected=model(torch.from_numpy(calibration[:1])).numpy()
    actual=session(path).run(None,{'image':calibration[:1]})[0]
    if not np.allclose(expected,actual,atol=1e-3,rtol=1e-3):raise RuntimeError('Float export parity failure')
    baseline=evaluate(path,root/'bundle/source_val',target/'float_evaluation.json')
    bstats=compare_metrics(baseline['rows'],baseline['rows'])
    results=[]
    for config in protocol['ptq_variants']:
        output=target/(config['id']+'.onnx');start=time.monotonic()
        result=dict(config=config,status='failed',motivation='Compare calibration/weight granularity under fixed source validation; all Conv heads included')
        try:
            images=alternative if config.get('calibration')=='s11' else calibration
            args={k:v for k,v in config.items() if k not in ('id','calibration')}
            result['audit']=ptq(path,output,images,**args)
            result['evaluation']=evaluate(output,root/'bundle/source_val',target/(config['id']+'_evaluation.json'))
            stats=compare_metrics(result['evaluation']['rows'],baseline['rows'])
            result['comparison']=stats
            result['acceptable_accuracy']=stats['pooled_delta']>=-.01 and stats['equal_event_delta']>=-.01 and result['evaluation']['water_IoU']>=baseline['water_IoU']-.02
            result['status']='evaluated'
        except Exception as error:
            result['reason']=repr(error);result['traceback']=traceback.format_exc()
        result['elapsed_seconds']=time.monotonic()-start
        atomic_json(target/(config['id']+'_receipt.json'),result);results.append(result)
        print(name,config['id'],result['status'],result.get('comparison',{}).get('pooled_delta'),flush=True)
    candidates=[r for r in results if r['status']=='evaluated' and r['audit']['jp50_qdq_compatible']]
    selected=max(candidates,key=lambda r:r['comparison']['pooled']['mIoU']) if candidates else None
    need_qat=not any(r['acceptable_accuracy'] for r in candidates)
    qresult=qat(root,name,protocol,train=need_qat)
    record=dict(status='complete',name=name,parameters=parameters,float=baseline,float_event=bstats,
        float_export_max_abs=float(np.max(np.abs(expected-actual))),float_audit=graph_audit(path),variants=results,
        selected=selected['config']['id'] if selected else None,qat=qresult,
        int4='blocked_native_INT8_not_yet_demonstrated',native_TRT='blocked_workstation_builder_CUDA_error35',
        full_integer_execution='not_demonstrated',new_framework=False,external_evaluation=False)
    atomic_json(root/'receipts'/(name+'.json'),record)


def collect(root,protocol):
    rows=[]
    for name in protocol['models']:
        result=read(root/'receipts'/(name+'.json'))
        rows.append(dict(model=name,variant='float',parameters=result['parameters'],mIoU=result['float']['mIoU'],water_IoU=result['float']['water_IoU'],equal_event_mIoU=result['float_event']['equal_event_mIoU'],delta=0,CPU_ms=result['float']['latency_cpu_ms_mean'],bytes=result['float']['model_bytes'],status='float'))
        for r in result['variants']:
            ev=r.get('evaluation',{});co=r.get('comparison',{})
            rows.append(dict(model=name,variant=r['config']['id'],parameters=result['parameters'],mIoU=ev.get('mIoU'),water_IoU=ev.get('water_IoU'),equal_event_mIoU=co.get('equal_event_mIoU'),delta=co.get('pooled_delta'),CPU_ms=ev.get('latency_cpu_ms_mean'),bytes=ev.get('model_bytes'),status=r['status']))
    with (root/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(root/'receipts/collect.json',dict(status='complete',rows=rows,native_INT8=False,int4='blocked',jetson='pending_device_execution'))
    # The device bundle checks data, source checkpoints and exports without
    # requiring access to the workstation absolute paths.
    atomic_json(root/'bundle/SHA256.json',{str(p.relative_to(root/'bundle')):file_hash(p) for p in (root/'bundle').rglob('*') if p.is_file() and p.name!='SHA256.json'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','assets','model','collect'])
    parser.add_argument('--root',type=Path)
    parser.add_argument('--model',choices=['B0','R4'])
    args=parser.parse_args()
    if args.action=='init':return initialize()
    root=args.root.resolve();protocol=read(root/'protocol.json')
    code=root/'code'
    sys.path.insert(0,str(code))
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    # Every source file used by a queued worker must retain its frozen identity.
    for name,digest in protocol['code_hashes'].items():
        if file_hash(code/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    if args.action=='assets':assets(root,protocol)
    elif args.action=='model':experiment(root,args.model,protocol)
    else:collect(root,protocol)


if __name__=='__main__':main()
