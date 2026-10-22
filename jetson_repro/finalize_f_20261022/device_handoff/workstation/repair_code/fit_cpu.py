#!/usr/bin/env python3
"""Bounded corrected mixed QAT. Source-only gates; no calibration after fitting."""
import argparse,copy,json,os,sys,time,traceback,math
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLCONFIGDIR']='/tmp/b0_mixed_repair_mpl'
from diagnose_unfolded import ROOT,OLD,REPO,q,x,rows,score
import numpy as np

def evaluate_export(model,path,out,items,torch,ji,parent):
    import onnx
    from onnxsim import simplify
    torch.onnx.export(model,torch.zeros(1,6,512,512),str(path),opset_version=13,input_names=['image'],output_names=['logits'],dynamo=False)
    graph,okay=simplify(onnx.load(path));assert okay;graph.ir_version=min(graph.ir_version,10);onnx.save(graph,path)
    # TensorRT8.4 requires a separately identified linear Resize deployment variant.
    from subprocess import run,DEVNULL,STDOUT
    cubic=any(n.op_type=='Resize' and any(a.name=='mode' and a.s==b'cubic' for a in n.attribute) for n in graph.graph.node)
    deployment=out/'qat_linear.onnx' if cubic else path
    if cubic:
        with (out/'linear_transform.log').open('w') as log:run([sys.executable,str(REPO/'jetson_repro/make_trt84_variant.py'),'--model',str(path),'--out',str(deployment),'--receipt',str(out/'LINEAR_TRANSFORM.json'),'--resize-mode','linear'],check=True,stdin=DEVNULL,stdout=log,stderr=STDOUT)
    session=ji.session(deployment);rs=[]
    for image,mask,rec in items:
        z=session.run(None,{'image':image})[0];assert np.isfinite(z).all();valid=(mask>=0)&(mask<2);pred=z[0].argmax(0);cm=np.bincount((mask[valid]*2+pred[valid]).astype('int64'),minlength=4).reshape(2,2);rs.append(dict(file=rec['name'],event=rec['event'],confusion=cm.tolist()))
    a=q.evaluate_rows(rs,ji);b=q.evaluate_rows(parent,ji);drops={k:b[k]-a[k] for k in ('mIoU','equal_event_mIoU','water_IoU')}
    result=dict(evaluation=a,baseline=b,drops=drops,accuracy_pass=all(v<=.02 for v in drops.values()),onnx=x.pin(deployment),graph_variant='linear Resize, scalar activation QDQ, per-channel weight QDQ, floating attention and classifier',training_graph=x.pin(path))
    return result

def main(seed):
    out=ROOT/f'fit_cpu_s{seed}';out.mkdir(exist_ok=False);result=dict(seed=seed,status='FAILED_EXECUTION',attempt_resolved=True,updates=0);started=time.time()
    try:
        protocol=x.read(ROOT/'FIT_PROTOCOL.json')
        for p in protocol['code']:assert x.sha(p['path'])==p['sha256']
        plan=x.read(OLD/'workstation/PROTOCOL.json');entry=next(v for v in plan['entries'] if v['seed']==seed)
        torch,model,ji,ds,inv,ev=q.runtime(x.read(entry['config']['path']));items=list(q.source_items(ds,inv,ev));parent=x.read(OLD/f'workstation/B0_s{seed}/FLOAT_EVALUATION.json')['rows'];assert rows(model,items,torch)==parent
        from mixed_smooth import prepare_parent,prepare_quant
        from models.compression_quant import FixedQATConv2d,freeze_batch_norm,load_compression_qat
        images=[torch.from_numpy(v[0]) for v in items];smooth,scales=prepare_parent(model,images);x.write(out/'SCALES.json',scales,True);checks=[]
        with torch.no_grad():
            for image,mask,rec in items:
                a=model(torch.from_numpy(image)).numpy();b=smooth(torch.from_numpy(image)).numpy();checks.append(dict(file=rec['name'],passed=bool(np.allclose(a,b,atol=.001,rtol=.001)),max_abs=float(abs(a-b).max())))
        construction=all(v['passed'] for v in checks);x.write(out/'FLOAT_CONSTRUCTION.json',dict(passed=construction,atol=.001,rtol=.001,images=89,rows=checks),True)
        if not construction:result.update(status='BLOCKED_CONSTRUCTION');return
        qat=prepare_quant(smooth,images);head=scales['head'];assert not isinstance(qat.get_submodule(head),FixedQATConv2d)
        assert not any(type(m).__name__ in ('LiteMLAQAT','QATMatMul') for m in qat.modules())
        q0=score(qat,items,torch,ji,parent);x.write(out/'Q0.json',q0,True)
        inherited=x.read(plan['inherited_protocol']['path']);load=q.streaming_samples(inherited);training=[load(rec) for rec in inherited['qat_train']];assert len(training)==512
        optimizer=torch.optim.AdamW([p for p in qat.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-5);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10);generator=torch.Generator().manual_seed(1729)
        losses=[];readouts=[];best=None
        for epoch in range(1,11):
            qat.train();freeze_batch_norm(qat)
            for mod in qat.modules():
                if isinstance(mod,FixedQATConv2d):mod.fake_quant_enabled=epoch>1
            order=torch.randperm(512,generator=generator).tolist()
            for step in range(128):
                batch=[training[i] for i in order[step*4:step*4+4]];im=torch.stack([r[0] for r in batch]);mask=torch.stack([r[1] for r in batch]);assert ((mask>=0)&(mask<2)).any()
                logits=qat(im);loss=torch.nn.functional.cross_entropy(logits,mask,ignore_index=-1);assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in qat.parameters());torch.nn.utils.clip_grad_norm_(qat.parameters(),1);optimizer.step();losses.append(float(loss.detach()));result['updates']+=1
                if step%16==0:x.write(out/'PROGRESS.json',dict(epoch=epoch,updates=result['updates'],max_updates=1280,loss=losses[-1],elapsed_seconds=time.time()-started))
            scheduler.step();qat.eval()
            for mod in qat.modules():
                if isinstance(mod,FixedQATConv2d):mod.fake_quant_enabled=True
            measured=score(qat,items,torch,ji,parent);measured.update(epoch=epoch,updates=result['updates']);readouts.append(measured);x.write(out/'SOURCE_HISTORY.json',readouts)
            print(seed,'epoch',epoch,'drops',measured['drops'],'accuracy',measured['accuracy_pass'],flush=True)
            torch.save(dict(model=qat.state_dict(),scales=scales,updates=result['updates'],protocol=protocol),out/'checkpoint_latest.pt')
            if epoch>=2 and measured['accuracy_pass']:
                export_dir=out/f'candidate_epoch{epoch}';export_dir.mkdir();actual=evaluate_export(qat,export_dir/'qat.onnx',export_dir,items,torch,ji,parent);x.write(export_dir/'SOURCE_GATE.json',actual,True)
                if actual['accuracy_pass']:
                    torch.save(dict(model=qat.state_dict(),scales=scales,updates=result['updates'],protocol=protocol),out/'checkpoint_selected.pt')
                    parities=[]
                    for image,mask,rec in items:
                        try:check=ji.qat_export_parity(qat,export_dir/'qat.onnx',image)
                        except RuntimeError as error:check=dict(strict_logit_parity_pass=False,unrounded_graph_parity_pass=False,error=str(error))
                        parities.append(dict(file=rec['name'],**check))
                    structural=all(p.get('strict_logit_parity_pass') or p.get('unrounded_graph_parity_pass') for p in parities);strict=all(p['strict_logit_parity_pass'] for p in parities)
                    x.write(export_dir/'EXPORT_PARITY.json',dict(strict_all89=strict,unrounded_structure_all89=structural,rows=parities),True)
                    if structural:
                        best=dict(source_gate=actual,checkpoint=x.pin(out/'checkpoint_selected.pt'),strict_logit_parity_all89=strict,unrounded_structure_all89=structural,export_parity=x.pin(export_dir/'EXPORT_PARITY.json'));break
        torch.save(dict(model=qat.state_dict(),scales=scales,updates=result['updates'],protocol=protocol),out/'checkpoint_final.pt')
        x.write(out/'TRAINING_RECEIPT.json',dict(updates=result['updates'],losses=losses,epochs=len(readouts),head_float=True,attention_matmul_float=True,quantized_convs=sum(isinstance(m,FixedQATConv2d) for m in qat.modules()),native_execution=False),True)
        if best:result.update(status='SOURCE_ACCURACY_RECOVERED',selected=best,acceptable=True)
        else:
            final=out/'final_export';final.mkdir();actual=evaluate_export(qat,final/'qat.onnx',final,items,torch,ji,parent);x.write(final/'SOURCE_GATE.json',actual,True);result.update(status='SOURCE_ACCURACY_NOT_RECOVERED',acceptable=False,final_source_gate=actual)
    except Exception as error:result.update(error=repr(error),traceback=traceback.format_exc())
    finally:result['elapsed_seconds']=time.time()-started;x.write(out/'RESULT.json',result,True);print(seed,result['status'],result['updates'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);main(p.parse_args().seed)
