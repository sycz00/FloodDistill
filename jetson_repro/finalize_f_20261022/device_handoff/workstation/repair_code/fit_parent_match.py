#!/usr/bin/env python3
"""Bounded seed91 QAT matching of the frozen genuine parent on the same512 images."""
import copy,time,traceback
from diagnose_unfolded import ROOT,OLD,q,x,rows,score
from fit_cpu import evaluate_export

def main():
    seed=91;origin=ROOT/'fit_cpu_s91';previous=x.read(origin/'RESULT.json');assert previous['status']=='SOURCE_ACCURACY_NOT_RECOVERED'
    assert x.read(ROOT/'mixed_exceptions_s91_v2/RESULT.json')['status']=='MIXED_EXCEPTIONS_NOT_RECOVERED'
    assert x.read(ROOT/'continue_cpu_s91/RESULT.json')['status']=='SOURCE_ACCURACY_NOT_RECOVERED'
    protocol=x.read(ROOT/'PARENT_MATCH_PROTOCOL.json')
    for pin in protocol['code']:assert x.sha(pin['path'])==pin['sha256']
    assert x.sha(protocol['starting_checkpoint']['path'])==protocol['starting_checkpoint']['sha256']
    out=ROOT/'parent_match_cpu_s91';out.mkdir(exist_ok=False);start=time.time();result=dict(seed=seed,attempt_resolved=True,status='FAILED_EXECUTION',updates=1280,additional_updates=0)
    try:
        entry=next(v for v in x.read(OLD/'workstation/PROTOCOL.json')['entries'] if v['seed']==seed)
        torch,parent,ji,ds,inv,ev=q.runtime(x.read(entry['config']['path']));items=list(q.source_items(ds,inv,ev));original=x.read(OLD/'workstation/B0_s91/FLOAT_EVALUATION.json')['rows'];assert rows(parent,items,torch)==original
        from mixed_smooth import InputScaledConv
        from models.compression_quant import FixedQATConv2d,freeze_batch_norm,load_compression_qat
        from quantization.qat import _set_module
        saved=torch.load(origin/'checkpoint_final.pt',map_location='cpu',weights_only=False);target=copy.deepcopy(parent).eval()
        for rec in saved['scales']['records']:_set_module(target,rec['name'],InputScaledConv(target.get_submodule(rec['name']),torch.tensor(rec['scale'])))
        target=load_compression_qat(target,saved['model']).eval();assert rows(target,items,torch)==x.read(origin/'SOURCE_HISTORY.json')[-1]['evaluation']['rows']
        for parameter in parent.parameters():parameter.requires_grad_(False)
        for filename in ('SCALES.json','FLOAT_CONSTRUCTION.json'):x.write(out/filename,x.read(origin/filename),True)
        inherited=x.read(x.read(OLD/'workstation/PROTOCOL.json')['inherited_protocol']['path']);load=q.streaming_samples(inherited);training=[load(rec) for rec in inherited['qat_train']];assert len(training)==512
        optimizer=torch.optim.AdamW([p for p in target.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-5);generator=torch.Generator().manual_seed(1729);history=[];losses=[];best=None
        for epoch in range(1,5):
            target.train();freeze_batch_norm(target)
            for m in target.modules():
                if isinstance(m,FixedQATConv2d):m.fake_quant_enabled=True
            order=torch.randperm(512,generator=generator).tolist()
            for step in range(128):
                batch=[training[i] for i in order[step*4:step*4+4]];image=torch.stack([v[0] for v in batch]);mask=torch.stack([v[1] for v in batch]);assert ((mask>=0)&(mask<2)).any()
                with torch.no_grad():reference=parent(image)
                logits=target(image);valid=(mask>=0)&(mask<2);temperature=2.
                kl=torch.nn.functional.kl_div(torch.log_softmax(logits/temperature,dim=1),torch.softmax(reference/temperature,dim=1),reduction='none').sum(1)
                margin=(logits[:,1]-logits[:,0])-(reference[:,1]-reference[:,0])
                loss=temperature**2*kl[valid].mean()+.01*margin[valid].square().mean();assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in target.parameters());torch.nn.utils.clip_grad_norm_(target.parameters(),1);optimizer.step();losses.append(float(loss.detach()));result['updates']+=1;result['additional_updates']+=1
                if step%16==0:x.write(out/'PROGRESS.json',dict(epoch=epoch,updates=result['updates'],additional_updates=result['additional_updates'],max_additional_updates=512,elapsed_seconds=time.time()-start))
            target.eval();measured=score(target,items,torch,ji,original);measured.update(epoch=epoch,updates=result['updates']);history.append(measured);x.write(out/'SOURCE_HISTORY.json',history);print(epoch,measured['drops'],measured['accuracy_pass'],flush=True)
            state=dict(model=target.state_dict(),scales=saved['scales'],updates=result['updates'],protocol=protocol,parent_protocol=saved['protocol']);torch.save(state,out/'checkpoint_latest.pt')
            if not measured['accuracy_pass']:continue
            dest=out/f'candidate_epoch{epoch}';dest.mkdir();gate=evaluate_export(target,dest/'qat.onnx',dest,items,torch,ji,original);x.write(dest/'SOURCE_GATE.json',gate,True)
            if not gate['accuracy_pass']:continue
            parity=[dict(file=rec['name'],**ji.qat_export_parity(target,dest/'qat.onnx',image)) for image,mask,rec in items];strict=all(p['strict_logit_parity_pass'] for p in parity);structure=all(p.get('strict_logit_parity_pass') or p.get('unrounded_graph_parity_pass') for p in parity);x.write(dest/'EXPORT_PARITY.json',dict(strict_all89=strict,unrounded_structure_all89=structure,rows=parity),True)
            if not structure:continue
            torch.save(state,out/'checkpoint_selected.pt');best=dict(source_gate=gate,checkpoint=x.pin(out/'checkpoint_selected.pt'),strict_logit_parity_all89=strict,unrounded_structure_all89=structure,export_parity=x.pin(dest/'EXPORT_PARITY.json'));break
        torch.save(dict(model=target.state_dict(),scales=saved['scales'],updates=result['updates'],protocol=protocol),out/'checkpoint_final.pt')
        x.write(out/'TRAINING_RECEIPT.json',dict(updates=result['updates'],parent_updates=1280,additional_updates=result['additional_updates'],losses=losses,epochs=len(history),head_float=True,attention_matmul_float=True,quantized_convs=56,native_execution=False,optimizer='fresh AdamW1e-5 WD1e-5; constant LR; frozen BN; no new float warmup',calibration='fixed saved grids, no recalibration',loss='KL temperature2 to frozen genuine parent plus0.01 squared logit-margin error; same512 images; masks provide valid support only; no new foundation teacher inference'),True)
        result.update(status='SOURCE_ACCURACY_RECOVERED' if best else 'SOURCE_ACCURACY_NOT_RECOVERED',acceptable=bool(best))
        if best:result['selected']=best
    except Exception as error:result.update(error=repr(error),traceback=traceback.format_exc())
    finally:result['elapsed_seconds']=time.time()-start;x.write(out/'RESULT.json',result,True);print(result['status'],flush=True)
if __name__=='__main__':main()
