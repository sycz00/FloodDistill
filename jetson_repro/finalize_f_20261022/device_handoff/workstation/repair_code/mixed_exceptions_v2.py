#!/usr/bin/env python3
"""Finite source-only precision fallback after the bounded QAT recipe fails."""
import copy,time,json
from diagnose_unfolded import ROOT,OLD,q,x,rows,score
from fit_cpu import evaluate_export

def main():
    seed=91;origin=ROOT/f'fit_cpu_s{seed}';previous=x.read(origin/'RESULT.json')
    assert previous['status']=='SOURCE_ACCURACY_NOT_RECOVERED'
    assert all(x.read(ROOT/f'fit_cpu_s{s}/RESULT.json')['attempt_resolved'] for s in (42,64,91,107))
    protocol=x.read(ROOT/'MIXED_EXCEPTION_PROTOCOL_v2.json')
    for pin in protocol['code']:assert x.sha(pin['path'])==pin['sha256']
    out=ROOT/f'mixed_exceptions_s{seed}_v2';out.mkdir(exist_ok=False);started=time.time();result=dict(seed=seed,updates=previous['updates'],attempt_resolved=True,status='FAILED_EXECUTION')
    try:
        entry=next(v for v in x.read(OLD/'workstation/PROTOCOL.json')['entries'] if v['seed']==seed)
        torch,parent,ji,ds,inv,ev=q.runtime(x.read(entry['config']['path']));items=list(q.source_items(ds,inv,ev));original=x.read(OLD/f'workstation/B0_s{seed}/FLOAT_EVALUATION.json')['rows'];assert rows(parent,items,torch)==original
        from mixed_smooth import InputScaledConv
        from models.compression_quant import load_compression_qat,FixedQATConv2d
        from quantization.qat import _set_module
        saved=torch.load(origin/'checkpoint_final.pt',map_location='cpu',weights_only=False);target=copy.deepcopy(parent).eval()
        for rec in saved['scales']['records']:_set_module(target,rec['name'],InputScaledConv(target.get_submodule(rec['name']),torch.tensor(rec['scale'])))
        target=load_compression_qat(target,saved['model']).eval();assert rows(target,items,torch)==x.read(origin/'SOURCE_HISTORY.json')[-1]['evaluation']['rows']
        histories=[];arms=[]
        names=[n for n,m in target.named_modules() if isinstance(m,FixedQATConv2d)]
        for policy in protocol['policies']:
            chosen=[n for n in names if any(token in n for token in policy['contains'])];assert chosen
            candidate=copy.deepcopy(target).eval()
            for name in chosen:
                mod=candidate.get_submodule(name);plain=torch.nn.Conv2d(mod.in_channels,mod.out_channels,mod.kernel_size,stride=mod.stride,padding=mod.padding,dilation=mod.dilation,groups=mod.groups,bias=mod.bias is not None,padding_mode=mod.padding_mode)
                with torch.no_grad():
                    plain.weight.copy_(mod.weight)
                    if mod.bias is not None:plain.bias.copy_(mod.bias)
                _set_module(candidate,name,plain)
            measured=score(candidate,items,torch,ji,original);measured.update(policy=policy['id'],floating_conv_exceptions=chosen,deployment_quantized_convs=56-len(chosen));histories.append(measured);x.write(out/'SOURCE_HISTORY.json',histories)
            arms.append(dict(policy=policy['id'],drops=measured['drops'],torch_accuracy_pass=measured['accuracy_pass'],floating_conv_exceptions=chosen));x.write(out/'ARMS.json',arms)
            print(policy['id'],measured['drops'],measured['accuracy_pass'],flush=True)
            if not measured['accuracy_pass']:continue
            dest=out/policy['id'];dest.mkdir();gate=evaluate_export(candidate,dest/'qat.onnx',dest,items,torch,ji,original);x.write(dest/'SOURCE_GATE.json',gate,True)
            if not gate['accuracy_pass']:continue
            parities=[dict(file=rec['name'],**ji.qat_export_parity(candidate,dest/'qat.onnx',image)) for image,mask,rec in items]
            structural=all(p.get('strict_logit_parity_pass') or p.get('unrounded_graph_parity_pass') for p in parities);strict=all(p['strict_logit_parity_pass'] for p in parities)
            x.write(dest/'EXPORT_PARITY.json',dict(strict_all89=strict,unrounded_structure_all89=structural,rows=parities),True)
            if not structural:continue
            torch.save(dict(model=candidate.state_dict(),scales=saved['scales'],updates=saved['updates'],protocol=saved['protocol'],mixed_exception_protocol=protocol,floating_conv_exceptions=chosen),out/'checkpoint_selected.pt')
            training=x.read(origin/'TRAINING_RECEIPT.json');training.update(training_quantized_convs=56,quantized_convs=56-len(chosen),floating_conv_exceptions=chosen,additional_optimizer_updates=0,mixed_exception_policy=policy['id']);x.write(out/'TRAINING_RECEIPT.json',training,True)
            for filename in ('SCALES.json','FLOAT_CONSTRUCTION.json'):x.write(out/filename,x.read(origin/filename),True)
            result.update(status='SOURCE_ACCURACY_RECOVERED',acceptable=True,deployment_quantized_convs=56-len(chosen),floating_conv_exceptions=chosen,selected=dict(source_gate=gate,checkpoint=x.pin(out/'checkpoint_selected.pt'),strict_logit_parity_all89=strict,unrounded_structure_all89=structural,export_parity=x.pin(dest/'EXPORT_PARITY.json')));break
        else:result.update(status='MIXED_EXCEPTIONS_NOT_RECOVERED',acceptable=False)
    except Exception as error:
        import traceback
        result.update(error=repr(error),traceback=traceback.format_exc())
    finally:result['elapsed_seconds']=time.time()-started;x.write(out/'RESULT.json',result,True)
if __name__=='__main__':main()
