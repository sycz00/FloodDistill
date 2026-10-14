"""Source-only compression controls; fixed candidate grid, no OOD feedback."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import common10 as c

def calibrated_control(model,batches,precision,kind):
    import torch
    from quant10 import prepare,PerChannel,clipping_hooks
    from models.compression_quant import FixedQATConv2d
    from quantization.qat import _set_module
    samples={};handles=[];generator=torch.Generator().manual_seed(1729)
    for name,mod in model.named_modules():
        if not isinstance(mod,torch.nn.Conv2d):continue
        def hook(layer,inputs,key=name):
            x=inputs[0].detach().cpu().float();values=[]
            for channel in range(x.shape[1]):
                a=x[:,channel].reshape(-1);values.append(a[torch.randperm(len(a),generator=generator)[:min(500,len(a))]])
            samples.setdefault(key,[]).append(torch.stack(values))
        handles.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for batch in batches:model(batch['image'].to(c.device()))
    for handle in handles:handle.remove()
    q=prepare(model,batches,precision)
    for name,mod in list(q.named_modules()):
        if not isinstance(mod,FixedQATConv2d):continue
        a=torch.cat(samples[name],dim=1).abs();maximum=2**(mod.activation_bits-1)-1
        if kind=='percentile':
            scale=float(torch.quantile(a.reshape(-1),.9999))/maximum;mod._act_scale.fill_(max(scale,1e-8));mod._activation_scale_value=max(scale,1e-8)
        elif kind=='per_channel':
            scale=a.max(1).values.div(maximum).clamp_min(1e-8);_set_module(q,name,PerChannel(mod,scale))
    return q.to(c.device())

def prepare_qat_parent(parent,batches,precision):
    """Immutable prepared parents can be consumed by V11 before B2 selection."""
    torch,_,_,_=c.setup()
    from quant10 import prepare
    from evaluate10 import source
    q=prepare(c.load(parent),batches,precision)
    name=parent['id']+'__QAT_parent__'+precision
    receipt_path=c.ROOT/'qat_parents'/f'{parent["graph"]}_{precision}_s{parent["seed"]}.json'
    existing=c.read(receipt_path)
    if existing:
        assert existing['FP_parent']['sha256']==parent['sha256'],'Prepared FP parent changed'
        entry=existing['entry'];assert c.sha(entry['checkpoint'])==entry['sha256']
        saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
        assert c.state_hash(saved['model'])==c.state_hash(q.state_dict()),'Prepared quantizer state differs; immutable V11 input protected'
        metric=c.read(c.ROOT/'source_metrics'/(entry['id']+'.json'))
        assert metric and metric['checkpoint_sha256']==entry['sha256']
        return entry,metric
    path=c.ROOT/'quantized_states'/(name+'.pt');path.parent.mkdir(exist_ok=True)
    assert not path.exists(),'Checkpoint exists without atomic parent receipt; inspect partial attempt'
    temporary=path.with_name(path.name+'.tmp');torch.save(dict(model=q.state_dict(),parent_sha256=parent['sha256']),temporary)
    __import__('os').replace(temporary,path)
    entry=dict(parent,id=name,checkpoint=str(path),sha256=c.sha(path),quantized=True,precision=precision,fp_reference=parent['id'],sealed=True)
    metric=source(entry,model=q)
    c.save(c.ROOT/'quantized_entries'/(name+'.json'),dict(entry,control='QAT_unchanged_baseline'))
    c.save(receipt_path,dict(entry=entry,FP_parent=parent))
    return entry,metric

def core_parents():
    """Exact R4 W4A4 starting states, independent of the A5 size decision."""
    tic=time.monotonic();torch,cj,_,_=c.setup()
    from stages10 import inventory
    from evaluate10 import source
    spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=1729),augment=False)
    indices=c.read(c.V9/'ptq_manifest.json')['records'][0]['calibration_indices']
    batches=[dict(image=ds[i]['image'][None].float()) for i in indices]
    parents=[x for x in inventory().values() if x['graph']=='R4' and x['method']=='lkd' and not x.get('quantized')]
    assert len(parents)==4 and {x['seed'] for x in parents}==set(c.SEEDS)
    gaps=[];rows=[]
    for parent in sorted(parents,key=lambda x:x['seed']):
        baseline=c.read(c.ROOT/'source_metrics'/(parent['id']+'.json'))
        if not baseline or baseline.get('checkpoint_sha256')!=parent['sha256']:
            baseline=source(parent)
        entry,metric=prepare_qat_parent(parent,batches,'W4A4')
        gap=metric['equal_event_water_IoU']-baseline['equal_event_water_IoU'];gaps.append(gap)
        rows.append(dict(seed=parent['seed'],parent=parent,entry=entry,source_gap=gap,
                         parent_receipt=str(c.ROOT/'qat_parents'/f'R4_W4A4_s{parent["seed"]}.json')))
    admission=dict(graph='R4',precision='W4A4',source_gap=float(np.mean(gaps)),
                   admitted=bool(np.mean(gaps)<-.01),rule='Original V10 four-seed source deficit worse than1pp; V11 applies its separate fixed gate')
    c.save(c.ROOT/'core_qat_parent_preparation.json',dict(outcome='completed',seconds=time.monotonic()-tic,
           rows=rows,qat_admission=[admission],source_only=True,calibration_ids=indices,
           calibration_manifest_sha256=c.sha(c.V9/'ptq_manifest.json'),
           recipe='Same parent preparation as full controls10; split only to remove irrelevant A5 scheduling dependency',
           full_calibration_controls_completed=False,external_scores_used=False))

def main():
    torch,cj,old,v=c.setup();from stages10 import inventory
    from evaluate10 import source
    from quant10 import prepare
    from models.compression_quant import precision_map,FixedQATConv2d
    spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=1729),augment=False);files=list(ds.files);indices=c.read(c.V9/'ptq_manifest.json')['records'][0]['calibration_indices'];bank16=[dict(image=ds[i]['image'][None].float()) for i in indices];byname={p.name:i for i,p in enumerate(files)};bank64=[dict(image=ds[byname[name]]['image'][None].float()) for name in c.read(c.ROOT/'calibration64.json')['files']]
    items=inventory();selection=c.read(c.ROOT/'smallest_viable_selection.json');graphs=['R4']+([selection['graph']] if selection and selection['graph']!='R4' else[]);controls=[];continuations=[]
    for graph in graphs:
        parents=[x for x in items.values() if x['graph']==graph and x['method']==('lkd' if graph=='R4' else 'KDoff') and not x.get('quantized')]
        if {x['seed'] for x in parents}!=set(c.SEEDS):continue
        for precision in ['W8A8','W4A8','W4A4']:
            candidates=[]
            for kind in ['minmax','percentile','per_channel']:
                for count,batches in [(16,bank16),(64,bank64)]:
                    rows=[]
                    for parent in parents:
                        name=parent['id']+'__'+precision+f'__{kind}{count}';q=calibrated_control(c.load(parent),batches,precision,kind) if kind!='minmax' else prepare(c.load(parent),batches,precision)
                        path=c.ROOT/'quantized_states'/(name+'.pt');path.parent.mkdir(exist_ok=True);torch.save(dict(model=q.state_dict(),parent_sha256=parent['sha256']),path)
                        entry=dict(parent,id=name,checkpoint=str(path),sha256=c.sha(path),quantized=True,precision=precision,calibration_range=kind,calibration_count=count,fp_reference=parent['id'],control='calibration',sealed=False)
                        result=source(entry,model=q);rows.append(dict(entry=entry,equal_event_water_IoU=result['equal_event_water_IoU']))
                    candidates.append(dict(kind=kind,count=count,mean_equal_event_water_IoU=float(np.mean([x['equal_event_water_IoU'] for x in rows])),rows=rows))
            # Select one calibration control per precision using four-seed source mean only.
            chosen=max(candidates,key=lambda x:x['mean_equal_event_water_IoU'])
            for row in chosen['rows']:
                entry=dict(row['entry'],sealed=True,source_selection='best ofsix fixed range/calibration choices, four-seed S11-validation mean')
                c.save(c.ROOT/'quantized_entries'/(entry['id']+'.json'),entry)
            controls.append(dict(graph=graph,precision=precision,candidates=candidates,selected=dict(kind=chosen['kind'],count=chosen['count']),source_only=True))
            if precision!='W4A4':continue
            for parent in parents:
                uniform=prepare(c.load(parent),bank16,'W4A4');sensitivity=[]
                for name,mod in uniform.named_modules():
                    if not isinstance(mod,FixedQATConv2d) or mod.weight_bits!=4:continue
                    single=prepare(c.load(parent),bank16,'W8A8');target=dict(single.named_modules())[name];target._w_scale.mul_(127/7);target._weight_bits.fill_(4);target._weight_bits_value=4;target._act_scale.mul_(127/7);target._activation_bits.fill_(4);target._activation_bits_value=4;target._activation_scale_value=float(target._act_scale[0])
                    meta=dict(parent,id=parent['id']+'__single4__'+name.replace('.','_'));result=source(meta,model=single,write_margins=False);sensitivity.append(dict(layer=name,source_equal_event_water_IoU=result['equal_event_water_IoU']))
                protected=[x['layer'] for x in sorted(sensitivity,key=lambda x:x['source_equal_event_water_IoU'])[:2]]
                for name in protected:
                    target=dict(uniform.named_modules())[name];target._w_scale.mul_(7/127);target._weight_bits.fill_(8);target._weight_bits_value=8;target._act_scale.mul_(7/127);target._activation_bits.fill_(8);target._activation_bits_value=8;target._activation_scale_value=float(target._act_scale[0])
                name=parent['id']+'__W4A4__protected2';path=c.ROOT/'quantized_states'/(name+'.pt');torch.save(dict(model=uniform.state_dict(),parent_sha256=parent['sha256']),path)
                entry=dict(parent,id=name,checkpoint=str(path),sha256=c.sha(path),quantized=True,precision='W4A4protected2',fp_reference=parent['id'],control='layer_protection',protected=protected,sensitivity=sensitivity,sealed=True);c.save(c.ROOT/'quantized_entries'/(name+'.json'),entry)
        required_precision='W4A4' if graph=='R4' else 'W4A8'
        if selection and selection['graph']=='R4':precisions=['W4A4','W4A8']
        else:precisions=[required_precision]
        for precision in precisions:
            gaps=[]
            for parent in parents:
                base=c.read(c.ROOT/'source_metrics'/(parent['id']+'.json'));entry,result=prepare_qat_parent(parent,bank16,precision);gaps.append(result['equal_event_water_IoU']-base['equal_event_water_IoU'])
            continuations.append(dict(graph=graph,precision=precision,source_gap=float(np.mean(gaps)),admitted=bool(np.mean(gaps)<-.01),rule='Four-seed S11-validation equal-event gap worse than1pp; no target outcomes'))
    c.save(c.ROOT/'compression_control_selection.json',dict(outcome='completed',controls=controls,qat_admission=continuations,source_only=True))
    inventory()
if __name__=='__main__':main()
