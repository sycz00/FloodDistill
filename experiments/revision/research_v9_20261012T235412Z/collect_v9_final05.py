"""Detailed final/interim analysis and frontier, using the current frozen manifest."""
from pathlib import Path
from collections import defaultdict
import json,subprocess,io
import numpy as np
import common_v9 as c
from policy_final03 import activate
activate()
read=c.read
def current_read(path,default=None):
    if Path(path)==c.ROOT/'manifest.json':return read(c.ROOT/'manifest_final05.json',default)
    return read(path,default)
c.read=current_read
import collect_v9_final03 as base

def main():
    base.main()
    counts,strata,dry,hists,decode,joint=base.read_metrics();rows=[]
    settings=c.read(c.ROOT/'source_operating_points.json',{})
    for ds in ['s11val','s11','sturm','wf','wf2']:
        for seed in c.SEEDS:
            fp=f'fl_R4_sel_s{seed}__selected';qp=f'PTQ_s{seed}';tau=settings.get(qp,{}).get('tau');v=f'tau={tau:g}' if tau is not None else None
            events=sorted({ev for d,m,ev,var in counts if d==ds and m in [fp,qp]})
            for event in events:
                a=counts.get((ds,fp,event,'deployed'));b=counts.get((ds,qp,event,'deployed'));z=counts.get((ds,qp,event,v)) if v else None
                if a is None or b is None:continue
                A=base.e.scores(a);B=base.e.scores(b);Z=base.e.scores(z) if z else None
                if A['water_IoU'] is None or B['water_IoU'] is None:continue
                d=B['water_IoU']-A['water_IoU'];dt=Z['water_IoU']-A['water_IoU'] if Z and Z['water_IoU'] is not None else None
                rows.append(dict(dataset=ds,seed=seed,event=event,FP_water_IoU=A['water_IoU'],PTQ_water_IoU=B['water_IoU'],PTQ_minus_FP=d,source_tau=tau,source_threshold_minus_FP=dt,absolute_gap_removed=abs(d)-abs(dt) if dt is not None else None,qualification='Threshold selected on S11 validation only;GPU PTQ simulator; no equivalence claim'))
    base.csvout('ptq_operating_point_by_event.csv',rows)
    torch,cj,ev=c.setup();frontier=[]
    for graph in ['R2','R3','R4','A3','A0']:
        model=c.build(graph);params=sum(p.numel() for p in model.parameters());buf=io.BytesIO();torch.save(dict(model=model.state_dict()),buf)
        compatible=[x for x in c.read(c.V7/'reuse_models.json',[]) if x['graph']==graph]
        frontier.append(dict(graph=graph,parameters=params,FP32_parameter_bytes=params*4,FP32_model_container_bytes=len(buf.getvalue()),native_engine_bytes=None,precision='FP32',checkpoint_ids=[x['id'] for x in compatible],history='Archived updates/early stopping remain distinct; no matched-budget retraining implied'))
    for record in c.read(c.ROOT/'ptq_manifest.json')['records']:
        p=c.ROOT/'ptq_states'/f"q_s{record['seed']}.pt";entry=dict(id='ptq',graph='R4',quantized=True,checkpoint=str(p),sha256=c.sha(p));model=c.load_model(entry,'cpu');packed=0;scales=0
        for name,module in model.named_modules():
            for key,param in module.named_parameters(recurse=False):
                bits=int(module._weight_bits) if key=='weight' and hasattr(module,'_weight_bits') else 32;packed+=(param.numel()*bits+7)//8
            for key,buffer in module.named_buffers(recurse=False):
                if key.startswith('_'):scales+=buffer.numel()*buffer.element_size()
        frontier.append(dict(graph='R4',seed=record['seed'],parameters=sum(x.numel() for x in model.parameters()),precision='W4A8body/W8A8stem/FP32head',theoretical_packed_parameters_bytes=packed,quantizer_metadata_tensor_bytes=scales,serialized_simulator_state_bytes=p.stat().st_size,native_engine_bytes=None,measured_native_kernel_ms=None,readout_route='GPU: CPU simulator parity failed forparent64',history=record['origin']))
    base.csvout('model_size_frontier.csv',frontier)
    ds_models=defaultdict(list)
    for ds,m,event,var in counts:
        if var=='deployed':ds_models[(ds,m)].append(event)
    family=defaultdict(list);methods={x['id']:x.get('method',x['id']) for x in c.read(c.ROOT/'final_models.json',{}).get('models',[])}
    for (ds,m),events in ds_models.items():
        cc=[base.e.scores(counts[(ds,m,ev,'deployed')]) for ev in set(events)];score=base.avg([x['water_IoU'] for x in cc]);family[(ds,methods.get(m,m))].append(score)
    summary=[dict(dataset=ds,family=f,n_models=len(vals),mean_equal_event_water_IoU=base.avg(vals),across_seed_SD=float(np.std([x for x in vals if x is not None],ddof=1)) if sum(x is not None for x in vals)>1 else None) for (ds,f),vals in family.items()]
    base.csvout('family_summary.csv',summary)
    dry_summary=[]
    dg=defaultdict(list)
    for row in dry:dg[(row['dataset'],row['model'],row['variant'])].append(row)
    for (ds,m,v),rr in dg.items():
        dry_summary.append(dict(dataset=ds,model=m,variant=v,reference_dry_windows=len(rr),any_water=sum(x['any_water'] for x in rr),alarms_001=sum(x['alarm_001'] for x in rr),alarms_01=sum(x['alarm_01'] for x in rr),alarms_05=sum(x['alarm_05'] for x in rr)))
    base.csvout('dry_alarm_summary.csv',dry_summary)
    per_window=[]
    for m,seconds in decode.items():
        inference=c.read(c.ROOT/'normal_windows.json',{}).get('windows',[])+c.read(c.ROOT/'sealed_windows.json',{}).get('rows',[]);n=sum(not x.get('omit') for x in inference)
        per_window.append(dict(model=m,cpu_decode_seconds=seconds,approx_windows=n,approx_ms_per512=seconds/n*1000 if n else None,qualification='CPU EDT+physical redicision only; excludes model forward; actual per-slot timing stored separately'))
    base.csvout('decode_cpu_cost.csv',per_window)
    with open(c.ROOT/'report_v9.md','a') as f:
        f.write('\n## Detailed analysis and scope qualifications\n\n')
        if summary:
            f.write('| Set | Family | Models | Equal-event water IoU | Seed SD |\n|---|---|---:|---:|---:|\n')
            for row in summary:
                mean=row['mean_equal_event_water_IoU'];sd=row['across_seed_SD'];f.write(f"|{row['dataset']}|{row['family']}|{row['n_models']}|{100*mean:.3f} pp|{100*sd:.3f} pp|\n" if mean is not None and sd is not None else f"|{row['dataset']}|{row['family']}|{row['n_models']}|{100*mean if mean is not None else 'undefined'}|not a seed sample|\n")
        f.write('\nThese family means are descriptive. The primary whole-event interval and mechanism/dry gates are in scientific_gates.json; no claim follows from a mean alone. Every model/event/seed is retained in the CSVs. PTQ event-specific gap removal uses source thresholds only (ptq_operating_point_by_event.csv). model_size_frontier.csv separates FP parameters, serialized model containers, theoretical packed parameters, metadata and unavailable native engine/kernel measurements.\n\n')
        gate=c.read(c.ROOT/'sealed_cost_admission04.json')
        if gate:f.write('Full sealed matrix cost admission: '+json.dumps(gate)+'\n\n')
        f.write('The original and repaired whole-smoke attempts are preserved and charged. The standalone smoke missed the Trainer seed reset; the actual training adapter already used it. TF32 and PTQ numerical failures were diagnosed before sealed labels. Future GPU inference resets TF32 flags after model construction, since the teacher factory can reset matmul precision. The immutable source teacher masks used cuDNN TF32off with factory matmul precision medium; both paired arms share the exact cached target bytes.\n\n')
        f.write('The primary B0 checkpoints retain two legacy metadata descriptions: data_position_state.completed_epochs divides by156 and the scheduler sentence mentions156 updates. For these252-image runs the authoritative receipt/config/training_state/epoch records show15 updates per epoch,50 epochs and750 completed updates. Both arms follow that identical executed recipe; no exact mid-epoch resume is claimed.\n\n')
        f.write('GDACS native264-pixel patches are resized224, so the training index/band width operates on about11.79m pixels; the native WF512-window decode operates on10m pixels. Teacher16-pixel tokens are not16-pixel decoder logits. EfficientViT-B0 uses its inspected H/8 segmentation head before full-resolution interpolation. STURM paper masks include both mapped water classes; WF excludes permanent water with JRC3. Spectral strata overlap and the component halfwidth definition does not measure every thin local branch. Ordinary independent dry imagery and the11-window human review remain open.\n\n')
        f.write('Authors must incorporate verified corrections, consolidate the evidence under consistent protocols, write the Discussion and response, and qualify any contribution by the sealed gates. If E1 fails, present the inherited-failure characterization and retained negative controls. If simpler E2/E3 explains a gain, retain that explanation; a nonsignificant difference does not establish equivalence. No manuscript was changed or submitted.\n')
    subprocess.run(['/usr/bin/python3',str(c.ROOT/'render_report_v9.py')],check=True,timeout=45)
    receipt=c.read(c.ROOT/'collection_receipt.json');receipt['report_sha256']=c.sha(c.ROOT/'report_v9.md');receipt['detailed_analysis']=True;c.save(c.ROOT/'collection_receipt.json',receipt)

if __name__=='__main__':main()
