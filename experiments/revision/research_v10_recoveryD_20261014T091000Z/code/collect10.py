"""Reviewer tables and whole-event uncertainty from executed readouts only."""
import csv,json,time,statistics
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
import common10 as c
from evaluate10 import scores,ap

def csvout(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status','reason']
    with open(c.ROOT/name,'w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def mean(values):
    a=[x for x in values if x is not None and np.isfinite(x)];return float(np.mean(a)) if a else None
def bootstrap(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return None,None
    rng=np.random.default_rng(1729);m=a[rng.integers(0,len(a),(10000,len(a)))].mean(1);return tuple(map(float,np.percentile(m,[2.5,97.5])))

def main():
    state=c.read(c.ROOT/'state.json',{});inventory=c.read(c.ROOT/'readout_models.json',{}).get('models',c.read(c.ROOT/'historical_models.json',[]));byid={x['id']:x for x in inventory};cells={};strata={};shared={};joint={};spectra={};ceilings={};calibration={};decode_times={};dry=[];clips=defaultdict(lambda:np.zeros(2,np.int64));histograms={}
    for mode in ['normal_A','normal_B','normal_C','normal','sealed']:
        for path in (c.ROOT/'metrics'/mode).glob('slot*.json'):
            a=c.read(path)
            for field,destination in [('event_counts',cells),('strata',strata),('shared',shared),('joint',joint),('spectra',spectra),('ceilings',ceilings),('calibration',calibration),('decode_times',decode_times)]:
                for row in a.get(field,[]):destination[tuple(row['key'])]=np.array(row['counts'])
            dry.extend(a.get('dry',[]))
            for row in a.get('clipping',[]):clips[(row['dataset'],row['model'],row['event'],row['layer'])]=np.array([row['clipped'],row['activations']])
        for path in (c.ROOT/'metrics'/mode).glob('hist_slot*.npz'):
            with np.load(path,allow_pickle=False) as a:
                for key,value in zip(a['keys'],a['histograms']):histograms[tuple(json.loads(str(key)))]=value.copy()
    dry=list({(r['dataset'],r['model'],r['name']):r for r in dry}.values())
    calibration_rows=[]
    for dataset,model in {(k[0],k[1]) for k in calibration}:
        values=sum(v for k,v in calibration.items() if (k[0],k[1])==(dataset,model));n=values[:,0].sum();ece=float(np.abs(values[:,1]-values[:,2]).sum()/n) if n else None;brier=float(values[:,3].sum()/n) if n else None
        calibration_rows.append(dict(dataset=dataset,model=model,teacher_edge_band_pixels=int(n),Brier=brier,ECE_15bins=ece,band='Within2nativepixels of frozen teacher hard-mask edge; reference-valid'))
    for entry in inventory:
        source=c.read(c.ROOT/'source_metrics'/(entry['id']+'.json'),{});values=source.get('teacher_edge_band_calibration_bins')
        if values is not None:
            values=np.array(values);n=values[:,0].sum();calibration_rows.append(dict(dataset='s11val',model=entry['id'],teacher_edge_band_pixels=int(n),Brier=float(values[:,3].sum()/n),ECE_15bins=float(np.abs(values[:,1]-values[:,2]).sum()/n)))
    csvout('mixed_pixel_calibration.csv',calibration_rows or [dict(status='DEFERRED',reason='No optional mixed-pixel calibration readout')]);csvout('decode_cpu_time.csv',[dict(dataset=k[0],model=k[1],event=k[2],tiles=int(v[1]),CPU_decode_seconds_per_tile=float(v[0]/v[1]),scope='Workstation CPU during bounded parallel readout; no deployment latency claim') for k,v in decode_times.items() if v[1]])
    bymodel=defaultdict(dict);eventrows=[]
    for (dataset,model,event,variant),cell in cells.items():
        bymodel[(dataset,model,variant)][event]=cell;entry=byid.get(model,{})
        eventrows.append(dict(dataset=dataset,model=model,graph=entry.get('graph'),method=entry.get('method'),seed=entry.get('seed'),variant=variant,event=event,tp=int(cell[0]),fp=int(cell[1]),fn=int(cell[2]),tn=int(cell[3]),**scores(cell)))
    summaries=[];summary_lookup={}
    for (dataset,model,variant),events in bymodel.items():
        entry=byid.get(model,{});pooled=scores(sum(events.values()));per=[scores(cell) for cell in events.values()];h=[value for (ds,name,_),value in histograms.items() if ds==dataset and name==model]
        row=dict(dataset=dataset,model=model,graph=entry.get('graph'),method=entry.get('method'),seed=entry.get('seed'),variant=variant,events=len(events),parameters=entry.get('parameters'),MACs_per_pixel=entry.get('MACs_per_pixel'),FP32_bytes=entry.get('FP32_bytes'),precision=entry.get('precision','FP32'),effective_precision=entry.get('effective_precision',entry.get('precision','FP32')),precision_degenerate=entry.get('precision_degenerate',False),precision_qualification=entry.get('precision_qualification'),native_integer_execution=False,
            **{'pooled_'+key:value for key,value in pooled.items()},**{'equal_event_'+key:mean([x[key] for x in per]) for key in pooled},pooled_AP=ap(sum(h)) if h else None,equal_event_AP=mean([ap(x) for x in h]))
        alarms=[x for x in dry if x['dataset']==dataset and x['model']==model];row.update(reference_dry_windows=len(alarms),dry_alarm_001=mean([x['alarm_001'] for x in alarms]),dry_alarm_01=mean([x['alarm_01'] for x in alarms]),dry_alarm_05=mean([x['alarm_05'] for x in alarms]));summaries.append(row);summary_lookup[(dataset,model,variant)]=row
    csvout('supervised_vs_distilled.csv',summaries);csvout('supervised_vs_distilled_by_event.csv',eventrows)
    rule_comparisons=[]
    for (dataset,model,variant),row in summary_lookup.items():
        if byid.get(model,{}).get('graph')=='rule' or variant!='deployed':continue
        hh=[h for (ds,name,_),h in histograms.items() if ds==dataset and name==model]
        for rule in ['MNDWI','WI2015','AWEIsh']:
            reference=summary_lookup.get((dataset,rule,'deployed'))
            if not reference:continue
            precision=None
            if hh and reference['pooled_recall'] is not None:
                h=sum(hh)[::-1];tp=h[:,1].cumsum();fp=h[:,0].cumsum();recall=tp/tp[-1] if tp[-1] else np.zeros(len(tp));index=int(np.argmin(abs(recall-reference['pooled_recall'])));precision=float(tp[index]/(tp[index]+fp[index])) if tp[index]+fp[index] else None
            wins=[]
            for event,a in bymodel[(dataset,model,'deployed')].items():
                b=bymodel.get((dataset,rule,'deployed'),{}).get(event)
                if b is not None and scores(a)['water_IoU'] is not None and scores(b)['water_IoU'] is not None:wins.append(scores(a)['water_IoU']>scores(b)['water_IoU'])
            rule_comparisons.append(dict(dataset=dataset,model=model,rule=rule,model_precision_at_rule_pooled_recall=precision,rule_pooled_recall=reference['pooled_recall'],event_win_fraction=mean(wins),rule_equal_event_water_IoU=reference['equal_event_water_IoU'],model_equal_event_water_IoU=row['equal_event_water_IoU'],PR_scope='4096-bin descriptive pooled curve; no target threshold is deployed'))
    csvout('learned_vs_rules.csv',rule_comparisons)
    # Anomalies are reported and excluded from confirmatory contrasts; training choices never change.
    anomalies=[]
    unets=[row['pooled_mIoU'] for row in summaries if row['dataset']=='s11' and row['graph']=='UNet' and row['variant']=='deployed']
    flagged=set()
    if len(unets)==4 and abs(mean(unets)-.862)>2*.018:
        flagged.update(x['id'] for x in inventory if x['graph']=='UNet');anomalies.append(dict(group='UNet_SUP',mean_s11_pooled_mIoU=mean(unets),expected=.862,tolerance=.036,reason='Outside TableI two-SD acceptance; no contrasts'))
    for entry in inventory:
        if 'matched_lkd_clip5_valid_N2500' in entry['id']:flagged.add(entry['id']);anomalies.append(dict(group=entry['id'],reason='Previously flagged3pp+paper anomaly'))
    csvout('anomalies.csv',anomalies)
    def group(graph,method):
        options=[x for x in inventory if x.get('graph')==graph and x.get('method')==method and not x.get('quantized') and not x.get('precision_degenerate') and not x.get('control') and not x.get('source_reference') and x['id'] not in flagged]
        preferred=[x for x in options if x.get('matched_new_training')]
        if preferred:options=preferred
        return {x.get('seed'):x['id'] for x in options}
    def contrast(name,left,right,margin=None,threshold=.01):
        output=[]
        for dataset in sorted({key[0] for key in bymodel}):
            differences=[];seedmean=[]
            for seed in c.SEEDS:
                aid=left.get(seed);bid=right.get(seed)
                if not aid or not bid or aid in flagged or bid in flagged:continue
                a=bymodel.get((dataset,aid,'deployed'),{});b=bymodel.get((dataset,bid,'deployed'),{});pairs={event:scores(a[event])['water_IoU']-scores(b[event])['water_IoU'] for event in a.keys()&b.keys() if scores(a[event])['water_IoU'] is not None and scores(b[event])['water_IoU'] is not None}
                if pairs:differences.append(pairs);seedmean.append(mean(list(pairs.values())))
            events=set.intersection(*(set(x) for x in differences)) if differences else set();values=[mean([x[event] for x in differences]) for event in sorted(events)];lo,hi=bootstrap(values);estimate=mean(seedmean);complete=len(seedmean)==4
            passed=complete and estimate is not None and ((estimate>=threshold and lo is not None and lo>0 and sum(x>0 for x in seedmean)>=3) if margin is None else lo is not None and lo>-margin)
            output.append(dict(contrast=name,dataset=dataset,delta_equal_event_water_IoU=estimate,event_bootstrap_low=lo,event_bootstrap_high=hi,across_seed_SD=float(np.std(seedmean,ddof=1)) if len(seedmean)>1 else None,seeds=len(seedmean),per_seed=seedmean,paired_events=len(events),minimum_effect=threshold,noninferiority_margin=margin,parameter_ratio=mean([byid.get(left.get(s),{}).get('parameters',0)/byid[right[s]]['parameters'] for s in c.SEEDS if right.get(s) in byid and byid[right[s]].get('parameters')]),status='PASSED' if passed and dataset=='wf2' else 'SCIENTIFIC_GATE_FAILED' if complete and dataset=='wf2' else 'DESCRIPTIVE' if complete else 'UNAVAILABLE',event_is_unit=True,qualification=c.read(c.ROOT/'final_evaluation_freeze.json',{}).get('scientific_look')))
        return output
    comparisons=[]
    for graph in ['R2','R4','A3','A0']:comparisons+=contrast(graph+'_LKD_minus_SUP',group(graph,'lkd'),group(graph,'SUP'))
    for graph in ['UNet','B0']:comparisons+=contrast('R4_LKD_minus_'+graph+'_SUP',group('R4','lkd'),group(graph,'SUP'),margin=.015)
    for rule in ['MNDWI','WI2015','AWEIsh']:comparisons+=contrast('R4_LKD_minus_'+rule,group('R4','lkd'),{seed:rule for seed in c.SEEDS})
    for graph in ['R2','R3','R4_micro','R4']:
        for method in ['SUP','CE']:comparisons+=contrast(graph+'_KDoff_minus_'+method,group(graph,'KDoff'),group(graph,method),threshold=.005 if method=='CE' else .01)
    csvout('contrasts.csv',comparisons)
    grid=[];decomposition=[];frontier=[];controlrows=[];qat=[];indices=[]
    for row in summaries:
        entry=byid.get(row['model'],{})
        if entry.get('matched_new_training') and entry.get('method') in ['SUP','CE','KDoff'] and entry.get('graph') in ['R2','R3','R4_micro','R4']:grid.append(row)
        if entry.get('quantized') or entry.get('method')=='lkd':frontier.append(dict(row,fp_reference=entry.get('fp_reference'),serialized_simulator_bytes=Path(entry['checkpoint']).stat().st_size,theoretical_packed_weight_bytes=entry.get('packed_bytes'),scope='GPU FP32 fake-quant simulator; native low-bit engine/kernel unavailable' if entry.get('quantized') else 'FP32 parent; historical completed updates may differ'))
        if entry.get('control'):controlrows.append(row)
        if str(entry.get('method')).startswith('d'):qat.append(row)
        if entry.get('graph')=='rule':indices.append(row)
    csvout('size_supervision_grid.csv',grid);csvout('compression_frontier.csv',frontier);csvout('compression_controls.csv',controlrows);csvout('qat_pairs.csv',qat);csvout('index_baselines.csv',indices)
    sharedrows=[];spectrarows=[];ceilingrows=[]
    for (dataset,model,event,stratum),values in shared.items():
        cell=strata.get((dataset,model,event,stratum),np.zeros(4));sharedrows.append(dict(dataset=dataset,model=model,event=event,stratum=stratum,shared_missed_water=int(values[0]),shared_false_water=int(values[1]),inherited_error=int(values[2]),compression_induced=int(values[3]),inherited_induced_intersection=int(values[4]),other_error=int(values[5]),valid_pixels=int(cell.sum()),share_misses=values[0]/cell[2] if cell[2] else None,share_false_water=values[1]/cell[1] if cell[1] else None))
    for (dataset,model,event,kind),values in spectra.items():spectrarows.append(dict(dataset=dataset,model=model,event=event,kind=kind,pixels=int(values[-1]),**{f'band_mean_{i}':float(values[i]/values[-1]) if values[-1] else None for i in range(6)}))
    for (dataset,model,event,kind),cell in ceilings.items():ceilingrows.append(dict(dataset=dataset,model=model,event=event,removed_error_stratum=kind,unattainable_reference_oracle=True,**scores(cell)))
    for (dataset,model,event,stratum),cell in strata.items():
        entry=byid.get(model,{});decomposition.append(dict(dataset=dataset,model=model,graph=entry.get('graph'),seed=entry.get('seed'),method=entry.get('method'),event=event,stratum=stratum,tp=int(cell[0]),fp=int(cell[1]),fn=int(cell[2]),tn=int(cell[3]),dark_land_false_water=cell[1]/(cell[1]+cell[3]) if cell[1]+cell[3] else None,**scores(cell)))
        if entry.get('matched_new_training') and entry.get('method') in ['SUP','CE','KDoff'] and entry.get('graph') in ['R2','R3','R4_micro','R4'] and not entry.get('control'):
            grid.append(dict(dataset=dataset,model=model,graph=entry['graph'],method=entry['method'],seed=entry['seed'],event=event,stratum=stratum,parameters=entry.get('parameters'),MACs_per_pixel=entry.get('MACs_per_pixel'),FP32_bytes=entry.get('FP32_bytes'),**scores(cell)))
    csvout('size_supervision_grid.csv',[x for x in grid if 'stratum' in x])
    for dataset,model,kind in sorted({(k[0],k[1],k[3]) for k in ceilings}):
        values=[v for k,v in ceilings.items() if (k[0],k[1],k[3])==(dataset,model,kind)]
        ceilingrows.append(dict(dataset=dataset,model=model,event='POOLED',removed_error_stratum=kind,unattainable_reference_oracle=True,**scores(sum(values))))
        ceilingrows.append(dict(dataset=dataset,model=model,event='EQUAL_EVENT',removed_error_stratum=kind,unattainable_reference_oracle=True,water_IoU=mean([scores(x)['water_IoU'] for x in values]),mIoU=mean([scores(x)['mIoU'] for x in values])))
    csvout('shared_failures.csv',sharedrows);csvout('shared_spectra.csv',spectrarows);csvout('ceilings.csv',ceilingrows);csvout('decomposition_results.csv',decomposition);csvout('joint_errors.csv',[dict(dataset=k[0],model=k[1],event=k[2],reference_teacher_rule_model_codes=json.dumps(v.tolist())) for k,v in joint.items()]);csvout('dry_window_alarms.csv',dry)
    clipping=[]
    for (dataset,model,event,layer),values in clips.items():
        parent=byid.get(model,{}).get('fp_reference');a=summary_lookup.get((dataset,model,'deployed'));pair=bymodel.get((dataset,parent,'deployed'),{});cell=bymodel.get((dataset,model,'deployed'),{}).get(event);delta=scores(cell)['water_IoU']-scores(pair[event])['water_IoU'] if cell is not None and event in pair and scores(cell)['water_IoU'] is not None and scores(pair[event])['water_IoU'] is not None else None
        clipping.append(dict(dataset=dataset,model=model,event=event,layer=layer,clipped=int(values[0]),activations=int(values[1]),fraction=values[0]/values[1] if values[1] else None,quant_minus_FP_event_water_IoU=delta,scope='All activations on registered padded inputs; unmasked internal features'))
    associations=defaultdict(list)
    for row in clipping:
        if row['fraction'] is not None and row['quant_minus_FP_event_water_IoU'] is not None:associations[(row['dataset'],row['model'],row['layer'])].append(row)
    correlation={key:float(spearmanr([x['fraction'] for x in rr],[x['quant_minus_FP_event_water_IoU'] for x in rr]).statistic) if len(rr)>2 else None for key,rr in associations.items()}
    for row in clipping:row['Spearman_descriptive']=correlation.get((row['dataset'],row['model'],row['layer']))
    csvout('activation_clipping.csv',clipping)
    attribution=[]
    for pair in c.read(c.ROOT/'operating_point_pairs.json',{}).get('pairs',[]):
        matched=pair['reference']+'__matched_to__'+pair['target']
        for dataset in sorted({key[0] for key in bymodel}):
            a=summary_lookup.get((dataset,matched,'deployed'));b=summary_lookup.get((dataset,pair['target'],'deployed'))
            if not a or not b or a['equal_event_water_IoU'] is None or b['equal_event_water_IoU'] is None:continue
            delta=b['equal_event_water_IoU']-a['equal_event_water_IoU'];attribution.append(dict(**pair,dataset=dataset,arm_minus_source_matched_reference=delta,within_03pp=abs(delta)<=.003,interpretation='Compatible with operating-point explanation' if abs(delta)<=.003 else 'Residual effect remains; not proof of mechanism'))
    csvout('operating_point_attribution.csv',attribution)
    csvout('offline_vs_live_kd.csv',contrast('R4_KDoff_minus_live_LKD',group('R4','KDoff'),group('R4','lkd')))
    import decisions10
    decisions10.collect(byid,bymodel,strata,shared,summary_lookup,comparisons,contrast,group,csvout)
    csvout('contrasts.csv',comparisons)
    checklist=[dict(job=name,status=value['status'],reason=value.get('reason'),evidence=str(c.ROOT/'receipts'/f'{name}.json')) for name,value in state.get('jobs',{}).items()];csvout('experiment_checklist.csv',checklist)
    sealed_done=all((c.ROOT/'metrics/sealed'/f'slot{i}.json').exists() for i in range(3));title='Completed bounded collection' if state.get('finished') else 'INTERIM';done=sum(x['status']=='COMPLETED' for x in checklist);pending=sum(x['status'] not in ['COMPLETED','UNAVAILABLE','DEFERRED','TECHNICAL_FAILED','SOURCE_GATE_NOT_MET'] for x in checklist)
    text=f'# V10 research follow-up — {title}\n\nCollected {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}. Sealed readout complete: {sealed_done}. Completed jobs: {done}; pending: {pending}.\n\n'
    text+='The frozen protocol, code hashes, four-seed recipes and job receipts define what was executed. No missing/deferred arm is treated as negative scientific evidence. Source validation selects checkpoints, calibration, thresholds, viability and QAT admission. Exposed readouts are descriptive; sealed claims require the declared event-level gates and complete seed coverage.\n\n'
    text+='The user-approved offline target correction stores FP32 logits for eight clean geometric views; geometric targets match fresh teacher forwards and only photometric jitter differs from live KD. The unchanged reference loss is used. Historical GPU PTQ range tensors and CPU regeneration differ slightly; GPU FP32 simulation is disclosed, and strict W8A8 includes stem and head. No simulator result establishes native low-bit latency, energy or losslessness.\n\n'
    text+='## Observed comparisons\n\n'
    for row in comparisons:
        if row['dataset']!='wf2':continue
        delta=row['delta_equal_event_water_IoU'];text+=f"- {row['contrast']}: {100*delta if delta is not None else 'unavailable'} pp; event interval [{row['event_bootstrap_low']}, {row['event_bootstrap_high']}], seeds={row['seeds']}, status={row['status']}.\n"
    text+='\n## What this means for the reviewer answers\n\nR1.1/R2.1: judge practical distillation value, smallest viability and compression mechanisms from the complete controlled contrasts; familiar components and shared misses alone do not establish novelty or intrinsic six-band limits. R1.2/R2.4: UNet/B0 and matched small supervised comparisons, with initialization/data/pretraining confounds and anomaly flags. R2.5: all fixed spectral rules, pooled/equal-event/AP and independently referenced flood-scene dry-window alarms. R1.6: false alarms and cloud margins remain explicit; teacher-referenced pre-event tests do not replace independently verified dry imagery. R2.6: strict all-layer W8A8 simulator and optional QDQ runtime are separate forms of evidence. OFA remains an existing ablation.\n\n'
    text+='## What must be done\n\nUse contrasts.csv with per-seed signs, whole-event bootstrap intervals and complete coverage; distinguish source-selected operating shifts from residual spatial/data effects. Retain unfavorable outcomes and calibration/QAT null results. Qualify noninferiority with the prespecified1.5pp margin. Validate independent ordinary dry references before unattended deployment claims. Authors still need a coherent Discussion, consistent tables and point-by-point response; no manuscript was changed or submitted.\n\n'
    text+='## Checklist and limitations\n\n'+json.dumps(checklist,indent=2)+'\n\nShared inherited/compression-induced errors overlap and the intersection is reported. Oracles use reference labels and are unattainable. AP uses4096 fixed bins. Pixel-width definitions and clouds follow registered masks. Parameter/MAC/bytes columns disclose cost conventions. Missing N251 sidecar configs are not replaced by obsolete checkpoints.\n'
    (c.ROOT/'report_v10.md').write_text(text);(c.ROOT/'negative_results.md').write_text('Cloud-shadow geometry was dropped and remains future work. OFA is a neutral existing ablation. Shared misses do not prove spectral invisibility. Historical numerical/technical failures, failed source gates, deferred optional work and scientific null results remain distinct. See experiment_checklist.csv and contrasts.csv for executed evidence.\n')
    c.save(c.ROOT/'collection_receipt.json',dict(outcome='completed',at=time.time(),interim=not bool(state.get('finished')),sealed_readout_complete=sealed_done,report_sha256=c.sha(c.ROOT/'report_v10.md')))
    c.save(c.ROOT/'evidence_index.json',dict(files={str(path.relative_to(c.ROOT)):c.sha(path) for path in c.ROOT.glob('*') if path.is_file() and path.suffix in ['.json','.csv','.md'] and path.name not in ['state.json','scheduler_status.json','control.json','evidence_index.json']}))
if __name__=='__main__':main()
