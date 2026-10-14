"""Declared mechanistic gates; exposed evidence is descriptive, no decisions feed training."""
import numpy as np
from collections import defaultdict
import common10 as c
from evaluate10 import scores

def avg(a):
    a=[x for x in a if x is not None and np.isfinite(x)];return float(np.mean(a)) if a else None

def collect(byid,bymodel,strata,shared,lookup,comparisons,contrast,group,csvout):
    datasets=['s11','sturm','wf','wf2'];gates=[]
    selected=c.read(c.ROOT/'smallest_viable_selection.json',{}).get('graph')
    if selected:
        rows=[r for r in comparisons if r['contrast']==selected+'_KDoff_minus_SUP'];sealed=next((r for r in rows if r['dataset']=='wf2'),None);positive=sum(r['delta_equal_event_water_IoU'] is not None and r['delta_equal_event_water_IoU']>0 and r['seeds']==4 for r in rows if r['dataset'] in ['s11','sturm','wf'])
        gates.append(dict(gate='A5 smallest viability depends on distillation',graph=selected,status='PASSED' if sealed and sealed['status']=='PASSED' and positive>=2 else 'SCIENTIFIC_GATE_FAILED' if sealed and sealed['seeds']==4 else 'UNAVAILABLE',positive_exposed_sets=positive))
    def rate(dataset,model,part,error):
        values=[]
        for (ds,name,event,label),a in strata.items():
            if ds!=dataset or name!=model or label!=part:continue
            numerator,denominator=(a[1],a[1]+a[3]) if error=='false_water' else (a[2],a[0]+a[2])
            if denominator:values.append(float(numerator/denominator))
        return avg(values)
    failures=[]
    candidates=[('representation','R2_log_CE','R2_CE'),('context','R2_ctx_CE','R2_CE'),('context','R4_micro_CE','R3_CE'),('data','R2_hardneg_CE','R2_CE'),('capacity','R4_CE','R2_CE'),('capacity','R4_CE','R3_CE')]
    for attribution,arm,baseline in candidates:
        for part,error in [('dark','false_water'),('bright','miss_water')]:
            passed_sets=[]
            for ds in datasets:
                ratios=[];consistent=0
                for seed in c.SEEDS:
                    a=rate(ds,f'{arm}_s{seed}',part,error);b=rate(ds,f'{baseline}_s{seed}',part,error);easy=rate(ds,f'R2_easyneg_CE_s{seed}',part,error)
                    if a is None or b is None or b==0:continue
                    ratios.append(1-a/b);consistent+=int(a<=.5*b and (attribution!='data' or easy is not None and a<easy))
                passed=len(ratios)==4 and consistent>=3 and avg(ratios)>=.5
                if passed:passed_sets.append(ds)
                failures.append(dict(attribution=attribution,arm=arm,baseline=baseline,dataset=ds,stratum=part,error=error,seeds=len(ratios),relative_reduction_mean=avg(ratios),consistent_seeds=consistent,local_gate=passed,scope='Teacher-referenced hard-negative cohort; reference-valid event rates'))
            cross='wf2' in passed_sets and sum(ds in passed_sets for ds in ['s11','sturm','wf'])>=2
            if attribution=='capacity':
                # Exclusive capacity attribution also needs no size-matched variant to satisfy the same gate.
                variants=[r for r in failures if r['attribution'] in ['representation','context','data'] and r['stratum']==part and r['error']==error]
                if any(sum(x['local_gate'] for x in variants if x['arm']==a and x['dataset'] in ['s11','sturm','wf'])>=2 and any(x['local_gate'] for x in variants if x['arm']==a and x['dataset']=='wf2') for a in {x['arm'] for x in variants}):cross=False
            gates.append(dict(gate='B1 '+attribution,arm=arm,baseline=baseline,stratum=part,status='PASSED' if cross else 'SCIENTIFIC_GATE_FAILED' if any(r['seeds']==4 and r['dataset']=='wf2' for r in failures if r['arm']==arm and r['stratum']==part) else 'UNAVAILABLE',sets_passing=passed_sets))
    csvout('failure_attribution_gates.csv',failures)
    # Concentration uses induced error density relative to pixel share, including overlaps explicitly.
    concentration=[];by_family=defaultdict(list)
    for ds,model,event,part in shared:
        entry=byid.get(model,{})
        if not entry.get('quantized') or part=='all' or entry.get('source_reference'):continue
        totals=shared.get((ds,model,event,'all'));pixels=strata[(ds,model,event,part)].sum();allpixels=strata[(ds,model,event,'all')].sum();induced=shared[(ds,model,event,part)][3]
        ratio=float((induced/totals[3])/(pixels/allpixels)) if totals[3] and pixels and allpixels else None
        concentration.append(dict(dataset=ds,model=model,seed=entry.get('seed'),event=event,stratum=part,induced_errors=int(induced),induced_share=float(induced/totals[3]) if totals[3] else None,pixel_share=float(pixels/allpixels) if allpixels else None,concentration=ratio))
        by_family[(ds,entry.get('graph'),entry.get('precision'),entry.get('control'),part,entry.get('seed'))].append(ratio)
    csvout('compression_concentration.csv',concentration)
    local={}
    for ds,graph,precision,control,part in {(k[0],k[1],k[2],k[3],k[4]) for k in by_family}:
        values=[avg(by_family.get((ds,graph,precision,control,part,s),[])) for s in c.SEEDS];local[(ds,graph,precision,control,part)]=sum(x is not None and x>=2 for x in values)>=3 and all(x is not None for x in values)
    for graph,precision,control,part in {(k[1],k[2],k[3],k[4]) for k in local}:
        cross=local.get(('wf2',graph,precision,control,part),False) and sum(local.get((ds,graph,precision,control,part),False) for ds in ['s11','sturm','wf'])>=2
        gates.append(dict(gate='B2 compression concentration',graph=graph,precision=precision,control=control,stratum=part,status='PASSED' if cross else 'SCIENTIFIC_GATE_FAILED' if ('wf2',graph,precision,control,part) in local else 'UNAVAILABLE'))
    controls=[]
    def induced_rate(ds,name,part):
        values=[]
        for (d,m,e,p),a in shared.items():
            if d==ds and m==name and p==part:
                n=strata[(d,m,e,p)].sum()
                if n:values.append(float(a[3]/n))
        return avg(values)
    for name,entry in byid.items():
        if entry.get('control') not in ['calibration','layer_protection','threshold_refit']:continue
        parent=entry.get('fp_reference');precision=entry.get('precision');unchanged=next((x['id'] for x in byid.values() if x.get('fp_reference')==parent and x.get('precision')==precision and x.get('quantized') and not x.get('control') and not x.get('source_reference')),None)
        if entry['control']=='threshold_refit':unchanged=entry.get('source_reference')
        if not unchanged:continue
        for ds in datasets:
            a=lookup.get((ds,name,'deployed'));b=lookup.get((ds,unchanged,'deployed'))
            if not a or not b:continue
            iou=a['equal_event_water_IoU']-b['equal_event_water_IoU'] if a['equal_event_water_IoU'] is not None and b['equal_event_water_IoU'] is not None else None
            for part in ['dark','bright','shore<=2','cloud<=50','landlike_water']:
                ar=induced_rate(ds,name,part);br=induced_rate(ds,unchanged,part);reduction=1-ar/br if ar is not None and br else None
                alarm=a.get('dry_alarm_01');basealarm=b.get('dry_alarm_01');alarm_ok=alarm is None and basealarm is None or alarm is not None and basealarm is not None and alarm-basealarm<=.01
                controls.append(dict(dataset=ds,model=name,seed=entry['seed'],control=entry['control'],unchanged_PTQ=unchanged,stratum=part,relative_induced_error_reduction=reduction,delta_equal_event_water_IoU=iou,delta_dry_alarm_01=alarm-basealarm if alarm is not None and basealarm is not None else None,local_gate=reduction is not None and reduction>=.5 and iou is not None and iou>=-.003 and alarm_ok,dry_windows_available=alarm is not None))
    csvout('compression_control_gates.csv',controls)
    for graph,control,part in {(byid[r['model']]['graph'],r['control'],r['stratum']) for r in controls}:
        byset={}
        for ds in datasets:
            rows=[r for r in controls if r['dataset']==ds and byid[r['model']]['graph']==graph and r['control']==control and r['stratum']==part]
            # Do not mix precisions or calibration choices into a four-seed vote.
            for precision in {byid[r['model']].get('precision') for r in rows}:
                chosen=[r for r in rows if byid[r['model']].get('precision')==precision]
                byset[(ds,precision)]={r['seed'] for r in chosen if r['local_gate']}
        for precision in {p for _,p in byset}:
            passed=len(byset.get(('wf2',precision),set()))>=3 and sum(len(byset.get((ds,precision),set()))>=3 for ds in ['s11','sturm','wf'])>=2
            gates.append(dict(gate='B2 control reduces induced stratum error',graph=graph,precision=precision,control=control,stratum=part,status='PASSED' if passed else 'SCIENTIFIC_GATE_FAILED' if ('wf2',precision) in byset else 'UNAVAILABLE'))
    tolerances=[]
    for entry in byid.values():
        if entry.get('precision')!='W8A8' or entry.get('source_reference'):continue
        a=c.read(c.ROOT/'source_metrics'/(entry['id']+'.json'));b=c.read(c.ROOT/'source_metrics'/(str(entry.get('fp_reference'))+'.json'))
        if not a or not b:continue
        gap=a['equal_event_water_IoU']-b['equal_event_water_IoU'];tolerances.append(dict(model=entry['id'],seed=entry.get('seed'),source_delta=gap,within_1pp=gap>=-.01,within_2pp=gap>=-.02,within_3pp=gap>=-.03,native_lossless_claim=False))
    csvout('strict_W8A8_source_tolerances.csv',tolerances)
    # Paired QAT rescue requires both baselines and source tolerance. Every admitted seed remains visible.
    qat=[x for x in byid.values() if x.get('control') in ['QAT_d0','QAT_d1','QAT_d2']]
    for graph,precision in {(x['graph'],x['precision']) for x in qat}:
        def ids(method):return {x['seed']:x['id'] for x in qat if x['graph']==graph and x['precision']==precision and x['method']==method}
        parents={s:c.read(c.ROOT/'qat_parents'/f'{graph}_{precision}_s{s}.json',{}).get('entry',{}).get('id') for s in c.SEEDS}
        for method in ['d1','d2']:
            rows=contrast(f'{graph}_{precision}_{method}_minus_d0',ids(method),ids('d0'),threshold=.005)+contrast(f'{graph}_{precision}_{method}_minus_PTQ',ids(method),parents,threshold=.005);comparisons.extend(rows)
            gaps=[]
            for seed,name in ids(method).items():
                a=c.read(c.ROOT/'source_metrics'/(name+'.json'));fp=byid[name].get('fp_reference');b=c.read(c.ROOT/'source_metrics'/(str(fp)+'.json'))
                if a and b:gaps.append(a['equal_event_water_IoU']-b['equal_event_water_IoU'])
            sealed=[r for r in rows if r['dataset']=='wf2'];gates.append(dict(gate='B2 QAT rescue',graph=graph,precision=precision,method=method,source_gap=avg(gaps),status='PASSED' if len(sealed)==2 and all(r['status']=='PASSED' for r in sealed) and len(gaps)==4 and avg(gaps)>=-.01 else 'SCIENTIFIC_GATE_FAILED' if sealed and all(r['seeds']==4 for r in sealed) else 'UNAVAILABLE'))
        comparisons.extend(contrast(f'{graph}_{precision}_d2_minus_d1',ids('d2'),ids('d1'),threshold=.005))
    C1=[]
    for other in ['F0','F2']:
        rows=contrast('C1_F1_minus_'+other,group('R4','F1'),group('R4',other),threshold=.005);comparisons.extend(rows);C1.extend(rows)
    sealed=[r for r in C1 if r['dataset']=='wf2'];sturm=[r for r in C1 if r['dataset']=='sturm']
    gates.append(dict(gate='C1 F1 beats F0 and F2 without STURM loss',status='PASSED' if len(sealed)==2 and all(r['status']=='PASSED' for r in sealed) and len(sturm)==2 and all(r['seeds']==4 and r['delta_equal_event_water_IoU']>=-.005 for r in sturm) else 'SCIENTIFIC_GATE_FAILED' if sealed and all(r['seeds']==4 for r in sealed) else 'UNAVAILABLE'))
    if c.read(c.ROOT/'decode_rule_selection.json',{}).get('outcome')=='completed':
        base=group('R4','lkd');decode_ids={s:name+'__decode_gate' for s,name in base.items()}
        for s,name in base.items():
            byid[decode_ids[s]]=dict(byid[name],id=decode_ids[s])
            for ds in datasets:
                if (ds,name,'decode') in bymodel:bymodel[(ds,decode_ids[s],'deployed')]=bymodel[(ds,name,'decode')]
        rows=contrast('C2_decode_minus_deployed_R4',decode_ids,base,threshold=.005);comparisons.extend(rows);sealed=next((r for r in rows if r['dataset']=='wf2'),None);sturm=next((r for r in rows if r['dataset']=='sturm'),None)
        gates.append(dict(gate='C2 decode rule improves without STURM loss',status='PASSED' if sealed and sealed['status']=='PASSED' and sturm and sturm['seeds']==4 and sturm['delta_equal_event_water_IoU']>=-.005 else 'SCIENTIFIC_GATE_FAILED' if sealed and sealed['seeds']==4 else 'UNAVAILABLE'))
        for name in decode_ids.values():
            byid.pop(name)
            for ds in datasets:bymodel.pop((ds,name,'deployed'),None)
    csvout('decision_gates.csv',gates)
    csvout('mixed_pixel_results.csv',[dict(row) for row in lookup.values() if byid.get(row['model'],{}).get('method') in ['F0','F1','F2','F3','F4']] or [dict(status='DEFERRED',reason='No completed mixed-pixel readout')])
    csvout('decode_rule_results.csv',[dict(row) for row in lookup.values() if row['variant']=='decode'] or [dict(status='DEFERRED',reason='No completed decode readout')])
    ort=c.read(c.ROOT/'strict_int8_ort.json',{});csvout('strict_int8_ort.csv',ort.get('rows',[dict(status=ort.get('outcome','DEFERRED'),reason=ort.get('reason','Optional job not executed'))]))
