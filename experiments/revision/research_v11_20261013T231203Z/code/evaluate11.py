"""One final V11 readout over frozen V9 windows; no selection from target scores."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import common11 as c

def paired_interval(values, level=.95):
    """Columns=seeds, rows=whole events; seed-mean paired estimand."""
    values=np.asarray(values,dtype=float)
    assert values.ndim==2 and values.shape[1]==4 and len(values)>0 and np.isfinite(values).all()
    assert 0 < level < 1
    event_difference=values.mean(1)
    rng=np.random.default_rng(1729)
    draws=event_difference[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    alpha=(1-level)/2
    return [float(x) for x in np.quantile(draws,[alpha,1-alpha])]

def forward_order(entries):
    """Resolve aliases independent of manifest order, rejecting cycles/missing bases."""
    byid={entry['id']:entry for entry in entries}
    if len(byid)!=len(entries):raise ValueError('Duplicate model ID in final freeze')
    ordered=[];done=set();active=set()
    def visit(name):
        if name not in byid:raise ValueError('Missing alias source '+name)
        if name in active:raise ValueError('Cyclic alias '+name)
        if name in done:return
        active.add(name);entry=byid[name]
        if entry.get('alias_of'):visit(entry['alias_of'])
        active.remove(name);done.add(name);ordered.append(entry)
    for entry in entries:visit(entry['id'])
    return ordered

def alias_root(name,byid):
    seen=set()
    while byid[name].get('alias_of'):
        if name in seen:raise ValueError('Cyclic alias '+name)
        seen.add(name);name=byid[name]['alias_of']
    return name

def water_iou(cell):
    if cell is None:return None
    tp,fp,fn,_=map(int,cell)
    return tp/(tp+fp+fn) if tp+fp+fn else None

def contrast_rows(cells,entries):
    """Paired, equally weighted event readouts; all undefined unions are disclosed.

    The two co-primary contrasts share the intersection of defined D/U/A events
    across every seed. This is independent of the sign or magnitude of scores.
    """
    result=[]
    for point in sorted({e.get('point','core') for e in entries}):
        primary=[e for e in entries if e.get('point','core')==point and e.get('endpoint')=='primary600']
        groups={arm:{e['seed']:e['id'] for e in primary if e['arm']==arm}
                for arm in ['D','U','A','L','N','P','Q0','F0']}
        # Optional threshold aliases use Q0's score with a source-only tau.
        for arm in ['Q0_tau_D','Q0_tau_U','Q0_tau_A']:
            groups[arm]={e['seed']:e['id'] for e in entries if e.get('point','core')==point and e['arm']==arm}
        if set(groups['D'])!=set(c.SEEDS):continue
        for dataset in sorted({key[0] for key in cells}):
            all_events=sorted({key[2] for key in cells if key[0]==dataset})
            def defined(event,arms):
                return all(set(groups[arm])==set(c.SEEDS) and all(
                    water_iou(cells.get((dataset,groups[arm][seed],event))) is not None
                    for seed in c.SEEDS) for arm in arms)
            coprimary=[event for event in all_events if defined(event,['D','U','A'])]
            for baseline in ['U','A','L','N','P','Q0','F0','Q0_tau_D','Q0_tau_U','Q0_tau_A']:
                if set(groups[baseline])!=set(c.SEEDS):continue
                kept=coprimary if baseline in ['U','A'] else [event for event in all_events if defined(event,['D',baseline])]
                omitted=[event for event in all_events if event not in kept]
                row=dict(dataset=dataset,point=point,endpoint='primary600',contrast='D-'+baseline,
                         events=len(kept),total_events=len(all_events),undefined_events=len(omitted),
                         included_events=json.dumps(kept),undefined_event_ids=json.dumps(omitted),
                         event_policy='common D/U/A defined union across all four seeds' if baseline in ['U','A'] else 'pairwise common defined union across all four seeds',
                         bootstrap_draws=10000,bootstrap_seed=1729,
                         interval_unit='whole events after averaging seed-paired differences; not pixel or training-seed bootstrap')
                if not kept:
                    row.update(status='UNDEFINED_NO_COMMON_EVENT_UNION',difference=None,D_mean=None,baseline_mean=None)
                    result.append(row);continue
                absolute={arm:np.array([[water_iou(cells[(dataset,groups[arm][seed],event)])
                                        for seed in c.SEEDS] for event in kept]) for arm in ['D',baseline]}
                values=absolute['D']-absolute[baseline];means=values.mean(0);eventmean=values.mean(1)
                ci=paired_interval(values);sim=paired_interval(values,.975) if baseline in ['U','A'] else [None,None]
                row.update(status='completed',difference=float(values.mean()),D_mean=float(absolute['D'].mean()),
                           baseline_mean=float(absolute[baseline].mean()),seed_SD=float(means.std(ddof=1)),
                           D_seed_SD=float(absolute['D'].mean(0).std(ddof=1)),
                           baseline_seed_SD=float(absolute[baseline].mean(0).std(ddof=1)),
                           CI95_low=ci[0],CI95_high=ci[1],
                           co_primary_bonferroni975_low=sim[0],co_primary_bonferroni975_high=sim[1],
                           co_primary_family='D-U and D-A only' if baseline in ['U','A'] else None,
                           worst_event=kept[int(eventmean.argmin())],worst_difference=float(eventmean.min()),
                           lower_decile_mean=float(np.sort(eventmean)[:max(1,math.ceil(.1*len(eventmean)))].mean()))
                for i,seed in enumerate(c.SEEDS):
                    row[f'seed{seed}']=float(means[i]);row[f'D_seed{seed}']=float(absolute['D'][:,i].mean())
                    row[f'baseline_seed{seed}']=float(absolute[baseline][:,i].mean())
                result.append(row)
    return result

def load(entry,torch,c10):
    assert c.sha(entry['checkpoint'])==entry['sha256'], entry['id']
    payload=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
    state=payload['model']
    model=c10.build(entry['graph'])
    if any(k.endswith('._w_scale') for k in state):
        from quant10 import load as load_quant
        model=load_quant(model,state)
        enabled=entry.get('arm')!='F0' and entry.get('quantized',True)
        for module in model.modules():
            if hasattr(module,'fake_quant_enabled'):module.fake_quant_enabled=enabled
    else:model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False)
    if entry.get('quantized'):
        quantizers=[m for m in model.modules() if hasattr(m,'fake_quant_enabled')]
        assert quantizers and all(m.fake_quant_enabled for m in quantizers),'Declared low precision disabled'
    return model

def evaluate():
    torch,c10,v=c.setup()
    from evaluate10 import counts,scores,hist,ap,strata,probability_bins
    frozen=c.read(c.ROOT/'final_evaluation_freeze.json')
    assert frozen and frozen['outcome']=='completed' and frozen['models']
    assert not (c.ROOT/'evaluation/started.json').exists(),'Final phase already started: do not silently reread target labels'
    for item in frozen['membership']:
        assert c.sha(item['path'])==item['sha256']
    rows=[]
    for item in frozen['membership']:
        obj=c.read(item['path'])
        rows.extend(r for r in obj.get('windows',obj.get('rows',[])) if not r['omit'] and r['dataset']!='s11val')
    entries=forward_order(frozen['models']);byid={e['id']:e for e in entries}
    assert len(byid)==len(entries)
    models={e['id']:load(e,torch,c10) for e in entries if not e.get('alias_of')}
    # Count actual Conv2d arithmetic on the first existing evaluation forward.
    # This adds neither a target read nor another model forward. Fake quantizer
    # rounding/clipping and non-convolution operations are separately disclosed.
    conv_macs=defaultdict(int);mac_handles=[]
    for name,model in models.items():
        def mac_hook(module,inputs,output,key=name):
            conv_macs[key]+=int(output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size))
        for module in model.modules():
            if isinstance(module,torch.nn.Conv2d):mac_handles.append(module.register_forward_hook(mac_hook))
    out=c.ROOT/'evaluation';out.mkdir(exist_ok=True)
    c.save(out/'started.json',dict(at=time.time(),freeze_sha256=c.sha(c.ROOT/'final_evaluation_freeze.json'),windows=len(rows),models=len(models),look='non-confirmatory'))
    cells=defaultdict(lambda:np.zeros(4,np.int64))
    parts=defaultdict(lambda:np.zeros(4,np.int64))
    histograms=defaultdict(lambda:np.zeros((4096,2),np.int64))
    calibration=defaultdict(lambda:np.zeros((15,4),float))
    harm=defaultdict(lambda:np.zeros(4,np.int64))
    timing=defaultdict(lambda:[0.,0])
    forward_timing=defaultdict(lambda:[0.,0])
    dry=[]
    started=time.monotonic();cpu_started=time.process_time()
    with torch.inference_mode(), (c.ROOT/'label_access.jsonl').open('a',buffering=1) as accesses:
        for i,row in enumerate(rows):
            t=time.monotonic()
            with np.load(row['path'],allow_pickle=False) as payload:
                x=payload['image'].copy();y=payload['mask'].copy()
                cloud=payload['cloud'].copy() if 'cloud' in payload and payload['cloud'].size else None
            accesses.write(json.dumps(dict(at=time.time(),path=row['path'],dataset=row['dataset'],event=row['event'],purpose='single final V11 frozen-model readout',derived_reference=True))+'\n')
            rho=v.reflectance(x);regions=strata(y,rho,cloud);valid=y!=-1
            margins={};predictions={};tile_histograms={};tile_calibration={}
            for entry in entries:
                name=entry['id'];tic=time.monotonic()
                if entry.get('alias_of'):m=margins[entry['alias_of']]
                else:
                    z=models[name](torch.from_numpy(x)[None]);assert torch.isfinite(z).all()
                    m=(z[0,1]-z[0,0]).numpy()
                    forward_timing[name][0]+=time.monotonic()-tic;forward_timing[name][1]+=1
                assert m.shape==y.shape,'Retain native registered 512 geometry'
                margins[name]=m;predictions[name]=m>entry.get('threshold',0.)
                timing[name][0]+=time.monotonic()-tic;timing[name][1]+=1
            if i==0:
                for handle in mac_handles:handle.remove()
            for entry in entries:
                name=entry['id'];m=margins[name];p=predictions[name];key=(row['dataset'],name,row['event'])
                cells[key]+=counts(p,y)
                base=alias_root(name,byid)
                if base not in tile_histograms:
                    tile_histograms[base]=hist(m[valid],y[valid].astype(np.int64))
                    tile_calibration[base]=probability_bins(m,y,np.ones(y.shape,bool))
                histograms[key]+=tile_histograms[base]
                calibration[key]+=tile_calibration[base]
                for region,mask in regions.items():parts[key+(region,)]+=counts(p,y,mask)
                ref=predictions.get(entry.get('fp_reference'))
                if ref is not None:
                    # Reference-conditioned corrections; teacher/shared-error categories need not be disjoint.
                    harm[key]+=np.array([np.count_nonzero(valid&(y==0)&ref&~p),np.count_nonzero(valid&(y==1)&~ref&p),np.count_nonzero(valid&(y==0)&~ref&p),np.count_nonzero(valid&(y==1)&ref&~p)])
                if valid.any() and not np.any(y[valid]==1):
                    n=int(valid.sum());water=int(np.count_nonzero(p&valid))
                    dry.append(dict(dataset=row['dataset'],model=name,event=row['event'],window=row['name'],valid_pixels=n,false_water_pixels=water,alarm_001=water>=.001*n,alarm_01=water>=.01*n,alarm_05=water>=.05*n))
            if i%100==0:
                print('final V11',i,'/',len(rows),'seconds',time.monotonic()-started,flush=True)
                c.save(out/'progress.json',dict(at=time.time(),completed_windows=i+1,total_windows=len(rows),elapsed_seconds=time.monotonic()-started))
    event_rows=[]
    for (dataset,name,event),cell in cells.items():
        entry=byid[name]
        event_rows.append(dict(dataset=dataset,model=name,point=entry.get('point','core'),arm=entry['arm'],seed=entry['seed'],step=entry['step'],endpoint=entry.get('endpoint','primary600'),event=event,tp=int(cell[0]),fp=int(cell[1]),fn=int(cell[2]),tn=int(cell[3]),reference_water_pixels=int(cell[0]+cell[2]),valid_pixels=int(cell.sum()),water_union_pixels=int(cell[:3].sum()),water_IoU_status='defined' if cell[:3].sum() else 'undefined_empty_union',**scores(cell),AP_binned=ap(histograms[(dataset,name,event)])))
    c.csvout(c.ROOT/'repair_by_event.csv',event_rows)
    c.csvout(out/'strata.csv',[dict(dataset=ds,model=name,event=event,stratum=part,tp=int(a[0]),fp=int(a[1]),fn=int(a[2]),tn=int(a[3]),**scores(a)) for (ds,name,event,part),a in parts.items()])
    c.csvout(out/'dry_window_alarms.csv',dry)
    c.csvout(c.ROOT/'correction_harm.csv',[dict(dataset=ds,model=name,arm=byid[name]['arm'],seed=byid[name]['seed'],event=event,corrected_false_water=int(a[0]),corrected_missed_water=int(a[1]),new_false_water=int(a[2]),new_missed_water=int(a[3])) for (ds,name,event),a in harm.items()])
    c.save(out/'calibration.json',dict(bins=15,rows=[dict(dataset=ds,model=name,event=event,counts_sum_p_sum_y_sum_squared_error=a.tolist()) for (ds,name,event),a in calibration.items()]))
    summaries=[]
    for ds,name in sorted({(ds,name) for ds,name,event in cells}):
        keys=[k for k in cells if k[:2]==(ds,name)];vals=[scores(cells[k])['water_IoU'] for k in keys];vals=[v for v in vals if v is not None]
        totals=sum(cells[k] for k in keys);alarms=[x for x in dry if (x['dataset'],x['model'])==(ds,name)]
        summaries.append(dict(dataset=ds,model=name,point=byid[name].get('point','core'),endpoint=byid[name].get('endpoint','primary600'),arm=byid[name]['arm'],seed=byid[name]['seed'],threshold=byid[name].get('threshold',0.),equal_event_water_IoU=float(np.mean(vals)) if vals else None,pooled_water_IoU=scores(totals)['water_IoU'],pooled_mIoU=scores(totals)['mIoU'],AP_binned=ap(sum(histograms[k] for k in keys)),events=len(keys),defined_water_IoU_events=len(vals),undefined_water_IoU_events=len(keys)-len(vals),valid_pixels=int(totals.sum()),reference_water_pixels=int(totals[0]+totals[2]),dry_windows=len(alarms),dry_alarm_001=float(np.mean([x['alarm_001'] for x in alarms])) if alarms else None,dry_alarm_01=float(np.mean([x['alarm_01'] for x in alarms])) if alarms else None,dry_alarm_05=float(np.mean([x['alarm_05'] for x in alarms])) if alarms else None))
    c.csvout(out/'summary.csv',summaries)
    seed_summaries=[]
    for ds,point,endpoint,arm in sorted({(r['dataset'],r['point'],r['endpoint'],r['arm']) for r in summaries}):
        group=[r for r in summaries if (r['dataset'],r['point'],r['endpoint'],r['arm'])==(ds,point,endpoint,arm)]
        row=dict(dataset=ds,point=point,endpoint=endpoint,arm=arm,seeds=len(group),complete_seed_set=sorted(r['seed'] for r in group)==sorted(c.SEEDS))
        for metric in ['equal_event_water_IoU','pooled_water_IoU','pooled_mIoU','AP_binned','dry_alarm_001','dry_alarm_01','dry_alarm_05']:
            values=[r[metric] for r in group if r[metric] is not None]
            row[metric+'_mean']=float(np.mean(values)) if values else None
            row[metric+'_seed_SD']=float(np.std(values,ddof=1)) if len(values)>1 else None
            row[metric+'_defined_seeds']=len(values)
        seed_summaries.append(row)
    c.csvout(out/'seed_summary.csv',seed_summaries)
    contrasts=contrast_rows(cells,entries)
    c.csvout(c.ROOT/'contrasts.csv',contrasts)
    inference_cost=[]
    for name,t in timing.items():
        base=alias_root(name,byid);forward=forward_timing[base]
        # Threshold aliases share one forward in this batch, but deployment still
        # needs that forward; an alias's few microseconds are not model latency.
        standalone=forward[0]+t[0] if name!=base else t[0]
        model=models[base]
        inference_cost.append(dict(model=name,alias_of=byid[name].get('alias_of'),forward_model=base,
                                   seconds_in_batch=t[0],tiles=t[1],mean_seconds=standalone/t[1],
                                   mean_shared_forward_seconds=forward[0]/forward[1],
                                   training_checkpoint_bytes=Path(byid[base]['checkpoint']).stat().st_size,
                                   model_state_tensor_bytes=sum(p.numel()*p.element_size() for p in model.state_dict().values()),
                                   parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                                   buffer_tensor_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
                                   parameter_count=sum(p.numel() for p in model.parameters()),
                                   convolution_MACs_first_canonical_tile=conv_macs[base],
                                   convolution_MACs_scope='Conv2d multiply-accumulates only; excludes nonlinearities, interpolation, quantizer rounding/clipping and thresholding',
                                   precision_map=json.dumps({key:dict(weight_bits=getattr(module,'weight_bits',None),activation_bits=getattr(module,'activation_bits',None),fake_quant_enabled=getattr(module,'fake_quant_enabled',None),storage_dtype=str(module.weight.dtype)) for key,module in model.named_modules() if isinstance(module,torch.nn.Conv2d)}),
                                   precision=byid[name].get('precision'),native_low_bit_kernel=False,
                                   latency_scope='model forward and threshold on cached normalized tensors; excludes normalization, shared file IO and metric computation; full evaluation wall/CPU cost recorded separately'))
    c.csvout(out/'inference_cost.csv',inference_cost)
    c.save(out/'receipt.json',dict(outcome='completed',windows=len(rows),models=len(models),seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu_started,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,freeze_sha256=c.sha(c.ROOT/'final_evaluation_freeze.json'),AP='4096 fixed bins [-32,32]; approximate ranking AP',labels='original V9 derived native reference masks, one read per window',exposure='non-confirmatory',runtime='CPU fake quantization; no native INT4/INT8 speed claim'))

if __name__=='__main__':evaluate()
