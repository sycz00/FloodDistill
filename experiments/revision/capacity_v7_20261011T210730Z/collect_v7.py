"""Idempotent CPU-only collection from actual v7 state and verified reuse records."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from evaluate_v7 import atomic_json, file_hash, now, read, scores, sum_mechanisms

SEEDS=(42,64,91,107)
METRICS=('mIoU','water_IoU','water_precision','water_recall')
TERMINAL={'completed','reused','technical_failure','blocked','deferred_budget',
          'failed_scientific_gate','not_run','unsupported'}


def optional(path, default=None):
    return read(path) if Path(path).is_file() else default


def write_text(path, text):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(text);os.replace(tmp,path)


def csv_write(path, rows, fields=None):
    fields=fields or sorted(set().union(*(set(row) for row in rows))) if rows else fields or ['status']
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value for key,value in row.items()})
    os.replace(tmp,path)


def source_metrics(receipt, kind, reused=None):
    reused=reused or {}; source=receipt.get('source_validation',{})
    direct=reused.get(kind+'_source_metrics') or source.get(kind+'_metrics') or receipt.get(kind+'_metrics')
    if direct:
        values={key:direct.get(key) for key in METRICS}
        if values['water_precision'] is None:values['water_precision']=direct.get('precision')
        if values['water_recall'] is None:values['water_recall']=direct.get('recall')
        if values['water_IoU'] is None:
            values['water_IoU']=direct.get('wIoU')
            if values['water_IoU'] is None and len(direct.get('iou_per_class',[]))>1: values['water_IoU']=direct['iou_per_class'][1]
        return values
    if kind=='selected':
        return dict(mIoU=source.get('best_val_mIoU'),water_IoU=None,water_precision=None,water_recall=None)
    rows=[row for row in receipt.get('epochs',[]) if row.get('phase')=='val']
    if not rows:return {key:None for key in METRICS}
    row=rows[-1];ious=row.get('iou_per_class',[])
    return dict(mIoU=row.get('mIoU'),water_IoU=ious[1] if len(ious)>1 else None,water_precision=None,water_recall=None)


def family_for(job, receipt, lr_selections):
    graph=receipt.get('graph',job.get('graph','unknown'))
    method=receipt.get('arm',receipt.get('method',job.get('method','unknown')))
    if method in ('C0','C1','C2','C3','C4','C5','HC','HL'):return method
    if graph in ('A0','A1') and method.lower()=='lkd':
        selection=lr_selections.get(graph)
        return graph+'_LKD_reanchored' if selection and float(receipt.get('lr',0))==float(selection['lr']) else f'{graph}_LKD_lr{receipt.get("lr",job.get("lr"))}'
    return graph+'_'+method.upper()


def paired_interval(differences):
    values=[value for value in differences if value is not None and math.isfinite(value)]
    result=dict(n_pairs=len(values),mean_difference_pp=statistics.mean(values) if values else None,
                sample_sd_pp=statistics.stdev(values) if len(values)>1 else None,
                ci95_low_pp=None,ci95_high_pp=None,
                uncertainty_type='four paired training seeds, df3 Student-t' if len(values)==4 else 'descriptive only; incomplete four-seed pairing')
    if len(values)==4:
        half=3.182446305284263*result['sample_sd_pp']/2
        result.update(ci95_low_pp=result['mean_difference_pp']-half,ci95_high_pp=result['mean_difference_pp']+half)
    return result


def bootstrap(differences):
    import numpy as np
    values=np.asarray([value for value in differences if value is not None and math.isfinite(value)],dtype=float)
    if not len(values):return dict(n_events=0,mean_difference_pp=None,ci95_low_pp=None,ci95_high_pp=None)
    rng=np.random.default_rng(1729)
    samples=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(n_events=len(values),mean_difference_pp=float(values.mean()),
                ci95_low_pp=float(np.percentile(samples,2.5)),ci95_high_pp=float(np.percentile(samples,97.5)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--reason',default='status')
    args=parser.parse_args();root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    # This process is always CPU-only, including plotting imports.
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mpl_cache'))
    state=optional(root/'state.json',{});manifest=optional(root/'manifest_initial.json',{})
    reuse_path=root/'provenance/reuse_index.json'
    if not reuse_path.exists():reuse_path=root/'reuse_index.json'
    reuse=optional(reuse_path,{'records':[]});a0_selection=optional(root/'selections/A0_lr.json')
    a1_selection=optional(root/'selections/A1_lr.json');lr_selections={'A0':a0_selection,'A1':a1_selection}
    optional_reuse_paths=sorted(root.glob('optional_*_reuse_index.json'))
    reuse_records=reuse.get('records',[])+[record for path in optional_reuse_paths for record in read(path).get('records',[])]
    promotion=optional(root/'selections/promotion.json');budget=optional(root/'budget_ledger.json',{})
    inputs=[Path(__file__),root/'state.json',root/'manifest_initial.json',reuse_path,root/'budget_ledger.json',
            root/'selections/A0_lr.json',root/'selections/A1_lr.json',root/'selections/promotion.json',root/'diagnostics.json']+optional_reuse_paths
    terminal_files=sorted((root/'terminal').glob('*.json'))
    deployment_files=sorted((root/'receipts').glob('*deploy*.json'))+sorted(root.glob('*deployment*.json'))
    deployment_files+= [Path(record.get('job',record)['receipt']) for record in state.get('jobs',{}).values()
        if record.get('job',record).get('role') in ('deploy','deployment') and record.get('job',record).get('receipt')
        and Path(record.get('job',record)['receipt']).is_file()]
    deployment_files=sorted(set(deployment_files))
    records=[];errors=[];inputs+=terminal_files+deployment_files
    for record in reuse_records:
        if not record.get('verified',record.get('status')=='reused'):
            errors.append('Blocked historical reuse: '+str(record.get('family'))+' seed'+str(record.get('seed')));continue
        path=Path(record['receipt_path']);terminal_path=Path(record['terminal_receipt_path'])
        if (record.get('receipt_sha256') and file_hash(path)!=record['receipt_sha256']) or (
                record.get('terminal_receipt_sha256') and file_hash(terminal_path)!=record['terminal_receipt_sha256']):
            errors.append('Changed reuse receipt bytes: '+str(path));continue
        receipt=read(path);terminal=read(terminal_path)
        if terminal.get('checkpoint_sha256')!=record['best_sha256']:
            errors.append('Changed reuse checkpoint identity: '+str(path));continue
        refreshed_path=root/'terminal'/(path.stem+'__selected.json')
        refreshed=optional(refreshed_path,{})
        if refreshed.get('checkpoint_sha256')==record['best_sha256'] and refreshed.get('mechanism_counts'):
            verified_mechanisms={}
            for dataset,counts in refreshed['mechanism_counts'].items():
                if refreshed['terminal_results'][dataset]['per_image_counts']==terminal['terminal_results'][dataset]['per_image_counts']:
                    verified_mechanisms[dataset]=counts
                else:errors.append('Reused mechanism evaluation prediction/count parity failed: '+str(path)+' '+dataset)
            terminal={**terminal,'mechanism_counts':verified_mechanisms}
            inputs.append(refreshed_path)
        family=record['family']
        if family in ('R4_LKD_selected','A3_LKD_selected','A3N_LKD_selected'):family=family.removesuffix('_selected')
        if family=='A0_LKD_recipe' and a0_selection and float(a0_selection['lr'])==5e-5:family='A0_LKD_reanchored'
        if family=='A1_LKD_recipe' and a1_selection and float(a1_selection['lr'])==5e-5:family='A1_LKD_reanchored'
        records.append(dict(id=path.stem,family=family,seed=record['seed'],graph=record['graph'],method=record['method'],
            status='reused',receipt=receipt,receipt_path=str(path),reuse=record,terminals={'selected':terminal},
            reason='verified historical checkpoint and per-image counts',job={}))
        inputs += [path,terminal_path]
    all_terminal={}
    for path in terminal_files:
        payload=read(path)
        if not payload.get('smoke_only') and payload.get('terminal_results'):
            all_terminal[path.stem]=payload
    jobs=state.get('jobs',{})
    if not jobs:
        jobs={job['id']:dict(job=job,status='not_run') for job in manifest.get('jobs',[])}
    for jid,job_state in jobs.items():
        job=job_state.get('job',job_state)
        if job.get('role')!='train':continue
        path=Path(job.get('receipt',job.get('output','/nonexistent')))
        receipt=optional(path,{})
        if receipt.get('outcome')=='smoke_completed' or receipt.get('smoke_only'):continue
        status=job_state.get('status','not_run')
        if status=='reused' and not receipt:
            existing=next((record for record in records if record['graph']==job.get('graph') and record['seed']==job.get('seed')
                and record['method'].lower()==job.get('method','').lower()),None)
            if existing:
                existing['reason']='verified historical checkpoint reused by '+jid
                continue
        if status=='completed' and receipt.get('outcome')!='completed':
            status='technical_failure';errors.append('Completed state lacks successful training receipt: '+jid)
        terminal={kind:all_terminal.get(path.stem+'__'+kind) for kind in ('selected','final')}
        for kind,payload in terminal.items():
            expected=receipt.get(('best' if kind=='selected' else 'last')+'_sha256')
            if payload and payload['checkpoint_sha256']!=expected:
                errors.append('Terminal checkpoint mismatch: '+jid+' '+kind);terminal[kind]=None
        if path.exists():inputs.append(path)
        records.append(dict(id=jid,family=family_for(job,receipt,lr_selections),
            seed=receipt.get('seed',job.get('seed')),graph=receipt.get('graph',job.get('graph')),
            method=receipt.get('method',job.get('method')),status=status,receipt=receipt,receipt_path=str(path),
            reuse={},terminals=terminal,reason=job_state.get('reason'),job=job))
    # A0 selected seed42 is one compatible grid/reused checkpoint, not another run.
    for graph,selection in lr_selections.items():
        if not selection:continue
        path=Path(selection['receipt']);receipt=optional(path)
        if receipt and not any(record['family']==graph+'_LKD_reanchored' and record['seed']==42 for record in records):
            matching=next((record for record in records if record['receipt'].get('best_sha256')==receipt.get('best_sha256')),None)
            if matching:records.append({**matching,'family':graph+'_LKD_reanchored','id':matching['id']+'_selected_alias'})
    fingerprint=hashlib.sha256(json.dumps({str(path):file_hash(path) for path in sorted(set(inputs)) if path.is_file()},sort_keys=True).encode()).hexdigest()
    previous=optional(root/'completion_receipt.json',{})
    prior_outputs=previous.get('outputs',{})
    if previous.get('input_fingerprint')==fingerprint and previous.get('collection_reason')==args.reason and prior_outputs and all(
            Path(value['path']).is_file() and file_hash(value['path'])==value['sha256'] for value in prior_outputs.values()):
        print('Collection unchanged:',root/'report_v7.md');return
    result_rows=[];event_rows=[];mechanism_rows=[];family_source=defaultdict(dict);family_ext=defaultdict(dict);family_events=defaultdict(dict)
    seen=set()
    for record in records:
        receipt=record['receipt'];status=record['status'];family=record['family'];seed=record['seed']
        key=(family,seed,receipt.get('best_sha256')) if receipt.get('best_sha256') else (family,seed,record['id'])
        if key in seen:continue
        seen.add(key)
        common=dict(run_id=record['id'],family=family,graph=record['graph'],method=record['method'],seed=seed,
            training_status=status,reason=record['reason'],training_receipt=record['receipt_path'],
            completed_updates=receipt.get('completed_updates'),declared_updates=record['job'].get('updates',
                1560 if record['method'] in ('C0','C1','C2','C3','C4','C5','HC','HL') else 7800),
            selected_updates=receipt.get('selected_updates'),lr=receipt.get('lr',record['job'].get('lr')),
            selected_checkpoint_path=receipt.get('best_path'),selected_checkpoint_sha256=receipt.get('best_sha256'),
            final_checkpoint_path=receipt.get('last_path'),final_checkpoint_sha256=receipt.get('last_sha256'),metric_unit='percent',
            hard_target_definition='cached_teacher_pseudolabel_focal',soft_target_definition='live_augmented_teacher_KL' if record['method'] not in ('ce','CE','HC','HL') else record['method'])
        for kind in ('selected','final'):
            metrics=source_metrics(receipt,kind,record['reuse'])
            rule_scalars={}
            if record['method'] in ('C1','C3','C4','C5') and status in ('completed','reused'):
                checkpoint_path=receipt.get(('best' if kind=='selected' else 'last')+'_path')
                if checkpoint_path and Path(checkpoint_path).is_file():
                    import torch
                    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                    if 'alpha' in payload['model'] and 'beta' in payload['model']:
                        alpha=float(payload['model']['alpha']);beta=float(payload['model']['beta'])
                        rule_scalars=dict(rule_alpha=alpha,rule_beta=beta,rule_a=8*math.tanh(alpha),rule_b=8*math.tanh(beta),
                            rule_coefficient_unit='additive binary logit margin')
                    del payload
            result_rows.append({**common,'checkpoint_kind':kind,'dataset':'source_validation','domain':'full',
                'readout_status':status if metrics.get('mIoU') is not None else 'not_run',
                **rule_scalars,
                **{key:value*100 if value is not None else None for key,value in metrics.items()}})
            if metrics.get('mIoU') is not None:family_source[(family,kind)][seed]=metrics
            terminal=record['terminals'].get(kind)
            for dataset in ('s11','sturm','wf'):
                for domain in ('full','known_overlap_excluded'):
                    readout=terminal.get('terminal_results',{}).get(dataset,{}).get(domain) if terminal else None
                    metrics=readout.get('micro',{}) if readout else {}
                    event_scores=[scores(cell)['water_IoU'] for cell in readout.get('event',{}).values()] if readout else []
                    event_scores=[value for value in event_scores if value is not None]
                    result_rows.append({**common,'checkpoint_kind':kind,'dataset':dataset,'domain':domain,
                        'readout_status':('reused' if status=='reused' else 'completed') if readout else 'not_run',
                        'n_images':readout.get('n_images',terminal['terminal_results'][dataset].get('n_images')) if readout else None,
                        'n_events':readout.get('n_events') if readout else None,
                        'event_macro_water_IoU':statistics.mean(event_scores)*100 if event_scores else None,
                        **{key:metrics.get(key)*100 if metrics.get(key) is not None else None for key in METRICS}})
                    if not readout:
                        mechanism_rows.append({**common,'checkpoint_kind':kind,'dataset':dataset,'domain':domain,
                            'event_id':None,'level':'pooled','readout_status':'not_run','reason':'No same-pixel teacher/rule/student mechanism receipt'})
                        continue
                    family_ext[(family,kind,dataset,domain)][seed]=metrics
                    event_values={}
                    for event,cell in readout['event'].items():
                        values=scores(cell);event_values[event]=values['water_IoU']
                        event_rows.append({**common,'checkpoint_kind':kind,'dataset':dataset,'domain':domain,'event_id':event,
                            'readout_status':'reused' if status=='reused' else 'completed','background_TP':cell[0][0],
                            'water_TP':cell[1][0],'water_FP':cell[1][1],'water_FN':cell[1][2],
                            'water_union_pixels':sum(cell[1]),'valid_pixels':cell[0][0]+sum(cell[1]),
                            **{key:value*100 if value is not None else None for key,value in values.items()},
                            'uncertainty_unit':'whole event; trained checkpoints conditional'})
                    family_events[(family,kind,dataset,domain)][seed]=event_values
                    per_image=terminal.get('mechanism_counts',{}).get(dataset,{})
                    if not per_image:
                        mechanism_rows.append({**common,'checkpoint_kind':kind,'dataset':dataset,'domain':domain,
                            'event_id':None,'level':'pooled','readout_status':'not_run','reason':'No same-pixel teacher/rule/student mechanism receipt'})
                        continue
                    groups=defaultdict(list)
                    for item in per_image.values():
                        if domain=='known_overlap_excluded' and item['omit']:continue
                        counts={key:value for key,value in item.items() if key not in ('event','map','omit')}
                        groups[item['event']].append(counts);groups['__pooled__'].append(counts)
                    for event,count_rows in groups.items():
                        counts=sum_mechanisms(count_rows)
                        mechanism_row={**common,'checkpoint_kind':kind,'dataset':dataset,'domain':domain,'event_id':event,
                            'level':'pooled' if event=='__pooled__' else 'event','readout_status':'completed',
                            'count_unit':'integer pixels on same reference grid','rate_unit':'fraction',
                            'conditioning_frequency_denominator':'rule_available_valid_pixels',**counts}
                        for rate in ('correction','harm','rescue'):
                            denominator=counts[rate+'_denominator'];mechanism_row[rate+'_rate']=counts[rate+'_numerator']/denominator if denominator else None
                            mechanism_row[rate+'_conditioning_frequency']=denominator/counts['rule_available_pixels'] if counts['rule_available_pixels'] else None
                        mechanism_row['signed_teacher_rule_benefit']=counts['signed_benefit_numerator']/counts['disagreements'] if counts['disagreements'] else None
                        mechanism_rows.append(mechanism_row)
    # Fixed comparators produced during the shared reference pass do not create
    # training-seed records or spend a second teacher inference command.
    for path in terminal_files:
        payload=read(path)
        if not payload.get('reference_only') or payload.get('smoke_only'):continue
        family='teacher' if payload['reference']=='teacher' else 'MNDWI>0'
        for dataset,dataset_result in payload.get('terminal_results',{}).items():
            for domain in ('full','known_overlap_excluded'):
                readout=dataset_result[domain];metrics=readout['micro']
                values=[scores(cell)['water_IoU'] for cell in readout['event'].values()];values=[value for value in values if value is not None]
                common=dict(run_id=payload['id'],family=family,graph=family,method='fixed_reference',seed=None,
                    training_status='not_run',training_receipt=None,checkpoint_kind='selected',dataset=dataset,domain=domain,
                    readout_status='completed',metric_unit='percent',n_images=readout.get('n_images',dataset_result['n_images']),
                    n_events=readout['n_events'],selected_checkpoint_sha256=payload['checkpoint_sha256'])
                result_rows.append({**common,**{key:metrics.get(key)*100 if metrics.get(key) is not None else None for key in METRICS},
                    'event_macro_water_IoU':statistics.mean(values)*100 if values else None})
                family_ext[(family,'selected',dataset,domain)][None]=metrics
                event_values={}
                for event,cell in readout['event'].items():
                    metrics=scores(cell);event_values[event]=metrics['water_IoU']
                    event_rows.append({**common,'event_id':event,'water_TP':cell[1][0],'water_FP':cell[1][1],
                        'water_FN':cell[1][2],'water_union_pixels':sum(cell[1]),'valid_pixels':cell[0][0]+sum(cell[1]),
                        **{key:value*100 if value is not None else None for key,value in metrics.items()}})
                family_events[(family,'selected',dataset,domain)][None]=event_values
    contrasts=[('R4_LKD','A0_LKD_reanchored'),('R4_LKD','A3_LKD'),('R4_LKD','R4_CE'),
        ('A1_LKD_reanchored','A0_LKD_reanchored'),
        ('A3_LKD','A3_CE'),('C3','C0'),('C1','C0'),('C2','C0'),('C3','C1'),('C3','C2'),('C3','C4'),('C3','C5'),('HL','HC'),('C0','HL')]
    contrasts += [(family,reference) for family in ('R4_LKD','A3_LKD','A3N_LKD','C0','C3','C4','C5') for reference in ('teacher','MNDWI>0')]
    comparison_rows=[];event_differences=[]
    for left,right in contrasts:
        for kind in ('selected','final'):
            contexts=[('source_validation','full',family_source.get((left,kind),{}),family_source.get((right,kind),{}))]
            contexts += [(dataset,domain,family_ext.get((left,kind,dataset,domain),{}),family_ext.get((right,kind,dataset,domain),{}))
                for dataset in ('s11','sturm','wf') for domain in ('full','known_overlap_excluded')]
            for dataset,domain,lhs,rhs in contexts:
                for metric in METRICS:
                    paired=[]
                    for seed in SEEDS:
                        lv=lhs.get(seed,{}).get(metric);rv=rhs.get(seed,rhs.get(None,{})).get(metric)
                        if lv is not None and rv is not None:paired.append(100*(lv-rv))
                    comparison_rows.append(dict(left=left,right=right,checkpoint_kind=kind,dataset=dataset,domain=domain,
                        metric=metric,unit='percentage_points',status='completed' if len(paired)==4 else 'inconclusive_pilot' if paired else 'not_run',
                        **paired_interval(paired)))
                if dataset=='source_validation':continue
                lm=family_events.get((left,kind,dataset,domain),{});rm=family_events.get((right,kind,dataset,domain),{})
                paired_seeds=sorted(seed for seed in lm if seed in SEEDS and (seed in rm or None in rm));deltas=[]
                if paired_seeds:
                    events=sorted(set.intersection(*(set(lm[seed])&set(rm.get(seed,rm.get(None,{}))) for seed in paired_seeds)))
                    for event in events:
                        values=[100*(lm[seed][event]-rm.get(seed,rm.get(None,{}))[event]) for seed in paired_seeds
                            if lm[seed][event] is not None and rm.get(seed,rm.get(None,{}))[event] is not None]
                        if values:
                            delta=statistics.mean(values);deltas.append(delta)
                            event_differences.append(dict(left=left,right=right,checkpoint_kind=kind,dataset=dataset,domain=domain,event_id=event,
                                mean_difference_pp=delta,n_checkpoint_pairs=len(values),uncertainty='descriptive event difference'))
                comparison_rows.append(dict(left=left,right=right,checkpoint_kind=kind,dataset=dataset,domain=domain,
                    metric='event_macro_water_IoU',unit='percentage_points',n_checkpoint_pairs=len(paired_seeds),
                    status='completed' if deltas else 'not_run',**bootstrap(deltas),
                    uncertainty_type='10000 whole-event bootstrap,seed1729, conditional on trained checkpoints; excludes training uncertainty'))
    deployment_rows=[]
    for path in deployment_files:
        payload=read(path)
        if payload.get('smoke_only'):continue
        deployment_rows+=payload.get('rows',[])
    for graph in ('R4','A3N','A3'):
        for precision in ('FP32','FP16','INT8'):
            if not any(row.get('graph')==graph and row.get('precision')==precision for row in deployment_rows):
                deployment_rows.append(dict(graph=graph,seed=42,precision=precision,status='not_run',reason='No completed measured engine receipt'))
    deployment_rows.append(dict(graph='A3',precision='W4A8',engine='integer_domain_simulator',status='simulation_only',
        reason='Historical adopted simulation only; no packed integer deployment claim',latency_p50_ms=None,artifact_bytes=None))
    csv_write(root/'results.csv',result_rows);csv_write(root/'comparisons.csv',comparison_rows)
    csv_write(root/'event_results.csv',event_rows);csv_write(root/'mechanism_counts.csv',mechanism_rows,
        None if mechanism_rows else ['family','seed','dataset','domain','event_id','readout_status','correction_numerator','correction_denominator','harm_numerator','harm_denominator','rescue_numerator','rescue_denominator'])
    csv_write(root/'deployment.csv',deployment_rows);csv_write(root/'event_differences.csv',event_differences)
    plots=make_plots(root,result_rows,deployment_rows,event_differences,mechanism_rows)
    statuses=Counter(record['status'] for record in records);bundles=state.get('bundles',{})
    complete=state.get('worker_status') in ('completed','finished','stopped','budget_closed','deadline','failed','technical_failure') and not state.get('active_job')
    gate_status='not_run' if promotion is None else 'passed' if promotion.get('passed') else 'failed_scientific_gate'
    lines=['# FloodDistill v7', '', 'This report reflects saved artifacts and the actual worker state. Readouts marked not_run have not been established.', '',
        f'Worker: {state.get("worker_status","not_run")}; collection reason: {args.reason}; prospective C3 gate: {gate_status}.',
        f'Training records: {dict(statuses)}. Historical aliases do not add independent seeds.', '',
        'The declared questions are whether R4 survives a source-selected, better-trained A0 anchor; whether matched soft supervision helps R4; whether additive MNDWI and disagreement weighting beat both controls; and what supported artifacts cost at runtime.', '',
        'CE retains focal loss on cached teacher hard pseudo-labels. LKD queries a frozen teacher on augmented images. Continuations use identical selected R4 parents, fresh AdamW, and 1,560 updates; they do not establish improved scratch training.', '',
        'IoU is the overlap of predicted and reference regions divided by their union. It is not pixel accuracy. Main readouts preserve the registered 512 grid, valid masks, established event IDs, and both full and overlap-excluded domains. Exclusion does not reverse prior research exposure.', '',
        'Undefined empty-union water IoUs remain missing and are omitted from event means. Pooled pixel counts and equally weighted event scores are separate. Four paired seeds use sample SD and df=3 Student-t 95% intervals; two-seed pilots are descriptive. Event intervals use 10,000 whole-event resamples with seed 1729, conditional on trained checkpoints.', '']
    for lhs,rhs in [('R4_LKD','A0_LKD_reanchored'),('R4_LKD','R4_CE'),('C3','C0'),('C3','C4'),('C3','C5')]:
        row=next(row for row in comparison_rows if row['left']==lhs and row['right']==rhs and row['checkpoint_kind']=='selected' and row['dataset']=='source_validation' and row['metric']=='mIoU')
        if row['n_pairs']:
            interval=f"; 95% CI [{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}]" if row['ci95_low_pp'] is not None else '; descriptive only'
            lines.append(f"{lhs} minus {rhs}: {row['mean_difference_pp']:+.2f} source mIoU points, n={row['n_pairs']}{interval}.")
        else:lines.append(f'{lhs} minus {rhs}: not_run; the required paired source receipts are missing.')
    lines+=['',f'C3 prospective gate: {gate_status}.']
    if promotion:
        failed=[key for key,value in promotion.get('criteria',{}).items() if not value]
        lines.append('Failed criteria: '+(', '.join(failed) if failed else 'none')+'.')
        lines.append('C3 exceeded both prespecified control margins only if both corresponding frozen source criteria passed; no external scores determined promotion.')
    else:lines.append('A gain exceeding C4 and C5 has not yet been established. The gate and control claims remain unknown.')
    lines+=['','All completed arms are retained, including failures of the novelty hypothesis. Final checkpoints and source scores are retained; final external scores require a separate reservation and remain not_run otherwise.',
        f'Mechanism summaries: {sum(row["readout_status"]=="completed" for row in mechanism_rows)} completed event/pooled records. Correction, teacher-following harm, rescue, conditioning frequencies, and signed teacher-versus-rule benefit use aggregated integer counts; empty groups stay undefined. Unlabelled TRAIN cannot establish correctness.',
        'Deployment rows distinguish completed native artifacts, unsupported attempts, not_run work, and historical simulation_only. Artifact bytes, synchronized latency, and actual memory are reported only when measured. Parameter count is not substituted for storage, latency, or energy.', '',
        f'Budget state: {json.dumps(budget,sort_keys=True)}', '', 'Bundle dispositions:']
    for name,value in bundles.items():lines.append(f"- {name}: {value.get('status','not_run')} ({value.get('reason') or 'no additional reason'}).")
    lines+=['','Completed/reused selected readouts (mean across available seeds; external sets remain research exposed):', '',
        '| Family | Source mIoU | STURM pooled water IoU | WF pooled water IoU | WF equal-event water IoU | Seeds |',
        '|---|---:|---:|---:|---:|---:|']
    def mean_field(family,dataset,field):
        candidates=[row[field] for row in result_rows if row['family']==family and row['dataset']==dataset
            and row['domain']=='full' and row['checkpoint_kind']=='selected' and row.get(field) is not None]
        return f'{statistics.mean(candidates):.2f}' if candidates else 'not_run'
    for family in sorted({record['family'] for record in records}):
        source=family_source.get((family,'selected'),{})
        if not source:continue
        lines.append(f"| {family} | {mean_field(family,'source_validation','mIoU')} | {mean_field(family,'sturm','water_IoU')} | {mean_field(family,'wf','water_IoU')} | {mean_field(family,'wf','event_macro_water_IoU')} | {len(source)} |")
    lines+=['','Historical cohort: regenerated approved N2500 with 127 replacement tiles; dataset SHA256 80f9e13c136abe8169b8a5a064e265717b8d241ee879d945fd279ffbbed52711. This is not an exact reproduction of the historical GDACS subset.',
        'Six historical tiny-rung runs stopped under inherited patience; they retain their original identity and fixed-budget exception. No claim is made that the joint external capacity floor or a universal small-model KD law has been found.',
        'Known ingredients include spectral-index integration, selective/disagreement distillation, and learned quantizer step sizes. No first-ever novelty claim is made; failure of a declared gate is not redefined as success.', '',
        f'Reproduce CPU analysis: `{manifest.get("python","/home/schmalstieg/miniconda3/envs/esi/bin/python")} {root}/collect_v7.py --run-dir {root} --reason status`.',
        'Data tables: results.csv, comparisons.csv, event_results.csv, mechanism_counts.csv, deployment.csv. Missing metric cells are explicit empty CSV values with status.']
    if errors:lines+=['','Collection integrity issues:']+['- '+error for error in errors]
    write_text(root/'report_v7.md','\n'.join(lines)+'\n')
    control_prefix=f'{manifest.get("python","/home/schmalstieg/miniconda3/envs/esi/bin/python")} {root}/run_overnight.py'
    handover=[f'Run: {root}',f'Worker status: {state.get("worker_status","not_run")}',f'Deadline: {state.get("deadline_at")}',
        f'Heartbeat: {state.get("heartbeat_at")}',f'Active job: {state.get("active_job")}',f'Report: {root}/report_v7.md',
        f'Status: {control_prefix} status --run-dir {root}',f'Pause: {control_prefix} pause --run-dir {root}',
        f'Resume: {control_prefix} resume --run-dir {root}',f'Stop: {control_prefix} stop --run-dir {root}',
        'Host must remain powered and awake. Closing the editor is separate from host sleep.',
        f'C3 gate: {gate_status}; collection integrity issues: {len(errors)}.']
    write_text(root/'handover.md','\n\n'.join(handover)+'\n')
    outputs=['report_v7.md','results.csv','comparisons.csv','event_results.csv','mechanism_counts.csv','deployment.csv','event_differences.csv','handover.md']+plots
    atomic_json(root/'completion_receipt.json',dict(status='complete' if complete else 'snapshot',
        outcome='completed' if complete and not errors else 'technical_failure' if errors else 'in_progress',
        completed_at=now(),collection_reason=args.reason,input_fingerprint=fingerprint,worker_status=state.get('worker_status'),
        programme_finished=complete,all_scientific_runs_completed=complete and all(record['status'] in ('completed','reused') for record in records),
        counts=dict(statuses),gate_status=gate_status,integrity_errors=errors,units='metrics percent; differences percentage points; counts integer pixels',
        outputs={name:dict(path=str(root/name),sha256=file_hash(root/name)) for name in outputs}))
    print('Collected',len(records),'training records;',len(result_rows),'readouts;',len(errors),'integrity issues')


def make_plots(root,results,deployment,event_differences,mechanisms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for row in deployment:
        if row.get('status')!='completed' or row.get('mIoU') is None:continue
        for axis,field,label in [(axes[0],'artifact_bytes','Actual artifact bytes'),(axes[1],'latency_p50_ms','Synchronized p50 latency(ms)')]:
            if row.get(field) is not None:
                axis.scatter(row[field],row['mIoU']);axis.annotate(row['graph']+' '+row['precision'],(row[field],row['mIoU']),fontsize=7)
        axes[0].set_xlabel('Actual artifact bytes');axes[1].set_xlabel('Synchronized p50 latency(ms)')
    for axis in axes:
        axis.set_ylabel('Source validation mIoU(%)')
        if not axis.collections:axis.text(.5,.5,'No completed measured artifacts',ha='center',transform=axis.transAxes)
    fig.tight_layout();fig.savefig(root/'accuracy_runtime.png',dpi=140);plt.close(fig);plots.append('accuracy_runtime.png')
    fig,axis=plt.subplots(figsize=(8,4))
    values=[row for row in event_differences if row['left']=='C3' and row['right']=='C0' and row['checkpoint_kind']=='selected' and row['domain']=='full']
    if values:
        values=sorted(values,key=lambda row:(row['dataset'],row['event_id']))
        axis.scatter(range(len(values)),[row['mean_difference_pp'] for row in values]);axis.axhline(0,color='grey',linewidth=.7)
        axis.set_xticks(range(len(values)),[row['dataset']+':'+row['event_id'] for row in values],rotation=90,fontsize=6)
    else:axis.text(.5,.5,'C3−C0 event readouts pending',ha='center',transform=axis.transAxes)
    axis.set_ylabel('C3−C0 water IoU(points)');fig.tight_layout();fig.savefig(root/'event_differences.png',dpi=140);plt.close(fig);plots.append('event_differences.png')
    fig,axis=plt.subplots(figsize=(7,4))
    values=[row for row in mechanisms if row['readout_status']=='completed' and row['level']=='pooled' and row['domain']=='full' and row['checkpoint_kind']=='selected']
    if values:
        for row in values:
            if row['correction_rate'] is not None and row['harm_rate'] is not None:
                axis.scatter(row['correction_rate'],row['harm_rate'],label=row['family']+' '+row['dataset'],s=20)
        handles,labels=axis.get_legend_handles_labels()
        if handles:axis.legend(handles[:12],labels[:12],fontsize=6)
    else:axis.text(.5,.5,'Mechanism counts pending',ha='center',transform=axis.transAxes)
    axis.set_xlabel('Correction rate(fraction)');axis.set_ylabel('Teacher-following harm rate(fraction)')
    fig.tight_layout();fig.savefig(root/'correction_harm.png',dpi=140);plt.close(fig);plots.append('correction_harm.png')
    return plots


if __name__=='__main__':main()
