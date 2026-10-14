"""Independent CPU audit of selected v7 external counts, mechanisms and deployment.

No frozen modules are imported. Readouts never select models or trigger training.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
import numpy as np

RUN=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
SEEDS=(42,64,91,107)
DATASETS=('s11','sturm','wf')
DOMAINS=('full','known_overlap_excluded')
METRICS=('mIoU','water_IoU','water_precision','water_recall')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,rows):
    fields=sorted(set().union(*(set(row) for row in rows))) if rows else ['status']
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(value) if isinstance(value,(dict,list)) else value for key,value in row.items()})

def scores(cell):
    cell=np.asarray(cell,dtype=np.int64);unions=cell.sum(1)
    iou=[float(cell[index,0]/unions[index]) if unions[index] else None for index in (0,1)]
    values=[value for value in iou if value is not None]
    tp,fp,fn=(int(value) for value in cell[1]);ratio=lambda n,d:n/d if d else None
    return dict(mIoU=sum(values)/len(values) if values else None,water_IoU=iou[1],
                water_precision=ratio(tp,tp+fp),water_recall=ratio(tp,tp+fn))

def interval(values):
    values=[value for value in values if value is not None]
    mean=st.mean(values) if values else None;sd=st.stdev(values) if len(values)>1 else None
    half=3.182446305284263*sd/2 if len(values)==4 else None
    return dict(n_pairs=len(values),mean_pp=mean,sample_sd_pp=sd,
        lo95_pp=mean-half if half is not None else None,hi95_pp=mean+half if half is not None else None,
        uncertainty='four training seeds, paired df3 Student-t' if len(values)==4 else 'descriptive; fewer than four training seeds')

def bootstrap(values):
    values=np.asarray([value for value in values if value is not None],dtype=float)
    if not len(values):return dict(n_events=0,mean_pp=None,lo95_pp=None,hi95_pp=None)
    rng=np.random.default_rng(1729)
    draws=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(n_events=len(values),mean_pp=float(values.mean()),lo95_pp=float(np.quantile(draws,.025)),
        hi95_pp=float(np.quantile(draws,.975)),uncertainty='10000 whole-event bootstrap seed1729; conditional on fixed trained checkpoints')

def cell_sum(values):return sum((np.asarray(value,dtype=np.int64) for value in values),np.zeros((2,3),dtype=np.int64))


results=list(csv.DictReader((RUN/'results.csv').open()))
source=[row for row in results if row['checkpoint_kind']=='selected' and row['dataset']=='source_validation'
    and row['readout_status'] in ('completed','reused')]
family_ids=defaultdict(dict)
for row in source:family_ids[row['family']][int(row['seed'])]=row['run_id']
reuse=read(RUN/'reuse_index.json')['records']+read(RUN/'optional_A1_reuse_index.json')['records']
reuse_by_id={Path(record['receipt_path']).stem:record for record in reuse if record['verified']}
terminal={path.stem:read(path) for path in (RUN/'terminal').glob('*.json') if not read(path).get('smoke_only')}
reference_manifest=read(RUN/'reference_cache/manifest.json')
families=defaultdict(dict)
for family,ids in family_ids.items():
    for seed,rid in ids.items():
        payload=terminal.get(rid+'__selected')
        if payload is None and rid in reuse_by_id:payload=read(reuse_by_id[rid]['terminal_receipt_path'])
        if payload:families[family][seed]=payload
for key,label in [('teacher','teacher'),('rule','v7_rule_R')]:
    families[label][None]=terminal['v7_reference_'+key]

integrity={'errors':[],'terminal_checked':0,'per_image_cells_checked':0,'reused_parity':[],
           'metrics_max_absolute_delta_fraction':0.,'mechanism_cells_checked':0}
unique_payload={payload['id']:payload for records in families.values() for payload in records.values()}
for rid,payload in unique_payload.items():
    assert payload['status']=='complete',rid
    integrity['terminal_checked']+=1
    for dataset,readout in payload['terminal_results'].items():
        expected={'s11':90,'sturm':2675,'wf':785}[dataset]
        images=readout['per_image_counts'];assert len(images)==expected,(rid,dataset,len(images))
        metadata={row['filename']:row for row in reference_manifest['datasets'][dataset]['rows']}
        assert set(images)==set(metadata),(rid,dataset,'image identity mismatch')
        full_maps=defaultdict(list);full_events=defaultdict(list);held_maps=defaultdict(list);held_events=defaultdict(list)
        for filename,cell in images.items():
            array=np.asarray(cell);assert array.shape==(2,3) and array.dtype.kind in 'iu' and (array>=0).all()
            assert array[0,1]==array[1,2] and array[0,2]==array[1,1]
            integrity['per_image_cells_checked']+=1
            row=metadata[filename];full_maps[row['map']].append(cell);full_events[row['event']].append(cell)
            if not row['omit']:held_maps[row['map']].append(cell);held_events[row['event']].append(cell)
        for domain,maps,events in [('full',full_maps,full_events),('known_overlap_excluded',held_maps,held_events)]:
            aggregate=readout[domain]
            assert set(maps)==set(aggregate['map']) and set(events)==set(aggregate['event'])
            for event,cells in events.items():assert np.array_equal(cell_sum(cells),aggregate['event'][event]),(rid,dataset,domain,event)
            for map_id,cells in maps.items():assert np.array_equal(cell_sum(cells),aggregate['map'][map_id]),(rid,dataset,domain,map_id)
            actual=scores(cell_sum(images.values()) if domain=='full' else cell_sum(cell for key,cell in images.items() if not metadata[key]['omit']))
            for metric,value in actual.items():
                saved=aggregate['micro'][metric]
                assert (value is None)==(saved is None)
                if value is not None:
                    delta=abs(value-saved);assert delta<1e-12
                    integrity['metrics_max_absolute_delta_fraction']=max(integrity['metrics_max_absolute_delta_fraction'],delta)
        per_image_mechanisms=payload.get('mechanism_counts',{}).get(dataset,{})
        if per_image_mechanisms:
            assert set(per_image_mechanisms)==set(images)
            for filename,item in per_image_mechanisms.items():
                valid=int(np.asarray(images[filename])[0,0])+sum(images[filename][1])
                assert item['valid_pixels']==valid
                assert item['rule_available_pixels']+item['rule_unavailable_pixels']==valid
                assert item['disagreements']==item['correction_denominator']+item['harm_denominator']
                assert item['signed_benefit_numerator']==item['correction_denominator']-item['harm_denominator']
                assert item['both_correct_pixels']+item['correction_denominator']+item['harm_denominator']+item['rescue_denominator']==item['rule_available_pixels']
                for group in ('correction','harm','rescue'):assert 0<=item[group+'_numerator']<=item[group+'_denominator']
                integrity['mechanism_cells_checked']+=1
for rid,record in reuse_by_id.items():
    current=terminal.get(rid+'__selected')
    if current is None:continue
    old=read(record['terminal_receipt_path']);assert current['checkpoint_sha256']==old['checkpoint_sha256']==record['best_sha256']
    comparisons={dataset:dict(images=len(old['terminal_results'][dataset]['per_image_counts']),
        mismatches=sum(old['terminal_results'][dataset]['per_image_counts'].get(key)!=value for key,value in current['terminal_results'][dataset]['per_image_counts'].items())) for dataset in DATASETS}
    assert all(row['mismatches']==0 for row in comparisons.values())
    integrity['reused_parity'].append(dict(run_id=rid,graph=record['graph'],seed=record['seed'],datasets=comparisons))

summary=[]
for family,records in families.items():
    for dataset in DATASETS:
        for domain in DOMAINS:
            stats=dict(family=family,dataset=dataset,domain=domain,n_training_seeds=len(records) if None not in records else 0,
                checkpoint_count=len(records),n_events=len(next(iter(records.values()))['terminal_results'][dataset][domain]['event']))
            for metric in METRICS:
                values=[100*payload['terminal_results'][dataset][domain]['micro'][metric] for payload in records.values()]
                stats[metric+'_mean_pct']=st.mean(values);stats[metric+'_sample_sd_pct']=st.stdev(values) if len(values)>1 else None
            event_means=[]
            for payload in records.values():
                values=[scores(cell)['water_IoU'] for cell in payload['terminal_results'][dataset][domain]['event'].values()]
                values=[100*value for value in values if value is not None];event_means.append(st.mean(values))
            stats.update(event_macro_water_IoU_mean_pct=st.mean(event_means),
                event_macro_water_IoU_sample_sd_pct=st.stdev(event_means) if len(event_means)>1 else None)
            summary.append(stats)
write('external_selected_summary.csv',summary)

contrasts=[('R4_LKD','A0_LKD_reanchored'),('R4_LKD','A1_LKD_reanchored'),('R4_LKD','R4_CE'),('A3_LKD','A3_CE'),
    ('A0_LKD_reanchored','A0_LKD_recipe'),('A1_LKD_reanchored','A1_LKD_recipe'),
    ('C3','C0'),('C3','C4'),('C3','C5'),('HL','HC'),('C0','HL')]
contrasts += [(family,reference) for family in ('R4_LKD','R4_CE','A0_LKD_reanchored','A1_LKD_reanchored','A3_LKD','A3N_LKD','C0','C3','HC','HL') for reference in ('teacher','v7_rule_R')]
contrast_rows=[];event_differences=[]
for left,right in contrasts:
    lhs,rhs=families[left],families[right]
    seeds=sorted(seed for seed in lhs if seed in SEEDS and (seed in rhs or None in rhs))
    for dataset in DATASETS:
        for domain in DOMAINS:
            for metric in METRICS:
                values=[100*(lhs[seed]['terminal_results'][dataset][domain]['micro'][metric]-rhs.get(seed,rhs.get(None))['terminal_results'][dataset][domain]['micro'][metric]) for seed in seeds]
                contrast_rows.append(dict(left=left,right=right,dataset=dataset,domain=domain,metric=metric,**interval(values)))
            maps=[(lhs[seed]['terminal_results'][dataset][domain]['event'],rhs.get(seed,rhs.get(None))['terminal_results'][dataset][domain]['event']) for seed in seeds]
            events=sorted(set.intersection(*(set(lmap)&set(rmap) for lmap,rmap in maps)))
            values=[];seed_event_means=defaultdict(list)
            for event in events:
                differences=[]
                for seed,(lmap,rmap) in zip(seeds,maps):
                    lvalue,rvalue=scores(lmap[event])['water_IoU'],scores(rmap[event])['water_IoU']
                    if lvalue is not None and rvalue is not None:
                        difference=100*(lvalue-rvalue);differences.append(difference);seed_event_means[seed].append(difference)
                if differences:
                    value=st.mean(differences);values.append(value)
                    event_differences.append(dict(left=left,right=right,dataset=dataset,domain=domain,event_id=event,
                        mean_difference_pp=value,checkpoint_pairs=len(differences)))
            contrast_rows.append(dict(left=left,right=right,dataset=dataset,domain=domain,metric='event_macro_water_IoU',
                **bootstrap(values),checkpoint_pairs=len(seeds)))
            contrast_rows.append(dict(left=left,right=right,dataset=dataset,domain=domain,metric='event_macro_water_IoU_training_seed_uncertainty',
                **interval([st.mean(values) for values in seed_event_means.values()])))
write('external_selected_contrasts.csv',contrast_rows);write('selected_per_event_differences.csv',event_differences)

mechanism_rows=[];mechanism_events=[]
for family,records in families.items():
    eligible={seed:payload for seed,payload in records.items() if payload.get('mechanism_counts')}
    if not eligible:continue
    for dataset in DATASETS:
        for domain in DOMAINS:
            grouped=defaultdict(dict)
            for seed,payload in eligible.items():
                acc=defaultdict(lambda:defaultdict(int))
                for item in payload['mechanism_counts'][dataset].values():
                    if domain!='full' and item['omit']:continue
                    counts={key:value for key,value in item.items() if key not in ('event','map','omit')}
                    for event in (item['event'],'__pooled__'):
                        for key,value in counts.items():acc[event][key]+=int(value)
                for event,counts in acc.items():grouped[event][seed]=counts
            for event,by_seed in grouped.items():
                counts=next(iter(by_seed.values()))
                fixed_keys=['valid_pixels','rule_available_pixels','rule_unavailable_pixels','disagreements','signed_benefit_numerator','correction_denominator','harm_denominator','rescue_denominator','both_correct_pixels']
                assert all(all(row[key]==counts[key] for key in fixed_keys) for row in by_seed.values())
                row=dict(family=family,dataset=dataset,domain=domain,event_id=event,checkpoint_count=len(by_seed),
                    **{key+'_per_checkpoint':counts[key] for key in fixed_keys},
                    disagreement_frequency=counts['disagreements']/counts['rule_available_pixels'] if counts['rule_available_pixels'] else None,
                    signed_teacher_rule_benefit=counts['signed_benefit_numerator']/counts['disagreements'] if counts['disagreements'] else None)
                for group in ('correction','harm','rescue'):
                    rates=[item[group+'_numerator']/item[group+'_denominator'] for item in by_seed.values() if item[group+'_denominator']]
                    row[group+'_rate_mean']=st.mean(rates) if rates else None
                    row[group+'_rate_sample_sd']=st.stdev(rates) if len(rates)>1 else None
                    row[group+'_numerator_sum_over_checkpoints']=sum(item[group+'_numerator'] for item in by_seed.values())
                    row[group+'_denominator_sum_over_checkpoints']=sum(item[group+'_denominator'] for item in by_seed.values())
                    row[group+'_conditioning_frequency']=counts[group+'_denominator']/counts['rule_available_pixels'] if counts['rule_available_pixels'] else None
                (mechanism_rows if event=='__pooled__' else mechanism_events).append(row)
write('mechanism_selected_summary.csv',mechanism_rows);write('mechanism_selected_events.csv',mechanism_events)

deployment=read(RUN/'receipts/deployment.json');deployment_rows=[]
for row in deployment['rows']:
    result=dict(row)
    result['accuracy_geometry']='89 S11 source-validation images at registered512; TRAIN224-to512 interpolation applies only to timing input'
    result['measurement_limits']=['Single selected seed42 on RTX4090; no Jetson or energy measurements',
        'Total PyTorch allocator peaks include live image/logits tensors from earlier source-validation loops; not isolated model peaks',
        'FP32 denotes float32 parameters/input; actual recorded matmul TF32 and cuDNN policies determine arithmetic']
    if row['status']=='completed':
        path=Path(row['artifact_path']);result['artifact_size_matches']=path.stat().st_size==row['artifact_bytes'];result['artifact_sha_matches']=sha(path)==row['artifact_sha256']
        assert result['artifact_size_matches'] and result['artifact_sha_matches']
        actual=scores(row['source_confusion_counts'])
        for metric,value in actual.items():assert abs(100*value-row[metric])<1e-10
        result['allocator_baseline_mib']=row['inference_baseline_allocated_bytes']/2**20
        result['peak_allocator_mib']=row['peak_allocated_bytes']/2**20
        result['incremental_allocator_peak_mib']=row['peak_incremental_allocated_bytes']/2**20
        result['actual_matmul_tf32']=row['backend_policy']['cuda_matmul_allow_tf32']
        result['actual_cudnn_tf32']=row['backend_policy']['cudnn_allow_tf32']
        result['precision_policy_label_is_complete']=not (row['backend_policy']['policy']=='deployment_tf32_off' and row['backend_policy']['cuda_matmul_allow_tf32'])
    deployment_rows.append(result)
write('deployment_independent_audit.csv',deployment_rows)

dataset_balance=[]
for dataset in DATASETS:
    for domain in DOMAINS:
        readout=families['R4_LKD'][42]['terminal_results'][dataset][domain]
        water={event:int(cell[1][0])+int(cell[1][2]) for event,cell in readout['event'].items()}
        total_water=sum(water.values());leader=max(water,key=water.get)
        mechanism=next(row for row in mechanism_rows if row['family']=='R4_LKD' and row['dataset']==dataset and row['domain']==domain)
        events=[row for row in mechanism_events if row['family']=='R4_LKD' and row['dataset']==dataset and row['domain']==domain]
        benefits=[row['signed_teacher_rule_benefit'] for row in events if row['signed_teacher_rule_benefit'] is not None]
        dataset_balance.append(dict(dataset=dataset,domain=domain,n_events=len(water),n_images=readout.get('n_images',families['R4_LKD'][42]['terminal_results'][dataset]['n_images']),
            reference_water_pixels=total_water,largest_water_event=leader,largest_water_event_pixels=water[leader],largest_water_event_share=water[leader]/total_water,
            reference_water_by_event=water,valid_pixels=mechanism['valid_pixels_per_checkpoint'],
            rule_unavailable_valid_pixels=mechanism['rule_unavailable_pixels_per_checkpoint'],
            teacher_rule_disagreement_pixels=mechanism['disagreements_per_checkpoint'],
            signed_teacher_rule_benefit_pooled=mechanism['signed_teacher_rule_benefit'],
            signed_teacher_rule_benefit_equal_event=st.mean(benefits),teacher_benefit_positive_events=sum(value>0 for value in benefits),
            teacher_benefit_negative_events=sum(value<0 for value in benefits),
            teacher_rule_by_event=[{key:row[key] for key in ('event_id','disagreements_per_checkpoint','correction_denominator_per_checkpoint','harm_denominator_per_checkpoint','signed_teacher_rule_benefit')} for row in events]))
write('external_dataset_balance.csv',dataset_balance)
integrity['reused_parity_images']=sum(row['images'] for record in integrity['reused_parity'] for row in record['datasets'].values())
(OUT/'external_mechanism_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
(OUT/'external_mechanism_summary.json').write_text(json.dumps(dict(external_summary=summary,contrasts=contrast_rows,
    mechanisms=mechanism_rows,mechanism_event_summaries=mechanism_events,dataset_balance=dataset_balance,
    deployment=deployment_rows,integrity=integrity),indent=2)+'\n')
input_paths=[RUN/name for name in ('results.csv','reuse_index.json','optional_A1_reuse_index.json','reference_cache/manifest.json','receipts/deployment.json')]
input_paths += [path for path in (RUN/'terminal').glob('*.json') if not read(path).get('smoke_only')]
input_paths += [Path(record['terminal_receipt_path']) for record in reuse]
output_paths=sorted(path for path in OUT.glob('*') if path.is_file() and path.name in (
    'external_selected_summary.csv','external_selected_contrasts.csv','selected_per_event_differences.csv',
    'mechanism_selected_summary.csv','mechanism_selected_events.csv','deployment_independent_audit.csv',
    'external_dataset_balance.csv','external_mechanism_integrity.json','external_mechanism_summary.json'))
(OUT/'external_analysis_receipt.json').write_text(json.dumps(dict(status='completed',device='cpu',gpu_jobs_launched=0,
    python=sys.version,numpy=np.__version__,analysis_source_sha256=sha(Path(__file__)),
    input_sha256={str(path):sha(path) for path in sorted(set(input_paths))},
    output_sha256={str(path):sha(path) for path in output_paths},
    uncertainty_contract='sample SD; Student-t95%df3 only with4seeds;10000whole-event resamples seed1729; pilots descriptive',
    integer_count_checks=integrity),indent=2)+'\n')
print(json.dumps({key:value for key,value in integrity.items() if key!='reused_parity'},indent=2))
print('Wrote selected-only external, mechanism and deployment audits to',OUT)
