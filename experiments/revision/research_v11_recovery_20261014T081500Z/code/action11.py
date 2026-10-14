"""Small source-only decisions around existing V11/V10 entry points."""
import argparse
import json
import math
from pathlib import Path
import time
import common11 as c

def evaluation_profile(profile):
    """Time the real batched final readout on source windows only."""
    import numpy as np
    torch,c10,v=c.setup(8)
    from evaluate11 import load,forward_batch,prepare_metric_support,tile_readout,EVALUATION_BATCH_SIZE
    from evaluate10 import strata
    c.source_access_guard()
    model=load(profile['checkpoints'][-1],torch,c10)
    # Fixed workload, no scoring-based choice of arm or model count.
    entries=[dict(id=f'profile_{i}',fp_reference='profile_0') for i in range(36)]
    entries += [dict(id=f'alias_{i}',alias_of=f'profile_{i%36}',fp_reference='profile_0',threshold=.01) for i in range(40)]
    models={f'profile_{i}':model for i in range(36)}
    rows=[r for r in c.read(c.V9/'normal_windows.json')['windows'] if r['dataset']=='s11val' and not r['omit']][:3*EVALUATION_BATCH_SIZE]
    durations=[]
    with torch.inference_mode():
        for offset in range(0,len(rows),EVALUATION_BATCH_SIZE):
            group=rows[offset:offset+EVALUATION_BATCH_SIZE]
            start=time.monotonic();data=[]
            for row in group:
                with np.load(row['path'],allow_pickle=False) as f:
                    data.append((f['image'].copy(),f['mask'].copy(),f['cloud'].copy() if 'cloud' in f and f['cloud'].size else None))
            margins,_=forward_batch(models,np.stack([x for x,y,cloud in data]),torch)
            for index,(x,y,cloud) in enumerate(data):
                support=prepare_metric_support(y,strata(y,v.reflectance(x),cloud))
                tile_readout(entries,{name:values[index] for name,values in margins.items()},y,support)
            durations.append((time.monotonic()-start)/len(group))
    result=dict(outcome='completed',source_only=True,source_windows=[r['path'] for r in rows],seconds_per_window=durations,conservative_seconds_per_window=max(durations)*1.5,forward_models=36,metric_variants=76,inference_batch_size=EVALUATION_BATCH_SIZE,reason='Exact shared batched forward/metric kernel on three source batches; maximum per-tile time times1.5, collection margin separate')
    c.save(c.ROOT/'evaluation_cost_profile.json',result)
    return result

def admission():
    profile=c.read(c.ROOT/'repair/core/profile.json')
    manifest=c.read(c.ROOT/'manifest.json')
    state=c.read(c.ROOT/'state.json',{})
    core=profile['conservative_28_run_core_hours']
    # Profile includes data, scoring, selection, source evaluations and checkpoints.
    # Preserve a conservative slowdown margin; never pad U's actual timing.
    per_run=core*3600/8/28*1.20
    measured=evaluation_profile(profile)
    windows=sum(sum(not row['omit'] and row['dataset']!='s11val' for row in c.read(c.V9/name).get(key,[])) for name,key in [('normal_windows.json','windows'),('sealed_windows.json','rows')])
    evaluation_seconds=windows*measured['conservative_seconds_per_window']+300
    evaluation_core_hours=evaluation_seconds*8/3600
    collection_margin=1800
    first=state.get('first_compute_at',state.get('first_compute_heavy_at',time.time())) or time.time()
    deadline=min(state.get('absolute_deadline',time.time()+16*3600),first+8*3600)-collection_margin
    core_cap=min(j['max_seconds'] for j in manifest['jobs'].values() if j.get('training_rule'))-30
    reasons=[]
    if not math.isfinite(per_run) or per_run>core_cap:reasons.append('Measured complete 600-step per-arm bound exceeds frozen per-job ceiling')
    if not math.isfinite(evaluation_seconds) or evaluation_seconds>manifest['jobs']['30_final_evaluation']['max_seconds']-30:reasons.append('Full primary evaluation estimate plus300s initialization/bootstrap margin exceeds protected allocation and30s shutdown margin')
    if time.time()+(math.ceil(per_run)+30)*28+math.ceil(evaluation_seconds)+30+300>deadline:reasons.append('Complete matched block and evaluation do not fit remaining execution window')
    result=dict(outcome='completed',admitted=not reasons,reason='; '.join(reasons) if reasons else 'Complete 28-run block and 36-model primary readout fit frozen caps',
                per_run_seconds_bound=per_run if math.isfinite(per_run) else None,
                core_core_hours_bound=per_run*28*8/3600 if math.isfinite(per_run) else None,
                evaluation_seconds_bound=evaluation_seconds if math.isfinite(evaluation_seconds) else None,
                evaluation_core_hours_bound=evaluation_core_hours if math.isfinite(evaluation_core_hours) else None,
                evaluated_windows=windows,primary_models_before_deduplication=36,
                profile_sha256=c.sha(c.ROOT/'repair/core/profile.json'),source_only=True)
    c.save(c.ROOT/'profile_admission.json',result)

def optional_decision():
    selection=c.read(c.ROOT/'repair/core/selection.json',{})
    inputs=c.read(c.ROOT/'repair_inputs.json',{})
    rows=[]
    for arm in ['D','U']:
        metrics=[]
        for seed in c.SEEDS:
            path=c.ROOT/f'repair/core/{arm}_s{seed}'
            a=c.read(path/'source_step0200.json');b=c.read(path/'source_step0600.json')
            fp=c.read(c.ROOT/f'repair/core/FP_s{seed}/source_step0000.json')
            if a and b and fp:metrics.append((b['equal_event_water_IoU']-a['equal_event_water_IoU'],b['equal_event_water_IoU']>=fp['equal_event_water_IoU']-.01))
        rows.append(dict(arm=arm,seeds=len(metrics),improvement_200_600=sum(x[0] for x in metrics)/4 if len(metrics)==4 else None,reaches_tolerance=all(x[1] for x in metrics) if len(metrics)==4 else None))
    c.save(c.ROOT/'optional_decision.json',dict(outcome='completed',source_only=True,extension=dict(status='DEFERRED',reason='Optional extension was not admitted in the frozen finite add-on; protect the complete fixed 600-step primary block and final evaluation',source_criteria=rows),transfer=dict(status='DEFERRED',reason='No competing architecture choice; optional second complete matrix not reserved in this bounded core-first queue',V10_selection_receipt=str(c.V10/'smallest_viable_selection.json')),conditioner=dict(status='CHECK_DEPENDENCY_AND_GEOMETRY',reason='Independent optional action next; missing parents or deployable geometry defer it')))

def freeze():
    """Keep primary endpoints and source-selected states; deduplicate identical states."""
    selection=c.read(c.ROOT/'repair/core/selection.json')
    if not selection:
        c.save(c.ROOT/'final_evaluation_freeze.json',dict(outcome='completed',models=[],reason='No complete four-seed primary matrix; no external label access'))
        return
    inputs=c.read(c.ROOT/'repair_inputs.json')
    entries=[]
    for seed in c.SEEDS:
        parent=inputs['parents'][str(seed)]
        for key,arm in [('FP_parent','FP'),('entry','Q0')]:
            e=dict(parent[key],arm=arm,step=0,updates=0,point='core',endpoint='primary600',quantized=arm=='Q0')
            entries.append(e)
    for entry in selection['primary']:
        entries.append(dict(entry,step=entry['updates'],endpoint='primary600'))
    # Original scientific identities and all common-update source selections remain
    # recorded even where a protected budget excludes secondary external forwards.
    primary_hashes={e['sha256'] for e in entries}
    secondary=[]
    for e in selection['source_selected']:
        secondary.append(dict(e,external_status='ALIAS_PRIMARY' if e['sha256'] in primary_hashes else 'DEFERRED_PROTECTED_PRIMARY_EVALUATION'))
    # Threshold controls use source receipts only. Aliases share the corresponding
    # forward, and the timing profile above includes their metric cost.
    import numpy as np
    for e in list(entries):
        metric_path=e.get('source_metrics')
        if e['arm'] not in ['FP','Q0'] and metric_path:
            m=c.read(metric_path)
            entries.append(dict(e,id=e['id']+'_source_tau',alias_of=e['id'],arm=e['arm']+'_source_tau',endpoint='source_threshold_control',threshold=m['source_matched_tau'],threshold_source_sha256=c.sha(metric_path)))
    for seed in c.SEEDS:
        q0=next(e for e in entries if e['arm']=='Q0' and e['seed']==seed)
        qmetric_path=c.ROOT/f'repair/core/Q0_s{seed}/source_step0000.json'
        qm=c.read(qmetric_path);h=np.load(qm['source_histogram'],allow_pickle=False).sum(1)
        above=np.r_[h[::-1].cumsum()[::-1],0]
        for arm in ['D','U','A']:
            target=next(e for e in entries if e['arm']==arm and e['seed']==seed)
            tm=c.read(target['source_metrics']);desired=tm['source_positive_fraction']*int(h.sum())
            index=min(range(4097),key=lambda i:(abs(float(above[i])-desired),abs(i-2048),i))
            entries.append(dict(q0,id=q0['id']+'_tau_'+arm,alias_of=q0['id'],arm='Q0_tau_'+arm,endpoint='source_threshold_control',threshold=index/64-32,threshold_policy='S11 validation fixed histogram; approximate target source positive fraction; ties closest zero then lower; clipping and ties can prevent exact matching',binned_source_positive_fraction=float(above[index]/h.sum()),target_source_positive_fraction=tm['source_positive_fraction'],binned_fraction_mismatch=float(above[index]/h.sum()-tm['source_positive_fraction']),target_source_sha256=c.sha(target['source_metrics']),reference_source_sha256=c.sha(qmetric_path)))
    seen={}
    for e in entries:
        identity=(e['graph'],e.get('state_sha256',e['sha256']),e.get('quantized',False))
        if not e.get('alias_of'):
            if identity in seen:e['alias_of']=seen[identity]
            else:seen[identity]=e['id']
        assert c.sha(e['checkpoint'])==e['sha256']
    membership=[dict(path=str(c.V9/n),sha256=c.sha(c.V9/n)) for n in ['normal_windows.json','sealed_windows.json']]
    c.save(c.ROOT/'final_evaluation_freeze.json',dict(outcome='completed',at=time.time(),models=entries,secondary=secondary,membership=membership,selection_sha256=c.sha(c.ROOT/'repair/core/selection.json'),source_only_selection=True,exposure='non-confirmatory research readout; no sealed score inspected to choose method',precision='CPU fake quantization; low-bit kernels not claimed'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['admission','optional','freeze']);args=parser.parse_args()
    c.source_access_guard()
    {'admission':admission,'optional':optional_decision,'freeze':freeze}[args.action]()

if __name__=='__main__':main()
