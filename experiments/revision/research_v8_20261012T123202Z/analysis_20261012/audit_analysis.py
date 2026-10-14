"""Independent CPU audit. Preserves all worker code, receipts and original tables."""
from pathlib import Path
import collections
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import torch

BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import head_solver as hs


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False)
def score(cell):
    a=np.asarray(cell,dtype=np.int64);union=a.sum(1)
    iou=np.divide(a[:,0],union,out=np.full(2,np.nan),where=union>0)
    return dict(mIoU=float(np.nanmean(iou)),water_IoU=float(iou[1]),
        water_precision=float(a[1,0]/(a[1,0]+a[1,1])) if a[1,0]+a[1,1] else None,
        water_recall=float(a[1,0]/(a[1,0]+a[1,2])) if a[1,0]+a[1,2] else None)


def main():
    torch.set_num_threads(2)
    m=read(BASE/'manifest_initial.json');state=read(BASE/'state.json');c=read(BASE/'config.json');cache=read(BASE/'support_cache/manifest.json')
    errors=[];checks={};checks['frozen_inputs']=len(m['frozen_inputs'])
    for path,expected in m['frozen_inputs'].items():
        if sha(path)!=expected:errors.append('Frozen hash changed: '+path)
    checks['child_manifests']=0
    for b in state['bundles'].values():
        if sha(b['manifest'])!=b['manifest_sha256']:errors.append('Child manifest changed: '+b['manifest'])
        checks['child_manifests']+=1
    attempts=[read(p) for p in (BASE/'attempts').glob('*.json')]
    preflight=read(BASE/'preflight_ledger.json')
    gpu_seconds=sum(a['command_elapsed_seconds'] for a in attempts if a['gpu'])
    pre_seconds=sum(a['command_elapsed_seconds'] for a in preflight)
    budget=read(BASE/'ledger.json')
    assert abs(gpu_seconds+pre_seconds-budget['incremental_spent_seconds'])<1e-9
    assert all(a['exit_code']==0 and a.get('finished_at') for a in attempts)
    checks.update(job_attempts=len(attempts),preflight_attempts=len(preflight),gpu_seconds=gpu_seconds,preflight_seconds=pre_seconds,
        incremental_hours=(gpu_seconds+pre_seconds)/3600,cumulative_hours=budget['cumulative_gpu_seconds']/3600,
        remaining_cumulative_hours=(108000-budget['cumulative_gpu_seconds'])/3600,
        launched_at=state['launched_at'],finished_at=state['finished_at'],
        worker_wall_minutes=(datetime.fromisoformat(state['finished_at'])-datetime.fromisoformat(state['launched_at'])).total_seconds()/60)
    for jid,rec in state['jobs'].items():
        assert rec['status']=='completed'
        for out in rec['job']['outputs']:
            value=read(out['path']);assert value['outcome']=='completed'
    a=pd.read_csv(BASE/'adaptation_results.csv')
    assert len(a)==180 and set(a.status)=={'completed'} and not a.duplicated(['event','seed','support_draw','arm']).any()
    parent_states={p['seed']:torch.load(p['checkpoint'],map_location='cpu',weights_only=True)['model'] for p in c['parents']}
    qstates={int(s):torch.load(v['q_checkpoint'],map_location='cpu',weights_only=True)['model'] for s,v in cache['parents'].items()}
    checked=0;frozen_tensors=0
    for row in a.itertuples():
        assert sha(row.artifact_path)==row.artifact_sha256
        obj=torch.load(row.artifact_path,map_location='cpu',weights_only=True)
        base=qstates[row.seed] if row.arm.startswith('Q') else parent_states[row.seed]
        for name,tensor in obj['model'].items():
            if name in ['cls.weight','cls.bias']:continue
            assert torch.equal(tensor,base[name]),(row.arm,name)
            frozen_tensors+=1
        if row.arm in ['F0','Qbase']:
            assert all(torch.equal(v,base[k]) for k,v in obj['model'].items())
        checked+=1
    checks.update(adaptation_checkpoint_hashes=checked,unchanged_extractor_tensor_comparisons=frozen_tensors)
    detail=[*read(BASE/'adaptation_fp_details.json'),*read(BASE/'adaptation_q_details.json')]
    numerical=[];stability=[];support=[]
    for seed,entry in cache['parents'].items():
        seed=int(seed);mean=np.asarray(entry['mean']);scale=np.asarray(entry['scale']);assert np.min(scale)>=.001
        for rec in entry['records']:
            assert sha(rec['path'])==rec['sha256']
            if rec['role']!='target_support':continue
            z=np.load(rec['path']);y=z['mask'];assert (y!=-1).all() and len(y)<=4096
            p=z['p'];pp=z['pp'];weight=hs.reliability(p,pp);correct=(p>.5)==y
            pf=hs.design(z['f'],mean,scale);pq=hs.design(z['q'],mean,scale)
            for cls in [0,1]:
                take=y==cls;wrong=take&~correct;right=take&correct
                stability.append(dict(event=rec['event'],seed=seed,draw=rec['draw'],reference_class=cls,
                    sampled_pixels=int(take.sum()),teacher_errors=int(wrong.sum()),mean_weight_wrong=float(weight[wrong].mean()) if wrong.any() else None,
                    mean_weight_correct=float(weight[right].mean()) if right.any() else None,
                    weighted_error_share=float(weight[wrong].sum()/weight[take].sum()) if take.any() else None,
                    stability_mean_difference=float(np.abs(p[take]-pp[take]).mean()) if take.any() else None))
            support.append(dict(event=rec['event'],seed=seed,draw=rec['draw'],chip=rec['filename'],
                sampled_pixels=len(y),sampled_water_pixels=int((y==1).sum()),sampled_teacher_water=int((p>.5).sum()),
                sampled_teacher_errors=int((~correct).sum()),sampled_water_fraction=float((y==1).mean()),
                available_chip_label_pixels=rec['valid_label_pixels'],available_chip_label_area_km2=rec['label_area_m2']/1e6,
                clipped_target_fraction=float((np.abs(hs.target(p))==8).mean()),
                teacher_margin_mean=float(hs.target(p).mean()),student_margin_mean=float(z['mf'].mean()),
                standardized_feature_distortion_RMS=float(np.sqrt(np.mean((pq-pf)**2)))))
            choice=read(BASE/f'choice_s{seed}.json')
            episode=[r for r in detail if r['event']==rec['event'] and r['seed']==seed and r['draw']==rec['draw']]
            for row in episode:
                arm=row['arm'];info=row['fit']
                if 'objective_gradient_max' not in info:continue
                quant=arm.startswith('Q');phi=pq if quant else pf;weights=weight if arm in ['F4','Q1','Q3'] else None
                tar=hs.target(p)
                if arm=='F1':phi=np.column_stack([z['mf'],np.ones(len(y))])
                if arm=='B_supervised':tar=np.where(y==1,np.log(19),-np.log(19))
                if arm=='B_teacher_label':tar=.5*np.where(y==1,np.log(19),-np.log(19))+.5*tar
                delta,observed=hs.solve(phi,tar-z['mq' if quant else 'mf'],weights,
                    pq-pf if arm in ['Q2','Q3'] else None,choice['lambda_'],1 if arm in ['Q2','Q3'] else 0)
                error=float(np.max(np.abs(np.asarray(info['undamped_delta'])-observed['undamped_delta'])))
                assert error<1e-9 and observed['objective_gradient_max']<1e-8
                numerical.append(dict(event=rec['event'],seed=seed,draw=rec['draw'],arm=arm,
                    recomputed_undamped_error=error,gradient_max=observed['objective_gradient_max'],eta=observed['eta']))
    table('support_audit.csv',support);table('reliability_audit.csv',stability);table('solver_audit.csv',numerical)
    checks['recomputed_ridge_calibration_solves']=len(numerical)
    for d in detail:
        row=a[(a.event==d['event'])&(a.seed==d['seed'])&(a.support_draw==d['draw'])&(a.arm==d['arm'])].iloc[0]
        scores=score(d['counts'])
        assert all(v is None or abs(row[k]-v)<1e-12 for k,v in scores.items())
    checks['adaptation_scores_recomputed_from_integer_counts']=len(detail)
    a['key']=a.event+'_'+a.seed.astype(str)+'_'+a.support_draw.astype(str)
    fp=a[a.arm=='F0'].set_index('key');qb=a[a.arm=='Qbase'].set_index('key')
    a['delta_FP_pp']=[100*(r.mIoU-fp.loc[r.key].mIoU) for r in a.itertuples()]
    a['delta_parent_pp']=[100*(r.mIoU-(qb if r.arm.startswith('Q') else fp).loc[r.key].mIoU) for r in a.itertuples()]
    a['water_delta_parent_pp']=[100*(r.water_IoU-(qb if r.arm.startswith('Q') else fp).loc[r.key].water_IoU) for r in a.itertuples()]
    event=a.groupby(['event','arm'])[['mIoU','water_IoU','water_precision','water_recall','delta_FP_pp','delta_parent_pp','water_delta_parent_pp']].mean().reset_index()
    event.to_csv(OUT/'adaptation_events.csv',index=False)
    arms=[]
    for arm,rows in a.groupby('arm'):
        arms.append(dict(arm=arm,n_episodes=len(rows),mIoU_pct=100*rows.mIoU.mean(),water_IoU_pct=100*rows.water_IoU.mean(),
            water_precision_pct=100*rows.water_precision.mean(),water_recall_pct=100*rows.water_recall.mean(),
            delta_parent_pp=rows.delta_parent_pp.mean(),delta_FP_pp=rows.delta_FP_pp.mean(),water_delta_parent_pp=rows.water_delta_parent_pp.mean(),
            positive_episodes=int((rows.delta_parent_pp>1e-10).sum()),negative_episodes=int((rows.delta_parent_pp< -1e-10).sum()),
            min_delta_pp=rows.delta_parent_pp.min(),max_delta_pp=rows.delta_parent_pp.max(),
            median_solver_ms=1000*rows.solve_optimization_seconds.median(),median_component_cost_ms=1000*rows.cold_end_to_end_seconds.median()))
    table('adaptation_summary.csv',arms)
    effects=[]
    indexed=a.set_index(['key','arm'])
    pairs=[('F4','F3'),('F4','F1'),('F4','F2_20'),('Q3','Q0'),('Q3','Q1'),('Q3','Q2'),('Q2','Q0'),('B_teacher_label','B_supervised'),('B_teacher_label','B_supervised_SGD20')]
    for candidate,control in pairs:
        for key in fp.index:
            u=indexed.loc[(key,candidate)];v=indexed.loc[(key,control)]
            effects.append(dict(event=u.event,seed=int(u.seed),draw=int(u.support_draw),candidate=candidate,control=control,
                mIoU_difference_pp=100*(u.mIoU-v.mIoU),water_IoU_difference_pp=100*(u.water_IoU-v.water_IoU)))
    table('paired_controls.csv',effects)
    joint=read(BASE/'joint_confusions.json')['records'];class_rows=[]
    for row in joint:
        j=np.asarray(row['joint_by_reference_class']);tc=np.asarray(row['teacher_counts']);sc=np.asarray(row['student_counts']);oc=np.asarray(row['unattainable_oracle_counts'])
        assert tc[:,0].sum()-sc[:,0].sum()==j[:,1].sum()-j[:,2].sum()
        assert j.sum()==tc[0,0]+tc[1].sum()
        assert oc[:,0].sum()==j[:,:3].sum()
        for cls in [0,1]:
            n=int(j[cls].sum());class_rows.append(dict(dataset=row['dataset'],seed=row['seed'],event=row['event'],scope=row['scope'],
                comparator=row['comparator'],reference_class=cls,valid_pixels=n,both_correct=int(j[cls,0]),teacher_correct_student_wrong=int(j[cls,1]),
                student_correct_teacher_wrong=int(j[cls,2]),both_wrong=int(j[cls,3]),G=float(j[cls,1]/n) if n else None,H=float(j[cls,2]/n) if n else None))
    table('joint_errors_by_class.csv',class_rows);checks['joint_event_cells_recomputed']=len(joint)
    h=pd.read_csv(BASE/'teacher_headroom.csv')
    header=h[(h.scope=='full')].groupby(['dataset','comparator','aggregation'])[['G','H','teacher_water_IoU','student_water_IoU','oracle_water_IoU','teacher_mIoU','student_mIoU','oracle_mIoU']].mean().reset_index()
    header.to_csv(OUT/'headroom_summary.csv',index=False)
    # Correct the first parser's unanchored 'ep' expression, which matched 'step'.
    curves=[];crossings=[];receipts=read(BASE/'ofa_training_receipts.json')
    for rec in receipts:
        jid=rec['run_id'];n=100 if '_N100_' in jid else 2500;updates=n//16
        log=Path('/data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006/queue_prompt4_release/logs')/(jid+'.attempt1.log')
        obs=[]
        for line in log.read_text(errors='replace').splitlines():
            match=re.search(r'^\s*\[val\]\s+step=(\d+)\s+mIoU=([0-9.]+)',line)
            if match:obs.append((int(match[1])+1,float(match[2])))
        assert obs
        arch=jid[len('O4_train_'):].split('_matched_')[0].split('_historical_')[0]
        method='ofa' if '_ofa_' in jid else 'lkd'
        best=rec['checkpoint_checks']['best']['epochs_completed'];last=rec['checkpoint_checks']['last']['epochs_completed']
        for ep,quality in obs:
            curves.append(dict(run_id=jid,architecture=arch,method=method,distinct_scenes=n,epoch=ep,
                optimizer_updates=ep*updates,mIoU=quality,selected=ep==best,final=ep==last,
                log_sha256=sha(log),score_precision='logged four decimals',elapsed_at_validation_seconds=None))
        for threshold in [.70,.75,.80,.85]:
            i=next((i for i,(ep,qv) in enumerate(obs) if qv>=threshold),None)
            crossings.append(dict(run_id=jid,architecture=arch,method=method,distinct_scenes=n,threshold=threshold,
                first_observed_update=obs[i][0]*updates if i is not None else None,
                previous_observed_update=obs[i-1][0]*updates if i is not None and i else 0 if i==0 else None,
                status='observed_logged_crossing' if i is not None else 'censored_unreached',
                crossing_time_seconds=None,total_training_wrapper_seconds=rec['total_measured_command_seconds'],
                isolated_training_time_seconds=None,selected_update=best*updates,final_update=last*updates,
                graph_identity=arch+'; original EfficientViT activation where applicable',
                objective_mask='legacy_all_pixels' if 'allvalid' in jid else 'valid_only'))
    table('ofa_learning_curves_corrected.csv',curves);table('ofa_learning_efficiency_corrected.csv',crossings)
    checks['corrected_curve_observations']=len(curves);checks['corrected_curve_selected_points']=sum(r['selected'] for r in curves);checks['corrected_curve_final_points']=sum(r['final'] for r in curves)
    dep=[]
    for name in ['measure_FP32','measure_FP16']:
        receipt=read(BASE/'receipts'/f'{name}.json');assert sha(receipt['artifact_path'])==receipt['artifact_sha256']
        module=torch.jit.load(receipt['artifact_path'],map_location='cpu');dtypes=sorted({str(v.dtype) for v in module.parameters()})
        assert dtypes==['torch.float16' if name=='measure_FP16' else 'torch.float32']
        dep.append(dict(precision=receipt['precision'],actual_parameter_dtypes=','.join(dtypes),artifact_bytes=receipt['deployed_artifact_bytes'],
            p50_ms=receipt['p50_ms'],p95_ms=receipt['p95_ms'],incremental_MiB=receipt['incremental_allocator_bytes']/2**20,
            source_mIoU_pct=100*receipt['source_scores']['mIoU'],source_water_IoU_pct=100*receipt['source_scores']['water_IoU'],
            reported_post_accuracy_matmul_tf32=receipt['backend']['cuda_matmul_allow_tf32'],reported_post_accuracy_cudnn_tf32=receipt['backend']['cudnn_allow_tf32'],
            reported_post_accuracy_matmul_precision=receipt['backend']['float32_matmul_precision'],
            timing_setting='strict configured before tracing; code replay shows loaders reset flags after timing; no contemporaneous timing flag receipt'))
    table('deployment_audit.csv',dep)
    # Derive full-chip reliability errors without counting the same teacher twice across parents.
    diag=cache['teacher_stability_diagnostics'];water=[r for r in diag if r['reference_class']==1 and not r['support'].startswith('dev_')]
    confident=sum(r['confident_stable'] for r in water);wrong=sum(r['confident_stable_wrong'] for r in water)
    issues=[dict(type='reporting',finding='OFA parser matched ep inside step, shifting every recovered epoch/update one epoch early; selected/final markers were wrong',remedy='Corrected tables preserve original logs and outputs'),
        dict(type='provenance',finding='Deployment backend record is captured after loaders reset flags; FP16 record hardcodes float32 input/checkpoint dtype',remedy='Artifact dtype audited; report recorded post-accuracy flags separately from configured timing flags; future receipt must record both phases'),
        dict(type='scope',finding='Pilot evaluates one map per event and only 92 disjoint query chips, not full event/benchmark transfer for adapted heads',remedy='Report exploratory selected-map scope explicitly'),
        dict(type='cost',finding='cold_end_to_end_seconds is a component sum excluding teacher model construction, input preparation, transformations and folding/serialization',remedy='Treat as reused-engine component estimate, not measured cold end-to-end adaptation'),
        dict(type='prerequisite',finding='No competent QAT or source validation quality frontier was measured for the new Q backbone; floating PTQ simulation only',remedy='Keep quantized accuracy exploratory and defer native deployment claim')]
    save('audit_summary.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),checks=checks,errors=errors,issues=issues,
        target_confident_stable_water_pixels=confident,target_confident_stable_water_errors=wrong,target_confident_stable_water_error_fraction=wrong/confident,
        target_query_chips=sum(len(e['queries']) for e in c['episodes']),target_events=len(c['episodes']),
        labels_available_six_unique_support_chips=sum(r['available_chip_label_pixels'] for r in support if r['seed']==42),
        plots=['adaptation_control_effects.png','headroom_estimands.png','ofa_learning_curves_corrected.png']))
    figures(pd.DataFrame(arms),event,header,pd.DataFrame(curves))
    print(json.dumps(dict(checks=checks,errors=errors,issues=len(issues),confident_stable_water_error_fraction=wrong/confident),indent=2))


def figures(arms,event,header,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['F1','F2_1','F2_5','F2_20','F3','F4','B_supervised','B_teacher_label','B_supervised_SGD20','Q0','Q1','Q2','Q3']
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1,1.5]})
    s=arms.set_index('arm').loc[selected]
    axes[0].barh(s.index,s.delta_parent_pp,color=['#468fa8' if v>=0 else '#b75e62' for v in s.delta_parent_pp]);axes[0].axvline(0,color='black',lw=.7)
    axes[0].set_xlabel('Mean paired mIoU change (pp)');axes[0].set_title('Each arm vs corresponding frozen parent')
    matrix=event[event.arm.isin(selected)].pivot(index='arm',columns='event',values='delta_parent_pp').reindex(selected)
    im=axes[1].imshow(matrix.to_numpy(),cmap='RdBu',vmin=-3,vmax=3,aspect='auto');axes[1].set_xticks(range(3),matrix.columns);axes[1].set_yticks(range(len(selected)),selected)
    for i in range(len(selected)):
        for j in range(3):axes[1].text(j,i,f'{matrix.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8)
    axes[1].set_title('Every exploratory event; four paired outcomes/event');fig.colorbar(im,ax=axes[1],label='mIoU pp');fig.tight_layout();fig.savefig(OUT/'adaptation_control_effects.png',dpi=170);plt.close(fig)
    w=header[(header.dataset=='wf')&(header.comparator=='T|S')];fig,ax=plt.subplots(figsize=(7,4));x=np.arange(2);width=.23
    for i,(col,label) in enumerate([('teacher_water_IoU','Teacher'),('student_water_IoU','R4'),('oracle_water_IoU','Unattainable oracle')]):
        ax.bar(x+(i-1)*width,100*w[col].to_numpy(),width,label=label)
    ax.set_xticks(x,['Equal-event','Pooled pixels']);ax.set_ylabel('Water IoU (%)');ax.set_title('WF joint headroom: averages across two R4 parents');ax.legend();fig.tight_layout();fig.savefig(OUT/'headroom_estimands.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,n in zip(axes,[100,2500]):
        subset=curves[(curves.architecture=='efficientvit_b0')&(curves.distinct_scenes==n)]
        for jid,rows in subset.groupby('run_id'):
            color='#337ea0' if rows.method.iloc[0]=='lkd' else '#d4813b'
            ax.plot(rows.optimizer_updates,100*rows.mIoU,color=color,alpha=.65)
            selected=rows[rows.selected];ax.scatter(selected.optimizer_updates,100*selected.mIoU,color=color,s=22)
        ax.set(xlabel='Optimizer updates',ylabel='Source validation mIoU (%)',title=f'EfficientViT N{n}: corrected observed epochs')
        ax.plot([],[],color='#337ea0',label='LKD');ax.plot([],[],color='#d4813b',label='OFA');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'ofa_learning_curves_corrected.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
