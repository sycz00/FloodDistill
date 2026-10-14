"""CPU report/closure collection, including partial, busy and failed campaigns."""
import csv,json,math,statistics,time,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import common_v9 as c
import edge_v9 as e

def csvout(name,rows,fields=None):
    if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
    if not fields:fields=['status','reason']
    with open(c.ROOT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def avg(vals):
    a=[x for x in vals if x is not None];return float(np.mean(a)) if a else None

def bootstrap(diffs):
    a=np.asarray(diffs,float);a=a[np.isfinite(a)]
    if not len(a):return None,None
    rng=np.random.default_rng(1729);means=np.mean(a[rng.integers(0,len(a),size=(10000,len(a)))],axis=1);lo,hi=np.percentile(means,[2.5,97.5]);return float(lo),float(hi)

def read_metrics():
    counts={};strata={};dry={};hist={};decode={};joint=[]
    for mode in ['source','normal','sealed']:
        for p in (c.ROOT/'metrics'/mode).glob('slot*.json'):
            a=c.read(p)
            for row in a['event_counts']:counts[(row['dataset'],row['model'],row['event'],row['variant'])]=row['counts']
            for row in a['strata']:strata[(row['dataset'],row['model'],row['stratum'])]=row['counts']
            for row in a['dry']:dry[(row['dataset'],row['model'],row['name'],row['variant'])]=row
            decode.update(a['decode_cpu_seconds']);joint.extend(a['joint'])
        for p in (c.ROOT/'metrics'/mode).glob('hist_slot*.npz'):
            with np.load(p) as a:
                for k,h in zip(a['keys'],a['histograms']):hist[tuple(json.loads(str(k)))]=h.copy()
    return counts,strata,list(dry.values()),hist,decode,joint

def main():
    manifest=c.read(c.ROOT/'manifest.json',{});state=c.read(c.ROOT/'state.json',{});status=c.read(c.ROOT/'scheduler_status.json',{});counts,strat,dry,hists,decode,joint=read_metrics();models=c.read(c.ROOT/'final_models.json',{}).get('models',[]);byid={x['id']:x for x in models}
    rows=[];eventrows=[];by_model=defaultdict(dict)
    for (ds,model,event,variant),cell in counts.items():
        metrics=e.scores(cell);eventrows.append(dict(dataset=ds,model=model,seed=byid.get(model,{}).get('seed'),event=event,variant=variant,tp=cell[0],fp=cell[1],fn=cell[2],tn=cell[3],**metrics));by_model[(ds,model,variant)][event]=np.array(cell)
    for (ds,model,variant),ev in by_model.items():
        pooled=e.scores(sum(ev.values()));per=[e.scores(x) for x in ev.values()];rows.append(dict(dataset=ds,model=model,seed=byid.get(model,{}).get('seed'),method=byid.get(model,{}).get('method',model),variant=variant,events=len(ev),
            **{'pooled_'+k:v for k,v in pooled.items()},**{'equal_event_'+k:avg([x[k] for x in per]) for k in pooled}))
    lookup={(x['dataset'],x['model'],x['variant']):x for x in rows};csvout('edge_arms_by_event.csv',eventrows);csvout('edge_arms_results.csv',rows)
    fs=[]
    totals={(ds,m):sum(v['counts'][2] for v in []) for ds,m,_ in []}
    for (ds,model,label),cell in strat.items():
        total=by_model.get((ds,model,'deployed'),{});miss=sum(int(x[2]) for x in total.values());fs.append(dict(dataset=ds,model=model,stratum=label,tp=cell[0],fp=cell[1],fn=cell[2],tn=cell[3],valid_pixels=sum(cell),share_all_missed_water=cell[2]/miss if miss else None,**e.scores(cell)))
    csvout('failure_strata.csv',fs);csvout('dry_window_alarms.csv',dry)
    aps=[]
    from evaluate_v9 import ap
    groups=defaultdict(dict)
    for (ds,m,event),h in hists.items():groups[(ds,m)][event]=h
    for (ds,m),events in groups.items():
        h=sum(events.values());val=ap(h);ruleh=sum(groups.get((ds,'MNDWI'),{}).values()) if groups.get((ds,'MNDWI')) else None
        matched=None;recall=None
        if ruleh is not None:
            rc=lookup.get((ds,'MNDWI','deployed'),{}).get('pooled_recall');recall=rc
            if rc is not None and h[:,1].sum():
                tp=np.cumsum(h[::-1,1]);fp=np.cumsum(h[::-1,0]);r=tp/tp[-1];i=int(np.searchsorted(r,rc));i=min(i,len(r)-1);matched=float(tp[i]/(tp[i]+fp[i])) if tp[i]+fp[i] else None
        aps.append(dict(dataset=ds,model=m,pooled_AP=val,equal_event_AP=avg([ap(x) for x in events.values()]),precision_at_MNDWI_recall=matched,MNDWI_recall=recall,protocol='Approximate4096 fixed bins; clipped margins[-32,32]; bin ties'))
    csvout('threshold_free_ap.csv',aps);apmap={(x['dataset'],x['model']):x for x in aps}
    ood=[]
    for r in rows:
        if r['variant']!='deployed':continue
        event=by_model[(r['dataset'],r['model'],'deployed')];rule=by_model.get((r['dataset'],'MNDWI','deployed'),{});shared=sorted(set(event)&set(rule));diff=[e.scores(event[k])['water_IoU']-e.scores(rule[k])['water_IoU'] for k in shared if e.scores(event[k])['water_IoU'] is not None and e.scores(rule[k])['water_IoU'] is not None];ci=bootstrap(diff)
        ood.append(dict(**r,**{k:v for k,v in apmap.get((r['dataset'],r['model']),{}).items() if k not in ['dataset','model']},events_better_than_MNDWI=sum(x>0 for x in diff),paired_events=len(diff),vs_MNDWI_mean_water_IoU=avg(diff),vs_MNDWI_event_bootstrap_low=ci[0],vs_MNDWI_event_bootstrap_high=ci[1],exposure='sealed evaluation' if r['dataset'].startswith('wf2') else 'research exposed'))
    csvout('ood_comparison_table.csv',ood)
    settings=c.read(c.ROOT/'source_operating_points.json',{});op=[]
    for ds,m,_ in by_model:
        if (ds,m,'deployed') not in by_model or any(x['dataset']==ds and x['model']==m for x in op):continue
        settings_m=settings.get(m);tau=settings_m.get('tau') if settings_m else None;available={float(v.split('=')[1]):ev for (d,n,v),ev in by_model.items() if d==ds and n==m and v.startswith('tau=')}
        oracle=[]
        for event in by_model[(ds,m,'deployed')]:
            candidates=[(e.scores(ev[event])['water_IoU'],t) for t,ev in available.items() if event in ev and e.scores(ev[event])['water_IoU'] is not None]
            if candidates:
                best=max(candidates,key=lambda x:x[0]);oracle.append(best[0])
        for variant in ['deployed',f'tau={tau:g}' if tau is not None else'', 'em','anchor']:
            x=lookup.get((ds,m,variant))
            if x:op.append(dict(**x,source_selected_tau=tau,event_oracle_equal_water_IoU=avg(oracle),oracle='Unattainable event-labelled threshold grid; analysis only',selection='S11 validation only',rule_quantization='Inherited50 half-unit margin bins for EM/anchor'))
    csvout('operating_point_analysis.csv',op)
    labelrows=[]
    for seed in c.SEEDS:
        human=c.read(c.ROOT/'scientific_receipts'/f'human_s{seed}.json');teacher=c.read(c.ROOT/'scientific_receipts'/f'teacher_s{seed}.json')
        paired=bool(human and teacher and human.get('outcome')=='completed' and teacher.get('outcome')=='completed')
        if paired:
            assert human['initial_base_sha256']==teacher['initial_base_sha256'];assert human['first_image_hashes']==teacher['first_image_hashes']
        for ds in ['s11val','s11','sturm','wf','wf2']:
            h=lookup.get((ds,f'human_s{seed}','deployed'),{});t=lookup.get((ds,f'teacher_s{seed}','deployed'),{})
            delta=t.get('equal_event_water_IoU')-h.get('equal_event_water_IoU') if h.get('equal_event_water_IoU') is not None and t.get('equal_event_water_IoU') is not None else None
            labelrows.append(dict(seed=seed,dataset=ds,status='COMPLETED' if paired and h and t else 'PENDING_OR_UNAVAILABLE',human_equal_event_water_IoU=h.get('equal_event_water_IoU'),teacher_label_equal_event_water_IoU=t.get('equal_event_water_IoU'),paired_difference=delta,initial_and_input_hashes_equal=paired,objective='True CE in both;252 identical images;750 updates; source-only earliest-best checkpoint',qualification='Task-adapted teacher saw these source-training images; not unseen annotation-free evidence'))
    csvout('same_image_label_swap.csv',labelrows)
    gates=[]
    for arm in ['E1','E2','E3','D']:
        perseed=[];per_event=[];thin=[];sturm=[]
        for seed in c.SEEDS:
            model=f'E0_s{seed}' if arm=='D' else f'{arm}_s{seed}';variant='D' if arm=='D' else'deployed';a=by_model.get(('wf2',model,variant),{});b=by_model.get(('wf2',f'E0_s{seed}','deployed'),{});es=sorted(set(a)&set(b));diff={event:e.scores(a[event])['water_IoU']-e.scores(b[event])['water_IoU'] for event in es if e.scores(a[event])['water_IoU'] is not None and e.scores(b[event])['water_IoU'] is not None};per_event.append(diff);perseed.append(avg(list(diff.values())))
            if arm!='D':
                x=strat.get(('s11',model,'water_halfwidth_<=2'));y=strat.get(('s11',f'E0_s{seed}','water_halfwidth_<=2'))
                if x and y and e.scores(x)['recall'] is not None and e.scores(y)['recall'] is not None:thin.append(e.scores(x)['recall']-e.scores(y)['recall'])
            x=lookup.get(('sturm',model,variant));y=lookup.get(('sturm',f'E0_s{seed}','deployed'))
            if x and y and x['equal_event_water_IoU'] is not None and y['equal_event_water_IoU'] is not None:sturm.append(x['equal_event_water_IoU']-y['equal_event_water_IoU'])
        complete=all(x is not None for x in perseed);es=set.intersection(*(set(x) for x in per_event)) if per_event else set();mean_event=[avg([x[event] for x in per_event]) for event in sorted(es)];lo,hi=bootstrap(mean_event);mean=avg(perseed);sd=float(np.std([x for x in perseed if x is not None],ddof=1)) if sum(x is not None for x in perseed)>1 else None
        drydiff=[]
        for seed in c.SEEDS:
            model=f'E0_s{seed}' if arm=='D' else f'{arm}_s{seed}';variant='D' if arm=='D' else'deployed';aa=[x['alarm_01'] for x in dry if x['dataset']=='wf2' and x['model']==model and x['variant']==variant];bb=[x['alarm_01'] for x in dry if x['dataset']=='wf2' and x['model']==f'E0_s{seed}' and x['variant']=='deployed']
            if aa and bb:drydiff.append(avg(aa)-avg(bb))
        passed=complete and mean>=.005 and lo is not None and lo>0 and sum(x>0 for x in perseed)>=3 and len(sturm)==4 and avg(sturm)>=-.005 and len(drydiff)==4 and avg(drydiff)<=.01 and (arm=='D' or(len(thin)==4 and avg(thin)>=.10))
        gates.append(dict(arm=arm,status='PASSED' if passed else 'SCIENTIFIC_GATE_FAILED' if complete else 'UNAVAILABLE',sealed_delta=mean,event_bootstrap_low=lo,event_bootstrap_high=hi,across_seed_SD=sd,positive_seeds=sum(x>0 for x in perseed if x is not None),paired_events=len(es),S11_thin_recall_delta=avg(thin),STURM_delta=avg(sturm),sealed_dry_alarm_1percent_delta=avg(drydiff),qualification='Seed-mean whole-event bootstrap10000 seed1729; seedSD separate; nonsignificance never equivalence'))
    c.save(c.ROOT/'scientific_gates.json',gates)
    csvout('edge_label_changes.csv',[dict(key=k,value=v) for k,v in c.read(c.ROOT/'edge_label_manifest.json',{}).get('counts',{}).items()])
    seal=c.read(c.ROOT/'sealed_wf2.json',{});csvout('wf2_exclusions.csv',[dict(scene=x['scene'],event=x['event'],reasons=json.dumps(x['reasons'])) for x in seal.get('excluded',[])])
    csvout('alternative_teacher_comparison.csv',[dict(status='UNAVAILABLE',checkpoint='/home/schmalstieg/repos/OLD/FD/tmp_runs_removed/runs_repro_copy/s11_unet_ce_N252_s42/best.pt',reason='Checkpoint has model/optimizer/epoch/metrics only; independent 252-human-label training lineage not verified; not replaced by pseudo-trained UNet ensemble')])
    frontier=[]
    for ds in ['s11val','s11','sturm','wf','wf2']:
        for seed in c.SEEDS:
            fp=lookup.get((ds,f'fl_R4_sel_s{seed}__selected','deployed'),{});q=lookup.get((ds,f'PTQ_s{seed}','deployed'),{});tau=settings.get(f'PTQ_s{seed}',{}).get('tau');qt=lookup.get((ds,f'PTQ_s{seed}',f'tau={tau:g}'),{}) if tau is not None else{}
            if fp and q:
                d=q['equal_event_water_IoU']-fp['equal_event_water_IoU'];dt=qt['equal_event_water_IoU']-fp['equal_event_water_IoU'] if qt else None
                frontier.append(dict(dataset=ds,seed=seed,graph='R4',precision='W4A8 body/W8A8 stem/FP32head',FP_equal_water_IoU=fp['equal_event_water_IoU'],PTQ_equal_water_IoU=q['equal_event_water_IoU'],deployed_delta=d,source_selected_tau=tau,source_threshold_delta=dt,absolute_gap_reduction=abs(d)-abs(dt) if dt is not None else None,source_tolerance_1pp=ds=='s11val' and d>=-.01,source_tolerance_2pp=ds=='s11val' and d>=-.02,source_tolerance_3pp=ds=='s11val' and d>=-.03,native_kernel=False,interpretation='Exploratory declared loss tolerances; no lossless/equivalence claim'))
    for m in c.read(c.V7/'reuse_models.json',[]):
        if m['graph'] in ['R2','R3','R4','A3','A0']:frontier.append(dict(graph=m['graph'],seed=m['seed'],model=m['id'],checkpoint=m['checkpoint'],sha256=m['sha256'],precision='FP32 historical fixed graph',qualification='Reuse compatible archived readouts; early-stop/update histories remain distinct'))
    csvout('full_precision_frontier.csv',frontier)
    checks=[]
    for name,job in manifest.get('jobs',{}).items():
        rec=c.read(c.ROOT/'command_receipts'/f'{name}.json');checks.append(dict(block=job['block'],job=name,status=rec['status'] if rec else state.get('jobs',{}).get(name,'QUEUED'),seeds=job.get('seed'),paired_controls=job.get('paired_controls'),reason=rec.get('reason') if rec else None,evidence=str(c.ROOT/'command_receipts'/f'{name}.json')))
    branches=c.read(c.ROOT/'conditional_admission.json',{}).get('branches',manifest.get('conditional_branches',[]))
    for x in branches:checks.append(dict(block=x['name'],status=x['status'],reason=x['reason'],paired_controls=x.get('controls'),evidence='conditional_admission.json' if (c.ROOT/'conditional_admission.json').exists() else'manifest.json'))
    checks.append(dict(block='11-window independent dry review',status='UNAVAILABLE',reason='No independent human verification records; teacher/rule labels do not substitute',evidence=str(c.ROOT.parent/'prompt3_dry_verification/annotations.template.json')))
    csvout('experiment_checklist.csv',checks)
    negative='''# Preserved negative results and withdrawn directions

V8 head adaptation remains a research-exposed negative pilot. Unchanged F4/Q3 and original V9 head matrices are retired. G<H all-pixel error counts do not establish a universal IoU/update bound. Disagreement coverage is future work.

OOD-labelled archive calibration is withdrawn. Every V9 checkpoint, threshold, temperature and rule setting uses S11 validation only. Event-optimal target thresholds are explicitly unattainable diagnostics.

Claude's exposed cloud-band spectral re-decision lost 2–6 pp on WF and 1–2 pp on S11; it is not retried. Low vegetation-stratum recall is an observation, not proof of spectral invisibility. None of the unavailable or deferred blocks is negative scientific evidence.

Teacher16-pixel tokens do not imply16-pixel decoder logits: the inspected teacher decoder and R4 both produce H/4 logits before upsampling. An E1 gain alone cannot establish a160m versus40m output-resolution mechanism. Ordinary dry unattended imagery remains unverified.
''';(c.ROOT/'negative_results.md').write_text(negative)
    closure='''# Reviewer closure status

The actual letter is paper/review.docx. R1.1/R2.1 need restrained novelty language and the measured edge-control gates. R1.2/R2.4 reuse MobileNet/LightFormer comparisons and add the new sealed readout. R2.3 retains bounded OFA update evidence, with no universal wall-clock speed claim. R1.3/R2.2 require data roles, the author's STURM history, benchmark exposure and the paired label-source results. R1.4 requires a consolidated component table. R1.5 requires surface-water/flood-application terminology. R2.5 uses threshold-free comparison, event uncertainty and failure strata. R1.6/R2.6 need independent-reference dry-window diagnostics, cautious deployment/power wording and the ordinary-dry limitation.

The manuscript and response remain author work; this queue does not submit or edit them. Numerical corrections are in revision_corrections.csv. STURM did not choose OFA/EfficientViT; null additional teacher/OOD/saliency investigations remain author-reported development exposure.
''';(c.ROOT/'reviewer_closure_status.md').write_text(closure);(c.ROOT/'reviewer_response_map.md').write_text(closure)
    finished=bool(state.get('finished')) or state.get('phase')=='COLLECTING';title='Final collection' if finished else 'INTERIM: queue is not complete'
    budget=status.get('budget',{});completed=sum(x['status']=='COMPLETED' for x in checks);failed=[x for x in checks if x['status']=='TECHNICAL_FAILED'];deferred=[x for x in checks if x['status'].startswith(('DEFERRED','UNAVAILABLE'))]
    text=[f'# V9 research report — {title}',f'Collected {c.q.now()}. Run `{c.ROOT.name}`.',f'Completed checklist entries: {completed}. Technical failures: {len(failed)}. Deferred/unavailable entries: {len(deferred)}.',f'GPU budget snapshot: {json.dumps(budget)}. Initial GPU wait: {status.get("wait_seconds")} seconds. Concurrent GPU-command time is summed; CPU time and waits are separate.',
        f'Sealed metadata: {len(seal.get("scenes",[]))} retained scenes, {seal.get("events")} events; {len(seal.get("excluded",[]))} exclusions; {seal.get("input_windows_upper_bound")} input-only upper windows. Actual scored windows are determined only in the final reference read. Prior code/log inspection found no retained executed train/val GT read; no historical OS open trace exists.',
        '## Executed evidence', 'The experiment_checklist.csv records every required and conditional block, controls, state, evidence and deferral reason. same_image_label_swap.csv contains four paired true-CE controls. edge_arms_results.csv and edge_arms_by_event.csv preserve all outcomes. failure_strata.csv reports recall, false water and share of misses. ood_comparison_table.csv and threshold_free_ap.csv distinguish pooled versus equal-event comparisons. operating_point_analysis.csv separates source selection, label-free rules and unattainable event oracles. dry_window_alarms.csv reports independent-reference flood-scene dry land at0.1/1/5% area thresholds. full_precision_frontier.csv keeps simulator accuracy separate from native deployment.',
        '## Declared scientific gates',json.dumps(gates,indent=2),'No attribution claim follows merely because two arms are nonsignificantly different. Positive findings require the complete four-seed paired protocol and the declared mechanism/dry-alarm safeguards.',
        '## What must be done', 'Authors must apply verified numerical/terminology corrections, reconcile the teacher baseline under one protocol, consolidate component evidence and write the Discussion and point-by-point response. Present OOD transfer at the event level and retain unfavorable outcomes. Independently verify ordinary dry imagery before making unattended false-alarm claims. Establish source training provenance for the conventional UNet before the alternative-teacher control. Native low-bit deployment remains unsupported on this workstation simulator.',
        '## Limitations', 'The teacher saw the252 source images. The cohort was selected using Prithvi. Earlier public benchmarks and rule-family discovery are research-exposed. The sealed set loses scenes/events under conservative overlap exclusions and is not automatically the prompt’s130-event population. Teacher token size differs from decoder output stride. AP uses declared4096-bin ties, EM/anchor inherited50 bins; floating-point CPU/GPU parity is documented by smoke receipts. Seed variability and whole-event uncertainty are distinct.',
        '## Failures and deferrals',json.dumps(failed+deferred,indent=2)]
    (c.ROOT/'report_v9.md').write_text('\n\n'.join(text)+'\n')
    if (c.ROOT/'prior_art.md').exists():
        with open(c.ROOT/'report_v9.md','a') as f:f.write('\n'+(c.ROOT/'prior_art.md').read_text())
    subprocess.run(['/usr/bin/python3',str(c.ROOT/'render_report_v9.py')],check=True,timeout=45)
    evidence={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.glob('*') if p.is_file() and p.suffix in ['.json','.csv','.md','.py'] and p.name not in ['evidence_index.json','scheduler_status.json','state.json']};c.save(c.ROOT/'evidence_index.json',dict(created=c.q.now(),files=evidence,immutable_parent_v8_pdf_sha256=c.sha(c.V8/'report_v8_codex_audit_20261012.pdf')))
    c.save(c.ROOT/'collection_receipt.json',dict(outcome='completed',report=str(c.ROOT/'report_v9.md'),report_sha256=c.sha(c.ROOT/'report_v9.md'),completed_at=c.q.now(),interim=not finished))

if __name__=='__main__':main()
