"""Validate and package prompt5's bounded evidence/drafts, with live partial status."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import json
from pathlib import Path
import sys
import textwrap
import zipfile
import numpy as np

ROOT=Path('/home/schmalstieg/repos/FloodDistill')
OUT=ROOT/'experiments/revision/prompt5_closure'
SNAP=Path('/data1/schmalstieg/flooddistill/revision_20261005/snapshots/95f47d14ae77a2cf')
sys.path.insert(0,str(SNAP))
from evaluation.revision_eval import scores,sha256
def js(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def csvwrite(p,records):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def main():
    closure=rows(OUT/'reviewer_closure_table.csv')
    required={f'R1.{i}' for i in range(1,7)}|{f'R2.{i}' for i in range(1,6)}|{'R2 final','Editor'}
    assert {r['reviewer_ID'] for r in closure}==required and len(closure)==13
    current=rows(OUT/'tables/current_full_by_seed.csv')
    expected=rows(ROOT/'experiments/revision/prompt4_report_20261007/tables/original_matched_by_seed.csv')
    print('existing-row schema',list(expected[0]))
    metric_names=('mIoU','water_IoU','background_IoU','water_precision','water_recall','water_F1')
    # Compare newly collected full-domain cells with all existing compatible V4 rows.
    lookup={(r['run_id'],r['dataset'],r['condition'],r['metric']):float(r['value']) for r in expected}
    maxdiff=0.;checked=0
    for r in current:
        condition={'S11_TEST512_LEGACY':'clean','STURM_LEGACY512':'clean','WF_LEGACY512':'clean',
                   'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1':'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1:clean',
                   'P3_WF_NATIVE_ZERO_PAD512_V1':'P3_WF_NATIVE_ZERO_PAD512_V1'}[r['protocol']]
        for metric in metric_names:
            key=(r['run_id'],r['dataset'],condition,metric)
            assert key in lookup,key
            delta=abs(float(r[metric])-lookup[key]);maxdiff=max(maxdiff,delta);checked+=1
    assert checked==1140 and maxdiff<1e-12
    wf=rows(OUT/'sensitivity_wf_complete/by_seed.csv')
    groups=defaultdict(list)
    for r in wf:groups[(r['dataset'],r['protocol'],r['condition'],r['domain'])].append(r)
    for group,rr in groups.items():
        assert len({(r['valid_pixels'],r['water_pixels'],r['n_maps'],r['n_events']) for r in rr})==1,group
    clean={(r['run_id'],r['protocol'],r['domain']):r for r in wf if r['condition']=='clean'}
    drops=[]
    for r in wf:
        base=clean[(r['run_id'],r['protocol'],r['domain'])]
        for metric in metric_names:
            a,b=float(base[metric]),float(r[metric])
            drops.append({**{k:r[k] for k in ('run_id','architecture','variant','budget','seed','dataset','protocol','condition','domain')},
                'metric':metric,'clean':a,'condition_value':b,'absolute_drop_pp':100*(a-b),
                'relative_drop_percent':100*(a-b)/a if a else None})
    csvwrite(OUT/'tables/WF_sensitivity_all_conditions_and_drops.csv',drops)
    gate=js(ROOT/'experiments/revision/prompt3_evaluation_gate.json');assessment=js(OUT/'overlap_scored_support_assessment.json')
    excluded=js(OUT/'overlap_exclusion_plan.json')
    windows=[w for w in gate['worldfloods']['windows'] if w['map'] in excluded['exclude_complete_WorldFloods_maps']]
    bbox_union=set()
    for r in assessment['records']:
        if r['dataset']!='WorldFloods':continue
        c0,r0,width,height=r['window_bbox']; c1,r1=c0+width,r0+height
        for w in windows:
            c,rr,ww,hh=w['window']
            if max(c,c0)<min(c+ww,c1) and max(rr,r0)<min(rr+hh,r1):bbox_union.add(w['filename'])
    refs=js(ROOT/'experiments/revision/prompt2_reference_manifest.json')['STURM']
    retained=[r for name,r in refs.items() if name not in excluded['exclude_complete_STURM_chips']]
    units={'STURM':{'full_chips':len(refs),'retained_chips':len(retained),
        'full_maps':len({r['map_id'] for r in refs.values()}),'retained_maps':len({r['map_id'] for r in retained}),
        'full_events':len({r['event'] for r in refs.values()}),'retained_events':len({r['event'] for r in retained}),
        'excluded_chips':excluded['exclude_complete_STURM_chips']},
        'WF':{'full_windows':len(gate['worldfloods']['windows']),
            'removed_complete_map_windows':len(windows),'retained_windows':len(gate['worldfloods']['windows'])-len(windows),
            'removed_map_valid_pixels':sum(w['valid_native'] for w in windows),
            'removed_map_water_pixels':sum(w['water_native'] for w in windows),
            'source_bbox_intersecting_retained_windows':sorted(bbox_union),
            'bbox_count_limit':'5 windows intersect patch bounding boxes; scored overlap established for the source pairs as a whole. Individual scored-window counts not inferred solely from these bounding boxes.',
            'excluded_maps':excluded['exclude_complete_WorldFloods_maps']}}
    assert units['WF']['removed_map_valid_pixels']==6336397 and units['WF']['removed_map_water_pixels']==223618
    dump(OUT/'overlap_unit_counts.json',units)
    spec=js(OUT/'sensitivity_spec.json'); completed=[]
    for r in spec['runs']:
        path=Path(r['subset_counts'])
        if not path.exists():continue
        receipt=js(path);assert receipt['status']=='complete'
        assert receipt['run_id']==r['run_id'] and receipt['exclusion_plan_sha256']==sha256(OUT/'overlap_exclusion_plan.json')
        assert receipt['evaluation_sha256']==sha256(r['evaluation'])
        assert set(receipt['cells'])==set(excluded['exclude_complete_STURM_chips'])
        completed.append(r['run_id'])
    complete_provenance=OUT/'sensitivity_complete/provenance.json'
    sensitivity_done=complete_provenance.exists() and js(complete_provenance)['status']=='complete'
    sensitivity_metric_cells=0
    if sensitivity_done:
        assert len(completed)==38
        provenance=js(complete_provenance)
        assert provenance['n_runs']==38 and provenance['mode']=='all'
        assert provenance['spec_sha256']==sha256(OUT/'sensitivity_spec.json')
        sr=rows(OUT/'sensitivity_complete/by_seed.csv')
        pooled=[r for r in rows(OUT/'sensitivity_complete/confusion_counts.csv') if r['level']=='pooled']
        lookup_counts={(r['run_id'],r['dataset'],r['protocol'],r['condition'],r['domain']):r for r in pooled}
        for r in sr:
            key=tuple(r[k] for k in ('run_id','dataset','protocol','condition','domain'))
            c=lookup_counts[key];tp,fp,fn,tn=[int(c[k]) for k in ('TP','FP','FN','TN')]
            recomputed=scores(np.asarray([[tn,fn,fp],[tp,fp,fn]],dtype=np.int64))
            for metric in metric_names:
                assert abs(recomputed[metric]-float(r[metric]))<1e-12;sensitivity_metric_cells+=1
        domain_groups=defaultdict(list)
        for r in sr:domain_groups[(r['dataset'],r['protocol'],r['condition'],r['domain'])].append(r)
        for key,rr in domain_groups.items():
            assert len({(r['valid_pixels'],r['water_pixels'],r['n_maps'],r['n_events']) for r in rr})==1,key
        # Compact paired effect table with seed and event uncertainty from one named collector.
        ps=rows(OUT/'sensitivity_complete/paired_seed_summary.csv');pe=rows(OUT/'sensitivity_complete/paired_event_summary.csv');effects=[]
        for r in ps:
            if r['contrast']!='OFA_minus_matched_KD' or r['metric']!='mIoU' or r['protocol']=='legacy512':continue
            e=next(e for e in pe if all(e[k]==r[k] for k in ('contrast','architecture','budget','dataset','protocol','condition','domain','metric')))
            effects.append({**{k:r[k] for k in ('architecture','budget','dataset','protocol','domain')},
                'pooled_effect_pp':100*float(r['mean_difference']),'paired_seed_SD_pp':100*float(r['sample_SD_ddof1']),
                'event_macro_effect_pp':100*float(e['mean_event_difference']),
                'event_CI_low_pp':100*float(e['ci95_low']),'event_CI_high_pp':100*float(e['ci95_high']),
                'n_events':e['n_events'],'source':'sensitivity_complete/paired_seed_summary.csv + paired_event_summary.csv'})
        assert len(effects)==16;csvwrite(OUT/'tables/overlap_OFA_effects_full_and_excluded.csv',effects)
        numerical=js(OUT/'numerical_provenance.json')
        numerical.update(status='complete_declared_prompt5_numerical_scope',STURM_exclusion_source=str(complete_provenance),
            STURM_exclusion_provenance_sha256=sha256(complete_provenance))
        dump(OUT/'numerical_provenance.json',numerical)
    state=js(OUT/'queue/manifest.state.json')
    terminal_receipts=0;outputs_verified=0
    if sensitivity_done:
        assert state['worker_status']=='finished' and len(state['jobs'])==39
        for jid,record in state['jobs'].items():
            assert record['status']=='succeeded' and len(record['attempts'])==1,jid
            attempt=record['attempts'][0];receipt=js(attempt['receipt_path'])
            assert receipt['exit_code']==0 and receipt['job_identity_sha256']==record['identity_sha256'],jid
            for output in receipt['output_validation']:
                assert sha256(output['path'])==output['sha256'],output['path']
                assert Path(output['path']).stat().st_size==output['bytes'];outputs_verified+=1
            terminal_receipts+=1
        dump(OUT/'completed_count_workload_receipt.json',{'status':'complete','manifest':str(OUT/'queue/manifest.json'),
            'manifest_sha256':sha256(OUT/'queue/manifest.json'),'state_sha256':sha256(OUT/'queue/manifest.state.json'),
            'all_nodes_succeeded':terminal_receipts,'fixed_chip_inference_jobs':38,'collectors':1,
            'worker_status':'finished','worker_stopped_at':state.get('stopped_at'),
            'new_training':0,'new_QAT':0,'new_corruption_inference':0,'no_remaining_runnable_GPU_work':True})
    sources=js(OUT/'numerical_provenance.json')['sources']
    for p,receipt in sources.items():assert sha256(p)==receipt['sha256'],p
    report={'status':'passed_available_scope','captured_utc':datetime.now(timezone.utc).isoformat(),
        'reviewer_rows':13,'current_full_seed_rows':len(current),'metric_cells_compared_to_V4':checked,
        'max_absolute_difference':maxdiff,'WF_count_domains_verified':len(groups),
        'WF_degradation_rows':len(drops),'source_hashes_checked':len(sources),
        'completed_fixed_chip_receipts_verified':len(completed),'required_fixed_chip_receipts':38,
        'STURM_exclusion_scope':'complete' if sensitivity_done else 'pending',
        'sensitivity_metric_cells_recomputed_from_integer_counts':sensitivity_metric_cells,
        'terminal_receipts_verified':terminal_receipts,'terminal_outputs_hash_and_size_verified':outputs_verified,
        'original_manuscripts_preserved':{
            str(ROOT.parent/'OLD/FloodDistill_v3/docs/main.pdf'):sha256(ROOT.parent/'OLD/FloodDistill_v3/docs/main.pdf'),
            str(ROOT.parent/'OLD/FloodDistill_v3/docs/supplement.pdf'):sha256(ROOT.parent/'OLD/FloodDistill_v3/docs/supplement.pdf')}}
    assert list(report['original_manuscripts_preserved'].values())==['4afa035c3ea9c30566834ec1d27cbbb1ebdfe835e2865a8f3e07827d51cd4575','1494e669f6e0edd95f0d07f53aff1fdcd2d802a8100b54ca999d61e594cc518e']
    dump(OUT/'validation.json',report)
    status=js(OUT/'closure_status.json')
    lines=['FloodDistill prompt5 — Reviewer closure evidence and proposed revision',
        'Captured: '+report['captured_utc'],
        'Draft package prepared. Submitted manuscripts are unchanged. No new training or QAT.',
        'Completed: scored-footprint assessment,38-comparator full-domain tables, WF sensitivity, numerical reconciliation,13response drafts.',
        'Pending: STURM3-chip exclusion count recovery/collector; author history; exact reviewer letter; device ancestry for retained unqualified numbers.',
        'Queue status is a capture, not a promise that these counts remain current: '+str(status['queue_counts']),
        'Worker: '+status['worker_status']+'; heartbeat '+str(status['last_worker_heartbeat']),
        'The existing detached scheduler checks admission every5seconds; foreign compute processes are never used or stopped.',
        'The worker runs independently of the editor parent on an awake host, with existing global/per-UUIDlocks and no automatic retries.',
        'Original V4 finished records remain intact:44newtraining succeeded;611successful,30externalblocked,1historicalfailedcollector node.',
        'No eight extra regeneratedN100nativeWF evaluations were added. Usable common-protocol collection remains separate.',
        '', 'Evidence and status distinctions',
        'Actual reviewer letter was not found; numbered request summaries follow prompt5 and are not claimed as verbatim quotations.',
        'Only submitted PDFs were found: exact replacement passages have page/section/table anchors. All edits are proposed, not incorporated.',
        'Known different-date overlap is input/geographic exposure, not proven evaluation-ground-truth supervision. Excluding it does not prove an untouched benchmark.',
        'WF excludes1fullmap:18→17maps,11events unchanged; removed6,336,397valid and223,618waterpixels.',
        f"STURM metadata:2675→2672chips,{units['STURM']['full_maps']}→{units['STURM']['retained_maps']}maps,{units['STURM']['full_events']}→{units['STURM']['retained_events']}events; scored removal786,432valid/80,512waterpixels. Numerical sensitivity awaits38receipts.",
        'WF OFA N2500full/heldout mIoU77.128/76.998%; teacher77.171/77.027%. OFA-minusKD full/heldout+0.438/+0.440pp.',
        'N100WF+0.697/+0.704pp; MBV3+0.353/+0.359; LF-0.887/-0.902. No universalOFA gain follows.',
        'WF legacyh0.20 full/heldout OFA74.759/74.744 versusMNDWI25.673/26.176%; retainedt1tilt-30 MNDWI75.54 versusOFA53.91%.',
        'Native clean and legacy robustness are different geometry domains. STURM exclusion robustness is unavailable; no new corruption sweep is run.',
        'Student TableII source/display crossings are separate from teacher baseline provenance. HistoricalST/WFteacher caches numerically support the printedrow but lack checkpoint/runtime ancestry; S11printedsource remains untraced.',
        'Existing official/remapped teacher verification supports the current verified baseline; no hardware cause is invented.',
        'All four seeds retained. N2500selectedsteps936–7800/completed7176–7800; N100selected168–300/completed300. Same policy does not imply equal actual updates.',
        'Dry clarification addresses the reviewer limit once incorporated; independent dry reliability remains unmeasured. Deployment captions preserve mixed precision/FP16faster and qualify the retained historical values.',
        '', 'Files to review',
        'reviewer_closure_table.csv — all13comment scopes, anchors, evidence, limitations, action and status.',
        'main_manuscript_proposed_edits.txt — M1–M16replacement passages, including proposedDiscussion.',
        'supplement_proposed_edits.txt — S1–S8anchored edits and numerical reconciliation.',
        'point_by_point_response_draft.txt — all13proposed responses; no submission or incorporation claimed.',
        'data_roles.csv / ablation_map.csv / remaining_inputs.txt — roles, fixed/changed comparisons and factual input list.',
        'tables/ — full count-derived sixmetrics, seed/event comparators,12OFAeffects,15studentroundingrows and3teacherlegacyrows.',
        'sensitivity_wf_complete/ — clean native and all compatible saved legacy perturbation counts/metrics/full/heldout pairing.',
        'sensitivity_complete/ — dependent output once all38STchip receipts pass, not an existing complete result until its provenance sayscomplete.',
        'OFA_matched_effects.pdf/.svg — full benchmarks, paired seedSD versus conditional eventCI.',
        'queue/manifest.json / manifest.state.json / logs and receipts — finite39node workload and automaticcollector.',
        'validation.json / numerical_provenance.json / package_source_index.json — verification and hashes.',
        '', 'Proposed current table (full benchmarks; matched/compatible controls, float checkpoints)',
        'Student/budget: S11mIoU; correctedSTURM; nativeWF. Mean±sampleSD over4seeds; fixed comparators noSD.']
    if sensitivity_done:
        lines=[line.replace('Pending: STURM3-chip exclusion count recovery/collector; author history;',
                           'Remaining factual inputs: author history;')
               .replace('Numerical sensitivity awaits38receipts.','Numerical sensitivity is complete from all38validated chip receipts.')
               .replace('dependent output once all38STchip receipts pass, not an existing complete result until its provenance sayscomplete.',
                        'complete nativeWF/correctedSTURM full/heldout counts, metrics, pairedseed and event summaries.') for line in lines]
        lines.insert(4,'Both common-exclusion sensitivities completed. All39queue nodes succeeded; worker finished with no failures or remaining runnableGPUwork.')
        lines.insert(20,'STURM full/heldout OFA76.026/76.031%, teacher75.392/75.395%; OFA-minusKD-0.023/-0.021pp. N100 +0.116/+0.117pp with retainedeventCI spanningzero.')
    for r in rows(OUT/'tables/main_table_current_display.csv'):
        lines.append(f"{r['architecture']} N{r['budget']} {r['objective_variant']}: {r['s11_mIoU']}; {r['sturm_mIoU']}; {r['wf_mIoU']}")
    lines+=['','OFA-minusKD full-domain effects (percentage points): pooled mean ± paired seedSD; event macro mean[95%CI]']
    for r in rows(OUT/'tables/OFA_matched_effects.csv'):
        lines.append(f"{r['architecture']} N{r['budget']} {r['dataset']}: {float(r['OFA_minus_KD_pooled_mIoU_pp']):+.3f} ± {float(r['paired_seed_SD_pp']):.3f}; event {float(r['event_macro_difference_pp']):+.3f} [{float(r['event_CI95_low_pp']):+.3f},{float(r['event_CI95_high_pp']):+.3f}]")
    lines+=['','Reviewer closure table']
    for r in closure:lines.append(r['reviewer_ID']+' — '+r['request']+' | '+r['status']+' | proposed:'+r['draft_anchor'])
    # Use the existing report style; retain separate editable-text drafts as the source.
    from importlib.util import spec_from_file_location,module_from_spec
    sp=spec_from_file_location('prompt5_draft_helper',ROOT/'scripts/revision_prompt5_drafts.py');mod=module_from_spec(sp);sp.loader.exec_module(mod)
    lines=[mod.readable(line) for line in lines]
    for name,label in [('data_roles.csv','Data role table'),('ablation_map.csv','Ablation map')]:
        lines+=['',label]
        for r in rows(OUT/name):lines.append(mod.readable(' | '.join(k+': '+str(v) for k,v in r.items())))
    for n in ('point_by_point_response_draft.txt','main_manuscript_proposed_edits.txt','supplement_proposed_edits.txt','remaining_inputs.txt'):
        lines+=['','Source document: '+n,'']+(OUT/n).read_text().splitlines()
    (OUT/'prompt5_report.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    wrapped=[]
    for line in lines:wrapped.extend(textwrap.wrap(line,width=100,break_long_words=True,break_on_hyphens=False) or [''])
    with PdfPages(OUT/'prompt5_report.pdf',metadata={'Title':'FloodDistill prompt5 reviewer closure draft package',
              'Subject':'Proposed edits and bounded sensitivity; status explicitly partial where applicable'}) as pdf:
        pages=0
        for start in range(0,len(wrapped),63):
            pages+=1;fig=plt.figure(figsize=(8.27,11.69))
            fig.text(.06,.973,'FloodDistill | prompt5 reviewer closure drafts',fontsize=10,color='#28516a')
            fig.text(.06,.946,'\n'.join(wrapped[start:start+63]),fontsize=8,family='DejaVu Sans Mono',va='top',linespacing=1.3)
            fig.text(.94,.025,str(pages),fontsize=8,ha='right');pdf.savefig(fig);plt.close(fig)
    index=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and not '/queue/' in str(path) and path.name!='package_source_index.json':
            index.append({'path':str(path.relative_to(OUT)),'sha256':sha256(path),'bytes':path.stat().st_size})
    dump(OUT/'package_source_index.json',{'captured_utc':report['captured_utc'],'files':index,
        'source_inputs':sources,'note':'Queue state/receipts continue changing; this is a draft/report capture. Original report source hashes preserved.'})
    archive=OUT.parent/'prompt5_closure_review_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and '/__pycache__/' not in str(path):z.write(path,'prompt5_closure/'+str(path.relative_to(OUT)))
    print(json.dumps({'validation':report,'pdf_pages':pages,'zip':str(archive)}))

if __name__=='__main__':main()
