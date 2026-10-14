"""Sourced prompt5 tables and figures, using existing integer confusion counts."""
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/home/schmalstieg/repos/FloodDistill')
OUT=ROOT/'experiments/revision/prompt5_closure'
V4=ROOT/'experiments/revision/prompt4_report_20261007'
SNAP=Path('/data1/schmalstieg/flooddistill/revision_20261005/snapshots/95f47d14ae77a2cf')
sys.path.insert(0,str(SNAP))
from evaluation.revision_eval import scores,sha256
from evaluation.stats_scene_level import boot_ci

SOURCES={}
def js(path):
    path=Path(path);SOURCES[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size}
    return json.loads(path.read_text())
def rows(path):
    path=Path(path);SOURCES[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size}
    return list(csv.DictReader(path.open()))
def write(name,x):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def csvwrite(name,x):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def rnd(x):return str(Decimal(str(x)).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP))
def full_tables():
    spec=js(OUT/'sensitivity_spec.json'); mm={};ee={};summary=[];counts=[];byseed=[]
    metrics=('mIoU','water_IoU','background_IoU','water_precision','water_recall','water_F1')
    for r in spec['runs']:
        canonical=js(r['evaluation']);native=js(r['native']);valid=js(r['sturm_valid'])
        assert all(x['status']=='complete' for x in (canonical,native,valid))
        valid_clean=valid.get('cells',{}).get('clean',valid)
        variants=[('s11','S11_TEST512_LEGACY',canonical['cells']['s11__clean']['scene'],canonical['cells']['s11__clean']['event']),
                  ('sturm','STURM_LEGACY512',canonical['cells']['sturm__clean']['scene'],canonical['cells']['sturm__clean']['event']),
                  ('wf','WF_LEGACY512',canonical['cells']['wf__clean']['scene'],canonical['cells']['wf__clean']['event']),
                  ('sturm','P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1',valid_clean['native']['map'],valid_clean['native']['event']),
                  ('wf','P3_WF_NATIVE_ZERO_PAD512_V1',native['native']['map'],native['native']['event'])]
        for dataset,protocol,maps,events in variants:
            m={k:np.asarray(v,dtype=np.int64) for k,v in maps.items()};e={k:np.asarray(v,dtype=np.int64) for k,v in events.items()}
            total=sum(m.values());assert np.array_equal(total,sum(e.values()))
            assert all((a>=0).all() and a[0,1]==a[1,2] and a[0,2]==a[1,1] for a in m.values())
            mm[(r['run_id'],dataset,protocol)]=m;ee[(r['run_id'],dataset,protocol)]=e
            meta={k:r[k] for k in ('run_id','architecture','variant','budget','seed')}
            byseed.append({**meta,'dataset':dataset,'protocol':protocol,'domain':'full','n_maps':len(m),
                'n_events':len(e),'valid_pixels':int(total[0,0]+total[1].sum()),'water_pixels':int(total[1,0]+total[1,2]),**scores(total)})
            counts.append({**meta,'dataset':dataset,'protocol':protocol,'domain':'full','TP':int(total[1,0]),
                'FP':int(total[1,1]),'FN':int(total[1,2]),'TN':int(total[0,0])})
    groups=defaultdict(list)
    for r in byseed:groups[tuple(r[k] for k in ('architecture','variant','budget','dataset','protocol'))].append(r)
    for key,rr in sorted(groups.items()):
        assert len(rr)==(1 if key[0] in ('teacher','index:mndwi_zero') else 4)
        for metric in metrics:
            v=[r[metric] for r in rr]
            summary.append({**dict(zip(('architecture','variant','budget','dataset','protocol'),key)),
                'domain':'full','metric':metric,'n_seeds':len(v),'mean':float(np.mean(v)),
                'sample_SD_ddof1':float(np.std(v,ddof=1)) if len(v)>1 else None,
                'display_mean':rnd(np.mean(v)),'display_SD':rnd(np.std(v,ddof=1)) if len(v)>1 else '',
                'n_maps':rr[0]['n_maps'],'n_events':rr[0]['n_events'],
                'valid_pixels':rr[0]['valid_pixels'],'water_pixels':rr[0]['water_pixels']})
    csvwrite('tables/current_full_by_seed.csv',byseed);csvwrite('tables/current_full_summary.csv',summary)
    csvwrite('tables/current_full_pooled_counts.csv',counts)
    paired=[];ev=[]
    for group in sorted({(r['architecture'],r['variant'],r['budget']) for r in spec['runs'] if r['seed']!='fixed'}):
        a,v,b=group;students=[r for r in spec['runs'] if (r['architecture'],r['variant'],r['budget'])==group]
        for fixed in ('teacher','index:mndwi_zero'):
            fixedid=next(r['run_id'] for r in spec['runs'] if r['architecture']==fixed)
            for dataset,protocol in [('s11','S11_TEST512_LEGACY'),('sturm','STURM_LEGACY512'),('wf','WF_LEGACY512'),
                                     ('sturm','P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1'),('wf','P3_WF_NATIVE_ZERO_PAD512_V1')]:
                for metric in metrics:
                    d=[];events=defaultdict(list)
                    for student in students:
                        sid=student['run_id'];lc,rc=mm[(sid,dataset,protocol)],mm[(fixedid,dataset,protocol)]
                        assert set(lc)==set(rc)
                        # Match the reference pixel counts in every map before pairing.
                        for name in lc:
                            ll,rr=lc[name],rc[name]
                            assert ll[0,0]+ll[1].sum()==rr[0,0]+rr[1].sum()
                            assert ll[1,0]+ll[1,2]==rr[1,0]+rr[1,2]
                        d.append(scores(sum(lc.values()))[metric]-scores(sum(rc.values()))[metric])
                        le,re=ee[(sid,dataset,protocol)],ee[(fixedid,dataset,protocol)]
                        assert set(le)==set(re)
                        for name in le:
                            lv,rv=scores(le[name])[metric],scores(re[name])[metric]
                            events[name].append(None if lv is None or rv is None else lv-rv)
                    meta={'architecture':a,'variant':v,'budget':b,'comparator':fixed,'dataset':dataset,'protocol':protocol,'domain':'full','metric':metric}
                    paired.append({**meta,'n_seeds':4,'mean_difference':float(np.mean(d)),
                                   'sample_SD_ddof1':float(np.std(d,ddof=1))})
                    vals=[float(np.mean(v)) for v in events.values() if len(v)==4 and all(x is not None for x in v)]
                    lo,hi=boot_ci(np.asarray(vals)) if len(vals)>=5 else (None,None)
                    ev.append({**meta,'n_defined_events':len(vals),'mean_event_difference':float(np.mean(vals)),
                               'ci95_low':lo,'ci95_high':hi,'uncertainty':'Event bootstrap conditional on fixed checkpoints; separate from sample seed SD.'})
    csvwrite('tables/current_full_comparator_paired_seed.csv',paired)
    csvwrite('tables/current_full_comparator_paired_event.csv',ev)
    return summary

def rounding(summary):
    src=ROOT/'paper_source_data/Table_II_Deployment_N2500/Table_II_values.csv';assembled=rows(src)
    raw=rows(V4/'tables/original_original_score_reproduction_summary.csv');out=[]
    objective={'ce':'CE','dfkd':'Logit-KD','ofa':'OFA (ours)','ce_unet':'CE','dfkd_unet':'Logit-KD'}
    for r in raw:
        if r['metric']!='mIoU' or r['budget']!='2500':continue
        model=('UNet-ResNet50' if r['architecture']=='unet_resnet50' else 'EfficientViT-B0')+', '+objective[r['declared_objective']]
        ds={'s11':'S11','sturm':'STURM','wf':'WF-fo'}[r['dataset']]
        paper=next(x for x in assembled if (x['model'],x['benchmark'])==(model,ds))
        historical=float(r['recovered_original_mean']);current=float(r['new_legacy_mean'])
        pp=r['printed_mean'];cross=rnd(historical)!=rnd(current)
        source_discrepancy=rnd(historical)!=rnd(float(pp))
        reason=('Recovered aggregate conflicts with printed value; source/display issue.' if source_discrepancy else
                'Small evaluator drift crosses three-decimal display boundary.' if cross else 'Same three-decimal mean display.')
        if model=='EfficientViT-B0, CE' and ds=='WF-fo':reason='Raw 0.7534745 rounds to0.753; assembled four-decimal0.7535 double-rounds to0.754; current evaluator crosses boundary to0.754.'
        if model=='UNet-ResNet50, Logit-KD':reason+=' Assembled CSV differs from archived seed means; do not use it as the unrounded source.'
        out.append({'model':model,'dataset':ds,'printed_mean':pp,'printed_SD':r['printed_reported_SD'],
            'assembled_CSV_mean':paper['mean'],'assembled_CSV_SD':paper['sd_ddof1'],
            'archived_seed_mean':historical,'archived_seed_sample_SD':r['recovered_original_sample_SD'],
            'archived_display':rnd(historical),'rescored_legacy_mean':current,
            'rescored_legacy_sample_SD':r['new_legacy_sample_SD'],'revised_display_mean':rnd(current),
            'revised_display_SD':rnd(r['new_legacy_sample_SD']),
            'rescoring_delta_pp':100*(current-historical),'classification':reason,
            'named_revised_source':str(V4/'tables/original_original_score_reproduction_summary.csv'),
            'rounding':'Nearest0.001; Decimal ROUND_HALF_UP from full-precision declared mean/SD, no intermediate rounding.'})
    assert len(out)==15;csvwrite('tables/Table_II_student_reconciliation.csv',out)
    old=Path('/home/schmalstieg/repos/OLD/esi_project/runs')
    official=js(old/'teacher_eval_check/teacher_official_results.json');verify=js(old/'teacher_ft_verify_results_STURM_ORIG.json')
    printed={'s11':(.897,.822),'sturm':(.755,.643),'wf':(.771,.641)};teacher=[];traces=[]
    spec=js(OUT/'sensitivity_spec.json');run=next(r for r in spec['runs'] if r['architecture']=='teacher');canonical=js(run['evaluation'])
    for ds in printed:
        metrics=scores(sum(np.asarray(c,dtype=np.int64) for c in canonical['cells'][ds+'__clean']['scene'].values()))
        historic=verify[{'s11':'s11_test','sturm':'sturm_s2','wf':'wf'}[ds]]
        teacher.append({'dataset':ds,'protocol':'legacy512','printed_mIoU':printed[ds][0],
            'printed_water_IoU':printed[ds][1],'historical_verification_mIoU':historic['mIoU'],
            'historical_verification_water_IoU':historic['iou_per_class'][1],
            'verified_current_mIoU':metrics['mIoU'],'verified_current_water_IoU':metrics['water_IoU'],
            'current_minus_printed_mIoU_pp':100*(metrics['mIoU']-printed[ds][0]),
            'revised_display_mIoU':rnd(metrics['mIoU']),'n_checkpoint_instances':1,'SD':'not applicable',
            'printed_source_status':('Historical S11 printed source untraced; not explained by rounding.' if ds=='s11' else
                'Historical per-map cache supports printed rounded value; its checkpoint/runtime ancestry is not verified.'),
            'verified_count_source':run['evaluation']})
        if ds!='s11':
            cached=js(old/f'perscene/{ds}__teacher__s-__clean.json');current=canonical['cells'][ds+'__clean']['scene']
            assert set(cached['scene'])==set(current)
            delta=max(int(np.abs(np.asarray(cached['scene'][name])-np.asarray(current[name])).max()) for name in current)
            ct=sum(np.asarray(v,dtype=np.int64) for v in cached['scene'].values())
            nt=sum(np.asarray(v,dtype=np.int64) for v in current.values())
            traces.append({'dataset':ds,'same_map_IDs':True,'max_absolute_count_difference':delta,
                           'old_count_source':str(old/f'perscene/{ds}__teacher__s-__clean.json'),
                           'old_pooled_metrics':scores(ct),'current_pooled_metrics':scores(nt),
                           'same_valid_pixel_total':int(ct[0,0]+ct[1].sum())==int(nt[0,0]+nt[1].sum()),
                           'same_reference_water_total':int(ct[1,0]+ct[1,2])==int(nt[1,0]+nt[1,2]),
                           'old_cache_mIoU_rounds_to_printed':rnd(scores(ct)['mIoU'])==rnd(printed[ds][0]),
                           'old_cache_water_IoU_rounds_to_printed':rnd(scores(ct)['water_IoU'])==rnd(printed[ds][1]),
                           'cache_checkpoint_runtime_identity':'Absent in old per-scene cache; filename is not a checkpoint receipt; do not combine with the separately retained scalar verification.'})
    csvwrite('tables/teacher_baseline_reconciliation.csv',teacher)
    old_eval=old.parent/'evaluation/perscene_eval.py'
    old_code=old_eval.read_text();SOURCES[str(old_eval)]={'sha256':sha256(old_eval),'bytes':old_eval.stat().st_size}
    assert '"runs/teacher_ft/best.pt"' in old_code
    write('teacher_trace.json',{'status':'verified_baseline_reused_printed_external_values_numerically_traced_ancestry_incomplete',
        'legacy_teacher_counts':run['evaluation'],'identity':canonical['identity'],
        'historical_official_S11_test':official['test_mIoU'],'historical_official_S11_validation':official['val_mIoU'],
        'historical_remapped_verification':verify,'historical_count_checks':traces,
        'model_input_protocol':'Pinned public300M task teacher, strict adapted load; six S11 bands/normalization, fixed class masks, pixel-pooled counts; STURM paper512, WF legacy partial-window resizing.',
        'printed_provenance':'Retained old per-map caches numerically support printed STURM0.755/WF0.771 and waterIoU0.643/0.641. S11 printed0.897/0.822 remains untraced. Separate official/remapped scalar verification agrees with the current verified baseline.',
        'old_evaluator_candidate':{'path':str(old_eval),'sha256':sha256(old_eval),
            'family_teacher_checkpoint':'runs/teacher_ft/best.pt','checkpoint_present':(old/'teacher_ft/best.pt').exists(),
            'loader':'build_teacher(pretrained=False), then torch.load/checkpoint model strict=True',
            'limit':'Code records this candidate path; caches lack immutable executed checkpoint/hash/config receipts, so code alone does not prove their generating ancestry.'},
        'cause':'Same valid/reference-water totals but different predictions in retained caches. Different checkpoint route is a candidate supported by old code, not proven executed ancestry; no hardware cause inferred.',
        'student_max_0_0192pp_scope':'28 declared original student checkpoint comparisons only',
        'revision_policy':'Preserve printed external numbers as count-supported historical references with incomplete checkpoint ancestry, S11 as untraced; use verified protocol-matched baseline in revised numerical comparisons.',
        'extra_teacher_inference':'Only3 missing exclusion chip counts in the finite sensitivity; no further teacher inference justified.'})

def architecture():
    seed=rows(V4/'tables/original_paired_seed_summary.csv');event=rows(V4/'tables/original_paired_event_summary.csv');out=[]
    conditions={'s11':'clean','sturm':'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1:clean','wf':'P3_WF_NATIVE_ZERO_PAD512_V1'}
    for r in seed:
        if r['contrast'] not in ('N100_OFA_minus_clipping_matched_KD','OFA_minus_clipping_matched_KD') or r['metric']!='mIoU' or r['condition']!=conditions[r['dataset']]:continue
        e=next(e for e in event if all(e[k]==r[k] for k in ('contrast','architecture','budget','dataset','condition','metric')))
        out.append({'architecture':r['architecture'],'budget':r['budget'],'dataset':r['dataset'],'protocol':r['condition'],
            'OFA_minus_KD_pooled_mIoU_pp':100*float(r['mean']),'paired_seed_SD_pp':100*float(r['sample_SD_ddof1']),
            'n_seeds':4,'event_macro_difference_pp':100*float(e['mean_event_metric_difference']),
            'event_CI95_low_pp':100*float(e['ci95_low']),'event_CI95_high_pp':100*float(e['ci95_high']),
            'n_events':e['n_events_with_four_seed_pairs'],'domain':'full',
            'seed_source':str(V4/'tables/original_paired_seed_summary.csv'),'event_source':str(V4/'tables/original_paired_event_summary.csv')})
    assert len(out)==12;csvwrite('tables/OFA_matched_effects.csv',out)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    names=[]
    for r in out:names.append(r['architecture'].replace('efficientvit_b0','EfficientViT').replace('mobilenet_v3_small','MobileNetV3-Small').replace('lightformer_resnet18','LightFormer')+f" N{r['budget']} / {r['dataset']}")
    for ax,kind in zip(axes,['seed','event']):
        for i,r in enumerate(out):
            value=r['OFA_minus_KD_pooled_mIoU_pp'] if kind=='seed' else r['event_macro_difference_pp']
            error=r['paired_seed_SD_pp'] if kind=='seed' else [[value-r['event_CI95_low_pp']],[r['event_CI95_high_pp']-value]]
            ax.errorbar(value,i,xerr=error,fmt='o',color={'s11':'#276FBF','sturm':'#986626','wf':'#27734D'}[r['dataset']],capsize=3)
        ax.axvline(0,color='grey',lw=.8);ax.set_xlabel('OFA minus matched KD (mIoU percentage points)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(out)),names,fontsize=8);axes[0].invert_yaxis()
    axes[0].set_title('Pooled pixel effect ± paired seed SD (4 seeds)')
    axes[1].set_title('Event macro effect and conditional 95% bootstrap CI')
    fig.suptitle('Original restored cohort; full benchmarks, corrected STURM / native WF',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'OFA_matched_effects.pdf');fig.savefig(OUT/'OFA_matched_effects.svg');plt.close(fig)

def historical_membership():
    p=Path('/home/schmalstieg/repos/OLD/FloodDistill_v3/data/index_files_curations/diversity_N252_index.json')
    x=js(p);a=js(OUT/'overlap_scored_support_assessment.json')
    configs=sorted(Path('/home/schmalstieg/repos/OLD/esi_project/wandb').glob('run-20260825_*/files/config.yaml'))
    evidence=[]
    for path in configs:
        text=path.read_text()
        if 'pseudo_' in text and '_N252_s' in text:
            lines=text.splitlines();names=[line.strip() for line in lines if 'mapped_diversity_N252_index.json' in line]
            if names:
                SOURCES[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size}
                evidence.append({'path':str(path),'index_in_executed_arguments':'mapped_diversity_N252_index.json',
                    'scope':'Retained runs/repro_full execution config; some use pseudo_gdacs_v2_regen. Immutable link to the TableI generating checkpoints not established.'})
    write('historical_N252_membership.json',{'status':'specific_membership_chain_incomplete','available_index':str(p),
        'available_index_sha256':sha256(p),'n_entries':len(x['files']),
        'intersecting_current_sources_in_available_index':sorted({r['training_file'] for r in a['records'] if r['training_file'] in x['files']}),
        'executed_argument_records':evidence,
        'missing':'Mapped index and source mapping/hash receipts, plus an immutable link from the retained execution configs to TableI generating checkpoints, were not recovered in the identified project data locations.',
        'decision':'Do not substitute current FPS first252 or available unmapped IDs for proven executed membership; retain TableI as historical and qualify its independence/statistics. No new TableI training or sensitivity inference.'})

def main():
    summary=full_tables();rounding(summary);architecture();historical_membership()
    wf=rows(OUT/'sensitivity_wf_complete/by_seed.csv');held=next(r for r in wf if r['domain']=='heldout' and r['protocol']=='P3_WF_NATIVE_ZERO_PAD512_V1')
    full=next(r for r in wf if r['domain']=='full' and r['protocol']=='P3_WF_NATIVE_ZERO_PAD512_V1')
    write('numerical_provenance.json',{'status':'complete_full_tables_and_WF_sensitivity_STURM_exclusion_pending',
        'sources':SOURCES,'rounding':'ROUND_HALF_UP to0.001 independently on full-precision mean and sampleSD; fixed comparators have no seedSD.',
        'full_current_comparators':38,'full_protocols':5,'n_current_full_seed_rows':190,
        'WF_exclusion':{'maps':[int(full['n_maps']),int(held['n_maps'])],'events':[int(full['n_events']),int(held['n_events'])],
            'removed_valid_pixels':int(full['valid_pixels'])-int(held['valid_pixels']),
            'removed_water_pixels':int(full['water_pixels'])-int(held['water_pixels'])},
        'distinctions':'Original printed references; archived student seed metrics; original strict checkpoint rescoring; current recovered recipe; clipping-matched controls; protocol corrections; separate regenerated cohort; historical device-only claims.'})
    print(json.dumps({'full_tables':'complete','OFA_effect_rows':12,'student_reconciliation_rows':15,
                      'teacher_reconciled_separately':True,'WF_sensitivity':'complete','STURM_sensitivity':'awaiting fixed-chip receipts'}))

if __name__=='__main__':main()
