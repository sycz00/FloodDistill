"""Fixed-chip inference and count-only held-out collection for prompt5."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

SNAPSHOT=Path('/data1/schmalstieg/flooddistill/revision_20261005/snapshots/95f47d14ae77a2cf')
sys.path.insert(0,str(SNAPSHOT))
from evaluation.revision_eval import scores, atomic_json, sha256

def load(p): return json.loads(Path(p).read_text())
def csv_out(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def infer(spec_path,out):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from data.ood_datasets import STURMFloodDataset
    from data.channel_adapter import sturm_s2_adapter
    import evaluation.perscene_eval as pe
    from evaluation.revision_native_border import confusion, sturm_invalid
    spec=load(spec_path); plan=load(spec['exclusion_plan'])
    assert sha256(spec['exclusion_plan'])==spec['exclusion_plan_sha256']
    assert torch.cuda.is_available(), 'CUDA required; no CPU substitution'
    dev=torch.device('cuda:0')
    original=load(spec['evaluation'])
    assert original['status']=='complete'
    assert original['identity']['runtime_packages']==spec['runtime_packages']
    assert {name:importlib.metadata.version(name) for name in spec['runtime_packages']}==spec['runtime_packages']
    ck=spec.get('checkpoint')
    if ck: assert sha256(ck)==original['identity']['checkpoint_sha256']
    ds=STURMFloodDataset(spec['sturm_root'],sensor='S2',img_size=512,
                        normalize='original',class_mapping='paper')
    names=[p[0].name for p in ds.samples]
    assert len(ds)==2675
    excluded=plan['exclude_complete_STURM_chips']
    selected=[names.index(fn) for fn in excluded]
    # Preserve the original STURM batch8 contexts, including unscored neighbours.
    contexts=sorted({j for i in selected for j in range(i//8*8,min(i//8*8+8,len(ds)))})
    assert contexts==spec['batch_context_indices']
    loader=DataLoader(Subset(ds,contexts),batch_size=8,shuffle=False,num_workers=0)
    adapter=sturm_s2_adapter(target_channels=6).to(dev).eval()
    model=pe.load(spec['arch'],ck or '',dev).eval()
    out=Path(out)
    assert not out.exists(), 'Existing terminal result cannot be overwritten'
    started=time.time(); cells={}
    with torch.inference_mode():
        for b in loader:
            logits=model(adapter(b['image'].to(dev)))
            logits=logits['out'] if isinstance(logits,dict) else logits[0] if isinstance(logits,(list,tuple)) else logits
            assert logits.shape[-2:]==(512,512) and torch.isfinite(logits).all()
            prediction=logits.argmax(1).cpu().numpy()
            for i,fn in enumerate(b['filename']):
                if fn not in excluded: continue
                index=names.index(fn); _,invalid=sturm_invalid(ds,index)
                mask=b['mask'][i].numpy().copy(); mask[invalid]=-1
                cell=confusion(prediction[i],mask)
                expected=spec['expected_chip_counts'][fn]
                assert int(cell[0,0]+cell[1].sum())==expected['valid_pixels']
                assert int(cell[1,0]+cell[1,2])==expected['water_pixels']
                cells[fn]={'confusion':cell.tolist(),'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
                           'valid_pixels':expected['valid_pixels'],'water_pixels':expected['water_pixels']}
                assert cells[fn]['mask_sha256']==expected['mask_sha256']
    assert set(cells)==set(excluded)
    atomic_json(out,{'status':'complete','run_id':spec['run_id'],'spec_sha256':sha256(spec_path),
        'exclusion_plan_sha256':spec['exclusion_plan_sha256'],
        'checkpoint_sha256':original['identity']['checkpoint_sha256'],
        'evaluation_sha256':sha256(spec['evaluation']),'runtime_packages':spec['runtime_packages'],
        'reader_snapshot':str(SNAPSHOT),'batch_size':8,'batch_context_indices':contexts,
        'scored_chips':3,'forwarded_context_chips':len(contexts),'device':'cuda:0',
        'precision':'float32, AMP disabled, inherited evaluator defaults',
        'tf32':{'matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn':torch.backends.cudnn.allow_tf32},
        'cells':cells,'started_unix':started,'completed_unix':time.time()})

def selected_runs(comparison):
    fixed={'teacher','index:mndwi_zero'}
    primary={'matched_focal_g2_clip5','matched_lkd_clip5_valid','historical_ofa_clip5_allvalid'}
    return [r for r in comparison['runs'] if r['architecture'] in fixed or
            r['run_id'].startswith('O4_train_') and (
            r['architecture']=='efficientvit_b0' and (r['budget']==2500 and r['variant'] in primary or
                r['budget']==100 and r['variant'] in {'matched_lkd_clip5_valid','matched_ofa_clip5_valid'}) or
            r['architecture'] in {'mobilenet_v3_small','lightformer_resnet18'} and r['budget']==2500)]

def collect(spec_path,out,mode):
    import numpy as np
    from evaluation.stats_scene_level import boot_ci
    spec=load(spec_path); plan=load(spec['exclusion_plan'])
    assert sha256(spec['exclusion_plan'])==spec['exclusion_plan_sha256']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metric_rows=[]; count_rows=[]; matrices={}; unavailable=[]
    map_events=spec['STURM_map_events']
    for item in spec['runs']:
        meta={k:item[k] for k in ('run_id','architecture','variant','budget','seed')}
        canonical=load(item['evaluation'])
        assert canonical['status']=='complete'
        if mode in ('wf','all'):
            native=load(item['native']); assert native['status']=='complete'
            variants=[('wf','P3_WF_NATIVE_ZERO_PAD512_V1','clean',native['native']['map'])]
            # Reuse available primary map counts on the same exclusions; no new sweep.
            for key,cell in canonical['cells'].items():
                if key.startswith('wf__'):
                    variants.append(('wf','legacy512',key.split('__',1)[1],cell['scene']))
        else: variants=[]
        if mode in ('sturm','all'):
            corrected=load(item['sturm_valid']); assert corrected['status']=='complete'
            complete=corrected.get('cells',{}).get('clean',corrected)['native']['map']
            full={k:np.asarray(v,dtype=np.int64) for k,v in complete.items()}
            receipt=load(item['subset_counts']); assert receipt['status']=='complete'
            assert receipt['run_id']==item['run_id'] and receipt['exclusion_plan_sha256']==spec['exclusion_plan_sha256']
            assert receipt['checkpoint_sha256']==canonical['identity']['checkpoint_sha256']
            assert receipt['evaluation_sha256']==sha256(item['evaluation'])
            assert set(receipt['cells'])==set(plan['exclude_complete_STURM_chips'])
            retained={k:v.copy() for k,v in full.items()}
            for fn,c in receipt['cells'].items():
                retained[spec['STURM_chip_maps'][fn]]-=np.asarray(c['confusion'],dtype=np.int64)
            assert all((v>=0).all() for v in retained.values())
            removed=sum(full.values())-sum(retained.values())
            assert int(removed[0,0]+removed[1].sum())==spec['STURM_removed_valid_pixels']
            assert int(removed[1,0]+removed[1,2])==spec['STURM_removed_water_pixels']
            variants.append(('sturm','P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1','clean',full))
            matrices[(item['run_id'],'sturm','P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1','clean','heldout')]=retained
            unavailable.append({'run_id':item['run_id'],'dataset':'sturm','scope':'overlap-excluded robustness',
                'status':'not_available','reason':'Saved perturbation counts aggregate maps, not the3excludedchips; no new corruption inference was run.'})
        for dataset,protocol,condition,raw in variants:
            full={k:np.asarray(v,dtype=np.int64) for k,v in raw.items()}
            retained=({k:v for k,v in full.items() if k not in plan['exclude_complete_WorldFloods_maps']}
                      if dataset=='wf' else matrices[(item['run_id'],dataset,protocol,condition,'heldout')])
            for domain,data in [('full',full),('heldout',retained)]:
                data={k:v for k,v in data.items() if int(v[0,0]+v[1].sum())>0}
                matrices[(item['run_id'],dataset,protocol,condition,domain)]=data
                events=defaultdict(lambda:np.zeros((2,3),dtype=np.int64))
                for name,conf in data.items(): events[name.split('_')[0] if dataset=='wf' else map_events[name]]+=conf
                total=sum(data.values()); metrics=scores(total)
                row={**meta,'dataset':dataset,'protocol':protocol,'condition':condition,'domain':domain,
                     'n_maps':len(data),'n_events':len(events),'valid_pixels':int(total[0,0]+total[1].sum()),
                     'water_pixels':int(total[1,0]+total[1,2]),**metrics}
                metric_rows.append(row)
                for level,units in [('map',data),('event',events),('pooled',{'all':total})]:
                    for name,a in units.items():
                        count_rows.append({**meta,'dataset':dataset,'protocol':protocol,'condition':condition,
                            'domain':domain,'level':level,'unit':name,'TP':int(a[1,0]),'FP':int(a[1,1]),
                            'FN':int(a[1,2]),'TN':int(a[0,0]),**scores(a)})
    metric_names=('mIoU','water_IoU','background_IoU','water_precision','water_recall','water_F1')
    grouped=defaultdict(list)
    for r in metric_rows: grouped[tuple(r[k] for k in ('architecture','variant','budget','dataset','protocol','condition','domain'))].append(r)
    summary=[]
    for key,rr in sorted(grouped.items()):
        n=1 if key[0] in ('teacher','index:mndwi_zero') else 4
        assert len(rr)==n and len({r['seed'] for r in rr})==n
        for metric in metric_names:
            values=[r[metric] for r in rr];assert all(v is not None for v in values)
            summary.append({**dict(zip(('architecture','variant','budget','dataset','protocol','condition','domain'),key)),
                'metric':metric,'n_seeds':n,'mean':float(np.mean(values)),
                'sample_SD_ddof1':float(np.std(values,ddof=1)) if n>1 else None,
                'n_maps':rr[0]['n_maps'],'n_events':rr[0]['n_events'],
                'valid_pixels':rr[0]['valid_pixels'],'water_pixels':rr[0]['water_pixels']})
    # Pooled paired-seed effects and event-macro effects have distinct uncertainty.
    paired=[]; event_summary=[]
    run_by_id={r['run_id']:r for r in spec['runs']}
    contrasts=[]
    for architecture,budget in sorted({(r['architecture'],r['budget']) for r in spec['runs'] if r['run_id'].startswith('O4_train_')}):
        ofa='historical_ofa_clip5_allvalid' if architecture=='efficientvit_b0' and budget==2500 else 'matched_ofa_clip5_valid'
        left=[r for r in spec['runs'] if (r['architecture'],r['budget'],r['variant'])==(architecture,budget,ofa)]
        if not left:continue
        for name,right_variant in [('OFA_minus_matched_KD','matched_lkd_clip5_valid')]:
            pairs=[]
            for l in left:
                r=next(r for r in spec['runs'] if (r['architecture'],r['budget'],r['variant'],r['seed'])==(architecture,budget,right_variant,l['seed']))
                pairs.append((l['run_id'],r['run_id'],l['seed']))
            contrasts.append((name,architecture,budget,pairs))
    for group in sorted({(r['architecture'],r['budget'],r['variant']) for r in spec['runs'] if r['run_id'].startswith('O4_train_')}):
        aa,bb,vv=group; rr=[r for r in spec['runs'] if (r['architecture'],r['budget'],r['variant'])==group]
        for comparator in ('teacher','index:mndwi_zero'):
            fixed=next(r for r in spec['runs'] if r['architecture']==comparator)
            contrasts.append((vv+'_minus_'+comparator,aa,bb,[(r['run_id'],fixed['run_id'],r['seed']) for r in rr]))
    for contrast,arch,budget,pairs in contrasts:
        assert len(pairs)==4
        keys=sorted({k[1:] for k in matrices if k[0]==pairs[0][0] and k[3]=='clean'})
        for dataset,protocol,condition,domain in keys:
            for metric in metric_names:
                differences=[]; by_event=defaultdict(list)
                for left,right,seed in pairs:
                    lm=matrices[(left,dataset,protocol,condition,domain)]; rm=matrices[(right,dataset,protocol,condition,domain)]
                    assert set(lm)==set(rm)
                    differences.append(scores(sum(lm.values()))[metric]-scores(sum(rm.values()))[metric])
                    ls=defaultdict(lambda:np.zeros((2,3),dtype=np.int64)); rs=defaultdict(lambda:np.zeros((2,3),dtype=np.int64))
                    for name in lm:
                        event=name.split('_')[0] if dataset=='wf' else map_events[name]
                        ls[event]+=lm[name];rs[event]+=rm[name]
                    for event in ls:
                        av,bv=scores(ls[event])[metric],scores(rs[event])[metric]
                        by_event[event].append(av-bv if av is not None and bv is not None else None)
                assert len(differences)==4
                common={'contrast':contrast,'architecture':arch,'budget':budget,'dataset':dataset,'protocol':protocol,
                        'condition':condition,'domain':domain,'metric':metric}
                paired.append({**common,'n_seeds':4,'mean_difference':float(np.mean(differences)),
                               'sample_SD_ddof1':float(np.std(differences,ddof=1))})
                values=[float(np.mean(v)) for v in by_event.values() if len(v)==4 and all(x is not None for x in v)]
                lo,hi=boot_ci(np.asarray(values)) if len(values)>=5 else (None,None)
                event_summary.append({**common,'n_events':len(values),'mean_event_difference':float(np.mean(values)) if values else None,
                    'ci95_low':lo,'ci95_high':hi,'uncertainty':'Event resampling conditional on fixed checkpoints; not four independent geographic replications.'})
    csv_out(out/'by_seed.csv',metric_rows);csv_out(out/'confusion_counts.csv',count_rows)
    csv_out(out/'summary.csv',summary);csv_out(out/'paired_seed_summary.csv',paired)
    csv_out(out/'paired_event_summary.csv',event_summary)
    atomic_json(out/'provenance.json',{'status':'complete','mode':mode,'spec_sha256':sha256(spec_path),
        'exclusion_plan_sha256':spec['exclusion_plan_sha256'],'n_runs':len(spec['runs']),
        'no_new_training':True,'no_new_corruption_inference':True,'per_seed_rows':len(metric_rows),
        'unavailable_robustness':unavailable,'exclusions':plan,
        'remaining_scope':'STURM clean sensitivity requires all38 fixed-chip receipts' if mode=='wf' else 'Declared sensitivity complete; independent dry/device/history limitations remain.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('infer','collect'))
    p.add_argument('--spec',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--mode',choices=('wf','sturm','all'),default='all');a=p.parse_args()
    a.spec=a.spec.resolve();a.out=a.out.resolve()
    if a.action=='infer':infer(a.spec,a.out)
    else:collect(a.spec,a.out,a.mode)

if __name__=='__main__':main()
