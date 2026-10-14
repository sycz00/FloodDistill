"""Prepare the finite prompt5 chip-count workload; never modify the V4 queue."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np

ROOT=Path('/home/schmalstieg/repos/FloodDistill')
OUT=ROOT/'experiments/revision/prompt5_closure'
OLD=Path('/data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006')
SNAP=Path('/data1/schmalstieg/flooddistill/revision_20261005/snapshots/95f47d14ae77a2cf')
PYTHON='/home/schmalstieg/miniconda3/envs/esi/bin/python'
sys.path.insert(0,str(ROOT/'scripts'))
from revision_prompt5_counts import selected_runs
from data.ood_datasets import STURMFloodDataset
from evaluation.revision_native_border import sturm_invalid

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),f'Will not overwrite frozen input: {p}'
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def main():
    q=OUT/'queue';q.mkdir(parents=True,exist_ok=True)
    plan=OUT/'overlap_exclusion_plan.json';p=read(plan)
    refs=read(ROOT/'experiments/revision/prompt2_reference_manifest.json')['STURM']
    comparison=OLD/'queue_prompt4/specs/original_data_comparisons.json'
    runs=selected_runs(read(comparison));assert len(runs)==38
    assert sum(r['architecture'] in ('teacher','index:mndwi_zero') for r in runs)==2
    old=read(OLD/'queue_prompt4_release/manifest.json')
    frozen=q/'frozen';frozen.mkdir(exist_ok=True)
    for source in ('scripts/revision_prompt5_counts.py','experiments/revision_queue.py'):
        dest=frozen/Path(source).name
        if dest.exists():assert sha(dest)==sha(ROOT/source)
        else:shutil.copyfile(ROOT/source,dest)
    helper=frozen/'revision_prompt5_counts.py'
    teacher=read(next(r['evaluation'] for r in runs if r['architecture']=='teacher'))
    stroot=teacher['spec']['roots']['sturm']
    ds=STURMFloodDataset(stroot,sensor='S2',img_size=512,normalize='original',class_mapping='paper')
    names=[v[0].name for v in ds.samples]
    indices=[names.index(fn) for fn in p['exclude_complete_STURM_chips']]
    contexts=sorted({j for i in indices for j in range(i//8*8,min(i//8*8+8,len(ds)))})
    expected={}
    for fn,i in zip(p['exclude_complete_STURM_chips'],indices):
        mask=ds[i]['mask'].numpy().copy();_,invalid=sturm_invalid(ds,i);mask[invalid]=-1
        expected[fn]={'valid_pixels':int((mask!=-1).sum()),'water_pixels':int((mask==1).sum()),
                      'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()}
    common_inputs=[{'path':str(plan),'sha256':sha(plan)},{'path':str(helper),'sha256':sha(helper)}]
    # Pin actual context arrays as well as the recovered immutable scientific sources.
    for i in contexts:
        for path in ds.samples[i]:
            common_inputs.append({'path':str(path),'sha256':sha(path)})
    for path in sorted(SNAP.rglob('*.py')):
        common_inputs.append({'path':str(path),'sha256':sha(path)})
    # Reuse the dependency-code pins from the old identity receipts.
    dep_inputs={v['path']:v for j in old['jobs'] for v in j.get('inputs',[])
                if '/repos/_deps/' in v['path'] or '/.cache/torch/hub/checkpoints/' in v['path']
                or 'efficientvit_seg_b0_cityscapes.pt' in v['path'] or 'model.safetensors' in v['path']}
    common_inputs.extend(dep_inputs.values())
    collector_runs=[]; jobs=[]
    for r in runs:
        canonical=read(r['evaluation']);assert canonical['status']=='complete'
        ck=canonical['spec'].get('checkpoint')
        if ck=='hf':raise ValueError('Expected pinned local teacher checkpoint')
        jid='P5_chips_'+r['run_id'];sp=q/'specs'/f'{jid}.json';result=OUT/'chip_counts'/f'{r["run_id"]}.json'
        spec={'run_id':r['run_id'],'arch':r['architecture'],'checkpoint':ck,
              'evaluation':r['evaluation'],'sturm_root':stroot,
              'runtime_packages':canonical['identity']['runtime_packages'],
              'exclusion_plan':str(plan),'exclusion_plan_sha256':sha(plan),
              'batch_context_indices':contexts,'expected_chip_counts':expected,
              'purpose':'Exactly3 missing corrected chip counts;16 contexts preserving original batch8; no new training or corruption.'}
        write(sp,spec)
        inputs=common_inputs+[{'path':str(sp),'sha256':sha(sp)},
                              {'path':r['evaluation'],'sha256':sha(r['evaluation'])}]
        if ck:inputs.append({'path':ck,'sha256':canonical['identity']['checkpoint_sha256']})
        jobs.append({'id':jid,'kind':'command','gpu':True,'cwd':str(SNAP),'deps':[],
                     'argv':['/usr/bin/ionice','-c','2','-n','7',PYTHON,str(helper),'infer',
                             '--spec',str(sp),'--out',str(result)],
                     'inputs':inputs,'outputs':[{'path':str(result),'type':'json','equals':
                         {'status':'complete','run_id':r['run_id'],'scored_chips':3,'forwarded_context_chips':16,
                          'exclusion_plan_sha256':sha(plan)}}]})
        collector_runs.append({k:r[k] for k in ('run_id','architecture','variant','budget','seed','evaluation','native','sturm_valid')})
        collector_runs[-1]['subset_counts']=str(result)
    spec={'exclusion_plan':str(plan),'exclusion_plan_sha256':sha(plan),'runs':collector_runs,
          'STURM_map_events':{r['map_id']:r['event'] for r in refs.values()},
          'STURM_chip_maps':{fn:refs[fn]['map_id'] for fn in p['exclude_complete_STURM_chips']},
          'STURM_removed_valid_pixels':sum(v['valid_pixels'] for v in expected.values()),
          'STURM_removed_water_pixels':sum(v['water_pixels'] for v in expected.values())}
    cs=OUT/'sensitivity_spec.json';write(cs,spec)
    inputs=common_inputs+[{'path':str(cs),'sha256':sha(cs)}]
    for r in collector_runs:
        for k in ('evaluation','native','sturm_valid'):inputs.append({'path':r[k],'sha256':sha(r[k])})
    jobs.append({'id':'P5_collect_heldout','kind':'command','gpu':False,'cwd':str(SNAP),
                 'deps':[j['id'] for j in jobs],
                 'env':{'CUDA_VISIBLE_DEVICES':''},
                 'argv':[PYTHON,str(helper),'collect','--spec',str(cs),'--out',str(OUT/'sensitivity_complete'),'--mode','all'],
                 'inputs':inputs,'outputs':[{'path':str(OUT/'sensitivity_complete/provenance.json'),
                     'type':'json','equals':{'status':'complete','n_runs':38,'mode':'all',
                                            'exclusion_plan_sha256':sha(plan)}}]})
    manifest={k:old[k] for k in ('schema_version','gpu_lock','gpu_devices','headroom_mb','host_limits','nice','poll_seconds')}
    manifest.update(run_id='prompt5_fixed_chip_counts',jobs=jobs,
        scientific_scope='38 frozen comparators x3 scored chips only; count-only dependent collector; no V4 reruns',
        scheduling_authorization='prompts/prompt5.md; existing authorized dual-GPU UUID scheduling',
        env={**old['env'],'PYTHONPATH':f'{SNAP}:/home/schmalstieg/repos/_deps/efficientvit:/home/schmalstieg/repos/_deps/flooddistill_revision_python',
             'PYTHONDONTWRITEBYTECODE':'1','NO_ALBUMENTATIONS_UPDATE':'1','MPLCONFIGDIR':'/tmp/flooddistill_prompt5_matplotlib'})
    manifest['host_limits']={**manifest['host_limits'],'disk_path':str(OUT)}
    write(q/'manifest.json',manifest)
    write(OUT/'missing_count_workload.json',{'status':'prepared_not_started','n_fixed_comparators':38,
        'n_GPU_jobs':38,'n_training_jobs':0,'n_scored_chips_per_job':3,'n_forwarded_context_chips_per_job':16,
        'missing_cells':'Per-chip clean STURM confusion counts absent; saved map counts reused for every other chip.',
        'mask_counts':expected,'mask_removed_valid':spec['STURM_removed_valid_pixels'],
        'mask_removed_water':spec['STURM_removed_water_pixels'],
        'manifest':str(q/'manifest.json'),'collector':str(OUT/'sensitivity_complete'),
        'batch_size':8,'precision':'float32, AMP disabled','retry_policy':'none',
        'persistence':'existing detached-session queue; survives editor parent closure on awake host; not a reboot service'})
    from experiments.revision_queue import load_manifest
    loaded=load_manifest(q/'manifest.json');assert len(loaded['jobs'])==39
    print(json.dumps({'jobs':39,'comparators':len(runs),'context_indices':contexts,
                      'removed_valid':spec['STURM_removed_valid_pixels'],'removed_water':spec['STURM_removed_water_pixels']}))

if __name__=='__main__':main()
