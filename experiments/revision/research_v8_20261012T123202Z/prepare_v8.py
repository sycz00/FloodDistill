"""CPU-only evidence reconciliation and prospective job preparation."""
import json
import os
from pathlib import Path
import hashlib
import shutil
import sys
from datetime import datetime, timezone
import audited_queue as q

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
V7=ROOT.parent/'capacity_v7_20261011T210730Z'
CAPSULE=ROOT.parent/'capacity_v4_20261008/frozen/297031f5f4ae1df5'
RUNGS=ROOT.parent/'capacity_v4_20261008/code/v5'
PYTHON='/home/schmalstieg/miniconda3/envs/esi/bin/python'


def read(p): return json.loads(Path(p).read_text())


def prepare():
    sys.path.insert(0,str(CAPSULE))
    import rasterio
    spec=read(V7/'resolved_spec.json')
    refs=read(V7/'reference_cache/manifest.json')
    assert refs['status']=='complete' and not refs['smoke_only']
    for d in refs['datasets'].values(): assert q.file_hash(d['path'])==d['sha256']
    parents=[x for x in read(V7/'reuse_models.json') if x['graph']=='R4' and x['seed'] in [42,64]]
    for p in parents: assert q.file_hash(p['checkpoint'])==p['sha256']
    windows=[]
    for i,row in enumerate(refs['datasets']['wf']['rows']):
        name=row['map']; rc=row['filename'].split('#r')[1].split('_c'); r,c=map(int,rc)
        with rasterio.open(Path(spec['roots']['wf'])/'test/S2'/name) as src:
            w,h=min(512,src.width-c),min(512,src.height-r)
            crs=str(src.crs); bounds=list(src.window_bounds(rasterio.windows.Window(c,r,w,h)))
        windows.append(dict(index=i,filename=row['filename'],map=name,event=row['event'],
            window=[c,r,w,h],bounds=bounds,crs=crs,omit=row['omit'],input_sha256=row['input_sha256'],mask_sha256=row['mask_sha256']))
    episodes=[]
    for event in sorted({w['event'] for w in windows}):
        for name in sorted({w['map'] for w in windows if w['event']==event and not w['omit']}):
            group=[w for w in windows if w['map']==name and not w['omit']]
            full=[w for w in group if w['window'][2:]==[512,512]]
            if len(full)<3: continue
            supports=sorted(full,key=lambda w:hashlib.sha256(w['filename'].encode()).hexdigest())[:2]
            queries=[w for w in group if w not in supports]
            # One map, disjoint native windows; no resampling of support/query geometry.
            for s in supports:
                for v in queries+supports:
                    if s==v: continue
                    a,b,c,d=s['window'];e,f,g,h=v['window']
                    assert a+c<=e or e+g<=a or b+d<=f or f+h<=b
            episodes.append(dict(event=event,map=name,supports=supports,queries=queries,
                known_training_overlap='excluded by existing scored-support footprint screen',
                prior_exposure='research-exposed exploratory event',area_per_pixel_m2=100))
            break
        if len(episodes)==3: break
    assert episodes, 'No verified disjoint target episodes'
    # Source-only labelled development opportunity, separate from target episodes.
    sys.path.insert(0,str(ROOT))
    import evaluate_v7 as ev
    ev.setup(CAPSULE,RUNGS)
    import scripts.compression_jobs as cj
    val,_=cj.loaders(spec)
    dev=[]
    by_event={}
    for i,p in enumerate(val.dataset.image_files): by_event.setdefault(Path(p).name.split('_')[0],[]).append((i,Path(p)))
    for event,items in sorted(by_event.items()):
        found=None
        for ai,(i,p) in enumerate(items):
            with rasterio.open(p) as src: ab=src.bounds;acrs=src.crs
            for j,v in items[ai+1:]:
                with rasterio.open(v) as src: bb=src.bounds;bcrs=src.crs
                if acrs==bcrs and (ab.right<=bb.left or bb.right<=ab.left or ab.top<=bb.bottom or bb.top<=ab.bottom):
                    found=dict(event=event,support_index=i,query_index=j,support=p.name,query=v.name);break
            if found: break
        if found: dev.append(found)
        if len(dev)==2: break
    train=cj.pseudo(spec,False)
    train_ids=sorted(range(len(train)),key=lambda i:hashlib.sha256(train.files[i].name.encode()).hexdigest())[:32]
    config=dict(spec=spec,capsule=str(CAPSULE),rungs_dir=str(RUNGS),v7=str(V7),parents=parents,
        reference_manifest=str(V7/'reference_cache/manifest.json'),episodes=episodes,source_development=dev,
        train_feature_indices=train_ids,train_feature_ids=[train.files[i].name for i in train_ids],
        seeds=[42,64],support_draws=[0,1],lambda_grid=[.001,.01,.1],lambda_default=.01,
        lr_grid=[.001,.01,.1],gamma=1,temperature=4,feature_scale_floor=.001,
        quantization='W4A8_fixed_PTQ_simulation_stem_W8A8_float32_head',calibration_indices=train_ids[:16],
        precision_status='simulation_only',max_support_pixels=4096)
    q.atomic_json(ROOT/'config.json',config)
    historical=read(V7/'historical_budget.json')
    v7pre=read(V7/'preflight_ledger.json')
    attempts=[read(p) for p in (V7/'attempts').glob('*.json')]
    spent=historical['cumulative_gpu_seconds']+sum(a['command_elapsed_seconds'] for a in v7pre)
    spent+=sum(a.get('command_elapsed_seconds',0) for a in attempts if a.get('gpu') and a.get('finished_at'))
    assert abs(spent-read(V7/'budget_ledger.json')['cumulative_gpu_seconds'])<.01
    # Newer capacity/research workers must be counted and their live reservations respected.
    newer=[];reserved=0
    for directory in ROOT.parent.iterdir():
        if directory in (ROOT,V7) or not directory.is_dir():continue
        for fn in ('manifest_initial.json','manifest.json'):
            p=directory/fn
            if not p.exists():continue
            m=read(p)
            if m.get('frozen_at','')<=read(V7/'manifest_initial.json')['frozen_at']:continue
            lp=directory/'budget_ledger.json'
            if lp.exists():
                b=read(lp);s=read(directory/'state.json') if (directory/'state.json').exists() else {}
                spent+=b.get('incremental_spent_seconds',0)+b.get('active_elapsed_seconds',0)
                if q.alive(s.get('worker')):reserved+=b.get('reserved_gpu_seconds',0)+b.get('protected_collection_seconds',0)
                newer.append(str(lp))
            break
    q.atomic_json(ROOT/'budget_reconciliation.json',dict(at=q.now(),verified_spent_seconds=spent,
        other_reserved_seconds=reserved,incremental_cap_seconds=min(14400,max(0,108000-spent-reserved)),
        v7_attempts=len(attempts),newer_ledgers=newer,counting='elapsed GPU command, including failed/preflight work'))
    sources=[REPO/'AGENT.md',REPO/'prompts/prompt_v8_research_focus.md',
        V7/'report_v7_codex_audit_20261012.md',V7/'budget_ledger.json',V7/'reuse_index.json',
        V7/'analysis_20261012/operational_audit.json',REPO/'experiments/revision/prompt5_closure/reviewer_closure_table.csv',
        REPO/'experiments/revision/prompt5_closure/tables/OFA_matched_effects.csv',
        REPO/'experiments/revision/prompt3_sources/main_raw.txt',REPO/'experiments/revision/prompt3_sources/supplement_raw.txt',
        REPO/'experiments/revision/prompt5_closure/overlap_exclusion_plan.json',Path(spec['index']),Path(spec['teacher'])]
    index=dict(created_at=q.now(),files=[dict(path=str(p.resolve()),sha256=q.file_hash(p)) for p in sources],
        parents=[{**p,'actual_sha256':q.file_hash(p['checkpoint']), 'training_updates':7800,
            'selection':'S11 validation strict mIoU earliest tie'} for p in parents],
        cohort_sha256='1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60',
        missing=['Actual reviewer letter not located; mappings inherit supplied concern summaries','test.txt not located'],
        limitations=['All external sets research-exposed','Teacher inherits 252 labelled source scenes',
            'Original-activation EfficientViT OFA evidence separated from repaired ReLU6 capacity graphs',
            'Historical generation-time cohort receipt unavailable','No verified independent dry-scene references'])
    q.atomic_json(ROOT/'evidence_index.json',index)
    # Inventory only names/available metadata; no new download, training or network dependency.
    bases=[Path('/home/schmalstieg/.cache/huggingface/hub'),REPO/'models',Path('/data1/schmalstieg/flooddistill/data')]
    hits=[]
    for base in bases:
        for p in base.glob('*'):
            if any(s in p.name.lower() for s in ['terramind','croma','s1s2','sentinel1','sar']):hits.append(str(p))
    q.atomic_json(ROOT/'second_teacher_inventory.json',dict(status='blocked_missing_inputs',installed_candidates=hits,
        missing=['Verified installed task-adapted TerraMind/CROMA water head','Ready compatible co-registered SAR/optical inference protocol and paired acquisition receipts'],
        preparation='Locate/licence task checkpoint; establish paired dates, co-registration and preprocessing; cost inference before admission. No full teacher training authorized.'))
    print(json.dumps(dict(run_dir=str(ROOT),events=[(e['event'],len(e['queries'])) for e in episodes],
        source_dev=dev,budget_hours=(108000-spent-reserved)/3600)))


def freeze():
    if (ROOT/'manifest_initial.json').exists():raise RuntimeError('immutable manifest already exists')
    b=read(ROOT/'budget_reconciliation.json');pre=read(ROOT/'preflight_ledger.json') if (ROOT/'preflight_ledger.json').exists() else []
    preseconds=sum(a.get('command_elapsed_seconds',0) for a in pre)
    env={'HF_HUB_OFFLINE':'1','WANDB_MODE':'disabled','NO_ALBUMENTATIONS_UPDATE':'1',
        'MPLCONFIGDIR':str(ROOT/'mpl_cache'),'PYTHONPATH':'/home/schmalstieg/repos/_deps/efficientvit:/home/schmalstieg/repos/_deps/flooddistill_revision_python'}
    bundles=[]
    def add(bundle,jobs):
        rows=[]
        for name,task,seconds,gpu,deps in jobs:
            out=ROOT/'receipts'/f'{name}.json'
            rows.append(dict(id=name,bundle=bundle,role=task,gpu=gpu,max_seconds=seconds,cwd=str(REPO),env=env,
                required_free_mb=9000 if task in ['cache','headroom'] else 4000,deps=deps,
                argv=[PYTHON,str(ROOT/'jobs_v8.py'),task,'--run-dir',str(ROOT),'--out',str(out)],
                receipt=str(out),outputs=[dict(path=str(out),type='json',equals={'outcome':'completed'})]))
        bundles.append(dict(id=bundle,jobs=rows))
    add('cpu_recovery',[('cpu_evidence','recover',600,False,[])])
    add('joint_error',[('joint_errors','headroom',1800,True,[])])
    add('support_cache',[('support_features','cache',1800,True,[])])
    add('rapid_adaptation',[('fp_adaptation','adapt_fp',1800,True,['support_features'])])
    add('compression',[('quant_adaptation','adapt_q',1800,True,['support_features','fp_adaptation']),
        ('measure_FP32','bench_fp32',300,True,[]),('measure_FP16','bench_fp16',300,True,[])])
    # Collection reserve is protected throughout and admitted last.
    add('collection',[('collect_receipt','collect',120,False,[])])
    inputs={str(p):q.file_hash(p) for p in ROOT.glob('*.py')}
    for p in [ROOT/'config.json',ROOT/'frozen_protocol.md',ROOT/'evidence_index.json',
              Path(read(ROOT/'config.json')['rungs_dir'])/'rungs.py']:
        inputs[str(p)]=q.file_hash(p)
    inputs.update(read(V7/'manifest_initial.json')['frozen_inputs'])
    # Do not bind this new run to obsolete historical source worker controls.
    m=dict(schema_version=1,experiment='v8_single_solve_adaptation',frozen_at=q.now(),
        python=PYTHON,gpu_index=0,gpu_uuid='GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc',
        gpu_lock='/data1/schmalstieg/flooddistill/revision_gpu.lock',
        gpu_uuid_lock='/data1/schmalstieg/flooddistill/revision_gpu_GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc.lock',
        experiment_lock=str(ROOT.parent/'.v8_single_solve.lock'),headroom_mb=3072,
        host_limits={**read(V7/'manifest_initial.json')['host_limits'],'disk_path':str(ROOT)},
        historical_gpu_seconds=b['verified_spent_seconds'],preflight_gpu_seconds=preseconds,
        incremental_cap_seconds=b['incremental_cap_seconds'],wall_seconds=28800,
        collection_reserve_seconds=2160,retry_reserve_seconds=0,bundles=bundles,frozen_inputs=inputs,
        conditional_allowlist=[],deferred=['Second task-adapted teacher: missing inputs','Competent full QAT: deferred_budget; PTQ simulation pilot only','Native INT8: unsupported installed backend'])
    m['identity_sha256']=q.digest({k:m[k] for k in ['experiment','frozen_inputs','bundles']})
    q.atomic_json(ROOT/'manifest_initial.json',m);q.atomic_json(ROOT/'manifest.json',m)
    (ROOT/'start_overnight.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+PYTHON+' '+str(ROOT/'run_overnight.py')+' start --run-dir '+str(ROOT)+'\n')
    os.chmod(ROOT/'start_overnight.sh',0o700)
    print(json.dumps(dict(manifest_sha256=q.file_hash(ROOT/'manifest_initial.json'),preflight_seconds=preseconds,cap_seconds=m['incremental_cap_seconds'])))


if __name__=='__main__':
    prepare() if len(sys.argv)==1 or sys.argv[1]=='prepare' else freeze()
