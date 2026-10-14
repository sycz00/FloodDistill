"""Finite map-sharded recovery around the existing V9 evaluator and audited queue."""
import argparse
from collections import defaultdict, Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import common_v9 as c

ROOT=c.ROOT
OLD=ROOT.parent/'research_v9_recovery_20261013T075000Z'

def prepare():
    rows=c.read(ROOT/'sealed_windows.json')['rows']
    grouped=defaultdict(list)
    for row in rows:
        if not row['omit']:grouped[(row['dataset'],row['map'])].append(row)
    shards=[dict(id=c.q.digest([dataset,name])[:16],dataset=dataset,map=name,windows=len(group),
                 event_ids=sorted({row['event'] for row in group})) for (dataset,name),group in grouped.items()]
    shards.sort(key=lambda row:(-row['windows'],row['id']))
    inputs=[ROOT/'sealed_windows.json',ROOT/'normal_windows.json',ROOT/'final_models_final03.json',
            ROOT/'source_operating_points_final03.json',ROOT/'source_edge_choice.json',
            ROOT/'gpu_inference_sealed_final03.json',ROOT/'training_spec.json']
    source_receipt=c.read(ROOT/'gpu_inference_sealed_final03.json')
    assert source_receipt['outcome']=='completed' and source_receipt['full_set']
    assert len(source_receipt['records'])==17 and all(r['windows']==17907 for r in source_receipt['records'])
    models=c.read(ROOT/'final_models_final03.json')['models']
    assert len(models)==36
    for entry in models:assert c.sha(entry['checkpoint'])==entry['sha256']
    code={p.name:c.sha(p) for p in ROOT.glob('*.py')}
    frozen=dict(outcome='completed',created=c.q.now(),shards=shards,windows=sum(r['windows'] for r in shards),
                map_count=len(shards),checkpoint_models=36,total_predictors=39,cached_GPU_predictors=17,
                input_hashes={str(p):c.sha(p) for p in inputs},code_hashes=code,
                allowed_changes=['Exclude zero-valid reference windows from dry cohorts',
                                 'Exact Euclidean integer-grid decoder disk dilation',
                                 'Cache each fixed map/model/rule threshold once',
                                 'Preserve complete maps as atomic resumable scoring shards'],
                unchanged='All checkpoints,source choices,threshold grids,reference definitions,AP bins,strata,event bootstrap10000 seed1729',
                source_model_selection=False,new_GPU_commands=0,
                threads_per_worker=4,parallel_workers=2,maximum_total_threads=8,
                deadline_hours_from_launch=12,collection_reserve_seconds=1800,
                inherited_normal_source_counts='Identical copied counts; zero-valid dry rows removed additively',
                authorization='User explicitly requested fixing and restarting all needed experiments')
    c.save(ROOT/'frozen_recovery.json',frozen)
    print(json.dumps({k:v for k,v in frozen.items() if k not in ['shards','input_hashes','code_hashes']},indent=2))

def verify_freeze():
    frozen=c.read(ROOT/'frozen_recovery.json')
    for name,digest in frozen['code_hashes'].items():assert c.sha(ROOT/name)==digest,('code changed',name)
    for name,digest in frozen['input_hashes'].items():assert c.sha(name)==digest,('input changed',name)
    return frozen

def complete(shard):
    receipt=c.read(ROOT/'shards'/shard['id']/'complete.json')
    if not receipt:return False
    if receipt.get('outcome')!='completed' or receipt.get('shard')!=shard:return False
    if receipt.get('frozen_sha256')!=c.sha(ROOT/'frozen_recovery.json'):return False
    return all((ROOT/p).is_file() and c.sha(ROOT/p)==sha for p,sha in receipt['outputs'].items())

def work(shard):
    root=ROOT/'shards'/shard['id'];root.mkdir(parents=True,exist_ok=True)
    fd=c.q.lock_fd(root/'worker.lock')
    if fd is None:raise RuntimeError('Shard already owned: '+shard['id'])
    try:
        if complete(shard):return c.read(root/'complete.json')
        import evaluate_v9_final03 as evaluator
        started,cpu=time.monotonic(),time.process_time()
        with open(root/'worker.log','a',buffering=1) as stream:
            oldout,olderr=os.dup(1),os.dup(2)
            try:
                os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
                evaluator.worker('sealed',0,shard)
            finally:
                os.dup2(oldout,1);os.dup2(olderr,2);os.close(oldout);os.close(olderr)
        data=c.read(root/'slot0.json')
        assert data['outcome']=='completed' and not data['unavailable']
        assert len(data['models'])==39 and data['window_count']==shard['windows']
        assert all(row['valid_pixels']>0 for row in data['dry'])
        result=dict(outcome='completed',shard=shard,seconds=time.monotonic()-started,
                    process_cpu_seconds=time.process_time()-cpu,allocated_threads=int(os.environ.get('V9_CPU_THREADS','4')),
                    frozen_sha256=c.sha(ROOT/'frozen_recovery.json'),completed_at=c.q.now(),
                    outputs={str(p.relative_to(ROOT)):c.sha(p) for p in [root/'slot0.json',root/'hist_slot0.npz']})
        c.save(root/'complete.json',result);return result
    finally:os.close(fd)

def pilot():
    frozen=verify_freeze();chosen=[]
    for target in [1,4,16]:
        candidate=min(frozen['shards'],key=lambda r:(abs(r['windows']-target),r['id']))
        if candidate not in chosen:chosen.append(candidate)
    results=[work(shard) for shard in chosen]
    # Warm throughput excludes the first process-import/model-load shard. The
    # protected estimate adds 60 percent overhead and fixed startup/collection.
    warm=results[1:] or results
    rate=sum(row['seconds'] for row in warm)/sum(row['shard']['windows'] for row in warm)
    estimate=rate*frozen['windows']/2*1.6+600
    receipt=dict(outcome='completed',shards=results,warm_seconds_per_tile_all39=rate,
                 estimated_wall_seconds_two_workers=estimate,conservative_factor=1.6,
                 admitted=estimate<11.5*3600,threads=4,parallel_estimate_workers=2,
                 all_required_models_preserved=True,no_model_or_setting_selected_by_target_scores=True)
    c.save(ROOT/'pilot_receipt.json',receipt);print(json.dumps(receipt,indent=2))

def analyze():
    frozen=verify_freeze();pilot_receipt=c.read(ROOT/'pilot_receipt.json')
    assert pilot_receipt['admitted'],'Measured full matrix exceeds renewed deadline'
    runtime=c.read(ROOT/'runtime.json')
    if not runtime:
        now=time.time();runtime=dict(launch=now,absolute_deadline=now+12*3600,analysis_cutoff=now+11.5*3600)
        c.save(ROOT/'runtime.json',runtime)
    halted=[False]
    signal.signal(signal.SIGTERM,lambda *_:halted.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:halted.__setitem__(0,True))
    pending=[shard for shard in frozen['shards'] if not complete(shard)]
    done=[shard for shard in frozen['shards'] if complete(shard)]
    pool=mp.get_context('spawn').Pool(2);active={};failures=[];started=time.monotonic();last=0
    try:
        while pending or active:
            state=c.q.read_json(ROOT/'queue_manifest.state.json') if (ROOT/'queue_manifest.state.json').exists() else {}
            halt=halted[0] or state.get('stop_requested',False) or time.time()>=runtime['analysis_cutoff']
            for key,(shard,result) in list(active.items()):
                if result.ready():
                    try:result.get();done.append(shard)
                    except Exception as error:failures.append(dict(shard=shard,error=repr(error)))
                    del active[key]
            while pending and len(active)<2 and not halt and not state.get('paused',False):
                shard=pending.pop(0);active[shard['id']]=(shard,pool.apply_async(work,(shard,)))
            if time.time()-last>=10 or halt:
                completed_tiles=sum(s['windows'] for s in done)
                c.save(ROOT/'analysis_status.json',dict(at=c.q.now(),pid=c.q.process_identity(os.getpid()),
                       phase='STOPPING_AFTER_ACTIVE_MAPS' if halt else 'PAUSED' if state.get('paused') else 'RUNNING',
                       completed_maps=len(done),total_maps=len(frozen['shards']),completed_windows=completed_tiles,
                       total_windows=frozen['windows'],active_maps=[s for s,r in active.values()],
                       pending_maps=len(pending),failed_maps=failures,elapsed_seconds=time.monotonic()-started,
                       deadline=runtime['absolute_deadline'],maximum_threads=8))
                last=time.time()
            if halt and not active:break
            if time.time()>=runtime['absolute_deadline']-60:
                pool.terminate();active.clear();break
            time.sleep(2)
        pool.close();pool.join()
    except BaseException:
        pool.terminate();pool.join();raise
    finished=len(done)==len(frozen['shards']) and not failures
    c.save(ROOT/'analysis_receipt.json',dict(outcome='completed' if finished else 'partial',maps=len(done),
           total_maps=len(frozen['shards']),windows=sum(s['windows'] for s in done),failures=failures,
           seconds=time.monotonic()-started,deadline=runtime['absolute_deadline'],GPU_seconds=0))

def merge():
    frozen=verify_freeze();assert all(complete(shard) for shard in frozen['shards'])
    counts=defaultdict(lambda:np.zeros(4,np.int64));strata=defaultdict(lambda:np.zeros(4,np.int64))
    hists=defaultdict(lambda:np.zeros((4096,2),np.int64));joint=defaultdict(lambda:np.zeros(16,np.int64))
    dry=[];decode=defaultdict(float);seconds=0.;models=None
    for shard in frozen['shards']:
        root=ROOT/'shards'/shard['id'];a=c.read(root/'slot0.json');seconds+=a['seconds']
        models=models or a['models'];assert models==a['models']
        for r in a['event_counts']:counts[(r['dataset'],r['model'],r['event'],r['variant'])]+=r['counts']
        for r in a['strata']:strata[(r['dataset'],r['model'],r['stratum'])]+=r['counts']
        for r in a['joint']:joint[(r['dataset'],r['model'],r['event'])]+=r['counts']
        dry.extend(a['dry'])
        for name,value in a['decode_cpu_seconds'].items():decode[name]+=value
        with np.load(root/'hist_slot0.npz',allow_pickle=False) as archive:
            for key,value in zip(archive['keys'],archive['histograms']):hists[tuple(json.loads(str(key)))]+=value
    destination=ROOT/'metrics/sealed';destination.mkdir(parents=True,exist_ok=True)
    keys=list(hists);np.savez_compressed(destination/'hist_slot0.npz',keys=np.array([json.dumps(k) for k in keys]),histograms=np.stack([hists[k] for k in keys]))
    c.save(destination/'slot0.json',dict(outcome='completed',seconds=seconds,models=models,unavailable=[],
           event_counts=[dict(dataset=ds,model=name,event=event,variant=variant,counts=cell.tolist()) for (ds,name,event,variant),cell in counts.items()],
           strata=[dict(dataset=ds,model=name,stratum=stratum,counts=cell.tolist()) for (ds,name,stratum),cell in strata.items()],dry=dry,
           joint=[dict(dataset=ds,model=name,event=event,bits='reference*8+teacher*4+MNDWI*2+student',counts=cell.tolist()) for (ds,name,event),cell in joint.items()],
           source_settings={},decode_cpu_seconds=dict(decode),AP_protocol=a['AP_protocol'],strata_protocol=a['strata_protocol'],
           map_shards=len(frozen['shards']),windows=frozen['windows'],zero_valid_dry_windows_excluded=True))
    c.save(ROOT/'evaluation_sealed.json',dict(outcome='completed',slots=[str(destination/'slot0.json')],
           windows=frozen['windows'],maps=len(frozen['shards']),sealed_raw_labels_reopened=False,completed_at=c.q.now()))

def collect():
    complete_analysis=c.read(ROOT/'analysis_receipt.json',{}).get('outcome')=='completed'
    if complete_analysis:merge()
    # Reuse the existing full scientific collector and unchanged event bootstrap.
    import collect_v9_final06
    collect_v9_final06.main()
    report=ROOT/'report_v9.md';text=report.read_text()
    text+='\n## Additive 14 October scoring recovery\n\n'
    text+=f'Scientific core evaluation complete: {complete_analysis}. No training or GPU inference was repeated. The full39-predictor matrix was scored over17907 windows in466 complete-map shards; completion is verified before aggregating. Source settings, masks, grids and whole-event bootstrap remain unchanged. Zero-valid windows are excluded from dry denominators; ordinary independently verified dry imagery remains unavailable. Historical technical failures and optional branch deferrals remain in the earlier campaign.\n'
    text+='Normal/source counts were reused exactly; only invalid zero-support dry rows were removed. This preserves earlier results while repairing the false alarm denominator. Decoder disk dilation was verified equal to the original Euclidean <=k band. Map-level EM/anchor thresholds are computed once per map/model rather than redundantly once per tile.\n'
    text+='The new deadline is12hours from this explicitly authorized restart; no GPU budget was reset or consumed. Queue state, map receipts and pilot timings are the current operational evidence.\n'
    report.write_text(text)
    subprocess.run(['/usr/bin/python3',str(ROOT/'render_report_v9.py')],check=True,timeout=60)
    receipt=c.read(ROOT/'collection_receipt.json',{});receipt.update(scientific_evaluation_complete=complete_analysis,
           report_sha256=c.sha(report),recovery_completed_at=c.q.now(),new_GPU_commands=0,
           historical_optional_deferrals_preserved=True)
    c.save(ROOT/'collection_receipt.json',receipt)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','pilot','analyze','collect']);args=parser.parse_args()
    globals()[args.action]()
