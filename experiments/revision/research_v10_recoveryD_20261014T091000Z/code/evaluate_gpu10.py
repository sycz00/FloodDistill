"""CUDA model execution and batched exact confusion counts; registered support unchanged."""
import json,time,hashlib,os,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import common10 as c

CHECKPOINT_WINDOWS=25
FLOAT_TOTALS={'spectra','calibration','decode_times'}
SHAPES={'cells':(4,),'hists':(4096,2),'groups':(4,),'joint':(16,),
        'shared':(6,),'spectra':(7,),'ceilings':(4,),'calibration':(15,4),
        'decode_times':(2,),'clipping':(2,)}

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def empty_aggregates():
    result={name:defaultdict(lambda shape=shape,name=name:np.zeros(shape,np.float64 if name in FLOAT_TOTALS else np.int64))
            for name,shape in SHAPES.items()}
    result['dry']=[]
    return result

def save_shard(path, protocol, ids, start, aggregates, seconds):
    """Atomic integer deltas plus cumulative floating totals preserve sum order."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={'metadata':np.array(json.dumps(dict(schema=1,protocol=protocol,row_ids=ids,
        start=start,end=start+len(ids),seconds=seconds,float_policy='cumulative totals preserve original sequential addition'))),
        'dry':np.array(json.dumps(aggregates['dry']))}
    for name,shape in SHAPES.items():
        values=aggregates[name];keys=list(values)
        arrays[name+'_keys']=np.array([json.dumps(k) for k in keys],dtype=str)
        arrays[name]=np.stack([values[k] for k in keys]) if keys else np.empty((0,*shape),np.float64 if name in FLOAT_TOTALS else np.int64)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('wb') as out:
        np.savez(out,**arrays);out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)

def committed_shards(directory, protocol, row_ids):
    paths=sorted(Path(directory).glob('chunk_*.npz'));position=0;seconds=0.
    for path in paths:
        with np.load(path,allow_pickle=False) as data:meta=json.loads(str(data['metadata']))
        assert meta['schema']==1 and meta['protocol']==protocol,'Resume protocol/input/model/settings mismatch'
        assert meta['start']==position and meta['end']>position,'Noncontiguous or duplicate resume shard'
        assert meta['row_ids']==row_ids[position:meta['end']],'Resume row identity/order mismatch'
        position=meta['end'];seconds+=meta['seconds']
    assert position<=len(row_ids)
    return paths,position,seconds

def restore_shards(paths, floating_only=False):
    aggregates=empty_aggregates()
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            for name in SHAPES:
                if floating_only and name not in FLOAT_TOTALS:continue
                if name in FLOAT_TOTALS:aggregates[name].clear()
                for key,value in zip(data[name+'_keys'],data[name]):
                    key=tuple(json.loads(str(key)))
                    if name in FLOAT_TOTALS:aggregates[name][key]=value.copy()
                    else:aggregates[name][key]+=value
            if not floating_only:aggregates['dry'].extend(json.loads(str(data['dry'])))
    return aggregates

def profile_final(output_name="final_readout_profile.json", require_final_freeze=True):
    """Fixed source-only full final bundle, before any final target-mask access."""
    if require_final_freeze:
        frozen=c.read(c.ROOT/'final_evaluation_freeze.json',{})
        assert frozen.get('outcome')=='completed'
        assert frozen['model_manifest_sha256']==c.sha(c.ROOT/'readout_models.json')
        assert frozen['source_settings_sha256']==c.sha(c.ROOT/'operating_points.json')
    started=time.monotonic();result=evaluate('sealed',0,source_profile=True)
    rows=[r for r in c.read(c.V9/'sealed_windows.json')['rows'] if not r['omit'] and r['dataset']!='s11val']
    events=sorted({(r['dataset'],r['event']) for r in rows})
    sizes=[sum((r['dataset'],r['event']) in set(events[i::3]) for r in rows) for i in range(3)]
    # First two fixed source windows warm CUDA kernels/caches. Charge their full
    # cost once in startup, rather than multiplying cold setup by every target tile.
    measured=result['profile_window_seconds'][2:]
    assert len(measured)==10
    per_window=(max(measured)+result['profile_checkpoint_seconds']/12)*1.5
    startup=max(0,time.monotonic()-started-sum(measured))
    limits=[math.ceil(n*per_window+startup*1.5+300) for n in sizes]
    receipt=dict(outcome='completed',source_only=True,target_labels_read=False,
        frozen_model_manifest_sha256=c.sha(c.ROOT/'readout_models.json'),
        frozen_settings_sha256=c.sha(c.ROOT/'operating_points.json'),slot_windows=sizes,
        slot_seconds=limits,total_seconds=sum(limits),measured_seconds_per_window=measured,
        conservative_seconds_per_window=per_window,startup_serialization_seconds=startup,
        warmup_windows=2,timed_windows=10,warmup_seconds=sum(result['profile_window_seconds'][:2]),
        source_windows=result['profile_source_paths'],evaluated_models=result['evaluated_models'],
        reason='Exact final model/metric bundle on first12 fixed S11val windows; first2 warmup charged once, max remaining10 plus amortized atomic checkpoint times1.5 plus1.5 startup and300s each slot',
        seconds=time.monotonic()-started)
    c.save(c.ROOT/output_name,receipt)
    if output_name=='final_readout_profile.json':c.save(c.ROOT/'source_runtime_profile.json',receipt)
    return receipt

def confusion_batch(pred,y,parts):
    import torch
    water=torch.stack([((y==1)&support).reshape(-1) for support in parts]).float();land=torch.stack([((y==0)&support).reshape(-1) for support in parts]).float();p=pred.reshape(len(pred),-1).float()
    tp=p@water.T;fp=p@land.T;fn=water.sum(1)[None]-tp;tn=land.sum(1)[None]-fp
    return torch.stack([tp,fp,fn,tn],-1).to(torch.int64)

def evaluate(mode,slot,phase=None,source_profile=False):
    torch,_,_,v=c.setup();from evaluate10 import strata,hist,probability_bins,is_dry_reference
    from quant10 import clipping_hooks
    entries=c.read(c.ROOT/'readout_models.json')['models'];selected=[x for x in entries if mode!='sealed' or x.get('sealed',True)];output_entries=selected
    settings=c.read(c.ROOT/'operating_points.json',{})
    if mode=='normal' and phase in ['B','C']:
        done=set()
        for previous in ['A','B'] if phase=='C' else ['A']:
            for index in range(3):done.update(c.read(c.ROOT/'metrics'/('normal_'+previous)/f'slot{index}.json',{}).get('evaluated_models',[]))
        output_entries=[x for x in selected if x['id'] not in done]
        if phase=='C' and settings.get('decode_rule'):output_entries=selected
        required={x['id'] for x in output_entries}|{'teacher','MNDWI','WI2015','AWEIsh'}
        while True:
            before=len(required)
            for entry in entries:
                if entry['id'] in required:required.update(a for a in [entry.get('source_reference'),entry.get('fp_reference')] if a)
            if before==len(required):break
        selected=[x for x in entries if x['id'] in required]
    selected=sorted(selected,key=lambda x:bool(x.get('source_reference')))
    inputs=c.read(c.V9/('sealed_windows.json' if mode=='sealed' else 'normal_windows.json'));rows=inputs['rows'] if mode=='sealed' else inputs['windows'];rows=[x for x in rows if not x['omit'] and x['dataset']!='s11val'];events=sorted({(x['dataset'],x['event']) for x in rows});own=set(events[slot::3]);rows=[x for x in rows if (x['dataset'],x['event']) in own]
    cache=Path('/data1/schmalstieg/flooddistill')/('revision_'+c.V9.name)/'gpu_predictions'/('sealed_tf32_off_gpu_ptq' if mode=='sealed' else 'normal_tf32_off_gpu_ptq')
    if source_profile:
        rows=[r for r in c.read(c.V9/'normal_windows.json')['windows'] if r['dataset']=='s11val' and not r['omit']][:12]
        assert len(rows)==12,'Source runtime profile needs twelve fixed source windows'
        cache=cache.parent/'normal_tf32_off_gpu_ptq'
    out=c.ROOT/'metrics'/(mode+'_'+phase if phase and mode=='normal' else mode)
    if source_profile:out=c.ROOT/'source_runtime_probe'
    row_ids=[fingerprint(r) for r in rows]
    protocol=fingerprint(dict(mode=mode,slot=slot,phase=phase,source_profile=source_profile,
        selected=selected,output_entries=output_entries,settings=settings,row_ids=row_ids,
        model_manifest_sha256=c.sha(c.ROOT/'readout_models.json'),
        membership_sha256=c.sha(c.V9/('normal_windows.json' if source_profile or mode!='sealed' else 'sealed_windows.json')),
        evaluator_sha256=c.sha(__file__),metric_code_sha256=c.sha(Path(__file__).with_name('evaluate10.py')),
        reference_code_sha256=c.sha(v.__file__),common_code_sha256=c.sha(c.__file__),cache_root=str(cache)))
    journal=out/f'resume_slot{slot}'
    shard_paths,position,previous_seconds=([],0,0.) if source_profile else committed_shards(journal,protocol,row_ids)
    existing=c.read(out/f'slot{slot}.json',{}) if not source_profile else {}
    if existing.get('outcome')=='completed':
        assert existing.get('resume_protocol_sha256')==protocol,'Completed readout protocol mismatch'
        if (out/f'hist_slot{slot}.npz').exists():return existing
    models={};counters={};handles=[]
    for entry in selected:
        if entry.get('source_reference'):continue
        reusable=entry.get('v9_cache_id') and (cache/entry['v9_cache_id']).exists()
        if source_profile and reusable:
            reusable=all((cache/entry['v9_cache_id']/(Path(r['path']).stem+'.npy')).exists() for r in rows)
        if entry['graph']!='rule' and not reusable:
            models[entry['id']]=c.load(entry)
            if entry.get('quantized'):
                a,b=clipping_hooks(models[entry['id']]);counters[entry['id']]=a;handles.extend(b)
    aggregates=restore_shards(shard_paths[-1:],floating_only=True) if shard_paths else empty_aggregates()
    cells,hists,groups,joint,shared,spectra,ceilings,calibration,decode_times,clipping=(aggregates[n] for n in SHAPES)
    dry=aggregates['dry'];start=time.monotonic();chunk_start=position;chunk_time=start;profile_times=[]
    with torch.no_grad():
        for i in range(position,len(rows)):
            row=rows[i];window_started=time.monotonic()
            with np.load(row['path'],allow_pickle=False) as a:x=a['image'].copy();y=a['mask'].copy();cloud=a['cloud'];cloud=cloud.copy() if cloud.size else None
            if mode=='sealed' and not source_profile:
                with open(c.ROOT/f'label_access_slot{slot}.jsonl','a') as output:output.write(json.dumps(dict(path=row['path'],event=row['event'],purpose='one final CUDA V10 readout of derived V9 labels; raw GT not reopened',mask_access_once=False,completed_shards_never_reread=True,maximum_uncommitted_tail_rereads=24,resume_protocol_sha256=protocol,row_id=row_ids[i]))+'\n')
            rho=v.reflectance(x);indices,available=v.indices(rho)
            if cloud is None:
                import edge_v9 as edge;cloud=edge.cloud_test(rho)
            parts=strata(y,rho,cloud);labels=list(parts);support=torch.stack([torch.from_numpy(parts[label]).to(c.device()) for label in labels]);Y=torch.from_numpy(y).to(c.device());valid=y!=-1;reference=y[valid].astype(np.int64);X=torch.from_numpy(x)[None].to(c.device());margins={};predictions={}
            for entry in selected:
                name=entry['id']
                if entry.get('source_reference'):m=margins[entry['source_reference']]
                elif entry['graph']=='rule':m=torch.from_numpy(np.where(available,indices[name]*{'MNDWI':32,'WI2015':.5,'AWEIsh':32}[name],0).astype(np.float32)).to(c.device())
                elif name in models:z=models[name](X);assert torch.isfinite(z).all();m=z[0,1]-z[0,0]
                else:m=torch.from_numpy(np.load(cache/entry['v9_cache_id']/(Path(row['path']).stem+'.npy'),allow_pickle=False)).to(c.device())
                margins[name]=m;predictions[name]=m>entry.get('matched_tau',0.)
            teacher=predictions['teacher'];rule=predictions['MNDWI'];Twrong=teacher!=(Y==1);mask_matrix=support.reshape(len(labels),-1).float();V=(Y!=-1);R=Y==1;rho_gpu=torch.from_numpy(rho.reshape(6,-1)).to(c.device(),dtype=torch.float64)
            for begin in range(0,len(output_entries),8):
                chunk=output_entries[begin:begin+8];P=torch.stack([predictions[e['id']] for e in chunk]);M=torch.stack([margins[e['id']] for e in chunk]);C=confusion_batch(P,Y,support).cpu().numpy();pred_np=P.cpu().numpy();margin_np=M.cpu().numpy()
                wrong=P!=R;inherited=wrong&Twrong;induced=torch.stack([wrong[j]&(predictions[e['fp_reference']]==R) if e.get('fp_reference') in predictions else torch.zeros_like(R) for j,e in enumerate(chunk)]);miss=V&R&~P&~teacher&~rule;false=V&~R&P&teacher&rule
                shared_arrays=[miss,false,V&inherited,V&induced,V&inherited&induced,V&wrong&~inherited&~induced];S=torch.stack([a.reshape(len(chunk),-1).float()@mask_matrix.T for a in shared_arrays],-1).to(torch.int64).cpu().numpy()
                spectral=[torch.cat([a.reshape(len(chunk),-1).to(torch.float64)@rho_gpu.T,a.sum((-2,-1))[:,None]],1).cpu().numpy() for a in [miss,false]]
                codes=8*R.long()+4*teacher.long()+2*rule.long();offset=torch.arange(len(chunk),device=c.device())[:,None]*16;J=torch.bincount((codes[V][None]+P[:,V].long()+offset).flatten(),minlength=len(chunk)*16).reshape(len(chunk),16).cpu().numpy()
                oracle={}
                for label in ['shore<=2','cloud<=50','landlike_water']:
                    if label in parts:oracle[label]=confusion_batch(torch.where(torch.from_numpy(parts[label]).to(c.device()),R,P),Y,[torch.ones_like(R)]).cpu().numpy()[:,0]
                for j,entry in enumerate(chunk):
                    name=entry['id'];key=(row['dataset'],name,row['event']);cell=C[j,labels.index('all')];cells[key+('deployed',)]+=cell;hists[key]+=hist(margin_np[j][valid],reference);q=settings.get(name,{}).get('matched_tau')
                    if q is not None:cells[key+('source_matched_tau',)]+=confusion_batch((M[j]>q)[None],Y,[torch.ones_like(R)]).cpu().numpy()[0,0]
                    if settings.get('decode_rule'):
                        d=settings['decode_rule'];tic=time.monotonic();decoded=v.decode(pred_np[j],indices[d['index']],available,d['k'],d['theta']);elapsed=time.monotonic()-tic;decode_times[key][0]+=elapsed;decode_times[key][1]+=1;cells[key+('decode',)]+=confusion_batch(torch.from_numpy(decoded)[None].to(c.device()),Y,[torch.ones_like(R)]).cpu().numpy()[0,0]
                    if entry.get('method') in ['F0','F1','F2','F3','F4']:calibration[key]+=probability_bins(margin_np[j],y,v.edge_distance(teacher.cpu().numpy())<=2)
                    for si,label in enumerate(labels):groups[key+(label,)]+=C[j,si];shared[key+(label,)]+=S[j,si]
                    for kind,a in zip(['shared_missed_water','shared_false_water'],spectral):spectra[key+(kind,)]+=a[j]
                    for kind,a in oracle.items():ceilings[key+(kind,)]+=a[j]
                    if is_dry_reference(y):
                        n=int(valid.sum());water=int(cell[1]);dry.append(dict(dataset=row['dataset'],model=name,event=row['event'],name=row['name'],valid_pixels=n,water_pixels=water,any_water=bool(water),alarm_001=water>=.001*n,alarm_01=water>=.01*n,alarm_05=water>=.05*n))
                    if row.get('same_day_sentinel2'):cells[('wf2_same_day_S2',name,row['event'],'deployed')]+=cell
                    if row.get('label_satellite'):cells[(row['dataset']+'_label_'+str(row['label_satellite']),name,row['event'],'deployed')]+=cell
            for name,a in counters.items():
                for layer,values in a.items():key=(row['dataset'],name,row['event'],layer);clipping[key][0]+=int(values[0]);clipping[key][1]+=int(values[1]);values[0]=0;values[1]=0
            elapsed_window=time.monotonic()-window_started
            if source_profile:profile_times.append(elapsed_window)
            if not source_profile and ((i+1)%CHECKPOINT_WINDOWS==0 or i+1==len(rows)):
                now=time.monotonic();path=journal/f'chunk_{chunk_start:08d}_{i+1:08d}.npz'
                save_shard(path,protocol,row_ids[chunk_start:i+1],chunk_start,aggregates,now-chunk_time)
                shard_paths.append(path);chunk_start=i+1;chunk_time=time.monotonic()
                # Integer deltas reset; cumulative floats retain original addition order.
                for name in SHAPES:
                    if name not in FLOAT_TOTALS:aggregates[name].clear()
                dry.clear()
                elapsed=previous_seconds+time.monotonic()-start
                c.save(out/f'progress_slot{slot}.json',dict(outcome='running',protocol_sha256=protocol,
                    committed_windows=i+1,total_windows=len(rows),elapsed_seconds=elapsed,
                    estimated_remaining_seconds=elapsed/(i+1)*(len(rows)-i-1),at=time.time(),
                    resume_count=len(shard_paths),maximum_uncommitted_tail_rereads=24))
            if i%25==0:print('CUDA',mode,slot,i+1,'/',len(rows),'elapsed',round(time.monotonic()-start,1),flush=True)
    if not source_profile:
        aggregates=restore_shards(shard_paths)
        cells,hists,groups,joint,shared,spectra,ceilings,calibration,decode_times,clipping=(aggregates[n] for n in SHAPES)
        dry=aggregates['dry']
    out.mkdir(parents=True,exist_ok=True);keys=list(hists)
    if not source_profile:
        temporary=out/f'hist_slot{slot}.npz.tmp.{os.getpid()}'
        with temporary.open('wb') as output:
            np.savez_compressed(output,keys=np.array([json.dumps(k) for k in keys]),histograms=np.stack([hists[k] for k in keys]) if keys else np.empty((0,4096,2),np.int64))
        os.replace(temporary,out/f'hist_slot{slot}.npz')
    pack=lambda a:[dict(key=list(k),counts=np.asarray(v).tolist()) for k,v in a.items()]
    result=dict(outcome='completed',device='cuda',TF32=False,evaluated_models=[e['id'] for e in output_entries],windows=len(rows),seconds=previous_seconds+time.monotonic()-start,resume_protocol_sha256=protocol,committed_windows=len(rows),maximum_uncommitted_tail_rereads=24,event_counts=pack(cells),strata=pack(groups),joint=pack(joint),shared=pack(shared),spectra=pack(spectra),ceilings=pack(ceilings),calibration=pack(calibration),decode_times=pack({k:np.array(a) for k,a in decode_times.items()}),dry=dry,clipping=[dict(dataset=k[0],model=k[1],event=k[2],layer=k[3],clipped=a[0],activations=a[1]) for k,a in clipping.items()],AP='4096 fixed bins[-32,32]; monotone index scaling',shared_classes='Inherited and induced overlap; intersection retained')
    if source_profile:
        checkpoint_started=time.monotonic()
        save_shard(out/'source_checkpoint_probe.npz',protocol,row_ids,0,aggregates,time.monotonic()-start)
        result.update(profile_window_seconds=profile_times,profile_source_paths=[r['path'] for r in rows],source_only=True,
                      profile_checkpoint_seconds=time.monotonic()-checkpoint_started)
    else:
        c.save(out/f'slot{slot}.json',result)
        c.save(out/f'progress_slot{slot}.json',dict(outcome='completed',protocol_sha256=protocol,
            committed_windows=len(rows),total_windows=len(rows),elapsed_seconds=result['seconds'],
            estimated_remaining_seconds=0,at=time.time(),maximum_uncommitted_tail_rereads=24))
    for h in handles:h.remove()
    return result
