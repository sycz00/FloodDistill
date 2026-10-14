"""CUDA model execution and batched exact confusion counts; registered support unchanged."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import common10 as c

def confusion_batch(pred,y,parts):
    import torch
    water=torch.stack([((y==1)&support).reshape(-1) for support in parts]).float();land=torch.stack([((y==0)&support).reshape(-1) for support in parts]).float();p=pred.reshape(len(pred),-1).float()
    tp=p@water.T;fp=p@land.T;fn=water.sum(1)[None]-tp;tn=land.sum(1)[None]-fp
    return torch.stack([tp,fp,fn,tn],-1).to(torch.int64)

def evaluate(mode,slot,phase=None):
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
    models={};counters={};handles=[]
    for entry in selected:
        if entry.get('source_reference'):continue
        reusable=entry.get('v9_cache_id') and (cache/entry['v9_cache_id']).exists()
        if entry['graph']!='rule' and not reusable:
            models[entry['id']]=c.load(entry)
            if entry.get('quantized'):
                a,b=clipping_hooks(models[entry['id']]);counters[entry['id']]=a;handles.extend(b)
    cells=defaultdict(lambda:np.zeros(4,np.int64));hists=defaultdict(lambda:np.zeros((4096,2),np.int64));groups=defaultdict(lambda:np.zeros(4,np.int64));joint=defaultdict(lambda:np.zeros(16,np.int64));shared=defaultdict(lambda:np.zeros(6,np.int64));spectra=defaultdict(lambda:np.zeros(7,np.float64));ceilings=defaultdict(lambda:np.zeros(4,np.int64));calibration=defaultdict(lambda:np.zeros((15,4)));decode_times=defaultdict(lambda:[0.,0]);dry=[];clipping=defaultdict(lambda:[0,0]);start=time.monotonic()
    with torch.no_grad():
        for i,row in enumerate(rows):
            with np.load(row['path'],allow_pickle=False) as a:x=a['image'].copy();y=a['mask'].copy();cloud=a['cloud'];cloud=cloud.copy() if cloud.size else None
            if mode=='sealed':
                with open(c.ROOT/f'label_access_slot{slot}.jsonl','a') as output:output.write(json.dumps(dict(path=row['path'],event=row['event'],purpose='one final CUDA V10 readout of derived V9 labels; raw GT not reopened',mask_access_once=True))+'\n')
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
            if i%100==0:print('CUDA',mode,slot,i,'/',len(rows),'elapsed',round(time.monotonic()-start,1),flush=True)
    out=c.ROOT/'metrics'/(mode+'_'+phase if phase and mode=='normal' else mode);out.mkdir(parents=True,exist_ok=True);keys=list(hists);np.savez_compressed(out/f'hist_slot{slot}.npz',keys=np.array([json.dumps(k) for k in keys]),histograms=np.stack([hists[k] for k in keys]) if keys else np.empty((0,4096,2),np.int64));pack=lambda a:[dict(key=list(k),counts=v.tolist()) for k,v in a.items()]
    c.save(out/f'slot{slot}.json',dict(outcome='completed',device='cuda',TF32=False,evaluated_models=[e['id'] for e in output_entries],windows=len(rows),seconds=time.monotonic()-start,event_counts=pack(cells),strata=pack(groups),joint=pack(joint),shared=pack(shared),spectra=pack(spectra),ceilings=pack(ceilings),calibration=pack(calibration),decode_times=pack({k:np.array(a) for k,a in decode_times.items()}),dry=dry,clipping=[dict(dataset=k[0],model=k[1],event=k[2],layer=k[3],clipped=a[0],activations=a[1]) for k,a in clipping.items()],AP='4096 fixed bins[-32,32]; monotone index scaling',shared_classes='Inherited and induced overlap; intersection retained'))
    for h in handles:h.remove()
