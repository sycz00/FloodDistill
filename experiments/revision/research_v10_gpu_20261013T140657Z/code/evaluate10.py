"""Shared V10 evaluator, native registered support and one sealed label read per window."""
import argparse,time,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
import common10 as c

def counts(pred,y,where=None):
    valid=y!=-1
    if where is not None:valid &= where
    p=pred[valid];w=y[valid]==1
    return np.array([np.count_nonzero(p&w),np.count_nonzero(p&~w),np.count_nonzero(~p&w),np.count_nonzero(~p&~w)],np.int64)
def scores(cell):
    tp,fp,fn,tn=map(int,cell);water=tp/(tp+fp+fn) if tp+fp+fn else None;land=tn/(tn+fp+fn) if tn+fp+fn else None
    return dict(water_IoU=water,mIoU=np.mean([x for x in [water,land] if x is not None]) if water is not None or land is not None else None,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
def ap(h):
    z=h[::-1];tp=z[:,1].cumsum();fp=z[:,0].cumsum();total=tp[-1]
    return float(np.sum(z[:,1]/total*np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0))) if total else None
def hist(m,y):
    i=np.clip(np.floor((m+32)*64),0,4095).astype(np.int64);return np.bincount(i*2+y,minlength=8192).reshape(4096,2)
def probability_bins(m,y,band):
    valid=(y!=-1)&band;p=1/(1+np.exp(-np.clip(m[valid],-80,80)));w=(y[valid]==1).astype(float);b=np.minimum((p*15).astype(int),14)
    return np.stack([np.bincount(b,weights=a,minlength=15) for a in [np.ones(len(p)),p,w,(p-w)**2]],1)
def source(entry,model=None,write_margins=True):
    torch,_,_,v=c.setup();model=c.load(entry) if model is None else model;rows=[x for x in c.read(c.V9/'normal_windows.json')['windows'] if x['dataset']=='s11val'];cells=defaultdict(lambda:np.zeros(4,np.int64));margins=[];metadata=[];calibration=np.zeros((15,4))
    with torch.no_grad():
        for row in rows:
            with np.load(row['path'],allow_pickle=False) as a:x=a['image'].copy();y=a['mask'].copy()
            z=model(torch.from_numpy(x)[None].to(c.device()));assert torch.isfinite(z).all();m=(z[0,1]-z[0,0]).cpu().numpy();cells[row['event']]+=counts(m>0,y);valid=y!=-1;margins.append(m[valid]);metadata.append((row['event'],int(valid.sum())))
            if entry.get('method') in ['F0','F1','F2','F3','F4']:
                teacher_path=Path('/data1/schmalstieg/flooddistill/revision_'+c.V9.name)/'gpu_predictions/normal_tf32_off_gpu_ptq/teacher'/(Path(row['path']).stem+'.npy')
                teacher_margin=np.load(teacher_path,allow_pickle=False);calibration+=probability_bins(m,y,v.edge_distance(teacher_margin>0)<=2)
    out=c.ROOT/'source_margins'/(entry['id']+'.npy');out.parent.mkdir(exist_ok=True);
    if write_margins:np.save(out,np.concatenate(margins).astype(np.float32))
    equal=np.mean([scores(cell)['water_IoU'] for cell in cells.values() if scores(cell)['water_IoU'] is not None])
    result=dict(outcome='completed',model=entry['id'],checkpoint_sha256=entry['sha256'],equal_event_water_IoU=float(equal),pooled=scores(sum(cells.values())),event_counts={k:v.tolist() for k,v in cells.items()},margins=str(out) if write_margins else None,source_only=True,threshold=0,valid_support='S11 validation reference-valid pixels only')
    if calibration[:,0].sum():result['teacher_edge_band_calibration_bins']=calibration.tolist()
    c.save(c.ROOT/'source_metrics'/(entry['id']+'.json'),result);return result

def strata(y,rho,cloud):
    water=y==1;land=y==0;dw=ndi.distance_transform_edt(~land) if land.any() else np.full(y.shape,np.inf);dl=ndi.distance_transform_edt(~water) if water.any() else np.full(y.shape,np.inf);distance=np.where(water,dw,dl)
    label,n=ndi.label(water,np.ones((3,3),bool));mx=ndi.maximum(dw,label,index=np.arange(1,n+1)) if n else [];half=np.r_[0,mx][label]
    vis=rho[:3].mean(0);ndvi=(rho[3]-rho[2])/np.maximum(rho[3]+rho[2],1e-6)
    result={'all':np.ones(y.shape,bool),'shore<=2':distance<=2,'shore3-10':(distance>2)&(distance<=10),'shore>10':distance>10,'thin_water<=2':water&(half<=2),'vegetated':ndvi>.4,'dark':(vis<.06)&(rho[3]<.08)&(rho[4]<.06),'bright':vis>.2}
    if cloud is not None:
        cd=ndi.distance_transform_edt(~cloud) if cloud.any() else np.full(y.shape,np.inf)
        for bound in [5,20,50]:result[f'cloud<={bound}']=cd<=bound
    result['landlike_water']=water&(rho[3]>=.08)&(rho[4]>=.06)
    return result

def evaluate(mode,slot,phase=None):
    if c.device().type=='cuda':
        import evaluate_gpu10
        return evaluate_gpu10.evaluate(mode,slot,phase)
    torch,_,_,v=c.setup();manifest=c.read(c.ROOT/'readout_models.json');entries=manifest['models'];selected=[x for x in entries if mode!='sealed' or x.get('sealed',True)]
    output_entries=selected
    if mode=='normal' and phase in ['B','C']:
        done=set()
        for previous in ['A','B'] if phase=='C' else ['A']:
            for index in range(3):done.update(c.read(c.ROOT/'metrics'/('normal_'+previous)/f'slot{index}.json',{}).get('evaluated_models',[]))
        output_entries=[x for x in selected if x['id'] not in done]
        if phase=='C' and c.read(c.ROOT/'operating_points.json',{}).get('decode_rule'):output_entries=selected
        required={x['id'] for x in output_entries}|{'teacher','MNDWI','WI2015','AWEIsh'}
        for entry in output_entries:
            required.update(x for x in [entry.get('source_reference'),entry.get('fp_reference')] if x)
        # References can themselves be operating-point views.
        while True:
            before=len(required)
            for entry in entries:
                if entry['id'] in required:required.update(x for x in [entry.get('source_reference'),entry.get('fp_reference')] if x)
            if len(required)==before:break
        selected=[x for x in entries if x['id'] in required]
    selected=sorted(selected,key=lambda x:bool(x.get('source_reference')))
    inputs=c.read(c.V9/('sealed_windows.json' if mode=='sealed' else 'normal_windows.json'));rows=inputs['rows'] if mode=='sealed' else inputs['windows'];rows=[x for x in rows if not x['omit'] and x['dataset']!='s11val'];events=sorted({(x['dataset'],x['event']) for x in rows});own=set(events[slot::3]);rows=[x for x in rows if (x['dataset'],x['event']) in own]
    cached_root=Path('/data1/schmalstieg/flooddistill')/('revision_'+c.V9.name)/'gpu_predictions'/('sealed_tf32_off_gpu_ptq' if mode=='sealed' else 'normal_tf32_off_gpu_ptq')
    settings=c.read(c.ROOT/'operating_points.json',{});models={};clip_counters={};clip_hooks=[]
    from quant10 import clipping_hooks
    for entry in selected:
        if entry.get('source_reference'):continue
        path=c.DATA/'unet_gpu_predictions'/entry['id']
        reusable=(entry.get('v9_cache_id') and (cached_root/entry['v9_cache_id']).exists()) or (mode=='sealed' and c.read(c.ROOT/'gpu_model_receipts'/(entry['id']+'.json'),{}).get('outcome')=='completed')
        if entry['graph']!='rule' and not reusable:
            models[entry['id']]=c.load(entry)
            if entry.get('quantized'):
                counters,handles=clipping_hooks(models[entry['id']]);clip_counters[entry['id']]=counters;clip_hooks.extend(handles)
    cells=defaultdict(lambda:np.zeros(4,np.int64));hists=defaultdict(lambda:np.zeros((4096,2),np.int64));groups=defaultdict(lambda:np.zeros(4,np.int64));joint=defaultdict(lambda:np.zeros(16,np.int64));shared=defaultdict(lambda:np.zeros(6,np.int64));spectra=defaultdict(lambda:np.zeros(7,np.float64));ceilings=defaultdict(lambda:np.zeros(4,np.int64));calibration=defaultdict(lambda:np.zeros((15,4)));decode_times=defaultdict(lambda:[0.,0]);dry=[];clipping=defaultdict(lambda:[0,0]);start=time.monotonic()
    for i,row in enumerate(rows):
        with np.load(row['path'],allow_pickle=False) as data:x=data['image'].copy();y=data['mask'].copy();cloud=data['cloud'];cloud=cloud.copy() if cloud.size else None
        if mode=='sealed':
            with open(c.ROOT/f'label_access_slot{slot}.jsonl','a') as output:output.write(json.dumps(dict(path=row['path'],event=row['event'],purpose='one final V10 evaluation, derived V9 reference cache; raw GT never reopened',mask_access_once=True))+'\n')
        rho=v.reflectance(x);indices,available=v.indices(rho)
        if cloud is None:
            import edge_v9 as edge
            cloud=edge.cloud_test(rho)
        parts=strata(y,rho,cloud);valid=y!=-1;reference=y[valid].astype(np.int64);margins={};predictions={}
        with torch.no_grad():
            for entry in selected:
                name=entry['id']
                if entry.get('source_reference'):m=margins[entry['source_reference']]
                elif entry['graph']=='rule':m=indices[name]*({'MNDWI':32,'WI2015':.5,'AWEIsh':32}[name]);m=np.where(available,m,0)
                elif name in models:
                    z=models[name](torch.from_numpy(x)[None].to(c.device()));assert torch.isfinite(z).all();m=(z[0,1]-z[0,0]).cpu().numpy()
                else:
                    base=c.DATA/'unet_gpu_predictions'/name if mode=='sealed' and c.read(c.ROOT/'gpu_model_receipts'/(name+'.json'),{}).get('outcome')=='completed' else cached_root/entry.get('v9_cache_id',name)
                    m=np.load(base/(Path(row['path']).stem+'.npy'),allow_pickle=False)
                margins[name]=m;predictions[name]=m>entry.get('matched_tau',0.)
        teacher=predictions['teacher'];rule=predictions['MNDWI'];teacher_wrong=teacher!=(y==1)
        for entry in output_entries:
            name=entry['id'];m=margins[name];p=predictions[name];key=(row['dataset'],name,row['event']);cell=counts(p,y);cells[key+('deployed',)]+=cell;hists[key]+=hist(m[valid],reference)
            tau=settings.get(name,{}).get('matched_tau')
            if tau is not None:cells[key+('source_matched_tau',)]+=counts(m>tau,y)
            decode=settings.get('decode_rule')
            if decode:
                timer=time.monotonic();decoded=v.decode(p,indices[decode['index']],available,decode['k'],decode['theta']);elapsed=time.monotonic()-timer
                decode_times[key][0]+=elapsed;decode_times[key][1]+=1;cells[key+('decode',)]+=counts(decoded,y)
            if entry.get('method') in ['F0','F1','F2','F3','F4']:calibration[key]+=probability_bins(m,y,v.edge_distance(teacher)<=2)
            wrong=p!=(y==1);fp_parent=predictions.get(entry.get('fp_reference'));inherited=wrong&teacher_wrong;induced=wrong&(fp_parent==(y==1)) if fp_parent is not None else np.zeros(y.shape,bool)
            q=8*reference+4*teacher[valid].astype(int)+2*rule[valid].astype(int)+p[valid].astype(int);joint[key]+=np.bincount(q,minlength=16)
            miss=valid&(y==1)&~p&~teacher&~rule;false=valid&(y==0)&p&teacher&rule
            for label,part in parts.items():
                groups[key+(label,)]+=counts(p,y,part)
                shared[key+(label,)]+=np.array([np.count_nonzero(miss&part),np.count_nonzero(false&part),np.count_nonzero(valid&inherited&part),np.count_nonzero(valid&induced&part),np.count_nonzero(valid&inherited&induced&part),np.count_nonzero(valid&wrong&~inherited&~induced&part)])
            for label,sel in [('shared_missed_water',miss),('shared_false_water',false)]:spectra[key+(label,)]+=np.r_[rho[:,sel].sum(1,dtype=np.float64),sel.sum()]
            for label in ['shore<=2','cloud<=50','landlike_water']:
                if label not in parts:continue
                oracle=p.copy();oracle[valid&parts[label]]=y[valid&parts[label]]==1;ceilings[key+(label,)]+=counts(oracle,y)
            if not reference.any():
                n=int(valid.sum());water=int(cell[1]);dry.append(dict(dataset=row['dataset'],model=name,event=row['event'],name=row['name'],valid_pixels=n,water_pixels=water,any_water=bool(water),alarm_001=water>=.001*n,alarm_01=water>=.01*n,alarm_05=water>=.05*n))
            if row.get('same_day_sentinel2'):cells[('wf2_same_day_S2',name,row['event'],'deployed')]+=cell
            if row.get('label_satellite'):cells[(row['dataset']+'_label_'+str(row['label_satellite']),name,row['event'],'deployed')]+=cell
        for name,counters in clip_counters.items():
            for layer,values in counters.items():key=(row['dataset'],name,row['event'],layer);clipping[key][0]+=int(values[0]);clipping[key][1]+=int(values[1]);values[0]=0;values[1]=0
        if i%100==0:print(mode,slot,i,'/',len(rows),flush=True)
    out=c.ROOT/'metrics'/(mode+'_'+phase if phase and mode=='normal' else mode);out.mkdir(parents=True,exist_ok=True);keys=list(hists);np.savez_compressed(out/f'hist_slot{slot}.npz',keys=np.array([json.dumps(k) for k in keys]),histograms=np.stack([hists[k] for k in keys]) if keys else np.empty((0,4096,2),np.int64))
    pack=lambda data:[dict(key=list(k),counts=v.tolist()) for k,v in data.items()]
    c.save(out/f'slot{slot}.json',dict(outcome='completed',evaluated_models=[x['id'] for x in output_entries],windows=len(rows),seconds=time.monotonic()-start,event_counts=pack(cells),strata=pack(groups),joint=pack(joint),shared=pack(shared),spectra=pack(spectra),ceilings=pack(ceilings),calibration=pack(calibration),decode_times=pack({k:np.array(a) for k,a in decode_times.items()}),dry=dry,clipping=[dict(dataset=key[0],model=key[1],event=key[2],layer=key[3],clipped=a[0],activations=a[1]) for key,a in clipping.items()],AP='4096 fixed bins[-32,32]; monotone index scaling',shared_classes='inherited and compression-induced overlap; intersection reported explicitly; other excludes both'))
    for handle in clip_hooks:handle.remove()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['source','normal','sealed']);p.add_argument('--model');p.add_argument('--slot',type=int);p.add_argument('--phase',choices=['A','B','C']);args=p.parse_args()
    if args.mode=='source':source(next(x for x in c.read(c.ROOT/'readout_models.json')['models'] if x['id']==args.model))
    else:evaluate(args.mode,args.slot,args.phase)
if __name__=='__main__':main()
