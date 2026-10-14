"""CPU readouts with exact threshold counts and declared 4096-bin AP."""
import argparse,json,time,sys,os,subprocess,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
import common_v9 as c
from policy_final02 import activate
activate()
import edge_v9 as e

TAUS=np.arange(-3,1.001,.5)
COARSE_EDGES=np.arange(-12,12.001,.5)
NAMES=['1','2','3','4-5','6-10','11-20','>20']

def groups(mask,rho,cloud):
    valid=mask!=-1;water=mask==1;land=mask==0
    dw=ndi.distance_transform_edt(~land) if land.any() else np.full(mask.shape,np.inf)
    dl=ndi.distance_transform_edt(~water) if water.any() else np.full(mask.shape,np.inf)
    dist=np.where(water,dw,dl);ed=np.searchsorted([1,2,3,5,10,20],dist,side='left').astype(np.uint8)
    lab,n=ndi.label(water,np.ones((3,3),bool));mx=ndi.maximum(dw,lab,index=np.arange(1,n+1)) if n else[];half=np.r_[0,mx][lab];thin=np.where(water,np.searchsorted([2,5],half,side='left')+1,0).astype(np.uint8)
    vis=rho[:3].mean(0);ndvi=(rho[3]-rho[2])/np.maximum(rho[3]+rho[2],1e-6)
    spectral=((vis<.06)&(rho[3]<.08)&(rho[4]<.06)).astype(np.uint8)+2*(vis>.20).astype(np.uint8)+4*(ndvi>.4).astype(np.uint8)
    out=[('reference_edge',ed,[f'reference_edge_{x}' for x in NAMES]),('halfwidth',thin,['nonwater','water_halfwidth_<=2','water_halfwidth_2-5','water_halfwidth_>5']),('spectral',spectral,[f'spectral_combination_{i}' for i in range(8)])]
    for prefix,cl in [('input_cloud',e.cloud_test(rho)),('reference_cloud',cloud)]:
        if cl is None:continue
        d=ndi.distance_transform_edt(~cl) if cl.any() else np.full(mask.shape,np.inf);code=np.searchsorted([5,20,50],d,side='left').astype(np.uint8)
        out.append((prefix,code,[prefix+'_'+x for x in ['<=5','6-20','21-50','>50']]))
    return [(prefix,arr[valid],names) for prefix,arr,names in out]

def hist2(margin,y,bins=4096):
    idx=np.clip(np.floor((margin+32)*64),0,bins-1).astype(np.int32)
    return np.bincount(idx*2+y,minlength=bins*2).reshape(bins,2).astype(np.int64)

def threshold_counts(margin,y):
    b=np.searchsorted(TAUS,margin,side='left')
    h=np.bincount(b*2+y,minlength=(len(TAUS)+1)*2).reshape(len(TAUS)+1,2)
    lo=np.cumsum(h,axis=0)[:-1];hi=h.sum(0)-lo
    return np.stack([hi[:,1],hi[:,0],lo[:,1],lo[:,0]],axis=1)

def coarse(margin,mnd,avail,mask):
    mb=np.searchsorted(COARSE_EDGES,margin,side='left');mi=np.clip(np.floor((mnd+1)*10),0,19).astype(np.int32);mi[~avail]=20
    ys=np.where(mask==-1,0,mask+1);H=np.bincount(((mb*21+mi)*3+ys).reshape(-1),minlength=50*21*3).reshape(50,21,3)
    return H

def ap(hist):
    z=hist[::-1];tp=np.cumsum(z[:,1]);fp=np.cumsum(z[:,0]);total=tp[-1]
    if total==0:return None
    return float(np.sum(z[:,1]/total*np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0)))

def equal(ev):
    vals=[e.scores(x)['water_IoU'] for x in ev.values()];vals=[x for x in vals if x is not None];return float(np.mean(vals)) if vals else None

def rule_module():
    import importlib.util
    spec=importlib.util.spec_from_file_location('v9_source_rules',c.PROBE/'probe_oprules.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.EDGES=COARSE_EDGES;return m

def calibrate(H,meta):
    """Fit temperature, prior and both rule grids on SOURCE labels only."""
    r=rule_module();src=r.source_calibration(H[:,None],['model'])['model'];centers=np.r_[COARSE_EDGES[0]-.25,(COARSE_EDGES[:-1]+COARSE_EDGES[1:])/2,COARSE_EDGES[-1]+.25]
    candidates={
        'em':[dict(lo=lo,hi=hi) for lo in [-4.,-2.,-1.] for hi in [0.,2.,4.]],
        'anchor':[dict(a=a,lo=lo,min_anchor=minimum) for a in [.2,.3,.4,.5] for lo in [-4.,-2.] for minimum in [100,1000]]}
    chosen={}
    for rule,grid in candidates.items():
        scores=[]
        for params in grid:
            events=defaultdict(lambda:np.zeros(4,np.int64))
            for h,row in zip(H,meta):
                b=r.rule_tau(rule,params,h.sum(2),centers,src);events[row['event']]+=r.counts_at(h.sum(1),b)
            scores.append(equal(events))
        index=max(range(len(grid)),key=lambda j:scores[j]);chosen[rule]=dict(params=grid[index],source_equal_event_water_IoU=scores[index])
    return dict(temperature_prior=src,rules=chosen,source='S11 validation only; inherited probe grid and bin semantics')

def worker(mode,slot):
    torch,cj,ev=c.setup();c.install_label_guard();start=time.monotonic();spec=c.read(c.ROOT/'training_spec.json');entries=c.read(c.ROOT/'final_models.json')['models']
    models=[dict(id='teacher',graph='teacher',method='teacher',seed=None)]+entries+[dict(id='MNDWI',graph='rule',method='rule',seed=None),dict(id='NDWI',graph='rule',method='rule',seed=None)]
    models=models[slot::2];device='cpu';loaded={x['id']:c.load_model(x,device) for x in models if x['graph'] not in ['B0','teacher','rule']}
    wm=c.read(c.ROOT/('sealed_windows.json' if mode=='sealed' else 'normal_windows.json'));windows=wm['rows'] if mode=='sealed' else wm['windows'];windows=[x for x in windows if not x['omit'] and (mode!='source' or x['dataset']=='s11val')]
    predpart='sealed_tf32_off' if mode=='sealed' else'normal_tf32_off';destination=c.ROOT/'metrics'/mode;destination.mkdir(parents=True,exist_ok=True)
    # Every model has independent event, scene and window counts; seeds are not pooled.
    counts=defaultdict(lambda:np.zeros(4,np.int64));strat=defaultdict(lambda:np.zeros(4,np.int64));hists=defaultdict(lambda:np.zeros((4096,2),np.int64));sceneH=defaultdict(lambda:np.zeros((50,21,3),np.int64));dry=[];windowH=defaultdict(list);windowmeta=defaultdict(list);joint=defaultdict(lambda:np.zeros(16,np.int64));unavailable=set()
    choice=c.read(c.ROOT/'source_edge_choice.json');settings=c.read(c.ROOT/'source_operating_points.json',{});decode_seconds=defaultdict(float)
    for wi,row in enumerate(windows):
        with np.load(row['path']) as a:x=a['image'].copy();mask=a['mask'];cl=a['cloud'];cloud=cl if cl.size else None
        rho=e.reflectance(x);mnd,available,ndwi,ndavail=e.indices(rho);valid=mask!=-1;y=mask[valid].astype(np.int64);parts=groups(mask,rho,cloud);meta=dict(dataset=row['dataset'],event=row['event'],map=row['map'],name=row['name'],same_day_sentinel2=row.get('same_day_sentinel2',False))
        trpath=c.DATA/'gpu_predictions'/predpart/'teacher'/(Path(row['path']).stem+'.npy');teacher=np.load(trpath)>0 if trpath.exists() else None
        for entry in models:
            name=entry['id'];key=(row['dataset'],name,row['event']);variants={}
            if entry['graph']=='rule':margin=(mnd if name=='MNDWI' else ndwi)*32;pred=((mnd>0)&available) if name=='MNDWI' else ((ndwi>0)&ndavail)
            elif entry['graph'] in ['B0','teacher']:
                path=c.DATA/'gpu_predictions'/predpart/name/(Path(row['path']).stem+'.npy')
                if not path.exists():unavailable.add(name);continue
                margin=np.load(path);pred=margin>0
            else:
                with torch.no_grad():z=loaded[name](torch.from_numpy(x)[None]);cj.finite_output(loaded[name],(torch.from_numpy(x)[None],),z);margin=(z[0,1]-z[0,0]).numpy();pred=margin>0
            base=e.counts(pred,mask);variants['deployed']=base
            if entry['graph']!='rule':
                tg=threshold_counts(margin[valid],y)
                for tau,cell in zip(TAUS,tg):variants[f'tau={float(tau):g}']=cell
                tic=time.perf_counter();decoded=e.decode(pred,mnd,available,choice['k'],choice['theta']);decode_seconds[name]+=time.perf_counter()-tic;variants['D']=e.counts(decoded,mask)
                h,w=row.get('native_shape',[512,512]);H=coarse(margin[:h,:w],mnd[:h,:w],available[:h,:w],mask[:h,:w]);sceneH[(row['dataset'],name,row['map'])]+=H;windowH[name].append(H.sum(1));windowmeta[name].append(meta)
                if teacher is not None:
                    j=(8*y+4*teacher[valid].astype(int)+2*((mnd>0)&available)[valid].astype(int)+pred[valid].astype(int));joint[key]+=np.bincount(j,minlength=16)
            for variant,cell in variants.items():
                counts[key+(variant,)]+=cell
                if row.get('same_day_sentinel2'):counts[('wf2_same_day_S2',name,row['event'],variant)]+=cell
            hists[key]+=hist2(margin[valid],y)
            q=2*(y==0)+(~pred[valid]).astype(np.int8)
            for prefix,codes,names in parts:
                cells=np.bincount(codes.astype(np.int32)*4+q,minlength=len(names)*4).reshape(len(names),4)[:,[0,2,1,3]]
                for label,cell in zip(names,cells):strat[(row['dataset'],name,label)]+=cell
                if prefix=='spectral':
                    for bit,label in [(1,'spectral_dark'),(2,'spectral_bright'),(4,'spectral_vegetated')]:strat[(row['dataset'],name,label)]+=cells[[j for j in range(8) if j&bit]].sum(0)
                    strat[(row['dataset'],name,'spectral_other')]+=cells[0]
            if not y.any():
                n=len(y)
                for variant,cell in variants.items():
                    if variant in ['deployed','D'] or variant.startswith('tau='):dry.append(dict(**meta,model=name,variant=variant,valid_pixels=n,water_pixels=int(cell[1]),any_water=bool(cell[1]),alarm_001=bool(cell[1]>=.001*n),alarm_01=bool(cell[1]>=.01*n),alarm_05=bool(cell[1]>=.05*n)))
        if wi%100==0:print(mode,'slot',slot,wi,'/',len(windows),flush=True)
    # Label-free rules are selected on source and applied per complete scene.
    rules=rule_module();centers=np.r_[COARSE_EDGES[0]-.25,(COARSE_EDGES[:-1]+COARSE_EDGES[1:])/2,COARSE_EDGES[-1]+.25]
    source={}
    for entry in models:
        name=entry['id']
        if entry['graph']=='rule' or name in unavailable or not windowH[name]:continue
        if mode=='source':
            Hs=np.stack([sceneH[('s11val',name,row['map'])] for row in windowmeta[name]])
            selected=calibrate(Hs,windowmeta[name]);eqs={tau:equal({event:cell for (ds,n,event,v),cell in counts.items() if ds=='s11val' and n==name and v==f'tau={float(tau):g}'}) for tau in TAUS}
            tau=max(TAUS,key=lambda t:eqs[t] if eqs[t] is not None else-1);selected.update(tau=float(tau),source_tau_grid={str(t):v for t,v in eqs.items()});source[name]=selected
        else:selected=settings.get(name)
        if selected:
            for hc,row in zip(windowH[name],windowmeta[name]):
                ds=row['dataset'];mp=row['map'];H=sceneH[(ds,name,mp)]
                for rule in ['em','anchor']:
                    b=rules.rule_tau(rule,selected['rules'][rule]['params'],H.sum(2),centers,selected['temperature_prior']);cell=rules.counts_at(hc,b);counts[(ds,name,row['event'],rule)]+=cell
                    if row.get('same_day_sentinel2'):counts[('wf2_same_day_S2',name,row['event'],rule)]+=cell
                    n=int(hc[:,1:].sum())
                    if n and hc[:,2].sum()==0:dry.append(dict(**row,model=name,variant=rule,valid_pixels=n,water_pixels=int(cell[1]),any_water=bool(cell[1]),alarm_001=bool(cell[1]>=.001*n),alarm_01=bool(cell[1]>=.01*n),alarm_05=bool(cell[1]>=.05*n)))
    # Partial prediction coverage is not a comparable full-set model result.
    counts={k:v for k,v in counts.items() if k[1] not in unavailable};strat={k:v for k,v in strat.items() if k[1] not in unavailable};hists={k:v for k,v in hists.items() if k[1] not in unavailable};dry=[x for x in dry if x['model'] not in unavailable]
    histkeys=list(hists);np.savez_compressed(destination/f'hist_slot{slot}.npz',histograms=np.stack([hists[k] for k in histkeys]) if histkeys else np.empty((0,4096,2),np.int64),keys=np.array([json.dumps(k) for k in histkeys]))
    c.save(destination/f'slot{slot}.json',dict(outcome='completed',seconds=time.monotonic()-start,models=[x['id'] for x in models],unavailable=sorted(unavailable),
        event_counts=[dict(dataset=ds,model=n,event=event,variant=v,counts=cell.tolist()) for (ds,n,event,v),cell in counts.items()],
        strata=[dict(dataset=ds,model=n,stratum=s,counts=cell.tolist()) for (ds,n,s),cell in strat.items()],dry=dry,
        joint=[dict(dataset=ds,model=n,event=event,bits='reference*8+teacher*4+MNDWI*2+student',counts=cell.tolist()) for (ds,n,event),cell in joint.items()],
        source_settings=source,decode_cpu_seconds=dict(decode_seconds),AP_protocol='4096 fixed margin bins over [-32,32], clipped overflow; approximate tied-bin AP; rules scaled by32; monotone logits rank',
        strata_protocol='Euclidean opposite known class distance; contiguous bins; component maximum inscribed halfwidth, not local branch width; spectral strata overlap'))

def run(mode):
    ps=[subprocess.Popen([sys.executable,str(Path(__file__)),mode,'--slot',str(i)],env={**os.environ,'CUDA_VISIBLE_DEVICES':''}) for i in range(2)]
    codes=[p.wait() for p in ps]
    if any(codes):raise RuntimeError('CPU evaluator worker failure '+str(codes))
    if mode=='source':
        settings={}
        for i in range(2):settings.update(c.read(c.ROOT/'metrics/source'/f'slot{i}.json')['source_settings'])
        c.save(c.ROOT/'source_operating_points.json',settings)
    c.save(c.ROOT/f'evaluation_{mode}.json',dict(outcome='completed',slots=[str(c.ROOT/'metrics'/mode/f'slot{i}.json') for i in range(2)],sealed_GT_paths_unopened=True,completed_at=c.q.now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['source','normal','sealed']);p.add_argument('--slot',type=int);a=p.parse_args()
    if a.slot is None:run(a.mode)
    else:worker(a.mode,a.slot)
