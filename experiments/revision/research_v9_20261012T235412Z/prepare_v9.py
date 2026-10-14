"""CPU preparation. This entry point never opens WorldFloods train/val GT."""
import csv,json,math,re,sys,time,hashlib,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import common_v9 as c
import edge_v9 as e

def corrections():
    dst=c.ROOT/'wrapup_verification';dst.mkdir(exist_ok=True)
    source=c.ROOT.parent/'reviewer_wrapup_20261012'
    for name in ['verify_numbers.py','make_figures.py']:shutil.copy2(source/name,dst/name)
    with open(dst/'verification.log','w') as f:
        subprocess.run([c.PYTHON,str(dst/'verify_numbers.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    checks=c.read(dst/'verification.json');rows=[]
    for x in checks:
        rows.append(dict(claim=x['check'],original=x['printed_in_paper'],correction=x['recomputed'],status='V: re-derived',
            protocol_graph_seed=x['check'],evidence=x['raw_source'],evidence_sha256=c.sha(dst/'verification.json'),qualification=x['verdict']))
    for name,old,new,qual in [
        ('OFA auxiliary training head','omitted','11,368 parameters','R: inherited; not deployed'),
        ('Crop accounting','86 chips lose >50% water','200 chips; 73.6% removed water; 21 dry','R: existing crop audit'),
        ('Teacher spatial scale','160 m output versus R4 40 m','16-pixel tokens; teacher decoder and R4 both H/4','Architecture inspection; token size is not decoder output stride'),
        ('Vegetation spectral invisibility','invisible to these bands','low observed recall in NDVI>0.4 stratum','J: observations do not establish lack of spectral information'),
        ('G<H adaptation bound','teacher updates bounded by no-update','no universal IoU or optimization bound follows','J: retire adaptation as requested; avoid unsupported mathematical inference'),
        ('STURM development exposure','unspecified','Additional teacher-derived/OOD/saliency investigations gave null results; did not choose OFA or EfficientViT','Author-reported; exposure remains'),
        ('Four-bit lossless','lossless','paired differences and uncertainty; no equivalence established','J: seed SD is not a significance test'),
        ('OFA speed','universal speedup','budget-dependent update evidence; no demonstrated end-to-end universal speedup','V7/V8 audited masking and timing boundaries'),
        ('OOD archive calibration','allow OOD-labelled calibration','withdrawn; only S11 validation calibrates','Author instruction 13 October')]:
        rows.append(dict(claim=name,original=old,correction=new,status='R' if qual.startswith('R:') else 'J',protocol_graph_seed='see qualification',
            evidence=str(c.REPO/'prompts/prompt_v9_overnight.md'),evidence_sha256=c.sha(c.REPO/'prompts/prompt_v9_overnight.md'),qualification=qual))
    with open(c.ROOT/'revision_corrections.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prior_access_audit():
    pattern=r'WorldFloods(?:v2|Dataset).*(?:/(?:train|val)/gt|split\s*=\s*[\x22\x27](?:train|val))|WorldFloodsv2/(?:train|val)/gt|--wf-split[ =]+(?:train|val)'
    cmd=['rg','-n','--no-heading','-i',pattern,str(c.ROOT.parent),str(c.REPO/'scripts'),str(c.REPO/'evaluation'),'-g','*.py','-g','*.json','-g','*.jsonl','-g','*.log','-g','*.sh','-g','!research_v9_20261012T235412Z/**']
    out=subprocess.run(cmd,capture_output=True,text=True)
    lines=[x for x in out.stdout.splitlines() if str(c.ROOT) not in x]
    (c.ROOT/'prior_label_access_search.txt').write_text('\n'.join(lines)+'\n')
    # A recorded prior target GT hash/receipt constitutes exposure. Static generic
    # dataset implementations alone are not evidence of an executed read.
    exposures=[]
    for line in lines:
        if re.search(r'WorldFloodsv2/(train|val)/gt/',line) and re.search(r'\.(jsonl?|log):',line):
            paths=re.findall(r'/data1/[^\s\x22\x27]+/(?:train|val)/gt/[^\s\x22\x27,]+',line)
            exposures.extend(Path(x).stem for x in paths)
    c.save(c.ROOT/'prior_label_access_audit.json',dict(searched_command=cmd,exit_code=out.returncode,hits=len(lines),exposed_scenes=sorted(set(exposures)),
        qualification='Retained code/manifests/logs/receipts inspected; no historical OS file-open trace exists. Absence of a retained executed GT read is not proof of every past access.',
        probe_scripts={p.name:c.sha(p) for p in c.PROBE.glob('*.py')},probe_execution='Probe scripts instantiate WorldFloods test only; no train/val GT reads recorded.'))
    if out.returncode not in (0,1):raise RuntimeError(out.stderr)
    return set(exposures)

def footprint(path):
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Polygon
    with rasterio.open(path) as f:
        tr=f.transform;xs=[];ys=[]
        corners=[(0,0),(f.width,0),(f.width,f.height),(0,f.height),(0,0)]
        for a,b in zip(corners[:-1],corners[1:]):
            for t in range(64):
                x,y=tr*(a[0]+(b[0]-a[0])*t/64,a[1]+(b[1]-a[1])*t/64);xs.append(x);ys.append(y)
        x,y=Transformer.from_crs(f.crs,'EPSG:4326',always_xy=True).transform(xs,ys)
        return Polygon(zip(x,y)),dict(crs=f.crs.to_string(),transform=list(tr),width=f.width,height=f.height)

def date(s):
    m=re.search(r'(20\d\d)[-]?(\d\d)[-]?(\d\d)',str(s))
    return datetime(*map(int,m.groups()),tzinfo=timezone.utc).date().toordinal() if m else None

def seal():
    import rasterio,numpy as np
    from shapely.strtree import STRtree
    wf=Path('/data1/schmalstieg/flooddistill/data/WorldFloodsv2')
    exposed=prior_access_audit();pool=Path('/data1/schmalstieg/flooddistill/data/pseudo_gdacs_v2/pool_features.csv')
    geometries=[];info=[];missing=[]
    with open(pool) as f:
        for row in csv.DictReader(f):
            path=Path('/data1/schmalstieg/flooddistill/data/flood_unlabeled_v2')/row['location']/row['scene']/'B2.tif'
            dt=date(row['scene'])
            if path.exists():
                geom,_=footprint(path);geometries.append(geom);info.append(('GDACS_pool',dt,row['file']))
            else:missing.append(dict(tile=re.search(r'_T(\d\d[A-Z]{3})',row['scene']).group(1),date=dt,file=row['file']))
    s11=Path('/data1/schmalstieg/flooddistill/data/sen1floods11')
    images=sorted(s11.rglob('*_S2Hand.tif'))
    assert len(images)>=431,len(images)
    for p in images:
        geom,_=footprint(p);geometries.append(geom);info.append(('S11_hand_all_unknown_date',None,p.name))
    for p in sorted((wf/'test/S2').glob('*.tif')):
        geom,_=footprint(p);meta=c.read(wf/'test/meta'/(p.stem+'.json'));geometries.append(geom);info.append(('WF_test',date(meta.get('s2_date')),p.name))
    tree=STRtree(geometries);scenes=[];excluded=[];input_windows=0
    # Shapely 1 and 2 return different query types.
    lookup={id(g):i for i,g in enumerate(geometries)}
    for split in ['train','val']:
        for p in sorted((wf/split/'S2').glob('*.tif')):
            geom,grid=footprint(p);mp=wf/split/'meta'/(p.stem+'.json');meta=c.read(mp);dt=date(meta.get('s2_date'));reasons=[]
            for hit in tree.query(geom):
                i=int(hit) if isinstance(hit,(int,np.integer)) else lookup[id(hit)]
                src,sd,nm=info[i]
                if geom.intersects(geometries[i]) and (sd is None or dt is None or abs(dt-sd)<=3):reasons.append(dict(source=src,id=nm,days=None if sd is None or dt is None else abs(dt-sd)))
            for m in missing:
                # Conservative MGRS square bounds for missing raw pool footprints.
                try:
                    import mgrs
                    from shapely.geometry import box
                    lat,lon=mgrs.MGRS().toLatLon(m['tile']);g=box(lon-2,lat-2,lon+2,lat+2)
                    if geom.intersects(g) and (dt is None or m['date'] is None or abs(dt-m['date'])<=3):reasons.append(dict(source='GDACS_missing_footprint_conservative_MGRS',id=m['file']))
                except ImportError:raise RuntimeError('Missing pool footprints require verified MGRS fallback; cannot seal optimistically')
            if p.stem in exposed:reasons.append(dict(source='prior_reference_access',id=p.name))
            gp=wf/split/'gt'/p.name;st=gp.stat();jp=wf/split/'JRC'/p.name
            if not jp.exists():
                candidates=list((wf/split).glob('**/'+p.name));jp=next((x for x in candidates if 'jrc' in str(x).lower()),None)
            row=dict(split=split,scene=p.stem,event=str(meta.get('ems_code') or p.stem.split('_')[0]),image=str(p),reference=str(gp),reference_stat=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns),
                metadata=str(mp),metadata_sha256=c.sha(mp),grid=grid,label_satellite=meta.get('satellite','unknown'),s2_date=meta.get('s2_date'),label_date=meta.get('satellite date'),
                label_lag_days=None if date(meta.get('satellite date')) is None or dt is None else date(meta.get('satellite date'))-dt,
                same_day_sentinel2='Sentinel-2' in str(meta.get('satellite','')) and date(meta.get('satellite date'))==dt,JRC=str(jp) if jp else None)
            if reasons:excluded.append(dict(**row,reasons=reasons));continue
            windows=[]
            # Input-only validity screen does not inspect any reference pixels.
            with rasterio.open(p) as src:
                for r in range(0,src.height,512):
                    for col in range(0,src.width,512):
                        w=min(512,src.width-col);h=min(512,src.height-r)
                        x=src.read(indexes=list(range(1,14)),window=rasterio.windows.Window(col,r,w,h))
                        if np.count_nonzero(np.any(x!=0,axis=0))>.01*w*h:windows.append([col,r,w,h])
            row['input_nonzero_windows']=windows;input_windows+=len(windows);scenes.append(row)
    c.save(c.ROOT/'sealed_wf2.json',dict(version=1,created=c.q.now(),GT_content_unopened=True,reference_hash_policy='hash bytes from the single final read',
        input_windows_upper_bound=input_windows,events=len(set(x['event'] for x in scenes)),scenes=scenes,excluded=excluded,
        screening=dict(GDACS_pool_rows=23720,raw_footprints=len(info)-len(images)-18,missing_raw=len(missing),S11_hand_chips=len(images),S11_unknown_dates='conservative spatial exclusion regardless date',date_window_days=3),
        subgroup='same-calendar-day Sentinel-2 labels',protocol='Native512; all13band-zero invalid; cloud excluded; permanent water ignored; label validity >1% of native window; zero pad invalid.'))
    print('sealed metadata:',len(scenes),'scenes',input_windows,'input-only upper windows',flush=True)

def source_choice():
    data=c.read(c.PROBE/'refine_events.json')['s11val'];vals=[]
    for k in [1,2,3,5]:
        for theta in [-.1,0.,.1,.2]:
            per=[]
            for seed in c.SEEDS:
                es=[]
                for event,cells in data.items():
                    key=next(key for key in cells if key.startswith(f's{seed}|k{k}|') and abs(float(key.split('|')[-1])-theta)<1e-8)
                    es.append(e.scores(cells[key])['water_IoU'])
                per.append(sum(es)/len(es))
            vals.append(dict(k=k,theta=theta,source_equal_event_water_IoU=sum(per)/4))
    best=max(vals,key=lambda x:x['source_equal_event_water_IoU'])
    assert (best['k'],best['theta'])==(2,.1),best
    c.save(c.ROOT/'source_edge_choice.json',dict(**best,grid=vals,selection='Only S11 validation; complete four parent seeds; earliest grid tie',evidence_sha256=c.sha(c.PROBE/'refine_events.json'),receipt_sha256=c.sha(c.PROBE/'probe_counts/receipt.json')))

def edge_banks(spec):
    import torch,numpy as np
    choice=c.read(c.ROOT/'source_edge_choice.json');files=c.read(spec['index'])['files'];out=c.ROOT/'edge_labels';out.mkdir(exist_ok=True)
    # Threshold candidates are global cohort controls, never per-image fits.
    grid=np.round(np.arange(0,1.001,.01),2);change=np.zeros(len(grid),dtype=np.int64);desired=0;tot=Counter();hashes={}
    for i,name in enumerate(files):
        p=Path(spec['pseudo'])/name if 'pseudo' in spec else Path(spec['pseudo_dir'])/name
        a=torch.load(p,map_location='cpu',weights_only=True);mask=a['mask'].numpy();conf=a['confidence'].float().numpy()
        assert np.isin(mask,[0,1]).all() and np.isfinite(conf).all() and conf.min()>=.5 and conf.max()<=1
        band=e.edge_distance(mask==1)<=choice['k'];m,av,_,_=e.indices(e.reflectance(a['image'].numpy()));E1=mask.copy();E1[band&av]=(m[band&av]>choice['theta']);E2=mask.astype(np.int8);E2[band]=-1
        pw=np.where(mask==1,conf,1-conf);desired+=int(np.count_nonzero(E1!=mask));pix=pw[band];original=mask[band]
        for j,t in enumerate(grid):change[j]+=np.count_nonzero((pix>t)!=original)
        tot.update(dict(pixels=mask.size,band=int(band.sum()),E1_changed=int((E1!=mask).sum()),E1_land_to_water=int(((E1==1)&(mask==0)).sum()),E1_water_to_land=int(((E1==0)&(mask==1)).sum()),confidence_half_ties=int((conf==.5).sum())))
        torch.save(dict(E1=torch.from_numpy(E1.astype(np.int8)),E2=torch.from_numpy(E2),band=torch.from_numpy(band)),out/name)
        if i%250==0:print('edge labels',i,flush=True)
    ix=min(range(len(grid)),key=lambda j:(abs(int(change[j])-desired),abs(grid[j]-.5),grid[j]));pstar=float(grid[ix])
    for name in files:
        p=Path(spec.get('pseudo',spec.get('pseudo_dir')))/name;a=torch.load(p,map_location='cpu',weights_only=True);b=torch.load(out/name,weights_only=True);mask=a['mask'];conf=a['confidence'].float();pw=torch.where(mask==1,conf,1-conf);E3=mask.to(torch.int8).clone();E3[b['band']]=(pw[b['band']]>pstar).to(torch.int8);b['E3']=E3
        for arm in ['E1','E2','E3']:assert torch.equal(b[arm][~b['band']],mask[~b['band']].to(torch.int8))
        torch.save(b,out/name);hashes[name]=c.sha(out/name)
    c.save(c.ROOT/'edge_label_manifest.json',dict(choice=choice,pstar=pstar,threshold_grid=grid.tolist(),threshold_changed_pixels=change.tolist(),E3_changed_pixels=int(change[ix]),E1_changed_pixels=desired,
        matching='nearest total changed-pixel count; ties closest .5 then lower threshold; counts may not match exactly; directional changes differ',confidence='cached maximum-class float16; pwater=c if teacher maskwater else 1-c; strict >; ties documented',counts=dict(tot),files=hashes))

def source_images(spec):
    import torch,albumentations as A
    from data.sen1floods11 import get_dataloaders,NORM_MEAN,NORM_STD
    ds=get_dataloaders(spec['roots']['s11'],batch_size=16,num_workers=0,img_size=224,eval_img_size=512)[0].dataset
    ds.transform=A.Compose([A.Resize(224,224),A.Normalize(mean=NORM_MEAN,std=NORM_STD,max_pixel_value=1.),A.pytorch.transforms.ToTensorV2()])
    out=c.ROOT/'source_images';out.mkdir(exist_ok=True);files=[];manifest=[]
    for i in range(len(ds)):
        x=ds[i];name=f'source_{i:04d}.pt';torch.save(dict(image=x['image'],mask=x['mask'],source=str(x.get('filename',i))),out/name);files.append(name);manifest.append(dict(file=name,source=str(x.get('filename',i)),sha256=c.sha(out/name)))
    assert len(files)==252
    c.save(out/'index.json',dict(files=files,n_selected=252));c.save(c.ROOT/'source_image_manifest.json',dict(files=manifest,preprocessing='Full 512 source footprint resized224 bilinear image/nearest mask; six bands; DN*.0001; frozen S11 normalization; no augmentation during caching',paired_valid_support='human reference mask !=-1 in both arms',old_anchor_reuse=False))

def main():
    start=time.monotonic();c.install_label_guard();torch,cj,ev=c.setup();spec=c.read(c.V7/'training_spec.json');c.save(c.ROOT/'training_spec.json',spec)
    corrections();source_choice();source_images(spec);edge_banks(spec);seal()
    c.save(c.ROOT/'cpu_preparation.json',dict(outcome='completed',seconds=time.monotonic()-start,GT_unopened=True,files={p.name:c.sha(p) for p in c.ROOT.glob('*.json') if p.name!='cpu_preparation.json'}))

if __name__=='__main__':main()
