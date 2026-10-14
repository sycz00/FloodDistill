"""Registered native inputs/reference caches; sealed references are read once here."""
import time,os,json
from pathlib import Path
import common_v9 as c
from policy_final02 import activate
activate()

def normal(spec):
    import torch,numpy as np
    from evaluation import revision_native_border as rb
    from data.channel_adapter import sturm_s2_adapter
    torch,cj,ev=c.setup();sets,maps,events=ev.datasets(spec);val,_=cj.loaders(spec)
    sets=[('s11val',val.dataset,4)]+sets
    for name,ds,bs in sets:
        for i in range(len(ds)):
            a=rb.native_item(ds,i) if name=='wf' else ds[i];x=a['image'].float();mask=np.asarray(a['mask']).copy();fn=Path(a.get('filename',str(ds.image_files[i]) if name=='s11val' else '')).name
            cloud=None
            if name=='sturm':
                x=sturm_s2_adapter(target_channels=6)(x[None])[0];_,invalid=rb.sturm_invalid(ds,i);mask[invalid]=-1;mp=maps[fn];event=events[fn];omit=fn in spec['excluded_sturm_chips']
            elif name=='wf':
                mp=fn.split('#')[0];event=mp.split('_')[0];omit=mp in spec['excluded_wf_maps']
                import rasterio
                ip,lp,w=ds.samples[i]
                with rasterio.open(lp) as f:cloud=f.read(1,window=rasterio.windows.Window(*w))==2
            else:mp=fn;event=fn.split('_')[0];omit=False
            native_shape=list(mask.shape)
            if mask.shape!=(512,512):mask=np.pad(mask,((0,512-mask.shape[0]),(0,512-mask.shape[1])),constant_values=-1)
            if cloud is not None and cloud.shape!=(512,512):cloud=np.pad(cloud,((0,512-cloud.shape[0]),(0,512-cloud.shape[1])))
            yield dict(dataset=name,index=i,name=fn,map=mp,event=event,omit=omit,image=x.numpy(),mask=mask.astype(np.int8),cloud=cloud,same_day_sentinel2=False,native_shape=native_shape)

def write_normal():
    import numpy as np
    start=time.monotonic();c.install_label_guard();c.setup();spec=c.read(c.ROOT/'training_spec.json');rows=[];out=c.DATA/'normal_cache';out.mkdir(parents=True,exist_ok=True)
    for a in normal(spec):
        name=f"{a['dataset']}_{a['index']:05d}.npz";p=out/name
        # Float32 inputs preserve the registered loader exactly.
        np.savez(p,image=a.pop('image'),mask=a.pop('mask'),cloud=a.pop('cloud') if a['cloud'] is not None else np.empty((0,),bool))
        rows.append(dict(**a,path=str(p)))
        if len(rows)%250==0:print('normal cache',len(rows),flush=True)
    c.save(c.ROOT/'normal_windows.json',dict(windows=rows,seconds=time.monotonic()-start,counts={s:sum(x['dataset']==s for x in rows) for s in ['s11val','s11','sturm','wf']}))

def freeze_models():
    models=[x for x in c.read(c.V7/'reuse_models.json') if x['graph']=='R4']
    for seed in c.SEEDS:
        r=c.read(c.V7/'receipts'/f'R4_CE_s{seed}.json');models.append(dict(id=f'E0_s{seed}',graph='R4',seed=seed,method='E0',checkpoint=r['best_path'],sha256=r['best_sha256']))
    for arm in ['E1','E2','E3','human','teacher']:
        for seed in c.SEEDS:
            r=c.read(c.ROOT/'scientific_receipts'/f'{arm}_s{seed}.json')
            if r and r.get('outcome')=='completed':models.append(dict(id=f'{arm}_s{seed}',graph='B0' if arm in ['human','teacher'] else 'R4',seed=seed,method=arm,checkpoint=r['best_path'],sha256=r['best_sha256']))
    inv=c.read(c.REPO/'experiments/revision/prompt4_original_checkpoint_inventory.json')
    runs=inv.get('runs',[]) if isinstance(inv,dict) else inv
    for x in runs:
        original_id=x.get('original_id',x.get('id',''))
        if 'O4_original_efficientvit_b0_ofa_N2500' in original_id:
            ck=x.get('checkpoint',x.get('path',x.get('checkpoint_path')))
            if ck and Path(ck).exists():models.append(dict(id=original_id,graph='B0',seed=x.get('seed'),method='paper_ofa',checkpoint=ck,sha256=c.sha(ck),qualification='historical original checkpoint; training recipe not a matched new control'))
    for seed in c.SEEDS:
        qp=c.ROOT/'ptq_states'/f'q_s{seed}.pt'
        if qp.exists():models.append(dict(id=f'PTQ_s{seed}',graph='R4',seed=seed,method='PTQ',quantized=True,checkpoint=str(qp),sha256=c.sha(qp)))
    for x in models:assert c.sha(x['checkpoint'])==x['sha256']
    c.save(c.ROOT/'final_models.json',dict(frozen_at=c.q.now(),models=models,source_only_selection=True,sealed_manifest_sha256=c.sha(c.ROOT/'sealed_wf2.json'),code={p.name:c.sha(p) for p in c.ROOT.glob('*.py')}))
    return models

def write_sealed():
    import rasterio,numpy as np
    once=c.install_label_guard(final=True);start=time.monotonic();manifest=c.read(c.ROOT/'sealed_wf2.json');models=c.read(c.ROOT/'final_models.json');assert models['sealed_manifest_sha256']==c.sha(c.ROOT/'sealed_wf2.json')
    assert not (c.ROOT/'label_access.jsonl').exists(),'Sealed stage cannot reopen references on retry'
    settings=c.read(c.ROOT/'source_operating_points.json')
    assert settings,'Source operating points must be frozen before final GT access'
    c.save(c.ROOT/'final_evaluation_freeze.json',dict(frozen_at=c.q.now(),models_sha256=c.sha(c.ROOT/'final_models_final02.json'),source_settings_sha256=c.sha(c.ROOT/'source_operating_points_final02.json'),sealed_manifest_sha256=c.sha(c.ROOT/'sealed_wf2.json')))
    out=c.DATA/'sealed_cache';out.mkdir(parents=True,exist_ok=True);rows=[]
    for scene in manifest['scenes']:
        p=Path(scene['reference']);st=p.stat();assert dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==scene['reference_stat']
        raw=once(p);assert raw.shape[0]==2
        with rasterio.open(scene['image']) as src:
            assert raw.shape[1:]==(src.height,src.width)
            jrc=rasterio.open(scene['JRC']) if scene['JRC'] else None
            if jrc is None:raise RuntimeError('Permanent-water JRC mask missing; no silent protocol substitution: '+scene['scene'])
            try:
                assert src.crs==jrc.crs and src.transform==jrc.transform and (src.width,src.height)==(jrc.width,jrc.height)
                for window in scene['input_nonzero_windows']:
                    col,r,w,h=window;win=rasterio.windows.Window(*window);allbands=src.read(indexes=list(range(1,14)),window=win)
                    cloud=raw[0,r:r+h,col:col+w]==2;lw=raw[1,r:r+h,col:col+w];permanent=jrc.read(1,window=win)==3
                    mask=np.where(lw>=1,lw-1,-1).astype(np.int8);mask[cloud|permanent|np.all(allbands==0,axis=0)]=-1
                    assert np.isin(mask,[-1,0,1]).all()
                    if np.count_nonzero(mask!=-1)<=.01*w*h:continue
                    rho=np.clip(allbands[[1,2,3,8,11,12]].astype(np.float32)*.0001,0,1)
                    import edge_v9 as e
                    x=(rho-e.MEAN[:,None,None])/e.STD[:,None,None];x=np.pad(x,((0,0),(0,512-h),(0,512-w)),constant_values=0)
                    mask=np.pad(mask,((0,512-h),(0,512-w)),constant_values=-1);cloud=np.pad(cloud,((0,512-h),(0,512-w)))
                    idx=len(rows);fp=out/f'wf2_{idx:05d}.npz';np.savez(fp,image=x,mask=mask,cloud=cloud)
                    rows.append(dict(dataset='wf2',index=idx,name=f"{scene['scene']}#r{r}_c{col}",map=scene['scene'],event=scene['event'],omit=False,path=str(fp),same_day_sentinel2=scene['same_day_sentinel2'],label_satellite=scene['label_satellite'],label_lag_days=scene['label_lag_days'],native_shape=[h,w]))
            finally:jrc.close()
        print('sealed final reference read',scene['scene'],'retained',len(rows),flush=True)
        c.save(c.ROOT/'sealed_windows_progress.json',dict(rows=rows,reference_scene_last=scene['scene'],window_count=len(rows),finished=False))
    c.save(c.ROOT/'sealed_windows.json',dict(rows=rows,seconds=time.monotonic()-start,reference_files_opened=len(manifest['scenes']),window_count=len(rows),events=len(set(x['event'] for x in rows)),opened_once=True))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='normal':write_normal()
    elif sys.argv[1]=='freeze':freeze_models()
    elif sys.argv[1]=='sealed':write_sealed()
