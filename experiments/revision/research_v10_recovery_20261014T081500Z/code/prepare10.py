"""Read-only historical inventory and TRAIN-only statistics/cohort preparation."""
import csv,random,time
from pathlib import Path
import common10 as c

def main():
    torch,cj,old,v=c.setup();spec=c.read(c.V9/'training_spec.json');items={}
    def add(entry):
        if 'N251' in entry['checkpoint']:raise AssertionError('Obsolete checkpoint')
        assert c.sha(entry['checkpoint'])==entry['sha256']
        items[entry['id']]=entry
    for entry in c.read(c.V9/'final_models_final03.json')['models']:
        if not entry.get('quantized'):add(dict(entry,origin='V9/V7 frozen inventory',matched_new_training=False))
    source=c.V5/'v5/results'
    for graph in ['R1','R2','R3','A0']:
        for seed in c.SEEDS:
            choices=[source/f'fl_{graph}_sel_s{seed}.json',source/f'm_lkd_{graph}_s{seed}.json',source/f'm_ofa_{graph}_s{seed}.json']
            for p in choices:
                if not p.exists():continue
                a=c.read(p);checkpoint=a.get('best_path')
                if checkpoint and Path(checkpoint).exists():add(dict(id=p.stem,graph=graph,seed=seed,method=a.get('method','lkd'),checkpoint=checkpoint,sha256=a['best_sha256'],origin=str(p),completed_updates=a.get('completed_updates'),matched_new_training=False))
    for graph,prefix in [('A0','m_sup_A0'),('A3','m_sup_A3'),('R2','mf_sup')]:
        for seed in c.SEEDS:
            p=source/f'{prefix}_s{seed}.json';a=c.read(p);assert a and a['graph']==graph;add(dict(id=f'{graph}_SUP_legacy_s{seed}',graph=graph,seed=seed,method='SUP',checkpoint=a['best_path'],sha256=a['best_sha256'],origin=str(p),completed_updates=a.get('completed_updates'),matched_new_training=False))
    for seed in c.SEEDS:
        p=c.REPO/f'runs/original_runs/V2/gt_s11_matched/sen_ce_N252_s{seed}/best.pt';assert p.exists();add(dict(id=f'B0_SUP_paper_s{seed}',graph='B0',seed=seed,method='SUP',checkpoint=str(p),sha256=c.sha(p),origin='TableI N252 human',matched_new_training=False))
    for seed in c.SEEDS:
        for graph,p in [('A0',c.V5/'results'/f'A0_FP_s{seed}.json'),('R2',source/f'mf_ce_s{seed}.json')]:
            a=c.read(p)
            if a and a.get('best_path') and Path(a['best_path']).exists():add(dict(id=f'{graph}_'+('LKD_stage2' if graph=='A0' else 'CE_legacy')+f'_s{seed}',graph=graph,seed=seed,method='lkd' if graph=='A0' else 'CE',checkpoint=a['best_path'],sha256=a['best_sha256'],origin=str(p),completed_updates=a.get('completed_updates'),matched_new_training=False))
    c.save(c.ROOT/'historical_models.json',list(items.values()))
    files=c.read(spec['index'])['files'];cohort=set(files);poolpath=Path('/data1/schmalstieg/flooddistill/data/pseudo_gdacs_v2');pool=list(csv.DictReader(open(poolpath/'pool_features.csv')));byname={x['file']:x for x in pool}
    pre=[x for x in pool if x['file'] not in cohort and x['location'].startswith('GDACS_') and x['is_flood_scene']=='0' and (poolpath/x['file']).exists()]
    pre.sort(key=lambda x:x['file']);rng=random.Random(1729);heldout=rng.sample(pre,500);heldnames={x['file'] for x in heldout}
    c.save(c.ROOT/'heldout_pre_event_pool.json',dict(files=[str(poolpath/x['file']) for x in heldout],criteria='GDACS location; is_flood_scene0; outside originalN2500; deterministic1729; never target labels',pool_features_sha256=c.sha(poolpath/'pool_features.csv')))
    hard=[x for x in pre if x['file'] not in heldnames and float(x['flood_frac'])<.01 and float(x['mndwi_water_frac'])>=.05]
    easy=[x for x in pre if x['file'] not in heldnames and float(x['flood_frac'])<.01 and float(x['mndwi_water_frac'])<.01]
    removed=set(rng.sample(files,250));negative={}
    for name,candidates in [('hard',hard),('easy',easy)]:
        if len(candidates)<250:
            negative[name]=dict(outcome='unavailable',reason=f'Only{len(candidates)} eligible pre-event candidate tiles');continue
        candidates.sort(key=lambda x:x['file']);selected=rng.sample(candidates,250);new=[x for x in files if x not in removed]+[x['file'] for x in selected];out=c.DATA/'negative_cohorts'/name;out.mkdir(parents=True,exist_ok=True)
        for file in new:
            original=Path(spec['pseudo'])/file if file in cohort else poolpath/file
            target=out/file
            if not target.exists():target.symlink_to(original)
        index=out/'index.json';c.save(index,dict(files=new,n_selected=2500));negative[name]=dict(outcome='completed',index=str(index),pseudo=str(out),removed=sorted(removed),added=[x['file'] for x in selected],heldout_disjoint=True)
    c.save(c.ROOT/'negative_cohorts.json',negative)
    total=torch.zeros(6,dtype=torch.float64);squares=total.clone();pixels=0
    for file in files:
        x=torch.load(Path(spec['pseudo'])/file,weights_only=True)['image'].float()[None];log=v.log_reflectance(x).double()[0].reshape(6,-1);total+=log.sum(1);squares+=(log*log).sum(1);pixels+=log.shape[1]
    mean=total/pixels;std=(squares/pixels-mean.square()).clamp_min(1e-8).sqrt();c.save(c.ROOT/'log_train_statistics.json',dict(mean=mean.tolist(),std=std.tolist(),pixels=pixels,source='OriginalN2500 TRAIN images only',index_sha256=c.sha(spec['index'])))
    scores={f:sum(float(byname[f][f'band_mean_{i}']) for i in range(3))/3 for f in files}
    selected64=list(c.read(c.V9/'ptq_manifest.json')['records'][0]['calibration_indices']);names=[files[i] for i in selected64]
    for candidates in [sorted(files,key=lambda f:scores[f]),sorted(files,key=lambda f:-scores[f]),sorted(files,key=lambda f:-float(byname[f].get('patch_cloud_prob') or 0))]:
        for file in candidates:
            if file not in names:names.append(file)
            if len(names)%16==0 and len(names)>16:break
    names=names[:64];assert len(names)==64
    c.save(c.ROOT/'calibration64.json',dict(files=names,criteria='16 adopted IDs plus48 TRAIN tiles stratified dark/bright/cloud source features; no target labels'))
    c.save(c.ROOT/'preparation.json',dict(outcome='completed',models=len(items),source_images=252,GDACS_images=2500,heldout_tiles=500,negative_candidates=dict(hard=len(hard),easy=len(easy)),historical_artifacts_immutable=True,N251_artifacts_read=False))
if __name__=='__main__':main()
