"""Finite source-only decisions and variant construction for the frozen V10 matrix."""
import argparse,json,random,time,copy
from pathlib import Path
import numpy as np
import common10 as c

def entry_from_receipt(path):
    a=c.read(path);return dict(id=path.stem,graph=a['graph'],method=a['method'],seed=a['seed'],checkpoint=a['best_path'],sha256=a['best_sha256'],parameters=a['parameters'],completed_updates=a['completed_updates'],matched_new_training=True,sealed=True,quantized=a.get('quantized',False),precision=a.get('precision') or 'FP32',fp_reference=a.get('fp_reference'),control=a.get('control'),cohort=a.get('cohort'))

def inventory():
    spec=c.read(c.V9/'training_spec.json');items={x['id']:x for x in c.read(c.ROOT/'historical_models.json')}
    items['teacher']=dict(id='teacher',graph='teacher',method='teacher',seed=None,checkpoint=spec['teacher'],sha256=spec['teacher_sha256'],v9_cache_id='teacher',sealed=True)
    for x in items.values():
        if x['id'] in {a['id'] for a in c.read(c.V9/'final_models_final03.json')['models'] if a['graph']=='B0'}:x['v9_cache_id']=x['id']
        x.setdefault('sealed',x['graph'] not in ['R1','R3','A3N'])
    for p in sorted((c.ROOT/'scientific_receipts').glob('*.json')):
        a=c.read(p)
        if a.get('outcome')=='completed':items[p.stem]=entry_from_receipt(p)
    for p in sorted((c.ROOT/'quantized_entries').glob('*.json')):
        x=c.read(p);items[x['id']]=x
    for name in ['MNDWI','WI2015','AWEIsh']:items[name]=dict(id=name,graph='rule',method='rule',seed=None,sealed=True)
    previous={x['id']:x for x in c.read(c.ROOT/'readout_models.json',{}).get('models',[])}
    for name,entry in items.items():
        old=previous.get(name,{})
        if old.get('sha256')==entry.get('sha256'):
            for key in ['parameters','FP32_bytes','MACs_per_pixel','MAC_scope','packed_bytes','quantizer_metadata_bytes']:
                if key in old:entry[key]=old[key]
    c.save(c.ROOT/'readout_models.json',dict(models=list(items.values()),created=time.time(),source_only_decisions=True))
    return items

def source_all(slot=None):
    from evaluate10 import source
    items={x['id']:x for x in c.read(c.ROOT/'readout_models.json')['models']}
    for i,(name,x) in enumerate(sorted(items.items())):
        if slot is not None and i%3!=slot:continue
        if x['graph']=='rule' or x['graph']=='teacher' or x.get('source_reference'):continue
        path=c.ROOT/'source_metrics'/(name+'.json');old=c.read(path)
        if old and old.get('checkpoint_sha256')==x['sha256']:continue
        source(x)

def choose_smallest():
    torch,cj,old,v=c.setup();items=inventory();sizes=['R2','R3','R4_micro','R4'];base=[]
    for seed in c.SEEDS:
        source=c.read(c.ROOT/'source_metrics'/f'R4_KDoff_s{seed}.json');assert source, 'A5 KDoff source rows required'
        base.append(source['equal_event_water_IoU'])
    teacher=old.teacher_cpu(c.read(c.V9/'training_spec.json'));models={f'{g}_KDoff_s{s}':c.load(items[f'{g}_KDoff_s{s}']) for g in sizes for s in c.SEEDS};stats={name:[0,0] for name in models}
    with torch.no_grad():
        for path in c.read(c.ROOT/'heldout_pre_event_pool.json')['files']:
            a=torch.load(path,weights_only=True);x=a['image'].float()[None];p=torch.softmax(teacher(x),1)[0,1].numpy();rho=v.reflectance(x[0].numpy());dark=(rho[:3].mean(0)<.06)&(rho[3]<.08)&(rho[4]<.06)&(p<.1)&np.any(rho!=0,axis=0)
            for name,model in models.items():
                z=model(x);pred=(z[0,1]>z[0,0]).numpy();stats[name][0]+=int((pred&dark).sum());stats[name][1]+=int(dark.sum())
    rates={name:x[0]/x[1] if x[1] else None for name,x in stats.items()};ref=[rates[f'R4_KDoff_s{s}'] for s in c.SEEDS];assert all(x is not None for x in ref),'No teacher-referenced dark pixels; viability undefined'
    rows=[];selected='R4'
    for graph in sizes:
        vals=[c.read(c.ROOT/'source_metrics'/f'{graph}_KDoff_s{s}.json')['equal_event_water_IoU'] for s in c.SEEDS];alarms=[rates[f'{graph}_KDoff_s{s}'] for s in c.SEEDS];qualifies=np.mean(vals)>=np.mean(base)-.01 and np.mean(alarms)<=2*np.mean(ref)
        rows.append(dict(graph=graph,source_equal_event_water_IoU=float(np.mean(vals)),teacher_referenced_dark_false_water=float(np.mean(alarms)),qualifies=bool(qualifies)))
    selected=next(x['graph'] for x in rows if x['qualifies']);c.save(c.ROOT/'smallest_viable_selection.json',dict(outcome='completed',graph=selected,rows=rows,source_only=True,heldout_pool_sha256=c.sha(c.ROOT/'heldout_pre_event_pool.json'),sealed_metrics_used=False))

def frontier():
    torch,cj,old,v=c.setup();from quant10 import prepare
    from models.compression_quant import precision_map
    items=inventory();spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=1729),augment=False);ids=c.read(c.V9/'ptq_manifest.json')['records'][0]['calibration_indices'];batches=[dict(image=ds[i]['image'][None].float()) for i in ids];created=[]
    parents=[x for x in items.values() if x['graph'] in ['R1','R2','R3','R4','A0','A3','A3N'] and x['method']=='lkd' and not x.get('quantized')]
    # Four retained seeds are required per rung; do not silently substitute another objective.
    for graph in ['R1','R2','R3','R4','A0','A3','A3N']:
        group=[x for x in parents if x['graph']==graph]
        if {x['seed'] for x in group}!=set(c.SEEDS):c.save(c.ROOT/'unavailable'/f'frontier_{graph}.json',dict(outcome='unavailable',reason='No verified complete four-seed LKD parent group'));continue
        for parent in group:
            for precision in ['W8A8','W4A8','W4A4']:
                name=parent['id']+'__'+precision;meta=c.ROOT/'quantized_entries'/(name+'.json')
                if meta.exists():continue
                q=prepare(c.load(parent),batches,precision);out=c.ROOT/'quantized_states'/(name+'.pt');out.parent.mkdir(exist_ok=True);torch.save(dict(model=q.state_dict(),parent_sha256=parent['sha256'],precision=precision),out)
                entry=dict(parent,id=name,checkpoint=str(out),sha256=c.sha(out),precision=precision,precision_map=precision_map(q),quantized=True,fp_reference=parent['id'],sealed=graph in ['R2','R4'] and precision in ['W8A8','W4A8'],origin='CPU adopted simulator; native kernels unavailable')
                c.save(meta,entry);created.append(name)
    c.save(c.ROOT/'frontier_preparation.json',dict(outcome='completed',new_points=created,source_only_calibration=True,calibration_ids=ids))

def matched_points():
    c.setup()
    import v10_reference as v
    items=inventory();settings={};pairs=[]
    for seed in c.SEEDS:
        reference=f'E0_s{seed}'
        if reference not in items:continue
        for target in [f'E1_s{seed}',f'E3_s{seed}']+[name for name,x in items.items() if x.get('seed')==seed and (name.startswith(('R2_log','R2_ctx','R4_micro','R2_hardneg','R2_easyneg')) or x.get('control'))]:
            reference=f'E0_s{seed}' if target.startswith(('E1_','E3_')) else f'R2_CE_s{seed}' if target.startswith(('R2_log','R2_ctx','R2_hardneg','R2_easyneg')) else f'R3_CE_s{seed}' if target.startswith('R4_micro') else items[target].get('fp_reference',f'E0_s{seed}')
            a=c.read(c.ROOT/'source_metrics'/(reference+'.json'));b=c.read(c.ROOT/'source_metrics'/(target+'.json'))
            if not a or not b:continue
            ma=np.load(a['margins'],mmap_mode='r');mb=np.load(b['margins'],mmap_mode='r');fraction=float(np.mean(mb>0));tau=v.source_matched_tau(ma,fraction);pairs.append(dict(reference=reference,target=target,seed=seed,tau=tau,target_fraction=fraction,source_only=True))
            # Distinct reference views preserve every attribution comparison.
            copyentry=dict(items[reference],id=reference+'__matched_to__'+target,source_reference=reference,matched_tau=tau,matched_target=target,fp_reference=reference,method='operating_point_reference',sealed=True)
            c.save(c.ROOT/'quantized_entries'/(copyentry['id']+'.json'),copyentry);settings[copyentry['id']]=dict(matched_tau=tau)
    # B2(a): refit the quantized threshold to its own FP parent's source fraction.
    for name,entry in list(items.items()):
        if not entry.get('quantized') or entry.get('source_reference'):continue
        reference=entry.get('fp_reference');a=c.read(c.ROOT/'source_metrics'/(str(reference)+'.json'));b=c.read(c.ROOT/'source_metrics'/(name+'.json'))
        if not a or not b:continue
        fraction=float(np.mean(np.load(a['margins'],mmap_mode='r')>0));tau=v.source_matched_tau(np.load(b['margins'],mmap_mode='r'),fraction)
        copyentry=dict(entry,id=name+'__refit_to_FP',source_reference=name,matched_tau=tau,control='threshold_refit',sealed=entry.get('sealed',False))
        c.save(c.ROOT/'quantized_entries'/(copyentry['id']+'.json'),copyentry);settings[copyentry['id']]=dict(matched_tau=tau)
    c.save(c.ROOT/'operating_point_pairs.json',dict(outcome='completed',pairs=pairs));existing=c.read(c.ROOT/'operating_points.json',{});existing.update(settings);c.save(c.ROOT/'operating_points.json',existing)

def final_freeze():
    items=inventory();c.save(c.ROOT/'final_evaluation_freeze.json',dict(outcome='completed',at=time.time(),models=list(items.values()),model_manifest_sha256=c.sha(c.ROOT/'readout_models.json'),source_settings_sha256=c.sha(c.ROOT/'operating_points.json'),V9_sealed_windows_sha256=c.sha(c.V9/'sealed_windows.json'),code_hashes=c.read(c.ROOT/'code_freeze.json')['files'],no_target_selection=True,scientific_look='second look non-confirmatory' if c.read(c.ROOT/'code_freeze.json')['V9_sealed_metrics_exist_at_freeze'] else 'Protocol/code frozen before V9 sealed metrics existed; no later target adaptation'))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','source','smallest','frontier','matched','freeze']);p.add_argument('--slot',type=int);args=p.parse_args()
    if args.action=='source':source_all(args.slot)
    else:globals()[{'inventory':'inventory','smallest':'choose_smallest','frontier':'frontier','matched':'matched_points','freeze':'final_freeze'}[args.action]]()
if __name__=='__main__':main()
