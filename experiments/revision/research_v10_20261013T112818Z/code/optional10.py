"""Frozen Tier C source procedures, admitted only before final sealed freeze."""
import argparse,time
from pathlib import Path
import numpy as np
import common10 as c

def targets():
    torch,cj,old,v=c.setup();spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=42),augment=False);out=c.DATA/'mixed_targets';out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,path in enumerate(ds.files):
        a=torch.load(path,weights_only=True);x=a['image'].numpy();mask=a['mask'];teacher=mask.numpy()==1;fraction,_=v.unmix_fraction(x,teacher,k=2,radius=7,min_px=8,max_residual=1.0);prob=np.where(teacher,a['confidence'].numpy(),1-a['confidence'].numpy());band=v.edge_distance(teacher)<=2;hard=teacher.astype(np.float32);available=np.isfinite(fraction)
        t1=hard.copy();t1[band&available]=fraction[band&available];t2=hard.copy();t2[band]=prob[band];t3=hard.copy();t3[band&available]=(fraction[band&available]>.5);t4=hard.copy();t4[band]=prob[band]>.2
        target=out/path.name;torch.save({name:torch.from_numpy(value) for name,value in zip(['F0','F1','F2','F3','F4'],[hard,t1,t2,t3,t4])},target);rows.append(dict(file=str(target),sha256=c.sha(target)))
        if i%200==0:print('mixed-pixel targets',i,'/',len(ds),flush=True)
    c.save(c.ROOT/'mixed_target_manifest.json',dict(outcome='completed',files=rows,definitions='Unchanged v10_reference.unmix_fraction; teacher-probability confidence cache; unchanged labels outside2pxband',source_only=True))

def decode():
    torch,_,_,v=c.setup();from evaluate10 import counts,scores
    parents=[m for m in c.read(c.ROOT/'readout_models.json')['models'] if m['graph']=='R4' and m['method']=='lkd'];assert len(parents)==4;models=[c.load(m) for m in parents];grid=[dict(index=index,k=k,theta=theta) for index,thetas in [('MNDWI',[-.1,0,.1,.2]),('WI2015',[-5,0,5,10]),('AWEIsh',[-.05,0,.05,.1])] for k in [1,2,3] for theta in thetas];cells=[{} for _ in grid];rows=[r for r in c.read(c.V9/'normal_windows.json')['windows'] if r['dataset']=='s11val'];tic=time.monotonic()
    with torch.no_grad():
        for row in rows:
            with np.load(row['path']) as a:x=a['image'].copy();y=a['mask'].copy()
            maps,available=v.indices(v.reflectance(x));preds=[(lambda z:(z[0,1]>z[0,0]).numpy())(model(torch.from_numpy(x)[None])) for model in models]
            for candidate,eventcounts in zip(grid,cells):
                for seed,p in enumerate(preds):
                    key=(seed,row['event']);cell=counts(v.decode(p,maps[candidate['index']],available,candidate['k'],candidate['theta']),y);eventcounts[key]=eventcounts.get(key,np.zeros(4,np.int64))+cell
    values=[float(np.mean([scores(v)['water_IoU'] for v in ec.values() if scores(v)['water_IoU'] is not None])) for ec in cells];choice=grid[int(np.argmax(values))];settings=c.read(c.ROOT/'operating_points.json',{});settings['decode_rule']=choice;c.save(c.ROOT/'operating_points.json',settings);c.save(c.ROOT/'decode_rule_selection.json',dict(outcome='completed',choice=choice,grid=[dict(candidate,source_mean_equal_event_water_IoU=value) for candidate,value in zip(grid,values)],seconds=time.monotonic()-tic,source_only=True))

def ort():
    import importlib.util
    available=all(importlib.util.find_spec(name) is not None for name in ['onnx','onnxruntime'])
    if not available:
        c.save(c.ROOT/'strict_int8_ort.json',dict(outcome='unavailable',reason='Existing ONNX/ORT dependencies unavailable; no installs/downloads authorized'));return
    torch,cj,_,_=c.setup();from quantization.qat import export_qat_qdq_onnx
    import onnx,onnxruntime as ort
    from evaluate10 import counts,scores
    parents=[x for x in c.read(c.ROOT/'readout_models.json')['models'] if x['graph'] in ['R2','R4'] and x.get('precision')=='W8A8' and x.get('method')=='lkd'];rows=[]
    spec=c.read(c.V9/'training_spec.json');ds=cj.pseudo(dict(spec,seed=1729),augment=False);ids=c.read(c.V9/'ptq_manifest.json')['records'][0]['calibration_indices'];calibration=[dict(image=ds[i]['image'][None].float()) for i in ids]
    for parent in parents:
        model=c.load(parent);out=c.ROOT/'ort'/parent['id'];out.mkdir(parents=True,exist_ok=True)
        try:
            result=export_qat_qdq_onnx(model,torch.zeros(1,6,512,512),str(out),skip_last_n=0,trt_compat='modern',cal_batches=calibration,output_qdq_scale='calibrated')
            path=result[0] if isinstance(result,tuple) else result
            options=ort.SessionOptions();options.intra_op_num_threads=8;options.inter_op_num_threads=1;options.optimized_model_filepath=str(out/'optimized.onnx')
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);graph=onnx.load(str(out/'optimized.onnx'));integer=sum(node.op_type in ['QLinearConv','ConvInteger'] for node in graph.graph.node)
            events={};agreement=0;pixels=0
            for row in c.read(c.V9/'normal_windows.json')['windows']:
                if row['dataset'] not in ['s11val','s11']:continue
                with np.load(row['path']) as data:x=data['image'].copy()[None];y=data['mask'].copy()
                with torch.no_grad():z=model(torch.from_numpy(x)).numpy()
                native=session.run(None,{session.get_inputs()[0].name:x})[0];a=z[0,1]>z[0,0];b=native[0,1]>native[0,0];valid=y!=-1;agreement+=int((a[valid]==b[valid]).sum());pixels+=int(valid.sum());key=(row['dataset'],row['event']);old=events.setdefault(key,[np.zeros(4,np.int64),np.zeros(4,np.int64)]);old[0]+=counts(a,y);old[1]+=counts(b,y)
            for (dataset,event),(sim,native) in events.items():rows.append(dict(model=parent['id'],dataset=dataset,event=event,simulator_water_IoU=scores(sim)['water_IoU'],ORT_water_IoU=scores(native)['water_IoU'],pixel_agreement=agreement/pixels,integer_conv_nodes=integer,native_integer_execution=integer>0,scope='Workstation CPU QDQ runtime; no target hardware latency claim'))
        except Exception as exc:rows.append(dict(model=parent['id'],status='unavailable',reason=repr(exc),native_integer_execution=False))
    c.save(c.ROOT/'strict_int8_ort.json',dict(outcome='completed',rows=rows,all_failures_retained=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['targets','decode','ort']);args=p.parse_args();globals()[args.action]()
