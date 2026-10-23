#!/usr/bin/env python3
"""Full native inventories, confusion counts, dry estimands and declared timing."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
from validate_results import validate_rows,validate_dry
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(o,indent=1,allow_nan=False)+'\n');t.replace(p)
def scores(c):
    tn,fp,fn,tp=[int(v) for v in np.asarray(c).ravel()]
    div=lambda a,b:a/b if b else None
    wi=div(tp,tp+fp+fn);li=div(tn,tn+fp+fn)
    return dict(water_iou=wi,miou=float(np.mean([v for v in (wi,li) if v is not None])) if wi is not None or li is not None else None,precision=div(tp,tp+fp),recall=div(tp,tp+fn))
def summarize(rows):
    groups={e:np.sum([r['counts_tn_fp_fn_tp'] for r in rows if r['event']==e],axis=0) for e in sorted({r['event'] for r in rows})}
    event=[dict(event=e,counts_tn_fp_fn_tp=c.tolist(),**scores(c)) for e,c in groups.items()]
    mean=lambda k:float(np.mean([r[k] for r in event if r[k] is not None])) if any(r[k] is not None for r in event) else None
    return dict(images=len(rows),events=len(groups),valid_pixels=sum(sum(r['counts_tn_fp_fn_tp']) for r in rows),pooled=scores(np.sum([r['counts_tn_fp_fn_tp'] for r in rows],axis=0)),equal_event={k:mean(k) for k in ('water_iou','miou','precision','recall')},per_event=event,rows=rows)

def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--io',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--sturm-root',type=Path);p.add_argument('--wf-root',type=Path);p.add_argument('--only-dry',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--source-reference',type=Path);p.add_argument('--quant-source-reference',type=Path);a=p.parse_args(argv)
    if not a.only_dry and (a.sturm_root is None or a.wf_root is None):p.error('--sturm-root and --wf-root are required unless --only-dry is used')
    return a

def main():
    a=parse_args()
    # Preserve the known Jetson TLS import order before albumentations/cv2.
    import torch
    import tensorrt as trt
    import pycuda.driver,pycuda.autoinit
    if not trt.__version__.startswith('8.4.'):raise RuntimeError('Target is TensorRT8.4; no implicit runtime migration')
    import integer_tools,device_runtime,dataset_io
    a.out.mkdir(parents=True,exist_ok=a.resume)
    identity=dict(engine_sha256=sha(a.engine),io_sha256=sha(a.io) if a.io else None,only_dry=a.only_dry,source_reference_sha256=sha(a.source_reference) if a.source_reference else None,quant_source_reference_sha256=sha(a.quant_source_reference) if a.quant_source_reference else None)
    meta=a.out/'IDENTITY.json'
    if meta.exists():assert json.loads(meta.read_text())==identity
    else:save(meta,identity)
    if (a.out/'COMPLETE.json').exists():
        for ds in (() if a.only_dry else ('s11val','s11test','bolivia','wf','sturm512')):
            validate_rows(ROOT,ds,json.loads((a.out/(ds+'.json')).read_text())['rows'])
        validate_dry(ROOT,json.loads((a.out/'dry159.json').read_text())['rows'])
        if not a.only_dry:
            assert (a.out/'timing.json').exists() and (a.out/'wf_dry.json').exists()
        return
    runner=device_runtime.IntegerRunner(a.engine,json.loads(a.io.read_text())) if a.io else integer_tools.TrtRunner(a.engine)
    datasets={}
    def check_source_wiring(ds,result):
        if ds!='s11val':return
        if a.quant_source_reference:
            from repair_policy import source_gate
            gate=source_gate(json.loads(a.quant_source_reference.read_text()),result)
            gate['reference_sha256']=sha(a.quant_source_reference);save(a.out/'NATIVE_SOURCE_GATE.json',gate)
            if not gate['passed']:raise RuntimeError('Quantized model failed the0.02 native source gate; stop external evaluation')
        if not a.source_reference:return
        ref=json.loads(a.source_reference.read_text());delta=abs(result['pooled']['miou']-ref['mIoU']);passed=delta<=.005
        save(a.out/'SOURCE_WIRING.json',dict(passed=passed,pooled_mIoU_abs_difference=delta,tolerance=.005,reference_sha256=sha(a.source_reference)))
        if not passed:raise RuntimeError('Native FP32 source wiring exceeds0.005; stop external evaluation for this engine')
    for ds in (() if a.only_dry else ('s11val','s11test','bolivia','wf','sturm512')):
        existing=a.out/(ds+'.json')
        if existing.exists():
            data=json.loads(existing.read_text());validate_rows(ROOT,ds,data['rows']);datasets[ds]=summarize(data['rows']);check_source_wiring(ds,datasets[ds]);continue
        rows=[]
        for image,mask,rec in dataset_io.images(ds,a.sturm_root,a.wf_root):
            z=runner(image[None].astype(np.float32));assert z.shape==(1,2,512,512) and np.isfinite(z).all()
            pred=z[0].argmax(0);valid=(mask>=0)&(mask<2)
            c=np.bincount((2*mask[valid].astype(np.int64)+pred[valid]).ravel(),minlength=4)
            assert int(c.sum())==rec['valid_pixels'] and int(c[2]+c[3])==rec['water_pixels']
            rows.append(dict(name=rec['name'],event=rec['event'],map_id=rec['map_id'],input_sha256=rec['input_sha256'],mask_sha256=rec['mask_sha256'],counts_tn_fp_fn_tp=c.tolist(),**{k:rec[k] for k in ('input_profile','canonical_input_sha256') if k in rec}))
        validate_rows(ROOT,ds,rows)
        datasets[ds]=summarize(rows);save(a.out/(ds+'.json'),datasets[ds]);check_source_wiring(ds,datasets[ds]);print(ds,len(rows),'complete',flush=True)
    dry=[]
    for image,valid,rec in (() if (a.out/'dry159.json').exists() else dataset_io.dry()):
        z=runner(image[None].astype(np.float32));assert z.shape==(1,2,512,512) and np.isfinite(z).all();pred=z[0].argmax(0)
        row={k:rec[k] for k in ('tile_id','group','decision','stratum')}
        for name,sl in (('central264',(slice(124,388),slice(124,388))),('full512',(slice(None),slice(None)))):
            support=valid[sl].astype(bool);nv=int(support.sum());nw=int(((pred[sl]==1)&support).sum())
            row[name]=dict(valid_pixels=nv,false_water_pixels=nw,false_water_fraction=nw/nv if nv else None,alarm_gt_1pct=bool(100*nw>nv) if nv else None)
        dry.append(row)
    if not dry:dry=json.loads((a.out/'dry159.json').read_text())['rows']
    validate_dry(ROOT,dry);save(a.out/'dry159.json',dict(rows=dry,estimands='author49 central264 primary; author49 full512 sensitivity; map159 central264 sensitivity; ordinary and stress separate'))
    if a.only_dry:
        save(a.out/'COMPLETE.json',dict(status='DRY_EXTENSION_COMPLETE',dry159=True,engine_sha256=sha(a.engine),heldback97='NOT_READ'));return
    wf_dry=[r for r in datasets['wf']['rows'] if r['counts_tn_fp_fn_tp'][2]+r['counts_tn_fp_fn_tp'][3]==0]
    assert len(wf_dry)==200 and len({r['event'] for r in wf_dry})==9
    save(a.out/'wf_dry.json',dict(images=200,events=9,rows=[dict(name=r['name'],event=r['event'],valid_pixels=sum(r['counts_tn_fp_fn_tp']),false_water_pixels=r['counts_tn_fp_fn_tp'][1],alarm_gt_1pct=100*r['counts_tn_fp_fn_tp'][1]>sum(r['counts_tn_fp_fn_tp'])) for r in wf_dry]))
    # Match the previous boundary: one fixed S11 val image, wall includes host encoding/transfers.
    path=sorted((ROOT/'assets/source_val').glob('*.npz'))[-1]
    with np.load(path,allow_pickle=False) as z:image=z['image'][None].astype(np.float32)
    for _ in range(20):runner(image)
    wall=[]
    for _ in range(100):
        t=time.perf_counter();runner(image);wall.append(1000*(time.perf_counter()-t))
    kernel=runner.kernel_benchmark(image,20,100)
    quantiles=lambda values:{'p'+str(p):float(np.percentile(values,p)) for p in (50,95,99)}
    save(a.out/'timing.json',dict(image=path.name,image_sha256=sha(path),wall_ms=quantiles(wall),kernel_ms=quantiles(kernel),raw_wall_ms=wall,raw_kernel_ms=kernel,wall_boundary='H2D, inference, D2H, copied logits and host quantization for integer IO; excludes normalization, disk, argmax',engine_bytes=a.engine.stat().st_size,device_memory_bytes=int(runner.engine.device_memory_size),engine_sha256=sha(a.engine)))
    save(a.out/'COMPLETE.json',dict(status='ALL_DECLARED_DATASETS_COMPLETE',datasets=list(datasets),dry159=True,wf_dry200=True,heldback97='NOT_READ',engine_sha256=sha(a.engine),io_sha256=sha(a.io) if a.io else None,tensorrt=trt.__version__))

if __name__=='__main__':main()
