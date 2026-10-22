import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from repair_graph import transform
P=Path(__file__).resolve().parent
D=Path('/home/schmalstieg/repos/FloodDistill/jetson_repro/finalize_f_20261022/device_handoff')
sys.path.insert(0,str(D/'runtime'))
import dataset_io
from evaluate_native import summarize
from device_runtime import encode
dataset_io.ROOT=Path('/home/schmalstieg/repos/MajorRev/revision_package_20261021/8_jetson_export_20261022')
def session(path,semantic=False):
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 if semantic:opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
 return ort.InferenceSession(str(path),opt,providers=['CPUExecutionProvider'])
models=json.loads((D/'RUN_MANIFEST.json').read_text())['engines']
for seed in [107,42,64,91]:
 e=next(m for m in models if m['architecture']=='R4_micro' and m['precision']=='pure' and m['seed']==seed)
 out=P/'models'/(e['id']+'_aligned16');out.mkdir(parents=True,exist_ok=False)
 path=out/'model.onnx';changes=transform(D/e['onnx'],path)
 old=session(D/e['onnx']);new=session(path);semantic_old=session(D/e['onnx'],True);semantic_new=session(path,True)
 io=json.loads((D/e['io']).read_text());rows=[];parity=[]
 start=time.time()
 for image,mask,rec in dataset_io.source('s11val'):
  x=encode(image[None],io['host_input_scale'],io['host_input_zero_point'])
  a=old.run(None,{'image':x})[0];b=new.run(None,{'image':x})[0]
  c=semantic_old.run(None,{'image':x})[0];d=semantic_new.run(None,{'image':x})[0]
  valid=(mask>=0)&(mask<2);pred=b[0].argmax(0)
  counts=np.bincount((mask[valid]*2+pred[valid]).astype(np.int64),minlength=4)
  rows.append(dict(name=rec['name'],event=rec['event'],input_sha256=rec['input_sha256'],mask_sha256=rec['mask_sha256'],counts_tn_fp_fn_tp=counts.tolist()))
  parity.append(dict(name=rec['name'],optimized_exact=bool(np.array_equal(a,b)),optimized_max_abs_int8=int(np.abs(a.astype(int)-b.astype(int)).max()),semantic_exact=bool(np.array_equal(c,d)),semantic_max_abs_int8=int(np.abs(c.astype(int)-d.astype(int)).max()),argmax_agreement=float(np.mean(a.argmax(1)==b.argmax(1)))))
  if len(rows)==1:print(seed,'first',parity[-1],flush=True)
 result=summarize(rows);baseline=json.loads((D/e['source_reference']).read_text());metrics=dict(mIoU=result['pooled']['miou'],equal_event_mIoU=result['equal_event']['miou'],water_IoU=result['pooled']['water_iou']);drops={k:baseline[k]-v for k,v in metrics.items()}
 report=dict(seed=seed,source_model=e,recipe='zero-pad16_dense_block_diagonal',graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),changes=changes,evaluation=result,baseline=baseline,metrics=metrics,drops=drops,accuracy_pass=all(v<=.02 for v in drops.values()),parity=parity,optimized_exact_all89=all(r['optimized_exact'] for r in parity),semantic_exact_all89=all(r['semantic_exact'] for r in parity),seconds=time.time()-start,native_execution='PENDING_JETSON',weights_fit=False,scales_recalibrated=False)
 (out/'REPAIR_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print(seed,'DONE',report['accuracy_pass'],report['optimized_exact_all89'],report['semantic_exact_all89'],drops,'seconds',report['seconds'],flush=True)
 assert report['accuracy_pass']
