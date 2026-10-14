#!/usr/bin/env python3
"""Evaluate QAT runtime drift, reconcile retained failures, and package evidence."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();sys.path.insert(0,str(root/'code'))
    sys.path.insert(0,str(root/'code/scripts'))
    import torch,numpy as np
    import jetson_integer_prepare as runner
    spec=importlib.util.spec_from_file_location('quantization.jetson_integer',root/'diagnostics/quantizer.py')
    quantizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(quantizer)
    sys.modules['quantization.jetson_integer']=quantizer
    from quantization.manual import fold_bn
    from models.compression_quant import load_compression_qat
    from experiments.revision_queue import atomic_json,file_hash
    from quantization.jetson_integer import session,scores,compare_metrics,graph_audit
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    protocol=read(root/'protocol.json');target=root/'bundle/B0/qat128'
    payload=torch.load(target/'checkpoint_qat.pt',map_location='cpu',weights_only=True)
    model=load_compression_qat(fold_bn(runner.load_model('B0',protocol)),payload['model']).eval()
    rows=[];conf=np.zeros((2,2),np.int64)
    for path in ([] if (target/'torch_QAT_evaluation.json').exists() else sorted((root/'bundle/source_val').glob('*.npz'))):
        with np.load(path,allow_pickle=False) as a:
            x=torch.from_numpy(a['image'].copy())[None];mask=a['mask'];event=str(a['event'])
        with torch.no_grad():z=model(x).numpy()
        assert np.isfinite(z).all()
        pred=z[0].argmax(0);valid=(mask>=0)&(mask<2)
        c=np.bincount((mask[valid]*2+pred[valid]).ravel(),minlength=4).reshape(2,2)
        rows.append(dict(file=path.name,event=event,confusion=c.tolist()));conf+=c
    reference=read(target/'torch_QAT_evaluation.json') if (target/'torch_QAT_evaluation.json').exists() else dict(rows=rows,**scores(conf),backend='PyTorch CPU fixed-scale fake quant; floating convolution arithmetic')
    atomic_json(target/'torch_QAT_evaluation.json',reference)
    exported=read(target/'runtime_evaluation.json') if (target/'runtime_evaluation.json').exists() else runner.evaluate(target/'model_qdq.onnx',root/'bundle/source_val',target/'runtime_evaluation.json')
    drift=compare_metrics(exported['rows'],reference['rows'])
    parent=read(root/'bundle/B0/float_evaluation.json');comparison=compare_metrics(exported['rows'],parent['rows'])
    image=np.load(next((root/'bundle/calibration').glob('*.npy')))[None]
    with torch.no_grad():expected=model(torch.from_numpy(image)).numpy()
    semantic=session(target/'model_qdq.onnx',disable_optimization=True).run(None,{'image':image})[0]
    actual=session(target/'model_qdq.onnx').run(None,{'image':image})[0]
    ablation=read(root/'B0_QAT_float_ablation_parity.json')
    assert ablation['original_tolerance_pass'], 'Unrounded model structure/weights parity failed'
    accepted=comparison['pooled_delta']>=-.01 and comparison['equal_event_delta']>=-.01 and exported['water_IoU']>=parent['water_IoU']-.02
    validated=dict(status='runtime_evaluated_with_logit_parity_failure',updates=payload['updates'],smoke_only=False,
        checkpoint_sha256=file_hash(target/'checkpoint_qat.pt'),model_sha256=file_hash(target/'model_qdq.onnx'),
        reference=reference,evaluation=exported,comparison=comparison,paired_runtime_drift=drift,
        semantic_max_abs=float(np.max(np.abs(expected-semantic))),optimized_max_abs=float(np.max(np.abs(expected-actual))),
        optimized_argmax_agreement=float(np.mean(expected.argmax(1)==actual.argmax(1))),
        strict_logit_parity_pass=False,unrounded_graph_parity=ablation,acceptable_accuracy=accepted,
        audit=graph_audit(target/'model_qdq.onnx'),
        interpretation='Strict logit parity remains failed. Unrounded graph matches. Sparse early rounding discontinuities amplify tiny floating-kernel differences; this is a supported inference from saved layer diagnostics. Evaluate and select the actual exported runtime, never alias it to PyTorch QAT.')
    atomic_json(target/'runtime_validation.json',validated)
    summary=[];models={}
    for name in ['R4','B0']:
        folder=root/'bundle'/name;baseline=read(folder/'float_evaluation.json')
        model=runner.load_model(name,protocol);parameters=sum(p.numel() for p in model.parameters())
        architecture=dict(parameters=parameters,convolution_layers=sum(isinstance(m,torch.nn.Conv2d) for m in model.modules()),
            BatchNorm_layers=sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),
            LayerNorm_layers=sum(isinstance(m,torch.nn.LayerNorm) for m in model.modules()),
            state_shapes={n:list(v.shape) for n,v in model.state_dict().items()})
        atomic_json(folder/'architecture.json',architecture)
        def add(variant,ev,co,status,acceptable=None,audit=None):
            summary.append(dict(model=name,variant=variant,parameters=parameters,mIoU=ev.get('mIoU'),
                water_IoU=ev.get('water_IoU'),equal_event_mIoU=co.get('equal_event_mIoU'),
                pooled_delta=co.get('pooled_delta'),equal_event_delta=co.get('equal_event_delta'),
                CPU_ms=ev.get('latency_cpu_ms_mean'),ONNX_bytes=ev.get('model_bytes'),
                status=status,acceptable_source_accuracy=acceptable,
                integer_execution='not_demonstrated',CPU_kernels=json.dumps(ev.get('optimized_operators',{}))))
        add('float',baseline,compare_metrics(baseline['rows'],baseline['rows']),'float')
        variants=[]
        for path in sorted(folder.glob('*_receipt.json')):
            r=read(path);variants.append(r)
            add(r['config']['id'],r.get('evaluation',{}),r.get('comparison',{}),r['status'],r.get('acceptable_accuracy'),r.get('audit'))
        qat=read(folder/'qat_smoke/repaired_validation.json') if name=='R4' else validated
        add('QAT_1update_smoke' if name=='R4' else 'QAT_128updates',qat['evaluation'],qat['comparison'],qat['status'],qat.get('acceptable_accuracy'))
        candidates=[r for r in variants if r['status']=='evaluated' and r['audit']['jp50_qdq_compatible']]
        selected=max(candidates,key=lambda r:r['evaluation']['mIoU']) if candidates else None
        deploy=selected['config']['id']+'.onnx' if selected else None
        if name=='B0' and accepted:deploy='qat128/model_qdq.onnx'
        models[name]=dict(parameters=parameters,baseline=baseline,variants=variants,QAT=qat,
            recommended_device_probe=deploy,device_status='pending_native_build_accuracy_latency_memory_and_kernel_audit',
            original_failure_receipt=str(root/'attempts'/(name+'.attempt1.json')))
        atomic_json(root/'receipts'/(name+'_final.json'),models[name])
    with (root/'results_final.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    atomic_json(root/'final_status.json',dict(workstation='completed_with_preserved_negative_results',models={n:{k:v for k,v in row.items() if k not in ['baseline','variants','QAT']} for n,row in models.items()},
        native_INT8='not_demonstrated; TensorRT builder blocked before parse/calibration by CUDA35',
        INT4='not_started_gate_closed',Jetson='pending_access_to_device; historical environment retained',
        QAT_B0='128 updates complete; source runtime measured; strict logit parity failed',
        QAT_R4='one-update smoke/reload/export/source runtime measured; no efficacy claim',
        original_queues='original model jobs failed strict parity; repair collector blocked by B0 parity; retained unchanged',
        reconciliation='final_status.json and receipts/*_final.json are the final reconciled evidence; no failed job was relabeled succeeded'))
    atomic_json(root/'bundle/deployment_manifest.json',{n:dict(model=n,candidate=row['recommended_device_probe'],
        float_model=n+'/model_float.onnx',candidate_model=n+'/'+row['recommended_device_probe'] if row['recommended_device_probe'] else None,
        status='candidate_for_native_probe',full_integer=False,checkpoint=protocol['models'][n]['sha256']) for n,row in models.items()})
    # Carry the final evidence, failures and source recipe with the portable arrays.
    for name in ['final_status.json','results_final.csv','workstation_TRT_preflight.json','execution_recipe_audit.json','CPU_resource_admission.json','B0_QAT_layer_parity.json','B0_QAT_float_ablation_parity.json']:
        shutil.copy2(root/name,root/'bundle'/name)
    print(json.dumps({n:dict(float=row['baseline']['mIoU'],candidate=row['recommended_device_probe'],QAT_mIoU=row['QAT']['evaluation']['mIoU']) for n,row in models.items()},indent=2))


if __name__=='__main__':main()
