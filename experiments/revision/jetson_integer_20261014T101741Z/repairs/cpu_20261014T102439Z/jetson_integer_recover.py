#!/usr/bin/env python3
"""Additive technical repairs; original failed artifacts and receipts stay intact."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','ptq','qat','collect'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repair',type=Path)
    parser.add_argument('--model',choices=['R4','B0'])
    args=parser.parse_args();root=args.root.resolve()
    source=Path(__file__).resolve().parents[1]
    if args.action=='freeze':
        repair=root/'repairs'/('cpu_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        repair.mkdir(parents=True)
        for name in ['scripts/jetson_integer_recover.py','scripts/jetson_integer_prepare.py','quantization/jetson_integer.py']:
            shutil.copy2(source/name,repair/Path(name).name)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in repair.glob('*.py')}
        (repair/'protocol.json').write_text(json.dumps(dict(created=datetime.now(timezone.utc).isoformat(),hashes=hashes,
            motivation='Repair scalar INT32-bias representation and distinguish ONNX semantic parity from optimized runtime numerics; retain existing artifacts',
            old_protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
            new_B0_variants=['Conv-only Percentile99.99','Conv+residual+mm1 Percentile99.99; attention mm2 and normalizer Add excluded'],
            calibration_count=32,source_selection_only=True,additional_training=False),indent=2))
        print(repair);return
    repair=args.repair.resolve();recipe=json.loads((repair/'protocol.json').read_text())
    for name,digest in recipe['hashes'].items():assert hashlib.sha256((repair/name).read_bytes()).hexdigest()==digest
    sys.path.insert(0,str(root/'code'))
    runner=module('prepare_repaired',repair/'jetson_integer_prepare.py')
    q=module('quantization.jetson_integer',repair/'jetson_integer.py');sys.modules['quantization.jetson_integer']=q
    protocol=json.loads((root/'protocol.json').read_text())
    import torch,numpy as np
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from experiments.revision_queue import atomic_json,file_hash
    name=args.model;target=root/'bundle'/name if name else None
    if args.action=='ptq':
        variants=[dict(id='repair_minmax_pt',method='MinMax',per_channel=False,symmetric=True,operators=['Conv','MatMul','Add'])]
        if name=='B0':
            import onnx
            graph=onnx.load(target/'model_float.onnx')
            mm=[n.name for n in graph.graph.node if n.op_type=='MatMul']
            assert len(mm)==8, len(mm)
            producer={out:n for n in graph.graph.node for out in n.output}
            normalizer=[]
            for n in graph.graph.node:
                if n.op_type=='Div':
                    p=producer[n.input[1]]
                    assert p.op_type=='Add';normalizer.append(p.name)
            variants += [dict(id='conv_only_percentile',method='Percentile',per_channel=True,symmetric=True,operators=['Conv']),
                dict(id='mm1_only_percentile',method='Percentile',per_channel=True,symmetric=True,operators=['Conv','MatMul','Add'],nodes_to_exclude=mm[1::2]+normalizer)]
        baseline=json.loads((target/'float_evaluation.json').read_text())
        images=np.stack([np.load(p,allow_pickle=False) for p in sorted((root/'bundle/calibration').glob('*.npy'))])
        results=[]
        for config in variants:
            output=target/(config['id']+'.onnx');start=time.monotonic()
            record=dict(config=config,status='failed',additional_training=False,parent_protocol_sha256=recipe['old_protocol_sha256'])
            try:
                record['audit']=q.ptq(target/'model_float.onnx',output,images,**{k:v for k,v in config.items() if k!='id'})
                record['evaluation']=runner.evaluate(output,root/'bundle/source_val',target/(config['id']+'_evaluation.json'))
                stats=q.compare_metrics(record['evaluation']['rows'],baseline['rows']);record['comparison']=stats
                record['acceptable_accuracy']=stats['pooled_delta']>=-.01 and stats['equal_event_delta']>=-.01 and record['evaluation']['water_IoU']>=baseline['water_IoU']-.02
                record['status']='evaluated'
            except Exception as e:
                import traceback
                record['reason']=repr(e);record['traceback']=traceback.format_exc()
            record['elapsed_seconds']=time.monotonic()-start
            atomic_json(target/(config['id']+'_receipt.json'),record);results.append(record)
            print(name,config['id'],record['status'],record.get('comparison',{}).get('pooled_delta'),flush=True)
        atomic_json(repair/(name+'_ptq.json'),dict(status='complete',variants=results))
    elif args.action=='qat':
        from quantization.manual import fold_bn
        from models.compression_quant import load_compression_qat
        directory=target/'qat128'
        if not directory.exists():directory=target/'qat_smoke'
        checkpoint=directory/'checkpoint_qat.pt';assert checkpoint.exists()
        model=fold_bn(runner.load_model(name,protocol))
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=load_compression_qat(model,payload['model']).eval()
        image=np.load(next((root/'bundle/calibration').glob('*.npy')))[None]
        with torch.no_grad():expected=model(torch.from_numpy(image)).numpy()
        path=directory/'model_qdq.onnx';assert path.exists()
        semantic=q.session(path,disable_optimization=True).run(None,{'image':image})[0]
        optimized=q.session(path).run(None,{'image':image})[0]
        assert np.allclose(expected,semantic,atol=1e-3,rtol=1e-3),'Semantic ONNX parity failed'
        result=dict(status='validated',updates=payload['updates'],smoke_only=payload['updates']==1,
            original_checkpoint_sha256=file_hash(checkpoint),additional_training=False,
            semantic_max_abs=float(np.max(np.abs(expected-semantic))),
            optimized_max_abs=float(np.max(np.abs(expected-optimized))),
            optimized_argmax_agreement=float(np.mean(expected.argmax(1)==optimized.argmax(1))),
            semantics_tolerance=dict(atol=.001,rtol=.001),audit=q.graph_audit(path),
            interpretation='Export semantics match; optimizer numerical changes disclosed and executable accuracy measured separately')
        result['evaluation']=runner.evaluate(path,root/'bundle/source_val',directory/'repaired_evaluation.json')
        baseline=json.loads((target/'float_evaluation.json').read_text())
        result['comparison']=q.compare_metrics(result['evaluation']['rows'],baseline['rows'])
        atomic_json(directory/'repaired_validation.json',result);atomic_json(repair/(name+'_qat.json'),result)
    else:
        for name in ['R4','B0']:
            target=root/'bundle'/name
            baseline=json.loads((target/'float_evaluation.json').read_text());variants=[]
            for path in sorted(target.glob('*_receipt.json')):variants.append(json.loads(path.read_text()))
            candidates=[r for r in variants if r['status']=='evaluated' and r['audit']['jp50_qdq_compatible']]
            selected=max(candidates,key=lambda r:r['evaluation']['mIoU']) if candidates else None
            smoke=target/'qat_smoke/repaired_validation.json';trained=target/'qat128/repaired_validation.json'
            qat_path=trained if trained.exists() else smoke
            qat=json.loads(qat_path.read_text()) if qat_path.exists() else dict(status='failed_or_pending')
            record=dict(status='complete',name=name,parameters=12230 if name=='R4' else sum(p.numel() for p in runner.load_model(name,protocol).parameters()),
                float=baseline,float_event=q.compare_metrics(baseline['rows'],baseline['rows']),variants=variants,
                selected=selected['config']['id'] if selected else None,qat=qat,
                full_integer_execution='not_demonstrated',int4='blocked',native_TRT='blocked_workstation_CUDA_error35',
                original_job_failures_retained=True,additional_training=False)
            atomic_json(root/'receipts'/(name+'_reconciled.json'),record)
        rows=[]
        for name in ['R4','B0']:
            a=json.loads((root/'receipts'/(name+'_reconciled.json')).read_text())
            rows.append(dict(model=name,variant='float',mIoU=a['float']['mIoU'],water_IoU=a['float']['water_IoU'],equal_event_mIoU=a['float_event']['equal_event_mIoU'],pooled_delta=0,CPU_ms=a['float']['latency_cpu_ms_mean'],bytes=a['float']['model_bytes'],status='float'))
            for r in a['variants']:
                ev=r.get('evaluation',{});co=r.get('comparison',{})
                rows.append(dict(model=name,variant=r['config']['id'],mIoU=ev.get('mIoU'),water_IoU=ev.get('water_IoU'),equal_event_mIoU=co.get('equal_event_mIoU'),pooled_delta=co.get('pooled_delta'),CPU_ms=ev.get('latency_cpu_ms_mean'),bytes=ev.get('model_bytes'),status=r['status']))
        import csv
        with (root/'results_reconciled.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        atomic_json(repair/'collect.json',dict(status='complete',rows=rows))


if __name__=='__main__':main()
