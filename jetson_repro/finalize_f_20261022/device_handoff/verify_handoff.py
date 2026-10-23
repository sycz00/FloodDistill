#!/usr/bin/env python3
"""Offline checks for deployment identities, full inventories and failure handling."""
import ast,contextlib,copy,csv,hashlib,io,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import onnx
import collect_native
from validate_results import validate_rows,benchmark_receipts_valid
import prepare_inputs
import precision_islands
import dataset_io
from evaluate_native import parse_args

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dry_check=prepare_inputs.verify(dry_only=True);assert dry_check['arrays']==159
    dry_manifest=read(ROOT/'inventories/dry159_manifest.json')['rows']
    assert {f.stem for f in (ROOT/'assets/dry159').glob('*.npz')}=={r['tile_id'] for r in dry_manifest}
    dry_count=0
    for image,valid,rec in dataset_io.dry():
        assert image.shape==(6,512,512) and image.dtype==np.float32 and np.isfinite(image).all()
        assert valid.shape==(512,512) and valid.dtype==np.bool_
        assert int(valid.sum())==rec['full_valid_pixels'] and int(valid[124:388,124:388].sum())==rec['central_valid_pixels']
        dry_count+=1
    assert dry_count==159
    accepted=[r for r in dry_manifest if r['decision']=='kept_dry']
    assert len(accepted)==49 and sum(r['group']=='ordinary' for r in accepted)==28 and sum(r['group']=='stress' for r in accepted)==21
    dry_args=parse_args(['--engine','fixture.engine','--out','fixture_output','--only-dry'])
    assert dry_args.only_dry and dry_args.sturm_root is None and dry_args.wf_root is None
    with contextlib.redirect_stderr(io.StringIO()):
        try:parse_args(['--engine','fixture.engine','--out','fixture_output'])
        except SystemExit as error:assert error.code==2
        else:raise AssertionError('Full evaluation admitted without external dataset roots')
    m=read(ROOT/'RUN_MANIFEST.json');models=m['engines'];slots=m['primary_matrix'];assert len(models)==51 and len(slots)==48
    assert {(s['architecture'],s['seed'],s['precision']) for s in slots}=={(a,s,p) for a in ('R4','R4_micro','B0') for s in (42,64,91,107) for p in ('fp32','fp16','mixed','pure')}
    assert sum(s['engine'] is None for s in slots)==4 and all(s['architecture']=='B0' and s['precision']=='pure' for s in slots if s['engine'] is None)
    for e in models:
        path=ROOT/e['onnx'];assert sha(path)==e['onnx_sha256'];g=onnx.load(path);onnx.checker.check_model(g)
        assert [d.dim_value for d in g.graph.input[0].type.tensor_type.shape.dim]==[1,6,512,512]
        assert [d.dim_value for d in g.graph.output[0].type.tensor_type.shape.dim]==[1,2,512,512]
        if e['precision']=='pure':
            assert g.graph.input[0].type.tensor_type.elem_type==onnx.TensorProto.INT8 and g.graph.output[0].type.tensor_type.elem_type==onnx.TensorProto.INT8
            assert sha(ROOT/e['io'])==e['io_sha256'];o=read(ROOT/e['io']);assert o['host_input_scale']>0 and o['output_scale']>0 and o['host_input_zero_point']==o['output_zero_point']==0
        else:assert g.graph.input[0].type.tensor_type.elem_type==onnx.TensorProto.FLOAT
        if e['precision'] in ('fp32','fp16'):assert not any(n.op_type in ('QuantizeLinear','DequantizeLinear') for n in g.graph.node)
        if e['method']=='QAT':assert isinstance(e['strict_logit_parity_all89'],bool) and e['unrounded_structure_all89']
        if e['architecture']=='B0':
            spec=read(ROOT/e['fp32_islands']);assert sha(ROOT/e['fp32_islands'])==e['fp32_islands_sha256']
            assert spec==precision_islands.from_onnx(path)
            assert len(spec['required_compute_nodes'])==13 and spec['attention_matmuls']==8 and spec['normalizers']==4
            if e['method']=='PTQ':assert e['source_accuracy_pass'] is False
            if e['method']=='QAT':
                assert e['source_accuracy_pass'] is True and e['qat_series']=='mixed_smooth_repair'
                gate=read(ROOT/e['source_gate']);assert gate['accuracy_pass'] and all(v<=.02 for v in gate['drops'].values())
                assert gate['onnx']['sha256']==e['onnx_sha256']
                summaries={}
                for key in ('evaluation','baseline'):
                    rows=[dict(name=r['file'],event=r['event'],counts_tn_fp_fn_tp=np.asarray(r['confusion']).ravel().tolist()) for r in gate[key]['rows']]
                    validate_rows(ROOT,'s11val',rows,imported=True);summary=collect_native.summarize(rows)
                    values=dict(mIoU=summary['pooled']['miou'],equal_event_mIoU=summary['equal_event']['miou'],water_IoU=summary['pooled']['water_iou'])
                    assert all(abs(values[k]-gate[key][k])<1e-12 for k in values);summaries[key]=values
                assert all(abs(summaries['baseline'][k]-summaries['evaluation'][k]-v)<1e-12 for k,v in gate['drops'].items())

                producers={t:n for n in g.graph.node for t in n.output};head=next(n for n in g.graph.node if n.name==spec['classifier'])
                assert producers.get(head.input[0]) is None or producers[head.input[0]].op_type!='DequantizeLinear'
                assert head.input[1] in {i.name for i in g.graph.initializer}

        ref=read(ROOT/e['source_reference']);assert len(ref['rows'])==89
        if e.get('reuse'):
            reuse=e['reuse'];assert sha(ROOT/reuse['engine'])==reuse['engine_sha256'];b=read(ROOT/reuse['build_receipt']);assert b['engine_sha256']==reuse['engine_sha256'] and b['model_sha256']==e['onnx_sha256']
            if e['precision']=='pure':assert b['precision_audit']['full_integer_tensor_path']
            for ds,spec in reuse['evaluations'].items():
                d=read(ROOT/spec['path']);assert (d.get('engine_sha256') or d.get('model_sha256'))==reuse['engine_sha256']
    # Native certificates must reject FP16 normalization, missing classifiers and stale policies.
    sample=next(e for e in models if e['architecture']=='B0' and e['method']=='QAT');spec=read(ROOT/sample['fp32_islands'])
    layers={'Layers':[{'Name':n,'Outputs':[{'Name':n+':0','Format/Datatype':'Row major linear FP32'}]} for n in spec['required_compute_nodes']]}
    audit=precision_islands.inspect(layers,spec);assert audit['passed']
    assert precision_islands.contains_node('PWN(/block/MatMul, /block/Add)', '/block/MatMul')
    assert not precision_islands.contains_node('/block/MatMul_1', '/block/MatMul')
    from types import SimpleNamespace
    class Layer:
        def __init__(self,name):self.name=name;self.type='LayerType.MATRIX_MULTIPLY';self.num_outputs=1;self.tensor=SimpleNamespace(name=name+':0',is_shape_tensor=False,dtype='float32');self.types={}
        def get_output(self,index):return self.tensor
        def set_output_type(self,index,value):self.types[index]=value
    native_layers=[Layer(n) for n in spec['node_names']]
    network=SimpleNamespace(num_layers=len(native_layers),get_layer=lambda i:native_layers[i])
    flags=[];cleared=[];config=SimpleNamespace(set_flag=flags.append,clear_flag=cleared.append);trt=SimpleNamespace(float32='float32',int32='int32',bool='bool',BuilderFlag=SimpleNamespace(OBEY_PRECISION_CONSTRAINTS='OBEY',PREFER_PRECISION_CONSTRAINTS='PREFER'))
    precision_islands.constrain(network,config,trt,spec)
    assert flags==['OBEY'] and all(r.precision=='float32' and r.types[0]=='float32' for r in native_layers)
    flags.clear();result=precision_islands.constrain(network,config,trt,spec,mode='prefer')
    assert flags==['PREFER'] and cleared==['OBEY'] and result['mode']=='prefer' and all(r.precision=='float32' and r.types[0]=='float32' for r in native_layers)
    # Embedded negative PTQ controls contain Q/DQ nodes: never demand FLOAT Q outputs.
    quantizer=Layer('/diagnostic/QuantizeLinear');quantizer.type='LayerType.QUANTIZE';quantizer.tensor.dtype='int8'
    dequantizer=Layer('/diagnostic/DequantizeLinear');dequantizer.type='LayerType.DEQUANTIZE'
    test_spec=copy.deepcopy(spec);test_spec['node_names']+=['/diagnostic/QuantizeLinear','/diagnostic/DequantizeLinear']
    test_layers=native_layers+[quantizer,dequantizer];test_network=SimpleNamespace(num_layers=len(test_layers),get_layer=lambda i:test_layers[i])
    precision_islands.constrain(test_network,config,trt,test_spec)
    assert not quantizer.types and not dequantizer.types and not hasattr(quantizer,'precision') and not hasattr(dequantizer,'precision')

    absent=SimpleNamespace(num_layers=0,get_layer=lambda i:None)
    try:precision_islands.constrain(absent,config,trt,spec)
    except ValueError:pass
    else:raise AssertionError('Missing FP32 compute constraints admitted')

    bad=copy.deepcopy(layers);bad['Layers'][0]['Outputs'][0]['Format/Datatype']='Row major linear FP16'
    assert not precision_islands.inspect(bad,spec)['passed']
    bad=copy.deepcopy(layers);bad['Layers']=[r for r in bad['Layers'] if r['Name']!=spec['classifier']]
    assert not precision_islands.inspect(bad,spec)['passed']
    certificate=dict(status='built',model_sha256=sample['onnx_sha256'],fp32_islands_sha256=sample['fp32_islands_sha256'],fp32_island_audit=audit,fp32_constraints=dict(required=spec['required_compute_nodes'],mode=sample.get('island_constraint_mode','obey')))
    assert precision_islands.validate_certificate(certificate,sample,ROOT,layers)
    for bad in (dict(certificate,fp32_constraints=dict(required=spec['required_compute_nodes'],mode='obey')),dict(certificate,fp32_islands_sha256='0'*64),dict(certificate,fp32_island_audit={}),dict(certificate,status='fp32_island_certification_failed')):
        try:precision_islands.validate_certificate(bad,sample,ROOT,layers)
        except ValueError:pass
        else:raise AssertionError('Invalid B0 native precision certificate admitted')
    parity=list((ROOT/'provenance/float_parity').glob('*.json'));assert len(parity)==12
    for p in parity:
        d=read(p);assert d['atol']==d['rtol']==.001 and len(d['semantic'])==len(d['optimized'])==89 and all(r['strict_logit_parity_pass'] for r in d['semantic']+d['optimized'])
    for p in ROOT.rglob('*.py'):
        if '__pycache__' not in p.parts:ast.parse(p.read_text(),feature_version=(3,8))
    for name in ('runtime/run_remaining.py','runtime/evaluate_native.py','runtime/collect_native.py','runtime/preflight.py','runtime/integer_tools.py','runtime/bench_trtexec.py','runtime/bench_power_steady.py','runtime/probe_int8_attention.py','runtime/chain_evaluations.py','prepare_inputs.py'):
        subprocess.run([sys.executable,str(ROOT/name),'--help'],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(ROOT/'runtime/device_runtime.py'),'--self-test'],check=True)
    with tempfile.TemporaryDirectory(prefix='jetson_handoff_fixture_') as folder:
        temp=Path(folder)
        for n in ('RUN_MANIFEST.json','models','prior_device_bundle','inventories'):(temp/n).symlink_to(ROOT/n)
        out=temp/'results/MISSING';out.mkdir(parents=True);(out/'STATUS.json').write_text('[]\n');collect_native.ROOT=temp
        assert collect_native.main('MISSING')==1
        receipt=read(out/'COLLECTION_RECEIPT.json');assert receipt['observed_complete_primary_cells']==0 and not receipt['all_declared_attempts_resolved'] and not receipt['invalid_results']
        rows=list(csv.DictReader((out/'accuracy.csv').open()));assert len(rows)==80
        # The compact handoff retains actual old raw accuracy; compare independent saved summary.
        summary=ROOT/'provenance/prior_accuracy.csv';prior=list(csv.DictReader(summary.open()));dsmap=dict(s11val='val',s11test='test',bolivia='bolivia',wf='wf',sturm512='sturm')
        for r in rows:
            p=next(v for v in prior if v['seed']==r['seed'] and v['precision']==r['precision'] and v['dataset']==dsmap[r['dataset']])
            for new,old in [('pooled_mIoU','pooled_mIoU'),('pooled_water_IoU','pooled_water_IoU'),('equal_event_water_IoU','event_water_IoU'),('precision_water','water_precision'),('recall_water','water_recall')]:assert abs(float(r[new])-float(p[old]))<=.00000051
        # A cached row with altered mask support must be rejected, not counted as complete.
        e=next(z for z in models if z['id']=='R4_s42_fp32');data=collect_native.imported(e,'s11val')['rows'];bad=copy.deepcopy(data);bad[0]['counts_tn_fp_fn_tp'][0]+=1
        try:validate_rows(temp,'s11val',bad,imported=True)
        except ValueError:pass
        else:raise AssertionError('Changed confusion support was admitted')
        # Bad imports cannot overwrite an existing asset or publish mismatching bytes.
        prepare_inputs.ROOT=temp;dest=temp/'assets/example.npz';dest.parent.mkdir();dest.write_bytes(b'keep')
        try:prepare_inputs.write_file('assets/example.npz','0'*64,io.BytesIO(b'wrong'))
        except ValueError:pass
        else:raise AssertionError('Conflicting asset overwritten')
        assert dest.read_bytes()==b'keep'
        try:prepare_inputs.write_file('assets/new.npz','0'*64,io.BytesIO(b'wrong'))
        except ValueError:pass
        else:raise AssertionError('Wrong asset hash admitted')
        assert not (temp/'assets/new.npz').exists()
    profile=dict(runA_returncode=0,runB_returncode=0,throughput_qps_runA=100);steady=dict(returncode=0,steady_samples=20,throughput_qps=100,steady_cpu_gpu_cv_mean_w=8,steady_energy_mJ_per_image=80);state=dict(profile_returncode=0,steady_returncode=0)
    assert benchmark_receipts_valid(profile,steady,state)
    assert not benchmark_receipts_valid(dict(profile,runA_returncode=1),steady,state)
    assert not benchmark_receipts_valid(profile,dict(steady,steady_samples=0),state)
    assert not benchmark_receipts_valid(profile,dict(steady,steady_energy_mJ_per_image=None),state)
    assert not benchmark_receipts_valid(profile,dict(steady,steady_energy_mJ_per_image=1),state)
    report=dict(status='PASS_OFFLINE_HANDOFF',engine_entries=51,primary_slots=48,unavailable_b0_pure=4,reused_native_engines=16,new_build_attempts=35,full_source_float_parity_records=12,old_native_accuracy_rows_reproduced=80,wrong_asset_hash_and_overwrite_guard='PASS',changed_cached_counts_guard='PASS',failed_or_incomplete_benchmark_guard='PASS',missing_native_work_guard='PASS',python38_syntax_and_cli='PASS',native_execution='PENDING_JETSON',source_input_preflight=read(ROOT/'provenance/HANDOFF_INPUT_PARITY.json')['status'],b0_fp32_island_negative_guards='PASS',b0_repaired_source_candidates=4,quantizer_structural_precision_guard='PASS',committed_dry_arrays=159,dry_array_hashes_and_valid_support='PASS',dry_only_without_external_roots='PASS')
    (ROOT/'provenance/HANDOFF_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
