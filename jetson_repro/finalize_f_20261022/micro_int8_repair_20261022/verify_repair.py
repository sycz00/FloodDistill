#!/usr/bin/env python3
"""Offline graph, frozen-source, and negative admission checks. No native claims."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from evaluate_native import summarize
from validate_results import validate_rows
from repair_policy import integer_certificate, source_gate


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rejected(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return
    raise AssertionError('Invalid admission fixture accepted')


def verify_collapsed_evaluator(report, model):
    """Exercise the actual evaluator with an explicitly fake all-zero device."""
    import evaluate_native
    import dataset_io
    import device_runtime
    from unittest.mock import patch

    class ZeroFixtureRunner:
        def __init__(self, *args):
            pass

        def __call__(self, image):
            return np.zeros((1, 2, 512, 512), np.int8)

    original_images = dataset_io.images
    reads = []

    def source_only(ds, sturm_root, wf_root):
        reads.append(ds)
        if ds != 's11val':
            raise AssertionError('Collapsed fixture reached external evaluation')
        return original_images(ds, sturm_root, wf_root)

    modules = {n: types.ModuleType(n) for n in ('tensorrt', 'pycuda', 'pycuda.driver', 'pycuda.autoinit')}
    modules['tensorrt'].__version__ = '8.4.FIXTURE_NOT_NATIVE'
    modules['pycuda'].__path__ = []
    with tempfile.TemporaryDirectory(prefix='micro_collapsed_evaluator_fixture_') as folder:
        temp = Path(folder)
        (temp / 'inventories').symlink_to(ROOT / 'inventories')
        engine = temp / 'FAKE_NOT_AN_ENGINE'
        engine.write_bytes(b'OFFLINE_GUARD_FIXTURE_ONLY')
        baseline = temp / 'baseline.json'
        baseline.write_text(json.dumps(dict(pooled=dict(miou=report['baseline']['mIoU'], water_iou=report['baseline']['water_IoU']), equal_event=dict(miou=report['baseline']['equal_event_mIoU']))))
        argv = ['fixture', '--engine', str(engine), '--io', str(ROOT / model['io']),
                '--out', str(temp / 'out'), '--sturm-root', str(temp / 'unused'),
                '--wf-root', str(temp / 'unused'), '--quant-source-reference', str(baseline)]
        with patch.dict(sys.modules, modules), patch.object(evaluate_native, 'ROOT', temp), patch.object(device_runtime, 'IntegerRunner', ZeroFixtureRunner), patch.object(dataset_io, 'images', source_only), patch.object(sys, 'argv', argv):
            try:
                evaluate_native.main()
            except RuntimeError as error:
                assert 'native source gate' in str(error)
            else:
                raise AssertionError('Collapsed evaluator fixture was admitted')
        assert reads == ['s11val']
        assert not json.loads((temp / 'out/NATIVE_SOURCE_GATE.json').read_text())['passed']
        assert not (temp / 'out/COMPLETE.json').exists()


def main():
    manifest = json.loads((ROOT / 'RUN_MANIFEST.json').read_text())
    assert len(manifest['engines']) == 8 and len(manifest['primary_matrix']) == 4
    assert {s['seed'] for s in manifest['primary_matrix']} == {42, 64, 91, 107}
    assert not manifest['unsupported_slots']
    inventory_counts = {'s11val': 89, 's11test': 90, 'bolivia': 15, 'wf': 836, 'sturm512': 2675}
    for ds, count in inventory_counts.items():
        assert len((ROOT / 'inventories' / (ds + '.jsonl')).read_text().splitlines()) == count
    results = []
    collapsed_fixture_done = False
    for model in manifest['engines']:
        path = ROOT / model['onnx']
        assert sha(path) == model['onnx_sha256']
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        assert [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim] == [1, 6, 512, 512]
        assert [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim] == [1, 2, 512, 512]
        if model['precision'] == 'fp32':
            assert not any(n.op_type in ('QuantizeLinear', 'DequantizeLinear') for n in graph.graph.node)
            continue
        assert model['mode'] == 'no_fp16'
        assert graph.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT8
        assert graph.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.INT8
        assert sha(ROOT / model['io']) == model['io_sha256']
        io = json.loads((ROOT / model['io']).read_text())
        assert io['variant_sha256'] == model['onnx_sha256']
        assert io['host_input_scale'] > 0 and io['output_scale'] > 0
        assert io['host_input_zero_point'] == io['output_zero_point'] == 0
        report = json.loads((ROOT / model['source_gate']).read_text())
        assert report['graph_sha256'] == model['onnx_sha256']
        assert len(report['parity']) == len(report['unrounded_all89']['rows']) == 89
        assert report['unrounded_all89']['atol'] == report['unrounded_all89']['rtol'] == .001
        assert report['unrounded_all89']['passed'] and all(r['passed'] for r in report['unrounded_all89']['rows'])
        assert not report['weights_fit'] and not report['scales_recalibrated']
        for node in graph.graph.node:
            if node.op_type in ('Conv', 'ConvTranspose'):
                assert next((a.i for a in node.attribute if a.name == 'group'), 1) == 1
        validate_rows(ROOT, 's11val', report['evaluation']['rows'])
        summary = summarize(report['evaluation']['rows'])
        base_rows = [dict(name=r['file'], event=r['event'], counts_tn_fp_fn_tp=np.asarray(r['confusion']).ravel().tolist()) for r in report['baseline']['rows']]
        validate_rows(ROOT, 's11val', base_rows, imported=True)
        gate = source_gate(summarize(base_rows), summary)
        assert gate['passed'] and report['accuracy_pass']
        for key, expected in [('mIoU', summary['pooled']['miou']), ('equal_event_mIoU', summary['equal_event']['miou']), ('water_IoU', summary['pooled']['water_iou'])]:
            assert abs(report['metrics'][key] - expected) < 1e-12
            assert abs(report['drops'][key] - (report['baseline'][key] - expected)) < 1e-12
        # A plausible-looking all-background result must block external evaluation.
        collapsed = copy.deepcopy(report['evaluation']['rows'])
        for row in collapsed:
            tn, fp, fn, tp = row['counts_tn_fp_fn_tp']
            row['counts_tn_fp_fn_tp'] = [tn + fp, 0, fn + tp, 0]
        assert not source_gate(summarize(base_rows), summarize(collapsed))['passed']
        bad = copy.deepcopy(summary)
        bad['pooled']['water_iou'] = float('nan')
        rejected(source_gate, summary, bad)
        for group, key in [('pooled', 'miou'), ('equal_event', 'miou'), ('pooled', 'water_iou')]:
            bad = copy.deepcopy(summary)
            bad[group][key] -= .020001
            assert not source_gate(summary, bad)['passed']
        layer = {'Layers': [{'Name': 'fixture', 'Inputs': [{'Format/Datatype': 'Row major linear Int8'}], 'Outputs': [{'Format/Datatype': 'Row major linear Int8'}]}]}
        build = dict(status='built', model_sha256=model['onnx_sha256'], precision_audit={'full_integer_tensor_path': True}, linear_host_bindings=True, integer_host_bindings=True)
        assert integer_certificate(build, layer, model)
        for dtype in ['Row major linear Float', 'Row major linear FP16', '', 'unrecognized']:
            bad = copy.deepcopy(layer)
            bad['Layers'][0]['Outputs'][0]['Format/Datatype'] = dtype
            rejected(integer_certificate, build, bad, model)
        rejected(integer_certificate, dict(build, model_sha256='0' * 64), layer, model)
        rejected(integer_certificate, dict(build, linear_host_bindings=False), layer, model)
        rejected(integer_certificate, dict(build, integer_host_bindings=False), layer, model)
        if not collapsed_fixture_done and (ROOT / 'assets/source_val').is_dir():
            verify_collapsed_evaluator(report, model)
            collapsed_fixture_done = True
        results.append(dict(seed=model['seed'], drops=report['drops'], unrounded_parity=True,
                            optimized_exact_all89=report['optimized_exact_all89'], semantic_exact_all89=report['semantic_exact_all89']))
    for p in ROOT.rglob('*.py'):
        if '__pycache__' not in p.parts:
            ast.parse(p.read_text(), feature_version=(3, 8))
    for name in ['run_remaining.py', 'evaluate_native.py', 'collect_native.py', 'preflight.py', 'integer_tools.py']:
        subprocess.run([sys.executable, str(ROOT / 'runtime' / name), '--help'], check=True, stdout=subprocess.DEVNULL)
    import collect_native
    with tempfile.TemporaryDirectory(prefix='micro_repair_missing_fixture_') as folder:
        temp = Path(folder)
        for name in ('RUN_MANIFEST.json', 'models', 'inventories'):
            (temp / name).symlink_to(ROOT / name)
        out = temp / 'results/MISSING'
        out.mkdir(parents=True)
        (out / 'STATUS.json').write_text('[]\n')
        collect_native.ROOT = temp
        assert collect_native.main('MISSING') == 1
        receipt = json.loads((out / 'COLLECTION_RECEIPT.json').read_text())
        assert receipt['matrix_slots'] == 4 and receipt['observed_complete_primary_cells'] == 0
        assert not receipt['all_declared_attempts_resolved'] and not receipt['invalid_results']
    proof = dict(status='PASS_OFFLINE_MICRO_REPAIR', repair_candidates=4, exact_float_baselines=4,
                 source_images_each=89, results=results, collapsed_water_guard='PASS',
                 each_metric_0p02_guard='PASS', floating_or_unknown_inspector_guard='PASS',
                 model_identity_guard='PASS', packed_or_float_binding_guard='PASS', missing_native_results_guard='PASS',
                 evaluator_stops_before_external_guard='PASS' if collapsed_fixture_done else 'NOT_RUN_NO_SOURCE_ARRAYS',
                 python38_syntax_and_cli='PASS', native_execution='PENDING_JETSON')
    (ROOT / 'provenance/OFFLINE_VALIDATION.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
