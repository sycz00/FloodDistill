#!/usr/bin/env python3
"""Probe whether TensorRT 8.4 can execute EfficientViT's attention in INT8.

In the shipped B0 QAT export there is no Q/DQ anywhere around the eight LiteMLA
MatMuls or the normalization Divs, so TensorRT has no scales for them and runs them
FP32 (or FP16 when that flag is set). Before any further QAT is spent on quantizing
the attention, it is worth knowing whether this TensorRT version could run such a
graph in INT8 at all.

These are activation x activation ("dynamic") batched MatMuls, which is the hard
case: both operands are runtime tensors, not weights. Shapes mirror the real engine:

    stage2 mm1 [1,8,17,1024] x [1,8,1024,16] -> [1,8,17,16]
    stage2 mm2 [1,8,17,16]   x [1,8,16,1024] -> [1,8,17,1024]
    stage3 mm1 [1,16,17,256] x [1,16,256,16] -> [1,16,17,16]
    stage3 mm2 [1,16,17,16]  x [1,16,16,256] -> [1,16,17,256]

Note the mm1 reduction length (1024 / 256): with INT8 operands and INT32
accumulation the worst case is about 1024 * 127 * 127 = 1.65e7, well inside INT32,
so integer accumulation is numerically safer here than FP16 ever was.
Deployment candidates are untouched; this is a capability probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integer_tools import inspect_layers  # noqa: E402

SCALE = np.float32(1.0 / 64.0)


def qdq_chain(tag, src, out):
    return [helper.make_node('QuantizeLinear', [src, 'scale', 'zero'], [tag + '_q'], name=tag + '_Q'),
            helper.make_node('DequantizeLinear', [tag + '_q', 'scale', 'zero'], [out], name=tag + '_DQ')]


def build_graph(a_shape, b_shape, with_div):
    inits = [numpy_helper.from_array(np.array(SCALE, np.float32), 'scale'),
             numpy_helper.from_array(np.array(0, np.int8), 'zero')]
    nodes = qdq_chain('a', 'a_in', 'a_dq') + qdq_chain('b', 'b_in', 'b_dq')
    nodes.append(helper.make_node('MatMul', ['a_dq', 'b_dq'], ['mm'], name='Attn_MatMul'))
    last = 'mm'
    if with_div:
        # LiteMLA divides the aggregated numerator by its last row; approximate the
        # shape interaction with a broadcast Div by a quantized activation.
        nodes += qdq_chain('d', 'd_in', 'd_dq')
        nodes.append(helper.make_node('Div', ['mm', 'd_dq'], ['dv'], name='Attn_Div'))
        last = 'dv'
    nodes += qdq_chain('out', last, 'y')
    inputs = [helper.make_tensor_value_info('a_in', TensorProto.FLOAT, list(a_shape)),
              helper.make_tensor_value_info('b_in', TensorProto.FLOAT, list(b_shape))]
    out_shape = list(a_shape[:-1]) + [b_shape[-1]]
    if with_div:
        inputs.append(helper.make_tensor_value_info(
            'd_in', TensorProto.FLOAT, list(a_shape[:-2]) + [1, b_shape[-1]]))
    graph = helper.make_graph(
        nodes, 'probe_attn', inputs,
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, out_shape)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    return model


def strict_build(path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(path).read_bytes()):
        return dict(status='parse_failed',
                    reason='; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.max_workspace_size = 512 * 1048576
    config.set_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    structural = {'CONSTANT', 'SHAPE', 'SHUFFLE', 'QUANTIZE', 'DEQUANTIZE', 'IDENTITY'}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if str(layer.type).split('.')[-1] in structural:
            continue
        layer.precision = trt.int8
        for j in range(layer.num_outputs):
            out = layer.get_output(j)
            if not out.is_shape_tensor and out.dtype not in (trt.int32, trt.bool):
                layer.set_output_type(j, trt.int8)
    try:
        engine = builder.build_engine(network, config)
    except Exception as error:
        return dict(status='build_raised', reason=repr(error))
    if engine is None:
        return dict(status='no_engine_under_int8_constraints')
    context = engine.create_execution_context()
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    audit = inspect_layers(layers)
    rows = layers['Layers'] if isinstance(layers, dict) else layers
    return dict(status='built', full_integer_tensor_path=audit['full_integer_tensor_path'],
                floating_compute=[r.get('Name') for r in audit['floating_compute_layers']],
                boundary_reformats=len(audit['boundary_reformats']),
                layer_types=sorted({r.get('LayerType') for r in rows}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ('stage2_mm1', (1, 8, 17, 1024), (1, 8, 1024, 16), False),
        ('stage2_mm2', (1, 8, 17, 16), (1, 8, 16, 1024), False),
        ('stage3_mm1', (1, 16, 17, 256), (1, 16, 256, 16), False),
        ('stage3_mm2', (1, 16, 17, 16), (1, 16, 16, 256), False),
        ('stage2_mm2_with_div', (1, 8, 17, 16), (1, 8, 16, 1024), True),
    ]
    results = []
    for tag, a, b, with_div in cases:
        path = args.out_dir / (tag + '.onnx')
        model = build_graph(a, b, with_div)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        outcome = strict_build(path)
        outcome.update(case=tag, a_shape=list(a), b_shape=list(b), with_div=with_div)
        results.append(outcome)
        print('%-22s -> %-34s int8_path=%s floating=%s' % (
            tag, outcome['status'], outcome.get('full_integer_tensor_path'),
            outcome.get('floating_compute')))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(dict(
        purpose='can TensorRT 8.4 run QDQ activation-by-activation batched MatMul (LiteMLA '
                'attention) under INT8 precision constraints',
        note='capability probe on synthetic graphs; deployment candidates untouched',
        activation_scale=float(SCALE), results=results), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
