#!/usr/bin/env python3
"""Probe which upsampling operators TensorRT 8.4 will execute in INT8.

The R4 engine is INT8 on every convolution and pointwise layer, and floating only
on its three bilinear Resize layers. `--mode strict` proves TensorRT 8.4 has no
INT8-obedient Resize, in either `linear` or `nearest` mode. The practical question
is therefore what to replace Resize with. R4's upsampling factors are 2, 2 and 4 -
all integer - so bilinear upsampling is exactly expressible as a transposed
convolution with fixed weights, which is an op family TensorRT does quantize.

This builds minimal QDQ graphs and reports, per candidate op, whether a strict
INT8 build succeeds and whether the inspector shows an all-INT8 tensor path.
Nothing here touches the deployment candidates; it is a capability probe.
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

ACT_SCALE = np.float32(1.0 / 64.0)
WEIGHT_SCALE = np.float32(1.0 / 128.0)


def bilinear_kernel(scale):
    """1D bilinear weights for integer upsampling by `scale` (kernel 2*scale)."""
    size = 2 * scale
    centre = (size - 1) / 2.0
    return np.array([1.0 - abs(i - centre) / scale for i in range(size)], np.float32)


def qdq(name, src, scale, zero, out):
    return [helper.make_node('QuantizeLinear', [src, scale, zero], [name + '_q'], name=name + '_Q'),
            helper.make_node('DequantizeLinear', [name + '_q', scale, zero], [out], name=name + '_DQ')]


def build_graph(kind, channels, size, scale):
    inits = [
        numpy_helper.from_array(np.array(ACT_SCALE, np.float32), 'act_scale'),
        numpy_helper.from_array(np.array(0, np.int8), 'act_zero'),
    ]
    nodes = qdq('in', 'image', 'act_scale', 'act_zero', 'x')

    if kind == 'resize_linear' or kind == 'resize_nearest':
        sizes = np.array([1, channels, size * scale, size * scale], np.int64)
        inits.append(numpy_helper.from_array(sizes, 'sizes'))
        nodes.append(helper.make_node(
            'Resize', ['x', '', '', 'sizes'], ['y'], name='Up',
            mode='linear' if kind.endswith('linear') else 'nearest',
            coordinate_transformation_mode='half_pixel', nearest_mode='floor'))
    else:
        groups = channels if kind == 'deconv_depthwise' else 1
        k1 = bilinear_kernel(scale)
        kernel2d = np.outer(k1, k1).astype(np.float32)
        if groups == channels:
            weight = np.tile(kernel2d[None, None], (channels, 1, 1, 1))
        else:
            weight = np.zeros((channels, channels, 2 * scale, 2 * scale), np.float32)
            for c in range(channels):
                weight[c, c] = kernel2d
        # TRT 8.4's ONNX parser rejects INT8 weight initializers ("invalid weights type
        # of Int8"), so use the float-weight Q/DQ workaround the deployment exports use:
        # a float initializer rounded onto the quantization grid, then Q -> DQ.
        wq = np.clip(np.round(weight / WEIGHT_SCALE), -127, 127).astype(np.float32) * WEIGHT_SCALE
        inits += [numpy_helper.from_array(wq, 'w_f'),
                  numpy_helper.from_array(np.array(WEIGHT_SCALE, np.float32), 'w_scale'),
                  numpy_helper.from_array(np.array(0, np.int8), 'w_zero')]
        nodes.append(helper.make_node('QuantizeLinear', ['w_f', 'w_scale', 'w_zero'],
                                      ['w_q'], name='W_Q'))
        nodes.append(helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zero'],
                                      ['w'], name='W_DQ'))
        nodes.append(helper.make_node(
            'ConvTranspose', ['x', 'w'], ['y'], name='Up', group=groups,
            kernel_shape=[2 * scale, 2 * scale], strides=[scale, scale],
            pads=[scale // 2, scale // 2, scale // 2, scale // 2]))

    nodes += qdq('out', 'y', 'act_scale', 'act_zero', 'logits')
    graph = helper.make_graph(
        nodes, 'probe_' + kind,
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, channels, size, size])],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, channels, None, None])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    return model


def strict_build(model_path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(model_path).read_bytes()):
        return dict(status='parse_failed',
                    reason='; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.max_workspace_size = 512 * 1048576
    config.set_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    structural = {'CONSTANT', 'SHAPE', 'SHUFFLE', 'QUANTIZE', 'DEQUANTIZE', 'IDENTITY'}
    forced = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        kind = str(layer.type).split('.')[-1]
        if kind in structural:
            continue
        layer.precision = trt.int8
        for j in range(layer.num_outputs):
            out = layer.get_output(j)
            if not out.is_shape_tensor and out.dtype not in (trt.int32, trt.bool):
                layer.set_output_type(j, trt.int8)
        forced.append(kind)
    try:
        engine = builder.build_engine(network, config)
    except Exception as error:
        return dict(status='build_raised', reason=repr(error), forced=forced)
    if engine is None:
        return dict(status='no_engine_under_int8_constraints', forced=forced)
    context = engine.create_execution_context()
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    audit = inspect_layers(layers)
    return dict(status='built', forced=forced,
                full_integer_tensor_path=audit['full_integer_tensor_path'],
                floating_compute=[r.get('Name') for r in audit['floating_compute_layers']],
                boundary_reformats=len(audit['boundary_reformats']),
                layer_types=sorted({r.get('LayerType') for r in
                                    (layers['Layers'] if isinstance(layers, dict) else layers)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # (channels, input size, scale) mirror R4's three real upsampling sites.
    cases = [('resize_linear', 16, 32, 2), ('resize_nearest', 16, 32, 2),
             ('deconv_depthwise', 16, 32, 2), ('deconv_dense', 16, 32, 2),
             ('deconv_depthwise', 2, 128, 4), ('deconv_dense', 2, 128, 4)]
    results = []
    for kind, channels, size, scale in cases:
        tag = '%s_c%d_s%d_x%d' % (kind, channels, size, scale)
        path = args.out_dir / (tag + '.onnx')
        model = build_graph(kind, channels, size, scale)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        outcome = strict_build(path)
        outcome.update(case=tag, op=kind, channels=channels, input_size=size, scale=scale)
        results.append(outcome)
        print('%-28s -> %-34s int8_path=%s %s' % (
            tag, outcome['status'], outcome.get('full_integer_tensor_path'),
            ('floating=%s' % outcome.get('floating_compute')) if outcome.get('floating_compute') else ''))

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(dict(
        purpose='which upsampling op TensorRT 8.4 will execute under INT8 precision constraints',
        note='capability probe on synthetic graphs; deployment candidates untouched',
        activation_scale=float(ACT_SCALE), weight_scale=float(WEIGHT_SCALE),
        results=results), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
