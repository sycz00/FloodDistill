#!/usr/bin/env python3
"""Replace bilinear Resize with an exactly equivalent fixed-weight ConvTranspose.

Motivation. The R4 INT8 engine is INT8 on every convolution and pointwise layer and
floating only on its three bilinear Resize layers plus the reformats that feed them,
because TensorRT 8.4 has no INT8 Resize implementation in any mode
(`diagnostics/int8_upsample_probe.json`). Dense (groups=1) ConvTranspose *is* INT8.

Exactness. For an integer upsampling factor s, bilinear interpolation with the
`half_pixel` / align_corners=False convention is the transposed convolution with the
separable triangular kernel of size 2s, stride s and padding s//2. This is an identity,
not an approximation, in the interior. The kernel values are also exactly representable
on an INT8 grid at scale 1/128: for s=2 the 1D weights are {0.25, 0.75} giving products
in {0.0625, 0.1875, 0.5625} -> {8, 24, 72}; for s=4 they are {k/8} giving products that
are multiples of 1/64 -> even integers with max 98. No weight is rounded away, so no
retraining or recalibration is involved.

Known difference. Resize clamps at the border; ConvTranspose implicitly zero-pads. The
two therefore disagree on a thin border ring. `--edge replicate` pre-pads the input by
one pixel with edge replication (Slice + Concat, both data-movement ops) and widens the
convolution padding to compensate, which removes that difference. `--edge zero` keeps the
plain form so the two can be compared. Either way this is a graph change and so a NEW
variant requiring source validation; it is not an amendment of the deployment candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WEIGHT_SCALE = np.float32(1.0 / 128.0)


def bilinear_1d(scale):
    """align_corners=False bilinear weights for integer upsampling by `scale`."""
    size = 2 * scale
    factor = (size + 1) // 2
    centre = factor - 1 if size % 2 else factor - 0.5
    grid = np.arange(size, dtype=np.float64)
    return 1.0 - np.abs(grid - centre) / factor


def resize_geometry(model, input_shape=(1, 6, 512, 512)):
    """Resolve each Resize node's input shape and integer upsampling factor via ORT."""
    import onnxruntime as ort
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    wanted = []
    for node in model.graph.node:
        if node.op_type != 'Resize':
            continue
        wanted.append(node.input[0])
        wanted.append(node.output[0])
    for name in wanted:
        if name not in existing:
            probe.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    session = ort.InferenceSession(probe.SerializeToString(), providers=['CPUExecutionProvider'])
    outputs = session.run(None, {session.get_inputs()[0].name:
                                 np.zeros(input_shape, dtype=np.float32)})
    values = dict(zip([o.name for o in session.get_outputs()], outputs))
    geometry = {}
    for node in model.graph.node:
        if node.op_type != 'Resize':
            continue
        src, dst = values[node.input[0]], values[node.output[0]]
        if src.shape[1] != dst.shape[1]:
            raise RuntimeError('Resize changes channels: ' + node.name)
        sh, sw = dst.shape[2] / src.shape[2], dst.shape[3] / src.shape[3]
        if sh != sw or sh != int(sh):
            raise RuntimeError('Non-integer or anisotropic factor on ' + node.name)
        geometry[node.name] = dict(channels=int(src.shape[1]), in_hw=list(src.shape[2:]),
                                   out_hw=list(dst.shape[2:]), scale=int(sh))
    return geometry


def dense_bilinear_weight(channels, scale):
    k1 = bilinear_1d(scale)
    k2 = np.outer(k1, k1)
    weight = np.zeros((channels, channels, 2 * scale, 2 * scale), np.float64)
    for c in range(channels):
        weight[c, c] = k2
    return weight


def quantize_exact(weight):
    """Round onto the INT8 grid and verify nothing was actually rounded."""
    grid = weight / WEIGHT_SCALE
    rounded = np.round(grid)
    if np.abs(grid - rounded).max() > 1e-9:
        raise RuntimeError('bilinear weights are not exact on the INT8 grid')
    if np.abs(rounded).max() > 127:
        raise RuntimeError('bilinear weights exceed INT8 range')
    return (rounded * WEIGHT_SCALE).astype(np.float32), int(np.abs(rounded).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--edge', choices=['zero', 'replicate'], default='replicate')
    args = parser.parse_args()
    if args.out.exists() or args.receipt.exists():
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')

    model = onnx.load(str(args.model))
    geometry = resize_geometry(model)
    if not geometry:
        parser.error('no Resize nodes found')

    graph = model.graph
    replaced = []
    for node in list(graph.node):
        if node.op_type != 'Resize':
            continue
        info = geometry[node.name]
        channels, scale = info['channels'], info['scale']
        source, target = node.input[0], node.output[0]
        tag = node.name.strip('/').replace('/', '_') or 'resize'

        weight = dense_bilinear_weight(channels, scale)
        weight_f, peak = quantize_exact(weight)
        w_name = tag + '_bilinear_w'
        graph.initializer.extend([
            numpy_helper.from_array(weight_f, w_name),
            numpy_helper.from_array(np.array(WEIGHT_SCALE, np.float32), tag + '_w_scale'),
            numpy_helper.from_array(np.array(0, np.int8), tag + '_w_zero')])

        new_nodes = [
            helper.make_node('QuantizeLinear', [w_name, tag + '_w_scale', tag + '_w_zero'],
                             [tag + '_w_q'], name=tag + '_W_Q'),
            helper.make_node('DequantizeLinear', [tag + '_w_q', tag + '_w_scale', tag + '_w_zero'],
                             [tag + '_w_dq'], name=tag + '_W_DQ')]

        conv_input, pad = source, scale // 2
        if args.edge == 'replicate':
            # Edge-replicate by one pixel on each side using Slice + Concat, then absorb
            # the extra ring by widening the deconvolution padding by `scale`.
            for axis, (lo_name, hi_name) in ((2, ('_top', '_bot')), (3, ('_left', '_right'))):
                a = np.array([axis], np.int64)
                graph.initializer.extend([
                    numpy_helper.from_array(np.array([0], np.int64), tag + lo_name + '_s'),
                    numpy_helper.from_array(np.array([1], np.int64), tag + lo_name + '_e'),
                    numpy_helper.from_array(np.array([-1], np.int64), tag + hi_name + '_s'),
                    numpy_helper.from_array(np.array([np.iinfo(np.int64).max], np.int64),
                                            tag + hi_name + '_e'),
                    numpy_helper.from_array(a, tag + lo_name + '_a'),
                    numpy_helper.from_array(a, tag + hi_name + '_a')])
                new_nodes += [
                    helper.make_node('Slice', [conv_input, tag + lo_name + '_s',
                                               tag + lo_name + '_e', tag + lo_name + '_a'],
                                     [tag + lo_name], name=tag + lo_name + '_Slice'),
                    helper.make_node('Slice', [conv_input, tag + hi_name + '_s',
                                               tag + hi_name + '_e', tag + hi_name + '_a'],
                                     [tag + hi_name], name=tag + hi_name + '_Slice'),
                    helper.make_node('Concat', [tag + lo_name, conv_input, tag + hi_name],
                                     [tag + '_pad%d' % axis], axis=axis,
                                     name=tag + '_Concat%d' % axis)]
                conv_input = tag + '_pad%d' % axis
            pad = scale // 2 + scale

        new_nodes.append(helper.make_node(
            'ConvTranspose', [conv_input, tag + '_w_dq'], [target], name=tag + '_Deconv',
            group=1, kernel_shape=[2 * scale, 2 * scale], strides=[scale, scale],
            pads=[pad, pad, pad, pad]))

        index = list(graph.node).index(node)
        graph.node.remove(node)
        for offset, new in enumerate(new_nodes):
            graph.node.insert(index + offset, new)
        replaced.append(dict(resize_node=node.name, channels=channels, scale=scale,
                             in_hw=info['in_hw'], out_hw=info['out_hw'],
                             deconv=tag + '_Deconv', kernel=2 * scale, stride=scale,
                             pads=pad, peak_int8_weight=peak, edge=args.edge))

    onnx.checker.check_model(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))
    record = dict(
        status='derived_variant', recipe='integer_upsample_deconv_%s' % args.edge,
        is_registered_candidate=False,
        source_model=str(args.model),
        source_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        variant_model=str(args.out),
        variant_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        weight_scale=float(WEIGHT_SCALE), weights_exact_on_int8_grid=True,
        replaced=replaced,
        rationale='TensorRT 8.4 has no INT8 Resize in any mode; dense ConvTranspose is INT8. '
                  'For integer factors, bilinear == transposed convolution with the separable '
                  'triangular kernel, so the substitution is an identity in the interior.',
        caveat='Graph change: a new variant requiring CPU source validation against the '
               'unpatched candidate. Border handling differs from Resize unless '
               'edge=replicate.')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'replaced'}, indent=2))
    for item in replaced:
        print('  %-12s C=%-3d x%d  %s -> %s  kernel=%d stride=%d pads=%d peak_w=%d' % (
            item['resize_node'], item['channels'], item['scale'], item['in_hw'],
            item['out_hw'], item['kernel'], item['stride'], item['pads'],
            item['peak_int8_weight']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
