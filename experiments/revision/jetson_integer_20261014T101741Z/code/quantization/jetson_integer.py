"""Static QDQ PTQ for the preserved JetPack 5.0.2 deployment target.

QDQ describes rounding; it does not prove integer kernel execution. All graph
coverage reports deliberately leave that claim to the device engine inspector.
"""
import copy
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)


def session(path, optimized_path=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    if optimized_path:
        options.optimized_model_filepath = str(optimized_path)
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


class ArrayReader(CalibrationDataReader):
    def __init__(self, images):
        self.images = images
        self.rewind()

    def get_next(self):
        try:
            return {'image': next(self.iterator).astype(np.float32, copy=False)[None]}
        except StopIteration:
            return None

    def rewind(self):
        self.iterator = iter(self.images)


def jp50_weights(model):
    """Use float weight -> Q -> DQ, the existing TRT8.4 workaround.

    Validate numerical identity after transformation. The source INT8 array is
    exactly reconstructed as q * scale, so this is a representation change.
    """
    initializers = {i.name: i for i in model.graph.initializer}
    nodes = []
    removed = set()
    for node in model.graph.node:
        if node.op_type == 'DequantizeLinear' and node.input[0] in initializers:
            init = initializers[node.input[0]]
            if init.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.INT32):
                value = numpy_helper.to_array(init)
                scale = numpy_helper.to_array(initializers[node.input[1]])
                zero = numpy_helper.to_array(initializers[node.input[2]])
                axis = next((a.i for a in node.attribute if a.name == 'axis'), 1)
                shape = [1] * value.ndim
                if scale.ndim:
                    shape[axis] = scale.size
                weight = ((value.astype(np.float32) - zero.reshape(shape)) * scale.reshape(shape)).astype(np.float32)
                if init.data_type == onnx.TensorProto.INT32:
                    # TRT8.4 Q/DQ accepts INT8, not ORT's INT32 bias DQ.
                    # Preserve the rounded bias value as a float initializer;
                    # TensorRT quantizes constant bias for its INT32 accumulator.
                    model.graph.initializer.append(numpy_helper.from_array(weight, node.output[0]))
                    removed.add(init.name)
                    continue
                fname, qname = init.name + '_jp50_float', init.name + '_jp50_quantized'
                model.graph.initializer.append(numpy_helper.from_array(weight, fname))
                attrs = {'axis': axis} if scale.ndim else {}
                nodes.append(helper.make_node('QuantizeLinear', [fname] + list(node.input[1:]),
                                               [qname], name=qname, **attrs))
                node.input[0] = qname
                removed.add(init.name)
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Keep an initializer if another node still consumes it.
    used = {name for node in nodes for name in node.input}
    kept = [i for i in model.graph.initializer if i.name not in removed or i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.ir_version = min(model.ir_version, 10)  # ORT1.19 / ONNX1.17 target
    onnx.checker.check_model(model)
    return model


def ptq(float_path, output, images, *, method='MinMax', per_channel=True,
        symmetric=True, operators=('Conv', 'MatMul', 'Add')):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = onnx.load(float_path)
    # Shape inference before calibration, as recommended by ORT; avoid automatic
    # optimizer rewrites that make matching checkpoints to graph nodes difficult.
    source = onnx.shape_inference.infer_shapes(source)
    source.ir_version = min(source.ir_version, 10)
    prepared = output.with_name(output.stem + '_prepared.onnx')
    onnx.save(source, prepared)
    raw = output.with_name(output.stem + '_ort.onnx')
    quantize_static(str(prepared), str(raw), ArrayReader(images), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8 if symmetric else QuantType.QUInt8,
        weight_type=QuantType.QInt8, per_channel=per_channel,
        calibrate_method=getattr(CalibrationMethod, method), op_types_to_quantize=list(operators),
        extra_options={'ActivationSymmetric': symmetric, 'WeightSymmetric': True,
                       'CalibTensorRangeSymmetric': symmetric, 'CalibPercentile': 99.99,
                       'DedicatedQDQPair': False})
    model = onnx.load(raw)
    onnx.save(jp50_weights(model) if symmetric else model, output)
    original, target = session(raw), session(output)
    x = np.asarray(images[0], np.float32)[None]
    a, b = original.run(None, {'image': x})[0], target.run(None, {'image': x})[0]
    if not np.array_equal(a, b):
        raise RuntimeError('JetPack weight representation changed output: ' + str(np.max(np.abs(a-b))))
    return graph_audit(output)


def graph_audit(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    counts = Counter(n.op_type for n in model.graph.node)
    learned = []
    barriers = []
    for node in model.graph.node:
        if node.op_type in ('Conv', 'MatMul', 'Add', 'Mul', 'Div', 'Resize', 'LayerNormalization'):
            inputs_qdq = [producers.get(inp) is not None and producers[inp].op_type == 'DequantizeLinear'
                          for inp in node.input[:2 if node.op_type in ('Conv', 'MatMul', 'Add') else 1]]
            output_qdq = any(c.op_type == 'QuantizeLinear' for c in consumers.get(node.output[0], []))
            row = dict(name=node.name, op=node.op_type, qdq_inputs=inputs_qdq, output_qdq=output_qdq)
            learned.append(row)
            if node.op_type in ('Div', 'Resize', 'LayerNormalization') or not all(inputs_qdq):
                barriers.append(row)
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    invalid = []
    weight_names = {n.input[1] for n in model.graph.node if n.op_type == 'Conv'}
    for n in model.graph.node:
        if n.op_type not in ('QuantizeLinear', 'DequantizeLinear'):
            continue
        scale, zp = inits.get(n.input[1]), inits.get(n.input[2])
        if scale is None or not np.isfinite(scale).all() or (scale <= 0).any():
            invalid.append(n.name + ': missing/nonpositive scale')
        if zp is None or zp.dtype != np.int8 or np.any(zp):
            invalid.append(n.name + ': nonzero or non-INT8 zero point')
        if scale is not None and scale.size > 1:
            # Only constant convolution weights may use per-channel scales.
            dq_output_is_weight = any(out in weight_names for out in n.output)
            q_output_is_weight = any(c.op_type == 'DequantizeLinear' and
                any(out in weight_names for out in c.output) for c in consumers.get(n.output[0], []))
            if not (dq_output_is_weight or q_output_is_weight):
                invalid.append(n.name + ': per-channel activation scale')
    return dict(path=str(path), bytes=Path(path).stat().st_size, ir_version=model.ir_version,
                opsets={o.domain: o.version for o in model.opset_import}, operators=dict(counts),
                compute_nodes=learned, precision_barriers=barriers,
                jp50_qdq_compatible=not invalid, incompatibilities=invalid,
                full_integer_execution='unverified_requires_device_engine_audit',
                note='Float host IO, INT32 bias/accumulation and shape indices must be disclosed separately.')


def scores(confusion):
    conf = np.asarray(confusion, np.int64)
    union = conf.sum(0) + conf.sum(1) - conf.diagonal()
    iou = np.divide(conf.diagonal(), union, out=np.full(2, np.nan), where=union > 0)
    return {'mIoU': float(np.nanmean(iou)),
            'water_IoU': float(iou[1]) if np.isfinite(iou[1]) else None,
            'confusion': conf.tolist()}


def compare_metrics(rows, reference, seed=1729, replicates=2000):
    """Paired whole-event bootstrap; both pooled and equal-event estimands."""
    events = sorted({r['event'] for r in rows})
    a, b = [], []
    for event in events:
        a.append(np.sum([r['confusion'] for r in rows if r['event'] == event], axis=0))
        b.append(np.sum([r['confusion'] for r in reference if r['event'] == event], axis=0))
    a, b = np.asarray(a), np.asarray(b)
    macro_a = np.mean([scores(v)['mIoU'] for v in a])
    macro_b = np.mean([scores(v)['mIoU'] for v in b])
    rng = np.random.default_rng(seed)
    pooled, macro = [], []
    for _ in range(replicates):
        ids = rng.integers(0, len(events), len(events))
        pooled.append(scores(a[ids].sum(0))['mIoU']-scores(b[ids].sum(0))['mIoU'])
        macro.append(np.mean([scores(a[i])['mIoU']-scores(b[i])['mIoU'] for i in ids]))
    return dict(pooled=scores(a.sum(0)), equal_event_mIoU=float(macro_a), events=events,
                pooled_delta=scores(a.sum(0))['mIoU']-scores(b.sum(0))['mIoU'],
                equal_event_delta=float(macro_a-macro_b),
                pooled_delta_whole_event_CI95=np.quantile(pooled, [.025,.975]).tolist(),
                equal_event_delta_whole_event_CI95=np.quantile(macro, [.025,.975]).tolist(),
                bootstrap_replicates=replicates, bootstrap_seed=seed)
