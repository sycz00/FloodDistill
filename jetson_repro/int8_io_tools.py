#!/usr/bin/env python3
"""Give a QDQ graph an integer input/output interface, and evaluate it.

Why. After bilinear Resize is replaced by a fixed-weight ConvTranspose
(`make_integer_upsample_variant.py`) the only non-INT8 layers left in the R4 engine
are the host-IO boundary reformats: TensorRT must convert the FP32 `image` binding to
INT8 on entry and the INT8 result back to FP32 on exit. The audit counts those, so
`full_integer_tensor_path` stays False even though no arithmetic is floating.

`convert` moves those two conversions out of the engine:

  input   the leading QuantizeLinear is deleted and `image` becomes an INT8 binding
          feeding the existing DequantizeLinear. Host-side quantization with the same
          scale is mathematically the same operation, just outside the engine.
  output  a QuantizeLinear is appended so `logits` becomes an INT8 binding. Its scale
          is taken from the Dequantize that feeds the final ConvTranspose: bilinear
          weights are a partition of unity, so the upsampled logits cannot exceed the
          input range and that scale can never saturate.

Argmax is unaffected by a shared per-tensor output scale, because a single positive
scale is monotonic and both classes share it - so the segmentation decision is
preserved exactly even though the returned logits are integers.

This changes the model interface, so it is a NEW variant requiring source validation,
exactly as the handover requires. `run` is the matching runner: the float-IO runner in
integer_tools.py deliberately refuses integer bindings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def initializer_map(graph):
    return {i.name: numpy_helper.to_array(i) for i in graph.initializer}


def convert(args):
    model = onnx.load(str(args.model))
    graph = model.graph
    inits = initializer_map(graph)
    image = graph.input[0].name

    lead = [n for n in graph.node if n.op_type == 'QuantizeLinear' and n.input[0] == image]
    if len(lead) != 1:
        raise RuntimeError('expected exactly one QuantizeLinear on %s, found %d' % (image, len(lead)))
    lead = lead[0]
    in_scale = float(inits[lead.input[1]])
    in_zero = int(inits[lead.input[2]]) if len(lead.input) > 2 else 0
    if in_zero != 0:
        raise RuntimeError('non-zero input zero point is not supported by this path')
    consumers = [n for n in graph.node if lead.output[0] in n.input]
    if [n.op_type for n in consumers] != ['DequantizeLinear']:
        raise RuntimeError('leading Quantize does not feed exactly one Dequantize')
    for node in consumers:
        for i, name in enumerate(node.input):
            if name == lead.output[0]:
                node.input[i] = image
    graph.node.remove(lead)
    graph.input[0].type.tensor_type.elem_type = TensorProto.INT8

    logits = graph.output[0].name
    producer = [n for n in graph.node if logits in n.output]
    if len(producer) != 1:
        raise RuntimeError('expected exactly one producer of %s' % logits)
    producer = producer[0]
    # Scale for the output Q: the Dequantize feeding the final upsample.
    feed = producer.input[0]
    src = [n for n in graph.node if feed in n.output]
    while src and src[0].op_type not in ('DequantizeLinear',):
        feed = src[0].input[0]
        src = [n for n in graph.node if feed in n.output]
    if not src:
        raise RuntimeError('could not trace the final upsample back to a Dequantize')
    out_scale = float(inits[src[0].input[1]])

    producer.output[0] = logits + '_f'
    graph.initializer.extend([
        numpy_helper.from_array(np.array(out_scale, np.float32), 'logits_out_scale'),
        numpy_helper.from_array(np.array(0, np.int8), 'logits_out_zero')])
    graph.node.append(helper.make_node(
        'QuantizeLinear', [logits + '_f', 'logits_out_scale', 'logits_out_zero'],
        [logits], name='logits_Q'))
    graph.output[0].type.tensor_type.elem_type = TensorProto.INT8

    onnx.checker.check_model(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))
    record = dict(status='derived_variant', recipe='integer_io',
                  is_registered_candidate=False,
                  source_model=str(args.model),
                  source_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  variant_model=str(args.out),
                  variant_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                  input_binding='int8', output_binding='int8',
                  host_input_scale=in_scale, host_input_zero_point=in_zero,
                  output_scale=out_scale, output_zero_point=0,
                  argmax_preserved='a single shared positive output scale is monotonic, '
                                   'so per-pixel class order is unchanged',
                  caveat='changes the model interface; new variant requiring source validation')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


class Int8IoRunner:
    """TRT 8.4 runner for an engine with INT8 input and output bindings."""

    def __init__(self, path, in_scale):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  context kept alive with the buffers
        if not hasattr(np, 'bool'):
            np.bool = bool
        self.cuda = cuda
        self.in_scale = np.float32(in_scale)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('engine deserialization failed; rebuild on this Jetson')
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.device, self.dtypes = [], [], [], []
        self.input_id = self.output_id = None
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            host = cuda.pagelocked_empty(shape, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.host.append(host)
            self.device.append(device)
            self.dtypes.append(dtype)
            self.bindings.append(int(device))
            if self.engine.binding_is_input(i):
                self.input_id = i
            else:
                self.output_id = i
        self.int8_io = (self.dtypes[self.input_id] == np.int8,
                        self.dtypes[self.output_id] == np.int8)

    def __call__(self, image_float):
        src = self.host[self.input_id]
        if src.dtype == np.int8:
            # The same operation the deleted QuantizeLinear performed, on the host.
            q = np.rint(image_float / self.in_scale)
            np.copyto(src, np.clip(q, -127, 127).astype(np.int8))
        else:
            np.copyto(src, image_float.astype(np.float32))
        self.cuda.memcpy_htod_async(self.device[self.input_id], src, self.stream)
        if not self.context.execute_async_v2(self.bindings, self.stream.handle):
            raise RuntimeError('TensorRT execution failed')
        self.cuda.memcpy_dtoh_async(self.host[self.output_id], self.device[self.output_id],
                                    self.stream)
        self.stream.synchronize()
        return self.host[self.output_id].copy()

    def kernel_times(self, warmup, iterations):
        for _ in range(warmup):
            self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.stream.synchronize()
        times = []
        for _ in range(iterations):
            start, end = self.cuda.Event(), self.cuda.Event()
            start.record(self.stream)
            self.context.execute_async_v2(self.bindings, self.stream.handle)
            end.record(self.stream)
            end.synchronize()
            times.append(float(start.time_till(end)))
        return times


def run(args):
    meta = json.loads(Path(args.io_receipt).read_text())
    runner = Int8IoRunner(args.model, meta['host_input_scale'])
    confusion = np.zeros((2, 2), np.int64)
    rows = []
    assets = sorted(Path(args.assets).glob('*.npz'))
    if not assets:
        raise RuntimeError('no NPZ evaluation assets')
    for path in assets:
        with np.load(path, allow_pickle=False) as item:
            image, mask = item['image'][None].astype(np.float32), item['mask']
            logits = runner(image)
            if logits.shape != (1, 2) + mask.shape:
                raise RuntimeError('unexpected logits geometry from ' + path.name)
            # argmax over integer logits: a shared positive scale preserves class order.
            pred = logits[0].astype(np.int32).argmax(0)
            valid = (mask >= 0) & (mask < 2)
            conf = np.bincount((mask[valid] * 2 + pred[valid]).ravel(),
                               minlength=4).reshape(2, 2)
            confusion += conf
            rows.append(dict(file=path.name, event=str(item['event']), confusion=conf.tolist()))
    union = confusion.sum(0) + confusion.sum(1) - confusion.diagonal()
    iou = np.divide(confusion.diagonal(), union, out=np.full(2, np.nan), where=union > 0)
    wall = []
    for _ in range(args.warmup):
        runner(image)
    for _ in range(args.iterations):
        start = time.perf_counter()
        runner(image)
        wall.append((time.perf_counter() - start) * 1000)
    kernel = runner.kernel_times(args.warmup, args.iterations)
    record = dict(status='evaluated', backend='trt_int8_io', model=str(args.model),
                  model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  int8_input=bool(runner.int8_io[0]), int8_output=bool(runner.int8_io[1]),
                  confusion=confusion.tolist(), mIoU=float(np.nanmean(iou)),
                  water_IoU=float(iou[1]), rows=rows,
                  host_quantization='input quantized on host with scale %.8g (the deleted '
                                    'QuantizeLinear); output argmax taken on integer logits'
                                    % meta['host_input_scale'],
                  kernel_ms_mean=float(np.mean(kernel)), kernel_ms_p50=float(np.median(kernel)),
                  kernel_ms_p95=float(np.quantile(kernel, .95)),
                  inference_wall_ms_mean=float(np.mean(wall)),
                  inference_wall_ms_p50=float(np.median(wall)),
                  inference_wall_ms_p95=float(np.quantile(wall, .95)),
                  engine_bytes=Path(args.model).stat().st_size,
                  engine_device_memory_bytes=int(runner.engine.device_memory_size),
                  warmup=args.warmup, iterations=args.iterations)
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'rows'}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    c = sub.add_parser('convert')
    c.add_argument('--model', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--receipt', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', type=Path, required=True)
    r.add_argument('--assets', type=Path, required=True)
    r.add_argument('--io-receipt', type=Path, required=True)
    r.add_argument('--receipt', type=Path, required=True)
    r.add_argument('--warmup', type=int, default=20)
    r.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if Path(args.receipt).exists():
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')
    return convert(args) if args.action == 'convert' else run(args)


if __name__ == '__main__':
    sys.exit(main())
