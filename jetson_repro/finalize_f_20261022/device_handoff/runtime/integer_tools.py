#!/usr/bin/env python3
"""Python3.8/TRT8.4 build, precision audit and fixed-NPZ evaluation.

No environment updates; no FP32 substitution after a requested build failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(temp), str(path))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def inspect_layers(layers):
    """Conservative audit. Tensor formats are evidence, not kernel accumulator types.

    Unknown/missing inspector information fails certification. Floating reformats
    are disclosed including host IO. Shape-only integer constants are allowed.
    """
    rows = layers.get('Layers', []) if isinstance(layers, dict) else layers
    float_rows, boundary_rows, unknown, int8 = [], [], [], []
    for row in rows:
        tensors = row.get('Inputs', []) + row.get('Outputs', [])
        formats = [str(t.get('Format/Datatype', '')) for t in tensors]
        if any(re.search(r'FP32|FP16|Float|Half|BF16', f, re.I) for f in formats):
            float_rows.append(row)
            kind=str(row.get('LayerType','')).lower()
            names=[t.get('Name') for t in tensors]
            if kind=='reformat' and any(n in ('image','logits') for n in names):
                boundary_rows.append(row)
        elif not formats or any(not f for f in formats):
            unknown.append(row)
        if any(re.search(r'INT8|Int8', f) for f in formats):
            int8.append(row)
    return dict(layer_count=len(rows), int8_tensor_layers=len(int8), floating_layers=float_rows,
                boundary_reformats=boundary_rows,
                floating_compute_layers=[r for r in float_rows if r not in boundary_rows],
                unknown_layers=unknown, full_integer_tensor_path=bool(rows) and not float_rows and not unknown,
                full_integer_arithmetic='not_certified_by_tensor_formats_alone',
                remaining_validation='Inspect tactics/kernel implementation for floating arithmetic; INT32 accumulators/bias are permitted.')


def build(args):
    record = dict(status='unsupported', model_sha256=sha(args.model), mode=args.mode,
                  host=platform.platform(), python=sys.version, started=time.time(),
                  calibration='embedded QDQ; no random ranges and no calibrator', fp16_enabled=False)
    start = time.monotonic()
    Path(args.engine).parent.mkdir(parents=True,exist_ok=True)
    try:
        import tensorrt as trt
        record['tensorrt'] = trt.__version__
        # Deliberately reject later APIs instead of silently changing build semantics.
        if not trt.__version__.startswith('8.4.'):
            raise RuntimeError('This device builder requires the preserved TensorRT8.4.x; found ' + trt.__version__)
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(Path(args.model).read_bytes()):
            raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        # Public host buffers are dense NCHW; internal INT8 layouts remain free.
        for i in range(network.num_inputs):network.get_input(i).allowed_formats=1 << int(trt.TensorFormat.LINEAR)
        for i in range(network.num_outputs):network.get_output(i).allowed_formats=1 << int(trt.TensorFormat.LINEAR)
        config = builder.create_builder_config()
        config.max_workspace_size = int(args.workspace_mib) * 1048576
        # fp32/fp16 are float reference arms: they must NOT set the INT8 flag, otherwise
        # TensorRT may still place quantizable layers in INT8 and the comparison is void.
        if args.mode not in ('fp32', 'fp16'):
            config.set_flag(trt.BuilderFlag.INT8)
            record['int8_enabled'] = True
        else:
            record['int8_enabled'] = False
        config.clear_flag(trt.BuilderFlag.TF32)
        if args.mode in ('mixed', 'fp16'):
            config.set_flag(trt.BuilderFlag.FP16)
            record['fp16_enabled'] = True
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        forced = []
        if args.mode in ('strict', 'strict_except_resize', 'strict_compute_only'):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            structural = {'CONSTANT', 'SHAPE', 'SHUFFLE', 'QUANTIZE', 'DEQUANTIZE', 'IDENTITY'}
            # TensorRT 8.4 has no INT8-obedient Resize implementation. Forcing INT8 on a
            # Resize aborts the build in the optimizer (isPolymorphic assertion), so the
            # registered bilinear upsampling makes 'strict' unsatisfiable by construction.
            # 'strict_except_resize' is a SEPARATE diagnostic recipe that still demands
            # INT8 obedience for every other compute layer and leaves Resize to the
            # builder, where the audit then discloses it as a floating compute layer.
            exempt = set(structural)
            if args.mode == 'strict_except_resize':
                exempt.add('RESIZE')
            if args.mode == 'strict_compute_only':
                # Slice/Concat/Gather move data, they do not compute. Forcing an INT8
                # output type on them makes TensorRT clone the feeding Dequantize and
                # then fail to find an implementation for the clone, which says nothing
                # about arithmetic precision. Exempting them keeps the constraint on
                # every genuine compute layer (Convolution, Deconvolution, Activation,
                # ElementWise, MatrixMultiply, Resize, ...) and lets the audit report
                # what the data-movement layers actually carry.
                exempt |= {'SLICE', 'CONCATENATION', 'GATHER'}
            record['unforced_layer_types'] = sorted(exempt)
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                kind = str(layer.type).split('.')[-1]
                if kind in exempt:
                    continue
                layer.precision = trt.int8
                for j in range(layer.num_outputs):
                    output = layer.get_output(j)
                    if not output.is_shape_tensor and output.dtype not in (trt.int32, trt.bool):
                        layer.set_output_type(j, trt.int8)
                forced.append(dict(name=layer.name, type=kind))
        record['forced_layers'] = forced
        if args.fp32_islands:
            import precision_islands
            island_spec=json.loads(args.fp32_islands.read_text());record['fp32_islands_sha256']=sha(args.fp32_islands)
            record['fp32_constraints']=precision_islands.constrain(network,config,trt,island_spec,mode='prefer' if args.mode=='mixed' else 'obey')
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError('Builder returned no engine; see build log for unsupported precision/operator')
        record['bindings']=[dict(name=engine.get_binding_name(i),shape=list(engine.get_binding_shape(i)),input=engine.binding_is_input(i),dtype=str(engine.get_binding_dtype(i)),vectorized_dim=engine.get_binding_vectorized_dim(i),components_per_element=engine.get_binding_components_per_element(i),format=engine.get_binding_format_desc(i)) for i in range(engine.num_bindings)]
        record['linear_host_bindings']=all(b['vectorized_dim']==-1 and b['components_per_element']==1 for b in record['bindings'])
        record['integer_host_bindings']=all(engine.get_binding_dtype(i)==trt.int8 for i in range(engine.num_bindings))
        if not record['linear_host_bindings']:raise RuntimeError('Dense host runner requires LINEAR bindings')
        context = engine.create_execution_context()
        inspector = engine.create_engine_inspector()
        inspector.execution_context = context
        layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        save(str(args.engine) + '.layers.json', layers)
        record['precision_audit'] = inspect_layers(layers)
        record['engine_device_memory_bytes'] = int(engine.device_memory_size)
        # Preserve even a non-strict engine for diagnosing the failed certification.
        Path(args.engine).write_bytes(bytes(engine.serialize()))
        record['engine_bytes'] = Path(args.engine).stat().st_size
        record['engine_sha256'] = sha(args.engine)
        record['status'] = 'built'
        if args.fp32_islands:
            record['fp32_island_audit']=precision_islands.inspect(layers,island_spec)
            if not record['fp32_island_audit']['passed']:record['status']='fp32_island_certification_failed'
        if args.mode == 'strict' and not record['precision_audit']['full_integer_tensor_path']:
            record['status'] = 'strict_certification_failed'
        if args.mode == 'strict_compute_only':
            audit = record['precision_audit']
            record['certification'] = (
                'INT8 obedience demanded for every compute layer; data-movement layers '
                '(Slice/Concat/Gather) left to the builder. floating_compute_layers=%d, '
                'boundary_reformats=%d. Zero floating compute with only host-IO boundary '
                'reformats means all arithmetic is INT8; it is still not a kernel-level '
                'certificate.' % (len(audit['floating_compute_layers']),
                                  len(audit['boundary_reformats'])))
        if args.mode == 'strict_except_resize':
            record['certification'] = ('partial: INT8 obedience demanded for every compute layer '
                                       'except RESIZE; resize stays floating by construction, so '
                                       'this is not a fully integer engine')
        record['full_integer_execution'] = 'unverified_kernel_audit_required'
    except Exception as error:
        record['status'] = 'failed'
        record['reason'] = repr(error)
    record['elapsed_seconds'] = time.monotonic() - start
    save(args.receipt, record)
    print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'built' else 2


def _restore_numpy_aliases_for_trt84():
    """TensorRT 8.4's Python bindings call `np.bool` inside `nptype()`.

    NumPy removed that alias in 1.24, and the preserved environment pins NumPy
    1.24.4, so every `trt.nptype(...)` raises AttributeError and no engine can be
    bound. `np.bool` was only ever an alias for the builtin `bool` - NumPy's own
    removal note states restoring it "will not modify any behavior" - so this
    reinstates it instead of editing the JetPack system package or downgrading
    NumPy. Engine arithmetic and dtypes are untouched.
    """
    if not hasattr(np, 'bool'):
        np.bool = bool


class TrtRunner:
    def __init__(self, path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # Context kept alive while device buffers exist.
        _restore_numpy_aliases_for_trt84()
        self.cuda = cuda
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed; rebuild on this Jetson')
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.device = [], [], []
        self.input_id, self.output_id = None, None
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            if any(x < 0 for x in shape):
                raise RuntimeError('Static bindings required')
            if self.engine.get_binding_vectorized_dim(i)!=-1 or self.engine.get_binding_components_per_element(i)!=1:raise RuntimeError('Packed binding cannot use the dense host runner')
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            host = cuda.pagelocked_empty(shape, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.host.append(host)
            self.device.append(device)
            self.bindings.append(int(device))
            if self.engine.binding_is_input(i):
                self.input_id = i
            else:
                self.output_id = i
        if self.engine.num_bindings != 2 or self.input_id is None or self.output_id is None:
            raise RuntimeError('Expected one image input and one logits output')

    def __call__(self, image):
        if image.shape != self.host[self.input_id].shape:
            raise ValueError('Input geometry mismatch')
        if self.host[self.input_id].dtype != np.float32:
            raise RuntimeError('INT8 IO needs explicit scale encoding; float IO runner cannot accept it')
        np.copyto(self.host[self.input_id], image)
        self.cuda.memcpy_htod_async(self.device[self.input_id], self.host[self.input_id], self.stream)
        if not self.context.execute_async_v2(self.bindings, self.stream.handle):
            raise RuntimeError('TensorRT execution failed')
        self.cuda.memcpy_dtoh_async(self.host[self.output_id], self.device[self.output_id], self.stream)
        self.stream.synchronize()
        return self.host[self.output_id].copy()

    def kernel_benchmark(self, image, warmup, iterations):
        self(image)
        for _ in range(warmup):
            if not self.context.execute_async_v2(self.bindings, self.stream.handle):
                raise RuntimeError('Benchmark enqueue failed')
        self.stream.synchronize()
        times = []
        for _ in range(iterations):
            start, end = self.cuda.Event(), self.cuda.Event()
            start.record(self.stream)
            if not self.context.execute_async_v2(self.bindings, self.stream.handle):
                raise RuntimeError('Benchmark enqueue failed')
            end.record(self.stream)
            end.synchronize()
            times.append(float(start.time_till(end)))
        return times


def run(args):
    if args.backend == 'trt':
        runner = TrtRunner(args.model)
    else:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        session = ort.InferenceSession(str(args.model), options, providers=['CPUExecutionProvider'])
        runner = lambda x: session.run(None, {'image': x})[0]
    rows, confusion = [], np.zeros((2, 2), np.int64)
    assets = sorted(Path(args.assets).glob('*.npz'))
    if not assets:
        raise RuntimeError('No NPZ evaluation assets')
    for path in assets:
        with np.load(path, allow_pickle=False) as item:
            image, mask = item['image'][None].astype(np.float32), item['mask']
            logits = runner(image)
            if logits.shape != (1, 2) + mask.shape or not np.isfinite(logits).all():
                raise RuntimeError('Invalid logits from ' + path.name)
            pred = logits[0].argmax(0)
            valid = (mask >= 0) & (mask < 2)
            conf = np.bincount((mask[valid]*2 + pred[valid]).ravel(), minlength=4).reshape(2, 2)
            confusion += conf
            rows.append(dict(file=path.name, event=str(item['event']), confusion=conf.tolist()))
    union = confusion.sum(0) + confusion.sum(1) - confusion.diagonal()
    iou = np.divide(confusion.diagonal(), union, out=np.full(2, np.nan), where=union > 0)
    for _ in range(args.warmup):
        runner(image)
    wall = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        runner(image)
        wall.append((time.perf_counter() - start)*1000)
    record = dict(status='evaluated', backend=args.backend, model_sha256=sha(args.model),
        confusion=confusion.tolist(), mIoU=float(np.nanmean(iou)), rows=rows,
        batch=1, shape=list(image.shape), warmup=args.warmup, iterations=args.iterations,
        inference_wall_ms_mean=float(np.mean(wall)), inference_wall_ms_p50=float(np.median(wall)),
        inference_wall_ms_p95=float(np.quantile(wall, .95)),
        wall_boundary='includes H2D/D2H for TRT; excludes normalization/disk IO',
        input='already normalized six-band float32; host quantization boundaries disclosed',
        host=platform.platform(), engine_bytes=Path(args.model).stat().st_size)
    if args.backend == 'trt':
        times = runner.kernel_benchmark(image, args.warmup, args.iterations)
        record['kernel_ms_mean'] = float(np.mean(times))
        record['kernel_ms_p50'] = float(np.median(times))
        record['kernel_ms_p95'] = float(np.quantile(times, .95))
        record['engine_device_memory_bytes'] = int(runner.engine.device_memory_size)
    save(args.receipt, record)
    print(json.dumps({k:v for k,v in record.items() if k != 'rows'}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--mode', choices=['strict', 'strict_except_resize', 'strict_compute_only',
                                      'no_fp16', 'mixed', 'fp32', 'fp16'], default='strict')
    p.add_argument('--workspace-mib', type=int, default=1024)
    p.add_argument('--fp32-islands',type=Path)
    p = sub.add_parser('run')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--backend', choices=['ort', 'trt'], default='ort')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.receipt.exists() or (args.action=='build' and args.engine.exists()):
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')
    if args.action=='build':
        return build(args)
    if args.iterations<=0 or args.warmup<0:
        parser.error('iterations must be positive and warmup nonnegative')
    try:
        return run(args)
    except Exception as error:
        save(args.receipt,dict(status='failed',backend=args.backend,reason=repr(error),
                              model=str(args.model),no_backend_substitution=True))
        print(repr(error),file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
