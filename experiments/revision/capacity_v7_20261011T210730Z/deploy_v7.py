"""Measure actual reloaded TorchScript FP32/FP16 artifacts on the current GPU.

The teacher is never constructed. Calibration IDs come only from TRAIN. INT8 is
reported unsupported when no installed, graph-supported native exporter exists;
simulation is never relabelled as an integer deployment engine.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import importlib.util
from pathlib import Path
import time
import os
import subprocess
import sys
from evaluate_v7 import atomic_json, backend_record, file_hash, load_model, now, read, scores, setup


def native_int8(entry, rungs_dir, train, calibration_ids, xcpu, source, artifact_dir, warmup, iterations, smoke):
    """Installed TensorRT10 PTQ, calibrated only on the declared TRAIN IDs.

    Native engine inspection must show INT8 tensor formats; a builder flag alone
    is insufficient evidence of integer execution. Remaining kernels are listed.
    """
    import json
    import numpy as np
    import torch
    import tensorrt as trt
    from quantization.onnx_qdq import export_float_onnx
    from evaluation import revision_native_border as rb
    row=dict(id=entry['id'],graph=entry['graph'],seed=42,precision='INT8',engine='tensorrt_native_ptq',
        checkpoint_sha256=entry['sha256'],status='unsupported',tensorrt_version=trt.__version__,
        input_shape='1x6x512x512',batch_size=1,warmup=warmup,timed_iterations=iterations,
        calibration_dataset='TRAIN',calibration_tiles=2 if smoke else len(calibration_ids),
        calibration_resampling='TRAIN224 to512 bilinear for fixed-shape engine calibration; no evaluation resampling',
        timing_boundary='synchronized engine forward; excludes input transfer and normalization',teacher_loaded=False,smoke_only=smoke)
    if not hasattr(trt,'IInt8EntropyCalibrator2') or not hasattr(trt.BuilderFlag,'INT8'):
        row['reason']='Installed TensorRT lacks legacy native INT8 PTQ calibrator/builder API';return row
    messages=[]
    class CaptureLogger(trt.ILogger):
        def __init__(self):super().__init__()
        def log(self,severity,message):
            if severity<=trt.ILogger.Severity.WARNING:
                messages.append(str(message));print('[TRT]',message,file=sys.stderr,flush=True)
    logger=CaptureLogger()
    model=load_model(entry,rungs_dir).float().eval()
    onnx_path=artifact_dir/(entry['id']+'_float.onnx')
    export_float_onnx(model,xcpu,str(onnx_path),static=True)
    with torch.inference_mode():
        expected=model(xcpu.cuda()).argmax(1).cpu().numpy()
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers())
    del model;gc.collect();torch.cuda.empty_cache()
    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__();self.ids=calibration_ids[:2] if smoke else calibration_ids;self.position=0
            self.input=torch.empty((1,6,512,512),device='cuda',dtype=torch.float32)
        def get_batch_size(self):return 1
        def get_batch(self,names):
            if self.position>=len(self.ids):return None
            item=train[self.ids[self.position]];self.position+=1
            x=item['image'].float()[None]
            x=torch.nn.functional.interpolate(x,size=(512,512),mode='bilinear',align_corners=False)
            self.input.copy_(x);torch.cuda.synchronize();return [self.input.data_ptr()]
        def read_calibration_cache(self):return None
        def write_calibration_cache(self,cache):
            (artifact_dir/(entry['id']+'_TRAIN_calibration.cache')).write_bytes(cache)
    try:builder_instance=trt.Builder(logger)
    except Exception as error:
        detail='; '.join(messages[-10:])
        raise RuntimeError('Native TensorRT builder failed: '+detail+'; '+str(error)) from error
    with builder_instance as builder:
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,logger)
        if not parser.parse(onnx_path.read_bytes()):
            row['reason']='ONNX parse unsupported: '+'; '.join(str(parser.get_error(i)) for i in range(parser.num_errors));return row
        config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
        config.set_flag(trt.BuilderFlag.INT8);config.set_flag(trt.BuilderFlag.FP16)
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        calibrator=Calibrator();config.int8_calibrator=calibrator
        engine_bytes=builder.build_serialized_network(network,config)
        row['calibration_tiles_used']=calibrator.position
        if engine_bytes is None:
            row['reason']='Installed TensorRT could not build this native PTQ graph; see per-job builder log';return row
    engine_path=artifact_dir/(entry['id']+'_INT8.trt');engine_path.write_bytes(engine_bytes)
    del calibrator,config,parser,network
    gc.collect();torch.cuda.empty_cache()
    runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:raise RuntimeError('Saved native engine cannot be deserialized')
    context=engine.create_execution_context();inspector=engine.create_engine_inspector();inspector.execution_context=context
    layer_json=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    inspector_path=artifact_dir/(entry['id']+'_INT8_layers.json');inspector_path.write_text(layer_json+'\n')
    parsed=json.loads(layer_json);layers=parsed.get('Layers',[]) if isinstance(parsed,dict) else parsed
    int8_layers=[layer for layer in layers if any('int8' in str(tensor.get('Format/Datatype','')).lower()
        for tensor in layer.get('Inputs',[])+layer.get('Outputs',[]))]
    row.update(artifact_path=str(engine_path),artifact_bytes=engine_path.stat().st_size(),artifact_sha256=file_hash(engine_path),
        engine_layer_inspection_path=str(inspector_path),native_int8_layers=len(int8_layers),total_engine_layers=len(layers),
        parameter_bytes=parameter_bytes,buffer_bytes=buffer_bytes,
        operator_fallback=[dict(name=layer.get('Name'),type=layer.get('LayerType'),inputs=layer.get('Inputs'),outputs=layer.get('Outputs'))
                           for layer in layers if layer not in int8_layers],
        precision_description='native INT8 with explicitly inspected FP16/FP32 fallback; source parameter bytes before compile')
    if not int8_layers:
        row['reason']='Engine inspection shows no INT8 tensor/kernel formats; cannot claim integer execution';return row
    names=[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names=[name for name in names if engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT]
    output_names=[name for name in names if engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT]
    if len(input_names)!=1 or len(output_names)!=1:raise RuntimeError('Expected one engine input and output')
    dtype_map={trt.float32:torch.float32,trt.float16:torch.float16,trt.int8:torch.int8,trt.int32:torch.int32}
    input_name,output_name=input_names[0],output_names[0]
    inputs=xcpu.cuda().to(dtype_map[engine.get_tensor_dtype(input_name)])
    outputs=torch.empty(tuple(engine.get_tensor_shape(output_name)),device='cuda',dtype=dtype_map[engine.get_tensor_dtype(output_name)])
    context.set_tensor_address(output_name,outputs.data_ptr())
    def execute(x):
        if not x.is_contiguous():x=x.contiguous()
        context.set_tensor_address(input_name,x.data_ptr())
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError('Native engine execution failed')
        return outputs
    with torch.inference_mode():
        actual=execute(inputs);torch.cuda.synchronize()
        if not torch.isfinite(actual).all():raise RuntimeError('Nonfinite native INT8 output')
        row['fp32_prediction_parity_disagreement_pixels']=int((actual.argmax(1).cpu().numpy()!=expected).sum())
        row['fp32_prediction_parity_valid_pixels']=512*512
        row['fp32_prediction_parity_boundary']='fixed hash-selected TRAIN timing tile; all pixels'
        for _ in range(warmup):execute(inputs)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated();timings=[]
        peak_device_used_bytes=0
        for _ in range(iterations):
            torch.cuda.synchronize();tick=time.perf_counter();execute(inputs);torch.cuda.synchronize()
            timings.append((time.perf_counter()-tick)*1000)
            free,total=torch.cuda.mem_get_info();peak_device_used_bytes=max(peak_device_used_bytes,total-free)
        free_bytes,total_bytes=torch.cuda.mem_get_info()
        row.update(latency_p50_ms=float(np.percentile(timings,50)),latency_p95_ms=float(np.percentile(timings,95)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            inference_baseline_allocated_bytes=baseline,peak_incremental_allocated_bytes=torch.cuda.max_memory_allocated()-baseline,
            device_used_bytes_at_inference=total_bytes-free_bytes,engine_device_memory_bytes=engine.device_memory_size,
            peak_device_used_bytes_at_synchronized_inference=peak_device_used_bytes,
            memory_measurement='PyTorch allocator bytes plus current total CUDA used bytes and engine-reported context bytes; native TensorRT allocation is outside PyTorch allocator')
        pooled=np.zeros((2,3),dtype=np.int64)
        for index in range(1 if smoke else len(source)):
            item=source[index];x=item['image'][None].cuda().to(inputs.dtype)
            logits=execute(x);torch.cuda.synchronize()
            if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite native engine source output')
            pooled+=rb.confusion(logits.argmax(1)[0].cpu().numpy(),item['mask'].numpy())
        row.update({key:value*100 if value is not None else None for key,value in scores(pooled.tolist()).items()})
        row.update(status='completed',source_accuracy_dataset='S11_source_validation',source_images=1 if smoke else len(source),
                   source_accuracy_unit='percent',source_confusion_counts=pooled.tolist(),quantized_execution_measured=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'spec', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('models', 'capsule', 'rungs-dir'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-seconds',type=float,default=840,help='Finish partial measurements before outer900s job timeout')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    spec = read(args.spec)
    capsule = args.capsule or Path(spec['capsule']).resolve()
    rungs_dir = args.rungs_dir or Path(spec['rungs_dir']).resolve()
    torch = setup(capsule, rungs_dir,backend_policy='deployment_tf32_off')
    import numpy as np
    import scripts.compression_jobs as cj
    from evaluation import revision_native_border as rb
    start = time.monotonic()
    reuse_path=args.run_dir/'provenance/reuse_index.json'
    if not reuse_path.exists():reuse_path=args.run_dir/'reuse_index.json'
    records = read(args.models) if args.models else read(reuse_path)['records']
    if isinstance(records, dict): records = records.get('models', records.get('records', []))
    entries = []
    for graph in ('R4', 'A3N', 'A3'):
        candidates = [record for record in records if record.get('graph') == graph and record.get('seed') == 42
                      and record.get('method', 'lkd').lower() == 'lkd']
        if not candidates: continue
        # Optimized anchor only; deterministic preference for largest declared LR.
        selected = max(candidates, key=lambda record: (float(record.get('lr', 0)), str(record.get('family', ''))))
        entries.append(dict(id=graph+'_s42', graph=graph, method='lkd', seed=42,
            checkpoint=selected.get('checkpoint', selected.get('best_path')),
            sha256=selected.get('sha256', selected.get('best_sha256'))))
    train = cj.pseudo(spec, augment=False)
    calibration_ids = sorted(range(len(train)), key=lambda index: file_hash_name(train.files[index].name))[:128]
    calibration = dict(dataset='TRAIN', rule='128 smallest SHA256 of cohort file names',
        ids=[train.files[index].name for index in calibration_ids], used_for_quantizer=False,
        reason='FP artifacts use no quantizer; native INT8 PTQ attempts use these TRAIN tiles only')
    artifact_dir = args.run_dir / ('smoke_deployment_artifacts' if args.smoke else 'deployment_artifacts')
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_dir/'train_calibration_ids.json', calibration)
    item = train[calibration_ids[0]]
    xcpu = item['image'].float()[None]
    # Timing at the registered 512 grid. TRAIN is 224; this declared benchmark
    # sample resampling does not alter any evaluation input or mask.
    xcpu = torch.nn.functional.interpolate(xcpu, size=(512,512), mode='bilinear', align_corners=False)
    val, _ = cj.loaders(spec)
    source = val.dataset
    warmup, iterations = (2, 3) if args.smoke else (20, 100)
    available = {}
    for package in ('tensorrt', 'onnx', 'onnxruntime', 'torchao'):
        try:
            available[package] = importlib.util.find_spec(package) is not None
        except (ValueError, ModuleNotFoundError): available[package] = False
    backend_versions={distribution.metadata['Name']:distribution.version for distribution in importlib.metadata.distributions()
        if distribution.metadata['Name'].lower().startswith('tensorrt')}
    try:
        driver_version=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL,timeout=5).strip().splitlines()[0]
    except Exception:driver_version='not_detected'
    rows = []
    for entry in entries:
        references = None
        for precision, dtype in [('FP32', torch.float32), ('FP16', torch.float16)]:
            row = dict(id=entry['id'], graph=entry['graph'], seed=42, precision=precision,
                engine='torchscript_cuda', status='not_run', checkpoint_sha256=entry['sha256'],
                input_shape='1x6x512x512', batch_size=1, warmup=warmup, timed_iterations=iterations,
                torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                backend_policy={**backend_record('deployment_tf32_off'),'grad_context':'torch.inference_mode',
                                'input_dtype':str(dtype),'checkpoint_parameter_dtype':str(dtype)},
                device=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                timing_boundary='synchronized model forward; excludes transfer and normalization; rule computation included when present',
                timing_input='fixed hash-selected TRAIN tile bilinearly resized224_to512; evaluation inputs unchanged',
                teacher_loaded=False, smoke_only=args.smoke,
                source_accuracy_dataset='S11_source_validation', empty_union_convention='undefined and omitted from mean')
            model = None; artifact = None; inputs = None;handles=[];failed_operators=[];mixed=set()
            try:
                model = load_model(entry, rungs_dir).to(dtype=dtype).eval()
                inputs = xcpu.cuda().to(dtype)
                def observe(module, _inputs, output):
                    if dtype == torch.float16 and torch.is_tensor(output) and output.dtype == torch.float32:
                        mixed.add(type(module).__name__)
                for name,module in model.named_modules():
                    handles.append(module.register_forward_hook(observe))
                    if isinstance(module,torch.nn.Conv2d):
                        def check_dtype(module,_inputs,module_name=name):
                            if _inputs and torch.is_tensor(_inputs[0]) and _inputs[0].dtype!=module.weight.dtype:
                                failed_operators.append(dict(module=module_name,type='Conv2d',input_dtype=str(_inputs[0].dtype),weight_dtype=str(module.weight.dtype)))
                        handles.append(module.register_forward_pre_hook(check_dtype))
                with torch.inference_mode():
                    expected = model(inputs)
                for handle in handles: handle.remove()
                if not torch.isfinite(expected).all(): raise RuntimeError('Nonfinite eager logits in requested precision')
                row['operator_fallback'] = ','.join(sorted(mixed)) if mixed else 'none_observed_on_trace_sample'
                row['parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
                row['buffer_bytes'] = sum(p.numel()*p.element_size() for p in model.buffers())
                with torch.inference_mode():
                    traced = torch.jit.trace(model, inputs, check_trace=True)
                    artifact_path = artifact_dir / (entry['id']+'_'+precision+'.torchscript.pt')
                    torch.jit.save(traced, str(artifact_path)); del traced
                    artifact = torch.jit.load(str(artifact_path), map_location='cuda').eval()
                    actual = artifact(inputs)
                if not torch.isfinite(actual).all(): raise RuntimeError('Nonfinite reloaded artifact logits')
                row.update(artifact_path=str(artifact_path), artifact_bytes=artifact_path.stat().st_size,
                    artifact_sha256=file_hash(artifact_path), eager_artifact_max_abs_logit_difference=float((actual-expected).abs().max()),
                    eager_artifact_prediction_disagreement_pixels=int((actual.argmax(1)!=expected.argmax(1)).sum()))
                del model, actual, expected; model = None
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    for _ in range(warmup): artifact(inputs)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    baseline = torch.cuda.memory_allocated(); timings = []
                    for _ in range(iterations):
                        torch.cuda.synchronize(); tick=time.perf_counter()
                        output = artifact(inputs); torch.cuda.synchronize()
                        timings.append((time.perf_counter()-tick)*1000)
                        if not torch.isfinite(output).all(): raise RuntimeError('Nonfinite timed output')
                    row.update(latency_p50_ms=float(np.percentile(timings,50)), latency_p95_ms=float(np.percentile(timings,95)),
                        latency_mean_ms=float(np.mean(timings)), inference_baseline_allocated_bytes=baseline,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_incremental_allocated_bytes=torch.cuda.max_memory_allocated()-baseline)
                    del output
                    pooled = np.zeros((2,3), dtype=np.int64); predictions = []; changed = total = 0
                    for index in range(1 if args.smoke else len(source)):
                        record = source[index]; image = record['image'][None].cuda().to(dtype)
                        logits = artifact(image)
                        if not torch.isfinite(logits).all(): raise RuntimeError(f'Nonfinite source logits at index{index}')
                        pred=logits.argmax(1)[0].cpu().numpy(); mask=record['mask'].cpu().numpy()
                        pooled += rb.confusion(pred,mask); predictions.append(pred.astype(np.uint8))
                        if references is not None:
                            valid=mask!=-1; changed+=int(((pred!=references[index])&valid).sum()); total+=int(valid.sum())
                    row.update({key: value*100 if value is not None else None for key,value in scores(pooled.tolist()).items()})
                    row.update(source_images=len(predictions), source_accuracy_unit='percent',
                        fp32_prediction_parity_disagreement_pixels=changed, fp32_prediction_parity_valid_pixels=total,
                        fp32_prediction_parity_disagreement_fraction=changed/total if total else None,
                        source_confusion_counts=pooled.tolist(), status='completed')
                    if precision=='FP32': references=predictions
            except Exception as error:
                row.update(status='unsupported' if precision=='FP16' else 'technical_failure',
                    reason=type(error).__name__+': '+str(error),unsupported_operators=failed_operators,
                    fp32_operator_outputs_observed=sorted(mixed))
                if precision=='FP16' and entry['graph'] in ('A3','A3N') and failed_operators:
                    row['diagnosis']='Pinned EfficientViT UpSampleLayer.forward casts half features to float before subsequent half convolution; LiteMLA linear attention also accumulates and returns float. Strict all-half artifact unsupported without a separately identified inference adapter.'
                    try:
                        import inspect
                        from efficientvit.models.nn.ops import UpSampleLayer
                        source_path=Path(inspect.getsourcefile(UpSampleLayer))
                        row['diagnosed_operator_source_path']=str(source_path);row['diagnosed_operator_source_sha256']=file_hash(source_path)
                    except Exception:pass
            finally:
                for handle in handles:handle.remove()
                del model, artifact, inputs
                gc.collect(); torch.cuda.empty_cache()
            rows.append(row)
            atomic_json(args.out,dict(status='running',outcome='in_progress',rows=rows,smoke_only=args.smoke,elapsed_seconds=time.monotonic()-start))
            print(entry['id'], precision, row['status'], flush=True)
        int8_row=dict(id=entry['id'],graph=entry['graph'],seed=42,precision='INT8',engine=None,status='unsupported',
            backend_policy={**backend_record('deployment_tf32_off'),'grad_context':'torch.inference_mode',
                            'precision_description':'native TensorRT PTQ with inspected FP16/FP32 fallback'},
            native_backend_probe=available,checkpoint_sha256=entry['sha256'],reason='TensorRT and ONNX required for existing native exporter',
            native_backend_versions=backend_versions,cuda_driver_version=driver_version,pytorch_cuda_runtime=torch.version.cuda,
            calibration_dataset='TRAIN',calibration_ids_path=str(artifact_dir/'train_calibration_ids.json'),
            quantized_execution_measured=False,smoke_only=args.smoke)
        if time.monotonic()-start>args.max_seconds-120:
            int8_row.update(status='deferred_budget',reason='Native engine build120s reservation does not fit remaining protected deployment time')
        elif available['tensorrt'] and available['onnx']:
            try:int8_row.update(native_int8(entry,rungs_dir,train,calibration_ids,xcpu,source,artifact_dir,warmup,iterations,args.smoke))
            except Exception as error:
                int8_row.update(status='unsupported',reason=type(error).__name__+': '+str(error),
                    cuda_initialization_error35=('error: 35' in str(error) or 'error 35' in str(error)))
        rows.append(int8_row)
        if int8_row.get('calibration_tiles_used',0):
            calibration['used_for_quantizer']=True
            atomic_json(artifact_dir/'train_calibration_ids.json',calibration)
        atomic_json(args.out,dict(status='running',outcome='in_progress',rows=rows,smoke_only=args.smoke,elapsed_seconds=time.monotonic()-start))
        gc.collect();torch.cuda.empty_cache()
        if time.monotonic()-start>args.max_seconds:break
    atomic_json(args.out, dict(status='complete', outcome='completed', completed_at=now(), rows=rows,
        elapsed_seconds=time.monotonic()-start, smoke_only=args.smoke, calibration=calibration,
        unsupported_does_not_block_training=True))


def file_hash_name(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__': main()
