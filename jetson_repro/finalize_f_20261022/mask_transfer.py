#!/usr/bin/env python3
"""Jetson follow-up: reuse buffers, send INT8 input, return a CUDA uint8 mask.

Keeps the engine unchanged. Reports quantization, H2D, engine, mask and D2H
boundaries separately. Native validation must compare every mask with the same
engine's original logits/argmax path on the complete supplied source inventory.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np

def encode_into(image,scale,scratch,dest):
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive finite shared input scale required')
    if image.dtype!=np.float32 or scratch.dtype!=np.float32 or dest.dtype!=np.int8:raise TypeError('float32 -> int8 only')
    np.divide(image,np.float32(scale),out=scratch)
    np.rint(scratch,out=scratch);np.clip(scratch,-127,127,out=scratch)
    np.copyto(dest,scratch,casting='unsafe')

def host_mask(logits):return (logits[0,1]>logits[0,0]).astype(np.uint8)

class MaskRunner:
    def __init__(self,engine,io):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import integer_tools
        from pycuda.compiler import SourceModule
        self.base=integer_tools.TrtRunner(engine);b=self.base
        import tensorrt as trt
        if not trt.__version__.startswith('8.4.'):raise RuntimeError('Validate this adapter separately for other TensorRT versions')
        if any(b.engine.get_binding_format(i)!=trt.TensorFormat.LINEAR for i in range(b.engine.num_bindings)):
            raise ValueError('Linear unpadded bindings required for this mask kernel')
        it,ot=b.host[b.input_id],b.host[b.output_id]
        if it.shape!=(1,6,512,512) or ot.shape!=(1,2,512,512):raise ValueError('Static six-band 512 binary engine required')
        if it.dtype==np.int8:
            if io is None or io['host_input_zero_point']!=0:raise ValueError('INT8 input needs the matching symmetric IO receipt')
            self.scale=io['host_input_scale'];self.scratch=np.empty(it.shape,np.float32)
        elif it.dtype!=np.float32:raise TypeError('Only FLOAT or INT8 input supported')
        if ot.dtype==np.int8:
            if io is None or io['output_zero_point']!=0 or np.ndim(io['output_scale']) or io['output_scale']<=0:raise ValueError('Shared positive output scale required')
            ctype='signed char'
        elif ot.dtype==np.float32:ctype='float'
        else:raise TypeError('Only FLOAT or INT8 output supported')
        source='''extern "C" __global__ void binary_mask(const TYPE *z, unsigned char *m, int n) {
            int i=blockIdx.x*blockDim.x+threadIdx.x;
            if(i<n) m[i]=(unsigned char)(z[n+i]>z[i]);
        }'''.replace('TYPE',ctype)
        self.module=SourceModule(source,no_extern_c=True);self.kernel=self.module.get_function('binary_mask')
        self.host=b.cuda.pagelocked_empty((512,512),np.uint8);self.device=b.cuda.mem_alloc(self.host.nbytes)
        self.marks=[b.cuda.Event() for _ in range(5)]
    def __call__(self,image):
        b=self.base;start=time.perf_counter()
        if image.shape!=b.host[b.input_id].shape or image.dtype!=np.float32 or not np.isfinite(image).all():raise ValueError('finite float32 input required')
        if b.host[b.input_id].dtype==np.int8:encode_into(image,self.scale,self.scratch,b.host[b.input_id])
        else:np.copyto(b.host[b.input_id],image)
        encoded=time.perf_counter()
        marks=self.marks;marks[0].record(b.stream)
        b.cuda.memcpy_htod_async(b.device[b.input_id],b.host[b.input_id],b.stream);marks[1].record(b.stream)
        if not b.context.execute_async_v2(b.bindings,b.stream.handle):raise RuntimeError('Engine enqueue failed')
        marks[2].record(b.stream)
        self.kernel(b.device[b.output_id],self.device,np.int32(512*512),block=(256,1,1),grid=(1024,1,1),stream=b.stream);marks[3].record(b.stream)
        b.cuda.memcpy_dtoh_async(self.host,self.device,b.stream);marks[4].record(b.stream);b.stream.synchronize()
        mask=self.host.copy();end=time.perf_counter()
        return mask,dict(host_prepare_ms=1000*(encoded-start),h2d_ms=marks[0].time_till(marks[1]),engine_ms=marks[1].time_till(marks[2]),mask_ms=marks[2].time_till(marks[3]),d2h_ms=marks[3].time_till(marks[4]),wall_ms=1000*(end-start))
    def reference(self,image):
        b=self.base
        if b.host[b.input_id].dtype==np.int8:encode_into(image,self.scale,self.scratch,b.host[b.input_id])
        else:np.copyto(b.host[b.input_id],image)
        b.cuda.memcpy_htod_async(b.device[b.input_id],b.host[b.input_id],b.stream)
        if not b.context.execute_async_v2(b.bindings,b.stream.handle):raise RuntimeError('Reference enqueue failed')
        b.cuda.memcpy_dtoh_async(b.host[b.output_id],b.device[b.output_id],b.stream);b.stream.synchronize()
        return host_mask(b.host[b.output_id])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--engine',type=Path);p.add_argument('--build-receipt',type=Path);p.add_argument('--io',type=Path);p.add_argument('--assets',type=Path);p.add_argument('--expected-images',type=int);p.add_argument('--out',type=Path);p.add_argument('--iterations',type=int,default=100);a=p.parse_args()
    if a.self_test:
        x=np.array([-200,-1.5,-.5,.5,1.5,200],np.float32);out=np.empty(x.shape,np.int8);encode_into(x,1,np.empty_like(x),out);assert out.tolist()==[-127,-2,0,0,2,127]
        rng=np.random.default_rng(42);x=rng.normal(size=(1,6,32,32)).astype(np.float32);out=np.empty(x.shape,np.int8);encode_into(x,.0321,np.empty_like(x),out);assert np.array_equal(out,np.clip(np.rint(x/np.float32(.0321)),-127,127).astype(np.int8))
        for dtype in (np.float32,np.int8):
            z=np.array([[[[1,2,3]],[[1,3,2]]]],dtype=dtype);assert np.array_equal(host_mask(z),z.argmax(1)[0])
        print('CPU encoding and tie fixtures PASS; CUDA path NOT TESTED');return
    for k in ('engine','build_receipt','assets','expected_images','out'):
        if getattr(a,k) is None:p.error('--'+k.replace('_','-')+' required')
    if a.out.exists():raise FileExistsError('Use a fresh receipt path')
    files=sorted(a.assets.glob('*.npz'))
    if len(files)!=a.expected_images or a.expected_images<=0 or a.iterations<=0:raise ValueError('Inventory/iteration count mismatch')
    build=json.loads(a.build_receipt.read_text());io=json.loads(a.io.read_text()) if a.io else None
    if build.get('status')!='built' or build['engine_sha256']!=sha(a.engine):raise ValueError('Engine/build receipt identity mismatch')
    if io and build['model_sha256']!=io['variant_sha256']:raise ValueError('IO scales belong to another model')
    runner=MaskRunner(a.engine,io);checks=[]
    for path in files:
        with np.load(path,allow_pickle=False) as data:image=data['image'][None].astype(np.float32);mask=data['mask']
        reference=runner.reference(image);actual,timing=runner(image)
        valid=(mask>=0)&(mask<2);conf=np.bincount((2*mask[valid]+actual[valid]).astype(np.int64),minlength=4).reshape(2,2)
        changes=int((reference!=actual).sum());checks.append(dict(file=path.name,sha256=sha(path),changed_pixels=changes,confusion=conf.tolist()))
        if changes:raise RuntimeError('Mask parity failed '+path.name)
    for _ in range(20):runner(image)
    times=[runner(image)[1] for _ in range(a.iterations)]
    summary={k:{f'p{p}':float(np.percentile([r[k] for r in times],p)) for p in (50,95,99)} for k in times[0]}
    result=dict(status='PASS_ALL_SOURCE_MASKS_EXACT',engine_sha256=sha(a.engine),build_receipt_sha256=sha(a.build_receipt),io_receipt_sha256=sha(a.io) if a.io else None,n_images=len(files),checks=checks,timings=summary,raw_timing_samples=times,timing_image=files[-1].name,wall_boundary='finite check, reusable-buffer host encoding, H2D, engine, CUDA argmax, uint8 D2H, synchronize, returned mask copy; excludes disk and reflectance normalization',instrumentation='Five reusable CUDA timing events; compare wall values with an equally instrumented baseline before claiming speedup',input_bytes=runner.base.host[runner.base.input_id].nbytes,output_bytes=runner.host.nbytes)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
