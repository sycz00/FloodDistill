#!/usr/bin/env python3
"""Additive device measurement adapter; Python3.8/TensorRT8.4 only."""
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import integer_tools

def encode(image,scale,zero):
    # Match the author's int8_io_tools.Int8IoRunner, including its narrow range.
    scale=np.float32(scale)
    if scale.ndim or not np.isfinite(scale) or scale<=0:raise ValueError('Input quantization scale must be a finite positive scalar')
    if zero!=0:raise ValueError('The author integer-I/O path requires a zero input zero point')
    image=np.asarray(image,dtype=np.float32)
    return np.clip(np.rint(image/scale),-127,127).astype(np.int8)

class IntegerRunner(integer_tools.TrtRunner):
    def __init__(self,path,io):
        super().__init__(path)
        self.scale=np.float32(io['host_input_scale']);self.zero=io['host_input_zero_point']
        encode(np.zeros(1,np.float32),self.scale,self.zero)
        output_scale=np.asarray(io['output_scale'])
        if output_scale.ndim or not np.isfinite(output_scale) or output_scale<=0 or io['output_zero_point']!=0:
            raise RuntimeError('Integer output requires one finite positive shared scale and zero zero point')
        if io['input_binding']!='int8' or io['output_binding']!='int8':raise RuntimeError('IO receipt is not the author INT8 interface')
        if self.host[self.input_id].dtype!=np.int8 or self.host[self.output_id].dtype!=np.int8:
            raise RuntimeError('Integer-I/O contract requires both INT8 bindings')
    def __call__(self,image):
        if image.shape!=self.host[self.input_id].shape:raise ValueError('Input shape mismatch')
        np.copyto(self.host[self.input_id],encode(image,self.scale,self.zero))
        self.cuda.memcpy_htod_async(self.device[self.input_id],self.host[self.input_id],self.stream)
        if not self.context.execute_async_v2(self.bindings,self.stream.handle):raise RuntimeError('TensorRT execute failure')
        self.cuda.memcpy_dtoh_async(self.host[self.output_id],self.device[self.output_id],self.stream)
        self.stream.synchronize();return self.host[self.output_id].copy()

def bench(args):
    if args.receipt.exists():raise RuntimeError('Preserve existing receipt; choose a new attempt directory')
    if args.warmup<0 or args.iterations<=0 or args.expected_images<=0:raise ValueError('Require nonnegative warmup and positive iterations/image count')
    runner=IntegerRunner(args.engine,json.loads(args.io_receipt.read_text())) if args.io_receipt else integer_tools.TrtRunner(args.engine)
    rows=[];conf=np.zeros((2,2),np.int64)
    assets=sorted(args.assets.glob('*.npz'))
    if len(assets)!=args.expected_images:raise RuntimeError('Source inventory count mismatch')
    for p in assets:
        with np.load(p,allow_pickle=False) as z:
            image=z['image'][None].astype(np.float32);mask=z['mask'];event=str(z['event'])
        if image.shape!=(1,6,512,512) or mask.shape!=(512,512) or not np.isfinite(image).all():raise RuntimeError('Invalid prepared source geometry or image values')
        logits=runner(image)
        if logits.shape!=(1,2,512,512) or not np.isfinite(logits).all():raise RuntimeError('Invalid logits')
        valid=(mask>=0)&(mask<2);pred=logits[0].argmax(0)
        c=np.bincount((mask[valid]*2+pred[valid]).astype(np.int64),minlength=4).reshape(2,2)
        conf+=c;rows.append(dict(file=p.name,event=event,source_sha256=integer_tools.sha(p),confusion=c.tolist()))
    for _ in range(args.warmup):runner(image)
    wall=[]
    for _ in range(args.iterations):
        start=time.perf_counter();runner(image);wall.append(1000*(time.perf_counter()-start))
    kernel=runner.kernel_benchmark(image,args.warmup,args.iterations)
    union=conf.sum(0)+conf.sum(1)-conf.diagonal()
    iou=np.divide(conf.diagonal(),union,out=np.full(2,np.nan),where=union>0)
    result=dict(status='evaluated',backend='TensorRT',model_sha256=integer_tools.sha(args.engine),
        batch=1,shape=[1,6,512,512],rows=rows,confusion=conf.tolist(),mIoU=float(np.nanmean(iou)),water_IoU=float(iou[1]) if np.isfinite(iou[1]) else None,
        warmup=args.warmup,iterations=args.iterations,engine_bytes=args.engine.stat().st_size,
        engine_device_memory_bytes=int(runner.engine.device_memory_size),integer_io=bool(args.io_receipt),
        wall_boundary='includes H2D, inference, D2H and explicit host input quantization for integer I/O; excludes normalization, disk I/O and argmax',
        timing_input=dict(file=assets[-1].name,sha256=integer_tools.sha(assets[-1]),sampling='Repeated execution of the final lexicographically sorted prepared image; percentiles are across iterations, not images'),
        timing_samples_ms=dict(kernel=kernel,wall=wall),
        host_input_encoding='float32 scale; rint round-to-even; clip [-127,127]; int8; zero point 0' if args.io_receipt else 'prepared float32 input',
        io_receipt_sha256=integer_tools.sha(args.io_receipt) if args.io_receipt else None,
        software=dict(python=platform.python_version(),numpy=np.__version__,tensorrt=__import__('tensorrt').__version__))
    for name,v in [('kernel',kernel),('wall',wall)]:
        for p in [50,95,99]:result[name+'_ms_p'+str(p)]=float(np.percentile(v,p))
    integer_tools.save(args.receipt,result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--engine',type=Path)
    p.add_argument('--assets',type=Path);p.add_argument('--expected-images',type=int);p.add_argument('--receipt',type=Path)
    p.add_argument('--io-receipt',type=Path);p.add_argument('--warmup',type=int,default=20);p.add_argument('--iterations',type=int,default=100);a=p.parse_args()
    if a.self_test:
        assert encode(np.array([-200.,-1.5,-.5,.5,1.5,200.]),1,0).tolist()==[-127,-2,0,0,2,127]
        print('Integer boundary rounding/clipping fixture passed');return
    for name in ('engine','assets','expected_images','receipt'):
        if getattr(a,name) is None:p.error('--'+name.replace('_','-')+' is required unless --self-test is used')
    bench(a)

if __name__=='__main__':main()
