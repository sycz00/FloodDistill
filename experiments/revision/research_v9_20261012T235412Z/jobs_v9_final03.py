"""Finite GPU commands. No target-dependent admission or optimization."""
import argparse,json,os,time
from pathlib import Path
import common_v9 as c
from policy_final03 import activate
activate()

def smoke():
    import numpy as np
    torch,cj,ev=c.setup('cuda');c.install_label_guard();spec=c.read(c.ROOT/'training_spec.json')
    from utils.seeding import seed_basic
    from train_v9 import state_hash
    from data_v9 import dataset
    seed_basic(42);teacher=c.teacher_cpu(spec);r4=c.build('R4');initial=state_hash(r4.state_dict());anchor=c.read(c.V7/'receipts/R4_CE_s42.json');assert initial==anchor['initial_base_sha256'],(initial,anchor['initial_base_sha256'])
    loader=torch.utils.data.DataLoader(dataset(spec,'ce',42),batch_size=16,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    hashes=[];batch=None
    for i,x in enumerate(loader):
        hashes.append(state_hash({k:x[k] for k in ['image','mask']}));batch=x
        if i==1:break
    assert hashes==anchor['first_minibatch_hashes'][:2],(hashes,anchor['first_minibatch_hashes'])
    results={};r4=r4.cuda();b0=c.build('B0').cuda();teacher=teacher.cuda();val,_=cj.loaders(spec);xb=next(iter(val))['image'];trainx=batch['image'].cuda()
    def bench(name,model,x,training=False):
        torch.cuda.reset_peak_memory_stats();model.train(training);torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(3):
            if training:
                z=model(x);loss=z.square().mean();loss.backward();model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():z=model(x)
        torch.cuda.synchronize();seconds=(time.perf_counter()-t)/3
        results[name]=dict(seconds_per_batch=seconds,batch_size=len(x),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    bench('R4_train',r4,trainx,True);bench('B0_train',b0,trainx,True);bench('teacher_train_geometry',teacher,trainx)
    bench('teacher_native',teacher,xb.cuda());bench('B0_native',b0,xb.cuda());bench('R4_native',r4,xb.cuda())
    del teacher,b0,r4;torch.cuda.empty_cache()
    m=c.load_model(next(x for x in c.read(c.V7/'reuse_models.json') if x['graph']=='R4'),'cpu')
    with torch.no_grad():cpu=m(xb).numpy();gpu=m.cuda()(xb.cuda()).cpu().numpy()
    dif=[int(np.count_nonzero((a[1]>a[0])!=(b[1]>b[0]))) for a,b in zip(cpu,gpu)]
    assert max(dif)<=2,('FP CPU/GPU parity outside documented tolerance',dif)
    import edge_v9 as edge
    refs=c.read(c.V7/'reference_cache/manifest.json') if (c.V7/'reference_cache/manifest.json').exists() else None
    # The source real-batch MNDWI formula must agree with frozen GPU formula.
    rb,av=ev.rule_arrays(xb.cuda(),spec)
    for i,x in enumerate(xb):
        mnd,available,_,_=edge.indices(edge.reflectance(x.numpy()));assert np.array_equal((mnd>0)&available,rb[i]);assert np.array_equal(available,av[i])
    qchecks=[]
    for seed in c.SEEDS:
        p=c.ROOT/'ptq_states'/f'q_s{seed}.pt'
        if not p.exists():continue
        entry=dict(id=f'PTQ_s{seed}',graph='R4',quantized=True,checkpoint=str(p),sha256=c.sha(p));m=c.load_model(entry,'cpu')
        with torch.no_grad():a=m(xb).numpy();b=m.cuda()(xb.cuda()).cpu().numpy()
        qdiff=[int(np.count_nonzero((x[1]>x[0])!=(y[1]>y[0]))) for x,y in zip(a,b)]
        assert max(qdiff)<=2,('quantized CPU/GPU parity failed',seed,qdiff)
        qchecks.append(dict(seed=seed,different_pixels_per_image=qdiff));del m;torch.cuda.empty_cache()
    c.save(c.ROOT/'smoke.json',dict(outcome='completed',initial_R4_sha256=initial,first_minibatch_hashes=hashes,FP_parity_different_pixels=dif,PTQ_parity=qchecks,throughput=results,backend=ev.backend_record(),sealed_GT_unopened=True))

def source_teacher():
    torch,cj,ev=c.setup('cuda');c.install_label_guard();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;spec=c.read(c.ROOT/'training_spec.json');model=c.teacher_cpu(spec).cuda();out=c.ROOT/'source_teacher';out.mkdir(exist_ok=True);rows=[]
    paths=sorted((c.ROOT/'source_images').glob('*.pt'))
    with torch.no_grad():
        for i in range(0,len(paths),16):
            ps=paths[i:i+16];samples=[torch.load(p,weights_only=True) for p in ps];x=torch.stack([a['image'] for a in samples]).cuda();z=model(x);cj.finite_output(model,(x,),z);mask=z.argmax(1).cpu()
            for p,a,y in zip(ps,samples,mask):
                y[a['mask']==-1]=-1;assert torch.equal(y!=-1,a['mask']!=-1)
                torch.save(dict(mask=y.to(torch.int8),teacher_sha256=spec['teacher_sha256'],input_sha256=c.sha(p)),out/p.name);rows.append(dict(file=p.name,sha256=c.sha(out/p.name)))
    c.save(c.ROOT/'source_teacher_manifest.json',dict(outcome='completed',files=rows,teacher_sha256=spec['teacher_sha256'],images=252,teacher_inputs='Unaugmented resized224 source training images; common reference-valid support; downstream common augmentation',sealed_GT_unopened=True))

def ptq():
    import shutil
    torch,cj,ev=c.setup();c.install_label_guard();config=c.read(c.V8/'config.json');spec=c.read(c.ROOT/'training_spec.json');out=c.ROOT/'ptq_states';out.mkdir(exist_ok=True);records=[];parents=[x for x in c.read(c.V7/'reuse_models.json') if x['graph']=='R4']
    from models.compression_quant import prepare_compression_qat,precision_map
    ds=cj.pseudo(dict(spec,seed=1729));calib=[{'image':ds[i]['image'][None].float()} for i in config['calibration_indices']]
    for parent in parents:
        seed=parent['seed'];old=c.V8/'support_cache'/f'q_s{seed}.pt';dest=out/f'q_s{seed}.pt'
        if old.exists():
            a=torch.load(old,map_location='cpu',weights_only=True);assert a['parent_sha256']==parent['sha256'];shutil.copy2(old,dest);pmap=a['precision'];origin='reused V8 exact bytes'
        else:
            model=c.load_model(parent,'cpu');model=prepare_compression_qat(model,calib,weight_bits=4,activation_bits=8,calibration_seed=1729);stem=model.stem[0];stem._w_scale.mul_(7/127);stem._weight_bits.fill_(8);stem._weight_bits_value=8;model.eval();pmap=precision_map(model)
            torch.save(dict(model=model.state_dict(),precision=pmap,parent_sha256=parent['sha256']),dest);origin='same frozen16 training calibration IDs, existing adopted PTQ'
        records.append(dict(seed=seed,sha256=c.sha(dest),parent=parent,origin=origin,precision=pmap,calibration_indices=config['calibration_indices'],native_kernel=False))
    c.save(c.ROOT/'ptq_manifest.json',dict(outcome='completed',records=records,calibration_receipt_sha256=c.sha(c.V8/'config.json'),precision='W4A8 body/W8A8 stem/FP32 head; adopted fake quantization; no LSQ or native speed claim'))

def infer(which):
    import numpy as np
    torch,cj,ev=c.setup('cuda');c.install_label_guard();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;spec=c.read(c.ROOT/'training_spec.json');models=c.read(c.ROOT/'final_models.json')['models']
    wm=c.read(c.ROOT/('normal_windows.json' if which=='normal' else 'sealed_windows.json'));windows=wm['windows'] if which=='normal' else wm['rows']
    selected=[dict(id='teacher',graph='teacher',checkpoint=spec['teacher'],sha256=spec['teacher_sha256'])]+[x for x in models if x['graph']=='B0' or x.get('quantized')]
    out=c.DATA/'gpu_predictions'/(which+'_tf32_off_gpu_ptq');out.mkdir(parents=True,exist_ok=True);records=[];begin=time.monotonic()
    for entry in selected:
        model=c.teacher_cpu(spec).cuda() if entry['graph']=='teacher' else c.load_model(entry,'cuda');destination=out/entry['id'];destination.mkdir(exist_ok=True);n=0;tic=time.monotonic()
        with torch.no_grad():
            for start in range(0,len(windows),4 if entry['graph']=='teacher' else 8):
                group=windows[start:start+(4 if entry['graph']=='teacher' else 8)];images=[]
                for row in group:
                    with np.load(row['path']) as a:images.append(torch.from_numpy(a['image'].copy()))
                x=torch.stack(images).cuda();z=model(x);cj.finite_output(model,(x,),z);margin=(z[:,1]-z[:,0]).cpu().numpy()
                for row,m in zip(group,margin):np.save(destination/(Path(row['path']).stem+'.npy'),m.astype(np.float32));n+=1
                if n%400<8:print(which,entry['id'],n,'/',len(windows),flush=True)
        torch.cuda.synchronize();records.append(dict(model=entry['id'],windows=n,seconds=time.monotonic()-tic,checkpoint_sha256=entry['sha256'],directory=str(destination)));del model;torch.cuda.empty_cache()
    c.save(c.ROOT/f'gpu_inference_{which}.json',dict(outcome='completed',seconds=time.monotonic()-begin,records=records,backend=ev.backend_record('deployment_tf32_off'),full_set=True,stored_margin_dtype='float32'))

def timing(session):
    import numpy as np
    torch,cj,ev=c.setup('cuda');c.install_label_guard();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    spec=c.read(c.ROOT/'training_spec.json');parent=next(x for x in c.read(c.V7/'reuse_models.json') if x['graph']=='R4' and x['seed']==42);val,_=cj.loaders(spec);sample=next(iter(val));records=[]
    for dtype in ['FP32','FP16']:
        model=c.load_model(parent,'cuda');model=model.half() if dtype=='FP16' else model.float();x=sample['image'].cuda().to(next(model.parameters()).dtype)
        with torch.no_grad():
            for _ in range(5):model(x)
            times=[]
            for _ in range(30):
                torch.cuda.synchronize();tic=time.perf_counter();model(x);torch.cuda.synchronize();times.append((time.perf_counter()-tic)*1000/len(x))
            cells=np.zeros(4,np.int64)
            for batch in val:
                images=batch['image'].cuda().to(next(model.parameters()).dtype);z=model(images);pred=(z[:,1]>z[:,0]).cpu().numpy()
                import edge_v9 as edge
                for y,p in zip(batch['mask'].numpy(),pred):cells+=edge.counts(p,y)
        records.append(dict(dtype=dtype,input_dtype=str(x.dtype),parameter_dtype=str(next(model.parameters()).dtype),latency_ms_per_image_median=float(np.median(times)),latency_samples=times,accuracy_boundary=ev.backend_record('deployment_tf32_off'),timing_boundary=ev.backend_record('deployment_tf32_off'),source_validation_counts=cells.tolist(),source_validation_images=89,transfer_and_preprocessing_included=False));del model;torch.cuda.empty_cache()
    c.save(c.ROOT/f'timing_session{session}.json',dict(outcome='completed',session=session,fresh_pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],records=records,scope='same-process paireddtype measurement within fresh session; workstation only; no comparison to V7 route; no native low-bit claims'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['smoke','source_teacher','ptq','normal','sealed','timing']);p.add_argument('--session',type=int);a=p.parse_args()
    if a.job=='smoke':smoke()
    elif a.job=='source_teacher':source_teacher()
    elif a.job=='ptq':ptq()
    elif a.job=='timing':timing(a.session)
    else:infer(a.job)
