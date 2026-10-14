"""Optional idle-GPU sealed UNet inference; original shared budget and UUID locks."""
import fcntl,os,time,sys,signal
from pathlib import Path
import numpy as np
import common10 as c

def main():
    # This subprocess owns its own session; only its PID may be terminated by its guardian.
    
    for path in [c.CAPSULE,c.RUNGS,c.V9]:sys.path.insert(0,str(path))
    import common_v9 as old
    import scheduler_v9 as queue
    items=c.read(c.ROOT/'readout_models.json')['models'];entries=[x for x in items if x['graph']=='UNet'];assert len(entries)==4
    rows=c.read(c.V9/'sealed_windows.json')['rows'];ceiling=1800;start=time.monotonic();history={uuid:[] for uuid in old.GPUS};chosen=None;lock=None
    spec=dict(peak_mib=12000,teacher=True)
    for observation in range(3):
        telemetry=queue.telemetry()
        for uuid in old.GPUS:
            if uuid in telemetry:telemetry[uuid]['at']=time.time();history[uuid].append(telemetry[uuid])
        if observation<2:time.sleep(15)
    for uuid in old.GPUS:
        row=queue.telemetry().get(uuid,{})
        ok,_,_=queue.availability(row,spec,history[uuid],time.time())
        if not ok:continue
        fd=os.open('/data1/schmalstieg/flooddistill/revision_gpu_'+uuid+'.lock',os.O_CREAT|os.O_RDWR,0o664)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:os.close(fd);continue
        with queue.Ledger() as ledger:
            fresh=queue.telemetry().get(uuid,{})
            if not queue.availability(fresh,spec,history[uuid],time.time())[0]:os.close(fd);continue
            if ledger.spend()+ledger.reserved()+ceiling>108000:os.close(fd);continue
            key=str(c.ROOT)+':UNet_GPU_final';ledger.data['reservations'][key]=dict(run=str(c.ROOT),block='UNet_GPU_final',remaining_seconds=ceiling,created=queue.c.q.now())
        chosen=uuid;lock=fd;break
    receipt_path=c.ROOT/'gpu_inference_receipt.json'
    if chosen is None:
        c.save(receipt_path,dict(outcome='unavailable',reason='No idle authorized GPU with safe headroom/shared budget; final evaluator uses CPU UNet',seconds=0));return
    os.environ['CUDA_VISIBLE_DEVICES']=chosen
    begin=time.monotonic();outcome='technical_failed';reason=None;completed=[]
    try:
        # CUDA imports/initialization occur only after UUID qualification and reservation.
        def stop_owned(*_):raise RuntimeError('Owned GPU stop requested')
        signal.signal(signal.SIGTERM,stop_owned)
        torch,_,old,_=c.setup()
        assert torch.cuda.is_available()
        valrows=[x for x in c.read(c.V9/'normal_windows.json')['windows'] if x['dataset']=='s11val'][:4]
        for entry in entries:
            model=c.load(entry);sample=[]
            for row in valrows:
                with np.load(row['path'],allow_pickle=False) as a:sample.append(torch.from_numpy(a['image'].copy()))
            with torch.no_grad():cpu=[model(x[None])[0] for x in sample]
            model=model.cuda();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            with torch.no_grad():gpu=[model(x[None].cuda())[0].cpu() for x in sample]
            differences=[int(((a[1]>a[0])!=(b[1]>b[0])).sum()) for a,b in zip(cpu,gpu)];assert max(differences)<=2,('Source CPU/GPU parity',entry['id'],differences)
            output=c.DATA/'unet_gpu_predictions'/entry['id'];output.mkdir(parents=True,exist_ok=True);lastcheck=0
            with torch.no_grad():
                for i,row in enumerate(rows):
                    if time.monotonic()-begin>=ceiling-15:raise TimeoutError('GPU command ceiling; remaining model readout falls back to CPU')
                    if time.monotonic()-lastcheck>=30:
                        telemetry=queue.telemetry().get(chosen,{});foreign=[x for x in telemetry.get('processes',[]) if x['pid']!=os.getpid()]
                        if not telemetry.get('known') or foreign:raise RuntimeError('Foreign contention; only owned inference is stopped')
                        lastcheck=time.monotonic()
                    with np.load(row['path'],allow_pickle=False) as a:x=torch.from_numpy(a['image'].copy())[None].cuda()
                    z=model(x);assert torch.isfinite(z).all();margin=(z[0,1]-z[0,0]).cpu().numpy().astype(np.float32);path=output/(Path(row['path']).stem+'.npy');temporary=path.with_name(path.name+'.tmp.npy');np.save(temporary,margin);os.replace(temporary,path)
                    if i%1000==0:print('UNet GPU',entry['id'],i,'/',len(rows),flush=True)
            torch.cuda.synchronize();c.save(c.ROOT/'gpu_model_receipts'/(entry['id']+'.json'),dict(outcome='completed',checkpoint_sha256=entry['sha256'],windows=len(rows),gpu_uuid=chosen,source_CPU_GPU_different_pixels=differences,input_dtype='float32',TF32=False));completed.append(entry['id']);del model;torch.cuda.empty_cache()
        outcome='completed'
    except Exception as exc:reason=repr(exc)
    finally:
        elapsed=time.monotonic()-begin
        c.save(receipt_path,dict(outcome=outcome,reason=reason,seconds=elapsed,gpu_uuid=chosen,completed_models=completed,CPU_fallback_for_incomplete_models=True,labels_not_read=True,performance_claim=False))
        with queue.Ledger() as ledger:
            ledger.data['charges'][str(receipt_path)]=dict(seconds=elapsed,receipt=str(receipt_path),sha256=c.sha(receipt_path));ledger.data['reservations'].pop(str(c.ROOT)+':UNet_GPU_final',None)
        os.close(lock)
if __name__=='__main__':main()
