"""Resume immutable prediction cache; never reopen sealed raw references."""
import json, os, time
from pathlib import Path
import numpy as np
import common_v9 as c
from policy_final03 import activate
from backend_record_recovery import install
activate()
install()

def main():
    torch, cj, ev = c.setup('cuda')
    c.install_label_guard()
    spec = c.read(c.ROOT / 'training_spec.json')
    windows = c.read(c.ROOT / 'sealed_windows.json')['rows']
    models = c.read(c.ROOT / 'final_models.json')['models']
    selected = [dict(id='teacher', graph='teacher', checkpoint=spec['teacher'], sha256=spec['teacher_sha256'])] + [x for x in models if x['graph']=='B0' or x.get('quantized')]
    destination_root = c.DATA / 'gpu_predictions/sealed_tf32_off_gpu_ptq'
    begin = time.monotonic()
    records = []
    for entry in selected:
        destination = destination_root / entry['id']
        destination.mkdir(parents=True, exist_ok=True)
        assert c.sha(entry['checkpoint']) == entry['sha256']
        cached = set()
        fingerprints = {}
        for row in windows:
            path = destination / (Path(row['path']).stem + '.npy')
            if not path.exists():
                continue
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (512,512) or not np.isfinite(array).all():
                raise RuntimeError('Invalid cached prediction: ' + str(path))
            cached.add(path.name)
            fingerprints[path.name] = c.sha(path)
        c.save(c.ROOT / 'resumed_prediction_hashes' / (entry['id']+'.json'), dict(checkpoint_sha256=entry['sha256'], predictions=fingerprints, prior_job='sealed_readout_retry01', timings_excluded=True))
        missing = [row for row in windows if Path(row['path']).stem+'.npy' not in cached]
        tic = time.monotonic()
        if missing:
            model = c.teacher_cpu(spec).cuda() if entry['graph']=='teacher' else c.load_model(entry,'cuda')
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            batch_size = 4 if entry['graph']=='teacher' else 8
            # Preserve original batching boundaries even for a partial last model.
            with torch.no_grad():
                for start in range(0,len(windows),batch_size):
                    group = windows[start:start+batch_size]
                    if all(Path(row['path']).stem+'.npy' in cached for row in group):
                        continue
                    images = []
                    for row in group:
                        with np.load(row['path'],allow_pickle=False) as data:
                            images.append(torch.from_numpy(data['image'].copy()))
                    x = torch.stack(images).cuda()
                    z = model(x)
                    cj.finite_output(model,(x,),z)
                    margins = (z[:,1]-z[:,0]).cpu().numpy()
                    for row, margin in zip(group,margins):
                        path = destination / (Path(row['path']).stem+'.npy')
                        temporary = path.with_name(path.name+'.tmp.npy')
                        np.save(temporary, margin.astype(np.float32))
                        os.replace(temporary,path)
                    if start % 800 == 0:
                        print(entry['id'], start, '/', len(windows), 'cached',len(cached),flush=True)
            torch.cuda.synchronize()
            backend = ev.backend_record('deployment_tf32_off')
            del model
            torch.cuda.empty_cache()
        else:
            backend = dict(policy='deployment_tf32_off', provenance='original immutable inference command; reused accuracy only, timings excluded')
        records.append(dict(model=entry['id'], windows=len(windows), reused_windows=len(cached), newly_required_windows=len(missing), seconds=time.monotonic()-tic, checkpoint_sha256=entry['sha256'], directory=str(destination), actual_model_boundary=backend, inherited_latency_measurements_excluded=True))
        print('COMPLETE',entry['id'],'reused',len(cached),'new',len(missing),flush=True)
    c.save(c.ROOT/'gpu_inference_sealed.json',dict(outcome='completed',seconds=time.monotonic()-begin,records=records,full_set=True,stored_margin_dtype='float32',resume_plan_sha256=c.sha(c.ROOT/'sealed_resume_plan02.json'),sealed_raw_labels_reopened=False))

if __name__=='__main__':
    main()
