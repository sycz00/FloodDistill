import sys,os,time,json
from pathlib import Path
import numpy as np
base=Path('/home/schmalstieg/repos/FloodDistill/experiments/revision');old=base/'research_v11_recovery_20261014T081500Z';out=Path(__file__).resolve().parent
sys.path.insert(0,str(old/'code'))
import common11 as c
from evaluate11 import load,forward_batch,prepare_metric_support,tile_readout,EVALUATION_BATCH_SIZE
threads=int(sys.argv[1]);started,cpu=time.monotonic(),time.process_time();torch,c10,v=c.setup(threads);c.source_access_guard()
from evaluate10 import strata
profile=c.read(old/'repair/core/profile.json');entry=profile['checkpoints'][-1];model=load(entry,torch,c10)
entries=[dict(id=f'profile_{i}',fp_reference='profile_0') for i in range(36)]+[dict(id=f'alias_{i}',alias_of=f'profile_{i%36}',fp_reference='profile_0',threshold=.01) for i in range(40)]
models={f'profile_{i}':model for i in range(36)}
rows=[r for r in c.read(c.V9/'normal_windows.json')['windows'] if r['dataset']=='s11val' and not r['omit']][:16]
batches=[]
with torch.inference_mode():
 for offset in range(0,len(rows),4):
  batch_start=time.monotonic();data=[]
  for row in rows[offset:offset+4]:
   with np.load(row['path'],allow_pickle=False) as f:data.append((f['image'].copy(),f['mask'].copy(),f['cloud'].copy() if 'cloud' in f and f['cloud'].size else None))
  read_seconds=time.monotonic()-batch_start;forward_start=time.monotonic()
  margins,_=forward_batch(models,np.stack([x for x,y,cloud in data]),torch)
  forward_seconds=time.monotonic()-forward_start;metric_start=time.monotonic()
  for index,(x,y,cloud) in enumerate(data):
   support=prepare_metric_support(y,strata(y,v.reflectance(x),cloud))
   tile_readout(entries,{name:values[index] for name,values in margins.items()},y,support)
  metric_seconds=time.monotonic()-metric_start
  batches.append(dict(warmup=offset==0,seconds_per_window=(time.monotonic()-batch_start)/4,read_seconds=read_seconds,forward_seconds=forward_seconds,metric_seconds=metric_seconds))
  print(json.dumps(batches[-1]),flush=True)
receipt=dict(outcome='completed',threads=torch.get_num_threads(),affinity=sorted(os.sched_getaffinity(0)),batch_size=4,models=36,threshold_variants=76,source_only=True,target_labels_read=False,source_windows=[r['path'] for r in rows],batches=batches,conservative_seconds_per_window=max(b['seconds_per_window'] for b in batches[1:])*1.5,wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,GPU_seconds=0,checkpoint_sha256=entry['sha256'],evaluator_sha256=c.sha(old/'code/evaluate11.py'))
(out/f'threads{threads}.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
