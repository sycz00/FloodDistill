import os,sys,argparse,json,copy,time,fcntl
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--uuid',required=True);a=p.parse_args();uuid=a.uuid
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'8_jetson_export_20261022';REPO=ROOT.parents[2]/'FloodDistill';F=REPO/'experiments/revision/controlled512_v2_20261016T213414Z/revision_finalize_20261019T141008Z'
locks=[]
for path in ('/tmp/weather-next-numerical-gpu0.lock','/tmp/flooddistill_revision_gpu.lock'):
 f=Path(path).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
sys.path.insert(0,str(F/'code'));import experiment_quant as q
os.environ['CUDA_VISIBLE_DEVICES']=uuid
import numpy as np
plan=q.x.read(OLD/'workstation/PROTOCOL.json');entry=next(v for v in plan['entries'] if v['seed']==42)
torch,model,ji,ds,inv,ev=q.runtime(q.x.read(entry['config']['path']));assert torch.cuda.is_available() and torch.cuda.device_count()==1
items=list(q.source_items(ds,inv,ev));gpu=copy.deepcopy(model).to('cuda:0').eval();checks=[]
with torch.no_grad():
 for im,mask,rec in items:
  a=model(torch.from_numpy(im)).numpy();b=gpu(torch.from_numpy(im).cuda()).cpu().numpy();checks.append(dict(file=rec['name'],passed=bool(np.allclose(a,b,atol=.001,rtol=.001)),max_abs=float(abs(a-b).max())))
result=dict(status='PASS' if all(r['passed'] for r in checks) else 'BLOCKED_GPU_FLOAT_PARITY',uuid=uuid,logical_device='cuda:0',device_name=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,images=89,atol=.001,rtol=.001,rows=checks,max_memory_bytes=torch.cuda.max_memory_allocated())
q.x.write(ROOT/'GPU_PREFLIGHT.json',result,True);print(result['status'],max(v['max_abs'] for v in checks),flush=True)
