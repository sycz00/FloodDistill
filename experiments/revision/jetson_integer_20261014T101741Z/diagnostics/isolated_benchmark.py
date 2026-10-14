import os,sys,time,json,resource
from pathlib import Path
import numpy as np,onnxruntime as ort
from statistics import median
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
model,assets,output=map(Path,sys.argv[1:])
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.log_severity_level=3
s=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
with np.load(sorted(assets.glob('*.npz'))[-1],allow_pickle=False) as a:x=a['image'][None].astype('float32')
for _ in range(20):s.run(None,{'image':x})
times=[]
for _ in range(100):
 t=time.perf_counter();s.run(None,{'image':x});times.append((time.perf_counter()-t)*1000)
output.write_text(json.dumps(dict(model=str(model),backend='ORT CPUExecutionProvider',batch=1,shape=list(x.shape),mean_ms=float(np.mean(times)),p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),warmup=20,iterations=100,threads=2,CPU_affinity=sorted(os.sched_getaffinity(0)),isolated_process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,memory_boundary='includes Python/NumPy/ORT imports, session, input/output and intermediates; not isolated weights or Jetson memory',timing_boundary='session.run; excludes disk IO and preprocessing; host under ordinary unrelated workload'),indent=2))
