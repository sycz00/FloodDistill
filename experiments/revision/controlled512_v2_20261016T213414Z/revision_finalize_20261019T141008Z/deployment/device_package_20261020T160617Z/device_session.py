#!/usr/bin/env python3
"""Run a copied, hash-verified package on the author's Jetson; retain every attempt."""
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;manifest=json.loads((root/'MANIFEST.json').read_text())
    for item in manifest['files']:
        if sha(root/item['path'])!=item['sha256']:raise RuntimeError('Package changed: '+item['path'])
    import tensorrt as trt
    if not trt.__version__.startswith('8.4.'):raise RuntimeError('Requires preserved TensorRT8.4; no automatic version changes')
    out=root/'outputs'/a.attempt
    out.mkdir(parents=True,exist_ok=False)
    env=dict(python=sys.version,platform=platform.platform(),tensorrt=trt.__version__,power='NOT_MEASURED',clock_changes=False)
    try:
        import pycuda.driver as cuda
        cuda.init()
        env['cuda']=dict(toolkit=list(cuda.get_version()),driver=cuda.get_driver_version(),
            devices=[dict(index=i,name=cuda.Device(i).name(),total_memory_bytes=cuda.Device(i).total_memory()) for i in range(cuda.Device.count())])
    except Exception as error:env['cuda']={'error':repr(error)}
    try:
        from importlib.metadata import version,PackageNotFoundError
        env['packages']={}
        for name in ['numpy','onnx','pycuda']:
            try:env['packages'][name]=version(name)
            except PackageNotFoundError:env['packages'][name]='NOT_INSTALLED'
    except ImportError as error:env['packages']={'error':repr(error)}
    for label,argv in [('power_mode',['nvpmodel','-q']),('clocks',['jetson_clocks','--show']),('l4t',['cat','/etc/nv_tegra_release'])]:
        try:
            r=subprocess.run(argv,capture_output=True,text=True);env[label]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except OSError as e:env[label]={'error':repr(e)}
    save(out/'environment.json',env)
    for split,count in [('val',89),('test',90),('bolivia',15)]:
        paths=sorted((a.assets_root/('source_'+split)).glob('*.npz'))
        if len(paths)!=count:raise RuntimeError('Missing prepared source_'+split)
        actual={p.name:sha(p) for p in paths}
        if actual!=manifest['assets'][split]:raise RuntimeError('Asset hash mismatch: '+split)
    outcomes=[]
    for model in manifest['models']:
        if not model['admitted']:
            outcomes.append(dict(id=model['id'],status='BLOCKED',reason=model['reason']));save(out/'status.json',outcomes);continue
        d=out/model['id'];d.mkdir()
        engine=d/'model.engine';build=d/'build.json'
        argv=[sys.executable,str(root/'integer_tools.py'),'build','--model',str(root/model['onnx']),
              '--engine',str(engine),'--receipt',str(build),'--mode',model['mode']]
        with (d/'build.log').open('w') as log:r=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            outcomes.append(dict(id=model['id'],status='FAILED_BUILD',returncode=r.returncode));save(out/'status.json',outcomes);continue
        if model.get('integer_io'):
            audit=json.loads(build.read_text())['precision_audit']
            if not audit['full_integer_tensor_path']:
                outcomes.append(dict(id=model['id'],status='FAILED_TENSOR_PRECISION_CERTIFICATE'));save(out/'status.json',outcomes);continue
        for split,count in [('val',89),('test',90),('bolivia',15)]:
            cmd=[sys.executable,str(root/'device_runtime.py'),'--engine',str(engine),'--assets',str(a.assets_root/('source_'+split)),
                 '--expected-images',str(count),'--receipt',str(d/(split+'.json'))]
            if model.get('integer_io'):cmd+=['--io-receipt',str(root/model['io_receipt'])]
            with (d/(split+'.log')).open('w') as log:r=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            outcomes.append(dict(id=model['id'],split=split,status='EVALUATED' if r.returncode==0 else 'FAILED_EVALUATION',returncode=r.returncode))
            save(out/'status.json',outcomes)
        save(out/'status.json',outcomes)
    save(out/'COMPLETE.json',dict(status='DEVICE_ATTEMPTS_RESOLVED',outcomes=outcomes,power='NOT_MEASURED'))

if __name__=='__main__':main()
