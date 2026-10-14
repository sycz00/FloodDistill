"""Small V9 adapters around the frozen research capsule and audited queue helpers."""
import json
import os
from pathlib import Path
import sys
import hashlib
import audited_queue as q

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
V7=ROOT.parent/'capacity_v7_20261011T210730Z'
V8=ROOT.parent/'research_v8_20261012T123202Z'
PROBE=ROOT.parent/'novelty_probe_20261012'
CAPSULE=ROOT.parent/'capacity_v4_20261008/frozen/297031f5f4ae1df5'
RUNGS=ROOT.parent/'capacity_v4_20261008/code/v5'
PYTHON='/home/schmalstieg/miniconda3/envs/esi/bin/python'
SEEDS=[42,64,91,107]
GPUS=['GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc','GPU-eadba879-32d7-d9cf-1214-f9e16ec4c5cb']
DATA=Path('/data1/schmalstieg/flooddistill')/('revision_'+ROOT.name)

def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def sha(p):return q.file_hash(p)
def save(p,value):q.atomic_json(p,value)

def setup(device='cpu'):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(CAPSULE));sys.path.insert(0,str(RUNGS))
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('WANDB_MODE','disabled')
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
    import torch
    torch.set_num_threads(2)
    if device=='cuda' and os.environ.get('V9_COLOCATION')=='1':
        peak=float(os.environ['V9_PEAK_MIB'])
        total=torch.cuda.get_device_properties(0).total_memory/2**20
        torch.cuda.set_per_process_memory_fraction(min(.95,(peak+512)/total),0)
    import scripts.compression_jobs as cj
    import evaluate_v7 as ev
    return torch,cj,ev

def teacher_cpu(spec):
    from models.teacher import build_teacher
    assert sha(spec['teacher'])==spec['teacher_sha256']
    return build_teacher(variant='prithvi_eo_v2_300_tl',num_classes=2,pretrained=False,freeze=True,official_ckpt=spec['teacher']).eval().requires_grad_(False)

def build(graph):
    if graph=='B0':
        from models.student import build_student
        return build_student('efficientvit_b0',pretrained=False,in_channels=6,num_classes=2)
    import rungs
    return rungs.build(graph)

def load_model(entry,device='cpu'):
    import torch
    assert sha(entry['checkpoint'])==entry['sha256'],entry['id']
    payload=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
    if entry.get('quantized'):
        from models.compression_quant import load_compression_qat
        model=load_compression_qat(build('R4'),payload['model'])
    elif entry['graph']=='UNet':
        from models.student import build_student
        model=build_student('unet_resnet50',pretrained=False,in_channels=6,num_classes=2)
        model.load_state_dict(payload['model'],strict=True)
    else:
        model=build(entry['graph']);model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)

def install_label_guard(final=False):
    """Deny sealed GT file reads in all jobs except the one final reader."""
    import rasterio
    allowed={};opened=set();active=[None]
    roots=[str(Path('/data1/schmalstieg/flooddistill/data/WorldFloodsv2')/s/'gt')+'/' for s in ['train','val']]
    def protected(path):
        try:return any(str(Path(path).resolve()).startswith(r) for r in roots)
        except (TypeError,OSError):return False
    def hook(event,args):
        if event=='open' and args and protected(args[0]):
            path=str(Path(args[0]).resolve())
            if not final or active[0]!=path or path in opened:raise PermissionError('sealed reference access denied: '+path)
            opened.add(path)
    sys.addaudithook(hook)
    original=rasterio.open
    def guarded(path,*args,**kwargs):
        if protected(path):raise PermissionError('sealed GT paths may only be loaded once through the final byte reader')
        return original(path,*args,**kwargs)
    rasterio.open=guarded
    def once(path):
        import io
        path=str(Path(path).resolve())
        if not final or path in opened:raise PermissionError('sealed label reopened: '+path)
        st=Path(path).stat();active[0]=path
        try:
            with open(path,'rb') as f:data=f.read()
        finally:active[0]=None
        with rasterio.io.MemoryFile(data) as mem:
            with mem.open() as src:arrays=src.read()
        row=dict(at=q.now(),path=path,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),mtime_ns=st.st_mtime_ns,
            purpose='final evaluation after every checkpoint and setting is frozen',opened_once=True)
        with open(ROOT/'label_access.jsonl','a',buffering=1) as f:f.write(json.dumps(row)+'\n')
        return arrays
    return once
