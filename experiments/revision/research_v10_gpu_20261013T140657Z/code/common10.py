"""V10 narrow adapters; historical runs are read-only."""
import json,sys,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT.parents[2]
V9=ROOT.parent/'research_v9_recovery_20261013T075000Z'
V7=ROOT.parent/'capacity_v7_20261011T210730Z'
V5=ROOT.parent/'capacity_v4_20261008'
CAPSULE=V5/'frozen/297031f5f4ae1df5'
RUNGS=V5/'code/v5'
DATA=Path('/data1/schmalstieg/flooddistill')/('revision_'+ROOT.name)
SEEDS=[42,64,91,107]
PYTHON='/home/schmalstieg/miniconda3/envs/esi/bin/python'
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for a in iter(lambda:f.read(2**20),b''):h.update(a)
    return h.hexdigest()
def save(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def setup(threads=8):
    sys.dont_write_bytecode=True
    for p in [CAPSULE,RUNGS,V9,ROOT/'reference']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('WANDB_MODE','disabled');os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
    import torch
    torch.set_num_threads(threads)
    if os.environ.get('V10_COMPUTE_DEVICE')=='cuda':
        assert torch.cuda.is_available(),'Authorized GPU job must initialize CUDA'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.95,13*2**30/total),0)
    import common_v9 as old
    
    if not getattr(old,"_v10_guard",False):
        old.install_label_guard();old._v10_guard=True
        def prohibit_obsolete(event,args):
            if event=="open" and args and isinstance(args[0],(str,bytes)) and "N251" in str(args[0]):raise PermissionError("Obsolete N251 artifact access prohibited")
        sys.addaudithook(prohibit_obsolete)
    import scripts.compression_jobs as cj
    import v10_reference as v
    return torch,cj,old,v
def state_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def build(graph,pretrained=False):
    import torch
    if graph=='UNet':
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name='resnet50',encoder_weights='imagenet' if pretrained else None,in_channels=6,classes=2,activation=None)
    if graph=='B0':
        from models.student import build_student
        return build_student('efficientvit_b0',pretrained=False,in_channels=6,num_classes=2)
    if graph in ['R2_log','R2_ctx','R4_micro']:
        import v10_reference as v
        stats=read(ROOT/'log_train_statistics.json',{})
        return v.build_tiny(graph,log_mean=stats.get('mean') if graph=='R2_log' else None,log_std=stats.get('std') if graph=='R2_log' else None)
    import rungs
    return rungs.build(graph)
def load(entry):
    torch,_,_,_=setup()
    assert 'N251' not in entry['checkpoint'],'Obsolete N251 checkpoint prohibited'
    assert sha(entry['checkpoint'])==entry['sha256'],entry['id']
    
    if entry['graph']=='teacher':
        import common_v9 as old
        return old.teacher_cpu(read(V9/'training_spec.json')).to(device())
    a=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True);model=build(entry['graph'])
    if entry.get('quantized'):
        from quant10 import load
        model=load(model,a['model'])
    else:model.load_state_dict(a['model'],strict=True)
    return model.to(device()).eval().requires_grad_(False)

def device():
    import torch
    return torch.device('cuda:0' if os.environ.get('V10_COMPUTE_DEVICE')=='cuda' else 'cpu')
