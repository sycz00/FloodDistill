"""GPU-specific numerical and device risks before training admission."""
import time
import common10 as c

def main():
    tic=time.monotonic();torch,cj,old,v=c.setup();assert c.device().type=='cuda'
    from models.compression_quant import FixedQATConv2d
    from quant10 import prepare,PerChannel,load,clipping_hooks
    from quantization.qat import _set_module
    from training.losses import focal_loss,DistillLoss
    class Teacher(torch.nn.Module):
        def __init__(self,z):super().__init__();self.z=z
        def forward(self,x):return self.z
    torch.manual_seed(1729);z=torch.randn(2,2,8,8,device=c.device(),requires_grad=True);y=torch.randint(0,2,(2,8,8),device=c.device());y[0,0,0]=-1;teacher_z=torch.randn_like(z)
    soft=v.soft_focal_loss(z,(y==1).float(),y!=-1);assert torch.allclose(soft,focal_loss(z,y,gamma=2,ignore_index=-1),atol=1e-6)
    from training.losses import kd_loss
    offline=v.offline_kd_loss(z,y,teacher_z);reference=.5*focal_loss(z,y,2.,-1)+.5*kd_loss(z,teacher_z,4.,valid_mask=y!=-1);assert torch.allclose(offline,reference,atol=1e-6);assert torch.isfinite(offline);offline.backward();assert torch.isfinite(z.grad).all()
    tiny=c.build('R2').to(c.device()).eval();x=torch.randn(2,6,32,32,device=c.device());q=prepare(tiny,[dict(image=x)],'W4A4');stem=next(m for m in q.modules() if isinstance(m,FixedQATConv2d));assert stem.weight_bits==stem.activation_bits==8
    for name,module in list(q.named_modules()):
        if isinstance(module,FixedQATConv2d):_set_module(q,name,PerChannel(module,torch.full((module.in_channels,),float(module._act_scale[0]),device=c.device())))
    restored=load(c.build('R2'),{k:t.cpu() for k,t in q.state_dict().items()}).to(c.device()).eval()
    with torch.no_grad():assert torch.equal(q(x),restored(x))
    strict=prepare(c.build('R2').to(c.device()).eval(),[dict(image=x)],'W8A8');assert not any(type(m)==torch.nn.Conv2d for m in strict.modules());counters,handles=clipping_hooks(strict)
    with torch.no_grad():strict(torch.ones_like(x)*1000)
    assert sum(int(a[0]) for a in counters.values())>0
    for handle in handles:handle.remove()
    from evaluate_gpu10 import confusion_batch
    from evaluate10 import counts
    ytoy=torch.randint(-1,2,(16,16),device=c.device());ptoy=torch.rand(8,16,16,device=c.device())>.5;stoy=torch.rand(4,16,16,device=c.device())>.3
    matrix=confusion_batch(ptoy,ytoy,stoy).cpu().numpy();reference=__import__('numpy').array([[counts(p,ytoy.cpu().numpy(),part) for part in stoy.cpu().numpy()] for p in ptoy.cpu().numpy()]);assert __import__('numpy').array_equal(matrix,reference)
    spec=c.read(c.V9/'training_spec.json');teacher_cpu=old.teacher_cpu(spec);teacher_gpu=old.teacher_cpu(spec).to(c.device());ds=cj.pseudo(dict(spec,seed=42),augment=False);checks=[]
    with torch.no_grad():
        for i in range(5):
            x=ds[i]['image'].float();images=torch.stack([(x.flip(-1) if flip else x).rot90(k,(-2,-1)) for flip in [False,True] for k in range(4)]);GPU=teacher_gpu(images.to(c.device())).cpu();CPU=teacher_cpu(images);delta=float((GPU-CPU).abs().max());assert delta<.001,(i,delta)
            checks.append(dict(tile=ds.files[i].name,CPU_GPU_logit_max_difference=delta,views=8))
    from evaluate10 import source
    anchor=next(e for e in c.read(c.ROOT/'historical_models.json') if e['graph']=='R4' and e['method']=='lkd' and e['seed']==42)
    source_score=source(anchor);saved_reference=c.read(c.ROOT/'verification.json')['source_validation_mIoU'];GPU_mIoU=source_score['pooled']['mIoU'];assert abs(GPU_mIoU-saved_reference)<.001
    c.save(c.ROOT/'GPU_preflight.json',dict(outcome='completed',seconds=time.monotonic()-tic,GPU_device=torch.cuda.get_device_name(0),CUDA_runtime=torch.version.cuda,FP32=True,TF32=False,soft_focal_hard_parity=True,offline_KD_finite_gradient=True,quantized_reload_bit_identical=True,GPU_batched_counts_CPU_bit_identical=True,strict_W8A8=True,GPU_existing_source_mIoU=GPU_mIoU,CPU_GPU_source_mIoU_delta=GPU_mIoU-saved_reference,synthetic_clipping=True,CPU_GPU_teacher_checks=checks,raw_sealed_labels_opened=False))
if __name__=='__main__':main()
