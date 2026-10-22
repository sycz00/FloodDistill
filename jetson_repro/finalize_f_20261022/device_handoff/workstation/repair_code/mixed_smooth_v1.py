"""Power-of-two variant of existing manual.apply_smooth_quant; scalar activation QDQ."""
import copy,torch
from torch import nn
from quantization.qat import _set_module,prepare_qat,QATConv2d,LiteMLAQAT
from models.compression_quant import FixedQATConv2d,freeze_batch_norm

class InputScaledConv(nn.Module):
    def __init__(self,conv,scale):
        super().__init__();self.conv=copy.deepcopy(conv);self.register_buffer('input_scale',scale.reshape(1,-1,1,1).clone())
        w=self.conv.weight;groups=conv.groups;cin=w.shape[1];cout=w.shape[0]//groups
        with torch.no_grad():w.copy_((w.reshape(groups,cout,cin,*w.shape[2:])*scale.reshape(groups,1,cin,1,1)).reshape_as(w))
    def forward(self,image):return self.conv(image/self.input_scale)

def prepare_parent(model,images):
    target=copy.deepcopy(model).eval();convs=[(n,m) for n,m in target.named_modules() if isinstance(m,nn.Conv2d)];assert convs[-1][1].out_channels==2
    chosen=convs[:-1];maxima={};hooks=[]
    def make(name):
        def capture(module,args):
            maximum=args[0].detach().abs().amax((0,2,3)).cpu();maxima[name]=torch.maximum(maxima[name],maximum) if name in maxima else maximum
        return capture
    for name,m in chosen:hooks.append(m.register_forward_pre_hook(make(name)))
    with torch.no_grad():
        for image in images:target(image)
    for hook in hooks:hook.remove()
    records=[]
    for name,m in chosen:
        w=m.weight.detach().cpu();groups=m.groups;cin=w.shape[1];cout=w.shape[0]//groups
        weight_max=w.reshape(groups,cout,cin,*w.shape[2:]).abs().amax((1,3,4)).reshape(-1);act_max=maxima[name]
        ideal=act_max.clamp(min=1e-8).sqrt()/weight_max.clamp(min=1e-8).sqrt();scale=torch.pow(2.,torch.round(torch.log2(ideal)).clamp(-16,16));scale=torch.where((act_max>0)&(weight_max>0),scale,torch.ones_like(scale))
        _set_module(target,name,InputScaledConv(m,scale));records.append(dict(name=name,scale=scale.tolist(),act_max=act_max.tolist(),weight_max=weight_max.tolist()))
    freeze_batch_norm(target);return target.eval(),dict(alpha=.5,power_of_two=True,head=convs[-1][0],scaled_convs=len(chosen),records=records)

def prepare_quant(model,images):
    torch.manual_seed(1729);target=prepare_qat(model,[dict(image=image) for image in images],skip_last_n=1,calibration_type='per_tensor')
    for name,m in list(target.named_modules()):
        if isinstance(m,LiteMLAQAT):_set_module(target,name,m.litemla)
    for name,m in list(target.named_modules()):
        if isinstance(m,QATConv2d):_set_module(target,name,FixedQATConv2d(m,m._act_scale,m._act_zp,m._w_scale,weight_bits=8,activation_bits=8))
    freeze_batch_norm(target);return target.eval()
