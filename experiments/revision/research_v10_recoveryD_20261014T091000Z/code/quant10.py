"""Adopted simulator plus explicit strict all-layer INT8 and clipping instrumentation."""
import copy
import common10 as c

def _per_channel_class():
    import torch
    from models.compression_quant import FixedQATConv2d
    class PerChannelImpl(FixedQATConv2d):
        def __init__(self,module,scale):
            super().__init__(module,module._act_scale,module._act_zp,module._w_scale,weight_bits=module.weight_bits,activation_bits=module.activation_bits)
            self.register_buffer('_input_channel_scale',scale.float().clone())
        def forward(self,x):
            if self.fake_quant_enabled:
                x=torch.fake_quantize_per_channel_affine(x,self._input_channel_scale,torch.zeros_like(self._input_channel_scale,dtype=torch.int32),1,-2**(self.activation_bits-1),2**(self.activation_bits-1)-1)
                w=torch.fake_quantize_per_channel_affine(self.weight,self._w_scale,torch.zeros(self.out_channels,dtype=torch.int32,device=self.weight.device),0,-2**(self.weight_bits-1),2**(self.weight_bits-1)-1)
            else:w=self.weight
            return self._conv_forward(x,w,self.bias)
    return PerChannelImpl

def PerChannel(module,scale):return _per_channel_class()(module,scale)

def load(model,state):
    from models.compression_quant import load_compression_qat
    from quantization.qat import _set_module
    filtered={k:v for k,v in state.items() if not k.endswith('._input_channel_scale')}
    q=load_compression_qat(model,filtered)
    for key,scale in state.items():
        if key.endswith('._input_channel_scale'):
            name=key[:-len('._input_channel_scale')];_set_module(q,name,PerChannel(dict(q.named_modules())[name],scale))
    return q

def prepare(model,batches,precision,calibration_seed=1729):
    import torch
    from models.compression_quant import prepare_compression_qat,FixedQATConv2d,freeze_batch_norm
    from quantization.qat import prepare_qat,_set_module,QATConv2d
    if precision=='W8A8':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(calibration_seed)
            q=prepare_qat(model,batches,skip_last_n=0,calibration_type='per_tensor')
            for name,m in list(q.named_modules()):
                if isinstance(m,QATConv2d):_set_module(q,name,FixedQATConv2d(m,m._act_scale,m._act_zp,m._w_scale,weight_bits=8,activation_bits=8))
        freeze_batch_norm(q);q.eval()
        assert not any(type(m)==torch.nn.Conv2d for m in q.modules()),'Strict point contains a floating convolution'
        return q.to(c.device())
    bits=4 if precision=='W4A4' else 8
    q=prepare_compression_qat(model,batches,weight_bits=4,activation_bits=bits,calibration_seed=calibration_seed)
    quantized=[m for m in q.modules() if isinstance(m,FixedQATConv2d)]
    if not quantized:
        # R1's sole convolution is also its logits head. The declared FP32-head
        # exception takes precedence; mixed-bit requests collapse to the FP32
        # parent. Do not manufacture a 4-bit body or quantize the excluded head.
        convs=[m for m in q.modules() if isinstance(m,torch.nn.Conv2d)]
        assert len(convs)==1 and convs[0].out_channels==2,'Unexpected empty mixed-precision graph'
        return q.to(c.device()).eval()
    stem=quantized[0]
    stem._w_scale.mul_(7/127);stem._weight_bits.fill_(8);stem._weight_bits_value=8
    if stem.activation_bits==4:
        stem._act_scale.mul_(7/127);stem._activation_scale_value=float(stem._act_scale[0])
    stem._activation_bits.fill_(8);stem._activation_bits_value=8
    return q.to(c.device()).eval()

def clipping_hooks(model):
    import torch
    from models.compression_quant import FixedQATConv2d
    counts={};handles=[]
    for name,module in model.named_modules():
        if not isinstance(module,FixedQATConv2d):continue
        def hook(mod,inputs,key=name):
            x=inputs[0].detach();bits=mod.activation_bits;scale=mod._input_channel_scale.view(1,-1,1,1) if hasattr(mod,'_input_channel_scale') else mod._activation_scale_value;lo=-2**(bits-1)*scale;hi=(2**(bits-1)-1)*scale
            outside=((x<lo)|(x>hi)).sum();n=x.numel();old=counts.setdefault(key,[0,0]);old[0]=old[0]+outside;old[1]+=n
        handles.append(module.register_forward_pre_hook(hook))
    return counts,handles
