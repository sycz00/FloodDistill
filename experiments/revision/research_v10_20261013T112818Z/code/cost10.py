"""Cost annotations from actual registered graphs; no latency or energy claims."""
import common10 as c
def main():
    torch,_,_,_=c.setup();inventory=c.read(c.ROOT/'readout_models.json');cache=c.read(c.ROOT/'graph_costs.json',{})
    for entry in inventory['models']:
        if entry['graph']=='rule':entry.update(parameters=0,MACs_per_pixel=None,FP32_bytes=0);continue
        model=c.load(entry);parameters=sum(p.numel() for p in model.parameters());entry.update(parameters=parameters,FP32_bytes=parameters*4)
        graph=entry['graph']
        if graph not in cache:
            with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as profiler:
                model(torch.zeros(1,6,512,512))
            flops=sum(event.flops or 0 for event in profiler.key_averages());cache[graph]=dict(parameters=parameters,MACs_per_pixel=flops/2/(512*512),scope='Profiled Conv2d/mm/bmm FLOPs divided by2; unsupported elementwise/pooling/quantizer operations excluded;512x512 CPU graph')
        entry['MACs_per_pixel']=cache[graph]['MACs_per_pixel'];entry['MAC_scope']=cache[graph]['scope']
        packed=0;metadata=0
        for name,module in model.named_modules():
            for key,parameter in module.named_parameters(recurse=False):
                bits=module.weight_bits if key=='weight' and hasattr(module,'weight_bits') else 32;packed+=(parameter.numel()*bits+7)//8
            for key,buffer in module.named_buffers(recurse=False):
                if key.startswith('_'):metadata+=buffer.numel()*buffer.element_size()
        entry['packed_bytes']=packed;entry['quantizer_metadata_bytes']=metadata
        del model
    c.save(c.ROOT/'graph_costs.json',cache);c.save(c.ROOT/'readout_models.json',inventory)
if __name__=='__main__':main()
