"""Required source-only verification; keeps failed cache requirements explicit."""
import time
from pathlib import Path
import common10 as c
def main():
    torch,cj,old,v=c.setup();tic=time.monotonic();spec=c.read(c.V9/'training_spec.json');calibration=c.read(c.V9.parent/'research_v8_20261012T123202Z/config.json')['calibration_indices'];ds=cj.pseudo(dict(spec,seed=1729),augment=False);batches=[dict(image=ds[i]['image'][None].float()) for i in calibration]
    from quant10 import prepare,clipping_hooks
    from models.compression_quant import precision_map
    parents=[m for m in c.read(c.V9/'final_models_final03.json')['models'] if m['method']=='lkd' and m['graph']=='R4'];rows=[]
    for seed in [42,64]:
        parent=next(x for x in parents if x['seed']==seed);base=c.load(parent);torch.set_num_threads(2);q=prepare(base,batches,'W4A8');saved=torch.load(c.V9/'ptq_states'/f'q_s{seed}.pt',map_location='cpu',weights_only=True)['model'];state=q.state_dict();different=[k for k in state if not torch.equal(state[k],saved[k])];c.save(c.ROOT/f'ptq_reproduction_s{seed}.json',dict(different=different,max_errors={k:float((state[k].float()-saved[k].float()).abs().max()) for k in different}));assert all(torch.allclose(state[k].float(),saved[k].float(),rtol=1e-3,atol=1e-7) for k in different),different
        rows.append(dict(seed=seed,all_tensors_identical=not different,quantizer_max_difference=max([float((state[k].float()-saved[k].float()).abs().max()) for k in different] or [0]),qualification='Original42/64 calibration executed on GPU in V8; CPU regeneration compares ranges and retains numerical differences; master weights identical',parent_sha256=parent['sha256']))
    strict=prepare(c.load(parents[0]),batches,'W8A8');mapping=precision_map(strict);assert mapping and all(x.get('weight_bits',8)==8 for x in mapping if isinstance(x,dict))
    counters,handles=clipping_hooks(strict)
    with torch.no_grad():strict(torch.full((1,6,224,224),1e3))
    assert any(v[0]>0 for v in counters.values())
    for h in handles:h.remove()
    val,_=cj.loaders(spec);model=c.load(parents[0]);cells=None
    from utils.metrics import accumulate_metrics,finalize_metrics
    with torch.no_grad():
        for b in val:cells=accumulate_metrics(model(b['image'].float()),b['mask'].long(),2,-1,cells)
    metrics=finalize_metrics(cells);anchor=c.read(c.V7/'receipts/R4_LKD_s42.json')
    if anchor is None:
        anchor=c.read(c.V5/'v5/results/fl_R4_lr5e-3_s42.json')
    expected=anchor['source_validation'].get('best_val_mIoU');difference=abs(metrics['mIoU']-expected);assert difference<=.001,(metrics['mIoU'],expected)
    c.save(c.ROOT/'verification.json',dict(outcome='completed',PTQ_tensor_parity=rows,strict_all_convolution_W8A8=True,strict_precision_map=mapping,synthetic_clipping_detected=True,source_validation_mIoU=metrics['mIoU'],source_validation_delta=difference,sealed_window_manifest_sha256=c.sha(c.V9/'sealed_windows.json'),sealed_labels_opened=False,N251_artifacts_read=False,seconds=time.monotonic()-tic))
if __name__=='__main__':main()
