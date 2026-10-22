#!/usr/bin/env python3
"""Reconstruct selected mixed QAT checkpoints with fixed saved grids, never recalibrate."""
import copy,json
from diagnose_unfolded import ROOT,OLD,q,x,rows
import numpy as np

def main(seed):
    results=[]
    for seed in (seed,):
        folder=ROOT/f'fit_cpu_s{seed}'
        if (ROOT/f'mixed_exceptions_s{seed}_v2/RESULT.json').exists() and x.read(ROOT/f'mixed_exceptions_s{seed}_v2/RESULT.json').get('acceptable'):folder=ROOT/f'mixed_exceptions_s{seed}_v2'
        if (ROOT/f'continue_cpu_s{seed}/RESULT.json').exists() and x.read(ROOT/f'continue_cpu_s{seed}/RESULT.json').get('acceptable'):folder=ROOT/f'continue_cpu_s{seed}'
        if (ROOT/f'parent_match_cpu_s{seed}/RESULT.json').exists() and x.read(ROOT/f'parent_match_cpu_s{seed}/RESULT.json').get('acceptable'):folder=ROOT/f'parent_match_cpu_s{seed}'
        result=x.read(folder/'RESULT.json');assert result['acceptable']
        plan=x.read(OLD/'workstation/PROTOCOL.json');entry=next(v for v in plan['entries'] if v['seed']==seed)
        torch,parent,ji,ds,inv,ev=q.runtime(x.read(entry['config']['path']));items=list(q.source_items(ds,inv,ev))
        checkpoint=result['selected']['checkpoint'];assert x.sha(checkpoint['path'])==checkpoint['sha256']
        saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
        from mixed_smooth import InputScaledConv
        from quantization.qat import _set_module
        from models.compression_quant import load_compression_qat,FixedQATConv2d
        target=copy.deepcopy(parent).eval()
        for record in saved['scales']['records']:
            conv=target.get_submodule(record['name']);_set_module(target,record['name'],InputScaledConv(conv,torch.tensor(record['scale'])))
        target=load_compression_qat(target,saved['model']).eval()
        quantized=sum(isinstance(m,FixedQATConv2d) for m in target.modules());assert quantized==result.get('deployment_quantized_convs',56)
        assert not isinstance(target.get_submodule(saved['scales']['head']),FixedQATConv2d)
        expected=x.read(folder/'SOURCE_HISTORY.json')[-1]['evaluation']['rows'];assert rows(target,items,torch)==expected
        gate=result['selected']['source_gate'];assert x.sha(gate['onnx']['path'])==gate['onnx']['sha256']
        session=ji.session(gate['onnx']['path']);actual=[]
        for image,mask,rec in items:
            logits=session.run(None,{'image':image})[0];assert np.isfinite(logits).all();pred=logits[0].argmax(0);valid=(mask>=0)&(mask<2)
            cm=np.bincount((mask[valid]*2+pred[valid]).astype('int64'),minlength=4).reshape(2,2)
            actual.append(dict(file=rec['name'],event=rec['event'],confusion=cm.tolist()))
        assert actual==gate['evaluation']['rows']
        results.append(dict(seed=seed,checkpoint_sha256=checkpoint['sha256'],onnx_sha256=gate['onnx']['sha256'],full89_torch_counts_reproduced=True,full89_ort_counts_reproduced=True,quantized_convs=quantized,total_convs=57,head_float=True,attention_float=True,recalibration=False,optimizer_updates=saved['updates']))
        print(seed,'PASS_RELOAD_AND_DEPLOYMENT_REPRODUCTION',flush=True)
    x.write(ROOT/f'provenance/RELOAD_VERIFICATION_s{seed}.json',dict(status='PASS',rows=results),True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,64,91,107],required=True);main(parser.parse_args().seed)
