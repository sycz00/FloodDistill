"""Focused regression checks for zero-valid dry alarms and exact faster decoding."""
import json,time
from pathlib import Path
import numpy as np
import edge_v9 as edge
import evaluate_v9_final03 as evaluate

def main():
    started=time.monotonic();cpu=time.process_time();rng=np.random.default_rng(1729)
    assert not evaluate.valid_dry_reference(np.empty(0,np.int64))
    assert evaluate.valid_dry_reference(np.zeros(8,np.int64))
    assert not evaluate.valid_dry_reference(np.array([0,1,0]))
    checks=0
    for bits in range(512):
        pred=((bits>>np.arange(9))&1).reshape(3,3).astype(bool)
        mnd=rng.uniform(-1,1,(3,3));valid=rng.random((3,3))>.2
        for k in [1,2,3,5]:
            np.testing.assert_array_equal(edge.decode(pred,mnd,valid,k,.1),edge.decode_reference(pred,mnd,valid,k,.1));checks+=1
    timings={}
    for shape in [(128,128),(512,512)]:
        pred=rng.random(shape)>.65;mnd=rng.uniform(-1,1,shape);valid=rng.random(shape)>.1
        for method in ['decode','decode_reference']:
            tic=time.monotonic();out=getattr(edge,method)(pred,mnd,valid,2,.1);timings[str(shape)+'_'+method]=time.monotonic()-tic
        np.testing.assert_array_equal(edge.decode(pred,mnd,valid,2,.1),edge.decode_reference(pred,mnd,valid,2,.1));checks+=1
    result=dict(outcome='completed',status='passed',zero_valid_never_dry=True,exact_decode_comparisons=checks,
                wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
                timings=timings,threads=2,target_labels_opened=False)
    evaluate.c.save(Path(__file__).resolve().parent/'regression_tests.json',result);print(json.dumps(result))

if __name__=='__main__':main()
