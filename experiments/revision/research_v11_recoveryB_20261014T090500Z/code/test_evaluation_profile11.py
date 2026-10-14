"""Fixed source-only warmup and exact final allocation are part of cost admission."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import action11 as a
import evaluate11 as e
sys.path.insert(0,str(a.c.V10/'code'))
import evaluate10 as old

class ProfileTests(unittest.TestCase):
    def test_one_fixed_warmup_three_fixed_measured_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            for i in range(16):
                path=root/f'source_{i}.npz'
                np.savez(path,image=np.ones((6,2,2),np.float32),mask=np.zeros((2,2),np.int8),cloud=np.empty(0,bool))
                rows.append(dict(path=str(path),dataset='s11val',omit=False))
            rows.append(dict(path='/must/not/open/target.npz',dataset='external',omit=False))
            (root/'manifest.json').write_text(json.dumps(dict(jobs={'30_final_evaluation':dict(threads=2)})))
            (root/'normal_windows.json').write_text(json.dumps(dict(windows=rows)))
            torch=types.SimpleNamespace(get_num_threads=lambda:2,inference_mode=contextlib.nullcontext)
            reference=types.SimpleNamespace(reflectance=lambda x:x)
            forwards=[];metric_calls=[]
            def forward(models,images,module):
                forwards.append(images.shape)
                return {name:np.zeros((4,2,2),np.float32) for name in models},{}
            def readout(entries,margins,y,support):metric_calls.append((len(entries),len(margins)))
            with mock.patch.object(a.c,'ROOT',root),mock.patch.object(a.c,'V9',root),\
                 mock.patch.object(a.c,'setup',return_value=(torch,None,reference)) as setup,\
                 mock.patch.object(a.c,'source_access_guard'),mock.patch.object(e,'load',return_value=object()),\
                 mock.patch.object(e,'forward_batch',side_effect=forward),mock.patch.object(e,'tile_readout',side_effect=readout),\
                 mock.patch.object(old,'strata',side_effect=lambda y,x,c:dict(all=np.ones(y.shape,bool))):
                result=a.evaluation_profile(dict(checkpoints=[{}]))
            setup.assert_called_once_with(2)
            self.assertEqual(forwards,[(4,6,2,2)]*4)
            self.assertEqual(metric_calls,[(76,36)]*16)
            self.assertEqual(result['warmup_windows'],4);self.assertEqual(result['measured_windows'],12)
            self.assertEqual(len(result['seconds_per_window']),3)
            self.assertEqual(result['source_windows'],[r['path'] for r in rows[:16]])
            self.assertEqual(result['threads'],2)
            self.assertEqual(result['conservative_seconds_per_window'],1.5*max(result['seconds_per_window']))
            self.assertGreaterEqual(result['startup_seconds'],result['warmup_seconds'])

if __name__=='__main__':unittest.main(verbosity=2)
