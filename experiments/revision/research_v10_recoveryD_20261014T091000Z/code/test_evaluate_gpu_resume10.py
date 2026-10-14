"""CPU-only synthetic durability and exact aggregate recovery checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import types
import numpy as np
import evaluate_gpu10 as e

class ResumeTests(unittest.TestCase):
    def test_end_to_end_committed_rows_not_reread_after_crash(self):
        import torch
        torch.set_num_threads(2)
        class Teacher(torch.nn.Module):
            def forward(self,x):return torch.stack([-x[:,0],x[:,0]],1)
        reference=types.SimpleNamespace(__file__=__file__,reflectance=lambda x:x,
            indices=lambda x:({n:x[0] for n in ['MNDWI','WI2015','AWEIsh']},np.ones(x.shape[1:],bool)))
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'run';root.mkdir();v9=base/'inputs';v9.mkdir()
            entries=[dict(id='teacher',graph='toy')]+[dict(id=n,graph='rule') for n in ['MNDWI','WI2015','AWEIsh']]
            (root/'readout_models.json').write_text(json.dumps(dict(models=entries)))
            (root/'operating_points.json').write_text('{}')
            rows=[]
            for i in range(5):
                image=np.ones((6,4,4),np.float32)*.1;image[0,::2]=-1
                mask=np.zeros((4,4),np.int8);mask[1:3,1:3]=1;mask[0,0]=-1
                path=base/f'tile{i}.npz';np.savez(path,image=image,mask=mask,cloud=np.zeros(mask.shape,bool))
                rows.append(dict(path=str(path),dataset='synthetic',event='one_event',name=str(i),omit=False))
            (v9/'sealed_windows.json').write_text(json.dumps(dict(rows=rows)))
            original=e.save_shard
            def crash_after_commit(*args,**kwargs):
                original(*args,**kwargs);raise RuntimeError('synthetic process interruption')
            with mock.patch.object(e.c,'ROOT',root),mock.patch.object(e.c,'V9',v9),\
                 mock.patch.object(e.c,'setup',return_value=(torch,None,None,reference)),\
                 mock.patch.object(e.c,'device',return_value=torch.device('cpu')),\
                 mock.patch.object(e.c,'load',return_value=Teacher()),mock.patch.object(e,'CHECKPOINT_WINDOWS',2):
                with mock.patch.object(e,'save_shard',side_effect=crash_after_commit):
                    with self.assertRaisesRegex(RuntimeError,'interruption'):e.evaluate('sealed',0)
                self.assertEqual(len((root/'label_access_slot0.jsonl').read_text().splitlines()),2)
                result=e.evaluate('sealed',0)
                self.assertEqual(result['windows'],5)
                accesses=[json.loads(line)['path'] for line in (root/'label_access_slot0.jsonl').read_text().splitlines()]
                self.assertEqual(accesses,[r['path'] for r in rows])
                self.assertEqual(e.evaluate('sealed',0),result)
                self.assertEqual(len((root/'label_access_slot0.jsonl').read_text().splitlines()),5)
                # Uninterrupted comparison in a separate additive root.
                other=base/'uninterrupted';other.mkdir()
                for name in ['readout_models.json','operating_points.json']:(other/name).write_bytes((root/name).read_bytes())
                with mock.patch.object(e.c,'ROOT',other):expected=e.evaluate('sealed',0)
                for name in ['event_counts','strata','joint','shared','spectra','ceilings','calibration','dry','clipping']:
                    self.assertEqual(result[name],expected[name],name)
                with np.load(root/'metrics/sealed/hist_slot0.npz') as actual,np.load(other/'metrics/sealed/hist_slot0.npz') as exp:
                    np.testing.assert_array_equal(actual['keys'],exp['keys']);np.testing.assert_array_equal(actual['histograms'],exp['histograms'])

    def test_integer_deltas_and_exact_sequential_float_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=e.empty_aggregates();expected=e.empty_aggregates();ids=['a','b','c']
            for i,value in enumerate([1e16,1.,-1e16]):
                for name in e.SHAPES:
                    increment=np.full(e.SHAPES[name],value if name in e.FLOAT_TOTALS else i+1,
                                      np.float64 if name in e.FLOAT_TOTALS else np.int64)
                    a[name][('event','model')]+=increment;expected[name][('event','model')]+=increment
                row={'valid_pixels':i+1};a['dry'].append(row);expected['dry'].append(row)
                e.save_shard(root/f'chunk_{i:08d}_{i+1:08d}.npz','protocol',ids[i:i+1],i,a,.5)
                for name in e.SHAPES:
                    if name not in e.FLOAT_TOTALS:a[name].clear()
                a['dry'].clear()
                if i==1:
                    # A process restart restores latest cumulative floats only.
                    paths,pos,seconds=e.committed_shards(root,'protocol',ids)
                    self.assertEqual(pos,2);self.assertEqual(seconds,1.)
                    a=e.restore_shards(paths[-1:],floating_only=True)
            paths,pos,seconds=e.committed_shards(root,'protocol',ids)
            self.assertEqual(pos,3);self.assertEqual(seconds,1.5)
            actual=e.restore_shards(paths)
            for name in e.SHAPES:np.testing.assert_array_equal(actual[name]['event','model'],expected[name]['event','model'])
            self.assertEqual(actual['dry'],expected['dry'])
            self.assertEqual(float(actual['spectra']['event','model'][0]),0.)

    def test_reject_changed_recipe_rows_and_noncontiguous_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=e.empty_aggregates()
            e.save_shard(root/'chunk_00000000_00000001.npz','p',['a'],0,a,1.)
            with self.assertRaisesRegex(AssertionError,'protocol'):e.committed_shards(root,'different',['a'])
            with self.assertRaisesRegex(AssertionError,'row identity'):e.committed_shards(root,'p',['different'])
            e.save_shard(root/'chunk_00000002_00000003.npz','p',['c'],2,a,1.)
            with self.assertRaisesRegex(AssertionError,'Noncontiguous'):e.committed_shards(root,'p',['a','b','c'])

    def test_partial_atomic_write_ignored_and_no_valid_dry_window(self):
        from evaluate10 import is_dry_reference
        self.assertFalse(is_dry_reference(np.full((8,8),-1,np.int8)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'chunk_00000000_00000025.npz.tmp.1').write_bytes(b'interrupted')
            self.assertEqual(e.committed_shards(root,'p',['a']),([],0,0.))

    def test_exact_confusion_with_invalid_and_empty_support(self):
        import torch
        torch.set_num_threads(2)
        from evaluate10 import counts
        rng=np.random.default_rng(348);y=rng.choice([-1,0,1],size=(23,29)).astype(np.int8)
        masks=[np.ones(y.shape,bool),rng.random(y.shape)>.5,np.zeros(y.shape,bool)]
        pred=rng.random((7,*y.shape))>.5
        actual=e.confusion_batch(torch.from_numpy(pred),torch.from_numpy(y),[torch.from_numpy(m) for m in masks]).numpy()
        expected=np.stack([[counts(p,y,m) for m in masks] for p in pred])
        np.testing.assert_array_equal(actual,expected)

if __name__=='__main__':unittest.main(verbosity=2)
