"""Synthetic-only evaluator checks: paired events, aliases, storage and one-pass readout."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import evaluate11 as e


class EvaluationTests(unittest.TestCase):
    def test_whole_event_bootstrap_and_seed_pairing(self):
        values=np.array([[.2,.1,.3,.2],[-.2,-.1,-.3,-.2],[.01,.01,.01,.01]])
        ci=e.paired_interval(values)
        np.testing.assert_allclose(ci,e.paired_interval(values[:,::-1]))
        self.assertLess(ci[0],0);self.assertGreater(ci[1],0)
        wider=e.paired_interval(values,.975)
        self.assertLessEqual(wider[0],ci[0]);self.assertGreaterEqual(wider[1],ci[1])
        np.testing.assert_allclose(e.paired_interval(np.ones((1,4))*.125),[.125,.125])
        with self.assertRaises(AssertionError):e.paired_interval(np.empty((0,4)))

    def test_common_coprimary_support_and_absolute_differences(self):
        entries=[dict(id=f'{arm}_{seed}',arm=arm,seed=seed,endpoint='primary600')
                 for arm in ['D','U','A'] for seed in e.c.SEEDS]
        cells={}
        for entry in entries:
            for event in ['water','dry']:
                # Every water event has a 10-pixel union; the A dry union is
                # undefined. Both co-primary contrasts must drop that event.
                value=np.array([8 if entry['arm']=='D' else 7,0,2 if entry['arm']=='D' else 3,5])
                if event=='dry':value=np.array([0,0 if entry['arm']=='A' else 1,0,15])
                cells['synthetic',entry['id'],event]=value
        rows=e.contrast_rows(cells,entries)
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(row['events'],1);self.assertEqual(row['undefined_events'],1)
            self.assertEqual(json.loads(row['included_events']),['water'])
            self.assertAlmostEqual(row['D_mean'],.8);self.assertAlmostEqual(row['baseline_mean'],.7)
            self.assertAlmostEqual(row['difference'],.1)
            self.assertAlmostEqual(row['co_primary_bonferroni975_low'],.1)
        for key in cells:cells[key]=np.array([0,0,0,20])
        rows=e.contrast_rows(cells,entries)
        self.assertTrue(all(row['status']=='UNDEFINED_NO_COMMON_EVENT_UNION' for row in rows))
        self.assertTrue(all(row['difference'] is None for row in rows))

    def test_alias_order_missing_and_cycles(self):
        entries=[dict(id='b',alias_of='a'),dict(id='a',alias_of='base'),dict(id='base')]
        self.assertEqual([x['id'] for x in e.forward_order(entries)],['base','a','b'])
        self.assertEqual(e.alias_root('b',{x['id']:x for x in entries}),'base')
        for bad in [[dict(id='a',alias_of='b')],
                    [dict(id='a',alias_of='b'),dict(id='b',alias_of='a')],
                    [dict(id='a'),dict(id='a')]]:
            with self.assertRaises(ValueError):e.forward_order(bad)

    def test_synthetic_single_pass_end_to_end(self):
        import torch
        torch.set_num_threads(2)
        v10=Path(__file__).resolve().parents[2]/'research_v10_gpu_20261013T183509Z/code'
        sys.path.insert(0,str(v10))
        class Adapter:
            @staticmethod
            def build(graph):return torch.nn.Conv2d(6,2,1)
        class Reference:
            @staticmethod
            def reflectance(x):return np.asarray(x,dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=np.ones((6,2,2),np.float32)*.1
            image[0]=np.array([[1,-1],[-1,-1]])
            mask=np.array([[1,0],[0,-1]],np.int8)
            sample=root/'synthetic.npz';np.savez(sample,image=image,mask=mask,cloud=np.empty(0,bool))
            membership=root/'windows.json'
            membership.write_text(json.dumps(dict(windows=[dict(path=str(sample),dataset='synthetic',event='event_a',name='tile',omit=False)])))
            entries=[]
            for arm in ['D','U','A']:
                model=Adapter.build('toy')
                with torch.no_grad():
                    model.weight.zero_();model.bias.zero_()
                    if arm=='D':model.weight[1,0,0,0]=1
                    else:model.bias[1]=-1
                checkpoint=root/f'{arm}.pt'
                # Real checkpoints may also carry optimizer payload: never call
                # full file length deployed model bytes.
                torch.save(dict(model=model.state_dict(),optimizer={'unused':torch.zeros(100)}),checkpoint)
                for seed in e.c.SEEDS:
                    entry=dict(id=f'{arm}_{seed}',arm=arm,graph='toy',seed=seed,step=600,
                               endpoint='primary600',checkpoint=str(checkpoint),sha256=e.c.sha(checkpoint),
                               quantized=False,precision='FP32')
                    if seed!=42:entry['alias_of']=f'{arm}_42'
                    entries.append(entry)
            entries.append(dict(entries[0],id='Q0_tau_D_42',arm='Q0_tau_D',endpoint='source_threshold_control',
                                alias_of='D_42',threshold=.25))
            # Reverse order to exercise alias resolution before its base appears.
            freeze=dict(outcome='completed',models=list(reversed(entries)),membership=[dict(path=str(membership),sha256=e.c.sha(membership))])
            (root/'final_evaluation_freeze.json').write_text(json.dumps(freeze))
            with mock.patch.object(e.c,'ROOT',root),mock.patch.object(e.c,'setup',return_value=(torch,Adapter,Reference)):
                e.evaluate()
                with self.assertRaises(AssertionError):e.evaluate()
            accesses=(root/'label_access.jsonl').read_text().strip().splitlines()
            self.assertEqual(len(accesses),1)
            contrasts=list(csv.DictReader((root/'contrasts.csv').open()))
            self.assertEqual(len(contrasts),2)
            self.assertTrue(all(float(row['difference'])==1 for row in contrasts))
            costs={row['model']:row for row in csv.DictReader((root/'evaluation/inference_cost.csv').open())}
            self.assertEqual(int(costs['D_42']['model_state_tensor_bytes']),56)
            self.assertGreater(int(costs['D_42']['training_checkpoint_bytes']),56)
            self.assertEqual(int(costs['D_42']['convolution_MACs_first_canonical_tile']),48)
            self.assertEqual(costs['D_64']['forward_model'],'D_42')
            self.assertGreater(float(costs['D_64']['mean_seconds']),float(costs['D_64']['mean_shared_forward_seconds']))
            receipt=json.loads((root/'evaluation/receipt.json').read_text())
            self.assertEqual(receipt['models'],3);self.assertEqual(receipt['windows'],1)


if __name__=='__main__':
    started,cpu=time.monotonic(),time.process_time()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationTests))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
                 failures=len(result.failures),errors=len(result.errors),wall_seconds=time.monotonic()-started,
                 process_cpu_seconds=time.process_time()-cpu,allocated_threads=2,
                 real_target_labels_opened=False,synthetic_only=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    print(json.dumps(receipt))
    if os.environ.get('V11_EVALUATION_TEST_RECEIPT'):
        Path(os.environ['V11_EVALUATION_TEST_RECEIPT']).write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
