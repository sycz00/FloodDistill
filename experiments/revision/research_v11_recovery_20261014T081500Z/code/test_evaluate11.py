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

V10=Path(__file__).resolve().parents[2]/'research_v10_recoveryB_20261014T084000Z/code'
sys.path.insert(0,str(V10))


class EvaluationTests(unittest.TestCase):
    def test_indexed_metrics_exact_reference_parity(self):
        from evaluate10 import counts,hist,probability_bins
        rng=np.random.default_rng(238)
        entries=e.forward_order([dict(id='view',alias_of='base',threshold=.125,fp_reference='ref'),
                                dict(id='base',fp_reference='ref'),dict(id='ref')])
        for kind in ['mixed','dry','invalid','all_water']:
            y=rng.choice([-1,0,1],size=(37,41),p=[.1,.6,.3]).astype(np.int8)
            if kind=='dry':y[y==1]=0
            if kind=='invalid':y.fill(-1)
            if kind=='all_water':y.fill(1)
            regions={'all':np.ones(y.shape,bool),'none':np.zeros(y.shape,bool),'random':rng.random(y.shape)>.7}
            support=e.prepare_metric_support(y,regions)
            margins={name:rng.normal(size=y.shape).astype(np.float32) for name in ['base','ref']}
            result,timing=e.tile_readout(entries,margins,y,support)
            self.assertEqual(set(timing),{'view','base','ref'})
            valid=y!=-1;ref=margins['ref']>0
            for entry in entries:
                name=entry['id'];m=margins[entry.get('alias_of',name)];p=m>entry.get('threshold',0);actual=result[name]
                np.testing.assert_array_equal(actual['counts'],counts(p,y))
                for region,mask in regions.items():np.testing.assert_array_equal(actual['strata'][region],counts(p,y,mask))
                np.testing.assert_array_equal(actual['histogram'],hist(m[valid],y[valid].astype(np.int64)))
                np.testing.assert_array_equal(actual['calibration'],probability_bins(m,y,np.ones(y.shape,bool)))
                if entry.get('fp_reference'):
                    expected=[np.count_nonzero(valid&(y==0)&ref&~p),np.count_nonzero(valid&(y==1)&~ref&p),
                              np.count_nonzero(valid&(y==0)&~ref&p),np.count_nonzero(valid&(y==1)&ref&~p)]
                    np.testing.assert_array_equal(actual['harm'],expected)
                self.assertEqual(actual['dry'] is not None,kind=='dry')
                if kind=='dry':
                    self.assertEqual(actual['dry']['false_water_pixels'],int(np.count_nonzero(p&valid)))
                    self.assertEqual(actual['dry']['valid_pixels'],int(valid.sum()))

    def test_fixed_batch_forward_native_geometry_and_frozen_quantizer(self):
        import torch
        torch.set_num_threads(2)
        sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
        from models.compression_quant import FixedQATConv2d
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            conv=torch.nn.Conv2d(6,2,3,padding=1)
            q=FixedQATConv2d(conv,torch.tensor([.05]),torch.tensor([0.]),torch.ones(2)*.02,
                             weight_bits=4,activation_bits=8).eval().requires_grad_(False)
            model=torch.nn.Sequential(q).eval().requires_grad_(False)
            images=np.random.default_rng(1729).normal(size=(4,6,512,512)).astype(np.float32)
            before={k:v.clone() for k,v in model.state_dict().items()}
            batch,seconds=e.forward_batch({'quant':model},images,torch)
            singles=np.concatenate([e.forward_batch({'quant':model},x[None],torch)[0]['quant'] for x in images])
            self.assertEqual(batch['quant'].shape,(4,512,512));self.assertGreater(seconds['quant'],0)
            np.testing.assert_allclose(batch['quant'],singles,rtol=1e-5,atol=1e-6)
            np.testing.assert_array_equal(batch['quant']>0,singles>0)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            with self.assertRaises(AssertionError):e.forward_batch({'quant':model},np.zeros((5,6,2,2),np.float32),torch)

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
        v10=Path(__file__).resolve().parents[2]/'research_v10_recoveryB_20261014T084000Z/code'
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
            # Five distinct inputs exercise a full batch and a one-tile tail.
            windows=[]
            for i in range(5):
                path=root/f'synthetic_{i}.npz';np.savez(path,image=image,mask=mask,cloud=np.empty(0,bool))
                windows.append(dict(path=str(path),dataset='synthetic',event='event_a',name=f'tile{i}',omit=False))
            membership.write_text(json.dumps(dict(windows=windows)))
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
            self.assertEqual(len(accesses),5)
            self.assertEqual(len({json.loads(line)['path'] for line in accesses}),5)
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
            self.assertEqual(receipt['models'],3);self.assertEqual(receipt['windows'],5)
            self.assertEqual(receipt['inference_batch_size'],4)


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
