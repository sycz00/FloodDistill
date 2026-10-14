"""Bounded CPU checks of the two science changes in this recovery snapshot."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['V10_COMPUTE_DEVICE']='cpu'
os.environ['V10_WORKER_CPU_THREADS']='2'
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common10 as c
torch,_,_,_=c.setup(2)
from quant10 import prepare
from models.compression_quant import FixedQATConv2d,precision_map


class RecoveryScienceTests(unittest.TestCase):
    def test_empty_reference_support_is_not_a_dry_scene(self):
        import numpy as np
        from evaluate10 import is_dry_reference
        self.assertFalse(is_dry_reference(np.full((4,4),-1)))
        self.assertTrue(is_dry_reference(np.array([[0,-1],[0,0]])))
        self.assertFalse(is_dry_reference(np.array([[0,-1],[1,0]])))

    def test_r1_fp_head_precedence_and_strict_int8(self):
        torch.manual_seed(1729)
        model=c.build('R1').eval()
        image=torch.randn(2,6,8,8)
        original=c.state_hash(model.state_dict())
        with torch.no_grad():expected=model(image)
        for precision in ['W4A4','W4A8']:
            quant=prepare(copy.deepcopy(model),[dict(image=image)],precision)
            self.assertFalse(any(isinstance(m,FixedQATConv2d) for m in quant.modules()))
            self.assertEqual(precision_map(quant)['status'],'floating_reference')
            self.assertEqual(c.state_hash(quant.state_dict()),original)
            with torch.no_grad():torch.testing.assert_close(quant(image),expected,rtol=0,atol=0)
        strict=prepare(copy.deepcopy(model),[dict(image=image)],'W8A8')
        self.assertEqual(len([m for m in strict.modules() if isinstance(m,FixedQATConv2d)]),1)
        self.assertFalse(any(type(m)==torch.nn.Conv2d for m in strict.modules()))
        self.assertEqual(c.state_hash(model.state_dict()),original)

    def test_r4_precision_map_is_unchanged(self):
        torch.manual_seed(1729)
        model=c.build('R4').eval();image=torch.randn(1,6,32,32)
        quant=prepare(model,[dict(image=image)],'W4A4')
        convs=[m for m in quant.modules() if isinstance(m,torch.nn.Conv2d)]
        self.assertIsInstance(convs[0],FixedQATConv2d)
        self.assertEqual((convs[0].weight_bits,convs[0].activation_bits),(8,8))
        self.assertIs(type(convs[-1]),torch.nn.Conv2d)
        self.assertTrue(all((m.weight_bits,m.activation_bits)==(4,4) for m in convs[1:-1]))

    def test_early_parent_receipt_protects_consumed_checkpoint(self):
        import controls10
        torch.manual_seed(3)
        q=prepare(c.build('R4').eval(),[dict(image=torch.randn(1,6,32,32))],'W4A4')
        parent=dict(id='test_parent',graph='R4',method='lkd',seed=42,sha256='fixed_parent')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def fake_source(entry,model=None):
                metric=dict(checkpoint_sha256=entry['sha256'],equal_event_water_IoU=.8)
                c.save(root/'source_metrics'/(entry['id']+'.json'),metric)
                return metric
            with patch.object(c,'ROOT',root),patch.object(c,'load',return_value=None),patch('quant10.prepare',return_value=q),patch('evaluate10.source',side_effect=fake_source):
                entry,_=controls10.prepare_qat_parent(parent,[],'W4A4')
                before=Path(entry['checkpoint']).stat().st_mtime_ns
                again,_=controls10.prepare_qat_parent(parent,[],'W4A4')
                self.assertEqual(entry,again)
                self.assertEqual(before,Path(entry['checkpoint']).stat().st_mtime_ns)
                with torch.no_grad():next(q.parameters()).add_(.01)
                with self.assertRaisesRegex(AssertionError,'immutable V11 input protected'):
                    controls10.prepare_qat_parent(parent,[],'W4A4')
                self.assertEqual(c.sha(entry['checkpoint']),entry['sha256'])


if __name__=='__main__':unittest.main()
