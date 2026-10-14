"""CPU-only snapshot path and historical architecture compatibility regression."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',V10_COMPUTE_DEVICE='cpu',V10_WORKER_CPU_THREADS='2')
import pathlib,json,unittest
import common10 as c
torch,_,_,_=c.setup(2)

class ReferencePathTest(unittest.TestCase):
    def test_micro_uses_frozen_sibling_rungs_and_historical_state(self):
        import rungs
        receipt=c.read(c.ROOT/'smoke_receipts/R4_micro_CE_s42.json')
        self.assertEqual(c.sha(receipt['best_path']),receipt['best_sha256'])
        payload=torch.load(receipt['best_path'],map_location='cpu',weights_only=False)
        model=c.build('R4_micro').eval();reference=rungs.ContextConv(widths=(4,4,6,8),dec=4).eval()
        model.load_state_dict(payload['model'],strict=True);reference.load_state_dict(payload['model'],strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()),receipt['parameters'])
        image=torch.randn(1,6,32,32)
        with torch.no_grad():
            actual=model(image);expected=reference(image)
        self.assertEqual(tuple(actual.shape),(1,2,32,32))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
