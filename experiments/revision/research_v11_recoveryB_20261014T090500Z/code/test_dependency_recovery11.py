"""Recovery accepts exact early parents, but never incomplete cache receipts."""
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import common11 as c
import dependency11 as d


class RecoveryDependencies(unittest.TestCase):
    def test_source_guard_allows_cost_profile_but_blocks_results(self):
        script = '''
import tempfile
from pathlib import Path
from unittest.mock import patch
import common11 as c
with tempfile.TemporaryDirectory() as folder:
    c.ROOT=Path(folder)/'v11'; c.V9=Path(folder)/'v9'; c.V10=Path(folder)/'v10'
    c.ROOT.mkdir()
    c.save(c.V9/'normal_windows.json',{'windows':[]})
    c.save(c.V9/'sealed_windows.json',{'rows':[]})
    c.source_access_guard()
    c.save(c.ROOT/'evaluation_cost_profile.json',{'source_only':True})
    for p in [c.ROOT/'evaluation/results.json', c.V10/'metrics/a.json', c.ROOT/'contrasts.csv']:
        try: open(p,'r')
        except PermissionError: pass
        else: raise AssertionError('Target result guard did not block '+str(p))
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_early_core_receipt_and_cache_atomicity(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(c, 'V10', Path(folder)):
            root = Path(folder)
            for name in ['teacher_logit_cache_manifest.json', 'core_qat_parent_preparation.json']:
                c.save(root/name, {'outcome': 'completed'})
            for name in ['01_teacher_cache', '02_core_R4_parents']:
                stage = root/'stage_results'/(name+'.json')
                c.save(stage, {'outcome': 'completed'})
                c.save(root/'receipts'/(name+'.json'),
                       {'outcome': 'completed', 'result_sha256': c.sha(stage)})
            for seed in c.SEEDS:
                c.save(root/'qat_parents'/f'R4_W4A4_s{seed}.json', {'parent': seed})
            self.assertEqual(d.v10_ready('repair'), [])
            self.assertFalse((root/'compression_control_selection.json').exists())
            c.save(root/'stage_results/01_teacher_cache.json', {'outcome': 'incomplete'})
            self.assertTrue(any('hash verification' in x for x in d.v10_ready('repair')))


if __name__ == '__main__':
    unittest.main()
