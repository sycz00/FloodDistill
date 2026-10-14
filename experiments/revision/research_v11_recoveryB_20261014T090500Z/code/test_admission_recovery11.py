"""Whole-matrix CPU and wall admission with measured thread allocations."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import action11 as a


class AdmissionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = dict(cap_core_hours=64, max_parallel_jobs=4,
            max_owned_threads=8, blocks={'collection': {'reserve_core_hours': 1.4}},
            jobs={f'train_{i}': dict(training_rule='D', threads=2,
                max_seconds=900, block='core') for i in range(28)})
        self.manifest['jobs']['30_final_evaluation'] = dict(threads=4,
            max_seconds=21600, block='evaluation')
        self.profile = dict(threads=2,
            conservative_28_run_core_hours=100*28*2/3600)
        self.state = dict(first_compute_at=1000, execution_deadline=5000,
            absolute_deadline=30000, charges={}, active={}, jobs={})

    def tearDown(self):
        self.tmp.cleanup()

    def admission(self):
        values = {'manifest.json': self.manifest, 'profile.json': self.profile,
                  'state.json': self.state, 'normal_windows.json': {'windows': []},
                  'sealed_windows.json': {'rows': []}}
        with patch.object(a.c, 'ROOT', self.root), patch.object(a.c, 'V9', self.root), \
             patch.object(a.c, 'read', side_effect=lambda p,*args: values[Path(p).name]), \
             patch.object(a.c, 'sha', return_value='fixed'), \
             patch.object(a.time, 'time', return_value=1000), \
             patch.object(a, 'evaluation_profile', return_value={
                 'conservative_seconds_per_window': 1, 'warmup_seconds': 5}):
            a.admission()
        return a.c.read(self.root/'profile_admission.json')

    def test_parallel_fits_but_serial_does_not_reset_deadline(self):
        result = self.admission()
        self.assertTrue(result['admitted'], result['reason'])
        self.assertEqual(result['parallel_training_jobs'], 4)
        self.assertEqual(result['training_wall_seconds_bound'], 1050)
        self.manifest['max_parallel_jobs'] = 1
        self.assertFalse(self.admission()['admitted'])

    def test_prior_cost_and_whole_matrix_are_counted(self):
        self.state['charges']['prior'] = {'reserved_core_seconds': 63*3600}
        result = self.admission()
        self.assertFalse(result['admitted'])
        self.assertIn('cumulative CPU', result['reason'])

    def test_profile_allocation_must_match_every_arm(self):
        self.profile['threads'] = 4
        with self.assertRaisesRegex(ValueError, 'every paired arm'):
            self.admission()


if __name__ == '__main__':
    unittest.main()
