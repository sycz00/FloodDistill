"""Measured limits reserve complete blocks and survive controller recovery."""
import copy
import tempfile
import unittest
from pathlib import Path
import worker11 as w


class MeasuredReservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = dict(cap_core_hours=64, measured_reservations=dict(
            job='profile', receipt='profile.json', evaluation_job='eval'), jobs={
            'profile': dict(threads=1, max_seconds=60, block='smoke'),
            'D': dict(threads=8, max_seconds=640, block='core', training_rule='D'),
            'U': dict(threads=8, max_seconds=640, block='core', training_rule='U'),
            'eval': dict(threads=8, max_seconds=21600, block='evaluation')}, blocks={
            'smoke': dict(job_ids=['profile'], reserve_core_hours=1),
            'core': dict(job_ids=['D','U'], reserve_core_hours=3),
            'evaluation': dict(job_ids=['eval'], reserve_core_hours=48, protected=True)})
        self.state = w.fresh_state(self.manifest, 1000)
        self.state['jobs']['profile']['status'] = 'COMPLETED'
        w.q.atomic_json(self.root/'profile.json', dict(admitted=True,
            per_run_seconds_bound=120.2, evaluation_seconds_bound=3600.2))

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_block_and_idempotent_active_recovery(self):
        original = copy.deepcopy(self.manifest)
        effective = w.measured_manifest(self.root, self.manifest, self.state)
        self.assertEqual(self.manifest, original)
        self.assertEqual(effective['jobs']['D']['max_seconds'], 151)
        self.assertEqual(self.state['blocks']['evaluation']['remaining_core_seconds'], 3631*8)
        self.assertIsNotNone(w.allocate(effective, self.state, 'D', 1001))
        self.assertEqual(self.state['active']['D']['max_seconds'], 151)
        self.assertEqual(self.state['blocks']['core']['remaining_core_seconds'], 151*8)
        before = w.account(self.state, 1002)
        resumed = w.measured_manifest(self.root, self.manifest, self.state)
        self.assertEqual(resumed, effective)
        self.assertEqual(w.account(self.state, 1002), before)
        self.assertLess(before['total_committed_core_seconds'], 64*3600)

    def test_profile_is_bound_once(self):
        w.measured_manifest(self.root, self.manifest, self.state)
        w.q.atomic_json(self.root/'profile.json', dict(admitted=True,
            per_run_seconds_bound=121, evaluation_seconds_bound=3600.2))
        with self.assertRaisesRegex(ValueError, 'profile changed'):
            w.measured_manifest(self.root, self.manifest, self.state)
        w.q.atomic_json(self.root/'profile.json', dict(admitted=False))
        with self.assertRaisesRegex(ValueError, 'profile changed'):
            w.measured_manifest(self.root, self.manifest, self.state)
        (self.root/'profile.json').unlink()
        with self.assertRaisesRegex(ValueError, 'profile changed'):
            w.measured_manifest(self.root, self.manifest, self.state)

    def test_reject_overmaximum_and_unapproved_profile(self):
        w.q.atomic_json(self.root/'profile.json', dict(admitted=True,
            per_run_seconds_bound=630, evaluation_seconds_bound=3600))
        with self.assertRaisesRegex(ValueError, 'frozen maximum'):
            w.measured_manifest(self.root, self.manifest, self.state)
        w.q.atomic_json(self.root/'profile.json', dict(admitted=False))
        self.assertIs(w.measured_manifest(self.root, self.manifest, self.state), self.manifest)
        self.assertNotIn('runtime_limits', self.state)


if __name__ == '__main__':
    unittest.main()
