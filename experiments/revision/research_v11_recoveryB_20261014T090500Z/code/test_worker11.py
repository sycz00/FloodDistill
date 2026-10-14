"""Mock-clock checks for the V11-only admission and recovery changes."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import worker11 as w


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for directory in ('logs', 'receipts', 'running', 'stage_results'):
            (self.root / directory).mkdir()
        self.manifest = dict(python='/usr/bin/python3', cwd=str(self.root),
            cap_core_hours=64, blocks={
                'core': dict(job_ids=['D', 'U'], reserve_core_hours=1),
                'evaluation': dict(job_ids=['eval'], reserve_core_hours=14, protected=True),
                'collection': dict(job_ids=[], reserve_core_hours=4, protected=True)}, jobs={
                name: dict(argv=['/usr/bin/true'], deps=[] if name == 'D' else ['D'],
                           threads=2, max_seconds=300, block='evaluation' if name == 'eval' else 'core',
                           compute_heavy=True, outputs=[dict(path=str(self.root / (name + '.json')))])
                for name in ['D', 'U', 'eval']})
        self.state = w.fresh_state(self.manifest, 1000)
        self.state['manifest_sha256'] = 'frozen'
        self.idle = dict(known=True, host_threads=48, safe_threads=40,
                         free_ram_gib=64, load1=0, external_reserved_threads=24)

    def tearDown(self):
        self.directory.cleanup()

    def tick(self, at, telemetry=None, dependency=None):
        return w.tick(self.root, self.manifest, self.state, at, launch=False,
                      telemetry_fn=telemetry or (lambda _: self.idle),
                      dependency_fn=dependency or (lambda *_: {'status': 'READY'}))

    def test_busy_600_second_recheck_then_idle_admission(self):
        calls = []
        def busy(_):
            calls.append(True)
            return dict(self.idle, external_reserved_threads=40)
        self.tick(1000, busy)
        self.assertEqual(self.state['phase'], 'WAITING_RESOURCES')
        self.assertEqual(self.state['next_check_at'], 1600)
        self.assertEqual(w.account(self.state, 1000)['hardware_occupied_threads'], 0)
        self.tick(1599)
        self.assertFalse(self.state['active'])
        self.tick(1600)
        self.assertEqual(list(self.state['active']), ['D'])

    def test_unknown_telemetry_waits(self):
        self.tick(1000, lambda _: {'known': False})
        self.assertEqual(self.state['phase'], 'WAITING_RESOURCES')
        self.assertFalse(self.state['active'])

    def test_lock_then_recheck_race_never_admits(self):
        readings = iter([self.idle] + [dict(self.idle, external_reserved_threads=40)] * 3)
        self.tick(1000, lambda _: next(readings))
        self.assertFalse(self.state['active'])
        self.assertNotIn('core', self.state['blocks'])

    def test_changed_dependency_at_locked_recheck_waits(self):
        counter = [0]
        def checker(*_):
            counter[0] += 1
            return {'status': 'READY' if counter[0] == 1 else 'WAITING_DEPENDENCY'}
        self.tick(1000, dependency=checker)
        self.assertFalse(self.state['active'])

    def test_block_to_active_is_not_double_reserved(self):
        self.assertTrue(w.reserve_block(self.manifest, self.state, 'core', 1000))
        before = w.account(self.state, 1000)['total_committed_core_seconds']
        token = w.allocate(self.manifest, self.state, 'D', 1000)
        self.assertIsNotNone(token)
        self.assertEqual(before, w.account(self.state, 1100)['total_committed_core_seconds'])
        self.assertIsNone(w.allocate(self.manifest, self.state, 'D', 1100))
        w.charge(self.state, 'D', dict(outcome='COMPLETED', seconds=100,
                                      measured_process_cpu_seconds=23), 1100)
        after = w.account(self.state, 1100)
        self.assertEqual(after['completed_charged_core_seconds'], 200)
        self.assertEqual(after['total_committed_core_seconds'], before - 400)
        self.assertEqual(after['measured_process_cpu_seconds'], 23)

    def test_cumulative_charge_prevents_overbudget_new_block(self):
        self.state['charges']['old'] = dict(reserved_core_seconds=46 * 3600)
        self.assertFalse(w.reserve_block(self.manifest, self.state, 'core', 1000))
        self.assertEqual(w.account(self.state, 1000)['total_committed_core_seconds'], 64 * 3600)

    def test_restart_reattaches_surviving_guardian_no_duplicate(self):
        self.tick(1000)
        token = self.state['jobs']['D']['token']
        guardian = {'pid': 13, 'boot_id': 'boot', 'start_ticks': '19'}
        w.q.atomic_json(self.root / 'running/D.json', dict(token=token, guardian=guardian))
        old_deadline = self.state['execution_deadline']
        with patch.object(w.q, 'alive', side_effect=lambda identity: identity == guardian):
            resumed = copy.deepcopy(self.state)
            w.reconcile(self.root, self.manifest, resumed, 1500)
        self.assertEqual(resumed['jobs']['D']['status'], 'RUNNING')
        self.assertEqual(resumed['jobs']['D']['attempts'], 1)
        self.assertEqual(resumed['execution_deadline'], old_deadline)
        self.assertEqual(list(resumed['active']), ['D'])

    def test_pid_reuse_never_signals_foreign_process(self):
        with patch.object(w.q, 'alive', return_value=False), patch.object(w.os, 'killpg') as kill:
            w.owned_stop(dict(pid=12, start_ticks='old', boot_id='old'))
        kill.assert_not_called()

    def test_completed_receipt_reconciles_once(self):
        self.tick(1000)
        token = self.state['jobs']['D']['token']
        receipt = dict(token=token, outcome='COMPLETED', seconds=90, measured_process_cpu_seconds=20)
        w.q.atomic_json(w.receipt_path(self.root, 'D', token), receipt)
        w.reconcile(self.root, self.manifest, self.state, 1100)
        w.reconcile(self.root, self.manifest, self.state, 1200)
        self.assertEqual(self.state['jobs']['D']['status'], 'COMPLETED')
        self.assertEqual(len(self.state['charges']), 1)
        self.assertEqual(w.account(self.state, 1200)['completed_charged_core_seconds'], 180)

    def test_deadline_collects_partial_results(self):
        result = self.tick(self.state['prerequisite_deadline'])
        self.assertEqual(result, 'COLLECT')
        with patch.object(w.time, 'time', return_value=self.state['prerequisite_deadline']):
            w.collect(self.root, self.manifest, self.state, self.state['reason'])
        self.assertTrue(self.state['finished'])
        self.assertTrue(all(row['status'] == 'DEFERRED' for row in self.state['jobs'].values()))
        self.assertEqual(w.account(self.state, self.state['prerequisite_deadline'])['planned_reserved_remaining_core_seconds'], 0)

    def test_stop_collects_without_waiting_for_600_second_scan(self):
        self.state['next_check_at'] = 1600
        w.q.atomic_json(self.root / 'control.json', {'stop': True})
        self.assertEqual(self.tick(1001), 'COLLECT')

    def test_stop_waits_for_owned_guardian(self):
        self.tick(1000)
        guardian = dict(pid=1, boot_id='b', start_ticks='2')
        self.state['jobs']['D']['guardian'] = guardian
        w.q.atomic_json(self.root / 'control.json', {'stop': True})
        with patch.object(w.q, 'alive', side_effect=lambda identity: identity == guardian), patch.object(w, 'owned_stop') as stop:
            self.assertIsNone(self.tick(1010))
        stop.assert_called_once_with(guardian)
        self.assertEqual(self.state['phase'], 'STOPPING')

    def test_finished_v9_string_rows_release_configured_maximum(self):
        path = self.root / 'old.json'
        w.q.atomic_json(path, dict(finished=1234, phase='FINISHED_WITH_DEFERRALS', active={}, jobs={'old': 'DEFERRED'}))
        self.manifest['external_workers'] = [dict(name='old', state_path=str(path), max_threads=16)]
        self.assertEqual(w.external_allocation(self.manifest)[0], 0)
        w.q.atomic_json(path, dict(finished=False, jobs={'old': 'PENDING'}))
        self.assertEqual(w.external_allocation(self.manifest)[0], 16)

    def test_manifest_protects_complete_block_and_no_gpu(self):
        w.validate_manifest(self.manifest)
        self.manifest['jobs']['D']['device'] = 'cuda'
        with self.assertRaises(ValueError):
            w.validate_manifest(self.manifest)


if __name__ == '__main__':
    unittest.main()
