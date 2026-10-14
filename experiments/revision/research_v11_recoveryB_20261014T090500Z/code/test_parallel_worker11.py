"""Focused finite-block CPU parallelism tests; no scientific commands launched."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import worker11 as w


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for d in ['logs','running','receipts']:(self.root/d).mkdir()
        self.jobs={f't{i}':dict(threads=2,max_seconds=600,block='core',training_rule='fixed',deps=[],argv=['/usr/bin/true'],outputs=[{'path':str(self.root/f't{i}.json')}]) for i in range(28)}
        self.jobs['select']=dict(threads=8,max_seconds=300,block='core',deps=list(self.jobs),argv=['/usr/bin/true'],outputs=[{'path':str(self.root/'select.json')}])
        self.jobs['eval']=dict(threads=8,max_seconds=600,block='evaluation',deps=['select'],argv=['/usr/bin/true'],outputs=[{'path':str(self.root/'eval.json')}])
        self.manifest=dict(jobs=self.jobs,max_parallel_jobs=4,max_owned_threads=8,cap_core_hours=64,blocks={'core':dict(job_ids=list(self.jobs)[:-1],reserve_core_hours=10),'evaluation':dict(job_ids=['eval'],reserve_core_hours=2,protected=True)},python='/usr/bin/python3',cwd=str(self.root))
        self.state=w.fresh_state(self.manifest,1000);self.state['manifest_sha256']='frozen'
        self.telemetry=dict(known=True,host_threads=48,safe_threads=40,free_ram_gib=64,load1=0,external_reserved_threads=32)
    def tearDown(self):self.temp.cleanup()
    def dependency(self,root,manifest,job,state):
        return {'status':'READY' if all(state['jobs'][n]['status']=='COMPLETED' for n in job.get('deps',[])) else 'WAITING_DEPENDENCY'}
    def tick(self,at):
        with patch.object(w.os,'sched_getaffinity',return_value=set(range(48))):
            w.tick(self.root,self.manifest,self.state,at,launch=False,telemetry_fn=lambda _:self.telemetry,dependency_fn=self.dependency)
    def test_four_two_thread_jobs_disjoint_and_eight_thread_cap(self):
        for at in range(1000,1004):self.tick(at)
        self.assertEqual(len(self.state['active']),4)
        cpus=[cpu for n in self.state['active'] for cpu in self.state['jobs'][n]['cpus']]
        self.assertEqual(len(cpus),8);self.assertEqual(len(set(cpus)),8)
        self.assertEqual(w.account(self.state,1004)['hardware_occupied_threads'],8)
        self.state['next_check_at']=1004;self.tick(1004);self.assertEqual(len(self.state['active']),4)
    def test_next_independent_launch_does_not_wait600(self):
        self.tick(1000);self.assertEqual(self.state['next_check_at'],1000)
        self.tick(1001);self.assertEqual(len(self.state['active']),2)
    def test_serial_selection_and_evaluation_cannot_overlap_training(self):
        self.tick(1000)
        self.assertFalse(w.parallel_capacity(self.manifest,self.state,self.jobs['select']))
        self.assertFalse(w.parallel_capacity(self.manifest,self.state,self.jobs['eval']))
    def test_whole_block_wall_28_runs_four_slots_then_selection(self):
        self.assertEqual(w.block_wall_seconds(self.manifest,self.state,self.manifest['blocks']['core'],1000),7*600+300)
        self.manifest['max_parallel_jobs']=2
        for n in self.jobs:
            if self.jobs[n].get('training_rule'):self.jobs[n]['threads']=4
        self.assertEqual(w.block_wall_seconds(self.manifest,self.state,self.manifest['blocks']['core'],1000),14*600+300)
    def test_core_seconds_not_discounted_for_concurrency(self):
        self.tick(1000);before=w.account(self.state,1000)['total_committed_core_seconds']
        self.tick(1001);after=w.account(self.state,1001)['total_committed_core_seconds']
        self.assertEqual(before,after)
    def test_internal_dependencies_force_serial_training_chain(self):
        for i in range(1,28):self.jobs[f't{i}']['deps']=[f't{i-1}']
        self.assertEqual(w.block_wall_seconds(self.manifest,self.state,self.manifest['blocks']['core'],1000),28*600+300)
    def test_serial_default_and_external_host_reservations_preserved(self):
        self.manifest.pop('max_parallel_jobs');self.tick(1000)
        self.state['next_check_at']=1001;self.tick(1001);self.assertEqual(len(self.state['active']),1)
        self.telemetry['external_reserved_threads']=39
        self.state['next_check_at']=1002;self.tick(1002);self.assertEqual(len(self.state['active']),1)
    def test_remaining_active_jobs_count_in_wall_schedule(self):
        for at in range(1000,1004):self.tick(at)
        remaining=w.block_wall_seconds(self.manifest,self.state,self.manifest['blocks']['core'],1100)
        self.assertGreaterEqual(remaining,7*600+300-100)
        self.assertLessEqual(remaining,7*600+300-97)

if __name__=='__main__':unittest.main()
