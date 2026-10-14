"""Source-profile dynamic readout admission; uses only temporary mocked ledgers."""
import contextlib
import math
import unittest
from unittest.mock import patch
from test_worker_recoveryB10 import RecoveryTests,Ledger
import worker_gpu_fast10 as w

class ReadoutTests(RecoveryTests):
    def prepare_profile(self,seconds=(100,200,300)):
        w.c.save(self.root/'readout_models.json',{'models':[{'id':'fixed'}]})
        w.c.save(self.root/'operating_points.json',{'fixed':{'tau':0}})
        profile=dict(outcome='completed',source_only=True,slot_seconds=list(seconds),frozen_model_manifest_sha256=w.c.sha(self.root/'readout_models.json'),frozen_settings_sha256=w.c.sha(self.root/'operating_points.json'))
        w.c.save(self.root/'final_readout_profile.json',profile)
        stage=self.root/'stage_results/profile.json';w.c.save(stage,dict(outcome='completed',profile_sha256=w.c.sha(self.root/'final_readout_profile.json')))
        w.c.save(self.root/'receipts/profile.json',dict(outcome='completed',result_sha256=w.c.sha(stage)))
        w.c.save(self.root/'final_evaluation_freeze.json',dict(model_manifest_sha256=profile['frozen_model_manifest_sha256'],source_settings_sha256=profile['frozen_settings_sha256']))
        w.c.save(self.root/'GPU_budget_policy.json',{'cumulative_cap_seconds':1000})
        jobs={str(i):dict(slot=i,dynamic_readout_profile='final_readout_profile.json',max_seconds=2400) for i in range(3)}
        state=dict(deadline=4000,jobs={name:dict(status='PENDING') for name in jobs})
        manifest={'scheduler':{'final_readout_jobs':list(jobs),'final_readout_profile_job':'profile'}}
        return jobs,state,manifest
    def test_profile_cannot_change_modelset_or_nonfinite_ceiling(self):
        self.prepare_profile();w.readout_profile(self.root/'final_readout_profile.json')
        w.c.save(self.root/'readout_models.json',{'models':[]})
        with self.assertRaises(AssertionError):w.readout_profile(self.root/'final_readout_profile.json')
        self.prepare_profile((100,math.inf,300))
        with self.assertRaises(AssertionError):w.readout_profile(self.root/'final_readout_profile.json')
    def test_full_block_reserved_before_first_slot(self):
        jobs,state,manifest=self.prepare_profile();ledger=Ledger(300,{w.protected_key():dict(remaining_seconds=200)})
        with patch.object(w.gq,'Ledger',side_effect=lambda:contextlib.nullcontext(ledger)),patch.object(w.time,'time',return_value=1000):
            job,reason=w.final_readout_job('0',jobs['0'],state,jobs,manifest)
        self.assertIsNone(reason);self.assertEqual(job['max_seconds'],100)
        self.assertEqual(ledger.reserved(),600);self.assertEqual(ledger.spend(),300)
    def test_block_budget_failure_admits_no_partial_slot(self):
        jobs,state,manifest=self.prepare_profile();ledger=Ledger(500,{w.protected_key():dict(remaining_seconds=200)})
        with patch.object(w.gq,'Ledger',side_effect=lambda:contextlib.nullcontext(ledger)),patch.object(w.time,'time',return_value=1000):
            job,reason=w.final_readout_job('0',jobs['0'],state,jobs,manifest)
        self.assertIsNone(job);self.assertIn('whole',reason);self.assertEqual(ledger.reserved(),200)
    def test_second_slot_does_not_double_reserve_active_slot(self):
        jobs,state,manifest=self.prepare_profile();ledger=Ledger(300,{w.protected_key():dict(remaining_seconds=500),'active':dict(remaining_seconds=100)})
        state['jobs']['0'].update(status='RUNNING',started_at=990,effective_max_seconds=100)
        with patch.object(w.gq,'Ledger',side_effect=lambda:contextlib.nullcontext(ledger)),patch.object(w.time,'time',return_value=1000):
            job,reason=w.final_readout_job('1',jobs['1'],state,jobs,manifest)
        self.assertIsNone(reason);self.assertEqual(job['max_seconds'],200);self.assertEqual(ledger.reserved(),600)
    def test_cutoff_failure_admits_no_partial_slot(self):
        jobs,state,manifest=self.prepare_profile();state['deadline']=3399
        with patch.object(w.time,'time',return_value=1000):job,reason=w.final_readout_job('0',jobs['0'],state,jobs,manifest)
        self.assertIsNone(job);self.assertIn('cutoff',reason)
    def test_transfer_remaining_predecessor_cohort_without_double_reserve(self):
        ledger=Ledger(100,{'old:cohort:R4':dict(remaining_seconds=200)})
        w.initialize_protection(ledger,{'scheduler':{'cohort_reservation_provider':'old','protected_gpu_seconds':0}},1000)
        self.assertEqual(ledger.reserved(),200);self.assertNotIn('old:cohort:R4',ledger.data['reservations'])
        self.assertIn(str(self.root)+':cohort:R4',ledger.data['reservations'])

if __name__=='__main__':unittest.main()
