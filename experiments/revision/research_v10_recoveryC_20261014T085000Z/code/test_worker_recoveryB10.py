"""CPU-only mock tests for successor admission; no real GPU ledger mutations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import worker_gpu_fast10 as w


class Ledger:
    def __init__(self,spent=0,reservations=None):
        self.data={'reservations':reservations or {}};self.actual=spent
    def spend(self):return self.actual
    def reserved(self):return sum(x['remaining_seconds'] for x in self.data['reservations'].values())


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.root_patch=patch.object(w.c,'ROOT',self.root);self.root_patch.start()
    def tearDown(self):self.root_patch.stop();self.directory.cleanup()
    def test_cohort_reserves_all_remaining_seeds_once(self):
        ledger=Ledger(100);job=dict(cohort_group='R4',max_seconds=50,cohort_remaining_seconds=200)
        self.assertFalse(w.budget_fits(ledger,job,299));self.assertTrue(w.budget_fits(ledger,job,300))
        w.reserve_job(ledger,'seed1',job);self.assertEqual(ledger.reserved(),200)
        job['cohort_remaining_seconds']=150
        self.assertTrue(w.budget_fits(ledger,job,300));w.reserve_job(ledger,'seed2',job)
        self.assertEqual(ledger.reserved(),200);self.assertEqual(w.cohort_coverage(ledger,job),100)
        w.undo_reservation(ledger,'seed2');self.assertEqual(ledger.reserved(),200)
        self.assertEqual(w.cohort_coverage(ledger,job),150)
    def test_protected_transfer_is_idempotent_and_no_charge(self):
        provider='old:protected_final';ledger=Ledger(100,{provider:dict(remaining_seconds=200)})
        manifest={'scheduler':dict(protected_gpu_seconds=200,protected_reservation_provider=provider)}
        w.initialize_protection(ledger,manifest,300);w.initialize_protection(ledger,manifest,300)
        self.assertNotIn(provider,ledger.data['reservations']);self.assertEqual(ledger.reserved(),200)
        self.assertEqual(ledger.spend(),100)
    def test_sealed_transfer_preserves_total_and_undo(self):
        ledger=Ledger(100,{w.protected_key():dict(remaining_seconds=200)})
        job=dict(action='evaluate',mode='sealed',max_seconds=100)
        self.assertTrue(w.budget_fits(ledger,job,300));w.reserve_job(ledger,'final',job)
        self.assertEqual(ledger.reserved(),200);w.undo_reservation(ledger,'final')
        self.assertEqual(ledger.reserved(),200)
    def provider(self,smoke=False):
        source=self.root/'old';source.mkdir();output=source/'checkpoint.pt';output.write_bytes(b'test only')
        scientific=dict(outcome='completed',status='complete',best_path=str(output),last_path=str(output),best_sha256=w.c.sha(output),last_sha256=w.c.sha(output))
        stage=source/'stage.json';w.c.save(stage,{'outcome':'completed'})
        receipt=source/'receipt.json';w.c.save(receipt,dict(outcome='completed',result_sha256=w.c.sha(stage)))
        result=source/'scientific.json';w.c.save(result,scientific)
        destination=self.root/('smoke_receipts' if smoke else 'scientific_receipts')/'seed.json'
        return {'reuse_pending_receipt':dict(path=str(receipt),stage_result=str(stage),scientific_receipt=str(result),publish_scientific_receipt=str(destination))},destination
    def test_pending_provider_waits_without_duplicate(self):
        self.assertIsNone(w.pending_provider({'reuse_pending_receipt':{'path':str(self.root/'absent')}}))
    def test_pending_provider_publishes_correct_smoke_destination(self):
        job,path=self.provider(smoke=True);self.assertEqual(w.pending_provider(job)['status'],'COMPLETED')
        self.assertTrue(path.exists());self.assertFalse((self.root/'scientific_receipts/seed.json').exists())
    def test_provider_corruption_rejected(self):
        job,_=self.provider();Path(job['reuse_pending_receipt']['stage_result']).write_text('{}')
        with self.assertRaises(AssertionError):w.pending_provider(job)
    def test_failed_provider_does_not_retry(self):
        job,_=self.provider();w.c.save(job['reuse_pending_receipt']['path'],{'outcome':'technical_failed'})
        self.assertEqual(w.pending_provider(job)['status'],'UNAVAILABLE')
    def test_prior_active_slot_and_uuid_reserved_and_pause_required(self):
        provider=self.root/'old';w.c.save(provider/'state.json',{'jobs':{'run':dict(status='RUNNING',slot=0,gpu_uuid='gpu',guardian={'pid':1})}})
        jobs={'inherited':{'inherited_active_provider':str(provider)}}
        with self.assertRaises(AssertionError):w.inherited_active(jobs)
        w.c.save(provider/'control.json',{'pause':True})
        with patch.object(w.q,'alive',return_value=True):
            rows=w.inherited_active(jobs);self.assertEqual([r['slot'] for r in rows],[0]);self.assertEqual(rows[0]['gpu_uuid'],'gpu')
    def test_whole_cohort_defer_and_remaining_excludes_active_provider(self):
        jobs={'a':{'max_seconds':100},'b':{'max_seconds':100},'c':{'max_seconds':100,'reuse_pending_receipt':{}}}
        state={'jobs':{'a':{'status':'RUNNING'},'b':{'status':'PENDING'},'c':{'status':'COMPLETED'}}}
        manifest={'scheduler':{'cohorts':{'group':['a','b','c']}}};job={'cohort_group':'group'}
        self.assertEqual(w.cohort_remaining(job,jobs,state,manifest),100)
        w.defer_cohort('b',job,jobs,state,manifest,'budget')
        self.assertEqual(state['jobs']['a']['status'],'RUNNING');self.assertEqual(state['jobs']['b']['status'],'DEFERRED')


if __name__=='__main__':unittest.main()
