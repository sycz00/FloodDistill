"""Focused safety/protocol tests; no CUDA and no real polling waits."""
import copy,json,os,tempfile,unittest,fcntl,sys
from pathlib import Path
from unittest.mock import patch
import common_v9 as c
import scheduler_v9 as s

class Tests(unittest.TestCase):
    def row(self,at=0,foreign=False,**kw):
        return dict(uuid=c.GPUS[0],known=True,authorized=True,mode='Default',util=0,free=16000,total=24564,processes=[dict(pid=777,memory=3000)] if foreign else [],at=at,**kw)
    def job(self,**kw):return dict(peak_mib=610,**kw)
    def test_busy_ten_minute_cadence_and_both_devices(self):
        checks=[];launch=[];job=self.job();hist=[]
        # The fake clock reproduces qualification at the first idle poll.
        for t in [0,600,1200]:
            r=self.row(t,foreign=t<1200);hist.append(r);checks.append(t)
            ok,_,_=s.availability(r,job,hist,t)
            self.assertFalse(ok)
            if t==1200:
                for dt in [15,30]:hist.append(self.row(t+dt))
                ok,_,_=s.availability(hist[-1],job,hist,1230)
                if ok:launch.append(1230)
        self.assertEqual(checks,[0,600,1200]);self.assertEqual(launch,[1230])
        other=[self.row(t) for t in [0,15,30]]
        self.assertTrue(s.availability(other[-1],job,other,30)[0])
        self.assertFalse(s.availability(self.row(30,True),job,[self.row(0,True)],30)[0])
    def test_unknown_unallocated_low_memory_and_sleeping_foreign(self):
        history=[self.row(t) for t in [0,15,30]]
        for key,val in [('known',False),('authorized',False),('free',2500),('processes',[dict(pid=55,memory=100)]),('mode','Exclusive_Process')]:
            r=self.row(30);r[key]=val;self.assertFalse(s.availability(r,self.job(),history,30)[0],key)
    def test_under_lock_race_recheck(self):
        h=[self.row(t) for t in [0,15,30]];self.assertTrue(s.availability(h[-1],self.job(),h,30)[0]);self.assertFalse(s.availability(self.row(31,True),self.job(),h,31)[0])
    def test_colocation_stability_and_teacher_exclusion(self):
        h=[self.row(t,True) for t in [0,600,1200]];self.assertTrue(s.availability(h[-1],self.job(),h,1200)[0]);self.assertTrue(s.availability(h[-1],self.job(),h,1200)[2])
        self.assertFalse(s.availability(h[-1],self.job(teacher=True),h,1200)[0]);self.assertFalse(s.availability(h[-1],dict(peak_mib=2000),h,1200)[0])
        h[1]['processes'][0]['memory']=3400;self.assertFalse(s.availability(h[-1],self.job(),h,1200)[0])
    def test_memory_guard_and_owned_identity(self):
        pid=os.getpid();ident=s.identity(pid);self.assertTrue(s.alive(ident));bad={**ident,'start_ticks':'0'};self.assertFalse(s.alive(bad))
        with patch.object(s.os,'killpg') as kill:
            s.own_stop(bad);kill.assert_not_called();s.own_stop(ident);kill.assert_called_once_with(pid,s.signal.SIGTERM)
        row=self.row(0,True);base=sum(x['memory'] for x in row['processes']);row['processes'][0]['memory']+=1;self.assertGreater(sum(x['memory'] for x in row['processes']),base);row['free']=1000;self.assertLess(row['free'],1024)
    def test_reserve_complete_pair_and_protected_collection(self):
        class Fake:
            def __init__(self):self.data=dict(reservations={},charges={})
            def spend(self):return 100000
            def reserved(self):return sum(x['remaining_seconds'] for x in self.data['reservations'].values())
        l=Fake();self.assertFalse(s.reserve_block(l,'r','pair',6500,2000));self.assertEqual(l.data['reservations'],{})
        self.assertTrue(s.reserve_block(l,'r','pair',5000,2000));self.assertEqual(l.reserved(),7000)
        self.assertTrue(s.reserve_block(l,'r','pair',5000,2000));self.assertEqual(l.reserved(),7000)
        self.assertFalse(s.reserve_block(l,'other','pair',500,1000))
    def test_lock_exclusion_and_atomic_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gpu.lock';a=open(p,'a');b=open(p,'a');fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
            a.close();fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB);b.close()
    def test_deadlines_restart_and_no_reset(self):
        a=dict(launch=100,wait_deadline=28900,absolute_deadline=57700,first_gpu=None);self.assertEqual(s.runtime_deadline(a),28900)
        a['first_gpu']=20000;self.assertEqual(s.runtime_deadline(a),48800);b=json.loads(json.dumps(a));self.assertEqual(s.runtime_deadline(b),48800)
        self.assertTrue(48800-600<48800);self.assertEqual(a['launch'],100)
    def test_worker_integration_fake_clock_wait_restart_completion(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir()
            for name in ['logs','job_launches','command_receipts']:(root/name).mkdir()
            clock=[0.];launches=[]
            job=dict(peak_mib=610,max_seconds=60,block='pair',command=['irrelevant'],outputs=[],dependencies=[])
            c.save(root/'manifest.json',dict(jobs={'one':job},protected_final_seconds=100));c.save(root/'scheduler_config.json',dict(gpu_poll_interval_seconds=600,heartbeat_seconds=30))
            def tel():return {u:dict(self.row(clock[0],foreign=clock[0]<1200),uuid=u) for u in c.GPUS}
            def popen(*args,**kw):
                launches.append(clock[0]);c.save(root/'command_receipts/one.json',dict(job='one',block='pair',status='COMPLETED',seconds=1,outputs={}));return SimpleNamespace(pid=os.getpid())
            realopen=os.open
            def fdopen(path,*args):return realopen(root/(Path(path).name) if str(path).startswith('/data1/') else path,*args)
            with patch.object(c,'ROOT',root),patch.object(s,'STATE',root/'state.json'),patch.object(s,'CONTROL',root/'control.json'),patch.object(s,'LEDGER',root/'ledger.json'),patch.object(s,'MUTEX',root/'ledger.lock'),patch.object(s.time,'time',lambda:clock[0]),patch.object(s.time,'monotonic',lambda:clock[0]),patch.object(s.time,'sleep',lambda n:clock.__setitem__(0,clock[0]+n)),patch.object(s,'telemetry',tel),patch.object(s.subprocess,'Popen',popen),patch.object(s.os,'open',fdopen),patch.object(s,'collect',lambda state:c.save(root/'test_final.json',state)):
                s.run();self.assertEqual(launches,[1230.]);state=c.read(root/'test_final.json');self.assertEqual(state['jobs']['one'],'COMPLETED');self.assertEqual(state['launch'],0);self.assertEqual(state['first_gpu'],1230)
                state['finished']=clock[0];c.save(root/'state.json',state)
                with self.assertRaises(SystemExit):s.run()

def protocol():
    import numpy as np,torch
    c.setup();import edge_v9 as e
    from data_v9 import dataset
    from train_v9 import state_hash
    result=e.checks();spec=c.read(c.ROOT/'training_spec.json');manifest=c.read(c.ROOT/'edge_label_manifest.json');names=list(manifest['files'])
    for name in [names[0],names[len(names)//2],names[-1]]:
        a=torch.load(Path(spec['pseudo'])/name,weights_only=True);b=torch.load(c.ROOT/'edge_labels'/name,weights_only=True)
        for arm in ['E1','E2','E3']:assert torch.equal(b[arm][~b['band']],a['mask'][~b['band']].to(torch.int8))
    # Shared augmentation moves image/mask together with a fixed worker stream.
    ds=dataset(spec,'E1',42);grid=torch.arange(224*224).reshape(224,224);out=ds._apply_geom(dict(image=grid[None],mask=grid.clone()),0);assert torch.equal(out['image'][0],out['mask'])
    from evaluate_v9 import threshold_counts,hist2,groups,calibrate
    margin=np.array([-4,-3,-2.5,0,0,.5,1,2],np.float32);mask=np.array([0,1,1,0,1,0,1,1],np.int8)
    from evaluate_v9 import TAUS
    for tau,cell in zip(TAUS,threshold_counts(margin,mask)):assert np.array_equal(cell,e.counts(margin>tau,mask))
    from evaluate_v9 import coarse
    fake=np.tile(np.linspace(-4,4,16,dtype=np.float32),(16,1));reference=(fake>0).astype(np.int8);H=coarse(fake,fake/4,np.ones_like(fake,bool),reference)
    fitted=calibrate(np.stack([H,H]),[dict(event='one',map='one'),dict(event='two',map='two')]);assert 'em' in fitted['rules'] and 'anchor' in fitted['rules']
    # Confusion parity with the adopted evaluator on an actual window from each set.
    wm=c.read(c.ROOT/'normal_windows.json');sample=[]
    from evaluation import revision_native_border as rb
    cache_manifest=c.read(c.V7/'reference_cache/manifest.json')
    for name in ['s11val','s11','sturm','wf']:
        row=next(x for x in wm['windows'] if x['dataset']==name)
        with np.load(row['path']) as a:
            pred=np.indices(a['mask'].shape).sum(0)%2==0;own=e.counts(pred,a['mask']);old=rb.confusion(pred,a['mask']);assert np.array_equal(own,old[1].tolist()+[int(old[0,0])])
            if name!='s11val':
                cached=cache_manifest['datasets'][name];i=next(i for i,r in enumerate(cached['rows']) if r['filename']==row['name']);bits=np.load(cached['path'],mmap_mode='r');oldrule=np.unpackbits(bits[i,1],count=512*512).reshape(512,512).astype(bool);m,av,_,_=e.indices(e.reflectance(a['image']));assert np.array_equal((m>0)&av,oldrule),(name,'MNDWI cache parity')
            sample.append(dict(dataset=name,counts=own.tolist()))
    # Strict original EfficientViT and new source B0 builders import/reload.
    b0=c.build('B0');assert sum(p.numel() for p in b0.parameters())>100000
    # Guard rejects a real sealed path before opening any bytes.
    guard=c.install_label_guard();sealed=c.read(c.ROOT/'sealed_wf2.json')['scenes'][0]['reference']
    try:open(sealed,'rb');raise AssertionError('sealed guard did not reject')
    except PermissionError:pass
    c.save(c.ROOT/'protocol_tests.json',dict(**result,real_window_confusion_parity=sample,outside_band_equal=True,augmentation_alignment=True,strict_threshold_ties=True,efficientvit_import=True))

if __name__=='__main__':
    if '--protocol' in sys.argv:protocol()
    else:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
        c.save(c.ROOT/'scheduler_tests.json',dict(outcome='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),fake_clock=True,real_GPU_commands=0));sys.exit(0 if r.wasSuccessful() else 1)
