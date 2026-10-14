"""Freeze the finite job matrix and exact host commands before any GPU job."""
import json,sys,time
from pathlib import Path
import common_v9 as c
from policy_final03 import activate
activate()

def conditional():
    import scheduler_v9 as s
    with s.Ledger() as l:
        available=min(14400,l.data['historical_seconds']+14400-l.spend())-7950
        result=[]
        for branch in c.read(c.ROOT/'manifest.json')['conditional_branches']:
            b=dict(branch)
            if not b.get('provenance_ok',True):b['status']='UNAVAILABLE'
            elif b['name']=='fresh FP32/FP16 timing sessions' and available>=b['minimum_gpu_seconds']:b['status']='READY';b['reason']='Verified freed budget admits all three fresh FP32/FP16 paired sessions'
            else:b['status']='DEFERRED_BUDGET'
            b['verified_free_after_protected_required_inference_seconds']=max(0,available)
            b['reason']+='; required normal+sealed inference remains protected; no benchmark scores are used'
            result.append(b)
    c.save(c.ROOT/'conditional_admission.json',dict(branches=result,created=c.q.now(),target_scores_unread=True))

def freeze():
    for name in ['logs','job_launches','command_receipts','scientific_receipts','runs']:(c.ROOT/name).mkdir(exist_ok=True)
    assert c.read(c.ROOT/'scheduler_tests.json')['outcome']=='passed';assert c.read(c.ROOT/'protocol_tests.json')['outcome']=='passed'
    spec=c.read(c.ROOT/'training_spec.json');V8=c.read(c.V8/'budget_ledger.json');assert V8['cumulative_gpu_seconds']==93490.88168258697 and V8['reserved_gpu_seconds']==0 and V8['active_elapsed_seconds']==0
    probe=c.read(c.PROBE/'probe_counts/receipt.json')
    for x in c.read(c.V7/'reuse_models.json'):
        if x['graph']=='R4':assert probe['models'][str(x['seed'])]==x['sha256']
    jobs={}
    def add(name,block,script,args,seconds,peak=0,deps=None,cpu=False,outputs=None,settled=False,teacher=False,seed=None):
        jobs[name]=dict(block=block,command=[c.PYTHON,str(c.ROOT/script)]+args,max_seconds=seconds,peak_mib=peak,device='cpu' if cpu else'gpu',dependencies=deps or[],outputs=[str(c.ROOT/x) for x in(outputs or[])],dependency_policy='settled' if settled else'completed',teacher=teacher,seed=seed)
    add('gpu_smoke','smoke','jobs_v9.py',['smoke'],180,5500,outputs=['smoke.json'],teacher=True)
    add('source_teacher_masks','source_masks','jobs_v9.py',['source_teacher'],180,3250,deps=['gpu_smoke'],outputs=['source_teacher_manifest.json'],teacher=True)
    for seed in c.SEEDS:
        for arm in ['human','teacher','E1','E2','E3']:
            primary=arm in ['human','teacher'];name=f'{arm}_s{seed}';args=['--capsule',str(c.CAPSULE),'--rungs-dir',str(c.RUNGS),'--spec',str(c.ROOT/'training_spec.json'),'--out',str(c.ROOT/'scientific_receipts'/f'{name}.json'),'--run-dir',str(c.ROOT/'runs'/name),'--graph','B0' if primary else'R4','--method',arm,'--lr','0.00005' if primary else'0.005','--seed',str(seed)]
            add(name,'label_swap' if primary else'edge_arms','train_v9.py',args,292.5 if primary else 420,1200 if primary else 800,deps=['source_teacher_masks'] if primary else['gpu_smoke'],outputs=[f'scientific_receipts/{name}.json'],seed=seed)
            jobs[name]['paired_controls']='human versus teacher, same seed' if primary else'E0,E1,E2,E3, same seed/initialization/images/batches/recipe'
    # Priority1 primary pairs, then edge arms; independent jobs may use both GPUs.
    jobs={k:jobs[k] for k in ['gpu_smoke','source_teacher_masks']+[f'{arm}_s{seed}' for seed in c.SEEDS for arm in ['human','teacher']]+[f'{arm}_s{seed}' for seed in c.SEEDS for arm in ['E1','E2','E3']]}
    train=list(jobs)[2:]
    add('conditional_admission','cpu_audit','freeze_v9.py',['conditional'],60,cpu=True,deps=train,settled=True,outputs=['conditional_admission.json'])
    for i in range(3):
        add(f'timing_session{i}','timing','jobs_v9.py',['timing','--session',str(i)],200,1000,deps=['conditional_admission'],outputs=[f'timing_session{i}.json']);jobs[f'timing_session{i}']['measurement']=True;jobs[f'timing_session{i}']['conditional_branch']='fresh FP32/FP16 timing sessions'
    add('freeze_models','cpu_audit','cache_v9.py',['freeze'],180,cpu=True,deps=[f'timing_session{i}' for i in range(3)],settled=True,outputs=['final_models.json'])
    add('normal_teacher_B0','normal_inference','jobs_v9.py',['normal'],1350,5500,deps=['freeze_models'],outputs=['gpu_inference_normal.json'],teacher=True)
    add('source_selection','cpu_source','evaluate_v9.py',['source'],1200,cpu=True,deps=['normal_teacher_B0'],outputs=['source_operating_points.json','evaluation_source.json'])
    add('final_reference_reader','cpu_final_cache','cache_v9.py',['sealed'],3600,cpu=True,deps=['source_selection'],outputs=['sealed_windows.json','final_evaluation_freeze.json'])
    add('sealed_teacher_B0','final','jobs_v9.py',['sealed'],6600,5500,deps=['final_reference_reader'],outputs=['gpu_inference_sealed.json'],teacher=True)
    add('normal_R4_PTQ_analysis','cpu_normal','evaluate_v9.py',['normal'],6600,cpu=True,deps=['final_reference_reader'],outputs=['evaluation_normal.json'])
    add('sealed_R4_PTQ_analysis','cpu_final','evaluate_v9.py',['sealed'],10800,cpu=True,deps=['sealed_teacher_B0','normal_R4_PTQ_analysis'],settled=True,outputs=['evaluation_sealed.json'])
    branches=[
        dict(name='B0 E1 transfer',status='UNAVAILABLE',provenance_ok=False,controls='four compatible B0 E0/E1 pairs',reason='Original B0 hard-label checkpoints lack full embedded matched input/initialization lineage; eight new N2500 B0 runs exceed protected freed-budget allowance',minimum_gpu_seconds=8000),
        dict(name='hard negatives A1/A2',status='DEFERRED_BUDGET',controls='E0 plus four A1/A2 pairs',reason='Eight complete N2500 trainings plus pool-feature/candidate-overlap preparation require3360s beyond core and full inference reservations',minimum_gpu_seconds=3360),
        dict(name='human S2 UNet teacher students',status='UNAVAILABLE',provenance_ok=False,controls='four matched Prithvi/UNet student pairs',reason='Located candidate lacks verified252-human-source training provenance; no new teacher training authorized',minimum_gpu_seconds=5400),
        dict(name='same-image soft-KD extension',status='DEFERRED_BUDGET',controls='four human/teacher-hard pairs with identical0.5*T^2 KL T4',reason='Eight full live-teacher CE+KL trainings need a conservative6000s paired allowance beyond protected core/inference; primary true-CE pairs take priority',minimum_gpu_seconds=6000),
        dict(name='R4 FP/QAT continuation',status='DEFERRED_BUDGET',controls='source-only LR pilot then matched FP/QAT parents42/64',reason='No competent verified tuned continuation recipe; bounded source pilot plus complete FP/QAT readouts need a separate1800s reservation; unchanged PTQ has priority',minimum_gpu_seconds=1800),
        dict(name='fresh FP32/FP16 timing sessions',status='DEFERRED_BUDGET',controls='three fresh comparable sessions',reason='Full scientific inference and collection protected; no co-located timing and no new native low-bit backend',minimum_gpu_seconds=600)]
    config=dict(gpu_poll_interval_seconds=600,heartbeat_seconds=30,initial_gpu_wait_hours=8,execution_hours_from_first_gpu_command=8,absolute_wall_hours=16,final_collection_reserve_seconds=600,max_technical_retries_per_job=1,termination_grace_seconds=15,gpu_uuid_lock_order='GPU UUID lock then shared ledger mutex',shared_ledger=str(c.ROOT.parent/'.v9_shared_gpu_ledger.json'),owned_process_identity='PID+proc start_ticks+bootID',automatic_retry='None without a diagnosed transient; scientific failure has no retry')
    c.save(c.ROOT/'scheduler_config.json',config)
    frozen={str(p):c.sha(p) for p in c.ROOT.glob('*.py')};frozen.update({str(c.ROOT/x):c.sha(c.ROOT/x) for x in ['training_spec.json','sealed_wf2.json','edge_label_manifest.json','source_image_manifest.json','source_edge_choice.json','ptq_manifest.json','normal_windows.json','scheduler_config.json']})
    c.save(c.ROOT/'manifest.json',dict(run=str(c.ROOT),created=c.q.now(),branch='research',historical_gpu_seconds=V8['cumulative_gpu_seconds'],imported_probe_gpu_seconds=14.6,imported_probe_sha256=c.sha(c.PROBE/'gpu_vram_check.json'),incremental_cap_seconds=14400,cumulative_cap_seconds=108000,protected_final_seconds=6600,jobs=jobs,conditional_branches=branches,frozen_files=frozen,sealed_labels_opened=False,seeds=c.SEEDS,
        E0_reuse={str(seed):c.sha(c.V7/'receipts'/f'R4_CE_s{seed}.json') for seed in c.SEEDS},scientific_gates='prompt_v9 section7; seed-mean whole-event bootstrap10000 seed1729; dry alarm1% primary;0.1/5% secondary'))
    protocol='''# Frozen V9 protocol

Four paired B0 seeds42/64/91/107: identical252 S11 source images/valid support, fixed224 resize, shared frozen multiplicative reflectance gain and geometry, same teacher-before-student RNG draws, AdamW5e-5/wd.05, true CE, clip5, FP32,50 epochs/750 updates, source mIoU every30 updates and earliest best. Both arms are fresh. The frozen task-adapted Prithvi masks use identical inputs before common augmentation. Teacher source-image exposure is explicit.

R4 E0 is the complete7800-update V7 scratch anchor. E1/E2/E3 retain the exact R4 CE-named historical FOCAL gamma2 recipe, AdamW.005/wd.05/clip5, batch16/workers2,50 epochs, source validation every312 updates. Only labels change before augmentation. E1 uses k2/theta.1 chosen on S11 validation; unavailable spectral pixels retain teacher labels. E2 ignores the preaugmentation edge band. E3 uses global pwater threshold matching total E1 changed pixels as closely as grid.01 allows; cached confidence is MAX class and half ties are preserved and reported. Initial R4 hash and first two E0 batches must match V7 in admitted smoke. All actual arms record image-only batch hashes and gradients.

16-pixel teacher tokens are verified separately from H/4 decoder logits; no160m-vs40m decoder advantage is claimed. Thin-water width is maximum inscribed halfwidth per8-connected reference component, not every local branch. Vegetation low recall is not physical invisibility. G<H is not a universal optimization bound. No head adaptation or OOD-labelled calibration runs.

WorldFloods-v2 train/val are evaluation-only. Metadata sealing retained468 scenes/116 events, not the unfiltered130-event prompt total. GDACS all23720 footprints, S11 all hand chips including Bolivia and WF test are screened; S11 unknown dates use conservative spatial exclusion. Labels are read once, in the final reference-cache stage after every checkpoint and S11-only operating setting is frozen. Guarded raw bytes are hashed from that read; only immutable derived evaluation caches are then read by predictors/scorers. Native512 windows use zero padding invalid, clouds excluded, JRC permanent3 ignored, all13band-zero invalid, >1% native validity. Same-day S2 subgroup is secondary.

Every model uses deployed margin>0, source-selected tau grid-3..1/.5, source-only EM/anchor settings with inherited50-bin semantics, and unattainable target event oracle on the declared grid. Decode D uses the same own-predicted-edge physical correction; CPU512-window time is separately measured. AP uses declared4096 fixed margin bins[-32,32], clipped overflow/ties; monotone MNDWI/NDWI score scaling. Report pooled and equal-event metrics, seedSD separate from event bootstrap, all seed/event outcomes, dry-window alarms0.1/1/5%.

PTQ uses unchanged V8 W4A8 body/W8A8stem/FP head; reuse42/64 bytes;91/107 same16 training IDs,1729 calibration seed. CPU evaluation requires admitted GPU parity smoke. Simulation is not native deployment or losslessness. Source loss tolerances1pp primary,2/3pp secondary are exploratory.

Full paired-block GPU reservations and protected6600s final inference take priority. Conditional branches and conservative costs/provenance blockers are frozen in manifest.json before any target scores; no target-derived admission. Failed/deferred blocks are not negative scientific evidence. No automatic technical retry without a diagnosis. All GPU commands, including failures/smoke/teacher inference, charge elapsed wall time; concurrent costs sum. CPU/waits are separate. Worker stops at completion or persisted deadlines and always invokes CPU collection.
'''
    (c.ROOT/'frozen_protocol.md').write_text(protocol)
    ctl=f'{c.PYTHON} {c.ROOT}/scheduler_v9.py'
    (c.ROOT/'CONTROLS.txt').write_text('\n'.join(ctl+' '+a for a in ['status','pause','resume','stop'])+'\n')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='conditional':conditional()
    else:freeze()
