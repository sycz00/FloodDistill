"""Freeze the finite core-first DAG. Safe to run only before worker launch."""
import json
from pathlib import Path
import shutil
import sys
import time
import common11 as c

def main():
    assert not (c.ROOT/'state.json').exists(),'Cannot change a launched queue'
    from worker11 import validate_manifest
    for name in ['logs','receipts','stage_results','running']:(c.ROOT/name).mkdir(exist_ok=True)
    jobs={}
    def add(name,script,args,output,*,deps=(),block='smoke',threads=1,seconds=90,heavy=False,**fields):
        command=[c.PYTHON,str(c.ROOT/'code'/script),*args]
        jobs[name]=dict(argv=[c.PYTHON,str(c.ROOT/'code/job11.py'),name],command=command,deps=list(deps),block=block,threads=threads,max_seconds=seconds,compute_heavy=heavy,outputs=[dict(path=str(c.ROOT/output),type='json',required_keys=['outcome']),dict(path=str(c.ROOT/'stage_results'/(name+'.json')),type='json',equals={'outcome':'completed'})],**fields)
    add('00_inventory','inventory11.py',[],'dependency_evidence_ledger.json',seconds=120)
    add('01_resolve','repair11.py',['--threads','1','resolve-inputs'],'repair_inputs.json',deps=['00_inventory'],seconds=180,dependency_kind='repair')
    jobs['01_resolve']['outputs'][0]['required_keys']=['schema','parents','cache_manifest']
    add('02_source_gate','repair11.py',['--threads','8','source-gate'],'repair/core/source_gate.json',deps=['01_resolve'],threads=8,seconds=900,heavy=True)
    add('03_profile_D_s42','repair11.py',['--threads','8','profile','--seed','42','--arm','D'],'repair/core/profile.json',deps=['02_source_gate'],threads=8,seconds=1200,heavy=True,source_gate=True)
    add('04_cost_admission','action11.py',['admission'],'profile_admission.json',deps=['03_profile_D_s42'],threads=8,seconds=180,heavy=True)
    train=[]
    for arm in ['D','U','A','L','N','P','F0']:
        for seed in c.SEEDS:
            name=f'10_train_{arm}_s{seed}';deps=['04_cost_admission']
            if arm != 'D':deps.append(f'10_train_D_s{seed}')
            add(name,'repair11.py',['--threads','8','train','--arm',arm,'--seed',str(seed)],f'repair/core/{arm}_s{seed}/receipt.json',deps=deps,block='core',threads=8,seconds=640,heavy=True,source_gate=True,profile_gate=True,training_rule=arm,seed=seed,updates=600)
            train.append(name)
    add('20_source_selection','repair11.py',['--threads','1','select'],'repair/core/selection.json',deps=train,block='core',seconds=90)
    add('21_optional_decision','action11.py',['optional'],'optional_decision.json',deps=['20_source_selection'],block='decisions',seconds=90,soft_dependencies=True)
    add('22_conditioner','conditioner11.py',['action','--output',str(c.ROOT/'conditioner_receipt.json')],'conditioner_receipt.json',deps=['21_optional_decision'],block='conditioner',threads=2,seconds=180,soft_dependencies=True,deferred_action=True)
    jobs['22_conditioner']['outputs'][1]['equals']={'outcome':'DEFERRED'}
    add('23_final_freeze','action11.py',['freeze'],'final_evaluation_freeze.json',deps=['22_conditioner'],block='decisions',seconds=120,soft_dependencies=True)
    add('30_final_evaluation','evaluate11.py',[],'evaluation/receipt.json',deps=['23_final_freeze'],block='evaluation',threads=8,seconds=21600,heavy=True,selected_models=True)
    blocks={name:dict(job_ids=[n for n,j in jobs.items() if j['block']==name],reserve_core_hours=cap,protected=name in ['evaluation','collection']) for name,cap in [('smoke',6),('core',40),('decisions',.1),('conditioner',12),('evaluation',48),('collection',1.4)]}
    old=[('research_v9_recovery_20261014T081500Z',8),('research_v10_recoveryB_20261014T084000Z',24)]
    charges=[dict(id='prelaunch_readonly_preparation',reserved_core_seconds=900,measured_process_cpu_seconds=None,reason='Conservative charge for short uninstrumented prelaunch inventory, file reads and code preparation; not claimed measured CPU time')]
    previous=c.read(c.ROOT.parent/'research_v11_20261013T231203Z/resource_ledger.json')
    charges.append(dict(id='prior_V11_run_cumulative_charge',reserved_core_seconds=previous['completed_charged_core_seconds'],measured_process_cpu_seconds=previous['measured_process_cpu_seconds'],reason='Carry forward all original V11 consumed budget; no reset'))
    previous_test=c.read(c.ROOT/'scheduler_launch_only/resource_ledger.json',{})
    if previous_test:charges.append(dict(id='initial_recovery_detach_test',reserved_core_seconds=previous_test['completed_charged_core_seconds'],measured_process_cpu_seconds=previous_test['measured_process_cpu_seconds'],reason='Completed first two-job detached check retained, before controller-restart test'))
    sentinel=c.read(c.ROOT/'scheduler_sentinel/verification_receipt.json',{})
    if sentinel:
        charges.append(dict(id='detached_recovery_sentinel',reserved_core_seconds=sentinel['charged_core_seconds'],measured_process_cpu_seconds=sentinel['measured_cpu_seconds'],reason='Verified two-job detached sentinel, controller restart, guardian reattachment and collection'))
    for name in ['conditioner_tests.json','repair_tests.json','worker_tests.json','evaluation_tests.json','targeted_recovery_tests.json','dependency_recovery_tests.json','runtime_reservation_tests.json','readout_optimization_tests.json','all_recovery_tests.json']:
        a=c.read(c.ROOT/name,{}) or c.read(c.ROOT/'code'/name,{})
        if a:
            elapsed=a.get('elapsed_seconds',a.get('wall_seconds',a.get('wall_time_seconds',0)))
            charge=a.get('total_charged_core_hours',0)*3600 or float(elapsed or 0)*2
            charges.append(dict(id=name,reserved_core_seconds=charge,measured_process_cpu_seconds=a.get('process_cpu_seconds'),reason='Recorded targeted CPU test; two-thread allocation; all timed/recorded prior attempts included where available'))
    for name in ['thread_profile.json','readout_thread_profile.json','batch_profile.json','readout_actual_Q0_batch_parity.json','prelaunch_evaluation_cost_profile.json']:
        a=c.read(c.ROOT/name,{})
        if a:charges.append(dict(id=name,reserved_core_seconds=8*float(a.get('elapsed_seconds',a.get('wall_seconds',0)))+80+float(a.get('failed_prior_attempt_reserved_core_seconds',0)),measured_process_cpu_seconds=None,reason='Conservative eight-thread elapsed plus10seconds setup for synthetic/source-only runtime profile'))
    manifest=dict(schema_version=11,python=c.PYTHON,cwd=str(c.REPO),dependency_module='dependency11',cap_core_hours=64,safe_host_threads=40,collection_block='collection',collection_threads=1,collect_argv=[c.PYTHON,str(c.ROOT/'code/collect11.py')],external_workers=[dict(name=name,state_path=str(c.ROOT.parent/name/('queue_manifest.state.json' if name.startswith('research_v9_') else 'state.json')),max_threads=threads) for name,threads in old],blocks=blocks,jobs=jobs,initial_charges=charges,env=dict(PYTHONPATH='/home/schmalstieg/repos/_deps/efficientvit:'+str(c.ROOT/'code')+':'+str(c.REPO)),scientific_contract=dict(seeds=c.SEEDS,core='R4 W4A4 body/W8A8 stem/FP32 head',updates=600,source_checkpoints=[0,50,200,600],source_only_gates=True,external_looks=1,optional_extension='not admitted; preserve primary/evaluation',optional_transfer='not admitted; preserve primary/evaluation',unavailable_conditioner='defer without blocking final evaluation'))
    manifest['measured_reservations']=dict(job='04_cost_admission',receipt='profile_admission.json',evaluation_job='30_final_evaluation',policy='Complete source-profile bounds plus30seconds shutdown; whole core and final readout reserved atomically before any main training')
    validate_manifest(manifest)
    c.save(c.ROOT/'manifest.json',manifest)
    files={str(p):c.sha(p) for p in (c.ROOT/'code').glob('*.py')}
    files[str(c.ROOT/'manifest.json')]=c.sha(c.ROOT/'manifest.json')
    for path,expected in c.read(c.ROOT/'initial_freeze.json')['files'].items():
        assert c.sha(path)==expected,path
        files[path]=expected
    # Freeze imported scientific and scheduling sources without copying or changing V10.
    for folder in [c.V10/'code',c.V10/'reference',c.V9]:
        for path in folder.glob('*.py'):files[str(path)]=c.sha(path)
    capsule=c.ROOT.parent/'capacity_v4_20261008/frozen/297031f5f4ae1df5'
    for path in capsule.rglob('*.py'):files[str(path)]=c.sha(path)
    for path in (c.ROOT.parent/'capacity_v4_20261008/code/v5').glob('*.py'):files[str(path)]=c.sha(path)
    c.save(c.ROOT/'code_freeze.json',dict(outcome='completed',at=time.time(),files=files,external_inputs_bound_only_after_completion=True))
    python=c.PYTHON;worker=c.ROOT/'code/worker11.py'
    commands=['V11 CPU-only detached queue; preserve existing host/allocation.','',*[f'{python} {worker} {action} --root {c.ROOT}' for action in ['status','pause','resume','stop']],f'tail -n 80 {c.ROOT}/logs/worker.log',f'tail -n 30 {c.ROOT}/availability.jsonl','',f'Restart lost worker only: setsid {python} {worker} run --root {c.ROOT} </dev/null >>{c.ROOT}/logs/worker.log 2>&1 &','Pause prevents new jobs; stop checkpoints/terminates only verified owned V11 processes and collects partial results. Resume does not reset deadlines. Resource/dependency checks every 600 seconds; heartbeat every 30 seconds.']
    (c.ROOT/'CONTROLS.txt').write_text('\n'.join(commands)+'\n')
    c.save(c.ROOT/'preparation.json',dict(outcome='completed',jobs=len(jobs),training_jobs=len(train),manifest_sha256=c.sha(c.ROOT/'manifest.json'),code_freeze_sha256=c.sha(c.ROOT/'code_freeze.json'),new_GPU_training=False))
    print('Prepared',len(jobs),'jobs including',len(train),'paired CPU continuations',flush=True)

if __name__=='__main__':main()
