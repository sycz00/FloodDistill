"""Finish Stage1 evidence, freeze the protocol, then launch the prescribed FP DAG."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml
BASE=Path(__file__).resolve().parent
common=json.loads((BASE/'shared_input_identity.json').read_text());capsule=Path(common['capsule'])
sys.path.insert(0,str(capsule))
from experiments import revision_queue as q
V2=BASE.parent/'compression_v2_20261007'
pin=lambda p:{'path':str(p),'sha256':q.file_hash(p)}
read=lambda p:json.loads(Path(p).read_text())
adopt=read(BASE/'simulator_adoption.json')
assert adopt['adopted_simulator']=='integer_domain_v4'
sens=[read(BASE/'results'/f'{cell}_sensitivity.json') for cell in ('A0_W4A8','A1_W4A8','A3_W4A8','A3_W4A4')]
reach=[read(BASE/'results'/f'optimizer_lr{label}.json') for label in ('1e-5','1e-4','5e-4')]
assert all(r['status']=='complete' for r in sens+reach)
assert all(r['completed_updates']==1560 for r in reach)
metrics=capsule/'evaluation/revision_eval.py'
protocol={'version':'capacity_v4_20261008','frozen_at':q.now(),'status':'frozen_before_Stage2_results',
    'task':'binary surface water; not flood increments','cohort_sha256':common['cohort_sha256'],
    'teacher_sha256':common['teacher_sha256'],'analysis_code_sha256':q.file_hash(metrics),
    'count_and_event_code_sha256':q.file_hash(capsule/'evaluation/compression_collect.py'),
    'graph_declarations':pin(BASE/'stage2_graphs_declared.json'),
    'simulator_adoption':pin(BASE/'simulator_adoption.json'),'seed_set':[42,64,91,107],
    'data_roles':{'fits':'TRAIN only','hyperparameters_checkpoint_selection':'S11 validation89',
                  'external':'S11-test/STURM/WF frozen secondary readouts; never method/gate tuning'},
    'FP_recipe':{'lr':5e-5,'weight_decay':.05,'batch':16,'drop_last':True,'epochs':50,
                  'updates':7800,'cosine':True,'AMP':False,'clip':5,'scratch':True,'activation':'ReLU6',
                  'validation_every_epochs':2,'strict_best_earlier_tie':True,
                  'inherited_early_stop_patience':20,'incomplete_budget_is_failure':True},
    'base_loss':'.5 focal(gamma2) + .5 T^2 KL(teacher||student), T4, valid pixels',
    'Stage3_recipe_grid':{'lr':[1e-4,5e-4],'epochs':[10,30],'maximum_configs':6,
        'selection':'hyperparameter selection on A3 W4A8 seed42 S11 validation only; no scientific superiority claim',
        'learned_steps':'LSQ-style only after primary-source inspection and tests; not yet implemented'},
    'hypotheses':{
        'H1':{'criterion':'median repaired preclip norm <=10x same-cell FP frozen-BN; max <=100x',
              'logger_independent_F64_deviation_max':.01},
        'H2':{'criterion':'conventional fixes recover >=50% A3 W4A4-to-W4A8 PTQ gap; TRAIN-ranked selective8 recovers >=50% A1 W4 weight damage'},
        'H3':{'criterion':'R2 minus A0 S11-test mIoU CI lower bound > -delta_NI; teacher comparisons on STURM/WF water-IoU with event CIs'},
        'H4':{'criterion':'A1 minus A3 pooled mIoU negative on STURM and WF over4seeds with paired-seed95% CI excluding0; descriptive secondary test, not a scheduling gate'},
        'H5':{'criterion':'3seed source gain >=max(.005,2SE) over strongest control, all same sign, waterIoU not worse, diagnostic changes; seed107 then conditional event CIs'}},
    'delta_NI':{'provisional_absolute':.02,'rule':'max(.02,2*measured median seed SD of waterIoU)',
                'sensitivity':[.01,.03],'never_loosen_after_external_results':True},
    'MDE_context_pp':{'pretrained_N2500_median_seed_SD':[.22,.20,.51],
        'pretrained_N2500_max_seed_SD':[.4,.7,.8],
        'conditional_event_halfwidth':[.9,1.3,1.9],
        'warning':'Historical pretrained estimates do not guarantee power for new scratch graphs; measure paired-seed variance'},
    'statistics':{'seed_SD':'sample ddof1','paired_seed_CI':'Student-t interval on paired differences; never pixels',
                   'event_bootstrap_resamples':10000,'bootstrap_seed':1729,
                   'bootstrap_unit':'event, conditional fixed checkpoints','no_prediction_pooling_across_seeds':True},
    'stage_caps_gpu_seconds':{'1':7200,'2':21600,'3':21600,'4':36000,'5':14400,'6':7200,'8':21600},
    'programme_target_gpu_seconds':108000,'sum_of_stage_maxima_is_not_a_global_reservation':True,
    'budget_enforcement':'actual completed command seconds plus outstanding hard-timeout reservations; explicit reallocation',
    'Stage5_unlock':'SB residual >=.01 at W8A8/W4A8 or >=.015 at4bit body; within.005 atW4A8 closes Stage5',
    'Stage8_order':['U6','U1','U5','U7_CPU_gate','U8','U3','U4','U7_runs'],
    'scope_limits':['simulation_only until supported engines measured','no externally selected graphs/seeds/recipe',
                    'no novelty claim before matched isolating controls and related-work audit'],
    'Stage1_optimizer_grid_rows':[{'lr':r['lr'],'selected_mIoU':r['validation']['selected']['micro']['mIoU'],
                                 'final_mIoU':r['validation']['final']['micro']['mIoU']} for r in reach]}
q.atomic_json(BASE/'protocol_v4.json',protocol)
(BASE/'protocol_v4.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False));(BASE/'protocol_v4.yaml').chmod(0o444)
rows=[]
for r in sens:
    for v in r['rows']:rows.append({'type':'sensitivity','cell':r['cell'],**{k:value for k,value in v.items() if not isinstance(value,dict)}})
for r in reach:rows.append({'type':'optimizer','cell':'A3_W4A8','lr':r['lr'],
    'selected_mIoU':r['validation']['selected']['micro']['mIoU'],'final_mIoU':r['validation']['final']['micro']['mIoU'],
    'selected_updates':r['selected_updates'],'completed_updates':r['completed_updates'],
    'codes_changed_fraction':r['movement']['selected']['codes_changed_fraction'],'norm_p50':r['norm_p50'],
    'norm_p95':r['norm_p95'],'norm_max':r['norm_max'],'clipped_fraction':r['clipped_fraction']})
keys=list(dict.fromkeys(k for r in rows for k in r))
with (BASE/'stage1.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
(BASE/'stage1_note.txt').write_text('Stage1 completed fixed-weight measurement repairs, source parity/rescore, TRAIN gradient/excess-loss audit, sensitivity maps/U6 screen, and3 equal-update optimizer-reach runs.\n'
    'Adopted integer-code forward with exact native STE backward; no old source was changed. W4A4 TRAIN forward disagreement remains about8%, so old source scores were rescored. Largest selected v2 source difference0.138pp.\n'
    'Fused QKV attribution jointly bypasses Q/K/V; no isolated V-only quantizer claim. Counterfactual gradient norms and class/boundary/teacher-confidence excess loss are diagnostics, not new benchmarks.\n'
    'All new scientific decisions use TRAIN/S11 validation; external benchmark scores have not been used. U6 remains a one-seed PTQ screen. Serialized/packed low-bit deployment remains unestablished.\n'
    'protocol_v4.yaml and the prescribed six graph declarations are frozen before Stage2. Full raw JSON retains every failed unit, repaired variant, selected/final score and movement diagnostic.\n')
template=read(BASE/'stage1_measurement/manifest.json')
directory=BASE/'stage2_fp';directory.mkdir(exist_ok=False)
shutil.copy2(BASE/'stage1_measurement/revision_queue.py',directory/'revision_queue.py')
version=BASE/'code/stage2';version.mkdir()
shutil.copy2(BASE/'stage2_fp.py',version/'stage2_fp.py');(version/'stage2_fp.py').chmod(0o444)
science=('models/compression.py','models/student.py','training/trainer.py','training/losses.py',
         'training/compression_loss.py','data/pseudo_labeled.py','data/sen1floods11.py','config.py','scripts/compression_jobs.py')
legacy=V2/'frozen/140df39f28974301'
assert all(q.file_hash(capsule/name)==q.file_hash(legacy/name) for name in science), 'Scientific source mismatch prevents reuse'
base_spec=read(V2/'pilot_final/specs/A0_FP_s42.json')
jobs=[];reused=[]
for graph in ('A0','A1','A2','A3','X1','X2'):
    for seed in (42,64,91,107):
        name=f'{graph}_FP_s{seed}'
        if seed==42 and graph in ('A0','A1','A3'):
            r=read(V2/'results'/f'{name}.json')
            assert r['completed_updates']==7800 and q.file_hash(r['best_path'])==r['best_sha256']
            assert q.file_hash(r['last_path'])==r['last_sha256']
            q.atomic_json(BASE/'results'/f'{name}.json',{**r,'reused_compatible_v2':True,
                'original_receipt':str(V2/'results'/f'{name}.json'),'original_receipt_sha256':q.file_hash(V2/'results'/f'{name}.json')})
            reused.append(name);continue
        spec={**base_spec,'graph':graph,'seed':seed,'run_dir':str(Path('/data1/schmalstieg/flooddistill/revision_20261005/capacity_v4_20261008')/name)}
        sp=directory/'specs'/f'{name}.json';q.atomic_json(sp,spec);sp.chmod(0o444)
        out=BASE/'results'/f'{name}.json'
        jobs.append({'id':name,'kind':'command','cwd':str(capsule),'gpu':True,'deps':[],
            'max_seconds':1020,'shutdown_grace_seconds':20,'required_free_mb':20000,
            'argv':[sys.executable,str(version/'stage2_fp.py'),'--capsule',str(capsule),'--spec',str(sp),
                    '--out',str(out),'--identity-manifest',str(BASE/'shared_input_identity.json'),
                    '--graph-declarations',str(BASE/'stage2_graphs_declared.json')],
            'inputs':[pin(p) for p in (version/'stage2_fp.py',sp,BASE/'shared_input_identity.json',BASE/'protocol_v4.yaml',BASE/'stage2_graphs_declared.json')],
            'outputs':[{'path':str(out),'type':'json','equals':{'status':'complete','outcome':'completed','completed_updates':7800},
                        'required_keys':['best_path','best_sha256','last_path','selected_updates','source_validation']}]})
template.update(jobs=jobs,gpu_budget_seconds=21600,stage='2 FP family,4seeds',
    scope='Six fixed graphs x4seeds;3 exact same-source/recipe/seed42 warmups reused;21 new finite runs',
    reused=reused,forecast_basis='v2 measured739.8/831.2/861.9 command seconds per FP warmup;1020s hard timeout',
    later_stages='not yet queued; Stage3 waits for Stage2 anchors')
q.atomic_json(directory/'manifest.json',template);(directory/'manifest.json').chmod(0o444)
validated=q.load_manifest(directory/'manifest.json')
q.initialize(directory/'manifest.json',directory/'manifest.state.json',validated)
result=q.start(directory/'manifest.json',directory/'manifest.state.json')
q.atomic_json(BASE/'stage2_launch_receipt.json',{'status':'launched','launch':result,
    'manifest':str(directory/'manifest.json'),'manifest_sha256':q.file_hash(directory/'manifest.json'),
    'protocol_sha256':q.file_hash(BASE/'protocol_v4.yaml'),'new_runs':len(jobs),'reused':reused,
    'reserved_gpu_seconds':sum(j['max_seconds'] for j in jobs),'no_Stage2_results_read_before_freeze':True})
q.atomic_json(BASE/'stage1_completion.json',{'status':'complete','outcome':'completed','next_stage':'2 launched',
    'protocol_sha256':q.file_hash(BASE/'protocol_v4.yaml'),'stage1_note':str(BASE/'stage1_note.txt')})
