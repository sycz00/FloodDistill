"""Freeze the finite v7 protocol and initial bundle templates after smoke gates."""
import argparse
import json
import math
from pathlib import Path
import sys
import audited_queue as q

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    spec=q.read_json(ROOT/'training_spec.json');idx=q.read_json(ROOT/'reuse_index.json')['records']
    a1=q.read_json(ROOT/'optional_A1_reuse_index.json') if (ROOT/'optional_A1_reuse_index.json').exists() else None
    budget=q.read_json(ROOT/'historical_budget.json');py='/home/schmalstieg/miniconda3/envs/esi/bin/python'
    capsule=spec['capsule'];rungs=spec['rungs_dir'];repo=str(ROOT.parents[2]);seeds=[42,64,91,107]
    env={'PYTHONPATH':'/home/schmalstieg/repos/_deps/efficientvit:/home/schmalstieg/repos/_deps/flooddistill_revision_python',
         'NO_ALBUMENTATIONS_UPDATE':'1','WANDB_MODE':'disabled','HF_HUB_OFFLINE':'1',
         'MPLCONFIGDIR':str(ROOT/'mpl_cache')}
    def common(name,bundle,role,argv,seconds,receipt):
        return {'id':name,'bundle':bundle,'role':role,'gpu':True,'cwd':repo,'env':env,'argv':argv,
                'max_seconds':seconds,'required_free_mb':20000,'deps':[],'receipt':str(receipt),
                'outputs':[{'path':str(receipt),'type':'json','equals':{'status':'complete'}}]}
    def train(name,bundle,graph,method,seed,lr,seconds):
        receipt=ROOT/'receipts'/(name+'.json')
        argv=[py,str(ROOT/'train_v7.py'),'--capsule',capsule,'--rungs-dir',rungs,'--spec',str(ROOT/'training_spec.json'),
              '--out',str(receipt),'--run-dir',str(ROOT/'runs'/name),'--graph',graph,'--method',method,
              '--seed',str(seed),'--lr',str(lr)]
        if method in ['C0','C1','C2','C3','C4','C5','HC','HL']:
            parent=next(r for r in idx if r['graph']=='R4' and r['seed']==seed)
            argv+=['--init-checkpoint',parent['best_path'],'--init-sha256',parent['best_sha256']]
        job=common(name,bundle,'train',argv,seconds,receipt)
        job.update(graph=graph,method=method,arm=method,seed=seed,lr=lr,
                   required_updates=1560 if method[0]=='C' or method in ['HC','HL'] else 7800)
        job['outputs'][0]['equals'].update(outcome='completed',completed_updates=job['required_updates'])
        job['outputs'][0]['finite']=['source_validation.best_val_mIoU','gradient_checks.max_post_clip_gradient_l2']
        job['outputs'][0]['ranges']={'source_validation.best_val_mIoU':[0,1]}
        return job
    diag=common('TRAIN_target_diagnostic','diagnostics','diagnostic',
        [py,str(ROOT/'diagnostics_v7.py'),'--capsule',capsule,'--spec',str(ROOT/'training_spec.json'),
         '--out',str(ROOT/'diagnostics.json'),'--max-train','128','--max-val','89','--seed','1729'],400,ROOT/'diagnostics.json')
    diag['outputs'][0]['finite']=['train.cached_live_augmented_disagreement_rate']
    bundles=[{'id':'diagnostics','jobs':[diag]},
             {'id':'A0_grid','future_reserve_seconds':4200,'jobs':[
                 train('A0_lr5e-4_s42','A0_grid','A0','lkd',42,5e-4,1400),
                 train('A0_lr5e-3_s42','A0_grid','A0','lkd',42,5e-3,1400)]},
             {'id':'A0_reanchor','condition':'A0_selected_lr','jobs':[
                 train(f'A0_selected_s{s}','A0_reanchor','A0','lkd',s,5e-3,1400) for s in seeds[1:]]},
             {'id':'R4_CE','jobs':[train(f'R4_CE_s{s}','R4_CE','R4','ce',s,5e-3,700) for s in seeds]},
             {'id':'pilot','jobs':[train(f'{arm}_s{s}','pilot','R4',arm,s,5e-4,450)
                                    for s in [42,64] for arm in ['C0','C1','C2','C3','C4','C5']]},
             {'id':'confirmation','condition':'C3_promotion','jobs':[
                 train(f'{arm}_s{s}','confirmation','R4',arm,s,5e-4,450)
                 for s in [91,107] for arm in ['C0','C3','C4','C5']]},
             {'id':'target_refresh','condition':'target_refresh','jobs':[
                 train(f'{arm}_s{s}','target_refresh','R4',arm,s,5e-4,450)
                 for s in [42,64] for arm in ['HC','HL']]},
             {'id':'optional_A3_CE','jobs':[train(f'A3_CE_optimized_s{s}','optional_A3_CE','A3','ce',s,5e-3,700)
                                         for s in seeds]}]
    ev=[py,str(ROOT/'evaluate_v7.py'),'--run-dir',str(ROOT),'--capsule',capsule,'--rungs-dir',rungs,
        '--spec',str(ROOT/'training_spec.json')]
    regular=common('terminal_selected','collection','eval',ev+['--out',str(ROOT/'receipts/terminal_selected.json'),'--receipts'],1000,ROOT/'receipts/terminal_selected.json')
    reference=common('reference_cache','collection','reference',ev+['--prepare-cache','--out',str(ROOT/'receipts/reference_cache.json')],700,ROOT/'receipts/reference_cache.json')
    mechanism=common('terminal_mechanisms','collection','eval',ev+['--mechanisms','--out',str(ROOT/'receipts/terminal_mechanisms.json'),'--receipts'],1000,ROOT/'receipts/terminal_mechanisms.json')
    mechanism['deps']=['reference_cache']
    deploy=common('deployment','collection','deploy',[py,str(ROOT/'deploy_v7.py'),'--run-dir',str(ROOT),
        '--capsule',capsule,'--rungs-dir',rungs,'--spec',str(ROOT/'training_spec.json'),
        '--models',str(ROOT/'reuse_index.json'),'--out',str(ROOT/'receipts/deployment.json')],900,ROOT/'receipts/deployment.json')
    if a1 and a1.get('status')=='complete' and not a1.get('errors') and all(rec['verified'] for rec in a1['records']):
        bundles.extend([
            {'id':'optional_A1_grid','future_reserve_seconds':4200,'jobs':[
                train('A1_lr5e-4_s42','optional_A1_grid','A1','lkd',42,5e-4,1400),
                train('A1_lr5e-3_s42','optional_A1_grid','A1','lkd',42,5e-3,1400)]},
            {'id':'optional_A1_reanchor','condition':'A1_selected_lr','jobs':[
                train(f'A1_selected_s{s}','optional_A1_reanchor','A1','lkd',s,5e-3,1400) for s in seeds[1:]]}])
    bundles.append({'id':'collection','condition':'completed_terminal','jobs':[regular,reference,mechanism,deploy]})
    # Reused selected anchors receive same-pixel mechanisms too; historical counts
    # remain the original reuse source and are checked before merging these readouts.
    reused_models=[{'id':Path(rec['receipt_path']).stem+'__selected','graph':rec['graph'],'method':rec['method'],
                   'seed':rec['seed'],'checkpoint':rec['best_path'],'sha256':rec['best_sha256']}
                  for rec in idx if rec['graph'] in ['R4','A3N','A3']]
    q.atomic_json(ROOT/'reuse_models.json',reused_models)
    regular['argv'][-1:-1]=['--models',str(ROOT/'reuse_models.json')]
    mechanism['argv'][-1:-1]=['--models',str(ROOT/'reuse_models.json')]
    prompt=ROOT.parents[2]/'prompts/prompt_v7_overnight.md'
    protocol={'version':'v7','frozen_before_v7_scientific_training':True,'source_prompt':str(prompt),
        'source_prompt_sha256':q.file_hash(prompt),'hypotheses':['A0 competent LR anchor','R4 cached focal CE vs live-soft LKD at common5e-3',
        'R4 additive spectral rule x disagreement factorial and controls, prospective continuation'],
        'data_roles':{'training_calibration_diagnostic_weight_design':'TRAIN only',
            'checkpoint_LR_promotion':'S11 validation only','external':'research-exposed S11test/STURM/WF secondary readouts'},
        'cohort_sha256':q.read_json(ROOT/'provenance.json')['cohort_sha256'],'teacher_sha256':spec['teacher_sha256'],
        'band_order':['B2','B3','B4','B8A','B11','B12'],'normalization_mean':[.1413,.138,.1235,.309,.2045,.1191],
        'normalization_std':[.0741,.0737,.0869,.118,.0977,.0766],
        'shared_recipe':{'batch_size':16,'workers':2,'precision':'FP32','optimizer':'AdamW','weight_decay':.05,
            'gradient_clip':5,'validation_frequency_updates':312,'selection':'strict source mIoU, earliest checkpoint tie',
            'scratch_updates':7800,'continuation_updates':1560,'patience_disabled':True,
            'cosine':'inherited epoch-stepped,156 updates per epoch, T_max50 scratch or10 continuation',
            'focal_gamma':2,'teacher_constructed_before_student':True,'continuation_BN':'loaded inherited buffers,train in all arms'},
        'rule':{'available':'finite nonnegative green and SWIR1, sum>1e-6 reflectance','r':'clip((green-swir)/(green+swir),-1,1), zero unavailable',
            'R':'r>0','a':'8*tanh(alpha)','b':'8*tanh(beta)','initialization':'alpha=beta=0',
            'logits':'land=z_land-(a*r+b)/2, water=z_water+(a*r+b)/2','added_parameters':2},
        'continuation_loss':{'T':4,'direction':'KL(teacher||student)','formula':'0.5*focal(cached)+0.5*T^2*mean_valid(w*KL(p||q))',
            'u':'1+2*disagreement','w':'u/mean_valid(u)','empty_valid':'graph-connected zero',
            'C4':'within image,teacher class,confidence[.5,.7),[.7,.9),[.9,1], dedicated RNG permutes binary mask',
            'C5':'within image,teacher class, same emphasis count, highest entropy on valid available; flattened-index tie'},
        'arms':{'C0':[False,'uniform'],'C1':[True,'uniform'],'C2':[False,'disagreement'],'C3':[True,'disagreement'],
            'C4':[True,'shuffled_disagreement'],'C5':[True,'teacher_entropy'],
            'HC':[False,'onehot_cached'],'HL':[False,'onehot_live']},
        'pilot_seeds':[42,64],'confirmation_seeds':[91,107],'confirmation_arms':['C0','C3','C4','C5'],
        'promotion_pp':{'mean_C3_C0_min':.5,'each_C3_C0_min':0,'mean_C3_C4_min':.2,'mean_C3_C5_min':.2,'mean_water_C3_C0_min':-.5},
        'promotion_integrity':'all finite loss/gradients, paired base/logits/minibatch, focused label alignment, budget gate',
        'target_refresh_trigger':'>=1% cached-vs-live hard disagreement on valid augmented TRAIN',
        'LR_grid':[5e-5,5e-4,5e-3],'LR_tie':'smaller LR','seed_set':seeds,
        'statistics':{'four_seed_CI':'paired Student-t95%,sampleSD,df3','two_seed_pilot':'descriptive',
            'event_CI':'10,000 whole event bootstrap,seed1729,checkpoint conditional','empty_union':'undefined water IoU omitted from event mean'},
        'budget':{'historical_gpu_seconds':budget['cumulative_gpu_seconds'],'incremental_cap_seconds':budget['incremental_cap_seconds'],
            'wall_seconds':28800,'protected_collection_seconds':3600,'technical_retry_reserve_seconds':2700},
        'deliberate_amendments':['new runs disable inherited patience;exact7800','continuations start checkpoint included',
            'local tracking only','TRAIN-only new calibration;legacy spec validation ID key only verifies inherited source split'],
        'deferred':['A1 LR reanchor unless its full bundle fits released reservations after optimized A3 CE',
            'C1/C2 seeds91/107 not separately reserved','final external checkpoints not separately reserved',
            'QAT/OFA grids,U1,U6,R1,temporal claims,new graphs,new data'],
        'deployment':'actual TorchScript FP32/FP16, native INT8 if installed backend supports; separate from simulation',
        'bundles':[{k:v for k,v in b.items() if k!='jobs'} for b in bundles]}
    if a.freeze:
        if (ROOT/'manifest_initial.json').exists():raise RuntimeError('initial manifest already frozen; never overwrite')
        import yaml
        (ROOT/'protocol_v7.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
        smokes=q.read_json(ROOT/'preflight_ledger.json')
        preflight=sum(r['command_elapsed_seconds'] for r in smokes)
        frozen={str(path):q.file_hash(path) for path in ROOT.glob('*.py')}
        for path in [ROOT/'protocol_v7.yaml',ROOT/'training_spec.json',ROOT/'reuse_index.json',ROOT/'provenance.json',
                     ROOT/'historical_budget.json',ROOT/'preflight_checks.json',ROOT/'checks/focused_cpu.json',ROOT/'queue_checks.json',
                     ROOT/'recovery_checks.json',ROOT/'reuse_models.json',ROOT/'launch_budget_check.json']:
            frozen[str(path)]=q.file_hash(path)
        identity=q.read_json(Path(capsule)/'source_identity.json')
        frozen.update({str(Path(capsule)/n):h for n,h in identity['files'].items()})
        for path in (Path('/home/schmalstieg/repos/_deps/efficientvit/efficientvit')).rglob('*.py'):
            frozen[str(path)]=q.file_hash(path)
        frozen[str(Path(rungs)/'rungs.py')]=q.file_hash(Path(rungs)/'rungs.py')
        if a1 and a1.get('status')=='complete' and not a1.get('errors') and all(rec['verified'] for rec in a1['records']):
            frozen[str(ROOT/'optional_A1_reuse_index.json')]=q.file_hash(ROOT/'optional_A1_reuse_index.json')
        manifest={'schema_version':1,'experiment':'capacity_v7','identity_sha256':q.digest(protocol),
            'frozen_at':q.now(),'python':py,'gpu_index':0,'gpu_uuid':'GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc',
            'gpu_lock':'/data1/schmalstieg/flooddistill/revision_gpu.lock',
            'gpu_uuid_lock':'/data1/schmalstieg/flooddistill/revision_gpu_GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc.lock',
            'experiment_lock':str(ROOT.parent/('.v7_'+q.digest(protocol)+'.lock')),
            'headroom_mb':2048,'host_limits':{'disk_path':str(ROOT),'max_load_per_cpu':1.,'min_available_ram_gib':16,'min_free_disk_gib':12},
            'historical_gpu_seconds':budget['cumulative_gpu_seconds'],'incremental_cap_seconds':budget['incremental_cap_seconds'],
            'preflight_gpu_seconds':preflight,'collection_reserve_seconds':3600,'retry_reserve_seconds':2700,'wall_seconds':28800,
            'reuse_A0_seed42':next(r['receipt_path'] for r in idx if r['graph']=='A0' and r['seed']==42),
            'frozen_inputs':frozen,'bundles':bundles}
        if a1 and a1.get('status')=='complete' and not a1.get('errors') and all(rec['verified'] for rec in a1['records']):
            manifest['reuse_A1_seed42']=next(rec['receipt_path'] for rec in a1['records'] if rec['seed']==42)
        q.atomic_json(ROOT/'manifest_initial.json',manifest)
        q.atomic_json(ROOT/'freeze_receipt.json',{'at':q.now(),'manifest_sha256':q.file_hash(ROOT/'manifest_initial.json'),
            'normalized_manifest_sha256':q.digest(manifest),'frozen_inputs_count':len(frozen),
            'preflight_gpu_seconds':preflight,'incremental_remaining_seconds':budget['incremental_cap_seconds']-preflight})
        q.atomic_json(ROOT/'failure_ledger.json',[{'job':r['job']['id'],'phase':'preflight','attempt':r['number'],
            'exit_code':r['exit_code'],'elapsed_seconds':r['command_elapsed_seconds'],
            'reason':r.get('error') or 'legacy source split identity key omitted; restored for frozen loader verification',
            'log_path':r['log_path'],'scientific_run':False} for r in smokes if r['exit_code']!=0])
        starter=ROOT/'start_overnight.sh'
        starter.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+py+' '+str(ROOT/'run_overnight.py')+' start --run-dir '+str(ROOT)+'\n')
        starter.chmod(0o755)
    else:
        q.atomic_json(ROOT/'manifest_preview.json',{'bundles':bundles,'protocol':protocol})
    print(json.dumps({'frozen':a.freeze,'bundles':[b['id'] for b in bundles],'jobs':sum(len(b['jobs']) for b in bundles)}))


if __name__=='__main__':main()
