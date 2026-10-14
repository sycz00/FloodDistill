"""Validated finite stage adapter. Scientific routines remain in their existing modules."""
import sys,time,subprocess,traceback
import common10 as c

def main():
    name=sys.argv[1];job=c.read(c.ROOT/'manifest.json')['jobs'][name];action=job['action'];result={'outcome':'completed'}
    if action=='train':
        command=[c.PYTHON,str(c.ROOT/'code/train10.py'),job['training']]+(['--smoke'] if job.get('smoke') else [])
        subprocess.run(command,check=True)
        path=c.ROOT/('smoke_receipts' if job.get('smoke') else 'scientific_receipts')/(job['training']+'.json')
        result=c.read(path);assert result and result['outcome']=='completed'
        assert c.sha(result['best_path'])==result['best_sha256'] and c.sha(result['last_path'])==result['last_sha256']
        result=dict(outcome='completed',scientific_receipt=str(path),sha256=c.sha(path))
    elif action=='parity':
        graph=job['graph'];a=c.read(c.ROOT/'smoke_receipts'/f'{graph}_CE_s42.json');b=c.read(c.ROOT/'smoke_receipts'/f'{graph}_KDoff_s42.json')
        assert a and b and a['initial_model_sha256']==b['initial_model_sha256'],'CE/KD initialization differs'
        assert a['first_image_hashes']==b['first_image_hashes'],'CE/KD shared cohort images differ'
        result.update(graph=graph,identical_initialization=True,identical_first_images=True)
    elif action=='SUP_parity':
        rows=[c.read(c.ROOT/'smoke_receipts'/f'{g}_SUP_s42.json') for g in ['R2','R3','R4_micro','R4']]
        assert all(rows);assert all(x['first_image_hashes']==rows[0]['first_image_hashes'] for x in rows),'Shared SUP cohort images differ'
        result.update(identical_first_images=True)
    elif action=='QAT_parity':
        prefix=job['prefix'];rows=[c.read(c.ROOT/'smoke_receipts'/(prefix+'_'+m+'_s42.json')) for m in ['d0','d1','d2']]
        assert all(rows);assert all(x['initial_model_sha256']==rows[0]['initial_model_sha256'] and x['first_image_hashes']==rows[0]['first_image_hashes'] for x in rows),'QAT initialization/images differ'
        result.update(identical_initialization=True,identical_first_images=True)
    elif action=='cache':
        import teacher_cache10 as module;module.main();result=c.read(c.ROOT/'teacher_logit_cache_manifest.json');assert result['outcome']=='completed' and len(result['files'])==2500
        assert all(x['maximum_geometric_target_difference']<.001 for x in result['fresh_forward_checks'])
    elif action in ['inventory','source','smallest','frontier','matched','freeze']:
        import stages10 as s
        if action=='source':s.source_all(job['slot']);result.update(slot=job['slot'])
        else:getattr(s,{'inventory':'inventory','smallest':'choose_smallest','frontier':'frontier','matched':'matched_points','freeze':'final_freeze'}[action])()
    elif action=='cost':
        import cost10;cost10.main()
    elif action=='controls':
        import controls10;controls10.main();result=c.read(c.ROOT/'compression_control_selection.json')
    elif action=='evaluate':
        import evaluate10 as e;e.evaluate(job['mode'],job['slot'],job.get('phase'))
        folder=job['mode']+'_'+job['phase'] if job.get('phase') else job['mode'];path=c.ROOT/'metrics'/folder/f"slot{job['slot']}.json";result=c.read(path)
        assert result['outcome']=='completed';result=dict(outcome='completed',metric_receipt=str(path),sha256=c.sha(path),windows=result['windows'])
    elif action=='GPU_preflight':
        import preflight_gpu10 as check;check.main();result=c.read(c.ROOT/'GPU_preflight.json');assert result['outcome']=='completed'
    elif action=='gpu':
        import gpu10;gpu10.main();result=c.read(c.ROOT/'gpu_inference_receipt.json');assert result
        # Optional GPU failure has an explicit CPU fallback, never blocks final labels.
        result=dict(outcome='completed',optional_gpu_outcome=result['outcome'],GPU_receipt=str(c.ROOT/'gpu_inference_receipt.json'))
    elif action in ['targets','decode','ort']:
        import optional10;getattr(optional10,action)()
    elif action=='collect':
        import collect10;collect10.main();result=c.read(c.ROOT/'collection_receipt.json');assert result
    else:raise ValueError(action)
    result.update(job=name,finished_at=time.time());c.save(c.ROOT/'stage_results'/(name+'.json'),result)
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
