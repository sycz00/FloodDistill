#!/usr/bin/env python3
"""Read historical artifacts without changing their capsules or derived analysis.

Run in the installed esi environment. All writes are confined to --out-dir.
Only the historical queue's guarded, stdlib-only parser is imported; training,
preparation and historical analysis entry points are never executed.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
CACHE = {}
SEEDS = (42, 64, 91, 107)
T95_DF3 = 3.182446305284263


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    p = Path(path).resolve()
    key = str(p)
    if key not in CACHE:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        CACHE[key] = h.hexdigest()
    return CACHE[key]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def check(path, expected):
    p = Path(path)
    actual = sha(p) if p.is_file() else None
    return dict(path=str(p.resolve()), sha256=expected, actual_sha256=actual,
                match=actual == expected)


def interval(values):
    avg, sd = statistics.mean(values), statistics.stdev(values)
    half = T95_DF3 * sd / math.sqrt(len(values))
    return dict(n=len(values), mean_pp=avg, sample_sd_pp=sd,
                ci95_low_pp=avg-half, ci95_high_pp=avg+half,
                paired_seed_differences_pp=values,
                uncertainty='paired training seeds, df=3 Student-t')


def queue_parser(path):
    # Inspected source is stdlib-only with a guarded main. Suppress bytecode writes.
    spec = importlib.util.spec_from_file_location('v7_readonly_queue_' + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget(base, out):
    archive = read(base/'audit_20261011/t0_verification.json')
    audit_time = archive['checked_at_utc']
    manifests = sorted(base.glob('*/manifest.json'))
    # Include newly created capacity queues since the audited baseline, but leave
    # this v7 run to its single live ledger (which adds the baseline once).
    for directory in sorted(base.parent.glob('capacity_*')):
        if directory.resolve() in (base.resolve(), out.resolve()):
            continue
        manifests += sorted(directory.rglob('manifest.json'))
    attempts, queues, unresolved = [], {}, []
    execution = {}
    for mp in manifests:
        sp = mp.with_suffix('.state.json')
        if not sp.is_file():
            continue
        m, s = read(mp), read(sp)
        if not isinstance(m.get('jobs'), list) or not isinstance(s.get('jobs'), dict):
            continue
        historical = mp.is_relative_to(base)
        ids = {j['id']: j for j in m['jobs']}
        qname = str(mp.parent.relative_to(base)) if historical else str(mp.parent)
        gpu_seconds = 0.0
        for jid, state in s['jobs'].items():
            job = ids.get(jid)
            if job is None:
                unresolved.append(f'{mp}: unknown job {jid}')
                continue
            for output in job.get('outputs', []):
                execution.setdefault(str(Path(output['path']).resolve()), []).append(
                    dict(manifest_path=str(mp.resolve()), job_id=jid, argv=job.get('argv'),
                         inputs=job.get('inputs', []), status=state.get('status')))
            if not job.get('gpu'):
                continue
            for attempt in state.get('attempts', []):
                if not historical and attempt.get('started_at', attempt.get('created_at', '')) <= audit_time:
                    continue
                elapsed = attempt.get('command_elapsed_seconds')
                if elapsed is None or not math.isfinite(float(elapsed)) or elapsed < 0:
                    unresolved.append(f'{mp}:{jid}: GPU attempt has no finite terminal elapsed')
                    continue
                row = {k: attempt.get(k) for k in ('number','token','created_at','started_at',
                    'finished_at','exit_code','receipt_path','gpu_device','command_elapsed_seconds')}
                row.update(queue=qname, job_id=jid, state_status=state.get('status'))
                receipt_path = attempt.get('receipt_path')
                if receipt_path and Path(receipt_path).is_file():
                    receipt = read(receipt_path)
                    row['attempt_receipt_sha256'] = sha(receipt_path)
                    row['state_receipt_elapsed_match'] = receipt.get('command_elapsed_seconds') == elapsed
                    if not row['state_receipt_elapsed_match']:
                        unresolved.append(f'{receipt_path}: state/receipt elapsed mismatch')
                attempts.append(row)
                gpu_seconds += elapsed
        canonical_match = None
        runner = mp.parent/'revision_queue.py'
        if historical and runner.is_file():
            parser = queue_parser(runner)
            canonical = parser.digest(parser.load_manifest(str(mp)))
            canonical_match = canonical == s.get('manifest_sha256')
            if not canonical_match:
                unresolved.append(f'{mp}: normalized manifest identity mismatch')
        queues[qname] = dict(manifest_path=str(mp.resolve()), manifest_sha256=sha(mp),
                            state_path=str(sp.resolve()), state_sha256=sha(sp),
                            canonical_manifest_match=canonical_match,
                            worker_status=s.get('worker_status'),
                            counts=dict(collections.Counter(j.get('status') for j in s['jobs'].values())),
                            gpu_command_seconds=gpu_seconds)
        if any(j.get('status') in ('running','launching') for j in s['jobs'].values()):
            unresolved.append(f'{mp}: active historical GPU/queue work must be reconciled')
    # Queue state, rather than scientific receipt elapsed, is the unit of spend:
    # it includes failures, startup, reuse-tagged GPU commands and inference.
    total = sum(q['gpu_command_seconds'] for q in queues.values())
    cap = max(0.0, min(8 * 3600, 30 * 3600 - total))
    result = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        audited_baseline_at_utc=audit_time, unit='GPU-tagged command elapsed seconds; not utilization',
        cumulative_gpu_seconds=total, cumulative_gpu_hours=total/3600,
        archived_gpu_hours=21.332773819681574,
        new_since_audit_gpu_seconds=sum(a['command_elapsed_seconds'] for a in attempts
            if (a.get('started_at') or a.get('created_at') or '') > audit_time),
        incremental_cap_seconds=cap, incremental_cap_hours=cap/3600,
        programme_cap_seconds=30*3600, failures_counted=sum(a.get('exit_code') not in (0,None) for a in attempts),
        queues=queues, attempts=attempts, unresolved=unresolved, verified=not unresolved)
    write(out/'historical_budget.json', result)
    return result, execution


def checkpoint_metadata(path, expected, model):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    cfg = payload['config']
    checks = {k: cfg.get(k) == v for k,v in expected.items()}
    model.load_state_dict(payload['model'], strict=True)
    return dict(config=cfg, config_checks=checks,
                initial_model_sha256=payload.get('initial_model_sha256'),
                loss_policy=payload.get('loss_policy'), training_state=payload.get('training_state'),
                scheduler=payload.get('scheduler'), metrics=payload.get('metrics'),
                optimizer_groups=[{k:v for k,v in group.items() if k != 'params'}
                                  for group in payload['optimizer']['param_groups']],
                graph_strict_load=True, parameters=sum(p.numel() for p in model.parameters()),
                graph_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                graph_buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
                epoch=payload.get('epoch'), resume_support=payload.get('resume_support'),
                checks_pass=all(checks.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--budget-only', action='store_true')
    args = parser.parse_args()
    base, out = Path(args.base).resolve(), Path(args.out_dir).resolve()
    if base == out or out.is_relative_to(base):
        raise ValueError('New output must be outside the historical run')
    out.mkdir(parents=True, exist_ok=True)
    ledger, execution = budget(base, out)
    if args.budget_only:
        print(json.dumps({k:ledger[k] for k in ('cumulative_gpu_hours','incremental_cap_hours','verified')}))
        return
    capsule = base/'frozen/297031f5f4ae1df5'
    identity = read(capsule/'source_identity.json')
    capsule_checks = [check(capsule/rel, h) for rel,h in identity['files'].items()]
    unlisted = sorted(str(p.relative_to(capsule)) for p in capsule.rglob('*')
        if p.is_file() and p.name!='source_identity.json' and '__pycache__' not in p.parts
        and str(p.relative_to(capsule)) not in identity['files'])
    shared = read(base/'shared_input_identity.json')
    shared_checks = [check(r['path'], r['sha256']) for r in shared['files']]
    artifact_checks = []
    for queue in ledger['queues'].values():
        for job in read(queue['manifest_path']).get('jobs', []):
            artifact_checks += [check(r['path'],r['sha256']) for r in job.get('inputs', [])]
    adoption = read(base/'simulator_adoption.json')
    simulator_check = check(adoption['source_file'], adoption['source_sha256'])
    original_spec = base/'stage2_fp/specs/A3_FP_s64.json'
    spec = read(original_spec)
    # Resolve path fields without changing the inherited bytes or interpreting
    # inherited validation-calibration IDs as the new TRAIN calibration policy.
    for key in ('pseudo','index','teacher','overlap_plan'):
        spec[key] = str(Path(spec[key]).resolve())
    spec['roots'] = {k:str(Path(v).resolve()) for k,v in spec['roots'].items()}
    spec.update(seed=42, run_dir=str(out/'runs'), workers=2)
    write(out/'resolved_spec.json', spec)
    cohort_receipt_path = Path(spec['index']).parent/'original_N2500_receipt.json'
    cohort = read(cohort_receipt_path)
    cohort_digest = digest({'index_sha256':sha(spec['index']), 'files':cohort['files']})
    cohort_pass = (cohort_digest == shared['cohort_sha256'] == cohort['dataset_sha256'] and
                   cohort['n_selected'] == 2500 and cohort['fps_replacements'] == 0)
    source_files = ['protocol_v4.yaml','protocol_v5_addendum.yaml','simulator_adoption.json',
                    'code/v5/train_v5c.py','code/v5/rungs.py','code/v5/terminal_v5.py',
                    'v5c/revision_queue.py','report_v5_codex_audit_20261011.md',
                    'audit_20261011/t0_verification.json','audit_20261011/report_completion_receipt.json',
                    'audit_20261011/supplement_summary.json','audit_20261011/selected_and_final_source.csv',
                    'v5/analysis/summary.json','v5/analysis/cells.csv']
    source_artifacts = [dict(path=str(base/rel),sha256=sha(base/rel)) for rel in source_files]
    numeric = []
    def value(rel):
        r = read(base/rel)
        if 'source_validation' in r:
            return r['source_validation']['best_val_mIoU']
        return r['validation']['selected']['micro']['mIoU']
    pairs = [('A3 LR repair', lambda s:value(f'v5/results/fl_A3_sel_s{s}.json')-
                               value(f'results/A3_FP_s{s}.json'),4.14),
             ('R4 selected LKD - A3 selected LKD', lambda s:value(f'v5/results/fl_R4_sel_s{s}.json')-
                               value(f'v5/results/fl_A3_sel_s{s}.json'),1.37),
             ('Q48 - F48', lambda s:value(f'v5/results/lb_Q48_A3_s{s}.json')-
                               value(f'v5/results/lb_F48_A3_s{s}.json'),-.25)]
    for name,fn,target in pairs:
        result=interval([100*fn(s) for s in SEEDS])
        numeric.append(dict(contrast=name,expected_rounded_pp=target,
                            within_0_01_pp=abs(result['mean_pp']-target)<=.01,**result))
    # Import only the model definitions after their pinned source hashes pass.
    common_pass = (not unlisted and all(r['match'] for r in capsule_checks+shared_checks+artifact_checks)
                   and simulator_check['match'] and cohort_pass and ledger['verified'])
    records=[]
    errors=[]
    if common_pass:
        inherited_pythonpath = read(base/'stage2_fp/manifest.json').get('env', {}).get('PYTHONPATH', '')
        for dependency_path in reversed(inherited_pythonpath.split(os.pathsep)):
            if dependency_path:
                sys.path.insert(0, dependency_path)
        sys.path.insert(0,str(capsule));sys.path.insert(0,str(base/'code/v5'))
        import rungs
        expected_cfg=dict(epochs=50,weight_decay=.05,task_loss='focal',gamma=2.,ignore_index=-1,
            alpha=.5,temperature=4.,use_amp=False,grad_clip=5.,scheduler='cosine',
            warmup_epochs=0,val_every_n_epochs=2,val_smoothing_window=1)
        baselines=[('A0_LKD_recipe','A0','results/A0_FP_s{seed}.json',5e-5),
                   ('R4_LKD_selected','R4','v5/results/fl_R4_sel_s{seed}.json',.005),
                   ('A3_LKD_selected','A3','v5/results/fl_A3_sel_s{seed}.json',.005),
                   ('A3N_LKD_selected','A3N','v5/results/fl_A3N_sel_s{seed}.json',.005)]
        for family,graph,template,lr in baselines:
            selection=base/f'v5/selections/lr_{graph}.json'
            if graph!='A0' and read(selection)['lr']!=lr:
                errors.append(f'{graph}: LR selection differs')
            model=rungs.build(graph)
            for seed in SEEDS:
                rp=base/template.format(seed=seed);receipt=read(rp)
                tp=base/'v5/terminal'/rp.name; terminal=read(tp)
                checks=[check(receipt[p+'_path'],receipt[p+'_sha256']) for p in ('best','last')]
                try:
                    metadata={p:checkpoint_metadata(receipt[p+'_path'],dict(expected_cfg,seed=seed,lr=lr),model)
                              for p in ('best','last')}
                    passed=all(c['match'] for c in checks) and all(x['checks_pass'] for x in metadata.values())
                    passed &= metadata['last']['training_state']['global_step']==receipt['completed_updates']==7800
                    passed &= metadata['best']['training_state']['global_step']==receipt['selected_updates']
                    passed &= abs(metadata['best']['metrics']['mIoU']-receipt['source_validation']['best_val_mIoU'])<1e-7
                    passed &= terminal['checkpoint_sha256']==receipt['best_sha256'] and terminal['graph']==graph
                    counts={k:len(v['per_image_counts']) for k,v in terminal['terminal_results'].items()}
                    passed &= counts==dict(s11=90,sturm=2675,wf=785)
                    for dataset in terminal['terminal_results'].values():
                        if any(len(c)!=2 or any(len(row)!=3 or any(not isinstance(x,int) or x<0 for x in row)
                                                for row in c) for c in dataset['per_image_counts'].values()):
                            passed=False
                    if not passed: errors.append(f'{family} seed {seed}: scientific identity failed')
                    source=receipt.get('reused',str(rp))
                    record=dict(status='reused' if passed else 'blocked',verified=bool(passed),family=family,
                        graph=graph,method='lkd',seed=seed,lr=lr,receipt_path=str(rp),receipt_sha256=sha(rp),
                        execution_receipt_path=source,execution_receipt_sha256=sha(source),
                        best_path=receipt['best_path'],best_sha256=receipt['best_sha256'],
                        last_path=receipt['last_path'],last_sha256=receipt['last_sha256'],
                        selected_updates=receipt['selected_updates'],completed_updates=receipt['completed_updates'],
                        source_validation=receipt['source_validation'],
                        selected_source_metrics=metadata['best']['metrics'],final_source_metrics=metadata['last']['metrics'],
                        terminal_receipt_path=str(tp),terminal_receipt_sha256=sha(tp),
                        terminal_alias_of=terminal.get('alias_of'),per_image_counts_location='terminal_results.{dataset}.per_image_counts',
                        per_image_counts_n=counts,per_image_csv=None,metadata=metadata,checkpoint_checks=checks,
                        execution_lineage=execution.get(str(Path(source).resolve()),[]),
                        normalization_code_sha256=sha(capsule/'data/sen1floods11.py'),
                        pseudo_augmentation_code_sha256=sha(capsule/'data/pseudo_labeled.py'),
                        graph_code_sha256=sha(base/'code/v5/rungs.py'),loss_code_sha256=sha(capsule/'training/losses.py'),
                        source_selection='S11 validation pooled mIoU, strict greater, earlier tie, every312 updates',
                        update_schedule='50 epoch-stepped cosine epochs x156 updates, AdamW, no warmup',
                        selected_checkpoint_external_status='reused',final_checkpoint_external_status='not_run')
                    records.append(record)
                except Exception as exc:
                    errors.append(f'{family} seed {seed}: {type(exc).__name__}: {exc}')
    else:
        errors.append('Shared pinned scientific input verification failed; checkpoint branches blocked')
    unique=len({r['best_sha256'] for r in records if r['verified']})
    reuse=dict(schema_version=1,records=records,independent_seed_set=list(SEEDS),
               distinct_selected_checkpoint_hashes=unique,terminal_aliases_not_independent=True,
               shared_inputs_verified=common_pass)
    write(out/'reuse_index.json',reuse)
    archived=read(base/'audit_20261011/t0_verification.json')
    audit=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='complete' if common_pass and not errors and all(x['within_0_01_pp'] for x in numeric) else 'blocked',
        read_only_historical_audit=True,capsule=str(capsule),capsule_source_identity_sha256=sha(capsule/'source_identity.json'),
        capsule_tree_digest=digest(identity['files']),capsule_tree_digest_match=digest(identity['files'])==identity['sha256'],
        capsule_checks=capsule_checks,capsule_unlisted_files=unlisted,shared_input_checks=shared_checks,
        queue_input_checks=artifact_checks,simulator_check=simulator_check,
        source_artifacts=source_artifacts,resolved_spec_path=str(out/'resolved_spec.json'),
        resolved_spec_sha256=sha(out/'resolved_spec.json'),inherited_spec_path=str(original_spec),
        inherited_spec_sha256=sha(original_spec),teacher_sha256=spec['teacher_sha256'],
        cohort_sha256=cohort_digest,cohort_verified=cohort_pass,cohort_receipt_path=str(cohort_receipt_path),
        cohort_provenance_limits='Original selected IDs restored, zero replacements; copied cache historical hashes and generation-time checkpoint receipt were unavailable',
        bands=['B2','B3','B4','B8A','B11','B12'],normalization_mean=[.1413,.1380,.1235,.3090,.2045,.1191],
        normalization_std=[.0741,.0737,.0869,.1180,.0977,.0766],
        input_storage='Already S11-normalized six-band float32 TRAIN tensors; do not normalize twice',
        numeric_checks=numeric,historical_budget_path=str(out/'historical_budget.json'),
        historical_budget_sha256=sha(out/'historical_budget.json'),
        cumulative_gpu_hours=ledger['cumulative_gpu_hours'],incremental_cap_hours=ledger['incremental_cap_hours'],
        reused_checkpoint_records=len(records),verified_checkpoint_records=sum(r['verified'] for r in records),
        historical_terminal_receipt_count=archived['terminal_receipt_count'],
        historical_early_stop_exceptions=archived['early_stops'],errors=errors)
    write(out/'provenance.json',audit)
    print(json.dumps({k:audit[k] for k in ('status','cumulative_gpu_hours','incremental_cap_hours',
                    'reused_checkpoint_records','verified_checkpoint_records','errors')}))
    for result in numeric:
        print(result['contrast'],result['mean_pp'],result['ci95_low_pp'],result['ci95_high_pp'])
    if audit['status']!='complete':sys.exit(1)


if __name__=='__main__':
    main()
