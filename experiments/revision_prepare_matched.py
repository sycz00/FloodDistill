"""Freeze prompt3's finite controls into the existing persistent queue."""
import copy
import datetime
import json
from pathlib import Path
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import revision_queue as q
from scripts.revision_queue_migrate import migrate

ROOT = Path('/data1/schmalstieg/flooddistill/revision_20261005')
COHORT = ROOT / 'regen_N2500_approved_20261005'
PY = '/home/schmalstieg/miniconda3/envs/esi/bin/python'
ARCHES = ['efficientvit_b0', 'mobilenet_v3_small', 'lightformer_resnet18']
SEEDS = [42, 64, 91, 107]


def cost_estimate(jobs, old_state, eval_jobs, training):
    """Measured completed durations plus explicit extrapolation for new paths."""
    observed = {}
    for name, record in old_state['jobs'].items():
        if not record['attempts']:
            continue
        attempt = record['attempts'][-1]
        if not attempt.get('started_at') or not attempt.get('finished_at'):
            continue
        seconds = (datetime.datetime.fromisoformat(attempt['finished_at']) -
                   datetime.datetime.fromisoformat(attempt['started_at'])).total_seconds()
        arch = next((a for a in ARCHES if a in name), 'comparator')
        category = 'training' if name.startswith('R_train_') else 'robustness' if name.startswith(('R_eval_', 'R_comparator_')) else 'other'
        objective = next((o for o in ['hard', 'logit', 'ofa'] if '_' + o + '_' in name), 'fixed')
        observed.setdefault((category, arch, objective), []).append(seconds)
    eval_durations = [v for (kind, _, _), values in observed.items() if kind == 'robustness' for v in values]
    robust_mean = statistics.mean(eval_durations)
    train_map = {t['run_id']: t for t in training}
    rows = []
    for job in jobs:
        if not job.get('gpu'):
            continue
        previous = old_state['jobs'].get(job['id'])
        if previous and (previous['status'] == 'succeeded' or previous.get('reason') == "'wf__g0'"):
            continue
        name = job['id']
        category, basis = 'additional', 'Boundary/validity adapter: model construction plus affected-input fraction; approximate until timed'
        seconds = 45.0
        if previous:
            category = 'original'
            arch = next((a for a in ARCHES if a in name), 'comparator')
            available = [v for (kind, a, _), values in observed.items() if kind == 'robustness' and a == arch for v in values]
            seconds = statistics.mean(available) if available else robust_mean
            basis = 'Mean completed canonical robustness duration for architecture; pooled observed fallback if this architecture has no completed readout'
        elif name in train_map:
            item = train_map[name]
            objective = 'hard' if item['objective'] == 'focal' else item['objective']
            values = observed[('training', item['architecture'], objective)]
            seconds = statistics.mean(values)
            basis = 'Mean current-cohort completed training duration, same architecture/objective family'
            if item['budget'] == 100:
                seconds = seconds * (6 / 156) + 120
                basis += '; N100 extrapolated by 6/156 updates plus 120s validation/checkpoint/load allowance'
        elif name in eval_jobs:
            entry = eval_jobs[name]
            seconds = robust_mean if len(entry['spec']['conditions']) > 1 else 85.0
            basis = 'Mean completed canonical full robustness' if len(entry['spec']['conditions']) > 1 else 'Clean-only readout: measured clean-cell duration plus model/load/checkpoint allowance'
        elif 'sturm_valid_' in name:
            spec = q.read_json(job['argv'][job['argv'].index('--spec') + 1])
            seconds = 15 + 1.5 * len(spec['conditions'])
        elif 'independent_dry_' in name:
            # Human-gated cost is kept separate from runnable GPU work.
            continue
        rows.append({'job': name, 'category': category, 'estimated_GPU_seconds': seconds, 'basis': basis})
    original = sum(r['estimated_GPU_seconds'] for r in rows if r['category'] == 'original') / 3600
    additional = sum(r['estimated_GPU_seconds'] for r in rows if r['category'] == 'additional') / 3600
    return {'captured_at': q.now(), 'original_remaining_GPU_hours': original,
            'additional_remaining_GPU_hours': additional, 'total_remaining_GPU_hours': original + additional,
            'ideal_two_GPU_wall_hours': (original + additional) / 2,
            'uncertainty': 'New paths extrapolated; allow roughly 25 percent variation plus foreign-work/resource waits. CPU serialized rule jobs and final collection add wall time. No sleep/reboot guarantee.',
            'observations': [{'kind': key[0], 'architecture': key[1], 'objective': key[2], 'n': len(v),
                              'mean_seconds': statistics.mean(v), 'min_seconds': min(v), 'max_seconds': max(v)} for key, v in sorted(observed.items())],
            'jobs': rows}


def prepare():
    pointer_path = ROOT / 'queue/active_queue.json'
    pointer = q.read_json(pointer_path)
    old_path = Path(pointer['active_manifest'])
    old_raw, old = q.read_json(old_path), q.load_manifest(old_path)
    old_state = q.read_json(old_path.with_name('manifest.state.json'))
    assert old_state['stop_requested'] and not q.alive(old_state.get('worker'))
    assert not any(r['status'] in {'running', 'launching'} for r in old_state['jobs'].values())
    loss_gate_path = REPO / 'experiments/revision/prompt3_loss_mask_gate.json'
    eval_gate_path = REPO / 'experiments/revision/prompt3_evaluation_gate.json'
    loss_gate, eval_gate = q.read_json(loss_gate_path), q.read_json(eval_gate_path)
    assert loss_gate['status'] == eval_gate['status'] == 'verified'
    # The actual mask audit permits reuse; never spend the conditional budget
    # merely to give an existing OFA run a new name.
    assert loss_gate['loss_mask_scan']['n2500_label_counts']['-1'] == 0
    assert loss_gate['conditional_T4_replacements_required'] == 0
    assert len(loss_gate['existing_ofa_reuse']) == 12
    old_snapshot = Path(old_raw['env']['PYTHONPATH'].split(':')[0])
    source = {str(p.relative_to(old_snapshot)): REPO / p.relative_to(old_snapshot)
              if (REPO / p.relative_to(old_snapshot)).is_file() else p
              for p in old_snapshot.rglob('*') if p.is_file() and p.name != 'source_identity.json'
              and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'}}
    additions = ['training/losses.py', 'experiments/revision_queue.py',
                 'experiments/revision_prepare_matched.py', 'scripts/revision_queue_migrate.py',
                 'evaluation/revision_collect.py', 'evaluation/revision_native_border.py',
                 'evaluation/revision_dry.py', 'scripts/revision_cohort_check.py',
                 'experiments/revision/prompt3_dry_reference.py',
                 'experiments/revision/prompt2_worldfloods_windows.json',
                 'experiments/revision/prompt3_evaluation_plan.json']
    for rel in additions:
        source[rel] = REPO / rel
    hashes = {rel: q.file_hash(p) for rel, p in sorted(source.items())}
    code_sha = q.digest(hashes)
    snapshot = ROOT / 'snapshots' / code_sha[:16]
    snapshot.mkdir(exist_ok=False)
    for rel, path in source.items():
        dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        dest.chmod(0o444)
    q.atomic_json(snapshot / 'source_identity.json', {'code_sha256': code_sha, 'files': hashes})
    queue_dir = COHORT / 'queue_prompt3'
    queue_dir.mkdir(exist_ok=False)
    spec_dir = queue_dir / 'specs'
    spec_dir.mkdir()
    wrapper = queue_dir / 'revision_queue.py'
    shutil.copyfile(REPO / 'experiments/revision_queue.py', wrapper)
    wrapper.chmod(0o444)
    receipts = {2500: COHORT / 'data/dataset_receipt.json', 100: COHORT / 'data/matched_N100_receipt.json'}
    datasets = {n: q.read_json(p) for n, p in receipts.items()}
    environment = copy.deepcopy(q.read_json(COHORT / 'environment.json'))
    environment.update(prompt='prompt3', source_sha256=code_sha,
                       parent_environment_sha256=q.file_hash(COHORT / 'environment.json'),
                       subset_N100_sha256=datasets[100]['dataset_sha256'])
    environment_path = queue_dir / 'environment.json'
    q.atomic_json(environment_path, environment)
    environment_path.chmod(0o444)
    env = {**old_raw['env'], 'PYTHONPATH': str(snapshot) + ':' + ':'.join(old_raw['env']['PYTHONPATH'].split(':')[1:])}
    recipe = copy.deepcopy(old_raw['fingerprints']['recipe'])
    recipe.update(status='Prompt3 predeclared matched controls',
                  grad_clip={'matched_focal': 5, 'matched_kd': 5, 'compatible_existing_ofa': 5},
                  main_kl_mask='target != -1; all approved pseudo-training pixels valid',
                  main_focal_gamma=2, focal_only_hard_coefficient=1,
                  teacher_initialization='Same teacher-before-student construction, including focal-only CPU initialization match',
                  OFA_reuse='12 current-cohort existing OFA; mask-equivalent on inspected data; historical initial hashes absent')
    fp = {**old_raw['fingerprints'], 'code': code_sha, 'environment': q.file_hash(environment_path), 'recipe': recipe}
    cache = {}
    def pin(path):
        path = Path(path).resolve()
        if str(path) not in cache:
            cache[str(path)] = q.file_hash(path)
        return {'path': str(path), 'sha256': cache[str(path)]}
    base = {}
    template = next(j for j in old_raw['jobs'] if j['id'] == 'R_train_efficientvit_b0_logit_N2500_s42')
    for item in template['inputs']:
        path = Path(item['path'])
        if path.is_relative_to(old_snapshot):
            path = snapshot / path.relative_to(old_snapshot)
        if path == COHORT / 'environment.json':
            path = environment_path
        base[str(path)] = pin(path)
    for path in [environment_path, loss_gate_path, eval_gate_path, *receipts.values(),
                 *[Path(d['index']) for d in datasets.values()],
                 *[snapshot / rel for rel in source]]:
        base[str(path)] = pin(path)
    base = list(base.values())
    jobs = copy.deepcopy(old_raw['jobs'])
    for job in jobs:
        job['env'] = {**old_raw['env'], **job.get('env', {})}
    def write_spec(name, data):
        path = spec_dir / (name + '.json')
        q.atomic_json(path, data)
        path.chmod(0o444)
        return path
    def command(name, script, args, outputs, deps=(), gpu=False, spec=None, fingerprints=None):
        job = {'id': name, 'kind': 'command', 'cwd': str(snapshot),
               'argv': ['/usr/bin/ionice', '-c', '2', '-n', '7', PY, str(snapshot / script), *map(str, args)],
               'env': copy.deepcopy(env), 'gpu': gpu, 'deps': list(deps), 'inputs': list(base),
               'outputs': outputs, 'fingerprints': fingerprints or fp}
        job['env']['PYTHONDONTWRITEBYTECODE'] = '1'
        if spec:
            job['inputs'].append(pin(spec))
        if gpu:
            job['required_free_mb'] = 16384
        else:
            job['env']['CUDA_VISIBLE_DEVICES'] = ''
        jobs.append(job)
        return job
    for name, path in [('P3_loss_mask_gate', loss_gate_path), ('P3_evaluation_geometry_gate', eval_gate_path)]:
        jobs.append({'id': name, 'kind': 'gate', 'cwd': str(snapshot), 'env': env,
                     'inputs': [pin(path)], 'outputs': [{'path': str(path), 'type': 'json',
                                                         'equals': {'status': 'verified'}, 'sha256': pin(path)['sha256']}]})
    command('P3_verify_N100', 'scripts/revision_cohort_check.py',
            ['data', '--receipt', receipts[100], '--out', queue_dir / 'N100_verified.json'],
            [{'path': str(queue_dir / 'N100_verified.json'), 'type': 'json', 'equals': {'status': 'verified', 'n_selected': 100}}])
    training, runs, first_ids = [], [], {}
    cells = [(arch, 'logit', 'matched_lkd_clip5_valid', 2500, 'T1') for arch in ARCHES]
    cells += [('efficientvit_b0', 'focal', 'matched_focal_g2_clip5', 2500, 'T2')]
    cells += [('efficientvit_b0', obj, variant, 100, 'T3') for obj, variant in
              [('logit', 'matched_lkd_clip5_valid'), ('ofa', 'matched_ofa_clip5_valid')]]
    for seed in SEEDS:
        for arch, objective, variant, budget, task in cells:
            name = f'P3_train_{arch}_{variant}_N{budget}_s{seed}'
            old_obj = 'hard' if objective == 'focal' else objective
            parent = q.read_json(COHORT / f'queue/specs/R_train_{arch}_{old_obj}_N2500_s{seed}.json')
            argv = copy.deepcopy(parent['argv'])
            argv[1] = str(snapshot / 'scripts/train.py')
            out = COHORT / 'training' / name
            def flag(key, value):
                if key in argv:
                    argv[argv.index(key) + 1] = str(value)
                else:
                    argv.extend([key, str(value)])
            for key, value in [('--n_samples', budget), ('--curated_index_file', datasets[budget]['index']),
                               ('--out_dir', out), ('--wandb_run_name', name), ('--grad_clip', 5.0)]:
                flag(key, value)
            if objective == 'focal':
                flag('--task_loss', 'focal'); flag('--gamma', 2.0)
                flag('--official_ckpt', old_raw['fingerprints'].get('teacher_path', parent.get('teacher_path', '')))
                if not argv[argv.index('--official_ckpt') + 1]:
                    teacher_parent = q.read_json(COHORT / f'queue/specs/R_train_{arch}_logit_N2500_s{seed}.json')['argv']
                    flag('--official_ckpt', teacher_parent[teacher_parent.index('--official_ckpt') + 1])
                argv.append('--match_teacher_initialization')
            else:
                argv.append('--kd_valid_only')
            expected = copy.deepcopy(parent['expected_config'])
            expected.update(out_dir=str(out), grad_clip=5.0)
            if objective == 'focal':
                expected.update(task_loss='focal', gamma=2.0)
            policy = {'main_hard': 'focal', 'main_focal_gamma': 2.0, 'grad_clip_norm': 5.0,
                      'main_kl_mask': None if objective == 'focal' else 'valid_only'}
            run_fp = {**fp, 'data': datasets[budget]['dataset_sha256']}
            metadata = {'run_id': name, 'architecture': arch, 'objective': objective, 'variant': variant,
                        'budget': budget, 'n_selected': budget, 'batch_size': 16, 'task': task,
                        'seed': seed, 'cohort': COHORT.name, 'dataset_receipt': str(receipts[budget]),
                        'argv': argv, 'out_dir': str(out), 'expected_config': expected,
                        'expected_loss_policy': policy, 'fingerprints': run_fp}
            spec = write_spec(name, metadata)
            deps = ['Q_dataset_verify', 'Q_evaluator_check', 'P3_loss_mask_gate']
            if budget == 100:
                deps.append('P3_verify_N100')
            path_key = (arch, objective, budget)
            if seed != 42:
                deps.append(first_ids[path_key])
            else:
                first_ids[path_key] = name
            outputs = [{'path': str(out / f), 'min_bytes': 1000} for f in ['best.pt', 'last.pt']]
            outputs += [{'path': str(out / 'verified.json'), 'type': 'json', 'equals': {'status': 'verified', 'run_id': name}}]
            command(name, 'scripts/revision_cohort_check.py', ['train', '--spec', spec, '--out', out / 'verified.json'],
                    outputs, deps, True, spec, run_fp)
            training.append(metadata)
    roots = q.read_json(COHORT / 'queue/specs/R_comparator_teacher.json')['roots']
    conditions = q.read_json(COHORT / 'queue/specs/R_comparator_teacher.json')['conditions']
    eval_jobs = {}
    for job in old_raw['jobs']:
        if job['id'].startswith(('R_eval_', 'R_comparator_')):
            s = q.read_json(Path(job['argv'][job['argv'].index('--spec') + 1]))
            eval_jobs[job['id']] = {'spec': s, 'output': job['outputs'][0]['path']}
    for item in training:
        name = item['run_id'].replace('P3_train_', 'P3_eval_')
        robust = item['architecture'] == 'efficientvit_b0' and item['budget'] == 2500
        s = write_spec(name, {'run_id': name, 'family': item['architecture'] + '_' + item['variant'] + '_N' + str(item['budget']),
                             'cohort': COHORT.name, 'arch': item['architecture'], 'seed': item['seed'],
                             'checkpoint': str(Path(item['out_dir']) / 'best.pt'), 'device': 'cuda:0',
                             'datasets': ['s11', 'sturm', 'wf'], 'roots': roots, 'num_workers': 2,
                             'conditions': conditions if robust else ['clean'],
                             'complete_reported_scalar_grid': True, 'fingerprints': item['fingerprints']})
        out = COHORT / 'evaluations' / (name + '.json')
        required = ['cells.wf__g0.90.micro'] if robust else []
        command(name, 'evaluation/revision_eval.py', ['--spec', s, '--out', out],
                [{'path': str(out), 'type': 'json', 'equals': {'status': 'complete'},
                  'finite': ['cells.' + d + '__clean.micro.mIoU' for d in ['s11', 'sturm', 'wf']], 'required_keys': required}],
                [item['run_id']], True, s, item['fingerprints'])
        eval_jobs[name] = {'spec': q.read_json(s), 'output': str(out)}
    # Reuse existing current-cohort CE and OFA rows; legacy no-clip KD stays a
    # named historical/current-cohort reference, outside matched contrast pairs.
    for arch in ARCHES:
        for seed in SEEDS:
            for objective, variant in [('hard', 'legacy_true_ce_clip0'), ('ofa', 'compatible_ofa_clip5_allvalid')]:
                run_id = f'R_train_{arch}_{objective}_N2500_s{seed}'
                runs.append({'run_id': run_id, 'architecture': arch, 'variant': variant, 'budget': 2500,
                             'seed': seed, 'evaluation': eval_jobs[run_id.replace('R_train_', 'R_eval_')]['output'],
                             'training_receipt': str(COHORT / 'training' / run_id / 'verified.json')})
    for item in training:
        runs.append({k: item[k] for k in ['run_id', 'architecture', 'variant', 'budget', 'seed']} |
                    {'evaluation': eval_jobs[item['run_id'].replace('P3_train_', 'P3_eval_')]['output'],
                     'training_receipt': str(Path(item['out_dir']) / 'verified.json')})
    for name in ['R_comparator_teacher', 'R_comparator_index_mndwi_zero']:
        runs.append({'run_id': name, 'architecture': eval_jobs[name]['spec']['arch'],
                     'variant': 'fixed_comparator', 'budget': 0, 'seed': 'fixed',
                     'evaluation': eval_jobs[name]['output']})
    native_jobs, validity_jobs = [], []
    retained = {r['evaluation']: r for r in runs if r['budget'] == 2500}
    for name, entry in eval_jobs.items():
        s, legacy_out = entry['spec'], entry['output']
        if legacy_out in retained or name in ['R_comparator_teacher', 'R_comparator_index_mndwi_zero']:
            native_id = name.replace('eval_', 'wf_native_').replace('comparator_', 'wf_native_')
            out = COHORT / 'native_border' / (native_id + '.json')
            if name == 'R_comparator_index_mndwi_zero':
                out = REPO / 'experiments/revision/prompt3_mndwi_worldfloods_native.json'
                done = q.read_json(out)
                canonical = q.read_json(legacy_out)['cells']['wf__clean']
                assert done['status'] == 'complete' and done['spec']['arch'] == s['arch']
                assert done['legacy']['map'] == canonical['scene'] and done['legacy']['event'] == canonical['event']
                assert done['legacy']['micro'] == canonical['micro']
                assert done['identity']['runtime_packages'] == eval_gate['runtime_packages']
                for rel, checksum in done['identity']['verified_source_sha256'].items():
                    assert q.file_hash(snapshot / rel) == checksum
                jobs.append({'id': native_id, 'kind': 'gate', 'cwd': str(snapshot), 'env': env,
                             'deps': [name, 'P3_evaluation_geometry_gate'], 'inputs': [pin(out)],
                             'outputs': [{'path': str(out), 'type': 'json', 'sha256': pin(out)['sha256'],
                                          'equals': {'status': 'complete', 'n_boundaries': 148, 'padding_counted': 0}}],
                             'reuse_reason': 'Completed CPU native-grid MNDWI pass; preserved original adapter execution hash. Reader/model/package identities and canonical clean map/event counts verified; no second boundary inference.'})
            else:
                ns = write_spec(native_id, {'arch': s['arch'], 'checkpoint': s['checkpoint'], 'device': s['device'],
                                            'roots': roots, 'legacy_result': legacy_out, 'gate': str(eval_gate_path), 'conditions': ['clean']})
                command(native_id, 'evaluation/revision_native_border.py', ['--mode', 'worldfloods', '--spec', ns, '--out', out],
                        [{'path': str(out), 'type': 'json', 'equals': {'status': 'complete', 'n_boundaries': 148, 'padding_counted': 0}}],
                        [name, 'P3_evaluation_geometry_gate'], s['device'].startswith('cuda'), ns)
            native_jobs.append(native_id)
            for run in runs:
                if run['evaluation'] == legacy_out:
                    run['native'] = str(out)
        if eval_gate['sturm']['mask_correction_required']:
            validity_id = name.replace('eval_', 'sturm_valid_').replace('comparator_', 'sturm_valid_')
            out = COHORT / 'sturm_valid' / (validity_id + '.json')
            vs = write_spec(validity_id, {'arch': s['arch'], 'checkpoint': s['checkpoint'], 'device': s['device'],
                                          'roots': roots, 'legacy_result': legacy_out, 'gate': str(eval_gate_path),
                                          'conditions': s['conditions'],
                                          **({'reuse_result': str(REPO / 'experiments/revision/prompt3_mndwi_sturm_finite.json')}
                                             if name == 'R_comparator_index_mndwi_zero' else {})})
            command(validity_id, 'evaluation/revision_native_border.py', ['--mode', 'sturm_valid', '--spec', vs, '--out', out],
                    [{'path': str(out), 'type': 'json', 'equals': {'status': 'complete'}}],
                    [name, 'P3_evaluation_geometry_gate'], s['device'].startswith('cuda'), vs)
            validity_jobs.append(validity_id)
            for run in runs:
                if run['evaluation'] == legacy_out:
                    run['sturm_valid'] = str(out)
                    run['sturm_conditions'] = s['conditions']
    comparisons = []
    for arch in ARCHES:
        pairs = [{'seed': seed, 'left': f'R_train_{arch}_ofa_N2500_s{seed}',
                  'right': f'P3_train_{arch}_matched_lkd_clip5_valid_N2500_s{seed}'} for seed in SEEDS]
        comparisons.append({'name': 'OFA_minus_clipping_matched_KD', 'architecture': arch, 'budget': 2500, 'pairs': pairs})
    for name, left, right, budget in [
        ('weighted_KD_minus_full_weight_focal', 'matched_lkd_clip5_valid', 'matched_focal_g2_clip5', 2500),
        ('N100_OFA_minus_clipping_matched_KD', 'matched_ofa_clip5_valid', 'matched_lkd_clip5_valid', 100)]:
        comparisons.append({'name': name, 'architecture': 'efficientvit_b0', 'budget': budget,
                            'pairs': [{'seed': seed, 'left': f'P3_train_efficientvit_b0_{left}_N{budget}_s{seed}',
                                       'right': f'P3_train_efficientvit_b0_{right}_N{budget}_s{seed}'} for seed in SEEDS]})
    for variant in ['matched_focal_g2_clip5', 'matched_lkd_clip5_valid', 'compatible_ofa_clip5_allvalid']:
        comparisons.append({'name': variant + '_minus_fixed_MNDWI', 'architecture': 'efficientvit_b0', 'budget': 2500,
                            'pairs': [{'seed': seed,
                                       'left': (f'R_train_efficientvit_b0_ofa_N2500_s{seed}' if variant.startswith('compatible')
                                                else f'P3_train_efficientvit_b0_{variant}_N2500_s{seed}'),
                                       'right': 'R_comparator_index_mndwi_zero'} for seed in SEEDS]})
    comparison_path = write_spec('matched_comparisons', {'runs': runs, 'comparisons': comparisons,
                                 'scope': 'Approved regenerated current cohort only; matched control gate recorded; CE and historical KD remain separately named'})
    # Dry reviewer gate and jobs are configured independently of model results.
    reference_root = REPO / 'experiments/revision/prompt3_dry_verification'
    dry_gate = reference_root / 'reference_gate.json'
    jobs.append({'id': 'P3_dry_reference_gate', 'kind': 'gate', 'cwd': str(snapshot), 'env': env,
                 'outputs': [{'path': str(dry_gate), 'type': 'json', 'equals': {'status': 'verified'}}]})
    dry_models = [(name, entry) for name, entry in eval_jobs.items() if name in ['R_comparator_teacher', 'R_comparator_index_mndwi_zero'] or
                  ('efficientvit_b0' in name and (name.startswith('P3_eval_') and '_N2500_' in name or '_ofa_N2500_' in name))]
    dry_jobs = []
    for name, entry in dry_models:
        s = entry['spec']; dry_id = name.replace('eval_', 'independent_dry_').replace('comparator_', 'independent_dry_')
        out = COHORT / 'independent_dry' / (dry_id + '.json')
        ds = write_spec(dry_id, {**s, 'run_id': dry_id, 'reference_kind': 'reviewed_worldfloods_chip',
                                'reference_manifest': str(reference_root / 'verified_manifest.json'),
                                'reference_gate': str(dry_gate), 'family': s['family']})
        train_dep = name.replace('P3_eval_', 'P3_train_').replace('R_eval_', 'R_train_')
        deps = ['P3_dry_reference_gate'] + ([train_dep] if train_dep in {j['id'] for j in jobs} else [])
        command(dry_id, 'evaluation/revision_dry.py', ['--spec', ds, '--out', out],
                [{'path': str(out), 'type': 'json', 'equals': {'status': 'complete'}, 'finite': ['false_positive_pixel_rate']}],
                deps, s['device'].startswith('cuda'), ds)
        dry_jobs.append(dry_id)
    jobs.append({'id': 'P3_target_device_artifact_gate', 'kind': 'gate', 'cwd': str(snapshot), 'env': env,
                 'blocked_reason': 'Four-seed matching device artifacts and an authorized accessible target Jetson workflow are absent; see prompt3_deployment_provenance.json',
                 'outputs': [{'path': str(REPO / 'experiments/revision/prompt3_deployment_provenance.json'), 'type': 'json', 'equals': {'work_status': 'evidence-ready'}}]})
    command('P3_collect', 'evaluation/revision_collect.py', ['--inputs', *[e['output'] for e in eval_jobs.values()],
             '--comparison_spec', comparison_path, '--out_dir', COHORT / 'analysis_prompt3',
             '--queue_manifest', queue_dir / 'manifest.json',
             '--reviewer_status', REPO / 'experiments/revision/prompt3_reviewer_status.json',
             '--status_out', REPO / 'experiments/revision/prompt3_run_status.json'],
            [{'path': str(COHORT / 'analysis_prompt3/matched_by_seed.csv')},
             {'path': str(COHORT / 'analysis_prompt3/paired_seed_summary.csv')},
             {'path': str(COHORT / 'analysis_prompt3/matched_provenance.json'), 'type': 'json'}],
            list(eval_jobs) + native_jobs + validity_jobs + dry_jobs, spec=comparison_path)
    jobs[-1]['run_after_terminal_deps'] = True
    def priority(job):
        name = job['id']
        if old_state['jobs'].get(name, {}).get('status') == 'succeeded': return (0, name)
        if job['kind'] == 'gate' or name in ['P3_verify_N100']: return (1, name)
        if name.startswith('R_comparator_') or name.startswith('R_eval_efficientvit_b0_'): return (2, name)
        if name.startswith('P3_train_') and '_N2500_' in name: return (3 if name.endswith('_s42') else 5, training.index(next(t for t in training if t['run_id'] == name)))
        if name.startswith('P3_eval_') and len(q.read_json(job['argv'][job['argv'].index('--spec') + 1])['conditions']) == 1: return (4, name)
        if 'wf_native_' in name or 'sturm_valid_' in name: return (6, name)
        if name.startswith('P3_train_'): return (7, training.index(next(t for t in training if t['run_id'] == name)))
        if name.startswith('P3_eval_'): return (8, name)
        if name.startswith('R_eval_'): return (9, name)
        return (10, name)
    jobs.sort(key=priority)
    estimate = cost_estimate(jobs, old_state, eval_jobs, training)
    q.atomic_json(queue_dir / 'cost_before_launch.json', estimate)
    manifest = copy.deepcopy(old_raw)
    manifest.update(jobs=jobs, env=env, manifest_revision='prompt3_finite_matched_controls', previous_queue=str(old_path),
                    fingerprints=fp, scientific_scope={'original_training_runs': 36, 'new_training_runs': 24,
                    'T1': 12, 'T2': 4, 'T3': 8, 'T4': 0, 'T4_reason': 'No ignored pseudo-training pixels; twelve existing OFA configurations compatible',
                    'absolute_new_training_ceiling': 36, 'conditions_frozen': conditions, 'seed_SD': 'sample ddof=1',
                    'no_accuracy_gate': True, 'data_receipts': {str(n): str(p) for n, p in receipts.items()},
                    'comparison_spec': str(comparison_path)}, resource_estimate=estimate)
    manifest_path = queue_dir / 'manifest.json'
    q.atomic_json(manifest_path, manifest); manifest_path.chmod(0o444)
    loaded = q.load_manifest(manifest_path)
    result = migrate(old_path, manifest_path)
    pointer.update(active_manifest=str(manifest_path), active_state=str(manifest_path.with_name('manifest.state.json')),
                   active_wrapper=str(wrapper), previous_manifest=str(old_path), prompt3_finite_matrix=True,
                   migration_receipt=str(queue_dir / 'migration.json'))
    q.atomic_json(pointer_path, pointer)
    q.atomic_json(reference_root / 'queue_resume.json', {'argv': [PY, str(wrapper), 'resume', '--manifest', str(manifest_path)]})
    q.atomic_json(queue_dir / 'preparation.json', {**result, 'status': 'verified', 'source_snapshot': str(snapshot),
                  'code_sha256': code_sha, 'new_training_runs': 24, 'new_full_budget_runs': 16, 'new_short_runs': 8,
                  'conditional_OFA_replacements': 0, 'compatible_OFA_partners': 12, 'jobs': len(jobs),
                  'comparison_spec': str(comparison_path), 'controls': {a: [PY, str(wrapper), a, '--manifest', str(manifest_path)]
                                                                      for a in ['status', 'pause', 'resume', 'stop']}})
    print(json.dumps(result | {'new_training_runs': 24, 'jobs': len(jobs), 'snapshot': str(snapshot)}))


if __name__ == '__main__':
    prepare()
