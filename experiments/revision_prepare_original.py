"""Append prompt4's original-data matrix to the existing finite detached queue."""
from __future__ import annotations

import copy
from datetime import datetime
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
ORIGINAL = ROOT / 'original_N2500_recovered_20261006'
REGEN = ROOT / 'regen_N2500_approved_20261005'
PY = '/home/schmalstieg/miniconda3/envs/esi/bin/python'
SEEDS = [42, 64, 91, 107]
ARCHES = ['efficientvit_b0', 'mobilenet_v3_small', 'lightformer_resnet18']


def prepare():
    pointer_path = ROOT / 'queue/active_queue.json'
    pointer = q.read_json(pointer_path)
    old_path = Path(pointer['active_manifest'])
    old_raw, old = q.read_json(old_path), q.load_manifest(old_path)
    state = q.read_json(old_path.with_name('manifest.state.json'))
    assert not q.alive(state.get('worker')) and state['worker_status'] in {'finished', 'stopped'}
    assert not any(r['status'] in {'running', 'launching', 'pending', 'failed'} for r in state['jobs'].values())
    assert state['stop_requested'], 'Record graceful stop before migrating the drained queue'
    data_gate_path = REPO / 'experiments/revision/prompt4_original_data_gate.json'
    inventory_path = REPO / 'experiments/revision/prompt4_original_checkpoint_inventory.json'
    data_gate, inventory = q.read_json(data_gate_path), q.read_json(inventory_path)
    assert data_gate['status'] == 'verified' and data_gate['selected_mapping_matches_prior_original_sources']
    assert data_gate['restoration_counts'] == {'restored': 127}
    assert data_gate['matched_OFA_reuse_mask_equivalent']
    assert inventory['status_counts'] == {'inference-ready': 28}
    old_snapshot = Path(old_raw['source_repair_snapshot'])
    source_identity = q.read_json(old_snapshot / 'source_identity.json')
    source = {rel: REPO / rel if (REPO / rel).is_file() else old_snapshot / rel
              for rel in source_identity['files']
              if '__pycache__' not in Path(rel).parts and Path(rel).suffix not in {'.pyc', '.pyo'}}
    for rel in ['experiments/revision_prepare_original.py', 'scripts/revision_original_data_check.py',
                'scripts/revision_original_checkpoints.py', 'evaluation/revision_eval.py',
                'evaluation/revision_collect.py', 'evaluation/revision_dry.py',
                'experiments/revision/prompt4_original_data_gate.json',
                'experiments/revision/prompt4_original_checkpoint_inventory.json',
                'experiments/revision/prompt4_original_score_set.json',
                'experiments/revision/prompt4_original_overlap_gate.json', 'prompts/prompt4.md', 'AGENT.md']:
        source[rel] = REPO / rel
    # Reader/model/loss execution remains unchanged; only bookkeeping adapters extend.
    for rel in ['models/student.py', 'models/teacher.py', 'models/lightformer.py',
                'data/pseudo_labeled.py', 'data/ood_datasets.py', 'data/channel_adapter.py',
                'data/sen1floods11.py', 'training/trainer.py', 'training/losses.py',
                'distillation/ofa_kd.py', 'scripts/train.py', 'evaluation/perscene_eval.py',
                'evaluation/revision_native_border.py']:
        assert q.file_hash(source[rel]) == source_identity['files'][rel], f'Unexpected scientific-source change: {rel}'
    hashes = {rel: q.file_hash(path) for rel, path in sorted(source.items())}
    code_sha = q.digest(hashes)
    snapshot = ROOT / 'snapshots' / code_sha[:16]
    snapshot.mkdir(exist_ok=False)
    for rel, path in source.items():
        dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        dest.chmod(0o444)
    q.atomic_json(snapshot / 'source_identity.json', {'code_sha256': code_sha, 'files': hashes})
    queue_dir = ORIGINAL / 'queue_prompt4'
    queue_dir.mkdir(exist_ok=False)
    specs = queue_dir / 'specs'
    specs.mkdir()
    wrapper = queue_dir / 'revision_queue.py'
    shutil.copyfile(snapshot / 'experiments/revision_queue.py', wrapper)
    wrapper.chmod(0o444)
    old_jobs = {j['id']: j for j in old_raw['jobs']}
    jobs = copy.deepcopy(old_raw['jobs'])
    env = {**old_raw['env'], 'PYTHONPATH': str(snapshot) + ':' + ':'.join(old_raw['env']['PYTHONPATH'].split(':')[1:]),
           'PYTHONDONTWRITEBYTECODE': '1'}
    receipts = {n: Path(data_gate['receipts'][str(n)]) for n in (2500, 100)}
    data = {n: q.read_json(path) for n, path in receipts.items()}
    teacher_template = q.read_json(REGEN / 'queue/specs/R_train_efficientvit_b0_logit_N2500_s42.json')
    teacher = teacher_template['argv'][teacher_template['argv'].index('--official_ckpt') + 1]
    environment = {**q.read_json(REGEN / 'environment.json'), 'prompt': 'prompt4',
                   'source_sha256': code_sha, 'parent_environment_sha256': q.file_hash(REGEN / 'environment.json'),
                   'original_data_digest': data[2500]['dataset_sha256'],
                   'historical_execution': 'Recovered v2 recipe in current environment; no exact old runtime/config claim'}
    environment_path = queue_dir / 'environment.json'
    q.atomic_json(environment_path, environment)
    environment_path.chmod(0o444)
    cache = {}
    def pin(path):
        path = Path(path).resolve()
        if str(path) not in cache:
            cache[str(path)] = q.file_hash(path)
        return {'path': str(path), 'sha256': cache[str(path)]}
    base = [pin(snapshot / rel) for rel in hashes]
    base += [pin(environment_path), *[pin(path) for path in receipts.values()],
             *[pin(Path(d['index'])) for d in data.values()], pin(data_gate_path), pin(inventory_path),
             pin(ORIGINAL / 'data/original_source_manifest.json')]
    # Keep the already verified pretrained dependencies and teacher pins.
    for item in old_jobs['R_train_efficientvit_b0_logit_N2500_s42']['inputs']:
        p = Path(item['path'])
        if not p.is_relative_to(ROOT):
            base.append(pin(p))
    base = list({p['path']: p for p in base}.values())
    fp = {**old_raw['fingerprints'], 'code': code_sha,
          'environment': q.file_hash(environment_path), 'data': data[2500]['dataset_sha256'],
          'cohort': ORIGINAL.name, 'authorization': 'prompts/prompt4.md',
          'training_originality': 'Copied original caches, unchanged labels; current-environment new training'}
    new_ids, priority = [], {}
    def write_spec(name, value):
        path = specs / (name + '.json')
        q.atomic_json(path, value)
        path.chmod(0o444)
        return path
    def command(name, script, argv, outputs, deps=(), gpu=False, spec=None, extra=(), order=(9, 0)):
        job = {'id': name, 'kind': 'command', 'cwd': str(snapshot),
               'argv': ['/usr/bin/ionice', '-c', '2', '-n', '7', PY, str(snapshot / script), *map(str, argv)],
               'env': copy.deepcopy(env), 'gpu': gpu, 'deps': list(dict.fromkeys(deps)),
               'inputs': list(base) + [pin(p) for p in extra], 'outputs': outputs, 'fingerprints': fp}
        if spec:
            job['inputs'].append(pin(spec))
        if gpu:
            job['required_free_mb'] = 16384
        else:
            job['env']['CUDA_VISIBLE_DEVICES'] = ''
        jobs.append(job)
        new_ids.append(name)
        priority[name] = order
        return job
    def json_output(path, equals=None):
        return {'path': str(path), 'type': 'json', 'equals': equals or {'status': 'complete'}}
    for name, path, status_key in [('O4_original_data_gate', data_gate_path, 'verified'),
                                   ('O4_original_checkpoint_gate', inventory_path, 'verified-main-checkpoints')]:
        jobs.append({'id': name, 'kind': 'gate', 'cwd': str(snapshot), 'env': env,
                     'inputs': [pin(path)], 'outputs': [dict(json_output(path, {'status': status_key}), sha256=pin(path)['sha256'])]})
        new_ids.append(name)
        priority[name] = (1, 0)
    for budget in (2500, 100):
        name = f'O4_verify_N{budget}'
        out = queue_dir / f'N{budget}_verified.json'
        command(name, 'scripts/revision_cohort_check.py', ['data', '--receipt', receipts[budget], '--out', out],
                [json_output(out, {'status': 'verified', 'n_selected': budget})], ['O4_original_data_gate'], order=(1, budget))
    roots = q.read_json(REGEN / 'queue/specs/R_comparator_teacher.json')['roots']
    full_conditions = q.read_json(REGEN / 'queue/specs/R_comparator_teacher.json')['conditions']
    full_expected = sorted(q.read_json(REGEN / 'evaluations/R_comparator_teacher.json')['cells'])
    clean_expected = ['s11__clean', 'sturm__clean', 'wf__clean']
    geometry_gate = REPO / 'experiments/revision/prompt3_evaluation_gate.json'
    first_original = inventory['runs'][0]
    assert first_original['seed'] == 42 and first_original['declared_objective'] == 'ce'
    first_name = 'O4_original_S11_functional_seed42'
    first_path = ORIGINAL / 'evaluations' / (first_name + '.json')
    first_spec = write_spec(first_name, {'run_id': first_name, 'family': first_original['original_id'],
        'cohort': 'copied_original_checkpoint_current_rescoring', 'arch': first_original['architecture'], 'seed': 42,
        'checkpoint': first_original['checkpoint'], 'device': 'cuda:0', 'datasets': ['s11'],
        'roots': roots, 'num_workers': 2, 'conditions': ['clean'], 'fingerprints': fp})
    command(first_name, 'evaluation/revision_eval.py', ['--spec', first_spec, '--out', first_path],
            [json_output(first_path, {'status': 'complete', 'identity.checkpoint_sha256': first_original['checkpoint_sha256'],
                                      'cells.s11__clean.n_tiles': 90})],
            ['O4_original_checkpoint_gate', 'Q_evaluator_check'], True, first_spec,
            [first_original['checkpoint']], (2, 0))
    for i, arch in enumerate(ARCHES):
        name = 'O4_protocol_' + arch
        out = queue_dir / (name + '.json')
        command(name, 'scripts/revision_protocol_check.py', ['--teacher', teacher, '--arch', arch,
            '--device', 'cuda:0', '--batch_size', 16, '--pseudo_pool', data[2500]['pool'],
            '--pseudo_index', data[2500]['index'], '--out', out],
            [json_output(out, {'status': 'verified', 'scientific_batch_size': 16, 'teacher_frozen_eval_no_grad': True})],
            [first_name, 'O4_verify_N2500'], True, order=(3, i))
    canonical_ids, correction_ids, score_runs, training_runs = [], [], [], []
    dependencies = {}
    def evaluate(name, spec, deps, order, extra=()):
        path = write_spec(name, spec)
        out = ORIGINAL / 'evaluations' / (name + '.json')
        required = ['cells.' + key + '.micro.mIoU' for key in (full_expected if len(spec['conditions']) > 1 else clean_expected)
                    if key.split('__', 1)[0] in spec['datasets']]
        command(name, 'evaluation/revision_eval.py', ['--spec', path, '--out', out],
                [dict(json_output(out), required_keys=required, finite=required)], deps, True, path, extra, order)
        canonical_ids.append(name)
        return str(out)
    def correct(run, legacy_result, clean_dep, sturm_dep, conditions, order):
        name = run['run_id'] + '_wf_native'
        out = ORIGINAL / 'native_border' / (name + '.json')
        spec = write_spec(name, {'arch': run['architecture'], 'checkpoint': run['checkpoint'], 'device': 'cuda:0',
            'roots': roots, 'legacy_result': legacy_result, 'gate': str(geometry_gate), 'conditions': ['clean']})
        command(name, 'evaluation/revision_native_border.py', ['--mode', 'worldfloods', '--spec', spec, '--out', out],
                [json_output(out, {'status': 'complete', 'n_boundaries': 148, 'padding_counted': 0})],
                [clean_dep, 'P3_evaluation_geometry_gate'], True, spec, order=order)
        run['native'] = str(out)
        correction_ids.append(name)
        deps = [name]
        name = run['run_id'] + '_sturm_valid'
        out = ORIGINAL / 'sturm_valid' / (name + '.json')
        spec = write_spec(name, {'arch': run['architecture'], 'checkpoint': run['checkpoint'], 'device': 'cuda:0',
            'roots': roots, 'legacy_result': run['evaluation'], 'gate': str(geometry_gate), 'conditions': conditions})
        command(name, 'evaluation/revision_native_border.py', ['--mode', 'sturm_valid', '--spec', spec, '--out', out],
                [json_output(out)], [sturm_dep, 'P3_evaluation_geometry_gate'], True, spec, order=order)
        run['sturm_valid'], run['sturm_conditions'] = str(out), conditions
        correction_ids.append(name)
        dependencies[run['run_id']] = [clean_dep, sturm_dep, *deps, name]
    for i, original in enumerate(inventory['runs']):
        name = original['original_id'] + '_eval'
        spec = {'run_id': name, 'family': original['original_id'].rsplit('_s', 1)[0],
                'cohort': 'copied_original_checkpoint_current_rescoring', 'arch': original['architecture'],
                'seed': original['seed'], 'checkpoint': original['checkpoint'], 'device': 'cuda:0',
                'datasets': ['s11', 'sturm', 'wf'], 'roots': roots, 'num_workers': 2,
                'conditions': ['clean'], 'fingerprints': {**fp, 'original_checkpoint_sha256': original['checkpoint_sha256']}}
        if i == 0:
            spec['reuse_results'] = [str(first_path)]
        out = evaluate(name, spec, [first_name, 'O4_original_checkpoint_gate'], (4, i), [original['checkpoint']])
        run = {**original, 'run_id': original['original_id'], 'variant': 'original_checkpoint_' + original['declared_objective'],
               'evaluation': out, 'canonical_expected_cells': clean_expected}
        correct(run, out, name, name, ['clean'], (4, 100 + i))
        score_runs.append(run)
    cells = [('efficientvit_b0', 'hard', 'historical_true_ce_clip0', 2500, ['A']),
             ('efficientvit_b0', 'logit', 'historical_lkd_clip0', 2500, ['A']),
             ('efficientvit_b0', 'ofa', 'historical_ofa_clip5_allvalid', 2500, ['A', 'B']),
             ('efficientvit_b0', 'logit', 'matched_lkd_clip5_valid', 2500, ['B']),
             ('efficientvit_b0', 'focal', 'matched_focal_g2_clip5', 2500, ['C'])]
    cells += [(a, obj, variant, 2500, ['D']) for a in ARCHES[1:] for obj, variant in
              [('logit', 'matched_lkd_clip5_valid'), ('ofa', 'matched_ofa_clip5_valid')]]
    cells += [('efficientvit_b0', obj, variant, 100, ['E']) for obj, variant in
              [('logit', 'matched_lkd_clip5_valid'), ('ofa', 'matched_ofa_clip5_valid')]]
    first_training = {}
    for seed in SEEDS:
        for arch, obj, variant, budget, groups in cells:
            if variant == 'matched_focal_g2_clip5':
                parent_id = f'P3_train_{arch}_{variant}_N2500_s{seed}'
            elif variant == 'matched_lkd_clip5_valid' or budget == 100:
                parent_id = f'P3_train_{arch}_{variant}_N{budget}_s{seed}'
            else:
                parent_id = f'R_train_{arch}_{obj}_N2500_s{seed}'
            parent_job = old_jobs[parent_id]
            parent = q.read_json(Path(parent_job['argv'][parent_job['argv'].index('--spec') + 1]))
            name = f'O4_train_{arch}_{variant}_N{budget}_s{seed}'
            out = ORIGINAL / 'training' / name
            argv = copy.deepcopy(parent['argv'])
            argv[1] = str(snapshot / 'scripts/train.py')
            def flag(key, value):
                if key in argv:
                    argv[argv.index(key) + 1] = str(value)
                else:
                    argv.extend([key, str(value)])
            clip = 0.0 if variant.endswith('clip0') else 5.0
            for key, value in [('--pseudo_dir', data[budget]['pool']), ('--curated_index_file', data[budget]['index']),
                               ('--n_samples', budget), ('--out_dir', out), ('--wandb_run_name', name), ('--grad_clip', clip)]:
                flag(key, value)
            matched = variant.startswith('matched_')
            if obj in ('ofa', 'logit') and matched and '--kd_valid_only' not in argv:
                argv.append('--kd_valid_only')
            expected = {**parent['expected_config'], 'out_dir': str(out), 'grad_clip': clip}
            policy = {'main_hard': 'ce' if obj == 'hard' else 'focal', 'main_focal_gamma': 2.0,
                      'grad_clip_norm': clip, 'main_kl_mask': None if obj in ('hard', 'focal') else 'valid_only' if matched else 'legacy_all_pixels'}
            metadata = {'run_id': name, 'architecture': arch, 'objective': obj, 'variant': variant,
                        'budget': budget, 'n_selected': budget, 'batch_size': 16, 'groups': groups, 'seed': seed,
                        'update_policy': {'epochs_ceiling': 50, 'drop_last': True, 'steps_per_epoch': budget // 16,
                                          'maximum_optimizer_steps': 50 * (budget // 16), 'early_stop_patience_checks': 20},
                        'selection_policy': {'split': 'S11_validation', 'n_chips': 89, 'metric': 'mIoU',
                                             'every_n_epochs': 2, 'smoothing_window': 1, 'checkpoint': 'best.pt'},
                        'gpu_assignment': 'First free authorized UUID under existing per-device lock; one physical GPU per run; score independent',
                        'cohort': ORIGINAL.name, 'dataset_receipt': str(receipts[budget]), 'argv': argv,
                        'out_dir': str(out), 'expected_config': expected, 'expected_loss_policy': policy,
                        'fingerprints': {**fp, 'data': data[budget]['dataset_sha256']}, 'parent_spec': str(parent_job['argv'][parent_job['argv'].index('--spec') + 1]),
                        'historical_limit': 'Recovered v2 recipe, current environment. Copied originals lack per-run executed config/hash receipts.',
                        'reuse': 'GroupA OFA also supplies GroupB: no ignored pixels under actual original augmentation; otherwise recipes identical' if groups == ['A', 'B'] else None}
            path = write_spec(name, metadata)
            deps = [f'O4_verify_N{budget}', first_name, 'O4_protocol_' + arch]
            key = (arch, variant, budget)
            if seed == 42:
                first_training[key] = name
            else:
                deps.append(first_training[key])
            order = (5 if seed == 42 and groups[0] in ['A', 'B', 'C'] else
                     6 if groups[0] in ['A', 'B', 'C'] else 7 if seed == 42 else 8, SEEDS.index(seed) * 20 + cells.index((arch, obj, variant, budget, groups)))
            command(name, 'scripts/revision_cohort_check.py', ['train', '--spec', path, '--out', out / 'verified.json'],
                    [{'path': str(out / p), 'min_bytes': 1000} for p in ['best.pt', 'last.pt']] +
                    [json_output(out / 'verified.json', {'status': 'verified', 'run_id': name})], deps, True, path, order=order)
            item = {**metadata, 'checkpoint': str(out / 'best.pt'), 'training_receipt': str(out / 'verified.json')}
            clean_name = name.replace('O4_train_', 'O4_eval_clean_')
            eval_spec = {'run_id': clean_name, 'family': arch + '_' + variant + '_N' + str(budget), 'cohort': ORIGINAL.name,
                        'arch': arch, 'seed': seed, 'checkpoint': item['checkpoint'], 'device': 'cuda:0',
                        'datasets': ['s11', 'sturm', 'wf'], 'roots': roots, 'num_workers': 2, 'conditions': ['clean'],
                        'training_receipt': item['training_receipt'], 'training_run_id': name, 'fingerprints': item['fingerprints']}
            clean_out = evaluate(clean_name, eval_spec, [name], (4, 300 + len(training_runs)))
            robust = arch == 'efficientvit_b0' and budget == 2500 and (variant.startswith('matched_') or obj == 'ofa')
            eval_name, eval_out = clean_name, clean_out
            if robust:
                eval_name = name.replace('O4_train_', 'O4_eval_robust_')
                robust_spec = {**eval_spec, 'run_id': eval_name, 'conditions': full_conditions,
                               'complete_reported_scalar_grid': True, 'reuse_results': [clean_out]}
                eval_out = evaluate(eval_name, robust_spec, [name, clean_name], (9, len(training_runs)))
            item['evaluation'], item['canonical_expected_cells'] = eval_out, full_expected if robust else clean_expected
            item['clean_evaluation'] = clean_out
            correct(item, clean_out, clean_name, eval_name, full_conditions if robust else ['clean'], (4 if not robust else 9, 500 + len(training_runs)))
            training_runs.append(item)
    assert len(training_runs) == 44 and sum(t['budget'] == 2500 for t in training_runs) == 36
    fixed = []
    for name, arch, native in [('R_comparator_teacher', 'teacher', 'R_wf_native_teacher'),
                                ('R_comparator_index_mndwi_zero', 'index:mndwi_zero', 'R_wf_native_index_mndwi_zero')]:
        legacy = old_jobs[name]['outputs'][0]['path']
        raw = q.read_json(legacy)
        assert raw['status'] == 'complete'
        for rel in ['data/ood_datasets.py', 'data/channel_adapter.py', 'data/sen1floods11.py', 'evaluation/perscene_eval.py', 'models/student.py', 'models/teacher.py']:
            assert raw['identity']['source_sha256'][rel] == hashes[rel]
        run = {'run_id': name, 'architecture': arch, 'variant': 'fixed_comparator', 'budget': 0, 'seed': 'fixed',
               'evaluation': legacy, 'native': old_jobs[native]['outputs'][0]['path'],
               'sturm_valid': old_jobs[name.replace('comparator_', 'sturm_valid_')]['outputs'][0]['path'],
               'sturm_conditions': full_conditions, 'canonical_expected_cells': full_expected,
               'reuse': 'Fixed checkpoint/index, unchanged loader/model/radiometry/masks/packages; no training-cohort dependence'}
        fixed.append(run)
        dependencies[name] = [name, native, name.replace('comparator_', 'sturm_valid_')]
    def run_id(arch, variant, budget, seed):
        return f'O4_train_{arch}_{variant}_N{budget}_s{seed}'
    contrasts = []
    for arch in ARCHES:
        ofa = 'historical_ofa_clip5_allvalid' if arch == 'efficientvit_b0' else 'matched_ofa_clip5_valid'
        contrasts.append({'name': 'OFA_minus_clipping_matched_KD', 'architecture': arch, 'budget': 2500,
            'pairs': [{'seed': s, 'left': run_id(arch, ofa, 2500, s), 'right': run_id(arch, 'matched_lkd_clip5_valid', 2500, s)} for s in SEEDS]})
    for label, left, right, budget in [('weighted_KD_minus_full_weight_focal', 'matched_lkd_clip5_valid', 'matched_focal_g2_clip5', 2500),
                                     ('N100_OFA_minus_clipping_matched_KD', 'matched_ofa_clip5_valid', 'matched_lkd_clip5_valid', 100),
                                     ('historical_OFA_minus_legacy_KD_with_clipping_confounded', 'historical_ofa_clip5_allvalid', 'historical_lkd_clip0', 2500),
                                     ('historical_KD_minus_true_CE_with_hard_family_confounded', 'historical_lkd_clip0', 'historical_true_ce_clip0', 2500)]:
        contrasts.append({'name': label, 'architecture': 'efficientvit_b0', 'budget': budget,
            'pairs': [{'seed': s, 'left': run_id('efficientvit_b0', left, budget, s), 'right': run_id('efficientvit_b0', right, budget, s)} for s in SEEDS]})
    for variant in ['matched_focal_g2_clip5', 'matched_lkd_clip5_valid', 'historical_ofa_clip5_allvalid']:
        contrasts.append({'name': variant + '_minus_fixed_MNDWI', 'architecture': 'efficientvit_b0', 'budget': 2500,
            'pairs': [{'seed': s, 'left': run_id('efficientvit_b0', variant, 2500, s), 'right': 'R_comparator_index_mndwi_zero'} for s in SEEDS]})
    for arm, variant in [('ce', 'historical_true_ce_clip0'), ('dfkd', 'historical_lkd_clip0'), ('ofa', 'historical_ofa_clip5_allvalid')]:
        contrasts.append({'name': 'new_recipe_training_minus_original_checkpoint_' + arm,
            'architecture': 'efficientvit_b0', 'budget': 2500,
            'pairs': [{'seed': s, 'left': run_id('efficientvit_b0', variant, 2500, s),
                       'right': f'O4_original_efficientvit_b0_{arm}_N2500_s{s}'} for s in SEEDS]})
    from evaluation.revision_native_border import WF_PROTOCOL, STURM_PROTOCOL
    corrected_clean = clean_expected + ['wf__' + WF_PROTOCOL, 'sturm__' + STURM_PROTOCOL + ':clean']
    # Only primary matched contrasts have two authorized robustness matrices.
    for contrast in contrasts:
        if (contrast['name'] not in ['OFA_minus_clipping_matched_KD', 'weighted_KD_minus_full_weight_focal']
                and not contrast['name'].endswith('_minus_fixed_MNDWI')) or contrast['architecture'] != 'efficientvit_b0':
            contrast['cell_keys'] = corrected_clean
        if contrast['name'] in ['OFA_minus_clipping_matched_KD', 'weighted_KD_minus_full_weight_focal', 'N100_OFA_minus_clipping_matched_KD']:
            contrast['require_matching_initialization'] = True
    reference_spec = write_spec('original_reference', {'runs': score_runs,
        'printed_table': str(REPO / 'experiments/revision/prompt4_published_Table_II_display.csv'),
        'printed_N100': str(REPO / 'paper_source_data/Figure_02_Objective_Ladder_vs_N/fig_ladder_vs_n_values.csv')})
    reference_inputs = [REPO / 'paper_source_data/Table_II_Deployment_N2500/Table_II_values.csv',
                        REPO / 'experiments/revision/prompt4_published_Table_II_display.csv',
                        REPO / 'experiments/revision/prompt4_published_reference_audit.json',
                        REPO / 'paper_source_data/Figure_02_Objective_Ladder_vs_N/fig_ladder_vs_n_values.csv',
                        *[Path(r['archive_metrics']) for r in score_runs]]
    original_pairs = []
    for budget in [2500, 100]:
        original_pairs.append({'name': 'original_checkpoint_OFA_minus_legacy_KD_with_clipping_confounded',
            'architecture': 'efficientvit_b0', 'budget': budget,
            'pairs': [{'seed': s, 'left': f'O4_original_efficientvit_b0_ofa_N{budget}_s{s}',
                       'right': f'O4_original_efficientvit_b0_dfkd_N{budget}_s{s}'} for s in SEEDS]})
    for contrast in original_pairs:
        contrast['cell_keys'] = corrected_clean
    comparison_spec = write_spec('original_data_comparisons', {'runs': score_runs + training_runs + fixed,
        'comparisons': contrasts, 'scope': 'Copied original checkpoint rescoring and new original-data training are separately named. Original caches unchanged. Recovered GroupA recipes use the current environment; exact historical execution is unverified.'})
    def collector(name, comparison_path, output, deps, original=False, status=False, order=(10, 0)):
        args = ['--matched_only', '--comparison_spec', comparison_path, '--out_dir', output]
        extra = []
        if original:
            args += ['--original_reference_spec', reference_spec]
            extra += [reference_spec, *reference_inputs]
        if status:
            args += ['--queue_manifest', queue_dir / 'manifest.json', '--reviewer_status', REPO / 'experiments/revision/prompt3_reviewer_status.json',
                     '--status_out', REPO / 'experiments/revision/prompt4_run_status.json']
        outputs = [{'path': str(output / 'matched_by_seed.csv')}, {'path': str(output / 'paired_seed_summary.csv')},
                   {'path': str(output / 'matched_provenance.json'), 'type': 'json', 'equals': {'status': 'complete'}}]
        if original:
            outputs += [{'path': str(output / 'original_score_reproduction_summary.csv')}]
        job = command(name, 'evaluation/revision_collect.py', args, outputs, deps, spec=comparison_path, extra=extra, order=order)
        job['run_after_terminal_deps'] = True
        return job
    original_only = write_spec('original_checkpoint_comparisons', {'runs': score_runs + fixed,
        'comparisons': original_pairs, 'scope': 'Original main-checkpoint rescoring only; historical clipping/data/source identities qualified'})
    collector('O4_collect_original_checkpoints', original_only, ORIGINAL / 'analysis_original_checkpoints',
              [d for r in score_runs for d in dependencies[r['run_id']]], original=True, order=(4, 900))
    for i, contrast in enumerate(contrasts):
        ids = {p[k] for p in contrast['pairs'] for k in ['left', 'right']}
        subset = [r for r in score_runs + training_runs + fixed if r['run_id'] in ids]
        name = 'O4_collect_pair_' + str(i)
        path = write_spec(name, {'runs': subset, 'comparisons': [contrast], 'scope': 'Original-data declared pair; all required seeds/protocols retained'})
        collector(name, path, ORIGINAL / 'analysis_original_data/per_comparison' / name,
                  [d for r in subset for d in dependencies[r['run_id']]], order=(4, 1000 + i))
    # Regenerated findings are a separate comparison file and table namespace.
    regen_spec_path = REGEN / 'queue_prompt3_source_only/specs/matched_comparisons.json'
    regen_runs = q.read_json(regen_spec_path)['runs']
    cross_runs, cross_pairs, cross_deps = [], [], []
    for original in training_runs:
        variant = original['variant']
        arch, budget, seed = original['architecture'], original['budget'], original['seed']
        if variant == 'historical_lkd_clip0':
            continue  # Historical no-clip KD is outside the previous matched spec.
        target_variant = {'historical_true_ce_clip0': 'legacy_true_ce_clip0',
                          'historical_ofa_clip5_allvalid': 'compatible_ofa_clip5_allvalid'}.get(variant, variant)
        if budget == 2500 and variant == 'matched_ofa_clip5_valid':
            target_variant = 'compatible_ofa_clip5_allvalid'
        match = next((r for r in regen_runs if r['architecture'] == arch and r['budget'] == budget and r['seed'] == seed and r['variant'] == target_variant), None)
        assert match, (arch, variant, budget, seed)
        left, right = copy.deepcopy(original), copy.deepcopy(match)
        left['variant'] += '_original_data'
        right['variant'] += '_regenerated_data'
        cross_runs.extend([left, right])
        cross_pairs.append({'architecture': arch, 'variant': variant, 'budget': budget, 'seed': seed,
                            'left': left['run_id'], 'right': right['run_id']})
        cross_deps += dependencies[left['run_id']]
    cross_comparisons = []
    for arch, variant, budget in sorted({(p['architecture'], p['variant'], p['budget']) for p in cross_pairs}):
        cross_comparisons.append({'name': variant + '_original_minus_regenerated_joint_cohort_cache_change',
            'architecture': arch, 'budget': budget,
            'cell_keys': corrected_clean,
            'require_matching_initialization': variant.startswith('matched_') and not (budget == 2500 and 'ofa' in variant),
            'pairs': [{k: p[k] for k in ['seed', 'left', 'right']} for p in cross_pairs
                      if (p['architecture'], p['variant'], p['budget']) == (arch, variant, budget)]})
    cross_path = write_spec('cross_cohort_comparisons', {'runs': list({r['run_id']: r for r in cross_runs}.values()),
        'comparisons': cross_comparisons,
        'scope': 'Original versus regenerated training, same declared recipe/environment/initialization policy and protocol. Joint cohort/cache change; cannot isolate exactly127scenes. Historical OFA initial tensor receipts absent; do not claim direct initial-tensor equality.'})
    cross_collector = collector('O4_collect_cross_cohort', cross_path, ORIGINAL / 'analysis_cross_cohort', cross_deps)
    # Independent dry reference stays a real external gate; fixed comparators reuse existing jobs.
    dry_root = REPO / 'experiments/revision/prompt3_dry_verification'
    dry_ids = []
    dry_outputs = [old_jobs[name]['outputs'][0]['path'] for name in ['R_independent_dry_teacher', 'R_independent_dry_index_mndwi_zero']]
    dry_ids += ['R_independent_dry_teacher', 'R_independent_dry_index_mndwi_zero']
    for item in training_runs:
        if item['architecture'] != 'efficientvit_b0' or item['budget'] != 2500 or item['variant'] not in ['matched_focal_g2_clip5', 'matched_lkd_clip5_valid', 'historical_ofa_clip5_allvalid']:
            continue
        name = item['run_id'].replace('O4_train_', 'O4_independent_dry_')
        out = ORIGINAL / 'independent_dry' / (name + '.json')
        spec = write_spec(name, {'run_id': name, 'family': item['architecture'] + '_' + item['variant'],
            'arch': item['architecture'], 'seed': item['seed'], 'checkpoint': item['checkpoint'], 'device': 'cuda:0',
            'training_receipt': item['training_receipt'], 'training_run_id': item['run_id'],
            'reference_manifest': str(dry_root / 'verified_manifest.json'), 'reference_gate': str(dry_root / 'reference_gate.json')})
        command(name, 'evaluation/revision_dry.py', ['--spec', spec, '--out', out],
                [dict(json_output(out), finite=['false_positive_pixel_rate'])],
                ['P3_dry_reference_gate', item['run_id']], True, spec, order=(11, len(dry_ids)))
        dry_ids.append(name)
        dry_outputs.append(str(out))
    out = ORIGINAL / 'analysis_original_dry'
    command('O4_collect_dry', 'evaluation/revision_collect.py', ['--dry_only', '--inputs', *dry_outputs, '--out_dir', out,
            '--queue_manifest', queue_dir / 'manifest.json', '--reviewer_status', REPO / 'experiments/revision/prompt3_reviewer_status.json',
            '--status_out', REPO / 'experiments/revision/prompt4_run_status.json'],
            [{'path': str(out / 'dry_provenance.json'), 'type': 'json', 'equals': {'status': 'complete'}},
             {'path': str(out / 'dry_by_checkpoint.csv')}], dry_ids, order=(11, 100))
    computation_ids = [t['run_id'] for t in training_runs] + canonical_ids + correction_ids + [first_name] + ['O4_protocol_' + a for a in ARCHES]
    final_out = ORIGINAL / 'analysis_original_data'
    final = collector('O4_collect_final', comparison_spec, final_out,
                      computation_ids + ['O4_collect_cross_cohort', 'O4_collect_original_checkpoints'] +
                      ['O4_collect_pair_' + str(i) for i in range(len(contrasts))], original=True, status=True)
    numerical_collectors = ['O4_collect_final', 'O4_collect_cross_cohort', 'O4_collect_original_checkpoints'] + ['O4_collect_pair_' + str(i) for i in range(len(contrasts))]
    command('O4_refresh_handoff', 'evaluation/revision_collect.py', ['--status_only', '--queue_manifest', queue_dir / 'manifest.json',
            '--reviewer_status', REPO / 'experiments/revision/prompt3_reviewer_status.json',
            '--status_out', final_out / 'queue_handoff.json', '--out_dir', final_out],
            [{'path': str(final_out / 'queue_handoff.json'), 'type': 'json'}],
            ['O4_collect_final'], order=(10, 100))['run_after_terminal_deps'] = True
    jobs.sort(key=lambda j: (-1, 0) if j['id'] not in priority else priority[j['id']])
    # Actual previous job durations establish costs; extrapolations are labelled.
    def duration(record):
        a = record['attempts'][-1]
        return (datetime.fromisoformat(a['finished_at']) - datetime.fromisoformat(a['started_at'])).total_seconds()
    cost_rows = []
    for item in training_runs:
        parent = state['jobs'][next(j for j in old_jobs if old_jobs[j].get('argv') and '--spec' in old_jobs[j]['argv'] and
                         old_jobs[j]['argv'][old_jobs[j]['argv'].index('--spec') + 1] == item['parent_spec'])]
        cost_rows.append({'job': item['run_id'], 'category': 'additional', 'estimated_GPU_seconds': duration(parent),
                          'basis': 'Same architecture/objective/budget measured completed regenerated-cohort duration; original labels/data differ'})
    for job in jobs:
        if job['id'] not in new_ids or not job.get('gpu') or job['id'].startswith('O4_train_') or job['id'] in dry_ids:
            continue
        name = job['id']
        arch = next((a for a in ARCHES if a in name), 'unet_resnet50' if 'unet_resnet50' in name else 'efficientvit_b0')
        if '_eval_robust_' in name:
            candidates = [r for n, r in state['jobs'].items() if n.startswith('P3_eval_') and arch in n and '_N2500_' in n]
            estimate = statistics.mean(duration(r) for r in candidates)
            basis = 'Measured current matched primary complete canonical matrix; clean cells now reused'
        elif name.endswith('_wf_native'):
            candidates = [r for n, r in state['jobs'].items() if 'wf_native_' in n and arch in n and r['status'] == 'succeeded' and r['attempts']]
            estimate = statistics.mean(duration(r) for r in candidates) if candidates else 90
            basis = 'Measured corresponding native-border adapter; unknown original UNet uses90s initial estimate'
        elif name.endswith('_sturm_valid'):
            spec = q.read_json(job['argv'][job['argv'].index('--spec') + 1])
            candidates = [r for n, r in state['jobs'].items() if n.startswith('P3_sturm_valid_') and arch in n and
                          ('_N100_' in n) == (len(spec['conditions']) == 1) and r['status'] == 'succeeded']
            estimate = statistics.mean(duration(r) for r in candidates) if candidates else 40
            basis = 'Measured affected-chip validity adapter, same condition cardinality where available'
        elif name.startswith('O4_protocol_'):
            estimate, basis = 35, 'Prior batch16 functional checks; no training result'
        elif name == first_name:
            estimate, basis = 12, 'Existing90chipS11clean timings plus checkpoint construction'
        else:
            outputs = [q.read_json(j['outputs'][0]['path']) for j in old_raw['jobs'] if j['id'].startswith('P3_eval_') and arch in j['id'] and '_N100_' in j['id']]
            if not outputs:
                outputs = [q.read_json(j['outputs'][0]['path']) for j in old_raw['jobs'] if j['id'].startswith('P3_eval_') and 'efficientvit_b0' in j['id'] and '_N100_' in j['id']]
            estimate = statistics.mean(sum(v['elapsed_seconds'] for k, v in p['cells'].items() if k.endswith('__clean')) for p in outputs) + 15
            if arch == 'unet_resnet50':
                estimate *= 2
            basis = 'Measured threeclean-cell inference plus15s load allowance; UNet2x extrapolation unmeasured until first readout'
        cost_rows.append({'job': name, 'category': 'additional', 'estimated_GPU_seconds': estimate, 'basis': basis})
    hours = sum(r['estimated_GPU_seconds'] for r in cost_rows) / 3600
    cost = {'captured_at': q.now(), 'original_remaining_GPU_hours': 0, 'additional_remaining_GPU_hours': hours,
            'total_remaining_GPU_hours': hours, 'ideal_two_GPU_wall_hours': hours / 2,
            'jobs': cost_rows, 'uncertainty': 'Measured same-machine family durations with declared extrapolations; allow25percent or more for original UNet readouts, reference hashing, serialized CPU collectors and resource waits. No sleep/reboot guarantee.'}
    q.atomic_json(queue_dir / 'cost_before_launch.json', cost)
    manifest = copy.deepcopy(old_raw)
    manifest.update(jobs=jobs, run_id='original_N2500_recovered_20261006_prompt4',
        manifest_revision='prompt4_original_data_reproduction_and_matched_controls', previous_queue=str(old_path),
        source_repair_snapshot=str(snapshot), fingerprints=old_raw['fingerprints'],
        scientific_scope={'programme': 'prompt4', 'original_checkpoint_set': str(inventory_path),
            'training_job_ids': [t['run_id'] for t in training_runs], 'computation_job_ids': computation_ids + numerical_collectors,
            'primary_result_dir': str(final_out), 'original_data_receipts': {str(n): str(p) for n, p in receipts.items()},
            'new_training_runs': 44, 'full_budget_runs': 36, 'N100_runs': 8, 'ceiling': 48,
            'GroupA_OFA_shared_with_B': 4, 'mask_equivalence_gate': str(data_gate_path),
            'conditions_frozen': full_conditions, 'seeds': SEEDS, 'no_accuracy_gate': True,
            'training_specifications': training_runs, 'comparison_spec': str(comparison_spec),
            'regenerated_result_dir': str(REGEN / 'analysis_prompt3_complete'),
            'regenerated_scope_status': 'computation-complete; independent dry/device/reference/history limits remain'},
        resource_estimate=cost)
    manifest['host_limits'] = {**manifest['host_limits'], 'disk_path': str(ORIGINAL)}
    manifest_path = queue_dir / 'manifest.json'
    q.atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    q.load_manifest(manifest_path)
    result = migrate(old_path, manifest_path)
    pointer.update(active_manifest=str(manifest_path), active_state=str(manifest_path.with_name('manifest.state.json')),
                   active_wrapper=str(wrapper), previous_manifest=str(old_path), original_data_cohort=ORIGINAL.name,
                   original_data_receipt=str(receipts[2500]), migration_receipt=str(queue_dir / 'migration.json'))
    q.atomic_json(pointer_path, pointer)
    q.atomic_json(dry_root / 'queue_resume.json', {'argv': [PY, str(wrapper), 'resume', '--manifest', str(manifest_path)]})
    preparation = {**result, 'created_utc': q.now(), 'snapshot': str(snapshot), 'code_sha256': code_sha,
        'new_training_runs': 44, 'original_rescores': 28, 'new_canonical_jobs': len(canonical_ids),
        'new_correction_jobs': len(correction_ids), 'additional_dry_jobs': 12, 'new_jobs': len(new_ids),
        'estimated_GPU_hours': hours, 'ideal_two_GPU_wall_hours': hours / 2,
        'automatic_evaluations': 'Every training node -> finalizedreceipt -> clean -> optionalprimaryrobust (reuseclean) -> corrections -> paired/finalcollectors',
        'controls': {a: [PY, str(wrapper), a, '--manifest', str(manifest_path)] for a in ['status', 'pause', 'resume', 'stop']}}
    q.atomic_json(queue_dir / 'preparation.json', preparation)
    print(json.dumps(preparation), flush=True)


if __name__ == '__main__':
    prepare()
