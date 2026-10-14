"""Freeze one bounded compression pilot around revision_queue.py (no new scheduler)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import sys
import platform

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import revision_queue as q
from models.compression import compression_graph_spec

PYTHON = '/home/schmalstieg/miniconda3/envs/esi/bin/python'
DATA = Path('/data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006/data')
TEACHER = Path('/home/schmalstieg/.cache/huggingface/hub/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-TL-Sen1Floods11/snapshots/918b9f140bb1783716664a2421ea3253d806017d/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt')
TEACHER_SHA = '3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60'
EXCLUSION = REPO / 'experiments/revision/prompt5_closure/overlap_exclusion_plan.json'


def pools(receipt, source):
    by_id = {row['file']: row for row in source['selected_source_rows']}
    groups = defaultdict(list)
    for row in receipt['files']:
        source_id = by_id[row['name']]['source'].split('/')[0]
        # Merge event/pre-event, sites and acquisition dates within a GDACS event.
        group = '_'.join(source_id.split('_')[:2]) if source_id.startswith('GDACS_') else source_id.split('_')[0]
        groups[group].append(row['name'])
    rng = random.Random(1729); ordered = sorted(groups); rng.shuffle(ordered)
    chosen = {'fit': [], 'interaction': []}
    for name, ids in (('fit', ordered[:256]), ('interaction', ordered[256:512])):
        for group in ids:
            file = rng.choice(sorted(groups[group]))
            chosen[name].append({'file': file, 'group': group, 'source': by_id[file]['source']})
    assert not ({row['group'] for row in chosen['fit']} & {row['group'] for row in chosen['interaction']})
    return chosen, {'available_source_event_groups': len(groups), 'one_image_per_group': True,
        'group_definition': 'GDACS event identifier merges all sites/date phases; nonGDACS events use conservative first prefix',
        'limitation': 'distinct GDACS events can revisit a geographic site; geographic independence is approximate'}


def snapshot(base):
    # stats_scene_level imports this helper when the terminal collector starts.
    files = [REPO / 'config.py', REPO / 'figures/fig_style.py']
    for name in ('models', 'training', 'quantization', 'distillation', 'data', 'evaluation', 'utils', 'scripts', 'experiments'):
        files += [p for p in (REPO / name).rglob('*.py') if '__pycache__' not in p.parts
                  and 'revision' not in p.relative_to(REPO / name).parts]
    hashes = {str(p.relative_to(REPO)): q.file_hash(p) for p in sorted(set(files))}
    target = base / 'frozen' / q.digest(hashes)[:16]
    if target.exists():
        identity = q.read_json(target / 'source_identity.json')
        if identity['sha256'] != q.digest(hashes) or identity['files'] != hashes:
            raise RuntimeError('existing immutable source snapshot identity differs')
        if any(q.file_hash(target / rel) != sha for rel, sha in hashes.items()):
            raise RuntimeError('existing immutable source snapshot bytes changed')
        return target, hashes
    target.mkdir(parents=True, exist_ok=False)
    for rel in hashes:
        dest = target / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dest); dest.chmod(0o444)
    q.atomic_json(target / 'source_identity.json', {'sha256': q.digest(hashes), 'files': hashes})
    return target, hashes


def prepare(base, inventory, phase, preflight=None, queue_name=None):
    base = base.resolve(); base.mkdir(parents=True, exist_ok=True)
    inv = q.read_json(inventory)
    assert len(inv['gpus']) == 2 and all('RTX 4090' in d['name'] for d in inv['gpus'])
    receipt = q.read_json(DATA / 'original_N2500_receipt.json')
    source = q.read_json(DATA / 'original_source_manifest.json')
    assert receipt['status'] == 'verified' and receipt['n_selected'] == 2500
    assert receipt['dataset_sha256'] == '1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60'
    chosen, group_info = pools(receipt, source)
    from data.sen1floods11 import get_dataloaders
    _, val, _ = get_dataloaders('/home/schmalstieg/data/sen1floods11', batch_size=4,
                               num_workers=2, img_size=224, eval_img_size=512, drop_last=False)
    calibration_ids = [Path(p).name for p in val.dataset.image_files]
    assert len(calibration_ids) == 89
    code, hashes = snapshot(base)
    if queue_name is not None and (Path(queue_name).name != queue_name or queue_name in ('', '.', '..')):
        raise ValueError('queue_name must be a single new directory name')
    queue_dir = base / (queue_name or phase); queue_dir.mkdir(exist_ok=False)
    wrapper = queue_dir / 'revision_queue.py'
    shutil.copyfile(code / 'experiments/revision_queue.py', wrapper); wrapper.chmod(0o444)
    data_inputs = []
    # Primary data-byte identity is checked once here and pinned per GPU job.
    if phase == 'pilot':
        for row in receipt['files']:
            p = Path(receipt['pool']) / row['name']
            if p.stat().st_size != row['bytes'] or q.file_hash(p) != row['sha256']:
                raise RuntimeError(f'original cohort cache bytes changed: {p}')
            data_inputs.append({'path': str(p), 'sha256': row['sha256']})
    pin = lambda path: {'path': str(Path(path).resolve()), 'sha256': q.file_hash(path)}
    identity_inputs = [pin(DATA / name) for name in
        ('original_N2500_index.json', 'original_N2500_receipt.json', 'original_source_manifest.json')]
    # The live lineage check has streamed the teacher bytes. Jobs independently
    # recheck this SHA before loading; avoid another preparation-time GB read.
    identity_inputs += [{'path': str(TEACHER), 'sha256': TEACHER_SHA}, pin(EXCLUSION)]
    verification = []
    if phase == 'pilot':
        required = [REPO / 'experiments/revision/compression_v2' / name for name in
            ('metric_check.json', 'graph_check.json', 'job_check.json', 'integration_check.json',
             'artifact_cpu_check.json', 'quant_graph_independent_check.json',
             'mobilenet_confirmation_cpu_check.json', 'mobilenet_artifact_cpu_check.json',
             'lineage_check.json', 'collector_check.json')]
        required += [REPO / 'experiments/revision/compression_v2_quant_checks.json',
                     REPO / 'experiments/revision/compression_v2_queue_checks/result.json']
        for evidence in required:
            checked = q.read_json(evidence)
            if checked.get('status') != 'passed':
                raise RuntimeError(f'required CPU verification has not passed: {evidence}')
            for source_path, tested_sha in checked.get('source_sha256', {}).items():
                if q.file_hash(source_path) != tested_sha:
                    raise RuntimeError(f'verification is stale for changed source: {source_path}; rerun {evidence}')
            if 'runner_source_sha256' in checked and checked['runner_source_sha256'] != hashes['experiments/revision_queue.py']:
                raise RuntimeError(f'detached queue check tested a different runner: {evidence}')
            verification.append(pin(evidence))
        for name in ('real_gpu_preflight', 'resource_timing', 'mb_resource_timing'):
            path = base / 'results' / (name + '.json')
            measured = q.read_json(path)
            if measured.get('status') != 'complete' or measured.get('outcome') != 'completed':
                raise RuntimeError(f'measured resource phase incomplete: {path}')
            verification.append(pin(path))
        dependencies = [path for root in ('efficientvit', 'flooddistill_revision_python')
                        for path in (REPO.parent / '_deps' / root).rglob('*.py')
                        if '__pycache__' not in path.parts and '.git' not in path.parts]
        dependency_pins = [pin(path) for path in sorted(dependencies)]
        versions = {}
        for name in ('torch', 'torchvision', 'numpy', 'scipy', 'timm', 'transformers', 'rasterio', 'albumentations'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'not installed'
        environment_path = queue_dir / 'environment.json'
        q.atomic_json(environment_path, {'status': 'recorded', 'python': sys.version,
            'python_executable': str(Path(sys.executable).resolve()), 'platform': platform.platform(),
            'package_versions': versions, 'dependency_source_pins': dependency_pins,
            'scope': 'Python package versions and imported external dependency source trees; not a full binary/container image'})
        environment_path.chmod(0o444)
        identity_inputs += [*verification, pin(environment_path), *dependency_pins]
    source_inputs = [{'path': str(code / rel), 'sha256': sha} for rel, sha in hashes.items()]
    common = {'seed': 42, 'workers': 2, 'pseudo': receipt['pool'], 'index': receipt['index'],
        'teacher': str(TEACHER), 'teacher_sha256': TEACHER_SHA, 'calibration_seed': 1729,
        'quantizer_calibration_ids': calibration_ids, 'feature_pools': chosen,
        'feature_pools_sha256': q.digest(chosen), 'smallest_tap_dimension': compression_graph_spec('A3')['width_list'][-1],
        'roots': {'s11': '/home/schmalstieg/data/sen1floods11', 'sturm': '/home/schmalstieg/data/STURM-Flood',
                  'wf': '/home/schmalstieg/data/WorldFloods'},
        'ptq_precisions': {'W8A8': [8, 8], 'W4A8': [4, 8], 'W4A4': [4, 4]},
        'protocols': {'s11': 'S11_TEST512_LEGACY', 'sturm': 'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1',
                      'wf': 'P3_WF_NATIVE_ZERO_PAD512_V1'},
        'overlap_plan': str(EXCLUSION),
        'excluded_sturm_chips': ['EMSR465_AOI01_07_11_1_1.tif', 'EMSR501_AOI01_10_05_2_1.tif', 'EMSR501_AOI01_10_05_2_2.tif'],
        'excluded_wf_maps': ['EMSR438_AOI02_DEL_MONIT03.tif']}
    # Resolve readers' real paths; the values above are checked against prior specs.
    reference = q.read_json('/data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006/queue_prompt4/specs/original_data_comparisons.json')
    old_run = next(iter(reference['runs'].values())) if isinstance(reference['runs'], dict) else reference['runs'][0]
    old_spec = q.read_json(old_run['evaluation'])['spec']
    if old_spec:
        common['roots'] = old_spec['roots']
    for root in common['roots'].values():
        if not Path(root).exists():
            raise FileNotFoundError(f'canonical reader root missing: {root}')
    env = {'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'WANDB_MODE': 'disabled', 'HF_HUB_OFFLINE': '1', 'NO_ALBUMENTATIONS_UPDATE': '1',
        'DATA_ROOT': '/home/schmalstieg/data', 'LIGHTFORMER_ROOT': '/home/schmalstieg/repos/_deps/lightformer_574577dd',
        'PYTHONPATH': str(code) + ':/home/schmalstieg/repos/_deps/efficientvit:/home/schmalstieg/repos/_deps/flooddistill_revision_python',
        'MPLCONFIGDIR': '/tmp/flooddistill_compression_matplotlib',
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    jobs, results, run_specs = [], {}, {}
    artifacts = Path('/data1/schmalstieg/flooddistill/revision_20261005') / base.name
    def command(name, action, extension=None, deps=(), seconds=None, gpu=True, after_terminal=False):
        spec = {**copy.deepcopy(common), **(extension or {})}
        spec_path = queue_dir / 'specs' / f'{name}.json'; q.atomic_json(spec_path, spec); spec_path.chmod(0o444)
        result = base / 'results' / f'{name}.json'
        job = {'id': name, 'kind': 'command', 'cwd': str(code),
            'argv': ['/usr/bin/ionice', '-c', '2', '-n', '7', PYTHON, str(code / 'scripts/compression_jobs.py'),
                     '--action', action, '--spec', str(spec_path), '--out', str(result)],
            'gpu': gpu, 'deps': list(deps), 'inputs': [*source_inputs, pin(spec_path), *identity_inputs,
                                                     *(data_inputs if gpu else [])],
            'outputs': [{'path': str(result), 'type': 'json', 'equals': {'status': 'complete'},
                         'required_keys': ['status', 'outcome']}]}
        if seconds is not None:
            job.update(max_seconds=seconds, shutdown_grace_seconds=45 if seconds > 60 else 2)
        if gpu:
            job['required_free_mb'] = int(preflight['peak_reserved_mb']) if preflight else 22000
        if after_terminal:
            job['run_after_terminal_deps'] = True
        jobs.append(job); results[name] = str(result); run_specs[name] = spec
        return result
    frontier = {name: {**compression_graph_spec(name), 'status': 'predeclared'} for name in ('A0', 'A1', 'A2', 'A3')}
    frontier['M0'] = {**compression_graph_spec('M0'), 'status': 'source_gated_second_family_seed42'}
    confirmation = {'status': 'source_gated_within_fixed_pilot_cap', 'seeds': [42, 64, 91, 107],
        'selection': 'A3W4A4; A1W4A8 fallback only if A3W4A4 is technically invalid',
        'second_family': {'architecture': 'mobilenet_v3_small_from_scratch',
                          'activations': 'native Hardswish/ReLU preserved', 'graph': 'M0',
                          'seeds_queued': [42], 'seeds_deferred': [64, 91, 107]},
        'beta_sensitivity': {'multipliers': [.5, 2.], 'methods': ['B2', 'B3'], 'same_updates': 1560},
        'published_low_bit_KD_baseline': 'faithful QFD/SQuaT implementation pending; no reproduction claimed',
        'OFA_comparator': 'existing matched float controls retained as references; ReLU6 matched OFA continuation pending',
        'time_matched_ordinary_KD': 'required before interpreting positive screening beyond equal-update B1 control'}
    if phase == 'preflight':
        command('real_gpu_preflight', 'preflight', seconds=300)
        gpu_budget = 300
    elif phase == 'timing':
        command('resource_timing', 'timing', seconds=180)
        gpu_budget = 180
    elif phase == 'mb_timing':
        command('mb_resource_timing', 'timing', {'graph': 'M0'}, seconds=180)
        gpu_budget = 180
    else:
        timing = q.read_json(base / 'results/resource_timing.json')
        if timing.get('status') != 'complete' or not timing.get('no_accuracy_computed'):
            raise RuntimeError('main pilot requires the measured source-only resource timing')
        terminal_sources = []
        source_job_names = ['verification_preflight']
        jobs.append({'id': 'verification_preflight', 'kind': 'gate', 'cwd': str(code),
                     'gpu': False, 'deps': [], 'inputs': verification,
                     'outputs': [{'path': item['path'], 'type': 'json',
                                  'equals': {'status': 'passed' if '/results/' not in item['path'] else 'complete'}}
                                 for item in verification]})
        results['verification_preflight'] = verification[0]['path']
        def primary_fp(graph):
            name = f'{graph}_FP_s42'; source_job_names.append(name); terminal_sources.append(name)
            command(name, 'warmup', {'graph': graph, 'run_dir': str(artifacts / name)},
                    deps=('mechanism_source_gate',) if graph == 'A0' else (), seconds=1600)
            ptq = f'{graph}_PTQ'; source_job_names.append(ptq)
            command(ptq, 'calibrate', {'graph': graph, 'warmup_receipt': results[name],
                    'transform_receipt': results['teacher_feature_transform']},
                    deps=(name, 'teacher_feature_transform'), seconds=300)
        def continuations(cell, interaction_name, extension, deps, *, seconds=480):
            for method in ('B1', 'B2', 'B3'):
                name = f'{cell}_{method}_s{extension.get("seed", 42)}'
                method_names.append(name); source_job_names.append(name); terminal_sources.append(name)
                arm = {**extension, 'method': method,
                       'interaction_receipt': results[interaction_name],
                       'transform_receipt': results['teacher_feature_transform'],
                       'run_dir': str(artifacts / name)}
                if method == 'B3':
                    arm['duplicate_receipt'] = results[f'{cell}_B2_s{extension.get("seed", 42)}']
                command(name, 'qat', arm, deps=deps, seconds=seconds)
        command('teacher_feature_transform', 'features', seconds=300)
        source_job_names.append('teacher_feature_transform')
        # Prioritize the two smaller fixed graphs and their paired continuations
        # before the A0 anchor. Ordering is independent of new accuracy.
        for graph in ('A1', 'A3'):
            primary_fp(graph)
        cells = [('A1', 'W4A8'), ('A3', 'W4A8'), ('A3', 'W8A8'), ('A3', 'W4A4')]
        cell_interactions = {}
        method_names = []
        for graph, precision in cells:
            cell = f'{graph}_{precision}'
            bits = common['ptq_precisions'][precision]
            interaction_name = f'{cell}_interaction'
            cell_interactions[cell] = interaction_name
            command(interaction_name, 'interaction', {'graph': graph, 'precision': precision, 'bits': bits,
                    'calibration_receipt': results[f'{graph}_PTQ'], 'transform_receipt': results['teacher_feature_transform']},
                    deps=(f'{graph}_PTQ',), seconds=240)
            source_job_names.append(interaction_name)
            continuations(cell, interaction_name, {'graph': graph, 'precision': precision, 'bits': bits},
                          (interaction_name,))
        hard = 'A3_W4A4'
        gate_deps = tuple(f'{hard}_{method}_s42' for method in ('B1', 'B2', 'B3'))
        fallback = 'A1_W4A8'
        fallback_deps = tuple(f'{fallback}_{method}_s42' for method in ('B1', 'B2', 'B3'))
        command('mechanism_source_gate', 'gate', {'interaction_receipt': results[cell_interactions[hard]],
                'primary_cell': {'graph': 'A3', 'precision': 'W4A4', 'bits': [4, 4], 'interaction_receipt': results[cell_interactions[hard]]},
                'fallback_cell': {'graph': 'A1', 'precision': 'W4A8', 'bits': [4, 8], 'interaction_receipt': results[cell_interactions[fallback]]},
                'fallback_method_receipts': [results[name] for name in fallback_deps],
                'method_receipts': [results[name] for name in gate_deps]}, deps=gate_deps + fallback_deps, gpu=False, after_terminal=True)
        source_job_names.append('mechanism_source_gate')
        for method in ('random', 'quant', 'independent', 'total'):
            name = f'selected_{method}_s42'; method_names.append(name); source_job_names.append(name); terminal_sources.append(name)
            command(name, 'qat', {'graph': 'A3', 'precision': 'W4A4', 'bits': [4, 4], 'method': method,
                'interaction_receipt': results[cell_interactions[hard]], 'transform_receipt': results['teacher_feature_transform'],
                'expansion_gate': results['mechanism_source_gate'], 'run_dir': str(artifacts / name)},
                deps=('mechanism_source_gate',), seconds=480)
        primary_fp('A0')
        conditional = {'conditional_gate': results['mechanism_source_gate'], 'select_gate_cell': True,
                       'graph': 'A3', 'precision': 'W4A4', 'bits': [4, 4]}
        for seed in (64, 91, 107):
            prefix = f'confirmation_s{seed}'
            fp_name = prefix + '_FP'; cal_name = prefix + '_PTQ'; inter_name = prefix + '_interaction'
            selected = {**conditional, 'seed': seed}
            command(fp_name, 'warmup', {**selected, 'run_dir': str(artifacts / fp_name)},
                    deps=('mechanism_source_gate',), seconds=1600)
            command(cal_name, 'calibrate', {**selected, 'warmup_receipt': results[fp_name],
                    'transform_receipt': results['teacher_feature_transform']},
                    deps=(fp_name,), seconds=300)
            command(inter_name, 'interaction', {**selected, 'calibration_receipt': results[cal_name],
                    'transform_receipt': results['teacher_feature_transform']},
                    deps=(cal_name,), seconds=240)
            source_job_names.extend((fp_name, cal_name, inter_name)); terminal_sources.append(fp_name)
            continuations('confirmation', inter_name, selected, (inter_name,))
        # A separate measured M0 step is a technical feasibility gate. It never
        # reads external accuracy and cannot alter the second family's recipe.
        mb_timing_path = base / 'results/mb_resource_timing.json'
        jobs.append({'id': 'M0_technical_preflight', 'kind': 'gate', 'cwd': str(code),
                     'gpu': False, 'deps': [], 'inputs': [],
                     'outputs': [{'path': str(mb_timing_path), 'type': 'json',
                                  'equals': {'status': 'complete', 'outcome': 'completed', 'batch_size': 16},
                                  'finite': ['peak_reserved_mb', 'feature_qat_step_seconds'],
                                  'ranges': {'peak_reserved_mb': [1, min(d['total_mb'] for d in inv['gpus']) - 2048]}}]})
        results['M0_technical_preflight'] = str(mb_timing_path)
        source_job_names.append('M0_technical_preflight')
        mb_selected = {'conditional_gate': results['mechanism_source_gate'],
                       'graph': 'M0', 'precision': 'W4A8', 'bits': [4, 8], 'seed': 42}
        mb_fp = 'M0_FP_s42'; mb_cal = 'M0_PTQ'; mb_inter = 'M0_W4A8_interaction'
        command(mb_fp, 'warmup', {**mb_selected, 'run_dir': str(artifacts / mb_fp)},
                deps=('mechanism_source_gate', 'M0_technical_preflight'), seconds=2400)
        command(mb_cal, 'calibrate', {**mb_selected, 'warmup_receipt': results[mb_fp],
                'transform_receipt': results['teacher_feature_transform']}, deps=(mb_fp,), seconds=300)
        command(mb_inter, 'interaction', {**mb_selected, 'calibration_receipt': results[mb_cal],
                'transform_receipt': results['teacher_feature_transform']}, deps=(mb_cal,), seconds=240)
        source_job_names.extend((mb_fp, mb_cal, mb_inter)); terminal_sources.append(mb_fp)
        continuations('M0_W4A8', mb_inter, mb_selected, (mb_inter,), seconds=600)
        if mb_timing_path.exists():
            mb_profile = q.read_json(mb_timing_path)
            if mb_profile.get('status') == 'complete':
                for job in jobs:
                    if job.get('gpu') and job['id'].startswith('M0_'):
                        job['required_free_mb'] = max(job['required_free_mb'], int(mb_profile['peak_reserved_mb']))
        for multiplier, suffix in ((.5, '0p5'), (2., '2')):
            for metric_key in ('B2', 'B3'):
                method = f'{metric_key}_beta{suffix}'; name = f'selected_{method}_s42'
                method_names.append(name); source_job_names.append(name); terminal_sources.append(name)
                command(name, 'qat', {**conditional, 'method': method, 'feature_metric_key': metric_key,
                        'beta_multiplier': multiplier, 'expansion_gate': results['mechanism_source_gate'],
                        'interaction_receipt': results[cell_interactions[hard]],
                        'transform_receipt': results['teacher_feature_transform'],
                        'run_dir': str(artifacts / name)}, deps=('mechanism_source_gate',), seconds=480)
        design_deps = tuple(source_job_names)
        command('source_design_freeze', 'freeze', {'receipts': [results[name] for name in design_deps]},
                deps=design_deps, gpu=False, after_terminal=True)
        terminal_names = []
        for name in terminal_sources:
            terminal_name = 'terminal_' + name; terminal_names.append(terminal_name)
            previous = run_specs[name]
            command(terminal_name, 'terminal', {'graph': previous['graph'], 'method': previous.get('method', 'B0'),
                    'seed': previous['seed'], 'precision': previous.get('precision', 'FP32'),
                    'quantized': name in method_names, 'method_receipt': results[name],
                    'design_gate': results['source_design_freeze'],
                    'artifact_dir': str(artifacts / 'exports' / terminal_name)},
                    deps=('source_design_freeze',),
                    seconds=330 if previous['graph'] == 'M0' or name not in method_names else 240)
        # Downstream collection runs even after a failure, retaining every frontier cell.
        all_jobs = tuple(j['id'] for j in jobs)
        frontier['A2']['status'] = 'deferred_predeclared_intermediate_budget'
        command('pilot_collection', 'collect', {'receipts': list(results.values()),
                'frontier': frontier, 'confirmation': confirmation,
                'queue_state': str(queue_dir / 'manifest.state.json')}, deps=all_jobs, gpu=False, after_terminal=True)
        gpu_budget = 43200 - 660  # Preflight300 + source timing180 + M0 timing180.
        fp_forecast = 50 * (156 * preflight['fp_step_seconds'] + timing['epoch_loader_seconds']) + 25 * timing['validation_pass_seconds']
        qat_forecast = (1560 * timing['feature_qat_step_seconds'] + 10 * timing['epoch_loader_seconds'] +
                        5 * timing['validation_pass_seconds'] + 32 * timing['diagnostic_step_seconds'])
        for job in jobs:
            if job['kind'] == 'command' and job['gpu'] and job['id'] not in terminal_names:
                job['deps'] = list(dict.fromkeys([*job['deps'], 'verification_preflight']))
    reservation = sum(j.get('max_seconds', 0) for j in jobs if j['gpu'])
    if reservation > gpu_budget:
        raise ValueError(f'pilot reservations {reservation}s exceed {gpu_budget}s')
    inv = q.read_json(inventory)
    assert len(inv['gpus']) == 2 and all('RTX 4090' in d['name'] for d in inv['gpus'])
    locks = Path('/data1/schmalstieg/flooddistill')
    manifest = {'schema_version': 1, 'jobs': jobs, 'env': env, 'poll_seconds': 5, 'nice': 10,
        'gpu_devices': [{'index': d['index'], 'uuid': d['uuid'],
                         'lock': str(locks / f"revision_gpu_{d['uuid']}.lock")} for d in inv['gpus']],
        'gpu_lock': str(locks / 'revision_gpu.lock'), 'gpu_budget_seconds': gpu_budget,
        'headroom_mb': 2048,
        'host_limits': {'min_available_ram_gib': 16, 'min_free_disk_gib': 12,
                        'max_load_per_cpu': 1.0, 'disk_path': str(artifacts.parent)},
        'resource_limit_basis': 'preflight host244GiB available,800GiB data disk free; reserve16GiB RAM for two teacher jobs and12GiB disk for this bounded artifacts/cache workload',
        'output_root': str(artifacts), 'fingerprints': {'source_sha256': q.digest(hashes),
            'cohort_sha256': receipt['dataset_sha256'], 'teacher_sha256': TEACHER_SHA,
            'prompt_sha256': q.file_hash(REPO / 'prompts/codex_compression_prompt_v2.txt')},
        'recipe': {'fp': 'matched FP32 AdamW5e-5 wd.05 batch16 drop_last50epochs cosine clip5 sourceval2epoch strict-best',
            'qat': 'fixed1560updates AdamW1e-5 wd1e-5 clip1 no_amp cosine10epochs allQDQenabled sourceval2epoch strict-best',
            'historical_qat_difference': 'paired mechanistic pilot usesQDQall10epochs ratherthan historical firstepochoff; all methods share this fixed choice',
            'feature_probe': '8x8 bilinear align_corners=False; nearest-exact centered class/valid support; equal availableclassmass thenequalimagemass',
            'history': 'author reports unsuccessful teacher-derived STURM criteria; didnotchoose architecture/OFA; externaldatasets not historically untouched'},
        'declared_frontier': frontier, 'confirmation': confirmation, 'group_info': group_info,
        'authorized_dual_gpu_basis': 'latest user explicitly requested scheduling across GPUs; overrides earlier single-device preference',
        'pilot_cap_seconds': 43200, 'reserved_main_gpu_seconds': reservation,
        'reserved_preflight_gpu_seconds': 660 if phase == 'pilot' else reservation,
        'enforcement': 'perownedcommand hardtimecaps including graceful checkpoint interval; no automaticGPUretry; summedreservations <=12GPUhours'}
    if phase == 'pilot':
        manifest['verification_evidence'] = verification
        manifest['environment_receipt'] = str(environment_path)
    if phase == 'pilot':
        manifest['resource_timing_basis'] = {'path': str(base / 'results/resource_timing.json'),
            'sha256': q.file_hash(base / 'results/resource_timing.json'),
            'fp_50epoch_forecast_seconds': fp_forecast, 'qat_10epoch_forecast_seconds': qat_forecast,
            'primary_fp_cap_seconds': 1600, 'primary_qat_cap_seconds': 480,
            'second_family_fp_cap_seconds': 2400, 'second_family_qat_cap_seconds': 600,
            'terminal_cap_seconds': {'efficientvit_qat': 240, 'native_fp_or_mobilenet': 330},
            'terminal_timing_basis': 'prior full clean S11/STURM/WF evaluation44 attempts:60.954-65.088s; current serial readers plus export/runtime are unmeasured and receive conservative extra time',
            'limitation': 'single measured source batch/epoch timing; file caches and device contention change wall time'}
    path = queue_dir / 'manifest.json'; q.atomic_json(path, manifest); path.chmod(0o444)
    q.load_manifest(path)
    q.atomic_json(queue_dir / 'preparation.json', {'status': 'verified', 'manifest': str(path),
        'manifest_sha256': q.file_hash(path), 'source_root': str(code), 'job_count': len(jobs),
        'reserved_gpu_seconds': reservation, 'available_gpu_seconds': gpu_budget, 'feature_pools': chosen,
        'group_info': group_info, 'frontier': frontier, 'confirmation': confirmation})
    print(json.dumps({'manifest': str(path), 'runner': str(wrapper), 'jobs': len(jobs),
                      'reserved_gpu_seconds': reservation}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--phase', choices=('preflight', 'timing', 'mb_timing', 'pilot'), required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--queue-name', help='New immutable queue directory; defaults to the phase name')
    args = parser.parse_args()
    if args.phase == 'pilot' and args.preflight is None:
        parser.error('pilot requires real measured preflight')
    prepare(args.base, args.inventory, args.phase, q.read_json(args.preflight) if args.preflight else None,
            queue_name=args.queue_name)


if __name__ == '__main__': main()
