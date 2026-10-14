"""Read completed revision receipts and export the requested results report.

This does no model inference, training, queue mutation, or writes to /data1.
The failed historical collector and all its inputs/outputs remain preserved.
"""
from __future__ import annotations

import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/data1/schmalstieg/flooddistill/revision_20261005')
ORIGINAL = BASE / 'original_N2500_recovered_20261006'
REGEN = BASE / 'regen_N2500_approved_20261005'
OUT = REPO / 'experiments/revision/prompt4_report_20261007'
RESULTS = ORIGINAL / 'analysis_original_data'
WF = 'P3_WF_NATIVE_ZERO_PAD512_V1'
ST = 'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1'
PROTOCOLS = [('s11', 'clean', 'S11 main512'), ('sturm', 'clean', 'STURM legacy512'),
             ('wf', 'clean', 'WorldFloods legacy512'), ('sturm', ST + ':clean', 'STURM finite support'),
             ('wf', WF, 'WorldFloods native borders')]
ARCH = {'efficientvit_b0': 'EffViT', 'mobilenet_v3_small': 'MBV3Small',
        'lightformer_resnet18': 'LightFormerR18', 'unet_resnet50': 'UNetR50',
        'teacher': 'Teacher', 'index:mndwi_zero': 'MNDWI'}
VARIANT = {'historical_true_ce_clip0': 'historical CE', 'historical_lkd_clip0': 'historical KD',
           'historical_ofa_clip5_allvalid': 'OFA (A/B shared)', 'matched_focal_g2_clip5': 'full focal',
           'matched_lkd_clip5_valid': 'matched KD', 'matched_ofa_clip5_valid': 'matched OFA',
           'fixed_comparator': 'fixed', 'original_checkpoint_ce': 'original CE',
           'original_checkpoint_dfkd': 'original KD', 'original_checkpoint_ofa': 'original OFA',
           'original_checkpoint_ce_unet': 'original CE', 'original_checkpoint_dfkd_unet': 'original KD'}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def export(path, records):
    if not records:
        raise ValueError(f'No rows for {path}')
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(records)


def num(x):
    return float(x) if x not in (None, '') else None


def pp(x, digits=3):
    v = num(x)
    return f'{100 * v:.{digits}f}' if v is not None else 'NA'


def mean_sd(r, digits=2):
    return pp(r['mean'], digits) + ('/' + pp(r['sample_SD_ddof1'], digits)
                                  if r['sample_SD_ddof1'] else '/--')


def label(r):
    return f"{ARCH.get(r['architecture'], r['architecture'])} {VARIANT.get(r['variant'], r['variant'])} N{r['budget']}"


def audit_outputs(manifest, state, queue_module):
    audit = []
    cache = {}
    for j in manifest['jobs']:
        record = state['jobs'][j['id']]
        if record['status'] != 'succeeded':
            continue
        # Every declared output is verified against the durable completion hash.
        expected = record.get('evidence') if j['kind'] == 'gate' else record['attempts'][-1]['output_validation']
        actual = []
        for spec in j['outputs']:
            p = Path(spec['path'])
            if p not in cache:
                cache[p] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
            entry = cache[p]
            assert entry['bytes'] >= spec.get('min_bytes', 1), p
            if spec.get('sha256'):
                assert entry['sha256'] == spec['sha256'], p
            if spec.get('type') == 'json':
                data = load(p)
                for key in spec.get('required_keys', []):
                    queue_module.field(data, key)
                for key, value in spec.get('equals', {}).items():
                    assert queue_module.field(data, key) == value, (p, key)
                for key in spec.get('finite', []):
                    v = queue_module.field(data, key)
                    assert isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for key, bounds in spec.get('ranges', {}).items():
                    v = queue_module.field(data, key)
                    assert math.isfinite(v) and bounds[0] <= v <= bounds[1]
            actual.append(entry)
        assert actual == expected, f'Completion evidence changed: {j["id"]}'
        audit.append({'job': j['id'], 'status': 'verified', 'outputs': actual})
    dump(OUT / 'completion_receipt_validation.json', {
        'status': 'verified', 'captured_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(),
        'scope': 'All succeeded jobs: declared output hashes, sizes and JSON constraints versus durable receipts. Not a re-execution or a repeated complete input-data scan.',
        'verified_jobs': len(audit), 'unique_output_files': len(cache),
        'unique_output_bytes': sum(x['bytes'] for x in cache.values()), 'jobs': audit})
    print(f'Verified {len(audit)} successful job receipts', flush=True)


def collect_common_cross_scope(collector):
    source = ORIGINAL / 'queue_prompt4/specs/cross_cohort_comparisons.json'
    spec = load(source)
    original = copy.deepcopy(spec)
    runs = {r['run_id']: r for r in spec['runs']}
    unavailable = []
    for comparison in spec['comparisons']:
        keep = []
        for cell in comparison['cell_keys']:
            missing = []
            for pair in comparison['pairs']:
                for side in ('left', 'right'):
                    r = runs[pair[side]]
                    required = 'native' if cell == 'wf__' + WF else 'sturm_valid' if '__' + ST in cell else 'evaluation'
                    if not r.get(required):
                        missing.append({'run_id': r['run_id'], 'required_artifact': required})
            if missing:
                unavailable.append({'contrast': comparison['name'], 'budget': comparison['budget'],
                    'cell': cell, 'missing_artifact_declarations': missing,
                    'status': 'not_available_in_regenerated_scope',
                    'reason': 'Regenerated matched N100 programme did not include native WorldFloods evaluation. Legacy WF remains available. No values substituted and no new inference authorized by this repair.'})
            else:
                keep.append(cell)
        comparison['cell_keys'] = keep
    assert len(unavailable) == 2
    assert all(x['budget'] == 100 and x['cell'] == 'wf__' + WF for x in unavailable)
    spec['scope'] = original['scope'] + ' Supplemental collection of shared protocols only; two unavailable N100 native-WF contrasts are explicitly retained in unavailable_cross_cohort_cells.json.'
    cross_dir = OUT / 'cross_cohort_common_protocols'
    cross_dir.mkdir(exist_ok=True)
    derived = OUT / 'cross_cohort_common_protocol_spec.json'
    dump(derived, spec)
    dump(OUT / 'unavailable_cross_cohort_cells.json', unavailable)
    collector.collect_matched(derived, cross_dir)
    assert load(cross_dir / 'matched_provenance.json')['status'] == 'complete'
    source_rows = rows(ORIGINAL / 'analysis_cross_cohort/paired_by_seed.csv')
    new_rows = rows(cross_dir / 'paired_by_seed.csv')
    keyed = lambda r: tuple(r[k] for k in ('contrast', 'architecture', 'budget', 'seed', 'dataset', 'condition', 'metric'))
    original_rows = {keyed(r): r for r in source_rows}
    assert all(r == original_rows[keyed(r)] and r['status'] == 'complete' for r in new_rows)
    assert len(source_rows) - len(new_rows) == 48
    dump(OUT / 'cross_cohort_collection_resolution.json', {
        'status': 'complete_for_common_protocol_scope', 'original_queue_job': 'O4_collect_cross_cohort_verified',
        'original_queue_status_preserved': 'failed', 'original_spec': str(source), 'original_spec_sha256': sha(source),
        'derived_spec': str(derived), 'derived_spec_sha256': sha(derived),
        'original_collector_command_exit_code': 0,
        'original_failure': 'Completion validator rejected partial provenance because 48 values from two undeclared regenerated N100 native-WF comparisons were unavailable.',
        'preserved_common_paired_rows': len(new_rows), 'unavailable_paired_metric_rows': 48,
        'new_training_or_evaluation_jobs': 0, 'queue_or_original_artifact_mutations': 0,
        'result_dir': str(cross_dir), 'unavailable': unavailable})
    print(f'Collected {len(new_rows)} unchanged cross-cohort paired rows; disclosed 48 unavailable rows', flush=True)
    return cross_dir


def validate_summaries(per_seed, summary, paired=False):
    import statistics
    keys = ('contrast', 'architecture', 'budget', 'dataset', 'condition', 'metric') if paired else (
        'architecture', 'variant', 'budget', 'dataset', 'condition', 'metric')
    grouped = collections.defaultdict(list)
    for r in per_seed:
        assert r['status'] == 'complete', r
        grouped[tuple(r[k] for k in keys)].append(r)
    for r in summary:
        values = grouped[tuple(r[k] for k in keys)]
        assert len({x['seed'] for x in values}) == len(values), r
        v = [num(x['difference' if paired else 'value']) for x in values]
        assert all(x is not None for x in v), r
        assert len(v) == int(r['expected_seeds']) == int(r['n_available_seeds']), r
        assert r['complete_declared_seed_row'] == 'True', r
        assert abs(statistics.mean(v) - num(r['mean'])) < 1e-12, r
        if len(v) > 1:
            assert abs(statistics.stdev(v) - num(r['sample_SD_ddof1'])) < 1e-12, r
    return len(summary)


def build_report(manifest_path, state_path, manifest, state, cross_dir):
    at = datetime.now(ZoneInfo('Europe/Berlin')).isoformat()
    summary = rows(RESULTS / 'matched_seed_summary.csv')
    paired = rows(RESULTS / 'paired_seed_summary.csv')
    events = rows(RESULTS / 'paired_event_summary.csv')
    reproduction = rows(RESULTS / 'original_score_reproduction_summary.csv')
    cross = rows(cross_dir / 'paired_seed_summary.csv')
    assert load(RESULTS / 'matched_provenance.json')['status'] == 'complete'
    assert load(RESULTS / 'original_score_reproduction_provenance.json')['status'] == 'complete'
    checks = {
        'matched_summary_groups': validate_summaries(rows(RESULTS / 'matched_by_seed.csv'), summary),
        'paired_summary_groups': validate_summaries(rows(RESULTS / 'paired_by_seed.csv'), paired, True),
        'common_cross_summary_groups': validate_summaries(rows(cross_dir / 'paired_by_seed.csv'), cross, True)}
    dump(OUT / 'summary_consistency_validation.json', {'status': 'verified', **checks,
        'checks': 'Unique seeds, complete expected seed groups, recomputed mean and sample SD; tolerance 1e-12.'})
    table_dir = OUT / 'tables'
    table_dir.mkdir(exist_ok=True)
    source_index = []
    for source_dir, prefix in ((RESULTS, 'original'), (REGEN / 'analysis_prompt3_complete', 'regenerated')):
        for p in sorted(source_dir.glob('*')):
            if not p.is_file() or p.suffix not in ('.csv', '.json'):
                continue
            copied = p.name not in ('confusion_counts.csv', 'paired_event_differences.csv', 'queue_handoff.json')
            dest = table_dir / (prefix + '_' + p.name)
            if copied:
                shutil.copyfile(p, dest)
            source_index.append({'source': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size,
                                 'report_copy': str(dest.relative_to(OUT)) if copied else None})
    training = []
    train_specs = manifest['scientific_scope']['training_specifications']
    for spec in train_specs:
        run_id = spec['run_id']
        job = next(j for j in manifest['jobs'] if j['id'] == run_id)
        record = state['jobs'][run_id]
        receipt = load(ORIGINAL / 'training' / run_id / 'verified.json')
        best, last = receipt['checkpoint_checks']['best'], receipt['checkpoint_checks']['last']
        attempt = record['attempts'][-1]
        training.append({'run_id': run_id, 'architecture': spec['architecture'], 'variant': spec['variant'],
            'budget': spec['budget'], 'seed': spec['seed'], 'status': record['status'],
            'selected_epoch': best['epochs_completed'], 'selected_optimizer_steps': best['optimizer_steps'],
            'completed_epochs': last['epochs_completed'], 'completed_optimizer_steps': last['optimizer_steps'],
            'initial_model_sha256': best['initial_model_sha256'], 'best_checkpoint_sha256': best['sha256'],
            'last_checkpoint_sha256': last['sha256'], 'elapsed_seconds': receipt['elapsed_seconds'],
            'gpu_uuid': attempt.get('gpu_device', {}).get('uuid'),
            'started_at': attempt['started_at'], 'finished_at': attempt['finished_at'],
            'spec_path': job['argv'][job['argv'].index('--spec') + 1],
            'code_sha256': job['fingerprints']['code'], 'data_sha256': job['fingerprints']['data']})
    training.sort(key=lambda r: (r['architecture'], r['budget'], r['variant'], r['seed']))
    export(table_dir / 'training_all_44.csv', training)
    clean = [r for r in summary if (r['dataset'], r['condition']) in {(d, c) for d, c, _ in PROTOCOLS}]
    export(table_dir / 'clean_all_metrics.csv', clean)
    lookup = {(r['architecture'], r['variant'], r['budget'], r['dataset'], r['condition'], r['metric']): r for r in summary}
    robustness = []
    primary = ['matched_focal_g2_clip5', 'matched_lkd_clip5_valid', 'historical_ofa_clip5_allvalid']
    for r in summary:
        if not (r['architecture'] in ('teacher', 'index:mndwi_zero') or
                r['architecture'] == 'efficientvit_b0' and r['budget'] == '2500' and r['variant'] in primary):
            continue
        if r['dataset'] == 's11' or r['condition'] == WF:
            continue
        clean_key = ST + ':clean' if r['condition'].startswith(ST + ':') else 'clean'
        ref = lookup[(r['architecture'], r['variant'], r['budget'], r['dataset'], clean_key, r['metric'])]
        value, base = num(r['mean']), num(ref['mean'])
        robustness.append({**r, 'clean_same_protocol_mean': base,
            'absolute_drop': base - value, 'relative_drop': (base - value) / base if base else None,
            'absolute_drop_percentage_points': 100 * (base - value),
            'interpretation': 'Four-seed checkpoint mean or one fixed comparator; same protocol/mask for clean and perturbed score.'})
    export(table_dir / 'robustness_all_conditions_metrics_and_drops.csv', robustness)
    jobs = []
    for j in manifest['jobs']:
        r = state['jobs'][j['id']]
        jobs.append({'job': j['id'], 'status': r['status'], 'gpu': j.get('gpu', False),
                     'attempts': len(r.get('attempts', [])), 'reason': r.get('reason'),
                     'deps': json.dumps(j.get('deps', [])), 'argv': json.dumps(j.get('argv'))})
    export(table_dir / 'queue_all_642_jobs.csv', jobs)
    audit = load(OUT / 'completion_receipt_validation.json')
    assert audit['verified_jobs'] == sum(r['status'] == 'succeeded' for r in state['jobs'].values())
    raw_counts = dict(collections.Counter(r['status'] for r in state['jobs'].values()))
    compute_counts = dict(collections.Counter(state['jobs'][i]['status'] for i in manifest['scientific_scope']['computation_job_ids']))
    evidence = {
        'captured_at': at, 'completion_class': 'computation-complete with explicit external-input blockers',
        'qualified_scope': 'All original training, declared numerical evaluations and original-data collectors complete; cross-cohort common-protocol collection completed supplementally. Historical queue failure preserved; no full dry/device/exact-history completion claim.',
        'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'state': str(state_path), 'state_sha256': sha(state_path),
        'raw_queue_states': raw_counts, 'worker_status': state['worker_status'],
        'worker_stopped_at': state['stopped_at'], 'new_training_states': {'succeeded': 44},
        'new_computation_scope_states': compute_counts, 'remaining_runnable_GPU_hours': 0,
        'external_blockers': load(REPO / 'experiments/revision/prompt4_missing_inputs.json')['independent_blockers'],
        'collector_resolution': str(OUT / 'cross_cohort_collection_resolution.json'),
        'unavailable_cross_cohort_cells': str(OUT / 'unavailable_cross_cohort_cells.json'),
        'report': str(OUT / 'prompt4_report.pdf'), 'full_text': str(OUT / 'prompt4_report.txt'),
        'tables': str(table_dir), 'source_index': str(OUT / 'source_index.json')}
    dump(OUT / 'verified_completion_status.json', evidence)
    dump(OUT / 'source_index.json', source_index)
    dump(REPO / 'experiments/revision/prompt4_run_status.json', evidence)
    lines = []
    def para(value=''):
        lines.extend(textwrap.wrap(str(value), width=106, break_long_words=False, break_on_hyphens=False) if value else [''])
        lines.append('')
    def section(value):
        lines.extend(['', value, '=' * min(106, len(value)), ''])
    def table(headers, data, widths):
        def fmt(values):
            return '  '.join(str(v).ljust(w) for v, w in zip(values, widths))
        lines.append(fmt(headers))
        lines.append(fmt(['-' * w for w in widths]))
        for r in data:
            assert all(len(str(v)) <= w for v, w in zip(r, widths)), r
            lines.append(fmt(r))
        lines.append('')
    para('FloodDistill prompt4: completed numerical experiments, report and analysis')
    para('Captured ' + at + ' Europe/Berlin. Requested report, not a manuscript or response letter.')
    section('1. Completion, verification and exact scope')
    stopped = datetime.fromisoformat(state['stopped_at']).astimezone(ZoneInfo('Europe/Berlin'))
    para(f"The detached worker finished at {stopped.isoformat()}. All 44 new training runs succeeded; all 84 canonical jobs (28 original rescoring, 44 new clean readouts, 12 primary robustness readouts), 144 original corrections, three real-data GPU protocol checks and the initial S11 functional check succeeded. All 13 original-data paired collectors, original-checkpoint collector and final original-data collector succeeded. No numerical GPU experiment failed or remains pending/running.")
    para(f"Raw frozen queue: {raw_counts}; worker status {state['worker_status']}. The 30 blocked nodes concern independent dry references/dependent dry analyses and target-device access. This is computation-complete with explicit external-input blockers, not complete for the entire requested scope. The regenerated study has 60 completed training runs and 60 canonical readouts, retained separately.")
    para(f"Verified the hashes/sizes/JSON constraints of all {audit['verified_jobs']} successful job receipts ({audit['unique_output_files']} unique outputs). Recomputed {sum(checks.values())} complete summary groups from per-seed values, checking unique seeds, means and sample SD. No training, inference, threshold, seed or scientific protocol was changed for this report.")
    para('The sole failed queue node is O4_collect_cross_cohort_verified. Its process exited 0 but its completeness validator correctly rejected 48 missing metric values: two N100 corrected-native WorldFloods comparisons asked for regenerated runs without that declared evaluation. The frozen failure, original specification and partial tables remain unchanged. A supplemental CPU collector limits contrasts to protocols available on both sides, preserves all 1,152 common paired rows exactly, and lists the two unavailable comparisons explicitly. This repairs result collection; it does not supply missing corrected N100 regenerated scores or turn the original queue record into a success.')
    section('2. Original data and checkpoint identity')
    para('All 2,500 selected original source/cache pairs were verified using the real six-band normalized loader, with bitwise equality between loaded raw inputs and cached tensors. All 127 formerly missing original IDs and their associated original caches were restored individually; zero selected sources/caches remain missing. Original hard labels were preserved; no replacement scenes or new pseudo-label generation was used.')
    receipts = manifest['scientific_scope']['original_data_receipts']
    for budget, path in sorted(receipts.items()):
        data = load(path)
        para(f'N={budget} receipt: {path}; SHA256 {sha(path)}.')
    para('N2500 dataset digest: 1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60. N100 digest: 0472b976e0fbecfee4d91a57b3faf092822be39f3090505cb1bb9339e29ba3a4. The supplied N100 index equals the verified FPS prefix of 100 records. Historical per-run subset ancestry remains unrecorded. These are distinct from the approved regenerated cohort with 127 replacement IDs.')
    para('Current hashes identify supplied bytes; absent historical per-cache hashes and cache-generation teacher receipts prevent proving their complete historical byte ancestry. The complete original selected-ID/source/cache manifest and 127-ID reconciliation remain in prompt4_original_data_gate.json and the two original data receipts.')
    para('All 28 predeclared original main checkpoints loaded strictly with finite expected keys and shapes: EfficientViT N2500 CE/KD/OFA (12), EfficientViT N100 KD/OFA (8), and UNet-ResNet50 N2500 CE/KD (8), seeds 42/64/91/107. No original MobileNet/LightFormer comparison checkpoints were supplied. OFA heads/optimizer states are unnecessary for main-model rescoring; inference readiness does not imply training-resume completeness. Checkpoint/paper mapping and all SHA256 values are retained in prompt4_original_checkpoint_inventory.json and original_score_reproduction_by_seed.csv.')
    para('The overlap screen found zero selected-source footprints intersecting S11 validation/test/Bolivia; two intersect S11 training footprints. Three STURM chips intersect two training sources (two pairs within 30 days), and one WorldFloods map intersects two sources. Whole-source footprint overlap does not prove overlap of scored content, label leakage, or historical score-driven choices. The author-history question remains unanswered; restoration cannot establish historical non-use.')
    section('3. Frozen recipe and completed training matrix')
    table(['Group', 'Cells actually run', 'Unique'], [
        ['A', 'EffViT N2500 CE clip0 / KD clip0 / OFA clip5, four seeds each', '12'],
        ['B', 'EffViT N2500 matched KD four seeds; shared A OFA four seeds', '4 new'],
        ['C', 'EffViT N2500 full-weight focal gamma2, clip5, four seeds', '4'],
        ['D', 'MBV3Small + LightFormerR18 N2500 matched KD/OFA, four seeds', '16'],
        ['E', 'EffViT N100 matched KD/OFA, four seeds each', '8']], [5, 76, 7])
    para('Total 44 unique new cells, below the ceiling of 48. A/B reuse was allowed only after all original training masks and actual augmentation scans showed zero ignored class pixels, making the declared main/branch valid-domain distinction equivalent here. This does not prove cloud or sensor validity.')
    para('Seeds 42,64,91,107; six bands B2,B3,B4,B8A,B11,B12; fixed normalization mean [0.1413,0.1380,0.1235,0.3090,0.2045,0.1191], SD [0.0741,0.0737,0.0869,0.1180,0.0977,0.0766]. Batch16, float32/no AMP, 224 training and 512 main evaluation; AdamW lr5e-5, weight decay0.05, cosine 50 epochs, no warmup, drop-last. Original pseudo hard masks plus frozen live teacher soft outputs on correspondingly augmented inputs. Teacher checkpoint SHA256 3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60.')
    para('Hard-only historical CE is true cross entropy. Matched focal-only is full-weight gamma2 focal. KD uses 0.5*focal+0.5*T^2*KL at T4; the focal-to-KD contrast also halves the hard coefficient. OFA adds 0.3*mean(four exit losses), teacher T4/student T1, target modulation exponent1.2, no branch T^2 factor. Gamma0 therefore does not make this branch standard symmetric-temperature KD. Matched focal/KD/OFA clip main-model gradients at norm5; historical KD/CE retain clip0. Historical OFA-versus-KD is clipping-confounded and kept separately.')
    para('Main initialization hashes are checked within eligible architecture/seed matched pairs. Alternative backbones retain ImageNet initialization and the frozen six-band adaptation; LightFormer native auxiliary loss is disabled uniformly. OFA exits are training-only and removed from inference. MBV3Small has 3,546,706 main parameters plus 168,792 OFA parameters. LightFormerR18 has 11,427,509 main inference parameters plus 176,008 OFA parameters; 111,232 native auxiliary parameters are unused. EfficientViT is approximately 0.7M main parameters with 11,368 training-only OFA parameters; backbone size is not matched.')
    para('Checkpoint selection uses only S11 validation89, pixel-pooled two-class mIoU, every2epochs, strict improvement retaining the earlier tie, patience20checks. No external-score selection or seed replacement. All readouts use finalized validation-selected best.pt, never last/intermediate. N2500 has 156 steps/epoch and ceiling7,800; N100 has6 and ceiling300, so these budgets are not compute matched. training_all_44.csv gives selected/completed epochs/steps, both checkpoint hashes, initial hash, GPU assignment and timings per seed.')
    table(['Architecture/budget/variant', 'Seeds', 'Done epochs', 'Selected steps', 'Done steps'], [
        [f"{ARCH[a]} N{b} {VARIANT[v]}", '4',
         f"{min(r['completed_epochs'] for r in rr)}-{max(r['completed_epochs'] for r in rr)}",
         f"{min(r['selected_optimizer_steps'] for r in rr)}-{max(r['selected_optimizer_steps'] for r in rr)}",
         f"{min(r['completed_optimizer_steps'] for r in rr)}-{max(r['completed_optimizer_steps'] for r in rr)}"]
        for (a,b,v), rr in sorted({k: [r for r in training if (r['architecture'],r['budget'],r['variant']) == k]
            for k in {(r['architecture'],r['budget'],r['variant']) for r in training}}.items())], [43, 5, 11, 15, 11])
    runtime = load(next(Path(r['evaluation']) for r in load(manifest['scientific_scope']['comparison_spec'])['runs'] if r['run_id'].startswith('O4_train_')))['identity']['runtime_packages']
    para('Current runtime packages: ' + json.dumps(runtime, sort_keys=True) + '. Code snapshot SHA256 95f47d14ae77a2cfa7d3ad5264b4712b5a4e97bae742bcf2cc39d9030568504c; per-run environment/data hashes are frozen in the manifest. GroupA is recovered-v2 recipe reproduction in this environment; executed historical per-run configs/source/runtime/selection chains are unavailable, so this is not exact historical execution.')
    section('4. Original checkpoint score reproduction')
    para('These are evaluations of the copied original checkpoints, not new training replicates. Scores below are mIoU percentages; Delta is new legacy scoring minus recovered old scalar aggregate in percentage points. Printed reported SD and recovered/new sample SD remain separately named in the attached tables. Water IoU and per-seed discrepancies are also included there. N100 references are underlying Figure02 source values, not a printed numeric table.')
    table(['Model/objective/budget', 'Data', 'Printed', 'Old raw', 'Rescored', 'Delta pp'], [
        [f"{ARCH[r['architecture']]} {r['declared_objective']} N{r['budget']}", r['dataset'],
         pp(r['printed_mean']), pp(r['recovered_original_mean']), pp(r['new_legacy_mean']), pp(r['new_minus_recovered_original'],4)]
        for r in reproduction if r['metric']=='mIoU'], [28, 5, 9, 9, 9, 11])
    table2 = [r for r in reproduction if r['budget']=='2500' and r['metric']=='mIoU']
    agreed = sum(round(num(r['new_legacy_mean']),3) == num(r['printed_mean']) for r in table2)
    max_diff = max(abs(num(r['new_minus_recovered_original'])) for r in reproduction if r['metric']=='mIoU')
    para(f'{agreed}/{len(table2)} TableII means agree with the printed three-decimal values. The largest aggregate legacy mIoU drift from recovered old scalars across the declared rows is {100*max_diff:.4f} percentage points. The UNetR50 KD STURM printed mean is 0.747 whereas the newly rescored mean is 0.746483; the recovered raw aggregate also rounds to 0.746. Keep this printed/source discrepancy visible. Near agreement and rounding do not establish bitwise prediction equivalence or reproduce training.')
    para('The initial EffViT CE seed42 S11 functional check differed by +0.0000837973 mIoU and +0.0001565149 water IoU in ratio units. Data ordering, strict model state, six-band/radiometry/normalization, geometry, masks, precision and pooling were checked within recovered evidence. Candidate v2 evaluation source has batch8 versus current S11 batch4, but is not a verified executed config. The old prediction/mask/runtime hash chain is absent; the cause remains unresolved. No hardware attribution or score-forcing change is supported.')
    section('5. Clean results: all architectures, originals, controls and fixed baselines')
    para('Every value is a percentage. mIoU and water IoU cells show mean/sample-SD(ddof1) over four seeds; precision/recall/F1 below show means. A fixed teacher or MNDWI cell has one measurement and no seed SD (--). Full SD for all six metrics is in clean_all_metrics.csv. Scores use each row own matched confusion-count domain. Legacy and corrected protocols are never averaged together.')
    for dataset, condition, title in PROTOCOLS:
        para(title + ' [' + condition + ']')
        groups = collections.defaultdict(dict)
        for r in clean:
            if (r['dataset'],r['condition']) == (dataset,condition):
                groups[(r['architecture'],r['variant'],r['budget'])][r['metric']] = r
        table(['Model', 'mIoU/SD', 'Water/SD', 'Prec', 'Recall', 'F1'], [
            [label(m['mIoU']), mean_sd(m['mIoU']), mean_sd(m['water_IoU']),
             pp(m['water_precision']['mean'],2), pp(m['water_recall']['mean'],2), pp(m['water_F1']['mean'],2)]
            for _,m in sorted(groups.items())], [40, 12, 12, 7, 7, 7])
    section('6. Paired OFA, hard-loss, clipping and retraining comparisons')
    para('The paired pooled-score difference is left minus right for the same seed, then averaged across four seeds. SD is the sample SD of these four differences, not an event uncertainty interval. This is distinct from subtracting unrelated archived aggregates.')
    contrast_names = sorted({r['contrast'] for r in paired})
    contrast_id = {name: f'C{i+1:02d}' for i,name in enumerate(contrast_names)}
    for name in contrast_names:
        para(contrast_id[name] + ': ' + name)
    table(['Pair/arch/budget', 'Data/protocol', 'mIoU delta/SD pp', 'Water delta/SD pp'], [
        [f"{contrast_id[r['contrast']]} {ARCH[r['architecture']]} N{r['budget']}",
         r['dataset'] + (' native' if r['condition']==WF else ' finite' if r['condition'].startswith(ST) else ' legacy'),
         mean_sd(r,3), mean_sd(next(x for x in paired if all(x[k]==r[k] for k in ('contrast','architecture','budget','dataset','condition')) and x['metric']=='water_IoU'),3)]
        for r in paired if r['metric']=='mIoU' and (r['dataset'],r['condition']) in {(d,c) for d,c,_ in PROTOCOLS}], [32, 14, 21, 22])
    para('EffViT N2500 matched OFA minus KD is -0.335pp S11, -0.023pp STURM legacy and +0.534pp WF legacy (+0.438pp native WF). The historical clipping-confounded OFA minus KD contrast instead gives +0.267pp S11; it cannot establish an isolated OFA benefit. OFA performance depends on the comparator recipe and benchmark.')
    para('At N100, matched OFA minus KD gives +0.695pp S11, +0.116pp STURM legacy and +0.944pp WF legacy (+0.696pp native WF). These small-budget results retain the direction of the original checkpoint comparison, but do not establish broad OFA superiority or equal-compute budget efficiency.')
    para('MobileNet N2500 OFA minus KD is -0.291pp S11, -0.052pp STURM and +0.307pp WF legacy. LightFormer gives +0.152pp S11, -0.182pp STURM and -1.043pp WF legacy. Architecture ports are feasible and controlled; their mixed/null/negative gains are scientific outcomes. A larger backbone improving S11 does not prove improved external transfer or edge deployment.')
    para('Weighted matched KD minus full-weight focal improves pooled mIoU by +0.749pp S11, +0.959pp STURM and +1.641pp WF legacy. This supports the weighted training recipe within this scope, while its changed hard coefficient prevents isolating KL alone.')
    para('New historical-recipe training versus original main-checkpoint scoring is a separate environment/initialization/training contrast. Historical CE/KD/OFA S11 differences are -0.701/-0.267/-0.248pp. The CE sample SD is now approximately1.28pp versus approximately0.16pp across original checkpoint scores. Copied checkpoint score agreement therefore does not imply training reproduction agreement; all declared seeds remain in the aggregates.')
    section('7. Event uncertainty and geographic limits')
    para('Event rows average each of the four checkpoints event-level metric differences, then resample events. They do not pool predictions, treat pixels or seeds as independent geography, or estimate an unconditional training-procedure confidence interval. S11 uses10 events, STURM29, WorldFloods11 in the principal clean comparisons. Mean event differences and pooled-score differences are different estimands and can have different signs. Intervals containing zero do not prove equivalence; no multiple-comparison superiority claim is made.')
    main_event = [r for r in events if r['metric']=='mIoU' and r['contrast'] in (
        'OFA_minus_clipping_matched_KD','N100_OFA_minus_clipping_matched_KD','weighted_KD_minus_full_weight_focal')
        and (r['dataset'],r['condition']) in {(d,c) for d,c,_ in PROTOCOLS}]
    table(['Pair/arch/budget', 'Data/protocol', 'Events', 'Event mean pp', '95% CI pp'], [
        [f"{contrast_id[r['contrast']]} {ARCH[r['architecture']]} N{r['budget']}",
         r['dataset']+(' native' if r['condition']==WF else ' finite' if r['condition'].startswith(ST) else ' legacy'),
         r['n_events_with_four_seed_pairs'], pp(r['mean_event_metric_difference']),
         pp(r['ci95_low'])+' to '+pp(r['ci95_high'])] for r in main_event], [31,14,6,14,24])
    para('EffViT N2500 event-macro OFA-minus-KD on S11 is -0.178pp (95%CI -0.967,+0.459); on native WF it is +0.965pp (+0.722,+1.215), conditional on these checkpoints. N100 S11 is +0.649pp (+0.130,+1.157), and native WF +0.848pp (+0.588,+1.127), while STURM remains unresolved by event uncertainty. LightFormer WF pooled mIoU decreases while its event-macro mean is near zero: event weighting and heterogeneous scenes matter. Full event tables for every predeclared contrast, metric and condition are attached.')
    section('8. Geometry, input validity and corrected external protocols')
    para('WorldFloods legacy evaluation resized148 partial windows among785. The native zero-pad512/cropped-output diagnostic preserves native border pixels and adds no padding counts; unaffected interiors are reused under the identity contract. Both legacy and native results remain reported. For primary matched N2500 focal/KD/OFA, native minus legacy clean mIoU is approximately +0.191/-0.229/-0.325pp. Consequently the OFA-minus-KD gain becomes +0.438pp instead of +0.534pp; geometry affects the magnitude but does not establish a different universal ranking.')
    para('STURM retains2,675 chips, inverse normalization0.04+0.12v, clipping[0,1], substitution of nonfinite values, broad B8 in place of B8A, class1/2 water and3/4/5 ignored. Ten chips have nonfinite selected-band input support at30,672 scored512 pixels. The finite-six-band support correction excludes only that support from scoring while keeping predictions/preprocessing. It is distinct from changing the dataset or native128 evaluation. Clipping changes selected inputs at543,510 finite scored512 pixels across350 chips; no unclipped transform sweep was added.')
    para('The prior centre-crop/content diagnostic found200 chips losing more than half their native water, 73.6263% pooled water removal, and21 originally positive chips becoming dry. This confounds content/geometry; it does not isolate label quality or geography. The comparison here preserves main512 protocol and does not turn a geometry diagnostic into a tuned replacement protocol.')
    section('9. MNDWI and robustness: all declared conditions retained')
    para('MNDWI threshold stays0 and fixed teacher/index outputs are reused only after runtime/model/reader/data identity checks. Primary robustness is limited to the three original-data EffViT N2500 matched focal/KD/shared-OFA families (four seeds each), teacher and MNDWI. S11 is clean-only. STURM has clean plus its declared27 perturbations; WF has clean plus26 (no g0.70 cell). Corrected STURM uses all declared conditions; native WF correction is clean-only. No robustness sweep was added for alternative backbones or N100.')
    para('Common positive scalar gain cancels in the ideal MNDWI ratio; float denominator stabilization/near-threshold cases can leave tiny numerical exceptions. Synthetic haze is an additive reflectance perturbation, not evidence from independently observed atmospheric conditions. Clean external MNDWI remains competitive: legacy mIoU74.669% STURM and76.267% WF, versus primary matched KD76.050%/76.919% and OFA76.027%/77.453%. Native-WF MNDWI is75.943%, matched KD76.690%, OFA77.128%. These are trade-offs rather than a universal rule-model defeat.')
    para('At haze h0.20, STURM legacy mIoU is41.935% MNDWI,66.795% focal,69.011% KD,68.164% OFA,74.657% teacher. On WF legacy it is25.673%,62.831%,70.840%,74.759%,75.770% respectively. Learned models retain a substantial advantage over the fixed rule under this synthetic perturbation; the teacher often retains more accuracy than students, and OFA is not uniformly the most robust student on STURM.')
    para('The tables below give every declared severity and comparator, with mIoU/water-IoU mean percentages. The companion CSV provides all six metrics, sample SD, same-protocol clean reference, absolute degradation and relative degradation, including improvements (negative drops). No failed/null applicable robustness cell is suppressed. The plots show all declared curves, not selected severities.')
    robustness_models = [('efficientvit_b0',v,'2500') for v in primary] + [('teacher','fixed_comparator','0'),('index:mndwi_zero','fixed_comparator','0')]
    for dataset, prefix, title in [('sturm','', 'STURM legacy512'),('sturm',ST+':','STURM finite support'),('wf','','WF legacy512')]:
        para(title + ': each cell mIoU / water IoU (%)')
        conditions = sorted({r['condition'] for r in robustness if r['dataset']==dataset and
            (r['condition'].startswith(prefix) if prefix else not r['condition'].startswith(ST))})
        data = []
        for condition in conditions:
            values = []
            for arch, variant, budget in robustness_models:
                k = (arch,variant,budget,dataset,condition)
                values.append(pp(lookup[(*k,'mIoU')]['mean'],2)+'/'+pp(lookup[(*k,'water_IoU')]['mean'],2))
            data.append([condition.removeprefix(ST+':'),*values])
        table(['Condition','Focal','KD','OFA','Teacher','MNDWI'],data,[15,13,13,13,13,13])
    section('10. Original versus regenerated: common protocols only')
    para('Contrasts are original minus regenerated four-seed results within the declared recipe/architecture/budget/scoring stratum. These change cohort and cached supervision jointly; they cannot isolate exactly127 source substitutions. Current matched initialization equality is required where receipt-backed and passes24 checks; the old CE/OFA initial tensor receipts are absent, so historical direct initial-tensor equality is unverified. No seeds are averaged across cohorts.')
    table(['Arch/budget/variant','Data/protocol','mIoU delta/SD pp','Water delta/SD pp'], [
        [f"{ARCH[r['architecture']]} N{r['budget']} {VARIANT[r['contrast'].split('_original_minus')[0]]}",
         r['dataset']+(' native' if r['condition']==WF else ' finite' if r['condition'].startswith(ST) else ' legacy'),
         mean_sd(r,3), mean_sd(next(x for x in cross if all(x[k]==r[k] for k in ('contrast','architecture','budget','dataset','condition')) and x['metric']=='water_IoU'),3)]
        for r in cross if r['metric']=='mIoU'], [41,14,21,22])
    para('The original-data restoration does not move every method in the same direction. EffViT matched N2500 KD changes +0.348pp S11, -0.008pp STURM and -0.078pp WF legacy; shared OFA changes -0.418pp, -0.062pp and +0.555pp respectively. N100 OFA changes +0.813pp S11 and -0.323pp WF legacy, while N100 KD changes +0.107pp and -1.381pp. Mixed signs and method/cohort interactions prevent attributing outcomes solely to127 images or claiming exact training replication.')
    para('Unavailable: N100 KD and OFA original-versus-regenerated native-WF comparisons, four seeds and six metrics each (48 paired metric rows). Only original N100 native results exist; no regenerated N100 native evaluation was declared. The original partial tables retain their null entries. The supplemental common-protocol tables are complete only for their explicit scope; no legacy value substitutes for a native score.')
    section('11. Reviewer evidence, supported conclusions and remaining action')
    evidence_map = [
        ('Reproduction', '28 strict checkpoints; printed/raw/new legacy and corrected rows;44 new training receipts.', 'Checkpoint scores nearly reproduce recovered scalars. Current-environment training is variable and not exact historical execution.', 'Use separate checkpoint-rescoring and retraining tables; retain printed/source discrepancy and missing historical ancestry.'),
        ('Architecture/OFA/modest gains', 'Matched main initialization, losses/clipping/masks/budget; MBV3Small and LightFormer controls, all four seeds.', 'Feasible OFA ports; benchmark-dependent gains, small-budget benefit, negative/null N2500 outcomes.', 'Describe observed scope; remove general OFA superiority and architecture-size equivalence claims.'),
        ('Loss and method description', 'Executed CE/focal/KD/OFA algebra and training-only head audit; full-weight focal control.', 'Historical CE is true CE; weighted KD contrast changes hard coefficient; branch gamma0 is not standard symmetric-temperature KD.', 'Correct loss equation/temperature/head descriptions; retain component attribution and do not invent new novelty or extra experiments.'),
        ('Data roles/leakage/curation', 'All2500 mapping/cache checks;127 restoration;N100 subset;overlap screen; existing descriptor audit.', 'Human hard labels versus teacher soft labels and pseudo hard labels have distinct roles; complete-source ancestry and historical non-use remain limited.', 'Describe maximum-class confidence/spatial dispersion rather than water-probability uncertainty; retain FPS/filter qualifications and footprint overlaps.'),
        ('Geometry/validity', '148 partial WF windows; native corrections; STURM finite-support counts; crop-content audit.', 'Geometry changes scores; finite-support changes are small but require named masks and counts.', 'Report legacy/corrected separately; preserve preprocessing and73.6263% crop-water removal qualification.'),
        ('MNDWI/robustness', 'Fixed threshold0, primary student outputs, all predeclared severities and paired/count-domain metrics.', 'Rule remains competitive clean and gain invariant; synthetic additive haze supports a conditional learned-model advantage.', 'Keep rule trade-offs and negative results; distinguish synthetic from independently observed atmosphere.'),
        ('Uncertainty/geography', 'Four-seed SD plus10/29/11-event checkpoint-conditional analyses.', 'Pooled scores and macro-event effects differ; no seed/pixel geographic independence or equivalence conclusion.', 'Report both estimands and CI scope; avoid unconditional/general superiority claims.'),
        ('Independent dry imagery', 'Prepared11-candidate independent-review package; gate still lacks approved annotations.', 'No independent dry-scene reliability estimate is complete. Six related held-out dry chips remain diagnostic only.', 'Provide genuine dispositions/assessability and required invalid-region masks; retain false-water fraction/alarm criteria and undefined IoU.'),
        ('Bounded deployment', 'Existing original float/QAT/mixed-engine audit; supplied1937 copied files have only .pt/.json; no new target artifacts.', 'Xavier-specific claims cannot be validated with RTX timing; unknown-seed engine does not establish four-seed ancestry.', 'Provide authorized accessible Xavier workflow, matching seed artifacts, named split/device timing logs and power-rail samples; no new QAT.'),
        ('Historical benchmark influence', 'Existing manuscript wording and one-time unanswered author-history question.', 'New tests and restoration cannot prove external scores never guided earlier choices.', 'Resolve the factual history once; no repeated question or unsupported clean-history declaration.')]
    export(OUT / 'reviewer_evidence_conclusions.csv', [dict(zip(('concern','evidence','supported_conclusion','remaining_action'),r)) for r in evidence_map])
    for concern, ev, conclusion, action in evidence_map:
        para(concern + '. Evidence: ' + ev + ' Conclusion: ' + conclusion + ' Remaining action: ' + action)
    section('12. External inputs and honest stop condition')
    para('No additional selected original imagery or pseudo caches are required. Do not resend the whole dataset. Numerical jobs are drained and the worker is finished; no new finite-matrix experiment is queued by this report. Remaining runnable GPU-hours are0; missing-input wait time has no numeric ETA. Training elapsed totals are ' + f"{sum(r['elapsed_seconds'] for r in training)/3600:.2f} GPU-hours across the44 new runs" + '; two-GPU wall time also depended on foreign workload/host headroom, evaluation and serialization.')
    para('Dry: genuine independent dispositions for11 candidates at experiments/revision/prompt3_dry_verification/annotations.template.json following review_instructions.json, assessability and prescribed masks. These unlock the reference gate,14 prepared regenerated/fixed jobs,12 original primary dry jobs and dry collection. No labels were invented from teacher-empty or MNDWI-empty predictions.')
    para('Historical exactness: executed per-run configs/commands, source/runtime exports, teacher-cache-generation receipt, selection histories and prediction/mask identity chain for the declared four seeds. These qualify exact execution and provenance claims; fully specified current-environment results are already complete.')
    para('Device: accessible already-authorized Xavier NX workflow; matching original float/QAT/ONNX/engine ancestry for all four seeds including107, engine-hashed device accuracy logs, split/aggregation identities, timing invocation/warmup/device settings and named raw power-rail traces. Mixed INT8/FP16 is not all-INT8. The prepared S11 device split is validation. QAT-stripped float latency export is not a genuine float-accuracy baseline. BN/scales are fixed while remaining model weights adapt during QAT; peak watts do not establish energy per image. No desktop substitute, portable-engine assumption or new QAT run was made.')
    para('History: the already-recorded question about whether external scores influenced prior method choices still needs a factual author answer; it is not reasked here. These external-input limits prevent saying every demand is complete.')
    section('13. Artifact index, commands and provenance')
    para('Report: prompt4_report.pdf and identical numeric narrative in prompt4_report.txt. Tables: tables/ includes full original and regenerated mean/SD, per-seed/paired rows, precision/recall/F1, every robustness condition/drop, original discrepancies and all44 training receipts. cross_cohort_common_protocols/ has the supplemental verified common-scope comparisons and event tables. unavailable_cross_cohort_cells.json retains absent contrasts.')
    para('source_index.json identifies each upstream final table by absolute path, size and SHA256. Large confusion-count/event-difference CSVs remain in their original result directories and are explicitly referenced rather than copied; the report package does not alter them. completion_receipt_validation.json and summary_consistency_validation.json are the report integrity checks. verified_completion_status.json is the latest report capture; frozen upstream queue_handoff.json retains its earlier needs-experiment status caused by the failed cross-scope node.')
    for title, path in [('Original checkpoint final collection',ORIGINAL/'analysis_original_checkpoints'),
                        ('Original-data numerical final collection',RESULTS),
                        ('Regenerated final collection',REGEN/'analysis_prompt3_complete'),
                        ('Historical partial cross collection',ORIGINAL/'analysis_cross_cohort'),
                        ('Active frozen queue manifest',manifest_path)]:
        para(title + ': ' + str(path))
    for action in ('status','pause','resume','stop'):
        para(action + ': /home/schmalstieg/miniconda3/envs/esi/bin/python ' + str(manifest_path.parent/'revision_queue.py') + ' ' + action + ' --manifest ' + str(manifest_path))
    para('These are existing queue controls, not an instruction to relaunch completed jobs. The worker checks resources approximately every5seconds when active; it has now drained its finite runnable scope. Frozen locks/UUID assignments, no foreign-workload interference and no editor/session dependency were retained. No logout/reboot/host-sleep execution guarantee is made.')
    report = '\n'.join(lines) + '\n'
    (OUT/'prompt4_report.txt').write_text(report)
    render_pdf(lines, summary, paired, lookup, robustness_models)
    # Preserve the previous user's operational capture before replacing the open handoff.
    handoff = REPO/'experiments/revision/prompt4_handoff.txt'
    archive = REPO/'experiments/revision/prompt4_captures'
    archive.mkdir(exist_ok=True)
    preserved = archive/'prompt4_handoff_before_final_report_20261007.txt'
    if not preserved.exists():
        shutil.copyfile(handoff,preserved)
    handoff.write_text('\n'.join([
        'Captured '+at+' Europe/Berlin.',
        'Completion class: computation-complete with explicit external-input blockers.',
        'All44 new training runs,84 declared canonical readouts,144 original corrections, protocol/functional checks,13 paired collectors and original-data final collector succeeded.',
        'Worker finished '+stopped.isoformat()+'. No pending/running numerical job; remaining runnable GPU-hours0.',
        'Raw historical queue states: '+str(raw_counts)+'. All611 successful output receipts revalidated.',
        'One historical cross-cohort collector failed its output completeness guard: regenerated N100 native-WF readouts were never declared. No numerical GPU job failed.',
        'Supplemental common-protocol collector now complete:1152 unchanged paired metric rows;48 unavailable native-WF N100 rows explicitly disclosed. Original queue failure and original artifacts remain preserved.',
        'Independent dry references/device workflow/historical exactness/benchmark-use history remain unresolved. The full requested scope is not claimed complete.',
        'Full report: '+str(OUT/'prompt4_report.pdf'),
        'Full text: '+str(OUT/'prompt4_report.txt'),
        'All tables: '+str(table_dir),
        'Latest verified report completion capture: '+str(OUT/'verified_completion_status.json'),
        'Missing cross-cohort cells: '+str(OUT/'unavailable_cross_cohort_cells.json'),
        'Original frozen numerical capture (preserved earlier failure-qualified status): '+str(RESULTS/'queue_handoff.json'),
        'Previous detailed handoff preserved: '+str(preserved),
        'No additional selected original sources/caches are needed; all127 prior missing IDs were restored. No new training/evaluation jobs or queue mutations were made.',
    ])+'\n')
    print(json.dumps({'status':evidence['completion_class'],'report':str(OUT/'prompt4_report.pdf'),
                      'raw_queue':raw_counts,'summary_checks':checks}),flush=True)


def render_pdf(lines, summary, paired, lookup, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    wrapped=[]
    for line in lines:
        wrapped.extend(textwrap.wrap(line,width=110,break_long_words=True,break_on_hyphens=False) or [''])
    count=0
    with PdfPages(OUT/'prompt4_report.pdf', metadata={'Title':'FloodDistill prompt4 completed numerical experiments',
            'Subject':'Full four-seed report with explicit external-input blockers and preserved collector failure'}) as pdf:
        for start in range(0,len(wrapped),68):
            count+=1
            fig=plt.figure(figsize=(8.27,11.69))
            fig.text(.065,.967,'FloodDistill | prompt4 results and analysis',fontsize=10,color='#28516a')
            fig.text(.065,.942,'\n'.join(wrapped[start:start+68]),fontsize=7.6,family='DejaVu Sans Mono',
                     va='top',linespacing=1.3)
            fig.text(.935,.025,str(count),ha='right',fontsize=8)
            pdf.savefig(fig); plt.close(fig)
        # Four-seed paired differences on clean and corrected protocols.
        fig,axes=plt.subplots(1,3,figsize=(11.69,7),sharey=True)
        comparisons=[('efficientvit_b0','100','N100_OFA_minus_clipping_matched_KD'),
                     ('efficientvit_b0','2500','OFA_minus_clipping_matched_KD'),
                     ('mobilenet_v3_small','2500','OFA_minus_clipping_matched_KD'),
                     ('lightformer_resnet18','2500','OFA_minus_clipping_matched_KD')]
        import numpy as np
        for ax,(dataset,condition,title) in zip(axes,[PROTOCOLS[0],PROTOCOLS[3],PROTOCOLS[4]]):
            rrs=[next(r for r in paired if (r['architecture'],r['budget'],r['contrast'],r['dataset'],r['condition'],r['metric'])==
                (a,b,c,dataset,condition,'mIoU')) for a,b,c in comparisons]
            ax.errorbar([100*num(r['mean']) for r in rrs],np.arange(4),
                        xerr=[100*num(r['sample_SD_ddof1']) for r in rrs],fmt='o',capsize=4)
            ax.axvline(0,color='#777',lw=1); ax.grid(axis='x',alpha=.25)
            ax.set_title(title,fontsize=10); ax.set_xlabel('OFA minus matched KD\n(percentage points)',fontsize=9)
        axes[0].set_yticks(np.arange(4),[f'{ARCH[a]} N{b}' for a,b,_ in comparisons])
        axes[0].invert_yaxis()
        fig.suptitle('Paired pooled mIoU: means and sample SD across four seeds',fontsize=13)
        fig.text(.12,.04,'Bars show seed SD, not confidence intervals or independent geographic replication.',fontsize=10)
        fig.tight_layout(rect=(.02,.09,.98,.92)); pdf.savefig(fig)
        fig.savefig(OUT/'paired_ofa_clean.png',dpi=180); plt.close(fig); count+=1
        colors=['#7570b3','#1b9e77','#d95f02','#333333','#e7298a']
        names=['Full focal','Matched KD','OFA','Teacher','MNDWI']
        for dataset,prefix,title in [('sturm',ST+':','STURM finite support'),('wf','','WorldFloods legacy512')]:
            fig,axs=plt.subplots(2,2,figsize=(11.69,8.27))
            for ax,(family,heading) in zip(axs.flat,[('gain','Common scalar gain'),('haze','Synthetic additive haze'),
                                                 ('vis','Visible-band gain'),('tilt','Spectral tilt')]):
                for i,(arch,variant,budget) in enumerate(models):
                    points=[]
                    for r in summary:
                        if (r['architecture'],r['variant'],r['budget'],r['dataset'],r['metric'])!=(arch,variant,budget,dataset,'mIoU'):
                            continue
                        condition=r['condition']
                        if prefix and not condition.startswith(prefix): continue
                        if not prefix and condition.startswith(ST): continue
                        c=condition.removeprefix(prefix)
                        match=None
                        if family=='gain': match=re.fullmatch(r'g([0-9.]+)',c)
                        if family=='haze': match=re.fullmatch(r'h([0-9.]+)',c)
                        if family in ('vis','tilt'): match=re.fullmatch(r't1'+family+r'([+-])(\d+)',c)
                        if match:
                            x=float(match[1]) if family in ('gain','haze') else int(match[2])/100*(1 if match[1]=='+' else -1)
                            points.append((x,100*num(r['mean'])))
                        if c=='clean' and family=='haze': points.append((0,100*num(r['mean'])))
                        if c=='t1unit' and family in ('vis','tilt'): points.append((0,100*num(r['mean'])))
                    points.sort()
                    ax.plot([p[0] for p in points],[p[1] for p in points],'-o',ms=3,label=names[i],color=colors[i])
                ax.set_title(heading); ax.set_ylabel('mIoU (%)'); ax.grid(alpha=.25)
                ax.set_xlabel('Scalar multiplier' if family=='gain' else 'Additive intensity' if family=='haze'
                              else 'Signed fractional gain')
            axs[0,0].legend(fontsize=8)
            fig.suptitle(title+': all declared perturbation curves',fontsize=13)
            fig.text(.10,.028,'Four-seed student means; fixed comparators have one evaluation. Native WF perturbations were not declared.',fontsize=9)
            fig.tight_layout(rect=(.02,.06,.98,.94)); pdf.savefig(fig)
            fig.savefig(OUT/(dataset+'_robustness.png'),dpi=180); plt.close(fig); count+=1
    dump(OUT/'report_render_validation.json',{'pages':count,'text_lines':len(wrapped),
        'report_sha256':sha(OUT/'prompt4_report.pdf'),'text_sha256':sha(OUT/'prompt4_report.txt')})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = load(BASE / 'queue/active_queue.json')
    manifest_path = Path(pointer['active_manifest'])
    state_path = Path(pointer['active_state'])
    manifest, state = load(manifest_path), load(state_path)
    assert manifest['scientific_scope']['new_training_runs'] == 44
    assert all(state['jobs'][i]['status'] == 'succeeded' for i in manifest['scientific_scope']['training_job_ids'])
    sys.path.insert(0, manifest['source_repair_snapshot'])
    from experiments import revision_queue as queue_module
    from evaluation import revision_collect as collector
    assert Path(collector.__file__).resolve().is_relative_to(Path(manifest['source_repair_snapshot']))
    audit_outputs(manifest, state, queue_module)
    cross_dir = collect_common_cross_scope(collector)
    build_report(manifest_path, state_path, manifest, state, cross_dir)


if __name__ == '__main__':
    main()
