#!/usr/bin/env python3
"""Regression checks for received device failures and immutable predecessor chaining."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from input_profiles import identify, validate_input, numerical_sturm, bilinear64, NUMERICAL
from repair_policy import source_gate
from evaluate_native import summarize
import chain_evaluations as chain


def read(path):
    return json.loads(Path(path).read_text())


def rejected(fn, *args):
    try:
        fn(*args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError):
        return
    raise AssertionError('Invalid admission fixture accepted')


def check_profiles():
    import cv2
    from data.sen1floods11_nodep import NORM_MEAN, NORM_STD
    inventory = [json.loads(line) for line in (ROOT / 'inventories/sturm512.jsonl').read_text().splitlines()]
    assert len(inventory) == 2675
    raw = np.random.default_rng(1729).uniform(0, 1, (6, 128, 128)).astype(np.float32)
    mean = np.asarray(NORM_MEAN, np.float32)[:, None, None]
    std = np.asarray(NORM_STD, np.float32)[:, None, None]
    reference = (bilinear64(raw) - mean) / std
    actual = cv2.resize(raw.transpose(1,2,0), (512,512)).transpose(2,0,1)
    actual = (actual - mean) / std
    rec = inventory[0]
    row = numerical_sturm(rec, actual, reference)
    assert row['input_profile'] == NUMERICAL
    validate_input(ROOT, 'sturm512', row, rec)
    # A new OpenCV hash is admitted by numerical agreement, not a hash allowlist.
    changed = numerical_sturm(rec, actual + np.float32(1e-5), reference)
    assert changed['input_sha256'] != row['input_sha256']
    validate_input(ROOT, 'sturm512', changed, rec)
    rejected(numerical_sturm, rec, actual + .001, reference)
    rejected(numerical_sturm, rec, actual[::-1], reference)
    rejected(validate_input, ROOT, 'sturm512', dict(row, input_max_abs_error=.001), rec)
    rejected(validate_input, ROOT, 'sturm512', dict(row, mask_sha256='0' * 64), rec)
    rejected(identify, ROOT, 'wf', rec, '0' * 64)
    for record in inventory:
        # Earlier completed measurements keep their old preprocessing receipts;
        # image hash equality is no longer an admission criterion.
        validate_input(ROOT, 'sturm512', dict(record, input_sha256='f' * 64), record)


def check_received_failures():
    from integer_tools import inspect_layers
    from validate_results import validate_rows
    layers = read(ROOT / 'provenance/DEVICE_MICRO107_LAYERS_20261022.json')
    audit = inspect_layers(layers)
    assert not audit['full_integer_tensor_path'] and audit['floating_compute_layers']
    native = read(ROOT / 'provenance/DEVICE_MICRO107_S11VAL_FAILURE_20261022.json')
    validate_rows(ROOT, 's11val', native['rows'])
    build = read(ROOT / 'provenance/DEVICE_MICRO107_BUILD_20261022.json')
    identity = read(ROOT / 'provenance/DEVICE_MICRO107_EVALUATION_IDENTITY_20261022.json')
    retired = next(m for m in read(ROOT / 'RUN_MANIFEST.json')['retired_micro_pure_20261023'] if m['seed'] == 107)
    assert build['model_sha256'] == retired['onnx_sha256']
    assert identity['engine_sha256'] == build['engine_sha256']
    parent = read(ROOT / 'models/R4_micro_s107_fp32/SOURCE_REFERENCE.json')
    baseline = dict(pooled=dict(miou=parent['mIoU'], water_iou=parent['water_IoU']), equal_event=dict(miou=parent['equal_event_mIoU']))
    assert not source_gate(baseline, native)['passed']
    rows = [json.loads(line) for line in (ROOT / 'provenance/DEVICE_B0_BUILD_DIAGNOSTICS_20261022.jsonl').read_text().splitlines() if line.startswith('{')]
    variants = {r['variant']: r for r in rows}
    assert not variants['islands_obey']['built'] and not variants['obey']['built']
    assert variants['islands_prefer']['built'] and variants['islands_prefer']['island_audit']


def check_chain(manifest):
    models = manifest['engines']
    tiny = next(m for m in models if m['id'] == 'R4_s42_fp32')
    failed = next(m for m in models if m['id'] == 'R4_s64_fp32')
    incomplete = next(m for m in models if m['id'] == 'R4_s91_fp32')
    changed = next(m for m in models if m['id'] == 'B0_s42_qat_mixed_repaired')
    oldchanged = dict(changed)
    del oldchanged['island_constraint_mode']
    assert not chain.same_recipe(oldchanged, changed)
    assert chain.same_recipe(tiny, dict(tiny))
    assert not chain.same_recipe(tiny, dict(tiny, onnx_sha256='0' * 64))
    with tempfile.TemporaryDirectory(prefix='followup_chain_fixture_NOT_NATIVE_') as directory:
        temp = Path(directory)
        new = temp / 'new'
        new.mkdir()
        oldroot = temp / 'old'
        oldroot.mkdir()
        old = oldroot / 'results/old_fixture'
        old.mkdir(parents=True)
        for name in ('inventories', 'models', 'prior_device_bundle'):
            (new / name).symlink_to(ROOT / name)
        chain.save(new / 'RUN_MANIFEST.json', dict(engines=[tiny, failed, incomplete, changed]))
        chain.save(oldroot / 'RUN_MANIFEST.json', dict(engines=[tiny, failed, incomplete, oldchanged]))
        oldstates = [dict(id=tiny['id'], status='MEASURED_PENDING_ACCEPTANCE'),
                     dict(id=failed['id'], status='FAILED_BENCHMARK'),
                     dict(id=incomplete['id'], status='MEASURED_PENDING_ACCEPTANCE'),
                     dict(id=changed['id'], status='FAILED_BUILD')]
        chain.save(old / 'STATUS.json', oldstates)
        chain.save(old / tiny['id'] / 'REUSE.json', tiny['reuse'])
        records = read(ROOT / 'inventories/dry159_manifest.json')['rows']
        rows = []
        for rec in records:
            row = {k: rec[k] for k in ('tile_id', 'group', 'decision', 'stratum')}
            for crop, key in [('central264', 'central_valid_pixels'), ('full512', 'full_valid_pixels')]:
                count = rec[key]
                row[crop] = dict(valid_pixels=count, false_water_pixels=0, false_water_fraction=0.0 if count else None, alarm_gt_1pct=False if count else None)
            rows.append(row)
        chain.save(old / tiny['id'] / 'evaluation/dry159.json', dict(rows=rows, fixture='OFFLINE_NOT_NATIVE'))
        chain.save(old / tiny['id'] / 'evaluation/COMPLETE.json', dict(engine_sha256=tiny['reuse']['engine_sha256'], fixture='OFFLINE_NOT_NATIVE'))
        chain.valid_complete(new, tiny, old / tiny['id'], oldstates[0])
        bad = copy.deepcopy(rows)
        bad[0]['central264']['valid_pixels'] += 1
        chain.save(old / tiny['id'] / 'evaluation/dry159.json', dict(rows=bad))
        rejected(chain.valid_complete, new, tiny, old / tiny['id'], oldstates[0])
        chain.save(old / tiny['id'] / 'evaluation/dry159.json', dict(rows=rows))
        chain.save(old / failed['id'] / 'failure_fixture.json', dict(fixture='Preserve this failure'))
        # Exercise a full legacy benchmark receipt which predates power engine hashes.
        mixed = next(m for m in manifest['engines'] if m['id'] == 'R4_s42_qat_mixed')
        dest = old / mixed['id']
        dest.mkdir()
        (dest / 'model.engine').write_bytes(b'OFFLINE_ENGINE_IDENTITY_FIXTURE_NOT_NATIVE')
        digest = chain.sha(dest / 'model.engine')
        chain.save(dest / 'build.json', dict(status='built', mode=mixed['mode'], model_sha256=mixed['onnx_sha256'], engine_sha256=digest))
        chain.save(dest / 'evaluation/COMPLETE.json', dict(engine_sha256=digest, io_sha256=None))
        chain.save(dest / 'evaluation/dry159.json', dict(rows=rows))
        for ds in chain.DATASETS:
            inv = [json.loads(line) for line in (ROOT / 'inventories' / (ds + '.jsonl')).read_text().splitlines()]
            fixture = [dict(name=r['name'], event=r['event'], input_sha256=r['input_sha256'], mask_sha256=r['mask_sha256'], counts_tn_fp_fn_tp=[r['valid_pixels']-r['water_pixels'], 0, r['water_pixels'], 0]) for r in inv]
            chain.save(dest / 'evaluation' / (ds + '.json'), dict(rows=fixture, fixture='OFFLINE_NOT_NATIVE'))
        chain.save(dest / 'evaluation/timing.json', dict(engine_sha256=digest))
        chain.save(dest / 'benchmark/bench.json', dict(engine_sha256=digest, runA_returncode=0, runB_returncode=0, throughput_qps_runA=100))
        power = dict(engine=str(dest / 'model.engine'), returncode=0, steady_samples=10, throughput_qps=100, steady_cpu_gpu_cv_mean_w=8, steady_energy_mJ_per_image=80)
        chain.save(dest / 'steady_power/power_steady.json', power)
        success = dict(profile_returncode=0, steady_returncode=0)
        chain.valid_complete(new, mixed, dest, success)
        chain.save(dest / 'steady_power/power_steady.json', dict(power, engine='/wrong/engine'))
        rejected(chain.valid_complete, new, mixed, dest, success)
        chain.save(dest / 'steady_power/power_steady.json', power)
        before = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()}
        plan = chain.import_previous(new, 'new_fixture', [old])
        actions = {p['id']: p['action'] for p in plan}
        assert actions[tiny['id']] == 'REUSE_VERIFIED_COMPLETE'
        assert actions[failed['id']] == 'RETAIN_SAME_RECIPE_FAILURE'
        assert actions[incomplete['id']] == actions[changed['id']] == 'RUN_CURRENT_RECIPE'
        imported = {s['id'] for s in read(new / 'results/new_fixture/STATUS.json')}
        assert imported == {tiny['id'], failed['id']}
        assert not (new / 'results/new_fixture' / changed['id']).exists()
        assert before == {str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()}
        rejected(chain.import_previous, new, 'new_fixture', [old])
        empty = oldroot / 'results/preflight_failed_without_model_states'
        empty.mkdir()
        fresh = chain.import_previous(new, 'after_failed_preflight', [empty])
        assert all(item['action'] == 'RUN_CURRENT_RECIPE' for item in fresh)
        assert read(new / 'results/after_failed_preflight/STATUS.json') == []
        # An active predecessor lock must delay the chain; held locks also block a resume.
        package_lock = oldroot / '.controller.lock'
        attempt_lock = old / 'controller.lock'
        package_lock.touch()
        attempt_lock.touch()
        ready, release = temp / 'READY', temp / 'RELEASE'
        code = "import fcntl,pathlib,sys,time; f=open(sys.argv[1],'r'); fcntl.flock(f,fcntl.LOCK_EX); pathlib.Path(sys.argv[2]).touch();\nwhile not pathlib.Path(sys.argv[3]).exists():time.sleep(.01)\n"
        argv = [sys.executable, '-c', code, str(package_lock), str(ready), str(release)]
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert ready.exists()
            chain.save(old / 'HEARTBEAT.json', dict(child_pid=proc.pid, argv=argv))
            assert chain.child_alive(old)
            from unittest.mock import patch
            waited = [];original_sleep=time.sleep

            def release_after_wait(seconds):
                waited.append(seconds)
                release.touch()
                deadline=time.monotonic()+5
                while proc.poll() is None and time.monotonic()<deadline:original_sleep(.01)
                assert proc.poll() is not None

            with patch.object(chain.time, 'sleep', release_after_wait):
                handles = chain.acquire_predecessors([old], temp / 'chain.json', temp / 'STOP')
            assert waited
            assert not chain.child_alive(old)
            contender = package_lock.open('r')
            try:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('Old resume could overlap the chain')
            contender.close()
            for handle in handles:
                handle.close()
        finally:
            release.touch()
            proc.wait(timeout=5)


def evaluate_sources(manifest, input_root):
    import onnxruntime as ort
    import dataset_io
    from device_runtime import encode
    original = dataset_io.ROOT
    results = []
    with tempfile.TemporaryDirectory(prefix='followup_source_fixture_') as directory:
        temp = Path(directory)
        (temp / 'inventories').symlink_to(ROOT / 'inventories')
        (temp / 'assets').symlink_to(input_root / 'assets')
        dataset_io.ROOT = temp
        try:
            for model in manifest['engines']:
                if not model.get('integer_repair_certificate'):
                    continue
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(str(ROOT / model['onnx']), options, providers=['CPUExecutionProvider'])
                io = read(ROOT / model['io'])
                report = read(ROOT / model['source_gate'])
                expected = {r['name']: r for r in report['evaluation']['rows']}
                for image, mask, rec in dataset_io.source('s11val'):
                    logits = session.run(None, {'image': encode(image[None], io['host_input_scale'], 0)})[0]
                    pred = logits[0].argmax(0)
                    valid = (mask >= 0) & (mask < 2)
                    counts = np.bincount((2 * mask[valid] + pred[valid]).astype(np.int64), minlength=4)
                    assert counts.tolist() == expected[rec['name']]['counts_tn_fp_fn_tp']
                results.append(dict(seed=model['seed'], images=89, actual_optimized_confusions_match=True))
        finally:
            dataset_io.ROOT = original
    assert len(results) == 4
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, help='Optional full353 source cache for fresh actual all89 ORT checks')
    args = parser.parse_args()
    manifest = read(ROOT / 'RUN_MANIFEST.json')
    assert len(manifest['engines']) == 51 and len(manifest['primary_matrix']) == 48
    assert sum(m.get('island_constraint_mode') == 'prefer' for m in manifest['engines']) == 8
    assert sum(bool(m.get('integer_repair_certificate')) for m in manifest['engines']) == 4
    check_profiles()
    check_received_failures()
    check_chain(manifest)
    source = evaluate_sources(manifest, args.input_root.resolve()) if args.input_root else 'NOT_RUN_NO_INPUT_ROOT'
    proof = dict(status='PASS_OFFLINE_DEVICE_FOLLOWUP', sturm_inventory=2675,sturm_normalized_absolute_tolerance=1e-4,image_hash_equality_required=False,
                 mask_and_unknown_input_rejection='PASS', changed_recipe_not_reused='PASS',
                 compatible_complete_reuse='PASS_OFFLINE_FIXTURE', prior_failure_preserved='PASS',
                 incomplete_not_promoted='PASS', predecessor_wait_and_resume_exclusion='PASS_REAL_PROCESS_FIXTURE',
                 failed_preflight_without_states='PASS', recorded_child_identity='PASS_REAL_PROCESS_FIXTURE',
                 received_micro_failure_rejected='PASS', received_b0_prefer_diagnostic='VERIFIED_RECEIPT_ONLY', legacy_power_receipt_identity='PASS',
                 source_evaluations=source, native_execution='PENDING_JETSON')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
