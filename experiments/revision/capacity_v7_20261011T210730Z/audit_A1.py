"""Read-only compatibility audit for the declared optional A1 re-anchor bundle."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
from audit_v7 import SEEDS, check, checkpoint_metadata, digest, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    base, run = Path(args.base).resolve(), Path(args.run_dir).resolve()
    capsule = base / 'frozen/297031f5f4ae1df5'
    provenance_path = run / 'provenance.json'
    provenance = read(provenance_path)
    errors, records = [], []
    common_pass = provenance.get('status') == 'complete' and provenance.get('cohort_verified') is True
    identity = read(capsule / 'source_identity.json')
    capsule_checks = [check(capsule / rel, expected) for rel, expected in identity['files'].items()]
    shared = read(base / 'shared_input_identity.json')
    shared_checks = [check(row['path'], row['sha256']) for row in shared['files']]
    common_pass &= all(row['match'] for row in capsule_checks + shared_checks)
    common_pass &= digest(identity['files']) == identity['sha256']
    dependencies = read(base / 'stage2_fp/manifest.json')['env']['PYTHONPATH']
    for path in reversed(dependencies.split(os.pathsep)):
        if path:
            sys.path.insert(0, path)
    sys.path.insert(0, str(capsule))
    sys.path.insert(0, str(base / 'code/v5'))
    if common_pass:
        import torch
        torch.set_num_threads(2)
        import rungs
        model = rungs.build('A1').cpu()
        expected_cfg = dict(epochs=50, weight_decay=.05, task_loss='focal', gamma=2.,
            ignore_index=-1, alpha=.5, temperature=4., use_amp=False, grad_clip=5.,
            scheduler='cosine', warmup_epochs=0, val_every_n_epochs=2,
            val_smoothing_window=1, lr=5e-5)
        for seed in SEEDS:
            rp = base / f'results/A1_FP_s{seed}.json'
            receipt = read(rp)
            tp = base / 'v5/terminal' / rp.name
            terminal = read(tp)
            try:
                checks = [check(receipt[key + '_path'], receipt[key + '_sha256']) for key in ('best', 'last')]
                # Seed42 is a v2 reuse alias; other seeds are direct v4 executions.
                original_path = Path(receipt.get('original_receipt', str(rp)))
                original_check = check(original_path, receipt.get('original_receipt_sha256', sha(rp)))
                original = read(original_path)
                metadata = {key: checkpoint_metadata(receipt[key + '_path'], dict(expected_cfg, seed=seed), model)
                            for key in ('best', 'last')}
                passed = all(row['match'] for row in checks) and original_check['match']
                passed &= all(row['checks_pass'] for row in metadata.values())
                passed &= receipt['status'] == 'complete' and receipt['outcome'] == 'completed'
                passed &= receipt['graph'] == 'A1' and original['graph'] == 'A1'
                passed &= metadata['last']['training_state']['global_step'] == receipt['completed_updates'] == 7800
                passed &= metadata['best']['training_state']['global_step'] == receipt['selected_updates']
                passed &= abs(metadata['best']['metrics']['mIoU'] - receipt['source_validation']['best_val_mIoU']) < 1e-7
                passed &= metadata['best']['initial_model_sha256'] == metadata['last']['initial_model_sha256']
                passed &= bool(metadata['best']['initial_model_sha256'])
                passed &= terminal['checkpoint_sha256'] == receipt['best_sha256'] and terminal['graph'] == 'A1'
                counts = {key: len(value['per_image_counts']) for key, value in terminal['terminal_results'].items()}
                passed &= counts == dict(s11=90, sturm=2675, wf=785)
                for dataset in terminal['terminal_results'].values():
                    if any(len(c) != 2 or any(len(row) != 3 or any(not isinstance(x, int) or x < 0 for x in row)
                        for row in c) for c in dataset['per_image_counts'].values()):
                        passed = False
                if not passed:
                    errors.append(f'A1 seed {seed}: scientific identity failed')
                records.append(dict(status='reused' if passed else 'blocked', verified=bool(passed),
                    family='A1_LKD_recipe', graph='A1', method='lkd', seed=seed, lr=5e-5,
                    receipt_path=str(rp), receipt_sha256=sha(rp), execution_receipt_path=str(original_path),
                    execution_receipt_sha256=sha(original_path), original_receipt_check=original_check,
                    best_path=receipt['best_path'], best_sha256=receipt['best_sha256'],
                    last_path=receipt['last_path'], last_sha256=receipt['last_sha256'],
                    selected_updates=receipt['selected_updates'], completed_updates=receipt['completed_updates'],
                    source_validation=receipt['source_validation'], selected_source_metrics=metadata['best']['metrics'],
                    final_source_metrics=metadata['last']['metrics'], terminal_receipt_path=str(tp),
                    terminal_receipt_sha256=sha(tp), terminal_alias_of=terminal.get('alias_of'),
                    per_image_counts_location='terminal_results.{dataset}.per_image_counts',
                    per_image_counts_n=counts, per_image_csv=None, metadata=metadata, checkpoint_checks=checks,
                    initial_model_sha256=metadata['best']['initial_model_sha256'],
                    historical_training_elapsed_seconds=receipt['elapsed_seconds'],
                    normalization_code_sha256=sha(capsule/'data/sen1floods11.py'),
                    pseudo_augmentation_code_sha256=sha(capsule/'data/pseudo_labeled.py'),
                    graph_code_sha256=sha(base/'code/v5/rungs.py'),
                    compression_graph_code_sha256=sha(capsule/'models/compression.py'),
                    loss_code_sha256=sha(capsule/'training/losses.py'),
                    source_selection='S11 validation pooled mIoU, strict greater, earlier tie, every312 updates',
                    update_schedule='50 epoch-stepped cosine epochs x156 updates, AdamW, no warmup',
                    selected_checkpoint_external_status='reused', final_checkpoint_external_status='not_run'))
            except Exception as exc:
                errors.append(f'A1 seed {seed}: {type(exc).__name__}: {exc}')
    else:
        errors.append('Current main provenance or pinned shared sources failed verification')
    elapsed = [record['historical_training_elapsed_seconds'] for record in records if record['verified']]
    result = dict(schema_version=1, status='complete' if common_pass and not errors and len(records) == 4 else 'blocked',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        read_only_historical_audit=True, gpu_used=False, graph='A1', family='A1_LKD_recipe',
        records=records, independent_seed_set=list(SEEDS),
        distinct_selected_checkpoint_hashes=len({record['best_sha256'] for record in records if record['verified']}),
        shared_inputs_verified=bool(common_pass), main_provenance_path=str(provenance_path),
        main_provenance_sha256=sha(provenance_path), capsule_checks=capsule_checks,
        shared_input_checks=shared_checks, errors=errors,
        initial_seed42_sha256=next((record['initial_model_sha256'] for record in records if record['seed'] == 42), None),
        historical_training_seconds=elapsed, historical_training_mean_seconds=statistics.mean(elapsed) if elapsed else None,
        historical_training_max_seconds=max(elapsed) if elapsed else None,
        historical_time_unit='scientific receipt elapsed; admission must use measured v7 smoke plus overhead and hard timeout')
    destination = run / 'optional_A1_reuse_index.json'
    if destination.exists():
        raise RuntimeError('Optional audit already exists; preserve it and use a separately identified audit')
    write(destination, result)
    print(json.dumps({key: result[key] for key in ('status', 'initial_seed42_sha256',
        'historical_training_mean_seconds', 'historical_training_max_seconds', 'errors')}))
    if result['status'] != 'complete':
        sys.exit(1)


if __name__ == '__main__':
    main()
