"""Independent receipt-level source analysis, preserving paired seeds and selection."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 64, 91, 107)
T95_DF3 = 3.182446305284263


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(metrics, key):
    if key == 'water_IoU':
        answer = metrics.get(key, metrics.get('wIoU'))
        return float(metrics['iou_per_class'][1] if answer is None else answer) * 100
    return float(metrics[key]) * 100


def paired(left, right, key, kind):
    assert set(left) == set(right) == set(SEEDS)
    differences = [value(left[seed][kind], key) - value(right[seed][kind], key) for seed in SEEDS]
    mean, sd = statistics.mean(differences), statistics.stdev(differences)
    half = T95_DF3 * sd / 2
    return dict(mean_pp=mean, sample_sd_pp=sd, ci95_low_pp=mean-half, ci95_high_pp=mean+half,
                paired_differences_pp=dict(zip(SEEDS, differences)), n_independent_seed_pairs=4,
                uncertainty='paired training-seed Student-t 95 percent CI, df3')


def main():
    groups = {}
    identities = {}

    def inherited(label, records, graph):
        selected = [record for record in records if record['graph'] == graph]
        assert len(selected) == 4 and {record['seed'] for record in selected} == set(SEEDS)
        assert all(record['verified'] and record['completed_updates'] == 7800 for record in selected)
        assert len({record['best_sha256'] for record in selected}) == 4
        groups[label] = {record['seed']: dict(selected=record['selected_source_metrics'],
            final=record['final_source_metrics'], parameters=record['metadata']['best']['parameters'],
            selected_updates=record['selected_updates'], completed_updates=7800,
            lr=record['lr'], receipt_path=record['receipt_path'],
            selected_checkpoint_sha256=record['best_sha256'], final_checkpoint_sha256=record['last_sha256'])
            for record in selected}
        for record in selected:
            identities[record['receipt_path']] = sha(record['receipt_path'])

    reuse = read(ROOT/'reuse_index.json')['records']
    inherited('A0_old_LKD', reuse, 'A0')
    inherited('A1_old_LKD', read(ROOT/'optional_A1_reuse_index.json')['records'], 'A1')
    inherited('R4_LKD', reuse, 'R4')
    inherited('A3_LKD', reuse, 'A3')
    inherited('A3N_LKD', reuse, 'A3N')

    def new(label, paths):
        records = {}
        for seed, path in paths.items():
            receipt = read(path)
            assert receipt['status'] == 'complete' and receipt['outcome'] == 'completed'
            assert receipt['seed'] == seed and receipt['scientific_run'] is True
            assert receipt['completed_updates'] == 7800
            records[seed] = dict(selected=receipt['source_validation']['selected_metrics'],
                final=receipt['source_validation']['final_metrics'], parameters=receipt['parameters'],
                selected_updates=receipt['selected_updates'], completed_updates=7800,
                lr=receipt['lr'], receipt_path=str(path), selected_checkpoint_sha256=receipt['best_sha256'],
                final_checkpoint_sha256=receipt['last_sha256'])
            identities[str(path)] = sha(path)
        assert set(records) == set(SEEDS) and len({r['selected_checkpoint_sha256'] for r in records.values()}) == 4
        groups[label] = records

    lr_grids = {}
    for graph in ('A0', 'A1'):
        selection = read(ROOT/'selections'/f'{graph}_lr.json')
        new(graph+'_reanchored_LKD', {seed: Path(selection['receipt']) if seed == 42
            else ROOT/'receipts'/f'{graph}_selected_s{seed}.json' for seed in SEEDS})
        candidates = [dict(lr=5e-5, selected_mIoU=value(groups[graph+'_old_LKD'][42]['selected'], 'mIoU'),
                           selected_updates=groups[graph+'_old_LKD'][42]['selected_updates'], status='reused')]
        for token in ('5e-4', '5e-3'):
            receipt = read(ROOT/'receipts'/f'{graph}_lr{token}_s42.json')
            candidates.append(dict(lr=receipt['lr'], selected_mIoU=receipt['source_validation']['best_val_mIoU']*100,
                selected_water_IoU=value(receipt['source_validation']['selected_metrics'], 'water_IoU'),
                selected_updates=receipt['selected_updates'], final_mIoU=value(receipt['source_validation']['final_metrics'], 'mIoU'),
                status='completed'))
        winning = max(candidates, key=lambda r: (r['selected_mIoU'], -r['lr']))
        assert winning['lr'] == selection['lr']
        lr_grids[graph] = dict(seed=42, candidates=candidates, chosen_lr=selection['lr'],
                              tie_break='lower LR after identical source mIoU')
    new('R4_CE', {seed: ROOT/'receipts'/f'R4_CE_s{seed}.json' for seed in SEEDS})
    new('A3_CE', {seed: ROOT/'receipts'/f'A3_CE_optimized_s{seed}.json' for seed in SEEDS})

    summaries, rows = {}, []
    for label, group in groups.items():
        summaries[label] = {'parameters': group[42]['parameters'], 'lr': group[42]['lr']}
        for kind in ('selected', 'final'):
            summaries[label][kind] = {metric: dict(mean_percent=statistics.mean(
                value(record[kind], metric) for record in group.values()), sample_sd_percent=statistics.stdev(
                value(record[kind], metric) for record in group.values())) for metric in ('mIoU', 'water_IoU')}
            for seed in SEEDS:
                record = group[seed]
                rows.append(dict(group=label, seed=seed, checkpoint_kind=kind,
                    mIoU_percent=value(record[kind], 'mIoU'), water_IoU_percent=value(record[kind], 'water_IoU'),
                    parameters=record['parameters'], lr=record['lr'], selected_updates=record['selected_updates'],
                    completed_updates=record['completed_updates'], checkpoint_sha256=record[kind+'_checkpoint_sha256'],
                    receipt_path=record['receipt_path']))

    pairs = [('A0_reanchored_LKD', 'A0_old_LKD'), ('A1_reanchored_LKD', 'A1_old_LKD'),
             ('R4_LKD', 'R4_CE'), ('A3_LKD', 'A3_CE'), ('R4_LKD', 'A0_reanchored_LKD'),
             ('R4_LKD', 'A1_reanchored_LKD'), ('R4_LKD', 'A3_LKD'), ('R4_LKD', 'A3N_LKD'),
             ('A3_LKD', 'A3N_LKD')]
    contrasts = []
    for left, right in pairs:
        for kind in ('selected', 'final'):
            for metric in ('mIoU', 'water_IoU'):
                contrasts.append(dict(left=left, right=right, checkpoint_kind=kind, metric=metric,
                                      **paired(groups[left], groups[right], metric, kind)))

    pilots, pilot_contrasts = {}, []
    for arm in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'HC', 'HL'):
        pilots[arm] = {}
        for seed in (42, 64):
            path = ROOT/'receipts'/f'{arm}_s{seed}.json'
            receipt = read(path)
            assert receipt['completed_updates'] == 1560 and receipt['gradient_checks']['updates_checked'] == 1560
            pilots[arm][seed] = {'selected_updates': receipt['selected_updates'],
                'completed_updates': 1560, 'selected_checkpoint_sha256': receipt['best_sha256'],
                'final_checkpoint_sha256': receipt['last_sha256']}
            for kind, key in (('start', 'start_metrics'), ('selected', 'selected_metrics'), ('final', 'final_metrics')):
                metrics = receipt['source_validation'][key]
                pilots[arm][seed][kind] = {metric: value(metrics, metric) for metric in ('mIoU', 'water_IoU')}
            identities[str(path)] = sha(path)
    pilot_pairs = [('C3','C0'), ('C3','C4'), ('C3','C5'), ('C1','C0'), ('C2','C0'),
                   ('C3','C2'), ('C4','C1'), ('C5','C1'), ('HL','HC'), ('C0','HL'), ('C0','HC')]
    for left, right in pilot_pairs:
        for kind in ('selected', 'final'):
            for metric in ('mIoU', 'water_IoU'):
                differences = [pilots[left][seed][kind][metric] - pilots[right][seed][kind][metric] for seed in (42,64)]
                pilot_contrasts.append(dict(left=left, right=right, checkpoint_kind=kind, metric=metric,
                    mean_pp=statistics.mean(differences), sample_sd_pp=statistics.stdev(differences),
                    paired_seed_differences_pp=dict(zip((42,64), differences)), n_independent_seed_pairs=2,
                    uncertainty='descriptive two-seed pilot; no inferential CI'))

    promotion = read(ROOT/'selections/promotion.json')
    diagnostics = read(ROOT/'diagnostics.json')
    output = dict(metric_unit='percent', contrast_unit='percentage_points', seed_set=list(SEEDS),
        independent_receipt_sha256=identities, groups=summaries, per_seed=rows, contrasts=contrasts,
        LR_grids=lr_grids, pilot_per_seed=pilots, pilot_contrasts=pilot_contrasts,
        prospective_gate=promotion, TRAIN_augmentation_diagnostics=diagnostics['train'],
        interpretation_limits=['Source scores selected on the same validation set; LR-selected recipe comparison.',
            'Scratch CE is cached teacher-mask focal, not human-labelled teacher-free training.',
            'Selected continuation starts are permitted; identical arm selections are checkpoint aliases, not independent discoveries.',
            'Final continuation source scores describe optimization movement; final external readouts not reserved.',
            'HC/HL/C0 two-seed continuations cannot retroactively attribute scratch KD gains.'])
    destination = Path(__file__).parent/'source_analysis.json'
    assert not destination.exists(), 'Preserve prior morning analysis'
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    with (destination.parent/'source_per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    compact = {'groups': {label: {'parameters': d['parameters'], 'selected_mIoU': d['selected']['mIoU']['mean_percent'],
        'final_mIoU': d['final']['mIoU']['mean_percent']} for label,d in summaries.items()},
        'selected_mIoU_contrasts': [d for d in contrasts if d['checkpoint_kind']=='selected' and d['metric']=='mIoU'],
        'pilot_final_mIoU_contrasts': [d for d in pilot_contrasts if d['checkpoint_kind']=='final' and d['metric']=='mIoU']}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
