"""Bounded Stage A audit: TRAIN augmentations and reference-labelled source regions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from v7_loss import NORM_MEAN, NORM_STD, RULE_EPSILON, rule_feature


class PairedAugmentedViews(Dataset):
    """Exact frozen augmentation once; cached masks and both images share geometry."""
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        idx = self.indices[offset]
        item = torch.load(self.dataset.files[idx], map_location='cpu', weights_only=True)
        x = item['image']
        # Actual frozen order: per-band reflectance gain, followed by shared geometry.
        xp = self.dataset._apply_mult_noise(x, idx)
        out = self.dataset._apply_geom({'image_geometry': x, 'image_photo': xp,
                                        'mask': item['mask'].clone()}, idx)
        out['tile'] = self.dataset.files[idx].name
        return out


def rates(counts, denominator):
    return {key: value / denominator if denominator else None for key, value in counts.items()}


@torch.no_grad()
def train_row(tile, cached, t_geometry, t_photo, image_photo):
    p_geometry, p_photo = F.softmax(t_geometry, 0), F.softmax(t_photo, 0)
    h_geometry, h_photo = p_geometry.argmax(0), p_photo.argmax(0)
    valid = cached != -1
    r, available = rule_feature(image_photo.unsqueeze(0))
    available, rule = available[0], r[0] > 0
    support = valid & available
    counts = {'geometry_cached_live_disagreement': int((valid & (cached != h_geometry)).sum()),
              'photometric_live_change': int((valid & (h_geometry != h_photo)).sum()),
              'cached_live_augmented_disagreement': int((valid & (cached != h_photo)).sum()),
              'teacher_rule_agreement': int((support & (h_photo == rule.long())).sum()),
              'teacher_rule_disagreement': int((support & (h_photo != rule.long())).sum())}
    n = int(valid.sum())
    row = {'tile': tile, 'valid_pixels': n, 'rule_available_valid_pixels': int(support.sum()),
           'rule_unavailable_valid_pixels': int((valid & ~available).sum()), **counts,
           'teacher_geometry_confidence_sum': float(p_geometry.amax(0)[valid].sum()),
           'teacher_photo_confidence_sum': float(p_photo.amax(0)[valid].sum()),
           'teacher_photo_entropy_sum': float(-(p_photo * F.log_softmax(t_photo, 0)).sum(0)[valid].sum()),
           'teacher_geometry_mean_confidence': float(p_geometry.amax(0)[valid].mean()) if n else None,
           'teacher_photo_mean_confidence': float(p_photo.amax(0)[valid].mean()) if n else None,
           'geometry_cached_live_disagreement_rate': counts['geometry_cached_live_disagreement'] / n if n else None,
           'photometric_live_change_rate': counts['photometric_live_change'] / n if n else None,
           'cached_live_augmented_disagreement_rate': counts['cached_live_augmented_disagreement'] / n if n else None}
    for cls, name in ((0, 'land'), (1, 'water')):
        cached_cls = valid & (cached == cls)
        geometry_cls = valid & (h_geometry == cls)
        photo_cls = support & (h_photo == cls)
        row[name + '_cached_pixels'] = int(cached_cls.sum())
        row[name + '_cached_live_disagreement'] = int((cached_cls & (cached != h_photo)).sum())
        row[name + '_geometry_live_pixels'] = int(geometry_cls.sum())
        row[name + '_photometric_live_change'] = int((geometry_cls & (h_geometry != h_photo)).sum())
        row[name + '_teacher_rule_available_pixels'] = int(photo_cls.sum())
        row[name + '_teacher_rule_disagreement'] = int((photo_cls & (h_photo != rule.long())).sum())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-train', type=int, default=128)
    p.add_argument('--max-val', type=int, default=89)
    p.add_argument('--seed', type=int, default=1729)
    a = p.parse_args()
    assert 0 < a.max_train <= 128 and 0 < a.max_val <= 89
    sys.path.insert(0, str(Path(a.capsule).resolve()))
    import scripts.compression_jobs as cj
    from utils.seeding import seed_basic
    from data.sen1floods11 import NORM_MEAN as frozen_mean, NORM_STD as frozen_std
    assert frozen_mean == NORM_MEAN and frozen_std == NORM_STD
    start = time.monotonic()
    spec = json.loads(Path(a.spec).read_text())
    spec['seed'] = a.seed
    seed_basic(a.seed)
    teacher = cj.teacher(spec)
    ds = cj.pseudo(spec, augment=True)
    indices = sorted(range(len(ds)), key=lambda i: hashlib.sha256(
        (str(a.seed) + ':' + ds.files[i].name).encode()).hexdigest())[:a.max_train]
    selected = [{'file': ds.files[i].name, 'sha256': cj.file_hash(ds.files[i])} for i in indices]
    views = PairedAugmentedViews(ds, indices)
    # Dedicated audit stream, no training process exists in this command.
    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(views, batch_size=4, shuffle=False, num_workers=spec['workers'],
                        pin_memory=True, generator=generator)
    train_rows = []
    with torch.no_grad():
        for batch in loader:
            geometry = batch['image_geometry'].cuda().float()
            photo = batch['image_photo'].cuda().float()
            t_geometry, t_photo = teacher(geometry), teacher(photo)
            cj.finite_output(teacher, (geometry,), t_geometry)
            cj.finite_output(teacher, (photo,), t_photo)
            for offset, tile in enumerate(batch['tile']):
                train_rows.append(train_row(tile, batch['mask'][offset].cuda(),
                    t_geometry[offset], t_photo[offset], photo[offset]))
    sum_keys = [key for key, value in train_rows[0].items()
                if isinstance(value, (int, float)) and not key.endswith('_rate')
                and 'mean_confidence' not in key]
    totals = {key: sum(row[key] for row in train_rows) for key in sum_keys}
    n = totals['valid_pixels']
    train_summary = {'tiles': len(train_rows), 'counts': totals,
        'geometry_cached_live_disagreement_rate': totals['geometry_cached_live_disagreement'] / n if n else None,
        'photometric_live_change_rate': totals['photometric_live_change'] / n if n else None,
        'cached_live_augmented_disagreement_rate': totals['cached_live_augmented_disagreement'] / n if n else None,
        'teacher_rule_disagreement_rate_available': totals['teacher_rule_disagreement']
            / totals['rule_available_valid_pixels'] if totals['rule_available_valid_pixels'] else None,
        'teacher_mean_confidence_geometry': totals['teacher_geometry_confidence_sum'] / n if n else None,
        'teacher_mean_confidence_photo': totals['teacher_photo_confidence_sum'] / n if n else None,
        'reference_correctness': 'unknown on unlabelled TRAIN; no signed benefit or correction claims'}
    val, _ = cj.loaders(spec)
    val_rows, seen = [], 0
    with torch.no_grad():
        for batch in val:
            if seen >= a.max_val:
                break
            images = batch['image'][:a.max_val - seen].cuda().float()
            masks = batch['mask'][:a.max_val - seen].cuda().long()
            logits = teacher(images)
            cj.finite_output(teacher, (images,), logits)
            hard = logits.argmax(1)
            r, available = rule_feature(images)
            rule = (r > 0).long()
            for offset in range(images.shape[0]):
                valid = masks[offset] != -1
                support = valid & available[offset]
                teacher_correct, rule_correct = hard[offset] == masks[offset], rule[offset] == masks[offset]
                both_correct = support & teacher_correct & rule_correct
                teacher_only = support & teacher_correct & ~rule_correct
                rule_only = support & ~teacher_correct & rule_correct
                both_wrong = support & ~teacher_correct & ~rule_correct
                row = {'tile': Path(val.dataset.image_files[seen]).name,
                       'valid_pixels': int(valid.sum()), 'rule_available_valid_pixels': int(support.sum()),
                       'both_correct': int(both_correct.sum()), 'teacher_correct_rule_wrong': int(teacher_only.sum()),
                       'rule_correct_teacher_wrong': int(rule_only.sum()), 'both_wrong': int(both_wrong.sum()),
                       'disagreements': int((teacher_only | rule_only).sum())}
                row['signed_disagreement_benefit'] = ((row['teacher_correct_rule_wrong']
                    - row['rule_correct_teacher_wrong']) / row['disagreements']) if row['disagreements'] else None
                val_rows.append(row)
                seen += 1
    val_totals = {key: sum(row[key] for row in val_rows)
                  for key in val_rows[0] if key not in ('tile', 'signed_disagreement_benefit')}
    val_totals['signed_disagreement_benefit'] = ((val_totals['teacher_correct_rule_wrong']
        - val_totals['rule_correct_teacher_wrong']) / val_totals['disagreements']) if val_totals['disagreements'] else None
    directory = Path(a.out).resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    for filename, rows in (('train_diagnostics_per_tile.csv', train_rows), ('source_rule_regions.csv', val_rows)):
        with (directory / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {'stage': 'A', 'seed': a.seed, 'sample_selection': 'ascending SHA256(seed:TRAIN filename)',
               'train_sample': selected, 'train_sample_sha256': hashlib.sha256(
                    json.dumps(selected, sort_keys=True).encode()).hexdigest(),
               'teacher_sha256': spec['teacher_sha256'], 'spec_sha256': cj.file_hash(a.spec),
               'norm_mean': NORM_MEAN, 'norm_std': NORM_STD, 'rule_epsilon_reflectance': RULE_EPSILON,
               'rule_threshold': 0., 'band_order': ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'],
               'augmentation_implementation': 'exact frozen PseudoLabeledDataset methods; dedicated audit worker stream',
               'augmentation_photo': "x'=k*x+(k-1)*mean/std, per-band k U(.8,1.2), p=.5",
               'prediction_grid_alignment': 'shared frozen geometry across cached mask, no-photo and photo image; full-grid predictions',
               'train': train_summary, 'source_validation': {'tiles': seen, 'counts': val_totals},
               'refresh_gate_threshold': .01,
               'target_refresh_gate': train_summary['cached_live_augmented_disagreement_rate'] >= .01,
               'external_data_used': False, 'source_labels_used_for_rule_tuning': False,
               'bounded_sample': a.max_train, 'source_validation_limit': a.max_val}
    cj.completed(a.out, start, **payload)
    print(json.dumps({'out': a.out, 'train_disagreement_rate': train_summary['cached_live_augmented_disagreement_rate'],
                      'target_refresh_gate': payload['target_refresh_gate']}))


if __name__ == '__main__':
    main()
