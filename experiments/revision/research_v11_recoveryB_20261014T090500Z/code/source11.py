"""Source-only V11 metrics, deficit gates and reference-conditional diagnostics."""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np


def deficit_gate(results):
    """Four-seed source gate; missing strata never count as negative evidence."""
    assert sorted(r['seed'] for r in results) == [42, 64, 91, 107]
    loss = float(np.mean([r['FP']['equal_event_water_IoU'] - r['Q0']['equal_event_water_IoU']
                          for r in results]))
    eligible, diagnostics = loss >= .003, {}
    for name in ['dark_land_false_water', 'bright_water_miss']:
        denominators = [int(r['Q0'][name]['denominator']) for r in results]
        rates = [r['Q0'][name]['rate'] - r['FP'][name]['rate'] for r in results
                 if r['Q0'][name]['rate'] is not None and r['FP'][name]['rate'] is not None]
        # The same reference pixels occur in four seeds; do not multiply the
        # contributing pixel count by four to meet the 1,000-pixel gate.
        assert len(set(denominators)) == 1
        enough = denominators[0] >= 1000 and len(rates) == 4
        difference = float(np.mean(rates)) if len(rates) == 4 else None
        passed = enough and difference >= .01
        eligible |= passed
        diagnostics[name] = dict(difference=difference, reference_pixels=denominators[0],
                                  sufficient_support=enough, passed=bool(passed))
    return dict(outcome='ADMITTED' if eligible else 'NO_MATERIAL_SOURCE_DEFICIT',
                mean_water_IoU_loss=loss, water_IoU_gate=loss >= .003,
                strata=diagnostics, source_only=True, seeds=[42, 64, 91, 107],
                backend='same CPU float32 backend for both paths; simulator only',
                rule='mean equal-event water IoU loss>=0.003 OR supported dark/bright error increase>=0.01')


def _rate(numerator, denominator):
    return dict(numerator=int(numerator), denominator=int(denominator),
                rate=float(numerator / denominator) if denominator else None)


def evaluate_source(model, inputs, root, entry, *, seed, arm, step):
    """Uses only registered S11-validation windows; no target evaluator outputs."""
    import torch
    import common10 as c
    import v10_reference as v
    from evaluate10 import counts, scores, strata, hist
    from repair11 import (read, sha, save, write_csv, paired_logits, priority_scores,
                          select_priority, scoring_state)
    root = Path(root)
    rows = [r for r in read(inputs['source_windows']['path'])['windows']
            if r['dataset'] == 's11val' and not r.get('omit', False)]
    assert rows and all(r['dataset'] == 's11val' for r in rows)
    baseline_entry = inputs['parents'][str(seed)]['FP_parent']
    parent = c.load(baseline_entry)
    teacher_files = {Path(r['path']).stem: r for r in inputs.get('source_teacher_predictions', [])}
    cells = defaultdict(lambda: np.zeros(4, np.int64))
    dark, bright = np.zeros(4, np.int64), np.zeros(4, np.int64)
    dry_alarms = np.zeros(3, np.int64)
    dry_n = 0
    diagnostic_rows, harm_rows, per_event = [], [], []
    tic, cpu = time.monotonic(), time.process_time()
    inference_seconds = 0.
    histogram = np.zeros((4096, 2), np.int64)
    at_boundary = np.zeros(4097, np.int64)
    clipped_below, clipped_above, below_or_at_lower, above_upper = 0, 0, 0, 0
    parent_positive, total_valid = 0, 0
    access_path = root / 'source_label_access.jsonl'
    import json
    with scoring_state(model), torch.no_grad(), open(access_path, 'a', buffering=1) as access:
        for row in rows:
            access.write(json.dumps(dict(path=row['path'], event=row['event'], model=entry['id'],
                                         purpose='S11 validation source gate/checkpoint/diagnostic',
                                         dataset='s11val', time=time.time())) + '\n')
            with np.load(row['path'], allow_pickle=False) as archive:
                x, y = archive['image'].copy(), archive['mask'].copy()
                cloud = archive['cloud'].copy() if 'cloud' in archive and archive['cloud'].size else None
            valid = y != -1
            image = torch.from_numpy(x)[None]
            forward_start = time.monotonic()
            logits = model(image)
            inference_seconds += time.monotonic() - forward_start
            assert tuple(logits.shape[-2:]) == tuple(y.shape) and torch.isfinite(logits).all()
            prediction = (logits[0, 1] > logits[0, 0]).numpy()
            parent_prediction = parent(image).argmax(1)[0].numpy() == 1
            margin_current = (logits[0, 1] - logits[0, 0]).numpy()
            histogram += hist(margin_current[valid], y[valid].astype(np.int64))
            values = margin_current[valid].astype(np.float64)
            clipped_below += int((values < -32).sum())
            clipped_above += int((values >= 32).sum())
            below_or_at_lower += int((values <= -32).sum())
            above_upper += int((values > 32).sum())
            positions = (values + 32) * 64
            exact = (positions >= 0) & (positions <= 4096) & (positions == np.floor(positions))
            at_boundary += np.bincount(positions[exact].astype(np.int64), minlength=4097)
            parent_positive += int(parent_prediction[valid].sum())
            total_valid += int(valid.sum())
            cell = counts(prediction, y)
            cells[row['event']] += cell
            parts = strata(y, v.reflectance(x), cloud)
            dark += counts(prediction, y, parts['dark'])
            bright += counts(prediction, y, parts['bright'])
            if valid.any() and not (y[valid] == 1).any():
                dry_n += 1
                fraction = float(prediction[valid].mean())
                dry_alarms += np.array([fraction >= threshold for threshold in [.001, .01, .05]])
            for label, reference_class in [('false_water', 0), ('missed_water', 1)]:
                support = valid & (y == reference_class)
                old_wrong, new_wrong = parent_prediction != (y == 1), prediction != (y == 1)
                harm_rows.append(dict(model=entry['id'], point=inputs['point'], arm=arm, seed=seed,
                                      step=step, event=row['event'], tile=row['name'], error=label,
                                      reference_pixels=int(support.sum()),
                                      corrected=int((support & old_wrong & ~new_wrong).sum()),
                                      newly_broken=int((support & ~old_wrong & new_wrong).sum()),
                                      baseline='starting FP parent', source_only=True))
            teacher_identity = teacher_files.get(Path(row['path']).stem)
            if teacher_identity is None or not entry.get('quantized'):
                continue
            assert sha(teacher_identity['file']) == teacher_identity['sha256']
            margin = np.load(teacher_identity['file'], allow_pickle=False)
            assert margin.shape == y.shape and np.isfinite(margin).all()
            teacher = torch.from_numpy(np.stack([-margin / 2, margin / 2]))[None]
            fp, quant = paired_logits(model, image)
            score = priority_scores(teacher, fp, quant)['D']
            selected, _, _ = select_priority(score, torch.from_numpy(valid)[None], teacher.argmax(1))
            chosen, d = selected[0].numpy(), score[0].numpy()
            t = margin > 0
            f, q = fp.argmax(1)[0].numpy() == 1, quant.argmax(1)[0].numpy() == 1
            truth = y == 1
            damage, benefit = valid & (f == truth) & (q != truth), valid & (f != truth) & (q == truth)
            code = ((t == truth).astype(np.int64) * 8 + (f == truth).astype(np.int64) * 4
                    + (parent_prediction == truth).astype(np.int64) * 2 + (q == truth).astype(np.int64))
            for stratum in ['all', 'dark', 'bright', 'shore<=2', 'shore3-10', 'shore>10']:
                for teacher_class in [0, 1]:
                    support = valid & parts[stratum] & (t == bool(teacher_class))
                    mask = chosen & support
                    joint = np.bincount(code[support], minlength=16)
                    diagnostic_rows.append(dict(model=entry['id'], point=inputs['point'], arm=arm,
                                                seed=seed, step=step, event=row['event'], tile=row['name'],
                                                stratum=stratum, teacher_class=teacher_class,
                                                valid_pixels=int(support.sum()), selected_pixels=int(mask.sum()),
                                                positive_d_mass=float(d[support & (d > 0)].sum(dtype=np.float64)),
                                                negative_d_mass=float(d[support & (d < 0)].sum(dtype=np.float64)),
                                                fp_correct_q_wrong=int((damage & support).sum()),
                                                selected_fp_correct_q_wrong=int((damage & mask).sum()),
                                                fp_wrong_q_correct=int((benefit & support).sum()),
                                                selected_fp_wrong_q_correct=int((benefit & mask).sum()),
                                                selected_teacher_wrong=int((mask & (t != truth)).sum()),
                                                oracle_selected_damage=min(int(mask.sum()), int((damage & support).sum())),
                                                oracle='reference-conditional undeployable upper bound at same count',
                                                joint_correctness_code='teacher*8+currentFP*4+startingFP*2+Q',
                                                joint_counts=json.dumps(joint.tolist()), source_only=True))
    for event, cell in cells.items():
        per_event.append(dict(model=entry['id'], point=inputs['point'], arm=arm, seed=seed,
                              step=step, event=event, tp=int(cell[0]), fp=int(cell[1]),
                              fn=int(cell[2]), tn=int(cell[3]), **scores(cell), source_only=True))
    diagnostic_dir = root / 'source_diagnostics' / entry['id']
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    np.save(diagnostic_dir / 'source_margin_histogram.npy', histogram, allow_pickle=False)
    write_csv(diagnostic_dir / 'priority_diagnostics.csv', diagnostic_rows)
    write_csv(diagnostic_dir / 'correction_harm.csv', harm_rows)
    write_csv(diagnostic_dir / 'repair_by_event.csv', per_event)
    event_iou = [r['water_IoU'] for r in per_event if r['water_IoU'] is not None]
    assert event_iou, 'No reference water denominator in source gate'
    above = np.r_[histogram.sum(1)[::-1].cumsum()[::-1], 0] - at_boundary
    above[0] = total_valid - below_or_at_lower
    above[-1] = above_upper
    bin_index = min(range(4097), key=lambda i: (abs(int(above[i]) - parent_positive), abs(i - 2048), i))
    return dict(outcome='completed', model=entry['id'], checkpoint_sha256=entry['sha256'],
                source_only=True, windows=len(rows), equal_event_water_IoU=float(np.mean(event_iou)),
                pooled=scores(sum(cells.values())), event_counts={k: a.tolist() for k, a in cells.items()},
                dark_land_false_water=_rate(dark[1], dark[1] + dark[3]),
                bright_water_miss=_rate(bright[2], bright[0] + bright[2]),
                dry_window_alarms={str(t): _rate(n, dry_n) for t, n in zip([.001, .01, .05], dry_alarms)},
                elapsed_seconds=time.monotonic() - tic, process_cpu_seconds=time.process_time() - cpu,
                model_inference_seconds=inference_seconds,
                per_model_source_tile_seconds=inference_seconds / len(rows),
                source_positive_fraction=float(sum(cells.values())[0:2].sum() / total_valid),
                parent_positive_fraction=float(parent_positive / total_valid),
                source_matched_tau=float(bin_index / 64 - 32),
                source_matched_positive_fraction=float(above[bin_index] / total_valid),
                source_matching_fraction_error=float((above[bin_index] - parent_positive) / total_valid),
                source_clipped_tail_counts=dict(below_minus32=clipped_below, at_or_above32=clipped_above),
                threshold_policy='Fixed4097 threshold boundaries[-32,32] spaced1/64; exact boundary-atom and tail counts; minimize source positive-fraction error to starting FP; ties closest to zero then lower threshold. Grid/ties/tails may prevent exact fraction matching; no error bound assumed.',
                source_histogram=str(diagnostic_dir / 'source_margin_histogram.npy'),
                priority_diagnostics='completed' if diagnostic_rows else 'not applicable to FP' if not entry.get('quantized') else 'UNAVAILABLE: no verified source teacher margins',
                backend='CPU float32 fixed fake quantization; no CPU/GPU identity claim',
                support='registered S11 validation reference-valid support; native class mapping and geometry')
