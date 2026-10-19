#!/usr/bin/env python3
"""Aggregate device receipts into one results table and a device ledger.

Reads the build/evaluation receipts written by integer_tools.py and emits
`results_device.csv` plus `device_ledger.json`. Nothing is recomputed from
engines: every number is copied from a receipt, and a job that failed stays
failed. Precision is reported from the engine inspector audit, never inferred
from the fact that a build succeeded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

# Workstation float parents (results_final.csv, variant=float), used for the source gates.
FLOAT_PARENT = {
    'R4': dict(mIoU=0.8801440638714488, water_IoU=0.7877983473760847,
               equal_event_mIoU=0.8705997603243271, parameters=12230),
    'B0': dict(mIoU=0.8594943460009921, water_IoU=0.7540168383041532,
               equal_event_mIoU=0.8531496664480486, parameters=704714),
}
GATE_MIOU, GATE_WATER = 0.01, 0.02


def metrics(receipt):
    """Pooled and equal-event IoU from a receipt's confusion matrices."""
    conf = np.array(receipt['confusion'], np.int64)
    union = conf.sum(0) + conf.sum(1) - conf.diagonal()
    iou = np.divide(conf.diagonal(), union, out=np.full(2, np.nan), where=union > 0)
    events = {}
    for row in receipt.get('rows', []):
        events.setdefault(row['event'], np.zeros((2, 2), np.int64))
        events[row['event']] += np.array(row['confusion'], np.int64)
    per_event = []
    for mat in events.values():
        uni = mat.sum(0) + mat.sum(1) - mat.diagonal()
        per_event.append(np.nanmean(np.divide(mat.diagonal(), uni,
                                              out=np.full(2, np.nan), where=uni > 0)))
    return dict(mIoU=float(np.nanmean(iou)), water_IoU=float(iou[1]),
                equal_event_mIoU=float(np.mean(per_event)) if per_event else None,
                events=len(events), scenes=len(receipt.get('rows', [])),
                labeled_pixels=int(conf.sum()))


def audit_summary(receipt):
    audit = receipt.get('precision_audit') or {}
    if not audit:
        return {}
    return dict(
        layer_count=audit.get('layer_count'),
        int8_tensor_layers=audit.get('int8_tensor_layers'),
        full_integer_tensor_path=audit.get('full_integer_tensor_path'),
        floating_layers=len(audit.get('floating_layers', [])),
        boundary_reformats=len(audit.get('boundary_reformats', [])),
        floating_compute_layers=len(audit.get('floating_compute_layers', [])),
        unknown_layers=len(audit.get('unknown_layers', [])),
        floating_compute_names=[r.get('Name') for r in audit.get('floating_compute_layers', [])],
    )


def classify(build, evaluation, backend='trt'):
    """Device ledger class. Deliberately conservative."""
    if build is None:
        # CPU reference rows carry no build; they are not a device engine result.
        if evaluation is None:
            return 'not_attempted'
        if evaluation.get('status') == 'evaluated':
            return 'cpu_reference_evaluated_no_engine'
        return 'cpu_reference_failed'
    status = build.get('status')
    if status != 'built':
        reason = str(build.get('reason', ''))
        if 'does not support cubic' in reason or 'UNSUPPORTED_NODE' in reason:
            return 'unsupported_operator_parse_failure'
        if 'CUDA initialization' in reason or 'factory function returned nullptr' in reason:
            return 'failed_device_cuda_unavailable'
        if status == 'strict_certification_failed':
            return 'built_but_strict_certification_failed'
        if 'Builder returned no engine' in reason:
            return 'failed_build_precision_constraints_unsatisfiable'
        return 'failed_build'
    audit = build.get('precision_audit') or {}
    if audit.get('unknown_layers'):
        return 'built_precision_unknown'
    integer_path = audit.get('full_integer_tensor_path')
    if evaluation is None or evaluation.get('status') != 'evaluated':
        base = 'built_not_evaluated'
    else:
        base = 'built_and_evaluated'
    if integer_path:
        return base + '_fully_integer_tensor_path_unverified_kernels'
    if build.get('fp16_enabled'):
        return base + '_mixed_int8_fp16'
    return base + '_int8_with_floating_resize_and_fp32_fallback'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-dir', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()

    jobs = []
    for model in ('R4', 'B0'):
        for attempt_dir in sorted((args.device_dir / 'outputs' / model).glob('attempt*')):
            for build_path in sorted(attempt_dir.glob('*_build.json')):
                variant = build_path.name[:-len('_build.json')]
                build = json.loads(build_path.read_text())
                evaluation, eval_path = None, None
                for cand_dir in sorted((args.device_dir / 'outputs' / model).glob('attempt*')):
                    cand = cand_dir / ('trt_eval_%s.json' % variant)
                    if cand.exists():
                        rec = json.loads(cand.read_text())
                        if rec.get('status') == 'evaluated' or evaluation is None:
                            evaluation, eval_path = rec, cand
                jobs.append(dict(model=model, attempt=attempt_dir.name, variant=variant,
                                 backend='trt', build=build, build_path=build_path,
                                 evaluation=evaluation, eval_path=eval_path))
        for attempt_dir in sorted((args.device_dir / 'outputs' / model).glob('attempt*')):
            for ort_path in sorted(attempt_dir.glob('ort*.json')):
                jobs.append(dict(model=model, attempt=attempt_dir.name,
                                 variant=ort_path.stem, backend='ort', build=None,
                                 build_path=None,
                                 evaluation=json.loads(ort_path.read_text()),
                                 eval_path=ort_path))

    rows, ledger = [], []
    for job in jobs:
        model, build, ev = job['model'], job['build'], job['evaluation']
        parent = FLOAT_PARENT[model]
        row = dict(model=model, attempt=job['attempt'], variant=job['variant'],
                   backend=job['backend'], parameters=parent['parameters'])
        met = {}
        if ev is not None and ev.get('status') == 'evaluated':
            met = metrics(ev)
            row.update(mIoU=met['mIoU'], water_IoU=met['water_IoU'],
                       equal_event_mIoU=met['equal_event_mIoU'],
                       scenes=met['scenes'], events=met['events'],
                       labeled_pixels=met['labeled_pixels'],
                       pooled_delta_vs_float_parent=met['mIoU'] - parent['mIoU'],
                       water_delta_vs_float_parent=met['water_IoU'] - parent['water_IoU'],
                       equal_event_delta_vs_float_parent=met['equal_event_mIoU'] - parent['equal_event_mIoU'],
                       kernel_ms_p50=ev.get('kernel_ms_p50'), kernel_ms_p95=ev.get('kernel_ms_p95'),
                       kernel_ms_mean=ev.get('kernel_ms_mean'),
                       wall_ms_p50=ev.get('inference_wall_ms_p50'),
                       wall_ms_p95=ev.get('inference_wall_ms_p95'),
                       wall_ms_mean=ev.get('inference_wall_ms_mean'),
                       engine_device_memory_bytes=ev.get('engine_device_memory_bytes'))
            gates = dict(
                pooled=row['pooled_delta_vs_float_parent'] > -GATE_MIOU,
                equal_event=row['equal_event_delta_vs_float_parent'] > -GATE_MIOU,
                water=row['water_delta_vs_float_parent'] > -GATE_WATER)
            row['source_gate_pass'] = all(gates.values())
            row['eval_status'] = 'evaluated'
        elif ev is not None:
            row['eval_status'] = ev.get('status')
            row['eval_reason'] = ev.get('reason')
        else:
            row['eval_status'] = 'not_evaluated'

        if build is not None:
            audit = audit_summary(build)
            row.update(build_status=build['status'], build_mode=build.get('mode'),
                       fp16_enabled=build.get('fp16_enabled'),
                       engine_bytes=build.get('engine_bytes'),
                       build_seconds=build.get('elapsed_seconds'),
                       build_reason=build.get('reason'),
                       forced_layer_count=len(build.get('forced_layers', []) or []),
                       **{k: v for k, v in audit.items() if k != 'floating_compute_names'})
        row['device_class'] = classify(build, ev, job['backend'])
        rows.append(row)
        ledger.append(dict(
            model=model, attempt=job['attempt'], variant=job['variant'], backend=job['backend'],
            device_class=row['device_class'],
            build_receipt=str(job['build_path']) if job['build_path'] else None,
            eval_receipt=str(job['eval_path']) if job['eval_path'] else None,
            build_status=(build or {}).get('status'),
            build_reason=(build or {}).get('reason'),
            eval_status=row.get('eval_status'), eval_reason=row.get('eval_reason'),
            precision_audit=audit_summary(build) if build else None,
            metrics=met or None,
            integer_execution_claim=('not_certified_by_tensor_formats_alone'
                                     if build and build.get('status') == 'built' else 'not_applicable')))

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.ledger.write_text(json.dumps(dict(
        gates=dict(pooled_mIoU_max_drop=GATE_MIOU, equal_event_mIoU_max_drop=GATE_MIOU,
                   water_IoU_max_drop=GATE_WATER,
                   note='engineering tolerances, not statistical equivalence'),
        float_parents=FLOAT_PARENT,
        float_parent_note='workstation values from results_final.csv; the float ONNX parents '
                          'were not in the bundle on this device, so no device float baseline '
                          'was measured for them',
        jobs=ledger), indent=2) + '\n')
    print('wrote %s (%d rows) and %s' % (args.csv, len(rows), args.ledger))
    return 0


if __name__ == '__main__':
    sys.exit(main())
