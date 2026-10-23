"""Independent native accuracy and integer-format admission for the repair."""
import math
import re


def source_gate(baseline, candidate):
    paths = [('pooled_mIoU', 'pooled', 'miou'),
             ('equal_event_mIoU', 'equal_event', 'miou'),
             ('pooled_water_IoU', 'pooled', 'water_iou')]
    drops = {}
    for label, group, key in paths:
        a, b = baseline[group][key], candidate[group][key]
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Undefined source accuracy ' + label)
        drops[label] = a - b
    counts = [sum(r['counts_tn_fp_fn_tp'][i] for r in candidate['rows']) for i in range(4)]
    # A collapsed all-background engine must fail even on a weak baseline.
    detects_water = counts[3] > 0
    return dict(passed=detects_water and all(v <= .02 for v in drops.values()),
                drops=drops, absolute_limit=.02, true_positive_water_pixels=counts[3],
                predicts_water=counts[1] + counts[3] > 0,
                accuracy_is_not_export_equivalence=True)


def integer_certificate(build, layers, model):
    if build.get('status') != 'built' or build.get('model_sha256') != model['onnx_sha256']:
        raise ValueError('Wrong build/model identity')
    if not build.get('linear_host_bindings') or not build.get('integer_host_bindings'):
        raise ValueError('Require linear unvectorized INT8 host bindings')
    if not build.get('precision_audit', {}).get('full_integer_tensor_path'):
        raise ValueError('Floating or unknown tensor formats in repaired pure engine')
    rows = layers.get('Layers', [])
    if not rows:
        raise ValueError('Missing detailed layer inspector')
    int8 = 0
    for row in rows:
        tensors = row.get('Inputs', []) + row.get('Outputs', [])
        if not tensors:
            raise ValueError('Missing tensor formats: ' + row.get('Name', '?'))
        for tensor in tensors:
            value = str(tensor.get('Format/Datatype', ''))
            if re.search(r'FP32|FP16|BF16|Float|Half', value, re.I):
                raise ValueError('Floating fallback: ' + row.get('Name', '?'))
            if not re.search(r'INT8|INT32|BOOL', value, re.I):
                raise ValueError('Unknown tensor format: ' + value)
            int8 += bool(re.search('INT8', value, re.I))
    if not int8:
        raise ValueError('No INT8 tensor evidence')
    return True
