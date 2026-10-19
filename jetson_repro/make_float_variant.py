#!/usr/bin/env python3
"""Strip Q/DQ from a QDQ graph to obtain a float graph for FP32/FP16 engines.

Why this exists. The workstation bundle that held `model_float.onnx` for both
candidates is absent from this device, and the handover forbids re-exporting from the
checkpoints under device torch 1.13. The only float graph obtainable here is the one
recovered by deleting the QuantizeLinear/DequantizeLinear pairs from the deployment
QDQ graph.

What this is, precisely. The exports use the float-weight Q/DQ workaround, so weight
initializers are float values already rounded onto the quantization grid. Removing
Q/DQ therefore yields **float arithmetic over grid-rounded weights**: a clean
precision ablation against the INT8 engine on identical weights, and the correct
baseline for an FP32/FP16/INT8 latency comparison. It is NOT the original float parent
(R4 0.880144, B0 0.859494 source mIoU) and must never be reported as such - those
parents were trained without any quantization grid.

Activation quantization disappears entirely, so accuracy may differ slightly in either
direction from both the INT8 graph and the true float parent.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import onnx


def strip_qdq(model):
    """Remove every QuantizeLinear -> DequantizeLinear pair, rewiring consumers."""
    graph = model.graph
    removed = []
    changed = True
    while changed:
        changed = False
        produced_by = {o: n for n in graph.node for o in n.output}
        consumers = collections.defaultdict(list)
        for node in graph.node:
            for name in node.input:
                consumers[name].append(node)
        for dq in list(graph.node):
            if dq.op_type != 'DequantizeLinear':
                continue
            q = produced_by.get(dq.input[0])
            if q is None or q.op_type != 'QuantizeLinear':
                continue
            # Only collapse when the Quantize feeds this Dequantize alone.
            if [c.name for c in consumers[q.output[0]]] != [dq.name]:
                continue
            source = q.input[0]
            for node in graph.node:
                for i, name in enumerate(node.input):
                    if name == dq.output[0]:
                        node.input[i] = source
            for out in graph.output:
                if out.name == dq.output[0]:
                    raise RuntimeError('graph output is produced by a Q/DQ pair: ' + out.name)
            graph.node.remove(dq)
            graph.node.remove(q)
            removed.append(dict(quantize=q.name, dequantize=dq.name))
            changed = True
            break
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.receipt.exists():
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')

    model = onnx.load(str(args.model))
    before = collections.Counter(n.op_type for n in model.graph.node)
    removed = strip_qdq(model)
    after = collections.Counter(n.op_type for n in model.graph.node)
    leftover = after.get('QuantizeLinear', 0) + after.get('DequantizeLinear', 0)
    if leftover:
        raise RuntimeError('%d Q/DQ nodes could not be collapsed' % leftover)
    onnx.checker.check_model(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))

    record = dict(
        status='derived_variant', recipe='float_from_qdq', is_registered_candidate=False,
        is_original_float_parent=False,
        source_model=str(args.model),
        source_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        variant_model=str(args.out),
        variant_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        qdq_pairs_removed=len(removed),
        ops_before=dict(before), ops_after=dict(after),
        semantics='float arithmetic over weights already rounded onto the quantization grid',
        caveat='NOT the original float parent; the bundle float ONNX is absent from this '
               'device and re-export under device torch 1.13 is disallowed. Use only as the '
               'FP32/FP16 arm of a precision ablation on identical weights.')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('ops_before', 'ops_after')},
                     indent=2))
    print('  ops before:', dict(before))
    print('  ops after :', dict(after))
    return 0


if __name__ == '__main__':
    sys.exit(main())
