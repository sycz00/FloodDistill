#!/usr/bin/env python3
"""Derive a TensorRT-8.4-parsable variant of a QDQ ONNX candidate.

This is a SEPARATE graph/precision recipe, not an amendment of the deployment
candidate. TensorRT 8.4 has no cubic Resize implementation, so the EfficientViT
SegHead `input_ops` bicubic upsampling (upstream default, carried by the
registered architecture) cannot be parsed at all. The cubic -> linear
substitution reuses `jetson/fix_qdq_onnx.py::patch_cubic_resize`, the repair this
repository already documents for the TRT 8.4 device path.

The substitution changes the registered model and must be source-validated on CPU
against the unpatched candidate before any device number is attributed to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent / 'jetson'))
from fix_qdq_onnx import patch_cubic_resize  # noqa: E402


def resize_modes(model):
    out = []
    for node in model.graph.node:
        if node.op_type != 'Resize':
            continue
        mode = next((a.s.decode() for a in node.attribute if a.name == 'mode'), None)
        out.append(dict(node=node.name, mode=mode))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--resize-mode', choices=['linear', 'nearest'], default='linear',
                        help='Target Resize mode. "linear" is the documented TRT 8.4 cubic '
                             'repair. "nearest" is a further, more invasive recipe used to '
                             'test whether TensorRT 8.4 can place Resize in INT8 at all; it '
                             'changes the registered interpolation much more than linear does.')
    args = parser.parse_args()
    if args.out.exists() or args.receipt.exists():
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')

    source = onnx.load(str(args.model))
    before = resize_modes(source)
    patched = patch_cubic_resize(source)
    if args.resize_mode == 'nearest':
        for node in patched.graph.node:
            if node.op_type != 'Resize':
                continue
            for attr in node.attribute:
                if attr.name == 'mode':
                    attr.s = b'nearest'
    after = resize_modes(patched)
    onnx.checker.check_model(patched)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(patched, str(args.out))

    changed = [b['node'] for b, a in zip(before, after) if b['mode'] != a['mode']]
    record = dict(
        status='derived_variant',
        recipe='trt84_%s_resize' % args.resize_mode,
        is_registered_candidate=False,
        source_model=str(args.model),
        source_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        variant_model=str(args.out),
        variant_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        resize_modes_before=before, resize_modes_after=after,
        changed_nodes=changed, changed_count=len(changed),
        reason='TensorRT 8.4 ONNX parser rejects mode="cubic" Resize (importResize: '
               '"This version of TensorRT does not support cubic interpolation!")',
        provenance='jetson/fix_qdq_onnx.py::patch_cubic_resize (documented TRT 8.4 repair)',
        caveat='Changes the registered interpolation; requires CPU source validation against '
               'the unpatched candidate. Device numbers from this graph must not be aliased '
               'with the registered candidate.')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
