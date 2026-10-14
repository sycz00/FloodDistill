"""Bounded export diagnostics/recovery from frozen checkpoints; no data or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from experiments.revision_queue import atomic_json, file_hash
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat, quantization_mode
from evaluation.compression_artifact import measure_artifact, _probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('diagnose', 'repair'), required=True)
    parser.add_argument('--terminal-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    receipt = json.loads(args.terminal_receipt.read_text())
    path = receipt['checkpoint_path']
    assert file_hash(path) == receipt['checkpoint_sha256']
    payload = torch.load(path, map_location='cpu', weights_only=True)
    graph = build_compression_student(receipt['graph'], pretrained=False)
    quantized = receipt['status_label'] == 'simulation_only'
    if quantized:
        graph = load_compression_qat(graph, payload['model'])
    else:
        graph.load_state_dict(payload['model'], strict=True)
    graph.cuda().eval()
    result = {'status': 'complete', 'outcome': 'completed',
              'terminal_receipt': str(args.terminal_receipt.resolve()),
              'terminal_receipt_sha256': file_hash(args.terminal_receipt),
              'checkpoint_path': path, 'checkpoint_sha256': receipt['checkpoint_sha256'],
              'graph': receipt['graph'], 'precision': receipt['precision'],
              'method': receipt['method'], 'new_training': False,
              'new_benchmark_evaluation': False, 'synthetic_export_probes_only': True}
    if args.action == 'diagnose':
        probes = []
        with torch.no_grad(), quantization_mode(graph, True):
            image = torch.zeros(1, 6, 512, 512, device='cuda')
            expected = graph(image)
            for optimized in (True, False):
                with torch.jit.optimized_execution(optimized):
                    loaded = torch.jit.load(receipt['artifact']['artifact']['unvalidated_file_path'],
                                           map_location='cuda').eval()
                    # Match original validation history, then the timed runtime
                    # warmup, so a cold profiling plan cannot hide fusion errors.
                    loaded(_probe(224, image.device, image.dtype))
                    sample = _probe(512, image.device, image.dtype)
                    loaded(sample)
                    for _ in range(20):
                        loaded(sample)
                    actual = loaded(image)
                    probes.append({'optimized_execution': optimized,
                        'max_absolute_logit_difference': float((expected - actual).abs().max()),
                        'allclose_original_tolerance': bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)),
                        'argmax_agreement': float((actual.argmax(1) == expected.argmax(1)).float().mean()),
                        'finite': bool(torch.isfinite(actual).all())})
        result['probes'] = probes
    else:
        assert args.artifact_dir is not None
        # Keep the original strict tolerances and exact argmax checks. Record
        # the execution setting in the measured backend rather than weakening
        # validation to accept an optimized graph with different numerics.
        with torch.jit.optimized_execution(False):
            artifact = measure_artifact(graph, args.artifact_dir, receipt['graph'], quantized)
        artifact['graph_executor_optimization'] = False
        artifact['runtime']['graph_executor_optimization'] = False
        if artifact['runtime'].get('status') == 'measured':
            artifact['runtime']['backend'] += ' (graph executor optimization disabled)'
        result['artifact'] = artifact
    result['elapsed_seconds'] = time.monotonic() - start
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in ('status', 'graph', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
