"""Static efficiency axis for every graph (CPU): parameters, MACs at 1x6x512x512, theoretical weight bytes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.utils.flop_counter import FlopCounterMode
import rungs

rows = {}
for g in ['A0', 'A1', 'A2', 'X1', 'X2', 'A3', 'A3N', 'R4', 'R3', 'R2', 'R1']:
    m = rungs.build(g).eval()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 6, 512, 512)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    macs = fc.get_total_flops() / 2
    rows[g] = {'parameters': n, 'MACs_512': macs, 'MACs_per_pixel': macs / 512 ** 2,
               'weight_bytes_fp32': 4 * n, 'weight_bytes_int8': n, 'weight_bytes_int4': n / 2,
               'per_pixel_model': rungs.per_pixel_check(rungs.build(g)) if g in ('R1', 'R2') else False}
    print(g, n, round(macs / 512 ** 2, 1))
Path(sys.argv[2]).write_text(json.dumps({'status': 'complete', 'note': 'MACs from torch FlopCounterMode (conv, matmul, '
    'interpolation excluded); theoretical bytes only, not serialized or measured', 'graphs': rows}, indent=1))
