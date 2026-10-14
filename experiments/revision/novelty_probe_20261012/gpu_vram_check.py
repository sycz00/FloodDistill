#!/usr/bin/env python3
"""Measure peak GPU memory of the V9 job types on ONE idle GPU (seconds of work).

Refuses to run if the selected GPU has any foreign compute process or >5 %
utilization. Writes gpu_vram_check.json with peak allocated/reserved memory,
the process memory seen by nvidia-smi (includes the CUDA context) and the
wall time, so the run can be charged to the V9 GPU ledger.

Run: CUDA_VISIBLE_DEVICES=<GPU-UUID> /home/schmalstieg/miniconda3/envs/esi/bin/python gpu_vram_check.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V7 = HERE.parent / 'capacity_v7_20261011T210730Z'
SPEC = json.loads((V7 / 'training_spec.json').read_text())

UUID = os.environ.get('CUDA_VISIBLE_DEVICES', '')
assert UUID.startswith('GPU-'), 'select the GPU by UUID'


def smi(query, kind='gpu'):
    flag = '--query-gpu' if kind == 'gpu' else '--query-compute-apps'
    out = subprocess.run(['nvidia-smi', f'{flag}={query}', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True, timeout=10).stdout
    return [line.split(', ') for line in out.strip().splitlines() if line.strip()]


def own_process_mib():
    for uuid, pid, mem in smi('gpu_uuid,pid,used_memory', 'apps'):
        if uuid == UUID and int(pid) == os.getpid():
            return int(mem)
    return None


def main():
    for uuid, util, used in smi('uuid,utilization.gpu,memory.used'):
        if uuid == UUID:
            assert int(util) < 5, f'GPU busy: {util}%'
    foreign = [r for r in smi('gpu_uuid,pid,used_memory', 'apps') if r[0] == UUID]
    assert not foreign, f'foreign compute process present: {foreign}'
    sys.dont_write_bytecode = True
    spec = SPEC
    sys.path.insert(0, spec['capsule']); sys.path.insert(0, str(V7))
    import torch
    from torch.nn import functional as F
    from train_v7 import build_model
    from models.student import build_student
    import scripts.compression_jobs as cj
    torch.backends.cudnn.benchmark = False
    dev = torch.device('cuda')
    results = {}
    t_all = time.monotonic()

    def measure(name, fn):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic(); fn(); torch.cuda.synchronize()
        results[name] = dict(peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                             peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
                             process_mib_nvidia_smi=own_process_mib(), seconds=round(time.monotonic() - t0, 2))
        print(name, results[name], flush=True)

    def train(model, size, batch, steps=4):
        model.train().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        for _ in range(steps):
            x = torch.randn(batch, 6, size, size, device=dev); y = torch.randint(0, 2, (batch, size, size), device=dev)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.to('cpu'); del opt

    def infer(model, size, batch, steps=3):
        model.eval().to(dev)
        with torch.no_grad():
            for _ in range(steps):
                model(torch.randn(batch, 6, size, size, device=dev))
        model.to('cpu')

    r4 = build_model('R4', 'lkd', spec['rungs_dir'])
    measure('R4_train_CE_b16_224', lambda: train(r4, 224, 16))
    measure('R4_infer_b8_512', lambda: infer(r4, 512, 8))
    b0 = build_student('efficientvit_b0', pretrained=False)
    measure('B0_train_CE_b16_224', lambda: train(b0, 224, 16))
    measure('B0_infer_b8_512', lambda: infer(b0, 512, 8))
    teacher = cj.teacher(spec)          # loads to cuda inside the helper
    teacher.to('cpu'); torch.cuda.empty_cache()
    measure('Prithvi_teacher_infer_b4_512', lambda: infer(teacher, 512, 4))
    measure('Prithvi_teacher_infer_b16_224', lambda: infer(teacher, 224, 16))
    total = time.monotonic() - t_all
    out = dict(gpu_uuid=UUID, measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), results=results,
               total_wall_seconds=round(total, 1), charge_to_v9_ledger_gpu_hours=round(total / 3600, 4),
               note='random inputs; peak memory, not accuracy; includes model transfer; context overhead in nvidia-smi column')
    (HERE / 'gpu_vram_check.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
