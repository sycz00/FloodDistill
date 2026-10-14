"""V11 paths and existing durable queue primitives; older runs remain read-only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[2]
V10 = ROOT.parent / 'research_v10_recoveryB_20261014T084000Z'
V9 = ROOT.parent / 'research_v9_recovery_20261013T075000Z'
PYTHON = '/home/schmalstieg/miniconda3/envs/esi/bin/python'
SEEDS = [42, 64, 91, 107]
sys.path.insert(0, str(V9))
import audited_queue as q
save = q.atomic_json
sha = q.file_hash

def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default

def csvout(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({k for row in rows for k in row})
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def setup(threads=None):
    threads = int(threads or os.environ.get('OMP_NUM_THREADS', '4'))
    assert 1 <= threads <= 8
    os.environ['V10_COMPUTE_DEVICE'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(V10 / 'code'))
    import common10
    torch, cj, old, reference = common10.setup(threads)
    # Never assign common10.ROOT or use its mutation-oriented inventory/source helpers.
    return torch, common10, reference

def source_access_guard():
    """Cover derived label caches as well as the original V9 raw-GT guard."""
    windows = read(V9 / 'normal_windows.json')['windows']
    prohibited = {str(Path(r['path']).resolve()) for r in windows if r['dataset'] != 's11val'}
    prohibited.update(str(Path(r['path']).resolve()) for r in read(V9 / 'sealed_windows.json')['rows'])
    roots = (str(ROOT / 'evaluation'), str(V10 / 'metrics'))
    prohibited.add(str(ROOT / 'contrasts.csv'))
    def guard(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        name = str(Path(os.fsdecode(args[0])).resolve())
        if name in prohibited or any(name == p or name.startswith(p + os.sep) for p in roots):
            raise PermissionError('V11 optimization/source selection cannot read evaluation labels/results: ' + name)
    sys.addaudithook(guard)
