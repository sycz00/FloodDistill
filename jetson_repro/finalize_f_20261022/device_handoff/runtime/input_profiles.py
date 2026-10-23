"""Admit exact received Jetson STURM bytes, without relaxing other input checks."""
import json
from pathlib import Path
from functools import lru_cache

JETSON = 'jetson_opencv454_received_20261022'


@lru_cache(maxsize=8)
def device_hashes(root):
    root = Path(root)
    path = root / 'inventories/sturm512_jetson_input_hashes.jsonl'
    canonical = {r['name']: r for r in (json.loads(line) for line in (root / 'inventories/sturm512.jsonl').read_text().splitlines())}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {}
    if len(rows) != 2675 or len(canonical) != 2675:
        raise ValueError('Require all2675 exact STURM device hashes')
    for row in rows:
        name = row['name']
        if name not in canonical or name in result or row['event'] != canonical[name]['event']:
            raise ValueError('Wrong or duplicate STURM device identity')
        if not all(row.get(k) is True for k in ('mask_ok', 'valid_ok', 'water_ok', 'shape_ok', 'finite')):
            raise ValueError('Received STURM mask/support evidence failed')
        digest = row['device_input_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid received STURM input hash')
        result[name] = digest
    if set(result) != set(canonical):
        raise ValueError('Incomplete STURM device input profile')
    return result


def identify(root, dataset, record, digest):
    if digest == record['input_sha256']:
        return dict(record, input_profile='canonical_workstation')
    if dataset == 'sturm512' and device_hashes(str(root))[record['name']] == digest:
        return dict(record, input_sha256=digest, canonical_input_sha256=record['input_sha256'], input_profile=JETSON)
    raise ValueError('Unrecognised input bytes ' + dataset + ' ' + record['name'])


def validate_input(root, dataset, row, record):
    if row.get('mask_sha256') != record['mask_sha256']:
        raise ValueError('Wrong mask hash ' + record['name'])
    identity = identify(root, dataset, record, row.get('input_sha256'))
    if identity['input_profile'] == JETSON:
        if row.get('input_profile') != JETSON or row.get('canonical_input_sha256') != record['input_sha256']:
            raise ValueError('Missing explicit Jetson input-profile provenance')
