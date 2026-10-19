#!/usr/bin/env python3
"""Rebuild the fixed Sen1Floods11 validation assets consumed by integer_tools.py.

The workstation bundle shipped pre-normalized `source_val/*.npz`. When that bundle
is absent, this regenerates byte-equivalent assets on the device from the official
loader so the evaluation semantics (band selection, scale, normalization, ignore
mask, 512-pixel geometry) stay identical to data/sen1floods11_nodep.py.

All 89 hand-labeled validation scenes are natively 13x512x512, so the official
eval A.Resize(512, 512) is an identity op and no interpolation is introduced here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# torch first: libgomp static-TLS guard documented in jetson_repro/README.md
import torch  # noqa: F401
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sen1floods11_nodep import NORM_MEAN, NORM_STD, Sen1Floods11Dataset  # noqa: E402

EXPECTED_SCENES = {'val': 89, 'test': 90, 'bolivia': 15}

# Sen1Floods11 hand-labeled split files. `bolivia` is the held-out-country split and is
# shipped as a CSV of S1Hand/LabelHand pairs rather than a plain id list.
SPLIT_FILES = {'train': 'flood_train_data.txt', 'val': 'flood_valid_data.txt',
               'test': 'flood_test_data.txt', 'bolivia': 'flood_bolivia_data.csv'}


def scene_ids(data_root, split):
    path = Path(data_root) / 'v1.1' / 'splits' / 'flood_handlabeled' / SPLIT_FILES[split]
    ids = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split(',')[0]
        for suffix in ('_S1Hand.tif', '_S2Hand.tif', '_LabelHand.tif'):
            if token.endswith(suffix):
                token = token[:-len(suffix)]
        ids.add(token)
    return ids


def retarget(dataset, data_root, split):
    """Point an official dataset instance at another split's scenes.

    Only the file lists change; `__getitem__` (band selection, constant scale,
    normalization, ignore-mask handling) is the official code path untouched.
    """
    import glob as _glob
    import os as _os
    root = Path(data_root) / 'v1.1' / 'data' / 'flood_events' / 'HandLabeled'
    ids = scene_ids(data_root, split)
    images = sorted(p for p in _glob.glob(str(root / 'S2Hand' / '*_S2Hand.tif'))
                    if _os.path.basename(p).replace('_S2Hand.tif', '') in ids)
    dataset.image_files = images
    dataset.label_files = [str(root / 'LabelHand' /
                               _os.path.basename(p).replace('_S2Hand.tif', '_LabelHand.tif'))
                           for p in images]
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test', 'bolivia'])
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()

    if args.receipt.exists():
        parser.error('Output already exists; use a distinct attempt directory to preserve evidence')

    transform = A.Compose([
        A.Resize(args.size, args.size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])
    base_split = 'val' if args.split in ('val', 'bolivia') else args.split
    dataset = Sen1Floods11Dataset(args.data_root, split=base_split, transform=transform)
    if args.split != base_split:
        dataset = retarget(dataset, args.data_root, args.split)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        name = item['filename']
        event = name.split('_')[0]
        image = item['image'].numpy().astype(np.float32)
        mask = item['mask'].numpy().astype(np.int32)
        if image.shape != (6, args.size, args.size) or mask.shape != (args.size, args.size):
            raise RuntimeError('Unexpected geometry for ' + name)
        if not np.isfinite(image).all():
            raise RuntimeError('Non-finite normalized input for ' + name)
        if not set(np.unique(mask).tolist()) <= {-1, 0, 1}:
            raise RuntimeError('Unexpected label values for ' + name)
        target = args.out / (name.replace('_S2Hand.tif', '') + '.npz')
        np.savez(target, image=image, mask=mask, event=event, filename=name)
        rows.append(dict(file=target.name, event=event, source=name,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         labeled_pixels=int((mask >= 0).sum()),
                         water_pixels=int((mask == 1).sum())))

    expected = EXPECTED_SCENES.get(args.split)
    if expected is not None and len(rows) != expected:
        raise RuntimeError('Expected %d scenes for split %s, wrote %d'
                           % (expected, args.split, len(rows)))

    events = sorted({r['event'] for r in rows})
    record = dict(status='prepared', split=args.split, scenes=len(rows), size=args.size,
                  events=events, event_count=len(events), data_root=args.data_root,
                  norm_mean=NORM_MEAN, norm_std=NORM_STD,
                  provenance='regenerated on device from official loader; workstation bundle absent',
                  resize_is_identity=True, rows=rows)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'rows'}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
