"""Numerical STURM input checks tolerate OpenCV interpolation roundoff."""
import json
import hashlib
import math
import numpy as np
from pathlib import Path

NUMERICAL = 'sturm_bilinear_normalized_atol_1e-4'
ATOL = 1e-4


def bilinear64(image, height=512, width=512):
    """Half-pixel bilinear interpolation with clamped borders, independent of OpenCV."""
    image = np.asarray(image, np.float64)
    _, h, w = image.shape
    x = np.clip((np.arange(width, dtype=np.float64) + .5) * w / width - .5, 0, w - 1)
    y = np.clip((np.arange(height, dtype=np.float64) + .5) * h / height - .5, 0, h - 1)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = x - ix, y - iy
    horizontal = image[:, :, ix] * (1 - wx) + image[:, :, np.minimum(ix + 1, w - 1)] * wx
    return horizontal[:, iy, :] * (1 - wy)[None, :, None] + horizontal[:, np.minimum(iy + 1, h - 1), :] * wy[None, :, None]


def sturm_reference(path):
    import rasterio
    from data.sen1floods11_nodep import NORM_MEAN, NORM_STD
    with rasterio.open(path) as source:
        image = source.read([1, 2, 3, 7, 8, 9]).astype(np.float32)
    image = np.nan_to_num(image, nan=0., posinf=0., neginf=0.)
    image = np.clip(np.float32(.04) + np.float32(.12) * image, 0., 1.)
    reference = bilinear64(image)
    mean = np.asarray(NORM_MEAN, np.float32).astype(np.float64)[:, None, None]
    std = np.asarray(NORM_STD, np.float32).astype(np.float64)[:, None, None]
    return (reference - mean) / std


def numerical_sturm(record, image, reference):
    image, reference = np.asarray(image), np.asarray(reference)
    if image.shape != (6, 512, 512) or reference.shape != image.shape or not np.isfinite(image).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid STURM input shape or nonfinite values')
    error = float(np.max(np.abs(image.astype(np.float64) - reference)))
    if error > ATOL:
        raise ValueError('STURM numerical preprocessing error exceeds1e-4: ' + record['name'] + ' max_abs=' + str(error))
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    return dict(record, input_sha256=digest, canonical_input_sha256=record['input_sha256'],
                input_profile=NUMERICAL, input_max_abs_error=error, input_atol=ATOL)



def identify(root, dataset, record, digest):
    if digest == record['input_sha256']:
        return dict(record, input_profile='canonical_workstation')
    raise ValueError('Unrecognised input bytes ' + dataset + ' ' + record['name'])


def validate_input(root, dataset, row, record):
    if row.get('mask_sha256') != record['mask_sha256']:
        raise ValueError('Wrong mask hash ' + record['name'])
    if dataset == 'sturm512':
        # Hashes are provenance only. Older completed rows retain their original
        # preprocessing receipts; current rows carry the measured numerical bound.
        if row.get('input_profile') == NUMERICAL:
            error = row.get('input_max_abs_error')
            if not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0 or error > ATOL or row.get('input_atol') != ATOL:
                raise ValueError('Missing or excessive STURM numerical preprocessing error')
        return
    identity = identify(root, dataset, record, row.get('input_sha256'))
