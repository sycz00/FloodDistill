"""Verify copied original IDs against their raw scenes and freeze unchanged caches."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.revision_queue import atomic_json, digest, file_hash


def check_overlap(data_root):
    """Actual restored raster footprints; geographic overlap is not test fitting."""
    from datetime import datetime
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform
    from shapely.strtree import STRtree
    source_path = data_root / 'original_source_manifest.json'
    source = json.loads(source_path.read_text())
    assert source['status'] == 'verified'
    reference_path = REPO / 'experiments/revision/prompt2_reference_manifest.json'
    refs = json.loads(reference_path.read_text())
    entries, polygons = [], []
    for dataset in ('S11', 'STURM', 'WorldFloods'):
        for name, row in refs[dataset].items():
            entries.append({'dataset': dataset, 'id': name, 'event': row['event'],
                            'split': row.get('split', 'external'), 'date': row.get('date')})
            polygon = Polygon(row['input_grid']['footprint_wgs84'])
            assert polygon.is_valid
            polygons.append(polygon)
    tree = STRtree(polygons)
    project = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True).transform
    def utc_date(s):
        if not s:
            return None
        value = datetime.fromisoformat(s.replace('Z', '+00:00'))
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    pairs = []
    for row in source['selected_source_rows']:
        polygon = Polygon(row['footprint_wgs84'])
        assert polygon.is_valid
        acquisition = datetime.strptime(row['acquisition'], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
        for index in tree.query(polygon, predicate='intersects'):
            reference = entries[index]
            overlap = polygon.intersection(polygons[index])
            area = transform(project, overlap).area
            if area <= 0:
                continue
            rd = utc_date(reference['date'])
            delta = abs((acquisition - rd).total_seconds()) / 86400 if rd else None
            pairs.append({**reference, 'training_file': row['file'], 'training_source': row['source'],
                          'training_acquisition': acquisition.isoformat(), 'overlap_m2_EPSG6933': area,
                          'fraction_of_training_footprint': area / transform(project, polygon).area,
                          'fraction_of_reference_footprint': area / transform(project, polygons[index]).area,
                          'acquisition_delta_days': delta})
    summary = {}
    for dataset in ('S11', 'STURM', 'WorldFloods'):
        for split in sorted({r['split'] for r in entries if r['dataset'] == dataset}):
            rows = [r for r in pairs if r['dataset'] == dataset and r['split'] == split]
            summary[dataset + ':' + split] = {'footprint_pairs': len(rows),
                'unique_benchmark_ids': len({r['id'] for r in rows}),
                'unique_training_sources': len({r['training_source'] for r in rows}),
                'pairs_within_2_days': sum(r['acquisition_delta_days'] is not None and r['acquisition_delta_days'] <= 2 for r in rows),
                'pairs_within_30_days': sum(r['acquisition_delta_days'] is not None and r['acquisition_delta_days'] <= 30 for r in rows)}
    atomic_json(REPO / 'experiments/revision/prompt4_original_overlap_gate.json', {
        'status': 'audit-complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_manifest': str(source_path), 'source_manifest_sha256': file_hash(source_path),
        'reference_manifest_sha256': file_hash(reference_path),
        'source_sha256': file_hash(Path(__file__)), 'training_footprint_coverage': 2500,
        'summary': summary, 'pairs': pairs,
        'ID_screen': 'Selected source IDs come from copied original source/cache mapping; S11 ID memberships are retained from the unchanged reference manifest',
        'limits': ['Native full raster footprints intersect; this does not establish overlap of scored valid pixels or identical image content.',
                   'Dates use original acquisition IDs and existing benchmark metadata.',
                   'Equal-area polygon area is a geographic screen, not a claim of independent events or exact leakage.',
                   'Restoration and this screen cannot determine historical benchmark influence on method choices.']})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlap_only', action='store_true')
    args = parser.parse_args()
    if args.overlap_only:
        check_overlap(args.out.resolve())
        return
    import numpy as np
    import rasterio
    from rasterio.warp import transform
    import torch
    from data.pseudo_labeled import PseudoLabeledDataset
    from data.unlabeled import FloodUnlabeledDataset
    torch.set_num_threads(2)
    pool = Path('/data1/schmalstieg/flooddistill/data/pseudo_gdacs_v2')
    raw = Path('/data1/schmalstieg/flooddistill/data/flood_unlabeled_v2')
    audit_path = REPO / 'experiments/revision/prompt4_original_data_gate.json'
    prior = json.loads((REPO / 'experiments/revision/protocol_audit.json').read_text())
    selections = {n: pool / f'diversity_N{n}_index.json' for n in (2500, 100)}
    indexes = {n: json.loads(p.read_text()) for n, p in selections.items()}
    names = indexes[2500]['files']
    assert len(names) == len(set(names)) == 2500
    assert len(indexes[100]['files']) == len(set(indexes[100]['files'])) == 100
    assert indexes[100]['files'] == names[:100], 'Original N100 index is not the recorded FPS prefix'
    csv_path = pool / 'source_manifest.csv'
    rows = list(csv.DictReader(csv_path.open()))
    mapping = {r['file']: r for r in rows}
    assert len(mapping) == len(rows), 'Duplicate cache names in original source manifest'
    # A source ID, rather than a numeric filename, chooses the raw loader item.
    loader = FloodUnlabeledDataset(str(raw), img_size=224, normalize=True)
    lookup = {str(p.relative_to(raw)): i for i, p in enumerate(loader.scenes)}
    assert len(lookup) == len(loader.scenes)
    errors, files, source_rows, all_sources = [], [], [], set()
    counts, optional = Counter(), Counter()
    native_nodata = Counter()
    root = args.out.resolve()
    root.mkdir(exist_ok=False, parents=True)
    frozen = root / 'pseudo'
    frozen.mkdir()
    for i, name in enumerate(names):
        row = mapping.get(name)
        path = pool / name
        try:
            assert row and path.is_file(), 'Selected cache or mapping missing'
            source = row['source']
            assert source in lookup and source not in all_sources, 'Missing or duplicate selected source'
            all_sources.add(source)
            before = path.stat()
            item = torch.load(path, map_location='cpu', weights_only=True)
            assert item['source'] == source, 'Cache/source manifest pairing mismatch'
            image, mask = item['image'], item['mask']
            assert image.shape == (6, 224, 224) and image.dtype == torch.float32
            assert mask.shape == (224, 224) and mask.dtype == torch.int64
            assert torch.isfinite(image).all()
            labels, numbers = torch.unique(mask, return_counts=True)
            assert set(labels.tolist()) <= {-1, 0, 1}
            counts.update(dict(zip(labels.tolist(), numbers.tolist())))
            optional.update(k for k in item if k not in {'image', 'mask', 'source'})
            for k in ('confidence', 'soft_logits'):
                if k in item:
                    assert torch.isfinite(item[k]).all(), f'Nonfinite {k}'
            assert 'soft_logits' not in item, 'Recover cached soft-target semantics before launching live-target controls'
            loaded = loader[lookup[source]]['image']
            assert torch.equal(loaded, image), 'Raw six-band loader differs from original cached tensor'
            band_rows = []
            for band in loader._BAND_ORDER:
                bp = raw / source / (band + '.tif')
                stat = bp.stat()
                with rasterio.open(bp) as src:
                    invalid = int(np.count_nonzero(src.read_masks(1) == 0))
                    native_nodata[band] += invalid
                    info = {'band': band, 'path': str(bp), 'sha256': file_hash(bp),
                            'bytes': stat.st_size, 'crs': str(src.crs),
                            'shape': [src.height, src.width], 'nodata': src.nodata,
                            'native_masked_pixels': invalid, 'transform': list(src.transform)}
                    if band == 'B2':
                        corners = [src.transform * p for p in [(0, 0), (src.width, 0),
                                   (src.width, src.height), (0, src.height), (0, 0)]]
                        lon, lat = transform(src.crs, 'EPSG:4326', *zip(*corners))
                        footprint = [list(p) for p in zip(lon, lat)]
                assert bp.stat().st_size == stat.st_size and bp.stat().st_mtime_ns == stat.st_mtime_ns
                band_rows.append(info)
            target = frozen / name
            shutil.copyfile(path, target)
            checksum = file_hash(target)
            assert checksum == file_hash(path)
            assert before.st_size == path.stat().st_size and before.st_mtime_ns == path.stat().st_mtime_ns
            target.chmod(0o444)
            files.append({'name': name, 'sha256': checksum, 'bytes': target.stat().st_size, 'source': source})
            source_rows.append({'file': name, 'source': source, 'source_manifest_row': row,
                                'raw_loader_index_for_verified_source': lookup[source],
                                'raw_to_cache_tensor_bit_equal': True, 'bands': band_rows,
                                'footprint_wgs84': footprint, 'acquisition': source.split('/')[1][:15]})
        except Exception as exc:
            errors.append({'file': name, 'source': row.get('source') if row else None,
                           'error': f'{type(exc).__name__}: {exc}'})
        if (i + 1) % 250 == 0:
            print(f'Original data checked {i + 1}/2500; errors={len(errors)}', flush=True)
    formerly_missing = prior['data_identity']['GDACS']['missing_original_sources']
    verified = {f['name']: f['source'] for f in files}
    restored = [{**r, 'status': 'restored' if verified.get(r['original_id']) == r['source'] else 'still_missing_or_mismatched'}
                for r in formerly_missing]
    historical = prior['data_identity']['GDACS']
    manifest = {'status': 'verified' if not errors else 'blocked', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'namespace': root.parent.name, 'selected_N2500': len(files), 'errors': errors,
                'original_pool': str(pool), 'original_source_root': str(raw),
                'index_sha256': {str(n): file_hash(p) for n, p in selections.items()},
                'index_matches_prior_historical_record': file_hash(selections[2500]) == historical['shipped_index']['sha256'],
                'source_manifest_sha256': file_hash(csv_path),
                'copied_manifest_matches_prior_record': file_hash(csv_path) == historical['shipped_manifest']['sha256'],
                'restoration_counts': dict(Counter(r['status'] for r in restored)), 'formerly_missing_127': restored,
                'label_pixel_counts': {str(k): counts[k] for k in (-1, 0, 1)},
                'optional_cache_key_counts': dict(optional), 'native_band_masked_pixels': dict(native_nodata),
                'raw_to_cache_tensor_checks': 'Every selected cached tensor equals the actual six-band normalized raw loader tensor bit for bit',
                'selected_source_rows': source_rows,
                'cache_semantics': 'Stored normalized six-band float32 image, teacher hard int64 mask, maximum-class float16 confidence; no cached soft logits. Train on stored bytes without another normalization; live teacher soft targets on augmented inputs.',
                'provenance_limits': ['Copied cache bytes now have hashes; no historical per-cache byte-hash receipt was supplied.',
                                      'Hard-cache teacher provenance is user-supplied original data plus existing generation code; generation-time teacher/checkpoint receipt is absent.',
                                      'Class-valid labels do not prove cloud/sensor-valid imagery; native raster validity is recorded separately.',
                                      'Restoration does not establish historical non-use of benchmark scores.'],
                'loader_sha256': {r: file_hash(REPO / r) for r in ['data/unlabeled.py', 'data/pseudo_labeled.py']}}
    atomic_json(root / 'original_source_manifest.json', manifest)
    if errors:
        atomic_json(audit_path, {k: v for k, v in manifest.items() if k != 'selected_source_rows'})
        raise RuntimeError(f'Original data identity blocked on {len(errors)} records; see {audit_path}')
    receipts = {}
    for n in (2500, 100):
        index = root / f'original_N{n}_index.json'
        shutil.copyfile(selections[n], index)
        index.chmod(0o444)
        subset = files[:n]
        identity = {'index_sha256': file_hash(index), 'files': subset}
        receipt = {'status': 'verified', 'dataset_sha256': digest(identity), **identity,
                   'pool': str(frozen), 'index': str(index), 'n_selected': n,
                   'n_distinct_sources': n, 'original_source_overlap': n, 'fps_replacements': 0,
                   'original_index_byte_reproduction': True,
                   'historical_byte_identity': 'Selection index matches supplied and previously recorded original bytes; cache historical hashes unavailable',
                   'hard_masks': 'Supplied original cached masks copied unchanged; no regeneration',
                   'normalization': 'Stored S11-normalized six-band float32 tensors, raw loader equality verified',
                   'source_audit': str(root / 'original_source_manifest.json'),
                   'authorization': 'prompts/prompt4.md original-data reproduction/control matrix'}
        if n == 2500:
            receipt['label_pixel_counts'] = manifest['label_pixel_counts']
        receipt_path = root / f'original_N{n}_receipt.json'
        atomic_json(receipt_path, receipt)
        receipts[n] = receipt_path
    # Exercise the actual augmentations, not a renamed or re-normalized dataset.
    augmentation = {}
    for n in (2500, 100):
        ds = PseudoLabeledDataset(str(frozen), n_samples=n, seed=42, use_curated=True,
            index_file=str(root / f'original_N{n}_index.json'), geom_aug=True, mult_noise=True)
        aug_counts = Counter()
        for i in range(len(ds)):
            item = ds[i]
            assert torch.isfinite(item['image']).all() and item['image'].shape == (6, 224, 224)
            labels, nums = torch.unique(item['mask'], return_counts=True)
            aug_counts.update(dict(zip(labels.tolist(), nums.tolist())))
        augmentation[str(n)] = {'n_checked': len(ds), 'label_counts': {str(k): aug_counts[k] for k in (-1, 0, 1)},
                                'finite_inputs': True, 'padding_introduced': False}
    gate = {k: v for k, v in manifest.items() if k != 'selected_source_rows'}
    gate.update(receipts={str(n): str(p) for n, p in receipts.items()}, augmentation=augmentation,
                data_manifest=str(root / 'original_source_manifest.json'),
                matched_OFA_reuse_mask_equivalent=all(r['label_counts']['-1'] == 0 for r in augmentation.values()),
                N100_identity='Supplied original N100 index, identical first100 of supplied original FPS ordering; execution ancestry remains separately qualified')
    atomic_json(audit_path, gate)
    print(json.dumps({'status': gate['status'], 'selected': len(files), 'restoration': gate['restoration_counts'],
                      'labels': gate['label_pixel_counts'], 'out': str(audit_path)}), flush=True)


if __name__ == '__main__':
    main()
