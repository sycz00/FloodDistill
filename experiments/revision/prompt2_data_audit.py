"""CPU-only membership, reference identity and footprint evidence for prompt2.

Reads native raster headers and small reference masks, never launches inference.
Outputs are separate from immutable experiment inputs and historical ledgers.
"""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import itertools
import json
import os

import numpy as np
import pyproj
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.strtree import STRtree

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'experiments/revision'
S11 = Path('/data1/schmalstieg/flooddistill/data/sen1floods11/v1.1')
STURM = Path('/data1/schmalstieg/flooddistill/data/STURM-Flood/Dataset')
WF = Path('/data1/schmalstieg/flooddistill/data/WorldFloodsv2/test')
RECEIPT = Path('/data1/schmalstieg/flooddistill/revision_20261005/regen_N2500_approved_20261005/data/dataset_receipt.json')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def identity(path, content=True):
    path = Path(path)
    st = path.stat()
    d = {'path': str(path), 'resolved_path': str(path.resolve()), 'bytes': st.st_size,
         'mtime_ns': st.st_mtime_ns}
    if content:
        d['sha256'] = sha(path)
    return d


def grid(path):
    with rasterio.open(path) as r:
        points = [r.transform * p for p in [(0, 0), (r.width, 0),
                                           (r.width, r.height), (0, r.height)]]
        poly = Polygon(points)
        wgs = transform(pyproj.Transformer.from_crs(r.crs, 'EPSG:4326', always_xy=True).transform, poly)
        area = transform(pyproj.Transformer.from_crs(r.crs, 'EPSG:6933', always_xy=True).transform, poly)
        return {'crs': str(r.crs), 'width': r.width, 'height': r.height,
                'affine': list(r.transform)[:6], 'bounds_wgs84': list(wgs.bounds),
                'footprint_wgs84': list(wgs.exterior.coords), 'band_descriptions': list(r.descriptions),
                'nodata': r.nodata if r.nodata is None or np.isfinite(r.nodata) else 'NaN',
                'native_res': list(r.res)}, area


def aligned(a, b):
    return (all(a[k] == b[k] for k in ['crs', 'width', 'height'])
            and np.allclose(a['affine'], b['affine'], rtol=1e-12, atol=1e-12))


def pixels(path, mapper):
    with rasterio.open(path) as r:
        raw = r.read(1).astype(np.int64)
        valid_read = r.read_masks(1) != 0
    mapped = mapper(raw)
    # S11 uses masked rioxarray read; its -1 replacement preserves signed labels.
    if mapper is s11map:
        mapped[~valid_read] = -1
    counts = {str(k): int(v) for k, v in zip(*np.unique(raw, return_counts=True))}
    return {'raw_class_counts': counts, 'valid_native': int((mapped != -1).sum()),
            'reference_water_native': int((mapped == 1).sum()),
            'ignored_native': int((mapped == -1).sum()),
            'mapped_native_sha256': hashlib.sha256(mapped.astype('<i8').tobytes()).hexdigest()}


def s11map(raw):
    return np.where(raw == 0, 0, np.where(raw == 1, 1, -1)).astype(np.int64)


def sturmmap(raw):
    return np.where(raw == 0, 0, np.where((raw == 1) | (raw == 2), 1, -1)).astype(np.int64)


def overlaps(left, right, same=False):
    geoms = [r['_geom'] for r in right]
    tree = STRtree(geoms)
    pairs = []
    for i, l in enumerate(left):
        for j in tree.query(l['_geom'], predicate='intersects'):
            j = int(j)
            if same and j <= i:
                continue
            r = right[j]
            inter = l['_geom'].intersection(r['_geom']).area
            if inter <= 1.0:
                continue
            ld, rd = l.get('date'), r.get('date')
            delta = None
            if ld and rd:
                dtl = datetime.datetime.fromisoformat(ld).replace(tzinfo=datetime.timezone.utc)
                dtr = datetime.datetime.fromisoformat(rd).replace(tzinfo=datetime.timezone.utc)
                delta = abs((dtl - dtr).total_seconds()) / 86400
            pairs.append({'left': l['id'], 'right': r['id'], 'left_event': l.get('event'),
                          'right_event': r.get('event'), 'intersection_m2': inter,
                          'fraction_left': inter / l['_geom'].area,
                          'fraction_right': inter / r['_geom'].area,
                          'acquisition_delta_days': delta, 'left_date': ld, 'right_date': rd})
    return {'n_pairs': len(pairs), 'n_left': len({x['left'] for x in pairs}),
            'n_right': len({x['right'] for x in pairs}), 'pairs': pairs}


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    refs = {'created_utc': now, 'scope': 'CPU reference/hash/header audit; no model predictions',
            'reader_hashes': {p: identity(REPO / p) for p in
                              ['data/sen1floods11.py', 'data/ood_datasets.py',
                               'data/channel_adapter.py', 'evaluation/perscene_eval.py',
                               'evaluation/revision_eval.py', 'utils/metrics.py']},
            'S11': {}, 'STURM': {}, 'WorldFloods': {}}
    groups = {}
    membership = {}
    # The release lacks S11_Metadata.geojson locally, so no acquisition date is inferred.
    for name in ['train', 'valid', 'test', 'bolivia']:
        p = S11 / f'splits/flood_handlabeled/flood_{name}_data.txt'
        ids = [s.strip() for s in p.read_text().splitlines() if s.strip()]
        membership[name] = {'manifest': identity(p), 'ids': ids, 'count': len(ids),
                            'duplicate_ids': [k for k, n in collections.Counter(ids).items() if n > 1]}
        records = []
        for chip in ids:
            image = S11 / f'data/flood_events/HandLabeled/S2Hand/{chip}_S2Hand.tif'
            label = S11 / f'data/flood_events/HandLabeled/LabelHand/{chip}_LabelHand.tif'
            g, geom = grid(image)
            lg, _ = grid(label)
            rec = {'id': chip, 'filename': image.name, 'event': chip.split('_')[0],
                   'split': name, 'image': identity(image, False), 'reference': identity(label),
                   'input_grid': g, 'reference_grid': lg, 'date': None,
                   'grids_match': all(g[k] == lg[k] for k in ['crs', 'width', 'height', 'affine']),
                   'grids_match_within_tolerance': aligned(g, lg),
                   **pixels(label, s11map)}
            refs['S11'][chip] = rec.copy()
            rec['_geom'] = geom
            records.append(rec)
        groups['s11_' + name] = records
    membership['intersections'] = {a + '__' + b: sorted(set(membership[a]['ids']) & set(membership[b]['ids']))
                                    for a, b in itertools.combinations(['train', 'valid', 'test', 'bolivia'], 2)}
    assigned = set().union(*(set(membership[n]['ids']) for n in ['train', 'valid', 'test', 'bolivia']))
    disk = {p.name.removesuffix('_S2Hand.tif') for p in (S11 / 'data/flood_events/HandLabeled/S2Hand').glob('*.tif')}
    membership['unassigned_image_ids'] = sorted(disk - assigned)
    membership['missing_image_ids'] = sorted(assigned - disk)
    membership['missing_metadata_geojson'] = not (S11 / 'Sen1Floods11_Metadata.geojson').exists()
    md = list(csv.DictReader(open(STURM / 'Sentinel2_metadata.csv')))
    sturm = []
    for row in md:
        tile = row['tile_id']
        image, label = STURM / 'Sentinel2/S2' / tile, STURM / 'Sentinel2/Floodmaps' / tile
        g, geom = grid(image)
        lg, _ = grid(label)
        rec = {'id': tile, 'event': row['ems_code'], 'map_id': row['floodmap_id'],
               'date': row['sentinel_timestamp'], 'image': identity(image, False),
               'reference': identity(label), 'input_grid': g, 'reference_grid': lg,
               'grids_match': all(g[k] == lg[k] for k in ['crs', 'width', 'height', 'affine']),
               'grids_match_within_tolerance': aligned(g, lg),
               **pixels(label, sturmmap)}
        refs['STURM'][tile] = rec.copy()
        rec['_geom'] = geom
        sturm.append(rec)
    groups['sturm'] = sturm
    wf = []
    for image in sorted((WF / 'S2').glob('*.tif')):
        label, jrc, meta = WF / 'gt' / image.name, WF / 'PERMANENTWATERJRC' / image.name, WF / 'meta' / (image.stem + '.json')
        m = json.load(open(meta)); g, geom = grid(image); lg, _ = grid(label); jg, _ = grid(jrc)
        with rasterio.open(label) as r:
            gt = r.read()
        with rasterio.open(jrc) as r:
            perm = r.read(1)
        valid = ((gt[1] == 1) | (gt[1] == 2)) & (gt[0] != 2) & (perm != 3)
        rec = {'id': image.name, 'event': m.get('ems_code'), 'date': m.get('s2_date', m.get('satellite date')),
               'image': identity(image, False), 'reference': identity(label), 'jrc_mask': identity(jrc),
               'metadata': identity(meta), 'input_grid': g, 'reference_grid': lg, 'jrc_grid': jg,
               'grids_match': all(g[k] == lg[k] == jg[k] for k in ['crs', 'width', 'height', 'affine']),
               'grids_match_within_tolerance': aligned(g, lg) and aligned(g, jg),
               'valid_before_spectral_nodata': int(valid.sum()),
               'reference_water_before_spectral_nodata': int(((gt[1] == 2) & valid).sum()),
               'jrc_class_counts': {str(k): int(v) for k, v in zip(*np.unique(perm, return_counts=True))},
               'jrc_product_year': m.get('permanent_water_image_path', 'unknown'),
               'jrc_product_year_basis': 'WorldFloods release metadata permanent_water_image_path; typically YYYY.tif',
               'spectral_nodata_limit': 'All-13-band-zero mask is applied by reader but not re-read for this reference-only audit'}
        refs['WorldFloods'][image.name] = rec.copy(); rec['_geom'] = geom; wf.append(rec)
    groups['wf'] = wf
    audits = {a + '__' + b: overlaps(groups[a], groups[b]) for a, b in
              itertools.combinations(['s11_train', 's11_valid', 's11_test', 's11_bolivia'], 2)}
    audits['sturm__wf'] = overlaps(sturm, wf)
    audits['sturm_internal'] = overlaps(sturm, sturm, True)
    audits['wf_internal'] = overlaps(wf, wf, True)
    for n in ['s11_train', 's11_valid', 's11_test']:
        for ext in ['sturm', 'wf']:
            audits[n + '__' + ext] = overlaps(groups[n], groups[ext])
    receipt = json.load(open(RECEIPT)); chosen = receipt['files']
    source_rows = {r['local_path']: r for r in csv.DictReader(open(REPO / 'data/flood_unlabeled_manifest.csv'))}
    cohort = [source_rows[r['source']] for r in chosen if r['source'] in source_rows]
    scenes = collections.defaultdict(list); locations = collections.defaultdict(list)
    for r in cohort:
        scenes[r['scene_system_index']].append(r['local_path'])
        locations[(r['event_id'], r['patch_idx'])].append(r['local_path'])
    gdacs = {'frozen_receipt': identity(RECEIPT), 'dataset_sha256': receipt['dataset_sha256'],
             'n_selected': len(chosen), 'original_source_overlap': receipt['original_source_overlap'],
             'replacement_count': receipt['fps_replacements'], 'original_byte_reproduction': False,
             'n_sources_with_metadata': len(cohort), 'raw_root_exists': Path('/home/schmalstieg/data/flood_unlabeled').exists(),
             'footprint_coverage': 0, 'n_acquisitions': len(scenes), 'n_events': len({r['event_id'] for r in cohort}),
             'repeated_acquisitions': {k:v for k,v in scenes.items() if len(v) > 1},
             'paired_location_acquisitions': {'|'.join(k):v for k,v in locations.items() if len(v) > 1},
             'limitation': 'Source centroid/acquisition identities available; raw rasters missing and resized .pt has no geotransform. Nominal 2.6km extent is not verified native footprint geometry.'}
    result = {'created_utc': now, 'membership': membership, 'footprint_checks': audits,
              'gdacs': gdacs, 'coverage': {k: len(v) for k,v in groups.items()},
              'method': 'Affine raster corner polygon -> EPSG6933 equal-area; STRtree intersections >1m2. Footprints bound imagery, not only scored pixels. Dates come from supplied metadata only.',
              'limitations': ['No exhaustive image-content hashing; optical identities use paths/sizes/mtimes.',
                              'Reference hashes identify original file and native mapped bitmap, not archived resized scoring-mask bitmap.',
                              'No GDACS native footprint coverage; no S11 acquisition metadata available.',
                              'Official S11 split and shared location names may entail dependence even where footprints are disjoint.',
                              'No overlap found by these checks is not proof of absence of historical test-guided choices.']}
    for name, obj in [('prompt2_reference_manifest.json', refs), ('prompt2_data_checks.json', result)]:
        p = OUT / name
        tmp = p.with_suffix('.json.tmp')
        with tmp.open('w') as f:
            json.dump(obj, f, indent=2, allow_nan=False)
        os.replace(tmp, p)
    print(json.dumps({'membership': {n: membership[n]['count'] for n in ['train','valid','test','bolivia']},
                      'footprint_intersections': {k:v['n_pairs'] for k,v in audits.items()},
                      'gdacs_metadata_coverage':len(cohort), 'reference_records':sum(len(refs[k]) for k in ['S11','STURM','WorldFloods'])}, indent=2))


if __name__ == '__main__':
    main()
