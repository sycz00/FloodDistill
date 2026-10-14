"""Extend existing centroid screen without asserting missing GDACS footprints."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/revision'


def xyz(lon, lat):
    a, b = np.radians(lon), np.radians(lat)
    return [np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)]


def date(s):
    return datetime.datetime.strptime(s[:15], '%Y%m%dT%H%M%S').replace(tzinfo=datetime.timezone.utc)


features = list(csv.DictReader(open(ROOT / 'data/index_files_curations/pool_features.csv')))
loc = {r['location']: r for r in features if r['lon'] and r['lat']}
shipped = {r['file']: r for r in csv.DictReader(open(ROOT / 'data/index_files_curations/source_manifest.csv'))}
idx = json.load(open(ROOT / 'data/index_files_curations/diversity_N2500_index.json'))
original = [shipped[n]['source'] for n in idx['files']]
receipt_path = Path('/data1/schmalstieg/flooddistill/revision_20261005/regen_N2500_approved_20261005/data/dataset_receipt.json')
receipt = json.load(open(receipt_path))
regenerated = [r['source'] for r in receipt['files']]
refs = json.load(open(BASE / 'prompt2_reference_manifest.json'))
external = {'sturm': [{'id':k, 'event':v['event'], 'date':v['date'],
                       'xyz':xyz(np.mean([p[0] for p in v['input_grid']['footprint_wgs84'][:-1]]),
                                 np.mean([p[1] for p in v['input_grid']['footprint_wgs84'][:-1]]))}
                      for k,v in refs['STURM'].items()],
            'wf': [{'id':k, 'event':v['event'], 'date':v['date'],
                    'xyz':xyz(np.mean([p[0] for p in v['input_grid']['footprint_wgs84'][:-1]]),
                              np.mean([p[1] for p in v['input_grid']['footprint_wgs84'][:-1]]))}
                   for k,v in refs['WorldFloods'].items()]}
result = {'scope':'Centroid screen, separate from actual footprint audit',
          'source_fingerprints': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [ROOT / 'data/index_files_curations/pool_features.csv',
                                   ROOT / 'data/index_files_curations/source_manifest.csv',
                                   ROOT / 'data/index_files_curations/diversity_N2500_index.json', receipt_path]},
          'cohorts': {},
          'limitations': ['Coordinates copied from event-acquisition descriptor to paired source sharing location ID where direct coordinate is absent.',
                          'Acquisition timestamp parsed from Sentinel source system index; date separation is inferred from this metadata.',
                          'These are centroids, not footprint intersection or exact image-content comparisons.',
                          'Raw GDACS rasters and their footprints are unavailable; test-time shared ground cannot be exhaustively excluded.']}
for name, sources in [('archived_original_N2500',original),('regenerated_approved_N2500',regenerated)]:
    points=[]; absent=[]
    acquisitions=collections.defaultdict(list); locations=collections.defaultdict(list)
    for s in sources:
        l, scene = s.split('/')
        acquisitions[scene].append(s); locations[l].append(s)
        if l not in loc:
            absent.append(s); continue
        r=loc[l]
        points.append({'source':s,'location':l,'event':l.rsplit('_',1)[0], 'date':date(scene),
                       'xyz':xyz(float(r['lon']),float(r['lat']))})
    tree=cKDTree([r['xyz'] for r in points]); screens={}
    for ds, cells in external.items():
        for km,days in [(25,30),(5,2)]:
            pairs=[]
            for e in cells:
                ed=datetime.datetime.fromisoformat(e['date']).replace(tzinfo=datetime.timezone.utc)
                for j in tree.query_ball_point(e['xyz'],2*np.sin(km/(2*6371.0088))):
                    p=points[j];dd=abs((p['date']-ed).total_seconds())/86400
                    if dd>days:continue
                    chord=np.linalg.norm(np.array(p['xyz'])-e['xyz'])
                    pairs.append({'benchmark_id':e['id'],'benchmark_event':e['event'],
                                  'training_source':p['source'],'training_event':p['event'],
                                  'distance_km':float(2*6371.0088*np.arcsin(min(1,chord/2))),
                                  'acquisition_delta_days':dd})
            screens[f'{ds}_{km}km_{days}days']={'n_pairs':len(pairs),
                   'n_benchmark_ids':len({r['benchmark_id'] for r in pairs}),
                   'events':sorted({r['benchmark_event'] for r in pairs}),'pairs':pairs}
    result['cohorts'][name]={'n_sources':len(sources),'n_sources_with_centroid':len(points),
                            'missing_centroid_sources':absent, 'n_acquisitions':len(acquisitions),
                            'n_events':len({s.split('/')[0].rsplit('_',1)[0] for s in sources}),
                            'repeated_acquisitions':{k:v for k,v in acquisitions.items() if len(v)>1},
                            'paired_location_acquisitions':{k:v for k,v in locations.items() if len(v)>1},
                            'screens':screens}
out=BASE/'prompt2_geographic_screen.json';out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:{'centroid_coverage':v['n_sources_with_centroid'],
                      'screens':{a:b['n_benchmark_ids'] for a,b in v['screens'].items()}}
                  for k,v in result['cohorts'].items()},indent=2))
