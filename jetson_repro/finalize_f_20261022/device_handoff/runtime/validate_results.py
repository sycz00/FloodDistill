"""Check raw result identities and full frozen inventory coverage before collection."""
import json
from pathlib import Path
import numpy as np

def validate_rows(root,ds,rows,imported=False):
    inv=[json.loads(line) for line in (Path(root)/'inventories'/(ds+'.jsonl')).read_text().splitlines()]
    expected={r['name']:r for r in inv};seen=set()
    if len(rows)!=len(inv):raise ValueError('Wrong image count '+ds)
    for row in rows:
        name=row['name']
        if ds in ('s11val','s11test','bolivia') and name.endswith('.npz'):name=name[:-4]+'_S2Hand.tif'
        if name not in expected or name in seen:raise ValueError('Unexpected or duplicate '+ds+' '+name)
        rec=expected[name];c=np.asarray(row['counts_tn_fp_fn_tp'])
        if c.shape!=(4,) or not np.issubdtype(c.dtype,np.integer) or (c<0).any():raise ValueError('Invalid confusion counts '+name)
        if row['event']!=rec['event'] or int(c.sum())!=rec['valid_pixels'] or int(c[2]+c[3])!=rec['water_pixels']:raise ValueError('Wrong event/mask support '+name)
        if not imported:
            from input_profiles import validate_input
            validate_input(root,ds,row,rec)
        seen.add(name)
    if seen!=set(expected):raise ValueError('Incomplete inventory '+ds)

def validate_dry(root,rows):
    expected={r['tile_id']:r for r in json.loads((Path(root)/'inventories/dry159_manifest.json').read_text())['rows']};seen=set()
    if len(rows)!=159:raise ValueError('Require all159 dry tiles')
    for row in rows:
        name=row['tile_id']
        if name not in expected or name in seen:raise ValueError('Duplicate/unexpected dry tile '+name)
        rec=expected[name]
        if any(row[k]!=rec[k] for k in ('group','decision','stratum')):raise ValueError('Wrong dry selection metadata '+name)
        for crop,key in [('central264','central_valid_pixels'),('full512','full_valid_pixels')]:
            v=row[crop];nv=v['valid_pixels'];nw=v['false_water_pixels']
            if nv!=rec[key] or not 0<=nw<=nv:raise ValueError('Wrong dry support '+name)
            if nv:
                if abs(v['false_water_fraction']-nw/nv)>1e-12 or v['alarm_gt_1pct']!=(100*nw>nv):raise ValueError('Wrong dry alarm '+name)
            elif v['false_water_fraction'] is not None or v['alarm_gt_1pct'] is not None:raise ValueError('Undefined dry support must remain undefined')
        seen.add(name)

def benchmark_receipts_valid(profile,steady,state):
    if any(v!=0 for v in (state.get('profile_returncode'),state.get('steady_returncode'),profile.get('runA_returncode'),profile.get('runB_returncode'),steady.get('returncode'))):return False
    if steady.get('steady_samples',0)<=0:return False
    values=[profile.get('throughput_qps_runA'),steady.get('throughput_qps'),steady.get('steady_cpu_gpu_cv_mean_w'),steady.get('steady_energy_mJ_per_image')]
    try:
        if not all(np.isfinite(v) and v>0 for v in values):return False
    except TypeError:return False
    return bool(np.isclose(values[3],1000*values[2]/values[1],rtol=1e-6,atol=1e-9))
