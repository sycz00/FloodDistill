"""Metadata-defined footprint assessment; no model inference or score selection."""
from __future__ import annotations
import json
import hashlib
import math
import sys
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'experiments/revision/prompt5_closure'

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def assess():
    import numpy as np
    import cv2
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds, Window
    from affine import Affine
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform
    from pyproj import Transformer
    # The supplied experimental snapshot defines the readers and masks.
    snapshot = Path('/data1/schmalstieg/flooddistill/revision_20261005/snapshots/95f47d14ae77a2cf')
    sys.path.insert(0,str(snapshot))
    from data.ood_datasets import WorldFloodsDataset
    OUT.mkdir(parents=True,exist_ok=True)
    prior_path=REPO/'experiments/revision/prompt4_original_overlap_gate.json'
    prior=load(prior_path)
    source_path=Path(prior['source_manifest']); source=load(source_path)
    ref_path=REPO/'experiments/revision/prompt2_reference_manifest.json'; refs=load(ref_path)
    gate_path=REPO/'experiments/revision/prompt3_evaluation_gate.json'; gate=load(gate_path)
    source_by_name={r['file']:r for r in source['selected_source_rows']}
    source_order=[r['file'] for r in source['selected_source_rows']]
    targets=[p for p in prior['pairs'] if p['dataset'] in ('STURM','WorldFloods')]
    records=[]
    excluded={'sturm':[], 'wf':[]}
    for pair in targets:
        src=source_by_name[pair['training_file']]
        reference=refs[pair['dataset']][pair['id']]
        b2=next(b for b in src['bands'] if b['band']=='B2')
        with rasterio.open(b2['path']) as ds:
            assert ds.crs.to_string()==b2['crs'] and (ds.height,ds.width)==tuple(b2['shape'])
            assert list(ds.transform)==b2['transform']
            assert sha(b2['path'])==b2['sha256']
        # Densify the actual training patch edges before CRS transformation.
        corners=src['footprint_wgs84']
        coords=[]
        for a,b in zip(corners[:-1],corners[1:]):
            coords.extend([(a[0]+(b[0]-a[0])*t/64,a[1]+(b[1]-a[1])*t/64) for t in range(64)])
        coords.append(coords[0]); polygon=Polygon(coords)
        grid=reference['reference_grid']; crs=grid['crs']; affine=Affine(*grid['affine'][:6])
        projected=transform(Transformer.from_crs('EPSG:4326',crs,always_xy=True).transform,polygon)
        classification='different-date geographic/input support overlap; no duplicate acquisition established'
        common={**pair,'FPS_rank_zero_based':source_order.index(pair['training_file']),
                'N100_current_member':source_order.index(pair['training_file'])<100,
                'N252_current_FPS_prefix_member':source_order.index(pair['training_file'])<252,
                'historical_N252_execution_membership':'not established by current-prefix membership',
                'actual_training_grid_shape':b2['shape'],'actual_training_grid_crs':b2['crs'],
                'training_grid_is_patch_not_whole_Sentinel2_scene':True,
                'support':'Current cached224 input is full B2 native264 patch resized, with whole-array flips/rot90 and per-channel gain; no stochastic spatial crop. Full potentially used patch retained conservatively for historical execution.',
                'classification':classification,'evaluation_label_use':'No evidence of evaluation-ground-truth use in pseudo supervision; exposure is to imagery and teacher-derived targets. Historical benchmark-guided decisions remain an author question.'}
        if pair['dataset']=='STURM':
            with rasterio.open(reference['reference']['path']) as ds:
                assert ds.crs.to_string()==crs and ds.transform==affine
                raw=ds.read(1)
            assert sha(reference['reference']['path'])==reference['reference']['sha256']
            mask=np.where(raw==0,0,np.where((raw==1)|(raw==2),1,-1))
            with rasterio.open(reference['image']['path']) as ds:
                assert ds.crs.to_string()==crs and ds.transform==affine
                finite=np.isfinite(ds.read([1,2,3,7,8,9])).all(axis=0)
            invalid512=cv2.resize((~finite).astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)>0
            mapped512=cv2.resize(mask.astype(np.int32),(512,512),interpolation=cv2.INTER_NEAREST)
            valid512=(mapped512!=-1)&~invalid512
            rec={**common,'map_id':reference['map_id'],'native_valid':int((mask!=-1).sum()),
                 'native_water':int((mask==1).sum()),'corrected512_chip_valid':int(valid512.sum()),
                 'corrected512_chip_water':int(((mapped512==1)&valid512).sum()),
                 'reference_sha256':sha(reference['reference']['path'])}
            for scale,shape,tr,m,v in [(1,mask.shape,affine,mask,mask!=-1),
                                      (4,(512,512),affine*Affine.scale(.25,.25),mapped512,valid512)]:
                for touched in (False,True):
                    overlap=rasterize([(mapping(projected),1)],out_shape=shape,transform=tr,
                                      all_touched=touched,dtype='uint8').astype(bool)
                    suffix=f'{"native" if scale==1 else "corrected512"}_{"touched" if touched else "center"}'
                    rec['intersect_valid_'+suffix]=int((overlap&v).sum())
                    rec['intersect_water_'+suffix]=int((overlap&v&(m==1)).sum())
            if rec['intersect_valid_corrected512_touched']:
                excluded['sturm'].append(pair['id'])
            records.append(rec)
        else:
            w=from_bounds(*projected.bounds,transform=affine)
            c0=max(0,math.floor(w.col_off)); r0=max(0,math.floor(w.row_off))
            c1=min(grid['width'],math.ceil(w.col_off+w.width)); r1=min(grid['height'],math.ceil(w.row_off+w.height))
            box=(c0,r0,c1-c0,r1-r0)
            dummy=SimpleNamespace(exclude_clouds=True,ignore_permanent_water=True,permanent_water_as='ignore',
                jrc_files={pair['id']:Path(reference['jrc_mask']['path'])})
            _, mask=WorldFloodsDataset._getitem_v2(dummy,Path(reference['image']['path']),
                Path(reference['reference']['path']),box)
            retained=np.zeros(mask.shape,dtype=bool)
            window_count=0
            for item in gate['worldfloods']['windows']:
                if item['map']!=pair['id']: continue
                c,r,ww,hh=item['window']; x0=max(c,c0); y0=max(r,r0); x1=min(c+ww,c1); y1=min(r+hh,r1)
                if x0<x1 and y0<y1:
                    retained[y0-r0:y1-r0,x0-c0:x1-c0]=True; window_count+=1
            rec={**common,'window_bbox':list(box),'overlapping_retained_windows':window_count,
                 'reference_sha256':sha(reference['reference']['path']),
                 'JRC_sha256':sha(reference['jrc_mask']['path'])}
            assert rec['reference_sha256']==reference['reference']['sha256']
            assert rec['JRC_sha256']==reference['jrc_mask']['sha256']
            for touched in (False,True):
                overlap=rasterize([(mapping(projected),1)],out_shape=mask.shape,
                    transform=affine*Affine.translation(c0,r0),all_touched=touched,dtype='uint8').astype(bool)
                scored=overlap&retained&(mask!=-1)
                suffix='touched' if touched else 'center'
                rec['intersect_valid_retained_native_'+suffix]=int(scored.sum())
                rec['intersect_water_retained_native_'+suffix]=int((scored&(mask==1)).sum())
            if rec['intersect_valid_retained_native_touched']:
                excluded['wf'].append(pair['id'])
            records.append(rec)
    excluded={k:sorted(set(v)) for k,v in excluded.items()}
    plan={'status':'frozen_before_comparative_score_access','created_utc':datetime.now(timezone.utc).isoformat(),
          'basis':'Original source-footprint metadata plus actual retained/scored mask support, common for all comparators; not chosen by accuracy.',
          'exclude_complete_STURM_chips':excluded['sturm'],'exclude_complete_WorldFloods_maps':excluded['wf'],
          'protocols':{'sturm':'P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1:clean','wf':'P3_WF_NATIVE_ZERO_PAD512_V1'},
          'prior_screen_sha256':sha(prior_path),'source_manifest_sha256':sha(source_path),
          'reference_manifest_sha256':sha(ref_path),'evaluation_gate_sha256':sha(gate_path),
          'geometry':'Densified actual264-pixel patch polygon transformed with always_xy; raster-centre and conservative all-touched intersections. Whole affected chips/maps excluded if scored support intersects.',
          'comparator_scope':'Teacher and MNDWI once; all4seeds of original-data EffViT N2500 focal/KD/OFA,EffViT N100 KD/OFA,MBV3Small/LightFormer N2500 KD/OFA; common exclusions even where N100 did not contain intersecting sources.'}
    plan_path=OUT/'overlap_exclusion_plan.json'
    if plan_path.exists():
        previous=load(plan_path)
        assert all(previous[k]==plan[k] for k in ('exclude_complete_STURM_chips','exclude_complete_WorldFloods_maps','source_manifest_sha256','reference_manifest_sha256','evaluation_gate_sha256'))
    else: dump(plan_path,plan)
    dump(OUT/'overlap_scored_support_assessment.json',{'status':'complete','records':records,
         'exclusion_plan':str(plan_path),'exclusion_plan_sha256':sha(plan_path),
         'counts':{'STURM_chips':len(excluded['sturm']),'WorldFloods_maps':len(excluded['wf'])},
         'limits':['Different acquisitions can expose similar geography; no duplicate acquisition was found in these pairs.',
                   'Exclusion does not prove every correlated example absent or resolve historical benchmark-guided choices.',
                   'Current FPS prefix does not prove executed historical N252 membership.'],
         'training_reader_sha256':sha(snapshot/'data/unlabeled.py'),
         'training_augmentation_reader_sha256':sha(snapshot/'data/pseudo_labeled.py')})
    print(json.dumps({'exclusions':excluded,'intersections':[
        {k:r[k] for k in r if k in ('id','training_file','N100_current_member','intersect_valid_corrected512_center',
         'intersect_water_corrected512_center','intersect_valid_retained_native_center','intersect_water_retained_native_center')}
        for r in records]},indent=2))

if __name__=='__main__': assess()
