"""Hash-check the exact frozen512 model inputs and reference masks, streaming images."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def inventory(ds):return [json.loads(l) for l in (ROOT/'inventories'/(ds+'.jsonl')).read_text().splitlines()]

def source(ds):
    split={'s11val':'val','s11test':'test','bolivia':'bolivia'}[ds];inv=inventory(ds)
    for rec in inv:
        name=rec['name'].replace('_S2Hand.tif','.npz')
        with np.load(ROOT/'assets'/('source_'+split)/name,allow_pickle=False) as z:image=z['image'];mask=z['mask']
        assert np.isin(mask,[-1,0,1]).all();mask=mask.astype(np.int8)
        assert ah(image)==rec['input_sha256'] and ah(mask)==rec['mask_sha256'],rec['name']
        yield image,mask,rec

def external(ds,sturm_root,wf_root):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import data.sen1floods11_nodep as nodep
    sys.modules['data.sen1floods11']=nodep
    from data.ood_datasets import get_sturm_flood_dataloader,get_worldfloods_dataloader
    from data.channel_adapter import sturm_s2_adapter,adapt_batch
    if ds=='sturm512':
        loader=get_sturm_flood_dataloader(str(sturm_root),sensor='S2',batch_size=1,num_workers=0,img_size=512,normalize='original',sim_gain=1.,sim_haze=0.,protocol='surface_water')
        adapter=sturm_s2_adapter(target_channels=6)
    elif ds=='wf':
        loader=get_worldfloods_dataloader(str(wf_root),split='test',batch_size=1,num_workers=0,img_size=512,tile_size=512,normalize='s11',sim_gain=1.,sim_haze=0.,protocol='surface_water');adapter=None
    else:raise ValueError(ds)
    inv=inventory(ds);by={r['name']:r for r in inv};seen=set()
    for batch in loader:
        if adapter is not None:batch=adapt_batch(batch,adapter)
        name=str(batch['filename'][0]);assert name in by and name not in seen,(ds,name)
        image=batch['image'][0].numpy();mask=batch['mask'][0].numpy()
        if mask.ndim==3 and mask.shape[0]==1:mask=mask[0]
        assert np.isin(mask,[-1,0,1]).all();mask=mask.astype(np.int8)
        rec=by[name];assert image.shape==(6,512,512) and mask.shape==(512,512)
        assert ah(image)==rec['input_sha256'],('input',ds,name,ah(image),rec['input_sha256'])
        assert ah(mask)==rec['mask_sha256'],('mask',ds,name,ah(mask),rec['mask_sha256'])
        seen.add(name);yield image,mask,rec
    assert seen==set(by),('incomplete inventory',ds,len(seen),len(by))

def images(ds,sturm_root=None,wf_root=None):
    yield from source(ds) if ds in ('s11val','s11test','bolivia') else external(ds,sturm_root,wf_root)

def dry():
    records=json.loads((ROOT/'inventories/dry159_manifest.json').read_text())['rows']
    assert len(records)==159
    for rec in records:
        with np.load(ROOT/'assets/dry159'/(rec['tile_id']+'.npz'),allow_pickle=False) as z:image=z['image'];valid=z['valid']
        assert ah(image)==rec['input_sha256'] and ah(valid)==rec['valid_sha256']
        yield image,valid,rec
