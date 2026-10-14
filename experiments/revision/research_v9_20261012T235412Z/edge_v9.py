"""Prespecified spectral rules, edge bands and input-only cloud diagnostics."""
import numpy as np
from scipy import ndimage as ndi

MEAN=np.array([.1413,.138,.1235,.309,.2045,.1191],np.float32)
STD=np.array([.0741,.0737,.0869,.118,.0977,.0766],np.float32)

def reflectance(x):return np.asarray(x,dtype=np.float32)*STD[:,None,None]+MEAN[:,None,None]
def indices(rho):
    g,s,n=rho[1],rho[4],rho[3]
    a=np.isfinite(g)&np.isfinite(s)&(g>=0)&(s>=0)&(g+s>1e-6)
    b=np.isfinite(g)&np.isfinite(n)&(g>=0)&(n>=0)&(g+n>1e-6)
    m=np.where(a,(g-s)/np.maximum(g+s,1e-6),0).clip(-1,1)
    nd=np.where(b,(g-n)/np.maximum(g+n,1e-6),0).clip(-1,1)
    return m,a,nd,b

def edge_distance(pred):
    pred=np.asarray(pred,dtype=bool)
    if pred.all() or not pred.any():return np.full(pred.shape,np.inf)
    return np.where(pred,ndi.distance_transform_edt(pred),ndi.distance_transform_edt(~pred))

def decode(pred,mndwi,available,k,theta):
    pred=np.asarray(pred,dtype=bool);band=(edge_distance(pred)<=k)&available
    out=pred.copy();out[band]=(mndwi>theta)[band]
    return out

def cloud_test(rho):
    b,g,r,n,s1,s2=rho;vis=(b+g+r)/3
    white=(abs(b-vis)+abs(g-vis)+abs(r-vis))/np.maximum(vis,1e-6)
    return (vis>.18)&(white<.7)&(b-.5*r-.08>0)&(s2>.03)&(s1>.12)

def strata(mask,rho,cloud=None):
    water=mask==1;land=mask==0;valid=mask!=-1
    dw=ndi.distance_transform_edt(~land) if land.any() else np.full(mask.shape,np.inf)
    dl=ndi.distance_transform_edt(~water) if water.any() else np.full(mask.shape,np.inf)
    distance=np.where(water,dw,dl);out={}
    for name,lo,hi in [('1',0,1),('2',1,2),('3',2,3),('4-5',3,5),('6-10',5,10),('11-20',10,20),('>20',20,np.inf)]:
        out['reference_edge_'+name]=valid&(distance>lo)&(distance<=hi)
    out['reference_edge_1-2']=valid&(distance<=2)
    # Inherited probe definition: maximum inscribed distance of each connected body.
    half=np.zeros(mask.shape,np.float32)
    if water.any() and land.any():
        lab,n=ndi.label(water,structure=np.ones((3,3),bool));mx=ndi.maximum(dw,lab,index=np.arange(1,n+1))
        half=np.r_[0,np.asarray(mx,np.float32)][lab]
    out['water_halfwidth_<=2']=water&(half>0)&(half<=2)
    out['water_halfwidth_2-5']=water&(half>2)&(half<=5)
    out['water_halfwidth_>5']=water&(half>5)
    vis=rho[:3].mean(0);ndvi=(rho[3]-rho[2])/np.maximum(rho[3]+rho[2],1e-6)
    out['spectral_dark']=valid&(vis<.06)&(rho[3]<.08)&(rho[4]<.06)
    out['spectral_bright']=valid&(vis>.20)
    out['spectral_vegetated']=valid&(ndvi>.4)
    out['spectral_other']=valid&~(out['spectral_dark']|out['spectral_bright']|out['spectral_vegetated'])
    for prefix,cl in [('input_cloud',cloud_test(rho)),('reference_cloud',cloud)]:
        if cl is None:continue
        distance=ndi.distance_transform_edt(~cl) if cl.any() else np.full(mask.shape,np.inf)
        for name,lo,hi in [('<=5',0,5),('6-20',5,20),('21-50',20,50),('>50',50,np.inf)]:
            out[prefix+'_'+name]=valid&(distance>lo)&(distance<=hi)
    return out

def counts(pred,mask):
    p=np.asarray(pred,bool);valid=mask!=-1;water=mask==1
    tp=int((p&water&valid).sum());fp=int((p&~water&valid).sum());fn=int((~p&water&valid).sum());tn=int((~p&~water&valid).sum())
    return np.array([tp,fp,fn,tn],np.int64)

def scores(c):
    tp,fp,fn,tn=map(int,c)
    ratio=lambda a,b:a/b if b else None
    w=ratio(tp,tp+fp+fn);l=ratio(tn,tn+fn+fp);defined=[v for v in [w,l] if v is not None]
    return dict(water_IoU=w,mIoU=sum(defined)/len(defined) if defined else None,
        precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),false_water_rate=ratio(fp,fp+tn))

def checks():
    mask=np.zeros((8,8),bool);mask[:,:4]=True
    assert (edge_distance(mask)<=1).sum()==16
    assert (edge_distance(mask)<=2).sum()==32
    rho=np.zeros((6,8,8),np.float32);rho[1]=.2;rho[4]=.1
    m,a,_,_=indices(rho);assert (m>0).all()
    out=decode(mask,m,a,2,.1);assert np.array_equal(out[edge_distance(mask)>2],mask[edge_distance(mask)>2])
    unavailable=np.zeros((8,8),bool);assert np.array_equal(decode(mask,m,unavailable,2,.1),mask)
    y=mask.astype(np.int8);y[0]=-1;assert counts(mask,y).sum()==56
    return dict(outcome='passed',synthetic_edge_distance=True,unavailable_preserves_prediction=True,ignore_support=True)
