import argparse,sys,json,time
from pathlib import Path
from diagnose_unfolded import ROOT,OLD,q,x,rows,score
import copy,numpy as np

def main(seed):
    out=ROOT/f'smooth_check_s{seed}';out.mkdir(exist_ok=False);plan=x.read(OLD/'workstation/PROTOCOL.json');e=next(v for v in plan['entries'] if v['seed']==seed)
    torch,model,ji,ds,inv,ev=q.runtime(x.read(e['config']['path']));items=list(q.source_items(ds,inv,ev));parent=x.read(OLD/f'workstation/B0_s{seed}/FLOAT_EVALUATION.json')['rows'];assert rows(model,items,torch)==parent
    from mixed_smooth import prepare_parent,prepare_quant
    images=[torch.from_numpy(z[0]) for z in items];smooth,scales=prepare_parent(model,images);checks=[]
    with torch.no_grad():
        for image,mask,r in items:
            a=model(torch.from_numpy(image)).numpy();b=smooth(torch.from_numpy(image)).numpy();checks.append(dict(file=r['name'],passed=bool(np.allclose(a,b,atol=.001,rtol=.001)),max_abs=float(np.max(abs(a-b)))))
    x.write(out/'FLOAT_CONSTRUCTION.json',dict(passed=all(v['passed'] for v in checks),images=89,rows=checks),True);x.write(out/'SCALES.json',scales,True)
    if not all(v['passed'] for v in checks):print('BLOCKED construction',seed,flush=True);return
    quant=prepare_quant(smooth,images);result=score(quant,items,torch,ji,parent);x.write(out/'Q0.json',result,True)
    torch.save(dict(state=quant.state_dict(),scales=scales),out/'quant_initial.pt');print(seed,'smooth',result['drops'],result['accuracy_pass'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);main(p.parse_args().seed)
