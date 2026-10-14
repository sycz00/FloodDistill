"""Fixed-weight damage/repair maps and TRAIN-ranked U6 mixed precision."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from collections import Counter

p=argparse.ArgumentParser()
for name in ('capsule','spec','cell','out','adoption'):p.add_argument('--'+name,required=True)
a=p.parse_args();sys.path.insert(0,a.capsule)
import numpy as np
import torch
from models.compression import build_compression_student
from models.compression_quant import load_compression_qat,rebind_compression_bits,FixedQATConv2d,_range
from experiments.revision_queue import atomic_json,file_hash
from training.losses import kd_loss
import scripts.compression_jobs as cj
from simulator_v4 import simulator
from scipy.stats import spearmanr
torch.set_num_threads(2);torch.manual_seed(1729);np.random.seed(1729)
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
start=time.monotonic();adoption=json.loads(Path(a.adoption).read_text())
assert adoption['adopted_simulator']=='integer_domain_v4'
spec=json.loads(Path(a.spec).read_text());spec['workers']=0
base=Path(spec['interaction_receipt']).parent;graph=a.cell.split('_')[0]
cal=json.loads((base/f'{graph}_PTQ.json').read_text())
assert file_hash(cal['tensor_path'])==cal['tensor_sha256']
payload=torch.load(cal['tensor_path'],map_location='cpu',weights_only=True)
q8=load_compression_qat(build_compression_student(graph,pretrained=False),payload['model']).cuda().eval()
weight_bits,activation_bits=(4,4) if a.cell.endswith('W4A4') else (4,8)
low=rebind_compression_bits(q8,weight_bits,activation_bits)
layers={name:mod for name,mod in low.named_modules() if isinstance(mod,FixedQATConv2d)}
high={name:mod for name,mod in q8.named_modules() if isinstance(mod,FixedQATConv2d)}
def set_bits(mod,w=None,act=None):
    if w is not None and w!=mod.weight_bits:
        mod._w_scale.mul_(_range(mod.weight_bits)[1]/_range(w)[1]);mod._weight_bits.fill_(w);mod._weight_bits_value=w
    if act is not None and act!=mod.activation_bits:
        mod._act_scale.mul_(_range(mod.activation_bits)[1]/_range(act)[1]);mod._activation_bits.fill_(act);mod._activation_bits_value=act;mod._activation_scale_value=float(mod._act_scale[0])
def val(model):
    with simulator(model,'integer'):return cj.validation(model,spec)['micro']
high_score=val(q8);low_score=val(low);rows=[]
for i,name in enumerate(layers):
    set_bits(high[name],w=4);damage=val(q8);set_bits(high[name],w=8)
    set_bits(layers[name],w=8);repair=val(low);set_bits(layers[name],w=4)
    row={'layer':name,'weights':layers[name].weight.numel(),
         'damage_one_weight_mIoU':high_score['mIoU']-damage['mIoU'],
         'repair_one_weight_mIoU':repair['mIoU']-low_score['mIoU'],
         'damage_one_weight_scores':damage,'repair_one_weight_scores':repair}
    if activation_bits==4:
        set_bits(high[name],act=4);dmg=val(q8);set_bits(high[name],act=8)
        set_bits(layers[name],act=8);rep=val(low);set_bits(layers[name],act=4)
        row.update(damage_one_activation_mIoU=high_score['mIoU']-dmg['mIoU'],
                   repair_one_activation_mIoU=rep['mIoU']-low_score['mIoU'])
    rows.append(row);print(json.dumps({'cell':a.cell,'layer':i+1,'total':len(layers)}),flush=True)
u6={'status':'not_applicable','reason':'Predeclared U6 held-out test is A1 W4A8'}
if a.cell=='A1_W4A8':
    teacher=cj.teacher(spec);teacher.eval().requires_grad_(False)
    train=cj.pseudo(spec,augment=False);by_name={p.name:i for i,p in enumerate(train.files)}
    pool=spec['feature_pools']['interaction'][:128]
    batches=[]
    for start_i in range(0,len(pool),16):
        samples=[train[by_name[row['file']]] for row in pool[start_i:start_i+16]]
        x=torch.stack([s['image'] for s in samples]).float().cuda()
        y=torch.stack([s['mask'] for s in samples]).long().cuda()
        with torch.no_grad():target=teacher(x).detach()
        batches.append((x.cpu(),y.cpu(),target.cpu()))
    def excess_sum(model):
        total=0.;n=0
        with simulator(model,'integer'),torch.no_grad():
            for x,y,target in batches:
                x,y,target=x.cuda(),y.cuda(),target.cuda();valid=y>=0;count=int(valid.sum())
                total+=float(kd_loss(model(x),target,4,valid))*count;n+=count
        return total/n
    base_kl=excess_sum(q8);label_free=[]
    for name in high:
        set_bits(high[name],w=4);score=excess_sum(q8)-base_kl;set_bits(high[name],w=8)
        label_free.append({'layer':name,'teacher_excess_KL':score})
    ranked=sorted(label_free,key=lambda r:(-r['teacher_excess_KL'],r['layer']))
    label_rank=sorted(rows,key=lambda r:(-r['damage_one_weight_mIoU'],r['layer']))
    top_overlap=len({r['layer'] for r in ranked[:4]}&{r['layer'] for r in label_rank[:4]})
    by_label={r['layer']:r['damage_one_weight_mIoU'] for r in rows}
    correlation=float(spearmanr([r['teacher_excess_KL'] for r in label_free],[by_label[r['layer']] for r in label_free]).statistic)
    allocations=[];rng=np.random.default_rng(1729);names=sorted(layers)
    def score_allocation(kind,selected,k,draw=None):
        for name in selected:set_bits(layers[name],w=8)
        score=val(low)
        extra=sum(layers[name].weight.numel()*4/8 for name in selected)
        allocations.append({'kind':kind,'k':k,'draw':draw,'layers':selected,'extra_theoretical_weight_bytes':extra,'scores':score})
        for name in selected:set_bits(layers[name],w=4)
    for k in (1,2,4):
        score_allocation('TRAIN_label_free',[r['layer'] for r in ranked[:k]],k)
        score_allocation('S11_val_label_rank_optimistic',[r['layer'] for r in label_rank[:k]],k)
        for draw in range(5):score_allocation('random_k',rng.choice(names,k,replace=False).tolist(),k,draw)
    structural=[name for name,mod in layers.items() if mod.weight.numel()<=64 or mod.out_channels<=8]
    score_allocation('structural_le64weights_or_le8outputs',structural,len(structural))
    u6={'status':'complete','TRAIN_images':128,'TRAIN_groups':len({r['group'] for r in pool}),
        'TRAIN_pool_role':'interaction/verification, disjoint from the PCA fit pool',
        'label_free_scores':label_free,'label_rank_same_validation_optimistic':label_rank,
        'top4_overlap':top_overlap,'spearman':correlation,'allocations':allocations,
        'gate_overlap_ge2':top_overlap>=2,'no_external_benchmark_access':True,
        'selection_scope':'single-seed PTQ screen only, no scientific method winner selected'}
atomic_json(a.out,{'status':'complete','outcome':'completed','stage':'1.4/U6','cell':a.cell,
    'simulator':'integer_domain_v4_exact_native_STE','high_reference':high_score,'low_reference':low_score,
    'rows':rows,'U6':u6,'no_training':True,'no_external_benchmark_access':True,
    'serialized_low_bit_engine_claim':False,'extra_bytes_scope':'theoretical weight payload only',
    'elapsed_seconds':time.monotonic()-start,'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20})
