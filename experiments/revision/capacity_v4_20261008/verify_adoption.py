"""Review Stage-1 simulator evidence against fixed criteria, retaining raw data."""
import hashlib
import json
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
identity=json.loads((BASE/'shared_input_identity.json').read_text())
sys.path[:0]=[identity['capsule'],str(BASE/'code/v1')]
import torch
from torch import nn
from models.compression_quant import FixedQATConv2d
from simulator_v4 import simulator
torch.set_num_threads(2);torch.manual_seed(1729)
unit=[]
for bits in (4,8):
    for seed in range(4):
        torch.manual_seed(seed)
        conv=nn.Conv2d(2,3,3,padding=1)
        m=FixedQATConv2d(conv,torch.tensor([.37]),torch.tensor([0],dtype=torch.int32),torch.full((3,),.17),weight_bits=bits,activation_bits=bits)
        x=(torch.randn(2,2,9,9)*4).requires_grad_(True)
        out=m(x);g=torch.autograd.grad(out.sum(),(x,m.weight,m.bias))
        with simulator(m):
            alt=m(x);h=torch.autograd.grad(alt.sum(),(x,m.weight,m.bias))
        assert torch.allclose(out,alt,rtol=1e-5,atol=1e-5)
        assert all(torch.allclose(left,right,rtol=1e-5,atol=1e-5) for left,right in zip(g,h))
        scaled=x.detach()/m._activation_scale_value
        ties=int(((scaled-scaled.floor()-.5).abs()<1e-6).sum())
        unit.append({'bits':bits,'seed':seed,'max_forward_difference':float((out-alt).detach().abs().max()),
                     'near_rounding_ties':ties,'STE_gradients_agree':True})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for p in sorted((BASE/'results').glob('*measurement.json')):
    d=json.loads(p.read_text());assert d['integer_norm_and_PTQ_gate']
    checks.append({'receipt':str(p),'sha256':sha(p),'TRAIN_argmax_agreement':d['augmented_argmax_agreement_mean'],
                   'PTQ_delta_mIoU':d['PTQ_parity_absolute_mIoU_difference']})
assert len(checks)==4
parity_path=BASE/'results/parity_all.json';d=json.loads(parity_path.read_text())
rows=[]
v2=BASE.parent/'compression_v2_20261007/results'
for row in d['rows']:
    path=v2/(row['job_id']+'.json');original=json.loads(path.read_text())
    pilot=original['validation']['micro']['mIoU'];fixed=row['validation']['integer']['micro']['mIoU']
    difference=fixed-pilot
    assert abs(difference)<=.0015
    rows.append({'job_id':row['job_id'],'original_receipt_sha256':sha(path),
        'checkpoint_sha256':row['checkpoint_sha256'],'recorded_v2_source_mIoU':pilot,
        'integer_source_mIoU':fixed,'delta_vs_recorded_v2':difference,
        'recomputed_legacy_TF32_disabled_mIoU':row['validation']['legacy']['micro']['mIoU']})
result={'status':'complete','outcome':'completed','adopted_simulator':'integer_domain_v4',
        'source_file':str(BASE/'code/v1/simulator_v4.py'),'source_sha256':sha(BASE/'code/v1/simulator_v4.py'),
        'unit_checks_nonbinary_scales_and_bias':unit,'fixed_batch_checks':checks,
        'original_parity_receipt_sha256':sha(parity_path),'source_rescore':rows,
        'maximum_delta_vs_recorded_v2_pp':100*max(abs(r['delta_vs_recorded_v2']) for r in rows),
        'TF32_enabled':False,'integer_exactness':'FP32 if conservative integer accumulation bound <2^24, otherwise float64',
        'forward_disagreement_not_hidden':'W4A4 augmented TRAIN disagreement remains about8%; v2 source comparisons rescored; original external conclusions not reused as adopted-simulator results',
        'numerical_backend_disclosure':'Disabling CUDA TF32 changes legacy rounding-residue logits; raw same-backend comparisons and comparisons against original recorded v2 scores both retained.',
        'stage1_complete':False,'remaining':['1.4 sensitivity maps/U6','1.6 optimizer reach','1.7 protocol freeze'],
        'no_external_scores_used':True}
(BASE/'simulator_adoption.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'adopted':result['adopted_simulator'],'max_recorded_v2_delta_pp':result['maximum_delta_vs_recorded_v2_pp']}))
