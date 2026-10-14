"""Export the independent source analysis as plot/report tables without GPU work."""
import csv
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
data = json.loads((HERE/'source_analysis.json').read_text())
data['graph_descriptions'] = {
    'A0': {'parameters':704714,'family':'EfficientViT-B0 segmentation','width_multiplier':1.,'eligible_repeat_multiplier':1.},
    'A1': {'parameters':190870,'family':'EfficientViT-B0 segmentation','width_multiplier':.5,'eligible_repeat_multiplier':1.},
    'A3': {'parameters':30922,'family':'EfficientViT-B0 segmentation','width_multiplier':.25,'eligible_repeat_multiplier':.5},
    'A3N': {'parameters':20250,'family':'A3 with LiteMLA context modules replaced by identity; local branches retained'},
    'R4': {'parameters':12230,'family':'declared compact spatial CNN',
           'structure':'stride2 six-band stem; depthwise separable residual stages at H/4,H/8,H/16; widths12,16,24,40; skip decoder16; full-resolution bilinear logits',
           'attention':False},
    'study_type':'fixed declared graphs and LR/intervention comparisons; no architecture search or NAS'}
pilot_summary = {}
for arm, seeds in data['pilot_per_seed'].items():
    assert set(seeds)=={'42','64'}
    pilot_summary[arm] = {}
    for kind in ('start','selected','final'):
        pilot_summary[arm][kind] = {metric:{'mean_percent':statistics.mean(row[kind][metric] for row in seeds.values()),
            'sample_sd_percent':statistics.stdev(row[kind][metric] for row in seeds.values()),'n':2,'uncertainty':'descriptive only'}
            for metric in ('mIoU','water_IoU')}
    pilot_summary[arm]['selected_mean_gain_over_start_pp'] = pilot_summary[arm]['selected']['mIoU']['mean_percent']-pilot_summary[arm]['start']['mIoU']['mean_percent']
    pilot_summary[arm]['final_mean_gain_over_start_pp'] = pilot_summary[arm]['final']['mIoU']['mean_percent']-pilot_summary[arm]['start']['mIoU']['mean_percent']
data['pilot_summary']=pilot_summary
data['pilot_selection_checks']={
    'all_C0_to_C5_seeds42_64_selected_start_zero':all(data['pilot_per_seed'][arm][str(seed)]['selected_updates']==0
        for arm in ('C0','C1','C2','C3','C4','C5') for seed in (42,64)),
    'HC_seed64_selected_update':data['pilot_per_seed']['HC']['64']['selected_updates'],
    'HC_seed64_selected_gain_pp':data['pilot_per_seed']['HC']['64']['selected']['mIoU']-data['pilot_per_seed']['HC']['64']['start']['mIoU']}
assert data['pilot_selection_checks']['all_C0_to_C5_seeds42_64_selected_start_zero']
assert data['pilot_selection_checks']['HC_seed64_selected_update']==936
destination=HERE/'source_stats.json'
assert not destination.exists()
destination.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def write(name, rows):
    path=HERE/name
    assert not path.exists()
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

families=[]
for name, group in data['groups'].items():
    for kind in ('selected','final'):
        for metric in ('mIoU','water_IoU'):
            families.append(dict(family=name,checkpoint_kind=kind,metric=metric,n_seeds=4,
                parameters=group['parameters'],lr=group['lr'],mean_percent=group[kind][metric]['mean_percent'],
                sample_sd_percent=group[kind][metric]['sample_sd_percent']))
write('source_families.csv',families)
contrasts=[]
for row in data['contrasts']:
    flat={key:value for key,value in row.items() if key!='paired_differences_pp'}
    flat.update({'difference_seed'+seed+'_pp':value for seed,value in row['paired_differences_pp'].items()})
    contrasts.append(flat)
write('source_paired_contrasts.csv',contrasts)
all_results=list(csv.DictReader((ROOT/'results.csv').open()))
pilots=[]
for arm,seeds in data['pilot_per_seed'].items():
    for seed,row in seeds.items():
        for kind in ('start','selected','final'):
            coefficient_row=next((r for r in all_results if r['run_id']==arm+'_s'+seed
                and r['checkpoint_kind']==kind and r['dataset']=='source_validation'),{})
            pilots.append(dict(arm=arm,seed=seed,checkpoint_kind=kind,mIoU_percent=row[kind]['mIoU'],
                water_IoU_percent=row[kind]['water_IoU'],selected_updates=row['selected_updates'],
                completed_updates=row['completed_updates'],source_gain_over_start_pp=row[kind]['mIoU']-row['start']['mIoU'],
                rule_a=coefficient_row.get('rule_a'),rule_b=coefficient_row.get('rule_b')))
write('pilot_selected_final.csv',pilots)
write('pilot_mean_source.csv',[dict(arm=arm,checkpoint_kind=kind,n_seeds=2,
    mean_mIoU_percent=summary[kind]['mIoU']['mean_percent'],mean_water_IoU_percent=summary[kind]['water_IoU']['mean_percent'],
    selected_mean_gain_over_start_pp=summary['selected_mean_gain_over_start_pp'],
    final_mean_gain_over_start_pp=summary['final_mean_gain_over_start_pp'])
    for arm,summary in pilot_summary.items() for kind in ('start','selected','final')])
print(json.dumps({'source_stats':str(destination),'pilot_selection_checks':data['pilot_selection_checks'],
    'pilot_final_means':{arm:summary['final']['mIoU']['mean_percent'] for arm,summary in pilot_summary.items()}},indent=2))
