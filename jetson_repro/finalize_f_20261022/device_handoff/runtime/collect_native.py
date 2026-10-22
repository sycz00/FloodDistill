#!/usr/bin/env python3
"""Collect every model/seed/precision cell; failed or absent seeds stay explicit."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from evaluate_native import summarize,scores
from validate_results import validate_rows,validate_dry,benchmark_receipts_valid
ROOT=Path(__file__).resolve().parents[1]
DATASETS=('s11val','s11test','bolivia','wf','sturm512')
def read(p):return json.loads(Path(p).read_text())
def csvout(p,rs):
    if not rs:return
    keys=list(dict.fromkeys(k for r in rs for k in r));p=Path(p)
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def imported(spec,ds):
    info=spec['reuse']['evaluations'][ds];o=read(ROOT/info['path']);rs=o['datasets'][info['key']]['rows'] if info.get('key') else o['rows']
    rows=[dict(name=r['file'],event=r['event'],counts_tn_fp_fn_tp=np.asarray(r['confusion']).ravel().tolist()) for r in rs]
    validate_rows(ROOT,ds,rows,imported=True)
    return summarize(rows)
def main(attempt):
    out=ROOT/'results'/attempt;manifest=read(ROOT/'RUN_MANIFEST.json');states={r['id']:r for r in read(out/'STATUS.json')}
    lookup={m['id']:m for m in manifest['engines']};evaluated={};accuracy=[];bench=[];dryrows=[];wfdry=[];coverage=[];invalid=[];complete_ids=set()
    for ident,m in lookup.items():
        dest=out/ident;state=states.get(ident,dict(status='NOT_RUN'))
        completion=dest/'evaluation/COMPLETE.json'
        try:
            expected_engine=m['reuse']['engine_sha256'] if m.get('reuse') else read(dest/'build.json')['engine_sha256']
            cert=read(ROOT/m['reuse']['build_receipt']) if m.get('reuse') else read(dest/'build.json')
            if cert['engine_sha256']!=expected_engine:raise ValueError('Engine/build identity mismatch')
            if m['precision']=='pure' and not cert['precision_audit']['full_integer_tensor_path']:raise ValueError('Integer certificate failed')
            if m.get('fp32_islands'):
                from precision_islands import validate_certificate
                validate_certificate(cert, m, ROOT, read(dest/'model.engine.layers.json'))
            if completion.exists() and read(completion)['engine_sha256']==expected_engine:complete_ids.add(ident)
        except (OSError,KeyError,ValueError) as error:
            if completion.exists():invalid.append(dict(id=ident,stage='completion/identity',reason=str(error)))
        for ds in DATASETS:
            p=dest/'evaluation'/(ds+'.json')
            try:
                s=imported(m,ds) if m.get('reuse') else read(p)
                if not m.get('reuse'):validate_rows(ROOT,ds,s['rows']);s=summarize(s['rows'])
            except OSError:continue
            except (KeyError,ValueError,AssertionError) as error:invalid.append(dict(id=ident,dataset=ds,reason=str(error)));continue
            expected=sum(1 for _ in (ROOT/'inventories'/(ds+'.jsonl')).open());assert s['images']==expected,(ident,ds,s['images'],expected)
            evaluated[ident,ds]=s
        dry=dest/'evaluation/dry159.json'
        if dry.exists():
            try:rs=read(dry)['rows'];validate_dry(ROOT,rs)
            except (OSError,KeyError,ValueError,AssertionError) as error:
                invalid.append(dict(id=ident,dataset='dry159',reason=str(error)));complete_ids.discard(ident);rs=[]
            for subset,crop in (('author49','central264'),('author49','full512'),('map159','central264')):
                for group in ('ordinary','stress'):
                    rr=[r for r in rs if r['group']==group and (subset=='map159' or r['decision']=='kept_dry')]
                    vals=[r[crop] for r in rr if r[crop]['false_water_fraction'] is not None]
                    dryrows.append(dict(id=ident,architecture=m['architecture'],seed=m['seed'],precision=m['precision'],method=m['method'],subset=subset,crop=crop,group=group,images=len(vals),alarms=sum(v['alarm_gt_1pct'] for v in vals),alarm_rate=float(np.mean([v['alarm_gt_1pct'] for v in vals])) if vals else None,false_water_fraction=float(np.mean([v['false_water_fraction'] for v in vals])) if vals else None))
        if (ident,'wf') in evaluated:
            rs=[r for r in evaluated[ident,'wf']['rows'] if sum(r['counts_tn_fp_fn_tp'][2:])==0];assert len(rs)==200
            for r in rs:
                c=r['counts_tn_fp_fn_tp'];wfdry.append(dict(id=ident,event=r['event'],name=r['name'],valid_pixels=sum(c),false_water_pixels=c[1],alarm_gt_1pct=100*c[1]>sum(c)))
        if m.get('reuse'):
            old=list(csv.DictReader((ROOT/'prior_device_bundle/01_report/benchmark.csv').open()));ref=next(r for r in old if r['seed']==str(m['seed']) and r['precision']==m['precision'])
            bench.append(dict(id=ident,architecture=m['architecture'],method=m['method'],measurement='reused21-22October',benchmark_evidence_valid=True,gpu_p50_ms=float(ref['gpu_compute_ms_median']),gpu_p95_ms=float(ref['gpu_compute_ms_p95']),gpu_p99_ms=float(ref['gpu_compute_ms_p99']),wall_p50_ms=float(ref['e2e_wall_ms_p50_device_runtime']),wall_p95_ms=float(ref['e2e_wall_ms_p95_device_runtime']),wall_p99_ms=float(ref['e2e_wall_ms_p99_device_runtime']),throughput=float(ref['throughput_img_per_s_trtexec']),steady_energy_mJ=float(ref['steady_energy_mJ_per_image']),**ref))
        elif (dest/'evaluation/timing.json').exists():
            t=read(dest/'evaluation/timing.json');profile=read(dest/'benchmark/bench.json') if (dest/'benchmark/bench.json').exists() else {};steady=read(dest/'steady_power/power_steady.json') if (dest/'steady_power/power_steady.json').exists() else {}
            build=read(dest/'build.json');audit=build.get('precision_audit',{})
            bench.append(dict(id=ident,architecture=m['architecture'],seed=m['seed'],precision=m['precision'],method=m['method'],measurement='new native attempt',benchmark_evidence_valid=benchmark_receipts_valid(profile,steady,state),gpu_p50_ms=t['kernel_ms']['p50'],gpu_p95_ms=t['kernel_ms']['p95'],gpu_p99_ms=t['kernel_ms']['p99'],wall_p50_ms=t['wall_ms']['p50'],wall_p95_ms=t['wall_ms']['p95'],wall_p99_ms=t['wall_ms']['p99'],engine_bytes=t['engine_bytes'],device_memory_bytes=t['device_memory_bytes'],int8_tensor_layers=audit.get('int8_tensor_layers'),layers=audit.get('layer_count'),full_integer_tensor_path=audit.get('full_integer_tensor_path'),throughput=profile.get('throughput_qps_runA'),steady_power_w=steady.get('steady_cpu_gpu_cv_mean_w'),steady_energy_mJ=steady.get('steady_energy_mJ_per_image'),power_rail=steady.get('power_rail'),fp32_islands_pass=build.get('fp32_island_audit',{}).get('passed'),fp32_islands_sha256=build.get('fp32_islands_sha256')))
        baseline=f"{m['architecture']}_s{m['seed']}_fp32"
        for ds in DATASETS:
            if (ident,ds) not in evaluated:continue
            s=evaluated[ident,ds];ref=evaluated.get((baseline,ds))
            row=dict(id=ident,architecture=m['architecture'],seed=m['seed'],precision=m['precision'],method=m['method'],dataset=ds,images=s['images'],events=s['events'],pooled_mIoU=s['pooled']['miou'],pooled_water_IoU=s['pooled']['water_iou'],equal_event_water_IoU=s['equal_event']['water_iou'],equal_event_mIoU=s['equal_event']['miou'],precision_water=s['pooled']['precision'],recall_water=s['pooled']['recall'],source_admission=m.get('source_admission'))
            if ref:
                drops=[ref['pooled']['miou']-s['pooled']['miou'],ref['equal_event']['miou']-s['equal_event']['miou'],ref['pooled']['water_iou']-s['pooled']['water_iou']]
                row.update(drop_pooled_mIoU=drops[0],drop_equal_event_mIoU=drops[1],drop_pooled_water_IoU=drops[2])
                if ds=='s11val':row.update(user_0p02_pass=all(d<=.02 for d in drops),historical_frozen_pass=all(d<=v for d,v in zip(drops,(.01,.01,.02))))
            accuracy.append(row)
    # Wiring is against the matching CPU float graph, including named linear graph variants.
    wiring=[]
    for m in manifest['engines']:
        if m['precision']!='fp32' or (m['id'],'s11val') not in evaluated:continue
        ref=read(ROOT/m['source_reference']);s=evaluated[m['id'],'s11val'];d=abs(s['pooled']['miou']-ref['mIoU']);wiring.append(dict(id=m['id'],pooled_mIoU_abs_difference=d,pass_0p005=d<=.005))
    # Fixed primary matrix:3 architectures x4 seeds x4 precision slots. No accepted-only means.
    for slot in manifest['primary_matrix']:
        ids=slot.get('engine');status=states.get(ids,{}).get('status','NOT_RUN') if ids else slot['status']
        complete=ids in complete_ids and all((ids,d) in evaluated for d in DATASETS) and (out/ids/'evaluation/dry159.json').exists()
        bm=next((r for r in bench if r['id']==ids),{})
        benchmark_ok=bool(bm) and bm.get('benchmark_evidence_valid',False) and (bool(bm.get('steady_power_w')) and bool(bm.get('engine_bytes')) and bool(bm.get('throughput') or bm.get('throughput_img_per_s_trtexec')))
        acc=next((r for r in accuracy if r['id']==ids and r['dataset']=='s11val'),{})
        source_ok=lookup.get(ids,{}).get('source_admission')=='SOURCE_QUALIFIED'
        coverage.append(dict(**slot,native_status=status,all_five_datasets_and_dry_complete=complete,benchmark_complete=benchmark_ok,source_admission=lookup.get(ids,{}).get('source_admission'),native_source_accuracy_pass=acc.get('user_0p02_pass'),workstation_source_accuracy_pass=lookup.get(ids,{}).get('source_accuracy_pass'),fp32_islands_required=bool(lookup.get(ids,{}).get('fp32_islands')),fp32_islands_pass=bm.get('fp32_islands_pass'),deployment_accuracy_qualified=bool(source_ok and acc.get('user_0p02_pass'))))
    intervals=[]
    for arch in ('B0','R4','R4_micro'):
        for precision in ('fp16','mixed','pure'):
            slots=[s for s in manifest['primary_matrix'] if s['architecture']==arch and s['precision']==precision];assert len(slots)==4
            for ds in DATASETS:
                if not all(s.get('engine') and (s['engine'],ds) in evaluated and (f"{arch}_s{s['seed']}_fp32",ds) in evaluated for s in slots):continue
                for metric in ('water_iou','precision','recall'):
                    arrays=[];names=None
                    for slot in sorted(slots,key=lambda s:s['seed']):
                        a={r['event']:r for r in evaluated[slot['engine'],ds]['per_event']};b={r['event']:r for r in evaluated[f"{arch}_s{slot['seed']}_fp32",ds]['per_event']};ev=sorted(a);assert ev==sorted(b)
                        if names is None:names=ev
                        assert names==ev
                        arrays.append([a[e][metric]-b[e][metric] if a[e][metric] is not None and b[e][metric] is not None else np.nan for e in ev])
                    a=np.array(arrays);valid=np.isfinite(a).all(0);d=a[:,valid].mean(0)
                    if not len(d):continue
                    lo=hi=None
                    if len(d)>1:
                        rng=np.random.default_rng(1729);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
                    intervals.append(dict(architecture=arch,precision=precision,dataset=ds,metric=metric,seeds=4,events=len(d),excluded_undefined_events=int((~valid).sum()),delta_pp=float(d.mean()*100),ci_low_pp=float(lo*100) if lo is not None else '',ci_high_pp=float(hi*100) if hi is not None else '',scope='whole-event conditional on4 fitted seeds; Bolivia descriptive; no equivalence claim'))
    summaries=[]
    for arch in ('B0','R4','R4_micro'):
        for precision in ('fp32','fp16','mixed','pure'):
            slots=[z for z in manifest['primary_matrix'] if z['architecture']==arch and z['precision']==precision];ids={z.get('engine') for z in slots}
            for ds in DATASETS:
                rows=[r for r in accuracy if r['id'] in ids and r['dataset']==ds]
                if len(rows)!=4:continue
                row=dict(architecture=arch,precision=precision,dataset=ds,seeds=4,qualified_seeds=sum(r['deployment_accuracy_qualified'] for r in coverage if r.get('engine') in ids))
                for key in ('pooled_mIoU','pooled_water_IoU','equal_event_water_IoU','equal_event_mIoU','precision_water','recall_water'):
                    vals=[r[key] for r in rows];row[key+'_mean']=float(np.mean(vals)) if all(v is not None for v in vals) else None;row[key+'_seed_sd']=float(np.std(vals,ddof=1)) if all(v is not None for v in vals) else None
                summaries.append(row)
    benchmark_summary=[]
    for arch in ('B0','R4','R4_micro'):
        for precision in ('fp32','fp16','mixed','pure'):
            ids={z.get('engine') for z in manifest['primary_matrix'] if z['architecture']==arch and z['precision']==precision}
            rr=[r for r in bench if r['id'] in ids]
            if len(rr)!=4:continue
            row=dict(architecture=arch,precision=precision,seeds=4)
            for key in ('gpu_p50_ms','gpu_p95_ms','gpu_p99_ms','wall_p50_ms','wall_p95_ms','wall_p99_ms','throughput','engine_bytes','device_memory_bytes','steady_power_w','steady_energy_mJ'):
                vv=[r.get(key) for r in rr];v=[float(z) for z in vv if z is not None and z!='']
                row[key+'_observed_seeds']=len(v)
                if len(v)==4:row[key+'_mean']=float(np.mean(v));row[key+'_seed_sd']=float(np.std(v,ddof=1))
            benchmark_summary.append(row)
    csvout(out/'benchmark_four_seed_summary.csv',benchmark_summary)
    csvout(out/'accuracy_four_seed_summary.csv',summaries)
    csvout(out/'accuracy.csv',accuracy);csvout(out/'benchmark.csv',bench);csvout(out/'dry_scene_summary.csv',dryrows);csvout(out/'wf_dry_per_image.csv',wfdry);csvout(out/'coverage_matrix.csv',coverage);csvout(out/'paired_event_intervals.csv',intervals);csvout(out/'wiring_check.csv',wiring)
    completed=sum(r['all_five_datasets_and_dry_complete'] and r['benchmark_complete'] for r in coverage);unsupported=sum(not r.get('engine') for r in coverage)
    all_resolved=set(states)==set(lookup) and all(s['status'] not in ('STARTED','NOT_RUN') for s in states.values())
    probe=read(out/'ATTENTION_PROBE_STATUS.json') if (out/'ATTENTION_PROBE_STATUS.json').exists() else {}
    report=dict(status='ALL_SUPPORTED_CELLS_OBSERVED' if completed+unsupported==48 and len(wiring)==12 and all(r['pass_0p005'] for r in wiring) and all_resolved and probe.get('returncode')==0 and not invalid else 'INCOMPLETE_OR_WIRING_FAILURE',matrix_slots=48,observed_complete_primary_cells=completed,unsupported_or_unavailable_slots=unsupported,engine_attempts=len(manifest['engines']),all_declared_attempts_resolved=all_resolved,invalid_results=invalid,accuracy_qualified_primary_cells=sum(r['deployment_accuracy_qualified'] for r in coverage),failed_attempts=[s for s in states.values() if s['status'].startswith('FAILED')],source_qualification_and_native_acceptance='Separate; neither overrides a recorded export/source failure',power='Use steady rail values; rail identity inferred; startup-diluted averages are secondary only',heldback97='NOT_READ')
    (out/'COLLECTION_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['status']=='ALL_SUPPORTED_CELLS_OBSERVED' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();raise SystemExit(main(a.attempt))
