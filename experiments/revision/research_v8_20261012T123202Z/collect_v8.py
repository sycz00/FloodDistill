"""Idempotent CPU collection on completion, failure, budget closure or stop."""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import audited_queue as q

ROOT=Path(__file__).resolve().parent


def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r));temp=Path(p).with_suffix('.tmp')
    with temp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temp,p)
def numeric(v):return float(v) if v not in ('',None) else None


def collect(reason):
    c=read(ROOT/'config.json',{});state=read(ROOT/'state.json',{});budget=read(ROOT/'budget_ledger.json',{})
    rows=[]
    for p in [ROOT/'adaptation_fp.csv',ROOT/'adaptation_q.csv']:
        if p.exists():rows+=csvread(p)
    existing={(r['event'],int(r['seed']),int(r['support_draw']),r['arm']) for r in rows}
    for e in c.get('episodes',[]):
        for s in c.get('seeds',[]):
            for d in c.get('support_draws',[]):
                for arm in ['F0','F1','F2_1','F2_5','F2_20','F3','F4','B_supervised','B_teacher_label','B_supervised_SGD20','Qbase','Q0','Q1','Q2','Q3']:
                    if (e['event'],s,d,arm) not in existing:
                        job='quant_adaptation' if arm.startswith('Q') else 'fp_adaptation'
                        rec=state.get('jobs',{}).get(job,{})
                        rows.append(dict(event=e['event'],seed=s,support_draw=d,arm=arm,status=rec.get('status','not_run'),
                            missing_reason=rec.get('reason') or 'Job has not produced terminal results'))
    csvwrite(ROOT/'adaptation_results.csv',rows,['event','seed','support_draw','arm','status']+list(dict.fromkeys(k for r in rows for k in r if k not in ['event','seed','support_draw','arm','status'])))
    comparisons=[]
    by={(r['event'],str(r['seed']),str(r['support_draw']),r['arm']):r for r in rows if r['status']=='completed'}
    for r in rows:
        if r['status']!='completed':continue
        comparator='Qbase' if r['arm'].startswith('Q') else 'F0'
        b=by.get((r['event'],str(r['seed']),str(r['support_draw']),comparator))
        if not b:continue
        for metric in ['mIoU','water_IoU','water_precision','water_recall']:
            a0,b0=numeric(r.get(metric)),numeric(b.get(metric))
            comparisons.append(dict(event=r['event'],seed=r['seed'],support_draw=r['support_draw'],arm=r['arm'],
                comparator=comparator,metric=metric,difference_pp=100*(a0-b0) if a0 is not None and b0 is not None else None,
                uncertainty='descriptive paired event/seed/support; no pixel independence claim'))
    csvwrite(ROOT/'comparisons.csv',comparisons,['event','seed','support_draw','arm','comparator','metric','difference_pp','uncertainty'])
    frontier=csvread(ROOT/'compression_anchors.csv') if (ROOT/'compression_anchors.csv').exists() else []
    for name in ['measure_FP32','measure_FP16']:
        r=read(ROOT/'receipts'/f'{name}.json')
        if r:
            frontier.append({**{k:r.get(k) for k in ['graph','seed','precision','parameters','artifact_path','artifact_sha256','deployed_artifact_bytes','p50_ms','p95_ms','fresh_process','incremental_allocator_bytes','edge_latency']},
                'status':'completed','source_mIoU':r['source_scores']['mIoU'],'source_water_IoU':r['source_scores']['water_IoU'],
                'backend':json.dumps(r['backend'],sort_keys=True)})
    for r in rows:
        if r['status']=='completed':
            frontier.append(dict(graph='R4',seed=r['seed'],precision=r.get('precision'),arm=r['arm'],event=r['event'],
                status='completed',artifact_path=r.get('artifact_path'),serialized_checkpoint_bytes=r.get('artifact_bytes'),
                deployed_artifact_bytes=None,water_IoU=r.get('water_IoU'),mIoU=r.get('mIoU'),
                cold_adaptation_seconds=r.get('cold_end_to_end_seconds'),
                limitation='FP32 master weights and quantizer metadata are not packed low-bit deployment; adaptation checkpoints are not measured engines'))
    csvwrite(ROOT/'compression_frontier.csv',frontier,['graph','seed','precision','status']+list(dict.fromkeys(k for r in frontier for k in r if k not in ['graph','seed','precision','status'])))
    deployment=[r for r in frontier if r.get('deployed_artifact_bytes')]
    deployment.append(dict(graph='R4',precision='native_INT8',status='unsupported',
        limitation='Inherited TensorRT CUDA initialization error35, cu13 runtime vs driver535.183.01; no new repair/download attempted'))
    csvwrite(ROOT/'deployment.csv',deployment)
    group=collections.defaultdict(list)
    for r in comparisons:
        if r['metric']=='mIoU' and r['difference_pp'] is not None:group[r['arm']].append(float(r['difference_pp']))
    means={k:statistics.mean(v) for k,v in group.items()}
    failed=[(k,v.get('reason')) for k,v in state.get('jobs',{}).items() if v['status'] in ['technical_failure','blocked']]
    negative=['Implementation/protocol issues are separate from measured method effects.']
    negative += [f'- {k}: {why}' for k,why in failed]
    negative += [f'- {arm}: mean descriptive query mIoU change {value:+.4f} pp; worsening episodes retained in comparisons.csv.' for arm,value in means.items() if value<0]
    negative += ['- Second teacher deferred: installed task head and compatible paired SAR/optical inference are not verified.',
        '- Competent full QAT deferred under the four-hour cap; new Q arms are conventional fixed-PTQ simulation controls.',
        '- Independent dry-reference assessment remains unresolved; adaptation does not resolve it.',
        '- Actual reviewer letter and test.txt not located; exact wording and additional demands remain unverified.',
        '- Original N100 versus N2500 confounds scene count and optimizer work; crossing time is missing when validation timestamps are absent.',
        '- Native INT8 is unsupported in the inherited backend. Edge latency is unmeasured.']
    (ROOT/'negative_results.md').write_text('\n'.join(negative)+'\n')
    report=['# V8 useful corrections, single-solve adaptation and compression',f'Collected {q.now()} after {reason}.',
        f"Worker state: {state.get('worker_status','not_launched')}; active job: {state.get('active_job')}. Results are partial until terminal receipts exist.",
        f"Cumulative GPU-command hours: {budget.get('cumulative_gpu_seconds',0)/3600:.6f}; incremental completed/preflight hours: {budget.get('incremental_spent_seconds',0)/3600:.6f}. Active elapsed and remaining reservations are in ledger.json.",
        '## Observations',f"Completed adaptation rows: {sum(r['status']=='completed' for r in rows)} of {len(rows)} predeclared rows."]
    if means:
        report+=['Mean descriptive query mIoU differences versus the corresponding frozen parent (pp):']+[f'- {a}: {v:+.4f}' for a,v in sorted(means.items())]
    else:report+=['No terminal adaptation result is available yet.']
    report += ['Every event, seed, support draw and arm remains in adaptation_results.csv. Equal-event differences are descriptive; no population CI is estimated from three events.',
        '## Interpretation and next causal test',
        'Useful teacher correction and harmful imitation are computed from same-pixel joint errors in teacher_headroom.csv, event_headroom.csv and joint_confusions.json. G-H is a pixel-accuracy identity; IoU is recomputed from integer confusion counts.',
        'Oracle outputs are unattainable label-aware diagnostics. Stability/confidence is a proposed proxy and can weight confidently wrong teacher predictions.',
        'If uniform ridge fails, inspect frozen-head expressiveness, support/query shift and target bias. If uniform ridge helps without reliability gain, retain the solver and reject this proxy in this setting. If calibration matches the solver, interpret it as an operating-point correction.',
        'Q0–Q3 retain the same fixed quantized backbone and float32 classifier. Q3 can support an interaction only if it exceeds Q0, Q1 and Q2 on matched episodes. This pilot uses fake quantization and establishes neither integer execution nor speedup.',
        'All external benchmarks are research-exposed. Adaptation uses no query labels for fitting or checkpoint selection. The teacher inherits 252 labelled source scenes; zero target-label adaptation is not a label-free pipeline.',
        'One-shot supervised arms use exactly one fully labelled support chip per episode; target_label_pixels and area disclose the annotation budget. A single solve excludes neither teacher queries nor feature extraction; cold and cached cost components are separate.',
        'Feature means/scales and quantizer ranges use TRAIN only. Lambda and SGD learning rate use disjoint S11 validation development pairs before target query scoring. Source labels used for hyperparameter selection are disclosed in choice_s42.json and choice_s64.json.',
        '## Provenance and limitations',
        'The restored original N2500 membership has zero replacements and SHA256 1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60. Historical generation-time byte identity remains unverified.',
        'V7 C3 is retained as a negative scientific result and is not repeated. Historic EfficientViT original-activation OFA curves remain separate from repaired ReLU6 capacity graphs.',
        'FP32/FP16 deployment measurements use separate fresh processes, actual recorded TF32 flags, a common input, 20 warmups and 100 synchronized timings. They are RTX4090 forward timings and exclude input transfer/normalization; edge deployment is unmeasured.',
        'Training checkpoint size is never substituted for deployed artifact size. Low-bit master checkpoints include floating-point tensors and quantizer metadata; no theoretical packed-size claim is made.',
        'See negative_results.md for missing reviewer letter, test.txt, independent dry references, second-teacher inputs and unsupported INT8.',
        '## Reproduction',f'`{sys.executable} {ROOT}/jobs_v8.py recover --out {ROOT}/receipts/recovered_again.json`',
        f'`{sys.executable} {ROOT}/collect_v8.py --reason manual`',
        'GPU task argv, input hashes, dependencies and timeouts are frozen in manifest.json; attempts and exact measured durations remain in the ledger.',
        '## Deliverables','evidence_index.json; frozen_protocol.md; manifest.json; ledger.json; status.json; joint_confusions.json; teacher_headroom.csv; event_headroom.csv; ofa_learning_efficiency.csv; ofa_learning_curves.csv; adaptation_results.csv; comparisons.csv; compression_frontier.csv; deployment.csv; reviewer_response_map.md; negative_results.md.']
    (ROOT/'report.md').write_text('\n\n'.join(report)+'\n')
    if (ROOT/'teacher_headroom.csv').exists():
        hr=csvread(ROOT/'teacher_headroom.csv');pooled=[r for r in hr if r['aggregation']=='pooled_pixel_counts' and r['comparator']=='T|S' and r['scope']=='full']
        lines=['# Teacher headroom','Observed pooled correction and harm rates below. They do not select any deployable threshold.']
        for r in pooled:lines.append(f"- {r['dataset']}, seed {r['seed']}: G={float(r['G']):.6f}, H={float(r['H']):.6f}; oracle water IoU={r['oracle_water_IoU']}; remaining both-wrong pixels={r['both_wrong_pixels']}.")
        lines += ['Inspect every event and class in event_headroom.csv. A pooled oracle improvement measures complementary information, not whether a frozen linear head or deployable router can access it.','Reliability-proxy errors on actual support reference labels are offline diagnostics in support_cache/manifest.json; they never fit a reliability model.']
        (ROOT/'headroom_interpretation.md').write_text('\n\n'.join(lines)+'\n')
    else:(ROOT/'headroom_interpretation.md').write_text('No completed joint-error inference receipt yet; no headroom conclusion is available.\n')
    plots(rows)
    q.atomic_json(ROOT/'completion_receipt.json',dict(at=q.now(),reason=reason,worker_status=state.get('worker_status'),
        complete=state.get('worker_status') in ['finished','stopped','budget_closed'],jobs={k:v['status'] for k,v in state.get('jobs',{}).items()},
        report=str(ROOT/'report.md'),report_sha256=q.file_hash(ROOT/'report.md'),failed=failed,
        budget=budget,adaptation_rows_completed=sum(r['status']=='completed' for r in rows)))
    (ROOT/'handover.md').write_text(f"Run: {ROOT}\n\nWorker: {state.get('worker')}; GPU {state.get('gpu_uuid')}; deadline {state.get('deadline_at')}.\n\n"+
        '\n'.join(f'`{sys.executable} {ROOT}/run_overnight.py {cmd} --run-dir {ROOT}`' for cmd in ['status','pause','resume','stop'])+
        f'\n\n`tail -f {ROOT}/worker.log`\n\nThe host must stay awake. Detachment survives the launcher exiting; it does not provide reboot/sleep persistence.\n')


def plots(rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        p=ROOT/'ofa_learning_curves.csv'
        if p.exists():
            curves=csvread(p)
            if curves:
                fig,ax=plt.subplots(figsize=(8,4))
                for jid in sorted({r['run_id'] for r in curves}):
                    rs=[r for r in curves if r['run_id']==jid]
                    ax.plot([float(r['optimizer_updates']) for r in rs],[100*float(r['source_validation_mIoU']) for r in rs],alpha=.45)
                ax.set(xlabel='Optimizer updates (observed)',ylabel='Source validation mIoU (%)',title='Historical matched LKD/OFA; per-validation timing unavailable')
                fig.tight_layout();fig.savefig(ROOT/'ofa_learning_efficiency.png',dpi=160);plt.close(fig)
        done=[r for r in rows if r['status']=='completed' and r.get('mIoU')]
        if done:
            fig,ax=plt.subplots(figsize=(9,4));arms=sorted({r['arm'] for r in done})
            for i,arm in enumerate(arms):
                rs=[r for r in done if r['arm']==arm]
                ax.scatter([i]*len(rs),[100*float(r['mIoU']) for r in rs],s=12,alpha=.6)
            ax.set_xticks(range(len(arms)),arms,rotation=60,ha='right');ax.set_ylabel('Query event mIoU (%)');ax.set_title('All exploratory seed/support/event outcomes')
            fig.tight_layout();fig.savefig(ROOT/'adaptation_events.png',dpi=160);plt.close(fig)
    except Exception as e:q.atomic_json(ROOT/'plot_failure.json',dict(error=str(e)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default=str(ROOT));p.add_argument('--reason',default='manual');a=p.parse_args();collect(a.reason)
