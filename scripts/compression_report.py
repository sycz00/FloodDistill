"""Render the completed bounded compression pilot from immutable receipts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/revision/compression_v2_20261007'
OUT = BASE / 'report_20261008'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    d = read(OUT / 'pilot_collection_verified.json')
    repairs = sorted(OUT.glob('repair_terminal_*.json'))
    assert len(repairs) == 12
    audit = {'status': 'passed_with_disclosed_limitations', 'checks': {}, 'support': {},
             'receipt_sha256': {}, 'queue_states': {}, 'gpu_command_seconds': {}}
    gpu_seconds = 0.
    for folder in ('preflight', 'timing', 'mb_timing', 'pilot_final',
                   'recovery_20261008', 'export_recovery_20261008', 'export_remaining_20261008'):
        p = BASE / folder / 'manifest.state.json'
        s = read(p)
        counts = dict(Counter(j['status'] for j in s['jobs'].values()))
        assert not set(counts) & {'pending', 'running', 'blocked'}
        audit['queue_states'][folder] = {'worker_status': s['worker_status'], 'job_states': counts,
                                        'state_sha256': sha(p)}
        seconds = 0.
        for j in s['jobs'].values():
            for a in j.get('attempts', []):
                if a.get('gpu_device'):
                    seconds += float(read(a['receipt_path']).get('command_elapsed_seconds', 0.))
        audit['gpu_command_seconds'][folder] = seconds
        gpu_seconds += seconds
    support = {}
    design = read(BASE / 'results/source_design_freeze.json')
    for path, digest in design['receipt_hashes'].items():
        assert sha(path) == digest
    gate = read(BASE / 'results/mechanism_source_gate.json')
    assert gate['eligible'] is False and gate['source_only'] and gate['no_external_scores_read']
    assert gate['technical_fallback_used'] is False
    for graph in ('A0', 'A1', 'A3'):
        warmup = read(BASE / 'results' / f'{graph}_FP_s42.json')
        assert warmup['completed_updates'] == 7800 and warmup['completed_epochs'] == 50
    for p in repairs:
        recovery = read(p); original = Path(recovery['terminal_receipt'])
        assert sha(original) == recovery['terminal_receipt_sha256']
        assert recovery['new_training'] is False and recovery['new_benchmark_evaluation'] is False
        t = read(original)
        freeze = read(original.with_suffix('.freeze.json'))
        assert datetime.fromisoformat(design['frozen_at']) < datetime.fromisoformat(freeze['at'])
        assert freeze['checkpoint_sha256'] == t['checkpoint_sha256']
        assert freeze['source_design_frozen']
        assert t['checkpoint_sha256'] == recovery['checkpoint_sha256']
        a = recovery['artifact']
        assert a['runtime']['status'] == 'measured'
        assert a['runtime']['graph_executor_optimization'] is False
        assert a['runtime']['warmups'] == 20 and a['runtime']['repeats'] == 200
        assert a['artifact']['status'] == 'exported_prediction_validated'
        audit['receipt_sha256'][str(original)] = sha(original)
        audit['receipt_sha256'][str(p)] = sha(p)
        for dataset, domains in t['terminal_results'].items():
            for domain, cell in domains.items():
                if not isinstance(cell, dict) or 'map' not in cell:
                    continue
                maps = sum((np.asarray(x, dtype=np.int64) for x in cell['map'].values()),
                           np.zeros((2, 3), dtype=np.int64))
                events = sum((np.asarray(x, dtype=np.int64) for x in cell['event'].values()),
                             np.zeros((2, 3), dtype=np.int64))
                assert np.array_equal(maps, events)
                key = dataset + '/' + domain
                gt = tuple((maps[:, 0] + maps[:, 2]).tolist())
                assert key not in support or support[key] == gt
                support[key] = gt
                ious = maps[:, 0] / (maps.sum(axis=1))
                assert abs(float(ious.mean()) - cell['micro']['mIoU']) < 1e-12
                assert abs(float(ious[1]) - cell['micro']['water_IoU']) < 1e-12
                assert len(cell['event']) == cell['n_events'] and len(cell['map']) == cell['n_maps']
    audit['support'] = {k: {'background_pixels': v[0], 'water_pixels': v[1],
                            'valid_pixels': sum(v)} for k, v in support.items()}
    assert sum(support['sturm/full']) == 674781328
    assert sum(support['sturm/known_overlap_excluded']) == 673994896
    assert sum(support['wf/full']) == 142159065
    assert sum(support['wf/known_overlap_excluded']) == 135822668
    common_states = {}
    for row in d['source_post_qat_diagnostics']:
        assert row['completed_updates'] == 1560
        key = (row['graph'], row['precision'])
        state_hash = row['common_quant_state_sha256']
        assert key not in common_states or common_states[key] == state_hash
        common_states[key] = state_hash
        x = row['final_residual_diagnostics']
        assert abs(x['Ce_trace'] - (x['A'] + x['Q'] + 2*x['X'])) < 1e-9
        assert abs(x['eta'] - x['chi_plus'] + x['chi_minus']) < 1e-9
        assert abs(x['eta']) <= 2*np.sqrt(x['rho']*(1-x['rho'])) + 1e-9
    for row in d['interaction_diagnostics']:
        m = row['metric']; ev = np.asarray(m['metric_eigenvalues'])
        assert ev.min() > 0 and ev.max()/ev.min() <= 4 + 1e-8
        assert abs(m['metric_trace'] - 32) < 1e-8
        losses = list(row['initial_feature_losses'].values())
        assert max(losses) - min(losses) < 1e-10
    for row in d['source_metrics']:
        if row.get('H') is not None:
            assert row['identity_verified']
            assert abs(row['accuracy_delta'] - row['B'] + row['H']) < 1e-12
    assert len(d['metrics']) == 72 and len(d['source_post_qat_diagnostics']) == 9
    assert all(x['checkpoint_integrity'] == 'verified' and x['artifact_integrity'] == 'verified'
               for x in d['metrics'])
    audit['checks'] = dict(queue_terminal=True, original_receipts_bound=True,
        twelve_measured_exports=True, checkpoint_and_artifact_hashes_verified=True,
        map_event_count_sums=True, pooled_mIoU_recomputed=True, identical_GT_support=True,
        known_overlap_exclusions=True, nine_complete_QAT_budgets=True,
        SPD_trace_condition_and_initial_loss_match=True, residual_and_GT_flip_identities=True,
        source_freeze_hashes_and_chronology=True, source_gate_negative=True,
        matched_common_quant_state=True, three_complete_FP_budgets=True)
    audit['total_gpu_command_seconds'] = gpu_seconds
    audit['total_gpu_command_hours'] = gpu_seconds / 3600
    audit['limitations'] = ['Native FP16 references remain blocked by float upsampling into half convolutions.',
        'One screening seed; no confirmation or matched-spectrum training after failed source gate.',
        'Fake quantization is FP32 simulation, with no packed low-bit deployment established.']
    (OUT / 'completion_audit.json').write_text(json.dumps(audit, indent=2) + '\n')

    lines = []
    def section(title, prose=''):
        lines.extend(['', title.upper(), '=' * len(title), prose, ''])
    def table(rows, fields):
        labels = [label for _, label, _ in fields]
        values = []
        for row in rows:
            values.append([fmt(row.get(key)) for key, _, fmt in fields])
        widths = [max(len(label), *(len(v[i]) for v in values)) for i, label in enumerate(labels)]
        lines.append('  '.join(label.ljust(w) for label, w in zip(labels, widths)))
        lines.extend('  '.join(v.ljust(w) for v, w in zip(row, widths)) for row in values)
        lines.append('')
    txt = lambda x: '-' if x is None else str(x)
    num = lambda x: '-' if x is None else f'{x:.6f}'
    pct = lambda x: '-' if x is None else f'{100*x:.3f}'

    section('Compression interaction distillation v2: full bounded-pilot report',
        '8 October 2026, Europe/Berlin. All completed pilot results and all intentional skipped conditions are included. '
        'This is a completed one-seed screening pilot, not a completed four-seed confirmation programme. '
        'B1 = ordinary task KD; B2 = isotropic feature MSE; B3 = uncertainty-shrunk interaction feature metric. '
        'Accuracy tables use percentage units; raw diagnostic tables use dimensionless fractions unless labelled otherwise.')
    section('Completion and failures',
        'The original detached queue finished on 7 October at 23:56 Berlin. Its historical state remains 98 succeeded and '
        '1 failed (pilot_collection); 62 successful nodes explicitly skipped work. Three FP students, nine QAT continuations, '
        'three PTQ maps, four interaction fits and twelve terminal checkpoints completed. No training or terminal accuracy '
        'restart is needed. The report collector failed because the frozen capsule omitted figures/fig_style.py, required '
        'by the event bootstrap helper. A new collection uses the same stored counts and includes that dependency. '
        'The original failed receipt and frozen capsule remain intact. Eight exports initially failed strict TorchScript '
        'logit validation; export-only recovery completed for all twelve under a common runtime configuration. '
        'All recovery workers finished. FP16 remains blocked and is never counted as a successful measurement.')
    table([{'queue': k, 'worker': v['worker_status'], 'states': str(v['job_states'])}
           for k, v in audit['queue_states'].items()],
          [('queue', 'Queue', txt), ('worker', 'Worker', txt), ('states', 'Job states', txt)])
    section('Scientific conclusion',
        'The interaction hypothesis is unsupported by this pilot. A1 W4A8, A3 W4A8 and A3 W8A8 have exact M=I after '
        'uncertainty shrinkage, so their B3 interventions equal B2 and duplicate training was skipped. Hard cell A3 W4A4 '
        'retained one weak anisotropic direction. Its source mIoU was B1 69.270%, B2 70.545%, B3 70.269%. '
        'B3 improves over B1 by 0.999 percentage points but falls below B2 by 0.276 points, missing the prespecified '
        '0.5 point gain over BOTH controls. The cell is technically valid; the technical-invalidity fallback is not invoked. '
        'Matched-spectrum controls, additional seeds, beta arms and second-family training were therefore intentionally '
        'not executed. Their missing outcomes cannot be treated as evidence for or against accuracy. '
        'A1 FP32 gives a useful generic architecture-size tradeoff; no interaction-specific compression benefit is established.')
    section('Cohort, teacher and frozen recipe',
        'The cached original N=2500 training cohort was restored, including 127 original sources; zero FPS replacements '
        'were introduced. All 2500 files (4,270,618,908 bytes) were rehashed at preflight. Pool manifest SHA256: '
        '1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60. '
        'This is distinct from earlier regenerated revision cohorts. Historical cache-generation identity gaps remain '
        'in lineage receipts; absent historical hashes cannot prove a historical byte-for-byte cache match. '
        'Frozen Prithvi EO 2.0 300M teacher SHA256: '
        '3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60. '
        'Bands B2/B3/B4/B8A/B11/B12; means [.1413,.138,.1235,.309,.2045,.1191], '
        'std [.0741,.0737,.0869,.118,.0977,.0766]. Cached normalized inputs are not normalized again. '
        'Train crops 224; validation/test geometry 512; two logits and existing valid/ignore masks. '
        'FP students are trained from scratch, FP32, seed42: AdamW lr5e-5 wd.05, batch16/drop-last, cosine, '
        '50 epochs/7800 updates, no AMP, clipping5. Teacher is frozen/eval and evaluated on the same augmented '
        'inputs (flips, rot90, per-channel gain .8-1.2). Task objective .5 focal(gamma2)+.5 T^2 KL(teacher||student), T4. '
        'S11 validation89 selects strict best pooled mIoU every2epochs, earlier ties, patience20 checks. '
        'QAT: AdamW lr1e-5 wd1e-5, clip1, fixed10epochs/1560updates, source selection every2epochs. '
        'QDQ is ON for all10epochs, differing explicitly from the legacy first-epoch-off schedule. '
        'Per cell, all arms start with the identical FP checkpoint, observed ranges, BN/fusion state, fixed adapter '
        'and data RNG; no terminal external outcome was used for selection or queue expansion.')
    section('Predeclared architecture frontier',
        'These are actual smaller dense graphs with legal attention dimensions, retained transition blocks and decoder '
        'geometry; all output two-class 512 maps. Native M0 preserves Hardswish/ReLU and existing decoder ReLU6, '
        'so it is a separate family. A2 was declared before results but its training is deferred. '
        'Existing historical V5 float controls have different graph/activation/lineage and are not exact matched controls.')
    params = {'A0':704714,'A1':190870,'A2':62277,'A3':30922,'M0':3546706}
    table([{'graph': k,'params': params[k], 'width':v.get('width_multiplier'),
            'depth':v.get('repeatable_depth_multiplier'), 'channels':str(v.get('width_list')),
            'status':'FP trained' if k in ('A0','A1','A3') else 'training deferred'}
           for k,v in d['frontier'].items()], [('graph','Graph',txt),('params','Params',txt),
           ('width','Width',txt),('depth','Repeat depth',txt),('channels','Channels',txt),('status','Status',txt)])
    section('Source validation and PTQ precision map',
        'The complete stored source rows below include common-checkpoint PTQ and selected QAT accuracy. '
        'FP warmups all completed7800updates; selected A0/A1/A3 updates5616/7800/4368. '
        'PTQ range calibration uses89 S11 validation images at512/batch1; human labels are not used to fit '
        'PCA/adapter/interaction. W4/W8 refer to floating-point fake quantization. Quantized convolution weights '
        'use per-output-channel symmetric ranges; activations per-tensor symmetric. Final convolution and '
        'attention matrix multiplications remain FP32. Full operator precision maps are in terminal and export receipts.')
    table(d['source_metrics'], [('job_id','Source receipt',txt),('precision','Precision',txt),
        ('mIoU','mIoU %',pct),('water_IoU','Water IoU %',pct),('water_precision','P %',pct),
        ('water_recall','R %',pct),('water_F1','F1 %',pct)])
    table(d['source_post_qat_diagnostics'], [('job_id','QAT',txt),('selected_updates','Selected',txt),
        ('completed_updates','Completed',txt)])
    section('Metric calibration and normalization',
        'A fixed256-image TRAIN fitting pool and disjoint256-image TRAIN interaction pool use one sample per '
        'event group, drawn from1404 available groups. Event IDs approximate geographic independence and can '
        'revisit sites. Teacher features are resampled to8x8 (bilinear/align_corners=False); class/valid masks '
        'use nearest-exact centered sampling. Available classes receive equal mass, then images equal mass. '
        'Teacher feature dimension1024, numerical rank1023; fixed PCA r32 retains92.524455% variance. '
        'Ridge adapter lambda1e-3*trace(Gram)/dimension, bias unpenalized; fit on FP quantization-bypass features '
        'with shared clamp/fusion. Teacher standardization, PCA and adapter remain frozen. '
        'With f=t-s_FP, q=s_FP-s_Q, e=f+q, use uncentered Cf/Cq/R; Ce=Cf+Cq+2R. '
        'Noise shrinkage uses200 whole-event bootstrap resamples (seed1729), 95% operator-norm delta, minimum64groups. '
        'M is SPD, trace32, condition<=4. Each c_K matches the initial fixed-input feature loss, while beta '
        'uses4x16 TRAIN minibatches to target weighted feature-gradient RMS=.1 task RMS on the last encoder stage. '
        'These rules do not equalize all subsequent feature/task gradients.')
    metricrows = []
    for row in d['interaction_diagnostics']:
        m=row['metric'];metricrows.append({'cell':row['job_id'], 'delta':m['delta'],
            'retained':m['retained_directions'],'distance':m['metric_distance_from_identity'],
            'cond':m['metric_condition_number'],'beta':row['beta_rule']['beta'],
            'c':m['initial_loss_scale'],'loss':row['initial_feature_losses']['B3']})
    table(metricrows,[('cell','Cell',txt),('delta','Delta',num),('retained','Kept',txt),
        ('distance','||M-I||F',num),('cond','Cond',num),('beta','Beta',num),('c','c_M',num),('loss','Initial loss',num)])
    hard = d['interaction_diagnostics'][-1]
    lines.append('Hard-cell spectrum: '+json.dumps(hard['metric']['metric_eigenvalues']))
    lines.append('Hard-cell matched loss scales: '+json.dumps(hard['loss_scales'],sort_keys=True))
    lines.append('Hard-cell split stability: '+json.dumps(hard['split_group_stability'],sort_keys=True))
    section('Raw interaction diagnostics before and after QAT',
        'Raw moments are separate from the shrunk training spectrum. A=trace(Cf), Q=trace(Cq), X=trace(R), '
        'eta=2X/(A+Q), rho=Q/(A+Q), Ce/(A+Q)=1+eta, eta=chi_plus-chi_minus. '
        'Negative eta indicates net cancellation, not proof of good labels. Opposing directions can cancel. '
        'On the hard cell B3 has less-negative eta, but larger Q and Ce than B2. '
        'The split-half operator-gap/projector overlap are descriptive directional instability, not selection tests. '
        'These diagnostics are not established OOD predictors.')
    raw=[]
    for row in d['interaction_diagnostics']:
        raw.append({'cell':row['job_id'],**row['metric']['diagnostics']})
    for row in d['source_post_qat_diagnostics']:
        raw.append({'cell':row['job_id'],**row['final_residual_diagnostics']})
    table(raw,[('cell','Moment receipt',txt),('A','A',num),('Q','Q',num),('X','X',num),
        ('Ce_trace','Ce',num),('eta','eta',num),('rho','rho',num),('chi_plus','chi+',num),('chi_minus','chi-',num)])
    section('Ground-truth flips and training gradients',
        'H=P(FP correct,Q wrong), B=P(FP wrong,Q correct), using the same post-QAT weights with quantization '
        'bypassed versus enabled, identical valid labels and decision rule. Accuracy_Q-Accuracy_FP=B-H exactly; '
        'IoU does not obey this identity. Teacher disagreement is a different diagnostic. '
        'Hard-cell B3 has5142 more harmful flips and13961 fewer repairs than B2: no GT mechanistic advantage. '
        'There are no independent fully dry references; no dry-scene safety claim. No invented boundary benchmark.')
    table([x for x in d['source_metrics'] if x.get('H') is not None],
        [('job_id','QAT receipt',txt),('harmful_count','H count',txt),('repaired_count','B count',txt),
         ('H','H',num),('B','B',num),('accuracy_delta','B-H',num),
         ('teacher_disagreement_fraction','Teacher disagreement',num)])
    gradients=[]
    for row in d['source_post_qat_diagnostics']:
        g=row['gradient_diagnostics']
        if g:
            ratios=[x['weighted_feature_gradient_rms']/max(x['task_gradient_rms'],1e-12) for x in g]
            gradients.append({'job':row['job_id'],'n':len(g),'min':min(ratios),'median':float(np.median(ratios)),
                              'max':max(ratios),'firsttask':g[0]['task_gradient_rms'],
                              'firstfeature':g[0]['feature_gradient_rms']})
    table(gradients,[('job','Gradient receipt',txt),('n','Records',txt),('min','Ratio min',num),
        ('median','Ratio median',num),('max','Ratio max',num),('firsttask','First task RMS',num),
        ('firstfeature','First feature RMS',num)])
    lines.append('B1 has no feature gradient. Full per-step diagnostics and epoch pre-clipping gradient norms, losses and clipped fractions are retained in pilot_collection_verified.json. Initial c_K matching is on fixed unaugmented fit inputs; augmented first-batch losses/gradients can differ. Large pre-clipping norms and frequent clipping constrain mechanistic interpretation.')
    section('Source paired analysis',
        'Conditional paired bootstrap:10000 whole-event resamples, seed1729. Source validation also selects '
        'checkpoints, so these intervals are descriptive and potentially optimistic. They do not provide '
        'independent validation or training-seed replication. Pooled micro deltas and event-macro intervals '
        'are distinct estimands; intervals below apply only to event-macro means.')
    for row in d['source_comparisons']:
        lines.append(json.dumps(row,sort_keys=True))
    section('Terminal benchmark protocols and all accuracy results',
        'S11 test90maps/10events, legacy512. Corrected STURM2675chips/46maps/29events with finite six-band '
        'valid mask. WF785 native zero-padded512windows/18maps/11events. Full and known-overlap-excluded '
        'domains are kept separate. STURM removes three preidentified chips (2672 remaining); WF removes '
        'EMSR438_AOI02_DEL_MONIT03 (733windows/17maps/11events). The author disclosed prior unsuccessful '
        'teacher-derived STURM criteria; these are terminal readouts after source design freeze, not a '
        'claim that external benchmarks were historically untouched. All models use identical GT support. '
        'S11 full/excluded are identical where no preidentified exclusions apply. Accuracy below is pooled '
        'valid-pixel micro scoring; map/event counts and conditional event intervals remain explicit.')
    table([{'domain':k,**v} for k,v in audit['support'].items()],
        [('domain','Domain',txt),('valid_pixels','Valid pixels',txt),('water_pixels','Water pixels',txt)])
    for dataset in ('s11','sturm','wf'):
        for domain in ('full','known_overlap_excluded'):
            rows=[x for x in d['metrics'] if x['dataset']==dataset and x['domain']==domain]
            lines.extend(['',dataset.upper()+' / '+domain])
            table(rows,[('job_id','Checkpoint',txt),('mIoU','mIoU %',pct),('water_IoU','Water IoU %',pct),
                ('water_precision','P %',pct),('water_recall','R %',pct),('water_F1','F1 %',pct),
                ('n_images','Images',txt),('n_maps','Maps',txt),('n_events','Events',txt)])
    section('Terminal paired effects and interpretation',
        'B3-B2 pooled mIoU: S11 test -1.530 points; full STURM -.560 points; full WF +.156 points. '
        'Known-overlap exclusion preserves the direction (STURM -.561, WF +.149). External event-macro '
        '95% intervals all cross zero. The small WF gain does not rescue the source gate. '
        'One seed, nonsignificance and heterogeneous external effects do not establish equivalence, '
        'generalization improvement or mechanism confirmation. No method redesign or expansion used these readouts.')
    table(d['comparisons'],[('dataset','Dataset',txt),('domain','Domain',txt),('control','B3 minus',txt),
        ('mIoU_pooled_delta','Micro delta %',pct),('event_macro_mIoU_delta','Macro delta %',pct),
        ('event_macro_mIoU_delta_ci95','Macro 95% CI',lambda x: '-' if x is None else f'[{100*x[0]:+.3f}, {100*x[1]:+.3f}]'),
        ('n_events','Events',txt)])
    section('Measured export and runtime frontier',
        'A1 FP32 is the smallest eligible measured artifact:1,183,268 bytes, source83.595%, versus A0 '
        '84.155%, inside the predeclared1 point tolerance. It is2.745x smaller by serialized inference bytes '
        '(A0 3,247,524bytes), supporting generic dense scaling. A3 FP32 falls outside tolerance; no fake-quant '
        'candidate qualifies. This source-only selector does not use test/OOD accuracy. '
        'All final measurements use loaded TorchScript/PyTorch CUDA on RTX4090, batch1x6x512x512 FP32, '
        'graph executor optimization disabled,20warmups and200 individually synchronized repeats. '
        'Latency is this backend/device result, not XavierNX or a packed INT4/INT8 kernel. Fake-quant exports '
        'retain floating weights plus quantizer overhead; simulated bitwidth does not equal compressed engine bytes. '
        'Memory is the process peak (including loaded model/probe/resident state) and additional forward allocation, '
        'not a whole-device hardware peak. MACs include convolution plus attention matmul at actual512 shape.')
    measured=[]; memory=[]
    for p in repairs:
        r=read(p);a=r['artifact'];run=a['runtime'];orig=read(r['terminal_receipt'])
        matching=next(x for x in d['metrics'] if x['job_id']==Path(r['terminal_receipt']).stem)
        measured.append({'job':Path(r['terminal_receipt']).stem,'params':orig['parameters'],
            'bytes':a['artifact']['serialized_bytes'],'median':run['latency_ms_median'],
            'p95':run['latency_ms_p95'],'macs':matching['total_macs']})
        memory.append({'job':Path(r['terminal_receipt']).stem,'peakalloc':run['process_peak_allocated_bytes'],
            'peakreserve':run['process_peak_reserved_bytes'],'extra':run['additional_peak_allocated_bytes'],
            'uuid':run['visible_gpu_uuid']})
    table(measured,[('job','Export',txt),('params','Params',txt),('bytes','Bytes',txt),('macs','MACs',txt),
        ('median','Median ms',num),('p95','P95 ms',num)])
    table(memory,[('job','Export',txt),('peakalloc','Peak allocated',txt),('peakreserve','Peak reserved',txt),
        ('extra','Additional allocated',txt)])
    section('Export repairs and unsupported FP16',
        'A warmed original optimized TorchScript plan on the A3 native artifact produced max logit difference '
        '.00070810318 on the zero probe, exceeding unchanged FP32 tolerance1e-5, despite identical argmax. '
        'Disabling graph executor optimization produced zero difference. The first cold diagnostic passed both '
        'plans; warm execution history was required to reproduce the mismatch. Export validation now warms20times '
        'before comparing all random224/random512/zero512 probes, strict closeness and exact predictions. '
        'Eight initially blocked exports were repaired without changing weights, training, masks or terminal '
        'accuracy; four previously valid exports were also remeasured using the same setting. Each repair is bound '
        'to the original terminal receipt SHA256 and checkpoint SHA256. Original files/receipts are preserved. '
        'Native FP16 still raises Input type FloatTensor and weight type HalfTensor because UpSampleLayer casts '
        'half inputs to float before a half convolution. No graph or precision workaround is silently introduced. '
        'FP16 accuracy/latency/frontier references remain unsupported; they are an outstanding deployment task.')
    section('Compute cost, tests and evidence scope',
        f'Total measured GPU-command wall time across original preflights/pilot and export-only recovery is '
        f'{gpu_seconds:.3f}s ({gpu_seconds/3600:.4f} summed GPU-hours), below the12GPU-hour bound. '
        'Command time includes process overhead and is not kernel-active utilization; two devices can overlap. '
        'Original main/preflight reserved upper bound42,960s; recovery reservations60+780+360s were admitted '
        'after completed actual costs, not appended as a claim that the original worst-case reservation remains. '
        'Training/evaluation budgets and precisions were unchanged. CPU collection/report rendering require no GPU. '
        'Capsule/import regression and export-override identity checks passed in collector_recovery_check_v2.json; '
        'existing preflight math/graph/QDQ/teacher/checkpoint tests are preserved. completion_audit.json verifies '
        'receipt binding, all12 measured artifact identities, pooled count-derived mIoU, equal map/event sums, '
        'identical support and exclusions,9 complete QAT budgets, SPD/trace/condition and initial-loss matching, '
        'residual bounds and the GT flip identity. This audit verifies completed evidence, not skipped experiments.')
    table([{'action':k,'seconds':v} for k,v in d['cost_seconds_by_action'].items()],
        [('action','Original action',txt),('seconds','Command seconds',num)])
    table([{'queue':k,'seconds':v} for k,v in audit['gpu_command_seconds'].items()],
        [('queue','GPU queue',txt),('seconds','Command seconds',num)])
    section('Outstanding evidence and novelty gates',
        'The full scientific programme remains unconfirmed. Random orientation, quant-only, independent-error '
        'and TOTAL-error metric training did not run after the negative source gate; four-seed confirmation '
        '(42/64/91/107), beta sensitivity, M0 family training and A2 are missing/deferred. Faithful QFD/SQuaT '
        'baseline, time-matched ordinary KD, matched ReLU6 OFA and focused primary-source related-work comparison '
        'remain required before novelty claims. These are not scheduled pending jobs in a live worker. '
        'No interaction advantage beyond MSE, packed low-bit benefit, independently dry safety or device-port '
        'performance is established. The strongest supported conclusions are legal generic dense scaling and '
        'a valid negative one-seed mechanism screen under the frozen protocol.')
    section('Complete recorded upstream job ledger',
        'The collector records98 upstream nodes; the original collector itself is the preserved99th failed node. '
        'Completed35 also contains calibration, interaction, terminal and technical preflight receipts; it is not '
        'a count of35 training experiments. All skipped reasons are retained in the outcomes CSV and JSON.')
    table(d['records'],[('job_id','Job',txt),('action','Action',txt),('outcome','Outcome',txt)])
    section('Reviewable artifact index',
        'compression_v2_full_report.txt and .pdf contain the same full narrative/tables; PDF adds three figures. '
        'pilot_collection_verified.json is the complete machine-readable collection, with full training diagnostics, '
        'precision-map references and checksummed file identities. Its *_metrics, *_source_metrics, '
        '*_comparisons, *_source_comparisons and *_outcomes CSVs provide reusable tables. '
        'completion_audit.json records count/hash checks and current terminal queue snapshots. '
        'verified_collection_spec.json binds all twelve export-only overrides; repair_terminal_*.json retain '
        'full warmed validation probes, precision maps, export SHA/size, runtime/device identity and FP16 errors. '
        'Original design/configs and queue live under pilot_final; frozen training capsule140df39f28974301 '
        'is immutable. Large weights/exports remain outside git under /data1/schmalstieg/flooddistill/revision_20261005/. '
        'Historical failures and the first recovered collection are preserved. The bundle includes the task '
        'specification, collection CSV/JSON, recovery receipts, audit, test receipt and report figures.')
    (OUT / 'compression_v2_full_report.txt').write_text('\n'.join(lines)+'\n')

    figures=[]
    fig, ax=plt.subplots(figsize=(9,5))
    labels=['Source val','S11 test','STURM full','WF full']
    x=np.arange(4)
    for i,method in enumerate(('B1','B2','B3')):
        source=next(r for r in d['source_metrics'] if r['job_id']==f'A3_W4A4_{method}_s42' and r['precision']=='W4A4')
        ys=[100*source['mIoU']]+[100*next(r for r in d['metrics'] if r['job_id']==f'terminal_A3_W4A4_{method}_s42' and r['dataset']==ds and r['domain']=='full')['mIoU'] for ds in ('s11','sturm','wf')]
        ax.bar(x+(i-1)*.24,ys,.24,label=method)
    ax.set(xticks=x,xticklabels=labels,ylabel='Pooled mIoU (%)',ylim=(0,100),title='A3 W4A4: completed one-seed screening')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'hard_cell_accuracy.png',dpi=180);figures.append(fig)
    fig, ax=plt.subplots(figsize=(9,5))
    cs=[r for r in d['source_comparisons'] if r['control']=='B2']+[r for r in d['comparisons'] if r['control']=='B2' and r['domain']=='full']
    for i,r in enumerate(cs):
        y=100*r['event_macro_mIoU_delta'];lo,hi=np.asarray(r['event_macro_mIoU_delta_ci95'])*100
        ax.errorbar(y,i,xerr=np.array([[y-lo],[hi-y]]),fmt='o',capsize=4)
    ax.axvline(0,color='black',lw=1);ax.set(yticks=np.arange(len(cs)),yticklabels=[r.get('dataset','source validation') for r in cs],xlabel='B3 minus B2 event-macro mIoU (percentage points)',title='Conditional paired-event 95% intervals; fixed selected checkpoints')
    fig.tight_layout();fig.savefig(OUT/'event_macro_intervals.png',dpi=180);figures.append(fig)
    fig, ax=plt.subplots(figsize=(11,5))
    for r in d['deployment_summary']['candidates']:
        if r['precision']=='FP16' or r['inference_artifact_bytes'] is None or r['source_validation_mIoU'] is None:continue
        ax.scatter(r['inference_artifact_bytes']/1024,100*r['source_validation_mIoU'],
                   marker='o' if r['precision']=='FP32' else 'x',
                   label=r['job_id'].removeprefix('terminal_').removesuffix('_s42'))
    anchor=d['deployment_summary']['anchors'][0]['source_validation_mIoU']
    ax.axhline(100*(anchor-.01),ls='--',color='black',label='A0 source minus1 point')
    ax.set(xscale='log',xlabel='Verified serialized artifact size (KiB, logarithmic)',ylabel='Source validation mIoU (%)',title='Measured FP32 and fake-quant frontier; source-only selection')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'measured_size_frontier.png',dpi=180);figures.append(fig)
    wrapped=[]
    for line in lines:
        wrapped += textwrap.wrap(line,width=119,replace_whitespace=False,drop_whitespace=False) or ['']
    with PdfPages(OUT/'compression_v2_full_report.pdf') as pdf:
        for start in range(0,len(wrapped),67):
            f=plt.figure(figsize=(11.69,8.27));f.text(.045,.95,'\n'.join(wrapped[start:start+67]),
                va='top',fontfamily='DejaVu Sans Mono',fontsize=7.3,linespacing=1.08)
            f.text(.95,.025,f'{start//67+1}',ha='right',fontsize=8);pdf.savefig(f);plt.close(f)
        for fig in figures:pdf.savefig(fig);plt.close(fig)
    included=list(OUT.glob('pilot_collection_verified*'))+repairs+list(OUT.glob('*.png'))
    included += [OUT/'compression_v2_full_report.pdf',OUT/'compression_v2_full_report.txt',
                 OUT/'completion_audit.json',OUT/'verified_collection_spec.json',OUT/'warm_export_diagnosis.json',
                 BASE/'collector_recovery_check_v2.json',ROOT/'prompts/codex_compression_prompt_v2.txt']
    with zipfile.ZipFile(OUT/'compression_v2_report_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in included:
            if p.is_file():z.write(p,p.name)
    (BASE/'compression_completion_handoff.txt').write_text(
        '2026-10-08 completed bounded screening pilot and full report.\n'
        'No active/pending training or terminal evaluation; no training restart required.\n'
        'Original pilot_collection failure preserved; new collector and twelve export-only repairs complete.\n'
        '62 upstream nodes intentionally skipped; FP16 references remain blocked.\n'
        'Source B3 hard cell below B2: expansion gate false; multi-seed/control programme not completed.\n'
        f'Full PDF: {OUT}/compression_v2_full_report.pdf\n'
        f'Full TXT: {OUT}/compression_v2_full_report.txt\n'
        f'Audit: {OUT}/completion_audit.json\n')
    print(json.dumps({'report':str(OUT/'compression_v2_full_report.pdf'),
                      'audit':audit['status'],'gpu_hours':gpu_seconds/3600,'text_lines':len(wrapped)}))

if __name__ == '__main__':
    main()
