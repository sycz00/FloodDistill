"""Always retain a factual partial report, including absent scientific results."""
import json
import csv
from pathlib import Path
import time
import common11 as c

TABLES = {
 'priority_diagnostics.csv':['point','arm','seed','step','event','stratum','selected','valid','status'],
 'repair_learning_curves.csv':['point','arm','seed','step','equal_event_water_IoU','elapsed_seconds','CPU_seconds','status'],
 'repair_by_event.csv':['dataset','point','arm','seed','step','event','water_IoU','mIoU','status'],
 'correction_harm.csv':['dataset','point','arm','seed','event','corrected_false_water','corrected_missed_water','new_false_water','new_missed_water','status'],
 'conditioner_results.csv':['form','seed','lambda','equal_event_water_IoU','status'],
 'contrasts.csv':['dataset','endpoint','contrast','difference','CI95_low','CI95_high','status']}

def main():
    state=c.read(c.ROOT/'state.json',{})
    ledger=c.read(c.ROOT/'resource_ledger.json',{})
    inventory=c.read(c.ROOT/'dependency_evidence_ledger.json',{})
    status=c.read(c.ROOT/'scheduler_status.json',{})
    finished=bool(state.get('finished') or state.get('phase')=='COLLECTING')
    for name, fields in TABLES.items():
        if not (c.ROOT/name).exists(): c.csvout(c.ROOT/name,[],fields)
    # Collect retained source diagnostics even after a partial block or deadline.
    for name in ['priority_diagnostics.csv','correction_harm.csv','repair_by_event.csv']:
        source_rows=[]
        for path in sorted((c.ROOT/'source_diagnostics').glob('*/'+name)):
            with path.open() as stream:
                source_rows.extend(dict(row,dataset='s11val') for row in csv.DictReader(stream))
        if source_rows:
            with (c.ROOT/name).open() as stream:
                external=[row for row in csv.DictReader(stream) if row.get('dataset')!='s11val']
            c.csvout(c.ROOT/name,source_rows+external)
    jobs=state.get('jobs',{})
    lines=['# V11 selective repair — '+('final operational report' if finished else 'interim report'),'',
      'This bounded add-on tests whether subtracting teacher mismatch to the current FP shadow improves quantization repair beyond uniform, absolute-KL, disagreement, and randomized class-matched supervision. The separate optional R4_micro classifier probe tests image conditioning against static and threshold controls. No outcome is presumed positive.', '',
      'V10 retains priority for reviewer-critical supervised comparisons and the size/precision frontier. The reviewer letter, paper, supplement and previous-roadmap corrections are indexed in `reviewer_evidence.json` and `history_corrections.md`.', '',
      f'Worker phase: **{state.get("phase", status.get("phase", "PREPARING"))}**. Report refreshed at UTC epoch {time.time():.3f}.', '',
      '## Execution evidence','', '| Job | Status | Reason |','|---|---|---|']
    for name,row in jobs.items():
        lines.append(f'| {name} | {row.get("status")} | {str(row.get("reason", "")).replace("|", "/")} |')
    if not jobs:lines.append('| Queue | PREPARING | No launched jobs claimed |')
    lines += ['', '## Scientific readout', '']
    gate=c.read(c.ROOT/'repair/core/source_gate.json')
    if gate:lines += ['Source gate: `repair/core/source_gate.json`. '+json.dumps({k:v for k,v in gate.items() if k!='results'},sort_keys=True)[:2500],'']
    else:lines += ['No source gate has completed. No V11 accuracy improvement, degradation, speed advantage or reviewer closure is established.','']
    evaluation=c.read(c.ROOT/'evaluation/receipt.json')
    if evaluation:
        lines+=['Final evaluation receipt: `evaluation/receipt.json`; raw absolute scores and paired differences are in `repair_by_event.csv`, `evaluation/summary.csv`, and `contrasts.csv`. Event uncertainty is separate from four-seed variation.','']
    else:lines+=['External V11 evaluation has not completed. Empty CSVs are schemas, not zero-valued scientific outcomes. Missing dependencies, cost deferrals and interrupted blocks are not negative scientific results.','']
    conditioner=c.read(c.ROOT/'conditioner_receipt.json')
    if conditioner:lines+=['Optional classifier status: '+json.dumps(conditioner,sort_keys=True)[:1800],'']
    lines+=['The optional 1,560-step extension and second precision point are deferred in this core-first launch. Source-selected checkpoints are retained; any secondary external readouts omitted to protect the primary matrix remain explicitly deferred in the final freeze.','']
    lines+=['## Resource and dependency accounting','',
      'CPU-only V11; at most eight allocated threads. Maximum cumulative reserved charge is 64 core-hours, with final evaluation and collection protected before core admission. Old workers are conservatively reserved at configured maxima until terminal. No GPU training or additional GPU reservation is made.', '',
      'Exact actual/active/reserved costs: `resource_ledger.json`. Process CPU time is recorded separately from elapsed time multiplied by assigned threads.', '',
      'Current prerequisite ledger: `dependency_evidence_ledger.json`. Frozen V10 cache policy uses float32 logits for eight fresh geometric views, with photometric jitter applied only to student inputs. No partial cache shard or unfinished checkpoint is admitted.', '',
      '## Interpretation and next step','',
      'When dependencies finish, source-only deficit and measured-cost gates determine whether the full paired block can run. D must beat U and A, with L/N/P retained, before subtraction-specific follow-up is warranted. If no complete block fits, retain the exact waiting/deferral cause and continue V10 first. Optional probes cannot consume the protected primary evaluation budget.', '',
      'All external benchmarks, including reused WorldFloods sealed membership, are treated as non-confirmatory research readouts. Checkpoint/settings freeze precedes the single external phase. Native masks, event grouping and the 512-pixel evaluation geometry are retained. Fake quantization quality does not establish native low-bit speed.', '',
      'Prior-art boundaries: [QAT-ACS](https://arxiv.org/abs/2306.07215), [GHOST](https://arxiv.org/abs/2301.00131), [Teacher Intervention](https://aclanthology.org/2023.eacl-main.64/), [Dynamic Filter Networks](https://arxiv.org/abs/1605.09673). Disagreement-based selection, teacher-assisted quantization and input-conditioned weights are established; this run makes no first-ever claim.', '',
      'Fresh-terminal commands and limits are in `CONTROLS.txt`. Detachment is verified separately in `launch_verification.json`; host shutdown/sleep or loss of allocation stops progress.']
    (c.ROOT/'report_v11.md').write_text('\n'.join(lines)+'\n')
    negative=['# V11 unavailable, deferred and unfavorable results','',
              'No score-based seed exclusion, external method selection, repeated OFA experiment, new quantizer search, teacher training, or GPU training is authorized.','']
    for name,row in jobs.items():
        if row.get('status') not in ('COMPLETED','PENDING','RUNNING'):
            negative.append(f'- {name}: {row.get("status")}; {row.get("reason", "see receipt")}')
    if not evaluation:negative+=['','No completed external readout; no scientific gain or failure inferred.']
    (c.ROOT/'negative_results.md').write_text('\n'.join(negative)+'\n')
    c.save(c.ROOT/'collection_receipt.json',dict(outcome='completed',at=time.time(),interim=not finished,report_sha256=c.sha(c.ROOT/'report_v11.md'),scientific_results_available=bool(evaluation)))

if __name__=='__main__':main()
