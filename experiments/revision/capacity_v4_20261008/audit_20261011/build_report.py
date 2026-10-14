"""Build the requested full audit report from reproduced v5 counts and receipts."""
import csv
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
summary = json.loads((BASE/'v5/analysis/summary.json').read_text())
audit = json.loads((OUT/'t0_verification.json').read_text())
extra = json.loads((OUT/'supplement_summary.json').read_text())
cells = {r['label']:r for r in summary['cells']}
readouts = list(csv.DictReader((OUT/'all_readouts.csv').open()))
total = sum(q['gpu_command_seconds'] for q in audit['queues'].values())/3600


def fmt_ci(d):
    return f"{d['mean']:+.2f} [{d.get('lo',d.get('lo95')):+.2f}, {d.get('hi',d.get('hi95')):+.2f}]"


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
                     + ['| '+' | '.join(map(str,r))+' |' for r in rows])


def cell_table(labels):
    return table(['Model / arm','Params','S11-val ± SD','S11-test','STURM mIoU / wIoU','WF mIoU / wIoU'],
        [[r['label'],f"{r['params']:,}",f"{r['val_mean']:.2f} ± {r['val_sd']:.2f}",f"{r['s11_mIoU_mean']:.2f}",
          f"{r['sturm_mIoU_mean']:.2f} / {r['sturm_wIoU_mean']:.2f}",
          f"{r['wf_mIoU_mean']:.2f} / {r['wf_wIoU_mean']:.2f}"] for r in [cells[k] for k in labels]])


fig,axes = plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
plt.rcParams.update({'font.size':9})
names = ['R1','R2','R3','R4','A3N','A3']
colors = dict(zip(names,['#7b8794','#b7791f','#5f8f50','#117f78','#7159a2','#3975b1']))
for g in names:
    r=cells[g+' LKD @sel lr']
    axes[0].errorbar(r['params'],r['val_mean'],yerr=r['val_sd'],fmt='o',color=colors[g],capsize=3)
    axes[0].annotate(g,(r['params'],r['val_mean']),xytext=(4,5),textcoords='offset points')
    axes[1].scatter(r['macs_px'],r['params'],s=65,color=colors[g])
    axes[1].annotate(g,(r['macs_px'],r['params']),xytext=(5,3),textcoords='offset points')
axes[0].set(xscale='log',xlabel='Parameters (log scale)',ylabel='S11-val mIoU (%)',
            title='Selected LR; four seeds, ± sample SD')
axes[0].set_xlim(9, 65000)
axes[1].set(yscale='log',xlabel='MACs per output pixel',ylabel='Parameters (log scale)',
            title='Memory and arithmetic give different rankings')
for i,g in enumerate(names):
    r=cells[g+' LKD @sel lr'];m=r['wf_wIoU_minus_teacher_event_mean']
    lo=r['wf_wIoU_minus_teacher_lo95'];hi=r['wf_wIoU_minus_teacher_hi95']
    axes[2].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=colors[g],capsize=3)
axes[2].axvline(0,color='black',lw=.9)
axes[2].axvline(-2,color='#999999',ls='--',lw=.8)
axes[2].set(yticks=range(len(names)),yticklabels=names,xlabel='WF event water-IoU minus teacher (pp)',
            title='Conditional event bootstrap; two-sided 95% CI')
for ax in axes:
    ax.grid(alpha=.18)
fig.savefig(OUT/'capacity_findings.png',dpi=180)
fig.savefig(OUT/'capacity_findings.pdf')
plt.close(fig)

report = f"""# FloodDistill: full v4–v5 audit and remaining experiment plan

11 October 2026, Europe/Berlin. Scope: the capacity/compression programme inherited from Codex and continued by Opus/Claude on 9–11 October, plus an independent CPU audit in this turn. The inherited report, code, manifests and checkpoints are preserved. This report records completed science separately from historical failed attempts and proposed work.

## 1. Current outcome and live status

The v5 experiment set has finished. The original corrected programme and handover follow-ups have not all finished. No capacity queue is running or waiting to launch further work. Both RTX 4090s were idle at the live check. This turn ran T0 and CPU analyses and produced this report; it launched no new GPU training.

The strongest established finding is that the old low-bit gradient explosion came from the simulation of rounding residues in LiteMLA. After repairing that measurement, the largest demonstrated accuracy improvement comes from changing A3's learning rate, not from a special quantization method. A small pure-convolutional model, R4, performs strongly; a 418-parameter per-pixel MLP, R2, satisfies the registered S11-validation floor rule. Neither establishes a teacher-equivalent external capacity floor.

Measured GPU-command time is **{total:.2f} GPU-hours**, leaving **{30-total:.2f}** under the approximate 30-hour cap. This is elapsed time summed across GPU-tagged commands, including failed attempts and some trivial jobs tagged GPU by the inherited queue; it is neither wall-clock duration nor measured accelerator utilization. CPU-only U1 fitting and this audit are outside that number.

## 2. Verification and scientific contract

T0 passes. The frozen `analyze_v5.py` was rerun; both regenerated tables are byte-identical to their saved versions. The analysis files were backed up first. The five required effects reproduce within 0.01 pp:

"""
report += table(['Required contrast','Recomputed mean and paired-seed 95% CI (pp)'],
    [[c['label'],fmt_ci(c)] for c in audit['numeric_checks']])
report += """

The A3 learning-rate contrast is not emitted by the frozen analyzer; this audit computed it directly from the four paired training receipts. This is a supplementary calculation, not a change to the frozen analysis.

Provenance checks passed for all 114 capsule source files, all 2,618 shared identity entries (including teacher, cohort tensors and code), all declared inputs in every checked queue, 232 checkpoint hash references in training receipts, and 110 terminal checkpoint hash references. There are no unlisted non-cache capsule files. Queue identities were verified using the runner's normalized manifest digest, rather than comparing the state digest with the raw JSON-file hash. The original smoke's per-image confusion counts also match v2 exactly: 90 S11-test + 2,675 STURM + 785 WF images, zero differences.

The 110 terminal receipts represent **103 distinct checkpoint byte hashes**, with seven aliases from reuse. They are not 110 independent trainings. All 23 populated analysis cells have four selection scores and four external readouts. Four R2 OFA receipts are intentional skips because its per-pixel graph has no OFA stages.

The shared recipe uses the restored original 2,500-tile cohort, six optical bands, 224-pixel training and 512-pixel main evaluation, a frozen 300M Prithvi teacher, and live soft targets for LKD/OFA. FP training is scratch/ReLU6/AdamW/cosine/FP32, ordinarily 7,800 updates. S11-val selects checkpoints and LR; S11-test/STURM/WF are research-exposed secondary readouts. They provide evidence but do not define an untouched confirmation set. All low-bit results remain **simulation_only**.

Seed means use four seeds, 42/64/91/107, and sample SD. Paired-seed CIs quantify training variation on a fixed benchmark. Event bootstraps average a model's four seeds within each event, then resample whole events 10,000 times with seed 1729; they quantify event variation conditional on those checkpoints. They do not replace the seed CIs or create independent pixel replicates. New supplementary event comparisons below are descriptive readouts, not new gates.

## 3. What happened, including failures and recoveries

"""
queue_rows=[]
for name,q in audit['queues'].items():
    counts=', '.join(f'{v} {k}' for k,v in q['counts'].items())
    queue_rows.append([name,q['worker_status'],counts,f"{q['gpu_command_seconds']/3600:.3f}"])
report += table(['Queue','Worker','Historical job states','GPU-command h'],queue_rows)
report += """

Historical failed/blocked states are retained. They are not evidence that the final repaired matrix is still waiting:

- Stage 1: Facts reproduced after the sandbox-CUDA failure was rerun on the host. Integer-domain forward arithmetic was added opt-in, and exact native STE gradients were restored in v3 after single-layer gradient accumulation differences were detected. Four fixed-batch cells, audits, parity, sensitivity maps and optimizer reach finished. The v4 protocol was frozen before Stage 2 scores.
- Stage 2: six of 21 new FP runs exceeded the original 1,020-second timeout. Each got one diagnosed retry in a fresh directory with a 1,400-second limit; all six succeeded. Together with three compatible seed-42 reuses, six graph families have four seeds. The handover attributes the timeouts to contention; the timeout receipts directly establish the incomplete attempts, while load is the reported explanation.
- Opus/Claude v5: a preregistered addendum changed the order and added E2 matched low-bit/FP continuations, E3 capacity rungs and LR grids, E4 training-method comparisons, and E1 external evaluation. These replace part of Stage 3 and part of Stage 4, rather than completing every original demand.
- Two reuse defects caused crashes: missing `graph_record.parameters` for the A3 reuse, then an exactly-7,800-update check for an early-stopped R1 reuse. New `train_v5b.py` and `train_v5c.py` preserved the original files and retried the affected dependency chains. A terminal import briefly changed the working directory and wrote a smoke output into the capsule; it was moved out. This audit independently confirms capsule integrity now.
- U1 was fitted on CPU and failed its external readout; U6 completed in Stage 1 and did not outperform a simpler structural allocation. The remaining exploration ideas have not run.

Six **unique** tiny-rung training runs stopped at 7,176–7,488 updates under inherited patience: R1 ×3, R2 ×1, R3 ×2. A seventh early-stop receipt aliases the R1 seed-42 grid run. `protocol_v4.yaml` contains both patience 20 and `incomplete_budget_is_failure: true`; the addendum describes matched 7,800 updates, but the v5c reuse code accepts fewer. Therefore these are successful executions under the inherited early-stop rule, with a fixed-budget compliance exception. They cannot be described as all completing exactly 7,800 updates. Do not silently reinterpret or relabel them. If the strict-budget reading is retained, finish/validate these endpoints as a separate amendment or report them as deviations. Their selected checkpoints came much earlier; extra training is needed primarily to restore the final-update comparison, and any later checkpoint selection would itself be a declared change.

## 4. Measurement repair and where quantization causes damage

The adopted simulator performs convolutions on integer codes and rescales after accumulation while retaining native STE masks/gradients. It preserves exact code cancellation. It is a scientific simulator repair, not a packed INT4 implementation or a custom hardware kernel.

"""
report += table(['Cell','Legacy median norm','Integer median / max','Median ratio to FP'],[
    ['A3 W8A8','7.5','3.38 / 5.40','0.9×'],['A3 W4A8','2.3e6','8.14 / 14.6','2.1×'],
    ['A3 W4A4','2.1e10','4.54 / 7.77','1.2×'],['A1 W4A8','1.2e4','10.6 / 20.5','2.7×']])
report += """

H1 passes the registered median ≤10× / maximum ≤100× FP criteria. Independent norm recomputation agrees within 1%. Selected v2 checkpoint source rescoring differs by at most 0.138 pp. The important limitation remains: integer and legacy simulators agree on only about 91.8% of augmented TRAIN argmax pixels at A3 W4A4. Near-equal validation mIoU does not make their forwards identical; old W4A4 conclusions require rescoring under the adopted simulator.

The gradient pathology and the forward accuracy damage have different causes. The former localizes to attention divisors; bypassing the stem input recovers approximately +4.2 pp for A1 W4A8 and +7.2 pp for A3 W4A4, while bypassing the joint QKV path offers little forward recovery. A V-only bypass was correctly left unresolved because Q/K/V share a fused quantized input.

U6 teacher-excess ranking overlaps the label ranking in 2/4 top layers. Its first 8-bit layer costs 18 extra theoretical bytes and recovers about 77% of A1's weight damage. The top-four allocation and the simple structural rule both score about 83.08 mIoU, but the structural rule uses 358 rather than 1,438 extra bytes. This is a useful negative-to-neutral result for the proposed ranking; it does not justify a new mixed-precision method claim.

## 5. Low bit: the matched control changes the interpretation

"""
report += cell_table(['A3 LKD (Stage2)','A3 F48','A3 Q48','A3 Q4b'])
report += """

F48 is the same 1,560-update, LR 5e-4 continuation with QDQ bypassed; Q48 is W4A8; Q4b is a 4-bit body with an 8-bit stem input exception. Both the LR and extra optimizer work are controlled in Q48 versus F48.

Q48−F48 on source validation is **−0.25 [−0.81, +0.32] pp**. There is no evidence of a quantization-specific accuracy gain. The earlier single-seed +3.2-pp “QAT beats FP” result did not isolate quantization from extra optimization. The matched comparison is consistent with a small cost. It does **not** prove strict equivalence or literally zero accuracy loss: that requires a registered margin and an appropriate CI-based equivalence/non-inferiority test.

The inherited report called all external comparisons null. That is correct for its paired-seed pooled contrasts, but incomplete for event-macro readouts. This audit finds:

"""
report += table(['Contrast','Benchmark','Paired-seed pooled wIoU CI (pp)','Conditional event wIoU CI (pp)'],
    [[name,bench,fmt_ci(extra['event_contrasts'][name][bench+' full']['paired_seed_pooled_water_IoU']),
      fmt_ci(extra['event_contrasts'][name][bench+' full']['conditional_event_water_IoU'])]
     for name in ['A3 Q48 - A3 F48','A3 Q4b - A3 Q48'] for bench in ['sturm','wf']])
report += """

The small WF event-level cost of Q48 is −0.71 pp, conditional on the four checkpoints; the four-seed pooled WF CI remains wide and includes zero. Q4b has a larger conditional WF event cost. These two uncertainty statements answer different questions and must both be retained. Q4b−Q48 on S11-val is −1.14 [−2.34, +0.07] pp; its mean fails the registered “within 1.0 pp” rule. No low-bit arm satisfies the teacher non-inferiority gate on WF.

## 6. Capacity and learning-rate results

Every E3 rung's worst seed-42 LR grid point was 5e-5; all except R1 selected 5e-3, while R1 selected 5e-4. The A3 four-seed difference selected LR−recipe LR is **+4.14 [+1.32, +6.96] pp on S11-val**, and +2.53 [+0.69, +4.37] on S11-test. This supports under-training of A3 with the inherited recipe. A0/A1/A2/X1/X2 did not get the grid: their optimization status is untested, so “all anchors are under-trained” is an inference, not an established result.

"""
report += cell_table([g+' LKD @sel lr' for g in ['R1','R2','R3','R4','A3N','A3']])
report += """

Fixed-LR graph-family context, with four seeds each:

"""
report += cell_table([g+' LKD (Stage2)' for g in ['A0','A1','A2','X1','X2']])
report += """

R4−A3 at their selected LR is +1.37 [+0.41,+2.33] source-val, +2.60 [+1.88,+3.32] S11-test mIoU and +2.84 [+1.53,+4.14] pooled WF water-IoU by paired seeds. Its WF conditional event advantage is +3.15 [+2.10,+4.34]. This is strong evidence for this declared pure-conv graph versus this A3 configuration. Comparing optimized R4 with recipe-LR A0 does not yet establish that smaller models are better.

A3N−A3 is +0.94 [+0.13,+1.74] S11-test and +1.71 [+0.11,+3.30] pooled WF water-IoU by paired seeds. Its conditional event WF difference is only +0.49 [−2.00,+3.11], while the conditional event STURM difference is +1.40 [+0.46,+2.52]. The justified statement is that attention provides no demonstrated advantage for these small, tested configurations. These results do not establish that context is unnecessary: A3N and R4 still have spatial convolutions.

R2 satisfies the registered **mean-based** floor rule: its four-seed source-val mean is 0.93 pp below A3, inside the 1.0-pp cutoff. That rule is not a CI-based equivalence test. R2 has the highest S11-test mean among the LKD rungs (90.21), and R2 SUP reaches 90.60; the teacher's pooled reference is 89.46. Thus a blanket “no student beats the teacher on S11-test” is incorrect. These pooled means do not establish geographic superiority, and the low-bit arms do not beat that teacher mean.

R2 loses −1.89 [−3.23,−0.55] pooled STURM water-IoU versus A3. Every declared rung fails the joint external teacher non-inferiority criterion at margins 1/2/3 pp because WF event uncertainty remains too wide or the deficit too large. The external knee remains **undefined**. On STURM alone, A3N passes the registered one-sided 2-pp test; this is specific to that benchmark and margin.

![Capacity, arithmetic and WF event uncertainty](audit_20261011/capacity_findings.png)

The arithmetic/parameter tradeoff is real. R2 has only 418 parameters but about 384 MACs/pixel; A3N has 20,250 parameters and about 144 MACs/pixel; A3 has 30,922 and about 164; R4 has 12,230 and about 417. These are static conv/matmul counts with interpolation excluded, not measured latency or energy. R4 is a strong accuracy/storage candidate; A3N is the lower-MAC context candidate. There is no single proven deployment winner. FP32 parameter payloads alone are 1,672 bytes for R2, 48,920 for R4, 81,000 for A3N and 123,688 for A3; serialized artifacts, buffers and runtime activations cost more. Uniform-bit theoretical counts are not achieved mixed-precision engine sizes.

## 7. What distillation adds, and what remains confounded

CE here is focal learning from **teacher hard pseudo-labels**, not teacher-free human supervision. LKD adds live soft targets. SUP uses 252 human-labelled S11 chips, not the 2,500-tile teacher-labelled pool. Comparing SUP with LKD changes training data, sample diversity and label source as well as soft-target use; it is not a same-input causal isolation of the teacher.

"""
report += cell_table(['A0 CE','A0 LKD (Stage2)','A0 OFA','A0 SUP',
    'A3 CE','A3 LKD (Stage2)','A3 OFA','A3 SUP','floor R2 CE','R2 LKD @sel lr','floor R2 SUP'])
report += """

A0 LKD−CE resolves on STURM: +2.72 [+0.94,+4.51] pooled water-IoU by paired seeds. Its conditional event difference is +3.19 [+1.21,+5.73] on STURM and +1.05 [+0.41,+1.70] on WF; the pooled WF paired-seed difference remains unresolved. R2 LKD−CE gives +5.42 [+3.49,+7.35] pooled STURM water-IoU and +4.72 [+2.18,+7.80] conditional event water-IoU. This supports a benchmark-dependent benefit from soft-target distillation, subject to repeating the comparisons at stronger common recipes.

No consistent OFA advantage over LKD is resolved. “OFA = LKD” overstates the evidence; the tests show no resolved gain, not proved equivalence. R2 OFA is undefined and skipped, not a failed implementation.

H7 (“the KD gain grows as capacity shrinks”) fails its registered primary S11-val test. The R2-versus-A0 STURM paired-seed interaction is +2.69 [+1.91,+3.48], but its conditional event interaction is +1.52 [−0.11,+3.72]. On WF the corresponding conditional event interaction is −0.97 [−2.05,−0.12], while its paired-seed pooled interaction is unresolved. These results do not support a uniform capacity law. R2 uses an LKD-selected LR of 5e-3 while A0 uses 5e-5, and A3's E4 rows also use 5e-5: T1/T2 are needed before making a general method claim.

R2 SUP is strongest in-domain and ahead of R2 LKD in pooled WF water-IoU, but behind on pooled STURM. The conditional event R2 LKD−SUP CIs include zero on both external datasets. Label-protocol differences are a plausible explanation, not an independently tested causal mechanism. Likewise, the attribution of WF deficits to missing teacher context or to turbid water requires a disagreement/stratified diagnostic; counts alone establish precision/recall differences, not their full cause.

## 8. Why event-level and overlap-excluded readouts matter

EMSR342 contributes **82.06% of WF reference-water pixels**. A model can exceed the teacher in pooled WF water-IoU and still trail it across equally weighted events. R4's pooled WF water-IoU is 65.95 versus the teacher's 64.32, but its mean event difference is **−2.95 [−6.99,−0.09] pp**, conditional on the checkpoints. Report both. Do not treat pooled dominance as a teacher-equivalence result.

The overlap-excluded readouts are present in the terminal receipts but absent from the inherited summary tables. This audit now supplies them for every populated cell:

"""
subset=[]
for family in ['R2 selected LKD','R4 selected LKD','A3N selected LKD','A3 selected LKD','A3 Q48']:
    for bench in ['sturm','wf']:
        pair=[next(r for r in readouts if r['family']==family and r['benchmark']==bench and r['domain']==dom)
              for dom in ['full','known_overlap_excluded']]
        subset.append([family,bench,*(f"{float(r['water_IoU_mean']):.2f}" for r in pair),
                       f"{float(pair[1]['teacher_event_delta_mean']):+.2f} [{float(pair[1]['teacher_event_delta_lo95']):+.2f}, {float(pair[1]['teacher_event_delta_hi95']):+.2f}]"])
report += table(['Model','Benchmark','Full pooled wIoU','Excluded pooled wIoU','Excluded event delta to teacher (pp)'],subset)
report += """

STURM excludes 3 of 2,675 chips; WF excludes 52 of 785 windows. These exclusions do not remove prior research exposure or create an untouched holdout. The displayed teacher-gap conclusion persists after exclusion. The complete file also includes precision, recall, sample SD and event-macro water-IoU for S11/STURM/WF.

## 9. Selected versus final-update outcomes

The inherited report largely presents selected checkpoints. Both selected and final-update source scores are now recorded:

"""
family_finals={}
for r in extra['final_source_results']:
    family_finals.setdefault(r['family'],[]).append(r)
report += table(['Model / arm','Selected S11-val mean','Final S11-val mean','Completed updates, four seeds'],
    [[label,f"{st.mean(r['selected_val_mIoU'] for r in rows):.2f}",
      f"{st.mean(r['final_val_mIoU'] for r in rows):.2f}",', '.join(str(r['completed_updates']) for r in rows)]
      for label,rows in family_finals.items()])
report += """

R1 is especially unstable: selected 84.30 versus final 77.26, with selected-score SD 5.92 pp. R4's selected 87.94 versus final 86.64 and A3's selected 86.57 versus final 86.28 reinforce the need to retain the checkpoint rule. This audit did not generate missing final-checkpoint external predictions: existing terminal results score selected artifacts only. Final external readout remains unrun where the original prompt demands both selected and final.

## 10. Novelty ledger, corrected interpretation

"""
report += table(['Candidate','Evidence now','Interpretation / next gate'],[
    ['Integer-domain QAT simulation','Gradient repair and parity verified','Strong measurement finding; priority claim needs full related-work/code check'],
    ['Log-domain stem','Proposed; no training','Conditional low-bit test after conventional clipping/unsigned/LSQ'],
    ['U1 gain-invariant index','Implemented hard-target fit fails WF','Do not retry for external scores; original soft-target proposal untested'],
    ['Flip-aware QAT','Not run','Only relevant if a residual hard-cell gap survives strong baseline'],
    ['Rule-residual student','Not run','One preregistered, controlled R4-first test; separate head and weighting effects'],
    ['Capacity-aware / uncertainty KD','Primary H7 not supported','Defer until well-trained matched-method comparisons'],
    ['Parameter floor vs compute floor','Measured','Useful framing, not a novel algorithm'],
    ['U6 label-free allocation','Structural allocation ties at fewer bytes','Negative-to-neutral; no method advantage claim'],
    ['Small pure-conv / no-attention graphs','R4 and A3N strong vs A3','Empirical finding for tested graphs; re-anchor A0/A1 before broader claim']])
report += """

U1's implemented zero-sum log-linear index scores 85.91 / 75.69 / 64.83 mIoU on S11-test / STURM / WF. WF MNDWI is 75.94; U1 water recall is 45.42% versus MNDWI 77.99%. It clearly fails the implemented one-shot recipe. However, `u1_index.py` fits teacher **hard masks**, whereas the original proposal called for soft-label logistic fitting. It also clamps reflectance to 1e-4 before logging: zero-sum gain invariance is exact only when clipping does not change which values hit the floor. The full proposed soft-target/exact-invariance version was not faithfully tested. Preserve the failed implementation and its target/floor disclosure; do not silently report a reproduction of the stronger proposal or tune it against WF.

Related-work checks in this audit were bounded. [LSQ](https://arxiv.org/abs/1902.08153) already learns quantizer step sizes with gradient scaling; it is a conventional baseline, not the new method. [Trio-ViT](https://arxiv.org/abs/2405.03882) already treats PTQ and acceleration of softmax-free efficient ViTs. Those primary papers make broad “first quantization treatment of linear attention” language untenable. The residue-division diagnosis may be a distinct contribution, but its priority is unverified. Onboard flood segmentation also has existing [WorldFloods/Φsat-1 work](https://ora.ox.ac.uk/objects/uuid%3A3befdca0-deb1-4cbe-aa30-763d8e55d82e); no first-onboard or first-tiny-model claim is warranted. A full priority search, including the other required comparisons, remains pending.

## 11. What still needs to run, in order

"""
report += table(['Priority / task','Concrete work','Why it matters','Forecast / status'],[
    ['T1: re-anchor','A0/A1 LR grid at s42; four seeds at selected LR; reuse compatible recipe runs','Removes optimization confound in size comparisons','~3–4 GPU-h; not queued'],
    ['T2: well-trained methods','A3/R4/A3N LKD vs CE; OFA on staged A3/A3N after smoke','Tests whether soft-target benefit survives competent optimization','~3–4 GPU-h; not queued'],
    ['T3: one novelty test','R4-first rule-residual + disagreement weighting; matched KL mass, random-weight control','Targets teacher–rule disagreement; tests a mechanism beyond architecture','≤3 GPU-h declared; not queued'],
    ['T4: remaining Stage 3','Re-anchored Q4 body/W4A4; clipping, proven unsigned grids, bounded LR/epoch grid, faithful LSQ','Establishes a strong conventional low-bit baseline','≤4 GPU-h declared; not queued'],
    ['Stage 6','Selected R4/A3N/A3 W4A8 exports, FP16 fix, warmed parity, bytes/latency/memory','Turns simulation results into defensible deployment evidence','≤2 GPU-h original cap; not queued'],
    ['Stage 4 completion','256/384/512 val scale diagnostic; common gain/haze/tilt robustness; teacher–rule disagreement counts','Tests geometry, robustness and what the teacher adds beyond MNDWI','Original robustness cap ≤3 GPU-h; not run'],
    ['Stage 7 / U8','Pair availability/co-registration, independent increment-labeler agreement, reference feasibility','Separates surface water from temporal flood increments','≤2 CPU-h plus any teacher inference accounted; not run'],
    ['Stage 8 remainder','U3/U4/U5/U7; U2 scale/robustness part still missing; U6 terminal readout incomplete','Exploration, uncertainty, curation and invariance hypotheses','Deferred; conditional gates and cap apply'],
    ['Stage 5','Only if strong SB leaves the preregistered hard-cell gap','Avoids mechanism arms on a problem conventional tools already solve','Closed until full Stage-3 gate; not run'],
    ['Final readouts / bookkeeping','Final artifacts externally scored, event CIs, model–rule lens, full prior art and failure/budget ledger','Closes original reporting demands','CPU counts reuse first; GPU only for missing predictions']])
report += f"""

T1 can require up to ten new FP trainings: two new seed-42 LR points plus three new-seed runs for each of A0 and A1 if a new LR wins. T2 can reuse the existing 12 selected-LR LKD checkpoints if identities match; it then needs 12 CE runs and eight OFA runs (A3/A3N). Existing CE runs take roughly 256–506 seconds depending on graph; OFA roughly 798–884 seconds. These observed costs support the forecasts, but new graph/method cells need smoke timing and hard reservations. A3N OFA must pass a staged-hook/finite-loss gate before it is scheduled; R4 OFA remains undefined.

The handover's approximate **4 + 4 + 3 + 4 = 15 GPU-hours** for T1–T4 already exceeds the **{30-total:.2f}** remainder by **{15-(30-total):.2f}**, before deployment and missing robustness. Completing T1/T2 first would leave only roughly 0.7–2.7 hours under these forecasts. Do not claim every follow-up fits the old cap. Reuse compatible checkpoints, release unused reservations explicitly, and measure new commands before choosing cuts. Under the current cap, comparison validity (T1/T2) has priority; T3/T4/deployment/robustness compete for the actual remainder. T1 does not re-anchor A2/X1/X2, so their width/depth story remains a fixed-recipe comparison unless a separate budgeted extension is declared.

## 12. Where improvement is most plausible

1. **Training recipe first.** The measured A3 +4.14-pp source effect is larger than the proposed auxiliary-method gains. Bring A0/A1 to an equally searched recipe before concluding that reduced capacity helps. Preserve the original LR results as the matched reproduction table.
2. **Pure-convolutional small models as the practical path.** Keep R4 for accuracy/storage and A3N for lower arithmetic. Quantize and benchmark their final selected artifacts on a supported target. Neither depends on LiteMLA divisors, but absence of that pathology does not prove general low-bit robustness.
3. **Residual low-bit activation damage.** Q4b's remaining deficit justifies the prescribed conventional baseline: stem precision, graph-proven unsigned grids, clipping, then learned steps. A log-domain stem is a hypothesis to test after those controls, not an established fix.
4. **Distill what the rule misses.** The rule-residual/disagreement test is the strongest proposed discriminating intervention. First cache teacher/rule predictions and quantify the four disagreement cells versus reference labels. Freeze γ, initialization, KL normalization, MDE and controls using TRAIN/source validation only. A combined head-plus-weighting experiment establishes a package effect; separate controls are needed to attribute it to the head or weighting. With tight budget, run R4 first; A3N replication is conditional on budget, never on favorable external selection.
5. **Evaluate robustness and transfer honestly.** Clean S11 accuracy does not explain haze, tilt or the 224→512 geometry issue. Run the common reduced robustness and scale diagnostics before making edge-reliability or context-sufficiency claims. Independent human dry references and genuinely new events remain unavailable/unauthorized rather than silently replaced by teacher-empty masks.

Do not restart a falsified U1 recipe, repeat unchanged v2 arms, add score-driven graphs, or expand every exploration branch. Current evidence supports a measurement correction, a training-recipe correction for A3, and a useful empirical small-model tradeoff. A new superior distillation method, a universal capacity floor, real low-bit deployment speedups, and flood-increment detection remain unestablished.

## 13. Reproduction, outputs and queue controls

Primary evidence: `audit_20261011/t0_verification.json`, `v5/analysis/summary.json`, `audit_20261011/supplement_summary.json`, `audit_20261011/all_readouts.csv`, and `audit_20261011/selected_and_final_source.csv`. The audit includes the five gate numbers, input/checkpoint hashes, per-queue elapsed time, early-stop and skip records. The inherited `report_v5.md` and novelty ledger are unchanged.

CPU reproduction from the repository root:

```bash
/home/schmalstieg/miniconda3/envs/esi/bin/python experiments/revision/capacity_v4_20261008/audit_20261011/verify_handover.py
/home/schmalstieg/miniconda3/envs/esi/bin/python experiments/revision/capacity_v4_20261008/audit_20261011/supplement_analysis.py
```

Existing queue controls, shown for v5c (all current queues are finished):

```bash
CAP_PY=/home/schmalstieg/miniconda3/envs/esi/bin/python
CAP_Q=experiments/revision/capacity_v4_20261008/v5c/revision_queue.py
CAP_M=experiments/revision/capacity_v4_20261008/v5c/manifest.json
"$CAP_PY" "$CAP_Q" status --manifest "$CAP_M"
"$CAP_PY" "$CAP_Q" pause --manifest "$CAP_M"
"$CAP_PY" "$CAP_Q" resume --manifest "$CAP_M"
"$CAP_PY" "$CAP_Q" stop --manifest "$CAP_M"
```

`resume` does not create any T1–T4 work or restart a finished worker. New stages need new immutable manifests. Existing poll intervals vary by queue; there is currently no active checker scheduling the handover follow-ups. This report does not imply future autonomous agent activity.
"""
path = BASE/'report_v5_codex_audit_20261011.md'
path.write_text(report)
print(path)
print('words',len(report.split()))
