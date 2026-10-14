"""Build the requested morning audit from independently recomputed CPU evidence.

Does not launch GPU work or mutate the frozen worker, receipts or collector.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st
from datetime import datetime
from zoneinfo import ZoneInfo
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
read = lambda path: json.loads(Path(path).read_text())
source = read(OUT/'source_stats.json')
external = read(OUT/'external_mechanism_summary.json')
integrity = read(OUT/'external_mechanism_integrity.json')
operational = read(OUT/'operational_audit.json')
state = read(RUN/'state.json')
budget = read(RUN/'budget_ledger.json')
provenance = read(RUN/'provenance.json')
failures = read(RUN/'failure_ledger.json')
assert state['worker_status'] == 'finished' and state['active_job'] is None
assert provenance['status'] == 'complete'
assert provenance['cohort_sha256'] == '1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60'
assert not source['prospective_gate']['passed']
assert not integrity['errors'] and not operational['errors']


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'---|'*len(headers)] +
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def ci(row, keys=('mean_pp', 'ci95_low_pp', 'ci95_high_pp')):
    return f"{row[keys[0]]:+.2f} [{row[keys[1]]:+.2f}, {row[keys[2]]:+.2f}]"


def sc(left, right, kind='selected', metric='mIoU'):
    return next(x for x in source['contrasts'] if x['left']==left and x['right']==right and
                x['checkpoint_kind']==kind and x['metric']==metric)


def es(family, dataset, domain='full'):
    family = 'v7_rule_R' if family == 'MNDWI>0' else family
    return next(x for x in external['external_summary'] if x['family']==family and
                x['dataset']==dataset and x['domain']==domain)


def ec(left, right, dataset, metric, domain='full', event=False):
    right = 'v7_rule_R' if right == 'MNDWI>0' else right
    rows=[x for x in external['contrasts'] if x['left']==left and x['right']==right and
          x['dataset']==dataset and x['domain']==domain and x['metric']==metric]
    return next(x for x in rows if ('whole-event' in x['uncertainty'])==event)


def mechanism(family, dataset, domain='full'):
    return next(x for x in external['mechanisms'] if x['family']==family and
                x['dataset']==dataset and x['domain']==domain and x['event_id']=='__pooled__')


plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
names=['A0_old_LKD','A0_reanchored_LKD','A1_old_LKD','A1_reanchored_LKD','A3_LKD','R4_LKD','R4_CE']
labels=['A0 old','A0 repaired','A1 old','A1 repaired','A3 LKD','R4 LKD','R4 CE']
for i,(name,label) in enumerate(zip(names,labels)):
    g=source['groups'][name];v=g['selected']['mIoU']
    axes[0].errorbar(v['mean_percent'],i,xerr=v['sample_sd_percent'],fmt='o',capsize=3,
                     color='#146a7c' if 'R4' in name else '#667685')
axes[0].set(yticks=range(len(names)),yticklabels=labels,xlabel='Source mIoU (%)',
            title='Four seeds; error bars are sample SD')
axes[0].invert_yaxis()
for name,label in [('A0_reanchored_LKD','A0'),('A1_reanchored_LKD','A1'),('A3_LKD','A3'),('R4_LKD','R4 LKD'),('R4_CE','R4 CE')]:
    g=source['groups'][name]
    axes[1].scatter(g['parameters'],g['selected']['mIoU']['mean_percent'],s=55)
    axes[1].annotate(label,(g['parameters'],g['selected']['mIoU']['mean_percent']),xytext=(4,5),textcoords='offset points')
axes[1].set(xscale='log',xlabel='Parameters (log scale; not artifact bytes)',ylabel='Source mIoU (%)',
            title='Fixed declared graphs at the competent recipe')
for ax in axes: ax.grid(alpha=.18)
fig.savefig(OUT/'source_reanchor.png',dpi=180);fig.savefig(OUT/'source_reanchor.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(10.5,4.1),layout='constrained')
arms=['C0','C1','C2','C3','C4','C5','HC','HL']
for i,arm in enumerate(arms):
    rows=source['pilot_per_seed'][arm]
    final=[x['final']['mIoU']-x['start']['mIoU'] for x in rows.values()]
    chosen=[x['selected']['mIoU']-x['start']['mIoU'] for x in rows.values()]
    ax.scatter([i-.08,i+.08],final,color='#b45236',s=35,zorder=3)
    ax.scatter(i,st.mean(chosen),marker='D',color='#146a7c',s=35,zorder=4)
ax.axhline(0,color='#555555',lw=.8)
ax.scatter([],[],color='#b45236',label='Final endpoints: two seed dots')
ax.scatter([],[],marker='D',color='#146a7c',label='Selected mean: starting checkpoint allowed')
ax.set(xticks=range(len(arms)),xticklabels=arms,ylabel='Source mIoU change from same-seed start (pp)',
       title='All pilot continuations trained; none selected an improved C0–C5 endpoint')
ax.grid(axis='y',alpha=.18);ax.legend(loc='lower right',frameon=False)
fig.savefig(OUT/'pilot_selected_final.png',dpi=180);fig.savefig(OUT/'pilot_selected_final.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.0),layout='constrained')
colors={'A0_LKD_reanchored':'#667685','R4_CE':'#b45236','teacher':'#8466a4','MNDWI>0':'#648149'}
for i,right in enumerate(colors):
    a=ec('R4_LKD',right,'wf','water_IoU')
    b=ec('R4_LKD',right,'wf','event_macro_water_IoU',event=True)
    for ax,row in zip(axes,[a,b]):
        ax.errorbar(row['mean_pp'],i,xerr=[[row['mean_pp']-row['lo95_pp']],[row['hi95_pp']-row['mean_pp']]],
                    fmt='o',color=colors[right],capsize=3)
for ax in axes:
    ax.axvline(0,color='#555555',lw=.8)
    ax.set(yticks=range(4),yticklabels=['Repaired A0','R4 cached-hard CE','Teacher','MNDWI'],
           xlabel='R4 LKD minus comparator, water IoU (pp)')
    ax.grid(alpha=.18)
axes[0].set_title('WF pooled pixels: four-seed uncertainty')
axes[1].set_title('WF equal events: conditional event uncertainty')
fig.savefig(OUT/'wf_estimands.png',dpi=180);fig.savefig(OUT/'wf_estimands.pdf');plt.close(fig)

local=lambda stamp:datetime.fromisoformat(stamp).astimezone(ZoneInfo('Europe/Berlin')).strftime('%d October %Y, %H:%M:%S CEST')
newh=budget['incremental_spent_seconds']/3600;totalh=budget['cumulative_gpu_seconds']/3600
wall=(datetime.fromisoformat(state['finished_at'])-datetime.fromisoformat(state['launched_at'])).total_seconds()/3600
params=source['graph_descriptions']

report=f"""# FloodDistill v7: audited overnight results and research direction

12 October 2026, Europe/Berlin. This report answers the morning status and analysis request using independently recomputed source receipts, integer external counts, mechanism denominators, deployment artifacts and worker state. It supersedes the interpretation and cohort sentence in the automatic `report_v7.md`; the original report, frozen code, checkpoints and receipts remain preserved.

**The result in one paragraph.** The overnight programme completed successfully as an execution: 34 new training runs, the diagnostics and all collection jobs finished within budget. Repaired A0 and A1 baselines substantially narrow the old capacity story. R4 remains a strong accuracy/storage candidate at 12,230 parameters, but its source superiority over competent A0 is unresolved. Additional live soft KD provides a modest pooled external benefit over cached-hard focal training, without a demonstrated source gain or a consistent event-level advantage. The proposed spectral contribution plus disagreement weighting fails its prospective gate: every C0–C5 arm selects its starting checkpoint, and final C3 is worse than its controls. The most defensible direction is to validate deployment and event/robustness tradeoffs of the established small student before pursuing another auxiliary KD mechanism.

**Status and issues.** Worker PID 1345926 finished at {local(state['finished_at'])}; the final collector completed at 04:09:03 CEST, before the 07:44:04 CEST deadline. Worker wall duration was {wall:.2f} hours. The live morning host probe found no GPU compute processes. No worker or training job remains running, and no future jobs are queued by this finished worker.

"""
dispositions=[]
for name,b in state['bundles'].items():
    count=sum(x['job']['bundle']==name and x['status']=='completed' for x in state['jobs'].values())
    dispositions.append([name,b['status'],count,b.get('reason') or 'Complete declared work'])
report+=table(['Bundle','Disposition','Completed jobs','Explanation'],dispositions)
report+=f"""

The 39 completed worker jobs comprise 34 scientific trainings plus one TRAIN diagnostic and four terminal/reference/mechanism/deployment commands. Eight C0/C3/C4/C5 confirmation jobs for seeds 91/107 were `not_run` because the C3 scientific gate failed. They are neither technical failures nor budget deferrals. Both optional bundles, optimized A3 CE and the complete A1 re-anchor, fit and finished. Twenty compatible historical training anchors were reused; aliases do not increase the number of independent seeds.

Incremental GPU-command spend was **{newh:.6f} hours**, including **487.541 seconds of preflight**, versus an eight-hour cap. Cumulative spend for the audited capacity programme is **{totalh:.6f} hours**, leaving **{30-totalh:.6f} hours** under 30 hours. This is summed elapsed command time including nested inference and failed work, not accelerator utilization or energy. It does not extend the expired v7 wall deadline or authorize resuming a finished manifest.

There were **zero overnight training failures, retries, OOM stops or nonfinite-gradient failures**. One preflight attempt failed before scientific training because a legacy source-split verification key was omitted; it was restored before freeze, the failed attempt was preserved, and its 13.067 seconds are counted. The evaluator's initial TF32 mismatch was repaired before launch; the final multi-model smoke and the full overnight anchor re-evaluation matched inherited integer counts.

The morning operational audit verifies all 272 frozen inputs, 68 new selected/final checkpoint hashes and configurations, all 20 reused anchor identities, all 2,500 restored cohort files and the teacher bytes. Eighteen scratch runs completed 7,800 updates and sixteen continuations completed 1,560, for 165,360 finite-gradient update checks. All four reused R4 LKDs completed 7,800 updates and have exactly the same initial model hashes as their new same-seed CE controls. The historical queue-state hashes are unchanged, with no post-launch work hidden from the ledger.

Three reporting/measurement problems remain material:

- The frozen collector hard-codes the wrong old cohort sentence. The actual v7 protocol and training receipts use the restored original N2500 cohort, zero replacements, SHA256 `1ac4ef17efae3ebb0d3fb8daad38b4541ef04627c07a60e1e2c42e00d2efbc60`. The automatic report's 127-replacement/`80f9...` statement is incorrect. Restored membership is verified; historical generation-time cache hashes/receipt were unavailable, so this is not proof of a byte-identical historical cache reproduction.
- The deployment policy label says `deployment_tf32_off`, but the recorded actual settings are cuDNN TF32 false, CUDA matmul TF32 true and matmul precision `medium`. FP32 below means float32 stored weights and inputs under those actual settings, not strict IEEE float32 arithmetic throughout.
- Absolute GPU allocator peaks include live source-validation image/logit tensors retained from earlier artifacts in the same process. They are process peaks with order-dependent baselines. Baseline-subtracted increments are retained as qualified observations; an isolated per-model memory ranking requires fresh-process measurement.

These problems do not invalidate the training completion receipts or the reproduced terminal integer counts. They do limit the automatic prose and deployment claims. The automatic collector's zero integrity errors cover its programmed hash/count checks, not these additional semantic and benchmark-isolation checks.

**What was compared and what the units mean.** A0 is the 704,714-parameter EfficientViT-B0 segmentation graph; A1 is its 190,870-parameter half-width rung. A3 has 30,922 parameters; A3N removes its LiteMLA context modules while retaining spatial convolutional branches and has 20,250. R4 is the declared 12,230-parameter compact spatial CNN: a six-band stride-two stem, depthwise separable residual stages and a skip decoder. It has no attention but does have spatial context. These are fixed declared graphs, not NAS.

The frozen Prithvi checkpoint, restored 2,500-tile pseudo-labelled TRAIN cohort, six-band order and normalization are unchanged. New scratch runs completed exactly 7,800 updates; continuations completed exactly 1,560. FP32 training, batch 16, AdamW, weight decay 0.05, gradient clipping 5, epoch-stepped cosine and source validation every 312 updates remain declared. S11 validation selects checkpoints and LR; all external sets are already research-exposed secondary readouts. Their scores did not change training gates or admission.

CE means the focal term on cached teacher hard masks. It is still teacher-derived supervision. LKD adds live teacher probabilities on the augmented image. These comparisons do not compare a foundation-model teacher against teacher-free human-labelled training. Scratch CE uses the common LR chosen for the LKD graph, rather than a separately optimized CE LR search. Historical R1/R2/R3 early-stop exceptions remain disclosed in inherited provenance; they are not silently relabelled as full-budget runs.

All scores below are percentages; differences are percentage points (pp). IoU is intersection divided by union, not pixel accuracy. Selected and final checkpoints are separate outcomes. Four-seed intervals use paired differences, sample SD and Student-t df=3; two-seed pilots are descriptive. Event intervals resample 10,000 whole events with seed 1729, conditional on the trained checkpoints. They do not quantify training uncertainty or treat pixels as independent samples. Empty-union event water IoUs stay undefined and are omitted from event means. Fixed teacher/rule predictions are not four independently trained comparators.

**The competent baseline repair.** Both A0 and A1 choose LR 0.005 from the declared 0.00005/0.0005/0.005 seed-42 grid, using source validation only. The chosen recipes then have all four seeds. A0 improves by {ci(sc('A0_reanchored_LKD','A0_old_LKD'))} pp and A1 by {ci(sc('A1_reanchored_LKD','A1_old_LKD'))} pp. The LR choice itself is validation-selected, so these are effects of the selected repair within the tested recipe, not an unbiased search-free estimate.

"""
grid=[]
for g,d in source['LR_grids'].items():
    for c in d['candidates']:grid.append([g,42,f"{c['lr']:g}",f"{c['selected_mIoU']:.4f}",c['selected_updates'],c['status']])
report+=table(['Graph','Grid seed','LR','Selected source mIoU','Selected update','Origin'],grid)
rows=[]
for name,label in [('A0_old_LKD','A0 old LKD'),('A0_reanchored_LKD','A0 repaired LKD'),('A1_old_LKD','A1 old LKD'),('A1_reanchored_LKD','A1 repaired LKD'),('A3N_LKD','A3N LKD'),('A3_LKD','A3 LKD'),('A3_CE','A3 cached-hard CE'),('R4_LKD','R4 LKD'),('R4_CE','R4 cached-hard CE')]:
    g=source['groups'][name];v=g['selected']['mIoU'];rows.append([label,f"{g['parameters']:,}",f"{g['lr']:g}",f"{v['mean_percent']:.2f} ± {v['sample_sd_percent']:.2f}",f"{g['selected']['water_IoU']['mean_percent']:.2f}",f"{g['final']['mIoU']['mean_percent']:.2f}"])
report+='\n\n'+table(['Four-seed family','Parameters','LR','Selected source mIoU ± SD','Selected water IoU','Final source mIoU'],rows)
report+="""

![Baseline repair and fixed-graph source comparison. Sample SD error bars are not confidence intervals.](analysis_20261012/source_reanchor.png)

The matched-objective primary comparison is R4 LKD minus repaired A0 LKD: **+0.62 pp, 95% CI [−0.04, +1.28]**. All four paired source differences are positive, but this interval does not exclude zero. R4 has 57.6 times fewer parameters; that is a parameter comparison, not a measured 57.6-times reduction in artifact bytes, runtime or memory. Against repaired A1 the matched source difference is +1.20 [0.59, 1.81]; against A3 it is +1.37 [0.41, 2.33].

The final-update R4 LKD minus repaired A0 contrast reverses to −0.33 [−0.81, +0.14]. Thus the defensible result belongs to the declared selected-checkpoint pipeline, not to every endpoint of training. R4 CE's selected source score is higher still, but comparing CE against A0 LKD changes the objective and cannot isolate an architecture effect. Within the same cached-hard objective R4 CE exceeds A3 CE by +1.54 [0.62, 2.46]. These findings support R4 as a practical tested graph; they do not show that smaller models generally dominate well-trained larger models.

**Does soft distillation help under the common recipe?**

"""
contrastrows=[]
for left,right,label in [('R4_LKD','R4_CE','R4 LKD − cached-hard CE'),('A3_LKD','A3_CE','A3 LKD − cached-hard CE'),('R4_LKD','A0_reanchored_LKD','R4 LKD − repaired A0'),('R4_LKD','A1_reanchored_LKD','R4 LKD − repaired A1')]:
    contrastrows.append([label,ci(sc(left,right)),ci(sc(left,right,metric='water_IoU'))])
report+=table(['Selected source comparison','mIoU difference and 95% CI (pp)','Water-IoU difference and 95% CI (pp)'],contrastrows)
report+="""

Neither R4 nor A3 has a demonstrated source-validation improvement from adding soft targets at the competent common LR. Nonsignificance is not evidence of equivalence or that soft targets are universally useless. The external results below show a smaller, domain-dependent benefit, chiefly a recall/precision tradeoff on STURM. The old universal claim that smaller models benefit more from KD remains unsupported.

**The controlled novelty pilot.** C0 is uniform LKD; C1 adds the two-parameter MNDWI contribution; C2 adds disagreement weighting; C3 combines both; C4 uses stratified shuffled disagreement; C5 uses teacher-entropy emphasis at matched emphasis counts. HC/HL are target-refresh controls, described next. Every arm/seed starts from the same selected R4 parent, with identical initial base logits and first augmented batches. All 12 pilot runs completed 1,560 updates with finite gradients.

The rule is computed from denormalized green/SWIR1 reflectance; invalid/negative/near-zero-sum pixels are rule-unavailable. The contribution changes the water-minus-land margin by a*r+b, with a=8*tanh(alpha), b=8*tanh(beta), both initialized to zero. KD weights are detached and mean-normalized over valid pixels; that normalization does not guarantee equal realized KL loss or gradient norms.

"""
descriptions={'C0':'Uniform LKD','C1':'Rule + uniform LKD','C2':'Disagreement weighting','C3':'Rule + disagreement','C4':'Rule + shuffled control','C5':'Rule + entropy control','HC':'Cached one-hot KD control','HL':'Live one-hot KD control'}
rows=[]
for a in arms:
    p=source['pilot_summary'][a];sels=', '.join(str(x['selected_updates']) for x in source['pilot_per_seed'][a].values())
    rows.append([a,descriptions[a],f"{p['selected']['mIoU']['mean_percent']:.4f}",sels,f"{p['final']['mIoU']['mean_percent']:.4f}",f"{p['final_mean_gain_over_start_pp']:+.4f}"])
report+=table(['Arm','Intervention','Selected source mean','Selected updates (42/64)','Final source mean','Final − start (pp)'],rows)
report+="""

The correct same-two-seed starting mean is **87.6666**, not the four-seed R4 mean 87.9399. All C0–C5 selected_updates are zero. The identical selected external and mechanism scores therefore describe the two parent models, rather than six independently improved methods. File hashes differ because checkpoint wrappers/metadata differ; their selected predictions are identical.

C3 minus C0/C4/C5 is exactly zero at selection. It misses the required +0.50 pp mean gain over C0 and +0.20 over each control. Integrity checks passed; the eight confirmation jobs were correctly not launched. At the final update, C3 minus C0 is −0.6093 pp, minus C4 −0.1995 and minus C5 −0.1123. Both weighting controls also fall below their starts. C1's final +0.0651 pp over C0 is descriptive and does not yield a selected gain. A failure of this two-seed, fixed continuation protocol does not rule out every spectral integration or selective KD design.

Training did change the model: final C3 a values are approximately 0.5522/0.5306 and b values −0.0893/−0.0826. The selected rule coefficients remain zero. Thus the result is not caused by an inactive correction or skipped optimizer; the learned endpoints simply did not win the registered source selection. Weight emphasis reduced effective sample size to about 75.6% of valid pixels. For C3 the saved weighted-KL logit gradient was about 12% stronger than its unweighted counterpart, despite mean weight one. These optimization diagnostics are dependent step samples and are descriptive.

![Pilot selected versus final endpoints. Two dots are the two seeds, without an inferential CI.](analysis_20261012/pilot_selected_final.png)

**Target refresh versus target softness.** On 128 hash-selected augmented TRAIN tiles, cached hard versus current live hard disagreement is **6.5327%** of valid pixels. Geometry-only discrepancy is 4.7912%; changing photometric gain changes live answers on 3.5191%. Those rates have different comparison sets and are not additive. Live teacher–rule disagreement on rule-available TRAIN pixels is 23.1376%. TRAIN is pseudo-labelled, so it supplies no independent correctness reference.

The ≥1% prospective trigger correctly admitted HC and HL. HC uses cached one-hot targets in the second term; HL uses live one-hot targets; C0 uses live soft targets. HC seed 64 selects update 936 for only +0.0078 pp source mIoU, while HC seed 42 and both HL/C0 seeds keep their starts. Final HL exceeds HC by +0.1623 pp, but final C0 minus HL is −0.0065 pp with mixed seed signs. All fall below their starts. This experiment establishes a target mismatch and tests the controls; it does not convincingly attribute a useful gain to refresh or softness. One-hot and soft targets also differ in gradient strength under the shared loss scaling.

**External transfer: pooled pixels and equal events.** The complete CSVs retain every completed arm, including seed-42 LR alternatives and negative outcomes. The table below summarizes the main four-seed selected families, plus one fixed teacher and rule. Pilot selected C0–C5/HL values are identical to their two-seed parents; HC's small changed selected model is retained separately in the full tables.

"""
families=['A0_LKD_reanchored','A1_LKD_reanchored','A3N_LKD','A3_LKD','A3_CE','R4_CE','R4_LKD','teacher','MNDWI>0']
labels={'A0_LKD_reanchored':'A0 repaired LKD','A1_LKD_reanchored':'A1 repaired LKD','A3N_LKD':'A3N LKD','A3_LKD':'A3 LKD','A3_CE':'A3 cached-hard CE','R4_CE':'R4 cached-hard CE','R4_LKD':'R4 LKD','teacher':'Teacher (fixed)','MNDWI>0':'MNDWI > 0 (fixed)'}
rows=[]
for f in families:
    a,b,c=[es(f,d) for d in ['s11','sturm','wf']]
    rows.append([labels[f],f"{a['water_IoU_mean_pct']:.2f}",f"{b['water_IoU_mean_pct']:.2f}",f"{b['event_macro_water_IoU_mean_pct']:.2f}",f"{c['water_IoU_mean_pct']:.2f}",f"{c['event_macro_water_IoU_mean_pct']:.2f}"])
report+=table(['Selected family','S11-test pooled water IoU','STURM pooled water IoU','STURM event water IoU','WF pooled water IoU','WF event water IoU'],rows)
report+="""

For R4 LKD minus cached-hard CE, pooled STURM water IoU improves **+1.17 [0.27, 2.07] pp** and pooled WF improves **+0.46 [0.03, 0.90] pp**, using four-seed uncertainty on each fixed benchmark. Their equal-event differences are +1.47 [−0.20, 3.60] on STURM and +0.60 [−0.60, 1.78] on WF, conditional on the four checkpoints. Those event intervals cross zero. The lower bounds near zero should not be oversold amid many secondary comparisons; no multiple-comparison correction or untouched confirmation was declared.

On STURM, LKD raises R4 recall from 76.49% to 80.08%, while precision drops from 78.98% to 77.14%. The fixed MNDWI rule has recall 87.03% and precision 71.32%. This is a precision/recall tradeoff, not general superiority in every operational criterion. Complete precision/recall, background IoU and mIoU values are retained in the appended data tables.

R4 LKD versus repaired A0 is +3.74 [1.74, 5.75] pooled S11-test water-IoU points, −0.63 [−1.33, 0.06] on STURM, and +1.82 [0.96, 2.68] on WF. Its WF equal-event difference from A0 is +0.86 [−0.61, 1.92]. Hence the small-model result survives as a useful tested tradeoff, not as uniform transfer dominance.

The teacher comparison changes with the estimand: R4 LKD gains **+1.63 [0.42, 2.83] pp pooled WF water IoU**, but loses **−2.95 [−6.99, −0.09] pp equal-event WF water IoU**. EMSR342 supplies 82.0617% of reference-water pixels, making pooled WF especially dominated by that event. There are 11 WF events, 29 STURM events and 10 S11-test events. Neither pooled gains nor a large number of pixels establishes a teacher-equivalent external capacity floor.

Relative to MNDWI, R4's pooled water-IoU difference is +17.46 on S11-test, +0.24 on STURM and +2.14 on WF. Equal-event differences are +11.93 [2.88, 21.62], −1.42 [−3.39, 0.58], and +7.53 [−0.98, 19.94], respectively. The cheap rule remains competitive on STURM and its high recall is operationally relevant. V7 adds clean-transfer and mechanism evidence; it did not run a new common-corruption or dry-scene experiment.

![WF shows why pooled-pixel and equal-event uncertainty cannot be substituted for each other.](analysis_20261012/wf_estimands.png)

Full and known-overlap-excluded domains remain separate. STURM exclusion removes three of 2,675 chips; WF removes 52 of 785 windows. The principal WF teacher deficit persists: R4 minus teacher event water IoU is −3.15 [−7.21, −0.22] after exclusion. Exclusion does not undo prior research exposure or create a clean holdout.

"""
rows=[]
for f in ['R4_LKD','R4_CE','A0_LKD_reanchored','teacher','MNDWI>0']:
    for d in ['sturm','wf']:
        a,b=[es(f,d,dom) for dom in ['full','known_overlap_excluded']]
        rows.append([labels[f],d,f"{a['water_IoU_mean_pct']:.2f}",f"{b['water_IoU_mean_pct']:.2f}",f"{a['event_macro_water_IoU_mean_pct']:.2f}",f"{b['event_macro_water_IoU_mean_pct']:.2f}"])
report+=table(['Family','Dataset','Pooled full','Pooled excluded','Event full','Event excluded'],rows)
report+="""

**What the teacher transfers beyond the spectral rule.** A correction is a student-correct prediction where the rule is wrong and teacher correct. Harm is a student-wrong prediction where the rule is correct and teacher wrong. Rescue is student-correct where both comparators are wrong. These use exactly the same evaluated valid/reference pixels; rule-unavailable pixels are excluded from these conditional groups and explicitly counted. Rates below average the four selected R4 seeds; the conditioning denominators are the fixed pixel counts per checkpoint, not four independent geographic samples.

"""
rows=[]
for d in ['s11','sturm','wf']:
    z=mechanism('R4_LKD',d)
    rows.append([d,f"{z['correction_denominator_per_checkpoint']:,}",f"{z['harm_denominator_per_checkpoint']:,}",f"{100*z['signed_teacher_rule_benefit']:+.2f}%",f"{100*z['correction_rate_mean']:.2f}%",f"{100*z['harm_rate_mean']:.2f}%",f"{100*z['rescue_rate_mean']:.2f}%"])
report+=table(['Dataset','Teacher-correct/rule-wrong pixels','Teacher-wrong/rule-correct pixels','Signed teacher benefit on disagreements','R4 correction','R4 harm','R4 rescue'],rows)
report+="""

Signed teacher benefit is (teacher-correct/rule-wrong count minus teacher-wrong/rule-correct count) divided by disagreements. On a valid binary disagreement exactly one comparator is correct. The teacher wins about 59.17% of STURM disagreements and 61.73% of WF disagreements, versus 85.11% in source validation. Thus useful teacher information beyond the rule is present, but so are substantial teacher mistakes. Positive pooled signed benefit is not an event-wise reliability guarantee.

The rescue-group denominators per checkpoint are 230,626 pixels on S11-test, 65,014,874 on STURM and 8,792,035 on WF. Rule-unavailable valid pixels number 51,676 of 20,517,367 on S11-test and zero on STURM/WF. Exact per-event conditioning frequencies, sample sizes and empty groups are retained in the mechanism CSVs; low rescue rates should be read against these denominators.

EMSR342 is the only WF event with negative teacher-versus-rule disagreement benefit: **−14.42%**, with 1,860,257 teacher wins and 2,487,382 rule wins. The other ten WF events have positive teacher benefit. WF R4 LKD correction is 90.84%, compared with 88.51% for R4 CE; its harm is also higher, 63.76% versus 61.47%. Both useful correction and teacher-following error increase. Those associations do not prove that the soft loss caused each individual error.

This diagnostic explains why blindly emphasizing every disagreement is not guaranteed to transfer useful knowledge. It does not explain the source pilot failure causally: the pilot failed on source selection, whereas the WF event diagnosis is a secondary external analysis. Selected C3 is its parent and has no new correction-versus-harm outcome. Reference labels do not establish shadow, turbidity or boundary causes without an appropriate stratified diagnostic; unlabelled TRAIN cannot supply correctness labels.

The independent external audit checked 56 terminal records, 198,800 integer per-image cells and 163,300 mechanism cells. All 42,600 image/model comparisons for the 12 re-evaluated historical R4/A3/A3N anchors match their saved counts exactly. Some old A0/A1 recipe anchors have no same-pixel mechanism pass and remain `not_run` there; that does not erase their existing accuracy readouts. These counts include repeated model evaluations of the same images and must not be called independent observations.

**Measured deployment and its limits.** Measurements use the existing selected seed-42 R4/A3N/A3 artifacts, batch one at 1×6×512×512 on one RTX 4090, teacher unloaded, 20 warmups and 100 synchronized forward timings. Input transfer and normalization are excluded. A hash-selected TRAIN tile was resized only for the timing input; source accuracy uses all 89 registered validation images unchanged. The declared INT8 calibration set is TRAIN-only; the builder failed before calibration and no calibrated native INT8 artifact was established. These are workstation measurements, not Jetson, onboard, energy or end-to-end processing measurements.

"""
rows=[]
for x in external['deployment']:
    if x['status']!='completed':continue
    rows.append([x['graph']+' '+x['precision'],f"{x['artifact_bytes']/1024:.2f}",f"{x['parameter_bytes']/1024:.2f}",f"{x['buffer_bytes']/1024:.2f}",f"{x['latency_p50_ms']:.3f} / {x['latency_p95_ms']:.3f}",f"{x['peak_incremental_allocated_bytes']/2**20:.3f}",f"{x['mIoU']:.4f}"])
report+=table(['Artifact (seed 42)','Actual artifact KiB','Parameter KiB','Buffer KiB','p50 / p95 ms','Qualified incremental allocator MiB','Source mIoU'],rows)
report+="""

R4 FP32 is smaller and faster than A3N/A3 among these measured artifacts. R4 FP16 reduces the serialized artifact from 185,419 to 162,411 bytes, only 12.4%, because serialization/runtime metadata are significant at this scale; halving parameter bytes does not halve the deployed artifact. Its p50 latency improves about 3.0%, from 1.318 to 1.279 ms, in one benchmark session. There is no demonstrated material FP16 throughput gain across repeated sessions or target hardware.

FP32 traced/reloaded artifacts reproduce their corresponding eager trace outputs exactly on the parity sample. R4 FP16 differs from the FP32 predictions on 571 of 20,294,725 valid validation pixels, **0.00281%**; source mIoU changes by approximately −0.00024 pp. This is close agreement, not exact FP16 parity. A3N and A3 strict FP16 are unsupported: pinned interpolation/attention paths promote activations to float32 before half-weight convolutions, producing a dtype mismatch. They were not silently measured as FP32 under an FP16 label.

All three native INT8 attempts are unsupported. The installed TensorRT cu13 builder fails CUDA initialization with error 35 (`cudaErrorInsufficientDriver`) on driver 535.183.01; no engine, INT8 artifact bytes or INT8 speedup were established. No new backend was installed. Existing W4A8/QAT results remain `simulation_only`. The native failure blocks that deployment branch, not the completed science.

Actual backend flags and the nonisolated-memory limitation stated above must accompany these measurements. The absolute allocator peaks must not be used to rank model memory. A3N has fewer static MACs in inherited analysis but is slower than R4 in this CUDA benchmark; MACs, parameters and latency are distinct resource estimands. A0 was not benchmarked in this deployment set, so neither a latency ratio nor an actual artifact-size ratio against A0 has been measured.

**What this supports for the research goal.** For cross-architecture foundation-model distillation for edge water segmentation, v7 strengthens a constrained empirical contribution: a declared very small spatial student remains competitive after repairing larger baselines, and the teacher transfers both useful rule corrections and errors. It also resolves a negative intervention result with matched controls. It does not establish a novel superior selective-KD method, a universal size/KD law, a joint external teacher-equivalent capacity floor, packed INT4/INT8 speedups, independently verified dry-scene reliability, or temporal flood increments. The labels and protocol primarily measure surface water.

Known ingredients predate this result. Spectral input integration is explicitly studied in [Bridging Spectral Indices and Deep Learning](https://link.springer.com/article/10.1007/s41976-026-00290-3), published 13 June 2026. [Disagreement-Guided Knowledge Distillation for Efficient Kidney Segmentation](https://www.mdpi.com/2076-3417/16/11/5573), published 2 June 2026, uses teacher–student disagreement rather than the teacher–rule mask tested here. It is related prior art, not a reproduction of this protocol. [LSQ](https://arxiv.org/abs/1902.08153) already learns quantizer steps with scaled gradients; [Trio-ViT](https://arxiv.org/abs/2405.03882) addresses quantization and acceleration of softmax-free efficient ViTs. This bounded primary-source check supports cautious novelty framing; it does not establish priority for a new combination or transfer reported speedups to this hardware.

**Where to go next, in order.**

1. Preserve the completed v7 evidence and use this corrected morning report. Keep R4 CE and R4 LKD as the two useful selected candidates rather than treating C3 as a success. Retain the repaired A0/A1 controls and report selected and final source results. Do not spend the remaining budget finishing confirmation for a failed gate or chasing a favorable external seed.
2. Close the measurement gaps before a new training method. In a new separately frozen measurement manifest, benchmark each artifact in a fresh process; record actual backend flags after all imports and model construction; measure live/peak allocated and reserved memory; repeat latency and full prediction parity. Use a supported backend compatible with the installed driver, or the actual edge device if accessible, with TRAIN-only calibration. This is the next concrete evidence needed for an edge-deployment claim. It is not an instruction to upgrade the current host stack or restart v7.
3. Test operational transfer of the established candidates. Use a prespecified small common robustness set for R4 CE/LKD, repaired A0, teacher and MNDWI, with inherited gain/haze/tilt severities, reflectance-space changes and unchanged masks. Keep equal-event precision/recall and correction/harm reporting alongside pooled scores. A clean event holdout is needed before a confirmatory transfer claim; current external sets remain exposed. Independent dry references are still required for dry-scene false alarms, and teacher-empty masks cannot substitute for them. V7 alone cannot close the original cross-architecture/OFA, dry-scene or real-atmosphere questions.
4. If low-bit novelty remains a priority, first determine whether it improves an actual constraint beyond the already-small FP16 artifact. Preserve the inherited matched FP continuation and adopted integer simulator. Establish structural stem precision, graph-verified signed/unsigned ranges and clipping, then a faithful LSQ baseline before testing a log-domain stem or flip-aware loss. The inherited Q48 versus F48 effect −0.25 [−0.81, 0.32] source points and unresolved four-bit activation damage remain separate evidence; v7 did not run new QAT training or validate a packed engine.
5. Revisit selective distillation only as a new prospective hypothesis. All pilots, including uniform C0, degrade at final updates, so the shared continuation optimizer/BN schedule may limit the test; this is a hypothesis, not an established cause. Any future control must separate a competent continuation schedule from the intervention, freeze source-only selection and stopping rules, and budget all seeds and collection before launch. The existing negative result must remain visible. Teacher confidence or disagreement alone is not independent truth, and WF labels must not be used to tune a new weighting rule.

Only about **4.13 cumulative GPU-command hours** remain under the audited 30-hour allowance. The original eight-hour v7 deadline has expired and its worker is terminal. Another broad training matrix, architecture search or unbounded QAT grid would compete with deployment and robustness within that remainder. No new GPU experiment was launched for this report; the next work should use a new bounded manifest after the research priority is chosen. These are recommendations, not queued or completed experiments.

**Reproduction and complete data.** The frozen protocol is `protocol_v7.yaml`; operational evidence is `state.json`, `attempts/`, `budget_ledger.json`, `failure_ledger.json`, `decision_log.jsonl`, `completion_receipt.json`, `provenance.json` and `launch_verification_receipt.json`. Source/evaluation identities and historical exceptions are retained there. The automatic report remains untouched for provenance; rerunning the old collector can reproduce its stale cohort prose, so this morning report is the corrected interpretation.

"""
report+=table(['Morning evidence','Purpose'],[
    ['analysis_20261012/source_stats.json; source_families.csv; source_paired_contrasts.csv','Independent selected/final source means, SD, paired contrasts and LR grids'],
    ['analysis_20261012/pilot_selected_final.csv; loss_diagnostics.csv','All two-seed pilot outcomes; descriptive optimization diagnostics'],
    ['analysis_20261012/external_selected_summary.csv; external_selected_contrasts.csv; selected_per_event_differences.csv','Full/excluded pooled and equal-event results; checkpoint-conditional event intervals'],
    ['analysis_20261012/mechanism_selected_summary.csv; mechanism_selected_events.csv','Integer-derived correction/harm/rescue rates and conditioning groups'],
    ['analysis_20261012/operational_audit.json; external_mechanism_integrity.json; deployment_independent_audit.csv','Independent execution, input/checkpoint/count parity and artifact identities; precision/memory caveats'],
    ['results.csv; comparisons.csv; event_results.csv; mechanism_counts.csv; deployment.csv','Automatic complete arm/seed readouts with explicit missing values']])
report+="""

CPU-only reproduction from the run directory uses a fresh sibling directory because the source and diagnostic helpers preserve existing outputs:

```bash
P=/home/schmalstieg/miniconda3/envs/esi/bin/python
V7_ANALYSIS_DIR=$(mktemp -d "$PWD/analysis_verify_XXXXXX")
cp analysis_20261012/source_analysis.py "$V7_ANALYSIS_DIR/"
cp analysis_20261012/export_source_tables.py "$V7_ANALYSIS_DIR/"
cp analysis_20261012/analyze_external_mechanisms.py "$V7_ANALYSIS_DIR/"
cp analysis_20261012/loss_diagnostic_summary.py "$V7_ANALYSIS_DIR/"
CUDA_VISIBLE_DEVICES= "$P" "$V7_ANALYSIS_DIR/source_analysis.py"
CUDA_VISIBLE_DEVICES= "$P" "$V7_ANALYSIS_DIR/export_source_tables.py"
CUDA_VISIBLE_DEVICES= "$P" "$V7_ANALYSIS_DIR/analyze_external_mechanisms.py"
CUDA_VISIBLE_DEVICES= "$P" "$V7_ANALYSIS_DIR/loss_diagnostic_summary.py"
```

The report/plot rebuilders use the delivered morning analysis and overwrite only the new morning report artifacts:

```bash
P=/home/schmalstieg/miniconda3/envs/esi/bin/python
CUDA_VISIBLE_DEVICES= "$P" analysis_20261012/build_morning_report.py
/usr/bin/python3 analysis_20261012/render_morning_pdf.py
```

The published fresh-directory CPU workflow was executed during this audit: all 16 checked JSON/CSV outputs reproduced byte-identically, including source, external, mechanism, deployment and loss-diagnostic tables. The verification receipt is `analysis_20261012/cpu_reproduction_check.json`.

Independent audit outputs are hash-bound in `analysis_20261012/morning_report_receipt.json`. The source, checkpoint and table evidence remains the authority; the report distinguishes completed/reused, failed scientific gate, unsupported, simulation-only and not-run outcomes without filling missing values or promoting a retrospective criterion.
"""

target=RUN/'report_v7_codex_audit_20261012.md'
target.write_text(report)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[RUN/n for n in ['state.json','budget_ledger.json','provenance.json','manifest_initial.json','failure_ledger.json','completion_receipt.json']]
inputs += [OUT/n for n in ['source_stats.json','external_mechanism_summary.json','external_mechanism_integrity.json','operational_audit.json','cpu_reproduction_check.json']]
receipt={'status':'complete','gpu_work_launched':False,'frozen_inputs_modified':False,
    'source_metrics_unit':'percent','differences_unit':'percentage_points',
    'input_sha256':{str(p):sha(p) for p in inputs},'report_path':str(target),'report_sha256':sha(target),
    'plots':{str(p):sha(p) for p in OUT.glob('*.png')},
    'automatic_report_corrected_by_new_report_only':True,
    'key_limits':['cohort prose corrected','actual TF32 flags disclosed','memory isolation limitation disclosed',
                  'all pilot starts selected','no new external confirmation','no native INT8 engine']}
(OUT/'morning_report_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'report':str(target),'words':len(report.split()),'input_files':len(inputs)}))
