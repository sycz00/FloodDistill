"""Build reviewable, section-anchored drafts without modifying submitted PDFs."""
from datetime import datetime,timezone
import csv
import json
import re
from pathlib import Path

ROOT=Path('/home/schmalstieg/repos/FloodDistill')
OUT=ROOT/'experiments/revision/prompt5_closure'
def read(name):return json.loads((OUT/name).read_text())
def rows(name):return list(csv.DictReader((OUT/name).open()))
def csvwrite(name,records):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def readable(content):
    protected={}
    def protect(match):
        number=len(protected);letters=''
        while True:
            letters=chr(65+number%26)+letters;number=number//26-1
            if number<0:break
        key='PROTECTEDTOKEN'+letters+'TOKEN'
        protected[key]=match.group(0);return key
    content=re.sub(r't1(?:tilt|vis)[+-]\d+|[gh]\d+\.\d+|\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?:B8A|W8A8|MobileNetV3-Small|LightFormer-ResNet18|UNet-ResNet50|EfficientViT-B0|EfficientViT|LightFormer|WorldFloods|mIoU|water_IoU|IoU)',protect,content)
    replacements={
        'teacher-scene':'teacher scene','originalmethodchoices':'original method choices',
        'benchmarkhistory':'benchmark history','currentcode':'current code',
        'dryacquisition':'dry acquisition','dry-acquisition':'dry-acquisition',
        'dryscenes':'dry scenes','dryscene':'dry scene','dryreliability':'dry reliability',
        'dryfalsealarms':'dry false alarms','dryfalsealarm':'dry false alarm',
        'drydataset':'dry dataset','dryreview':'dry review','drytruth':'dry truth',
        'teacher-dryscenes':'teacher-dry scenes','teacher-derivedtargets':'teacher-derived targets',
        'waterprecision':'water precision','waterpixels':'water pixels','surface-watersemantics':'surface-water semantics',
        'checkpointidentity':'checkpoint identity','checkpointancestry':'checkpoint ancestry',
        'deviceaccuracy':'device accuracy','devicetiming':'device timing','deviceinputs':'device inputs',
        'powerrail':'power rail','powerlogs':'power logs','buildidentity':'build identity',
        'full-domain':'full-domain','matchingchain':'matching chain','rawtimingpowerchain':'raw timing/power chain',
        'mappedindex':'mapped index','seedartifact':'seed artifact','mappedchain':'mapped chain',
        'eventintervals':'event intervals','eventbootstrap':'event bootstrap','pairedseed':'paired seed',
        'point-by-point':'point-by-point','source/cachematching':'source/cache matching',
        'falsepositive':'false positive','falsealarms':'false alarms','falsealarm':'false alarm',
        'groundtruth':'ground truth','benchmarkfeedback':'benchmark feedback','benchmarkfeedback':'benchmark feedback',
        'selectionstudy':'selection study','selectionmechanism':'selection mechanism',
        'commonrecipe':'common recipe','currentrecipe':'current recipe','frozenrecipe':'frozen recipe',
        'trainingonly':'training only','traininglabels':'training labels','label-supportedoutcomes':'label-supported outcomes',
        'humanhardlabels':'human hard labels','humanhard':'human hard','pseudohardtargets':'pseudo hard targets',
        'pseudohard':'pseudo hard','livesofttargets':'live soft targets','livesoft':'live soft',
        'hardtargets':'hard targets','softtargets':'soft targets','calibrationscales':'calibration scales',
        'all4seeds':'all 4 seeds','fourseed':'four seed','4seed':'4 seed','4timingseeds':'4 timing seeds',
        'seedSD':'seed SD','seedvariation':'seed variation','seedmeans':'seed means',
        'incompleteancestry':'incomplete ancestry','independentreview':'independent review',
        'independentdry':'independent dry','independentreference':'independent reference',
        'actualletter':'actual letter','exactletter':'exact letter','authorhistory':'author history',
        'authorinput':'author input','authorchoice':'author choice','alreadyasked':'already asked',
        'alreadyqualifies':'already qualifies','notrecovered':'not recovered','notresolved':'not resolved',
        'notincorporated':'not incorporated','universalgain':'universal gain','universalfirst':'universal first',
        'theoreticalnovelty':'theoretical novelty','firstclaim':'first claim','first-in-literatureassertion':'first-in-literature assertion',
        'newtheoreticalnovelty':'new theoretical novelty','unsupportedfirst':'unsupported first',
        'annotationbudget':'annotation budget','compactarchitecturecomparison':'compact architecture comparison',
        'alternativebackbones':'alternative backbones','architecturecomparison':'architecture comparison',
        'globallyoptimal':'globally optimal','globallyoptimality':'global optimality',
        'pretrainingdifferences':'pretraining differences','Unequalcapacity':'Unequal capacity',
        'actualadaptation':'actual adaptation','adaptationsizes':'adaptation sizes','main/exitsizes':'main/exit sizes',
        'auxiliary-loss':'auxiliary-loss','optionalreview':'optional review','clearerMethods':'clearer Methods',
        'matchingartifact':'matching artifact','noindependent':'no independent','noauthorized':'no authorized',
        'noXavier':'no Xavier','no datasetrequired':'no dataset required',
        'Benchmarksreported':'Benchmarks reported','floodfocused':'flood focused','floodfocusedbenchmarksdo':'flood focused benchmarks do',
        'existingpaperacknowledgement':'existing paper acknowledgement','newlyinundatedland':'newly inundated land',
        'temporal/referencewater':'temporal/reference-water','referencedwater':'reference water',
        'referencewater':'reference water','referencegate':'reference gate','labeltruth':'label truth',
        'confounded deltas':'confounded deltas','confoundedlabel':'confounded label',
        'retainedrecords':'retained records','metadata-fixed':'metadata-fixed',
        'allavailable':'all available','No globallyoptimal':'No globally optimal',
        'onlyashistorical':'only as historical','withoutcallingoldnumbersfalse':'without calling old numbers false',
        'mask/evaluatorreceipts':'mask/evaluator receipts','devicefloat':'device float',
        'teacherempty':'teacher empty','teacher-emptybaseline':'teacher-empty baseline',
        'thatlimitation':'that limitation','make thislimitationexplicit':'make this limitation explicit',
        'onceincorporated':'once incorporated','specificanalysispending':'specific analysis pending',
        'compactsurface-water':'compact surface-water','S11devicevalidation':'S11 device validation',
        'S11device':'S11 device','MNDWIthreshold':'MNDWI threshold','withnativeclean':'with native clean',
        'syntheticdiagnostics':'synthetic diagnostics','syntheticdiagnostic':'synthetic diagnostic',
        'realatmosphericvalidation':'real atmospheric validation','realatmospheric':'real atmospheric',
        'manuscriptbody':'manuscript body','oldengine':'old engine','newQAT':'new QAT',
        'rawnamed':'raw named','matchinglogs':'matching logs','newdata':'new data',
        'sourceexposure':'source exposure','source-footprintassessment':'source-footprint assessment',
        'hardtermweight':'hard-term weight','gradientpath':'gradient path','recoveredrecipe':'recovered recipe',
        'currentenvironment':'current environment','currentcontrols':'current controls',
        'historicaldevice':'historical device','historicalcontrol':'historical control',
        'historicalqualification':'historical qualification','historicalprovenance':'historical provenance',
        'precisionrecalltradeoff':'precision/recall trade-off','nooptimality':'no optimality',
        'recentarchitecture':'recent architecture','authorretain/omitchoice':'author retain/omit choice',
        'letterwording':'letter wording','checkpoint/mask/protocol/countsource':'checkpoint/mask/protocol/count source',
        'Current concern status':'Current concern status',
        'nativeWF':'native WF','correctedSTURM':'corrected STURM',
        'label-qualityisolation':'label-quality isolation','noSTresult':'no ST result',
        'pipelinecontribution':'pipeline contribution','not universalwin':'not a universal win',
        'onunattended':'on unattended','nofalsealarm':'no false-alarm',
        'externalgroundtruth':'external ground truth','source-footprint':'source-footprint',
        'matchedcontrols':'matched controls','ablationmap':'ablation map','possibleinformationleakage':'possible information leakage',
        'explicitlyqualify':'explicitly qualify','executedmapped':'executed mapped','activationreplacement':'activation replacement',
        'existingcomparison':'existing comparison','holdsfixed':'holds fixed','andimplemented':'and implemented',
        'sequentialhistorical':'sequential historical','anadditive':'an additive','filteringstudy':'filtering study',
        'submittedpaper':'submitted paper','watersegmentation':'water segmentation','unattendedfalse':'unattended false',
        'reliabilityvalueis':'reliability value is','limiteddiagnostic':'limited diagnostic','independentdispositions':'independent dispositions',
        'establishedcomponents':'established components','budgetquestion':'budget question','isadded':'is added',
        'boundedapplicationcontribution':'bounded application contribution','establishedmethods':'established methods',
        'andgeographyconfounders':'and geography confounders','geometryqualification':'geometry qualification',
        'demonstratedcropnumbers':'demonstrated crop numbers','theirnativewater':'their native water',
        'pooledwaterisremoved':'pooled water is removed','positivechipsbecome':'positive chips become',
        'remainconfounders':'remain confounders','separateconditional':'separate conditional','varybybackbone':'vary by backbone',
        'regeneratedcohortdoesnot':'regenerated cohort does not','collapsedwithoriginal':'collapsed with original',
        'attributeallcohortdifferences':'attribute all cohort differences','regeneratedseparate':'regenerated separate',
        'andlackofalternativebackbone':'and lack of alternative backbone','deviceprofiling':'device profiling',
        'remainseparate':'remain separate','whenlearnedmodelshelp':'when learned models help',
        'perturbationcellsand':'perturbation cells and','distinguishedfromlegacy':'distinguished from legacy',
        'derivedprecision':'derived precision','existingconditions':'existing conditions','deploymentresults':'deployment results',
        'needscorresponding':'needs corresponding','missingidentity':'missing identity','Missingidentity':'Missing identity',
        'arequalified':'are qualified','retainuntracedvalues':'retain untraced values','rail-unverifiedwatts':'rail-unverified watts',
        'orauthorsmayomit':'or authors may omit','thosevalues':'those values','desktopaccuracyis':'desktop accuracy is',
        'assignedto':'assigned to','isnotcalledtest':'is not called test','Specificdeviceinputlist':'Specific device input list',
        'semanticlimits':'semantic limits','deploymentevidence':'deployment evidence','closuretable':'closure table',
        'identifyactualsection':'identify actual section','replacementpassages':'replacement passages','arepreserved':'are preserved',
        'editablepatch':'editable patch','letterverification':'letter verification','isclaimed':'is claimed',
        'NumericalSTsensitivitywillbeinsertedonlyafterall':'Numerical ST sensitivity will be inserted only after all',
        'countreceiptsandcollectorpass':'count receipts and the collector pass','completepackage':'complete package',
        'responseentries':'response entries','section-anchoredpassages':'section-anchored passages',
        'publishedbenchmark':'published benchmark','exitparameters':'exit parameters','historicalkernel':'historical kernel',
        'quantitativepowerclaims':'quantitative power claims','usesvalidation':'uses validation',
        'copiedcheckpointrescoring':'copied checkpoint rescoring','versusregeneratedcohorts':'versus regenerated cohorts',
        'archivedseedmean':'archived seed mean','printedreferencevalues':'printed reference values',
        'whichbenchmarks':'which benchmarks','launchercomments':'launcher comments','trainingprerequisite':'training prerequisite',
        'uncompletedtemplate':'uncompleted template','noverifiedreference':'no verified reference',
        'revieweridentity':'reviewer identity','tileassessability':'tile assessability','blindlyresumed':'blindly resumed',
        'referenceimport':'reference import','unknownreferencepixels':'unknown reference pixels',
        'fixedalarmfractions':'fixed alarm fractions','retainedunqualified':'retained unqualified',
        'numericalclaims':'numerical claims','matchingoriginalfloat':'matching original float',
        'engineidentitychain':'engine identity chain','measurementsare':'measurements are',
        'inferredfrompeakwatts':'inferred from peak watts','authorizedtargethasbeensupplied':'authorized target has been supplied',
        'unsupportedprecision':'unsupported precision','oldnumbersfalse':'old numbers false',
        'dependentcountcollector':'dependent count collector','authorizedjobs':'authorized jobs','existingrobustness':'existing robustness',
        'scoredregions':'scored regions','selectedpatch':'selected patch','their actualadapted':'their actual adapted',
        'unresolvedFPSgain':'unresolved FPS gain','unresolvedFPS':'unresolved FPS',
        'changedhard':'changed hard','withoutlog':'without log','ofdry':'of dry','andlimits':'and limits',
        'andteacher':'and teacher','andgeography':'and geography','forlabel':'for label','andlive':'and live',
        'alreadylimits':'already limits','forthat':'for that','thefilter':'the filter','newdry':'new dry',
        'teacher-emptybaseline':'teacher-empty baseline','addressedonce':'addressed once',
        'isnot':'is not','asR':'as R','orunsupported':'or unsupported',
        'byimage':'by image','Differentimagery':'Different imagery','andgeography':'and geography',
        'andsensitivitycannotbeinferredfrom':'and sensitivity cannot be inferred from',
        'currentprefix':'current prefix','quantizationcalibration':'quantization calibration',
        'inconsistency':'inconsistency','archived/printed':'archived/printed',
        'retainunfavourable':'retain unfavourable','withoutforcing':'without forcing',
        'original N':'original N','S11test':'S11 test','WFmap':'WF map','WFretainedhazedifference':'WF retained haze difference',
        'kernelmeasurement':'kernel measurement','TableIVrepresentativetimingdoesnot':'Table IV representative timing does not',
        'requirefourtimingseeds':'require four timing seeds','residualoverlap':'residual overlap',
        'QATcalibration':'QAT calibration','STURMchips':'STURM chips','STchips':'ST chips','STchip':'ST chip',
        'plusauthor':'plus author','Effcontrols':'Eff controls','adaptedsize':'adapted size','auxpolicy':'aux policy',
        'Restoredsource':'Restored source','Ablationmap':'Ablation map','FPSnull':'FPS null',
        'Suppliedpaper':'Supplied paper','Newlyinundatedland':'Newly inundated land',
        'Proposedchanges':'Proposed changes','softroles':'soft roles','Conclusionacknowledgement':'Conclusion acknowledgement',
        'Teacher-emptybaseline':'Teacher-empty baseline','Existingpaperacknowledgement':'Existing paper acknowledgement',
        'TableIIIfour':'Table III four','all4':'all 4','forR':'for R',
        'andQAT':'and QAT','andOFA':'and OFA','Backgroundclass':'Background class','clipping to[':'clipping to [',
    }
    for old,new in sorted(replacements.items(),key=lambda v:-len(v[0])):content=content.replace(old,new)
    content=re.sub(r'(?<=[a-z])(?=\d)',' ',content)
    content=re.sub(r'(?<=\d)(?=[A-Za-z])',' ',content)
    content=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',content)
    content=re.sub(r'(?<=[,;:])(?=[A-Za-z])',' ',content)
    for key,value in protected.items():content=content.replace(key,value)
    content=re.sub(r'(?<=\d{3})(?=[a-z])',' ',content)
    content=re.sub(r'(mIoU|IoU)(?=\d)',r'\1 ',content)
    content=content.replace('EfficientViTOFA','EfficientViT OFA').replace('STURMmIoU','STURM mIoU')
    assert 'PROTECTEDTOKEN' not in content
    return content
def text(name,content):(OUT/name).write_text(readable(content.strip())+'\n')

ROLES=[
('Teacher task adaptation','252 S11 training scenes, human surface-water labels','Public frozen task-adapted300M teacher; current checkpoint pinned; adaptation performed before student comparisons','Human annotation budget252; no teacher training in revision','III-A p3; teacher_trace.json'),
('Direct hard supervision','S11 training human masks','Hard-only trueCE; S11 KD/OFA retain human hard masks and add frozen teacher soft targets','Not the same images or geometry as GDACS; teacher-independent only hard-onlyS11','III-C p5; TableI p9'),
('Pseudo hard supervision','Selected GDACS original caches','Teacher argmax masks on additional imagery, all inspected selected targets valid','Original selected2500 restored, current source/cache match; historical generation ancestry incomplete','III-A/B pp3-4; original_data gate'),
('Live soft supervision','Same augmented student training image','Frozen eval-mode teacher logits, no gradients; mainKD temperature4','Hard and soft teacher signals are not independent references','III-C p5; frozen training/losses.py'),
('Normalization','S11 training-image band statistics','Six fixed means/std reused on all benchmarks','No external recalibration; radiometric shift remains','AppendixA p14; data/channel_adapter.py'),
('Prefilter','Original pseudo pool of23720 tiles','Water fraction[.05,.85], stored max-class-confidence spatialSD>=.05; retained5987','Recorded index establishes executed constants, not their chronological predeclaration; dry tiles removed','III-B p4; original_N2500_index.json'),
('FPS descriptor/ranking','Same5987 filtered candidates','15D: pseudo fraction, confidence mean/SD, six image-band means/SD; empirical curation choices','Existing launcher comments cite external scores in selector rationale; author history unresolved. No stable full-budgetFPS gain in suppliedS2','III-B p4; supplementS2 pp1-2'),
('Selection alternatives','Filtered pool plus S11 validation for GLISTER surrogate','Existing selector study compares FPS/confidence/GLISTER/R2-Core/uniform; uniform-unfiltered changes candidate pool','Development/calibration involvement disclosed; different selector recipe not merged with matchedcontrols','SupplementS2 pp1-2'),
('Student checkpoint selection','89 S11 validation scenes','Every2epochs maximum pooled mIoU, strict> tie, patience20checks; ceiling50epochs','Validation legitimately informs selection; actual updates vary. S11test/external scores not used in current revision selection','IV-A/B pp6-7; training_all_44.csv'),
('Activation/QAT calibration','Historical deployment route; S11 validation images','Image-based calibration; GDACS pseudo targets for QAT; S11 validation labels select QAT checkpoint','Calibration batches carry masks but calibration consumes images. Weights adapt, fixed scales and BN; exact device ancestry absent','III-D pp5-6; supplementS7 pp6-7'),
('External numerical readout','STURM2675chips/46maps/29events; WF785windows/18maps/11events; S11test','Frozen readout; corrected finite STURM support and native WF clean; legacy panels separate','Different-date source exposure exists. Readout is not proof of historical non-use in development','IV-A p6; overlap assessment'),
('Overlap sensitivity','Metadata-defined3STURMchips and1WFmap','Same complete-unit exclusions for all38 comparators; pooled integer-count reconstruction','Known-overlap exclusion is not newly untouched evaluation or exhaustive geographic decorrelation','SupplementS1 addition; sensitivity_spec.json'),
('Development feedback','Historical manuscript disclosure and retained launcher comments','Submitted wording: Method choices were refined using the same public benchmarks reported in the paper.','Author must identify choices/benchmarks/timing; no clean-test history inferred from current code','VII Limitations p13; remaining_inputs.txt'),
]

ABLATIONS=[
('Prefilter','SuppliedS2 uniform filtered vs uniform unfiltered','Budget/selector family/training recipe within supplied study','Candidate eligibility, water prevalence and geographic mixture','Up to.048 clean mIoU cost without filter in supplied study; dry reliability sacrificed','SupplementS2 pp1-2; supplied selector source data','Historical study; not an additive pipeline contribution'),
('FPS','SuppliedS2 filtered selector family','Filtered candidate pool and within-study recipe','Selected imagery/descriptor-based ranking; GLISTER uses validation surrogate','Clean spread.012mIoU comparable to seed variation; no resolved full-budgetFPS advantage','SupplementS2 pp1-2/FigureS1','Do not reuse selector-study scores as new matched controls'),
('Historical hard-only vsKD','V4 original-data groupA and original rescoring','Teacher/source budget/optimizer and selection policy','TrueCE becomes.5focal+.5KL; historical KD/OFA clipping also differs','Historical CE->KD delta does not isolate addition of soft targets','tables/original_matched_seed_summary.csv; groupA recipes','Recovered recipe in current environment; not bitwise historical training'),
('Focal vsKD','V4 clipping-matched full focal vs.5focal+.5KL','Original pool/budget, batch16, FP32, AdamW, clipping5, checkpoint policy','Hard-term coefficient halves while soft term is introduced','Useful matched composite objective control, not a pure soft-only additive effect','prompt4 tables weighted_KD_minus_full_weight_focal','All4seeds retained; increased hard-only variability retained'),
('OFA vsKD','Original-data matched EfficientViT/MBV3/LF; EffN100','Backbone/budget, main focal+KD, valid reduction, clipping5, selection policy','Four training-only exits with.3mean branch loss; auxiliary trainable heads','Effects vary by backbone/budget/benchmark; see OFA_matched_effects.csv','tables/OFA_matched_effects.csv; OFA_matched_effects.pdf','Seed SD and event bootstrap are separate; no universal accuracy gain'),
('Activation replacement','SubmittedTableIII Hardswish baseline vs independently trainedReLU6 device-float row','Nominal selected pseudo pool and deployment target','Activation, initialization/training history, platform/evaluator chain','Historical sequential delta includes retraining; do not attribute entire difference to activation alone','MainIII-D pp5-6/TableIII p13; supplementS7','Historical device artifacts incompletely traced; no new ReLU6 training'),
('QAT','SubmittedTableIII matched device-float vs mixed engine if artifact chain recovered','Required same adapted activation/checkpoint ancestry, masks/split, target evaluator','Fixed-scale fake quantization plus QAT weight adaptation and mixed precision','Historical proposed contrast requires float/QAT/engine identity; latency-only strippedQAT export is not validated preQAT accuracy comparator','TableIII p13; supplementS7 pp6-7; deployment reinspection','Fourseed accuracy ancestry missing; representative TableIV timing need not have4timing seeds'),
]

MAIN=r"""
PROPOSED MAIN-MANUSCRIPT REPLACEMENTS — NOT INCORPORATED
Only submitted main.pdf and supplement.pdf were found. Page numbers refer to those supplied PDFs. No editable source patch or revised manuscript is claimed. Existing references already in the manuscript remain the attribution base. Exact reviewer quotations await the actual letter.

M1 — Abstract p1; replace the accuracy/deployment outcome sentences
“With a fixed task-adapted Prithvi teacher, additional Sentinel-2 imagery can supervise compact surface-water segmentation without new manual labels. In current-environment training on the restored original N=2,500 pool, EfficientViT-B0 OFA reaches 0.875 mIoU and 0.783 water IoU on the S11 test split, compared with 0.895 and 0.817 for the verified frozen teacher. On the full corrected STURM protocol, their mIoUs are 0.760 and 0.754; on native-geometry WorldFloods they are 0.771 and 0.772. These are protocol-conditional readouts, with known geographic exposure addressed by a separately reported exclusion sensitivity. OFA effects vary with architecture, budget and benchmark. The original Xavier NX deployment demonstrates a target-specific mixed INT8/FP16 configuration; numerical deployment results are retained only with their documented historical measurement scope.”
Source: current_full_summary.csv, current_full_comparator_paired_seed.csv. The STURM exclusion sensitivity is pending; this passage intentionally claims only full-domain scores. Do not attach these desktop checkpoints to the old engine. If authors retain5.57ms/1.5MB in the Abstract, first resolve the specific engine/timing chain in remaining_inputs.txt; use “historically reported1.5MB mixed INT8/FP16 TensorRT engine” with explicit historical qualification if no chain is recovered.

M2 — Introduction contributions p2; replace the novelty/contribution bullet text
“The contribution is an application study of annotation-budget amplification through an existing frozen task teacher, controlled transfer to compact segmentation students, and a target-specific deployment configuration. Pseudo-labeling, subset selection, knowledge distillation, OFA and quantization are established components attributed to the references already cited. The controlled comparisons identify where this combination helps and where it does not: they do not establish a universally superior objective, an optimal configuration for every backbone, or a first-in-literature theoretical mechanism.”
Retain the manuscript's existing geometry qualification and statement that the deployment does not establish universal infeasibility of larger teachers.

M3 — III-A p3, teacher accuracy paragraph; IV-C/TableI/TableII p9 and IV-F p10 teacher baseline references
“The frozen task teacher is evaluated separately from student-seed variation. Verified legacy clean mIoU is0.894572 on S11,0.753917 on STURM and0.774820 on WorldFloods, with water IoU0.816731,0.648660 and0.648150, respectively. Retained historical per-map caches reproduce the printed external values: STURM0.755021 and WF0.770579 round to0.755/0.771, and waterIoU0.643438/0.641028 rounds to0.643/0.641. Those caches have the same map IDs and pooled valid/reference-water totals as the verified readout but different predictions; historical pixel-mask byte identity is not established. The retained old evaluator names runs/teacher_ft/best.pt, which is absent; the caches lack executed checkpoint/hash/runtime receipts, so that code path is a candidate ancestry rather than a proven explanation. The S11 printed0.897/0.822 source remains untraced. Separate official/remapped scalar verification agrees closely with the current verified baseline. These discrepancies are not explained by the0.0192 percentage-point maximum for the28 student checkpoint rescores, and no hardware explanation is inferred.”
Use teacher_baseline_reconciliation.csv for exact displayed values. Historical per-map teacher caches carry the same map IDs but different predictions, without sufficient checkpoint/runtime receipts; they are not silently merged with the separate scalar-verification record. Corrected STURM/nativeWF teacher rows must use the matching corrected counts, not the legacy WF scalar. Historical TableI student results remain in their own panel; do not update its independence/statistical claims from N2500 membership.

M4 — III-B p4, definitions in Eq1/Eq2 and “fixed a priori” sentence
“For each cached tile, let q(u)=max_c p_T(c|u) denote the stored maximum-class confidence. The executed descriptors use mean(q) and the spatial standard deviation of q, together with teacher argmax water fraction and the six image-band means and standard deviations. These are15 descriptive features; spatial dispersion is not calibrated epistemic uncertainty. The retained index uses water fraction between0.05 and0.85 and confidence dispersion at least0.05. These constants and the FPS selection are documented from the retained index; the available records do not establish that their original choice preceded all benchmark feedback. Filtering excludes fully teacher-dry tiles.”
Retain FPS as a recorded selection mechanism. Replace any arbitrary-start description with the implementation's start at the first candidate after descending confidence-dispersion ordering, followed by greedy farthest-point selection in the standardized15D descriptor space. Do not infer a measured geographic-independence guarantee from spectral diversity. The original selection-study supplement already reports no resolved full-budgetFPS advantage; keep it.

M5 — III-C p5, main loss description and branch Eq4–6 explanation
“Historical hard-only finetuning labeledCE uses true cross-entropy. Historical logitKD and OFA use a focal hard term with gamma2, not the same hard-only loss: L_main=0.5 L_focal +0.5 T² KL(softmax(z_T/T)||softmax(z_S/T)), T=4. The dense KL reduction flattens pixels and divides by pixel count. The inherited mainKL includes ignored positions; matched revision controls restrict it to valid hard-target pixels. These reductions coincide on the inspected selected pseudo caches whose pixels are all valid, which is distinct from sensor-valid support. Cross-entropy and focal hard terms exclude target=-1. Historical hard-only/KD use clipping0; historicalOFA and the new matchedcontrols use global gradient clipping5.”
“Each of four OFA exits uses teacher probabilities softmax(z_T/4) and student probabilities softmax(z_exit), with no branchT² multiplier. For valid target y, the exit loss is -(1+p_T(y))^1.2 log p_exit(y) -sum_{c≠y}p_T(c)log p_exit(c), averaged over valid pixels. The total adds0.3 times the mean of the four exit losses. With a binary target, the second term has one non-target class and is not a separately normalized conditional-distributionKL. The hard target is a human annotation for S11 and teacher argmax for GDACS; live soft targets come from the same frozen teacher on the same augmented input.”
“The EfficientViT training exits contain11,368 trainable parameters and are removed for main-network inference. They comprise projection, BatchNorm, activation and classifier layers, rather than488 parameters of simple final classifiers.”
The full-weight focal-only matched control uses L_focal; its comparison withKD also changes the hard-term coefficient from1 to0.5. Preserve this qualification in the ablation map. Do not present measured gradient-path diagnostics as a proven general explanation of the observed budget effect.

M6 — III-D pp5–6; QAT description and heading consistency
Use “Target-specific mixed INT8/FP16 deployment” in deployment headings that currently imply an entirelyINT8 graph.
“QAT adapts the trainable model weights through fixed-scale fake quantization with a straight-through estimator; calibration scales and BatchNorm affine/running statistics are fixed for the retained route. QDQ is applied to convolution inputs and weights; output logits remain floating point, and LiteMLA operations remainFP16 where required. This produces a mixedINT8/FP16 TensorRT engine tied to XavierNX/JetPack5.0/TensorRT8.4.”
Preserve the existing acknowledgement that FP16 is faster. Retain activation failure observations as observations under this route, rather than universal impossibility of Hardswish quantization.

M7 — IV-A p6 and AppendixA pp14–15; STURM/WF definitions
“Our optical STURM readout uses2675chips grouped into46flood maps and29EMS events. Backgroundclass0 remains valid, classes1/2 are water and3–5 are ignored. The release inverse transform is0.04+0.12v, followed by clipping to[0,1] and fixed S11 normalization after selecting the six mapped bands, includingB8 in place ofB8A. Nonfinite input values are replaced by the inherited loader; the corrected score additionally excludes pixels affected by nonfinite support in any of the six selected bands under the executed bilinear interpolation. This correction removes30,672 previously scored pixels from10chips. Clipping also affects543,510finite scored pixel positions on the512×512tensor grid across350chips, without making those pixels invalid. Shared processing makes evaluation consistent across comparators; differently trained models can still respond differently to the same shift. Comparative conclusions remain conditional on this protocol.”
“The historical WF reader retained785windows and resized partial boundary windows to512×512. The corrected clean readout preserves native geometry: normalize each window, zero-pad to512, infer, crop output to its native extent and apply the same cloud/nodata/permanent-water reference mask. Interior counts are reused. Legacy boundary resizing and native-geometry clean results are separate panels.”
Data-role and footprint tables are supplied alongside these paragraphs. Neither the class mapping nor sensor transformation is a published-STURM baseline comparison.

M8 — IV-C p8, geometry diagnostic, replace the sentence containing “86 of the252chips”
“The center crop removes more than half the native positive area from200 of the252training chips, removes73.6263% of pooled native water pixels, and turns21 originally water-positive chips into dry chips. Consequently, this control changes content and class prevalence as well as extent, and cannot isolate label quality or geography.”
Retain the manuscript's existing qualification of image, geometry and geography confounding. Do not infer N252 heldout independence from the N2500 screen: some retained runs/repro_full execution configs name a regenerated pool and mapped_diversity_N252_index.json. Their immutable link to the TableI generating checkpoints and the mapped-source/hash chain were not recovered. The available unmapped index and current first252prefix exclude the four intersecting source IDs, but do not prove the actual executed source membership. TableI remains a historical controlled comparison with this provenance limit; its exact historical sensitivity/statistics are not newly reproduced.

M9 — IV-C p9/TableII caption, revised results layout
Use separate panels: A original checkpoints rescored under the legacy evaluator, B current-environment original-data matchedcontrols, C corrected-protocol readouts. tables/Table_II_student_reconciliation.csv supplies panelA with each new mean and SD from the same four saved checkpoint readouts. current_full_summary.csv supplies panelB/C; main_table_current_display.csv is the compact proposed table. Include the fixed teacher/MNDWI without seedSD. Retain archived/printed values in the supplement reconciliation rather than mixing old means or SDs into a new row.
“Student entries are means and sampleSD over seeds42,64,91,107. They are separate from uncertainty over maps/events. The restored2500 input/cache pairs establish current source matching, not bitwise reproduction of historical training. Every current control follows the same early-stopping policy; actual selected and completed optimizer steps are listed in the supplement.”
N2500 selected steps range936–7800 and completed steps7176–7800; N100 selected168–300 and completed300. All44training runs completed their declared stopping policy. No seed was discarded.

M10 — IV-C pp9–10, student–teacher comparison; replace affected gap/statistics paragraph
“Using the verified teacher and matching current original-data counts, EfficientViT N2500 OFA minus teacher mIoU is-1.974 percentage points on S11, +0.635 on correctedSTURM and-0.042 on nativeWF. The corresponding water-IoU differences are-3.374,+0.535 and-0.077 points. Conditional event-macro mIoU differences are-1.877points on S11 (95%bootstrapCI[-2.893,-0.682]),+0.330 on STURM ([-0.715,+1.415]) and-1.560 on nativeWF ([-2.863,-0.466]). Pooled pixels and event means answer different questions. An interval including zero is not evidence of equivalence; four training seeds are not four geographically independent studies.”
Source: current_full_comparator_paired_seed/event.csv. These comparisons concern current original-data retraining and full domains; they replace neither historical checkpoint means nor the pending STURM exclusion result. The teacher has one fixed checkpoint, with no fabricated training-seedSD. Old46-map/29-event p-values and intervals must remain historical or be replaced with these newly named conditional event statistics; do not retain old significance claims beside the revised baseline.

M11 — V training-size discussion p12 and architecture comparison addition near IV-C/TableII
“OFA's pooled mIoU effects relative to clipping-matchedKD are modest and depend on budget, backbone and benchmark. At N2500, EfficientViT differences are-0.335,-0.023,+0.438percentage points on S11/correctedSTURM/nativeWF; MobileNetV3-Small gives-0.291,-0.052,+0.353 and LightFormer-ResNet18 +0.152,-0.183,-0.887. At EfficientViT N100 they are+0.695,+0.116,+0.697. The12paired seedSDs and separate event intervals are reported in the supplement. The N100 S11/nativeWF event evidence supports a limited positive effect; correctedSTURM's event interval spans zero. The separately identified regenerated cohort also shows null/unfavourable effects; its difference cannot be attributed solely to the127 replacement images because pseudo hard targets and cohort semantics also differ.”
“The additional backbones port the same six-band training and selection recipe; LightFormer's native auxiliary loss/output is disabled consistently for these cells. Active main parameter counts are3,546,706 forMobileNetV3-Small and11,427,509 forLightFormer; separate OFA exits add168,792 and176,008. These models and their pretraining differ in capacity and ancestry. The common recipe does not establish an optimum forLightFormer or device efficiency for either alternative backbone.”
Use OFA_matched_effects.pdf and the compact effect table. The original supplied MobileNet architecture study atN5000 remains a separate historical panel; do not combine it with the N2500 controlled ports.

M12 — IV-F robustness pp10–11; practical MNDWI comparison
“With threshold zero fixed, nativeWF EfficientViT N2500 OFA has91.48%water precision and68.33%recall versus77.82%/77.99% forMNDWI. Its higher precision trades missed water for fewer false positives on the scored flood-focused benchmark; this is not a fully dry-acquisition false-alarm estimate. Under the supplied synthetic haze h0.20, legacyWF OFA mIoU is74.76% against25.67% forMNDWI. After excluding the intersectingWFmap the corresponding values are74.74% and26.18%, so that diagnostic advantage remains. Conversely, retained spectral perturbations such as t1tilt-30 favourMNDWI:75.54% versus53.91% forOFA on that same exclusion domain. All available severities, comparators and degradation values are retained in the supplied tables, including unfavourable learned-model outcomes. Common positive scalar gain largely cancels inMNDWI's ratio; numerical/clipping exceptions are interpreted using the fixed diagnostic implementation. These perturbations do not establish reliability under independently observed atmospheric conditions.”
Native clean and legacy robustness are explicitly different geometry protocols; do not mix their precision/recall or interpret one as the same altered benchmark cell.

M13 — VI/TableIII p13 and TableIV p14; deployment evidence qualification
“TablesIII–IV preserve historically reported XavierNX results under a mixedINT8/FP16 route. TableIII's four-seed device-accuracy means require matching float/QAT/engine and evaluator identities for all four seeds; those complete receipts are not available in the retained package. TableIV reports one representative timing configuration, which does not require four independent timing seeds. Its values describe GPU-kernel timing under the stated trtexec200iterations/20warm-up setting, excluding I/O and preprocessing, rather than end-to-end throughput. FP16 is faster in this configuration; the mixed engine is smaller. Raw rail-labelled power samples and settings were not recovered, so the submitted watts remain historical and rail-unverified; a combinedCPU/GPU/CV rail must not be relabeledGPU-only, and peak watts are not energy per image.”
The existingpreparedS11device path selects validation; do not label it test. A strippedQAT float export intended for latency cannot substantiate preQATfloat accuracy. Remove unqualified “GPU power”/engine-accuracy/cost attribution wherever no matching chain exists; retain the numerical values only in a historical/reference panel with this caption. Alternative conservative author choice: omit untraceable numerical deployment claims from the revised main tables while retaining the implementation description. Missing logs do not establish that the reported values are false. No desktop timings replace Xavier measurements and no new QAT is requested.

M14 — VII Limitations p13; retain benchmark-guided history and strengthen dry/terminology wording
“Our labels establish surface water, not newly inundated land. An inundation-increment product requires reference-water or temporal information. The prefilter excludes teacher-dry imagery, and neither the flood-focused benchmarks nor agreement with teacher-empty baselines measures independent dry-scene false alarms. No independent review dispositions were supplied for the existing11dry candidates across3events, so dry-acquisition reliability and unattended event detection remain unmeasured. These limitations are acknowledged without asserting that collecting a new dry dataset is necessary to interpret the present results.”
Retain “Method choices were refined using the same public benchmarks reported in the paper.” until an author supplies a factual, stage-specific correction. Current code verifies the revision's frozen selection/readout separation; it does not establish historical untouched tests. Add the footprint sensitivity paragraph from supplementS1, explicitly labelled partial until all STURM receipts exist.

M15 — Proposed Discussion, inserted before existing VII Limitations; preserve existing Limitations
“The results address a practical annotation-budget question: once a task teacher exists, it can generate targets for additional imagery without acquiring new manual labels. This supports an application contribution combining established components, rather than a new theoretical distillation principle. The matched-budget comparison remains conditional on different training imagery, crop content, geography and historical provenance; the crop diagnostic demonstrates how much water content changes and prevents attributing the historical difference to label source alone.

The controlled backbone comparisons also narrow the role ofOFA. At N2500 it is not a consistent accuracy improvement, while EfficientViTN100 supports limited positive effects onS11/WorldFloods. SeedSD, pooled-pixel effects and conditional event intervals represent different sources and units of variation. Architecture capacity and pretraining remain unequal, and a common recipe is not an optimality or deployment claim for every backbone. The restored original and regenerated cohorts are reported separately, with cohort differences extending beyond replaced imagery.

The source-footprint assessment confirms different-date exposure to scored geography in threeSTURMchips and oneWFmap. The exclusion was fixed from metadata before comparative scores and applied equally to all comparators, including N100 whose selected sources did not contain those intersections. WF exclusion leaves17maps and11events; it preserves the small conditional clean effects and the learned-model haze advantage while retaining perturbations that favourMNDWI. The STURM chip-removal result awaits the bounded fixed-checkpoint receipts and is not inferred from WF. Excluding known exposure does not erase historical benchmark-guided choices or prove a newly untouched benchmark. N252 executed membership lacks its mapped-index chain, so the N2500 screen cannot validate TableI's geographic independence.

TheMNDWI comparison illustrates an operational trade-off rather than a universal replacement for a spectral rule: the learned model raises precision and lowers recall on nativeWF, while the rule remains favourable under several spectral perturbations. These are fixed-threshold and synthetic diagnostics. They neither validate real atmospheric robustness nor measure false alarms on fully dry acquisitions. Teacher supervision also transfers its surface-water convention; temporal/reference-water information is necessary for a product describing new inundation.

Finally, the deployment result is specific to the mixedINT8/FP16 Xavier configuration and its historical evidence. FP16 is faster in the representative kernel measurement. Device accuracy needs matching artifact ancestry, and power needs named-rail logs; desktop training accuracy cannot be assigned to the old engine. The supported conclusion is a bounded compact-model study with explicit provenance and deployment limits, not validated unattended hydrological or event-detection operation.”
Before incorporation, replace the third paragraph's STURM pending sentence with the actual sensitivity result only after sensitivity_complete/provenance.json is complete. This is a proposed Discussion, not an incorporated revision.

M16 — VIII Conclusion p13–14; replace numerical/generic conclusion claims
“The study demonstrates how an existing task-adapted teacher can extend supervision beyond a fixed manual annotation set and train compact surface-water students. Controlled comparisons show thatOFA's contribution depends on budget, architecture and evaluation domain. Historical matched-budget claims remain bounded by imagery/geometry/geography and provenance differences; verified current readouts and exclusion sensitivities are reported separately. A target-specific mixedINT8/FP16 deployment is described with its historical measurement qualifications. Independent dry reliability, newly inundated-land detection and general operational robustness are not established.”
Terminology pass: use “surface-water segmentation” for label-supported outcomes throughout Abstract/Methods/Results/Conclusion. Use “human hard labels”, “teacher-derived pseudo hard targets”, and “live teacher soft targets” consistently; do not call teacher-empty orMNDWI-empty imagery independently dry.
"""

SUPP=r"""
PROPOSED SUPPLEMENT EDITS — NOT INCORPORATED
S1 — S1 data-overlap audit pp1–2; add after the centroid-screen results
“The original centroid/radius screens and the new actual-footprint check have different support definitions. The new check verifies actual264×264 selected image patches, densifies boundaries before CRS transformation, and intersects retained reference pixels under the fixed evaluation masks. It confirms scored-support intersection with STURMchips EMSR465_AOI01_07_11_1_1.tif, EMSR501_AOI01_10_05_2_1.tif and EMSR501_AOI01_10_05_2_2.tif, and with retained scored windows of WFmap EMSR438_AOI02_DEL_MONIT03.tif. The sources are different-date acquisitions: approximately935days forEMSR465,11.992days forEMSR501 and155.012/964.988days for the twoWFsource pairs. No duplicate acquisition is established. The source patches have potential full-array training support under resize/flips/rotations/channel gain, rather than an inferred whole Sentinel-2 scene or invented single stochastic crop. No evidence links external ground-truth labels to pseudo-label supervision; imagery/teacher-target exposure and possible benchmark-guided development remain distinct issues.”
“From metadata alone we froze complete-unit exclusions common to the fixed teacher, zero-thresholdMNDWI and36student checkpoints. Removing the completeWFmap changes18maps to17 while retaining11events, and removes6,336,397 valid and223,618water pixels under native clean masks. The3STURMchips contain786,432 scored valid pixels and80,512water pixels; their removal will leave2672chips with the remaining map/event counts recomputed, rather than discarding whole unaffected flood maps. The necessary per-chip counters are being recovered with fixed checkpoints because saved counters aggregate flood maps. No training or corruption sweep is added. Full benchmarks remain separate from this conservative sensitivity, and neither exclusion nor the current N100/currentFPS252-prefix absence proves complete historical independence.”
“Some retained runs/repro_full N252 execution configs refer to a regenerated pool and mapped_diversity_N252_index.json; their immutable link to the TableI generating checkpoints is not established. The available unmapped252index excludes the intersecting current IDs, but the executed mapping/hash chain is unavailable. Consequently TableI's actual membership and count-domain sensitivity have not been reestablished; those historical statistics are retained with that qualification.”
Source tables: overlap_scored_support_assessment.json; overlap_exclusion_plan.json; sensitivity_wf_complete/*; pending sensitivity_complete/*. Do not label pendingSTURM as finished. Prior centroid0/54/threshold findings remain historical screen results and are not overwritten by this patch.

S2 — S2 subset selection pp1–2, descriptor/provenance wording
Replace mean/std of teacher-water probability with mean/std of stored maximum-class confidence, matchingM4. Preserve supplied statements that clean selector spread is0.012mIoU and no stableFPSadvantage is resolved atN2500. “FPS and filter values were fixed for the reported comparisons; retained records do not establish their original chronological predeclaration. GLISTER's surrogate uses S11validation. The recorded launcher describes external-score-based selection rationale, which is disclosed separately from current frozen readout.” Do not invent an untouched-development history.

S3 — S3 spatial harmonization/evaluation scale pp2–3
AddM7's corrected STURM/nativeWF definitions and M8's crop counts. Caption legacy/native protocols explicitly; retain existing native-resolution and radiometric diagnostics as diagnostics, not replacement publishedbenchmark baselines. Replace any sharedshift-insensitivity statement with protocol consistency and possible differential model response.

S4 — S4 spectral reference p3
Add native-clean waterprecision/recall/F1/IoU from current_full_summary.csv and retained-WF values from sensitivity_wf_complete/summary.csv. Each row shares checkpoint/mask/protocol/count source. Keep fixedMNDWIthreshold0 and the existing epsilon/clipping diagnostics. Include every already declared perturbation cell, drops and null/unavailable flags; no threshold tuning.

S5 — S5 teacher semantics pp4–5
Retain the supplied surface-water convention and the distinction between proxies and independent annotations. Replace “false alarm” based on teacher-empty pre-event baselines with “positive prediction relative to a teacher-empty baseline”; add that this is teacher agreement, not independently verified absence of water. Independent11candidate dispositions remain absent; no dry-acquisition reliability score is claimed.

S6 — S6 OFA gradient-path analysis pp5–6
Retain measured diagnostics as conditional observations; useM5's implemented branch probabilities/reduction/weights and11,368exitparameters. Do not derive a general mechanism theorem or infer that deep supervision necessarily improves accuracy. Include OFA_matched_effects.csv/pdf and distinguish the12pairedseedSDs from eventbootstrapCIs.

S7 — S7 deployment implementation/export pp6–7
Retain existing mixedINT8/FP16, FP16speed and BN qualifications; insertM6 and M13. Annotate originalTableIIIdevice accuracy as historical/untraced for the complete4seedchain andTableIVas representative historicalkernel timing. Require named powerrail, settings and buildidentity for quantitativepowerclaims. Do not interpret QAT adaptation as training only calibration scales or use strippedQATfloat latency exports as validated preQATaccuracy. Existing code's S11device path usesvalidation.

S8 — S8 additional diagnostics/ablations pp7–11
Insert ablation_map.csv, data_roles.csv, main_table_current_display.csv and OFA_matched_effects.pdf. Architecture addition: original-data fourseedmatched N2500MobileNet/LF andEffN100 comparisons with their actual adapted sizes and separate training-onlyheads. Preserve historical suppliedN5000MobileNetstudy as a separate panel. Current recovered-recipe and matchedcontrols are distinct from original copiedcheckpointrescoring; original versusregeneratedcohorts remain separated.

Add a numerical-source reconciliation panel adjacent to originalTableII source discussion
“All revised means and sampleSD are generated from the same named fourseedsource with nearest0.001rounding from full precision, independently for each statistic. EfficientViTCE/WF raw0.7534745 rounds0.753, while the assembled0.7535 can double-round0.754 and the new0.753609 crosses to0.754; KD/WF raw0.766608 and rescore0.766415 cross0.767→0.766 with only-0.019225points drift. UNetKD/STURM raw0.746492 and new0.746483 both round0.746, rather than printed0.747; its assembledCSV0.7469 does not match the archivedseedmean. UNetKD/WF assembled0.765±0.0066 is also not the archived0.762321±0.002476 or new0.762313±0.002493; revised newdisplay is0.762±0.002 under the stated rule. The15row reconciliation lists every mean andSD. The teacher row is reconciled separately because student near-reproduction tolerance does not cover it.”
Source: Table_II_student_reconciliation.csv, teacher_baseline_reconciliation.csv and teacher_trace.json. The old external teacher count caches support the printed rounded STURM/WF values, but lack generating checkpoint/runtime ancestry; the old evaluator references an absent teacher_ft checkpoint. The S11 printed teacher source remains untraced. Preserve printedreferencevalues with explicit verification status; no evaluator tuning to recover display digits.
"""

COMMENTS=[
('R1.1','Explain the contribution beyond establishedKD, pseudo-labels, selection and quantization','I contributions p2; III pp3–6','Application/annotation-budget transfer and measured bounded deployment; component attribution already supplied','No universalfirst/theoreticalnovelty claim supported','Incorporate M2/M15/M16','M2/M15/M16','evidence-ready'),
('R1.2','Present/discuss recent compact architecture comparison andOFA on alternative backbones','IV-C/TableII p9; supplementS8 p10','All4seeds original N2500MBV3/LF KD/OFA and Effcontrols; adaptedsize/auxpolicy verified','Unequalcapacity/pretraining; commonrecipe not globallyoptimal; no alternativeedge measurements','IncorporateM11 and newtable/figure','M11; S8','evidence-ready'),
('R1.3','State data roles, selection/calibration/development choices and possibleinformationleakage','III-A/B pp3–4; IV-A p6; VII p13; supplementS1','Restoredsource/cachematching; actual3STchips/1WFmap scored intersections; WF count sensitivity complete','STchipcounts queued; benchmarkhistory/actualN252mappedchain unresolved; no untouchedclaim','Finish38chipreceipts/dependentcollector; authorhistory; incorporate roles/overlapparagraph','M4/M7/M8/M14/M15; S1/S2','requires-specific-analysis'),
('R1.4','Make filtering,FPS,KD/OFA,activationreplacement andQAT ablations understandable together','III-B/D pp4–6; VI/TableIII p13; supplementS2/S7/S8','Ablationmap identifies fixed and changed terms; filteringstudy/FPSnull; matchedcontrols complete','Sequential/confoundeddeltas not additive; deviceQATchain incomplete','Incorporate ablationmap and implementedlosscorrections; qualify devicecontrast','M5/M6/M9/M13; S2/S6/S7/S8','evidence-ready'),
('R1.5','Make surface-water/flood andteacher/pseudo-label terminology consistent','Abstractp1; IIIpp3–5; IV-Gp12; VII/VIIIp13','Suppliedpaper alreadylimits semantics; human/pseudohard/live softroles recovered','No independentnew-inundation truth or temporalwaterproduct validated','Apply terminology pass and preserve originalqualification','M1/M5/M14/M16; S5','writing-needed'),
('R1.6','Acknowledge removal ofdryscenes andlimits onunattendedfalsealarms','III-Bp4; VIILimitationsp13; VIIIConclusion','Existingpaperacknowledgement;11candidates/3events prepared, noindependentdispositions','Teacher-empty/MNDWI-empty is notdrytruth; no dryacquisition score','IncorporateM14/M16; optional reviewonly if laterprovided; no datasetrequired','M14/M16; S5','explicitly-limited-in-manuscript'),
('R2.1','Explain contribution beyond establishedcomponents','I/III/VII','Same boundedapplicationcontribution asR1.1; reusable establishedmethods attributed','No needfor new theory/first-in-literatureassertion','IncorporateM2/M15','M2/M15/M16','evidence-ready'),
('R2.2','Bound S11/GDACScomparison byimage,geometry andgeographyconfounders','IV-Cp8/TableIp9; supplementS3','Existinggeometryqualification; correctedcrop200/252,73.6263%waterremoved,21newdry','Cropnotcleanlabel-qualityisolation; mappedhistoricalN252membership incomplete','Correct count andqualifyhistoricalindependence withoutTableIrerun','M8/M15; S1/S3','explicitly-limited-in-manuscript'),
('R2.3','Explain modest andbudget-dependentOFAgains','V p12; III-Cp5','12OFA-minusKD effects withpairedseedSD/eventCI; limitedN100positiveS11/WF; regeneratedseparate','Not universalwin; eventSTN100CIspans0; no causal127imageattribution','IncorporateM11/M15 andmatched-effectsfigure','M11/M15; S6/S8','evidence-ready'),
('R2.4','Present/discuss compactarchitecturecomparison andOFAalternativebackbones','IV-C/TableIIp9; supplementS8','Same controlledMBV3/LFports asR1.2; all4seeds complete','Commonrecipe andsize/pretrainingdifferences remain','Incorporate table/effectplot/adaptationlimits','M11; S8','evidence-ready'),
('R2.5','Explain whenlearnedmodelshelp relativetoMNDWI','IV-Fpp10–11; supplementS4','Matched count-derivedprecision/recall; all existingconditions; WFretainedhazedifference remains','Syntheticdiagnostics/precisionrecalltradeoff; dryreliabilitynotmeasured; STretainedrobustnesscounts absent','IncorporateM12; retain favourable/unfavourableconditions andprotocol distinctions','M12/M15; S4','evidence-ready'),
('R2 final','Qualify hardware-specific andmixedINT8/FP16deploymentresults','Abstractp1; III-Dpp5–6; VI/TableIIIp13/TableIVp14; supplementS7','Bodyalreadyacknowledgesmixedprecision/FP16faster; unknown-seedengine/buildrecordexists','Complete4seeddeviceaccuracyancestry/rawtimingpowerchain unavailable; noauthorizedtarget supplied','Useconservative historicalqualification; authorretain/omitchoice; specificdeviceinputs forunqualifiednumbers','M1/M6/M13/M16; S7','device-input-needed'),
('Editor','Provide Discussion andpoint-by-point response identifyingchangesandlocations','ProposedDiscussionbeforeVIIp13; responsefile','All13responseentries andsection-anchoredpassages prepared','Actualletter absent; draft changes notincorporated; STsensitivitypartial','Incorporateproposedpassages, reconcileactualletterwording, refreshSTnumericalresultwhencomplete','M15; point_by_point_response_draft.txt','writing-needed'),
]

def main():
    csvwrite('data_roles.csv',[dict(zip(('role','data','verified_use','qualification','source_anchor'),r)) for r in ROLES])
    csvwrite('ablation_map.csv',[dict(zip(('component','existing_comparison','held_fixed','also_changes','supported_result','source','limitation'),r)) for r in ABLATIONS])
    sensitivity_done=(OUT/'sensitivity_complete/provenance.json').exists() and read('sensitivity_complete/provenance.json')['status']=='complete'
    closure=[]
    for r in COMMENTS:
        row=dict(zip(('reviewer_ID','request','manuscript_anchor','existing_evidence','scientific_limitation','remaining_action','draft_anchor','status'),r))
        for key in ('request','manuscript_anchor','existing_evidence','scientific_limitation','remaining_action','draft_anchor'):
            row[key]=readable(row[key])
        row.update(request_source='prompts/prompt5.md section3; supplied request summary, not a verified verbatim reviewer quotation',
            exact_request_verification='Actual editor/reviewer letter not located; awaiting supplied letter',
            final_draft_location='main_manuscript_proposed_edits.txt / supplement_proposed_edits.txt / point_by_point_response_draft.txt',
            draft_state='prepared; proposed, not incorporated',
            secondary_inputs='Author history forR1.3; deployment forR2final; actualletter wording for all entries; dryreview optional')
        if sensitivity_done and row['reviewer_ID']=='R1.3':
            row.update(status='author-input-needed',existing_evidence='Scored-support assessment and both exclusion sensitivities complete:38fixedchipreceipts and dependentcollector validated; role/source tables complete.',
                scientific_limitation='Benchmark-guided development history and actual historical N252 membership remain unresolved; known-overlap exclusion does not establish a new untouched test.',
                remaining_action='Incorporate proposed roles/overlap text and obtain author history; no remaining inference.')
        closure.append(row)
    csvwrite('reviewer_closure_table.csv',closure)
    main_src=MAIN;supp_src=SUPP
    if sensitivity_done:
        main_src=main_src.replace('The STURM exclusion sensitivity is pending; this passage intentionally claims only full-domain scores.',
            'These Abstract values identify full-domain scores. The completed common-exclusion sensitivity is reported separately in Supplement S1.')
        main_src=main_src.replace('the pending STURM exclusion result','the separately reported STURM exclusion result')
        main_src=main_src.replace('explicitly labelled partial until all STURM receipts exist','including the completed STURM exclusion result')
        main_src=main_src.replace('The STURM chip-removal result awaits the bounded fixed-checkpoint receipts and is not inferred from WF.',
            'STURM exclusion leaves2,672chips across46maps/29events. EfficientViT N2500OFA/teacher mIoU is76.031/75.395%, versus76.026/75.392% on the full corrected domain. OFA-minusKD remains-0.021points; N100 gives+0.117points, while its event interval still spans zero. These results reuse unaffected map counts and the38validated fixed-chip receipts.')
        main_src=main_src.replace("Before incorporation, replace the third paragraph's STURM pending sentence with the actual sensitivity result only after sensitivity_complete/provenance.json is complete. This is a proposed Discussion, not an incorporated revision.",
            'The sensitivity paragraph uses completed sensitivity_complete/provenance.json and its source tables. This is a proposed Discussion, not an incorporated revision.')
        supp_src=supp_src.replace('their removal will leave2672chips with the remaining map/event counts recomputed','their removal leaves2672chips, with46maps and29events retained')
        supp_src=supp_src.replace('The necessary per-chip counters are being recovered with fixed checkpoints because saved counters aggregate flood maps.',
            'All38fixed-checkpoint jobs and the dependent collector completed successfully. Each inferred16chips preserving original batch8contexts and scored exactly the3excluded chips; saved counters for all other chips were reused.')
        supp_src=supp_src.replace('pending sensitivity_complete/*. Do not label pendingSTURM as finished.',
            'completed sensitivity_complete/*. STURM excluded-domain robustness remains unavailable because the saved perturbation counters do not resolve these chips; no new corruption inference was run.')
        supp_src+='\nCompleted common-exclusion clean sensitivity: EfficientViT N2500OFA STURMmIoU76.026→76.031%, teacher75.392→75.395%, MNDWI74.670→74.694%. OFA-minusKD full/retained STURM effects are-0.023/-0.021points forEffN2500,-0.052/-0.051 forMBV3,-0.183/-0.181 forLF and+0.116/+0.117 forEffN100. RetainedN100 STURMeventmacro is+0.005points with95%CI[-0.143,+0.151]; no equivalence or consistent gain is inferred. Full/heldout metric, count, paired-seed and event tables are in sensitivity_complete.\n'
    text('main_manuscript_proposed_edits.txt',main_src);text('supplement_proposed_edits.txt',supp_src)
    text('remaining_inputs.txt',r"""
Remaining inputs and conservative closure path
1. Actual editor/reviewer letter: not found among identified sources. One location question was sent; no reply is assumed. All13comment IDs/scopes follow prompt5, with paraphrases explicitly distinguished from exact quotations. Supply the letter to verify verbatim requests; no new obligations are inferred from previous audit prompts.
2. Author benchmark history (already asked; not reasked): DidSTURM/WF scores influence architecture/filtering/loss/checkpoint selection/calibration or other originalmethodchoices, and whichbenchmarks doesVII's exactsentence refer to: “Method choices were refined using the same public benchmarks reported in the paper.” Retained launchercomments also cite external scores in FPSrationale. Conservative draft retains the disclosure; currentcode cannot establish historicalnon-use or contamination. An executedN252mappedindex/hash/source mapping would narrowTableI's provenance limit but is not a trainingprerequisite.
3. Optional independent dryreview: existing11candidates/3events have only an uncompletedtemplate and noverifiedreferencegate. Dispositions need revieweridentity/date/actualsource/whole-tileassessability with genuine clear-reference support; teacher/MNDWI predictions cannot certifydry. Manuscript limitation addressesR1.6 without newdata. The olddryqueue is not blindlyresumed: independentreferenceimport must be followed by the actual current-footprint eligibility gate and scoped original/fixed comparator scheduling; unknownreferencepixels stayignored, emptywaterIoU/recall stayundefined, and fixedalarmfractions are0,.001,.01. No drywork was added now.
4. Deployment, only for retainedunqualified numericalclaims: matchingoriginalfloat/QATcheckpoints, seed42/64/91/107 ONNX/engineidentitychain and on-deviceaccuracy masks/evaluator/split logs forTableIII; existing authorizedXavier workflow/target if measurementsare needed. TableIVneeds the representative engine/build/settings and original200iteration/20warm-up timingrecord, not4timingseeds. Neednamedpowerrail samples and power/clock/hardwarestate; noenergy/image inferredfrompeakwatts. The currenthost isx86_64, noXavier/trtexec, and noauthorizedtargethasbeensupplied. Latency-only strippedQATfloat export cannot validate preQATaccuracy. Conservative captions or omission resolve unsupportedprecision; absenceoflogs doesnot show the oldnumbersfalse.
5. Computationalremainingwork: exactly38frozen-checkpoint STURMchipcount jobs plus dependentcountcollector; no authorchoice is needed torunthese authorizedjobs onceGPUadmissionpasses. FullV4training/evaluations arepreserved and notrerun. WFclean/existingrobustness exclusion complete. No generated manuscript passage claims STexclusionsuccess beforeits receipt.
""")
    if sensitivity_done:
        path=OUT/'remaining_inputs.txt';lines=path.read_text().splitlines()
        lines=[line for line in lines if not line.startswith('5. ')]
        lines+=['5. Computational status: all38fixed-checkpoint STURM jobs and the dependent collector succeeded. The detached worker finished; no runnable prompt5GPU work remains. Both corrected clean sensitivities and all compatible savedWFrobustness exclusion summaries are complete. No new training, QAT or corruption sweep occurred. HistoricalV4blocked/failed records remain preserved.']
        text('remaining_inputs.txt','\n'.join(lines))
    response=['POINT-BY-POINT RESPONSE — REVIEWABLE DRAFT, NOT SUBMITTED OR INCORPORATED',
              'Request wording below is the prompt5 supplied summary. Exact quotations and numbering must be checked against the missing actual letter. Responses describe proposed revisions; change “we propose” to “we have revised” only after incorporation and verification. The numerical STURM exclusion remains pending.', '']
    answers={
    'R1.1':'We propose to frame the contribution as annotation-budget amplification with an existing task teacher, controlled compact-model transfer, and a qualified deployment study. We will attribute KD, pseudo-labeling, selection, OFA and quantization to established components already cited. We make no universal or first-in-literature claim. Proposed changes: M2, DiscussionM15, ConclusionM16.',
    'R1.2':'The original-data comparison now contains all four seeds of MobileNetV3-Small and LightFormer-ResNet18 KD/OFA under the common six-band recipe. OFA need not improve accuracy: the full corrected-domain effects are small and inconsistent. We propose the sourced current table and12row effect figure, with actual main/exitsizes, pretraining differences and auxiliary-loss policy disclosed. No globallyoptimal LightFormer or edge-performance claim is made. Proposed changes: M11 andS8.',
    'R1.3':'We propose a single data-role table specifying human traininglabels, pseudo hardtargets, live teacher softtargets, training-based normalization, prefilter/FPS, S11validation selection, QATcalibration and external readout. Actual selectedpatch footprints intersect scoredregions in3STURMchips and1WFmap on different dates; no duplicate acquisition or externalgroundtruth use is established. A common metadata-fixed exclusion is used. WF native sensitivity is complete (18→17maps,11events unchanged); the STchip-count inference and collector are queued, with no ranking conclusion yet. We retain the original benchmark-guided-development disclosure pending the alreadyasked authorhistory answer, and explicitlyqualify historicalN252membership because its executedmappedindex chain is missing. Proposed changes: M4/M7/M8/M14/M15 andS1/S2. Status: specificanalysispending plusauthorinput.',
    'R1.4':'We propose an ablationmap identifying what each existingcomparison holdsfixed and changes. It preserves the supplement’s unresolvedFPSgain, distinguishes trueCE from focal/KD, historical clipping from matchedcontrols, and recognizes changedhardtermweight. OFA exits andimplemented branch temperatures/reductions are corrected. Activation/QAT are sequentialhistoricaldevice steps with incompleteancestry, not anadditive decomposition. Proposed changes: M5/M6/M9/M13; ablation_map.csv; S2/S6/S7/S8.',
    'R1.5':'The submittedpaper alreadyqualifies surface-watersemantics. We propose a consistency pass using surface-watersegmentation forlabel-supportedoutcomes and distinguishing humanhardlabels, teacher-derived pseudohardtargets andlivesofttargets. Newlyinundatedland needs temporal/referencewater information. Proposedchanges:M1/M5/M14/M16 andS5.',
    'R1.6':'We propose clearer Methods/Discussion/Conclusionacknowledgement that thefilter removes teacher-dryscenes and floodfocusedbenchmarksdo not measure independentdryfalsealarms. The11candidate package has no genuine independentreview dispositions, so no dryreliabilityvalueis invented. A newdrydataset is not needed to make thislimitationexplicit. Teacher-emptybaseline agreement remains a limiteddiagnostic. Proposedchanges:M14/M15/M16 andS5. Clarification is ready; addressedonceincorporated.',
    'R2.1':'We propose the same bounded contribution framing asR1.1: established components supporting the application/annotation-budgetquestion, with controlled outcomes and limits. No newtheoreticalnovelty orunsupported firstclaim isadded. Proposedchanges:M2/M15/M16.',
    'R2.2':'We retain the existing geometryqualification and correct its demonstratedcropnumbers:200 of252chips lose>half theirnativewater,73.6263%pooledwaterisremoved and21positivechipsbecome dry. Differentimagery andgeography remainconfounders. The actual TableI N252 membership and checkpoint/source mapping identities are not recovered; its independence andsensitivitycannotbeinferredfromN2500/currentprefix. No TableItrainingis scheduled. Proposedchanges:M8/M15 andS1/S3.',
    'R2.3':'The12full-domain matchedOFAeffects are reportedwith all4seed differences and separateconditional eventintervals. EffN100 gives+0.695/+0.116/+0.697points onS11/correctedST/nativeWF, butSTeventCI[-0.144,+0.151]points spanszero. N2500effects varybybackbone andbenchmark. The separate regeneratedcohortdoesnot supporta universalgain andisnot collapsedwithoriginal. We donotinfer equivalence or attributeallcohortdifferences to127images. Proposedchanges:M11/M15 andS6/S8; OFA_matched_effects.pdf/csv.',
    'R2.4':'We propose the completed common-recipe MobileNet/LightFormerOFA controls and their actualadaptation details as the architecturecomparison. Unequalcapacity/pretraining andlackofalternativebackbone deviceprofiling areexplicit. ExistingN5000historicalMobileNetresults remainseparate. Proposedchanges:M11 andS8.',
    'R2.5':'Count-matched metrics show nativeWF OFAprecision/recall91.48/68.33% versusMNDWI77.82/77.99%, clarifyingthe falsepositive/missedwater trade-off. UnderexistinglegacyWFhaze h0.20, learned mIoU74.76% versus25.67% remains favourableafterWFexclusion (74.74/26.18%). Underretainedt1tilt-30 MNDWIwins75.54/53.91%. Allavailable perturbationcellsand drops are retained, threshold0fixed, withnativeclean distinguishedfromlegacy robustness. Theseresultsare syntheticdiagnostics, notdry-acquisition orrealatmosphericvalidation. Proposedchanges:M12/M15 andS4.',
    'R2 final':'We preserve the manuscriptbody’s existingmixedINT8/FP16 andFP16faster acknowledgements and propose consistentAbstract/headings/captions. TableIIIfourseeddeviceaccuracy needscorresponding seedartifact/evaluatorreceipts; TableIVrepresentativetimingdoesnot requirefourtimingseeds. Missingidentity/powerlogs arequalifiedwithoutcallingoldnumbersfalse. We retainuntracedvalues onlyashistorical, kernel-only representative measurements andrail-unverifiedwatts, orauthorsmayomit thosevalues. No desktopaccuracyis assignedto the oldengine, no RTXsubstitution/newQAT occurs, andS11devicevalidation isnotcalledtest. Proposedchanges:M1/M6/M13/M16 andS7. Specificdeviceinputlist attached.',
    'Editor':'A focused Discussion is proposed before existingVII Limitations, connecting annotationbudget, architecture/OFA, footprint/history, MNDWItrade-offs, dry/semanticlimits and deploymentevidence. The full13entry closuretable and this point-by-point draft identifyactualsection/table anchors and replacementpassages. OriginalPDFs arepreserved; neither editablepatch incorporation nor exactletterverification isclaimed. NumericalSTsensitivitywillbeinsertedonlyafterall38countreceiptsandcollectorpass. Proposedchanges:M15 and the completepackage.',
    }
    for row in closure:
        answer=answers[row['reviewer_ID']]
        if sensitivity_done:
            answer=answer.replace('the STchip-count inference and collector are queued, with no ranking conclusion yet.',
                'all38STchip-count jobs and the dependent collector succeeded. Excluding the3chips retains46maps/29events and gives EfficientViTOFA/teacher mIoU76.031/75.395%. Its OFA-minusKD is-0.021points; the N100eventinterval still spanszero.')
            answer=answer.replace('Status: specificanalysispending plusauthorinput.','Status: numerical analysis complete; author history remains unresolved.')
            answer=answer.replace('NumericalSTsensitivitywillbeinsertedonlyafterall38countreceiptsandcollectorpass.',
                'NumericalSTsensitivity is now supported by all38countreceipts and the successfulcollector; its source tables are attached.')
        response.extend([row['reviewer_ID']+' — '+row['request'],answer,
                         'Evidence: '+row['existing_evidence'],
                         'Draft state: proposed, not incorporated. Current concern status: '+row['status']+'.',''])
    response_text='\n'.join(response)
    if sensitivity_done:response_text=response_text.replace('The numerical STURM exclusion remains pending.','Both numerical overlap sensitivities are complete; source receipts and count tables are attached.')
    text('point_by_point_response_draft.txt',response_text)
    summary=rows('tables/current_full_summary.csv');display=[]
    groups=sorted({(r['architecture'],r['variant'],r['budget']) for r in summary})
    for a,v,b in groups:
        row={'architecture':a,'objective_variant':v,'budget':b,'cohort':'original_restored_current_environment' if b!='0' else 'fixed_comparator',
             'panel':'current matched/compatible OFA; full benchmarks; float checkpoints, not deployed engines'}
        for ds,proto in [('s11','S11_TEST512_LEGACY'),('sturm','P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1'),('wf','P3_WF_NATIVE_ZERO_PAD512_V1')]:
            r=next(r for r in summary if (r['architecture'],r['variant'],r['budget'],r['dataset'],r['protocol'],r['metric'])==(a,v,b,ds,proto,'mIoU'))
            row[ds+'_mIoU']=r['display_mean']+(' +/- '+r['display_SD'] if r['n_seeds']=='4' else ' (fixed)')
        row['source']='tables/current_full_summary.csv';display.append(row)
    csvwrite('tables/main_table_current_display.csv',display)
    current=datetime.now(timezone.utc).isoformat();state=read('queue/manifest.state.json')
    receipt={'captured_utc':current,'drafts':'complete_reviewable_proposals_not_incorporated',
        'letter':'actual wording/quotes unverified; scope mapped from prompt5',
        'full_current_tables':'complete','WF_sensitivity':'complete','STURM_sensitivity':'complete' if sensitivity_done else 'pending38fixedchipjobs_pluscollector',
        'queue_counts':{s:sum(r['status']==s for r in state['jobs'].values()) for s in ('pending','launching','running','succeeded','failed','blocked')},
        'worker_status':state['worker_status'],'last_worker_heartbeat':state.get('heartbeat_at'),
        'author_history':'pending, conservative disclosure retained','dry':'unmeasured; clarification drafted; independentreviewoptional',
        'device':'specificchain absent; conservativehistoricalqualification drafted','new_training':0,
        'scope_completion':('Authorized computation and reviewable draft package complete; unresolved inputs have conservative proposed wording. No manuscript incorporation/submission claimed.' if sensitivity_done else
                            'Draft/evidence work complete; bounded STURM sensitivity queued. Reviewer responses not incorporated or submitted.')}
    (OUT/'closure_status.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
