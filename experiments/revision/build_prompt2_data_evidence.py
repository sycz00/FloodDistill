"""Build the scoped claim-to-evidence data/evaluator ledger from read-only audits."""
from pathlib import Path
import collections, datetime, hashlib, json, subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/revision'
REV=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
base=json.load(open(OUT/'protocol_audit.json'))
checks=json.load(open(OUT/'prompt2_data_checks.json'))
geo=json.load(open(OUT/'prompt2_geographic_screen.json'))
refs=json.load(open(OUT/'prompt2_reference_manifest.json'))
loader=json.load(open(OUT/'prompt2_loader_membership.json'))
windows=json.load(open(OUT/'prompt2_worldfloods_windows.json'))
texts={k:(OUT/'prompt2_sources'/f'{k}_v3.txt').read_text() for k in ['main','supplement']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def page(doc,term):
 text=texts[doc];i=text.lower().find(term.lower());return text[:i].count('\f')+1 if i>=0 else None
def paper(doc,location,claim,term):
 original=next(x for x in base['manuscript_evidence'] if Path(x['path']).stem==doc)
 return {'document':original['path'],'sha256':original['sha256'],'page':page(doc,term),'location':location,'claim':claim}
def code(path,lines):
 p=ROOT/path if not path.startswith('/') else Path(path)
 return {'path':str(p),'lines':lines,'git_revision':REV if str(p).startswith(str(ROOT)) else None,'sha256':sha(p),'executed_history':'Current/source inspection; older checkpoint execution verified only where saved fingerprint/config matches'}
def art(name):
 p=OUT/name;return {'path':str(p),'sha256':sha(p)}
records=[]
def rec(i,reviewers,pap,cod,finding,classification,severity,affected,remedy,status='verified-complete',ev=None):
 records.append({'id':i,'reviewer_ids':reviewers,'paper':pap,'code_references':[code(p,l) for p,l in cod],
  'data_run_checkpoint_evidence':ev or [],'finding':finding,'classification':classification,'severity':severity,
  'affected_results':affected,'smallest_remedy':remedy,'status':status})
rec('P2-A01',['R1.3','R2.2'],paper('main','IV-A Datasets','Official S11 split252/89/90;15 Bolivia separately reserved','official split of 252'),
 [('data/sen1floods11.py',[19,124]),('/home/schmalstieg/miniconda3/envs/esi/lib/python3.10/site-packages/terratorch/datasets/sen1floods11.py',[45,111])],
 {'membership_counts':{n:checks['membership'][n]['count'] for n in ['train','valid','test','bolivia']},'ID_intersections':checks['membership']['intersections'],
  'no_duplicate_IDs':all(not checks['membership'][n]['duplicate_ids'] for n in ['train','valid','test','bolivia']),
  'unassigned_images':checks['membership']['unassigned_image_ids'],'missing_images':checks['membership']['missing_image_ids'],
  'current_loader_exact_membership_and_pairing':{k:{a:b for a,b in v.items() if a not in ['actual_image_ids','actual_label_ids']} for k,v in loader.items()},
  'historical_limit':'Official manifests verified now; this does not recover every older executed loader revision or calibration batch subset.'},
 'verified','low',['S11 controls/validation/test','current canonical evaluator'],'Retain explicit manifest and reference identities; no split rewrite.',ev=[art('prompt2_data_checks.json'),art('prompt2_loader_membership.json'),art('prompt2_reference_manifest.json')])
rec('P2-A02',['R1.3','R2.2'],paper('supplement','S1 Data-overlap and contamination audit','Centroid screening bounds possible training/evaluation overlap','centroids rather than footprint geometry'),[],
 {'S11_cross_split_footprint_pairs':{k:v['n_pairs'] for k,v in checks['footprint_checks'].items() if k.startswith('s11_') and '__s11_' in k},
  'S11_vs_external_footprint_pairs':{k:v['n_pairs'] for k,v in checks['footprint_checks'].items() if k.startswith('s11_') and k.endswith(('sturm','wf'))},
  'shared_location_groups':'Train/validation/test each contain Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA; Bolivia separate.',
  'acquisition_time_limit':'S11_Metadata.geojson is absent; S11 dates were not invented.',
  'result':'No positive-area overlap found by these native footprint checks. Shared locations/events remain dependence; no exhaustive imagery-content comparison.'},
 'verified','medium',['S11 training-seed uncertainty','geographic generalization interpretation'],'State chip split/footprint coverage and shared events; retain existing split.',ev=[art('prompt2_data_checks.json')])
rec('P2-A03',['R1.3','R2.2'],paper('main','III-B / IV-A GDACS pool','N2500 drawn from original filtered pool23720/5987','5,987 candidates'),[],
 {'current_cohort':'regen_N2500_approved_20261005','dataset_sha256':checks['gdacs']['dataset_sha256'],'original_source_overlap':2373,'replacements':127,
  'original_byte_reproduction':False,'frozen_receipt':checks['gdacs']['frozen_receipt'],
  'authorization_evidence':'Existing operational active_queue.json and dataset receipt record prior acceptance of regenerated cohort; the initiating conversation is not independently reconstructed here.'},
 'verified','high',['New local/current-environment architecture comparisons','training reproduction claim'],'Keep regenerated approved cohort distinct from original archived results and label its127 replacement sources.',ev=[checks['gdacs']['frozen_receipt'],art('protocol_audit.json')])
rec('P2-A04',['R1.3','R2.2'],paper('supplement','S1','54/2675 STURM chips within25km/30days;none within5km/2days','54 of the 2,675'),[],
 {'cohorts':{k:{'coverage_centroids':v['n_sources_with_centroid'],'total':v['n_sources'],'n_events':v['n_events'],'n_acquisitions':v['n_acquisitions'],
                   'shared_acquisition_groups':len(v['repeated_acquisitions']),'same_location_multiacquisition_groups':len(v['paired_location_acquisitions']),
                   'screens':{a:{b:c for b,c in q.items() if b!='pairs'} for a,q in v['screens'].items()}}
             for k,v in geo['cohorts'].items()},'native_GDACS_footprint_coverage':'0/2500: raw raster root missing; .pt inputs lack geotransform.',
  'result':'Original centroid finding54 is reproduced; regenerated current cohort has35. Tight-screen0 does not establish independence or no spatial overlap.',
  'coordinate_status':'inferred coordinate transfer within same location from stored event-row descriptor; timestamp inferred from Sentinel source index'},
 'inferred','high',['Leakage/contamination qualification','event independence','old versus new source correspondence'],'Report exact screen coverage, pairs and metadata basis; recover GDACS native footprints if available without acquiring a new dataset.',ev=[art('prompt2_geographic_screen.json'),art('prompt2_data_checks.json')])
rec('P2-A05',['R1.3','R2.2'],paper('supplement','S1','STURM/WorldFloods share3EMS activations and coincident ground at2AOIs','share three'),[],
 {'shared_activations':['EMSR419','EMSR422','EMSR438'],'positive_area_intersection_pairs':17,'overlapping_WF_AOIs':2,
  'details':'8 STURM chips overlap EMSR422_AOI06 with metadata timestamps7.52minutes apart;9 chips overlap EMSR438_AOI06 with9.983days separation. All17 STURM footprints are contained by corresponding WF scene bounds.',
  'EMSR419':'Shared activation, no actual raster-footprint intersection found.','sensitivity_status':'Manuscript reports excluding2AOIs moves WF<.001; exact underlying exclusion computation not located in shipped S1 artifacts.'},
 'verified','medium',['External benchmarks as replications','pooled uncertainty'],'Call benchmarks complementary; use independent-event grouping; preserve published sensitivity as reported pending its executed artifact.',ev=[art('prompt2_data_checks.json')])
rec('P2-A06',['R1.3','R2.2'],paper('main','IV-A evaluation/statistics','WorldFloods18AOIs/11events; STURM46maps/29events','18 scenes or 11 events'),
 [('evaluation/perscene_eval.py',[280,313])],
 {'WF_internal_positive_area_pairs':checks['footprint_checks']['wf_internal']['pairs'],
  'STURM_internal_pairs':checks['footprint_checks']['sturm_internal']['pairs'],
  'interpretation':'WorldFloods includes same-event AOIs sharing ground, including ZOMBA repeated whole footprint~1.993days apart. Pixel/map units are not independent events. Raster-bound overlap does not itself prove duplicated scorable labels or exact imagery.'},
 'verified','medium',['WorldFloods macro map statistics','STURM/WF event uncertainty'],'Prefer event grouping and document overlapping AOIs; do not delete cases or change canonical scores.',ev=[art('prompt2_data_checks.json')])
cache='/home/schmalstieg/.cache/huggingface/hub/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-TL-Sen1Floods11/snapshots/91ce9d38086a80b078a192b374df758b8855b732/config.yaml'
rec('P2-A07',['R1.3','R2.2'],paper('main','III-A Teacher','Released Prithvi task teacher adapted using252S11 training chips','fine-tuned on the 252'),
 [('models/teacher.py',[1,170]),(cache,[33,45,95,136])],
 {'released_checkpoint_tensor_identity':base['teacher_identity_audit']['recommended_pin'],
  'adapted_tensor_digest':base['teacher_identity_audit']['checkpoint_reports'][0]['remapped_tensor_content_sha256'],
  'local_release_config':{'path':cache,'sha256':sha(cache),'datamodule':'Sen1Floods11NonGeoDataModule','early_stop_monitor':'val/loss','selection_membership_snapshot':'not included'},
  'teacher_task_label_use':'Public model card says Sen1Floods11; release config uses official-split datamodule. Matches current intended split at framework level; actual teacher training member list and best-checkpoint selection rule are not preserved locally.',
  'pretraining':'Potential unlabeled foundation pretraining overlap not exhaustively audited; complete source acquisition membership unavailable.'},
 'unknown','medium',['Teacher reference','S11 KD/OFA interpretation','claims of exhaustive contamination absence'],'Retain pinnedteacher tensor identity; distinguish verified released config from inferred official membership and unknown pretraining overlap.',
 status='needs-analysis-only',ev=[art('protocol_audit.json'),{'url':'https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/98c11be/README.md','local_copy':str(OUT/'prompt2_sources/teacher_modelcard.md'),'sha256':sha(OUT/'prompt2_sources/teacher_modelcard.md')}])
rec('P2-A08',['R1.3','R2.2'],paper('main','Appendix A normalization / III-B embedding','Fixed S11-training mean/std; no fitting external statistics','no per-image statistics'),
 [('data/sen1floods11.py',[12,17]),('data/diversity_pseudo.py',[79,87,150,174]),('data/ood_datasets.py',[170,174,650,674])],
 {'normalization':base['recipe']['norm_mean'],'normalization_std':base['recipe']['norm_std'],
  'descriptor_scaling':'1st/99th percentiles over filtered GDACS candidate descriptors; external data not concatenated by this selection implementation.',
  'provenance_limit':'Values match rounded TerraTorch supplied S11 constants; exact original computation of training-statistics arrays unavailable. Other dataset normalization modes exist but primary explicit evaluator selects original STURM/s11 WorldFloods.'},
 'verified','low',['Main preprocessing','subset-selection protocol'],'Freeze explicit normalization mode and constants per run; qualify original-stat derivation evidence.',ev=[art('protocol_audit.json')])
rec('P2-A09',['R1.3','R2.2'],paper('supplement','S2 Subset selection','GLISTER validation objective uses S11validation','Sen1Floods11 validation split'),
 [('data/glister_pseudo.py',[187,208]),('quantization/ptq.py',[93,108]),('training/trainer.py',[245,302])],
 {'data_roles':{'S11_train':'human hard labels for supervised gradients; task teacher training intended same split',
                'GDACS_pseudo':'teacher hard masks all objectives, live teacher soft targets for LogitKD/OFA',
                'S11_validation':'checkpoint selection; GLISTER validation-label objective; calibration images only in calibration forward; QAT validation labels selection',
                'S11_test':'fixed reporting through test loader after selected checkpoint',
                'STURM_WorldFloods':'fixed evaluation loaders in declared primary runs; historical choice influence unresolved'},
  'image_vs_label_calibration':'Calibration collects images from S11val; QAT training data and labels followdeclared train loader. No test-set gradients or implicit adaptation found in primary forward evaluator.',
  'history_limit':'Current code/launchers establish allowed paths, not proof that earlier benchmark scores were never inspected.'},
 'verified','medium',['Selector comparisons','calibration provenance','S11 teacher-free labels'],'Explain every data role explicitly; validation use is legitimate but affects selection interpretation.',ev=[art('protocol_audit.json')])
rec('P2-A10',['R1.3','R2.2'],paper('main','VII Limitations, Benchmark-guided development','Method choices refined using same public benchmarks reported in paper','Benchmark-guided development'),[],
 {'verified_manuscript_statement':'Benchmark-guided development sentence conflicts with untouched-external-set account supplied in prompt1.',
  'user_account_classification':'user-reported','actual_historical_decisions':'unknown','history_limit':'Git history has only2cleaned-release commits; current code absence cannot prove no score-guided choices.',
  'precise_question':'Which method choices, if any, changed after STURM or WorldFloods scores were viewed? If none, which benchmark did the manuscript sentence refer to?'},
 'inconsistent','high',['Untouched held-out claim','generalization interpretation'],'Preserve contradictory evidence and obtain factual history; forward freeze does not retroactively clean development.',status='blocked',ev=[art('protocol_audit.json')])
rec('P2-B01',['R1.3','R1.5','R2.2'],paper('main','Appendix A STURM','Inverse normalization; B8substitute; NaN-chipguard removes~0.6%','guard removes'),
 [('data/ood_datasets.py',[530,545,639,695]),('data/channel_adapter.py',[74,91])],
 {'bands':'Actual native indices0,1,2,6,7,8; manuscript says index6 is broadB8 fed to narrowB8A slot. Source docstrings incorrectly name thisB8A.',
  'units_transform':'float32 raw values; nonfinite replaced0 BEFORE inverse0.04+0.12v; then clip[0,1]; perturbations before resize and fixed normalization.',
  'discrepancy':'Executable retains2675chips and replaces nonfinite values rather than dropping chips; inverse is clipped rather than un-clipped as described. No new processing-level inference was promoted to metadata fact.',
  'processing_level':'Paper distribution-based atmospheric-correction argument remains inferred; no verified product-level field locally.'},
 'inconsistent','medium',['All STURM scores','synthetic haze interpretation'],'Correct eventual methodological description to executed transform; preserve scoring and rankings.',ev=[art('protocol_audit.json'),art('prompt2_reference_manifest.json')])
rec('P2-B02',['R1.3','R1.5','R2.2','R2.5'],paper('main','IV-A Permanent-water harmonization','Permanent JRCpixels ignored; seasonal water retained','yearly-history layer'),
 [('data/ood_datasets.py',[95,107,266,286])],
 {'GT':'Two-band WorldFloodsv2 release clear/cloud andland/water product; lw1background/lw2water/lw0invalid, cloud==2ignored, all13spectral bands0ignored.',
  'reference_product':'Co-registered JRC annual layer PERMANENTWATERJRC; metadata permanent_water_image_path2018.tif/2019.tif/2020.tif by scene. JRC3ignored;2seasonal retained;0JRCnodata not automatically ignored if GT valid.',
  'reference_year_by_scene':{k:v['jrc_product_year'] for k,v in refs['WorldFloods'].items()},
  'semantics':'S11positive water includes permanentwater; STURM1/2positive,3/4/5ignored. External ignore masks narrow scoring; they do not convert a single-image model to newly flooded-land detector.',
  'deployment_role':'JRC is evaluation mask here, not modelinput; acquisition-year annual history is retrospective support, not established real-time prior.'},
 'verified','medium',['Flood-only external scores','operational new-inundation claim','surface-water semantics'],'Name exact release product/year/class map and evaluation-only role; qualify event-inundation usecase.',ev=[art('prompt2_reference_manifest.json')])
rec('P2-B03',['R1.3','R1.5','R2.2'],paper('supplement','S3 Evaluation-scale control','STURM128/224/512grid comparison; teacher parity protocol-specific','Evaluation-scale control'),
 [('data/ood_datasets.py',[624,635]),('data/sen1floods11.py',[48,59])],
 {'native_grids':'S11actual512x512; STURMactual128x128; WorldFloods10m512native windows withpartialedges.',
  'interpolation':'Albumentations Resize defaults bilinear image/nearest categorical mask; augmentation shares image/mask geometric transform.',
  'STURM_scale_readout':'128->512 nearestreference yields exact16xnative valid/reference-water counts. Upsampling creates no new physical detail; published teacher parity at512does not establish resolution-invariant parity.',
  'scale_artifact':str(ROOT/'paper_source_data/Supp_InText_Statistics/S3_harmonisation/sturm_scale_control.csv')},
 'verified','medium',['STURM teacher parity','geometry/R2.2interpretation'],'Retain main512protocol and separate native-grid diagnostic; re-use existing scale controls.',ev=[art('prompt2_reference_manifest.json'),{'path':str(ROOT/'paper_source_data/Supp_InText_Statistics/S3_harmonisation/sturm_scale_control.csv'),'sha256':sha(ROOT/'paper_source_data/Supp_InText_Statistics/S3_harmonisation/sturm_scale_control.csv')}])
rec('P2-B04',['R1.3','R2.2'],paper('main','IV-A WorldFloods geometry','785native512windows','785 native'),
 [('data/ood_datasets.py',[217,240,173,178,290,317])],
 {'actual_windows':windows['count'],'scenes':len(windows['scene_window_counts']),'partial_boundary_windows':windows['partial_window_count'],
  'native_pixels_retained_windows':windows['native_pixels_in_retained_windows'],'scoring_grid_pixels_before_mask':windows['full_scored_grid_pixels_before_mask'],
  'finding':windows['geometry_finding'],'padding':'No pad pixel is introduced; each edge window is stretched to512in both dimensions. Canonical pixel pooling counts transformed-grid samples, not native equal-area pixels.'},
 'inconsistent','medium',['WorldFloods micro pooling/physical-area interpretation'],'State actual border resize andgrid-weight semantics; preserve executable protocol, do not patch to improve scores.',ev=[art('prompt2_worldfloods_windows.json')])
rec('P2-B05',['R1.3','R2.2','R2.5'],paper('main','Appendix A / IV-A ignore semantics','Ignored pixels excluded consistently','ignore protocol'),
 [('data/ood_datasets.py',[266,286,682,696]),('utils/metrics.py',[15,70])],
 {'label_hashes':'All446S11/2675STURM/18WorldFloodsreferences and18JRCmask files have content hashes; native mapped S11/STURMbitmap hashes also stored.',
  'grid_alignment':'All image/reference grids agree within1e-12absolute/relative affine tolerance;445S11exactaffines differ onlyserializationroundoff(max7.151e-14pixel-scale).STURM/WFall exact.',
  'STURMclean_count_verification':{'fresh_native_valid_times16':674812000,'saved_MNDWI_valid':674812000,'fresh_native_water_times16':166972576,'saved_MNDWI_water':166972576},
  'WFmask_limit':'Reference-onlyauditvalid totals omit all13-band-zero spectralnodata mask; canonical savedinteger counts represent actual executed reader pixels.',
  'loss_limit':'Hardloss masks-1; softKD valid-pixel reduction is a distinct issue in2Caudit; identical metric masks do not prove identical loss reduction.'},
 'verified','low',['Clean/corrupted comparatorvalidpixel fairness','canonicalmaskcoverage'],'Attach reference/mask/config identities; preserve ignoredsignedlabels throughnearestresize and exposeactualcounts.',ev=[art('prompt2_reference_manifest.json')])
rec('P2-B06',['R1.3','R2.2','R2.5'],paper('main','IV-A Aggregation','Micro mIoU and macroper-event/map distinct;4seed mean','micro-aggregated'),
 [('evaluation/perscene_eval.py',[264,277,286,313]),('evaluation/revision_eval.py',[49,64,136,168]),('utils/metrics.py',[84,119])],
 {'class_order':'channel0background/channel1water;argmaxatfixeddecisionrule; spectralMNDWI>0asclass1.',
  'counts':'2x3per-classTP/FP/FN arrays retainTN=class0TP; precision/recallderivecounts, cannotderiveuniquelyfromIoU.',
  'evaluation_mode':'Loadedteacher/studenteval(),run@torch.no_grad();primary pathhasnoTTAfittingortrain-modeBatchNormupdate.',
  'grouping':'STURMmetadata46maps29EMSactivations;WF18AOIs11EMScodes;S11country/locationprefixisproxywithoutofficialeventmetadata.',
  'undefined_convention':'Newrevisionratio returnsNoneonzerodenominator andaverageonlydefinedclassIoUs;oldergenericfinalizer usesNaNandexcludesundefinedclasses. Historical displayedtablesretaintheiroriginalconventions.',
  'per_seed':'Eachcheckpointisindependentlyevaluated;seedmeanofper-seedscoreisnotprobabilityensemble.'},
 'verified','low',['Allreportedsegmentationmetrics','precision/recallreadout','uncertainty'],'Exportintegercountsandexplicitunit/mask/undefinedcontributors; aggregateeventsandtrainingseedsseparately.',ev=[art('prompt2_reference_manifest.json')])
rec('P2-B07',['R1.3','R2.2'],paper('main','IV-A/evaluationtableprovenance','Reported scores trace frozenbestcheckpoint','best checkpoint'),
 [('evaluation/revision_eval.py',[77,108]),('training/trainer.py',[258,302])],
 {'current_selection':'S11valmicro2classmIoU;every2epochsandfinalepoch;rollingwindow1;strictgreater,earlierretainedties;earlystop20nonimprovingchecks.',
  'historical_limit':'Oldercheckpoint/trainerresultscommonlylackexecutedcode/config/datahashandrawpredictionbitmaps. Colocation/familynamesassociatepaperrowsbutdo not prove exactexecutedidentity.',
  'new_saved_outputs':'Revisionoutputsrecordspec/checkpoint/sourcehashesandenvironment;countsretainoriginalmap/eventaggregates.',
  'mask_identity_limit':'Referencefile/nativebitmaphashesnewlyverified;historicalresizedscoringbitmaphashesnotretainedandcannotberecoveredfromIoUoraggregatecounts.Originalpredictionmapsabsent.'},
 'unknown','medium',['Archivedcleanreproduction','paperrowtocheckpointassociation'],'Keepcount-derivedreadoutslabelledsaved-artifactconditional;linkexplicitcurrentfingerprints;neverclaimolderexecutionfromcurrentcodescan.',status='needs-analysis-only',ev=[art('protocol_audit.json'),art('prompt2_reference_manifest.json')])
rec('P2-A11',['R1.3','R2.2','R1.1'],paper('main','III-B Eq1-2descriptor','muT/spatialstddefinedfromwaterprobability;stdcalleduncertainty','spatial mean of the'),
 [('data/pseudo_labeled.py',[383,409]),('data/diversity_pseudo.py',[32,58,79,87])],
 {'executed_confidence':'Generationstoresprobs.max(dim=1),notwaterclassprobability;selectorusesmean/stdofsavedmaxclassconfidence. Storedsampleconfidence0.5..1confirms.',
  'dispersion':'Spatialstdofconfidenceispixelvariation,notcalibratedmodeluncertainty;meanmaxclassprobabilityisconfidencebutnotmeanwaterprobability.',
  'fraction_denominator':'Storedregeneratedhardmasksreceiptcontainsno-1pixels;mask.meanisvalidforbinaryfullgridthere. Generalmask.meanwouldbeincorrectwith-1ignoredpixels.',
  'history_limit':'Knownoriginalsourcecodeanddescriptorvaluesneedlinkage;regeneratedmasksnotexactoriginalpoolandindividualwaterfractionmaynowcrosstheoriginalselectionthreshold.'},
 'inconsistent','medium',['Filtering/selectorinterpretation','claimeduncertaintycriterion'],'Describeexactmaxconfidence/spatialdispersionandoriginalselectionlineage;retainchosenIDswithoutregeneration.',ev=[art('protocol_audit.json')])
rec('P2-B08',['R1.6','R2.5'],paper('main','VII dryscene limitation','Entirelywater-freefalsealarmsuncharacterized','fully dry scenes'),[],
 {'S11human_test_zero_water_chips':6,'named_ids':[r['id'] for r in base['dry_reference']['S11_test_zero_label_pilot']],
  'independent_reference':'HumanS11LabelHandvalidpixelszero-positive;partialignoredareasretained.',
  'independent_event_gap':'TheseGhana/India/Nigeriacasescome fromeventlocationsalsousedtrain/val. Theyareheld-outchipdiagnostics,notindependentnew-eventdryset.',
  'teacher_empty':'Teacher-emptytraining/pre-eventcandidatescannotstandinforindependentdrytruth.'},
 'verified','high',['Dryfalsealarmoperationalclaim'],'Reuse6chipdiagnosticcountsandmarkbroaderindependentdryreferencegap;do not addnegativetraining.',status='blocked',ev=[art('protocol_audit.json'),art('prompt2_reference_manifest.json')])
result={'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_revision':REV,
        'scope':'Prompt2 sections2A-B; CPU-onlyread-onlysource/dataaudit, separateevidenceoutputs;noGPUtraining/inference',
        'classification_vocabulary':['verified','inconsistent','user-reported','inferred','unknown'],
        'key_findings':[
          'Actual S11 loaders match all official252/89/90 IDs and paired references;15Bolivia IDs are separately disjoint. No S11 split footprint overlap found, but the10locations are shared across train/validation/test.',
          'Approved regenerated N2500 differs from original by127sources; distinguish archived and current cohorts.',
          'Original GDACS centroid screen reproduces54nearbySTURMchips; current regenerated cohort35. Coverage2499/2500centroids,0/2500nativeGDACSfootprints; tight-screen0is notproof ofindependence.',
          'STURM andWorldFloods share3activations;17STURMfootprints overlap2WorldFloodsscenes. WorldFloods alsohas5internalfootprintoverlaps;event grouping matters.',
          'WorldFloods contains785windows from18scenes,including148partialbordersresized to512. No padding, butedgeground area receivesdifferentgridweight.',
          'STURM clipping/nonfinite replacement disagrees withmanuscript un-clipped/dropped-chipdescription; preserveexecutedprotocol.',
          'Reference masks haveperfilehashes/nativebitmaphashesnow, buthistoricalresizedscoringbitmaps andpredictionmapswere notsaved.',
          'Teacher releasedconfig matchesofficial-splitframework; exacthistoricalmembership/checkpointselection andfoundationpretrainingoverlap remainunknown.',
          'Manuscriptbenchmark-guideddevelopmentstatement stillconflicts withuntouchedexternalsetaccount; historicaldecisionevidence requires factual clarification.',
          'S11sixhuman-labelledzerowater testchipsprovideadiagnostic; anindependent-eventdryreference set remainsmissing.'
        ],'records':records,'artifacts':[art(n) for n in ['prompt2_data_checks.json','prompt2_loader_membership.json','prompt2_reference_manifest.json','prompt2_geographic_screen.json','prompt2_worldfloods_windows.json']],
        'unresolved_inputs':['Historical choices changed after viewing external benchmark scores','Native GDACS footprints/raw raster access','Exact released teacher executed membership/selection evidence','Independently labelled dry events outside training/calibration/selection']}
# Keep human-facing strings readable even when compact field text was assembled above.
(OUT/'prompt2_data_evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print('records',len(records),'output',OUT/'prompt2_data_evidence.json')
