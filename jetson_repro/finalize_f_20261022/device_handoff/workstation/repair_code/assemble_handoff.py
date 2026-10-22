#!/usr/bin/env python3
"""Install qualified mixed repair exports without overwriting historical graphs."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'device_handoff';sys.path.insert(0,str(D/'runtime'))
import precision_islands

def read(p):return json.loads(Path(p).read_text())
def write(p,o):Path(p).write_text(json.dumps(o,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=read(D/'RUN_MANIFEST.json');attempts=list(csv.DictReader((D/'B0_QAT_ATTEMPTS.csv').open()));summary=[];retired=[]
    for seed in (42,64,91,107):
        folder=R/f'fit_cpu_s{seed}'
        if (R/f'mixed_exceptions_s{seed}_v2/RESULT.json').exists() and read(R/f'mixed_exceptions_s{seed}_v2/RESULT.json').get('acceptable'):folder=R/f'mixed_exceptions_s{seed}_v2'
        if (R/f'continue_cpu_s{seed}/RESULT.json').exists() and read(R/f'continue_cpu_s{seed}/RESULT.json').get('acceptable'):folder=R/f'continue_cpu_s{seed}'
        if (R/f'parent_match_cpu_s{seed}/RESULT.json').exists() and read(R/f'parent_match_cpu_s{seed}/RESULT.json').get('acceptable'):folder=R/f'parent_match_cpu_s{seed}'
        result=read(folder/'RESULT.json');assert result['acceptable'] and result['status']=='SOURCE_ACCURACY_RECOVERED'
        choice=result['selected'];gate=choice['source_gate'];assert gate['accuracy_pass'] and choice['unrounded_structure_all89']
        assert sha(gate['onnx']['path'])==gate['onnx']['sha256'] and sha(choice['checkpoint']['path'])==choice['checkpoint']['sha256']
        previous=f'B0_s{seed}_qat_mixed';ident=f'B0_s{seed}_qat_mixed_repaired';entry=next(e for e in manifest['engines'] if e['id']==previous);retired.append(dict(entry))
        dest=D/'models'/ident;dest.mkdir(exist_ok=False);shutil.copy2(gate['onnx']['path'],dest/'model.onnx');shutil.copy2(D/entry['source_reference'],dest/'SOURCE_REFERENCE.json');write(dest/'SOURCE_GATE.json',gate)
        provenance=D/'workstation'/f'B0_mixed_repair_s{seed}';provenance.mkdir(exist_ok=False)
        for filename in ('RESULT.json','TRAINING_RECEIPT.json','SCALES.json','FLOAT_CONSTRUCTION.json'):
            shutil.copy2(folder/filename,provenance/filename)
        shutil.copy2(choice['export_parity']['path'],provenance/'EXPORT_PARITY.json')
        if folder.name.startswith(('mixed_exceptions','continue','parent_match')):
            parent=R/f'fit_cpu_s{seed}';negative=provenance/'parent_failed_fit';negative.mkdir()
            for n in ('RESULT.json','TRAINING_RECEIPT.json','SOURCE_HISTORY.json'):shutil.copy2(parent/n,negative/n)
            shutil.copy2(R/'MIXED_EXCEPTION_PROTOCOL_v2.json',provenance/'MIXED_EXCEPTION_PROTOCOL.json')
            diagnostic=R/f'mixed_exceptions_s{seed}_v2'
            for n in ('ARMS.json','RESULT.json'):shutil.copy2(diagnostic/n,negative/('precision_diagnostic_'+n))
            if folder.name.startswith(('continue','parent_match')):shutil.copy2(R/'CONTINUATION_PROTOCOL.json',provenance/'CONTINUATION_PROTOCOL.json')
            if folder.name.startswith('parent_match'):
                shutil.copy2(R/'PARENT_MATCH_PROTOCOL.json',provenance/'PARENT_MATCH_PROTOCOL.json')
                failed=provenance/'failed_hard_target_continuation';failed.mkdir()
                for n in ('RESULT.json','TRAINING_RECEIPT.json','SOURCE_HISTORY.json'):shutil.copy2(R/'continue_cpu_s91'/n,failed/n)
                attempts.append(dict(architecture='B0',seed=seed,series='failed_hard_target_continuation',updates=read(failed/'RESULT.json')['updates'],status='SOURCE_ACCURACY_NOT_RECOVERED',strict='NOT_REACHED',structure='NOT_REACHED',pooled_drop='',event_drop='',water_drop='',receipt=f'workstation/B0_mixed_repair_s{seed}/failed_hard_target_continuation/RESULT.json'))
            before=read(parent/'RESULT.json');drop=before['final_source_gate']['drops']
            attempts.append(dict(architecture='B0',seed=seed,series='mixed_smooth_initial_full56',updates=before['updates'],status=before['status'],strict='NOT_REACHED',structure='NOT_REACHED',pooled_drop=drop['mIoU'],event_drop=drop['equal_event_mIoU'],water_drop=drop['water_IoU'],receipt=f'workstation/B0_mixed_repair_s{seed}/parent_failed_fit/RESULT.json'))
        selected_folder=Path(gate['onnx']['path']).parent
        if (selected_folder/'LINEAR_TRANSFORM.json').exists():shutil.copy2(selected_folder/'LINEAR_TRANSFORM.json',provenance/'LINEAR_TRANSFORM.json')
        entry.update(id=ident,onnx=f'models/{ident}/model.onnx',onnx_sha256=sha(dest/'model.onnx'),source_reference=f'models/{ident}/SOURCE_REFERENCE.json',source_gate=f'models/{ident}/SOURCE_GATE.json',source_accuracy_pass=True,strict_logit_parity_all89=choice['strict_logit_parity_all89'],unrounded_structure_all89=True,source_admission='SOURCE_QUALIFIED' if choice['strict_logit_parity_all89'] else 'SOURCE_ACCURACY_PASS_STRICT_QDQ_DIAGNOSTIC',qat_series='mixed_smooth_repair',graph_variant=gate['graph_variant'],fp32_islands=f'models/{ident}/FP32_ISLANDS.json',training_receipt=f'workstation/B0_mixed_repair_s{seed}/TRAINING_RECEIPT.json',deployment_quantized_convs=result.get('deployment_quantized_convs',56),floating_conv_exceptions=result.get('floating_conv_exceptions',[]),qat_objective=read(folder/'TRAINING_RECEIPT.json').get('loss','CE on frozen cached teacher-hard512'))
        for slot in manifest['primary_matrix']:
            if slot.get('engine')==previous:slot['engine']=ident;slot['status']=entry['source_admission']
        attempts.append(dict(architecture='B0',seed=seed,series='mixed_smooth_repair',updates=result['updates'],status=result['status'],strict=choice['strict_logit_parity_all89'],structure=True,pooled_drop=gate['drops']['mIoU'],event_drop=gate['drops']['equal_event_mIoU'],water_drop=gate['drops']['water_IoU'],receipt=f'workstation/B0_mixed_repair_s{seed}/RESULT.json'))
        summary.append(dict(seed=seed,engine=ident,updates=result['updates'],drops=gate['drops'],accuracy_pass=True,strict_logit_parity_all89=choice['strict_logit_parity_all89'],unrounded_structure_all89=True,onnx_sha256=entry['onnx_sha256'],checkpoint=choice['checkpoint'],native_execution='PENDING_JETSON',deployment_quantized_convs=result.get('deployment_quantized_convs',56),floating_conv_exceptions=result.get('floating_conv_exceptions',[]),qat_objective=read(folder/'TRAINING_RECEIPT.json').get('loss','CE on frozen cached teacher-hard512')))
    for e in manifest['engines']:
        if e['architecture']=='B0':
            p=D/e['fp32_islands'];write(p,precision_islands.from_onnx(D/e['onnx']));e['fp32_islands_sha256']=sha(p)
    manifest['retired_diagnostics']=retired;manifest['b0_mixed_repair']='Original four failed QAT graphs retained as historical records; replaced primary entries have fresh identities. New source-qualified accuracy candidates retain strict parity diagnostics and require native FP32 island certification.'
    write(D/'RUN_MANIFEST.json',manifest)
    for name,rs in [('B0_QAT_ATTEMPTS.csv',attempts),('MODEL_COVERAGE.csv',manifest['primary_matrix'])]:
        keys=list(dict.fromkeys(k for r in rs for k in r))
        with (D/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rs)
    for source in ('FIT_PROTOCOL.json','provenance/RELOAD_VERIFICATION.json'):
        shutil.copy2(R/source,D/'provenance'/Path(source).name)
    write(D/'provenance/B0_MIXED_REPAIR.json',dict(status='FOUR_SEED_SOURCE_ACCURACY_RECOVERED',seeds=summary,precision='Per-seed QDQ Conv count and floating exceptions listed explicitly; one unquantized classifier; all8 attention MatMuls floating; no Conv output QDQ; FP32 attention/normalization/classifier required at native build',selection='first source-passing epoch after warmup and at least one quantized epoch; no external scores or new teacher inference',scope='Actual optimized ORT graph accuracy passes; strict QDQ equivalence and native engine acceptance remain separate'))
    source=read(D/'SOURCE_PACKAGE.json');source['mixed_repair_source']=str(R.relative_to(R.parents[2]));source['repair_protocol_sha256']=sha(R/'FIT_PROTOCOL.json');source['new_native_work']='PENDING_JETSON: four recovered mixed B0 exports plus FP32 attention/classifier constraints';write(D/'SOURCE_PACKAGE.json',source)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
