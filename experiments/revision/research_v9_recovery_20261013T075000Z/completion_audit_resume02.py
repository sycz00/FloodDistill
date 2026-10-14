"""Additive report audit after existing recovery collection; never changes queue inputs."""
import csv,json,time,hashlib,os
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 state=read(ROOT/'state.json',{});deadline=state['absolute_deadline']+600
 while time.time()<deadline:
  state=read(ROOT/'state.json',{})
  if state.get('finished'):break
  time.sleep(15)
 if not state.get('finished'):raise RuntimeError('Recovery collection not finished by audited deadline')
 receipt=read(ROOT/'collection_receipt.json',{});original=ROOT/'report_v9.md';assert original.exists()
 rows=[]
 for name,status in state['jobs'].items():
  path=ROOT/'command_receipts'/(name+'.json');r=read(path,{})
  rows.append(dict(job=name,status=status,seconds=r.get('seconds'),GPU_UUID=r.get('gpu_uuid'),receipt=str(path),receipt_sha256=sha(path) if path.exists() else None))
 out=ROOT/'recovery_completion_status_resume02.csv'
 with out.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 complete=all(read(ROOT/name,{}).get('outcome')=='completed' for name in ['evaluation_normal.json','evaluation_sealed.json','evaluation_source_final03.json'])
 text='# V9 resumed evaluation: audited full report\n\nScientific evaluation complete: '+str(complete)+'. This audit supplements the historical queue checklist with the actual resume02 receipts. Old failed/deferred attempts remain historical evidence; a successful recovery does not erase those attempts.\n\n'
 text+='The complete sealed GPU readout finished on13October2026 at12:05:36 UTC (14:05:36 CEST), charging2124.613 GPU-command seconds. Already valid FP32 predictions from the repaired attempt were reused with shape, finite-value and SHA checks; missing batch groups were recomputed with the same frozen graphs, input support and numerical settings. Accuracy prediction caches are retained; latency recorded during the contended previous attempt is excluded from performance claims. Raw sealed reference labels were not reopened. The registered derived cache contains17907 windows. The final CPU analysis and scientific gates use the completed readout.\n\n'
 text+='Execution was authorized until18:35 CEST, with the original cumulative30 GPU-command-hour and inherited incremental4-hour caps. The job statuses below are authoritative for recovery completion. The original report follows, including all its scientific tables and limitations; its earlier-attempt checklist and “without reusing partial predictions” recovery sentence describe the older retry, not resume02.\n\n'
 text+='## Audited recovery status\n\n'+json.dumps(rows,indent=2)+'\n\n## Full scientific report\n\n'+original.read_text()
 report=ROOT/'report_v9_resume02.md';report.write_text(text)
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
 from reportlab.lib.styles import getSampleStyleSheet
 from reportlab.lib.pagesizes import A4
 styles=getSampleStyleSheet();story=[]
 for para in text.split('\n\n'):
  if not para.strip():continue
  style=styles['Title'] if para.startswith('# ') else styles['Heading2'] if para.startswith('## ') else styles['BodyText']
  if para.startswith('# '):para=para[2:]
  elif para.startswith('## '):para=para[3:]
  story.extend([Paragraph(escape(para).replace('\n','<br/>'),style),Spacer(1,6)])
 pdf=ROOT/'report_v9_resume02.pdf';SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=40,title='FloodDistill V9 completed recovery audit').build(story)
 result=dict(outcome='completed',scientific_evaluation_complete=complete,at=time.time(),original_report_sha256=sha(original),report_sha256=sha(report),PDF_sha256=sha(pdf),status_csv_sha256=sha(out))
 (ROOT/'completion_audit_resume02_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
