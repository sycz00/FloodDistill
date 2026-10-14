import json,subprocess
from pathlib import Path
import common_v9 as c
import collect_v9_final06 as collector
collector.main()
state=c.read(c.ROOT/'state.json',{})
complete=all(c.read(c.ROOT/name,{}).get('outcome')=='completed' for name in ['evaluation_normal.json','evaluation_sealed.json','evaluation_source_final03.json'])
path=c.ROOT/'report_v9.md'
text=path.read_text().replace('V9 research report — Final collection','V9 research report — Recovery collection; scientific evaluation incomplete' if not complete else 'V9 research report — Completed scientific readout')
text+='\n## Recovery audit\n\nAll20 original training runs were completed and reused. The final teacher readout crashed only in backend precision telemetry, after3639 teacher windows; dependent readouts were unavailable. This fresh queue reruns the full prediction matrix without reusing partial predictions, retains original clocks and cumulative30 GPU-command-hour and incremental4-hour caps. Historical reports and failures remain unchanged.\n\n'
text+='Scientific evaluation complete: '+str(complete)+'. Operational collection does not imply complete experimental results.\n'
path.write_text(text)
subprocess.run(['/usr/bin/python3',str(c.ROOT/'render_report_v9.py')],check=True)
receipt=c.read(c.ROOT/'collection_receipt.json');receipt.update(report_sha256=c.sha(path),scientific_evaluation_complete=complete);c.save(c.ROOT/'collection_receipt.json',receipt)
files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.glob('*') if p.is_file() and p.suffix in ['.json','.csv','.md','.py','.pdf'] and p.name not in ['evidence_index.json','scheduler_status.json','state.json','control.json']}
c.save(c.ROOT/'evidence_index.json',dict(created=c.q.now(),files=files))
