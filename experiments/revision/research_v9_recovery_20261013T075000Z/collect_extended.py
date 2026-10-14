"""Collect the same scientific readouts and disclose the approved window extension."""
import runpy
import subprocess
import common_v9 as c

runpy.run_path(str(c.ROOT / 'collect_recovery.py'), run_name='__main__')
approval = c.read(c.ROOT / 'deadline_extension_approval.json')
report = c.ROOT / 'report_v9.md'
with report.open('a') as output:
    output.write('\n## Approved recovery window\n\n')
    output.write('The user explicitly approved extending execution to18:35 CEST on13October2026. '
                 'The original launch and first-GPU timestamps remain recorded. '
                 'This approval overrides the eight-hour execution cutoff only; '
                 'the original absolute wall limit, cumulative30 GPU-command-hour cap, '
                 'inherited4-hour V9 incremental cap, scientific settings and ten-minute collection reserve remain enforced. '
                 'Approval receipt: deadline_extension_approval.json.\n')
subprocess.run(['/usr/bin/python3', str(c.ROOT / 'render_report_v9.py')], check=True)
receipt = c.read(c.ROOT / 'collection_receipt.json')
receipt['report_sha256'] = c.sha(report)
receipt['approved_execution_deadline'] = approval['approved_execution_deadline_Europe_Berlin']
c.save(c.ROOT / 'collection_receipt.json', receipt)
index = c.read(c.ROOT / 'evidence_index.json')
for name in ['report_v9.md', 'report_v9_codex_audit_20261013.pdf', 'collection_receipt.json', 'deadline_extension_approval.json', 'manifest_extended.json']:
    index['files'][name] = c.sha(c.ROOT / name)
c.save(c.ROOT / 'evidence_index.json', index)
