"""Render the collected Markdown report using the existing system ReportLab."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
root=Path(__file__).resolve().parent;styles=getSampleStyleSheet();story=[]
for para in (root/'report_v9.md').read_text().split('\n\n'):
    if not para.strip():continue
    if para.startswith('# '):style=styles['Title'];para=para[2:]
    elif para.startswith('## '):style=styles['Heading2'];para=para[3:]
    else:style=styles['BodyText']
    story.extend([Paragraph(escape(para).replace('\n','<br/>'),style),Spacer(1,6)])
SimpleDocTemplate(str(root/'report_v9_codex_audit_20261013.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=40,title='FloodDistill V9 bounded research audit').build(story)
