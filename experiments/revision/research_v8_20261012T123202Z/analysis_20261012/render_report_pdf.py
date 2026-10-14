"""Render the saved report to PDF using the installed system ReportLab."""
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).resolve().parent.parent
source = BASE/'report_v8_codex_audit_20261012.md'
target = source.with_suffix('.pdf')
fontdir = Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('DV','DejaVuSans.ttf'),('DV-Bold','DejaVuSans-Bold.ttf'),('DV-Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
styles = getSampleStyleSheet()
body = ParagraphStyle('reportbody',fontName='DV',fontSize=8.8,leading=12.6,spaceAfter=7)
heads = {1:ParagraphStyle('reporttitle',parent=body,fontName='DV-Bold',fontSize=18,leading=22,spaceAfter=14),
         2:ParagraphStyle('reportsection',parent=body,fontName='DV-Bold',fontSize=12,leading=16,
                          spaceBefore=13,spaceAfter=8,keepWithNext=True)}
small = ParagraphStyle('reportcell',parent=body,fontSize=6.9,leading=9.2,spaceAfter=0)
code = ParagraphStyle('reportcode',fontName='DV-Mono',fontSize=6.4,leading=8.2,
                      backColor=colors.HexColor('#f1f4f7'),spaceBefore=3,spaceAfter=8)
bullet = ParagraphStyle('reportbullet',parent=body,leftIndent=10,firstLineIndent=-7)


def inline(text):
    text=escape(text)
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<link href="\2" color="#126e98">\1</link>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="DV-Mono" size="7.4">\1</font>',text)
    return text


doc = SimpleDocTemplate(str(target),pagesize=A4,rightMargin=36,leftMargin=36,
                        topMargin=38,bottomMargin=38,title='FloodDistill v8 research results and research direction, 12 October 2026',
                        author='Codex: independently verified experiment report')
width=A4[0]-72
story=[]
lines=source.read_text().splitlines()
i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():
        i+=1;continue
    if line.startswith('```'):
        block=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):
            block.append(lines[i]);i+=1
        story.append(Preformatted('\n'.join(block),code,maxLineLength=103));i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            vals=[v.strip() for v in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',v) for v in vals):
                rows.append(vals)
            i+=1
        n=len(rows[0])
        if n==2:fractions=[.39,.61]
        elif n==3:fractions=[.24,.32,.44]
        elif n==4:fractions=[.24,.15,.45,.16]
        elif n==5:fractions=[.25,.13,.14,.14,.34]
        elif n==6:fractions=[.23,.11,.16,.12,.19,.19]
        else:fractions=[1/n]*n
        data=[[Paragraph(inline(v),small) for v in row] for row in rows]
        tab=Table(data,colWidths=[width*f for f in fractions],repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),
            ('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),
            ('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#437588')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f8')]),
            ('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d4dbdf'))]))
        story+=[tab,Spacer(1,9)];continue
    if line.startswith('!['):
        match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
        pic=Image(str(BASE/match[2]));pic.drawHeight*=width/pic.drawWidth;pic.drawWidth=width
        story+=[pic,Paragraph(inline(match[1]),small),Spacer(1,9)];i+=1;continue
    if line.startswith('#'):
        level=len(line)-len(line.lstrip('#'))
        story.append(Paragraph(inline(line[level:].strip()),heads.get(level,heads[2])));i+=1;continue
    parts=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','![','- ')):
        parts.append(lines[i]);i+=1
    text=' '.join(parts)
    if text.startswith('- '):
        text='• '+text[2:];style=bullet
    else:style=body
    story.append(Paragraph(inline(text),style))


def footer(canvas,document):
    canvas.setFont('DV',7)
    canvas.setFillColor(colors.HexColor('#536470'))
    canvas.drawString(36,22,'FloodDistill · audited v8 research results · 12 October 2026')
    canvas.drawRightString(A4[0]-36,22,str(document.page))


doc.build(story,onFirstPage=footer,onLaterPages=footer)
import hashlib
import json
analysis = Path(__file__).resolve().parent
receipt_path = analysis/'report_receipt.json'
receipt = json.loads(receipt_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert receipt['report_sha256'] == sha(source)
receipt['pdf_path'] = str(target)
receipt['pdf_sha256'] = sha(target)
receipt['pdf_pages'] = doc.page
receipt['analysis_helper_sha256'] = {str(path):sha(path) for path in analysis.glob('*.py')}
receipt['table_sha256'] = {str(path):sha(path) for path in analysis.glob('*.csv')}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(target)
