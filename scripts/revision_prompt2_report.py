"""Export reviewer evidence from existing revision artifacts without model inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import subprocess
import textwrap
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data1/schmalstieg/flooddistill/revision_20261005')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_out(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extend_ledger(out, evidence, manifest, pointer, at):
    """Append reviewer records to the existing audit; CSV is a view, not another scheduler."""
    records = []
    for name, value in evidence.items():
        if not isinstance(value, dict):
            continue
        for r in value.get('records', value.get('ledger', [])):
            paper = r.get('paper', {})
            record = dict(r)
            record.update(source_evidence=name,
                paper_location=r.get('paper_location', paper.get('location')),
                claim=r.get('claim', paper.get('claim')),
                code_references=r.get('code_references', r.get('evidence', [])),
                data_run_checkpoint_evidence=r.get('data_run_checkpoint_evidence', r.get('evidence', [])))
            records.append(record)
    md = evidence.get('prompt2_mndwi_summary', {})
    if md:
        records.append(dict(id='P2-MNDWI',reviewer_ids=['R1.3','R2.5'],
            paper_location='Main spectral-rule comparison; RSE Table 3',
            claim='65.0 versus 74.39 water IoU represents a matched implementation deficit',
            classification='inconsistent',finding='Canonical90 reproduces65.0336%; all446 is59.3781%. Exact RSE grouping/masks/products unavailable; numerical and spatial probes do not explain the gap.',
            affected_results='S11 spectral competitiveness and protocol interpretation',severity='high',
            smallest_remedy='Report population/aggregation separately, preserve canonical learned-model holdout, and qualify the exact external-paper gap as unresolved.',status='verified-analysis; external-protocol gap unresolved',
            code_references=[{'path':str(REPO/'scripts/revision_mndwi_diagnostic.py'),'sha256':digest((REPO/'scripts/revision_mndwi_diagnostic.py').read_bytes())}],
            data_run_checkpoint_evidence=md.get('outputs'),root_cause=md.get('root_cause')))
    records.append(dict(id='P2-COUNTS',reviewer_ids=['R1.6','R2.5'],paper_location='Primary clean and robustness comparisons',
        claim='Precision and recall are needed at the existing fixed decision',classification='verified',
        finding='Available integer TP/FP/FN/TN exports and derived metrics validated; missing finite comparisons are already queued.',
        affected_results='Current archived-checkpoint evaluations, fixed rules; later architecture/teacher readouts pending',severity='high',
        smallest_remedy='Use exported counts with consistent aggregation; complete only existing missing inference.',status='available-counts verified; remaining-inference pending',
        code_references=[{'path':str(Path(__file__).resolve()),'sha256':digest(Path(__file__).read_bytes())}],
        data_run_checkpoint_evidence=[str(out/'confusion_counts.csv'),str(out/'metric_summaries.csv'),str(out/'evaluation_matrix.csv')]))
    audit_path = REPO/'experiments/revision/protocol_audit.json'
    immutable = {Path(i['path']).resolve() for j in manifest['jobs'] for i in j.get('inputs', [])}
    assert audit_path.resolve() not in immutable, 'Do not modify a frozen queue input'
    audit = json.loads(audit_path.read_text())
    prior = audit.get('reviewer_addendum_prompt2', {})
    if not prior:
        raw = audit_path.read_bytes()
        before = out/'sources'/('protocol_audit_before_prompt2_' + digest(raw)[:12] + '.json')
        before.write_bytes(raw)
        parent = dict(path=str(before),sha256=digest(raw))
    else:
        parent = prior['preserved_base_audit']
    audit['reviewer_addendum_prompt2'] = dict(at_local=at, preserved_base_audit=parent,
        report=str(out/'prompt2_report.html'), records=records,
        active_manifest=pointer['active_manifest'], evaluation_matrix=str(out/'evaluation_matrix.csv'),
        no_added_training=True, tracker_policy='Existing state/manifest is authoritative; report matrix is an audit view. No second worker.',
        pending_writing=evidence.get('prompt2_loss_deployment_evidence', {}).get('later_writing_needs', []))
    temporary = audit_path.with_suffix('.tmp')
    dump(temporary,audit)
    temporary.replace(audit_path)
    dump(out/'reviewer_claim_ledger.json',audit['reviewer_addendum_prompt2'])
    csv_out(out/'reviewer_claim_ledger.csv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records])
    return records


def ratio(a, b):
    return a / b if b else None


def metrics(tp, fp, fn, tn):
    water = ratio(tp, tp + fp + fn)
    background = ratio(tn, tn + fp + fn)
    defined = [v for v in (water, background) if v is not None]
    return dict(water_IoU=water, mIoU=statistics.mean(defined) if defined else None,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                false_positive_pixel_rate=ratio(fp, fp + tn))


def counts(cell):
    assert len(cell) == 2 and all(len(row) == 3 for row in cell)
    assert all(math.isfinite(n) and n >= 0 and int(n) == n for row in cell for n in row)
    assert cell[0][1] == cell[1][2] and cell[0][2] == cell[1][1]
    return tuple(int(n) for n in (cell[1][0], cell[1][1], cell[1][2], cell[0][0]))


def section(sections, title, paragraphs=(), headers=(), rows=()):
    sections.append(dict(title=title, paragraphs=list(paragraphs), headers=list(headers), rows=list(rows)))


def render(out, content):
    esc = lambda x: html.escape(str(x))
    body = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>' + esc(content['title']) + '</title>',
            '<style>body{font:15px/1.5 system-ui,sans-serif;color:#182b3a;max-width:1200px;margin:40px auto;padding:0 24px}'
            'h1{font-size:30px}h2{margin-top:35px;font-size:22px}h3{font-size:18px}p{max-width:105ch}'
            'table{border-collapse:collapse;width:100%;font-size:12px;margin:20px 0}th,td{padding:8px;text-align:left;vertical-align:top;border-bottom:1px solid #ccd7de;overflow-wrap:anywhere}'
            'th{background:#edf3f6}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}'
            '.meta{color:#526a7a}a{color:#075b88}@media print{body{margin:0;font-size:10pt}h2,h3{break-after:avoid}tr{break-inside:avoid}table{font-size:8pt}}</style><body>',
            '<h1>' + esc(content['title']) + '</h1>', '<p class="meta">' + esc(content['at_local']) + '</p>']
    for s in content['sections']:
        body.append('<h2>' + esc(s['title']) + '</h2>')
        body += ['<p>' + esc(p) + '</p>' for p in s['paragraphs']]
        if s['headers']:
            body.append('<table><thead><tr>' + ''.join('<th>' + esc(x) + '</th>' for x in s['headers']) + '</tr></thead><tbody>')
            body += ['<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in row) + '</tr>' for row in s['rows']]
            body.append('</tbody></table>')
    body.append('<h2>Evidence records and reviewer requirements</h2>')
    for name, evidence in content['audit_evidence'].items():
        body.append('<h3>' + esc(name.replace('_', ' ')) + '</h3>')
        # Keep all findings, qualifications, exact paths and fingerprints reviewable.
        if isinstance(evidence, dict):
            for key, value in evidence.items():
                body.append('<h4>' + esc(key.replace('_', ' ')) + '</h4>')
                body.append('<pre>' + esc(json.dumps(value, indent=2, ensure_ascii=False)) + '</pre>')
        else:
            body.append('<pre>' + esc(json.dumps(evidence, indent=2, ensure_ascii=False)) + '</pre>')
    body.append('<h2>Downloadable evidence</h2><ul>')
    body += ['<li><a href="' + esc(p.name) + '">' + esc(p.name) + '</a></li>' for p in sorted(out.iterdir()) if p.suffix in {'.csv', '.json', '.pdf'}]
    body.append('</ul>')
    source_links = [
        ('Prompt 1 base experiment instructions', '../../../prompts/prompt1.md'),
        ('Prompt 2 reviewer addendum', '../../../prompts/prompt2.md'),
        ('Available manuscript source text', '../prompt2_sources/main_v3.txt'),
        ('Available supplement source text', '../prompt2_sources/supplement_v3.txt'),
        ('Pinned official RSE repository', 'https://github.com/kipoju/udl4fl/tree/8f56cc7970ac9e561bfcd04a9ae0aa8e58c1c709'),
        ('RSE article DOI', 'https://doi.org/10.1016/j.rse.2025.114882'),
        ('Official LightFormer source', 'https://github.com/Chen-XiaoLv/LightFormer/tree/574577dd639403419f511e7bab15e82a87768672'),
        ('LightFormer article DOI', 'https://doi.org/10.1016/j.jag.2026.105331')]
    body.append('<h2>Source anchors</h2><ul>' + ''.join('<li><a href="' + esc(url) + '">' + esc(title) + '</a></li>' for title,url in source_links) + '</ul>')
    body.append('</body></html>')
    (out / 'prompt2_report.html').write_text('\n'.join(body))


def render_pdf(out, content):
    """A printable report with wrapped evidence cards, using the existing plotting stack."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out / 'prompt2_report.pdf', metadata={'Title': content['title']}) as pdf:
        fig, ax, page, y = None, None, 0, 0

        def new_page():
            nonlocal fig, ax, page, y
            if fig is not None:
                ax.text(.94, .035, str(page), fontsize=8, ha='right')
                pdf.savefig(fig)
                plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            page += 1
            y = .91
            ax.text(.075, .96, content['title'], fontsize=10, color='#28516a')
            ax.text(.075, .941, content['at_local'], fontsize=8, color='#526a7a')

        def lines(value, size=9, heading=False):
            nonlocal y
            if heading and y < .16:
                new_page()
            for line in textwrap.wrap(str(value), width=97 if size == 9 else 79, break_long_words=True, break_on_hyphens=False) or ['']:
                if y < .075:
                    new_page()
                ax.text(.075, y, line, fontsize=size, va='top', color='#172a39')
                y -= .0158 if size == 9 else .021
            y -= .009

        new_page()
        for s in content['sections']:
            lines(s['title'], 12, heading=True)
            for paragraph in s['paragraphs']:
                lines(paragraph)
            # Evidence cards keep wide tables readable in print.
            for row in s['rows']:
                lines('; '.join(f'{k}: {v}' for k, v in zip(s['headers'], row)))
        lines('Claim evidence appendix', 12, heading=True)
        lines('Each record below summarizes the claim, finding, status and remedy. Complete code/data/checkpoint identities, line numbers and SHA256 hashes are retained in reviewer_claim_ledger.json, its CSV view and the HTML evidence appendix.')
        for r in content.get('ledger_records', []):
            lines(r['id'] + '  ' + ', '.join(r.get('reviewer_ids', [])), 12, heading=True)
            lines('Location: ' + str(r.get('paper_location')))
            lines('Claim: ' + str(r.get('claim')))
            finding = r.get('summary', r.get('finding'))
            if isinstance(finding, (dict,list)):
                finding = json.dumps(finding, ensure_ascii=False)
            lines('Finding: ' + str(finding))
            lines('Evidence classification: ' + str(r.get('classification')) + '; severity: ' + str(r.get('severity')) + '; status: ' + str(r.get('status')))
            lines('Affected results: ' + str(r.get('affected_results')))
            lines('Smallest remedy: ' + str(r.get('smallest_remedy',r.get('remedy'))))
            refs = r.get('code_references', [])
            if isinstance(refs,list):
                concise = [str(ref.get('path')) + ':' + str(ref.get('line_start',ref.get('lines',''))) for ref in refs[:3] if isinstance(ref,dict)]
                if concise:
                    lines('Code references: ' + '; '.join(concise))
        ax.text(.94, .035, str(page), fontsize=8, ha='right')
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO / 'experiments/revision/prompt2_report')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = out / 'sources'
    sources.mkdir(exist_ok=True)
    pointer = json.loads((ROOT / 'queue/active_queue.json').read_text())
    manifest_path, state_path = map(Path, (pointer['active_manifest'], pointer['active_state']))
    manifest, state = [json.loads(p.read_text()) for p in (manifest_path, state_path)]
    dump(sources / 'manifest.json', manifest)
    dump(sources / 'state.json', state)
    at = datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds')
    source_records, scene_rows, summaries, matrix = [], [], [], []
    reference_path = REPO / 'experiments/revision/prompt2_reference_manifest.json'
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    reference_sha = digest(reference_path.read_bytes()) if reference_path.exists() else None
    reference_groups = {}
    for ds, namespace in (('s11','S11'), ('sturm','STURM'), ('wf','WorldFloods')):
        for item in reference.get(namespace, {}).values():
            ref_key = item['map_id'] if ds == 'sturm' else item.get('filename', item['id'])
            reference_groups.setdefault((ds, ref_key), []).append(item)
    if reference:
        (sources / reference_path.name).write_bytes(reference_path.read_bytes())
    paths = {Path(o['path']) for j in manifest['jobs'] for o in j.get('outputs', []) if '/evaluations/' in o['path'] and Path(o['path']).exists()}
    # Prepared rule resumes reuse the old cells; select one source per rule combination.
    old_rules = {ROOT / f'evaluations/P0_rule_{i}.json' for i in ('mndwi', 'ndwi')}
    paths |= old_rules
    paths = {p for p in paths if not p.name.startswith('R_comparator_index_')}
    for path in sorted(paths):
        raw = path.read_bytes()
        data = json.loads(raw)
        if 'cells' not in data:
            continue
        key = digest(raw)
        (sources / (path.stem + '_' + key[:10] + '.json')).write_bytes(raw)
        spec, identity = data['spec'], data['identity']
        fp = identity.get('fingerprints', spec.get('fingerprints', {}))
        source_records.append(dict(path=str(path), sha256=key, status=data['status'], identity=identity, spec=spec))
        protocol_id = digest(json.dumps({'code': identity.get('source_sha256'), 'roots': spec.get('roots'), 'protocol': fp.get('protocol')}, sort_keys=True).encode())
        for cell_key, cell in data['cells'].items():
            ds, condition = cell_key.split('__', 1)
            current = []
            for scene_id, cell_counts in cell['scene'].items():
                tp, fpositive, fn, tn = counts(cell_counts)
                row = dict(source=str(path), source_sha256=key, run_id=spec['run_id'],
                           model=spec['arch'], family=spec.get('family'), cohort=spec.get('cohort'),
                           checkpoint=spec.get('checkpoint'), checkpoint_sha256=identity.get('checkpoint_sha256'),
                           seed=spec.get('seed'), dataset=ds, split='canonical_test90' if ds == 's11' else 'existing_external_protocol',
                           condition=condition, image_or_map_id=scene_id,
                           event_id=scene_id.split('_')[0] if ds == 's11' else None,
                           grouping='chip' if ds == 's11' else 'existing map grouping; event counts retained in source',
                           TP=tp, FP=fpositive, FN=fn, TN=tn, valid_pixels=tp+fpositive+fn+tn,
                           reference_water_pixels=tp+fn, protocol_identity=protocol_id,
                           mask_identity='reader/config source hashes and dataset fingerprint in source provenance',
                           mask_bitmap_sha256=None, mask_identity_status='historical bitmap digest not exported; do not infer from counts',
                           **metrics(tp, fpositive, fn, tn))
                row.update(reference_empty=(tp+fn == 0),
                    legacy_zero_union_water_IoU=tp/max(tp+fpositive+fn,1),
                    legacy_zero_union_mIoU=(tp/max(tp+fpositive+fn,1)+tn/max(tn+fpositive+fn,1))/2)
                refs = reference_groups.get((ds, scene_id), [])
                if refs:
                    native_identity = [{k: item.get(k) for k in ('id', 'reference', 'mapped_native_sha256', 'jrc_mask')} for item in refs]
                    row.update(event_id=refs[0]['event'],
                               event_identity_status='country prefix proxy; exact event grouping unverified' if ds == 's11' else 'verified metadata event identifier',
                               reference_manifest_sha256=reference_sha,
                               reference_record_ids=';'.join(item['id'] for item in refs),
                               native_reference_identity_sha256=digest(json.dumps(native_identity,sort_keys=True).encode()),
                               mask_identity='native reference/group hashes + reader hashes in prompt2_reference_manifest.json; resized scoring bitmap hash absent')
                current.append(row)
                scene_rows.append(row)
            pooled = [sum(row[k] for row in current) for k in ('TP', 'FP', 'FN', 'TN')]
            computed = metrics(*pooled)
            for metric, saved in [('water_IoU', 'water_IoU'), ('mIoU', 'mIoU'), ('precision', 'water_precision'), ('recall', 'water_recall')]:
                original = cell['micro'][saved]
                assert (computed[metric] is None and original is None) or abs(computed[metric] - original) < 1e-12, (path, cell_key, metric)
            summary = dict(source=str(path), run_id=spec['run_id'], model=spec['arch'], family=spec.get('family'),
                           cohort=spec.get('cohort'), seed=spec.get('seed'), dataset=ds, condition=condition,
                           aggregation='micro: counts pooled within this checkpoint, dataset and condition',
                           n_images_or_maps=len(current), n_valid_units=sum(x['valid_pixels'] > 0 for x in current),
                           n_events=len(cell.get('event', {})), **dict(zip(('TP','FP','FN','TN'), pooled)), **computed)
            for metric in ('water_IoU','mIoU','precision','recall'):
                values = [x[metric] for x in current if x[metric] is not None]
                summary['macro_' + metric] = statistics.mean(values) if values else None
                summary['macro_' + metric + '_defined_units'] = len(values)
            legacy_units = [x for x in current if x['reference_water_pixels'] > 0]
            summary['legacy_reference_positive_units'] = len(legacy_units)
            for metric in ('water_IoU','mIoU'):
                values = [x['legacy_zero_union_' + metric] for x in legacy_units]
                summary['legacy_reference_positive_macro_' + metric] = statistics.mean(values) if values else None
            summaries.append(summary)
    matched_reference = {}
    for row in scene_rows:
        group = (row['dataset'], row['image_or_map_id'])
        current_reference = (row['valid_pixels'], row['reference_water_pixels'])
        if group in matched_reference:
            assert matched_reference[group] == current_reference, ('inconsistent valid/reference counts', group, row['source'], row['condition'])
        matched_reference[group] = current_reference
    dump(out / 'count_validation.json', dict(status='passed', count_rows=len(scene_rows),
        matched_reference_units=len(matched_reference), integer_nonnegative_counts=True,
        binary_FP_FN_symmetry=True, saved_metrics_max_absolute_tolerance=1e-12,
        retained_and_reference_pixels_identical_across_available_models_and_perturbations=True,
        undefined_metrics='null when denominator is zero; macro means include only defined scores and report contributing counts'))
    csv_out(out / 'confusion_counts.csv', scene_rows)
    csv_out(out / 'metric_summaries.csv', summaries)
    # Map the exact requested finite cohort to the existing queue; preserve missing seeds.
    for job in manifest['jobs']:
        jid = job['id']
        if not any(x in jid for x in ('archive_clean', 'R_eval_', 'R_comparator_', 'independent_dry')):
            continue
        outputs = [o['path'] for o in job.get('outputs', [])]
        present = [p for p in outputs if Path(p).exists()]
        cells = next((json.loads(Path(p).read_text()).get('cells', {}) for p in present if p.endswith('.json')), {})
        for ds in (('independent_dry' ,) if 'independent_dry' in jid else ('s11', 'sturm', 'wf')):
            measured = cells.get(ds+'__clean', {})
            status = 'verified-complete' if measured else 'blocked' if state['jobs'][jid]['status'] == 'blocked' else 'needs-inference'
            matrix.append(dict(task=jid, dataset=ds, queue_status=state['jobs'][jid]['status'], status=status,
                               output=outputs[0] if outputs else None, confusion_counts_available=bool(measured),
                               dependencies=';'.join(job.get('deps', []))))
    csv_out(out / 'evaluation_matrix.csv', matrix)
    training = []
    durations = defaultdict(list)
    for job in manifest['jobs']:
        if not job['id'].startswith('R_train_'):
            continue
        record = state['jobs'][job['id']]
        verified_path = next(Path(o['path']) for o in job['outputs'] if o['path'].endswith('/verified.json'))
        v = json.loads(verified_path.read_text()) if verified_path.exists() else {}
        r = v.get('results', {})
        training.append(dict(run_id=job['id'], status=record['status'], cohort=pointer['cohort'],
                             validation_mIoU=r.get('best_val_mIoU'), s11_test_mIoU=r.get('test_mIoU'),
                             best_checkpoint_sha256=v.get('checkpoint_checks', {}).get('best', {}).get('sha256'),
                             elapsed_seconds=v.get('elapsed_seconds'), checkpoint_selection='frozen S11 validation selected; test is readout'))
        if record['status'] == 'succeeded' and v.get('elapsed_seconds'):
            durations[job['id'].rsplit('_s',1)[0].replace('_N2500','')].append(v['elapsed_seconds'])
    csv_out(out / 'training_status.csv', training)
    estimates = []
    for row in training:
        if row['status'] not in {'pending', 'running'}:
            continue
        group = row['run_id'].rsplit('_s',1)[0].replace('_N2500','')
        vals = durations.get(group, [])
        if vals:
            estimates.append(dict(run_id=row['run_id'], measured_n=len(vals), estimated_seconds=statistics.mean(vals), min_observed=min(vals), max_observed=max(vals)))
    # Full durations for running jobs give a conservative estimate; admission waits remain unknown.
    estimate = dict(training_gpu_hours=sum(x['estimated_seconds'] for x in estimates)/3600,
                    ideal_two_gpu_wall_hours=sum(x['estimated_seconds'] for x in estimates)/7200,
                    basis=estimates, limitation='Training only; running jobs counted at full duration. Heavy evaluator timings and foreign workload/admission waits remain unknown; not a completion-time promise.')
    dump(out / 'time_estimate.json', estimate)
    evidence = {}
    for p in sorted((REPO/'experiments/revision').glob('prompt2*evidence.json')):
        raw = p.read_bytes()
        evidence[p.stem] = json.loads(raw)
        (sources / p.name).write_bytes(raw)
    for name in ('prompt2_mndwi_summary.json','prompt2_rule_reuse.json'):
        p = REPO/'experiments/revision'/name
        if p.exists():
            evidence[p.stem] = json.loads(p.read_text())
            (sources/p.name).write_bytes(p.read_bytes())
    sections = []
    queue_counts = Counter(x['status'] for x in state['jobs'].values())
    train_counts = Counter(x['status'] for x in training)
    section(sections, 'Execution scope and current outcome', [
        'This report executes the reviewer-only addendum in prompts/prompt2.md and covers its demands with measured results, evidence findings, linked queued inference or an explicit blocker. The direct user request authorizes this report despite the prompt files restricting narrative documentation. The manuscript and reviewer response have not been edited.',
        f'Evidence captured {at}. Repository {REPO}, branch research. Existing dirty work and historical artifacts are preserved. No training, fine tuning, QAT, architecture, dataset acquisition or hyperparameter search was added through prompt2.',
        f'Existing queue entries: {dict(queue_counts)}. Existing 36-run training matrix: {dict(train_counts)}. These training jobs were authorized earlier; their completion is not counted as new addendum training.',
        'Archived/current-evaluator results and regenerated-pool current-environment controls remain distinct. The active pointer records prior user authorization for 127 dataset replacements and scheduling across both GPUs. This addendum launched no independent GPU job and did not change that scheduler.'
    ])
    section(sections, 'Requirement coverage and deduplication', [
        'Evaluation_matrix.csv maps the exact existing inference jobs and their dependencies to the finite reviewer comparison. Compatible completed counts were exported; missing outputs remain needs-inference. The frozen teacher, remaining archived OFA seeds, all architecture controls and all existing robustness readouts are already in the same queue. No second worker or added training configuration is required.',
        'Each evidence record below includes the finding, source boundaries, severity/remedy and status. Verified limitations prepare later writing; pending evaluations do not resolve a reviewer concern. The later manuscript and response pass remains outstanding.',
        'The two fixed-rule queued comparisons reused 110 compatible completed dataset-condition cells after reader/index/mask/perturbation code and package identity checks. Only the omitted WorldFloods g0.90 cell per rule remains for those existing jobs. Original outputs and full reuse provenance are preserved.'
    ], ['Demand', 'Evidence or output', 'Status'], [
        ['§1 finite comparison, precision/recall', 'evaluation_matrix.csv, confusion_counts.csv, metric_summaries.csv', 'Available counts exported; remaining inference already queued'],
        ['§2A split, lineage, overlap and history', 'prompt2_data_evidence.json and appended protocol audit', 'Verified findings and bounded unknowns'],
        ['§2B preprocessing, masks, metric semantics', 'prompt2_data_evidence.json; count arithmetic validation', 'Current code verified; historical execution qualified'],
        ['§2C losses, fairness and component contrasts', 'prompt2_loss_deployment_evidence.json', 'Existing losses audited; no objective silently corrected'],
        ['§2D deployment and dry scenes', 'Deployment evidence and dry_diagnostics.csv', 'Existing execution evidence; independent dry references blocked'],
        ['§3 S11 MNDWI discrepancy', 'prompt2_mndwi_summary.json and agent count/paired exports', 'Bounded diagnosis; official evaluator limits retained'],
        ['§4 novelty and excluded experiments', 'Novelty records in loss/deployment evidence', 'Existing analyses inspected; no few-shot/XAI work started'],
        ['§5 priorities and later writing', 'Reviewer ledger and pending writing actions', 'Evidence recorded; manuscript/response pending'],
        ['§6 ledger, exports, tracker and handoff', 'Report, source snapshots, appended audit and active queue pointer', 'Operational additions produced; queued work not claimed complete']
    ])
    findings = []
    for name, data in evidence.items():
        if isinstance(data, dict):
            for item in data.get('key_findings', []):
                findings.append(str(item) if not isinstance(item, dict) else json.dumps(item, ensure_ascii=False))
    if findings:
        section(sections, 'Verified findings and claim limits', findings)
    md = evidence.get('prompt2_mndwi_summary', {})
    if md:
        rows = md.get('aggregation_table', [])
        section(sections, 'Sen1Floods11 MNDWI protocol comparison', [
            'The fixed rule uses floating-point B3 and B11 at threshold strictly above zero on the frozen reader. Canonical90 counts were reused, missing chips were processed once, and raw-radiometry/clamp/spatial diagnostics were declared before scoring. All446 is a fixed-rule protocol diagnostic including training, validation and Bolivia data, rather than an independent learned-model benchmark.',
            'The exact external-paper gap is unresolved. Expanding the population lowers the measured pooled score; per-chip averaging lowers it further. Raw float32 before the normalization roundtrip changes 343 valid canonical decisions and increases water IoU by only 0.00527 percentage points. Literal clamp removal and the specified spatial/tiling checks change no valid decisions. These checks do not establish a material implementation deficit.',
            'The pinned official source supplies inference code but no MNDWI benchmark evaluator. The official floodmap/event grouping, validity/cloud/nodata/GSW choices and exact input product preprocessing remain unknown. Country-prefix averages remain proxies; the external paper reported 74.39% water IoU, not two-class mIoU. B8/B8A does not enter MNDWI.',
            'The full declared variants, per-chip confusion counts, paired decision-disagreement counts and exact source/input identities are linked in the evidence below. The original archived score is matched to displayed precision; original historical executed source/prediction provenance remains reconstructed.'
        ], ['Population','Chips','Pooled water IoU','Per-chip mean water IoU','Defined chips','Undefined chips','Country-prefix proxy mean','Official RSE map grouping'], [
            [x['population'],int(x['n_chips']),f"{x['water_IoU']:.6f}",f"{x['per_chip_mean_water_IoU']:.6f}",int(x['n_defined_chip_water_IoU']),int(x['n_undefined_chip_water_IoU']),f"{x['location_proxy_macro_water_IoU']:.6f}",'unavailable'] for x in rows])
        section(sections, 'MNDWI paired decision checks', [
            'The following paired summaries compare the necessary declared variants to the identical baseline references. Zero means no changed decisions in that population; it is not a general claim about unavailable input products. The complete paired counts remain in prompt2_mndwi_pairs.csv.'
        ], ['Population','Variant','Valid pixels','Changed valid pixels','Changed ignored pixels','Background to water','Water to background'], [
            [x['population'],x['variant'],x['valid_pixels'],x['changed_valid'],x['changed_ignored'],x['reference0_variant1_valid'],x['reference1_variant0_valid']] for x in md.get('paired_summary', [])])
        for name in ('prompt2_mndwi_comparison.csv','prompt2_mndwi_counts.csv','prompt2_mndwi_pairs.csv','prompt2_mndwi_event_counts.csv'):
            p = REPO/'experiments/revision'/name
            if p.exists():
                (out/p.name).write_bytes(p.read_bytes())
    ld = evidence.get('prompt2_loss_deployment_evidence', {})
    if ld:
        pairs = ld.get('component_pair_ledger', [])
        section(sections, 'Component comparisons and causal limits', [
            'Each component contrast identifies both run sets and every changed factor. The hard-only versus KD contrast changes the hard loss from CE to focal and scales it by one half; it is not a clean one-factor test of soft targets. Different augmentation, initialization, geometry, budgets and evaluation paths prevent chaining unrelated ablations into an additive contribution total.'
        ], ['Component','Left runs','Right runs','Changed factors','Interpretation'], [
            [x['component'],x['left_runs'],x['right_runs'],'; '.join(x['changed_factors']),x.get('one_factor_claim')] for x in pairs])
        budgets = ld.get('OFA_vs_LogitKD_all_budget_pairs', [])
        section(sections, 'Existing OFA and Logit KD budget evidence', [
            'The following table recovers every existing budget from the paper-source cohort. Means and sample standard deviations describe four training seeds; paired deltas use corresponding seeds. No sweep, statistical search or favorable-seed exclusion was added. Exact historical checkpoint/data linkage remains qualified in the ledger.',
            'OFA has benefits at some limited/intermediate budgets and lacks consistent N2500 superiority. A fixed epoch budget changes optimizer-step counts with N, so sample-efficiency interpretation must include those counts. Null results do not show a feasibility failure or equivalence.'
        ], ['N','Metric','Seeds','Nominal optimizer steps','KD mean','KD SD','OFA mean','OFA SD','Paired OFA minus KD','Paired delta SD','OFA wins'], [
            [x['N'],x['metric'],x['n_seeds'],x['nominal_steps'],f"{x['mean']['logit_kd']:.6f}",f"{x['sample_sd_ddof1']['logit_kd']:.6f}",f"{x['mean']['ofa']:.6f}",f"{x['sample_sd_ddof1']['ofa']:.6f}",f"{x['paired_delta_mean']:+.6f}",f"{x['paired_delta_sample_sd']:.6f}",x['ofa_wins']] for x in budgets])
        geometry = ld.get('geometry_control', {}).get('summary', {})
        section(sections, 'Cropped S11 geometry evidence', [
            'The existing center264 crop changes water content as well as physical field of view. It does not isolate label quality or geography. The CPU label audit measures the lost-water/dry-chip effects without retraining or replacing the geometry experiment.'
        ], ['Quantity','Measured value'], list(geometry.items()))
        deployment = ld.get('deployment', {})
        table = deployment.get('existing_tableIII_rows', [])
        section(sections, 'Deployment accuracy and practical claims', [
            'The original activation, ReLU6, QAT, float export and mixed INT8/FP16 stages remain separate artifacts and execution paths. Accuracy from an undeployed checkpoint must not be attached to a different engine. The table below reaggregates existing stage evidence; the source ledger records checkpoint ancestry and missing raw device associations.',
            'QAT fixes quantization scale buffers in the inspected implementation and freezes BatchNorm while continuing to train other model weights. The claim that only INT8 scales adapt is inconsistent with the optimizer. Engine file size is distinct from runtime memory, kernel timing from end-to-end latency, and peak power from energy per image.',
            'Existing Jetson measurements have FP16 faster than the mixed engine. The teacher was not benchmarked on that device, so a measured teacher/student device speedup is unavailable. No RTX float inference, engine rebuild or new hardware benchmark was substituted.'
        ], ['Stage','Labels','Evaluator','Benchmark','mIoU mean','Sample SD','Seeds'], [[x['stage'],x['labels'],x['evaluator'],x['benchmark'],x['mean_micro_mIoU'],x['sd_ddof1'],x['n_seeds']] for x in table])
        section(sections, 'Existing Jetson timing evidence', [
            'These are archived measurements on the reported Jetson Xavier NX execution path. Timing and representative-engine identity remain subject to the ledger provenance limits; this addendum did not benchmark the current workstation.'
        ], ['Quantity','FP32','FP16','Mixed INT8 QAT'], [[x['metric'],x['FP32'],x['FP16'],x['INT8-QAT']] for x in deployment.get('existing_tableIV_rows', [])])
    clean = [x for x in summaries if x['condition'] == 'clean' and (x['model'] != 'index:ndwi_zero')]
    fmt = lambda x: 'undefined' if x is None else f'{x:.6f}'
    section(sections, 'Primary clean counts and error metrics', [
        'All rows use the fixed existing decision and pixel-pooled counts per checkpoint. Precision and recall are calculated from TP, FP and FN, rather than inferred from IoU. Individual seed predictions are never pooled into an ensemble. Tables include every available valid seed; missing seeds remain in evaluation_matrix.csv.',
        'The count exporter checks nonnegative integer counts, symmetric binary-class FP/FN identities and agreement with all saved micro metrics to 1e-12. Macro metrics average defined image/map scores and disclose contributing units. Companion legacy_reference_positive_macro fields preserve stats_scene_level.py filtering to reference-water-positive units and its zero-union class convention. These conventions must not be mixed. S11 country-prefix event groups are coarse; no independence or equivalence claim is made.',
        'Historical artifacts did not store valid-mask bitmap hashes. The export records exact source/output/checkpoint hashes, dataset/protocol identities and that limitation. Current source inspection cannot establish unknown historical training choices.'
    ], ['Run', 'Dataset', 'TP', 'FP', 'FN', 'TN', 'Water IoU', 'mIoU', 'Precision', 'Recall'], [
        [x['run_id'],x['dataset'],x['TP'],x['FP'],x['FN'],x['TN'],fmt(x['water_IoU']),fmt(x['mIoU']),fmt(x['precision']),fmt(x['recall'])] for x in clean])
    dry = []
    drypaths = list(ROOT.glob('P4_dry_diagnostic_*.json')) + [REPO/'runs/revision_20261005/dry_mndwi.json']
    for p in drypaths:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for x in d['scenes']:
            dry.append(dict(comparator=p.stem, **x, independent_reference_status='Human-empty chip diagnostic; event groups overlap train/validation; not independent held-out dry evaluation'))
    csv_out(out/'dry_diagnostics.csv', dry)
    section(sections, 'Dry scenes and robustness limits', [
        'Existing dry diagnostics cover six S11 test chips in four groups with human-labelled empty water masks. They permit interpretable false-water counts and prespecified any-water, 0.1% and 1% area alarms. Event groups overlap training/validation, so independent operational dry-scene generalization remains blocked by independent_dry_reference_gate. No negative images were added to training and the 5% filter was unchanged.',
        'All available clean and corrupted fixed-rule counts and metrics are exported in metric_summaries.csv. The existing gain_numerics.json investigates epsilon/clamping exceptions to mathematically invariant shared gain. The report retains all severities, moderate and extreme conditions, nulls and rule wins. Synthetic haze supports sensitivity to that defined perturbation only.',
        'Matched learned-model robustness is already queued and remains pending where counts are absent. Teacher advantage must not be called student advantage. Float RTX inference does not represent the Jetson mixed engine. Existing source JSON retains map/event counts for paired analyses; training-seed variability and checkpoint-conditional event uncertainty remain distinct.'
    ])
    section(sections, 'Existing training and architecture controls', [
        'The training table records current-environment regenerated N2500 controls separately from archived paper results. Validation selects checkpoints; S11 test is a fixed readout. Functional/budget verification does not prove that original-data training was reproduced. All architecture objectives and seeds remain in the declared matrix, including unfavorable runs.',
        'LightFormer native auxiliary supervision is uniformly disabled in this common-recipe adaptation; its unused native parameters are counted separately from main inference and OFA heads. This is not the published best-tuned LightFormer recipe. Loss differences, initialization, clipping and valid-pixel semantics are documented in the claim ledger rather than attributed wholesale to OFA.'
    ], ['Run','Status','Validation mIoU','S11 test mIoU'], [[x['run_id'],x['status'],fmt(x['validation_mIoU']),fmt(x['s11_test_mIoU'])] for x in training])
    wrapper, manifest_str, state_str = pointer['active_wrapper'], str(manifest_path), str(state_path)
    prefix = f'/home/schmalstieg/miniconda3/envs/esi/bin/python {wrapper}'
    commands = {a:f'{prefix} {a} --manifest {manifest_str} --state {state_str}' for a in ('status','pause','resume','stop')}
    section(sections, 'Source availability and evidence boundaries', [
        'The project-equivalent manuscript and supplement were found in OLD/FloodDistill_v3/docs, extracted and traced by file hashes and physical pages. The actual reviewer letter and supplied RSE/LightFormer article PDFs were not found in the inspected project/home paths. Reviewer IDs and task scope therefore come from prompt2 rather than a verified copy of the review letter.',
        'A bounded attempt to open the two article DOI sources could not retrieve full text. Official pinned code was inspected where available; repository inspection is not an exact reproduction of an unavailable article evaluator. The teacher modelcard and release configuration describe task semantics, but they do not establish executed member IDs or historical checkpoint-selection candidates.',
        'Raw GDACS native footprints are unavailable, and one selected centroid is missing from the proximity audit. No direct overlap found by the documented checks is not a proof of complete independence. Historical benchmark-guided method choices, calibration batch membership and some deployed artifact ancestry remain bounded unknowns in the ledger.'
    ])
    section(sections, 'Operational handoff and remaining work', [
        f'Worker identity at capture: {state.get("worker")}; status {state.get("worker_status")}. The existing detached Linux session uses durable state, process identities and locks. The two-job sentinel verifies parent-process independence; actual editor-close/logout tests are not claimed. Reboot is unsupported, and no jobs run while the machine sleeps or is powered off.',
        f'Measured remaining training workload is approximately {estimate["training_gpu_hours"]:.2f} GPU-hours, or {estimate["ideal_two_gpu_wall_hours"]:.2f} hours at ideal two-GPU scheduling. This counts active jobs at full duration and excludes pending inference, admission waits and other users workload; total queue completion time remains unknown.',
        'Pause prevents new launches and lets current jobs finish. Resume uses the existing worker/state reconciliation. Stop gracefully stops the worker after active jobs; it does not cancel training. Missing independent dry references, historical choice/calibration history and the official RSE evaluator remain scientific limitations where the ledger identifies them.',
        'Later writing must consolidate component contrasts, actual data roles, surface-water terminology, limited novelty/OFA gains, geometry confounds, hardware precision/timing limits and the Discussion, with reviewer IDs. No manuscript edits or reviewer-response draft are included.'
    ], ['Control','Exact command'], [[a,c] for a,c in commands.items()])
    provenance = dict(at_local=at, git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                      branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
                      manifest=str(manifest_path), manifest_sha256=digest(manifest_path.read_bytes()),
                      frozen_state_sha256=digest((sources/'state.json').read_bytes()), sources=source_records,
                      count_rows=len(scene_rows), metric_rows=len(summaries), status='Report and CPU exports completed; finite inference queue and independent dry-reference gap remain',
                      no_added_training=True, queue_mutations='No source, manifest or state edits; pending fixed-rule outputs contain verified reusable cells with lineage')
    dump(out/'provenance.json',provenance)
    content = dict(title='Flood distillation reviewer evidence report', at_local=at, sections=sections, audit_evidence=evidence)
    records = extend_ledger(out,evidence,manifest,pointer,at)
    content['reviewer_claim_records'] = len(records)
    content['ledger_records'] = records
    dump(out/'report_content.json',content)
    render(out,content)
    render_pdf(out,content)
    dump(REPO/'experiments/revision/prompt2_execution.json', dict(report=str(out/'prompt2_report.html'),
        provenance=str(out/'provenance.json'), queue_manifest=str(manifest_path), queue_counts=dict(queue_counts),
        training_counts=dict(train_counts), evaluation_matrix=str(out/'evaluation_matrix.csv'), at_local=at,
        statuses={'audits':'evidence recorded','counts':'available artifacts exported','remaining_inference':'existing queue','independent_dry':'blocked'}))
    print(json.dumps(dict(report=str(out/'prompt2_report.html'), count_rows=len(scene_rows), metric_rows=len(summaries), audits=list(evidence), training=dict(train_counts))))


if __name__ == '__main__':
    main()
