"""Predeclare retained original runs and strictly inspect their main checkpoints."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.revision_queue import atomic_json, file_hash, now


def main():
    import torch
    from models.student import build_student
    torch.set_num_threads(2)
    root = REPO / 'runs/original_runs'
    selected = []
    # Availability/claim order is fixed before opening any score values.
    for arch, arms, budget in [('efficientvit_b0', ['ce', 'dfkd', 'ofa'], 2500),
                               ('efficientvit_b0', ['dfkd', 'ofa'], 100),
                               ('unet_resnet50', ['ce_unet', 'dfkd_unet'], 2500)]:
        for arm in arms:
            for seed in [42, 64, 91, 107]:
                directory = root / 'V2/pseudo_v2_augmented' / f'pseudo_{arm}_aug_N{budget}_s{seed}'
                selected.append({'original_id': f'O4_original_{arch}_{arm}_N{budget}_s{seed}',
                                 'architecture': arch, 'declared_objective': arm, 'budget': budget,
                                 'seed': seed, 'checkpoint': str(directory / 'best.pt'),
                                 'archive_metrics': str(directory / 'eval_results_STURM_ORIG.json'),
                                 'paper_association': 'Primary pseudo-labelled/architecture table' if budget == 2500 else 'Supplement small-budget KD/OFA curve',
                                 'association_status': 'Candidate association from supplied run tree and recovered v2 launcher; no embedded run config or historical code/data receipt'})
    predeclared = REPO / 'experiments/revision/prompt4_original_score_set.json'
    if predeclared.exists():
        assert json.loads(predeclared.read_text())['runs'] == selected, 'Original score set changed'
    else:
        atomic_json(predeclared, {'created_utc': now(), 'status': 'predeclared', 'runs': selected,
                                 'order': 'architectures/objectives as listed; seeds42,64,91,107; no score gate',
                                 'excluded': 'Exploratory budgets, noaug/curation/transfer sweeps and new null experiments'})
    allfiles = [p for p in root.rglob('*') if p.is_file()]
    inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'kind': p.suffix} for p in allfiles]
    models, rows = {}, []
    for row in selected:
        row = dict(row)
        path = Path(row['checkpoint'])
        if not path.is_file():
            row.update(status='missing-checkpoint', error=str(path))
            rows.append(row)
            continue
        try:
            archive = json.loads(Path(row['archive_metrics']).read_text()) if Path(row['archive_metrics']).exists() else {}
            assert archive.get('arch') == row['architecture'], 'Associated evaluator architecture differs'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            state = payload['model']
            assert all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values())
            arch = row['architecture']
            if arch not in models:
                models[arch] = build_student(arch, num_classes=2, pretrained=False).eval()
            model = models[arch]
            expected = model.state_dict()
            assert set(expected) == set(state), 'Main-model tensor key mismatch'
            assert all(expected[k].shape == state[k].shape for k in expected), 'Main-model tensor shape mismatch'
            model.load_state_dict(state, strict=True)
            row.update(status='inference-ready', checkpoint_sha256=file_hash(path),
                       checkpoint_keys=list(payload), main_state_tensors=len(state), finite_main_tensors=True,
                       strict_main_reload=True, selected_epoch_recorded=payload.get('epoch', -1) + 1,
                       config_present='config' in payload, auxiliary_state_present=bool(payload.get('loss_fn')),
                       optimizer_present='optimizer' in payload,
                       selection_identity='File named best with stored epoch/validation metrics; selection rule and full training history not proven',
                       archive_metrics_sha256=file_hash(Path(row['archive_metrics'])) if archive else None,
                       archived_architecture=archive.get('arch'), archived_count_domain='Saved scalar metrics only; predictions and masks/count receipts absent',
                       normalization='Recovered v2 original STURM/WorldFloods evaluator; independent new count-domain rescoring required',
                       historical_objective_and_seed_identity='Associated recovered launcher and directory declaration; checkpoint has no embedded objective/seed config')
            del payload, state
        except Exception as exc:
            row.update(status='blocked-identity', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        print(row['original_id'], row['status'], flush=True)
    originals = {'created_utc': now(), 'status': 'verified-main-checkpoints' if all(r['status'] == 'inference-ready' for r in rows) else 'partial',
                 'predeclared_set': str(predeclared), 'predeclared_sha256': file_hash(predeclared),
                 'runs': rows, 'status_counts': dict(Counter(r['status'] for r in rows)),
                 'copied_inventory': inventory, 'copied_file_count': len(inventory),
                 'missing_original_compact_controls': ['No copied MobileNetV3-Small or LightFormer run/config found in declared original tree; new matched original-data controls remain separately authorized'],
                 'training_reproduction': 'Inference-ready originals are not new-machine training replicates or resumable OFA training states',
                 'missing_records': ['Executed per-run configs/commands with teacher version and source/environment identity',
                                     'Final/selected checkpoint receipts and training histories, including original OFA heads',
                                     'Original prediction/count/scoring-mask identities',
                                     'Original four-seed deployment artifact chains and target workflow'],
                 'source_sha256': {p: file_hash(REPO / p) for p in ['models/student.py', 'scripts/revision_original_checkpoints.py']},
                 'recipe_sources': ['/home/schmalstieg/repos/OLD/esi_project/experiments/v2_nsweep_pseudo_vs_gt.sh',
                                    '/home/schmalstieg/repos/OLD/esi_project/experiments/v2_nsweep_eval.sh']}
    atomic_json(REPO / 'experiments/revision/prompt4_original_checkpoint_inventory.json', originals)
    print(json.dumps({'status': originals['status'], 'run_statuses': originals['status_counts']}), flush=True)


if __name__ == '__main__':
    main()
