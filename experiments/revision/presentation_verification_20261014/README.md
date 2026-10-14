# Presentation verification, 14 Oct 2026

Claude's audit of the reviewer deck (`MajorRev/FloodDistill_reviewer_presentation.pptx` +
companion) against code, data and experiment outputs, following
`prompts/presentation_verification_claude.md` (Phase A audits + Phase B checkpoint).
Read-only: no training, no protocol changes, no edits to existing artifacts or to the deck.

| File | Content |
|---|---|
| `00_decision_report.md` | Verdict on the deck, Phase B decisions, proposed experiments P1–P7, approvals needed |
| `deck_corrections.md` | Slide-by-slide status and replacement wording |
| `reviewer_evidence_matrix.md` | Concern → claim → evidence → gap → action |
| `pipeline_ablation_plan.md` | R1.4 component map, activation/LR findings, minimal experiments |
| `dataset_protocol_audit.md` | Preprocessing, label semantics, teacher input contract, protocol options (approval) |
| `dry_scene_audit.md` | Dry-scene removal per stage, validity of false-alarm evidence, proposed dry set |
| `mndwi_evidence_audit.md` | Verified MNDWI comparisons, validity issues, conclusion table |
| `checks/`, `outputs/`, `run_all.sh` | Re-derivations (CPU, < 15 min; `SKIP_RADIOMETRY=1` skips the slowest) |

Tags: [V] re-derived here; [R] taken from existing reports/CSVs; [H] judgement.
