# Deck and companion: slide-by-slide verification

Scope: the slides that touch the five topics of `prompts/presentation_verification_claude.md`
(R1.4, dataset/label-source confounding, dry scenes, teacher→GDACS preprocessing, MNDWI).
Slides 5–9, 11, 16 and 18–19 were not re-audited, apart from numbers they share with the
slides below. Tags as in `00_decision_report.md`.

| Slide | Statement | Status | Evidence | Proposed text |
|---|---|---|---|---|
| 2 | Teacher fine-tuned on the 252 human labels; 2,500 pseudo-labelled tiles | ✔ [R] | HF `config.yaml` (S11 datamodule, RandomCrop 224, LR 5e-5); `diversity_N2500_index.json` | — |
| 10 | Matters: 10× scenes +2.6 (S11); soft targets +0.7 to +1.2; FPS vs random +0.2; pre-filter +0.1 / +0.3 / +4.0 / +31 | ✔ [V] | `reviewer_wrapup_20261012/figure_data/fig03, fig04`; `verification.json` | — |
| 10 | "ReLU6 'cost' vanishes with a validated learning rate (STURM 76.2 vs 74.8)" | ✘ [V] | `outputs/relu6_init_test.txt`, `outputs/relu6_vs_hs_paired.txt`, `OLD/esi_project/experiments/v2_repro_n2500.sh` | "Activation replacement, single factor (same pretrained init, LR, recipe; 4 seeds): STURM −0.8 / WF +0.8 pooled; per flood map −0.2 (p = 0.24) / +0.6 (p = 0.07). No consistent cost." |
| 10 notes | "The ReLU6 student was trained from scratch at the pretrained model's learning rate (5e-5)" | ✘ [V] | as above. The 76.2/74.8 pair is v7 A0 (scratch) at LR 5e-3 vs 5e-5 [V: `capacity_v7…/results.csv`] | "Separately: scratch-initialised B0 needs LR 5e-3 (+1.4 STURM, +2.2 S11-test mIoU vs 5e-5). That matters for the size ladder, not for the deployed student." |
| 10 / 4 | R1.4 status "Answered" | ✘ [V] | No QAT-vs-PTQ comparison for the deployed B0 exists (searched paper, supplement, prompt4/5, v4–v9) | "Mostly answered; QAT's isolated effect is being measured (P1)." |
| 10 | Figure row "Hardswish → ReLU6 (retrained from scratch), confounded, loses the ImageNet start" | ✘ [V] | as above | Mark as a filled, one-factor row |
| 12 | 68 % of the pool dropped (teacher < 5 %); 5,843 with none; no dry tile in training | ✔ [V] | `outputs/dry_funnel.txt` | Add: "Event tiles were already screened at download: each had to contain ≥ 1 % MNDWI new water, with up to 8 relocations; only pre-event baselines are ungated." |
| 12 | Pre-event 8,883 tiles, 1.9 ± 0.8 % (teacher-referenced) | ✔ [R] | wrap-up §5.6, `relabel_feasibility.py` | — |
| 12 / 15 | WF dry windows: MNDWI 136, teacher 5, paper student 8–16, 12k student 8–10 (of 206) | ✔ [V], caveat | `research_v9_recovery_20261013T075000Z/dry_alarm_summary.csv`; `outputs/wf_dry_by_event.txt` | Add: "206 windows from 9 events (82 % from three). Events with ≥ 1 alarm: MNDWI 8/9, teacher 2/9, student 5/9." |
| 12 | Six water-free S11 test chips, worst 0.2 %–43 % | ✔ [R] | wrap-up §5.6; V9 S11 rows: 6 positive-valid dry chips [V] | — |
| companion A3, limit 5 | "STURM's 28 dry windows: every method flags 22–24 → unlabelled water" | ✘ [V] | 20 of 28 have zero valid pixels; alarm rule true at n = 0. Real dry chips: 8; teacher 4/8, MNDWI 3/8 alarms | "STURM has only 8 water-free chips under our mapping (20 more contain only ignored permanent water); too few to measure false alarms." |
| 13 | D − A: S11 −0.2, STURM +0.8, WF +1.1; WF 4/4 seeds, STURM 3/4; seed sd 3.7–7.4; 750 updates | ✔ [V] | `same_image_label_swap.csv` | Add to D caveats: "teacher labels drawn on chips resized to 224 px (≈ 23 m), not at the teacher's native 10 m scale." |
| 13 | C: crop removes 73.6 % of water; 200 chips; 21 empty | ✔ [R] | wrap-up §5.7 | — |
| 14 | P/R/IoU table (S11, STURM, WF) | ✔ [V] | `prompt5_closure/tables/current_full_summary.csv` via fig06 CSV | — |
| 14 | "At MNDWI's recall … WF 85.6 vs 78.1 %, S11 73.5 vs **67.9** %, equal on STURM" | ✘ minor [V] | `threshold_free_ap.csv`: S11 MNDWI precision 67.5 % | "73.5 vs 67.5 %" |
| 14 notes | Pooled AP WF 88.3/88.4, STURM 82.7/77.7, S11 94.7/90.7; per-event WF 76.6/70.8 | ✔ [V] | `threshold_free_ap.csv` | Add: "per-event STURM AP 56.2 vs 55.5 (tie)." |
| 14 / 15 | "Inside the training distribution the student leads by 13 points" | ✔ number, caveat [V] | S11 split files: test chips come from the same 10 events as train | "On new chips from the 10 training events (S11 test)." Proposed: add the Bolivia held-out event (P7). |
| 15 | Events won 15/29, 6/11; haze 74.8 vs 25.7; darkening 54.2 vs 75.7; labeller 75.4/75.4 vs 71.5/76.8 | ✔ [R] | wrap-up fig07, fig08 data; paper S4 | — |
| 17 | Paper corrections list | add 2 [V] | — | Add: "§III-D/App. B 'ReLU6-from-scratch' → ReLU6 retrained from the pretrained B0 weights"; "§IV-A 'download and screening' → state the ≥ 1 % MNDWI new-water, ≤ 50 % permanent-water and ≤ 40 % cloud-probability gates with up to 8 relocations." |

## Code/documentation inconsistencies found (not in the deck)

- `data/ood_datasets.py`: the STURM band list comment says B8A for band 7; per the paper
  (App. A) STURM supplies B8. Comment only; numbers unaffected.
- Paper App. A: the STURM inverse is applied "without clipping". The loader clips
  reflectance to [0, 1] after the inverse, which binds only for stored values < −0.33.
- `experiments/repro_n2500.sh` (release) and the historical `v2_repro_n2500.sh` both build
  the ReLU6 student with pretrained weights. That is consistent with the checkpoints, not with
  the paper text.
- `OLD/esi_project/runs/repro_review/pseudo_ofa_relu6_*` used `--batch_size 8`, a recipe
  deviation flagged in `repro_full_train.sh`. These are *not* the Table III checkpoints
  (`runs/v2_repro`, deleted; eval JSONs remain). Use them for P1 only together with their own
  QAT children.
