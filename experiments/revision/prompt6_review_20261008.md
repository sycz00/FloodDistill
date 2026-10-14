# Review of `prompts/prompt6.md` against supervision and evidence

**Date:** 2026-10-08 · **Outputs of this review:** [`prompts/prompt6_corrected.md`](../../prompts/prompt6_corrected.md) (merged, corrected prompt) · [`prompts/prompt6_corrected_appendix.md`](../../prompts/prompt6_corrected_appendix.md) (conditional specs + creative hypotheses) · this file (change map) · addendum in [`supervisor_report_20261008.md`](supervisor_report_20261008.md) · receipts in [`supervisor_checks_20261008/receipts/`](supervisor_checks_20261008/receipts/).
**Tags:** **[V]** verified by me from raw artifacts or algebra · **[R]** taken from the student's report · **[H]** my judgment, untested.
Line numbers refer to `prompts/prompt6.md` as it stands (407 lines).

---

## 0. Verdict

1. **Aligned, and in places ahead of my v3.** prompt6 reaches the same core diagnosis independently (QAT gradients 10³–10¹⁰, LiteMLA quotient, signed 4-bit grids, one-seed pilot, research-exposed benchmarks) and adds good technical material I did not have: the exact Δ-identity, fixed-weight counterfactual quantizer probes, the A1/A3 width×depth factorial, signedness analysis, faithful-LSQ requirement, final-update reporting, deterministic replay, explicit inherited/differs/control/falsifier novelty assessment. Every number I could check was right.
2. **Not executable as written** — four structural problems: (a) scope: two new methods, a factorial, a rotation, LSQ, deployment, baselines, ~10 QAT continuations and a 12-GPU-h cap, mostly at **one seed**, with a 0.5 pp gate that the pilot itself shows is below noise (B3 vs B2, near-identical objectives, differ by 1.53 pp on S11-test); (b) it keeps the v2 QAT optimizer (lr 1e-5 × 1,560 updates) although it moves only 0.4–2.1 % of the 4-bit integer weight codes; (c) it lacks the capacity-floor / robustness / disagreement programme that gives the project a publishable question; (d) its flagship **Candidate A rests on a premise my probes show to be a float32-simulation artifact.**
3. **New measurements change several of its priorities** (§3): integer-domain arithmetic removes the gradient explosion; an 8-bit stem plus unsigned grids recovers 70 % of the W4A4 PTQ gap with no training; one layer explains 82 % of A1's 4-bit-weight damage; a 401-parameter per-pixel model beats the conv students on S11-val (pilot).
4. **The corrected prompt keeps prompt6's best content, re-orders it baseline-first, makes every gate power-aware, parks Candidate A and the rotation, cuts it from 5,010 to 4,458 words (+2,369 words of on-demand appendix), and adds your requested exploration track (§6 here, Stage 8 there).**

---

## 1. What prompt6 gets right (adopted)

| Item (prompt6 lines) | Why it is good | Where adopted |
|---|---|---|
| Objective: failure explanation + task-specific damage vs global feature metric (L8) | Right reframing of v2 | Task |
| Treat v2 as one seed, not 98 experiments (L33) | Correct and important | Facts / Ground rule 6 |
| Exact A1/A3 structure; A2 ≡ A3 in depth (L37) **[V]** | I had missed that A2 is a width-only rung | F9; Stage 2 X1/X2 |
| Report final-update as well as selected checkpoints (L85) | Selection over 5 evaluations inflates noise | Ground rule 3 |
| Research-exposed benchmarks; no external-driven gates (L89–91) | Stricter and more correct than my earlier wording | Ground rule 5 |
| Logger audit in float64, per-step distributions, module shares (L99–108) | Closes the "bad logging" case | 1.1 |
| Cases (a)–(d) and C_l collapse metric (L116–126) | Precisely what was needed; I computed C_l | 1.2, F2 |
| Fixed-weight counterfactual quantizer probes (L128–132) | Cheap causal tool | 1.2 |
| Signed vs unsigned 4-bit grids; stem step 1.434355 (L136, L146) **[V]** | Turned out to be the biggest forward effect | F5, Stage 3 |
| Faithful LSQ, no silent folding into new methods (L148) | Keeps attribution honest | Stage 3 |
| Δ-identity, H_T/B_T, GT flips with B−H (L204–219) **[V algebra]** | Task-level diagnostic; corrects "all displacement is harmful" | 1.5, appendix §B |
| Factorial B0/BN/BR/BNR logic and "correction-only comparison to v2" (L252–260) | Right way to separate effects | Stage 5 (reduced) |
| Backward-floor control for Candidate A (L194–196) | Its own control is what kills Candidate A | appendix §A |
| Deployment as separate workstream; three size quantities; no custom kernels (L321–332) | Honest | Stage 6 |
| Inherited / differs / control / falsifier novelty assessment (L386) | Excellent discipline | Outputs (ledger) |
| Reference list with caveats (L388–404) | 6 of 7 verified | appendix §F |

## 2. Claims in prompt6 that I verified

| prompt6 claim | Check | Result |
|---|---|---|
| L37 A1 `width .5, depth 1`; A3 `width .25, depth .5`; A2 and A3 resolve to identical repeats | `compression_graph_spec` | ✓ `depth_list` A1 `[1,2,2,2,2]`, A2 = A3 = `[0,1,1,1,1]`; attention dim 16/8/6/4 |
| L39–47 source-validation numbers | pilot report tables | ✓ |
| L53–55 pre-clip norm ranges (9.56e5–3.405e8; 5.297e9–1.300e10; 1.082e3–3.298e5) | per-epoch JSONs | ✓ exactly |
| L57 "does NOT prove zero optimizer movement" | code-flip / update measurement | ✓ correct caution — and the answer is: movement is tiny (0.4–2.1 % of W4 codes; max |Δw| ≤ 0.22 step) |
| L136 signed [−8,7] at a ReLU6 site gives 8 non-negative levels | `_range()` code | ✓ |
| L146 first-conv A3 W4A4 step ≈ 1.434355 | checkpoint `_act_scale` | ✓ identical value; **64.1 % of normalised input values round to 0** at that step |
| L181–186 surrogate VJP / stop-gradient form | algebra | ✓ forward equal, both local derivatives scaled by d/d̄ |
| L204–212 Δ = T²[(p_F−p_T)(b−a) + KL(Bern(p_F)‖Bern(p_Q))] | algebra | ✓ (softplus(b)−softplus(a) = KL(p_F‖p_Q) + p_F(b−a)) |
| L282–286 rotation: W′ = W·Uᵀ gives W′(Uz) = Wz, zero padding preserved | algebra | ✓ for orthogonal U |
| L392–404 references | web fetch | ✓ 6/7 (title, authors, venue); **MPT is AAAI 2026 and concerns model-update negative flips, not quantization**; **QuS not retrievable** (verification wall; search also surfaced *Trio-ViT*, unread) |

## 3. New evidence from this review (receipts)

| # | Finding | Numbers | Receipt |
|---|---|---|---|
| N1 | **The gradient explosion is a float32-simulation artifact.** Conv on integer codes then scale (exact zeros preserved) | W4A4 1.14e11 → 4.0; W4A8 1.29e8 → 4.4; W8A8 3.98 → 3.97. Eval parity: PTQ ≤ 0.03 pp; selected v2 QAT checkpoints ≤ 0.14 pp. Caveat: 7.5 % argmax disagreement on one augmented TRAIN batch at W4A4 | `int_domain_fakequant`, `int_domain_ptq_eval`, `int_domain_qat_eval`, `int_domain_divergence` |
| N2 | **Gradient concentration.** 25/25,088 and 12/12,544 attention positions hold ≈100 % of the squared qkv gradient; D_Q median ≈ 1e-7, |q| ≈ 5–8e-9 (rounding-residue scale); 64–83 % had healthy D_F (case b), 17–24 % already small (case a). With quantizers off, same weights: total grad² 6e-4 vs 2.7e19 | `attn_grad_concentration` |
| N3 | **Exact-zero denominators already exist in FP** (5–9 % of positions, attention dim 4) and are harmless; quantization collapse C = 6.5–8.2 % (W4A4), 3–4 % (W4A8), 1.5–1.7 % (W8A8) | `prompt6_probes` part 1 |
| N4 | **Forward damage is activation-side and mostly the stem** (A3 PTQ S11-val): W4A4 68.99; W8A4 68.84; stem 8-bit 76.23; unsigned grids 72.17; both 78.31; stem + qkv/aggreg 8-bit 75.89; W4A8 82.20; FP 82.70 | `prompt6_probes` part 2 |
| N5 | **A1's 4-bit-weight damage is localised.** One 4-channel stem depthwise conv (36 weights, +18 B) at 8 bit: 78.95 → 82.01 (random layer 79.01); top-2 82.79; top-4 82.98 (all-W8 82.66). Optimistic (same 89 images for ranking and scoring) | `a1_layer_sensitivity`, `a1_mixed_precision` |
| N6 | **Optimizer reach.** Integer-code change W4 0.4–2.1 %, W8 28–29 %; head moves 40–120× less at W4A4 than at W8A8 | `weight_code_flips`, `param_motion` |
| N7 | **Per-pixel pilot (S11-val only, single fit).** MNDWI 75.91/59.84 (mIoU/water-IoU); zero-sum log-linear index (6 params, exactly gain-invariant) 81.36/66.76; 401-param per-pixel MLP 85.45/74.15 vs A3 82.70, A1 83.60, A0 84.16. Metric code reproduces 82.699 and 84.155 | `distilled_index_pilot`, `metric_sanity` |
| N8 | Noise floor (seed SD, MDE, event CIs) | median seed SD 0.22/0.20/0.51 pp; event CI ±0.9/±1.3/±1.4–1.9 | `noise_floor` |

## 4. Change map: prompt6 → corrected prompt

Verdicts: **KEEP** · **FIX** (error or contradiction) · **ADD** (missing) · **PARK/CUT** (scope) · **ALIGN** (conflict with v3 resolved).

| prompt6 | Verdict | Change | Why / evidence | Now in |
|---|---|---|---|---|
| L1 "failed compression pilot" | FIX wording | "negative one-seed screening pilot" | Not a failure; 62 of 98 nodes were designed skips | title/Task |
| L4–10 task, objective | KEEP + ADD | Add capacity-floor study and an exploration track as co-primary goals | Without them the project has no publishable question beyond a corrected baseline (N7, F10) | Task |
| L12–29 provenance | KEEP | Compact; "bundle lacks weights" is obsolete on this machine → hash the bytes | weights exist under `/data1` | Ground rule 2 |
| L33 one seed ≠ 98 experiments | KEEP | | | Facts (intro) |
| L35–37 graphs | KEEP + ADD | Add X1/X2 as seeded Stage-2 rungs; A2 width-only | N/A, **[V]** | F9, Stage 2 |
| L51–59 gradient facts "to verify" | FIX | Replace "verify independently" by measured facts F1–F4; answer L57 (movement is tiny) | N1–N3, N6 | Facts |
| L61–91 protocol | KEEP + FIX | Keep constants. **Separate fitted quantities (TRAIN only) from recipe hyperparameters (S11-val allowed)**; V2 recipe only for the replay arm; strong baseline may change lr/epochs (labelled) | prompt6 L82 forbids moving beyond lr 1e-5, which cannot move W4 codes (N6) | Ground rules 3–4 |
| L89–91 research-exposed | KEEP | Strengthen: no external number may change a gate/margin/rung/seed | consistent with my report | Ground rule 5 |
| L93–97 Wave A budget/probe lists | KEEP | Folded into Stage 1 | | Stage 1 |
| L99–108 §4.1 logger audit | KEEP | Condensed; my independent probe already reproduces the magnitude, Codex must still close the logger | N1 | 1.1 |
| L110–132 §4.2 quotient, cases a–d, C_l, probes | KEEP + ADD | Add the answers already measured (N2, N3); extend to A1 W4A8; label probe scope | N2/N3 | 1.2 |
| L134–148 §4.3 quantizer weaknesses | KEEP + **PROMOTE** | Promote to the core of Stage 3; results N4 given as facts; add LSQ-if-needed | stem + unsigned = +9.3 pp, no training | F5, Stage 3 |
| L146 input-stem audit | ANSWERED | 64.1 % of input values → 0 at A4; 8-bit stem +7.2 pp | N4 | F5 |
| L150–158 §4.4 old feature statistic | CUT to one line | Identity recompute is redundant; per-PC explainability (F7) supersedes | N/A | F7 |
| (absent) | **ADD** | **Faithful integer-domain simulator + floors**; sensitivity sweeps; label-free mixed precision | N1, N5 | 1.3, 1.4 |
| (absent) | **ADD** | **Optimizer-reach audit and LR/epochs grid** | N6 | 1.6, Stage 3 |
| L160–199 Candidate A | **PARK** (appendix §A) | Unlock only if the simulator/floor fails or collapse persists with accuracy loss | premise ("collapse harms the deployed computation") is unsupported: the collapse positions are rounding residues and exact arithmetic removes the pathology (N1, N2). Math is correct **[V]** | appendix §A |
| L200–244 Candidate B | KEEP, conditional (Stage 5) | Gate on H_T/B_T; add rule-disagreement weighting control; ≥ 3 seeds | identity verified; the auxiliary re-weights the same KL term per pixel, so generic hard-pixel weighting is the right control (prompt6 already names it) | appendix §B, Stage 5 |
| L246–264 Wave B (one seed, ≤ 10 QAT runs, factorial) | FIX | ≥ 3 seeds; factorial reduced to SB vs SB+X; hard cell defined after strong PTQ | v2 B3−B2 = 1.53 pp at near-identical objectives; QAT runs cost ≈ 200 s | Stage 3/5 |
| L266–276 Wave C (2×2 graphs, seed 42) | FIX + MOVE | Move to Stage 2 with 4 seeds; do the per-layer sweep first | A1's PTQ damage is one layer (N5); the FP A1<A3 reversal needs seeds (F8: non-pretrained SD up to ±3.7 pp) | Stage 2, 1.4 |
| L278–290 rotation idea | **PARK** (appendix §C) | PTQ-only test allowed | 8-bit stem is standard and free; +7.2 pp already (N4) | appendix §C |
| L292–306 H1–H4, 0.5 pp one-seed gate, "seed 64 next" | FIX | Numeric H1–H5 in `protocol_v4.yaml`; gate = mean paired gain ≥ max(0.5 pp, 2 SE), same sign in 3 seeds, water-IoU not worse; 3 seeds up front | one-seed 0.5 pp is inside noise | 1.7, Stage 5, Statistics and gates |
| L306 SQuaT/QFD faithful baselines only after a source gate | KEEP | | | Stage 5 |
| L308–319 reporting | KEEP, condensed | | | Statistics and gates; Outputs |
| L321–332 deployment | KEEP, narrow | **ALIGN:** keep the pre-registered source-only selector (no retrospective change) and show a multi-benchmark frontier as secondary | my earlier "select on all three benchmarks" contradicted the research-exposed rule — corrected | Stage 6 |
| L326 "UpSampleLayer casts to float" | FIX | Verify, do not assume | student's claim, not verified by me | Stage 6 |
| L334–351 verification list | KEEP, condensed | Candidate-specific tests to appendix; add simulator tests + metric sanity | | Verification; appendix |
| L355–357 one GPU; 12 GPU-h cap | **ALIGN** | Two idle 4090s allowed (the v2 pilot actually ran on two); stage caps ≈ 30 GPU-h; explicit 12-h variant | Stage 2 alone is ≈ 6 GPU-h at 4 seeds | Budget |
| L368, L370 manifests, worker checks | KEEP + ADD | Manifest < 10 MB | v2 manifest 64 MB | Ground rule 8 |
| L374–386 eight deliverables | CUT to four | `protocol_v4.yaml`; per-stage CSV + ≤ 1-page note; `report_v4.md` ≤ 3 pages with novelty ledger; receipts | document sprawl | Outputs |
| L388–404 references | KEEP + annotate | verified/unverified; add Trio-ViT (unread); MPT relevance by analogy | web check | appendix §F |
| (absent) | **ADD** | **Capacity-floor ladder, reduced robustness, disagreement lens, scale diagnostic, H3** | N7, F10 | Stage 4 |
| (absent) | **ADD** | **Temporal feasibility audit + two-labeler agreement** | pool has 11,895 event / 11,825 baseline tiles | Stage 7 |
| (absent) | **ADD** | **Exploration track U1–U8** (your request) | §6 below | Stage 8 |

## 5. Conflicts between prompt6 and my v3 — and how they were resolved

| Topic | prompt6 | v3 / report | Resolution |
|---|---|---|---|
| Order | mechanisms first (Candidate A/B), baseline "warranted by Wave A" | baseline first | **Baseline first**, with prompt6's diagnostics; mechanisms gated (Stage 5) |
| QAT optimizer | preserve lr 1e-5, 1,560 updates | raise lr / steps | V2 replay keeps lr 1e-5; strong baseline searches a ≤ 6-config grid on S11-val; both labelled |
| Seeds | one seed screen, then seed 64 on success | 3–4 seeds | ≥ 3 seeds up front where cheap; 1–2 seeds = screen (kill/defer only) |
| Frontier selection | keep source-only selector, don't use A3's external score | "select on all three benchmarks" (my report, Must #4) | **prompt6 is right** — the benchmarks are research-exposed. Selection stays pre-registered; report all benchmarks as secondary. My report is corrected |
| GPUs | one RTX 4090 by default | both | both only if idle and no foreign process (v2 pilot used two); **your decision** |
| Budget | 12 GPU-h | ≈ 24–30 | stage caps ≈ 30; explicit 12-h cut order under *Budget*; **your decision** |
| Hard cell | implicit | smallest × lowest-bit (v2) | defined *after* strong PTQ: lowest precision still ≥ 1.5 pp from FP |

## 6. Creative and unconventional ideas (your request) — what is in Stage 8

prompt6 already contained three unconventional ideas (counterfactual normalisation gradients, anchored excess-loss hinge, spectral rotation). Two are kept as conditional/parked for the reasons above. The corrected prompt adds eight hypotheses, each tied to a verified fact, with a cheap first gate and a pre-registered falsifier (designs in appendix §D):

| ID | Idea | Why it is not just another tweak | Grounding | Honest novelty risk |
|---|---|---|---|---|
| U1 | **Distill the foundation model into a closed-form, exactly gain-invariant spectral index** (zero-sum log-linear in six bands; optionally a sparse formula) | Turns "model vs rule" into "what rule would the teacher write?"; interpretable, trivially deployable | N7: +5.5 pp mIoU over MNDWI on S11-val, 6 parameters, exact invariance | data-driven water indices exist; NDWI-like contrasts reportedly collapse on WF |
| U2 | **A ~400-parameter per-pixel network may be enough** (tile-free, streamable); the 224/512 protocol may penalise context models | Challenges the premise that edge models need spatial context | N7: 85.45 > 84.16 (A0) on S11-val | single fit, validation only; MACs/pixel (384) exceed A3's (164) — advantage is memory/streaming, not MACs |
| U3 | **Invariance by construction beats invariance by augmentation:** parameter-free radiometry-normalising stem (log-centering, dark-object subtraction) → robust beyond trained severities, better 4-bit stem | Tests extrapolation, which augmentation cannot give; attacks both robustness (R2.5) and the stem problem | N4 (64 % of stem inputs → 0), paper's haze/tilt asymmetry between rule and learned models | illumination-invariant features, DOS and instance norm are classical |
| U4 | **Learn only what the rule cannot:** MNDWI prior head; rule-disagreement-weighted KD | Spends tiny capacity on the teacher-added content | disagreement lens | residual learning / hard-example weighting are known |
| U5 | **Rule–net disagreement as a free uncertainty signal** for abstention | Zero-cost second member from physics | learned: high precision/low recall on WF, rule opposite | ensemble disagreement is classical; no dry-scene claim without human GT |
| U6 | **Label-free, sensitivity-guided mixed precision** (teacher excess loss ranks layers) | Allocation without labels at calibration time; matches the mixed INT8/FP16 engine | N5: one layer, +3.06 pp for 18 B | HAWQ/BRECQ-style sensitivity exists |
| U7 | **Disagreement-driven pool curation** | Data-centric use of the new lens where FPS showed no clear gain | CPU gate on concentration of {T≠R} mass | active-learning prior art |
| U8 | **Two independent labelers (foundation model × physical rule) as a noise estimator for flood-increment labels** | Cheap route to temporal supervision without increment ground truth | pool pairs 11,895/11,825 tiles | the idea is a feasibility audit, not a result |

Priority by information per cost: **U6 → U1 → U5 → U7-gate → U8** (all CPU/PTQ-only), then U3, U4. None may displace Stages 1–3. If you want one bet for novelty, my ranking is **U1/U2 (context-free + gain-invariant distilled models)** because the pilot already shows a surprising, cheaply testable signal, followed by **U3**.

## 7. Corrections to my own earlier deliverables

1. **Report "Must #4"** ("select frontier models on all three benchmarks") contradicted my own rule that no external number may influence selection. Corrected: selection stays validation-based and pre-registered; all benchmarks are reported with CIs as secondary.
2. **Report §3.2(c) bullet 2** implied that quantization creates the zero denominators. Measured: exact zeros exist in FP; quantization adds collapse to rounding-residue-scale values, and exact arithmetic removes the gradient pathology. Corrected in the addendum.
3. **Report item "quantiser rounds 55.7 % to zero"** understated the dominant effect: the **stem input** (64.1 % rounded to 0) accounts for +7.2 pp of the W4A4 PTQ gap.
4. `prompts/codex_next_prompt_v3.md` is superseded by `prompt6_corrected.md` (header added there).

## 8. Decisions for you

1. **GPUs:** both idle RTX 4090s (default in the corrected prompt, as the v2 pilot ran) or one?
2. **Budget:** ≈ 30 GPU-h (full) or prompt6's 12 GPU-h (cut order spelled out under *Budget, order and cuts*)?
3. **Exploration track:** start the CPU/PTQ-only gates (U6, U1, U5, U7-gate, U8) immediately in parallel (default), or only after Stage 3?
4. **Candidate B:** keep as a conditional Stage-5 arm (default), or drop it to keep the line lean? (My prior that it beats generic hard-pixel weighting is low.)

## 9. File map

| File | Role |
|---|---|
| `prompts/prompt6.md` | your prompt, untouched |
| `prompts/prompt6_corrected.md` | **the prompt to give Codex** (core, 4.5 k words) |
| `prompts/prompt6_corrected_appendix.md` | specs read on demand (§A–§F) |
| `prompts/codex_next_prompt_v3.md` | my earlier prompt, marked superseded |
| `experiments/revision/prompt6_review_20261008.md` | this change map |
| `experiments/revision/supervisor_report_20261008.md` | report + Addendum A (new findings, corrections) |
| `experiments/revision/supervisor_checks_20261008/` | scripts + receipts; `bash run_all.sh` (fast set, ~4 min) / `full` |
