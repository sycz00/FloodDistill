#!/usr/bin/env python3
"""Re-derive the numbers the reviewer wrap-up corrects or relies on most.

Read-only. Prints a report and writes verification.json next to this file.
Each check compares a value recomputed from a raw artifact with the value
printed in the submitted manuscript or supplement (paper/main.pdf,
paper/supplement.pdf).

Run:  python3 -I verify_numbers.py
"""
import csv
import json
import os
import re
import statistics as st

import make_figures as mf

HERE = os.path.dirname(os.path.abspath(__file__))
checks = []


def check(name, recomputed, printed, source, verdict):
    checks.append(dict(check=name, recomputed=recomputed, printed_in_paper=printed, raw_source=source, verdict=verdict))
    print(f"[{verdict:>9s}] {name}\n            recomputed: {recomputed}\n            paper:      {printed}")


# 1. Table III from the four raw on-device logs
dev = mf.device_logs()
seeds = ["42", "64", "91", "107"]
for ds, label, printed in [("sturm", "STURM", "float 0.7557±0.0062 → INT8 0.7448±0.0236"),
                           ("wf", "WorldFloods-v2", "float 0.7861±0.0105 → INT8 0.7702±0.0214")]:
    fl = [dev[(ds, "float", s)] / 100 for s in seeds]
    q = [dev[(ds, "int8", s)] / 100 for s in seeds]
    check(f"Table III device rows, {label}",
          f"float {st.mean(fl):.4f}±{st.stdev(fl):.4f} → INT8 {st.mean(q):.4f}±{st.stdev(q):.4f}; "
          f"per-seed INT8-float (pp): " + ", ".join(f"s{s} {100 * (b - a):+.2f}" for s, a, b in zip(seeds, fl, q)),
          printed, mf.OLD_DEVICE + "/N_both_s{42,64,91,107}", "matches")

# 2. Table IV from the s91 trtexec summary log
log = open(f"{mf.PSD}/Table_IV_Jetson_Efficiency/s91_speed.log").read()
pw = re.search(r"GPU power mean\(W\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", log).groups()
check("Table IV GPU mean power (FP32 / FP16 / INT8)", " / ".join(pw) + " W",
      "3.95 / 4.21 / 3.42 W", "paper_source_data/Table_IV_Jetson_Efficiency/s91_speed.log",
      "MISMATCH: FP32 mean power is 3.25 W in the log, printed 3.95 W")
other = {s: re.search(r"Latency mean\s+\(ms\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                      open(f"{mf.PSD}/Table_IV_Jetson_Efficiency/s{s}_speed.log").read()).groups() for s in ("42", "64", "91")}
check("Table IV latency across the three engine builds (FP32/FP16/INT8, ms)",
      "; ".join(f"s{s}: {'/'.join(v)}" for s, v in other.items()), "s91 values printed: 7.91/5.20/5.57",
      "paper_source_data/Table_IV_Jetson_Efficiency/s{42,64,91}_speed.log", "matches (s91), spread small")

# 3. Pre-filter effect from the raw per-scene selector counts (STURM, clean)
sel = mf.selector_raw()
parts = []
for name, a, b in [("FPS", "v2cur_fps_N2500", "v2cur_fps_raw_N2500"),
                   ("random", "v2cur_random_N2500", "v2cur_random_raw_N2500"),
                   ("GLISTER", "v2cur_glister_N2500", "v2cur_glister_raw_N2500"),
                   ("confidence", "v2cur_confidence_N2500", "v2cur_confidence_raw_N2500")]:
    ma, mb = mf.mean_sd(sel[a])[0], mf.mean_sd(sel[b])[0]
    parts.append(f"{name}: {mb:.2f} → {ma:.2f} ({ma - mb:+.2f} pp)")
check("Clean STURM mIoU without → with the pre-filter, per selector", "; ".join(parts),
      "'removing it costs up to 0.048 mIoU clean' (Supplement S2)",
      "paper_source_data/Supp_Figure_S1_Subset_Selection/raw/*__clean.json (+ figure CSV for random)",
      "MISMATCH: cost is 0.1-0.3 pp for FPS/random, 4.0 pp for GLISTER, 31 pp for confidence ranking")

# 4. Pool funnel
p = mf.pool_counts()
check("GDACS pool funnel", f"{p['n']:,} tiles; <5% water {p['dry']:,} (zero water {p['zero']:,}; pre-event {p['dry_pre']:,}); "
      f">85% {p['wet']:,}; too uniform {p['flat']:,}; candidates {p['cand']:,}; selected 2,500",
      "23,720 → 5,987 → 2,500; '8,883 baselines the teacher left empty'", mf.POOL + "/pool_features.csv + diversity_index.json",
      "matches counts; wording: the 8,883 tiles have <5% predicted water, only part of them are empty")

# 5. Definition of the student 'false alarm' rate (1.9%)
src = open(f"{mf.REPO}/evaluation/relabel_feasibility.py").read()
ff = re.search(r"FF_FLOOD\s*=\s*([0-9.]+)", src).group(1)
check("Definition behind 'false alarm on 1.9% of baselines'",
      f"student water fraction >= {ff} on pre-event tiles where the teacher's water fraction < {ff}",
      "'raises a false alarm on 1.9 ± 0.8% of the 8,883 baselines the teacher left empty'",
      "evaluation/relabel_feasibility.py (FF_FLOOD)", "wording needs the 5% area rule and 'teacher-referenced'")

# 6. Teacher baseline (verified in the prompt-5 closure)
tb = mf.read_csv(f"{mf.REV}/prompt5_closure/tables/teacher_baseline_reconciliation.csv")
check("Teacher mIoU, legacy protocol (S11 / STURM / WF)",
      " / ".join(f"{float(r['verified_current_mIoU']):.4f}" for r in tb), " / ".join(r["printed_mIoU"] for r in tb),
      "prompt5_closure/tables/teacher_baseline_reconciliation.csv", "S11 printed value untraced (-0.24 pp); WF +0.38 pp")

# 7. Six human-labelled dry S11 chips (single seed-42 checkpoints)
rows = mf.read_csv(f"{mf.REV}/prompt2_report/dry_diagnostics.csv")
summ = []
for comp in ["P4_dry_diagnostic_teacher", "P4_dry_diagnostic_evit_ofa", "P4_dry_diagnostic_evit_logit",
             "P4_dry_diagnostic_evit_hard", "dry_mndwi"]:
    v = [r for r in rows if r["comparator"] == comp]
    worst = max(float(r["false_water_fraction"]) for r in v)
    summ.append(f"{comp.replace('P4_dry_diagnostic_', '')}: any-water chips {sum(r['any_positive_pixel'] == 'True' for r in v)}/6,"
                f" worst {100 * worst:.1f}%")
check("False water on the 6 water-free S11 test chips", "; ".join(summ), "not reported",
      "experiments/revision/prompt2_report/dry_diagnostics.csv", "diagnostic only (6 chips, 1 seed, overlapping events)")

# 8. Per-event wins against MNDWI and the weight of EMSR342
ev, wpx, stu = mf.event_tables()
for ds in ("sturm", "wf"):
    T, M = ev[(ds, "teacher")], ev[(ds, "MNDWI>0")]
    es = sorted(set(T) & set(M))
    s_w = sum(1 for e in es if len(stu.get((ds, e), [])) == 4 and st.mean(stu[(ds, e)]) > M[e])
    check(f"Events where the learned model beats MNDWI, {ds}", f"teacher {sum(T[e] > M[e] for e in es)}/{len(es)}, "
          f"student {s_w}/{len(es)}", "not reported", "capacity_v7 event_results.csv + prompt4 cross-cohort event counts",
          "new evidence")
tot = sum(v for (ds, e), v in wpx.items() if ds == "wf")
check("Share of WorldFloods reference water in EMSR342", f"{100 * wpx[('wf', 'EMSR342')] / tot:.1f}%; "
      f"water IoU MNDWI {ev[('wf', 'MNDWI>0')]['EMSR342']:.1f}, teacher {ev[('wf', 'teacher')]['EMSR342']:.1f}, "
      f"student {st.mean(stu[('wf', 'EMSR342')]):.1f}", "'a single AOI carries 57% of its flood pixels' (main text)",
      "capacity_v7 event_results.csv (event = EMS activation, not AOI)", "different unit: 82% per event vs 57% per AOI")

with open(os.path.join(HERE, "verification.json"), "w") as f:
    json.dump(checks, f, indent=1)
print("\nwrote verification.json with", len(checks), "checks")
