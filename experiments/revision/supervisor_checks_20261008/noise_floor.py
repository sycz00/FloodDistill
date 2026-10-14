"""Seed noise and minimum detectable effect from EXISTING 4-seed results (CPU, read-only).

Inputs: prompt5_closure/tables/current_full_by_seed.csv, OFA_matched_effects.csv,
        current_full_comparator_paired_event.csv
Approximation: two-sided alpha=.05, power=.80, paired design, t-quantiles with n-1 dof:
  MDE ~= (t_{.975,n-1} + t_{.80,n-1}) * SD_diff / sqrt(n)
"""
import csv, math, statistics as st
from collections import defaultdict
T = "/home/schmalstieg/repos/FloodDistill/experiments/revision/prompt5_closure/tables/"
# t quantiles (dof = n-1): n=3,4,6,8
tq975 = {2: 4.303, 3: 3.182, 5: 2.571, 7: 2.365}
tq80 = {2: 1.061, 3: 0.978, 5: 0.920, 7: 0.896}
rows = list(csv.DictReader(open(T + "current_full_by_seed.csv")))
P = {"S11": "S11_TEST512_LEGACY", "STURM": "P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1", "WF": "P3_WF_NATIVE_ZERO_PAD512_V1"}
print("Seed SD (sample, ddof=1, 4 seeds) of pooled scores, percentage points  [full domain]")
print(f"{'model':46s} {'S11 mIoU':>9s} {'STURM mIoU':>10s} {'WF mIoU':>8s} | {'S11 wIoU':>8s} {'STURM wIoU':>10s} {'WF wIoU':>8s}")
g = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["domain"] != "full": continue
    for d, p in P.items():
        if r["protocol"] == p:
            g[(r["architecture"], r["variant"], r["budget"])][(d, "mIoU")].append(float(r["mIoU"]))
            g[(r["architecture"], r["variant"], r["budget"])][(d, "wIoU")].append(float(r["water_IoU"]))
sds = []
for k, v in sorted(g.items()):
    if len(v[("S11", "mIoU")]) < 4: continue
    f = lambda d, m: 100 * st.stdev(v[(d, m)])
    print(f"{'/'.join(k):46s} {f('S11','mIoU'):9.2f} {f('STURM','mIoU'):10.2f} {f('WF','mIoU'):8.2f} | {f('S11','wIoU'):8.2f} {f('STURM','wIoU'):10.2f} {f('WF','wIoU'):8.2f}")
    sds.append([f('S11','mIoU'), f('STURM','mIoU'), f('WF','mIoU')])
med = [st.median(c) for c in zip(*sds)]
mx = [max(c) for c in zip(*sds)]
print(f"{'median over ALL listed cells (incl. N=100)':46s} {med[0]:9.2f} {med[1]:10.2f} {med[2]:8.2f}")
print(f"{'max over ALL listed cells (incl. N=100)':46s} {mx[0]:9.2f} {mx[1]:10.2f} {mx[2]:8.2f}")
# N=2500-only summary (the 7 pretrained model/objective cells at the deployment budget)
sds2500 = []
for k, v in sorted(g.items()):
    if k[2] == "2500" and len(v[("S11", "mIoU")]) >= 4:
        sds2500.append([100 * st.stdev(v[("S11", "mIoU")]), 100 * st.stdev(v[("STURM", "mIoU")]), 100 * st.stdev(v[("WF", "mIoU")])])
med2 = [st.median(c) for c in zip(*sds2500)]; mx2 = [max(c) for c in zip(*sds2500)]
print(f"{'median over N=2500 cells (n=%d)' % len(sds2500):46s} {med2[0]:9.2f} {med2[1]:10.2f} {med2[2]:8.2f}")
print(f"{'max over N=2500 cells':46s} {mx2[0]:9.2f} {mx2[1]:10.2f} {mx2[2]:8.2f}")

print("\nPaired-seed SD of OFA - matched-KD differences (pp) -> approximate MDE for n seeds (80% power, alpha .05)")
print(f"{'cell':44s} {'SD_diff':>8s} {'MDE n=4':>8s} {'MDE n=8':>8s}")
for r in csv.DictReader(open(T + "OFA_matched_effects.csv")):
    sd = float(r["paired_seed_SD_pp"]); n = int(r["n_seeds"])
    mde4 = (tq975[3] + tq80[3]) * sd / math.sqrt(4)
    mde8 = (tq975[7] + tq80[7]) * sd / math.sqrt(8)
    print(f"{r['architecture']+' N'+r['budget']+' '+r['dataset']:44s} {sd:8.2f} {mde4:8.2f} {mde8:8.2f}")

print("\nEvent-level (conditional on fixed checkpoints) 95% CI half-widths, model - teacher, mIoU, pp:")
for r in csv.DictReader(open(T + "current_full_comparator_paired_event.csv")):
    if r["comparator"] == "teacher" and r["metric"] == "mIoU" and r["domain"] == "full" and r["variant"] in ("matched_lkd_clip5_valid",) and r["architecture"] == "efficientvit_b0" and r["budget"] == "2500":
        lo, hi = float(r["ci95_low"]), float(r["ci95_high"])
        print(f"  {r['dataset']:6s} n_events={r['n_defined_events']:>3s}  diff={100*float(r['mean_event_difference']):+.2f}  CI=[{100*lo:+.2f},{100*hi:+.2f}]  half-width={50*(hi-lo):.2f}")
