"""V9 dry-window alarms per event (WF test, deployed thresholds, >=1% rule): denominators and alarm counts by event
for MNDWI, teacher and the paper student (4 seeds). Read-only on research_v9_recovery_20261013T075000Z/dry_window_alarms.csv."""
import csv, collections, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["dataset"] == "wf" and r["variant"] == "deployed" and int(r["valid_pixels"]) > 0]
den = collections.Counter(r["event"] for r in rows if r["model"] == "teacher")
def by(model_pred):
    c = collections.Counter()
    for r in rows:
        if model_pred(r["model"]) and r["alarm_01"] == "True": c[r["event"]] += 1
    return c
mn = by(lambda m: m == "MNDWI"); te = by(lambda m: m == "teacher")
st = by(lambda m: m.startswith("O4_original_efficientvit_b0_ofa_N2500"))
print(f"{'event':10s} {'dry':>4s} {'MNDWI':>6s} {'teacher':>8s} {'student(sum 4 seeds)':>21s}")
for e, n in den.most_common():
    print(f"{e:10s} {n:4d} {mn[e]:6d} {te[e]:8d} {st[e]:21d}")
print(f"{'total':10s} {sum(den.values()):4d} {sum(mn.values()):6d} {sum(te.values()):8d} {sum(st.values()):21d}")
