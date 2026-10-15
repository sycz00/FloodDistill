"""Collect the four-seed QAT summaries into qat_seeds_summary.csv (fractions; deltas vs each seed's float parent)."""
import csv, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
rows = []
for f in sorted(HERE.glob("*_s*/summary.json")):
    r = json.loads(f.read_text())
    base = dict(model=r["model"], seed=r["seed"], parameters=r["parameters"])
    rows.append(dict(base, variant="float", mIoU=r["float"]["mIoU"], water_IoU=r["float"]["water_IoU"],
                     equal_event_mIoU=r["float"]["equal_event_mIoU"], pooled_delta=0.0, equal_event_delta=0.0,
                     acceptable="", strict_logit_parity=""))
    if "ptq_percentile_pc" in r:
        p = r["ptq_percentile_pc"]
        rows.append(dict(base, variant="PTQ percentile_pc", mIoU=p["mIoU"], water_IoU=p["water_IoU"], equal_event_mIoU="",
                         pooled_delta=p["pooled_delta"], equal_event_delta=p["equal_event_delta"], acceptable=p["acceptable"],
                         strict_logit_parity=""))
    q = r["qat"]
    rows.append(dict(base, variant=f"QAT{q['updates']}", mIoU=q["mIoU"], water_IoU=q["water_IoU"], equal_event_mIoU="",
                     pooled_delta=q["pooled_delta"], equal_event_delta=q["equal_event_delta"], acceptable=q["acceptable"],
                     strict_logit_parity=q["strict_logit_parity"]))
with open(HERE / "qat_seeds_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
