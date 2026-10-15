"""Aggregate the surface-water re-evaluation (CPU only). Reads results/*.npz, writes tables/ and summary.md.

Families = models.json family (4 seeds) or single models (teacher, rules). Seeds are averaged per unit
before paired event tests; events are the replication unit. Operating points for learned models are
selected on S11 validation only; spectral-rule thresholds were frozen on S11 train+val (calibration.json).
"""
from __future__ import annotations

import csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
TAB = RES.parent / "tables" if len(sys.argv) <= 1 else RES / "tables"
TAB.mkdir(parents=True, exist_ok=True)
REG = {m["id"]: m for m in json.loads((HERE / "models.json").read_text())["models"]}
RULES = ["MNDWI", "NDWI", "AWEInsh", "AWEIsh", "WI2015"]
NB = 4096


def family(mid):
    if mid.startswith(("rule:", "rulecal:")):
        return mid
    return REG[mid]["family"] if mid in REG else mid


def metrics(c):
    tp, fp, fn, tn = (float(x) for x in c)
    iou_w = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    iou_b = tn / (tn + fn + fp) if tn + fn + fp else float("nan")
    return dict(water_IoU=iou_w, background_IoU=iou_b, mIoU=np.nanmean([iou_w, iou_b]),
                precision=tp / (tp + fp) if tp + fp else float("nan"),
                recall=tp / (tp + fn) if tp + fn else float("nan"))


def ap_from_hist(h):
    """h: (2 labels, NB) counts; score bins ascending. Average precision (step) and PR arrays."""
    land, water = h[0][::-1].astype(float), h[1][::-1].astype(float)
    tp, fp = np.cumsum(water), np.cumsum(land)
    P = water.sum()
    if P == 0:
        return float("nan"), None, None
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    return ap, prec, rec


def wilcoxon_p(d):
    try:
        from scipy.stats import wilcoxon
        d = np.asarray([x for x in d if np.isfinite(x)])
        return float(wilcoxon(d, zero_method="pratt").pvalue) if len(d) >= 5 and np.any(d != 0) else float("nan")
    except Exception:
        return float("nan")


def boot_ci(d, n=10000, seed=1729):
    d = np.asarray([x for x in d if np.isfinite(x)])
    if len(d) < 3:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows_pool, rows_event, rows_pair, rows_ap, rows_dry, rows_op = [], [], [], [], [], []
    data = {p.stem: np.load(p, allow_pickle=False) for p in sorted(RES.glob("*.npz")) if not p.stem.endswith(".tmp")}
    # source-selected operating points (learned models): tau maximizing pooled water IoU on S11 val
    tau_bin = {}
    if "s11val" in data:
        d = data["s11val"]
        for mid in d["model_ids"]:
            key = f"hist__{mid}__s11"
            if mid.startswith(("rule", "rulecal")) or key not in d.files:
                continue
            h = d[key].sum(0)
            tp = np.cumsum(h[1][::-1])[::-1]; fp = np.cumsum(h[0][::-1])[::-1]; fn = h[1].sum() - tp
            iou = tp / np.maximum(tp + fp + fn, 1)
            tau_bin[str(mid)] = int(np.argmax(iou))
    for ds, d in data.items():
        if ds in ("s11train",):
            continue
        maps = [str(x) for x in d["mappings"]]; events = [str(x) for x in d["events"]]
        ev_list = [str(x) for x in d["event_list"]]
        ids = [str(x) for x in d["model_ids"]]
        fam = defaultdict(list)
        for mid in ids:
            fam[family(mid)].append(mid)
        for k, mp in enumerate(maps):
            # per family: pooled and equal-event metrics per seed
            ev_iou = {}
            for f, mids in fam.items():
                pooled = [metrics(d["conf__" + m][:, k].sum(0)) for m in mids]
                ev_seed = []
                for m in mids:
                    c = d["conf__" + m][:, k]
                    per = defaultdict(lambda: np.zeros(4))
                    for i, e in enumerate(events):
                        per[e] += c[i]
                    ev_seed.append({e: metrics(v)["water_IoU"] for e, v in per.items() if v[0] + v[1] + v[2] > 0})
                ev_mean = {e: float(np.nanmean([s.get(e, np.nan) for s in ev_seed])) for e in ev_list}
                ev_iou[f] = ev_mean
                eq = [np.nanmean(list(s.values())) for s in ev_seed]
                row = dict(dataset=ds, mapping=mp, family=f, n_models=len(mids))
                for mt in ("water_IoU", "mIoU", "precision", "recall"):
                    v = [p[mt] for p in pooled]
                    row[mt] = 100 * float(np.mean(v)); row[mt + "_sd"] = 100 * float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
                row["equal_event_water_IoU"] = 100 * float(np.mean(eq)); row["n_events"] = len(ev_list)
                rows_pool.append(row)
                for e, v in ev_mean.items():
                    rows_event.append(dict(dataset=ds, mapping=mp, family=f, event=e, water_IoU=100 * v))
            # paired event tests vs MNDWI (threshold 0)
            ref = ev_iou.get("rule:MNDWI")
            if ref:
                for f, ev in ev_iou.items():
                    if f == "rule:MNDWI":
                        continue
                    diff = [100 * (ev[e] - ref[e]) for e in ev_list if np.isfinite(ev[e]) and np.isfinite(ref[e])]
                    lo, hi = boot_ci(diff)
                    rows_pair.append(dict(dataset=ds, mapping=mp, family=f, n_events=len(diff),
                                          mean_event_diff_vs_MNDWI=float(np.mean(diff)) if diff else float("nan"),
                                          ci95_lo=lo, ci95_hi=hi, wins=int(np.sum(np.array(diff) > 0)),
                                          wilcoxon_p=wilcoxon_p(diff)))
            # AP and precision at MNDWI recall (histograms)
            hk = f"__{mp}"
            mndwi_h = d["hist__rule:MNDWI" + hk].sum(0) if ("hist__rule:MNDWI" + hk) in d.files else None
            mndwi_rec = None
            if mndwi_h is not None:
                c = d["conf__rule:MNDWI"][:, k].sum(0); mndwi_rec = c[0] / max(c[0] + c[2], 1)
            for f, mids in fam.items():
                if f.startswith("rulecal"):
                    continue
                aps, peq, pars = [], [], []
                for m in mids:
                    key = f"hist__{m}{hk}"
                    if key not in d.files:
                        continue
                    H = d[key]
                    ap, prec, rec = ap_from_hist(H.sum(0)); aps.append(ap)
                    peq.append(np.nanmean([ap_from_hist(H[e])[0] for e in range(H.shape[0])]))
                    if mndwi_rec is not None and prec is not None:
                        j = int(np.searchsorted(rec, mndwi_rec)); j = min(j, len(prec) - 1); pars.append(prec[j])
                if aps:
                    rows_ap.append(dict(dataset=ds, mapping=mp, family=f, pooled_AP=100 * float(np.mean(aps)),
                                        equal_event_AP=100 * float(np.mean(peq)),
                                        precision_at_MNDWI_recall=100 * float(np.mean(pars)) if pars else float("nan"),
                                        MNDWI_recall=100 * float(mndwi_rec) if mndwi_rec is not None else float("nan")))
            # source-selected operating point applied to this test set
            for f, mids in fam.items():
                vals = []
                for m in mids:
                    key = f"hist__{m}{hk}"
                    if m not in tau_bin or key not in d.files:
                        continue
                    h = d[key].sum(0); b = tau_bin[m]
                    tp = h[1][b:].sum(); fp = h[0][b:].sum(); fn = h[1][:b].sum()
                    vals.append(100 * tp / max(tp + fp + fn, 1))
                if vals:
                    rows_op.append(dict(dataset=ds, mapping=mp, family=f, water_IoU_at_S11val_operating_point=float(np.mean(vals))))
            # dry windows (valid pixels > 0, no water)
            any_valid = None
            for f, mids in fam.items():
                c0 = d["conf__" + mids[0]][:, k]
                valid = c0.sum(1); water = c0[:, 0] + c0[:, 2]
                dry = np.where((valid > 0) & (water == 0))[0]
                if any_valid is None:
                    any_valid = dry
                for thr_name, thr in (("any", 0.0), ("0.1%", 0.001), ("1%", 0.01), ("5%", 0.05)):
                    counts, ev_hit = [], []
                    for m in mids:
                        c = d["conf__" + m][:, k][dry]
                        fpix = c[:, 1]; n = c.sum(1)
                        alarm = fpix > thr * n if thr > 0 else fpix > 0
                        counts.append(int(alarm.sum()))
                        ev_hit.append(len({events[i] for i, a in zip(dry, alarm) if a}))
                    rows_dry.append(dict(dataset=ds, mapping=mp, family=f, dry_windows=len(dry),
                                         dry_events=len({events[i] for i in dry}), rule=thr_name,
                                         alarms_mean=float(np.mean(counts)), alarms_min=min(counts), alarms_max=max(counts),
                                         events_with_alarm_mean=float(np.mean(ev_hit))))
    for name, rows in (("pooled", rows_pool), ("per_event", rows_event), ("paired_vs_MNDWI", rows_pair),
                       ("ap", rows_ap), ("dry_windows", rows_dry), ("operating_point", rows_op)):
        if rows:
            with open(TAB / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", {n: len(r) for n, r in (("pooled", rows_pool), ("event", rows_event), ("pair", rows_pair),
                                         ("ap", rows_ap), ("dry", rows_dry), ("op", rows_op))}, "->", TAB)


if __name__ == "__main__":
    main()
