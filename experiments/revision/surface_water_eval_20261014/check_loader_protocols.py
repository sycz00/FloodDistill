"""Check the protocol switch in data/ood_datasets.py (default "surface_water" since 2026-10-15).

1. New default == the inputs and surface masks scored by eval_surface_water.py (WF 836 windows,
   STURM 2,675 chips at 512 and 128 px); "flood_p3" == the P3 masks of the same evaluator.
2. The old keyword arguments give byte-identical items to the loader before the change
   (git revision given by --old_rev), so existing paper/revision drivers are unchanged.
Writes loader_protocol_check.json next to this file.
"""
import argparse, importlib.util, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/schmalstieg/repos/FloodDistill")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import numpy as np, torch
import eval_surface_water as ev
from data import ood_datasets as new

WF = ev.DATA / "WorldFloodsv2"
ST = ev.DATA / "STURM-Flood/Dataset"


def load_old(rev):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:data/ood_datasets.py"],
                         capture_output=True, text=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / "ood_datasets_old.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("ood_datasets_old", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def same(a, b):
    return a["image"].shape == b["image"].shape and torch.equal(a["image"], b["image"]) \
        and torch.equal(a["mask"], b["mask"]) and a["filename"] == b["filename"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_rev", default="HEAD")
    ap.add_argument("--legacy_stride", type=int, default=1, help="check every n-th item for the legacy comparison")
    a = ap.parse_args()
    old = load_old(a.old_rev)
    out = {}

    # 1a. WorldFloods default / flood_p3 vs the surface-water evaluator
    ref = ev.WFTest()
    d_sw = new.WorldFloodsDataset(WF, split="test", img_size=512, tile_size=512)
    d_p3 = new.WorldFloodsDataset(WF, split="test", img_size=512, tile_size=512, protocol="flood_p3")
    assert d_sw.protocol == "surface_water" and len(d_sw) == len(ref) == 836 and len(d_p3) == 785
    p3_index = {(p.name, tuple(w)): i for i, (p, _, w) in enumerate(d_p3.samples)}
    bad = []
    for i in range(len(ref)):
        x, m, _ = ref[i]
        it = d_sw[i]
        ok = torch.equal(it["image"], x) and torch.equal(it["mask"], m[0].long())
        p, _, w = d_sw.samples[i]
        if (p.name, tuple(w)) in p3_index:
            ok &= torch.equal(d_p3[p3_index[(p.name, tuple(w))]]["mask"], m[1].long())
        else:
            ok &= bool((m[1] == -1).all())
        if not ok:
            bad.append(ref.names[i])
    out["wf_default_vs_evaluator"] = dict(windows=len(ref), flood_p3_windows=len(d_p3), mismatches=bad)
    print("WF default/flood_p3 vs evaluator:", len(bad), "mismatches", flush=True)

    # 1b. STURM default / flood_p3 vs the surface-water evaluator, 512 and 128 px
    for size in (512, 128):
        ref = ev.STURMSet(size)
        d_sw = new.STURMFloodDataset(ST, sensor="S2", img_size=size, normalize="original")
        d_p3 = new.STURMFloodDataset(ST, sensor="S2", img_size=size, normalize="original", protocol="flood_p3")
        bad = []
        for i in range(len(ref)):
            x, m, _ = ref[i]
            a_, b_ = d_sw[i], d_p3[i]
            if not (torch.equal(a_["image"][ev.STURM_SEL], x) and torch.equal(a_["mask"], m[0].long())
                    and torch.equal(b_["mask"], m[1].long())):
                bad.append(ref.names[i])
        out[f"sturm{size}_default_vs_evaluator"] = dict(chips=len(ref), mismatches=bad)
        print(f"STURM {size} default/flood_p3 vs evaluator:", len(bad), "mismatches", flush=True)

    # 2. old keyword arguments == loader at --old_rev
    cases = [
        ("wf", dict(permanent_water_as="ignore")),
        ("wf", dict(permanent_water_as="background")),
        ("wf", dict(ignore_permanent_water=True)),
        ("wf", dict(ignore_permanent_water=False)),
        ("sturm", dict(class_mapping="paper")),
        ("sturm", dict(class_mapping="published")),
        ("sturm", dict(class_mapping="operational")),
    ]
    for name, kw in cases:
        if name == "wf":
            mk = lambda m: m.WorldFloodsDataset(WF, split="test", img_size=512, tile_size=512, normalize="s11", **kw)
        else:
            mk = lambda m: m.STURMFloodDataset(ST, sensor="S2", img_size=512, normalize="original", **kw)
        o, n = mk(old), mk(new)
        assert o.samples == n.samples
        idx = range(0, len(o), a.legacy_stride)
        bad = [i for i in idx if not same(o[i], n[i])]
        key = f"legacy_{name}_" + "_".join(f"{k}={v}" for k, v in kw.items())
        out[key] = dict(checked=len(idx), total=len(o), mismatches=bad)
        print(key, len(idx), "checked,", len(bad), "mismatches", flush=True)

    out["old_rev"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", a.old_rev],
                                    capture_output=True, text=True).stdout.strip()
    out["all_pass"] = all(not v["mismatches"] for v in out.values() if isinstance(v, dict))
    (HERE / "loader_protocol_check.json").write_text(json.dumps(out, indent=1))
    print("ALL PASS" if out["all_pass"] else "FAILURES", flush=True)


if __name__ == "__main__":
    main()
