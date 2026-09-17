"""Evaluate trained runs on Sen1Floods11 and external flood benchmarks.

Current STURM normalize="original" results go to eval_results.json;
the legacy percentile protocol goes to eval_results_STURM_PERCENTILE.json.
Historical source-data filenames use a different convention; see
paper_source_data/README.md. New run families need --runs_reg and --n_filter
to bypass the independent approach and budget filters.
"""
import argparse, json, os, re, sys
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

# Disable wandb before any import that might auto-init it
os.environ["WANDB_MODE"] = "disabled"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from data.sen1floods11 import get_dataloaders
from data.ood_datasets import (
    get_sturm_flood_dataloader, get_floodplanet_dataloader, get_worldfloods_dataloader,
    get_s1s2water_dataloader, get_senforflood_dataloader,
)
from data.channel_adapter import sturm_s2_adapter
from models.student import build_student
from training.trainer import Trainer
from config import TrainConfig

# Config
# Dataset roots: $DATA_ROOT/<dataset>, see utils/paths.py and docs/DATA_LOCATION.md.
from utils.paths import dataset_root as _root

RUNS_BASE    = Path("runs/ablations")
DATA_ROOT    = _root("S11_ROOT")
STURM_ROOT   = _root("STURM_ROOT")
FP_ROOT      = _root("FP_ROOT")
WF_ROOT      = _root("WF_ROOT")
S1S2W_ROOT   = _root("S1S2W_ROOT")
SFF_ROOT     = _root("SFF_ROOT")
GPU          = 1
DEVICE       = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 8
NUM_WORKERS  = 4

# Architecture detection
def detect_arch(run_name: str, arch_override: str | None = None) -> str:
    # Match "unet" as a whole token: sen_ce_unet_* is a UNet student, but in bridge_ofa_unet4_*
    # the UNet is the teacher and the student is EfficientViT.
    if re.search(r"(^|_)unet(50)?(_resnet\d+)?(_|$)", run_name):
        return "unet_resnet50"
    if arch_override:
        return arch_override
    if "lrelu6" in run_name:          # must come before "relu6" check (lrelu6 contains relu6)
        return "efficientvit_b0_lrelu6"
    if "relu6" in run_name:
        return "efficientvit_b0_relu6"
    if "tanh" in run_name:
        return "efficientvit_b0_tanh"
    return "efficientvit_b0"

def eval_img_size(arch: str) -> int:
    return 512  # both architectures are evaluated at 512

# Dummy trainer config
def make_dummy_cfg() -> TrainConfig:
    return TrainConfig(
        ignore_index=-1,
        gpu_ids=[GPU],
        out_dir="/tmp/dummy_eval",
        use_amp=False,
    )

# Load model
def load_model(ckpt_path: Path, arch: str) -> torch.nn.Module:
    model = build_student(arch, num_classes=2, pretrained=False)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # trainer.py saves {"model": ...}; quantization/qat.py strip_qat saves {"student": ...}.
    state = payload.get("model", payload.get("student", payload))
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model

# Evaluate on a dataloader
def evaluate_dl(model, dl, adapter=None) -> dict:
    """Evaluate model on dl. Pass adapter for datasets with mismatched channel counts."""
    cfg = make_dummy_cfg()

    if adapter is not None:
        # adapt the images of every batch before the model sees them
        class AdaptedDL:
            def __init__(self, dl, adapter):
                self.dl = dl
                self.adapter = adapter
            def __iter__(self):
                for batch in self.dl:
                    batch = dict(batch)
                    batch["image"] = self.adapter(batch["image"].to(DEVICE))
                    yield batch
            def __len__(self):
                return len(self.dl)
            @property
            def dataset(self):
                return self.dl.dataset

        dl = AdaptedDL(dl, adapter)

    trainer = Trainer(
        model=model, train_dl=dl, val_dl=dl, test_dl=dl,
        cfg=cfg, num_classes=2, class_names=["background", "flood"],
    )
    return trainer.evaluate(model, dl)

# Teacher reference metrics (for the delta-to-teacher columns)
def load_teacher_refs(sturm_normalize: str) -> dict:
    """Load the teacher's OOD metrics for the same protocol if available; missing files give no delta columns.

    STURM: runs/teacher_eval_results[_STURM_PERCENTILE].json. WorldFloods flood-only:
    runs/teacher_wf_v2_floodonly.json. WorldFloods older protocol: runs/wf_v2_full_eval.json.
    """
    refs: dict[str, float] = {}
    p = Path("runs/teacher_eval_results_STURM_PERCENTILE.json"
             if sturm_normalize == "percentile" else "runs/teacher_eval_results.json")
    if p.exists():
        s = json.loads(p.read_text()).get("sturm_s2", {})
        if "mIoU" in s:
            refs["sturm_mIoU"] = s["mIoU"]
            refs["sturm_flood_IoU"] = s.get("iou_per_class", [None, None])[1]
    q = Path("runs/teacher_wf_v2_floodonly.json")
    if q.exists():
        d = json.loads(q.read_text())
        refs["wf_fo_mIoU"] = d.get("mIoU")
        refs["wf_fo_flood_IoU"] = d.get("flood_IoU", d.get("water_IoU"))
    q2 = Path("runs/wf_v2_full_eval.json")
    if q2.exists():
        d = json.loads(q2.read_text())
        t = ((d.get("per_seed", d)).get("Teacher (Prithvi)") or [{}])[0]
        refs["wf_mIoU"] = t.get("mIoU")
        refs["wf_flood_IoU"] = t.get("water_IoU")
    return {k: v for k, v in refs.items() if v is not None}


# Parse run name into group/N
def parse_group(run: str):
    """Return (group_label, N_or_None) for summary grouping."""
    if run.startswith("baseline_evit_b0"):
        return "baseline_evit_b0", None
    if run.startswith("baseline_unet50"):
        return "baseline_unet50", None
    if run.startswith("logit_kd"):
        return "logit_kd", None
    if run.startswith("dfkd_labeled_pseudo"):
        return "dfkd_labeled_pseudo", None
    if run.startswith("dfkd_pseudo_only"):
        return "dfkd_pseudo_only", None
    if re.match(r"dfkd_full_s", run):
        return "dfkd_full", None
    if re.match(r"ssfd_full_s", run):
        return "ssfd_full", None
    m = re.match(r"(dfkd|ssfd|ofa)_N(\d+)_s", run)
    if m:
        return m.group(1) + "_N", int(m.group(2))
    # All named variants; longest prefixes must come first to avoid partial matches
    m = re.match(
        r"(ofa_base_pcgrad|ofa_new_pcgrad|ofa_new"
        r"|spld_new_gated_pcgrad|spld_new_gated|spld_new|spld_pcgrad"
        r"|spld_tvar_pcgrad|spld_tvar"
        r"|dfkd_new|pseudo_sup)_N(\d+)_s",
        # r"|sotd_branch_replace|sotd_branch|sotd_replace|sotd"
        # r"|sppm_rep|sppm_br|sppm"
        # r"|gradmatch|ofa_edt"
        # r"|dlpb_c|dlpb_b|dlpb|coupled_c|coupled_b)_N(\d+)_s",
        run,
    )
    if m:
        return m.group(1) + "_N", int(m.group(2))
    m = re.match(r"ofa_full_s", run)
    if m:
        return "ofa_full", None
    m = re.match(r"ssfd_gradalign_n_(\d+)_s", run)
    if m:
        return "ssfd_gradalign_N", int(m.group(1))
    # Current naming, written by experiments/*.sh and docker/entrypoint.py:
    #   <prefix>_<arm>[_<tag>]_N<n>_s<seed>   e.g. sen_ofa_N252_s42, pseudo_ofa_aug_N2500_s42
    m = re.match(r"(.+?)_N(\d+)_s\d+$", run)
    if m:
        return m.group(1) + "_N", int(m.group(2))
    return run, None

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",        action="store_true", help="Re-evaluate even if cached")
    parser.add_argument("--skip-ood",     action="store_true", help="Skip STURM, FloodPlanet, WorldFlood, S1S2-Water, SenForFlood")
    parser.add_argument("--only_sturm",   action="store_true",
                        help="Fast path: evaluate STURM only, skip S11 (train/val/test) and all other OOD sets")
    parser.add_argument("--recompute_ood",action="store_true",
                        help="Recompute only FP and WF for cached runs (S11+STURM kept). "
                             "Use after fixing OOD normalization without re-running S11.")
    parser.add_argument("--run",          type=str, default=None, help="Evaluate a single run only")
    parser.add_argument("--runs_reg",  type=str, default=None, help="Filter runs whose name contains this substring")
    parser.add_argument("--n_filter",  type=int, default=None, help="Only evaluate runs with _N{n}_ in their name")
    parser.add_argument("--runs_base", type=str, default=None, help="Override runs base directory")
    parser.add_argument("--arch",      type=str, default=None, help="Force architecture for all runs")
    parser.add_argument("--gpu",       type=int, default=None, help="Override CUDA device index (default: GPU constant)")
    parser.add_argument("--sff_max_samples", type=int, default=None,
                        help="Limit SenForFlood to this many patches (deterministic). "
                             "2000 is enough for stable mIoU; use to speed up batch evaluation.")
    parser.add_argument("--sturm_normalize", type=str, default="original",
                        choices=["percentile", "clip", "original"],
                        help="STURM normalisation: 'original' (default, paper protocol, "
                             "eval_results.json) or 'percentile' (older protocol, "
                             "eval_results_STURM_PERCENTILE.json)")
    parser.add_argument("--sturm_gain", type=float, default=1.0,
                        help="Multiply STURM reflectance by this before normalisation (1.0 = off; "
                             "needs --sturm_normalize original)")
    parser.add_argument("--sturm_offset", type=float, default=0.0,
                        help="E5 radiometric robustness: additive reflectance offset applied "
                             "before the S11 normalization (0.0 = off).")
    parser.add_argument("--sturm_haze", type=float, default=0.0,
                        help="Additive haze, blue-band strength in reflectance (0.0 = off; needs "
                             "--sturm_normalize original)")
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Suffix for the result file name, e.g. one per perturbation level")
    parser.add_argument("--skip_s1s2w", action="store_true",
                        help="Skip S1S2-Water (not used in the paper)")
    parser.add_argument("--skip_wf", action="store_true",
                        help="Skip the WorldFloods-v2 evaluation (it is ON by default for "
                             "full evals; --only_sturm and --skip-ood also skip it).")
    parser.add_argument("--wf_include_permanent", action="store_true",
                        help="Count JRC permanent water as water on WorldFloods (keys wf_*); "
                             "default is flood-only (keys wf_fo_*)")
    args = parser.parse_args()

    # Allow GPU override via CLI (modifies module-level GPU/DEVICE used by load_model etc.)
    if args.gpu is not None:
        import sys as _sys
        _mod = _sys.modules[__name__]
        _mod.GPU    = args.gpu
        _mod.DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    runs_base = Path(args.runs_base) if args.runs_base else RUNS_BASE
    runs = sorted(runs_base.iterdir()) if not args.run else [runs_base / args.run]

    runs = [r for r in runs if r.is_dir()]
    _n_found = len(runs)
    if args.runs_reg:
        runs = [r for r in runs if args.runs_reg in r.name]
    if args.n_filter is not None:
        runs = [r for r in runs if f"_N{args.n_filter}_" in r.name]

    # WF cache-key prefix: legacy protocol keeps the historical wf_* keys; the
    # flood-only protocol writes wf_fo_* so the two never mix in one cache file.
    wfk = "wf" if args.wf_include_permanent else "wf_fo"
    teacher_refs = load_teacher_refs(args.sturm_normalize)
    if teacher_refs:
        print("[setup] teacher refs: " +
              "  ".join(f"{k}={v:.4f}" for k, v in sorted(teacher_refs.items())))

    # N values for evaluation and summary; runs without _N{n}_ in the name always pass.
    common_n_order = [20,40,60, 80 ,50, 100 , 150, 200, 250, 252, 300, 350, 400, 450, 500, 2500, 4000 ,5000]

    def _run_n(name: str):
        m = re.search(r'_N(\d+)_', name)
        return int(m.group(1)) if m else None

    # An explicit --n_filter is always kept.
    if args.n_filter is None:
        _before = len(runs)
        runs = [r for r in runs if _run_n(r.name) is None or _run_n(r.name) in common_n_order]
        _dropped_n = _before - len(runs)
    else:
        _dropped_n = 0

    # Approach whitelist (bypass when --runs_reg or --run is set explicitly)
    ALLOWED_APPROACHES = {
        "dfkd_new", "ofa_new", "spld_new", "spld_pcgrad",
        "ofa_new_pcgrad", "pseudo_sup", "spld_new_gated", "spld_new_gated_pcgrad",
        "spld_tvar", "spld_tvar_pcgrad",
        # sec. 45/50/51/52: lrelu6 and continuous-proxy alpha sweep
        "spld_pcgrad_cont", "spld_pcgrad_lrelu6", "spld_pcgrad_lrelu6_cont",
        "spld_pcgrad_cont_a2", "spld_pcgrad_cont_a5",
        "spld_pcgrad_cont_gated", "spld_pcgrad_cont_a2gated",
    }

    def _run_approach(name: str):
        m = re.match(r"(.+?)_N\d+_s", name)
        return m.group(1) if m else None

    _dropped_approach = 0
    if not args.runs_reg and not args.run:
        _before = len(runs)
        runs = [r for r in runs if _run_approach(r.name) in ALLOWED_APPROACHES]
        _dropped_approach = _before - len(runs)

    # Stop with a message if the filters above removed every run.
    if not runs:
        msg = [f"no runs to evaluate under {runs_base} ({_n_found} directories found)"]
        if _dropped_n:
            msg.append(f"  {_dropped_n} dropped by common_n_order - their _N<n>_ is not in "
                       f"that list. Add the budget, or pass --n_filter <n>.")
        if _dropped_approach:
            msg.append(f"  {_dropped_approach} dropped by ALLOWED_APPROACHES - the run-name "
                       f"prefix before _N<n>_s<seed> is not a known approach. Pass "
                       f"--runs_reg <substring> (or --run <name>) to bypass that whitelist.")
        if not _dropped_n and not _dropped_approach:
            msg.append("  the --run / --runs_reg / --n_filter filters matched nothing.")
        sys.exit("\n".join(msg))

    # Build S11 dataloaders once
    if not args.only_sturm:
        print("[setup] Loading S11 dataloaders (img_size=512) …")
        train_dl, val_dl, test_dl = get_dataloaders(
            data_root=str(DATA_ROOT), batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, eval_img_size=512,
        )
    else:
        print("[setup] --only_sturm: skipping S11 dataloaders")
        train_dl = val_dl = test_dl = None

    # Build OOD dataloaders once
    if not args.skip_ood:
        print(f"[setup] Loading STURM S2 dataloader (normalize={args.sturm_normalize}) …")
        sturm_dl = get_sturm_flood_dataloader(
            root=str(STURM_ROOT), sensor="S2", batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, img_size=512, normalize=args.sturm_normalize,
            sim_gain=args.sturm_gain, sim_offset=args.sturm_offset, sim_haze=args.sturm_haze,
        )
        if args.sturm_gain != 1.0 or args.sturm_offset != 0.0:
            print(f"[setup] E5 perturbation: reflectance -> {args.sturm_gain}*x + {args.sturm_offset}")
        if args.sturm_haze != 0.0:
            print(f"[setup] HAZE perturbation: reflectance += {args.sturm_haze}*w_band (blue→SWIR)")
        # STURM S2 has 9 bands; select the 6 that match S11 training bands
        sturm_adapter = sturm_s2_adapter(target_channels=6).to(DEVICE)

        # FloodPlanet - disabled for now; re-enable by restoring the loader below
        fp_dl = None
        # fp_dl = get_floodplanet_dataloader(
        # root=str(FP_ROOT), batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS, img_size=512,
        # )

        # WorldFloods-v2 in 512 px tiles; flood-only by default (keys wf_fo_*),
        # --wf_include_permanent for the older scoring (keys wf_*).
        wf_dl = None
        if not args.skip_wf and not args.only_sturm and WF_ROOT.exists():
            print("[setup] Loading WorldFloods-v2 dataloader "
                  f"({'legacy flood+permanent' if args.wf_include_permanent else 'FLOOD-ONLY'}) …")
            wf_dl = get_worldfloods_dataloader(
                root=str(WF_ROOT), split="test", batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS, img_size=512, tile_size=512, normalize="s11",
                ignore_permanent_water=not args.wf_include_permanent,
            )

        s1s2w_dl = None
        if S1S2W_ROOT.exists() and not args.only_sturm and not args.skip_s1s2w:
            print("[setup] Loading S1S2-Water dataloader …")
            s1s2w_dl = get_s1s2water_dataloader(
                root=str(S1S2W_ROOT), batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS, img_size=512,
            )

        # SenForFlood - disabled for now (bad OOD scores, enable later)
        sff_dl = None

    all_results = {}

    for run_dir in runs:
        run = run_dir.name
        ckpt = run_dir / "best.pt"
        # QAT run directories have no best.pt. model_qat_stripped.pt scores about 0.16 mIoU lower
        # in float; evaluate the QDQ ONNX (quantization/onnx_qdq.py) or the engine (trt_eval.py).
        if not ckpt.exists() and (run_dir / "model_qat_stripped.pt").exists():
            print(f"[skip]   {run}  — QAT run: evaluate model_qdq_trt_no_out.onnx via "
                  f"quantization/onnx_qdq.py::eval_onnx, not the stripped float weights")
            continue
        cache_name = ("eval_results_STURM_PERCENTILE.json"
                      if args.sturm_normalize == "percentile" else "eval_results.json")
        if args.out_suffix:
            cache_name = cache_name.replace(".json", f"{args.out_suffix}.json")
        cache = run_dir / cache_name

        if not ckpt.exists():
            print(f"[miss]   {run}  — no best.pt")
            continue

        if cache.exists() and not args.force:
            with open(cache) as f:
                r = json.load(f)

            # --recompute_ood: add missing OOD metrics, keep S11 + STURM.
            # Skips any dataset whose metrics are already present in the cache.
            if args.recompute_ood and not args.skip_ood:
                need_fp        = fp_dl    is not None and "fp_mIoU"    not in r
                need_wf        = wf_dl    is not None and f"{wfk}_mIoU" not in r
                need_s1s2w     = s1s2w_dl is not None and "s1s2w_mIoU" not in r
                need_sff       = sff_dl   is not None and "sff_mIoU"   not in r
                need_sturm_pa  = "sturm_pixel_acc" not in r

                if not any([need_fp, need_wf, need_s1s2w, need_sff, need_sturm_pa]):
                    print(f"[cached-ood] {run:<40}  all OOD metrics present, skipping")
                else:
                    arch = detect_arch(run, arch_override=args.arch)
                    try:
                        model = load_model(ckpt, arch)
                        def _flood_iou(m): return m["iou_per_class"][1]
                        def _flood_f1(m):  return m["f1_per_class"][1]

                        if need_sturm_pa:
                            print(f"[recompute_ood] {run} STURM pixel_acc …", end=" ", flush=True)
                            mspa = evaluate_dl(model, sturm_dl, adapter=sturm_adapter)
                            r["sturm_pixel_acc"] = round(mspa.get("pixel_acc", float("nan")), 6)
                            print(f"pixel_acc={mspa.get('pixel_acc', float('nan')):.4f}")

                        if need_fp:
                            print(f"[recompute_ood] {run} FP …", end=" ", flush=True)
                            mf = evaluate_dl(model, fp_dl)
                            r["fp_mIoU"]      = round(mf["mIoU"], 6)
                            r["fp_flood_IoU"] = round(_flood_iou(mf), 6)
                            r["fp_f1"]        = round(mf.get("f1", float("nan")), 6)
                            r["fp_flood_f1"]  = round(_flood_f1(mf), 6)
                            print(f"mIoU={mf['mIoU']:.4f}", end="  ")

                        if need_wf:
                            print(f"[recompute_ood] {run} WF({wfk}) …", end=" ", flush=True)
                            mw = evaluate_dl(model, wf_dl)
                            r[f"{wfk}_mIoU"]      = round(mw["mIoU"], 6)
                            r[f"{wfk}_flood_IoU"] = round(_flood_iou(mw), 6)
                            r[f"{wfk}_f1"]        = round(mw.get("f1", float("nan")), 6)
                            r[f"{wfk}_flood_f1"]  = round(_flood_f1(mw), 6)
                            print(f"mIoU={mw['mIoU']:.4f}")

                        if need_s1s2w:
                            print(f"[recompute_ood] {run} S1S2W …", end=" ", flush=True)
                            ms2 = evaluate_dl(model, s1s2w_dl)
                            r["s1s2w_mIoU"]      = round(ms2["mIoU"], 6)
                            r["s1s2w_flood_IoU"] = round(_flood_iou(ms2), 6)
                            r["s1s2w_f1"]        = round(ms2.get("f1", float("nan")), 6)
                            r["s1s2w_flood_f1"]  = round(_flood_f1(ms2), 6)
                            print(f"mIoU={ms2['mIoU']:.4f}")

                        if need_sff:
                            print(f"[recompute_ood] {run} SFF …", end=" ", flush=True)
                            msf = evaluate_dl(model, sff_dl)
                            r["sff_mIoU"]      = round(msf["mIoU"], 6)
                            r["sff_flood_IoU"] = round(_flood_iou(msf), 6)
                            r["sff_f1"]        = round(msf.get("f1", float("nan")), 6)
                            r["sff_flood_f1"]  = round(_flood_f1(msf), 6)
                            print(f"mIoU={msf['mIoU']:.4f}")

                        with open(cache, "w") as f:
                            json.dump(r, f, indent=2)
                        del model
                        torch.cuda.empty_cache()
                    except Exception as e:
                        import traceback
                        print(f"  ERROR: {e}")
                        traceback.print_exc()

            all_results[run] = r
            def _fmt4(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            s11_t  = r.get("s11_test_mIoU", "?")
            s11_f  = r.get("s11_test_flood_IoU", "?")
            sturm  = r.get("sturm_mIoU", "?")
            s1s2w  = r.get("s1s2w_mIoU", "?")
            if not args.recompute_ood:
                print(f"[cached] {run:<40}  S11_test={_fmt4(s11_t)}  flood={_fmt4(s11_f)}  STURM={_fmt4(sturm)}  S1S2W={_fmt4(s1s2w)}")
            continue

        print(f"\n[eval]   {run} …", flush=True)
        arch = detect_arch(run, arch_override=args.arch)
        esz  = eval_img_size(arch)

        try:
            model = load_model(ckpt, arch)
            r = {}

            def _flood_iou(m): return m["iou_per_class"][1]
            def _flood_f1(m):  return m["f1_per_class"][1]

            # S11: train / val / test
            s11_splits = [] if args.only_sturm else [("train", train_dl), ("val", val_dl), ("test", test_dl)]
            for split_name, dl in s11_splits:
                print(f"         S11 {split_name} …", end=" ", flush=True)
                m = evaluate_dl(model, dl)
                r[f"s11_{split_name}_mIoU"]     = round(m["mIoU"], 6)
                r[f"s11_{split_name}_flood_IoU"] = round(_flood_iou(m), 6)
                r[f"s11_{split_name}_pixel_acc"] = round(m.get("pixel_acc", float("nan")), 6)
                r[f"s11_{split_name}_f1"]        = round(m.get("f1", float("nan")), 6)
                r[f"s11_{split_name}_flood_f1"]  = round(_flood_f1(m), 6)
                print(f"mIoU={m['mIoU']:.4f}  flood={_flood_iou(m):.4f}")

            # OOD
            if not args.skip_ood:
                print("         STURM …", end=" ", flush=True)
                ms = evaluate_dl(model, sturm_dl, adapter=sturm_adapter)
                r["sturm_mIoU"]       = round(ms["mIoU"], 6)
                r["sturm_flood_IoU"]  = round(_flood_iou(ms), 6)
                r["sturm_pixel_acc"]  = round(ms.get("pixel_acc", float("nan")), 6)
                r["sturm_f1"]         = round(ms.get("f1", float("nan")), 6)
                r["sturm_flood_f1"]   = round(_flood_f1(ms), 6)
                dt = ""
                if "sturm_mIoU" in teacher_refs:
                    dt = f"  Δteacher={ms['mIoU'] - teacher_refs['sturm_mIoU']:+.4f}"
                print(f"mIoU={ms['mIoU']:.4f}  flood={_flood_iou(ms):.4f}{dt}")

                if fp_dl is not None:
                    print("         FP …",    end=" ", flush=True)
                    mf = evaluate_dl(model, fp_dl)
                    r["fp_mIoU"]      = round(mf["mIoU"], 6)
                    r["fp_flood_IoU"] = round(_flood_iou(mf), 6)
                    r["fp_f1"]        = round(mf.get("f1", float("nan")), 6)
                    r["fp_flood_f1"]  = round(_flood_f1(mf), 6)
                    print(f"mIoU={mf['mIoU']:.4f}  flood={_flood_iou(mf):.4f}")

                if wf_dl is not None:
                    print(f"         WF({wfk}) …", end=" ", flush=True)
                    mw = evaluate_dl(model, wf_dl)
                    r[f"{wfk}_mIoU"]      = round(mw["mIoU"], 6)
                    r[f"{wfk}_flood_IoU"] = round(_flood_iou(mw), 6)
                    r[f"{wfk}_f1"]        = round(mw.get("f1", float("nan")), 6)
                    r[f"{wfk}_flood_f1"]  = round(_flood_f1(mw), 6)
                    dt = ""
                    if f"{wfk}_flood_IoU" in teacher_refs:
                        dt = f"  Δteacher={_flood_iou(mw) - teacher_refs[f'{wfk}_flood_IoU']:+.4f}"
                    print(f"mIoU={mw['mIoU']:.4f}  flood={_flood_iou(mw):.4f}{dt}")

                if s1s2w_dl is not None:
                    print("         S1S2W …", end=" ", flush=True)
                    ms2 = evaluate_dl(model, s1s2w_dl)
                    r["s1s2w_mIoU"]      = round(ms2["mIoU"], 6)
                    r["s1s2w_flood_IoU"] = round(_flood_iou(ms2), 6)
                    r["s1s2w_f1"]        = round(ms2.get("f1", float("nan")), 6)
                    r["s1s2w_flood_f1"]  = round(_flood_f1(ms2), 6)
                    print(f"mIoU={ms2['mIoU']:.4f}  flood={_flood_iou(ms2):.4f}")

                if sff_dl is not None:
                    print("         SFF …",   end=" ", flush=True)
                    msf = evaluate_dl(model, sff_dl)
                    r["sff_mIoU"]      = round(msf["mIoU"], 6)
                    r["sff_flood_IoU"] = round(_flood_iou(msf), 6)
                    r["sff_f1"]        = round(msf.get("f1", float("nan")), 6)
                    r["sff_flood_f1"]  = round(_flood_f1(msf), 6)
                    print(f"mIoU={msf['mIoU']:.4f}  flood={_flood_iou(msf):.4f}")

            r["arch"] = arch
            with open(cache, "w") as f:
                json.dump(r, f, indent=2)
            all_results[run] = r

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Aggregate summary
    groups: dict[tuple, list] = defaultdict(list)
    for run, r in all_results.items():
        grp, N = parse_group(run)
        groups[(grp, N)].append(r)

    def agg(vals):
        vs = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vs:
            return None, None, 0
        return float(np.mean(vs)), float(np.std(vs, ddof=1) if len(vs) > 1 else 0.0), len(vs)

    print("\n" + "=" * 148)
    wf_lbl = "WF" if wfk == "wf" else "WFfo"
    if teacher_refs:
        print("Teacher refs: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(teacher_refs.items())))
        print("ΔT-ST = group STURM mIoU − teacher · ΔT-WF = group %s flood-IoU − teacher (protocol-matched)" % wf_lbl)
    header = ("%-30s %5s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %7s  %7s  n"
              % ("Group", "N", "S11tst", "S11fld", "S11val", "S11trn", "STURM", "STfld",
                 "FP", "FPfld", wf_lbl, wf_lbl + "fld", "ΔT-ST", "ΔT-WF"))
    print(header)
    print("-" * 148)
    summary = {}
    order = [
        ("baseline_evit_b0", None), ("baseline_unet50", None),
        ("logit_kd", None), ("dfkd_labeled_pseudo", None), ("dfkd_pseudo_only", None),
        ("dfkd_full", None), ("ssfd_full", None),
        *[("dfkd_N", n) for n in common_n_order],
        *[("ssfd_N", n) for n in common_n_order],
        *[("ssfd_gradalign_N", n) for n in common_n_order],
        *[("ofa_N", n) for n in common_n_order],
        *[("ofa_dw_N", n) for n in common_n_order],
        *[("ofa_uw_N", n) for n in common_n_order],
        *[("ofa_dm_N", n) for n in common_n_order],
        *[("ofa_rel_N", n) for n in common_n_order],
        ("ofa_full", None),
        *[("cakd_N", n) for n in common_n_order],
        *[("spld_N", n) for n in common_n_order],
        *[("sotd_N", n) for n in common_n_order],
        *[("sotd_branch_N", n) for n in common_n_order],
        *[("sotd_replace_N", n) for n in common_n_order],
        *[("sotd_branch_replace_N", n) for n in common_n_order],
        *[("sppm_N", n) for n in common_n_order],
        *[("sppm_br_N", n) for n in common_n_order],
        *[("sppm_rep_N", n) for n in common_n_order],
        *[("gradmatch_N", n) for n in common_n_order],
        *[("ofa_edt_N", n) for n in common_n_order],
        *[("spld_det_N", n) for n in common_n_order],
        *[("spld_pcgrad_N", n) for n in common_n_order],
        *[("ofa_base_pcgrad_N", n) for n in common_n_order],
        *[("ofa_new_N", n) for n in common_n_order],
        *[("ofa_new_pcgrad_N", n) for n in common_n_order],
        *[("dfkd_new_N", n) for n in common_n_order],
        *[("pseudo_sup_N", n) for n in common_n_order],
        *[("spld_new_N", n) for n in common_n_order],
        *[("spld_new_gated_N", n) for n in common_n_order],
        *[("spld_new_gated_pcgrad_N", n) for n in common_n_order],
        *[("spld_tvar_N", n) for n in common_n_order],
        *[("spld_tvar_pcgrad_N", n) for n in common_n_order],
        *[("dlpb_N", n) for n in common_n_order],
        *[("dlpb_b_N", n) for n in common_n_order],
        *[("dlpb_c_N", n) for n in common_n_order],
        *[("coupled_b_N", n) for n in common_n_order],
        *[("coupled_c_N", n) for n in common_n_order],
    ]
    # `order` is a legacy list of group names. Anything not in it used to vanish from the
    # summary without a word, which reads as "no results" rather than "not on my list".
    extra = sorted(set(groups) - set(order), key=lambda k: (str(k[0]), k[1] or 0))
    for key in list(order) + extra:
        if key not in groups:
            continue
        runs_list = groups[key]
        grp, N = key

        def col(metric):
            return agg([r.get(metric) for r in runs_list])

        t_m,  t_s,  n1 = col("s11_test_mIoU")
        tf_m, tf_s, _  = col("s11_test_flood_IoU")
        v_m,  v_s,  _  = col("s11_val_mIoU")
        tr_m, tr_s, _  = col("s11_train_mIoU")
        st_m, st_s, _  = col("sturm_mIoU")
        sf_m, sf_s, _  = col("sturm_flood_IoU")
        fp_m, fp_s, _  = col("fp_mIoU")
        ff_m, ff_s, _  = col("fp_flood_IoU")
        wf_m, wf_s, _  = col(f"{wfk}_mIoU")
        wff_m, wff_s, _ = col(f"{wfk}_flood_IoU")

        # delta to teacher, protocol-matched: STURM on mIoU, WF on flood-IoU
        dt_st = (st_m - teacher_refs["sturm_mIoU"]
                 if st_m is not None and "sturm_mIoU" in teacher_refs else None)
        dt_wf = (wff_m - teacher_refs[f"{wfk}_flood_IoU"]
                 if wff_m is not None and f"{wfk}_flood_IoU" in teacher_refs else None)

        def fmt(m, s):
            if m is None: return "  --- "
            return "%5.3f" % m if s == 0 else "%5.3f" % m

        def fmtd(d):
            return "   --- " if d is None else "%+6.3f" % d

        N_str = str(N) if N is not None else "all"
        print("%-30s %5s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %7s  %7s  %d"
              % (grp, N_str,
                 fmt(t_m, t_s), fmt(tf_m, tf_s),
                 fmt(v_m, v_s), fmt(tr_m, tr_s),
                 fmt(st_m, st_s), fmt(sf_m, sf_s),
                 fmt(fp_m, fp_s), fmt(ff_m, ff_s),
                 fmt(wf_m, wf_s), fmt(wff_m, wff_s),
                 fmtd(dt_st), fmtd(dt_wf),
                 n1))

        summary[f"{grp}@{N_str}"] = {
            "n": n1,
            "s11_test":  {"mean": t_m,  "std": t_s},
            "s11_test_flood": {"mean": tf_m, "std": tf_s},
            "s11_val":   {"mean": v_m,  "std": v_s},
            "s11_train": {"mean": tr_m, "std": tr_s},
            "sturm":     {"mean": st_m, "std": st_s},
            "sturm_flood": {"mean": sf_m, "std": sf_s},
            "fp":        {"mean": fp_m, "std": fp_s},
            "fp_flood":  {"mean": ff_m, "std": ff_s},
            wfk:         {"mean": wf_m, "std": wf_s},
            f"{wfk}_flood": {"mean": wff_m, "std": wff_s},
            "delta_teacher_sturm_mIoU": dt_st,
            f"delta_teacher_{wfk}_flood_IoU": dt_wf,
        }
    if teacher_refs:
        summary["_teacher_refs"] = teacher_refs

    out_path = runs_base / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary → {out_path}")


if __name__ == "__main__":
    main()
