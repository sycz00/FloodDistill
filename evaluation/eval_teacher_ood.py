# !/usr/bin/env python3
"""Evaluate the Prithvi teacher on S11 test, STURM, FloodPlanet and WorldFloods.

    python evaluation/eval_teacher_ood.py
    python evaluation/eval_teacher_ood.py --only s11,sturm --gpu 0
"""
import argparse, json, os, sys
import torch
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

_ap = argparse.ArgumentParser()
_ap.add_argument("--only", type=str, default="s11,sturm,fp,wf,s1s2w,sff",
                  help="Comma-separated subset of {s11,sturm,fp,wf,s1s2w,sff} to run")
_ap.add_argument("--ckpt", type=str, default="hf",
                  help="'hf' (default): public Prithvi-EO-2.0-300M Sen1Floods11 checkpoint; "
                       "or a local .pt")
_ap.add_argument("--gpu", type=int, default=1)
_ap.add_argument("--out", type=str, default=None,
                  help="Output JSON (default runs/teacher_eval_results.json; with 'percentile' "
                       "runs/teacher_eval_results_STURM_PERCENTILE.json)")
_ap.add_argument("--sturm_normalize", type=str, default="original",
                  choices=["percentile", "clip", "original"],
                  help="STURM normalisation: 'original' (default, paper protocol) or "
                       "'percentile' (older protocol, separate file)")
# Robustness perturbations (apply to STURM S2 and WF, in reflectance space before norm).
_ap.add_argument("--gain", type=float, default=1.0, help="E5 radiometric gain (1.0=off).")
_ap.add_argument("--offset", type=float, default=0.0, help="E5 additive offset (0.0=off).")
_ap.add_argument("--haze", type=float, default=0.0,
                  help="Haze: additive wavelength-weighted path radiance (Makarau 2014; 0.0=off).")
_ap.add_argument("--wf_size", type=int, default=256,
                  help="WF img_size AND tile_size (use 512 to match the student eval — the "
                       "default 256 is the legacy teacher setting, a resolution-parity gap).")
_ap.add_argument("--tag", type=str, default="",
                  help="Suffix appended to the output filename (before .json) for sweeps.")
_cli = _ap.parse_args()
_SELECTED = set(_cli.only.split(","))
if _cli.out is not None:
    _OUT = _cli.out
elif _cli.sturm_normalize == "percentile":
    _OUT = "runs/teacher_eval_results_STURM_PERCENTILE.json"
else:
    _OUT = "runs/teacher_eval_results.json"
if _cli.tag:
    _OUT = _OUT.replace(".json", f"{_cli.tag}.json")

from data.sen1floods11 import get_dataloaders
from data.ood_datasets import (
    get_sturm_flood_dataloader, get_floodplanet_dataloader, get_worldfloods_dataloader,
    get_s1s2water_dataloader, get_senforflood_dataloader,
)
from data.channel_adapter import sturm_s2_adapter
from models.teacher import build_teacher
from training.trainer import Trainer
from config import TrainConfig

# Paths
# Dataset roots: $DATA_ROOT/<dataset>, see utils/paths.py and docs/DATA_LOCATION.md.
from utils.paths import dataset_root as _root

TEACHER_CKPT = Path(_cli.ckpt)
DATA_ROOT    = _root("S11_ROOT")
STURM_ROOT   = _root("STURM_ROOT")
FP_ROOT      = _root("FP_ROOT")
WF_ROOT      = _root("WF_ROOT")
S1S2W_ROOT   = _root("S1S2W_ROOT")
SFF_ROOT     = _root("SFF_ROOT")
GPU          = _cli.gpu
DEVICE       = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
RESULTS      = {}
BATCH_SIZE   = 4   # teacher is large - smaller batch
NUM_WORKERS  = 4

def make_cfg():
    return TrainConfig(ignore_index=-1, gpu_ids=[GPU], out_dir="/tmp/dummy", use_amp=False)

def evaluate(model, dl, adapter=None):
    class AdaptedDL:
        def __init__(self, dl, adapter):
            self.dl, self.adapter = dl, adapter
        def __iter__(self):
            for batch in self.dl:
                batch = dict(batch)
                batch["image"] = self.adapter(batch["image"].to(DEVICE))
                yield batch
        def __len__(self): return len(self.dl)
        @property
        def dataset(self): return self.dl.dataset

    if adapter is not None:
        dl = AdaptedDL(dl, adapter)
    cfg = make_cfg()
    trainer = Trainer(model=model, train_dl=dl, val_dl=dl, test_dl=dl,
                      cfg=cfg, num_classes=2, class_names=["background", "flood"])
    return trainer.evaluate(model, dl)

def fmt(m):
    iou  = m["mIoU"]
    fiou = m["iou_per_class"][1]
    return f"mIoU={iou:.4f}  flood_IoU={fiou:.4f}"

print(f"Loading teacher from {TEACHER_CKPT} ...")
_official = str(TEACHER_CKPT) == "hf"
teacher = build_teacher(
    variant="prithvi_eo_v2_300_tl",
    num_classes=2,
    pretrained=False,
    freeze=True,
    official_ckpt="hf" if _official else None,
)
if not _official:
    payload = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    teacher.load_state_dict(state, strict=True)
teacher = teacher.to(DEVICE)
teacher.eval()
print("Teacher loaded.\n")

# S11 test
if "s11" in _SELECTED:
    print("[S11] Loading dataloaders ...")
    _, _, test_dl = get_dataloaders(str(DATA_ROOT), batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS, eval_img_size=512)
    print("[S11] Evaluating ...", end=" ", flush=True)
    m = evaluate(teacher, test_dl)
    print(fmt(m))
    RESULTS["s11_test"] = m

# STURM
if "sturm" in _SELECTED:
    print(f"[STURM] Loading dataloader (normalize={_cli.sturm_normalize}) ...")
    sturm_dl = get_sturm_flood_dataloader(str(STURM_ROOT), sensor="S2",
                                          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=512,
                                          normalize=_cli.sturm_normalize,
                                          sim_gain=_cli.gain, sim_offset=_cli.offset, sim_haze=_cli.haze)
    adapter = sturm_s2_adapter(target_channels=6).to(DEVICE)
    print("[STURM] Evaluating ...", end=" ", flush=True)
    m = evaluate(teacher, sturm_dl, adapter=adapter)
    print(fmt(m))
    RESULTS["sturm_s2"] = m

# FloodPlanet
if "fp" in _SELECTED:
    print("[FP] Loading dataloader ...")
    fp_dl = get_floodplanet_dataloader(str(FP_ROOT), batch_size=BATCH_SIZE,
                                       num_workers=NUM_WORKERS, img_size=512)
    print("[FP] Evaluating ...", end=" ", flush=True)
    m = evaluate(teacher, fp_dl)
    print(fmt(m))
    RESULTS["fp"] = m

# WorldFloods
if "wf" in _SELECTED:
    print("[WF] Loading dataloader ...")
    wf_dl = get_worldfloods_dataloader(str(WF_ROOT), split="test", batch_size=BATCH_SIZE,
                                       num_workers=NUM_WORKERS, img_size=_cli.wf_size, tile_size=_cli.wf_size,
                                       sim_gain=_cli.gain, sim_offset=_cli.offset, sim_haze=_cli.haze)
    print("[WF] Evaluating ...", end=" ", flush=True)
    m = evaluate(teacher, wf_dl)
    print(fmt(m))
    RESULTS["wf"] = m

# S1S2-Water (optional)
if "s1s2w" in _SELECTED:
    if S1S2W_ROOT.exists():
        print("[S1S2W] Loading dataloader ...")
        s1s2w_dl = get_s1s2water_dataloader(str(S1S2W_ROOT), batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS, img_size=512)
        print("[S1S2W] Evaluating ...", end=" ", flush=True)
        m = evaluate(teacher, s1s2w_dl)
        print(fmt(m))
        RESULTS["s1s2w"] = m
    else:
        print("[S1S2W] Skipped (not found at", S1S2W_ROOT, ")")

# SenForFlood (optional)
if "sff" in _SELECTED:
    if SFF_ROOT.exists():
        print("[SFF] Loading dataloader ...")
        sff_dl = get_senforflood_dataloader(str(SFF_ROOT), batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS, img_size=512)
        print("[SFF] Evaluating ...", end=" ", flush=True)
        m = evaluate(teacher, sff_dl)
        print(fmt(m))
        RESULTS["sff"] = m
    else:
        print("[SFF] Skipped (not found at", SFF_ROOT, ")")

out_path = Path(_OUT)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({"ckpt": str(TEACHER_CKPT), **RESULTS}, indent=2, default=str))
print(f"\n[done] → {out_path}")
