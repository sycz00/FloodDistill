"""Compare teacher and student correctness on valid disagreement pixels.

For binary predictions exactly one model is correct on each disagreement.
A student share above 50% is a local advantage on this slice, not proof of
general denoising. Both models see the same deterministic STURM batches at
512x512 with normalize="original" and the six-band adapter.
Defaults: EfficientViT OFA/Logit-KD and UNet Logit-KD at N=2500, four seeds.
Use --student to specify another series. Writes runs/disagreement_audit.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.channel_adapter import sturm_s2_adapter
from data.ood_datasets import get_sturm_flood_dataloader
from models.student import build_student
from models.teacher import build_teacher
from utils.paths import dataset_root


# Configuration

NUM_CLASSES = 2
IN_CHANNELS = 6
EVAL_SIZE   = 512          # STURM 128 to 512, as in the paper OOD protocol
BATCH_SIZE  = 8
NUM_WORKERS = 4
SEEDS       = [42, 64, 91, 107]

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "runs", "disagreement_audit.json")

# default student series, label: (arch, run-dir template under --runs_root)
DEFAULT_STUDENTS = {
    "EvIT-OFA":     ("efficientvit_b0", "pseudo_ofa_aug_N2500_s{seed}"),
    "UNet-LogitKD": ("unet_resnet50",   "pseudo_dfkd_unet_aug_N2500_s{seed}"),
    "EvIT-LogitKD": ("efficientvit_b0", "pseudo_dfkd_aug_N2500_s{seed}"),
}


# Model loading

def load_student(arch: str, ckpt_path: str, device) -> torch.nn.Module:
    model = build_student(arch=arch, in_channels=IN_CHANNELS,
                          num_classes=NUM_CLASSES, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_teacher(ckpt_path: str, device) -> torch.nn.Module:
    model = build_teacher(num_classes=NUM_CLASSES, official_ckpt=ckpt_path, freeze=True)
    return model.to(device).eval()


# Per-pixel prediction at mask resolution

def predict(model, images, mask_hw) -> torch.Tensor:
    logits = model(images)
    if logits.shape[-2:] != mask_hw:
        logits = F.interpolate(logits, size=mask_hw, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


# Audit accumulator

class Audit:
    """Counts for one student series (accumulated over all seeds & pixels)."""

    def __init__(self) -> None:
        self.n_valid          = 0    # valid (labelled) pixels
        self.n_disagree       = 0    # valid pixels where student != teacher
        self.student_right     = 0   # of disagreements, student matches GT
        self.teacher_right     = 0   # of disagreements, teacher matches GT
        # Breakdown by GT class on disagreement pixels: who wins where.
        self.dis_gt_flood     = 0    # disagreement pixels whose GT is flood (1)
        self.dis_gt_bg        = 0    # disagreement pixels whose GT is background (0)
        # (on a GT=flood disagreement "student_right" means the student predicted flood)
        self.student_right_flood = 0
        self.student_right_bg    = 0
        # Overall agreement-with-GT (context, all valid pixels).
        self.student_correct_all = 0
        self.teacher_correct_all = 0

    def update(self, s_pred, t_pred, mask) -> None:
        valid    = mask >= 0
        disagree = (s_pred != t_pred) & valid
        s_ok     = s_pred == mask
        t_ok     = t_pred == mask

        self.n_valid    += int(valid.sum())
        self.n_disagree += int(disagree.sum())
        self.student_right += int((disagree & s_ok).sum())
        self.teacher_right += int((disagree & t_ok).sum())

        dis_flood = disagree & (mask == 1)
        dis_bg    = disagree & (mask == 0)
        self.dis_gt_flood += int(dis_flood.sum())
        self.dis_gt_bg    += int(dis_bg.sum())
        self.student_right_flood += int((dis_flood & s_ok).sum())
        self.student_right_bg    += int((dis_bg & s_ok).sum())

        self.student_correct_all += int((valid & s_ok).sum())
        self.teacher_correct_all += int((valid & t_ok).sum())

    def summary(self) -> dict:
        nd = max(self.n_disagree, 1)
        return {
            "n_valid_px":            self.n_valid,
            "n_disagree_px":         self.n_disagree,
            "disagree_rate":         round(self.n_disagree / max(self.n_valid, 1), 6),
            "student_right":         self.student_right,
            "teacher_right":         self.teacher_right,
            "student_right_rate":    round(self.student_right / nd, 6),
            "teacher_right_rate":    round(self.teacher_right / nd, 6),
            # Where the student's disagreements are, and how it does in each regime:
            "disagree_gt_flood":     self.dis_gt_flood,
            "disagree_gt_bg":        self.dis_gt_bg,
            # winning a GT=flood disagreement = recovered a flood pixel the teacher missed,
            # winning a GT=bg one = removed a teacher FP
            "student_recall_gain_rate": round(
                self.student_right_flood / max(self.dis_gt_flood, 1), 6),
            "student_fp_suppress_rate": round(
                self.student_right_bg / max(self.dis_gt_bg, 1), 6),
            "student_acc_all_px":    round(self.student_correct_all / max(self.n_valid, 1), 6),
            "teacher_acc_all_px":    round(self.teacher_correct_all / max(self.n_valid, 1), 6),
        }


# Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sturm_root", default=str(dataset_root("STURM_ROOT")))
    ap.add_argument("--teacher_ckpt", default="hf",
                    help="'hf' (default) is the paper's teacher: the public "
                         "Prithvi-EO-2.0-300M Sen1Floods11 checkpoint. A path loads a local one.")
    ap.add_argument("--runs_root",
                    default=os.path.join(ROOT, "runs", "nsweep"),
                    help="Directory the default run-dir templates are resolved against.")
    ap.add_argument("--student", nargs=3, action="append", default=None,
                    metavar=("LABEL", "ARCH", "TEMPLATE"),
                    help="Student run directory with {seed} placeholder (names without '/' "
                         "are relative to --runs_root); can be repeated")
    ap.add_argument("--sturm_normalize", default="original",
                    choices=["original", "percentile", "clip"],
                    help="STURM normalisation ('original' is the paper protocol)")
    ap.add_argument("--out", default=OUT_JSON, help="Where to write the audit JSON.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    if args.student:
        students = {lab: (arch, tmpl) for lab, arch, tmpl in args.student}
    else:
        students = DEFAULT_STUDENTS

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[audit] device={device}  eval_size={EVAL_SIZE}")

    loader  = get_sturm_flood_dataloader(
        root=args.sturm_root, sensor="S2",
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=EVAL_SIZE,
        normalize=args.sturm_normalize,
    )
    adapter = sturm_s2_adapter(target_channels=IN_CHANNELS).to(device)

    print(f"[audit] loading teacher: {args.teacher_ckpt}")
    teacher = load_teacher(args.teacher_ckpt, device)

    # results[label] = {"per_seed": {seed: summary}, "pooled": summary(all seeds)}
    results: dict = {}

    for label, (arch, tmpl) in students.items():
        print(f"\n{'='*64}\n[audit] {label}  ({arch})")
        pooled  = Audit()
        per_seed = {}

        for seed in args.seeds:
            run_dir = tmpl.format(seed=seed)
            if os.path.sep not in run_dir:
                run_dir = os.path.join(args.runs_root, run_dir)
            ckpt = os.path.join(run_dir, "best.pt")
            if not os.path.exists(ckpt):
                print(f"  seed {seed}: checkpoint missing ({ckpt}) — skip")
                continue

            student = load_student(arch, ckpt, device)
            audit   = Audit()

            with torch.no_grad():
                for batch in loader:
                    images = batch["image"].to(device).float()
                    images = adapter(images)
                    mask   = batch.get("mask", batch.get("label"))
                    if mask is None:
                        continue
                    mask   = mask.to(device).long()
                    mask_hw = mask.shape[-2:]

                    t_pred = predict(teacher, images, mask_hw)
                    s_pred = predict(student, images, mask_hw)

                    audit.update(s_pred, t_pred, mask)
                    pooled.update(s_pred, t_pred, mask)

            s = audit.summary()
            per_seed[seed] = s
            print(f"  seed {seed}: disagree={s['n_disagree_px']:>10} px "
                  f"({100*s['disagree_rate']:.2f}%)  "
                  f"student_right={100*s['student_right_rate']:.2f}%  "
                  f"(recall_gain={100*s['student_recall_gain_rate']:.1f}%  "
                  f"fp_suppress={100*s['student_fp_suppress_rate']:.1f}%)")
            del student
            torch.cuda.empty_cache()

        # mean/std of the per-seed student_right_rate (the headline number)
        rates = [per_seed[k]["student_right_rate"] for k in per_seed]
        results[label] = {
            "arch":              arch,
            "per_seed":          per_seed,
            "pooled":            pooled.summary(),
            "student_right_rate_mean": round(statistics.mean(rates), 6) if rates else None,
            "student_right_rate_std":  round(statistics.stdev(rates), 6) if len(rates) > 1 else 0.0,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[audit] saved → {args.out}")

    # Markdown summary
    print("\n### Disagreement audit on STURM-S2 (student vs. Prithvi teacher)\n")
    print("On valid pixels where student and teacher disagree, who matches GT?\n")
    print("| Student | seeds | disagree rate | **student right %** | teacher right % "
          "| recall-gain % | FP-suppress % |")
    print("|---|---|---|---|---|---|---|")
    for label, r in results.items():
        p = r["pooled"]
        n = len(r["per_seed"])
        mean = r["student_right_rate_mean"]
        std  = r["student_right_rate_std"]
        srr  = f"{100*mean:.2f}±{100*std:.2f}" if mean is not None else "—"
        print(f"| {label} | {n} | {100*p['disagree_rate']:.2f}% | "
              f"**{srr}** | {100*p['teacher_right_rate']:.2f}% | "
              f"{100*p['student_recall_gain_rate']:.1f}% | "
              f"{100*p['student_fp_suppress_rate']:.1f}% |")
    print("\n> student right >50% ⇒ genuine denoising (student corrects the teacher); "
          "≈50% ⇒ disagreement is noise.")
    print("> student-right % (mean±std over seeds) is the headline; pooled columns "
          "aggregate all seeds' pixels.")
    print("> recall-gain % = of GT-flood disagreement pixels, share the student got "
          "right (floods it recovered that the teacher missed).")
    print("> FP-suppress % = of GT-background disagreement pixels, share the student "
          "got right (teacher false positives it removed).")


if __name__ == "__main__":
    main()
