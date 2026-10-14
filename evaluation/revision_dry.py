"""S11 diagnostic or explicitly reviewed original-reference WorldFloods chips."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WANDB_MODE", "disabled")


class ReviewedWorldFloodsDryDataset:
    """Native512, fully assessable chips admitted by a frozen human review gate."""

    def __init__(self, manifest_path, gate_path):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from data.sen1floods11 import NORM_MEAN, NORM_STD
        from evaluation.revision_eval import sha256

        self.manifest_path = Path(manifest_path).resolve()
        gate = json.loads(Path(gate_path).read_text())
        if gate.get("status") != "verified" or gate.get("verified_manifest_sha256") != sha256(self.manifest_path):
            raise RuntimeError("Independent dry reference gate is not verified for this manifest")
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema") != "prompt3_dry_verified_v1" or manifest.get("status") != "verified":
            raise RuntimeError("Require imported, frozen reviewed manifest")
        for key in ("candidate_manifest", "annotations_path"):
            hash_key = "candidate_manifest_sha256" if key == "candidate_manifest" else "annotations_sha256"
            if sha256(manifest[key]) != manifest[hash_key]:
                raise RuntimeError("Frozen reference/review source changed after import")
        self.rows = manifest["candidates"]
        if not 0 < len(self.rows) <= 60 or len({r["id"] for r in self.rows}) != len(self.rows):
            raise RuntimeError("Invalid bounded reviewed set")
        for row in self.rows:
            review = row.get("review", {})
            if (review.get("status") != "verified_dry" or review.get("fully_assessable") is not True
                    or review.get("valid_region") != "full_tile" or list(row["window"])[2:] != [512, 512]):
                raise RuntimeError("Scoring only admits independently reviewed full native512 dry chips")
            if not all(review.get(k) for k in ("reviewer", "reviewed_utc", "reference_source")):
                raise RuntimeError("Missing human reviewer/source identity")
            if sha256(row["label_copy_path"]) != row["label_copy_sha256"]:
                raise RuntimeError("Preserved original label copy changed")
        # Same WorldFloods band radiometry, Resize(identity), fixed S11 normalize
        # and tensor transform; permanent water is never removed from reference.
        self.transform = A.Compose([A.Resize(512, 512), A.Normalize(mean=list(NORM_MEAN),
                                    std=list(NORM_STD), max_pixel_value=1.0), ToTensorV2()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import rasterio
        from rasterio.windows import Window
        from data.ood_datasets import _WF_BAND_INDICES

        row = self.rows[index]
        win = Window(*row["window"])
        with rasterio.open(row["image_path"]) as src:
            raw = src.read(window=win)
            affine = list(src.window_transform(win))[:6]
            if str(src.crs) != row["crs"] or affine != row["native_affine"]:
                raise RuntimeError("Native dry-chip grid changed")
        with rasterio.open(row["original_reference_path"]) as src:
            gt = src.read(window=win)
            if str(src.crs) != row["crs"] or list(src.window_transform(win))[:6] != row["native_affine"]:
                raise RuntimeError("Native dry-chip reference grid changed")
        for array, prefix in ((raw, "input"), (gt, "label")):
            if (list(array.shape) != row[prefix + "_shape"] or str(array.dtype) != row[prefix + "_dtype"]
                    or hashlib.sha256(array.tobytes(order="C")).hexdigest() != row[prefix + "_window_sha256"]):
                raise RuntimeError("Native reviewed input/reference window changed")
        if (not np.all(np.isfinite(raw[:13])) or np.any(np.all(raw[:13] == 0, axis=0))
                or not np.all(gt[0] == 1) or not np.all(gt[1] == 1)):
            raise RuntimeError("Dry reference contains unknown/cloud/water or unusable spectrum")
        image = np.clip(raw[_WF_BAND_INDICES].astype(np.float32) * 0.0001, 0.0, 1.0)
        mask = np.zeros((512, 512), dtype=np.int64)
        result = self.transform(image=image.transpose(1, 2, 0), mask=mask)
        return {"image": result["image"].float(), "mask": result["mask"].long(),
                "filename": row["id"], "event_group": row["event"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.spec, args.out = args.spec.resolve(), args.out.resolve()
    spec = json.loads(args.spec.read_text())
    import torch
    from torch.utils.data import DataLoader, Subset
    from data.sen1floods11 import get_dataloaders
    from evaluation.revision_eval import NamedDataset, atomic_json, sha256
    from evaluation.perscene_eval import load

    reviewed = "reference_manifest" in spec
    if reviewed:
        dataset = ReviewedWorldFloodsDryDataset(spec["reference_manifest"], spec["reference_gate"])
        named = dataset
    else:
        _, _, test = get_dataloaders(spec["s11_root"], batch_size=2, num_workers=0,
                                     img_size=224, eval_img_size=512, drop_last=False)
        wanted = set(spec["chip_ids"])
        # Select by the reader's file identity, never by a tested model's predictions.
        dataset = test.dataset
        indices = [i for i, path in enumerate(dataset.image_files)
                   if Path(path).name.replace("_S2Hand.tif", "") in wanted]
        if len(indices) != len(wanted):
            raise RuntimeError(f"Reader membership mismatch: {len(indices)} != {len(wanted)}")
        named = NamedDataset(Subset(dataset, indices), [Path(dataset.image_files[i]).name for i in indices])
    dl = DataLoader(named, batch_size=2, shuffle=False, num_workers=0)
    dev = torch.device(spec.get("device", "cpu"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Declared CUDA unavailable")
    ckpt = spec.get("checkpoint", "")
    identity = {"spec_sha256": sha256(args.spec), "checkpoint_sha256": sha256(ckpt) if ckpt else None}
    if spec.get("training_receipt"):
        receipt = json.loads(Path(spec["training_receipt"]).read_text())
        if (receipt.get("status") != "verified" or receipt.get("run_id") != spec["training_run_id"] or
                receipt["checkpoint_checks"]["best"]["sha256"] != identity["checkpoint_sha256"]):
            raise RuntimeError("Dry evaluator checkpoint differs from finalized training receipt")
        identity["final_training_receipt_sha256"] = sha256(spec["training_receipt"])
    if reviewed:
        identity.update({"reference_manifest_sha256": sha256(spec["reference_manifest"]),
                         "reference_gate_sha256": sha256(spec["reference_gate"]),
                         "dry_evaluator_sha256": sha256(__file__),
                         "data_reader_sha256": sha256(Path(__file__).resolve().parents[1] / "data/ood_datasets.py"),
                         "normalization_source_sha256": sha256(Path(__file__).resolve().parents[1] / "data/sen1floods11.py"),
                         "input_window_sha256": {r["id"]: r["input_window_sha256"] for r in dataset.rows},
                         "reference_window_sha256": {r["id"]: r["label_window_sha256"] for r in dataset.rows}})
    else:
        identity["reference_sha256"] = {str(dataset.segmentation_mask_files[i]): sha256(dataset.segmentation_mask_files[i]) for i in indices}
    if args.out.exists():
        prior = json.loads(args.out.read_text())
        if prior.get("identity") != identity:
            raise RuntimeError("Existing dry diagnostic has different inputs")
        if prior.get("status") == "complete":
            return
    model = load(spec["arch"], ckpt, dev)
    scenes = []
    with torch.no_grad():
        for batch in dl:
            valid = batch["mask"] != -1
            if ((batch["mask"] != 0) & valid).any():
                raise RuntimeError("Dry human reference contains water/unknown valid class")
            prediction = model(batch["image"].to(dev)).argmax(1).cpu()
            for i, filename in enumerate(batch["filename"]):
                nvalid = int(valid[i].sum())
                if not nvalid:
                    raise RuntimeError("No valid dry reference pixels")
                nwater = int(((prediction[i] == 1) & valid[i]).sum())
                fraction = nwater / nvalid
                scenes.append({"chip": filename, "event_group": batch["event_group"][i] if reviewed else filename.split("_")[0],
                               "valid_pixels": nvalid, "false_water_pixels": nwater,
                               "false_water_fraction": fraction,
                               "any_positive_pixel": nwater > 0,
                               "diagnostic_area_0.1pct": fraction >= 0.001,
                               "diagnostic_area_1pct": fraction >= 0.01,
                               "water_IoU": None, "water_F1": None})
    events = {}
    for scene in scenes:
        row = events.setdefault(scene["event_group"], {"chips": 0, "valid_pixels": 0, "false_water_pixels": 0})
        row["chips"] += 1
        row["valid_pixels"] += scene["valid_pixels"]
        row["false_water_pixels"] += scene["false_water_pixels"]
    for row in events.values():
        row["false_water_fraction"] = row["false_water_pixels"] / row["valid_pixels"]
    atomic_json(args.out, {"status": "complete", "identity": identity, "spec": spec,
        "completed_unix": time.time(), "n_chips": len(scenes), "n_event_groups": len(events),
        "false_positive_pixel_rate": sum(s["false_water_pixels"] for s in scenes) / sum(s["valid_pixels"] for s in scenes),
        "scene_alarm_frequency": {k: sum(s[k] for s in scenes) / len(scenes)
                                  for k in ("any_positive_pixel", "diagnostic_area_0.1pct", "diagnostic_area_1pct")},
        "scenes": scenes, "events": events,
        "scope": ("Independently reviewed, original all-water-reference dry chips in existing WorldFloods acquisitions. Fixed native512; no JRC exclusion. Residual training-history/geography uncertainty persists; not an operational landscape sample."
                  if reviewed else "Held-out human-labelled dry chips; event groups overlap training/validation. Diagnostic only; independent-event dry-scene gate remains blocked.")})
    print(f"Dry-chip diagnostic complete: {len(scenes)} chips, {len(events)} event groups", flush=True)


if __name__ == "__main__":
    main()
