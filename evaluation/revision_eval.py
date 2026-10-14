"""Explicit, fingerprinted revision jobs around the existing per-scene evaluator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WANDB_MODE", "disabled")


class NamedDataset:
    """Attach reader file IDs without changing images, labels or transforms."""

    def __init__(self, dataset, filenames):
        self.dataset, self.filenames = dataset, filenames
        assert len(dataset) == len(filenames)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "filename": self.filenames[index]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def scores(cells):
    import numpy as np
    a = np.asarray(cells, dtype=np.float64)
    union = a.sum(1)
    iou = [float(a[i, 0] / union[i]) if union[i] else None for i in (0, 1)]
    ratio = lambda n, d: float(n / d) if d else None
    valid = sum(x is not None for x in iou)
    return {"mIoU": sum(x for x in iou if x is not None) / valid if valid else None,
            "water_IoU": iou[1], "background_IoU": iou[0],
            "water_precision": ratio(a[1, 0], a[1, 0] + a[1, 1]),
            "water_recall": ratio(a[1, 0], a[1, 0] + a[1, 2]),
            "water_F1": ratio(2 * a[1, 0], 2 * a[1, 0] + a[1, 1] + a[1, 2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    # Resolve before perscene_eval's existing import changes cwd to the code root.
    args.spec, args.out = args.spec.resolve(), args.out.resolve()
    spec = json.loads(args.spec.read_text())
    import numpy as np
    import torch
    from data.ood_datasets import get_sturm_flood_dataloader, get_worldfloods_dataloader
    from data.channel_adapter import sturm_s2_adapter
    import evaluation.perscene_eval as pe

    dev = torch.device(spec.get("device", "cpu"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Declared CUDA execution unavailable; refusing CPU substitution")
    if spec["arch"] == "teacher" and spec.get("checkpoint") == "hf":
        raise ValueError("Revision teacher identity must be a pinned local checkpoint")
    ckpt = spec.get("checkpoint", "")
    identity = {"spec_sha256": sha256(args.spec),
                "checkpoint_sha256": sha256(ckpt) if ckpt else None,
                "runtime_packages": {name: importlib.metadata.version(name) for name in
                                     ("torch", "torchvision", "terratorch", "numpy", "albumentations", "rasterio", "timm")},
                "fingerprints": spec.get("fingerprints", {}),
                "source_sha256": {str(p): sha256(p)
                                  for folder in ("models", "data", "evaluation", "utils")
                                  for p in sorted(Path(folder).glob("*.py"))}}
    if spec.get("training_receipt"):
        receipt_path = Path(spec["training_receipt"])
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "verified" or
                receipt.get("run_id") != spec["training_run_id"] or
                receipt["checkpoint_checks"]["best"]["sha256"] != identity["checkpoint_sha256"]):
            raise RuntimeError("Final validation-selected checkpoint does not match its successful training receipt")
        identity["final_training_receipt_sha256"] = sha256(receipt_path)
    model = pe.load(spec["arch"], ckpt, dev)
    conditions = spec.get("conditions", ["clean"])
    params = dict(pe.CONDS)
    # Same scalar perturbation implementation; complete the reported scalar grid.
    params["g1.10"] = (1.10, 0.0)
    params["g1.00"] = (1.0, 0.0)
    output = {"status": "running", "identity": identity, "spec": spec,
              "cells": {}, "summary": {}, "started_unix": time.time()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        prior = json.loads(args.out.read_text())
        if prior["identity"] != identity:
            raise RuntimeError("Existing output has different inputs; choose a new run ID")
        output = prior
        if output.get("status") == "complete":
            return
    for path in spec.get("reuse_results", []):
        prior = json.loads(Path(path).read_text())
        same = (prior.get("status") == "complete" and prior["spec"]["arch"] == spec["arch"] and
                all(prior["identity"][k] == identity[k] for k in
                    ("checkpoint_sha256", "runtime_packages", "source_sha256")))
        if not same:
            raise RuntimeError("Reusable evaluation has a different checkpoint/source/runtime identity")
        for key, cell in prior["cells"].items():
            dataset, condition = key.split("__", 1)
            if dataset in spec["datasets"] and condition in conditions:
                if Path(prior["spec"]["roots"][dataset]).resolve() != Path(spec["roots"][dataset]).resolve():
                    raise RuntimeError("Reusable evaluation has a different dataset root")
                output["cells"].setdefault(key, cell)
        output.setdefault("reused_results", []).append({"path": path, "sha256": sha256(path)})
    if "sturm" in spec["datasets"]:
        pe.STURM = spec["roots"]["sturm"]
        t2s, t2e = pe.sturm_maps()
    else:
        t2s, t2e = {}, {}
    for ds in spec["datasets"]:
        for condition in conditions:
            if ds == "s11" and condition != "clean":
                continue
            complete_scalar = (spec.get("complete_reported_scalar_grid", False)
                               and ds == "wf" and condition == "g0.90")
            if ds not in pe.COND_DATASETS.get(condition, {ds}) and not complete_scalar:
                continue
            key = ds + "__" + condition
            if key in output["cells"]:
                continue
            gain, haze = params[condition]
            bands = pe.BAND_CONDS.get(condition)
            if ds == "sturm":
                dl = get_sturm_flood_dataloader(spec["roots"][ds], sensor="S2", batch_size=8,
                    num_workers=spec.get("num_workers", 2), img_size=512,
                    normalize="original", sim_gain=gain, sim_haze=haze,
                    sim_gain_bands=None if bands is None else pe._sturm9(bands),
                    class_mapping="paper")
                adapter = sturm_s2_adapter(target_channels=6).to(dev)
                keys = (lambda fn: t2s.get(fn), lambda fn: t2e.get(fn))
            elif ds == "wf":
                dl = get_worldfloods_dataloader(spec["roots"][ds], split="test", batch_size=4,
                    num_workers=spec.get("num_workers", 2), img_size=512, tile_size=512,
                    normalize="s11", sim_gain=gain, sim_haze=haze, sim_gain_bands=bands,
                    permanent_water_as="ignore")
                adapter = None
                keys = (lambda fn: fn.split("#")[0],
                        lambda fn: fn.split("#")[0].split("_")[0])
            elif ds == "s11":
                from data.sen1floods11 import get_dataloaders
                _, _, dl = get_dataloaders(spec["roots"][ds], batch_size=4,
                    num_workers=spec.get("num_workers", 2), img_size=224,
                    eval_img_size=512, drop_last=False)
                from torch.utils.data import DataLoader
                dl = DataLoader(NamedDataset(dl.dataset, [Path(p).name for p in dl.dataset.image_files]),
                    batch_size=4, num_workers=spec.get("num_workers", 2), shuffle=False)
                adapter = None
                keys = (lambda fn: Path(fn).name, lambda fn: Path(fn).name.split("_")[0])
            else:
                raise ValueError(ds)
            start = time.monotonic()
            scene, event, count, dropped = pe.run(model, dl, dev, adapter, keys)
            if dropped or count != len(dl.dataset) or not scene:
                raise RuntimeError(f"Invalid scene coverage: {count}/{len(dl.dataset)}, dropped={dropped}")
            for cell in scene.values():
                if not np.isfinite(cell).all() or (cell < 0).any():
                    raise RuntimeError("Invalid confusion counts")
            micro = scores(sum(scene.values()))
            output["cells"][key] = {"dataset": ds, "condition": condition, "gain": gain,
                "haze": haze, "gain_bands": bands, "n_tiles": count,
                "scene": {k: v.tolist() for k, v in scene.items()},
                "event": {k: v.tolist() for k, v in event.items()},
                "micro": micro, "scene_metrics": {k: scores(v) for k, v in scene.items()},
                "event_metrics": {k: scores(v) for k, v in event.items()},
                "elapsed_seconds": time.monotonic() - start}
            atomic_json(args.out, output)
            print(f"{key}: {count} tiles, {len(scene)} scenes, mIoU={micro['mIoU']:.6f}, "
                  f"{time.monotonic() - start:.1f}s", flush=True)
    for ds in spec["datasets"]:
        clean = output["cells"][ds + "__clean"]["micro"]
        rows = {}
        for condition in conditions:
            if ds + "__" + condition not in output["cells"]:
                continue
            current = output["cells"][ds + "__" + condition]
            rows[condition] = {metric: {
                "clean": clean[metric], "corrupted": current["micro"][metric],
                "absolute_drop": clean[metric] - current["micro"][metric],
                "relative_drop": ((clean[metric] - current["micro"][metric]) / clean[metric]
                                  if clean[metric] else None)}
                for metric in ("mIoU", "water_IoU")
                if clean[metric] is not None and current["micro"][metric] is not None}
        output["summary"][ds] = {"degradation": rows,
            "worst_absolute_drop": {metric: max((row[metric]["absolute_drop"] for row in rows.values()
                                     if metric in row), default=None) for metric in ("mIoU", "water_IoU")},
            "scalar_gain_confusion_invariant": all(
                output["cells"][ds + "__" + cond]["scene"] == output["cells"][ds + "__clean"]["scene"]
                for cond in conditions if cond.startswith("g") and ds + "__" + cond in output["cells"])
                if spec["arch"].startswith("index:") else None,
            "uncertainty": "Per-event cells retained; no pixel-independent inference or equivalence claim"}
    output["status"] = "complete"
    output["completed_unix"] = time.time()
    atomic_json(args.out, output)


if __name__ == "__main__":
    main()
