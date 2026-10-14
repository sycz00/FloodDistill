"""Narrow input, fixed-checkpoint and completed-training checks for this cohort."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.revision_queue import atomic_json, digest, file_hash


def freeze_data(index, pool, out):
    import torch
    torch.set_num_threads(2)
    original = json.loads(index.read_text())
    names = original["files"]
    if len(names) != 2500 or len(set(names)) != 2500:
        raise ValueError("Expected exactly 2500 distinct selected files")
    out.mkdir(parents=True, exist_ok=False)
    frozen_pool = out / "pseudo"
    frozen_pool.mkdir()
    files, sources = [], set()
    counts = {str(k): 0 for k in (-1, 0, 1)}
    for i, name in enumerate(names):
        if Path(name).name != name:
            raise ValueError("Selection must contain pool-local file names")
        src, dst = pool / name, frozen_pool / name
        shutil.copyfile(src, dst)
        checksum = file_hash(dst)
        if checksum != file_hash(src):
            raise ValueError(f"Input changed during copying: {src}")
        sample = torch.load(dst, map_location="cpu", weights_only=True)
        image, mask = sample["image"], sample["mask"]
        if image.shape != (6, 224, 224) or mask.shape != (224, 224):
            raise ValueError(f"Unexpected geometry: {src}")
        if not torch.isfinite(image).all() or mask.dtype != torch.int64:
            raise ValueError(f"Invalid image/mask: {src}")
        labels, numbers = torch.unique(mask, return_counts=True)
        if not set(labels.tolist()) <= {-1, 0, 1}:
            raise ValueError(f"Unexpected mask labels: {src}")
        for label, number in zip(labels.tolist(), numbers.tolist()):
            counts[str(label)] += number
        source = sample.get("source")
        if not isinstance(source, str) or source in sources:
            raise ValueError(f"Missing or duplicate source identity: {src}")
        sources.add(source)
        files.append({"name": name, "sha256": checksum, "bytes": dst.stat().st_size,
                      "source": source})
        dst.chmod(0o444)
        if (i + 1) % 500 == 0:
            print(f"Frozen and checked {i + 1}/2500 tiles", flush=True)
    frozen_index = out / "mapped_N2500_index.json"
    shutil.copyfile(index, frozen_index)
    if file_hash(frozen_index) != file_hash(index):
        raise ValueError("Index changed while freezing")
    frozen_index.chmod(0o444)
    identity = {"index_sha256": file_hash(frozen_index), "files": files}
    atomic_json(out / "dataset_receipt.json", {
        "status": "verified", "dataset_sha256": digest(identity),
        "pool": str(frozen_pool), "index": str(frozen_index), **identity,
        "n_selected": len(files), "n_distinct_sources": len(sources),
        "label_pixel_counts": counts, "original_source_overlap": 2373,
        "fps_replacements": 127, "original_index_byte_reproduction": False,
        "authorization": "User accepted regenerated N2500 with 127 replacements on 2026-10-05",
        "hard_masks": "Existing regenerated teacher masks; hash frozen, not regenerated here",
        "normalization": "Existing stored six-band S11-normalized float32 images unchanged",
        "provenance": original.get("provenance", {}), "source_index": str(index),
    })


def verify_data(receipt):
    data = json.loads(receipt.read_text())
    if data["status"] != "verified" or data["n_selected"] not in (100, 2500):
        raise ValueError("Dataset receipt is not verified N100/N2500")
    if len(data["files"]) != data["n_selected"] or len({f["name"] for f in data["files"]}) != data["n_selected"]:
        raise ValueError("Dataset receipt membership/count mismatch")
    if file_hash(data["index"]) != data["index_sha256"]:
        raise ValueError("Frozen selection index changed")
    if digest({"index_sha256": data["index_sha256"], "files": data["files"]}) != data["dataset_sha256"]:
        raise ValueError("Dataset identity is inconsistent")
    for spec in data["files"]:
        if file_hash(Path(data["pool"]) / spec["name"]) != spec["sha256"]:
            raise ValueError(f"Frozen tile changed: {spec['name']}")
    return data


def evaluator_check(spec):
    rows = []
    for item in spec["comparisons"]:
        old = json.loads(Path(item["archive_metrics"]).read_text())
        new = json.loads(Path(item["rerun"]).read_text())
        if new["status"] != "complete":
            raise ValueError("Fixed-checkpoint evaluation incomplete")
        checkpoint = Path(new["spec"]["checkpoint"])
        if new["identity"]["checkpoint_sha256"] != file_hash(checkpoint):
            raise ValueError("Evaluated checkpoint identity changed")
        for dataset, prefix in (("s11", "s11_test"), ("sturm", "sturm"), ("wf", "wf_fo")):
            cell = new["cells"][dataset + "__clean"]
            if cell["n_tiles"] != {"s11": 90, "sturm": 2675, "wf": 785}[dataset]:
                raise ValueError("Evaluation data count mismatch")
            for metric, suffix in (("mIoU", "mIoU"), ("water_IoU", "flood_IoU")):
                a, b = old[prefix + "_" + suffix], cell["micro"][metric]
                delta = abs(a - b)
                if not math.isfinite(b) or delta > spec["absolute_tolerance"]:
                    raise ValueError(f"Fixed-checkpoint mismatch: {item['id']} {dataset} {metric}: {delta}")
                rows.append({"id": item["id"], "dataset": dataset, "metric": metric,
                             "archived": a, "rerun": b, "absolute_delta": delta,
                             "checkpoint_sha256": new["identity"]["checkpoint_sha256"]})
    return {"status": "verified", "comparisons": rows,
            "absolute_tolerance": spec["absolute_tolerance"],
            "tolerance_basis": spec["tolerance_basis"],
            "scope": "Same archived checkpoints and rounded saved evaluator metrics; no training accuracy gate"}


def training_check(spec):
    import torch
    torch.set_num_threads(2)
    run = Path(spec["out_dir"])
    result = json.loads((run / "results.json").read_text())
    for key in ("best_val_mIoU", "best_val_smoothed_mIoU", "test_mIoU"):
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"Invalid completed training metric: {key}")
    observations = {}
    for name in ("best", "last"):
        ckpt = torch.load(run / (name + ".pt"), map_location="cpu", weights_only=True)
        if ckpt["checkpoint_schema_version"] != 2:
            raise ValueError("Missing atomic training-state checkpoint")
        cfg = ckpt["config"]
        for key, expected in spec["expected_config"].items():
            if cfg[key] != expected:
                raise ValueError(f"Frozen training configuration changed: {key}")
        if not all(torch.isfinite(t).all() for t in ckpt["model"].values()):
            raise ValueError("Nonfinite model checkpoint")
        completed = ckpt["epoch"] + 1
        state = ckpt["training_state"]
        n_selected = spec.get("n_selected", 2500)
        steps_per_epoch = n_selected // spec.get("batch_size", 16)
        if state["next_epoch"] != completed or state["global_step"] != completed * steps_per_epoch:
            raise ValueError("Training optimizer-step budget does not match selected/drop_last/batch16")
        if ckpt["scheduler"]["last_epoch"] != completed:
            raise ValueError("Scheduler epoch budget mismatch")
        if name == "last" and completed != 50 and not state["selection"]["early_stop_reached"]:
            raise ValueError("Training ended before its declared budget without legitimate early stop")
        if name == "best" and abs(ckpt["metrics"]["mIoU"] - result["best_val_mIoU"]) > 1e-12:
            raise ValueError("Best checkpoint/validation result mismatch")
        loss = ckpt["loss_fn"]
        if (loss is None) != (spec["objective"] in ("hard", "focal")):
            raise ValueError("Hard/KD loss-state identity mismatch")
        if spec["objective"] == "ofa":
            if not all(any(k.startswith(f"exit_branches.{i}.") for k in loss) for i in range(4)):
                raise ValueError("OFA branches missing from checkpoint")
        for key, expected in spec.get("expected_loss_policy", {}).items():
            if ckpt.get("loss_policy", {}).get(key) != expected:
                raise ValueError(f"Loss policy identity changed: {key}")
        if spec.get("expected_loss_policy") and not ckpt.get("initial_model_sha256"):
            raise ValueError("New matched checkpoint lacks actual initialization hash")
        observations[name] = {"epochs_completed": completed, "optimizer_steps": state["global_step"],
                              "sha256": file_hash(run / (name + ".pt")),
                              "initial_model_sha256": ckpt.get("initial_model_sha256"),
                              "loss_policy": ckpt.get("loss_policy")}
        del ckpt
    return {"status": "verified", "run_id": spec["run_id"], "architecture": spec["architecture"],
            "objective": spec["objective"], "seed": spec["seed"], "cohort": spec["cohort"],
            "checkpoint_checks": observations, "results": result,
            "accuracy_acceptance_threshold": None,
            "scope": "Functional recipe/budget/checkpoint checks; null or unfavorable accuracy remains valid"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "data", "evaluator", "training", "train"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze_data(args.index, args.pool, args.out_dir)
        return
    started = time.monotonic()
    if args.action == "data":
        data = verify_data(args.receipt)
        result = {"status": "verified", "dataset_sha256": data["dataset_sha256"], "n_selected": data["n_selected"]}
    else:
        spec = json.loads(args.spec.read_text())
        if args.action == "evaluator":
            result = evaluator_check(spec)
        elif args.action == "training":
            result = training_check(spec)
        else:
            data = verify_data(Path(spec["dataset_receipt"]))
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Declared CUDA training unavailable; refusing CPU substitution")
            argv = spec["argv"]
            if argv[argv.index("--pseudo_dir") + 1] != data["pool"] or argv[argv.index("--curated_index_file") + 1] != data["index"]:
                raise ValueError("Training command does not use the frozen cohort")
            if spec.get("n_selected", 2500) != data["n_selected"] or int(argv[argv.index("--n_samples") + 1]) != data["n_selected"]:
                raise ValueError("Training command/spec budget does not match the frozen cohort")
            sys.argv = argv[1:]
            runpy.run_path(argv[1], run_name="__main__")
            result = training_check(spec)
    result["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.out, result)
    print(json.dumps({"status": result["status"], "action": args.action, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
