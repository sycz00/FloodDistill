"""CPU checks for real source-reader metadata and compression-job boundaries.

Actual data are inspected for IDs, support and shape only. Validation arithmetic
and gates use fabricated tensors/receipts; no real accuracy is inspected here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from distillation import compression_metric as cm
from experiments import compression_prepare as preparation
from experiments.revision_queue import atomic_json, file_hash, read_json, digest
from models.compression_quant import FixedQATConv2d
from scripts import compression_jobs as jobs


class PairedSourceToy(Dataset):
    image_files = [Path("Bolivia_1_S2Hand.tif"), Path("Bolivia_2_S2Hand.tif")]

    def __len__(self):
        return 2

    def __getitem__(self, i):
        targets = [torch.tensor([[0, 1], [1, -1]]), torch.tensor([[0, 1], [0, 1]])]
        fp = [torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0, 0], [0, 1]])]
        quant = [torch.tensor([[1, 1], [1, 0]]), torch.tensor([[0, 1], [0, 1]])]
        image = torch.zeros(6, 2, 2)
        image[0], image[1] = fp[i], quant[i]
        # Deliberately matches TerraTorch's lack of an inline filename key.
        return {"image": image, "mask": targets[i]}


class PairedBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = FixedQATConv2d(nn.Conv2d(1, 1, 1), torch.tensor([1.]),
                                       torch.tensor([0]), torch.tensor([1.]))

    def forward(self, x):
        prediction = x[:, int(self.quantizer.fake_quant_enabled)].long()
        return torch.nn.functional.one_hot(prediction, 2).permute(0, 3, 1, 2).float()


def source_metadata(s11_root: str, receipt_path: Path, source_path: Path) -> dict:
    from data.sen1floods11 import get_dataloaders, NORM_MEAN, NORM_STD
    _, val, test = get_dataloaders(s11_root, batch_size=4, num_workers=0,
                                  img_size=224, eval_img_size=512, drop_last=False)
    ids = [Path(p).name for p in val.dataset.image_files]
    assert len(ids) == len(set(ids)) == len(val.dataset) == 89 and len(test.dataset) == 90
    item = val.dataset[0]
    assert "image" in item and "mask" in item
    assert item["image"].shape == (6, 512, 512) and item["mask"].shape == (512, 512)
    receipt, source = read_json(receipt_path), read_json(source_path)
    selected, group_info = preparation.pools(receipt, source)
    repeated, repeated_info = preparation.pools(receipt, source)
    assert selected == repeated and group_info == repeated_info
    by_id = {row["file"]: row for row in source["selected_source_rows"]}
    assert len(selected["fit"]) == len(selected["interaction"]) == 256
    assert not ({row["file"] for row in selected["fit"]} & {row["file"] for row in selected["interaction"]})
    assert not ({row["group"] for row in selected["fit"]} & {row["group"] for row in selected["interaction"]})
    for rows in selected.values():
        for row in rows:
            actual_source = by_id[row["file"]]["source"]
            location = actual_source.split("/")[0]
            expected_event = "_".join(location.split("_")[:2]) if location.startswith("GDACS_") else location.split("_")[0]
            assert row["group"] == expected_event
            assert row["source"] in (actual_source, actual_source.split("/")[0]), \
                "pool source metadata does not describe its selected image"
    pseudo_spec = {"pseudo": receipt["pool"], "index": receipt["index"], "seed": 42,
                   "workers": 0, "feature_pools": selected}
    ds = jobs.pseudo(pseudo_spec, augment=False)
    assert len(ds) == 2500 and not ds.mult_noise and not ds.geom_aug and not ds._extra_keys
    fit = jobs.pool_loader(pseudo_spec, "fit", batch_size=2)
    sample = next(iter(fit))
    assert sample["image"].shape == (2, 6, 224, 224) and sample["mask"].shape == (2, 224, 224)
    assert "soft_logits" not in sample
    _, weights = cm.probe_features(sample["image"], sample["mask"])
    assert torch.allclose(weights.sum(1), torch.ones(2, dtype=torch.float64))
    return {"s11_root": s11_root, "source_val_images": len(ids), "terminal_test_ids_only": len(test.dataset),
            "source_val_ids_sha256": digest(ids), "first_source_val_id": ids[0],
            "raw_reader_item_keys": sorted(item), "source_shape": [6, 512, 512],
            "training_probe_input_shape": [6, 224, 224], "normalization_mean": NORM_MEAN,
            "normalization_std": NORM_STD, "feature_pool_sizes": {k: len(v) for k, v in selected.items()},
            "feature_pools_sha256": digest(selected), "group_info": group_info,
            "cohort_receipt_sha256": file_hash(receipt_path), "source_manifest_sha256": file_hash(source_path),
            "no_real_model_or_accuracy_evaluated": True}


def check(s11_root: str, receipt_path: Path, source_path: Path) -> dict:
    torch.set_num_threads(2)
    metadata = source_metadata(s11_root, receipt_path, source_path)
    raw_loader = DataLoader(PairedSourceToy(), batch_size=2, shuffle=False, num_workers=0)
    branch = PairedBranch()
    spec = {"workers": 0}
    # CPU only: replace this job's explicit CUDA transfers for toy arithmetic.
    with patch.object(jobs, "loaders", return_value=(raw_loader, None)), \
            patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
        result = jobs.validation(branch, spec, paired=True)
    flips = result["damage_repair"]
    assert result["n_images"] == 2 and len(result["event"]) == 1
    assert flips["valid_pixels"] == 7 and flips["harmful_count"] == 1 and flips["repaired_count"] == 2
    assert flips["fp_correct_count"] == 5 and flips["quantized_correct_count"] == 6
    assert flips["identity_verified"] and flips["accuracy_delta"] == 1 / 7
    assert set(result["scene"]) == {p.name for p in PairedSourceToy.image_files}
    with tempfile.TemporaryDirectory(prefix="flooddistill_compression_job_check_") as temporary:
        directory = Path(temporary)
        interaction = directory / "interaction.json"
        methods = [directory / f"B{i}.json" for i in (1, 2, 3)]
        gate_spec = {"interaction_receipt": str(interaction), "method_receipts": [str(p) for p in methods]}
        for nonidentity, effect, failed, expected in ((True, .004, False, False),
                                                       (True, .006, False, True),
                                                       (False, .1, False, False),
                                                       (True, .1, True, False)):
            atomic_json(interaction, {"nonidentity": nonidentity, "status": "complete", "outcome": "completed"})
            for i, path in enumerate(methods):
                record = {"status": "complete", "outcome": "completed",
                          "validation": {"micro": {"mIoU": .8 + (effect if i == 2 else 0.)}}}
                if failed and i == 2:
                    record = {"status": "failed", "outcome": "failed", "error": "synthetic technical failure"}
                atomic_json(path, record)
            gate = directory / "gate.json"
            jobs.source_gate(gate_spec, gate)
            assert read_json(gate)["eligible"] == expected
            assert read_json(gate)["no_external_scores_read"]
            assert not read_json(gate)["technical_fallback_used"], "no fallback cell was supplied"
        fallback_interaction = directory / "fallback_interaction.json"
        fallback_methods = [directory / f"fallback_B{i}.json" for i in (1, 2, 3)]
        atomic_json(fallback_interaction, {"status": "complete", "outcome": "completed", "nonidentity": True})
        for i, path in enumerate(fallback_methods):
            atomic_json(path, {"status": "complete", "outcome": "completed",
                               "validation": {"micro": {"mIoU": .8 + (.01 if i == 2 else 0.)}}})
        gate_spec.update(primary_cell={"graph": "A3", "precision": "W4A4", "bits": [4, 4],
                                       "interaction_receipt": str(interaction)},
                         fallback_cell={"graph": "A1", "precision": "W4A8", "bits": [4, 8],
                                        "interaction_receipt": str(fallback_interaction)},
                         fallback_method_receipts=[str(p) for p in fallback_methods])
        atomic_json(interaction, {"status": "complete", "outcome": "completed", "nonidentity": True})
        for i, path in enumerate(methods):
            atomic_json(path, {"status": "complete", "outcome": "completed",
                               "validation": {"micro": {"mIoU": .8 + (.002 if i == 2 else 0.)}}})
        jobs.source_gate(gate_spec, gate)
        assert not read_json(gate)["eligible"] and not read_json(gate)["technical_fallback_used"]
        assert read_json(gate)["selected_cell"]["graph"] == "A3", "negative valid cell must not select favorable fallback"
        atomic_json(methods[2], {"status": "failed", "outcome": "failed", "error": "synthetic numeric failure"})
        jobs.source_gate(gate_spec, gate)
        assert read_json(gate)["eligible"] and read_json(gate)["technical_fallback_used"]
        assert read_json(gate)["selected_cell"]["graph"] == "A1"
        atomic_json(interaction, {"status": "complete", "outcome": "completed", "nonidentity": False})
        atomic_json(methods[2], {"status": "complete", "outcome": "skipped", "reason": "identity duplicate"})
        jobs.source_gate(gate_spec, gate)
        assert not read_json(gate)["eligible"] and not read_json(gate)["technical_fallback_used"]
        assert read_json(gate)["selected_cell"]["graph"] == "A3"
        artifacts = directory / "tensor.pt"
        sha = jobs.save_tensor(artifacts, {"probe": torch.arange(4)})
        assert torch.equal(jobs.load_tensor(artifacts, sha)["probe"], torch.arange(4))
        artifacts.write_bytes(b"changed artifact")
        try:
            jobs.load_tensor(artifacts, sha)
            raise AssertionError("changed upstream artifact was accepted")
        except RuntimeError as exc:
            assert "hash changed" in str(exc)
        class StopOwner:
            stop_requested = True
        stopped = jobs.InterruptibleLoader(raw_loader, StopOwner())
        try:
            next(iter(stopped))
            raise AssertionError("stopped command admitted another training batch")
        except jobs.BudgetStop:
            pass
    return {"status": "passed", "device": "cpu", "real_source_metadata": metadata,
            "checks": ["real_source_reader_ids_shapes_and_filename_contract", "deterministic_disjoint_training_event_pools",
                       "pool_source_metadata_matches_selected_file", "training_only_probe_weights_no_cached_soft_targets",
                       "named_validation_wrapper_paired_counts_and_event_aggregation", "source_only_fixed_threshold_gate",
                       "technical_failure_prevents_mechanism_expansion", "upstream_tensor_sha_rejects_changed_artifact",
                       "budget_stop_prevents_next_training_batch", "only_technical_invalidity_uses_fixed_fallback",
                       "negative_or_identity_primary_never_selects_favorable_fallback"],
            "accuracy_scope": "fabricated two-image tensors only; no real accuracy inspected"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s11-root", default="/home/schmalstieg/data/sen1floods11")
    base = Path("/data1/schmalstieg/flooddistill/revision_20261005/original_N2500_recovered_20261006/data")
    parser.add_argument("--cohort-receipt", type=Path, default=base / "original_N2500_receipt.json")
    parser.add_argument("--source-manifest", type=Path, default=base / "original_source_manifest.json")
    parser.add_argument("--out", "--output", dest="out", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    inputs = [Path(__file__).resolve(), Path(jobs.__file__).resolve(), Path(preparation.__file__).resolve()]
    before = {str(path): file_hash(path) for path in inputs}
    try:
        with patch("torch.cuda.is_available", return_value=False):
            result = check(args.s11_root, args.cohort_receipt, args.source_manifest)
        after = {str(path): file_hash(path) for path in inputs}
        if before != after:
            raise RuntimeError("job/check/preparation sources changed while this check was running")
        result["source_sha256"] = before
    except Exception as error:
        if args.out:
            atomic_json(args.out, {"status": "failed", "device": "cpu",
                                   "error": f"{type(error).__name__}: {error}",
                                   "source_sha256": before,
                                   "elapsed_seconds": time.monotonic() - start})
        raise
    result["elapsed_seconds"] = time.monotonic() - start
    if args.out:
        atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
