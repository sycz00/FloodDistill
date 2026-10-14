"""Prompt3 evaluation-only geometry/validity variants; legacy sources stay intact."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WANDB_MODE", "disabled")
from evaluation.revision_eval import atomic_json, sha256, scores

WF_PROTOCOL = "P3_WF_NATIVE_ZERO_PAD512_V1"
STURM_PROTOCOL = "P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1"
SELECTED_STURM = [0, 1, 2, 6, 7, 8]
PACKAGES = ("torch", "torchvision", "terratorch", "numpy", "albumentations", "rasterio", "timm")


def digest_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def confusion(pred, mask):
    import numpy as np
    valid = mask != -1
    p, y = pred[valid].astype(bool), mask[valid] == 1
    tp, fp, fn, tn = [int(z.sum()) for z in (p & y, p & ~y, ~p & y, ~p & ~y)]
    return np.array([[tn, fn, fp], [tp, fp, fn]], dtype=np.int64)


def cell_valid(cell):
    return int(cell[0, 0] + cell[1].sum())


def native_item(dataset, index):
    """Reuse raw loader masks/radiometry, normalize without resize, then pad."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from albumentations.pytorch import ToTensorV2
    image_path, label_path, window = dataset.samples[index]
    assert window is not None and dataset.version == "v2"
    image, mask = dataset._getitem_v2(image_path, label_path, window)
    assert image.shape[1:] == mask.shape == (window[3], window[2])
    assert np.isfinite(image).all(), f"Nonfinite native WF input: {image_path} {window}"
    normalized = dataset.transform.transforms[1](image=image.transpose(1, 2, 0), mask=mask)
    result = ToTensorV2()(**normalized)
    x = result["image"].float()
    h, w = mask.shape
    padded = F.pad(x, (0, 512 - w, 0, 512 - h), mode="constant", value=0.0)
    padded_mask = np.pad(mask, ((0, 512 - h), (0, 512 - w)), constant_values=-1)
    assert torch.equal(padded[:, :h, :w], x)
    assert np.array_equal(padded_mask[:h, :w], mask)
    assert int((padded_mask != -1).sum()) == int((mask != -1).sum())
    return {"image": padded, "mask": mask, "padded_mask": padded_mask,
            "filename": f"{Path(image_path).name}#r{window[1]}_c{window[0]}", "height": h, "width": w}


def sturm_invalid(dataset, index):
    import cv2
    import numpy as np
    import rasterio
    with rasterio.open(dataset.samples[index][0]) as src:
        image = src.read().astype(np.float32)
    source = ~np.isfinite(image[SELECTED_STURM]).all(axis=0)
    # Same bilinear spatial footprint as the executed image interpolation.
    support = cv2.resize(source.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR) > 0
    return source, support


def audit(spec, out):
    import cv2
    import numpy as np
    import rasterio
    import torch
    from data.ood_datasets import WorldFloodsDataset, STURMFloodDataset
    started = time.monotonic()
    wf = WorldFloodsDataset(spec["roots"]["wf"], img_size=512, tile_size=512,
                           normalize="s11", permanent_water_as="ignore")
    old_windows = json.loads((ROOT / "experiments/revision/prompt2_worldfloods_windows.json").read_text())
    assert len(wf) == old_windows["count"] == 785
    windows, maps, rects = [], {}, defaultdict(list)
    reference_manifest = json.loads((ROOT / "experiments/revision/prompt2_reference_manifest.json").read_text())
    for index, (ip, lp, window) in enumerate(wf.samples):
        c, r, w, h = window
        partial = w != 512 or h != 512
        native = native_item(wf, index)
        if not partial:
            legacy = wf[index]
            assert torch.equal(legacy["image"], native["image"])
            assert np.array_equal(legacy["mask"].numpy(), native["mask"])
        if ip.name not in maps:
            with rasterio.open(ip) as src, rasterio.open(lp) as label, rasterio.open(wf.jrc_files[ip.name]) as jrc:
                assert src.crs == label.crs == jrc.crs
                assert src.transform == label.transform == jrc.transform
                assert (src.width, src.height) == (label.width, label.height) == (jrc.width, jrc.height)
                stat = ip.stat()
                maps[ip.name] = {"image_path": str(ip), "image_size": stat.st_size, "image_mtime_ns": stat.st_mtime_ns,
                                "reference_sha256": sha256(lp), "JRC_sha256": sha256(wf.jrc_files[ip.name]),
                                "height": src.height, "width": src.width, "crs": str(src.crs), "affine": list(src.transform)}
                archived = reference_manifest["WorldFloods"][ip.name]
                assert stat.st_size == archived["image"]["bytes"] and stat.st_mtime_ns == archived["image"]["mtime_ns"]
                assert maps[ip.name]["reference_sha256"] == archived["reference"]["sha256"]
                assert maps[ip.name]["JRC_sha256"] == archived["jrc_mask"]["sha256"]
        assert 0 <= c < maps[ip.name]["width"] and 0 <= r < maps[ip.name]["height"]
        assert c + w <= maps[ip.name]["width"] and r + h <= maps[ip.name]["height"]
        for pc, pr, pw, ph in rects[ip.name]:
            assert min(c + w, pc + pw) <= max(c, pc) or min(r + h, pr + ph) <= max(r, pr)
        rects[ip.name].append(window)
        with rasterio.open(ip) as src:
            affine = src.window_transform(rasterio.windows.Window(c, r, w, h))
            assert affine * (0, 0) == src.transform * (c, r)
        mask = native["mask"]
        windows.append({"index": index, "filename": native["filename"], "map": ip.name, "event": ip.name.split("_")[0],
                        "window": list(window), "partial": partial, "valid_native": int((mask != -1).sum()),
                        "water_native": int((mask == 1).sum()), "mask_sha256": digest_array(mask),
                        "normalized_native_input_sha256": digest_array(native["image"].numpy()),
                        "window_affine": list(affine)})
        if (index + 1) % 200 == 0:
            print(f"WF coverage: {index + 1}/{len(wf)}", flush=True)
    assert sum(w["partial"] for w in windows) == old_windows["partial_window_count"] == 148
    for name, record in maps.items():
        height, width = record["height"], record["width"]
        partition_pixels = sum(min(512, height - r) * min(512, width - c)
                               for r in range(0, height, 512) for c in range(0, width, 512))
        assert partition_pixels == height * width
        retained_pixels = sum(w * h for c, r, w, h in rects[name])
        record.update({"full_native_partition_pixels": partition_pixels,
                       "retained_native_pixels": retained_pixels,
                       "excluded_by_legacy_window_selection_pixels": partition_pixels - retained_pixels})
    sturm = STURMFloodDataset(spec["roots"]["sturm"], sensor="S2", img_size=512,
                              normalize="original", class_mapping="paper")
    sturm_rows, affected, clip_affected = [], [], []
    totals = defaultdict(int)
    for index, (ip, lp) in enumerate(sturm.samples):
        with rasterio.open(ip) as src, rasterio.open(lp) as label:
            raw = src.read().astype(np.float32)
            labels = label.read(1).astype(np.int64)
        mapped = np.where(labels == 0, 0, np.where(np.isin(labels, [1, 2]), 1, -1)).astype(np.int64)
        invalid9 = ~np.isfinite(raw).all(axis=0)
        invalid6 = ~np.isfinite(raw[SELECTED_STURM]).all(axis=0)
        support = cv2.resize(invalid6.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR) > 0
        finite_native = ~invalid6
        recovered = np.float32(0.04) + np.float32(0.12) * np.nan_to_num(raw, nan=0., posinf=0., neginf=0.)
        clipped = np.clip(recovered, 0., 1.)
        changed_native = np.any(clipped[SELECTED_STURM] != recovered[SELECTED_STURM], axis=0) & finite_native
        mask512 = cv2.resize(mapped, (512, 512), interpolation=cv2.INTER_NEAREST)
        valid512 = mask512 != -1
        changed512 = np.zeros((512, 512), dtype=bool)
        if changed_native.any():
            a = sturm.transform(image=clipped.transpose(1, 2, 0), mask=mapped)["image"].numpy()[SELECTED_STURM]
            b = sturm.transform(image=recovered.transpose(1, 2, 0), mask=mapped)["image"].numpy()[SELECTED_STURM]
            changed512 = np.any(a != b, axis=0) & ~support
            clip_affected.append(ip.name)
        row = {"index": index, "filename": ip.name,
               "nonfinite_any9_native": int(invalid9.sum()), "nonfinite_selected6_native": int(invalid6.sum()),
               "nonfinite_selected6_native_scored": int((invalid6 & (mapped != -1)).sum()),
               "nonfinite_selected6_nearest512_scored": int((cv2.resize(invalid6.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool) & valid512).sum()),
               "nonfinite_selected6_bilinear_support512_scored": int((support & valid512).sum()),
               "clipping_changed_selected6_native_finite": int(changed_native.sum()),
               "clipping_changed_selected6_native_finite_scored": int((changed_native & (mapped != -1)).sum()),
               "clipping_changed_actual_selected6_tensor512_finite_scored": int((changed512 & valid512).sum()),
               "valid512": int(valid512.sum()), "mask_native_sha256": digest_array(mapped),
               "nonfinite_source_mask_sha256": digest_array(invalid6), "exclusion512_mask_sha256": digest_array(support),
               "input_stat": {"size": ip.stat().st_size, "mtime_ns": ip.stat().st_mtime_ns}, "reference_sha256": sha256(lp)}
        sturm_rows.append(row)
        if row["nonfinite_selected6_bilinear_support512_scored"]:
            affected.append(ip.name)
        for key, value in row.items():
            if isinstance(value, int) and key != "index":
                totals[key] += value
        if (index + 1) % 500 == 0:
            print(f"STURM validity: {index + 1}/{len(sturm)}", flush=True)
    assert len(sturm) == 2675
    with out.with_name(out.stem + "_sturm_pixels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sturm_rows[0]))
        writer.writeheader(); writer.writerows(sturm_rows)
    plan_path = ROOT / "experiments/revision/prompt3_evaluation_plan.json"
    result = {"schema_version": 1, "status": "verified", "passed": True, "device": "cpu", "threads": 1,
              "created_unix": time.time(), "elapsed_seconds": time.monotonic() - started,
              "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
              "prior_reference_manifest_sha256": sha256(ROOT / "experiments/revision/prompt2_reference_manifest.json"),
              "source_sha256": {"data/ood_datasets.py": sha256(ROOT / "data/ood_datasets.py"),
                                 "data/channel_adapter.py": sha256(ROOT / "data/channel_adapter.py"),
                                 "evaluation/perscene_eval.py": sha256(ROOT / "evaluation/perscene_eval.py"),
                                 "evaluation/revision_native_border.py": sha256(Path(__file__))},
              "runtime_packages": {name: importlib.metadata.version(name) for name in PACKAGES},
              "worldfloods": {"protocol": WF_PROTOCOL, "maps": maps, "windows": windows,
                              "n_windows": len(windows), "n_partial": 148, "n_interiors": 637,
                              "native_valid_pixels": sum(w["valid_native"] for w in windows),
                              "interiors_bit_identical": True, "retained_window_overlap": False,
                              "padding_counted": 0, "affine_coverage_verified": True,
                              "legacy_interior_reuse_method": "Subtract legacy boundary counts from per-map totals only after fingerprint verification"},
              "sturm": {"protocol": STURM_PROTOCOL, "n_chips": len(sturm), "totals": dict(totals),
                        "affected_nonfinite_scored_chips": affected, "n_affected_nonfinite_scored_chips": len(affected),
                        "clipping_affected_input_chips": clip_affected, "n_clipping_affected_input_chips": len(clip_affected),
                        "mask_correction_required": bool(affected),
                        "correction": "Exclude every output pixel with bilinear contribution from a nonfinite selected-band source; preserve predictions, labels and all other masks",
                        "clipping_sensitivity_required": False,
                        "clipping_remedy": "Document shared executed clipping; no alternate-transform score selection",
                        "rows": sturm_rows}}
    atomic_json(out, result)
    print(json.dumps({"gate": str(out), "WF_windows": len(windows), "STURM_totals": dict(totals),
                      "affected_nonfinite_chips": len(affected)}, indent=2), flush=True)


def verify_legacy(spec, gate):
    """Require actual executed identities before reusing any aggregate counts."""
    legacy_path = Path(spec["legacy_result"])
    legacy = json.loads(legacy_path.read_text())
    assert legacy["status"] == "complete"
    assert legacy["spec"]["arch"] == spec["arch"]
    for dataset in ("wf", "sturm"):
        assert Path(legacy["spec"]["roots"][dataset]).resolve() == Path(spec["roots"][dataset]).resolve()
    assert legacy["cells"]["wf__clean"]["n_tiles"] == gate["worldfloods"]["n_windows"]
    assert set(legacy["cells"]["wf__clean"]["scene"]) == set(gate["worldfloods"]["maps"])
    checkpoint = spec.get("checkpoint", "")
    assert legacy["identity"]["checkpoint_sha256"] == (sha256(checkpoint) if checkpoint else None)
    names = ["data/ood_datasets.py", "data/sen1floods11.py", "data/channel_adapter.py", "evaluation/perscene_eval.py"]
    if not spec["arch"].startswith("index:"):
        names += ["models/student.py", "models/teacher.py", "models/lightformer.py"]
    for name in names:
        assert legacy["identity"]["source_sha256"][name] == sha256(ROOT / name), f"Legacy source mismatch: {name}"
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    assert packages == legacy["identity"]["runtime_packages"] == gate["runtime_packages"]
    for name in ("data/ood_datasets.py", "data/channel_adapter.py", "evaluation/perscene_eval.py"):
        assert gate["source_sha256"][name] == sha256(ROOT / name)
    from data.ood_datasets import _wf_detect_layout
    _, split = _wf_detect_layout(Path(spec["roots"]["wf"]), "test")
    for name, item in gate["worldfloods"]["maps"].items():
        ip = split / "S2" / name
        stat = ip.stat()
        assert stat.st_size == item["image_size"] and stat.st_mtime_ns == item["image_mtime_ns"]
        assert sha256(split / "gt" / name) == item["reference_sha256"]
        assert sha256(split / "PERMANENTWATERJRC" / name) == item["JRC_sha256"]
    sturm_root = Path(spec["roots"]["sturm"])
    prior_manifest = json.loads((ROOT / "experiments/revision/prompt2_reference_manifest.json").read_text())
    for item in gate["sturm"]["rows"]:
        name = item["filename"]
        image = sturm_root / "Sentinel2/S2" / name
        label = sturm_root / "Sentinel2/Floodmaps" / name
        stat = image.stat()
        assert stat.st_size == item["input_stat"]["size"] and stat.st_mtime_ns == item["input_stat"]["mtime_ns"]
        assert sha256(label) == item["reference_sha256"] == prior_manifest["STURM"][name]["reference"]["sha256"]
        assert stat.st_size == prior_manifest["STURM"][name]["image"]["bytes"]
        assert stat.st_mtime_ns == prior_manifest["STURM"][name]["image"]["mtime_ns"]
    return legacy, {"legacy_result_sha256": sha256(legacy_path), "checkpoint_sha256": legacy["identity"]["checkpoint_sha256"],
                    "verified_source_sha256": {name: sha256(ROOT / name) for name in names}, "runtime_packages": packages}


def infer_batch(model, images, device):
    import torch
    with torch.no_grad():
        out = model(torch.stack(images).to(device))
        out = out["out"] if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)
        assert out.shape[-2:] == (512, 512) and torch.isfinite(out).all()
        return out.argmax(1).cpu().numpy()


def export(out, output):
    import numpy as np
    rows = []
    for protocol, payload in (("legacy", output["legacy"]), (output["protocol"], output["native"])):
        cell = sum(np.asarray(c) for c in payload["map"].values())
        rows.append({"level": "pooled", "unit": "all", "protocol": protocol,
                     "TP": int(cell[1, 0]), "FP": int(cell[1, 1]), "FN": int(cell[1, 2]), "TN": int(cell[0, 0]),
                     "valid_pixels": cell_valid(cell), **scores(cell)})
    for level in ("map", "event"):
        for name in output["native"][level]:
            old, new = np.asarray(output["legacy"][level][name]), np.asarray(output["native"][level][name])
            for protocol, cell in (("legacy", old), (output["protocol"], new)):
                rows.append({"level": level, "unit": name, "protocol": protocol,
                             "TP": int(cell[1, 0]), "FP": int(cell[1, 1]), "FN": int(cell[1, 2]), "TN": int(cell[0, 0]),
                             "valid_pixels": cell_valid(cell), **scores(cell)})
    with out.with_suffix(".counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    old, new = output["legacy"]["micro"], output["native"]["micro"]
    output["paired_differences"] = {metric: new[metric] - old[metric] if new[metric] is not None and old[metric] is not None else None for metric in old}
    output["paired_valid_pixel_difference"] = output["native"]["valid_pixels"] - output["legacy"]["valid_pixels"]
    output["paired_map_event"] = {level: {name: {metric: scores(np.asarray(output["native"][level][name]))[metric] - scores(np.asarray(output["legacy"][level][name]))[metric]
                                                                 if scores(np.asarray(output["native"][level][name]))[metric] is not None and scores(np.asarray(output["legacy"][level][name]))[metric] is not None else None for metric in old}
                                          for name in output["native"][level]} for level in ("map", "event")}
    output["paired_unit_summary"] = {}
    for level, units in output["paired_map_event"].items():
        output["paired_unit_summary"][level] = {}
        for metric in old:
            values = [row[metric] for row in units.values() if row[metric] is not None]
            output["paired_unit_summary"][level][metric] = {"n_defined_units": len(values),
                "n_undefined_units": len(units) - len(values),
                "mean_delta": float(np.mean(values)) if values else None,
                "SD_ddof1": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "interpretation": "Descriptive paired unit spread; dependent maps and seed variability remain separate"}
    output["status"] = "complete"
    atomic_json(out, output)


def evaluate(mode, spec, spec_path, out):
    import numpy as np
    import torch
    from data.ood_datasets import WorldFloodsDataset, STURMFloodDataset
    import evaluation.perscene_eval as pe
    gate_path = Path(spec["gate"])
    gate = json.loads(gate_path.read_text())
    assert gate["passed"] and gate["status"] == "verified"
    legacy, identity = verify_legacy(spec, gate)
    identity.update({"spec_sha256": sha256(spec_path), "gate_sha256": sha256(gate_path), "adapter_sha256": sha256(Path(__file__))})
    if out.exists():
        prior = json.loads(out.read_text())
        assert prior["identity"] == identity, "Output identity changed; preserve existing result and choose new run ID"
        if prior["status"] == "complete":
            return
    device = torch.device(spec.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Declared CUDA unavailable")
    model = pe.load(spec["arch"], spec.get("checkpoint", ""), device)
    if mode == "worldfloods":
        ds = WorldFloodsDataset(spec["roots"]["wf"], img_size=512, tile_size=512, normalize="s11", permanent_water_as="ignore")
        assert len(ds) == gate["worldfloods"]["n_windows"]
        boundaries = [w["index"] for w in gate["worldfloods"]["windows"] if w["partial"]]
        old_boundary, new_boundary = defaultdict(lambda: np.zeros((2, 3), dtype=np.int64)), defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
        # Preserve legacy batch composition so its kernel shapes match canonical inference.
        target = set(boundaries)
        for start in sorted({index // 4 * 4 for index in boundaries}):
            indices = list(range(start, min(start + 4, len(ds))))
            batch = [ds[index] for index in indices]
            predictions = infer_batch(model, [b["image"] for b in batch], device)
            for index, item, pred in zip(indices, batch, predictions):
                if index in target:
                    old_boundary[item["filename"].split("#")[0]] += confusion(pred, item["mask"].numpy())
        for start in range(0, len(boundaries), 4):
            indices = boundaries[start:start + 4]
            items = [native_item(ds, index) for index in indices]
            predictions = infer_batch(model, [item["image"] for item in items], device)
            for index, item, pred in zip(indices, items, predictions):
                expected = gate["worldfloods"]["windows"][index]
                assert digest_array(item["mask"]) == expected["mask_sha256"]
                assert digest_array(item["image"].numpy()) == expected["normalized_native_input_sha256"]
                cell = confusion(pred[:item["height"], :item["width"]], item["mask"])
                assert cell_valid(cell) == expected["valid_native"]
                new_boundary[item["filename"].split("#")[0]] += cell
        clean = legacy["cells"]["wf__clean"]
        new_map, interiors = {}, {}
        for name, original in clean["scene"].items():
            residual = np.asarray(original, dtype=np.int64) - old_boundary[name]
            assert (residual >= 0).all(), f"Boundary subtraction incompatible with legacy counts: {name}"
            expected = sum(w["valid_native"] for w in gate["worldfloods"]["windows"] if w["map"] == name and not w["partial"])
            assert cell_valid(residual) == expected
            interiors[name] = residual.tolist()
            new_map[name] = residual + new_boundary[name]
        assert sum(cell_valid(c) for c in new_map.values()) == gate["worldfloods"]["native_valid_pixels"]
        new_event = defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
        for name, cell in new_map.items():
            new_event[name.split("_")[0]] += cell
        output = {"schema_version": 1, "identity": identity, "spec": spec, "protocol": WF_PROTOCOL,
                  "legacy": {"map": clean["scene"], "event": clean["event"], "micro": clean["micro"], "valid_pixels": sum(cell_valid(np.asarray(c)) for c in clean["scene"].values())},
                  "native": {"map": {k: v.tolist() for k, v in new_map.items()}, "event": {k: v.tolist() for k, v in new_event.items()}, "micro": scores(sum(new_map.values())), "valid_pixels": gate["worldfloods"]["native_valid_pixels"]},
                  "reused_interior_counts": interiors, "legacy_boundary_counts": {k: v.tolist() for k, v in old_boundary.items()},
                  "native_boundary_counts": {k: v.tolist() for k, v in new_boundary.items()}, "n_boundaries": len(boundaries), "padding_counted": 0}
        export(out, output)
    else:
        assert gate["sturm"]["mask_correction_required"], "No evidenced invalid-scored inputs; no correction run needed"
        pe.STURM = spec["roots"]["sturm"]
        map_of, event_of = pe.sturm_maps()
        adapter = __import__("data.channel_adapter", fromlist=["sturm_s2_adapter"]).sturm_s2_adapter(target_channels=6).to(device)
        conditions = spec.get("conditions", ["clean"])
        condition_outputs = {}
        reusable = {}
        if spec.get("reuse_result"):
            reuse_path = Path(spec["reuse_result"])
            prior_index = json.loads(reuse_path.read_text())
            assert prior_index["status"] == "complete" and prior_index["protocol"] == STURM_PROTOCOL
            assert prior_index["spec"]["arch"] == spec["arch"]
            for dataset in ("wf", "sturm"):
                assert Path(prior_index["spec"]["roots"][dataset]).resolve() == Path(spec["roots"][dataset]).resolve()
            for key in ("checkpoint_sha256", "verified_source_sha256", "runtime_packages", "gate_sha256"):
                assert prior_index["identity"][key] == identity[key], f"Reuse identity mismatch: {key}"
            prior_cells = prior_index.get("cells", {prior_index.get("condition"): {"path": str(reuse_path), "sha256": sha256(reuse_path)}})
            for condition, saved in prior_cells.items():
                if condition not in conditions:
                    continue
                cell_path = Path(saved["path"])
                assert sha256(cell_path) == saved["sha256"]
                cell = json.loads(cell_path.read_text())
                assert cell["status"] == "complete" and cell["protocol"] == STURM_PROTOCOL and cell["condition"] == condition
                for key in ("checkpoint_sha256", "verified_source_sha256", "runtime_packages", "gate_sha256"):
                    assert cell["identity"][key] == identity[key]
                canonical = legacy["cells"]["sturm__" + condition]
                assert cell["legacy"]["map"] == canonical["scene"] and cell["legacy"]["event"] == canonical["event"]
                assert cell["excluded_invalid_input_scored_pixels"] == gate["sturm"]["totals"]["nonfinite_selected6_bilinear_support512_scored"]
                assert cell["native"]["valid_pixels"] == cell["legacy"]["valid_pixels"] - cell["excluded_invalid_input_scored_pixels"]
                cell["reuse_provenance"] = {"source_index": str(reuse_path), "source_index_sha256": sha256(reuse_path),
                                            "source_cell": str(cell_path), "source_cell_sha256": saved["sha256"],
                                            "original_identity": cell["identity"],
                                            "compatibility": "Checkpoint/reader/package/gate/mask/protocol and per-condition legacy map/event counts match; spec scope/path and reuse-only adapter changes differ"}
                cell["identity"], cell["spec"] = identity, spec
                reusable[condition] = cell
        for condition in conditions:
            condition_out = out if len(conditions) == 1 else out.with_name(out.stem + "__" + condition + out.suffix)
            if condition in reusable and not condition_out.exists():
                export(condition_out, reusable[condition])
                print(f"Reused verified STURM correction: {condition}", flush=True)
            if condition_out.exists():
                prior = json.loads(condition_out.read_text())
                assert prior["identity"] == identity and prior["protocol"] == STURM_PROTOCOL and prior["condition"] == condition
                if prior["status"] == "complete":
                    condition_outputs[condition] = {"path": str(condition_out), "sha256": sha256(condition_out),
                                                    "legacy": prior["legacy"], "native": prior["native"],
                                                    "paired_differences": prior["paired_differences"],
                                                    "excluded_invalid_input_scored_pixels": prior["excluded_invalid_input_scored_pixels"],
                                                    "reused": "reuse_provenance" in prior}
                    continue
            assert legacy["cells"]["sturm__" + condition]["n_tiles"] == gate["sturm"]["n_chips"]
            ds = STURMFloodDataset(spec["roots"]["sturm"], sensor="S2", img_size=512, normalize="original", class_mapping="paper", sim_gain=(dict(pe.CONDS) | {"g1.00": (1., 0.), "g1.10": (1.10, 0.)})[condition][0], sim_haze=(dict(pe.CONDS) | {"g1.00": (1., 0.), "g1.10": (1.10, 0.)})[condition][1], sim_gain_bands=pe._sturm9(pe.BAND_CONDS[condition]) if condition in pe.BAND_CONDS else None)
            assert [p[0].name for p in ds.samples] == [row["filename"] for row in gate["sturm"]["rows"]]
            targets = {i for i, pair in enumerate(ds.samples) if pair[0].name in gate["sturm"]["affected_nonfinite_scored_chips"]}
            removed = defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
            for start in sorted({index // 8 * 8 for index in targets}):
                indices = list(range(start, min(start + 8, len(ds))))
                items = [ds[index] for index in indices]
                images = adapter(torch.stack([b["image"] for b in items]).to(device))
                predictions = infer_batch(model, list(images), device)
                for index, item, pred in zip(indices, items, predictions):
                    if index not in targets:
                        continue
                    _, support = sturm_invalid(ds, index)
                    assert digest_array(support) == gate["sturm"]["rows"][index]["exclusion512_mask_sha256"]
                    mask = item["mask"].numpy().copy()
                    mask[~support] = -1
                    removed[map_of[item["filename"]]] += confusion(pred, mask)
            clean = legacy["cells"]["sturm__" + condition]
            new_map = {name: np.asarray(cell, dtype=np.int64) - removed[name] for name, cell in clean["scene"].items()}
            assert all((c >= 0).all() for c in new_map.values())
            new_event = defaultdict(lambda: np.zeros((2, 3), dtype=np.int64))
            map_events = {map_name: event_of[tile] for tile, map_name in map_of.items()}
            assert all(map_events[map_name] == event_of[tile] for tile, map_name in map_of.items())
            for map_name, cell in new_map.items():
                new_event[map_events[map_name]] += cell
            excluded = sum(cell_valid(c) for c in removed.values())
            assert excluded == gate["sturm"]["totals"]["nonfinite_selected6_bilinear_support512_scored"]
            output = {"schema_version": 1, "identity": identity, "spec": spec, "protocol": STURM_PROTOCOL, "condition": condition,
                      "legacy": {"map": clean["scene"], "event": clean["event"], "micro": clean["micro"], "valid_pixels": sum(cell_valid(np.asarray(c)) for c in clean["scene"].values())},
                      "native": {"map": {k: v.tolist() for k, v in new_map.items()}, "event": {k: v.tolist() for k, v in new_event.items()}, "micro": scores(sum(new_map.values())), "valid_pixels": sum(cell_valid(c) for c in new_map.values())},
                      "excluded_invalid_input_scored_pixels": excluded, "removed_counts": {k: v.tolist() for k, v in removed.items()}, "n_affected_chips": len(targets)}
            export(condition_out, output)
            condition_outputs[condition] = {"path": str(condition_out), "sha256": sha256(condition_out),
                                            "legacy": output["legacy"], "native": output["native"],
                                            "paired_differences": output["paired_differences"],
                                            "excluded_invalid_input_scored_pixels": excluded}
        if len(conditions) > 1:
            index = {"schema_version": 1, "status": "complete", "identity": identity,
                     "protocol": STURM_PROTOCOL, "spec": spec, "cells": condition_outputs}
            if "clean" in condition_outputs:
                index.update({"legacy": condition_outputs["clean"]["legacy"], "native": condition_outputs["clean"]["native"]})
            atomic_json(out, index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "worldfloods", "sturm_valid"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.spec, args.out = args.spec.resolve(), args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.spec.read_text())
    import cv2
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.mode == "gate":
        audit(spec, args.out)
    else:
        evaluate(args.mode, spec, args.spec, args.out)


if __name__ == "__main__":
    main()
