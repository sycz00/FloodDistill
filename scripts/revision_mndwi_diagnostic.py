#!/usr/bin/env python3
"""Bounded prompt2 S11 index audit; never loads model weights or uses CUDA."""
from __future__ import annotations

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
os.environ["CUDA_VISIBLE_DEVICES"] = ""
OUT = ROOT / "experiments/revision"
CACHE = Path("/data1/schmalstieg/flooddistill/revision_20261005/evaluations/P0_rule_mndwi.json")
S11 = Path("/home/schmalstieg/data/sen1floods11")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import albumentations as A
    import cv2
    import numpy as np
    import rasterio
    import torch
    import terratorch.datasets.sen1floods11 as installed_reader
    from data.sen1floods11 import get_dataloaders
    from evaluation.perscene_eval import SpectralIndexPredictor
    from evaluation.revision_eval import scores

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    start = time.monotonic()
    plan_path = OUT / "prompt2_mndwi_plan.json"
    plan = json.loads(plan_path.read_text())
    assert len(plan["necessary_variants"]) == 4
    cache = json.loads(CACHE.read_text())
    assert cache["status"] == "complete" and cache["spec"]["arch"] == "index:mndwi_zero"
    original = cache["cells"]["s11__clean"]["scene"]
    _, _, dl = get_dataloaders(str(S11), batch_size=1, num_workers=0, drop_last=False)
    assert len(dl.dataset) == 90
    assert set(original) == {Path(p).name for p in dl.dataset.image_files}
    for source in ("data/sen1floods11.py", "evaluation/perscene_eval.py"):
        assert sha(ROOT / source) == cache["identity"]["source_sha256"][source], source
    all_ds = copy.copy(dl.dataset)
    base = S11 / "v1.1/data/flood_events/HandLabeled"
    all_ds.image_files = sorted(str(p) for p in (base / "S2Hand").glob("*_S2Hand.tif"))
    all_ds.segmentation_mask_files = [str(base / "LabelHand" / Path(p).name.replace("_S2Hand", "_LabelHand"))
                                      for p in all_ds.image_files]
    assert len(all_ds) == 446
    capture = {}

    def remember(image, **kwargs):
        capture["rho"] = image.copy()
        return image

    # Capture the same resized reflectance that enters the original Normalize.
    transforms = list(dl.dataset.transform.transforms)
    assert len(transforms) == 3 and isinstance(transforms[0], A.Resize) and isinstance(transforms[1], A.Normalize)
    all_ds.transform = A.Compose([transforms[0], A.Lambda(image=remember), *transforms[1:]])
    rule = SpectralIndexPredictor("mndwi_zero").eval()
    split_dir = S11 / "v1.1/splits/flood_handlabeled"
    splits = {name: set((split_dir / f"flood_{filename}_data.txt").read_text().split())
              for name, filename in (("train", "train"), ("validation", "valid"), ("test", "test"), ("Bolivia", "bolivia"))}
    assert [len(splits[k]) for k in ("train", "validation", "test", "Bolivia")] == [252, 89, 90, 15]
    assert sum(len(v) for v in splits.values()) == len(set.union(*splits.values())) == 446
    counts, pairs, inputs, per_chip = [], [], [], []

    def confusion(pred, mask):
        valid = mask != -1
        p, y = pred[valid], mask[valid] == 1
        tp, fp, fn, tn = [int(z.sum()) for z in (p & y, p & ~y, ~p & y, ~p & ~y)]
        return np.array([[tn, fn, fp], [tp, fp, fn]], dtype=np.int64)

    with torch.no_grad():
        for index in range(len(all_ds)):
            path = Path(all_ds.image_files[index])
            label = Path(all_ds.segmentation_mask_files[index])
            filename = path.name
            chip_id = filename.removesuffix("_S2Hand.tif")
            roles = [name for name, ids in splits.items() if chip_id in ids]
            assert len(roles) == 1
            item = all_ds[index]
            x, mask = item["image"], item["mask"].numpy()
            rho = capture.pop("rho")
            assert x.shape == (6, 512, 512) and mask.shape == (512, 512)
            assert set(np.unique(mask)) <= {-1, 0, 1}
            baseline = rule(x[None]).argmax(1)[0].numpy().astype(bool)
            recovered = (x[None] * rule.sd + rule.mu)[0].numpy()
            green, swir = rho[..., 1], rho[..., 4]
            den = recovered[1] + recovered[4]
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_index = (green - swir) / np.maximum(green + swir, np.float32(1e-6))
                literal_index = (recovered[1] - recovered[4]) / den
            # Native512->512 is identity, so threshold-before/after geometry agrees.
            spatial = cv2.resize((raw_index > 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
            assert np.array_equal(spatial, raw_index > 0)
            # 256 pixels/32 overlap: each retained pixel has a single final count.
            tiled = np.zeros((512, 512), dtype=bool)
            coverage = np.zeros((512, 512), dtype=np.uint8)
            disagreement = np.zeros((512, 512), dtype=bool)
            for y in (0, 224, 256):
                for z in (0, 224, 256):
                    patch = rule(x[None, :, y:y + 256, z:z + 256]).argmax(1)[0].numpy().astype(bool)
                    seen = coverage[y:y + 256, z:z + 256] > 0
                    disagreement[y:y + 256, z:z + 256] |= seen & (tiled[y:y + 256, z:z + 256] != patch)
                    tiled[y:y + 256, z:z + 256] = patch
                    coverage[y:y + 256, z:z + 256] += 1
            assert (coverage > 0).all() and not disagreement.any()
            predictions = {"baseline": baseline, "raw_float32": raw_index > 0,
                           "literal_unclamped_ratio": literal_index > 0,
                           "spatial512_identity": spatial,
                           "overlap256_32": tiled}
            cells = {variant: confusion(pred, mask) for variant, pred in predictions.items()}
            if filename in original:
                assert np.array_equal(cells["baseline"], np.array(original[filename])), filename
                cells["baseline"] = np.array(original[filename], dtype=np.int64)
            valid = mask != -1
            for variant, cell in cells.items():
                metrics = scores(cell)
                counts.append({"chip_id": chip_id, "filename": filename, "data_role": roles[0],
                               "canonical90": int(filename in original), "variant": variant,
                               "count_source": "reused_P0" if variant == "baseline" and filename in original else "current_CPU",
                               "TP": int(cell[1, 0]), "FP": int(cell[1, 1]), "FN": int(cell[1, 2]), "TN": int(cell[0, 0]),
                               "valid_pixels": int(valid.sum()), "ignored_pixels": int((~valid).sum()), **metrics})
                if variant != "baseline":
                    reference = predictions["raw_float32"] if variant == "spatial512_identity" else baseline
                    reference_id = "raw_float32" if variant == "spatial512_identity" else "baseline"
                    changed = predictions[variant] != reference
                    pairs.append({"chip_id": chip_id, "canonical90": int(filename in original),
                                  "reference": reference_id, "variant": variant,
                                  "valid_pixels": int(valid.sum()), "changed_valid": int((changed & valid).sum()),
                                  "changed_ignored": int((changed & ~valid).sum()),
                                  "reference0_variant1_valid": int((~reference & predictions[variant] & valid).sum()),
                                  "reference1_variant0_valid": int((reference & ~predictions[variant] & valid).sum())})
            with rasterio.open(path) as src, rasterio.open(label) as lab:
                assert src.width == src.height == lab.width == lab.height == 512
                assert src.crs == lab.crs
                # TIFF transforms differ in final floating digits in these products.
                # Measure corner displacement in pixels rather than exact float equality.
                corner_shift = max(max(abs(px - x0), abs(py - y0))
                                   for x0, y0 in ((0, 0), (512, 0), (0, 512), (512, 512))
                                   for px, py in [(~src.transform) * (lab.transform * (x0, y0))])
                raw_valid = (src.read_masks(3) > 0) & (src.read_masks(12) > 0)
                inputs.append({"chip_id": chip_id, "filename": filename, "data_role": roles[0],
                               "image_path": str(path), "image_resolved_path": str(path.resolve()),
                               "image_sha256": sha(path), "label_path": str(label), "label_sha256": sha(label),
                               "mask_tensor_sha256": hashlib.sha256(mask.astype("<i8").tobytes()).hexdigest(),
                               "image_shape": [src.count, src.height, src.width], "image_dtypes": src.dtypes,
                               "image_nodata": src.nodata, "label_nodata": lab.nodata,
                               "crs": str(src.crs), "transform": list(src.transform),
                               "label_transform": list(lab.transform), "maximum_corner_displacement_pixels": corner_shift,
                               "source_green_swir_nodata_with_valid_label": int((valid & ~raw_valid).sum()),
                               "denominator_zero_valid": int(((den == 0) & valid).sum()),
                               "denominator_negative_valid": int(((den < 0) & valid).sum()),
                               "denominator_clamped_valid": int(((den < 1e-6) & valid).sum()),
                               "recovered_nonfinite_valid": int((~np.isfinite(den) & valid).sum()),
                               "raw_green_swir_equal_valid": int(((green == swir) & valid).sum()),
                               "raw_vs_recovered_max_abs": float(np.max(np.abs(rho.transpose(2, 0, 1) - recovered)))})
            per_chip.append({"filename": filename, "location_prefix_proxy": chip_id.rsplit("_", 1)[0], "cells": cells})
            if (index + 1) % 100 == 0:
                print(f"{index + 1}/446 chips checked on CPU", flush=True)

    write_csv(OUT / "prompt2_mndwi_counts.csv", counts)
    write_csv(OUT / "prompt2_mndwi_pairs.csv", pairs)
    table, event_rows, paired_summary = [], [], []
    for population, members in (("canonical90", [r for r in per_chip if r["filename"] in original]), ("all446", per_chip)):
        for variant in predictions:
            total = sum(r["cells"][variant] for r in members)
            chip_scores = [scores(r["cells"][variant]) for r in members]
            water = [s["water_IoU"] for s in chip_scores if s["water_IoU"] is not None]
            grouped = {}
            for record in members:
                group = record["location_prefix_proxy"]
                grouped[group] = grouped.get(group, np.zeros((2, 3), dtype=np.int64)) + record["cells"][variant]
            group_scores = [scores(cell)["water_IoU"] for cell in grouped.values()]
            table.append({"population": population, "n_chips": len(members), "variant": variant,
                          "TP": int(total[1, 0]), "FP": int(total[1, 1]), "FN": int(total[1, 2]), "TN": int(total[0, 0]),
                          **scores(total), "per_chip_mean_water_IoU": float(np.mean(water)) if water else None,
                          "n_defined_chip_water_IoU": len(water), "n_undefined_chip_water_IoU": len(members) - len(water),
                          "per_chip_mean_mIoU_original_zero_union_convention": float(np.mean([np.mean(r["cells"][variant][:, 0] / np.maximum(r["cells"][variant].sum(1), 1)) for r in members])),
                          "n_location_prefix_proxies": len(grouped),
                          "location_proxy_macro_water_IoU": float(np.mean([s for s in group_scores if s is not None])),
                          "RSE_floodmap_macro_water_IoU": None,
                          "grouping_status": "filename-prefix proxy; official event/map correspondence unknown"})
            for group, cell in grouped.items():
                event_rows.append({"population": population, "variant": variant, "location_prefix_proxy": group,
                                   "grouping_status": "not_verified_RSE_event_or_map", "aggregation": "sum_chip_confusions_then_score",
                                   "n_chips": sum(r["location_prefix_proxy"] == group for r in members),
                                   "TP": int(cell[1, 0]), "FP": int(cell[1, 1]), "FN": int(cell[1, 2]), "TN": int(cell[0, 0]), **scores(cell)})
        for variant in predictions:
            if variant == "baseline":
                continue
            subset = [r for r in pairs if r["variant"] == variant and (population == "all446" or r["canonical90"])]
            paired_summary.append({"population": population, "variant": variant, "reference": subset[0]["reference"],
                                   **{k: sum(r[k] for r in subset) for k in ("valid_pixels", "changed_valid", "changed_ignored", "reference0_variant1_valid", "reference1_variant0_valid")}})
    write_csv(OUT / "prompt2_mndwi_comparison.csv", table)
    write_csv(OUT / "prompt2_mndwi_event_counts.csv", event_rows)

    def probe(g, s):
        den = g + s
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = np.float32(g - s) / np.maximum(np.float32(den), np.float32(1e-6)), np.float32(g - s) / np.float32(den)
        return {"green": g, "swir1": s, "denominator": den, "clamped_positive": bool(a > 0),
                "literal_positive": bool(b > 0), "literal_finite": bool(np.isfinite(b))}

    source_audit = json.loads((OUT / "prompt2_mndwi_sources/source_audit.json").read_text())
    a, b = table[0], next(r for r in table if r["population"] == "all446" and r["variant"] == "baseline")
    result = {"schema_version": 1, "status": "complete", "device": "cpu", "torch_threads": 1,
              "opencv_threads": 1, "dataloader_workers": 0, "elapsed_seconds": time.monotonic() - start,
              "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
              "cached_counts": {"path": str(CACHE), "sha256": sha(CACHE), "identity": cache["identity"],
                                "canonical90_reused_and_exact_current_pixel_counts_verified": True,
                                "historical_reproduction_status": "Current frozen reproduction agrees archived summary to five decimals; original historical prediction identity not supplied"},
              "sources": source_audit, "inputs": inputs,
              "reader_identity": {"installed_path": installed_reader.__file__, "installed_sha256": sha(installed_reader.__file__),
                                  "project_source_sha256": {p: sha(ROOT / p) for p in ("data/sen1floods11.py", "evaluation/perscene_eval.py", "scripts/revision_mndwi_diagnostic.py")},
                                  "package_versions": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "rasterio", "terratorch", "albumentations")}},
              "split_identity": {name: {"path": str(split_dir / f"flood_{filename}_data.txt"), "sha256": sha(split_dir / f"flood_{filename}_data.txt"), "n": len(splits[name])}
                                 for name, filename in (("train", "train"), ("validation", "valid"), ("test", "test"), ("Bolivia", "bolivia"))},
              "mask_semantics": "rioxarray masked=True; input NaN/nodata->0; label NaN/nodata->-1; valid label!=-1; no additional masks or clipping; A.Resize512 image bilinear/mask nearest is identity on native512",
              "paired_summary": paired_summary,
              "numeric_totals": {k: sum(r[k] for r in inputs) for k in ("source_green_swir_nodata_with_valid_label", "denominator_zero_valid", "denominator_negative_valid", "denominator_clamped_valid", "recovered_nonfinite_valid", "raw_green_swir_equal_valid")},
              "synthetic_arithmetic_probes": [probe(g, s) for g, s in ((.2, .1), (.1, .2), (0., 0.), (-.1, -.2), (1e-8, 0.))],
              "arithmetic_findings": {"float_cast_before_subtraction": True, "green_tensor_position": 1, "swir1_tensor_position": 4,
                                      "green_raster_1based_band": 3, "swir1_raster_1based_band": 12,
                                      "swapped_channel_probe_inverts_water_decision": True,
                                      "shared_positive_gain": "Mathematically cancels; pre-existing gain_numerics artifact reused, no duplicate gain run",
                                      "shared_positive_offset": "Numerator unchanged; zero-threshold sign unchanged while denominator stays positive",
                                      "per_band_zscore": "Never fed directly to MNDWI; original predictor restores reflectance first",
                                      "unsigned_underflow": "Reader casts tofloat32 before scaling/subtraction; synthetic uint16(100)-uint16(200) wraps but actual path avoids it"},
              "common_gain_reuse": {"path": "/data1/schmalstieg/flooddistill/revision_20261005/gain_numerics.json", "sha256": sha("/data1/schmalstieg/flooddistill/revision_20261005/gain_numerics.json")},
              "root_cause": {"status": "unresolved",
                             "canonical90_water_IoU": a["water_IoU"], "all446_pooled_water_IoU": b["water_IoU"],
                             "population_delta_percentage_points": 100 * (b["water_IoU"] - a["water_IoU"]),
                             "all446_per_chip_water_IoU": b["per_chip_mean_water_IoU"],
                             "RSE_reported_water_IoU": .7439,
                             "remaining_unknowns": ["official RSE floodmap/event grouping", "official index validity/cloud/nodata/GSW masks and denominators", "exact source input product preprocessing", "original historical prediction/run identity"],
                             "no_matched_implementation_deficit_established": True,
                             "B8_vs_B8A_cannot_affect_two_band_MNDWI": True,
                             "all446_holdout_warning": "Fixed-rule protocol diagnostic only; all446 includes training/validation material and teacher task data; no learned-model all446 evaluation performed"}}
    write_json(OUT / "prompt2_mndwi_diagnostic.json", result)
    print(json.dumps({"status": result["status"], "root_cause": result["root_cause"], "pairs": paired_summary}, indent=2))


if __name__ == "__main__":
    main()
