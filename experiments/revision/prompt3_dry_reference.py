"""Bounded model-blind dry-chip package and explicit human-reference import.

No model is loaded. Original WorldFloods all-water labels define candidates;
human review, not this script, opens the scoring gate.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "experiments/revision/prompt3_dry_verification"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def build(out):
    import numpy as np
    import rasterio
    from rasterio.windows import Window, bounds
    from rasterio.warp import transform_bounds
    from PIL import Image

    if (out / "manifest.json").exists():
        raise RuntimeError("Frozen candidate package exists; refusing to resample")
    refs_path = REPO / "experiments/revision/prompt2_reference_manifest.json"
    geo_path = REPO / "experiments/revision/prompt2_geographic_screen.json"
    checks_path = REPO / "experiments/revision/prompt2_data_checks.json"
    refs = json.loads(refs_path.read_text())
    checks = json.loads(checks_path.read_text())
    wf = refs["WorldFloods"]
    sturm_events = {r["event"] for r in refs["STURM"].values()}
    shared = sorted({r["event"] for r in wf.values()} & sturm_events)
    excluded_maps = {p["right"] for section in ("s11_train__wf", "s11_valid__wf")
                     for p in checks["footprint_checks"][section]["pairs"]}
    frame, reasons = [], Counter()
    for name, ref in sorted(wf.items()):
        if ref["event"] in shared:
            reasons["map_shared_STURM_event"] += 1
            continue
        if name in excluded_maps:
            reasons["map_intersects_S11_train_valid_footprint"] += 1
            continue
        with rasterio.open(ref["reference"]["path"]) as src:
            gt = src.read()
            for r in range(0, src.height, 512):
                for c in range(0, src.width, 512):
                    if r + 512 > src.height or c + 512 > src.width:
                        reasons["partial_boundary_window"] += 1
                        continue
                    tile = gt[:, r:r + 512, c:c + 512]
                    if not bool(np.all(tile[0] == 1)):
                        reasons["original_reference_cloud_or_unknown"] += 1
                        continue
                    if not bool(np.all(tile[1] == 1)):
                        reasons["original_all_water_reference_not_all_land"] += 1
                        continue
                    frame.append({"map": name, "event": ref["event"], "window": [c, r, 512, 512]})
    # Fixed metadata-only event round-robin; within-event order is seeded before
    # any spectral usability rejection. Never consult a tested prediction.
    rng = np.random.default_rng(42)
    groups = defaultdict(list)
    for candidate in frame:
        groups[candidate["event"]].append(candidate)
    events = sorted(groups)
    rng.shuffle(events)
    queues = {}
    for event in events:
        rng.shuffle(groups[event])
        queues[event] = deque(groups[event])
    order = []
    while any(queues.values()):
        for event in events:
            if queues[event]:
                order.append(queues[event].popleft())
    selected = []
    out.mkdir(parents=True, exist_ok=True)
    for candidate in order:
        if len(selected) >= 60:
            break
        ref = wf[candidate["map"]]
        win = Window(*candidate["window"])
        with rasterio.open(ref["image"]["path"]) as src:
            raw = src.read(window=win)
            profile = src.profile.copy()
            affine = src.window_transform(win)
            wgs84 = transform_bounds(src.crs, "EPSG:4326", *bounds(win, src.transform))
        if not np.all(np.isfinite(raw[:13])) or np.any(np.all(raw[:13] == 0, axis=0)):
            reasons["spectral_nonfinite_or_all13zero"] += 1
            continue
        with rasterio.open(ref["reference"]["path"]) as src:
            label = src.read(window=win)
            lp = src.profile.copy()
        ident = candidate["map"].replace(".tif", "") + f"__r{int(win.row_off)}_c{int(win.col_off)}"
        folder = out / "candidates" / ident
        folder.mkdir(parents=True, exist_ok=True)
        lp.update(width=512, height=512, transform=affine, count=2, compress="deflate")
        label_path = folder / "original_label.tif"
        with rasterio.open(label_path, "w", **lp) as dst:
            dst.write(label)
        Image.fromarray(np.full((512, 512), 255, np.uint8)).save(folder / "valid_region.png")
        # Fixed display mapping only; these PNGs never enter model evaluation.
        display = np.clip(raw[:13].astype(np.float32) * 1e-4 / 0.4, 0, 1)
        rgb = (display[[3, 2, 1]].transpose(1, 2, 0) * 255).astype(np.uint8)
        swir = (display[[11, 8, 3]].transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(folder / "true_colour.png")
        Image.fromarray(swir).save(folder / "swir_nir_red.png")
        metadata = json.loads(Path(ref["metadata"]["path"]).read_text())
        row = {**candidate, "id": ident, "image_path": ref["image"]["path"],
               "original_reference_path": ref["reference"]["path"],
               "original_reference_file_sha256": ref["reference"]["sha256"],
               "input_window_sha256": array_hash(raw), "input_dtype": str(raw.dtype),
               "input_shape": list(raw.shape), "label_window_sha256": array_hash(label),
               "label_dtype": str(label.dtype), "label_shape": list(label.shape),
               "label_copy_path": str(label_path), "label_copy_sha256": file_hash(label_path),
               "valid_region_path": str(folder / "valid_region.png"),
               "true_colour_path": str(folder / "true_colour.png"),
               "infrared_path": str(folder / "swir_nir_red.png"),
               "bounds_wgs84": list(wgs84), "native_affine": list(affine)[:6],
               "crs": str(profile["crs"]), "s2_acquisition_date": ref["date"],
               "reference_acquisition_date": metadata.get("satellite date"),
               "reference_satellite": metadata.get("satellite"),
               "aoi": metadata.get("aoi_code"), "reference_source": "WorldFloodsv2 original gt land/water and cloud bands; no JRC exclusion",
               "reference_metadata_path": ref["metadata"]["path"],
               "reference_metadata_sha256": ref["metadata"]["sha256"],
               "reference_water_pixels": 0, "valid_pixels": 512 * 512,
               "automated_eligibility": "all original labels clear and land; fully finite nonempty spectrum",
               "review_status": "pending", "model_predictions_visible": False}
        selected.append(row)
        atomic(folder / "metadata.json", row)
    manifest = {"schema": "prompt3_dry_candidates_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Reference-labelled dry-chip candidates within WorldFloodsv2 test acquisitions; neither fully dry acquisitions nor random operational landscapes",
                "candidate_cap": 60, "selection_seed": 42,
                "selection_policy": "full512 windows; original all-water land/clear labels; exclude shared STURM events and known S11 train/validation intersections; seeded event round-robin; reject unusable spectra without model predictions",
                "source_fingerprints": {str(p): file_hash(p) for p in (refs_path, geo_path, checks_path, Path(__file__))},
                "sampling_frame": frame, "ordered_eligible_frame_sha256": hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest(),
                "frame_count": len(frame), "exclusions": dict(reasons), "excluded_shared_STURM_events": shared,
                "selected_count": len(selected), "events": dict(Counter(r["event"] for r in selected)),
                "known_overlap": "No existing S11 train/validation raster footprint intersects these WF scenes. The existing original/current-N2500 WF centroid screens find 0 at25km/30days, but do not prove every candidate-window exclusion or absence of wider historical selection coverage.",
                "residual_uncertainty": "GDACS native footprints absent and one training centroid absent; benchmark-guided development history unresolved; label/reference and S2 dates can differ; repeated tiles within events remain dependent.",
                "status": "needs-reference-verification", "candidates": selected}
    atomic(out / "manifest.json", manifest)
    atomic(out / "annotations.template.json", {"candidate_manifest_sha256": file_hash(out / "manifest.json"),
           "status": "pending", "annotations": [{"id": r["id"], "status": "pending", "reviewer": "", "reviewed_utc": "",
               "reference_source": "", "valid_region": "full_tile", "fully_assessable": False,
               "notes": "Inspect RGB, SWIR and original reference; cloud/unknown is not land. Review date offset."} for r in selected]})
    print(json.dumps({"selected": len(selected), "frame": len(frame), "events": manifest["events"], "status": manifest["status"]}))


def freeze(out, annotations, resume_config):
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    review = json.loads(annotations.read_text())
    if review.get("candidate_manifest_sha256") != file_hash(manifest_path):
        raise RuntimeError("Review manifest fingerprint mismatch")
    rows = review.get("annotations", [])
    candidates = {r["id"]: r for r in manifest["candidates"]}
    if len(rows) != len(candidates) or {r["id"] for r in rows} != set(candidates):
        raise RuntimeError("Require exactly one disposition per frozen candidate")
    accepted = []
    for row in rows:
        if row.get("status") not in {"verified_dry", "water_present", "unassessable"}:
            raise RuntimeError("Pending or invented disposition cannot open reference gate")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("reviewer", "reviewed_utc", "reference_source")):
            raise RuntimeError("Every disposition needs reviewer, date and reference source")
        datetime.fromisoformat(row["reviewed_utc"].replace("Z", "+00:00"))
        if row["status"] == "verified_dry":
            if row.get("fully_assessable") is not True or row.get("valid_region") != "full_tile":
                raise RuntimeError("This adapter accepts only independently reviewed fully assessable tiles")
            candidate = candidates[row["id"]]
            if file_hash(candidate["label_copy_path"]) != candidate["label_copy_sha256"]:
                raise RuntimeError("Original-label copy changed during review")
            accepted.append({**candidate, "review": row})
    if not accepted:
        raise RuntimeError("No independently verified dry chips; keep gate blocked")
    verified_path = out / "verified_manifest.json"
    frozen = {"schema": "prompt3_dry_verified_v1", "status": "verified", "scope": manifest["scope"],
              "candidate_manifest": str(manifest_path), "candidate_manifest_sha256": file_hash(manifest_path),
              "annotations_path": str(annotations.resolve()), "annotations_sha256": file_hash(annotations),
              "frozen_utc": datetime.now(timezone.utc).isoformat(), "model_scoring_started": False,
              "candidates": accepted, "excluded_dispositions": dict(Counter(r["status"] for r in rows if r["status"] != "verified_dry")),
              "overlap_and_uncertainty": {k: manifest[k] for k in ("known_overlap", "residual_uncertainty")}}
    if verified_path.exists():
        prior = json.loads(verified_path.read_text())
        if prior.get("annotations_sha256") != frozen["annotations_sha256"] or prior.get("candidate_manifest_sha256") != frozen["candidate_manifest_sha256"]:
            raise RuntimeError("Verified set is already frozen; do not revise after scoring")
    else:
        atomic(verified_path, frozen)
    gate = {"status": "verified", "verified_manifest": str(verified_path), "verified_manifest_sha256": file_hash(verified_path),
            "candidate_manifest_sha256": file_hash(manifest_path), "annotations_sha256": file_hash(annotations),
            "n_verified_dry_chips": len(accepted), "scope": manifest["scope"]}
    atomic(out / "reference_gate.json", gate)
    if resume_config:
        config = json.loads(resume_config.read_text())
        argv = config.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise RuntimeError("Resume configuration must contain exact existing-worker argv")
        # This only resumes the already configured queue; it never creates jobs.
        subprocess.run(argv, cwd=config.get("cwd", str(REPO)), check=True)
    print(json.dumps(gate))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("build", "import-review"))
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--annotations", type=Path)
    ap.add_argument("--resume-config", type=Path)
    args = ap.parse_args()
    if args.action == "build":
        build(args.out.resolve())
    else:
        if not args.annotations:
            ap.error("import-review requires --annotations")
        freeze(args.out.resolve(), args.annotations.resolve(), args.resume_config)
