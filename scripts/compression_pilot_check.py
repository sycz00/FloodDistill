"""CPU structural/provenance checks of the frozen finite compression pilot DAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import revision_queue as q


def check(path: Path, preflight_manifest: Path | None = None,
          preflight_result: Path | None = None) -> dict:
    raw = q.read_json(path)
    manifest = q.load_manifest(path)
    jobs = {job["id"]: job for job in manifest["jobs"]}
    specs = {}
    actions = {}
    for name, job in jobs.items():
        if job['kind'] == 'gate':
            assert not job['gpu']
            assert job['outputs'] and all(entry.get('type') == 'json' for entry in job['outputs'])
            for entry in job.get('inputs', []):
                assert q.file_hash(entry['path']) == entry['sha256']
            continue
        argv = job["argv"]
        assert "--spec" in argv and "--action" in argv
        spec_path = Path(argv[argv.index("--spec") + 1])
        actions[name] = argv[argv.index("--action") + 1]
        specs[name] = q.read_json(spec_path)
        input_pins = {entry["path"]: entry["sha256"] for entry in job["inputs"]}
        assert input_pins[str(spec_path)] == q.file_hash(spec_path)
        assert job["kind"] == "command"
        assert not job.get("retry"), "Pilot must not automatically retry or silently spend a cap twice"
        assert job["outputs"] and all(entry.get("type") == "json" and
               entry.get("equals", {}).get("status") == "complete" for entry in job["outputs"]), \
               "Existing runner uses output.type=json and output.equals, not format/fields"
        if job["gpu"]:
            assert job["max_seconds"] > job["shutdown_grace_seconds"] > 0
            assert job["required_free_mb"] > 0
        # Only source/config hashes are read here; cohort/teacher bytes have a
        # separate live preparation check and are rechecked by each runner.
        for entry in job["inputs"]:
            source = Path(entry["path"])
            if source.suffix == ".py" or source == spec_path:
                assert q.file_hash(source) == entry["sha256"], f"Frozen source changed: {source}"
    limits = raw.get("host_limits")
    assert limits and "resource_limits" not in raw, "Runner host admission uses host_limits"
    assert all(limits[key] > 0 for key in ("min_available_ram_gib", "min_free_disk_gib", "max_load_per_cpu"))
    assert Path(limits["disk_path"]).is_dir()
    assert str(limits["disk_path"]).startswith("/data1/"), "Check the real artifact disk, not repository snapshot disk"
    devices = manifest["gpu_devices"]
    assert len(devices) == 2 and len({d["uuid"] for d in devices}) == 2
    assert len({d["lock"] for d in devices}) == 2
    assert manifest.get("headroom_mb", 2048) >= 2048
    inventory_path = path.parent.parent / "inventory.json"
    inventory = q.read_json(inventory_path)
    assert {d["uuid"] for d in devices} == {d["uuid"] for d in inventory["gpus"]}
    smallest_gpu_total = min(d["total_mb"] for d in inventory["gpus"])
    for job in jobs.values():
        if job["gpu"]:
            assert job["required_free_mb"] + manifest.get("headroom_mb", 2048) <= smallest_gpu_total, \
                "Impossible GPU admission requirement (possibly double-counted headroom)"
    reservation = sum(job["max_seconds"] for job in jobs.values() if job["gpu"])
    assert reservation == raw["reserved_main_gpu_seconds"]
    assert reservation <= raw["gpu_budget_seconds"]
    preflight_reserved = raw['reserved_preflight_gpu_seconds']
    if preflight_manifest:
        earlier = q.load_manifest(preflight_manifest)
        first_reservation = sum(job["max_seconds"] for job in earlier["jobs"] if job["gpu"])
        assert earlier["gpu_budget_seconds"] >= first_reservation
        assert preflight_reserved >= first_reservation
    assert reservation + preflight_reserved <= raw["pilot_cap_seconds"] == 43200
    if preflight_result:
        measured = q.read_json(preflight_result)
        assert measured["status"] == "complete" and measured["outcome"] == "completed"
        assert measured["batch_size"] == 16 and measured["eval_batch_size"] == 4
        for job in jobs.values():
            if job["gpu"]:
                assert job["required_free_mb"] >= measured["peak_reserved_mb"]
        assert measured["teacher_frozen"] and measured["no_accuracy_inspected"]
    recipe = raw["recipe"]
    assert "1560" in recipe["qat"] and "10epochs" in recipe["qat"]
    assert "allQDQenabled" in recipe["qat"]
    assert "historical" in recipe["historical_qat_difference"]
    assert "all10epochs" in recipe["historical_qat_difference"]

    def ancestors(name):
        answer = set()
        todo = list(jobs[name]["deps"])
        while todo:
            parent = todo.pop()
            if parent not in answer:
                answer.add(parent)
                todo.extend(jobs[parent]["deps"])
        return answer
    terminal_names = {name for name, action in actions.items() if action == "terminal"}
    assert terminal_names
    source_names = set(jobs) - terminal_names - {"pilot_collection"}
    for name in source_names:
        assert not (ancestors(name) & terminal_names), "External feedback enters a source decision"
    for name in terminal_names:
        assert "source_design_freeze" in ancestors(name), "External readout starts before source design closes"
        assert Path(specs[name]["method_receipt"]) == Path(next(
            job["outputs"][0]["path"] for job in jobs.values()
            if job["outputs"][0]["path"] == specs[name]["method_receipt"]))
        assert specs[name]["protocols"] == {
            "s11": "S11_TEST512_LEGACY", "sturm": "P3_STURM_FINITE_SIX_BAND_SUPPORT_MASK_V1",
            "wf": "P3_WF_NATIVE_ZERO_PAD512_V1"}
        assert '/exports/' in specs[name]['artifact_dir']
        source_name = name.removeprefix('terminal_')
        assert specs[name]['seed'] == specs[source_name]['seed']
    collection = jobs["pilot_collection"]
    assert collection["run_after_terminal_deps"]
    assert set(collection["deps"]) == set(jobs) - {"pilot_collection"}
    assert set(raw["declared_frontier"]) == {"A0", "A1", "A2", "A3", "M0"}
    assert raw["declared_frontier"]["A2"]["status"].startswith("deferred")
    all_specs = list(specs.values())
    common = all_specs[0]
    pools = common["feature_pools"]
    assert all(0 < len(pools[key]) <= 256 for key in ("fit", "interaction"))
    assert not ({r["file"] for r in pools["fit"]} & {r["file"] for r in pools["interaction"]})
    assert not ({r["group"] for r in pools["fit"]} & {r["group"] for r in pools["interaction"]})
    assert all(len({r["group"] for r in pools[key]}) == len(pools[key]) for key in pools)
    assert len(common["quantizer_calibration_ids"]) == 89
    assert len(set(common["quantizer_calibration_ids"])) == 89
    for spec in all_specs:
        assert spec["seed"] in (42, 64, 91, 107)
        assert spec["feature_pools_sha256"] == q.digest(spec["feature_pools"]) == common["feature_pools_sha256"]
        assert spec["quantizer_calibration_ids"] == common["quantizer_calibration_ids"]
        assert spec["teacher_sha256"] == "3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60"
    cells = {}
    for name, action in actions.items():
        if (action == "qat" and specs[name]["method"] in ("B1", "B2", "B3")
                and specs[name]['seed'] == 42 and specs[name]['graph'] != 'M0'):
            spec = specs[name]
            cells.setdefault((spec["graph"], spec["precision"]), {})[spec["method"]] = spec
    assert set(cells) == {("A1", "W4A8"), ("A3", "W4A8"), ("A3", "W8A8"), ("A3", "W4A4")}
    for methods in cells.values():
        assert set(methods) == {"B1", "B2", "B3"}
        assert len({v["interaction_receipt"] for v in methods.values()}) == 1
        assert len({v["run_dir"] for v in methods.values()}) == 3, "Methods share mutable training output directories"
    for seed in (64, 91, 107):
        prefix = f'confirmation_s{seed}'
        fp, cal, inter = (prefix + suffix for suffix in ('_FP', '_PTQ', '_interaction'))
        assert specs[cal]['warmup_receipt'] == jobs[fp]['outputs'][0]['path']
        assert specs[inter]['calibration_receipt'] == jobs[cal]['outputs'][0]['path']
        for name in (fp, cal, inter, *(f'confirmation_{method}_s{seed}' for method in ('B1', 'B2', 'B3'))):
            assert specs[name]['seed'] == seed
            assert specs[name]['conditional_gate'] == jobs['mechanism_source_gate']['outputs'][0]['path']
            assert specs[name]['select_gate_cell']
            assert 'mechanism_source_gate' in ancestors(name)
            if actions[name] == 'qat':
                assert specs[name]['interaction_receipt'] == jobs[inter]['outputs'][0]['path']
                assert 'expansion_gate' not in specs[name], 'Confirmation must keep its own independently fitted metric'
    mb = [name for name, config in specs.items() if config.get('graph') == 'M0' and actions[name] != 'terminal']
    assert mb and all(specs[name]['seed'] == 42 for name in mb)
    assert 'M0_technical_preflight' in ancestors('M0_FP_s42')
    assert all(not specs[name].get('select_gate_cell') for name in mb)
    beta = [name for name, config in specs.items() if 'feature_metric_key' in config]
    assert len(beta) == 4
    assert {(specs[name]['feature_metric_key'], specs[name]['beta_multiplier']) for name in beta} == {
        ('B2', .5), ('B3', .5), ('B2', 2.), ('B3', 2.)}
    assert 'verification_preflight' in jobs and raw['verification_evidence']
    for name, job in jobs.items():
        if job['gpu'] and name not in terminal_names:
            assert 'verification_preflight' in ancestors(name)
    return {"status": "passed", "manifest": str(path), "manifest_sha256": q.file_hash(path),
            "job_count": len(jobs), "gpu_jobs": sum(job["gpu"] for job in jobs.values()),
            "main_reserved_gpu_seconds": reservation, "preflight_reserved_gpu_seconds": preflight_reserved,
            "total_reserved_gpu_seconds": reservation + preflight_reserved,
            "cap_gpu_seconds": 43200, "terminal_job_count": len(terminal_names),
            "smallest_inventory_gpu_total_mb": smallest_gpu_total,
            "checks": ["real_runner_schema", "host_admission_keys", "uuid_device_locks_and_headroom",
                "finite_total_reservation", "frozen_source_and_specs", "paired_common_state_and_isolated_outputs",
                "source_before_terminal_without_feedback", "training_only_disjoint_group_pools",
                "fixed_calibration_image_ids", "complete_frontier_and_failure_collector"],
            "gpu_jobs_executed": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.manifest.resolve(), args.preflight_manifest, args.preflight_result)
    q.atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
