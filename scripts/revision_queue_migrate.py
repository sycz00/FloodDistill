"""Move the stopped finite queue to a new frozen wrapper, retaining run history."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import revision_queue as q


def migrate(old_path, new_path):
    old_path, new_path = Path(old_path).resolve(), Path(new_path).resolve()
    old_state_path = old_path.with_name("manifest.state.json")
    new_state_path = new_path.with_name("manifest.state.json")
    old, new = q.load_manifest(old_path), q.load_manifest(new_path)
    old_jobs = {j["id"]: j for j in old["jobs"]}
    new_jobs = {j["id"]: j for j in new["jobs"]}
    q.initialize(new_path, new_state_path, new)
    recovered, reused = [], []
    with q.edit_state(old_state_path) as prior:
        if not prior["stop_requested"] or q.alive(prior.get("worker")):
            raise ValueError("Old worker must finish its graceful stop before migration")
        if any(r["status"] in {"running", "launching"} for r in prior["jobs"].values()):
            raise ValueError("An old attempt is still active")
        with q.edit_state(new_state_path) as state:
            if q.alive(state.get("worker")) or any(r["attempts"] for r in state["jobs"].values()):
                raise ValueError("Destination queue already used")
            for name, job in new_jobs.items():
                if name not in old_jobs:
                    continue
                record = copy.deepcopy(prior["jobs"][name])
                if q.digest(job) != record["identity_sha256"] or q.digest(job) != q.digest(old_jobs[name]):
                    raise ValueError(f"Existing job identity changed: {name}")
                if record["status"] == "succeeded":
                    valid, evidence = q.validate_outputs(job)
                    expected = record.get("evidence") if job["kind"] == "gate" else record["attempts"][-1]["output_validation"]
                    if not valid or evidence != expected:
                        raise ValueError(f"Completed output changed: {name}")
                    reused.append(name)
                elif record["status"] == "failed" and record.get("reason") == "'wf__g0'":
                    attempt = record["attempts"][-1]
                    if attempt.get("exit_code") != 0:
                        raise ValueError(f"Cannot recover nonzero inference: {name}")
                    valid, evidence = q.validate_outputs(job)
                    if not valid or q.check_inputs(job):
                        raise ValueError(f"Cannot validate complete inference: {name}: {evidence}")
                    record["validation_recovery"] = {
                        "at": q.now(), "old_reason": record["reason"],
                        "old_output_validation": attempt.get("output_validation"),
                        "reason": "Decimal condition key was split by the old queue field accessor; no inference rerun",
                        "validator_sha256": q.file_hash(Path(q.__file__).resolve()),
                        "original_receipt_preserved": attempt.get("receipt_path"),
                    }
                    attempt["output_validation"] = evidence
                    record.update(status="succeeded", reason=None)
                    recovered.append(name)
                elif record.get("block_kind") == "dependency":
                    record.update(status="pending", reason=None, block_kind=None)
                state["jobs"][name] = record
            state.update(paused=True, stop_requested=False, migration={
                "at": q.now(), "source_manifest": str(old_path),
                "source_manifest_file_sha256": q.file_hash(old_path),
                "source_state_sha256": q.digest(prior), "worker_stopped": True,
                "reused_succeeded": reused, "recovered_validation_only": recovered,
                "all_existing_job_identities_preserved": True,
            })
    q.atomic_json(new_path.parent / "migration.json", q.read_json(new_state_path)["migration"])
    return {"manifest": str(new_path), "reused": len(reused), "recovered": recovered}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.old, args.new)))
