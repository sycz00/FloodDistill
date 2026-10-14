"""CPU-only checks of the existing runner's detached queue and time reservations."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def wait_until(callback, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = callback()
        if value:
            return value
        time.sleep(.05)
    raise TimeoutError("CPU queue sentinel exceeded its own check timeout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = args.out_dir.resolve() if args.out_dir else Path(tempfile.mkdtemp(prefix="compression-queue-check-"))
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "experiments" / "revision_queue.py"
    frozen = root / "revision_queue.py"
    shutil.copyfile(source, frozen)
    spec = importlib.util.spec_from_file_location("compression_queue_under_test", frozen)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    workers = []
    checks = {}
    def cli(manifest, action):
        result = subprocess.run([sys.executable, str(frozen), action, "--manifest", str(manifest)],
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout) if result.stdout else None
    def manifest_file(directory, jobs, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        queue.atomic_json(path, {"schema_version": 1, "poll_seconds": .1, "nice": 0,
                                "gpu_lock": str(root / "unused_gpu.lock"), "jobs": jobs, **kwargs})
        return path
    def state(path):
        return path.with_name("manifest.state.json")
    def terminal(path):
        if not state(path).exists():
            return None
        value = queue.read_json(state(path))
        return value if value["worker_status"] == "finished" else None
    def job(name, directory, program, deps=(), **kwargs):
        return {"id": name, "cwd": str(directory), "argv": [sys.executable, "-c", program],
                "gpu": False, "deps": list(deps),
                "outputs": [{"path": name + ".json", "type": "json", "equals": {"valid": True}}], **kwargs}

    try:
        # Exercise reservation validation without querying or using any GPU.
        directory = root / "validation"
        example = job("gpu", directory, "pass", gpu=True,
                      max_seconds=3, shutdown_grace_seconds=.5)
        valid = manifest_file(directory, [example], gpu_budget_seconds=3)
        queue.load_manifest(valid)
        invalids = [({"gpu_budget_seconds": 2}, {}),
                    ({"gpu_budget_seconds": 3}, {"max_seconds": 0}),
                    ({"gpu_budget_seconds": 3}, {"shutdown_grace_seconds": -1}),
                    ({"gpu_budget_seconds": 3}, {"shutdown_grace_seconds": float("nan")}),
                    ({"gpu_budget_seconds": 3}, {"retry": {"reason": "test", "exit_codes": [124]}}),
                    ({}, {"gpu": False, "max_seconds": -1}),
                    ({}, {"gpu": False, "max_seconds": 1, "shutdown_grace_seconds": float("inf")})]
        for i, (changes, job_changes) in enumerate(invalids):
            bad = directory / f"invalid{i}.json"
            data = queue.read_json(valid)
            data.pop("gpu_budget_seconds")
            data.update(changes)
            data["jobs"] = [{**example, **job_changes}]
            bad.write_text(json.dumps(data) + "\n")
            try:
                queue.load_manifest(bad)
            except ValueError:
                continue
            raise AssertionError(f"Invalid time reservation {i} was accepted")
        checks["finite_positive_reservations_and_gpu_budget_sum"] = True
        checks["budgeted_gpu_commands_cannot_retry"] = True

        directory = root / "detached"
        initiator = directory / "initiator.json"
        def sentinel(name):
            return ("import json,time;from pathlib import Path;time.sleep(.5);"
                    f"i=json.loads(Path({str(initiator)!r}).read_text());"
                    "p=Path('/proc/%s/stat'%i['pid']);"
                    "alive=p.exists() and p.read_text().split(') ')[1].split()[19]==i['start_ticks'];"
                    f"Path({name + '.json'!r}).write_text(json.dumps({{'valid':not alive,'initiator_exited':not alive}}))")
        manifest = manifest_file(directory, [job("one", directory, sentinel("one"),
                                      max_seconds=2, shutdown_grace_seconds=.5),
                     job("two", directory, sentinel("two"), deps=("one",),
                         max_seconds=2, shutdown_grace_seconds=.5)])
        starter = ("import json,os,subprocess,sys;from pathlib import Path;"
                   "p=os.getpid();Path(sys.argv[1]).write_text(json.dumps({'pid':p,'start_ticks':"
                   "Path('/proc/%s/stat'%p).read_text().split(') ')[1].split()[19]}));"
                   "subprocess.run([sys.executable,sys.argv[2],'start','--manifest',sys.argv[3]],check=True)")
        subprocess.run([sys.executable, "-c", starter, str(initiator), str(frozen), str(manifest)],
                       check=True, capture_output=True, text=True)
        workers.append(manifest)
        finished = wait_until(lambda: terminal(manifest))
        assert all(r["status"] == "succeeded" and len(r["attempts"]) == 1
                   for r in finished["jobs"].values())
        assert all(queue.read_json(directory / f"{name}.json")["initiator_exited"] for name in ("one", "two"))
        checks["two_jobs_after_initiating_process_exits"] = True
        checks["finite_detached_worker_terminates"] = True

        directory = root / "bounded"
        # Fork inherits the command session and lock FDs, like a DataLoader.
        # Parent checkpoints and exits on TERM; the child ignores TERM. The
        # reservation must kill the whole owned group rather than only parent.
        bounded = "\n".join([
            "import os,signal,time,json",
            "from pathlib import Path",
            "def stop(sig,frame):",
            "    Path('limited.json').write_text(json.dumps({'valid':True,'partial_checkpoint':True}))",
            "    raise SystemExit(0)",
            "signal.signal(signal.SIGTERM,stop)",
            "child=os.fork()",
            "if child==0:",
            "    signal.signal(signal.SIGTERM,signal.SIG_IGN)",
            "    Path('child.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/%s/stat'%os.getpid()).read_text().split(') ')[1].split()[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))",
            "    time.sleep(60)",
            "else:",
            "    time.sleep(60)",
        ])
        valid_output = lambda name: f"from pathlib import Path;Path({name + '.json'!r}).write_text('{{\"valid\":true}}')"
        manifest = manifest_file(directory, [job("limited", directory, bounded,
                                               max_seconds=2, shutdown_grace_seconds=.5),
                      job("dependent", directory, valid_output("dependent"), deps=("limited",)),
                      job("independent", directory, valid_output("independent"))])
        cli(manifest, "start")
        workers.append(manifest)
        finished = wait_until(lambda: terminal(manifest))
        limited = finished["jobs"]["limited"]
        assert limited["status"] == "failed" and len(limited["attempts"]) == 1
        attempt = limited["attempts"][0]
        receipt = queue.read_json(queue.paths(state(manifest), "limited", 1)[0])
        assert receipt["exit_code"] == 124 and receipt["budget_exhausted"]
        assert receipt["command_elapsed_seconds"] <= 2.5
        child = queue.read_json(directory / "child.json")
        wait_until(lambda: not queue.alive(child))
        lock = queue.lock_fd(queue.paths(state(manifest), "limited", 1)[1])
        assert lock is not None, "A child retained the owned command lock after its cap"
        os.close(lock)
        assert queue.read_json(directory / "limited.json")["partial_checkpoint"]
        assert finished["jobs"]["dependent"]["status"] == "blocked"
        assert not finished["jobs"]["dependent"]["attempts"]
        assert finished["jobs"]["independent"]["status"] == "succeeded"
        checks.update({"time_cap_saves_partial_checkpoint_but_marks_failure": True,
                       "term_ignoring_owned_child_is_killed": True,
                       "child_lock_descriptors_released": True,
                       "failed_budget_blocks_descendants": True,
                       "unrelated_cpu_job_completes": True,
                       "no_automatic_budget_retry": True})
        result = {"status": "passed", "checks": checks, "gpu_jobs_executed": 0,
                  "runner_source_sha256": queue.file_hash(source),
                  "tested_runner_sha256": queue.file_hash(frozen),
                  "artifacts": str(root),
                  "not_tested": ["actual_editor_closure", "logout", "reboot", "host_sleep", "GPU training"]}
        assert result["runner_source_sha256"] == result["tested_runner_sha256"], "Runner changed during verification"
        queue.atomic_json(root / "result.json", result)
        print(json.dumps(result, indent=2))
    finally:
        for manifest in workers:
            if state(manifest).exists() and queue.alive(queue.read_json(state(manifest)).get("worker")):
                cli(manifest, "stop")


if __name__ == "__main__":
    main()
