#!/usr/bin/env python3
"""CPU-only persistence/recovery checks; records exactly which guarantees passed."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from types import SimpleNamespace

import revision_queue as queue


def wait_for(test, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = test()
        if result:
            return result
        time.sleep(0.1)
    raise RuntimeError("queue smoke check timed out")


def cli(script, manifest, action):
    result = subprocess.run([sys.executable, str(script), action, "--manifest", str(manifest)],
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout) if result.stdout else None


def wait_state(path, predicate):
    def check():
        state = queue.read_json(path)
        return state if predicate(state) else None
    return wait_for(check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    root = Path(args.out_dir).resolve() if args.out_dir else Path(tempfile.mkdtemp(prefix="flooddistill-queue-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    script = Path(queue.__file__).resolve()
    workers = []
    checks = {}
    started = time.monotonic()
    try:
        sentinel_dir = root / "sentinel"
        sentinel_dir.mkdir()
        manifest = sentinel_dir / "manifest.json"
        initiator = sentinel_dir / "initiator.json"
        # Each job independently verifies the initiating process identity is gone.
        program = "import json,os,time;from pathlib import Path;time.sleep(float(__import__('sys').argv[3]));p=Path(__import__('sys').argv[1]);info=json.loads(Path(__import__('sys').argv[2]).read_text());f=Path('/proc/%s/stat'%info['pid']);alive=f.exists() and f.read_text().split(') ')[1].split()[19]==info['start_ticks'];p.write_text(json.dumps({'parent_exited':not alive,'pid':os.getpid()}))"
        jobs = [{"id": f"sentinel{i}", "argv": [sys.executable, "-c", program,
                                                   str(sentinel_dir / f"job{i}.json"), str(initiator), "0.6"],
                 "cwd": str(sentinel_dir), "gpu": False, "deps": ["sentinel1"] if i == 2 else [],
                 "outputs": [{"path": f"job{i}.json", "type": "json", "equals": {"parent_exited": True}}]}
                for i in (1, 2)]
        queue.atomic_json(manifest, {"schema_version": 1, "poll_seconds": 0.1, "jobs": jobs})
        starter_program = "import json,os,subprocess,sys;from pathlib import Path;p=os.getpid();Path(sys.argv[1]).write_text(json.dumps({'pid':p,'start_ticks':Path('/proc/%s/stat'%p).read_text().split(') ')[1].split()[19]}));subprocess.run([sys.executable,sys.argv[2],'start','--manifest',sys.argv[3]],check=True)"
        subprocess.run([sys.executable, "-c", starter_program, str(initiator), str(script), str(manifest)], check=True, capture_output=True)
        state_path = manifest.with_name("manifest.state.json")
        workers.append((manifest, state_path))
        sentinel = wait_state(state_path, lambda s: all(j["status"] == "succeeded" for j in s["jobs"].values()))
        assert all(len(j["attempts"]) == 1 for j in sentinel["jobs"].values())
        assert queue.read_json(sentinel_dir / "job2.json")["parent_exited"]
        checks["two_cpu_jobs_after_initiator_exit"] = True

        recovery_dir = root / "recovery"
        recovery_dir.mkdir()
        manifest = recovery_dir / "manifest.json"
        command = "import sys,time,json;from pathlib import Path;p=Path(sys.argv[1]);f=open(p,'a');f.write('launch\\n');f.close();time.sleep(2);Path(sys.argv[2]).write_text(json.dumps({'valid':True}))"
        jobs = [{"id": f"recovery{i}", "argv": [sys.executable, "-c", command,
                                                   str(recovery_dir / f"launch{i}.txt"), str(recovery_dir / f"job{i}.json")],
                 "cwd": str(recovery_dir), "gpu": False,
                 "deps": ["recovery1"] if i == 2 else [],
                 "outputs": [{"path": f"job{i}.json", "type": "json", "equals": {"valid": True}}]}
                for i in (1, 2)]
        queue.atomic_json(manifest, {"schema_version": 1, "poll_seconds": 0.1, "jobs": jobs})
        cli(script, manifest, "start")
        state_path = manifest.with_name("manifest.state.json")
        workers.append((manifest, state_path))
        running = wait_for(lambda: (s if (s := queue.read_json(state_path))["jobs"]["recovery1"]["status"] == "running" else None))
        first_worker = running["worker"]
        assert queue.alive(first_worker)
        os.kill(first_worker["pid"], signal.SIGKILL)  # Only this test's owned worker.
        wait_for(lambda: not queue.alive(first_worker))
        cli(script, manifest, "start")
        cli(script, manifest, "pause")
        paused = wait_for(lambda: (s if (s := queue.read_json(state_path))["jobs"]["recovery1"]["status"] == "succeeded" else None))
        assert paused["jobs"]["recovery2"]["status"] == "pending"
        assert (recovery_dir / "launch1.txt").read_text().splitlines() == ["launch"]
        checks["restart_does_not_duplicate_active_job"] = True
        checks["pause_leaves_active_job_running"] = True
        cli(script, manifest, "resume")
        wait_for(lambda: (recovery_dir / "launch2.txt").exists())
        cli(script, manifest, "stop")
        stopped = wait_for(lambda: (s if (s := queue.read_json(state_path))["worker_status"] == "stopped" else None))
        assert stopped["jobs"]["recovery2"]["status"] == "succeeded"
        assert all(len(j["attempts"]) == 1 for j in stopped["jobs"].values())
        checks["graceful_stop_waits_for_active_job"] = True

        gate_dir = root / "gates"
        gate_dir.mkdir()
        manifest = gate_dir / "manifest.json"
        (gate_dir / "bad.json").write_text('{"miou": "NaN"}')
        safe = {"id": "safe", "cwd": str(gate_dir), "argv": [sys.executable, "-c", "from pathlib import Path;Path('safe.txt').write_text('done')"],
                "outputs": [{"path": "safe.txt"}]}
        queue.atomic_json(manifest, {"schema_version": 1, "poll_seconds": 0.1, "jobs": [
            {"id": "gate", "kind": "gate", "cwd": str(gate_dir), "outputs": [{"path": "bad.json", "type": "json", "finite": ["miou"]}]},
            {**safe, "id": "dependent", "deps": ["gate"], "outputs": [{"path": "dependent.txt"}]}, safe]})
        cli(script, manifest, "start")
        state_path = manifest.with_name("manifest.state.json")
        workers.append((manifest, state_path))
        gate_state = wait_for(lambda: (s if (s := queue.read_json(state_path))["worker_status"] == "finished" else None))
        assert gate_state["jobs"]["gate"]["status"] == "blocked"
        assert gate_state["jobs"]["dependent"]["status"] == "blocked"
        assert not gate_state["jobs"]["dependent"]["attempts"]
        assert gate_state["jobs"]["safe"]["status"] == "succeeded"
        checks["invalid_gate_blocks_descendants_and_allows_independent_job"] = True
        checks["finite_worker_termination"] = True

        busy_dir = root / "busy"
        busy_dir.mkdir()
        manifest = busy_dir / "manifest.json"
        shared_lock = busy_dir / "shared-gpu.lock"
        gpu_lock = queue.lock_fd(shared_lock)
        assert gpu_lock is not None
        queue.atomic_json(manifest, {"schema_version": 1, "poll_seconds": 0.1,
                                    "gpu_lock": str(shared_lock), "jobs": [
            {**safe, "id": "gpu_wait", "cwd": str(busy_dir), "gpu": True,
             "outputs": [{"path": "gpu.txt"}]},
            {**safe, "cwd": str(busy_dir)}]})
        try:
            cli(script, manifest, "start")
            state_path = manifest.with_name("manifest.state.json")
            workers.append((manifest, state_path))
            busy = wait_state(state_path, lambda s: s["jobs"]["safe"]["status"] == "succeeded")
            assert busy["jobs"]["gpu_wait"]["status"] == "pending"
            assert busy["jobs"]["gpu_wait"]["reason"] == "shared GPU lock busy"
            assert not busy["jobs"]["gpu_wait"]["attempts"]
            cli(script, manifest, "stop")
            wait_state(state_path, lambda s: s["worker_status"] == "stopped")
        finally:
            os.close(gpu_lock)
        checks["GPU_lock_wait_allows_independent_CPU_job"] = True

        # Reuse must fail if a previously validated artifact changes after restart.
        queue.atomic_json(gate_dir / "bad.json", {"miou": 0.5})
        cli(script, gate_dir / "manifest.json", "resume")
        gate_path = gate_dir / "manifest.state.json"
        # The existing dependent command deliberately lacks its declared output.
        changed = wait_state(gate_path, lambda s: s["worker_status"] == "finished" and s["jobs"]["dependent"]["status"] == "failed")
        assert changed["jobs"]["gate"]["status"] == "succeeded"
        checks["zero_exit_without_expected_output_fails"] = True
        (gate_dir / "safe.txt").write_text("changed")
        cli(script, gate_dir / "manifest.json", "resume")
        changed = wait_state(gate_path, lambda s: s["jobs"]["safe"]["status"] == "failed")
        assert len(changed["jobs"]["safe"]["attempts"]) == 1
        checks["restart_rejects_changed_completed_artifact"] = True
        limits = {"host_limits": {"max_load_per_cpu": 1.5,
                  "min_available_ram_gib": 32, "min_free_disk_gib": 24,
                  "disk_path": str(root)}}
        gpu_job = {"gpu": True, "cwd": str(root)}
        with patch.object(queue.os, "sched_getaffinity", return_value=set(range(48))), \
             patch.object(queue.os, "getloadavg", return_value=(96, 0, 0)), \
             patch.object(queue.Path, "read_text", return_value="MemAvailable: 67108864 kB\n"), \
             patch.object(queue.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 2**30)):
            assert not queue.host_available(limits, gpu_job)[0]
            assert queue.host_available(limits, {"gpu": False})[0]
            with patch.object(queue.os, "getloadavg", return_value=(24, 0, 0)):
                assert queue.host_available(limits, gpu_job)[0]
                with patch.object(queue.Path, "read_text", return_value="MemAvailable: 1048576 kB\n"):
                    assert not queue.host_available(limits, gpu_job)[0]
                with patch.object(queue.shutil, "disk_usage", return_value=SimpleNamespace(free=2**30)):
                    assert not queue.host_available(limits, gpu_job)[0]
        checks["host_load_ram_disk_admission_and_CPU_progress"] = True
        checks["gpu_jobs_launched"] = False
        result = {"checks": checks, "elapsed_seconds": time.monotonic() - started, "artifacts": str(root),
                  "not_tested": ["actual_editor_closure", "logout", "reboot", "host_sleep", "GPU_training"]}
        queue.atomic_json(root / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        for manifest, state_path in workers:
            if state_path.exists() and queue.alive(queue.read_json(state_path).get("worker")):
                cli(script, manifest, "stop")


if __name__ == "__main__":
    main()
