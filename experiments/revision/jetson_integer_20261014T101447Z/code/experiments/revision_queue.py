#!/usr/bin/env python3
"""Finite local command queue. State/receipts are durable; jobs retain their locks.

Commands: init, start, run, status, pause, resume, stop; each accepts --manifest.
Linux only (the inspected project host). No scientific choices are made here.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone


TERMINAL = {"succeeded", "failed", "blocked"}
THREAD_ENV = {k: "2" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def lock_fd(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


@contextlib.contextmanager
def edit_state(path):
    fd = lock_fd(str(path) + ".lock", blocking=True)
    try:
        state = read_json(path) if Path(path).exists() else None
        yield state
        if state is not None:
            state["updated_at"] = now()
            atomic_json(path, state)
    finally:
        os.close(fd)


def process_identity(pid):
    """PID + boot ID + kernel start ticks prevents PID reuse reconciliation."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def alive(identity):
    return bool(identity and process_identity(identity["pid"]) == identity)


def resolved(path, cwd):
    candidate = Path(path).expanduser()
    return str((candidate if candidate.is_absolute() else Path(cwd) / candidate).resolve())


def load_manifest(path):
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("jobs"), list):
        raise ValueError("manifest requires schema_version=1 and jobs list")
    ids = [j["id"] for j in manifest["jobs"]]
    if len(ids) != len(set(ids)) or any(not i or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in i) for i in ids):
        raise ValueError("job IDs must be unique safe filenames")
    jobs = []
    for raw in manifest["jobs"]:
        job = dict(raw)
        job["kind"] = job.get("kind", "command")
        if job["kind"] not in {"command", "gate"}:
            raise ValueError("job kind must be command or gate")
        job["cwd"] = resolved(job.get("cwd", "."), Path(path).parent)
        if not Path(job["cwd"]).is_dir():
            raise ValueError(f"{job['id']}: cwd missing: {job['cwd']}")
        if job["kind"] == "command" and (not isinstance(job.get("argv"), list) or not job["argv"] or any(not isinstance(a, str) for a in job["argv"])):
            raise ValueError(f"{job['id']}: argv must be a nonempty string list")
        if job["kind"] == "command" and not Path(job["argv"][0]).is_absolute():
            raise ValueError(f"{job['id']}: argv[0] must be an absolute executable path")
        job["env"] = {"PATH": os.defpath, "HOME": str(Path.home()), "LANG": "C.UTF-8",
                      **THREAD_ENV, **manifest.get("env", {}), **job.get("env", {})}
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in job["env"].items()):
            raise ValueError("environment keys/values must be strings")
        job["deps"] = job.get("deps", [])
        if any(d not in ids or d == job["id"] for d in job["deps"]):
            raise ValueError(f"{job['id']}: invalid dependency")
        for key in ("inputs", "outputs"):
            job[key] = [dict(v, path=resolved(v["path"], job["cwd"])) for v in job.get(key, [])]
        if not job["outputs"]:
            raise ValueError(f"{job['id']}: declare expected outputs (exit code alone is insufficient)")
        for item in job["inputs"]:
            if len(item.get("sha256", "")) != 64:
                raise ValueError(f"{job['id']}: each input needs a sha256")
        retry = job.get("retry", {})
        if retry and (not retry.get("reason") or not isinstance(retry.get("exit_codes"), list)):
            raise ValueError("retry requires transient-failure reason and exit_codes; limit is one")
        if "max_seconds" in job:
            seconds = float(job["max_seconds"])
            grace = float(job.get("shutdown_grace_seconds", 30))
            if not math.isfinite(seconds) or not math.isfinite(grace) or grace <= 0 or seconds <= grace:
                raise ValueError(f"{job['id']}: command time limit must be finite and exceed positive shutdown grace")
        jobs.append(job)
    # Explicitly reject cycles rather than leaving an unattended worker idle forever.
    remaining, cleared = {j["id"]: set(j["deps"]) for j in jobs}, set()
    while remaining:
        ready = {i for i, deps in remaining.items() if deps <= cleared}
        if not ready:
            raise ValueError("dependency cycle")
        cleared |= ready
        remaining = {i: deps for i, deps in remaining.items() if i not in ready}
    manifest["jobs"] = jobs
    # A bounded pilot reserves each GPU command's worst-case elapsed time,
    # including its checkpoint-shutdown grace. No retry can spend it twice.
    if "gpu_budget_seconds" in manifest:
        budget = float(manifest["gpu_budget_seconds"])
        gpu_jobs = [j for j in jobs if j.get("gpu", False)]
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("gpu_budget_seconds must be finite and positive")
        for job in gpu_jobs:
            seconds = float(job.get("max_seconds", 0))
            grace = float(job.get("shutdown_grace_seconds", 30))
            if not math.isfinite(seconds) or not math.isfinite(grace) or seconds <= grace or grace <= 0:
                raise ValueError(f"{job['id']}: bounded GPU job needs max_seconds > shutdown grace > 0")
            if job.get("retry"):
                raise ValueError("Budgeted GPU commands cannot automatically retry")
        if sum(float(j["max_seconds"]) for j in gpu_jobs) > budget:
            raise ValueError("GPU command reservations exceed the declared pilot budget")
    manifest["gpu_lock"] = resolved(manifest.get("gpu_lock", "~/.cache/flooddistill/revision-gpu.lock"), Path(path).parent)
    devices = manifest.get("gpu_devices", [])
    if devices:
        if (len(devices) > 2 or len({d["uuid"] for d in devices}) != len(devices)
                or len({str(d["index"]) for d in devices}) != len(devices)):
            raise ValueError("Declare at most two distinct physical GPUs")
        manifest["gpu_devices"] = [dict(d, lock=resolved(d["lock"], Path(path).parent)) for d in devices]
        if any(not d["uuid"].startswith("GPU-") for d in devices):
            raise ValueError("GPU devices require pinned UUIDs")
        if any(j.get("gpu") and j.get("retry") for j in jobs):
            raise ValueError("Multi-GPU queue uses no automatic retries")
        for name in manifest.get("first_seed_jobs", []) + manifest.get("seed_expansion_jobs", []):
            if name not in ids:
                raise ValueError("Unknown first-seed or expansion job")
    return manifest


def field(value, key):
    """Read nested fields while retaining dots in condition IDs (e.g. g0.90).

    Longest dictionary-key match wins. Explicit segment lists avoid ambiguity
    for future specs; existing plain nested paths and list indices still work.
    """
    if isinstance(key, (list, tuple)):
        for part in key:
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value
    parts = key.split(".")
    while parts:
        if isinstance(value, list):
            value = value[int(parts.pop(0))]
            continue
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in value:
                value = value[candidate]
                parts = parts[end:]
                break
        else:
            raise KeyError(parts[0])
    return value


def validate_outputs(job):
    evidence = []
    try:
        for spec in job["outputs"]:
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size < spec.get("min_bytes", 1):
                raise ValueError(f"missing/empty output: {path}")
            if spec.get("sha256") and file_hash(path) != spec["sha256"]:
                raise ValueError(f"output hash mismatch: {path}")
            if spec.get("type", "file") == "json":
                data = read_json(path)
                for key in spec.get("required_keys", []):
                    field(data, key)
                for key in spec.get("finite", []):
                    number = field(data, key)
                    if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                        raise ValueError(f"invalid finite metric {key}: {path}")
                for key, bounds in spec.get("ranges", {}).items():
                    number = field(data, key)
                    if not math.isfinite(number) or not bounds[0] <= number <= bounds[1]:
                        raise ValueError(f"metric {key} outside {bounds}: {path}")
                for key, expected in spec.get("equals", {}).items():
                    if field(data, key) != expected:
                        raise ValueError(f"unexpected gate/metric {key}: {path}")
            elif spec.get("type", "file") != "file":
                raise ValueError("output type must be file or json")
            evidence.append({"path": str(path), "bytes": path.stat().st_size,
                             "sha256": file_hash(path)})
        return True, evidence
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        return False, str(error)


def check_inputs(job):
    for spec in job["inputs"]:
        try:
            if file_hash(spec["path"]) != spec["sha256"]:
                return f"input changed: {spec['path']}"
        except OSError as error:
            return str(error)
    return None


def gpu_available(manifest, job):
    """Check the selected physical GPU; foreign work on other GPUs is allowed."""
    try:
        target = str(manifest.get("gpu_index", 0))
        memory = subprocess.run(["nvidia-smi", "-i", target,
                                 "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True)
        rows = [line.strip() for line in memory.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("selected GPU must resolve to exactly one device")
        selected_uuid, free_text = [part.strip() for part in rows[0].split(",")]
        if not selected_uuid.startswith("GPU-"):
            raise ValueError("invalid selected GPU UUID")
        if manifest.get("gpu_uuid") and selected_uuid != manifest["gpu_uuid"]:
            raise ValueError("selected GPU identity changed")
        visible = job.get("env", {}).get("CUDA_VISIBLE_DEVICES", "")
        if visible not in {target, selected_uuid}:
            raise ValueError("CUDA_VISIBLE_DEVICES must select the admitted physical GPU only")
        free = int(free_text)
        probe = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        processes = []
        other_devices = []
        for line in probe.stdout.splitlines():
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 3 or not fields[0].startswith("GPU-"):
                raise ValueError("unrecognized GPU compute-process record")
            if fields[0] == selected_uuid:
                processes.append(line.strip())
            else:
                other_devices.append(line.strip())
        if processes:
            return False, "foreign compute on selected GPU " + selected_uuid + ": " + "; ".join(processes)
        required = int(job.get("required_free_mb", 0)) + int(manifest.get("headroom_mb", 2048))
        return free >= required, (f"selected GPU {target} {selected_uuid}: free={free} MiB, "
                                  f"required with headroom={required} MiB; "
                                  f"foreign compute on other GPUs={len(other_devices)}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"GPU probe failed: {error}") from error


def host_available(manifest, job):
    """Admission only; never alter a job's scientific recipe to fit resources."""
    limits = manifest.get("host_limits", {})
    if not limits or not job.get("gpu", False):
        return True, "host admission not required"
    cpus = len(os.sched_getaffinity(0))
    load = os.getloadavg()[0]
    available_kib = next(int(line.split()[1]) for line in
                         Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    free_gib = shutil.disk_usage(limits.get("disk_path", job["cwd"])).free / 2**30
    ram_gib = available_kib / 2**20
    reason = (f"host load1={load:.2f}/{cpus} CPUs; available RAM={ram_gib:.1f} GiB; "
              f"free disk={free_gib:.1f} GiB")
    allowed = (load / cpus <= limits.get("max_load_per_cpu", float("inf"))
               and ram_gib >= limits.get("min_available_ram_gib", 0)
               and free_gib >= limits.get("min_free_disk_gib", 0))
    return allowed, reason


def paths(state_path, job_id, attempt):
    root = Path(state_path).parent
    prefix = f"{job_id}.attempt{attempt}"
    return root / "attempts" / (prefix + ".json"), root / "attempts" / (prefix + ".lock"), root / "logs" / (prefix + ".log")


def initialize(manifest_path, state_path, manifest):
    if Path(state_path).exists():
        state = read_json(state_path)
        if state["manifest_sha256"] != digest(manifest):
            raise ValueError("existing state belongs to different frozen manifest; use a distinct state path")
        return
    fd = lock_fd(str(state_path) + ".lock", blocking=True)
    try:
        if Path(state_path).exists():
            if read_json(state_path)["manifest_sha256"] != digest(manifest):
                raise ValueError("existing state belongs to a different frozen manifest")
            return
        records = {}
        for job in manifest["jobs"]:
            records[job["id"]] = {"status": "blocked" if job.get("blocked_reason") else "pending",
                                  "reason": job.get("blocked_reason"), "block_kind": "manual" if job.get("blocked_reason") else None,
                                  "attempts": [], "identity_sha256": digest(job), "argv": job.get("argv"),
                                  "cwd": job["cwd"], "env_sha256": digest(job["env"]),
                                  "fingerprints": job.get("fingerprints", {}), "inputs": job["inputs"],
                                  "outputs": job["outputs"], "deps": job["deps"]}
        atomic_json(state_path, {"schema_version": 1, "manifest_path": str(manifest_path),
                                "manifest_sha256": digest(manifest), "created_at": now(), "jobs": records,
                                "queue_code_sha256": file_hash(Path(__file__).resolve()),
                                "paused": False, "stop_requested": False, "worker_status": "initialized"})
    finally:
        os.close(fd)


def reconcile(state, jobs, state_path):
    for job in jobs:
        record = state["jobs"][job["id"]]
        if record["status"] not in {"running", "launching"}:
            continue
        attempt = record["attempts"][-1]
        receipt_path, lock_path, _ = paths(state_path, job["id"], attempt["number"])
        # A migrated running command keeps its original immutable runner/receipt.
        receipt_path = Path(attempt.get("receipt_path", receipt_path))
        lock_path = receipt_path.with_suffix(".lock")
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        if receipt and receipt.get("token") != attempt["token"]:
            raise ValueError(f"foreign receipt for {job['id']}; refusing reconciliation")
        if receipt:
            attempt.update(receipt)
            if receipt.get("finished_at"):
                valid, evidence = validate_outputs(job) if receipt["exit_code"] == 0 else (False, "nonzero exit code")
                record["status"] = "succeeded" if valid else "failed"
                record["reason"] = None if valid else receipt.get("error", evidence)
                attempt["output_validation"] = evidence
                retry = job.get("retry", {})
                if not valid and receipt["exit_code"] in retry.get("exit_codes", []) and len(record["attempts"]) == 1:
                    record["status"] = "pending"
                    record["reason"] = "one transient retry: " + retry["reason"]
                continue
            record["status"] = "running"
        fd = lock_fd(lock_path)
        if fd is None:
            record["reason"] = "owned attempt lock remains held; no duplicate launch"
        else:
            os.close(fd)
            if receipt and (alive(receipt.get("runner")) or alive(receipt.get("command"))):
                record["reason"] = "owned PID remains alive; no duplicate launch"
            else:
                record.update(status="failed", reason="owned attempt ended without terminal receipt; no automatic retry")
                attempt.update(finished_at=now(), exit_code=None)


def reconcile_completed(state, jobs):
    """A matching name/state alone cannot authorize reuse after artifact mutation."""
    for job in jobs:
        record = state["jobs"][job["id"]]
        if record["status"] != "succeeded":
            continue
        valid, evidence = validate_outputs(job)
        expected = record.get("evidence") if job["kind"] == "gate" else record["attempts"][-1].get("output_validation")
        error = check_inputs(job)
        if not valid or evidence != expected or error:
            record.update(status="failed", reason=error or "completed output missing, invalid, or changed since validation")


def runner(args, manifest, state_path):
    job = next(j for j in manifest["jobs"] if j["id"] == args.job_id)
    receipt_path, _, log_path = paths(state_path, job["id"], args.attempt)
    receipt = {"token": args.token, "runner": process_identity(os.getpid()), "started_at": now(),
               "job_identity_sha256": digest(job), "argv": job["argv"], "cwd": job["cwd"],
               "env_sha256": digest(job["env"]), "log_path": str(log_path)}
    atomic_json(receipt_path, receipt)
    # These inherited descriptors keep both locks alive if the queue worker exits.
    fds = [fd for fd in (args.job_fd, args.gpu_fd, args.legacy_fd) if fd is not None]
    try:
        state = read_json(state_path)
        if state["manifest_sha256"] != digest(manifest) or state["jobs"][job["id"]]["identity_sha256"] != digest(job):
            raise ValueError("frozen manifest/job identity changed before runner startup")
        if state["queue_code_sha256"] != file_hash(Path(__file__).resolve()):
            raise ValueError("queue code changed before runner startup")
        attempt = state["jobs"][job["id"]]["attempts"][-1]
        if attempt["token"] != args.token:
            raise ValueError("attempt token changed before runner startup")
        runtime_env = dict(job["env"])
        device = attempt.get("gpu_device")
        if device:
            if device not in manifest.get("gpu_devices", []):
                raise ValueError("Attempt GPU is outside the frozen device inventory")
            runtime_env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
            receipt["gpu_device"] = device
        receipt["env_sha256"] = digest(runtime_env)
        atomic_json(receipt_path, receipt)
        error = check_inputs(job)
        if error:
            raise ValueError(error)
        os.nice(int(manifest.get("nice", 10)))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab", buffering=0) as log:
            command = subprocess.Popen(job["argv"], cwd=job["cwd"], env=runtime_env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=tuple(fds), start_new_session=True)
            receipt["command"] = process_identity(command.pid)
            atomic_json(receipt_path, receipt)
            limit = job.get("max_seconds")
            if limit is None:
                receipt["exit_code"] = command.wait()
            else:
                grace = float(job.get("shutdown_grace_seconds", 30))
                receipt["runtime_limit_seconds"] = float(limit)
                began = time.monotonic()
                try:
                    receipt["exit_code"] = command.wait(timeout=max(0.01, float(limit) - grace))
                except subprocess.TimeoutExpired:
                    receipt["budget_exhausted"] = True
                    receipt["error"] = "enforced command time reservation exhausted; descendants cannot treat partial training as complete"
                    # This session contains only this owned command and its
                    # descendants, never a foreign GPU process.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(command.pid, signal.SIGTERM)
                    try:
                        command.wait(timeout=max(0.01, float(limit) - (time.monotonic() - began)))
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(command.pid, signal.SIGKILL)
                        command.wait()
                    # A parent may exit on TERM while a child still owns GPU
                    # resources. Reap that owned group even in that case.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(command.pid, signal.SIGKILL)
                    receipt["exit_code"] = 124
                receipt["command_elapsed_seconds"] = time.monotonic() - began
        valid, evidence = validate_outputs(job) if receipt["exit_code"] == 0 else (False, "nonzero exit code")
        receipt["output_validation"] = evidence
        if not valid:
            receipt.setdefault("error", evidence)
    except Exception as error:
        receipt.update(exit_code=126, error=f"{type(error).__name__}: {error}")
    receipt["finished_at"] = now()
    atomic_json(receipt_path, receipt)


def launch(job, manifest, manifest_path, state_path, occupied=(), legacy_fd=None):
    allowed, host_reason = host_available(manifest, job)
    if not allowed:
        return "waiting for resources: " + host_reason
    gpu_fd = None
    device = None
    if job.get("gpu", False):
        reasons = []
        devices = manifest.get("gpu_devices") or [None]
        for candidate in devices:
            if candidate and candidate["uuid"] in occupied:
                reasons.append(f"owned job occupies {candidate['uuid']}")
                continue
            gpu_fd = lock_fd(candidate["lock"] if candidate else manifest["gpu_lock"])
            if gpu_fd is None:
                reasons.append("shared GPU lock busy" if candidate is None else f"device lock busy: {candidate['uuid']}")
                continue
            probe_manifest = {**manifest, "gpu_index": candidate["index"], "gpu_uuid": candidate["uuid"]} if candidate else manifest
            probe_job = {**job, "env": {**job["env"], "CUDA_VISIBLE_DEVICES": candidate["uuid"]}} if candidate else job
            try:
                available, reason = gpu_available(probe_manifest, probe_job)
            except Exception:
                os.close(gpu_fd)
                raise
            if available:
                device = candidate
                break
            os.close(gpu_fd)
            gpu_fd = None
            reasons.append(reason)
        if gpu_fd is None:
            return "; ".join(reasons)
    job_fd = None
    try:
        with edit_state(state_path) as state:
            if state["paused"] or state["stop_requested"]:
                return "control changed before launch"
            if state["queue_code_sha256"] != file_hash(Path(__file__).resolve()):
                raise ValueError("queue runner code changed; freeze a new queue instead of mutating active inputs")
            record = state["jobs"][job["id"]]
            attempt = len(record["attempts"]) + 1
            receipt, job_lock, _ = paths(state_path, job["id"], attempt)
            job_fd = lock_fd(job_lock)
            if job_fd is None:
                return "attempt lock busy"
            token = uuid.uuid4().hex
            record["attempts"].append({"number": attempt, "token": token, "created_at": now(), "receipt_path": str(receipt)})
            if device:
                record["attempts"][-1]["gpu_device"] = device
                record["attempts"][-1]["legacy_guard_retained"] = legacy_fd is not None
            record.update(status="launching", reason=None)
        command = [sys.executable, str(Path(__file__).resolve()), "_job", "--manifest", str(manifest_path),
                   "--state", str(state_path), "--job-id", job["id"], "--attempt", str(attempt),
                   "--token", token, "--job-fd", str(job_fd)]
        if gpu_fd is not None:
            command += ["--gpu-fd", str(gpu_fd)]
        if legacy_fd is not None and gpu_fd is not None:
            command += ["--legacy-fd", str(legacy_fd)]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True,
                                   pass_fds=tuple(fd for fd in (job_fd, gpu_fd, legacy_fd if gpu_fd is not None else None) if fd is not None))
        with edit_state(state_path) as state:
            state["jobs"][job["id"]]["attempts"][-1]["runner"] = process_identity(process.pid)
    finally:
        # Closing, rather than LOCK_UN, preserves the shared inherited flock.
        for fd in (job_fd, gpu_fd):
            if fd is not None:
                os.close(fd)
    return None


def worker(manifest_path, state_path, manifest):
    fd = lock_fd(str(state_path) + ".worker.lock")
    if fd is None:
        raise ValueError("a worker already holds this queue lock")
    stop_signal = [False]
    parallel = bool(manifest.get("gpu_devices"))
    legacy_fd = None
    gpu_ids = {j["id"] for j in manifest["jobs"] if j.get("gpu", False)}
    signal.signal(signal.SIGTERM, lambda *_: stop_signal.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop_signal.__setitem__(0, True))
    try:
        with edit_state(state_path) as state:
            reconcile_completed(state, manifest["jobs"])
        while True:
            candidates = []
            if parallel and legacy_fd is None:
                # Exclude old single-GPU wrappers after the adopted run releases
                # its global lock. New commands retain this lock across crashes.
                legacy_fd = lock_fd(manifest["gpu_lock"])
            with edit_state(state_path) as state:
                state.update(worker=process_identity(os.getpid()), worker_status="running", heartbeat_at=now())
                if stop_signal[0]:
                    state["stop_requested"] = True
                reconcile(state, manifest["jobs"], state_path)
                active_records = {i: r for i, r in state["jobs"].items() if r["status"] in {"launching", "running"}}
                active = bool(active_records)
                occupied = {r["attempts"][-1]["gpu_device"]["uuid"] for i, r in active_records.items()
                            if i in gpu_ids and r["attempts"][-1].get("gpu_device")}
                gpu_active = sum(i in gpu_ids for i in active_records)
                cpu_active = any(i not in gpu_ids for i in active_records)
                owned_guard_active = any(
                    (r["attempts"][-1].get("legacy_adopted") or r["attempts"][-1].get("legacy_guard_retained"))
                    and (alive(r["attempts"][-1].get("command")) or alive(r["attempts"][-1].get("runner")))
                    for r in active_records.values())
                if state["stop_requested"] and not active:
                    state.update(worker_status="stopped", stopped_at=now())
                    return
                if (parallel or not active) and not state["paused"] and not state["stop_requested"]:
                    for job in manifest["jobs"]:
                        record = state["jobs"][job["id"]]
                        if record["status"] != "pending":
                            continue
                        dependency_states = [state["jobs"][d]["status"] for d in job["deps"]]
                        after_terminal = job.get("run_after_terminal_deps", False)
                        if not after_terminal and any(s in {"failed", "blocked"} for s in dependency_states):
                            record.update(status="blocked", block_kind="dependency", reason="failed/blocked dependencies: " + ", ".join(d for d in job["deps"] if state["jobs"][d]["status"] in {"failed", "blocked"}))
                            continue
                        if any(s not in (TERMINAL if after_terminal else {"succeeded"}) for s in dependency_states):
                            continue
                        if job["id"] in manifest.get("seed_expansion_jobs", []) and any(
                                state["jobs"][i]["status"] not in TERMINAL for i in manifest.get("first_seed_jobs", [])):
                            record["reason"] = "waiting for all first-seed configurations to finish"
                            continue
                        if job["kind"] == "gate":
                            error = check_inputs(job)
                            valid, evidence = (False, error) if error else validate_outputs(job)
                            record.update(status="succeeded" if valid else "blocked", block_kind=None if valid else "gate_evidence",
                                          reason=None if valid else str(evidence), evidence=evidence, checked_at=now())
                        else:
                            # The runner verifies hashes once, after admission.
                            # Do not reread GB-scale weights/data on idle polls.
                            if parallel and job.get("gpu", False):
                                if cpu_active or gpu_active >= len(manifest["gpu_devices"]):
                                    continue
                                if legacy_fd is None and not owned_guard_active:
                                    record["reason"] = "legacy shared GPU lock busy"
                                    continue
                            elif active:
                                # Keep substantial CPU/disk jobs serialized.
                                continue
                            candidates.append(job)
                    if all(r["status"] in TERMINAL for r in state["jobs"].values()):
                        state.update(worker_status="finished", stopped_at=now())
                        return
            gpu_waited = False
            for candidate in candidates:
                if gpu_waited and candidate.get("gpu", False):
                    continue
                try:
                    if digest(load_manifest(manifest_path)) != digest(manifest):
                        raise ValueError("frozen manifest changed while worker was running")
                    reason = launch(candidate, manifest, manifest_path, state_path, occupied, legacy_fd)
                    if reason:
                        with edit_state(state_path) as state:
                            state["jobs"][candidate["id"]]["reason"] = reason
                        gpu_waited = bool(candidate.get("gpu", False))
                    else:
                        break
                except Exception as error:
                    with edit_state(state_path) as state:
                        record = state["jobs"][candidate["id"]]
                        # An uncertain launch must reconcile its attempt before retrying.
                        if record["status"] == "pending":
                            record.update(status="blocked", block_kind="launch", reason=f"{type(error).__name__}: {error}")
            time.sleep(max(0.1, min(float(manifest.get("poll_seconds", 5)), 60)))
    finally:
        if legacy_fd is not None:
            os.close(legacy_fd)
        os.close(fd)


def start(manifest_path, state_path):
    state = read_json(state_path)
    if alive(state.get("worker")):
        return {"worker": state["worker"], "already_running": True}
    # The worker lock closes the race between simultaneous starts.
    log = Path(state_path).parent / "worker.log"
    with open(log, "ab", buffering=0) as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", "--manifest", str(manifest_path), "--state", str(state_path)],
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=str(Path(manifest_path).parent), start_new_session=True, close_fds=True)
    return {"worker": process_identity(process.pid), "worker_log": str(log), "detached_session": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "start", "run", "status", "pause", "resume", "stop", "_job"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--token")
    parser.add_argument("--job-fd", type=int)
    parser.add_argument("--gpu-fd", type=int)
    parser.add_argument("--legacy-fd", type=int)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this thin wrapper supports the inspected Linux host only")
    manifest_path = Path(args.manifest).resolve()
    state_path = Path(args.state).resolve() if args.state else manifest_path.with_name(manifest_path.stem + ".state.json")
    manifest = load_manifest(manifest_path)
    if args.action == "_job":
        runner(args, manifest, state_path)
        return
    initialize(manifest_path, state_path, manifest)
    if args.action == "run":
        worker(manifest_path, state_path, manifest)
    elif args.action == "start":
        print(json.dumps(start(manifest_path, state_path), sort_keys=True))
    elif args.action in {"pause", "resume", "stop"}:
        with edit_state(state_path) as state:
            if args.action == "pause":
                state["paused"] = True
            elif args.action == "stop":
                state["stop_requested"] = True
            else:
                state.update(paused=False, stop_requested=False)
                for record in state["jobs"].values():
                    if record.get("block_kind") in {"gate_evidence", "dependency"}:
                        record.update(status="pending", reason=None, block_kind=None)
            state["last_control"] = {"action": args.action, "at": now()}
        print(json.dumps(start(manifest_path, state_path) if args.action == "resume" else {"control": args.action, "state": str(state_path)}, sort_keys=True))
    elif args.action == "status":
        state = read_json(state_path)
        counts = {status: sum(j["status"] == status for j in state["jobs"].values()) for status in ("pending", "launching", "running", "succeeded", "failed", "blocked")}
        print(json.dumps({"state": str(state_path), "counts": counts, "paused": state["paused"], "stop_requested": state["stop_requested"],
                          "worker_status": state["worker_status"], "worker_alive": alive(state.get("worker")),
                          "jobs": {i: {"status": j["status"], "reason": j.get("reason"), "attempts": len(j["attempts"])} for i, j in state["jobs"].items()}}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
