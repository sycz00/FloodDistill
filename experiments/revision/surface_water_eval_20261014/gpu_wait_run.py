"""Wait for an exclusively idle GPU under the shared revision rule, then run the surface-water evaluation.

Rule (same as V9/V10 scheduler_v9.availability + audited_queue.lock_fd): a UUID qualifies after >= 3 idle
observations (no compute process, utilization < 5 %, >= 16 GiB free) spanning >= 29 s; the job then takes the
non-blocking flock /data1/schmalstieg/flooddistill/revision_gpu_<uuid>.lock and holds it until the evaluation
exits. Running jobs are never touched. Create STOP in this folder to abort while waiting.
"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/home/schmalstieg/miniconda3/envs/esi/bin/python"
UUIDS = ["GPU-5a747234-86cd-0af0-db56-9a5e5b3c81fc", "GPU-eadba879-32d7-d9cf-1214-f9e16ec4c5cb"]
STATE = HERE / "state.json"


def save(**kw):
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    s.update(kw, updated=time.time())
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); os.replace(tmp, STATE)


def telemetry():
    q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True).stdout
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=15, check=True).stdout
    rows = {}
    for line in q.splitlines():
        u, util, free = [x.strip() for x in line.split(",")]
        rows[u] = dict(util=float(util), free=float(free), procs=0)
    for line in p.splitlines():
        if line.strip():
            rows[line.split(",")[0].strip()]["procs"] += 1
    return rows


def main():
    save(phase="waiting", pid=os.getpid(), started=time.time(), rule="3 idle observations >= 29 s + UUID flock")
    hist = {u: [] for u in UUIDS}
    deadline = time.time() + 24 * 3600
    while time.time() < deadline:
        if (HERE / "STOP").exists():
            save(phase="stopped_while_waiting"); return
        try:
            t = telemetry()
        except Exception as exc:
            save(last_error=str(exc)); time.sleep(20); continue
        now = time.time()
        for u in UUIDS:
            r = t.get(u)
            ok = r is not None and r["procs"] == 0 and r["util"] < 5 and r["free"] >= 16384
            hist[u] = (hist[u] + [now]) if ok else []          # continuous idle run since the first idle observation
            if len(hist[u]) >= 3 and now - hist[u][0] >= 29:
                fd = os.open(f"/data1/schmalstieg/flooddistill/revision_gpu_{u}.lock", os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd); hist[u] = []; continue
                if telemetry().get(u, {}).get("procs", 1) != 0:
                    fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd); hist[u] = []; continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=u, OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
                log = open(HERE / "logs" / "eval.out", "a")
                child = subprocess.Popen(["/usr/bin/ionice", "-c", "2", "-n", "7", PY, str(HERE / "eval_surface_water.py"),
                                          "--device", "cuda", "--workers", "4"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=str(HERE))
                save(phase="running", gpu_uuid=u, child_pid=child.pid, admitted=time.time())
                rc = child.wait()
                fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
                save(phase="finished" if rc == 0 else "failed", returncode=rc, finished=time.time())
                return
        save(phase="waiting", last_check=now, idle_history={u: len(h) for u, h in hist.items()})
        time.sleep(10)
    save(phase="gave_up_after_24h")


if __name__ == "__main__":
    main()
