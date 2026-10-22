#!/usr/bin/env python3
"""Step 4 benchmark, same method as jetson_repro/jetson/benchmark_jetson.py (paper section 5.3).

Per engine:
  run A  trtexec --loadEngine --batch=1 --iterations=200 --warmUp=20 --percentile=99
         --separateProfileRun --dumpProfile --exportProfile --exportTimes --exportLayerInfo
         (timing percentiles come from the exported per-iteration times)
  run B  the same trtexec timing command as benchmark_jetson.measure_power (no profiling),
         while the INA3221 is polled every 100 ms; average power and energy per image.
Power channel: ch2 of the ina3221 hwmon, the VDD_CPU_GPU_CV fallback of benchmark_jetson.py
(the channel labels are root-only on this board, so the label could not be read back).
ch1 (board input) is recorded alongside. An idle reading (5 s, no load) precedes each engine.
"""
import argparse, json, os, re, subprocess, threading, time, hashlib
import numpy as np

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HW = "/sys/class/hwmon/hwmon4"


def read_ch(ch):
    v = int(open(f"{HW}/in{ch}_input").read()); i = int(open(f"{HW}/curr{ch}_input").read())
    return v * i / 1000.0  # mW


def sample_power(stop, out, interval=0.1):
    while not stop.is_set():
        t = time.time()
        try:
            out.append((t, read_ch(2), read_ch(1)))
        except OSError as e:
            out.append((t, None, None)); out.append(repr(e)); return
        stop.wait(interval)


def pct(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                min=float(v.min()), max=float(v.max()))


def power_stats(samples):
    s = [x for x in samples if isinstance(x, tuple) and x[1] is not None]
    if not s:
        return None
    gpu = np.array([x[1] for x in s]) / 1000.0; vin = np.array([x[2] for x in s]) / 1000.0
    return dict(samples=len(s), cpu_gpu_cv_mean_w=float(gpu.mean()), cpu_gpu_cv_max_w=float(gpu.max()),
                cpu_gpu_cv_min_w=float(gpu.min()), vdd_in_ch1_mean_w=float(vin.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True, help="new folder for this engine")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20, help="ms, as trtexec --warmUp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    common = [TRTEXEC, f"--loadEngine={a.engine}", "--batch=1", f"--iterations={a.iterations}",
              f"--warmUp={a.warmup}"]
    runA = common + ["--percentile=99", "--separateProfileRun", "--dumpProfile",
                     f"--exportProfile={a.out}/profile.json", f"--exportTimes={a.out}/times.json",
                     f"--exportLayerInfo={a.out}/layer_info.json", "--profilingVerbosity=detailed"]
    rec = dict(engine=os.path.abspath(a.engine), engine_bytes=os.path.getsize(a.engine),
               engine_sha256=hashlib.sha256(open(a.engine, "rb").read()).hexdigest(),
               runA_cmd=" ".join(runA), runB_cmd=" ".join(common), started=time.time())
    # idle power
    idle = []; st = threading.Event(); th = threading.Thread(target=sample_power, args=(st, idle)); th.start()
    time.sleep(5); st.set(); th.join()
    rec["idle_power"] = power_stats(idle)
    r = subprocess.run(runA, capture_output=True, text=True, errors="replace")
    open(f"{a.out}/runA.log", "w").write(r.stdout + r.stderr)
    rec["runA_returncode"] = r.returncode
    if r.returncode == 0:
        try:
            t = json.load(open(f"{a.out}/times.json"))
            rec["times_keys"] = sorted(t[0].keys()) if t else []
            for out_key, key in (("gpu_compute_ms", "computeMs"), ("host_latency_ms", "latencyMs"),
                                 ("h2d_ms", "h2dMs"), ("d2h_ms", "d2hMs"), ("end_to_end_ms", "endToEndMs")):
                if t and key in t[0]:
                    rec[out_key] = pct([x[key] for x in t])
        except Exception as e:
            rec["times_error"] = repr(e)
        m = re.search(r"Throughput:\s*([\d.]+)\s*qps", r.stdout); rec["throughput_qps_runA"] = float(m.group(1)) if m else None
        m = re.search(r"Total Activation Memory:\s*(\d+)", r.stdout + r.stderr)
        rec["trtexec_activation_memory_bytes"] = int(m.group(1)) if m else None
    # power run
    samples = []; st = threading.Event(); th = threading.Thread(target=sample_power, args=(st, samples)); th.start()
    time.sleep(0.5)
    t0 = time.time(); r = subprocess.run(common, capture_output=True, text=True, errors="replace"); t1 = time.time()
    st.set(); th.join()
    open(f"{a.out}/runB_power.log", "w").write(r.stdout + r.stderr)
    rec["runB_returncode"] = r.returncode
    inrun = [x for x in samples if isinstance(x, tuple) and t0 <= x[0] <= t1]
    rec["power"] = power_stats(inrun)
    rec["power_samples_raw"] = [list(x) if isinstance(x, tuple) else x for x in samples]
    m = re.search(r"Throughput:\s*([\d.]+)\s*qps", r.stdout); qps = float(m.group(1)) if m else None
    rec["throughput_qps_runB"] = qps
    if rec["power"] and qps:
        rec["energy_mJ_per_image_cpu_gpu_cv"] = 1000.0 * rec["power"]["cpu_gpu_cv_mean_w"] / qps
        if rec["idle_power"]:
            rec["energy_mJ_per_image_above_idle"] = 1000.0 * (rec["power"]["cpu_gpu_cv_mean_w"] - rec["idle_power"]["cpu_gpu_cv_mean_w"]) / qps
    rec["power_rail"] = "ina3221 ch2 (VDD_CPU_GPU_CV per benchmark_jetson.py fallback; label root-only)"
    rec["finished"] = time.time()
    json.dump(rec, open(f"{a.out}/bench.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "power_samples_raw"}, indent=1))


if __name__ == "__main__":
    main()
