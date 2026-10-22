#!/usr/bin/env python3
"""Supplementary steady-state power for one engine (added after the paper-method run).

Why: benchmark_jetson.measure_power (and bench_trtexec.py run B, which copies it) averages
the INA3221 over the whole trtexec process. With --iterations=200 the inference phase lasts
~3 s, while process start-up (CUDA init, engine load) lasts 2-5 s at near-idle power and
differs between engines, so that average is diluted by different amounts per precision.

Here: trtexec --duration=30 --warmUp=2000; samples every 100 ms; only samples taken between
(time trtexec prints "Starting inference" + 3 s) and (process exit - 1 s) are averaged. The
"Starting inference" moment is taken from the first stdout line containing it, read live.
Energy per image = mean steady power / trtexec throughput of the same run.
"""
import argparse, json, os, re, subprocess, threading, time
import numpy as np

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HW = "/sys/class/hwmon/hwmon4"


def read_ch(ch):
    return int(open(f"{HW}/in{ch}_input").read()) * int(open(f"{HW}/curr{ch}_input").read()) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=int, default=30)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=False)
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append((time.time(), read_ch(2), read_ch(1))); stop.wait(0.1)

    th = threading.Thread(target=poll); th.start(); time.sleep(3)
    idle_end = time.time()
    cmd = [TRTEXEC, f"--loadEngine={a.engine}", "--batch=1", f"--duration={a.duration}", "--warmUp=2000",
           "--iterations=200"]
    t_start_inf = None; lines = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in p.stdout:
        lines.append(line)
        if t_start_inf is None and "Starting inference" in line:
            t_start_inf = time.time()
    p.wait(); t_exit = time.time(); time.sleep(1); stop.set(); th.join()
    open(f"{a.out}/trtexec.log", "w").write("".join(lines))
    txt = "".join(lines)
    m = re.search(r"Throughput:\s*([\d.]+)\s*qps", txt); qps = float(m.group(1)) if m else None
    idle = [s for s in samples if s[0] < idle_end]
    lo, hi = (t_start_inf or t_exit) + 3.0, t_exit - 1.0
    steady = [s for s in samples if lo <= s[0] <= hi]
    rec = dict(engine=os.path.abspath(a.engine), cmd=" ".join(cmd), returncode=p.returncode, throughput_qps=qps,
               window_s=[lo - (t_start_inf or 0), hi - (t_start_inf or 0)], steady_samples=len(steady),
               idle_cpu_gpu_cv_w=float(np.mean([s[1] for s in idle]) / 1000) if idle else None,
               power_rail="ina3221 ch2 (VDD_CPU_GPU_CV per benchmark_jetson.py fallback; label root-only)",
               samples_raw=[list(s) for s in samples], t_start_inference=t_start_inf, t_exit=t_exit)
    if steady:
        w = np.array([s[1] for s in steady]) / 1000; vin = np.array([s[2] for s in steady]) / 1000
        rec.update(steady_cpu_gpu_cv_mean_w=float(w.mean()), steady_cpu_gpu_cv_sd_w=float(w.std()),
                   steady_vdd_in_ch1_mean_w=float(vin.mean()))
        if qps:
            rec["steady_energy_mJ_per_image"] = 1000 * float(w.mean()) / qps
            if rec["idle_cpu_gpu_cv_w"] is not None:
                rec["steady_energy_mJ_per_image_above_idle"] = 1000 * (float(w.mean()) - rec["idle_cpu_gpu_cv_w"]) / qps
    json.dump(rec, open(f"{a.out}/power_steady.json", "w"), indent=1)
    print({k: v for k, v in rec.items() if k != "samples_raw"})


if __name__ == "__main__":
    main()
