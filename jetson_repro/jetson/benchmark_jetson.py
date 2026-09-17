#!/usr/bin/env python3
"""FP32 / FP16 / INT8-QAT TensorRT engine benchmark on the Jetson Xavier.

Builds FP32 and FP16 engines from the float ONNX with trtexec, then benchmarks those two
plus the INT8-QAT engine.

The INT8 engine is not built by this script, do that by hand with trtexec first:

    Step 0: fix the ONNX for TRT 8.4 (cubic Resize, INT64 chain, freeze shapes)
        python jetson/fix_qdq_onnx.py runs/quant/<run>/model_qdq_trt.onnx

    Step 1: build the INT8 QAT engine (from model_qdq_trt.onnx or model_qdq_trt_static.onnx)
        /usr/src/tensorrt/bin/trtexec \
            --onnx=runs/quant/<run>/model_qdq_trt.onnx \
            --int8 --fp16 \
            --saveEngine=runs/quant/<run>/model_int8_qat.engine \
            --workspace=2048

    Step 2: run this script, it compiles FP32/FP16 and benchmarks all three
        python jetson/benchmark_jetson.py --model_dir runs/quant/<run>/

Usage:
    python3 benchmark_jetson.py \
        --model_dir ~/models/qat_ofa_div_s42 \
        --img_size 512 \
        --iterations 500 \
        --measure_power

model_dir should contain:
    model_float_trt_base.onnx  compiled to model_fp32.engine and model_fp16.engine
    model_int8_qat.engine      pre-built with trtexec (Step 1)

FP16 accumulation: --fp16_mm2_fp32 pins the LiteMLA mm2 normalization (mm2, Slice, Add,
Div) to FP32 in the FP16 engine, because the un-normalized attention numerator overflows
FP16. This uses trtexec builder flags on the float ONNX, not a pre-converted fp16 ONNX,
which TRT 8.4 miscompiles into an attention collapse. FP32/INT8 are not affected.

trtexec: /usr/src/tensorrt/bin/trtexec  (JetPack 5.0 / TRT 8.4.x)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _run_streaming(cmd: list[str]) -> tuple[int, str]:
    """Run cmd, stream combined stdout/stderr to the terminal, return (rc, output)."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)

TRTEXEC_DEFAULT = "/usr/src/tensorrt/bin/trtexec"

# LiteMLA linear-attention MatMuls (mm1 = vk = v*k^T, mm2 = out = vk*q). This is where FP16
# overflows: mm2's un-normalized numerator reaches ~6.5e4 (the FP16 limit), turns into inf
# and then nan after the normalizer Div, and the flood output collapses. Both matmuls are
# pinned to FP32 with FP32 output (--layerPrecisions + --layerOutputTypes) so the large
# numerator is computed and stored in FP32.

# NB: the normalizer Add+Div is fused into one TRT POINTWISE layer whose name contains
# commas, which trtexec --layerPrecisions can't parse, so that one can't be pinned this way.
# The 8 MATRIX_MULTIPLY layers keep their ONNX node names in the built engine (confirmed in
# jetson/s42.log), so the names match. They are fixed by the EfficientViT-B0 architecture
# (context blocks in stages 2 and 3). To check, grep the trtexec log for "could not find
# layer"; --fp16_mm2_layers overrides the list.
DEFAULT_MM2_LAYERS = [
    f"/model/backbone/stages.{s}/op_list.{o}/context_module/main/{node}"
    for s in (2, 3) for o in (1, 2)
    for node in ("MatMul", "MatMul_1")
]


def _fp16_mm2_layer_names(model_dir, cli_layers: str | None) -> list[str]:
    """Resolve the mm2 layer names to pin to FP32.

        Priority: explicit --fp16_mm2_layers (comma-separated) > a sidecar
        `fp16_mm2_layers.txt` in model_dir (one name per line, written at export time)
        > the architecture-deterministic DEFAULT_MM2_LAYERS. No onnx dependency needed
        on the device.
    """
    if cli_layers:
        return [s.strip() for s in cli_layers.split(",") if s.strip()]
    sidecar = model_dir / "fp16_mm2_layers.txt"
    if sidecar.exists():
        names = [ln.strip() for ln in sidecar.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        if names:
            return names
    return list(DEFAULT_MM2_LAYERS)



# Engine compilation


def compile_engine(
    trtexec: str,
    onnx_path: str,
    engine_path: str,
    fp16: bool,
    workspace_mb: int,
    rebuild: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    if os.path.exists(engine_path) and not rebuild:
        print(f"  [skip] engine already exists: {engine_path}")
        return

    # No --minShapes/--optShapes/--maxShapes: both model_float_static.onnx and
    # model_float_trt_base.onnx have fully static shapes - shape profiles are for
    # dynamic-batch ONNXes only and will cause TRT 8.4 to reject static graphs.
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")
    if extra_args:
        cmd.extend(extra_args)

    print(f"  $ {' '.join(cmd)}")
    rc, output = _run_streaming(cmd)
    if rc != 0:
        print(f"\n  ERROR: trtexec exited with code {rc}", file=sys.stderr)
        for line in output.splitlines()[-40:]:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    print(f"  engine saved: {engine_path}")



# Latency / FPS benchmark


def _parse_trtexec_output(text: str) -> dict:
    results: dict = {}

    m = re.search(r'mean\s*=\s*([\d.]+)\s*ms', text)
    if m:
        results["lat_ms_mean"] = float(m.group(1))

    m = re.search(r'percentile\(99%\)\s*=\s*([\d.]+)\s*ms', text)
    if m:
        results["lat_ms_p99"] = float(m.group(1))

    m = re.search(r'min\s*=\s*([\d.]+)\s*ms', text)
    if m:
        results["lat_ms_min"] = float(m.group(1))

    m = re.search(r'max\s*=\s*([\d.]+)\s*ms', text)
    if m:
        results["lat_ms_max"] = float(m.group(1))

    m = re.search(r'Throughput:\s*([\d.]+)\s*qps', text)
    if m:
        results["fps"] = float(m.group(1))

    return results


def benchmark_engine(
    trtexec: str,
    engine_path: str,
    iterations: int,
    warmup: int,
    batch_size: int,
    profile_path: str | None = None,
) -> dict:
    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--batch={batch_size}",
        f"--iterations={iterations}",
        f"--warmUp={warmup}",
        "--percentile=99",
        "--separateProfileRun",
    ]
    if profile_path:
        cmd.append(f"--exportProfile={profile_path}")

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    combined = result.stdout + result.stderr

    stats = _parse_trtexec_output(combined)

    if "lat_ms_mean" not in stats:
        print("  WARNING: could not parse latency from trtexec output — showing last 30 lines:")
        for line in combined.splitlines()[-30:]:
            print(f"    {line}")

    stats["engine_size_mb"] = (
        round(os.path.getsize(engine_path) / (1 << 20), 2)
        if os.path.exists(engine_path) else None
    )
    return stats



# Power measurement (Jetson only, via tegrastats)


def _read_ina3221_power_mw() -> float | None:
    """Read VDD_CPU_GPU_CV (ch2) from sysfs INA3221 hwmon.

        Returns mW or None.  Requires read access to /sys/class/hwmon/hwmon*/;
        on Jetson Xavier NX these are root-only by default.  Grant access with:
            sudo chmod a+r /sys/class/hwmon/hwmon4/in[0-9]_input
            sudo chmod a+r /sys/class/hwmon/hwmon4/curr[0-9]_input
        Or add a udev rule:
            echo 'SUBSYSTEM=="hwmon", ATTR{name}=="ina3221", MODE="0444"' \
                | sudo tee /etc/udev/rules.d/99-ina3221.rules && sudo udevadm trigger
    """
    import glob
    for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name_file = os.path.join(hwmon, "name")
        try:
            with open(name_file) as f:
                if "ina3221" not in f.read().lower():
                    continue
        except OSError:
            continue
        # look for VDD_CPU_GPU_CV by label, then fall back to ch2 (Xavier NX default)
        target_ch = "2"
        for ch in ("1", "2", "3"):
            lf = os.path.join(hwmon, f"in{ch}_label")
            try:
                with open(lf) as f:
                    if "CPU_GPU" in f.read().upper():
                        target_ch = ch
                        break
            except OSError:
                pass
        v_file = os.path.join(hwmon, f"in{target_ch}_input")   # mV
        i_file = os.path.join(hwmon, f"curr{target_ch}_input") # mA
        try:
            v_mv = int(open(v_file).read())
            i_ma = int(open(i_file).read())
            return v_mv * i_ma / 1000.0  # mW
        except (OSError, ValueError):
            return None  # INA3221 found but not readable (needs root)
    return None


def measure_power(
    trtexec: str,
    engine_path: str,
    batch_size: int,
    iterations: int,
    warmup: int,
    interval_ms: int = 100,
) -> dict:
    """GPU power draw during a trtexec run.

        Tries tegrastats --logfile first (works on AGX Xavier, mW fields in the log), then
        polls the sysfs INA3221 (Xavier NX, if hwmon4 is readable). If neither works it
        prints how to grant access.
    """
    log_path = str(Path(engine_path).with_suffix("")) + "_power.txt"

    tstat_proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms), "--logfile", log_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    # sysfs polling thread (Xavier NX fallback)
    import threading
    sysfs_samples: list[float] = []
    sysfs_stop = threading.Event()

    def _poll_sysfs():
        while not sysfs_stop.is_set():
            v = _read_ina3221_power_mw()
            if v is not None:
                sysfs_samples.append(v)
            sysfs_stop.wait(interval_ms / 1000.0)

    sysfs_thread = threading.Thread(target=_poll_sysfs, daemon=True)
    sysfs_thread.start()

    subprocess.run(
        [trtexec, f"--loadEngine={engine_path}",
         f"--batch={batch_size}", f"--iterations={iterations}",
         f"--warmUp={warmup}"],
        capture_output=True,
    )

    sysfs_stop.set()
    tstat_proc.terminate()
    tstat_proc.wait()
    sysfs_thread.join(timeout=2)

    # 1. Try tegrastats log
    gpu_watts: list[float] = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                # JetPack 5.x (L4T 35.x): VDD_CPU_GPU_CV covers CPU+GPU+CV.
                m = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
                if not m:
                    # JetPack 4.x (L4T 32.x): GPU has its own power domain
                    m = re.search(r'(?:POM_5V_GPU|GPU)\s+(\d+)mW', line)
                if m:
                    gpu_watts.append(int(m.group(1)) / 1000.0)

    if gpu_watts:
        return {
            "gpu_power_mean_w":  round(sum(gpu_watts) / len(gpu_watts), 3),
            "gpu_power_max_w":   round(max(gpu_watts), 3),
            "gpu_power_min_w":   round(min(gpu_watts), 3),
            "gpu_power_samples": len(gpu_watts),
            "gpu_power_source":  "tegrastats",
        }

    # 2. Try sysfs INA3221
    if sysfs_samples:
        watts = [s / 1000.0 for s in sysfs_samples]
        return {
            "gpu_power_mean_w":  round(sum(watts) / len(watts), 3),
            "gpu_power_max_w":   round(max(watts), 3),
            "gpu_power_min_w":   round(min(watts), 3),
            "gpu_power_samples": len(watts),
            "gpu_power_source":  "sysfs-ina3221-VDD_CPU_GPU_CV",
        }

    # 3. Neither source worked
    sample = ""
    if os.path.exists(log_path):
        with open(log_path) as f:
            sample = f.readline().strip()
    print("  WARNING: no power readings available on this board.")
    print("  tegrastats did not emit mW fields (Xavier NX JP5 expected).")
    print("  To enable sysfs INA3221 power reads without root, run once:")
    print("    sudo chmod a+r /sys/class/hwmon/hwmon4/in[0-9]_input")
    print("    sudo chmod a+r /sys/class/hwmon/hwmon4/curr[0-9]_input")
    if sample:
        print(f"  tegrastats sample: {sample[:120]}")
    return {"gpu_power_mean_w": None, "gpu_power_max_w": None}



# Report formatting


def _fmt(v, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def print_comparison(fp32: dict, fp16: dict, int8: dict) -> None:
    def _speedup(baseline, other):
        if baseline.get("lat_ms_mean") and other.get("lat_ms_mean"):
            return baseline["lat_ms_mean"] / other["lat_ms_mean"]
        return None

    sp_fp16 = _speedup(fp32, fp16)
    sp_int8 = _speedup(fp32, int8)

    W = 86
    print("\n" + "=" * W)
    print(f"  {'Metric':<28}  {'FP32':>14}  {'FP16':>14}  {'INT8-QAT':>14}")
    print("-" * W)

    rows = [
        ("Latency mean  (ms)",     fp32.get("lat_ms_mean"),  fp16.get("lat_ms_mean"),  int8.get("lat_ms_mean")),
        ("Latency p99   (ms)",     fp32.get("lat_ms_p99"),   fp16.get("lat_ms_p99"),   int8.get("lat_ms_p99")),
        ("Latency min   (ms)",     fp32.get("lat_ms_min"),   fp16.get("lat_ms_min"),   int8.get("lat_ms_min")),
        ("Latency max   (ms)",     fp32.get("lat_ms_max"),   fp16.get("lat_ms_max"),   int8.get("lat_ms_max")),
        ("Throughput    (FPS)",    fp32.get("fps"),          fp16.get("fps"),          int8.get("fps")),
        ("Engine size   (MB)",     fp32.get("engine_size_mb"), fp16.get("engine_size_mb"), int8.get("engine_size_mb")),
        ("GPU power mean(W)",      fp32.get("gpu_power_mean_w"), fp16.get("gpu_power_mean_w"), int8.get("gpu_power_mean_w")),
        ("GPU power max (W)",      fp32.get("gpu_power_max_w"),  fp16.get("gpu_power_max_w"),  int8.get("gpu_power_max_w")),
    ]
    for metric, fval, hval, ival in rows:
        d = 1 if "FPS" in metric or "MB" in metric else 2
        print(f"  {metric:<28}  {_fmt(fval, d):>14}  {_fmt(hval, d):>14}  {_fmt(ival, d):>14}")

    print("-" * W)
    print(f"  {'Speedup vs FP32':<28}  {'1.00×':>14}  "
          f"{(f'{sp_fp16:.2f}×' if sp_fp16 else 'N/A'):>14}  "
          f"{(f'{sp_int8:.2f}×' if sp_int8 else 'N/A'):>14}")
    print("=" * W)
    if (int8.get("lat_ms_mean") and fp16.get("lat_ms_mean")
            and int8["lat_ms_mean"] > fp16["lat_ms_mean"]):
        print("  NOTE: INT8-QAT slower than FP16 is expected on Xavier NX (Volta, sm_72).")
        print("  At batch=1 the model is memory-bandwidth-bound, not compute-bound,")
        print("  so INT8's 2× theoretical compute advantage does not materialise.")
        print("  The gain vs FP32 still demonstrates correct INT8 quantization.")



# Main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compile FP32/FP16 TRT engines and benchmark FP32/FP16/INT8 on Jetson Xavier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_dir", required=True,
                    help="Dir containing model_float_trt_base.onnx and model_int8_qat.engine")
    ap.add_argument("--trtexec", default=TRTEXEC_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=200,
                    help="Warmup duration in milliseconds passed to trtexec --warmUp")
    ap.add_argument("--workspace_mb", type=int, default=2048)
    ap.add_argument("--measure_power", action="store_true",
                    help="Measure GPU power draw with tegrastats")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force recompile even if .engine files already exist")
    ap.add_argument("--fp16_mm2_fp32", action="store_true",
                    help="Pin the 4 LiteMLA mm2 MatMuls to FP32 in the FP16 engine "
                         "(FP32 accumulation for the attention normalizer). FP32/INT8 "
                         "engines are unaffected. Verify layer-name matches in the "
                         "trtexec log; adjust with --fp16_mm2_layers if needed.")
    ap.add_argument("--fp16_mm2_layers", default=None,
                    help="Comma-separated TRT layer names to pin to FP32 for the FP16 "
                         "engine (overrides the sidecar / built-in defaults).")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: model_dir/benchmark_results.json)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()

    # ONNX preference: trt_base (onnxsim-simplified+static) > static > dynamic
    for name in ("model_float_trt_base.onnx", "model_float_static.onnx", "model_float.onnx"):
        if (model_dir / name).exists():
            float_onnx = model_dir / name
            break
    else:
        print(f"ERROR: no float ONNX found in {model_dir}", file=sys.stderr)
        sys.exit(1)


    fp32_engine = model_dir / "model_fp32.engine"
    fp16_engine = model_dir / "model_fp16.engine"
    # Prefer PTQ engine (built by build_trt_engine.py - properly fused INT8, ~1.8 MB).
    # Fall back to trtexec-built engine if PTQ not present.
    int8_qat    = model_dir / "model_int8_qat.engine"
    int8_legacy = model_dir / "model_int8.engine"
    int8_engine = int8_qat if int8_qat.exists() else int8_legacy
    out_json    = Path(args.out) if args.out else model_dir / "benchmark_results.json"

    if not os.path.exists(args.trtexec):
        print(f"ERROR: trtexec not found at {args.trtexec}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*68}")
    print(f"  Jetson Xavier TRT Benchmark  ({float_onnx.name})")
    print(f"  model_dir: {model_dir}")
    print(f"  trtexec:   {args.trtexec}")
    print(f"  input:     {args.batch_size}×{args.channels}×{args.img_size}×{args.img_size}")
    print(f"  iters: {args.iterations}  warmup: {args.warmup}")
    if int8_engine == int8_qat:
        print(f"  INT8 engine: {int8_qat.name}  (QAT — properly fused)")
    else:
        print(f"  INT8 engine: {int8_legacy.name}  (trtexec fallback — may be slow)")
    print(f"{'='*68}\n")

    # 1. Compile FP32 engine
    print(f"[ 1/5 ] Compiling FP32 engine from {float_onnx.name} …")
    compile_engine(args.trtexec, str(float_onnx), str(fp32_engine),
                   fp16=False, workspace_mb=args.workspace_mb, rebuild=args.rebuild)

    # 2. Compile FP16 engine
    # Built from the float ONNX and not a pre-converted fp16 ONNX: TRT 8.4's fp16-ONNX parser
    # drops the mixed-precision fp32 fallback and LiteMLA attention collapses. Optionally the
    # LiteMLA mm2 normalization (mm2, Slice, Add, Div) is pinned to FP32 with builder precision
    # flags, so the attention normalizer accumulates in FP32 and everything else is normal
    # --fp16. FP32/INT8 engines are not affected.
    fp16_extra: list[str] = []
    if args.fp16_mm2_fp32:
        layers = _fp16_mm2_layer_names(model_dir, args.fp16_mm2_layers)
        if layers:
            specs = ",".join(f"{n}:fp32" for n in layers)
            fp16_extra = [
                "--precisionConstraints=obey",
                f"--layerPrecisions={specs}",
                f"--layerOutputTypes={specs}",
            ]
            print(f"\n[ 2/5 ] Compiling FP16 engine from {float_onnx.name} "
                  f"(pinning {len(layers)} LiteMLA mm2-normalization node(s) → FP32) …")
        else:
            print(f"\n[ 2/5 ] Compiling FP16 engine from {float_onnx.name} "
                  f"(--fp16_mm2_fp32 set but no mm2 layer names found — plain --fp16) …")
    else:
        print(f"\n[ 2/5 ] Compiling FP16 engine from {float_onnx.name} (plain --fp16) …")
    compile_engine(args.trtexec, str(float_onnx), str(fp16_engine),
                   fp16=True, workspace_mb=args.workspace_mb, rebuild=args.rebuild,
                   extra_args=fp16_extra)

    # 3. INT8 PTQ engine
    print(f"\n[ 3/5 ] INT8-QAT engine: {int8_engine.name}")
    if not int8_engine.exists():
        print("  ERROR: INT8 engine not found. Build it first with:", file=sys.stderr)
        print(f"    /usr/src/tensorrt/bin/trtexec --onnx={model_dir}/model_qdq_trt_no_out_static.onnx \\",
              file=sys.stderr)
        print(f"        --int8 --fp16 --saveEngine={int8_qat} --workspace=2048", file=sys.stderr)
        sys.exit(1)
    print(f"  engine found: {int8_engine}  "
          f"({int8_engine.stat().st_size / 1e6:.1f} MB)")

    # 4. Benchmark
    print(f"\n[ 4/5 ] Benchmarking …  ({args.iterations} iters, warmup={args.warmup})")

    print(f"\n  --- FP32 ---")
    fp32_stats = benchmark_engine(
        args.trtexec, str(fp32_engine),
        iterations=args.iterations, warmup=args.warmup, batch_size=args.batch_size,
        profile_path=str(model_dir / "profile_fp32.json"),
    )

    print(f"\n  --- FP16 ---")
    fp16_stats = benchmark_engine(
        args.trtexec, str(fp16_engine),
        iterations=args.iterations, warmup=args.warmup, batch_size=args.batch_size,
        profile_path=str(model_dir / "profile_fp16.json"),
    )

    print(f"\n  --- INT8-QAT ---")
    int8_stats = benchmark_engine(
        args.trtexec, str(int8_engine),
        iterations=args.iterations, warmup=args.warmup, batch_size=args.batch_size,
        profile_path=str(model_dir / "profile_int8.json"),
    )

    # 5. Power measurement (optional)
    if args.measure_power:
        print("\n[ 5/5 ] Measuring power draw with tegrastats …")
        print("  FP32:")
        fp32_stats.update(measure_power(args.trtexec, str(fp32_engine),
            batch_size=args.batch_size, iterations=args.iterations, warmup=args.warmup))
        print("  FP16:")
        fp16_stats.update(measure_power(args.trtexec, str(fp16_engine),
            batch_size=args.batch_size, iterations=args.iterations, warmup=args.warmup))
        print("  INT8-QAT:")
        int8_stats.update(measure_power(args.trtexec, str(int8_engine),
            batch_size=args.batch_size, iterations=args.iterations, warmup=args.warmup))
    else:
        print("\n[ 5/5 ] Power measurement skipped (pass --measure_power to enable)")

    # Print comparison
    print_comparison(fp32_stats, fp16_stats, int8_stats)

    # Save JSON
    def _speedup(baseline, other):
        if baseline.get("lat_ms_mean") and other.get("lat_ms_mean"):
            return round(baseline["lat_ms_mean"] / other["lat_ms_mean"], 3)
        return None

    output = {
        "device":           "jetson_xavier",
        "model_dir":        str(model_dir),
        "batch_size":       args.batch_size,
        "iterations":       args.iterations,
        "warmup":           args.warmup,
        "trtexec":          args.trtexec,
        "fp32":             fp32_stats,
        "fp16":             fp16_stats,
        "int8":             int8_stats,
        "int8_engine":      int8_engine.name,
        "speedup_fp16_vs_fp32": _speedup(fp32_stats, fp16_stats),
        "speedup_int8_vs_fp32": _speedup(fp32_stats, int8_stats),
        "speedup_int8_vs_fp16": _speedup(fp16_stats, int8_stats),
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_json}")
    print("\nNext step — mIoU evaluation:")
    print(f"  PYTHONPATH=/usr/lib/python3.8/dist-packages \\")
    print(f"  ~/miniforge3/envs/flood/bin/python jetson/eval_jetson.py \\")
    print(f"      --model_dir {model_dir} \\")
    print(f"      --engine {int8_engine}")


if __name__ == "__main__":
    main()
