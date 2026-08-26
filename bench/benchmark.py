"""Measure accuracy, latency, memory and energy for every model x precision.

Everything measured here runs through ONNX Runtime on the CPU execution
provider. No MPS, no CoreML: the question this benchmark answers is what a
CPU-only ministry server can do, so the accelerator is excluded explicitly
rather than merely left unused.

Measurement discipline:
  * thread counts are pinned (ORT intra/inter-op AND the OMP/MKL environment)
    and recorded in every row -- unpinned thread counts are the single most
    common reason CPU latency numbers fail to reproduce;
  * WARMUP inferences are discarded before timing begins, so first-call graph
    allocation and page faults do not land in the distribution;
  * at least MIN_RUNS timed inferences per configuration, reported as
    p50/p95/p99 plus mean and standard deviation rather than a single number;
  * peak RSS is sampled around the run;
  * the wall-clock window is recorded so bench.power can integrate sampled CPU
    package power over exactly the timed region.

Usage:
    OMP_NUM_THREADS=1 python -m bench.benchmark --model all --seeds 0,1,2,3,4
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import psutil

from bench import utils
from bench.config import MODEL_ZOO, RESULTS_DIR, SPLIT_CSV
from bench.data import EuroSATFold
from bench.export_onnx import ONNX_DIR, PRECISIONS
from bench.power import energy_joules, parse_log

BENCH_JSONL = os.path.join(RESULTS_DIR, "bench.jsonl")
POWER_LOG = os.path.join(RESULTS_DIR, "power_log.txt")

WARMUP = 50
MIN_RUNS = 1000
BATCH_SIZES = (1, 32)
# Single thread is the primary configuration: it is the reproducible one, and
# it is what a shared multi-tenant ministry server realistically grants a
# single inference process. The 4-thread rows show what headroom exists.
THREAD_CONFIGS = (1, 4)


def make_session(path: str, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def accuracy(session: ort.InferenceSession, x: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    for i in range(0, len(x), 64):
        logits = session.run(None, {"input": x[i:i + 64]})[0]
        correct += int((logits.argmax(1) == y[i:i + 64]).sum())
    return correct / len(y)


def time_inference(session, sample: np.ndarray, n_runs: int) -> dict:
    for _ in range(WARMUP):
        session.run(None, {"input": sample})

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    lat = np.empty(n_runs, dtype=np.float64)
    t_start = time.time()
    for i in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {"input": sample})
        lat[i] = (time.perf_counter() - t0) * 1000.0     # ms
    t_end = time.time()
    rss_peak = max(rss_before, proc.memory_info().rss)

    return {
        "n_timed_runs": n_runs,
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_std": float(lat.std(ddof=1)),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_p99": float(np.percentile(lat, 99)),
        "latency_ms_min": float(lat.min()),
        "wall_start": t_start,
        "wall_end": t_end,
        "wall_seconds": t_end - t_start,
        "rss_peak_bytes": rss_peak,
    }


def bench_one(name: str, seed: int, precision: str, x_test, y_test,
              power_samples, split_sha: str, env: dict) -> list[dict]:
    path = os.path.join(ONNX_DIR, f"{name}_seed{seed}_{precision}.onnx")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run bench.export_onnx first")

    rows = []
    acc = None
    for threads in THREAD_CONFIGS:
        session = make_session(path, threads)
        if acc is None:                     # accuracy is thread-count invariant
            acc = accuracy(session, x_test, y_test)
        for bs in BATCH_SIZES:
            sample = np.ascontiguousarray(x_test[:bs])
            n_runs = MIN_RUNS if bs == 1 else max(100, MIN_RUNS // bs)
            timing = time_inference(session, sample, n_runs)
            energy = energy_joules(power_samples, timing["wall_start"],
                                   timing["wall_end"])
            per_inf = None
            if energy["joules"] is not None:
                total_images = n_runs * bs
                per_inf = energy["joules"] / total_images
            rows.append({
                "kind": "bench",
                "model": name,
                "timm_id": MODEL_ZOO[name],
                "seed": seed,
                "precision": precision,
                "threads_intra_op": threads,
                "threads_inter_op": 1,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
                "batch_size": bs,
                "test_acc": acc,
                "onnx_bytes": os.path.getsize(path),
                "warmup_runs": WARMUP,
                **timing,
                "energy_source": "powermetrics cpu_power (Apple Silicon on-die telemetry)",
                "energy_joules_window": energy["joules"],
                "energy_mean_power_w": energy["mean_power_w"],
                "energy_samples": energy["n_samples"],
                "energy_window_coverage": energy["coverage"],
                "energy_joules_per_inference": per_inf,
                "energy_joules_per_1k_inferences":
                    per_inf * 1000 if per_inf is not None else None,
                "split_sha256": split_sha,
                "env": env,
            })
            print(f"  {name:19s} s{seed} {precision:12s} t{threads} b{bs:<2d} "
                  f"acc={acc:.4f} p95={timing['latency_ms_p95']:8.3f}ms "
                  f"E/1k={'n/a' if per_inf is None else f'{per_inf*1000:7.2f}J'}",
                  flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--precisions", default=",".join(PRECISIONS))
    ap.add_argument("--split", default=SPLIT_CSV)
    ap.add_argument("--power-log", default=POWER_LOG)
    args = ap.parse_args()

    split_sha = utils.verify_split(args.split)
    env = utils.environment()

    ds = EuroSATFold("test", args.split)
    x_test = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    y_test = np.array([ds[i][1] for i in range(len(ds))])

    power_samples = parse_log(args.power_log)
    if not power_samples:
        print(f"WARNING: no power samples in {args.power_log}. Latency and "
              f"accuracy will be recorded; energy columns will be null.\n"
              f"         Start the sampler first:  sudo ./scripts/energy_sampler.sh",
              flush=True)
    else:
        print(f"power log: {len(power_samples)} samples spanning "
              f"{(power_samples[-1][0]-power_samples[0][0])/60:.1f} min", flush=True)

    names = list(MODEL_ZOO) if args.model == "all" else args.model.split(",")
    for name in names:
        for seed in (int(s) for s in args.seeds.split(",")):
            for precision in args.precisions.split(","):
                for row in bench_one(name, seed, precision, x_test, y_test,
                                     power_samples, split_sha, env):
                    utils.append_jsonl(BENCH_JSONL, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
