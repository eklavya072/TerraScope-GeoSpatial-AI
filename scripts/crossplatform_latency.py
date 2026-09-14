"""Re-measure latency on whatever CPU this runs on, and compare the ORDER.

README's first limitation is the one most likely to change a conclusion:

    Single hardware platform. One Apple M2. Latency and energy rankings may
    differ on x86, on server-class CPUs with AVX-512, or under different memory
    bandwidth.

That caveat was never tested, because testing it appeared to need a second
machine. It does not. Latency depends on the shape of the input tensor, not on
its contents, so the seven committed ONNX graphs can be timed anywhere -- no
EuroSAT download, no checkpoints, no GPU. A free x86 CI runner is a second
platform.

What this can and cannot establish, stated plainly:

  * It CAN check whether the ORDER of configurations survives a change of
    instruction set. That order is the deployment claim -- "EfficientNet-Lite0
    int8 is the cheapest thing worth running" -- and if it inverts on x86, the
    recommendation is platform-specific and the README must say so.

  * It CANNOT produce publication-grade absolute numbers. CI runners are shared
    virtual machines with noisy neighbours and no thermal guarantees. Absolute
    milliseconds from here are indicative and are labelled as such wherever they
    are reported. The benchmark's own figures stay the Apple M2 ones.

  * It CANNOT measure energy or accuracy. Energy needs privileged on-die
    telemetry the runner does not expose; accuracy needs the corpus.

Rank agreement is Spearman's rho, computed here rather than imported so the CI
job needs onnxruntime and numpy and nothing else.

Usage:
    python scripts/crossplatform_latency.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import onnxruntime as ort

MODELS = "web/models"
SUMMARY = os.path.join("results", "summary.json")

# Mirrors bench/benchmark.py exactly. A cross-platform comparison run under a
# different protocol would compare the protocols, not the platforms.
WARMUP = 50
N_RUNS = 300
THREADS = 1
SHAPE = (1, 3, 64, 64)


def make_session(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def time_graph(path: str) -> dict:
    session = make_session(path)
    # Latency is shape-dependent, not content-dependent; a fixed seed keeps the
    # tensor identical across platforms so nothing but the CPU differs.
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(SHAPE, dtype=np.float32)

    for _ in range(WARMUP):
        session.run(None, {"input": sample})

    lat = np.empty(N_RUNS, dtype=np.float64)
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        session.run(None, {"input": sample})
        lat[i] = (time.perf_counter() - t0) * 1000.0

    return {"p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95))}


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, ties averaged. Returns 1.0 for identical orderings."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    if not os.path.isdir(MODELS):
        print(f"{MODELS} not found; run from the repository root", file=sys.stderr)
        return 2

    graphs = sorted(f for f in os.listdir(MODELS) if f.endswith(".onnx"))
    if not graphs:
        print(f"no .onnx graphs in {MODELS}", file=sys.stderr)
        return 2

    measured = json.load(open(SUMMARY))["measured"]

    rows = []
    for f in graphs:
        stem = f[:-len(".onnx")]
        model, _, precision = stem.partition("_seed0_")
        here = time_graph(os.path.join(MODELS, f))
        ref = measured.get(f"{model}|{precision}|t1|b1")
        rows.append({
            "model": model,
            "precision": precision,
            "here_p50": here["p50"],
            "here_p95": here["p95"],
            "m2_p95": ref["latency_p95_ms"]["mean"] if ref else None,
        })

    paired = [r for r in rows if r["m2_p95"] is not None]
    rho = spearman([r["here_p95"] for r in paired],
                   [r["m2_p95"] for r in paired]) if len(paired) > 1 else float("nan")

    order_here = [f"{r['model']}|{r['precision']}"
                  for r in sorted(paired, key=lambda r: r["here_p95"])]
    order_m2 = [f"{r['model']}|{r['precision']}"
                for r in sorted(paired, key=lambda r: r["m2_p95"])]
    identical = order_here == order_m2

    uname = platform.uname()
    header = (f"{uname.system} {uname.machine} | "
              f"onnxruntime {ort.__version__} | "
              f"{THREADS} intra-op thread, batch 1, {N_RUNS} timed runs "
              f"after {WARMUP} warm-ups")

    lines = [
        "### Cross-platform latency check",
        "",
        header,
        "",
        "Indicative only: CI runners are shared virtual machines. The benchmark's "
        "published figures are the Apple M2 ones. What is being tested here is "
        "the ORDER, which is what the deployment recommendation rests on.",
        "",
        "| Model | Precision | p50 (ms) | p95 (ms) | M2 p95 (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: r["here_p95"]):
        m2 = f"{r['m2_p95']:.2f}" if r["m2_p95"] is not None else "—"
        lines.append(f"| {r['model']} | {r['precision']} | "
                     f"{r['here_p50']:.2f} | {r['here_p95']:.2f} | {m2} |")
    lines += [
        "",
        f"**Rank agreement with the Apple M2: Spearman rho = {rho:.3f}** "
        f"over {len(paired)} paired configurations.",
        "",
        ("Ordering is identical on both platforms."
         if identical else
         "**Ordering differs between platforms.** Cheapest-first here: "
         + " < ".join(order_here) + ". On the M2: " + " < ".join(order_m2) + "."),
    ]
    report = "\n".join(lines)
    print(report)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fh:
            fh.write(report + "\n")

    # Divergence is a finding, not a build failure: this job exists to report
    # what a second platform does, and failing CI on it would create pressure to
    # stop looking. Only a broken run is an error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
