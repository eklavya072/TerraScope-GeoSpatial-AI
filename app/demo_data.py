"""Data access for the demo app. No Streamlit, no hardcoded results.

Every figure the app displays either comes from this module -- which reads the
committed benchmark artefacts -- or is measured live in the browser session.
There are deliberately no numeric literals here that describe a result: if a
value cannot be sourced from results/summary.json it is returned as None and the
UI prints "not measured" rather than inventing something.

Nothing here imports torch or tensorflow. The app runs on onnxruntime alone.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from bench.config import (CLASSES, INPUT_SIZE, NORM_MEAN, NORM_STD, SPLIT_CSV,
                          recipe_hash)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "results", "summary.json")
MODEL_DIR = os.path.join(ROOT, "app", "models_onnx")
SAMPLES_DIR = os.path.join(ROOT, "app", "samples")

PRECISIONS = ("fp32", "int8_dynamic", "int8_static")
# The configuration every reported headline uses; see PROTOCOL.md.
REPORT_THREADS = 1
REPORT_BATCH = 1

# Timing discipline for live measurement, mirroring bench/benchmark.py in spirit
# though not in scale -- a browser request cannot run for 20 seconds.
WARMUP_RUNS = 5
TIMED_RUNS = 20


# --------------------------------------------------------------- artefacts ---

def load_summary(path: str = SUMMARY_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_samples(directory: str = SAMPLES_DIR) -> dict:
    index = os.path.join(directory, "index.json")
    if not os.path.exists(index):
        return {"tiles": [], "note": "no sample tiles bundled"}
    with open(index) as fh:
        return json.load(fh)


def model_path(model: str, precision: str, seed: int = 0) -> str:
    return os.path.join(MODEL_DIR, f"{model}_seed{seed}_{precision}.onnx")


def available_models(summary: dict) -> list[str]:
    """Zoo members that have a shipped graph for every precision."""
    models = sorted({v["model"] for v in summary["measured"].values()})
    return [m for m in models
            if all(os.path.exists(model_path(m, p)) for p in PRECISIONS)]


def measured(summary: dict, model: str, precision: str,
             threads: int = REPORT_THREADS, batch: int = REPORT_BATCH) -> dict | None:
    """The measured row for one configuration, or None if it was not measured."""
    return summary["measured"].get(f"{model}|{precision}|t{threads}|b{batch}")


def accuracy_ci(summary: dict, model: str, precision: str) -> tuple | None:
    """(mean_pct, half_width_pct) for a configuration, or None."""
    row = measured(summary, model, precision)
    if not row:
        return None
    acc = row["test_acc"]
    half = acc.get("half_width")
    return (acc["mean"] * 100.0, half * 100.0 if half is not None else None)


def quantisation_delta(summary: dict, model: str, precision: str) -> dict | None:
    return summary.get("quantisation_delta", {}).get(f"{model}|{precision}")


def daily_co2e_grams(summary: dict, model: str, precision: str,
                     inferences_per_day: int) -> float | None:
    """CO2e for a daily inference volume, scaled from the measured per-1M figure.

    Scaling a measured figure is not the same as inventing one: the per-million
    value is measured, and the only arithmetic here is the volume the user chose.
    """
    row = measured(summary, model, precision)
    if not row or not row.get("co2e_g_per_1m"):
        return None
    return row["co2e_g_per_1m"]["mean"] * (inferences_per_day / 1_000_000)


def provenance(summary: dict) -> dict:
    """Everything the receipts screen needs, all read from the artefacts."""
    return {
        "split_csv": SPLIT_CSV,
        "split_sha256": summary.get("split_sha256"),
        "recipe_hash": summary.get("recipe_hash"),
        "expected_recipe_hash": recipe_hash(),
        "environment": summary.get("environment", {}),
        "grid_intensity": summary.get("grid_intensity_g_co2e_per_kwh"),
        "grid_intensity_source": summary.get("grid_intensity_source"),
        "energy_note": summary.get("energy_note"),
        "exclusions": summary.get("exclusions", {}),
        "thread_regime_confound": summary.get("thread_regime_confound", {}),
        "seeds": sorted({n for v in summary.get("accuracy_over_seeds", {}).values()
                         for n in [v.get("n")] if n}),
    }


# ------------------------------------------------------------ preprocessing ---

def preprocess(image: Image.Image) -> np.ndarray:
    """Exactly the benchmark's preprocessing: 64x64, ImageNet stats, NCHW float32.

    Constants come from bench.config so the app cannot drift from the pipeline
    that produced the accuracy figures it displays.
    """
    img = image.convert("RGB")
    if img.size != (INPUT_SIZE, INPUT_SIZE):
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(NORM_MEAN, dtype=np.float32)) / np.array(NORM_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


# ------------------------------------------------------------- live timing ---

@dataclass
class Prediction:
    model: str
    precision: str
    label: str
    confidence: float
    probabilities: np.ndarray
    latency_ms_median: float
    latency_ms_min: float
    timed_runs: int
    warmup_runs: int


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def classify_and_time(model: str, precision: str, tensor: np.ndarray,
                      warmup: int = WARMUP_RUNS,
                      runs: int = TIMED_RUNS) -> Prediction:
    """Run one model on one tile, timing it honestly.

    The session is built and released inside this call. Holding fifteen sessions
    open would exceed the deployment's memory budget -- ResNet-50 fp32 alone is
    ~162 MB resident -- and session construction plus the first inference are
    kept out of the timed window, since they measure graph setup rather than
    inference.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = REPORT_THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path(model, precision), opts,
                                   providers=["CPUExecutionProvider"])
    try:
        feed = {session.get_inputs()[0].name: tensor}
        for _ in range(warmup):
            session.run(None, feed)

        latencies = []
        logits = None
        for _ in range(runs):
            start = time.perf_counter()
            logits = session.run(None, feed)[0]
            latencies.append((time.perf_counter() - start) * 1000.0)

        probs = _softmax(np.asarray(logits, dtype=np.float64).ravel())
        top = int(probs.argmax())
        return Prediction(
            model=model, precision=precision,
            label=CLASSES[top], confidence=float(probs[top]),
            probabilities=probs,
            latency_ms_median=float(np.median(latencies)),
            latency_ms_min=float(np.min(latencies)),
            timed_runs=runs, warmup_runs=warmup,
        )
    finally:
        del session          # release before the next model is built


def pareto_frontier(points: list[dict], x_key: str = "energy",
                    y_key: str = "accuracy", label_key: str = "config") -> list[str]:
    """Labels on the frontier minimising x_key and maximising y_key.

    This duplicates bench.stats.pareto_frontier deliberately. That module
    imports scipy, which the demo's dependency set excludes to keep the
    deployed image small, so the app cannot call it. Duplicated logic is a
    liability, so tests/test_demo_app.py asserts the two implementations agree
    on the real measured data -- if they ever diverge, CI fails rather than the
    app quietly drawing a different frontier than the paper figure.
    """
    usable = [p for p in points
              if p.get(x_key) is not None and p.get(y_key) is not None]
    frontier = []
    for p in usable:
        dominated = any(
            q is not p and q[x_key] <= p[x_key] and q[y_key] >= p[y_key]
            and (q[x_key] < p[x_key] or q[y_key] > p[y_key])
            for q in usable)
        if not dominated:
            frontier.append(p[label_key])
    return frontier
