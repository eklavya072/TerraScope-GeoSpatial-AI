"""Measure what the missing scene identifier costs, instead of only warning about it.

EuroSAT tiles are cut from larger Sentinel-2 scenes. The corpus as redistributed
carries no scene identifier, so the committed split can be stratified by class
but cannot be grouped by scene, and spatially adjacent tiles may land in
different folds. The README has always said so. What it could not say is how
much it matters.

Two measurements, from the corpus plus committed artefacts only.

1. NEAREST-NEIGHBOUR SIMILARITY, WITH A CONTROL.

   For every test tile, the cosine similarity to its most similar TRAIN tile,
   computed on per-tile standardised pixels so that overall brightness cannot
   dominate the comparison.

   That number alone means nothing. Two Forest tiles are similar because forests
   are similar, not because they leaked. The control is the same statistic
   computed INSIDE the train fold: 2,700 randomly chosen train tiles, each
   against the other 18,899. Candidate-pool sizes match (18,900 vs 18,899), so
   the two distributions are directly comparable, and train tiles certainly do
   share scenes with other train tiles.

     test-vs-train well BELOW train-vs-train  ->  folds are meaningfully separated
     test-vs-train AT train-vs-train          ->  a test tile sits as close to the
                                                  training data as a same-scene
                                                  neighbour does

2. WHAT IT COSTS IN ACCURACY.

   The committed efficientnet_lite0 int8_static graph is evaluated on the whole
   test fold, then on the subset whose nearest train neighbour falls below each
   threshold. The gap between those is the inflation -- measured, not assumed.

This bounds obvious duplication. It cannot recover true scene membership, which
is not in the data; a pair from the same scene that happens to look different
will not be caught. Read it as a lower bound on leakage, not a clean bill.

Usage:
    python scripts/leakage_check.py            # writes results/leakage.json
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.config import CLASSES, NORM_MEAN, NORM_STD  # noqa: E402

CORPUS = os.path.join("data", "eurosat_rgb")
SPLIT = os.path.join("splits", "eurosat_split_seed42.csv")
GRAPH = os.path.join("web", "models", "efficientnet_lite0_seed0_int8_static.onnx")
OUT = os.path.join("results", "leakage.json")

THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
CONTROL_SEED = 42
CHUNK = 2048


def load_split():
    folds = {"train": [], "test": []}
    with open(SPLIT) as fh:
        for row in csv.DictReader(fh):
            if row["fold"] in folds:
                folds[row["fold"]].append((row["filename"], row["label"]))
    return folds


def load_pixels(items) -> np.ndarray:
    """(n, 12288) float32, per-tile standardised so cosine == correlation."""
    out = np.empty((len(items), 64 * 64 * 3), dtype=np.float32)
    for i, (name, _) in enumerate(items):
        with Image.open(os.path.join(CORPUS, name)) as im:
            out[i] = np.asarray(im.convert("RGB"), dtype=np.float32).ravel()
    out -= out.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out /= norms
    return out


def max_similarity(probe: np.ndarray, pool: np.ndarray,
                   exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """For each probe row, the best cosine similarity in pool, and its index."""
    best = np.full(len(probe), -np.inf, dtype=np.float32)
    arg = np.zeros(len(probe), dtype=np.int64)
    for start in range(0, len(pool), CHUNK):
        block = pool[start:start + CHUNK]
        sim = probe @ block.T
        if exclude_self:
            for i in range(len(probe)):
                j = i - start
                if 0 <= j < len(block):
                    sim[i, j] = -np.inf
        idx = sim.argmax(axis=1)
        val = sim[np.arange(len(probe)), idx]
        better = val > best
        best[better] = val[better]
        arg[better] = start + idx[better]
    return best, arg


def describe(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "median": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max())}


def accuracy_on(graph, pix_raw: np.ndarray, labels: np.ndarray,
                mask: np.ndarray) -> tuple[float, int]:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(graph, opts, providers=["CPUExecutionProvider"])
    x, y = pix_raw[mask], labels[mask]
    if len(y) == 0:
        return float("nan"), 0
    correct = 0
    for i in range(0, len(x), 64):
        logits = sess.run(None, {"input": x[i:i + 64]})[0]
        correct += int((logits.argmax(1) == y[i:i + 64]).sum())
    return correct / len(y), len(y)


def model_inputs(items) -> tuple[np.ndarray, np.ndarray]:
    """NCHW float32 with ImageNet statistics -- the graph's documented contract."""
    n = len(items)
    x = np.empty((n, 3, 64, 64), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(NORM_STD, dtype=np.float32).reshape(3, 1, 1)
    for i, (name, label) in enumerate(items):
        with Image.open(os.path.join(CORPUS, name)) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        x[i] = (a.transpose(2, 0, 1) - mean) / std
        y[i] = CLASSES.index(label)
    return x, y


def main() -> int:
    if not os.path.isdir(CORPUS):
        print(f"{CORPUS} not found; run `make data` first", file=sys.stderr)
        return 2

    folds = load_split()
    train, test = folds["train"], folds["test"]
    print(f"train={len(train)}  test={len(test)}", flush=True)

    print("loading pixels ...", flush=True)
    tr = load_pixels(train)
    te = load_pixels(test)

    print("test -> train nearest neighbour ...", flush=True)
    te_best, te_arg = max_similarity(te, tr, exclude_self=False)

    print("control: train -> train nearest neighbour ...", flush=True)
    rng = np.random.default_rng(CONTROL_SEED)
    pick = rng.choice(len(train), size=len(test), replace=False)
    # exclude_self is applied against the FULL train pool, so a probe never
    # matches itself; every other train tile stays a candidate.
    ctl_best = np.full(len(pick), -np.inf, dtype=np.float32)
    for start in range(0, len(tr), CHUNK):
        block = tr[start:start + CHUNK]
        sim = tr[pick] @ block.T
        for i, p in enumerate(pick):
            j = p - start
            if 0 <= j < len(block):
                sim[i, j] = -np.inf
        ctl_best = np.maximum(ctl_best, sim.max(axis=1))

    same_class = np.array(
        [train[a][1] == t[1] for a, t in zip(te_arg, test)], dtype=bool)

    print("evaluating the committed graph ...", flush=True)
    x, y = model_inputs(test)
    acc_all, n_all = accuracy_on(GRAPH, x, y, np.ones(len(test), dtype=bool))

    report = {
        "test_vs_train": describe(te_best),
        "control_train_vs_train": describe(ctl_best),
        "nearest_neighbour_same_class_fraction": float(same_class.mean()),
        "accuracy_full_test_fold": acc_all,
        "n_full_test_fold": n_all,
        "by_threshold": [],
    }

    for t in THRESHOLDS:
        flagged = te_best >= t
        clean = ~flagged
        acc_clean, n_clean = accuracy_on(GRAPH, x, y, clean)
        ctl_frac = float((ctl_best >= t).mean())
        report["by_threshold"].append({
            "threshold": t,
            # Validity check. If the flagged pairs were near-duplicates, they
            # should overwhelmingly share a class; if they do not, raw-pixel
            # similarity is picking up flat texture rather than duplication and
            # the leakage reading does not hold.
            "same_class_fraction": (float(same_class[flagged].mean())
                                    if flagged.any() else None),
            "test_tiles_at_or_above": int(flagged.sum()),
            "test_fraction": float(flagged.mean()),
            "control_fraction": ctl_frac,
            "accuracy_excluding_them": acc_clean,
            "n_excluding_them": n_clean,
            "accuracy_delta_pp": (acc_clean - acc_all) * 100.0,
        })

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print()
    print("similarity of a tile to its nearest neighbour in the train fold")
    print(f"  test  -> train : median {report['test_vs_train']['median']:.4f}"
          f"  p99 {report['test_vs_train']['p99']:.4f}"
          f"  max {report['test_vs_train']['max']:.4f}")
    print(f"  train -> train : median {report['control_train_vs_train']['median']:.4f}"
          f"  p99 {report['control_train_vs_train']['p99']:.4f}"
          f"  max {report['control_train_vs_train']['max']:.4f}   (control)")
    print()
    print(f"accuracy on the full test fold: {acc_all * 100:.2f}%  (n={n_all})")
    for r in report["by_threshold"]:
        print(f"  >= {r['threshold']:.2f}: {r['test_tiles_at_or_above']:5d} test tiles "
              f"({r['test_fraction'] * 100:5.2f}%)   control {r['control_fraction'] * 100:5.2f}%"
              f"   same-class {(r['same_class_fraction'] or 0) * 100:5.1f}%"
              f"   accuracy without them {r['accuracy_excluding_them'] * 100:.2f}% "
              f"({r['accuracy_delta_pp']:+.2f} pp)")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
