"""Generate the canonical train/val/test split for the TerraScope benchmark.

EuroSAT ships no official train/test split, so every published EuroSAT number
is measured against a fold definition the reader cannot see. This script fixes
that: it derives a stratified 70/20/10 split deterministically from a fixed
seed and writes it to a file that is committed to the repository. Every result
in this benchmark references that file by its sha256.

Determinism: filenames are sorted lexicographically before any random draw, and
classes are visited in sorted order, so the output depends only on the corpus
contents and the seed -- not on filesystem order, dict ordering or PYTHONHASHSEED.

Output:
    splits/eurosat_split_seed<seed>.csv         filename,label,fold
    splits/eurosat_split_seed<seed>.json        provenance metadata
    splits/eurosat_split_seed<seed>.csv.sha256  hash referenced by all results

Usage:
    python scripts/make_split.py [--seed 42]
"""

import argparse
import collections
import hashlib
import json
import os
import sys
from datetime import date

import numpy as np

CORPUS_ROOT = os.path.join("data", "eurosat_rgb")
MANIFEST = os.path.join(CORPUS_ROOT, "MANIFEST.sha256")
SPLIT_DIR = "splits"
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
EXPECTED_N = 27_000


def read_manifest() -> list[str]:
    if not os.path.exists(MANIFEST):
        sys.exit(f"missing {MANIFEST} -- run scripts/prepare_data.py first")
    with open(MANIFEST) as fh:
        return sorted(line.split("  ", 1)[1].strip() for line in fh if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = read_manifest()
    if len(files) != EXPECTED_N:
        sys.exit(f"corpus has {len(files)} tiles, expected {EXPECTED_N}")

    by_class: dict[str, list[str]] = collections.defaultdict(list)
    for rel in files:
        by_class[rel.split("/")[0]].append(rel)

    rng = np.random.default_rng(args.seed)
    rows: list[tuple[str, str, str]] = []
    for label in sorted(by_class):
        items = sorted(by_class[label])
        order = rng.permutation(len(items))
        n_train = int(round(len(items) * RATIOS["train"]))
        n_val = int(round(len(items) * RATIOS["val"]))
        for rank, idx in enumerate(order):
            fold = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
            rows.append((items[idx], label, fold))

    rows.sort()
    csv_path = os.path.join(SPLIT_DIR, f"eurosat_split_seed{args.seed}.csv")
    os.makedirs(SPLIT_DIR, exist_ok=True)
    with open(csv_path, "w") as fh:
        fh.write("filename,label,fold\n")
        for rel, label, fold in rows:
            fh.write(f"{rel},{label},{fold}\n")

    with open(csv_path, "rb") as fh:
        split_sha = hashlib.sha256(fh.read()).hexdigest()
    with open(MANIFEST, "rb") as fh:
        corpus_sha = hashlib.sha256(fh.read()).hexdigest()

    counts = collections.Counter(fold for _, _, fold in rows)
    per_class = {
        label: dict(collections.Counter(f for _, l, f in rows if l == label))
        for label in sorted(by_class)
    }
    meta = {
        "dataset": "EuroSAT (RGB), Helber et al. -- MIT licensed",
        "corpus_manifest_sha256": corpus_sha,
        "split_csv_sha256": split_sha,
        "seed": args.seed,
        "ratios": RATIOS,
        "stratified_by": "class label",
        "n_total": len(rows),
        "fold_counts": dict(counts),
        "per_class_counts": per_class,
        "generated": date.today().isoformat(),
        "generator": "scripts/make_split.py",
    }
    with open(os.path.join(SPLIT_DIR, f"eurosat_split_seed{args.seed}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(csv_path + ".sha256", "w") as fh:
        fh.write(f"{split_sha}  {os.path.basename(csv_path)}\n")

    print(f"wrote {csv_path}")
    print(f"  folds: {dict(counts)}")
    print(f"  split_csv_sha256={split_sha}")
    print(f"  corpus_manifest_sha256={corpus_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
