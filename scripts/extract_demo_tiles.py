"""Extract demo tiles for the Streamlit app from the TEST fold only.

Why the assertion matters: classifying a tile the models were trained on would
make the demo look better than the benchmark, which is the precise dishonesty
this repository exists to avoid. The committed split records every tile's fold,
so the mistake is avoidable -- and this script fails loudly rather than quietly
picking a train tile if the split ever changes shape.

Output:
    app/samples/<Class>_<n>.png     tiles, PNG (lossless; the source is JPEG)
    app/samples/index.json          true label and split provenance per tile

Usage:
    python scripts/extract_demo_tiles.py [--per-class 2]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from bench.config import CLASSES, CORPUS_ROOT, SPLIT_CSV

OUT_DIR = os.path.join("app", "samples")
REQUIRED_FOLD = "test"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7,
                    help="selection seed; deterministic given the split")
    args = ap.parse_args()

    if not os.path.isdir(CORPUS_ROOT):
        sys.exit(f"missing {CORPUS_ROOT} -- run scripts/prepare_data.py first")

    with open(SPLIT_CSV) as fh:
        rows = list(csv.DictReader(fh))

    by_class = defaultdict(list)
    for r in rows:
        if r["fold"] == REQUIRED_FOLD:
            by_class[r["label"]].append(r)

    import random
    rng = random.Random(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    index = []
    for label in CLASSES:
        pool = sorted(by_class[label], key=lambda r: r["filename"])
        if len(pool) < args.per_class:
            sys.exit(f"class {label} has only {len(pool)} test tiles")
        for row in rng.sample(pool, args.per_class):
            # HARD GUARANTEE: never ship a tile any model was trained on.
            assert row["fold"] == REQUIRED_FOLD, (
                f"{row['filename']} is in fold {row['fold']!r}, not "
                f"{REQUIRED_FOLD!r} -- refusing to ship a training tile as a demo")
            assert row["label"] == label

            src = os.path.join(CORPUS_ROOT, row["filename"])
            dest_name = os.path.basename(row["filename"]).replace(".jpg", ".png")
            dest = os.path.join(OUT_DIR, dest_name)
            with Image.open(src) as im:
                im.convert("RGB").save(dest, format="PNG", optimize=True)

            with open(src, "rb") as fh:
                source_sha = hashlib.sha256(fh.read()).hexdigest()
            index.append({
                "file": dest_name,
                "true_label": label,
                "split_path": row["filename"],
                "fold": row["fold"],
                "source_sha256": source_sha,
            })

    with open(os.path.join(SPLIT_CSV + ".sha256")) as fh:
        split_sha = fh.read().split()[0]

    meta = {
        "note": ("Every tile below is from the held-out TEST fold of the "
                 "committed split; no model in this repository was trained on "
                 "any of them."),
        "split_csv": SPLIT_CSV,
        "split_sha256": split_sha,
        "fold": REQUIRED_FOLD,
        "per_class": args.per_class,
        "selection_seed": args.seed,
        "tiles": sorted(index, key=lambda t: t["file"]),
    }
    with open(os.path.join(OUT_DIR, "index.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"wrote {len(index)} tiles to {OUT_DIR}/ (fold={REQUIRED_FOLD})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
