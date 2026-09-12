"""Choose demo tiles the models actually disagree about.

A picker that takes two tiles per class at random shows a demo where every
model agrees on almost everything, which hides the finding: quantisation does
not fail uniformly, and the cheap models fail in specific, visible ways. This
selects from the held-out test fold the tiles that split the zoo -- the ones
where the five architectures return different answers -- and keeps a few
unanimous tiles for contrast so the page is not a gallery of failures.

    python scripts/pick_demo_tiles.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.config import CLASSES, INPUT_SIZE, NORM_MEAN, NORM_STD  # noqa: E402

CORPUS = os.path.join(ROOT, "data", "eurosat_rgb")
SPLIT = os.path.join(ROOT, "splits", "eurosat_split_seed42.csv")
OUT_DIR = os.path.join(ROOT, "app", "samples")
WEB_ASSETS = os.path.join(ROOT, "web", "assets")

SPLIT_TILES = 9        # how many contested tiles to ship
CLEAN_TILES = 3        # plus a few the whole zoo agrees on
SCAN = 900             # candidates to score
SEED = 0


def tensor(path):
    im = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    a = np.asarray(im, np.float32) / 255.0
    a = (a - np.array(NORM_MEAN, np.float32)) / np.array(NORM_STD, np.float32)
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def main() -> None:
    sessions = {}
    for f in sorted(os.listdir(os.path.join(ROOT, "web", "models"))):
        if f.endswith(".onnx") and "int8_static" in f:
            sessions[f.split("_seed0_")[0]] = ort.InferenceSession(
                os.path.join(ROOT, "web", "models", f),
                providers=["CPUExecutionProvider"])

    test = [(r["filename"], r["label"]) for r in csv.DictReader(open(SPLIT))
            if r["fold"] == "test"]
    rng = np.random.default_rng(SEED)
    rng.shuffle(test)

    contested, clean = [], []
    for rel, label in test[:SCAN]:
        src = os.path.join(CORPUS, rel)
        if not os.path.exists(src):
            continue
        x = tensor(src)
        preds = {n: CLASSES[int(np.argmax(s.run(None, {s.get_inputs()[0].name: x})[0][0]))]
                 for n, s in sessions.items()}
        answers = len(set(preds.values()))
        wrong = sum(1 for v in preds.values() if v != label)
        row = (answers, wrong, rel, label)
        if answers > 1 and wrong:
            contested.append(row)
        elif answers == 1 and not wrong:
            clean.append(row)
        if len(contested) >= 60 and len(clean) >= 20:
            break

    contested.sort(key=lambda r: (-r[0], -r[1]))
    # one contested tile per class where possible, so the picker is not all
    # pasture; then fill by how badly the zoo splits.
    picked, seen = [], set()
    for row in contested:
        if row[3] not in seen:
            picked.append(row); seen.add(row[3])
        if len(picked) >= SPLIT_TILES:
            break
    for row in contested:
        if len(picked) >= SPLIT_TILES:
            break
        if row not in picked:
            picked.append(row)
    picked += clean[:CLEAN_TILES]

    for f in os.listdir(WEB_ASSETS):
        if f.endswith(".png") and "_" in f and f.split("_")[0] in CLASSES:
            os.remove(os.path.join(WEB_ASSETS, f))
    for f in os.listdir(OUT_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, f))

    tiles = []
    for answers, wrong, rel, label in picked:
        src = os.path.join(CORPUS, rel)
        name = os.path.splitext(os.path.basename(rel))[0] + ".png"
        Image.open(src).convert("RGB").save(os.path.join(OUT_DIR, name))
        Image.open(src).convert("RGB").save(os.path.join(WEB_ASSETS, name))
        tiles.append({
            "file": name,
            "fold": "test",
            "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
            "split_path": rel,
            "true_label": label,
            "distinct_answers": answers,
            "models_wrong": wrong,
        })

    old = json.load(open(os.path.join(OUT_DIR, "index.json")))
    old["tiles"] = tiles
    old["note"] = ("Tiles chosen from the held-out test fold for disagreement: "
                   "the count of distinct answers the five int8 models return, "
                   "and how many of them are wrong, are recorded per tile.")
    old["selection_seed"] = SEED
    json.dump(old, open(os.path.join(OUT_DIR, "index.json"), "w"), indent=1)

    print(f"{len(tiles)} tiles")
    for t in tiles:
        print(f"  {t['distinct_answers']} answers, {t['models_wrong']}/5 wrong  "
              f"{t['true_label']:22s} {t['file']}")


if __name__ == "__main__":
    main()
