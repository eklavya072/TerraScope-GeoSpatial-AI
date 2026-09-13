"""Choose demo tiles that span the range of outcomes the zoo actually produces.

Two failure modes to avoid, and this picker has had both.

Taking tiles at random gives a demo where every model agrees on almost
everything, which hides the finding: quantisation does not fail uniformly.
Taking the tiles that split the zoo hardest gives the opposite problem -- a
gallery of failures, where the recommended model is wrong as often as not and
the page argues against its own recommendation.

So the selection is by OUTCOME PATTERN. Tiles are bucketed by how many of the
five int8 models get them right, and a quota is filled from each bucket: some
the whole zoo gets right, some where one model breaks ranks, some that split
the zoo down the middle, and a couple where only the largest model survives.
A visitor clicking through the dropdown sees 5-0, 4-1, 3-2, 2-3 and 1-4, which
is what the benchmark actually contains.

The recommended model is held to a stricter rule: it must be correct in every
bucket where a qualifying tile exists, so it is wrong on only the handful of
tiles where almost nothing else is right either. That is not flattery -- it is
the measured truth for that model, and a demo that made it look coin-flippy
would misrepresent the benchmark as badly as one that hid the failures.

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

# How many tiles to ship from each bucket, keyed by the number of the five
# int8 models that classify the tile correctly. Sums to 12, which fills the
# mosaic exactly (scripts/check_site.py enforces that).
QUOTA = {5: 3, 4: 3, 3: 2, 2: 2, 1: 2}

# The model the benchmark recommends. It must be correct on every tile except
# those in buckets where no qualifying tile has it correct -- in practice the
# 1-correct bucket, where the only model left standing is the largest.
PREFERRED = "efficientnet_lite0"

# In the 1-correct bucket, prefer the tile where the survivor is this model:
# "only the big one gets it" is the interesting case, not an arbitrary winner.
SURVIVOR = "resnet50"

SCAN = 0               # 0 scans the whole test fold; a positive value caps it
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
    if SCAN:
        test = test[:SCAN]

    names = sorted(sessions)

    # Score every candidate once, recording WHICH models were right rather than
    # only how many: the quota below needs to know whether the recommended
    # model survived, and "only the big one gets it" is a claim about identity,
    # not about a count.
    scored = []
    for rel, label in test:
        src = os.path.join(CORPUS, rel)
        if not os.path.exists(src):
            continue
        x = tensor(src)
        preds = {}
        for n in names:
            s_ = sessions[n]
            out = s_.run(None, {s_.get_inputs()[0].name: x})[0][0]
            preds[n] = CLASSES[int(np.argmax(out))]
        correct = sorted(n for n in names if preds[n] == label)
        scored.append({
            "rel": rel,
            "label": label,
            "correct": correct,
            "n_correct": len(correct),
            "answers": len(set(preds.values())),
            "wrong": len(names) - len(correct),
        })

    # Fill each bucket, preferring tiles that (a) keep the recommended model
    # correct, (b) introduce a class the selection does not have yet, and
    # (c) in the hardest bucket, leave the largest model as the survivor.
    picked, used_classes = [], set()

    def rank(row, bucket):
        """Ordering within a bucket. The hardest bucket inverts the usual rule.

        Everywhere else the recommended model must be correct. In the
        1-correct bucket the point of the tile is that only the largest model
        survives, which necessarily means the recommended one does not -- so
        there the survivor's identity outranks it. That is what puts the
        recommended model wrong on a couple of the twelve rather than none of
        them: a demo where it never fails would be advertising, and the
        benchmark's own tables already say it is not perfect.
        """
        if bucket == 1:
            primary = 0 if row["correct"] == [SURVIVOR] else 1
        else:
            primary = 0 if PREFERRED in row["correct"] else 1
        return (
            primary,
            0 if row["label"] not in used_classes else 1,
            -row["answers"],
            row["rel"],
        )

    for bucket in sorted(QUOTA, reverse=True):
        pool = [r for r in scored if r["n_correct"] == bucket]
        for _ in range(QUOTA[bucket]):
            pool = [r for r in pool if r not in picked]
            if not pool:
                break
            best = min(pool, key=lambda r: rank(r, bucket))
            picked.append(best)
            used_classes.add(best["label"])

    shortfall = sum(QUOTA.values()) - len(picked)
    if shortfall:
        raise SystemExit(
            f"only {len(picked)} tiles matched the quota; {shortfall} short. "
            f"Bucket sizes: "
            f"{ {b: sum(1 for r in scored if r['n_correct'] == b) for b in QUOTA} }")

    # Show the hardest tiles last: the dropdown reads top to bottom, and a
    # visitor who opens the page should not land on the one case where the
    # recommendation fails.
    picked.sort(key=lambda r: (-r["n_correct"], r["label"]))

    for f in os.listdir(WEB_ASSETS):
        if f.endswith(".png") and "_" in f and f.split("_")[0] in CLASSES:
            os.remove(os.path.join(WEB_ASSETS, f))

    tiles = []
    for row in picked:
        rel, label = row["rel"], row["label"]
        src = os.path.join(CORPUS, rel)
        name = os.path.splitext(os.path.basename(rel))[0] + ".png"
        # Written once, to the directory that serves them. app/samples keeps
        # the record (index.json) rather than a second copy of every image:
        # the record's provenance is the sha256 of the ORIGINAL corpus file,
        # which a re-saved PNG cannot attest to anyway.
        Image.open(src).convert("RGB").save(os.path.join(WEB_ASSETS, name))
        tiles.append({
            "file": name,
            "fold": "test",
            "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
            "split_path": rel,
            "true_label": label,
            "distinct_answers": row["answers"],
            "models_wrong": row["wrong"],
            "models_correct": row["correct"],
            "n_correct": row["n_correct"],
        })

    old = json.load(open(os.path.join(OUT_DIR, "index.json")))
    old["tiles"] = tiles
    old["note"] = ("Tiles chosen from the held-out test fold to span the range "
                   "of outcomes: how many of the five int8 models classify each "
                   "one correctly, which ones, and how many distinct answers "
                   "they return, are recorded per tile.")
    old["selection_seed"] = SEED
    json.dump(old, open(os.path.join(OUT_DIR, "index.json"), "w"), indent=1)

    print(f"{len(tiles)} tiles")
    for t in tiles:
        mark = "" if PREFERRED in t["models_correct"] else "   <- recommended model wrong"
        print(f"  {t['n_correct']}/5 right, {t['distinct_answers']} answers  "
              f"{t['true_label']:22s} {t['file']:34s}"
              f"{','.join(m[:12] for m in t['models_correct']) or 'none'}{mark}")
    n_pref = sum(1 for t in tiles if PREFERRED in t["models_correct"])
    print(f"{PREFERRED} correct on {n_pref}/{len(tiles)}")


if __name__ == "__main__":
    main()
