"""The committed split is the reproducibility anchor. If it is wrong, nothing else matters."""

import collections
import csv
import hashlib
import os

import pytest

SPLIT = os.path.join("splits", "eurosat_split_seed42.csv")
EXPECTED_TOTAL = 27_000
EXPECTED_FOLDS = {"train": 18_900, "val": 5_400, "test": 2_700}


@pytest.fixture(scope="module")
def rows():
    with open(SPLIT) as fh:
        return list(csv.DictReader(fh))


def test_split_matches_its_committed_hash():
    """Every result row references this file by sha256; the sidecar must agree."""
    with open(SPLIT, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    expected = open(SPLIT + ".sha256").read().split()[0]
    assert digest == expected, "split file no longer matches its .sha256 sidecar"


def test_split_size_and_fold_counts(rows):
    assert len(rows) == EXPECTED_TOTAL
    counts = collections.Counter(r["fold"] for r in rows)
    assert dict(counts) == EXPECTED_FOLDS


def test_no_tile_appears_in_two_folds(rows):
    """The failure that would silently invalidate every accuracy figure."""
    names = [r["filename"] for r in rows]
    assert len(set(names)) == len(names), "a tile appears more than once"


def test_label_agrees_with_directory(rows):
    for r in rows:
        assert r["filename"].split("/")[0] == r["label"]


def test_split_is_stratified(rows):
    """Each fold must preserve the class distribution, or per-class accuracy
    comparisons across folds are measuring the split, not the model."""
    by_class = collections.defaultdict(collections.Counter)
    for r in rows:
        by_class[r["label"]][r["fold"]] += 1
    for label, folds in by_class.items():
        total = sum(folds.values())
        for fold, expected_frac in (("train", 0.70), ("val", 0.20), ("test", 0.10)):
            frac = folds[fold] / total
            assert abs(frac - expected_frac) < 0.01, (
                f"{label} fold {fold} is {frac:.3f}, expected ~{expected_frac}")


def test_metadata_matches_the_csv(rows):
    import json
    meta = json.load(open(os.path.join("splits", "eurosat_split_seed42.json")))
    assert meta["n_total"] == len(rows)
    assert meta["seed"] == 42
    with open(SPLIT, "rb") as fh:
        assert meta["split_csv_sha256"] == hashlib.sha256(fh.read()).hexdigest()
