"""Shared fixtures. Tests here run against COMMITTED artefacts only.

None of these tests need the 27,000-image corpus, which is fetched at prepare
time and never committed. That is deliberate: CI must be able to verify the
integrity of what ships in the repository without a multi-gigabyte download,
otherwise the checks get skipped in practice and stop protecting anything.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = "results"


def _load_jsonl(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="session")
def bench_rows():
    return _load_jsonl("bench.jsonl")


@pytest.fixture(scope="session")
def train_rows():
    return _load_jsonl("runs.jsonl")


@pytest.fixture(scope="session")
def summary():
    path = os.path.join(RESULTS, "summary.json")
    if not os.path.exists(path):
        pytest.skip("results/summary.json not present")
    with open(path) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def readme():
    with open("README.md") as fh:
        return fh.read()
