"""The demo app: it must run, and it must not invent numbers.

The app's whole justification is that it advertises a benchmark whose numbers
are traceable. A demo that displays a plausible-looking figure it made up would
discredit the thing it exists to show, so these tests check provenance as well
as function.
"""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "app"))

APP = "app_demo.py"
MODEL_DIR = os.path.join("app", "models_onnx")
SAMPLES = os.path.join("app", "samples")


# ------------------------------------------------------------- provenance ---

def test_no_result_shaped_literals_in_app_source():
    """Grep the app for float literals that look like results.

    Accuracies, energies and latencies must be read from summary.json or
    measured live. Structural numbers (widths, indices, run counts) are fine, so
    the check targets decimals with two or more fractional digits, which is the
    shape every result in this project takes.
    """
    # Lines defining these named structural constants may carry a decimal.
    # Everything else in the app may not.
    ALLOWED_CONSTANT_LINES = ("NEAR_EQUAL_ACCURACY_TOLERANCE",
                              "ACCURACY_SLIDER_STEP", "LATENCY_SLIDER_STEP")
    offenders = []
    for path in (APP, os.path.join("app", "demo_data.py")):
        with open(path) as fh:
            for lineno, line in enumerate(fh, start=1):
                code = line.split("#")[0]
                if any(name in code for name in ALLOWED_CONSTANT_LINES):
                    continue
                code = re.sub(r'"[^"]*"|\'[^\']*\'', "", code)   # drop strings
                # Two or more fractional digits is the shape every result in
                # this project takes (97.45, 3.18, 0.67). Unit conversions and
                # opacities (255.0, 1000.0, 0.9) carry one and are not results.
                for match in re.finditer(r"(?<![\w.])\d+\.\d{2,}", code):
                    offenders.append(f"{path}:{lineno}: {match.group()}")
    assert not offenders, (
        "result-shaped float literals in app source; read them from "
        f"summary.json instead: {offenders}")


def test_every_shipped_model_is_a_measured_configuration():
    summary = json.load(open(os.path.join("results", "summary.json")))
    measured = {(v["model"], v["precision"]) for v in summary["measured"].values()}
    for name in os.listdir(MODEL_DIR):
        if not name.endswith(".onnx"):
            continue
        model, _, precision = name[:-len(".onnx")].partition("_seed0_")
        assert (model, precision) in measured, (
            f"{name} is shipped but has no measured row to display")


def test_sample_tiles_are_all_from_the_test_fold():
    """The demo must never classify a tile a model was trained on."""
    import csv
    index = json.load(open(os.path.join(SAMPLES, "index.json")))
    assert index["fold"] == "test"

    folds = {}
    with open(os.path.join("splits", "eurosat_split_seed42.csv")) as fh:
        for row in csv.DictReader(fh):
            folds[row["filename"]] = (row["fold"], row["label"])

    assert index["tiles"], "no demo tiles bundled"
    for tile in index["tiles"]:
        fold, label = folds[tile["split_path"]]
        assert fold == "test", f"{tile['file']} is in the {fold} fold"
        assert label == tile["true_label"]


def test_sample_index_references_the_committed_split_hash():
    index = json.load(open(os.path.join(SAMPLES, "index.json")))
    expected = open(os.path.join(
        "splits", "eurosat_split_seed42.csv.sha256")).read().split()[0]
    assert index["split_sha256"] == expected


def test_app_frontier_matches_the_benchmark_implementation():
    """demo_data duplicates pareto_frontier to avoid a scipy dependency in the
    deployed app. The duplicate must not drift from the original."""
    import demo_data as dd
    from bench.stats import pareto_frontier as bench_frontier

    summary = dd.load_summary()
    rows = [{"config": f"{r['model']} {r['precision']}",
             "energy": r["energy_j_per_1k"]["mean"],
             "accuracy": r["test_acc"]["mean"] * 100}
            for r in summary["measured"].values()
            if r["threads"] == dd.REPORT_THREADS
            and r["batch_size"] == dd.REPORT_BATCH and r.get("energy_j_per_1k")]

    mine = sorted(dd.pareto_frontier(rows))
    theirs = sorted(bench_frontier(
        [{"label": r["config"], "energy": r["energy"], "acc": r["accuracy"]}
         for r in rows], "energy", "acc"))
    assert mine == theirs


def test_app_requirements_exclude_heavy_frameworks():
    # Comments explain why these are absent, so compare against code lines only.
    reqs = "\n".join(line.split("#")[0]
                     for line in open("requirements.txt")).lower()
    for banned in ("tensorflow", "torch", "keras", "scipy"):
        assert banned not in reqs, f"{banned} must not be an app dependency"
    for pinned in ("streamlit==", "onnxruntime==", "numpy=="):
        assert pinned in reqs, f"{pinned} must be pinned"


# ------------------------------------------------------------- behaviour ----

def test_preprocessing_matches_the_benchmark_pipeline():
    import numpy as np
    import demo_data as dd
    from PIL import Image
    from bench.config import INPUT_SIZE, NORM_MEAN, NORM_STD

    img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
    tensor = dd.preprocess(img)
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32
    expected = (1.0 - NORM_MEAN[0]) / NORM_STD[0]
    assert abs(float(tensor[0, 0, 0, 0]) - expected) < 1e-5


def test_live_classification_runs_and_reports_its_own_timing():
    pytest.importorskip("onnxruntime")
    import demo_data as dd
    from PIL import Image

    index = json.load(open(os.path.join(SAMPLES, "index.json")))
    tile = index["tiles"][0]
    img = Image.open(os.path.join(SAMPLES, tile["file"]))
    pred = dd.classify_and_time("efficientnet_lite0", "int8_static",
                                dd.preprocess(img))

    from bench.config import CLASSES
    assert pred.label in CLASSES
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.latency_ms_median > 0
    assert pred.timed_runs >= dd.TIMED_RUNS and pred.warmup_runs >= dd.WARMUP_RUNS


def test_unmeasured_configuration_returns_none_not_a_guess():
    import demo_data as dd
    summary = dd.load_summary()
    assert dd.measured(summary, "no_such_model", "fp32") is None
    assert dd.accuracy_ci(summary, "no_such_model", "fp32") is None
    assert dd.daily_co2e_grams(summary, "no_such_model", "fp32", 1_000_000) is None


def test_app_renders_all_four_screens_without_error():
    """Runs the real Streamlit script headlessly. st.tabs evaluates every tab
    body, so this exercises all four screens in one pass."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, f"app raised: {at.exception}"

    # Provenance is rendered through st.code, the honesty line through
    # st.warning, and the rest through markdown/captions -- gather them all.
    parts = []
    for group in (at.markdown, at.caption, at.warning, at.code, at.subheader,
                  at.success, at.error, at.info):
        parts.extend(str(element.value) for element in group)
    text = " ".join(parts)

    for expected in ("measured live on this server",   # provenance note
                     "held-out test fold",             # Classify guarantee
                     "Paired per-seed change",         # Quantisation deltas
                     "Pareto frontier",                # Deploy
                     "split sha256",                   # Method provenance
                     "excluded"):                      # Method exclusions
        assert expected in text, f"missing from rendered app: {expected!r}"


def test_provenance_note_is_present_and_not_dismissible():
    """The note sits in the page body above the tabs, so it is visible on every
    screen and cannot be dismissed. It must name both what is measured here and
    what is looked up, or a reader could mistake one for the other."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "measured live on this server" in body
    assert "looked up from the benchmark" in body
    assert "Apple M2" in body
    # Rendered as page copy, not a toast or an st.warning banner.
    assert not at.warning, "the note must not be a dismissible-looking banner"


def test_app_uses_no_emoji():
    """Emoji read as a tutorial project. The finding should carry the page."""
    for path in (APP, os.path.join("app", "demo_data.py")):
        text = open(path, encoding="utf-8").read()
        found = [c for c in text
                 if ord(c) > 0x2190 and c not in "\u2014\u00b7\u00d7\u2265\u00b1\u2264"]
        assert not found, f"{path} contains emoji/pictographs: {set(found)}"


def test_theme_and_stylesheet_are_present():
    """The palette is carried by config.toml plus one stylesheet; CSS lives in a
    file so the app source stays free of decimal literals."""
    config = open(os.path.join(".streamlit", "config.toml")).read()
    assert "primaryColor" in config and "backgroundColor" in config
    css = open(os.path.join("app", "style.css")).read()
    assert "--primary" in css and "Inter" in css
    # The app may wrap the stylesheet in a <style> tag; what it must not do is
    # carry CSS rules inline, which is how decimal literals creep back in.
    source = open(APP).read()
    assert "style.css" in source, "the stylesheet must be loaded from file"
    assert "font-family:" not in source and "border-radius:" not in source, \
        "CSS rules belong in app/style.css, not inline in the app source"
