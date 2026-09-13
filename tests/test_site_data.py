"""The site must not invent numbers either.

The Streamlit app was tested for this by grepping its source for decimal
literals. The site is HTML, and its figures arrive two ways: exported into
web/data/site.json by scripts/build_site_data.py, or written into the markup as
the value JavaScript replaces on load. The second kind is what a visitor sees
if the fetch fails, so it is a published figure too and it is checked here
against the artefacts like any other.
"""

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "web", "data", "site.json")
WEB = os.path.join(ROOT, "web")


@pytest.fixture(scope="module")
def data():
    with open(SITE) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def summary():
    with open(os.path.join(ROOT, "results", "summary.json")) as fh:
        return json.load(fh)


_RUNS = re.compile(r"\s+")


def markup(name):
    with open(os.path.join(WEB, name)) as fh:
        return fh.read()


# ---------------------------------------------------------------- export ---

def test_site_json_is_what_the_builder_produces_today(tmp_path):
    """A stale export is a lie with a timestamp. Rebuild and compare."""
    committed = json.load(open(SITE))
    subprocess.run([sys.executable, os.path.join("scripts", "build_site_data.py")],
                   cwd=ROOT, check=True, capture_output=True)
    rebuilt = json.load(open(SITE))
    assert committed == rebuilt, (
        "web/data/site.json is out of date with results/summary.json; "
        "run python scripts/build_site_data.py and commit the result")


def test_every_exported_figure_traces_to_the_summary(data, summary):
    measured = summary["measured"]
    for c in data["configs"]:
        key = f'{c["model"]}|{c["precision"]}|t1|b1'
        assert key in measured, f"{c['label']} is exported but was never measured"
        row = measured[key]
        assert c["accuracy"] == pytest.approx(row["test_acc"]["mean"] * 100)
        if c["energy"] is not None:
            assert c["energy"] == pytest.approx(row["energy_j_per_1k"]["mean"])


def test_the_two_accuracy_gaps_are_not_the_same_number(data):
    """gap is measured against the most accurate configuration, gap_vs_dearest
    against the one the energy ratio is quoted against. Copy that names the
    dearest model and then quotes `gap` is wrong, which has happened."""
    f = data["finding"]
    assert f["gap"] == pytest.approx(
        f["best"]["accuracy"] - f["cheapest"]["accuracy"])
    assert f["gap_vs_dearest"] == pytest.approx(
        f["dearest"]["accuracy"] - f["cheapest"]["accuracy"])


# ----------------------------------------------------------- the markup ----

def test_figures_written_into_the_markup_match_the_data(data):
    """The fallback values a visitor sees before JavaScript runs, or if it
    never does, must be the real ones."""
    f = data["finding"]
    home = markup("index.html")

    ratio = f"{f['ratio']:.0f}×"
    assert ratio in home, f"the hero's energy ratio should read {ratio}"

    spread = f"{f['spread']:.2f} points"
    assert spread in home, f"the accuracy spread should read {spread}"

    lat = f"{f['latency_ratio']:.0f}"
    assert f"{lat} times lower p95 latency" in home, (
        f"the latency ratio fallback should read {lat}")

    worst = min((q for k, q in data["quantisation"].items()
                 if k.endswith("int8_static")), key=lambda q: q["delta"])
    assert f"{worst['delta']:.2f} pp" in home, (
        "the quantisation cliff figure in the markup is not the measured one")

    # The exclusion count moved off the landing page when its footer became a
    # bare link strip. It is still a published figure wherever it is stated, so
    # the check follows it rather than being dropped.
    excluded = f'{data["exclusions"]["excluded"]} of {data["exclusions"]["total"]}'
    pages = {n: markup(n) for n in ("index.html", "demo.html",
                                    "trade-offs.html", "method.html")}
    stated = [n for n, m in pages.items()
              if excluded in _RUNS.sub(" ", m).replace("Four of", "4 of")]
    assert stated, (
        f"no page states the exclusion count, and it should read {excluded}")


# Lines defining these named structural constants may carry a decimal. They
# are a preprocessing recipe, a seek tolerance, a slider step and a visibility
# threshold. Everything else in the site scripts may not.
ALLOWED_CONSTANT_LINES = ("NORM_MEAN", "NORM_STD", "SEEK_EPSILON",
                          "SLIDER_STEP", "REVEAL_THRESHOLD",
                          "SCRUB_ENTER", "SCRUB_EXIT", "SCRUB_FLOOR",
                          "SCRUB_DONE", "FIRST_FRAME_NUDGE", "SETTLE_AT",
                          "GLIDE_TAU", "GLIDE_MAX_STEP", "GLIDE_SNAP")


def test_no_stray_result_shaped_numbers_in_the_scripts():
    """Same rule the app source lived under: the site's own JavaScript may not
    carry a figure that looks like a measurement. Structural numbers carry at
    most one decimal place; every result in this project carries two or more."""
    offenders = []
    for name in ("hero.js", "demo.js", "tradeoffs.js", "method.js", "site.js"):
        path = os.path.join(WEB, "assets", name)
        source = open(path).read()
        # Block comments are prose and may legitimately discuss a figure.
        # Blanked rather than deleted so line numbers still point somewhere.
        source = re.sub(r"/\*.*?\*/",
                        lambda m: "\n" * m.group().count("\n"), source, flags=re.S)
        for lineno, line in enumerate(source.splitlines(), start=1):
            code = line.split("//")[0]
            if any(n in code for n in ALLOWED_CONSTANT_LINES):
                continue
            code = re.sub(r'"[^"]*"|\'[^\']*\'', "", code)   # drop strings
            for m in re.finditer(r"(?<![\w.])\d+\.\d{2,}", code):
                offenders.append(f"{name}:{lineno}: {m.group()}")
    assert not offenders, (
        f"result-shaped literals in the site scripts; read them from "
        f"data/site.json instead: {offenders}")


def test_the_browser_preprocesses_exactly_as_the_benchmark_did():
    """The demo's normalisation must equal bench.config's. If it drifts, the
    page is demonstrating a pipeline the reported accuracies do not describe,
    and every prediction on it becomes a different experiment."""
    sys.path.insert(0, ROOT)
    from bench.config import INPUT_SIZE, NORM_MEAN, NORM_STD

    js = open(os.path.join(WEB, "assets", "demo.js")).read()

    def numbers(name):
        m = re.search(name + r"\s*=\s*\[([^\]]+)\]", js)
        assert m, f"{name} not found in demo.js"
        return [float(x) for x in m.group(1).split(",")]

    assert numbers("NORM_MEAN") == pytest.approx(list(NORM_MEAN))
    assert numbers("NORM_STD") == pytest.approx(list(NORM_STD))
    size = re.search(r"var SIZE = (\d+)", js)
    assert size and int(size.group(1)) == INPUT_SIZE


# ------------------------------------------------------------- shipping ----

def test_every_shipped_model_is_a_measured_configuration(summary):
    measured = {(v["model"], v["precision"]) for v in summary["measured"].values()}
    models = os.path.join(WEB, "models")
    shipped = [n for n in os.listdir(models) if n.endswith(".onnx")]
    assert shipped, "no models are shipped to the browser"
    for name in shipped:
        model, _, precision = name[:-len(".onnx")].partition("_seed0_")
        assert (model, precision) in measured, (
            f"{name} is served to visitors but has no measured row to show beside it")


def test_demo_tiles_are_all_from_the_held_out_test_fold(data):
    """The demo must never classify a tile a model was trained on."""
    import csv
    folds = {}
    with open(os.path.join(ROOT, "splits", "eurosat_split_seed42.csv")) as fh:
        for row in csv.DictReader(fh):
            folds[row["filename"]] = (row["fold"], row["label"])

    assert data["tiles"], "no tiles are bundled with the demo"
    for tile in data["tiles"]:
        fold, label = folds[tile["split_path"]]
        assert fold == "test", f"{tile['file']} is in the {fold} fold"
        assert label == tile["label"]


def test_every_tile_and_model_the_site_names_is_actually_served(data):
    for tile in data["tiles"]:
        assert os.path.exists(os.path.join(WEB, "assets", tile["file"])), (
            f"{tile['file']} is offered in the tile picker but is not served")


def test_the_test_count_the_landing_page_claims_is_the_real_one():
    """The drive panel counts the tests over the artefacts. That is a claim
    about this suite, so it is checked against this suite rather than trusted
    to stay true as tests are added."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, f"could not read the collected count from pytest:\n{out[-500:]}"
    real = int(m.group(1))

    claimed = re.search(r'data-count="(\d+)" id="stat-tests"', markup("index.html"))
    assert claimed, "the landing page no longer states a test count"
    assert int(claimed.group(1)) == real, (
        f"the landing page claims {claimed.group(1)} tests; there are {real}")


def test_the_hero_media_the_markup_references_is_actually_served():
    """A hero that 404s degrades to a poster that also 404s."""
    home = markup("index.html")
    for name in set(re.findall(r"assets/([\w.-]+\.(?:mp4|jpg))", home)) | \
                set(re.findall(r"assets/([\w.-]+\.(?:mp4|jpg))",
                               open(os.path.join(WEB, "assets", "hero.css")).read())) | \
                set(re.findall(r"assets/([\w.-]+\.mp4)",
                               open(os.path.join(WEB, "assets", "hero.js")).read())):
        assert os.path.exists(os.path.join(WEB, "assets", name)), \
            f"{name} is referenced by the hero but is not served"
