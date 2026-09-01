"""The commit hashes cited as evidence must actually resolve, in this order.

The pre-registration argument is the credibility keystone of this repository:
"don't take my word for it, check that the criteria were committed before the
measurements." That argument is only as good as the hashes it cites, and those
hashes are fragile in a specific way -- ANY history rewrite renumbers every
commit downstream of it. Rewriting author metadata across the history once
already turned the cited pre-registration commit into a dead reference, which
reads far worse to a reviewer than never having cited one.

So the citation is asserted here rather than trusted. If a rewrite invalidates
it again, CI says so instead of a reader discovering `fatal: Not a valid object
name` on the one claim the project asks to be checked.
"""

import json
import os
import re
import subprocess

import pytest

DOCS = ("README.md", "PROTOCOL.md")

# The pre-registration claim, stated as data so the test and the prose cannot
# drift apart. See PROTOCOL.md for what each of these is.
PREREGISTRATION = "bd06ff5"      # introduced bench/exclusion.py + PROTOCOL.md
FIRST_MEASUREMENTS = "1c9fa33"   # first commit carrying results/bench.jsonl


def _git(*args):
    return subprocess.run(("git",) + args, capture_output=True, text=True)


def _have_git_history():
    if not os.path.isdir(".git"):
        return False
    # actions/checkout defaults to a depth-1 clone, where no historical commit
    # is present and every assertion below would fail for the wrong reason.
    if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        return False
    return _git("rev-parse", "--verify", "HEAD").returncode == 0


needs_history = pytest.mark.skipif(
    not _have_git_history(),
    reason="no unshallowed git history available (set fetch-depth: 0 in CI)")


def _cited_hashes():
    """Backticked short hashes in the docs, minus the ones that aren't commits.

    The recipe hash is also a 12-character hex string in backticks, and the
    split digest is a 64-character one; neither is a git object.
    """
    with open(os.path.join("results", "summary.json")) as fh:
        not_a_commit = {json.load(fh)["recipe_hash"]}
    found = set()
    for doc in DOCS:
        with open(doc) as fh:
            found |= set(re.findall(r"`([0-9a-f]{7,12})`", fh.read()))
    return found - not_a_commit


@needs_history
def test_every_hash_cited_in_the_docs_resolves():
    cited = _cited_hashes()
    assert cited, "no commit hashes found in the docs -- has the citation been dropped?"
    for h in sorted(cited):
        r = _git("cat-file", "-t", h)
        assert r.returncode == 0 and r.stdout.strip() == "commit", (
            f"`{h}` is cited in {' or '.join(DOCS)} but does not resolve to a "
            f"commit. A history rewrite renumbers commits; re-cite the new hash.")


@needs_history
def test_the_preregistration_commits_are_the_ones_cited():
    for h in (PREREGISTRATION, FIRST_MEASUREMENTS):
        assert h in _cited_hashes(), f"{h} is no longer cited in the docs"


@needs_history
def test_preregistration_precedes_the_first_measurements():
    """The whole argument, as an assertion.

    --is-ancestor is the right check rather than comparing dates: commit dates
    can be forged, ancestry cannot be, and ancestry is what a reader verifies.
    """
    assert _git("merge-base", "--is-ancestor",
                PREREGISTRATION, FIRST_MEASUREMENTS).returncode == 0, (
        f"{PREREGISTRATION} is not an ancestor of {FIRST_MEASUREMENTS}; the "
        f"pre-registration claim in PROTOCOL.md no longer holds")


@needs_history
def test_exclusion_criteria_existed_at_the_preregistration_commit():
    """Citing the commit is not enough -- it must actually contain the criteria."""
    r = _git("show", f"{PREREGISTRATION}:bench/exclusion.py")
    assert r.returncode == 0, "bench/exclusion.py is absent from the cited commit"
    for threshold in ("MIN_ENERGY_COVERAGE", "MAX_P95_P50_RATIO",
                      "MAX_SAMPLE_GAP_RATIO", "MAX_BASELINE_DRIFT"):
        assert threshold in r.stdout, f"{threshold} was not pre-registered"


@needs_history
def test_no_measurement_windows_existed_at_the_preregistration_commit():
    """The claim that matters: no energy or latency window had been measured
    when the thresholds that judge them were fixed."""
    assert _git("cat-file", "-e",
                f"{PREREGISTRATION}:results/bench.jsonl").returncode != 0, (
        "results/bench.jsonl already existed at the pre-registration commit -- "
        "the criteria were NOT fixed before measurement")


@needs_history
def test_committed_thresholds_still_match_the_module_in_use():
    """A pre-registration that was quietly edited afterwards is not one."""
    from bench import exclusion
    old = _git("show", f"{PREREGISTRATION}:bench/exclusion.py").stdout
    for name in ("MIN_ENERGY_COVERAGE", "MAX_SAMPLE_GAP_RATIO",
                 "MAX_BASELINE_DRIFT", "MAX_P95_P50_RATIO", "MAX_RETRIES"):
        match = re.search(rf"^{name} = ([\d.]+)$", old, re.MULTILINE)
        assert match, f"{name} not found in the pre-registered module"
        assert float(match.group(1)) == float(getattr(exclusion, name)), (
            f"{name} has changed since pre-registration: "
            f"{match.group(1)} -> {getattr(exclusion, name)}")
