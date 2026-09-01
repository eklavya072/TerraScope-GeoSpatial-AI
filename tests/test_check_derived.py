"""The tolerant comparator must tolerate ULP noise and nothing else.

Loosening a check is how a check quietly stops working. scripts/check_derived.py
exists because scipy p-values differ in the last ULP across platforms, but if
that tolerance also swallowed a flipped significance flag or a moved accuracy
figure, the integrity job would be theatre. These tests pin the boundary from
both sides.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from check_derived import REL_TOL, compare  # noqa: E402


def test_identical_structures_compare_equal(summary):
    assert compare(summary, copy.deepcopy(summary)) == []


def test_ulp_noise_is_tolerated():
    """The exact values from the failing CI run that motivated this."""
    for mac, linux in ((0.0002170793405032607, 0.0002170793405032606),
                       (0.00013843176958077457, 0.00013843176958077454),
                       (0.24799826638585334, 0.24799826638585346),
                       (0.00014941444263864175, 0.00014941444263864173)):
        assert mac != linux, "these must be genuinely different floats"
        assert compare({"p": mac}, {"p": linux}) == []


def test_a_flipped_significance_flag_is_caught():
    """bool is a subclass of int; without explicit handling True/False could be
    compared numerically and 1 vs 0 is far outside tolerance anyway -- but the
    failure message must name it as a value change, not a float drift."""
    diffs = compare({"significant_holm": True}, {"significant_holm": False})
    assert len(diffs) == 1 and "True -> False" in diffs[0]


def test_a_real_p_value_change_is_caught():
    """Crossing a Holm threshold is the change that would matter most."""
    diffs = compare({"p": 0.0049}, {"p": 0.0051})
    assert len(diffs) == 1 and "rel_tol" in diffs[0]


def test_a_moved_accuracy_is_caught():
    """One tile out of 2,700 is 3.7e-4 -- 5 orders of magnitude above tolerance."""
    diffs = compare({"mean": 0.9812345679}, {"mean": 0.9816049383})
    assert len(diffs) == 1


def test_tolerance_boundary_behaves():
    base = 1.0
    assert compare({"x": base}, {"x": base * (1 + REL_TOL / 10)}) == []
    assert compare({"x": base}, {"x": base * (1 + REL_TOL * 10)}) != []


def test_structural_changes_are_caught():
    assert compare({"a": 1}, {"a": 1, "b": 2}) != []          # added
    assert compare({"a": 1, "b": 2}, {"a": 1}) != []          # removed
    assert compare({"a": [1, 2]}, {"a": [1, 2, 3]}) != []     # length
    assert compare({"a": "AC"}, {"a": "battery"}) != []       # string
    assert compare({"n": 5}, {"n": 4}) != []                  # counts


def test_exclusion_bookkeeping_cannot_drift_silently(summary):
    """The specific fields a reader would check first."""
    for field, changed in (("windows_total", 299), ("windows_excluded", 3)):
        mutated = copy.deepcopy(summary)
        mutated["exclusions"][field] = changed
        assert compare(summary, mutated) != [], f"{field} change went unnoticed"


def test_committed_summary_is_valid_json_and_has_the_expected_shape(summary):
    for key in ("accuracy_over_seeds", "measured", "pairwise_significance",
                "exclusions", "thread_regime_confound", "environment"):
        assert key in summary
    assert json.dumps(summary)
