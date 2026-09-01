"""Pre-registered rejection criteria (PROTOCOL.md)."""

from bench.exclusion import (MAX_P95_P50_RATIO, MIN_ENERGY_COVERAGE, evaluate)

CLEAN = {"energy_window_coverage": 0.99, "latency_ms_p50": 1.0, "latency_ms_p95": 1.1}


def test_clean_window_is_kept():
    assert evaluate(CLEAN)["excluded"] is False


def test_low_coverage_is_excluded():
    v = evaluate({**CLEAN, "energy_window_coverage": MIN_ENERGY_COVERAGE - 0.01})
    assert v["excluded"] and "coverage" in v["reasons"][0]


def test_latency_tail_is_excluded():
    v = evaluate({**CLEAN, "latency_ms_p95": MAX_P95_P50_RATIO + 0.2})
    assert v["excluded"] and "p95/p50" in v["reasons"][0]


def test_uninstrumented_criteria_do_not_exclude_a_window():
    """REGRESSION: the original module failed unevaluable criteria closed. Since
    two of the four were never instrumented, that marked all 300 windows
    excluded -- a result that tells a reader nothing. An uninstrumented
    criterion is a gap in the protocol, not evidence against a window.
    """
    v = evaluate(CLEAN)
    assert v["not_evaluated"], "uninstrumented criteria should be reported"
    assert v["excluded"] is False


def test_field_names_match_what_the_harness_records():
    """REGRESSION: the module expected latency_p95_ms while bench.benchmark
    writes latency_ms_p95, so the criterion silently never evaluated."""
    v = evaluate({"energy_window_coverage": 0.99,
                  "latency_ms_p50": 1.0, "latency_ms_p95": 2.0})
    assert v["excluded"], "p95/p50 = 2.0 must trip the contention criterion"
