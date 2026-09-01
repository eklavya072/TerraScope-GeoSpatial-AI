"""Confidence intervals, multiple-comparison correction and frontier logic."""

import math

from bench.stats import compare, holm_bonferroni, mean_ci, pareto_frontier


def test_mean_ci_uses_t_distribution_not_normal():
    """With n=5 the t critical value is 2.776; using z (1.96) would understate
    the interval by ~30% and manufacture significance."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = mean_ci(values)
    assert ci["mean"] == 3.0
    assert ci["n"] == 5
    # sd = 1.5811, sem = 0.7071, t(0.975, df=4) = 2.7764
    assert abs(ci["half_width"] - 0.7071 * 2.7764) < 1e-3


def test_mean_ci_single_value_has_no_interval():
    ci = mean_ci([0.97])
    assert ci["mean"] == 0.97 and ci["half_width"] is None


def test_identical_populations_are_not_significant():
    a = [0.97, 0.971, 0.972, 0.9715, 0.9705]
    assert compare(a, list(a))["p"] > 0.99


def test_clearly_separated_populations_are_significant():
    res = compare([0.98] * 5, [0.90] * 5)
    assert res["p"] < 0.05 and res["diff"] > 0


def test_holm_is_more_conservative_than_uncorrected():
    """The point of correction: a p just under 0.05 among many tests must not
    survive. With 10 comparisons the smallest Holm threshold is 0.005."""
    pairs = {(f"a{i}", f"b{i}"): {"diff": 0.001, "t": 1.0, "p": 0.04,
                                  "cohens_d": 0.1, "significant_uncorrected": True}
             for i in range(10)}
    out = holm_bonferroni(pairs)
    assert all(not v["significant_holm"] for v in out.values())
    assert all(v["significant_uncorrected"] for v in out.values())


def test_holm_thresholds_increase_with_rank():
    pairs = {(f"a{i}", "b"): {"diff": 0.0, "t": 0.0, "p": 0.001 * (i + 1),
                              "cohens_d": 0.0, "significant_uncorrected": True}
             for i in range(5)}
    out = holm_bonferroni(pairs)
    thresholds = [v["holm_threshold"] for v in
                  sorted(out.values(), key=lambda v: v["p"])]
    assert thresholds == sorted(thresholds)


def test_pareto_frontier_minimises_x_and_maximises_y():
    pts = [
        {"label": "cheap_accurate", "energy": 1.0, "acc": 0.97},
        {"label": "costly_accurate", "energy": 10.0, "acc": 0.98},
        {"label": "dominated", "energy": 5.0, "acc": 0.95},
    ]
    front = pareto_frontier(pts, "energy", "acc")
    assert "cheap_accurate" in front and "costly_accurate" in front
    assert "dominated" not in front


def test_pareto_ignores_points_with_missing_values():
    pts = [{"label": "ok", "energy": 1.0, "acc": 0.9},
           {"label": "no_energy", "energy": None, "acc": 0.99}]
    assert pareto_frontier(pts, "energy", "acc") == ["ok"]
