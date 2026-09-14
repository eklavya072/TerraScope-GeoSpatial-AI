"""Every headline number in the README, re-derived from summary.json.

This exists because a hand-typed README sentence once claimed MobileNetV3-Small
at 4 threads was "2.2x faster but ~47% MORE energy" when the measured values are
1.50x and 16% LESS -- wrong in magnitude AND direction. It was stale from a
smoke test taken before an energy bug was fixed, and it survived because the
generated tables were correct while the prose around them was not.

Generated tables are protected by scripts/render_readme.py. Prose is protected
by this file. A claim that appears in neither is a claim nobody is checking.
"""

import re
import statistics as st
import subprocess
import sys

import pytest


def key(model, precision):
    return f"{model}|{precision}|t1|b1"


@pytest.fixture(scope="module")
def measured(summary):
    return summary["measured"]


def _pct(x):
    return x * 100


def test_headline_accuracy_gap(readme, measured):
    r50 = measured[key("resnet50", "fp32")]["test_acc"]["mean"]
    lite = measured[key("efficientnet_lite0", "int8_static")]["test_acc"]["mean"]
    assert f"{_pct(r50) - _pct(lite):.2f}" == "0.67"
    assert "0.67 percentage points" in readme


def test_headline_energy_latency_and_disk_ratios(readme, measured):
    r50 = measured[key("resnet50", "fp32")]
    lite = measured[key("efficientnet_lite0", "int8_static")]
    energy = r50["energy_j_per_1k"]["mean"] / lite["energy_j_per_1k"]["mean"]
    latency = r50["latency_p95_ms"]["mean"] / lite["latency_p95_ms"]["mean"]
    disk = r50["onnx_mb"] / lite["onnx_mb"]
    assert round(energy) == 19 and "19× less energy" in readme
    assert round(latency) == 29 or round(latency) == 28
    assert "28× faster" in readme
    assert round(disk) == 25 and "25× less disk" in readme


def test_headline_absolute_values_appear_verbatim(readme, measured):
    r50 = measured[key("resnet50", "fp32")]
    lite = measured[key("efficientnet_lite0", "int8_static")]
    for value in (f"{_pct(r50['test_acc']['mean']):.2f}",
                  f"{_pct(lite['test_acc']['mean']):.2f}",
                  f"{r50['energy_j_per_1k']['mean']:.2f}",
                  f"{lite['energy_j_per_1k']['mean']:.2f}"):
        assert value in readme, f"{value} is not in the README"


def test_accuracy_spread_claim(readme, summary):
    accs = [v["mean"] for v in summary["accuracy_over_seeds"].values()]
    spread = _pct(max(accs)) - _pct(min(accs))
    assert f"{spread:.2f} pp spread" in readme


def test_significant_comparison_count(readme, summary):
    n = sum(1 for v in summary["pairwise_significance"].values()
            if v["significant_holm"])
    total = len(summary["pairwise_significance"])
    assert f"{n} of {total} pairwise comparisons" in readme


def test_quantisation_delta_claims(readme, summary):
    qd = summary["quantisation_delta"]
    lite = qd["efficientnet_lite0|int8_static"]
    assert f"{abs(lite['delta_mean']) * 100:.2f} ± {lite['delta_half_width'] * 100:.2f} pp" \
        in readme
    for model, expected in (("mobilenetv3_small|int8_static", "64.66"),
                            ("mobilevit_s|int8_static", "48.86")):
        assert f"{abs(qd[model]['delta_mean']) * 100:.2f}" == expected
        assert expected in readme


def test_four_thread_claims_match_the_data(readme, bench_rows):
    """The specific regression. Both directions and both batch sizes."""
    def mean_of(field, bs, threads):
        vals = [r[field] for r in bench_rows
                if r["model"] == "mobilenetv3_small" and r["precision"] == "fp32"
                and r["batch_size"] == bs and r["threads_intra_op"] == threads]
        return st.mean(vals)

    for bs, speed_txt, energy_txt in ((1, "1.50×", "16% less energy"),
                                      (32, "2.19×", "54% less energy")):
        speedup = (mean_of("latency_ms_p95", bs, 1) / mean_of("latency_ms_p95", bs, 4))
        e1 = mean_of("energy_joules_per_1k_inferences", bs, 1)
        e4 = mean_of("energy_joules_per_1k_inferences", bs, 4)
        change = (e4 / e1 - 1) * 100
        assert f"{speedup:.2f}×" == speed_txt
        assert change < 0, "4 threads used LESS energy; the README must not say more"
        assert f"{abs(change):.0f}% less energy" == energy_txt
        assert speed_txt in readme and energy_txt in readme


def test_no_stale_pre_fix_claims_survive(readme):
    for stale in ("47% *more* energy", "2.2× faster", "~47%"):
        assert stale not in readme, f"stale claim resurfaced: {stale}"


def test_exclusion_count_claim(readme, summary):
    ex = summary["exclusions"]
    assert f"{ex['windows_excluded']} of {ex['windows_total']} windows" in readme


def test_energy_is_labelled_estimated_everywhere(readme):
    """The distinction between a measurement and an estimate is the difference
    between a credible benchmark and a discredited one."""
    assert "estimated" in readme.lower()
    table = readme[readme.index("| Model | Precision"):]
    header = table[:table.index("\n")]
    assert "estimated" in header, "the results table header must say estimated"


def test_grid_intensity_assumption_is_stated(readme, summary):
    assert str(int(summary["grid_intensity_g_co2e_per_kwh"])) in readme


def test_every_readme_percentage_claim_is_plausible(readme):
    """Cheap guard against a decimal-point slip: no accuracy claim above 100%."""
    for match in re.finditer(r"(\d{2,3}\.\d{2})%", readme):
        assert float(match.group(1)) <= 100.0


# --------------------------------------------------------------------------
# The energy-vs-latency section. Its whole point is that the energy axis is
# mostly latency, stated with numbers -- so the numbers have to be on ONE basis.
# They were not: the latency row printed config-mean endpoints (0.285 -> 12.006)
# beside a per-window ratio (48.8x, which is 12.239/0.251), while the energy,
# power and correlation figures were all per-window. A table that silently mixes
# bases is the failure mode this whole section exists to argue against.
# --------------------------------------------------------------------------

def _t1b1(bench_rows):
    return [r for r in bench_rows
            if r["threads_intra_op"] == 1 and r["batch_size"] == 1]


def test_energy_latency_table_is_per_window_throughout(readme, bench_rows):
    rows = _t1b1(bench_rows)
    assert len(rows) == 75
    for field, low, high, ratio in (
            ("latency_ms_p50", "0.251", "12.239", "48.8"),
            ("energy_joules_per_1k_inferences", "1.56", "63.12", "40.5"),
            ("energy_mean_power_w", "4.86", "10.24", "2.1")):
        vals = [r[field] for r in rows]
        digits = len(low.split(".")[1])
        assert f"{min(vals):.{digits}f}" == low, f"{field} min drifted"
        assert f"{max(vals):.{digits}f}" == high, f"{field} max drifted"
        assert f"{max(vals) / min(vals):.1f}" == ratio, (
            f"{field} ratio must be computed on the SAME rows as its endpoints")
        assert f"{low} → {high}" in readme
        assert f"{ratio}×" in readme


def test_energy_latency_correlation_claim(readme, bench_rows):
    import numpy as np
    rows = _t1b1(bench_rows)
    r = np.corrcoef([x["latency_ms_p50"] for x in rows],
                    [x["energy_joules_per_1k_inferences"] for x in rows])[0, 1]
    assert f"r = {r:.3f}" in readme, f"measured r is {r:.4f}"


def test_power_by_precision_claim(readme, bench_rows):
    """The second-order term the section rests on: int8-static draws MORE power."""
    rows = _t1b1(bench_rows)
    mean_w = {p: st.mean(x["energy_mean_power_w"] for x in rows
                         if x["precision"] == p)
              for p in ("fp32", "int8_dynamic", "int8_static")}
    assert mean_w["int8_static"] > mean_w["fp32"], \
        "the section claims quantised models draw more power"
    for p in mean_w:
        assert f"{mean_w[p]:.2f} W" in readme, f"{p} power figure drifted"
    increase = (mean_w["int8_static"] / mean_w["fp32"] - 1) * 100
    assert f"{increase:.0f}%" in readme


def test_only_one_pair_reorders_between_latency_and_energy(readme, bench_rows):
    """'It reorders one pair' -- asserted, because it is the section's evidence
    that energy is not a pure restatement of latency."""
    import collections
    groups = collections.defaultdict(list)
    for r in _t1b1(bench_rows):
        groups[(r["model"], r["precision"])].append(r)
    by = {k: (st.mean(x["latency_ms_p50"] for x in v),
              st.mean(x["energy_joules_per_1k_inferences"] for x in v))
          for k, v in groups.items()}
    by_lat = [k for k in sorted(by, key=lambda k: by[k][0])]
    by_energy = [k for k in sorted(by, key=lambda k: by[k][1])]
    moved = [k for k, j in zip(by_lat, by_energy) if k != j]
    assert len(moved) == 2, f"{len(moved)} positions differ, not one swapped pair"
    assert "reorders one pair" in readme


def test_readme_test_count_matches_reality(readme):
    """The README said 58 tests while there were 82.

    The landing page has been guarded against exactly this since the site was
    built (tests/test_site_data.py asserts its stat matches a live collection),
    and its number stayed right. The README had no such guard, so the README's
    number is the one that rotted -- through 24 added tests, in three separate
    places, past every reader. An unchecked number is not a documented number.
    """
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, f"could not read the collected count from pytest:\n{out[-500:]}"
    real = int(m.group(1))

    claimed = re.findall(r"(\d+) tests", readme)
    assert claimed, "the README no longer states a test count"
    for c in claimed:
        assert int(c) == real, (
            f"the README claims {c} tests; there are {real}")
