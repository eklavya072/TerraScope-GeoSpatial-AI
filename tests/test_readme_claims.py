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
