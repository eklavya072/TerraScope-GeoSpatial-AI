"""Invariants the shipped results must satisfy.

These check the artefacts in results/ rather than re-running the benchmark, so
they catch a corrupted, partial or mislabelled result set in seconds.
"""

import collections

EXPECTED_MODELS = {"resnet50", "mobilenetv3_small", "mobilenetv3_large",
                   "efficientnet_lite0", "mobilevit_s"}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def test_every_model_has_five_seeds(train_rows):
    by = collections.defaultdict(set)
    for r in train_rows:
        by[r["model"]].add(r["seed"])
    assert set(by) == EXPECTED_MODELS
    for model, seeds in by.items():
        assert seeds == EXPECTED_SEEDS, f"{model} is missing seeds"


def test_all_results_reference_one_split_and_one_recipe(train_rows, bench_rows):
    """The central claim of the repository: every number is traceable to one
    committed fold definition and one training contract."""
    assert len({r["split_sha256"] for r in train_rows}) == 1
    assert len({r["recipe_hash"] for r in train_rows}) == 1
    assert len({r["split_sha256"] for r in bench_rows}) == 1
    assert ({r["split_sha256"] for r in train_rows}
            == {r["split_sha256"] for r in bench_rows})


def test_onnx_export_is_accuracy_neutral(train_rows, bench_rows):
    """Exported graphs must evaluate the same function as the PyTorch models.
    Any drift here would invalidate every latency and energy figure, because
    they would describe a different model than the accuracy column does.
    """
    torch_acc = {(r["model"], r["seed"]): r["test_acc"] for r in train_rows}
    onnx_acc = {(r["model"], r["seed"]): r["test_acc"] for r in bench_rows
                if r["precision"] == "fp32" and r["threads_intra_op"] == 1
                and r["batch_size"] == 1}
    assert len(onnx_acc) == 25
    for key, acc in onnx_acc.items():
        assert acc == torch_acc[key], f"{key}: ONNX {acc} != PyTorch {torch_acc[key]}"


def test_measurement_conditions_were_uniform(bench_rows):
    """A row measured on battery or under Low Power Mode is not comparable to
    one that was not; the whole matrix must share a regime."""
    assert {r["env"]["power_state"]["power_source"] for r in bench_rows} == {"AC"}
    modes = {v for r in bench_rows
             for v in (r["env"]["power_state"]["low_power_mode"] or {}).values()}
    assert modes == {"0"}, f"Low Power Mode was active: {modes}"
    assert len({r["env"]["onnxruntime"] for r in bench_rows}) == 1


def test_no_thermal_warning_during_measurement(bench_rows):
    for r in bench_rows:
        assert "No thermal warning" in str(r["env"]["power_state"]["thermal_pressure"])


def test_every_row_has_energy_and_adequate_coverage(bench_rows):
    for r in bench_rows:
        assert r["energy_joules_per_1k_inferences"] is not None
        assert r["energy_window_coverage"] > 0.9


def test_warmup_and_run_counts_meet_the_stated_protocol(bench_rows):
    """The protocol promises >=1,000 timed INFERENCES and a 20s target window.

    Batch-32 windows reach that with ~100 calls, so the invariant is on images,
    not calls. The 20s figure is a target rather than a floor: run counts are
    sized from a short probe of per-call cost, which can undershoot. Both facts
    are stated in PROTOCOL.md; this test pins them so the documentation cannot
    drift away from the data again.
    """
    for r in bench_rows:
        assert r["warmup_runs"] == 50
        assert r["n_timed_runs"] * r["batch_size"] >= 1000
        assert r["wall_seconds"] >= 16.5, "window too short for energy attribution"


def test_thread_counts_are_pinned_and_recorded(bench_rows):
    for r in bench_rows:
        assert r["threads_intra_op"] in (1, 4)
        assert r["threads_inter_op"] == 1
        assert r["omp_num_threads"] == "1"


def test_summary_environment_is_the_measurement_not_the_training_one(summary):
    """REGRESSION: the hardware table was built from the last TRAINING row, so
    the README reported the training session's date and 'battery', contradicting
    PROTOCOL.md's 'on AC power'."""
    assert summary["environment"]["power_state"]["power_source"] == "AC"
    assert summary["training_environment"]["timestamp_utc"] \
        != summary["environment"]["timestamp_utc"]


def test_exclusions_are_recorded_not_deleted(summary, bench_rows):
    ex = summary["exclusions"]
    assert ex["windows_total"] == len(bench_rows), \
        "excluded windows must remain in bench.jsonl"
    assert ex["windows_used"] + ex["windows_excluded"] == ex["windows_total"]


def test_multithread_confound_is_characterised(summary):
    """The 4-thread rows ship with a known core-placement confound; it must be
    declared in the machine-readable summary, not only in prose."""
    regimes = summary["thread_regime_confound"]
    assert regimes["t4"]["bimodal"] is True
    assert regimes["t1"]["bimodal"] is False, \
        "the 1-thread rows carry the headline figures and must be single-regime"


def test_regime_note_agrees_with_the_bimodal_flag(summary):
    """REGRESSION: the note was gated on the mere existence of a minority regime
    rather than on the bimodal threshold, so the 1-thread entry -- which carries
    every headline figure -- declared itself "confounded" while its own flag said
    otherwise. summary.json is the CC-BY-4.0 artefact deposited under the DOI, so
    a machine-readable contradiction there is worse than a prose typo.
    """
    for threads, regime in summary["thread_regime_confound"].items():
        confounded_note = "confounded" in regime["note"]
        assert confounded_note == regime["bimodal"], (
            f"{threads}: bimodal={regime['bimodal']} but note says "
            f"{'confounded' if confounded_note else 'clean'}")


def test_headline_thread_count_is_single_regime(summary):
    t1 = summary["thread_regime_confound"]["t1"]
    assert t1["bimodal"] is False
    assert "no core-placement confound" in t1["note"]


def test_energy_is_not_merely_latency_rescaled(bench_rows):
    """Pins the power/latency decomposition the README reports.

    If package power were constant, the energy column would carry no information
    beyond latency. It is not constant, and the README says so with numbers;
    this keeps those numbers honest.
    """
    import statistics as st
    sel = [r for r in bench_rows
           if r["threads_intra_op"] == 1 and r["batch_size"] == 1]
    powers = [r["energy_mean_power_w"] for r in sel]
    assert max(powers) / min(powers) > 2.0, "power varies more than 2x at 1 thread"

    by_prec = {}
    for prec in ("fp32", "int8_static"):
        vals = [r["energy_mean_power_w"] for r in sel if r["precision"] == prec]
        by_prec[prec] = st.mean(vals)
    assert by_prec["int8_static"] > by_prec["fp32"], (
        "int8-static draws MORE package power than fp32; if this inverts, the "
        "README's central caveat about quantisation is wrong")
