"""Energy integration. Every one of these is a regression test for a real bug.

The three defects these cover all produced plausible-looking output rather than
an error, which is why they survived until the numbers were read closely.
"""

import textwrap

from bench.power import PowerLog, energy_joules, parse_log

SAMPLE = textwrap.dedent("""\
    *** Sampled system activity (Mon Sep  1 22:40:00 2026 +0530) (200.00ms elapsed) ***

    **** Processor usage ****

    CPU Power: 5000 mW
    GPU Power: 0 mW

    *** Sampled system activity (Mon Sep  1 22:40:00 2026 +0530) (200.00ms elapsed) ***

    CPU Power: 5000 mW

    *** Sampled system activity (Mon Sep  1 22:40:01 2026 +0530) (200.00ms elapsed) ***

    CPU Power: 5000 mW
    """)


def _write(tmp_path, text):
    p = tmp_path / "power_log.txt"
    p.write_text(text)
    return str(p)


def test_parses_power_and_elapsed(tmp_path):
    samples = parse_log(_write(tmp_path, SAMPLE))
    assert len(samples) == 3
    assert all(p == 5000.0 for _, p, _ in samples)
    assert all(abs(dt - 0.2) < 1e-9 for _, _, dt in samples)


def test_energy_uses_elapsed_not_timestamp_difference(tmp_path):
    """REGRESSION: powermetrics timestamps are quantised to whole seconds while
    the sampler runs at 200ms, so samples share a timestamp. Integrating by
    timestamp difference gave a zero-width trapezoid and reported exactly 0 J.

    Three samples of 5 W covering 200 ms each must be 3 x 5 x 0.2 = 3.0 J.
    """
    samples = parse_log(_write(tmp_path, SAMPLE))
    result = energy_joules(samples, samples[0][0] - 1, samples[-1][0] + 1)
    assert abs(result["joules"] - 3.0) < 1e-9
    assert result["joules"] > 0, "zero-width integration regression"


def test_two_samples_sharing_a_timestamp_still_yield_energy(tmp_path):
    """The exact shape of the original bug: a window inside a single second."""
    samples = parse_log(_write(tmp_path, SAMPLE))
    same_second = [s for s in samples if s[0] == samples[0][0]]
    assert len(same_second) == 2
    # Both samples carry the same quantised timestamp; the window must still
    # yield 2 x 5 W x 0.2 s = 2.0 J rather than a zero-width trapezoid.
    result = energy_joules(same_second, samples[0][0] - 0.5, samples[0][0] + 0.5)
    assert result["joules"] is not None
    assert abs(result["joules"] - 2.0) < 1e-9


def test_powerlog_reads_appended_samples(tmp_path):
    """REGRESSION: the log was parsed once at startup while the sampler kept
    appending, so every measurement window fell after the last known sample and
    integrated over nothing. The whole matrix would have reported null energy.
    """
    path = _write(tmp_path, SAMPLE)
    log = PowerLog(path)
    first = len(log.samples)
    assert first == 3

    with open(path, "a") as fh:
        fh.write(SAMPLE)
    assert log.update() == 3, "appended samples were not picked up"
    assert len(log.samples) == first + 3


def test_powerlog_tolerates_a_partial_trailing_line(tmp_path):
    """The sampler may be mid-write when we read; a truncated line must be
    retried on the next update, not parsed as a fragment or dropped."""
    path = _write(tmp_path, SAMPLE)
    log = PowerLog(path)
    before = len(log.samples)

    with open(path, "a") as fh:
        fh.write("*** Sampled system activity (Mon Sep  1 22:40:02 2026 +0530) "
                 "(200.00ms elapsed) ***\n\nCPU Pow")
    log.update()
    assert len(log.samples) == before, "partial line should not yet parse"

    with open(path, "a") as fh:
        fh.write("er: 4000 mW\n")
    log.update()
    assert len(log.samples) == before + 1
    assert log.samples[-1][1] == 4000.0


def test_coverage_reports_gaps(tmp_path):
    samples = parse_log(_write(tmp_path, SAMPLE))
    # 0.6s of samples claimed over a 10s window -> coverage must be small
    result = energy_joules(samples, samples[0][0], samples[0][0] + 10)
    assert result["coverage"] < 0.1


def test_missing_log_is_not_an_error():
    assert parse_log("does/not/exist.txt") == []
    assert energy_joules([], 0, 1)["joules"] is None
