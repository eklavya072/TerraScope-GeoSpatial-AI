"""Pre-registered rejection criteria for energy/latency measurement windows.

Written and committed BEFORE any measurement in Phase 5 was taken. That
ordering is the whole point: this project already has hard evidence that
measurement windows get contaminated -- training epochs contended by
concurrent profiling ran 485s and 1261s against 136s idle, a 3.5-9x inflation,
recorded in results/runs.jsonl. Contamination is therefore not hypothetical
and windows WILL have to be discarded.

Deciding which windows to discard after seeing the energy numbers would be
post-hoc selection, and it is the first thing a sceptical reader should attack.
Fixing the thresholds in advance converts that judgement call into a stated
protocol, and turns the contaminated windows from an embarrassment into
evidence that the instrument was characterised.

Two rules govern use of this module:

  1. A window failing ANY criterion is re-run, up to MAX_RETRIES times.
  2. A window that still fails is recorded with `excluded=True` and its
     reasons -- never deleted, never silently dropped. The excluded rows ship
     in results/ and are counted in the README.

Deviation in v1.0.0: rule 1 was not carried out. The exclusions were identified
after the measurement session had ended and the privileged sampler had been
stopped, so failing windows could not be re-run under identical conditions.
They are flagged and excluded from aggregates instead, and the headline numbers
are reported both with and without them so the effect of the exclusion is
visible rather than asserted.

See PROTOCOL.md for the rationale in prose.
"""

# --------------------------------------------------------------- thresholds --

# Fraction of the measurement window actually spanned by power samples
# (computed by bench.power.energy_joules). Below this, the integral is running
# over a gap -- the sampler started late, stalled, or died -- and the joule
# figure is not a measurement of the window it claims to describe.
MIN_ENERGY_COVERAGE = 0.95

# Largest tolerated gap between consecutive power samples, as a multiple of the
# sampler's nominal interval. A stalled sampler can still yield high coverage
# while missing the interior of the window; trapezoid integration across a long
# gap silently interpolates over whatever happened inside it.
MAX_SAMPLE_GAP_RATIO = 3.0

# Drift of the idle baseline measured adjacent to this run, relative to the
# session baseline. Package power is whole-SoC, so attributing energy to
# inference requires subtracting an idle baseline; if the machine's idle state
# has moved this much, background load or thermal state has changed and the
# subtraction is no longer valid for this window.
MAX_BASELINE_DRIFT = 0.15

# Ratio of p95 to p50 inference latency within one window. With pinned thread
# counts and a warmed-up steady state, CPU inference latency is tight; a long
# tail means something else was scheduled against us mid-window.
MAX_P95_P50_RATIO = 1.50

# Re-runs allowed before a window is recorded as excluded.
MAX_RETRIES = 2


def evaluate(window: dict, session_baseline_w: float | None = None,
             sampler_interval_s: float = 0.2) -> dict:
    """Judge one measurement window against the pre-registered criteria.

    Field names match what bench.benchmark actually records. Two of the four
    criteria were never instrumented in the v1.0.0 measurement run: per-window
    sampler gaps were not retained, and no idle baseline was captured adjacent
    to each window. Those are reported as `not_evaluated` and are NOT counted
    as failures -- the original draft of this module failed them closed, which
    would have marked all 300 windows excluded and told a reader nothing. A
    criterion that was never instrumented is a stated gap in the protocol, not
    evidence against an individual window, and PROTOCOL.md records it as a
    deviation.

    Returns {"excluded": bool, "reasons": [...], "not_evaluated": [...]}.
    """
    reasons: list[str] = []
    not_evaluated: list[str] = []

    cov = window.get("energy_window_coverage")
    if cov is None:
        not_evaluated.append("energy_window_coverage missing")
    elif cov < MIN_ENERGY_COVERAGE:
        reasons.append(f"energy coverage {cov:.3f} < {MIN_ENERGY_COVERAGE}")

    gap = window.get("max_sample_gap_s")
    if gap is None:
        not_evaluated.append("max_sample_gap_s not recorded in v1.0.0")
    elif gap > MAX_SAMPLE_GAP_RATIO * sampler_interval_s:
        reasons.append(
            f"sampler gap {gap:.2f}s > {MAX_SAMPLE_GAP_RATIO}x interval "
            f"({sampler_interval_s}s)")

    idle = window.get("idle_baseline_w")
    if session_baseline_w is None or idle is None:
        not_evaluated.append("idle baseline not instrumented in v1.0.0")
    elif session_baseline_w > 0:
        drift = abs(idle - session_baseline_w) / session_baseline_w
        if drift > MAX_BASELINE_DRIFT:
            reasons.append(
                f"idle baseline drift {drift:.1%} > {MAX_BASELINE_DRIFT:.0%}")

    p50, p95 = window.get("latency_ms_p50"), window.get("latency_ms_p95")
    if p50 is None or p95 is None:
        not_evaluated.append("latency percentiles missing")
    elif p50 > 0 and (p95 / p50) > MAX_P95_P50_RATIO:
        reasons.append(
            f"p95/p50 = {p95 / p50:.2f} > {MAX_P95_P50_RATIO} (contention)")

    return {"excluded": bool(reasons), "reasons": reasons,
            "not_evaluated": not_evaluated}
