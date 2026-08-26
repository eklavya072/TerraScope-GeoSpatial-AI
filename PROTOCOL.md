# Measurement protocol (pre-registered)

**Committed before any Phase 5 measurement was taken.** Everything in this file
fixes a decision in advance that would otherwise be made after seeing results.
The commit timestamp is the evidence for that ordering; check it against the
first row in `results/bench.jsonl`.

## Why this exists

This project has hard evidence that measurement windows get contaminated.
Training epochs contended by concurrent profiling ran **485 s and 1261 s
against 136 s when idle** — a 3.5–9× inflation, recorded in
`results/runs.jsonl`. Contamination is therefore a certainty, not a risk, and
some measurement windows will have to be discarded.

Choosing *which* windows to discard after seeing the energy numbers would be
post-hoc selection. Pre-registering the thresholds converts that judgement into
a stated protocol, and turns the contaminated windows from a caveat into
evidence that the instrument was characterised before it was trusted.

## Rejection criteria

A window is rejected if **any** of the following holds. Thresholds live in
`bench/exclusion.py`; this file and that module must agree.

| Criterion | Threshold | What it catches |
|---|---|---|
| Energy window coverage | `< 0.95` | Sampler started late, stalled or died; the integral spans a gap |
| Max inter-sample gap | `> 3 ×` nominal interval | Sampler stalled mid-window; trapezoid integration interpolates over the stall |
| Idle baseline drift | `> 15%` vs session baseline | Background load or thermal state moved; baseline subtraction no longer valid |
| p95 / p50 latency ratio | `> 1.50` | Something was scheduled against the benchmark mid-window |

A criterion that **cannot be evaluated** (a missing field) counts as a failure,
not a pass. An unverifiable measurement is not a measurement.

## Actions

1. A failing window is **re-run**, up to **2 retries**.
2. A window still failing after retries is recorded with `excluded=True` and its
   reasons. **It is never deleted and never silently dropped.**
3. Excluded rows ship in `results/`, and the README reports how many windows
   were excluded and why. A benchmark that never reports an exclusion is either
   lucky or not looking.

## Session conditions

- One long-lived `powermetrics` sampler for the entire matrix
  (`scripts/energy_sampler.sh`), started once under `sudo`. Per-run invocation
  would put process-startup cost inside each window — a bias that scales with
  window length, and therefore correlates with the variable being measured.
- Nominal sampling interval **500 ms**, recorded alongside every result.
- `caffeinate` for the session; a sleep mid-matrix loses the trace.
- Idle protocol: **≥60 s** idle before the matrix, **≥60 s** after, **≥30 s**
  enforced between runs. Session baseline is taken from the leading idle period.
- Both **gross** and **baseline-subtracted** energy are reported. The headline
  figure is baseline-subtracted, and the table header says so.
- The raw sampler trace is archived in `results/` and in the Zenodo deposit. It
  is the evidence behind every energy figure.

## Energy claim wording

Energy is **CPU package power sampled via `powermetrics`, baseline-subtracted**
— on-die telemetry, not wall-plug metering, and package-level rather than
per-process. It is a real sensor reading, which a TDP-model estimate is not, but
it is not a socket measurement and must never be described as one.

Where `codecarbon` is run alongside, both columns are reported. Agreement
between two independent methods is a credibility argument; disagreement is a
finding, and goes in limitations rather than being reconciled away.

## Statistical protocol

- **≥5 seeds** per model; every accuracy figure reported as mean ± 95% Student-t
  confidence interval (`bench/stats.py: mean_ci`).
- Pairwise comparison by **Welch's t-test**, corrected across all *k(k−1)/2*
  pairs with **Holm–Bonferroni** (`bench/stats.py: holm_bonferroni`).
- The README states explicitly which differences survive correction and which
  do not. Differences inside the noise band are reported as indistinguishable —
  not ranked.
- No model is tuned harder than another to manufacture a gap. One recipe,
  enforced by `RECIPE` in `bench/config.py` and its hash in every result row.

## Known asymmetry

Four of five backbones report ImageNet channel statistics in timm's
`pretrained_cfg`; **`mobilevit_s.cvnets_in1k` reports `mean=(0,0,0)`,
`std=(1,1,1)`** (raw `[0,1]` inputs). Shared preprocessing — required by the
fairness rule — therefore departs from MobileViT's pretraining convention for
that model alone. Full fine-tuning is expected to absorb it. This is a genuine
asymmetry, recorded here and in the README limitations rather than described as
a clean alignment.
