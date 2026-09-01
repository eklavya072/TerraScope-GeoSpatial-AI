# Measurement protocol

The rejection criteria and the statistical rules below were committed **before**
any Phase 5 measurement was taken (commit `000d902`, which precedes every
commit carrying results). The commit ordering is the evidence; check it against
the first row of `results/bench.jsonl`.

This file describes what was **actually done**, and states every deviation from
the original pre-registration explicitly. A protocol document that describes an
experiment nobody ran is worse than no protocol document at all.

## Why pre-register rejection criteria

This project has hard evidence that measurement windows get contaminated.
Training epochs contended by concurrent profiling ran **485 s and 1261 s against
136 s when idle** — a 3.5–9× inflation, recorded in `results/runs.jsonl`.
Contamination is a certainty, not a risk, so some windows will have to be
discarded. Choosing *which* after seeing the energy numbers would be post-hoc
selection. Fixing the thresholds in advance converts that judgement into a
stated rule.

## Rejection criteria

Thresholds live in `bench/exclusion.py`; this file and that module must agree.

| Criterion | Threshold | Evaluated in v1.0.0? |
|---|---|---|
| Energy window coverage | `< 0.95` | **yes** |
| p95 / p50 latency ratio | `> 1.50` | **yes** |
| Max inter-sample gap | `> 3 ×` nominal interval | no — not retained per window |
| Idle baseline drift | `> 15%` vs session baseline | no — no per-window idle baseline captured |

Criteria that were never instrumented are reported as `not_evaluated`. They are
**not** counted as per-window failures: the original draft failed them closed,
which would have marked all 300 windows excluded and told a reader nothing. An
uninstrumented criterion is a gap in the protocol, recorded below, not evidence
against an individual window.

## Outcome in v1.0.0

**4 of 300 windows excluded.** They remain in `results/bench.jsonl` and are
listed in `results/summary.json` under `exclusions`; nothing is deleted.

| Window | Reason |
|---|---|
| `resnet50 int8_dynamic` s0 t4 b1 | p95/p50 = 1.65 > 1.50 (contention) |
| `mobilevit_s fp32` s0 t4 b1 | p95/p50 = 1.64 > 1.50 (contention) |
| `mobilevit_s int8_static` s3 t4 b32 | energy coverage 0.948 < 0.95 |
| `mobilevit_s fp32` s4 t1 b32 | energy coverage 0.949 < 0.95 |

None fall in the primary reporting configuration (1 thread, batch 1), so **no
headline figure changes** when they are removed — verified by recomputing the
summary with and without them.

## Session conditions as run

- One long-lived `powermetrics` sampler for the entire matrix
  (`scripts/energy_sampler.sh`), started once under `sudo`. Per-run invocation
  would put process-startup cost inside each window — a bias that scales with
  window length and therefore correlates with the variable being measured.
- Nominal sampling interval **200 ms**.
- `caffeinate` held for the session; a sleep mid-matrix loses the trace.
- The full matrix ran in **one session**, 22:40–00:47, on AC power with macOS
  Low Power Mode disabled. `pmset` recorded no thermal warning. Power source and
  Low Power Mode state are recorded in every result row.
- Each window is at least **1,000 inferences and at least 20 seconds**, after 50
  discarded warm-up inferences. The 20 s floor exists because `powermetrics`
  timestamps are quantised to whole seconds, so shorter windows cannot be
  attributed energy accurately.
- Thread counts pinned in the ORT session **and** in the environment
  (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS`), because BLAS pools ignore the session setting.

## Energy claim wording

Energy is **CPU package power sampled via `powermetrics` and integrated over the
timed window** — on-die telemetry, package-level, not per-process and not
wall-plug metering. It is a real sensor reading, which a TDP-model estimate is
not, but it is not a socket measurement and must never be described as one.

Figures are **gross**, not baseline-subtracted. The idle baseline measured over
338 s immediately after the matrix was **0.036 W**, which is **0.78% of the
lowest-power measurement window** (4.64 W) and less for every other window.
Baseline subtraction would therefore change no reported figure by more than
0.78%, well inside the seed-to-seed variation already reported.

## Statistical protocol

- **≥5 seeds** per model; every accuracy figure reported as mean ± 95% Student-t
  confidence interval (`bench/stats.py: mean_ci`).
- Pairwise comparison by **Welch's t-test**, corrected across all *k(k−1)/2*
  pairs with **Holm–Bonferroni** (`bench/stats.py: holm_bonferroni`).
- Quantisation deltas are **paired by seed**, since fp32 and int8 share a
  checkpoint and pairing removes the seed variation that would otherwise swamp a
  sub-point change.
- The README states explicitly which differences survive correction and which do
  not. Differences inside the noise band are reported as indistinguishable, not
  ranked.
- No model is tuned harder than another. One recipe, enforced by `RECIPE` in
  `bench/config.py` and hashed into every result row.

## Deviations from the pre-registration

Stated in full, because a deviation a reader discovers themselves costs more
credibility than the deviation itself.

1. **Failing windows were not re-run.** The pre-registration allowed 2 retries.
   The exclusions were identified after the session ended and the privileged
   sampler had been stopped, so failing windows could not be re-run under
   identical conditions. They are excluded from aggregates instead, and the
   headline numbers are reported with and without them (no change).
2. **No idle protocol between runs.** The pre-registration specified ≥60 s idle
   before and after the matrix and ≥30 s between runs. This was not enforced;
   runs proceeded back to back. Consequence: no per-window idle baseline, so the
   baseline-drift criterion could not be evaluated, and energy is reported gross.
   The magnitude of that consequence is quantified above (≤0.78%).
3. **Sampling interval 200 ms, not 500 ms.** Finer than pre-registered.
4. **The raw sampler trace is not archived in the repository.** It is ~250 MB
   per session, above GitHub's file limit. `results/power_summary.csv` ships the
   per-second means needed to audit the integration, and per-window joules are
   in `results/bench.jsonl`.
5. **`codecarbon` was not run alongside.** On Apple Silicon it has no RAPL to
   read and degrades to a hardcoded-TDP model whose output is a linear function
   of runtime. A second column that is arithmetically derived from elapsed time
   is not an independent method, so it would not have been the corroboration the
   pre-registration imagined. This is recorded in the README limitations.

## Known asymmetry

Four of five backbones report ImageNet channel statistics in timm's
`pretrained_cfg`; **`mobilevit_s.cvnets_in1k` reports `mean=(0,0,0)`,
`std=(1,1,1)`** (raw `[0,1]` inputs). Shared preprocessing — required by the
fairness rule — therefore departs from MobileViT's pretraining convention for
that model alone. Full fine-tuning is expected to absorb it. Recorded here and
in the README limitations rather than described as a clean alignment.
