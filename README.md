# TerraScope

**What does a percentage point of land-cover accuracy cost in energy?**

[![CI](https://github.com/eklavya072/TerraScope-GeoSpatial-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/eklavya072/TerraScope-GeoSpatial-AI/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-run%20the%20models-1b4332)](https://eklavya072.github.io/TerraScope-GeoSpatial-AI/)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/results%20data-CC--BY--4.0-blue)](LICENSE-DATA)
[![EuroSAT](https://img.shields.io/badge/dataset-EuroSAT%20(MIT)-green)](https://github.com/phelber/eurosat)

A **Green AI** benchmark: five architectures × three numeric precisions × five seeds,
trained on EuroSAT under one identical recipe and measured on CPU only — accuracy,
latency, memory and energy. 300 measurement windows. Every number regenerates from
committed data, and CI fails if any of them drifts.

---

## The finding

Accuracy is a weak discriminator. Energy is not.

|                        | ResNet-50 (fp32) | EfficientNet-Lite0 (int8) | difference |
| ---------------------- | ---------------: | ------------------------: | ---------: |
| Top-1 accuracy         |       **98.12%** |               **97.45%** | −0.67 pp |
| Energy per 1k images   |      **60.84 J** |                **3.18 J** | 19× less |
| p95 latency            |         12.24 ms |                   0.43 ms | 28× faster |
| Model on disk          |          94.0 MB |                    3.8 MB | 25× smaller |

Swapping the standard baseline for a quantised mobile model costs
**0.67 percentage points** of accuracy and buys **19× less energy**,
**28× faster** inference and **25× less disk**.

Across all five architectures accuracy spans a **1.25 pp spread** — of which
8 of 10 pairwise comparisons survive Holm–Bonferroni correction, so the differences
are real, just small. Energy spans a factor of 19.

> **On CPU-only hardware, deployment is an energy decision, not an accuracy one.**

Energy is **estimated** from on-die power telemetry, not metered at the wall.
That distinction is kept everywhere it appears — see [Limitations](#limitations).

---

## Why this question

Land-cover classification from satellite imagery is run by public-sector and
development organisations on the hardware they already own, which is usually a CPU
server without a GPU. UNDP's Accelerator Labs, for instance, apply earth-observation
models to generate land-use and land-cover maps in the field, and UNDP names
**Green Compute** as one of five foundations for national AI ecosystems.

Both of those make the same question load-bearing: *what does the accuracy actually
cost to run?* Model papers report accuracy. Deployments pay for joules, latency and
memory. This benchmark measures all four under one recipe so the trade-off can be
read off directly rather than guessed at.

That is the argument Schwartz et al. made in **[Green AI](https://doi.org/10.1145/3381831)**
(CACM, 2020): efficiency belongs in the evaluation, not in an appendix, because
reporting accuracy alone rewards whoever can spend the most compute. Strubell et al.
(ACL 2019) put numbers on the training side of that; Henderson et al.
([JMLR 2020](https://jmlr.org/papers/v21/20-312.html)) argued for systematic energy
and carbon reporting. This repository is a small, complete instance of what those
papers ask for, applied to **inference** — where a deployed model spends nearly all
of its lifetime energy, and where the published numbers are thinnest.

---

## Try it

**[Run the models in your browser →](https://eklavya072.github.io/TerraScope-GeoSpatial-AI/)**
No install. It executes the real ONNX graphs on a held-out test tile via WebAssembly
and times them on your machine; accuracy and energy are looked up from the committed
benchmark, never invented.

Or reproduce the whole thing locally:

```bash
make setup     # locked environment (uv + uv.lock)
make data      # fetch EuroSAT, write a sha256 manifest
make split     # regenerate the committed split, verified byte-identical
make train     # 5 architectures × 5 seeds, one shared recipe
make export    # ONNX fp32 + int8 dynamic/static
make bench     # latency, memory and energy matrix (CPU only)
make report    # rebuild every table, figure and README number
```

Training uses a GPU where available purely to make the matrix tractable. **All
benchmarking is CPU-only**, and no reported figure depends on the training device.

<details>
<summary>Energy measurement needs a privileged sampler</summary>

Apple Silicon exposes on-die power only to root, so a sampler runs alongside the
benchmark:

```bash
sudo ./scripts/energy_sampler.sh
```

Start it before `make bench` and leave it running. Without it the benchmark still
records accuracy, latency and memory, and reports every energy column as `null`
rather than substituting an estimate.
</details>

---

## Results

<!-- BEGIN:results_table -->
### Results (ONNX Runtime CPU EP, intra-op threads=1, batch=1)

Accuracy is the mean over 5 seeds with a Student-t 95% confidence interval. Energy figures are ESTIMATED from on-die power telemetry, not metered at the wall. CO2e assumes 481 gCO2e/kWh (world average grid carbon intensity, ~481 gCO2e/kWh).

| Model | Precision | Params | Accuracy % (mean ± 95% CI) | p95 latency (ms) | Model RSS (MB) | Model (MB) | Energy/1k inf (J, estimated) | CO2e/1M inf (g, estimated) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| efficientnet_lite0 | fp32 | 3.38M | 97.61 ± 0.13 | 3.74 | 31 | 13.5 | 19.28 | 2.58 |
| efficientnet_lite0 | int8_dynamic | 3.38M | 63.44 ± 3.97 | 4.43 | 19 | 3.6 | 25.75 | 3.44 |
| efficientnet_lite0 | int8_static | 3.38M | 97.45 ± 0.32 | 0.43 | 17 | 3.8 | 3.18 | 0.42 |
| mobilenetv3_large | fp32 | 4.21M | 97.10 ± 0.17 | 2.73 | 35 | 16.8 | 14.46 | 1.93 |
| mobilenetv3_large | int8_dynamic | 4.21M | 70.47 ± 8.10 | 2.88 | 19 | 4.4 | 19.32 | 2.58 |
| mobilenetv3_large | int8_static | 4.21M | 91.10 ± 1.11 | 0.52 | 21 | 4.7 | 3.60 | 0.48 |
| mobilenetv3_small | fp32 | 1.53M | 97.19 ± 0.10 | 1.40 | 18 | 6.1 | 6.97 | 0.93 |
| mobilenetv3_small | int8_dynamic | 1.53M | 14.57 ± 2.95 | 1.54 | 14 | 1.7 | 8.02 | 1.07 |
| mobilenetv3_small | int8_static | 1.53M | 32.53 ± 10.40 | 0.32 | 19 | 1.9 | 1.72 | 0.23 |
| mobilevit_s | fp32 | 4.94M | 98.36 ± 0.44 | 4.46 | 37 | 20.0 | 26.31 | 3.51 |
| mobilevit_s | int8_dynamic | 4.94M | 58.41 ± 10.82 | 3.91 | 30 | 5.5 | 24.53 | 3.28 |
| mobilevit_s | int8_static | 4.94M | 49.50 ± 9.13 | 2.11 | 28 | 5.8 | 13.77 | 1.84 |
| resnet50 | fp32 | 23.53M | 98.12 ± 0.30 | 12.24 | 162 | 94.0 | 60.84 | 8.13 |
| resnet50 | int8_dynamic | 23.53M | 80.49 ± 8.15 | 5.36 | 39 | 23.7 | 27.08 | 3.62 |
| resnet50 | int8_static | 23.53M | 97.22 ± 0.38 | 2.49 | 67 | 24.0 | 14.39 | 1.92 |
<!-- END:results_table -->

**ONNX export is accuracy-neutral.** All 25 exported fp32 graphs reproduce their
PyTorch checkpoint's test accuracy exactly — 25/25 identical, maximum difference
0.000000 pp. Latency and energy are measured on the exported graph while accuracy is
attributed to the model, so without this check every row would risk describing two
different models on one line.

---

## What quantisation costs

<!-- BEGIN:quantisation -->
### Accuracy cost of int8 quantisation

Paired per-seed differences against each model's own fp32 export, mean with a Student-t 95% confidence interval. Negative means quantisation lost accuracy.

| Model | Precision | fp32 % | int8 % | Δ (pp, mean ± 95% CI) |
|---|---|---:|---:|---:|
| efficientnet_lite0 | int8_dynamic | 97.61 | 63.44 | -34.17 ± 4.07 |
| efficientnet_lite0 | int8_static | 97.61 | 97.45 | -0.16 ± 0.23 |
| mobilenetv3_large | int8_dynamic | 97.10 | 70.47 | -26.63 ± 8.12 |
| mobilenetv3_large | int8_static | 97.10 | 91.10 | -6.01 ± 1.08 |
| mobilenetv3_small | int8_dynamic | 97.19 | 14.57 | -82.62 ± 2.92 |
| mobilenetv3_small | int8_static | 97.19 | 32.53 | -64.66 ± 10.34 |
| mobilevit_s | int8_dynamic | 98.36 | 58.41 | -39.94 ± 11.18 |
| mobilevit_s | int8_static | 98.36 | 49.50 | -48.86 ± 9.28 |
| resnet50 | int8_dynamic | 98.12 | 80.49 | -17.63 ± 7.89 |
| resnet50 | int8_static | 98.12 | 97.22 | -0.90 ± 0.50 |
<!-- END:quantisation -->

Quantisation is **architecture-specific, not a uniform tax.**

- **EfficientNet-Lite0 loses 0.16 ± 0.23 pp** — a confidence interval containing
  zero, so its int8 form is statistically indistinguishable from its fp32 parent
  while using 6× less energy. ResNet-50 loses 0.90 pp.
- **MobileNetV3-Small loses 64.66 pp and MobileViT-S 48.86 pp.** Post-training
  quantisation destroys them. Do not deploy either in int8 without
  quantisation-aware training.
- **int8 *dynamic* is strictly dominated** — worse accuracy *and* worse energy than
  fp32 for every model here. Reported because a negative result saves someone else
  the experiment.

<details>
<summary>We tried to recover MobileNetV3-Small, and could not</summary>

Per-channel weights, min-max / percentile / entropy calibration, restricting
quantisation to Conv/Gemm, excluding depthwise convolutions, signed and unsigned
activations, and batch sizes 1 and 64 all fail. Its hard-swish and squeeze-excite
activation distributions are the textbook case post-training quantisation cannot
represent; recovering them requires quantisation-aware training, which is out of
scope for a post-training benchmark.
</details>

---

## Are the accuracy differences real?

<!-- BEGIN:significance -->
### Which accuracy differences are statistically distinguishable?

Welch's t-test over seeds, Holm-Bonferroni corrected across all pairwise comparisons (family-wise alpha = 0.05).

| Comparison | Δ accuracy (pp) | p | Holm threshold | Distinguishable? |
|---|---:|---:|---:|---|
| efficientnet_lite0 vs mobilenetv3_small | +0.41 | 0.0001 | 0.0050 | **yes** |
| mobilenetv3_large vs resnet50 | -1.01 | 0.0001 | 0.0056 | **yes** |
| efficientnet_lite0 vs mobilenetv3_large | +0.50 | 0.0002 | 0.0063 | **yes** |
| mobilenetv3_small vs resnet50 | -0.93 | 0.0005 | 0.0071 | **yes** |
| mobilenetv3_large vs mobilevit_s | -1.25 | 0.0006 | 0.0083 | **yes** |
| mobilenetv3_small vs mobilevit_s | -1.16 | 0.0014 | 0.0100 | **yes** |
| efficientnet_lite0 vs resnet50 | -0.51 | 0.0063 | 0.0125 | **yes** |
| efficientnet_lite0 vs mobilevit_s | -0.75 | 0.0073 | 0.0167 | **yes** |
| mobilenetv3_large vs mobilenetv3_small | -0.09 | 0.2480 | 0.0250 | no |
| mobilevit_s vs resnet50 | +0.24 | 0.2565 | 0.0500 | no |

8 of 10 pairwise accuracy differences are statistically distinguishable after correction.
<!-- END:significance -->

Two cautions. *Statistically distinguishable* is not *operationally meaningful*: the
entire best-to-worst spread is 1.25 pp, and a difference can be reliable yet far too
small to justify changing a deployment. And these compare architectures under one
fixed recipe and budget — a model that trains poorly here might do better with tuning
it was deliberately not given.

### Accuracy versus energy

![Accuracy versus energy, with the Pareto frontier marked](results/pareto.png)

The marked frontier is the true mathematical one, so it includes
`mobilenetv3_small int8_static` purely because nothing is cheaper — at 32.5% accuracy
that configuration is useless in practice. Restricted to configurations above 97%
accuracy: **EfficientNet-Lite0 int8_static** (97.45%, 3.18 J/1k) leads,
**ResNet-50 int8_static** (97.22%, 14.39 J/1k) is dominated by it, and
**MobileViT-S fp32** (98.36%, 26.31 J/1k) buys the last 0.9 pp for 8.3× the energy.

---

## Deployment recommendation

For a CPU-only server classifying Sentinel-2 RGB tiles, deploy **EfficientNet-Lite0
quantised to int8 with static calibration.**

It gives up 0.67 pp against the ResNet-50 fp32 baseline and 0.90 pp against the most
accurate model measured, in exchange for 19× less energy, 28× lower p95 latency and a
3.8 MB artefact. On one thread it sustains roughly 2,300 images/second (0.43 ms p95),
and its 17 MB resident footprint leaves the machine free for other work.

If the last 0.9 pp genuinely matters — which, given the dataset caveats below, should
be argued rather than assumed — MobileViT-S fp32 is accuracy-optimal at 8.3× the
energy. Do **not** deploy a quantised MobileNetV3 or MobileViT without
quantisation-aware training.

<details>
<summary><b>Is the energy column just the latency column in different units?</b></summary>

A fair objection, answered with the data rather than deflected. Energy is power × time,
so if package power were constant the energy axis would carry nothing latency doesn't.

Across all 75 measurement windows at 1 thread, batch 1:

| Quantity | Range | Ratio |
|---|---|---|
| p50 latency | 0.251 → 12.239 ms | 48.8× |
| Energy per 1,000 inferences | 1.56 → 63.12 J | 40.5× |
| Mean package power | 4.86 → 10.24 W | **2.1×** |

Energy correlates with latency at **r = 0.983**. So most of the energy spread *is* the
latency spread, and we say so plainly rather than implying two independent findings.

What the remaining 2.1× buys is not nothing:

- **Quantised models draw systematically more power** — 5.51 W for fp32, 6.16 W for
  int8-dynamic, **6.95 W for int8-static**: a 26% increase for static int8 over fp32.
  Quantisation doesn't simply make the same work shorter; it makes the CPU work harder
  while it runs. A latency-only reading would overstate int8's energy advantage.
- **It reorders one pair.** By latency, `mobilenetv3_large int8_dynamic` beats
  `efficientnet_lite0 fp32`; by energy the order reverses. One swap in fifteen is a
  small effect, and reporting it as small is the honest framing.
- **Across thread counts they decouple properly.** Package power spans 3.7× among the
  4-thread windows against 2.1× at 1 thread — but that is also where core placement
  confounds the comparison, so no cross-model conclusion is drawn from it.

The honest summary: at a fixed thread count energy is largely a restatement of
latency, with a real but second-order power term that matters most when comparing
precisions. The energy axis earns its place because the deployment question *is*
energy, and because the power term moves opposite to the intuition that int8 is
uniformly cheaper — but it is not an independent axis, and this README does not
claim it is.

**Thread counts.** MobileNetV3-Small fp32 goes from 0.94 to 0.79 J/1k at 4 threads
(1.50× faster, 16% less energy) at batch 1, and from 6.64 to 3.06 J/1k
(2.19× faster, 54% less energy) at batch 32.
</details>

---

## Why you can trust these numbers

| | |
|---|---|
| **Pre-registered** | Rejection criteria were committed **before any measurement** — `bd06ff5` (26 Aug) added PROTOCOL.md and `bench/exclusion.py`; `1c9fa33` (2 Sep) is the first commit carrying results. Seven days apart, checkable with `git log bd06ff5..1c9fa33`. |
| **One recipe** | Identical training recipe, preprocessing, split and seed protocol for every architecture. The recipe is hashed into every result row, so a changed recipe cannot masquerade as the old one. |
| **Committed split** | EuroSAT ships no official split. Ours is deterministic, sha256-hashed and version-controlled — CI verifies the hash every run. |
| **Variance, not point estimates** | Five seeds per configuration, Student-t 95% confidence intervals, Welch's t-test with Holm–Bonferroni correction. |
| **Nothing hand-typed** | Every published table and headline number is generated from `results/bench.jsonl`. CI regenerates them and fails on drift. |
| **83 tests** | Run against committed artefacts — no GPU, no dataset download. |

<details>
<summary>Exclusions, and what was excluded</summary>

**4 of 300 windows were excluded** — two for latency p95/p50 > 1.50 (contention), two
for energy sample coverage below 0.95. None fall in the primary reporting
configuration, so no headline figure changes when they are removed; this was verified
by recomputing the summary both ways. Excluded rows remain in `results/bench.jsonl`
and are listed in `results/summary.json` — nothing is deleted.

[PROTOCOL.md](PROTOCOL.md) records five deviations from the original
pre-registration, including two criteria that were never instrumented and the fact
that failing windows were not re-run.

Energy is reported **gross**, not baseline-subtracted. The idle baseline measured over
338 s immediately after the matrix was 0.036 W — 0.78% of the lowest-power window, and
less for every other one, well inside seed-to-seed variation.
</details>

<details>
<summary>Training recipe and hardware</summary>

<!-- BEGIN:recipe -->
Every architecture is trained under this identical recipe. There is no
supported way to give one model a tuned recipe of its own.

| Setting | Value |
|---|---|
| augmentation | `['random_hflip', 'random_vflip', 'random_rot90']` |
| batch_size | `128` |
| early_stopping | `{'mode': 'max', 'monitor': 'val_acc', 'patience': 4, 'restore_best_weights': True}` |
| finetune | `full` |
| input_size | `64` |
| label_smoothing | `0.1` |
| lr | `0.0003` |
| max_epochs | `20` |
| norm_mean | `[0.485, 0.456, 0.406]` |
| norm_std | `[0.229, 0.224, 0.225]` |
| optimizer | `adamw` |
| schedule | `cosine` |
| warmup_epochs | `2` |
| weight_decay | `0.0001` |
<!-- END:recipe -->

<!-- BEGIN:hardware -->
All measurements in this repository come from ONE machine. Latency,
memory and energy figures are properties of the model AND this hardware;
they are not portable claims.

| Property | Value |
|---|---|
| CPU | Apple M2 |
| Cores | 8 physical / 8 logical |
| RAM | 8 GiB |
| OS | Darwin 23.6.0 (Darwin Kernel Version 23.6.0) |
| Python | 3.12.11 |
| PyTorch | 2.9.1 |
| ONNX Runtime | 1.29.0 |
| timm | 1.0.28 |
| NumPy | 2.5.2 |
| Measurement date (UTC) | 2026-09-01T17:09:50+00:00 |
| Power source during measurement | AC |
| macOS Low Power Mode | 0 (AC) |
| Training environment (affects no reported figure) | 2026-08-27T01:52:08+00:00, battery |

Split file: `splits/eurosat_split_seed42.csv`  
Split sha256: `b77443792ba4b11439ed220b3dea699ce61e48e0a0af49c51fb6a8cf43b0595d`  
Recipe hash: `746abf440ef1`

Inference is measured through ONNX Runtime's **CPU execution provider only**.
Apple's GPU (MPS) and CoreML providers are excluded deliberately, not merely
left unused: the question is what CPU-only hardware achieves. Training used
the GPU, which affects no reported figure -- training cost is not part of the
deployment claim being made.
<!-- END:hardware -->
</details>

```bash
make test      # run the suite against the committed artefacts
```

---

## Limitations

The conclusions above are bounded by these, and the most important is first.

- **Single hardware platform.** Every published figure comes from one Apple M2, and
  energy and accuracy still do. This is the limitation most likely to change a
  conclusion. *Partially tested:* CI re-times the seven committed graphs on an x86
  Linux runner on every push and reports rank agreement with the M2 ordering
  ([`scripts/crossplatform_latency.py`](scripts/crossplatform_latency.py)). That
  checks whether the deployment recommendation survives a change of instruction set
  — the part that would actually mislead someone — but a shared CI runner cannot
  produce publication-grade absolute numbers, and it measures neither energy nor
  accuracy.
- **Energy is estimated, not metered.** On-die CPU package power, sampled at 200 ms
  and integrated over each window. Excludes DRAM, display and PSU losses. `codecarbon`
  cannot serve as an independent check: it reads Intel RAPL, which Apple Silicon
  lacks, so it degrades to a hardcoded-TDP model whose output is a linear function of
  runtime — latency wearing a different unit.
- **CO₂e rests entirely on a stated assumption** — 481 gCO₂e/kWh, world average.
  Carbon scales linearly with it, so substituting your own grid is one
  multiplication: the recommended configuration's 0.42 g per million inferences
  becomes ~0.04 g on a grid at 50 gCO₂e/kWh and ~0.61 g at 700. National grids span
  roughly that range, a ~15× spread — wider than any difference this benchmark
  measures between models.
- **EuroSAT is near-saturated**, so architecture differences are small in absolute
  terms even when statistically reliable. That is itself the finding.
- **Geographic bias.** EuroSAT covers 34 European countries. Nothing here supports a
  claim about performance elsewhere — land cover, agriculture, settlement morphology
  and phenology all differ.
- **No scene-level split control.** EuroSAT tiles are cut from larger Sentinel-2
  scenes and the corpus carries no scene identifier, so spatially adjacent tiles may
  span folds. This inflates all accuracies relative to true generalisation to unseen
  geography. Every model is affected equally, so comparisons remain valid, but the
  absolute numbers are not a geographic-generalisation claim.
- **RGB only.** EuroSAT's 13-band multispectral form is not benchmarked.
- **One preprocessing convention.** Fairness requires identical preprocessing, so all
  five models use ImageNet channel statistics. Four report exactly those; MobileViT's
  config expects raw [0,1] inputs, so its numbers carry a caveat the others do not.
- **Early stopping interacts with fast convergence.** MobileViT-S reaches ~98%
  validation accuracy within one epoch; on two of five seeds the patience-4 rule fired
  at epochs 6 and 7. The rule is identical for every model, so the comparison is fair,
  but it explains MobileViT-S's wider confidence interval.
- **Training-time figures mix power regimes.** ResNet-50 seed 0 trained under Low
  Power Mode on battery. Accuracy is unaffected — it comes from checkpoints — but
  `train_seconds` is not comparable across rows, and no benchmark figure depends on it.

---

## The wider point

Green AI is usually argued at training time, where the headline numbers are largest.
But a model is trained once and served for years — nearly all of its lifetime energy
is spent on inference, and that is the part most benchmarks leave unmeasured.

Measured here, on this dataset and this hardware, the entire accuracy spread across
five architectures is 1.25 pp while energy spans a factor of 19. Wherever that shape
holds, reaching for the largest model that fits is not a cautious default — it is a
large and invisible energy bill for a difference that may not survive contact with
the deployment.

The contribution is not that EfficientNet-Lite0 wins; on another dataset or another
chip it may not. It is that the trade-off was made measurable, under one honest
recipe, and that every number here can be checked by anyone who clones the
repository. **Efficiency only becomes a design decision once someone reports it.**

---

## Repository map

```
bench/          the benchmark
  config.py       zoo, shared recipe, normalisation, grid intensity (single source of truth)
  data.py         split-driven loading; the ONLY way to obtain a fold
  models.py       one construction path, so no architecture gets special treatment
  train.py        one (model, seed) under the shared recipe
  export_onnx.py  ONNX fp32 + int8 dynamic/static, calibrated from the train fold only
  benchmark.py    CPU-only latency/accuracy/energy matrix
  power.py        powermetrics parsing and energy integration
  exclusion.py    pre-registered window rejection criteria
  stats.py        Student-t CIs, Welch tests, Holm correction, Pareto frontier
  report.py       aggregation -> summary.json, tables, Pareto figure
scripts/        data prep, split generation, isolated memory measurement, site build
tests/          the suite; runs on committed artefacts, no GPU or dataset
splits/         the committed split, its metadata and its sha256
results/        raw measurements, derived tables, summary.json, Pareto figure
web/            the demo site: static pages, vendored runtime, served ONNX graphs
```

Deeper documentation: **[PROTOCOL.md](PROTOCOL.md)** (measurement protocol, outcomes,
deviations) and **[DATASHEET.md](DATASHEET.md)** (*Datasheets for Datasets*, Gebru et al.).

---

## Licences and attribution

Code is MIT ([LICENSE](LICENSE)). Results data and the split file are CC-BY-4.0
([LICENSE-DATA](LICENSE-DATA)).

**EuroSAT** is distributed under the MIT licence:

> Helber, P., Bischke, B., Dengel, A., & Borth, D. *EuroSAT: A Novel Dataset and Deep
> Learning Benchmark for Land Use and Land Cover Classification.*

**Sentinel-2 / Copernicus.** EuroSAT is derived from Copernicus Sentinel-2 imagery,
provided under terms granting free access, including reproduction, distribution and
modification.

**ESA WorldCover** is *not* used here, so its attribution string is deliberately
omitted rather than included for completeness — printing an attribution for data one
has not used is a false provenance claim.
