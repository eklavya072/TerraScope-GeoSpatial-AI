# Datasheet: TerraScope EuroSAT benchmark split and results

Following the *Datasheets for Datasets* framework (Gebru et al., 2021). This
datasheet covers two artefacts produced by this repository:

1. **the benchmark split** — `splits/eurosat_split_seed42.csv`, a deterministic
   train/validation/test partition of the EuroSAT RGB corpus;
2. **the results data** — `results/runs.jsonl`, `results/bench.jsonl` and
   `results/summary.json`, the measured accuracy, latency, memory and energy
   figures.

It does **not** re-datasheet EuroSAT itself, which is the work of Helber et al.;
see *Attribution* below and the original paper for the imagery's own provenance.

---

## Motivation

**For what purpose was the dataset created?**
The split was created because EuroSAT ships no official train/test partition.
Every published EuroSAT accuracy figure is therefore measured against a fold
definition the reader cannot inspect, which makes cross-paper comparison
unsound and reproduction impossible. This split is generated deterministically
from a fixed seed and committed to the repository so that every number in this
benchmark references a fold definition anyone can check by hash.

The results data was created to answer one question: what land-cover
classification accuracy is achievable on CPU-only hardware, and what does each
accuracy point cost in energy?

**Who created it and who funded it?**
Created by the TerraScope repository maintainer as an independent, unfunded
project. No sponsor influenced the design or the reporting.

---

## Composition

**What do the instances represent?**
Each row of the split file is one EuroSAT RGB tile: a 64×64 pixel Sentinel-2
image patch, labelled with one of 10 land-cover classes (AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River, SeaLake).

**How many instances are there?**
27,000 tiles total — 18,900 train / 5,400 validation / 2,700 test (70/20/10).
The underlying class distribution is unbalanced by design of EuroSAT: 3,000
tiles each for AnnualCrop, Forest, HerbaceousVegetation, Residential and
SeaLake; 2,500 each for Highway, Industrial, PermanentCrop and River; 2,000 for
Pasture. The split is **stratified**, so each fold preserves those proportions.

**Does the dataset contain all possible instances or is it a sample?**
It covers the complete EuroSAT RGB corpus. It is itself a sample of Sentinel-2
coverage over Europe — see *Limitations*.

**What data does each instance consist of?**
The split file stores a relative path, a class label and a fold assignment. The
imagery is not redistributed here; `scripts/prepare_data.py` fetches it and
writes `data/eurosat_rgb/MANIFEST.sha256`, a per-tile sha256 manifest so that a
reader can verify they hold the same bytes we measured against.

**Is any information missing?**
The tiles carry no geolocation, acquisition timestamp, or Sentinel-2 tile ID in
the form redistributed by the mirror used here. This means the split **cannot**
control for geographic or temporal leakage between folds — see *Limitations*.

**Are there labels, and is there noise?**
Yes, one class label per tile, from EuroSAT. Label noise was not independently
audited; the classes Highway, River and PermanentCrop are visually confusable at
64×64 and are the usual source of residual error.

**Are there errors, sources of noise, or redundancies?**
EuroSAT tiles are cut from larger Sentinel-2 scenes, so tiles from the same
scene can be spatially adjacent and highly correlated. Because the corpus as
redistributed carries no scene identifier, a random split (including this one)
may place near-adjacent tiles in different folds. This inflates measured
accuracy relative to true generalisation to unseen geography.

**Is the dataset self-contained?**
The split file and manifest are self-contained and committed. The imagery is
fetched from an external mirror at prepare time; the manifest is what protects
against that mirror changing underneath a reader.

**Does it contain confidential, offensive, or personally identifiable data?**
No. It is 64×64 overhead satellite imagery at 10 m ground resolution, at which
individuals are not identifiable.

---

## Collection process

**How was the data acquired?**
The imagery was collected by Helber et al. from Copernicus Sentinel-2 Level-1C
products and released as EuroSAT. This project acquires it from the
`giswqs/EuroSAT_RGB` Hugging Face mirror, reading undecoded bytes so the files
written to disk are byte-identical to the upstream archive rather than re-encoded
through an image library.

**What was the sampling strategy for the split?**
Stratified random assignment within each class, 70/20/10, drawn from
`numpy.random.default_rng(42)`. Filenames are sorted lexicographically and
classes visited in sorted order before any draw, so the output depends only on
the corpus contents and the seed — not on filesystem ordering, dict iteration
order, or `PYTHONHASHSEED`. Verified byte-identical across repeated runs.

**Over what timeframe was the data collected?**
The EuroSAT imagery predates this project; see Helber et al. for acquisition
dates. The split and all results in this repository were generated in August
2026 on the hardware recorded in `results/summary.json`.

**Were ethical review processes conducted?**
Not applicable — no human subjects, no personal data.

---

## Preprocessing / cleaning / labelling

**Was any preprocessing done?**
For the corpus: none. Original JPEG bytes are preserved and hashed.

For training and evaluation, identically for every architecture:
- tiles used at their native 64×64 resolution (no upscaling to the backbones'
  224×224 pretraining size — that would multiply CPU inference cost roughly 12×
  while adding no information, and CPU cost is the quantity under study);
- channel normalisation with ImageNet-1k statistics, mean (0.485, 0.456, 0.406)
  and std (0.229, 0.224, 0.225), as distributed with torchvision and reported by
  timm's `pretrained_cfg` for these backbones;
- augmentation on the training fold only: random horizontal flip, vertical flip
  and 90° rotation. For nadir satellite imagery these are genuinely
  label-preserving, unlike colour jitter, whose safety depends on sensor
  radiometry.

**Was the raw data saved?**
Yes — the corpus on disk is the unmodified original bytes, verifiable against
`MANIFEST.sha256`.

**Is the preprocessing software available?**
Yes: `scripts/prepare_data.py`, `scripts/make_split.py`, `bench/data.py`.

---

## Uses

**What tasks has the dataset been used for?**
Training and evaluating five ImageNet-pretrained architectures under one shared
recipe, and measuring their CPU inference latency, memory footprint and energy
consumption at three precisions (fp32, int8 dynamic, int8 static).

**What (other) tasks could it be used for?**
Comparing new architectures under the same recipe; studying quantisation
sensitivity; as a worked example of split-pinned, variance-reporting benchmark
methodology.

**Is there anything about the composition that could lead to unfair treatment?**
EuroSAT covers 34 European countries. A model selected on this benchmark should
**not** be assumed to transfer to other continents, whose land-cover
appearance, agricultural patterns, settlement morphology and seasonal
phenology differ substantially. Treating a EuroSAT accuracy figure as a global
land-cover capability claim would systematically overstate performance outside
Europe.

**Are there tasks for which it should not be used?**
It should not be used to certify operational fitness for any deployment outside
European Sentinel-2 RGB imagery at 10 m resolution, nor as evidence about
multispectral (13-band) performance — this benchmark uses RGB only.

---

## Limitations

These apply to the results data specifically, and are restated in the README:

- **Single hardware platform.** All measurements come from one Apple M2
  (8 cores, 8 GB RAM, macOS 14.6.1). Latency and energy rankings may differ on
  x86, on server-class CPUs with AVX-512, or under different memory bandwidth.
- **Single dataset.** One dataset, one resolution, one sensor, RGB only.
- **Energy is estimated, not metered.** Figures come from Apple Silicon on-die
  CPU package power telemetry sampled by `powermetrics`, integrated over each
  timed window. This excludes DRAM, display and power-supply losses, and is not
  a wall-socket measurement. `codecarbon` cannot serve as an independent check
  on this hardware: it reads Intel RAPL, which Apple Silicon lacks, so it falls
  back to a hardcoded-TDP model whose output is a linear function of runtime.
- **CO₂e depends entirely on a stated assumption.** Carbon figures scale
  linearly with an assumed grid carbon intensity, which varies roughly 15×
  between a hydro/nuclear-heavy grid and a coal-heavy one. The assumption is
  printed next to every CO₂e number.
- **EuroSAT is near-saturated.** Strong models cluster in a narrow accuracy
  band, so architecture differences are often smaller than seed-to-seed noise.
  This is why every accuracy figure here is a mean over 5 seeds with a 95%
  confidence interval, and why pairwise differences are reported with
  Holm-Bonferroni correction.
- **Geographic bias.** EuroSAT is European. See *Uses* above.
- **No scene-level split control.** Spatially adjacent tiles from the same
  Sentinel-2 scene may span folds, which inflates accuracy relative to true
  generalisation to unseen geography. The corpus as redistributed carries no
  scene identifier, so this could not be controlled.
- **Thermal environment.** The benchmark runs on an actively-used laptop rather
  than a climate-controlled rack. Runs are executed in one session with
  background load minimised, and per-run timing variance is reported, but
  ambient temperature is not controlled and sustained-load throttling cannot be
  ruled out.
- **Preprocessing fidelity for MobileViT.** The fairness rule requires identical
  preprocessing for all architectures, so all five use ImageNet channel
  statistics. Four of the five backbones report exactly those statistics in
  their pretraining config; MobileViT's config instead expects raw [0,1] inputs.
  Full fine-tuning is expected to absorb the difference, but MobileViT's numbers
  carry this caveat that the others do not.

---

## Maintenance

**Who maintains it and how can they be contacted?**
The repository maintainer, via GitHub issues on the project repository.

**Will the dataset be updated?**
The split file is **frozen**. Changing it would invalidate every published
result, so any revision will ship as a new file under a new seed and a new
version tag rather than as an edit. Results data may gain rows as models or
hardware are added; existing rows are not rewritten.

**Is there an erratum process?**
Corrections are published as repository releases with notes describing what
changed and why. Because each result row carries the split hash, the recipe
hash and the full environment fingerprint, a superseded number can always be
traced to the exact conditions that produced it.

**How will older versions be supported?**
Released versions are archived on Zenodo with their own DOIs, so a citation to
a specific version remains resolvable after later revisions.

**Can others extend or contribute?**
Yes. Adding an architecture means adding one entry to `MODEL_ZOO` in
`bench/config.py` and running `make all`; the shared recipe applies
automatically, which is deliberate — there is no supported way to give a new
model a tuned recipe of its own.

---

## Attribution

EuroSAT is distributed under the MIT licence. Cite:

> Helber, P., Bischke, B., Dengel, A., & Borth, D. *EuroSAT: A Novel Dataset and
> Deep Learning Benchmark for Land Use and Land Cover Classification.*

EuroSAT is derived from Copernicus Sentinel-2 imagery. Copernicus data is
provided under terms granting free access, including reproduction, distribution
and modification.

The results data in `results/` is released under CC-BY-4.0. The code is MIT.
