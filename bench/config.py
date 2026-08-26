"""Single source of truth for the benchmark: zoo, recipe, preprocessing, paths.

Everything a result depends on lives here so that a reader can see the whole
experimental contract in one file. Changing anything in this module invalidates
previously recorded results; the recipe hash written into every result row is
derived from RECIPE, so a silent change is detectable after the fact.
"""

import hashlib
import json
import os

# ---------------------------------------------------------------- dataset ----

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

CORPUS_ROOT = os.path.join("data", "eurosat_rgb")
SPLIT_CSV = os.path.join("splits", "eurosat_split_seed42.csv")
RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join("results", "checkpoints")

# EuroSAT tiles are natively 64x64. We train and benchmark at native resolution:
# upscaling to the backbones' 224x224 pretrain size multiplies CPU inference
# cost by ~12x while adding no information, and CPU inference cost is the very
# quantity this benchmark exists to measure.
INPUT_SIZE = 64

# ImageNet-1k channel statistics (Deng et al. 2009), as distributed with
# torchvision. Verified against timm's `pretrained_cfg` for all five entries in
# the zoo below: FOUR of the five (resnet50, both mobilenetv3 variants,
# efficientnet_lite0) report exactly these values. `mobilevit_s.cvnets_in1k`
# does NOT -- it reports mean=(0,0,0), std=(1,1,1), i.e. raw [0,1] inputs.
#
# We nonetheless apply one shared normalisation to every architecture, because
# the fairness rule requires identical preprocessing. The consequence is that
# MobileViT-S alone is fine-tuned under a normalisation that departs from its
# pretraining convention; full fine-tuning over 20 epochs is expected to absorb
# this, but it remains a genuine asymmetry and is recorded as such in the README
# limitations rather than presented as a clean alignment.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ------------------------------------------------------------------- zoo ----

# timm identifiers pin both architecture and pretrained weights. The bare names
# on the left are what appears in results tables and the README.
MODEL_ZOO = {
    "resnet50":            "resnet50.a1_in1k",
    "mobilenetv3_small":   "mobilenetv3_small_100.lamb_in1k",
    "mobilenetv3_large":   "mobilenetv3_large_100.ra_in1k",
    "efficientnet_lite0":  "efficientnet_lite0.ra_in1k",
    # MobileViT-S is the transformer-hybrid slot (4.94M params, the closest
    # weight-class match to TinyViT-5M's 5.07M). TinyViT was measured at
    # 694-1421 s/epoch on this hardware against MobileViT-S's 171 s/epoch --
    # a 4-8x difference that would have consumed the entire compute budget.
    # Caveat recorded in the README limitations: MobileViT's ImageNet
    # pretrain_cfg expects raw [0,1] inputs rather than the ImageNet channel
    # statistics the other four backbones use, so the shared preprocessing
    # required by the fairness rule departs from its pretraining convention.
    "mobilevit_s":         "mobilevit_s.cvnets_in1k",
}

BASELINE = "resnet50"

# ---------------------------------------------------------------- recipe ----

# ONE recipe for every architecture. Fairness before speed: no per-model
# learning rates, no per-model epoch budgets, no per-model augmentation. If a
# model underperforms under this recipe, that is a finding about the model
# under a fixed budget, and it is reported as such rather than tuned away.
RECIPE = {
    "input_size": INPUT_SIZE,
    "batch_size": 128,
    "max_epochs": 20,
    "optimizer": "adamw",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "schedule": "cosine",
    "warmup_epochs": 2,
    "label_smoothing": 0.1,
    "early_stopping": {"monitor": "val_acc", "mode": "max", "patience": 4,
                       "restore_best_weights": True},
    "finetune": "full",   # all layers trainable for every model
    "augmentation": ["random_hflip", "random_vflip", "random_rot90"],
    "norm_mean": NORM_MEAN,
    "norm_std": NORM_STD,
}

SEEDS = [0, 1, 2, 3, 4]


def recipe_hash() -> str:
    """Stable short hash of the training contract, embedded in every result."""
    blob = json.dumps(RECIPE, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ------------------------------------------------------------------ carbon ----

# Converting measured joules to CO2e requires a grid carbon-intensity
# assumption, and that assumption dominates the result -- the same workload is
# roughly 15x more carbon-intensive on a coal-heavy grid than on a nuclear or
# hydro-heavy one. It is therefore stated explicitly here and reprinted next to
# every CO2e figure rather than buried in a library default.
#
# Default: world average electricity generation intensity. Override with the
# --grid-intensity flag on bench.report when targeting a specific grid.
GRID_INTENSITY_G_CO2E_PER_KWH = 481.0
GRID_INTENSITY_SOURCE = (
    "world average grid carbon intensity, ~481 gCO2e/kWh; substitute your own "
    "grid's figure -- national values range from under 50 (hydro/nuclear-heavy) "
    "to over 700 (coal-heavy), and this constant scales every CO2e number "
    "reported here linearly"
)
