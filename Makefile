# TerraScope benchmark. Every target runs inside the locked uv environment so
# that torch/onnxruntime versions -- which latency and energy numbers depend on
# -- cannot drift between a maintainer's machine and a reader's.

UV ?= uv
VENV := .venv-bench
PY := $(VENV)/bin/python
export UV_PROJECT_ENVIRONMENT := $(VENV)
export PYTHONPATH := $(CURDIR)

SEEDS ?= 0,1,2,3,4
MODEL ?= all
DEVICE ?= auto

.PHONY: help setup data split train bench report clean-results

help:
	@echo "make setup    install the locked benchmark environment"
	@echo "make data     download EuroSAT and write the sha256 manifest"
	@echo "make split    regenerate the committed deterministic split"
	@echo "make train    train MODEL=$(MODEL) over SEEDS=$(SEEDS)"
	@echo "make bench    full measurement matrix (accuracy, latency, memory, energy)"
	@echo "make report   rebuild the results table and Pareto curve from results/"

setup:
	$(UV) sync --frozen

data: setup
	$(PY) scripts/prepare_data.py

split: data
	$(PY) scripts/make_split.py --seed 42

train: setup
	$(PY) -m bench.train --model $(MODEL) --seeds $(SEEDS) --device $(DEVICE)

clean-results:
	rm -f results/runs.jsonl results/bench.jsonl
