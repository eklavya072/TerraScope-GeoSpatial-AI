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

.PHONY: help setup data split train train-resume export bench report all clean-results

help:
	@echo "make setup    install the locked benchmark environment"
	@echo "make data     download EuroSAT and write the sha256 manifest"
	@echo "make split    regenerate the committed deterministic split"
	@echo "make train    train MODEL=$(MODEL) over SEEDS=$(SEEDS)"
	@echo "make train-resume  resume an interrupted training matrix"
	@echo "make export   export checkpoints to ONNX fp32 + int8 dynamic/static"
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

# Resume an interrupted matrix: re-runs only the (model, seed) pairs that are
# not already recorded under the current recipe and split hashes.
train-resume: setup
	caffeinate -is $(PY) -m bench.train --model $(MODEL) --seeds $(SEEDS) \
	    --device $(DEVICE) --skip-done

export: setup
	$(PY) -m bench.export_onnx --model $(MODEL) --seeds $(SEEDS)

# Thread counts are pinned in the environment as well as in the ORT session:
# BLAS/OMP backends spawn their own pools and will silently ignore the session
# setting otherwise, which is the classic way CPU latency numbers stop
# reproducing on someone else's machine.
bench: setup
	@test -s results/power_log.txt || echo "NOTE: results/power_log.txt is empty -- \
energy columns will be null. Start 'sudo ./scripts/energy_sampler.sh' in another \
terminal first."
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
	$(PY) -m bench.benchmark --model $(MODEL) --seeds $(SEEDS)

# report regenerates the tables and figure, then injects them into README.md
# between generated markers, so no README number is ever hand-typed.
report: setup
	$(PY) -m bench.report
	$(PY) scripts/render_readme.py

# The whole pipeline from a clean clone.
all: data split train export bench report

clean-results:
	rm -f results/runs.jsonl results/bench.jsonl
