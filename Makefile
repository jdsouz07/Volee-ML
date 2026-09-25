# Shortcuts. Run `make` with no arguments to see them.
PY = .venv/bin/python

help:
	@echo "make setup     create the Python environment and install packages"
	@echo "make all       run every step (download -> features -> train -> evaluate -> Volee demo)"
	@echo "make data      step 1: download and clean pro matches"
	@echo "make tune      step 2a: tune Glicko and the margin-aware rating on validation years"
	@echo "make features  step 2: replay history into model-ready rows"
	@echo "make train     step 3: train the models"
	@echo "make evaluate  step 4: score them and write reports/results.md"
	@echo "make volee     step 5: run the model on Volee-format matches"
	@echo "make export    step 6: rebuild the live demo's data (docs/demo_data.json)"
	@echo "make demo      open the demo locally at http://localhost:8766"
	@echo "make test      run the tests"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

all:
	$(PY) -m volee_ml.pipeline

data:
	$(PY) -m volee_ml.data
tune:
	$(PY) -m volee_ml.tune
features:
	$(PY) -m volee_ml.features
train:
	$(PY) -m volee_ml.train
evaluate:
	$(PY) -m volee_ml.evaluate
volee:
	$(PY) -m volee_ml.volee
export:
	$(PY) -m volee_ml.export
demo:
	$(PY) -m http.server 8766 --directory docs
test:
	.venv/bin/pytest -q

.PHONY: help setup all data tune features train evaluate volee export demo test
