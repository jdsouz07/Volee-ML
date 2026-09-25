# Shortcuts. Run `make` with no arguments to see them.
PY = .venv/bin/python

help:
	@echo "make setup     create the Python environment and install packages"
	@echo "make all       run every step (download -> features -> train -> evaluate -> Volee demo)"
	@echo "make data      step 1: download and clean pro matches"
	@echo "make features  step 2: replay history into model-ready rows"
	@echo "make train     step 3: train the models"
	@echo "make evaluate  step 4: score them and write reports/results.md"
	@echo "make volee     step 5: run the model on Volee-format matches"
	@echo "make test      run the tests"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

all:
	$(PY) -m volee_ml.pipeline

data:
	$(PY) -m volee_ml.data
features:
	$(PY) -m volee_ml.features
train:
	$(PY) -m volee_ml.train
evaluate:
	$(PY) -m volee_ml.evaluate
volee:
	$(PY) -m volee_ml.volee
test:
	.venv/bin/pytest -q

.PHONY: help setup all data features train evaluate volee test
