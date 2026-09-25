"""
The live demo must give the same answer as the real model.

Three implementations of one forecast have to agree:
  1. the trained sklearn model (train.py),
  2. export.py's plain-Python version of the demo arithmetic,
  3. docs/model.js, which the web page actually runs.

Needs a trained model and the exported docs/demo_data.json, so it's skipped
on a fresh checkout until `make all` has been run.
"""

import json
import shutil
import subprocess

import joblib
import pandas as pd
import pytest

from volee_ml import config
from volee_ml.export import DEMO_JSON, demo_features, demo_probability
from volee_ml.train import model_inputs

MODEL = config.MODELS_DIR / "logistic_regression.joblib"
needs_artifacts = pytest.mark.skipif(not (MODEL.exists() and DEMO_JSON.exists()),
                                     reason="run `make all` first")


def _pairs(payload, n=25):
    ids = sorted(payload["players"])
    by_tour = {}
    for pid in ids:
        by_tour.setdefault(payload["players"][pid]["tour"], []).append(pid)
    pairs = []
    for tour_ids in by_tour.values():
        pairs += [(tour_ids[i], tour_ids[-1 - i]) for i in range(n)]
    return pairs


@needs_artifacts
def test_python_demo_matches_sklearn():
    payload = json.loads(DEMO_JSON.read_text())
    model = joblib.load(MODEL)
    for a, b in _pairs(payload):
        feats = demo_features(payload, a, b)
        expected = model.predict_proba(model_inputs(pd.DataFrame([feats]), for_linear=True))[0, 1]
        assert abs(demo_probability(payload, feats) - expected) < 1e-6


@needs_artifacts
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_demo_matches_python():
    payload = json.loads(DEMO_JSON.read_text())
    pairs = _pairs(payload)
    script = f"""
      const m = require({json.dumps(str(config.ROOT / 'docs' / 'model.js'))});
      const data = require({json.dumps(str(DEMO_JSON))});
      const pairs = {json.dumps(pairs)};
      console.log(JSON.stringify(pairs.map(([a, b]) => m.demoProbability(data, m.demoFeatures(data, a, b)).probability)));
    """
    js = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for (a, b), p_js in zip(pairs, js):
        assert abs(p_js - demo_probability(payload, demo_features(payload, a, b))) < 1e-9
