"""Checks the rating replay and the margin-aware score (ratings.py, evaluate.py)."""

import numpy as np
import pandas as pd

from volee_ml.evaluate import bootstrap_improvement
from volee_ml.features import build_features
from volee_ml.ratings import VOLEE, RatingSettings, replay, winner_score
from tests.test_no_leakage import _tiny_history


def test_margin_weight_one_is_plain_glicko():
    # w = 1 must reduce to the ordinary 1/0 score whatever the margin.
    assert winner_score(12, 2, 1.0) == 1.0
    assert winner_score(13, 12, 1.0) == 1.0


def test_margin_score_rewards_dominance():
    assert winner_score(12, 2, 0.75) > winner_score(13, 11, 0.75) > 0.75
    assert winner_score(12, 2, 0.0) == 12 / 14


def test_replay_with_volee_settings_matches_the_feature_pass():
    # Two independent code paths (features.py's loop and ratings.replay)
    # must agree on Volee's forecast for every match.
    history = _tiny_history()
    flips = np.array([f == 1 for f in build_features(history)["y"]])
    probs, _ = replay(history, VOLEE, flips)
    assert np.allclose(probs, build_features(history)["glicko_prob"].to_numpy())


def test_bootstrap_interval_contains_estimate_and_signs_are_right():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    good = np.where(y == 1, 0.7, 0.3)          # a model that's usually right
    coin = np.full(n, 0.5)                     # a coin flip
    df = pd.DataFrame({"y": y, "match_id": [f"t_{i // 20}_{i}" for i in range(n)]})
    est, lo, hi = bootstrap_improvement(df, good, coin, rounds=300)
    assert lo < est < hi and lo > 0            # the good model is clearly better
    est2, lo2, hi2 = bootstrap_improvement(df, coin, good, rounds=300)
    assert hi2 < 0                             # and the reverse is clearly worse
