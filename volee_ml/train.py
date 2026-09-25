"""
train.py — STEP 3: train the models on 2000–2019 and save them.

Run it:  python -m volee_ml.train

THE MODELS, FROM SIMPLEST TO MOST FLEXIBLE

  0. Volee's Glicko-2 (the baseline)
     No training at all — just read `glicko_prob`, the forecast Volee's
     rating system already makes. Everything else has to beat this.

  1. Logistic regression
     Learns one weight per feature and adds them up: "each extra point of
     form is worth this much, each year of age this much..." It can only
     learn straight-line effects, which makes it easy to read: the weights
     ARE the explanation. See `explain_logistic` below.

  2. Gradient boosting (HistGradientBoostingClassifier)
     Builds hundreds of small decision trees, each one fixing the mistakes of
     the ones before. It can learn bends and interactions ("rest matters,
     but only for older players"), at the cost of being harder to read.

Both learn from TRAIN only. Validation (2020–22) is for comparing them;
test (2023–26) is looked at once, in evaluate.py.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from volee_ml import config
from volee_ml.features import FEATURES, load_features, split

FEATURE_NAMES = list(FEATURES)


def to_logit(p: pd.Series) -> pd.Series:
    """Probability (0–1) -> log-odds (−∞ to +∞).

    Logistic regression adds things up on the log-odds scale, so a
    probability is more useful to it once converted. (Trees don't care.)
    """
    p = p.clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def model_inputs(df: pd.DataFrame, for_linear: bool = False) -> pd.DataFrame:
    """Select the feature columns, in a fixed order."""
    x = df[FEATURE_NAMES].copy()
    if for_linear:
        for col in ("glicko_prob", "margin_prob"):
            x[col] = to_logit(x[col])
    return x


def build_models() -> dict:
    return {
        "logistic_regression": make_pipeline(
            # Puts every feature on the same scale (mean 0, spread 1) so the
            # learned weights can be compared with each other.
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,      # small steps: slower, but more careful
            max_iter=400,            # up to 400 trees
            max_leaf_nodes=31,       # each tree stays small
            min_samples_leaf=200,    # a leaf must cover 200+ matches: stops it memorising
            l2_regularization=1.0,   # another brake on over-fitting
            early_stopping=False,    # we compare on our own time-based validation set instead
            random_state=config.RANDOM_SEED,
        ),
    }


def fit(train: pd.DataFrame) -> dict:
    models = build_models()
    models["logistic_regression"].fit(model_inputs(train, for_linear=True), train["y"])
    models["gradient_boosting"].fit(model_inputs(train), train["y"])
    return models


def predict(models: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every model's forecast that A wins, for each row of `df`."""
    return {
        "volee_glicko": df["glicko_prob"].to_numpy(),
        "standard_glicko": df["glicko_std_prob"].to_numpy(),
        "tuned_glicko": df["glicko_tuned_prob"].to_numpy(),
        "margin_glicko": df["margin_prob"].to_numpy(),
        "logistic_regression": models["logistic_regression"].predict_proba(model_inputs(df, for_linear=True))[:, 1],
        "gradient_boosting": models["gradient_boosting"].predict_proba(model_inputs(df))[:, 1],
    }


def explain_logistic(model) -> pd.Series:
    """The logistic regression's weights, biggest effect first.

    Because every feature was scaled to the same spread, a bigger number
    means a bigger influence. Positive = helps A win.
    """
    weights = model.named_steps["logisticregression"].coef_[0]
    return pd.Series(weights, index=FEATURE_NAMES).sort_values(key=abs, ascending=False)


if __name__ == "__main__":
    train, _, _ = split(load_features())
    print(f"Training on {len(train):,} matches ({config.TRAIN_YEARS[0]}–{config.TRAIN_YEARS[1]})...")
    models = fit(train)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
    print(f"Saved models to {config.MODELS_DIR}\n")
    print("Logistic regression weights (scaled; + helps A win):")
    print(explain_logistic(models["logistic_regression"]).round(3).to_string())
