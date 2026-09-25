"""
evaluate.py — STEP 4: measure every model honestly and write the report.

Run it:  python -m volee_ml.evaluate

HOW WE SCORE A FORECAST
  * Log loss   — THE main number. Punishes confident wrong answers hard:
                 saying 90% and being wrong costs far more than saying 55%
                 and being wrong. Lower is better. A coin flip scores 0.693.
  * Brier      — average squared gap between forecast and result. Lower is
                 better. A coin flip scores 0.25.
  * Accuracy   — how often the favourite won. Easy to understand, but it
                 ignores confidence, so it's not what we optimise.
  * AUC        — how well the model ranks likely winners above likely
                 losers. 0.5 = random, 1.0 = perfect.

We also check CALIBRATION: of all the matches a model called 70%, did about
70% actually happen? A well-calibrated model's forecasts can be shown to
users as real percentages. That's what `calibration.png` shows.

Outputs (committed to the repo): reports/results.md, reports/*.png
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from volee_ml import config
from volee_ml.features import FEATURES, load_features, split
from volee_ml.train import explain_logistic, model_inputs, predict

LABELS = {
    "volee_glicko": "Volee's Glicko-2 (baseline)",
    "standard_glicko": "Glicko-2, textbook start values",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}


def score(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, p > 0.5),
        "auc": roc_auc_score(y, p),
    }


def score_table(df: pd.DataFrame, preds: dict) -> pd.DataFrame:
    table = pd.DataFrame({LABELS[k]: score(df["y"].to_numpy(), p) for k, p in preds.items()}).T
    base = table.loc[LABELS["volee_glicko"], "log_loss"]
    table["vs_baseline"] = (base - table["log_loss"]) / base   # % log-loss improvement
    return table


def markdown(table: pd.DataFrame) -> str:
    lines = ["| Model | Log loss | Brier | Accuracy | AUC | Log loss vs Volee |",
             "|---|---|---|---|---|---|"]
    for name, r in table.iterrows():
        lines.append(f"| {name} | {r.log_loss:.4f} | {r.brier:.4f} | {r.accuracy:.1%} | {r.auc:.4f} | {r.vs_baseline:+.1%} |")
    return "\n".join(lines)


def plot_calibration(df: pd.DataFrame, preds: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    for key in ("volee_glicko", "logistic_regression", "gradient_boosting"):
        frac, mean = calibration_curve(df["y"], preds[key], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=LABELS[key])
    ax.set_xlabel("Forecast chance that A wins")
    ax.set_ylabel("How often A actually won")
    ax.set_title("Calibration on the test set (2023–2026)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(model, df: pd.DataFrame, path) -> pd.Series:
    """Permutation importance: shuffle one feature, see how much worse the
    model gets. A feature the model leans on hurts a lot when scrambled."""
    result = permutation_importance(
        model, model_inputs(df), df["y"], scoring="neg_log_loss",
        n_repeats=5, random_state=config.RANDOM_SEED,
    )
    importance = pd.Series(result.importances_mean, index=list(FEATURES)).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.plot.barh(ax=ax, color="#4b7f78")
    ax.set_xlabel("Log-loss increase when the feature is shuffled")
    ax.set_title("What the gradient-boosting model relies on (validation set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return importance.sort_values(ascending=False)


def subgroup(df: pd.DataFrame, preds: dict, mask: pd.Series, name: str) -> str:
    part = df[mask]
    sub = {k: v[mask.to_numpy()] for k, v in preds.items()}
    return f"**{name}** ({len(part):,} matches)\n\n" + markdown(score_table(part, sub))


if __name__ == "__main__":
    features = load_features()
    _, valid, test = split(features)
    models = {name: joblib.load(config.MODELS_DIR / f"{name}.joblib")
              for name in ("logistic_regression", "gradient_boosting")}

    valid_preds, test_preds = predict(models, valid), predict(models, test)
    valid_table, test_table = score_table(valid, valid_preds), score_table(test, test_preds)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration(test, test_preds, config.REPORTS_DIR / "calibration.png")
    importance = plot_importance(models["gradient_boosting"], valid, config.REPORTS_DIR / "feature_importance.png")
    weights = explain_logistic(models["logistic_regression"])

    # Where do the models disagree most with Glicko? Players Glicko is unsure about.
    newcomer = (test["rd_a"] > 90) | (test["rd_b"] > 90)

    report = f"""# Results

Generated by `python -m volee_ml.evaluate`. Log loss is the main metric (lower is better).

## Test set — {config.TEST_YEARS[0]}–{config.TEST_YEARS[1]}, looked at once

{markdown(test_table)}

## Validation set — {config.VALID_YEARS[0]}–{config.VALID_YEARS[1]}, used to compare models

{markdown(valid_table)}

## One slice of the test set

{subgroup(test, test_preds, newcomer, "Matches where at least one player's rating is still uncertain (RD over 90)")}

## What the models learned

Gradient boosting, permutation importance on validation (log-loss increase when shuffled):

| Feature | Importance |
|---|---|
""" + "\n".join(f"| {k} | {v:.4f} |" for k, v in importance.items()) + """

Logistic regression weights (features scaled to the same spread; positive helps A win):

| Feature | Weight |
|---|---|
""" + "\n".join(f"| {k} | {v:+.3f} |" for k, v in weights.items()) + """

![Calibration](calibration.png)
![Feature importance](feature_importance.png)
"""
    (config.REPORTS_DIR / "results.md").write_text(report)
    print(markdown(test_table))
    print(f"\nWrote {config.REPORTS_DIR / 'results.md'}")
