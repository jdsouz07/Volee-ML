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

IS THE IMPROVEMENT REAL? (confidence intervals)
A test set is one sample of history. If we'd happened to get slightly
different matches, would the model still win? The BOOTSTRAP answers that
without collecting new data: re-draw the test set many times from itself,
recompute the improvement each time, and see how much it wobbles. The middle
95% of those results is the 95% confidence interval. If the whole interval
is above zero, the improvement isn't luck.

One refinement: matches in the same tournament aren't independent (same
week, same conditions, same players). So we re-draw whole TOURNAMENTS, not
single matches. That's a "cluster bootstrap", and it gives honestly wider
intervals.

We also check CALIBRATION: of all the matches a model called 70%, did about
70% actually happen? That's what `calibration.png` shows.

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
    "tuned_glicko": "Glicko-2, tuned on validation",
    "margin_glicko": "Margin-aware Glicko-2 (new)",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}
BOOTSTRAP_ROUNDS = 2000


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def per_match_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def tournament_of(match_ids: pd.Series) -> np.ndarray:
    """'atp_2024-0339_300' -> 'atp_2024-0339': drop the match number."""
    return match_ids.str.rsplit("_", n=1).str[0].to_numpy()


def bootstrap_improvement(df: pd.DataFrame, p_new: np.ndarray, p_base: np.ndarray,
                          rounds: int = BOOTSTRAP_ROUNDS, seed: int = config.RANDOM_SEED) -> tuple[float, float, float]:
    """% log-loss improvement of p_new over p_base, with a 95% cluster-bootstrap interval.

    Returns (estimate, low, high). Positive means p_new is better.
    """
    y = df["y"].to_numpy()
    loss_new, loss_base = per_match_log_loss(y, p_new), per_match_log_loss(y, p_base)
    _, cluster = np.unique(tournament_of(df["match_id"]), return_inverse=True)
    n_clusters = cluster.max() + 1
    # Per-tournament totals, so one re-draw is a cheap weighted sum.
    sum_new = np.bincount(cluster, loss_new, n_clusters)
    sum_base = np.bincount(cluster, loss_base, n_clusters)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(rounds, n_clusters))       # tournaments, with replacement
    counts = np.apply_along_axis(np.bincount, 1, picks, minlength=n_clusters)
    new_totals, base_totals = counts @ sum_new, counts @ sum_base
    improvements = (base_totals - new_totals) / base_totals

    estimate = (loss_base.sum() - loss_new.sum()) / loss_base.sum()
    low, high = np.percentile(improvements, [2.5, 97.5])
    return estimate, low, high


def ci_table(df: pd.DataFrame, preds: dict) -> str:
    """Improvement over Volee's Glicko-2 with its 95% interval.

    (There's no separate column for the tuned Glicko: tune.py found Volee's
    current settings are already the best plain-Glicko settings, so the two
    baselines give the same forecasts to five decimal places.)
    """
    rows = ["| Model | Log-loss improvement vs Volee's Glicko-2 | 95% interval |", "|---|---|---|"]
    for key in ("margin_glicko", "logistic_regression", "gradient_boosting"):
        est, lo, hi = bootstrap_improvement(df, preds[key], preds["volee_glicko"])
        rows.append(f"| {LABELS[key]} | {est:+.2%} | {lo:+.2%} to {hi:+.2%} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_calibration(df: pd.DataFrame, preds: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    for key in ("volee_glicko", "margin_glicko", "logistic_regression"):
        frac, mean = calibration_curve(df["y"], preds[key], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=LABELS[key])
    ax.set_xlabel("Forecast chance that A wins")
    ax.set_ylabel("How often A actually won")
    ax.set_title("Calibration on the test set (2023–2026)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_by_year(df: pd.DataFrame, preds: dict, path) -> pd.DataFrame:
    """Improvement over Volee's Glicko-2, one bar per year, with 95% intervals.

    A result that only holds in one lucky year isn't a result. Validation
    years are shown lighter: the models were compared on them, so the test
    years (darker) are the stricter evidence.
    """
    rows = []
    for year in sorted(df["year"].unique()):
        mask = (df["year"] == year).to_numpy()
        part = df[mask]
        for key in ("margin_glicko", "logistic_regression"):
            est, lo, hi = bootstrap_improvement(part, preds[key][mask], preds["volee_glicko"][mask], rounds=1000)
            rows.append({"year": year, "model": key, "est": est, "lo": lo, "hi": hi})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    for i, key in enumerate(("margin_glicko", "logistic_regression")):
        part = table[table["model"] == key]
        x = np.arange(len(part)) + (i - 0.5) * width
        colors = ["#b9cfcb" if y < config.TEST_YEARS[0] else "#4b7f78" for y in part["year"]]
        if key == "logistic_regression":
            colors = ["#d9c2a0" if y < config.TEST_YEARS[0] else "#b07a2c" for y in part["year"]]
        ax.bar(x, part["est"] * 100, width, color=colors, label=LABELS[key])
        ax.errorbar(x, part["est"] * 100, yerr=[(part["est"] - part["lo"]) * 100, (part["hi"] - part["est"]) * 100],
                    fmt="none", color="black", capsize=3, linewidth=1)
    years = sorted(table["year"].unique())
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([f"{y}\n{'test' if y >= config.TEST_YEARS[0] else 'valid'}" for y in years])
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Log-loss improvement vs Volee (%)")
    ax.set_title("Improvement over Volee's Glicko-2, by year (95% intervals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return table


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


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features = load_features()
    _, valid, test = split(features)
    models = {name: joblib.load(config.MODELS_DIR / f"{name}.joblib")
              for name in ("logistic_regression", "gradient_boosting")}

    valid_preds, test_preds = predict(models, valid), predict(models, test)
    valid_table, test_table = score_table(valid, valid_preds), score_table(test, test_preds)

    both = pd.concat([valid, test])
    both_preds = {k: np.concatenate([valid_preds[k], test_preds[k]]) for k in valid_preds}

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration(test, test_preds, config.REPORTS_DIR / "calibration.png")
    plot_by_year(both, both_preds, config.REPORTS_DIR / "by_year.png")
    importance = plot_importance(models["gradient_boosting"], valid, config.REPORTS_DIR / "feature_importance.png")
    weights = explain_logistic(models["logistic_regression"])

    # Players the rating system is unsure about: new or returning.
    newcomer = (test["rd_a"] > 90) | (test["rd_b"] > 90)

    report = f"""# Results

Generated by `python -m volee_ml.evaluate`. Log loss is the main metric (lower is better).

## Test set — {config.TEST_YEARS[0]}–{config.TEST_YEARS[1]}, looked at once

{markdown(test_table)}

## Is it real? 95% confidence intervals on the test set

Log-loss improvement, with a tournament-level cluster bootstrap ({BOOTSTRAP_ROUNDS:,} re-draws).
An interval entirely above zero means the gain is not luck.

{ci_table(test, test_preds)}

![Improvement by year](by_year.png)

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
    print()
    print(ci_table(test, test_preds))
    print(f"\nWrote {config.REPORTS_DIR / 'results.md'}")
