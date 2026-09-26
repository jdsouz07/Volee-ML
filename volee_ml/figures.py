"""
figures.py — presentation charts for a portfolio, resume or slides.

Run it:  python -m volee_ml.figures      (after `make train`)

evaluate.py already draws the charts the report needs. These are the same
numbers, redrawn to be read in five seconds by someone who has never seen the
project: one message per picture, every series labelled directly, one colour
per model everywhere.

Outputs: reports/figures/*.png
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss

from volee_ml import config
from volee_ml.evaluate import bootstrap_improvement
from volee_ml.features import load_features, split
from volee_ml.train import explain_logistic, predict

OUT = config.REPORTS_DIR / "figures"

# One colour per model, in every chart. Checked for colour-blind separation;
# every series is also labelled in text, so colour is never the only cue.
COLOR = {
    "logistic_regression": "#2a78d6",
    "margin_glicko": "#eb6834",
    "gradient_boosting": "#1baf7a",
    "volee_glicko": "#8b8a86",
}
NAME = {
    "volee_glicko": "Volee's Glicko-2 (the app today)",
    "margin_glicko": "Margin-aware Glicko-2",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}
INK, SUB, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": SUB, "axes.titlecolor": INK,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 30,
    "xtick.color": SUB, "ytick.color": SUB,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.035, text, transform=ax.transAxes, color=SUB, fontsize=11, va="bottom")


def footer(fig, text: str = "Volee-ML · test set: 16,107 pro matches, 2023–2026, never used for training or tuning") -> None:
    fig.text(0.012, 0.012, text, color=SUB, fontsize=8.5)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
# 1. The headline: how much better than the app, and is it luck?
# ---------------------------------------------------------------------------
def headline(test: pd.DataFrame, preds: dict) -> None:
    keys = ["logistic_regression", "gradient_boosting", "margin_glicko"]
    rows = [(k, *bootstrap_improvement(test, preds[k], preds["volee_glicko"])) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (k, est, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo * 100, hi * 100], [y, y], color=COLOR[k], linewidth=6, solid_capstyle="round", alpha=0.35)
        ax.plot(est * 100, y, "o", color=COLOR[k], markersize=12, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi * 100 + 0.07, y, f"+{est:.2%}", va="center", color=INK, fontsize=12, fontweight="bold")
        ax.text(hi * 100 + 0.07, y - 0.28, f"95% CI +{lo:.2%} to +{hi:.2%}", va="center", color=SUB, fontsize=9)
    ax.axvline(0, color=SUB, linewidth=1)
    ax.text(0.04, -0.52, "0 = the app today", color=SUB, fontsize=9, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAME[k] for k, *_ in reversed(rows)], fontsize=11, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.1, 2.9)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Log-loss improvement over the app's rating system (%)  →  better")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("ML beats the app's rating system, and it isn't luck")
    subtitle(ax, "Dot = improvement on the held-out test set · bar = 95% tournament-level bootstrap interval")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.8, bottom=0.2)
    footer(fig)
    save(fig, "1_headline_improvement.png")


# ---------------------------------------------------------------------------
# 2. Where it matters for a club app: players the rating system isn't sure about
# ---------------------------------------------------------------------------
def new_players(test: pd.DataFrame, preds: dict) -> None:
    unsure = ((test["rd_a"] > 90) | (test["rd_b"] > 90)).to_numpy()
    keys = ["margin_glicko", "gradient_boosting", "logistic_regression"]
    base_all = log_loss(test["y"], preds["volee_glicko"])
    base_new = log_loss(test["y"][unsure], preds["volee_glicko"][unsure])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    groups = ["All test matches", f"Uncertain ratings\n(new or returning players, {unsure.sum():,} matches)"]
    width = 0.25
    for i, k in enumerate(keys):
        vals = [(base_all - log_loss(test["y"], preds[k])) / base_all * 100,
                (base_new - log_loss(test["y"][unsure], preds[k][unsure])) / base_new * 100]
        x = np.arange(2) + (i - 1) * (width + 0.02)
        ax.bar(x, vals, width, color=COLOR[k], edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.06, f"+{v:.1f}%", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(range(2))
    ax.set_xticklabels(groups, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Log-loss improvement vs the app (%)")
    ax.set_ylim(0, 3.7)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("The gain is biggest for newer players")
    subtitle(ax, "Most players in a club ladder are new or play rarely, which is exactly where plain Glicko-2 is weakest")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in keys]
    ax.legend(handles, [NAME[k] for k in keys], frameon=False, loc="upper left", fontsize=10)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.2)
    footer(fig)
    save(fig, "2_newer_players.png")


# ---------------------------------------------------------------------------
# 3. Every year, not one lucky one
# ---------------------------------------------------------------------------
def by_year(both: pd.DataFrame, preds: dict) -> None:
    keys = ["logistic_regression", "margin_glicko"]
    years = sorted(both["year"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axvspan(len([y for y in years if y < config.TEST_YEARS[0]]) - 0.5, len(years) - 0.5, color="#f0efec", zorder=0)
    for i, k in enumerate(keys):
        est, lo, hi = [], [], []
        for year in years:
            m = (both["year"] == year).to_numpy()
            e, l, h = bootstrap_improvement(both[m], preds[k][m], preds["volee_glicko"][m], rounds=1000)
            est.append(e * 100); lo.append(l * 100); hi.append(h * 100)
        x = np.arange(len(years)) + (i - 0.5) * 0.22
        ax.vlines(x, lo, hi, color=COLOR[k], linewidth=3, alpha=0.4)
        ax.plot(x, est, "o", color=COLOR[k], markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(x[-1] + 0.18, est[-1], NAME[k], color=INK, fontsize=10, va="center")
    ax.axhline(0, color=SUB, linewidth=1)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], color=INK)
    ax.text(len(years) - 0.55, 3.95, "held-out test years", ha="right", color=SUB, fontsize=9)
    ax.text(-0.45, 3.95, "validation years", color=SUB, fontsize=9)
    ax.set_xlim(-0.6, len(years) + 1.7)
    ax.set_ylim(-0.4, 4.2)
    ax.set_ylabel("Log-loss improvement vs the app (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("It wins every year, not just once")
    subtitle(ax, "Improvement over the app's Glicko-2 by year, with 95% intervals · above 0 = better than the app")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.12)
    footer(fig, "Volee-ML · 2020–2022 used to compare models, 2023–2026 held out")
    save(fig, "3_every_year.png")


# ---------------------------------------------------------------------------
# 4. Calibration: when it says 70%, it happens 70% of the time
# ---------------------------------------------------------------------------
def calibration(test: pd.DataFrame, preds: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.plot([0.1, 0.9], [0.1, 0.9], "--", color=SUB, linewidth=1)
    ax.text(0.86, 0.9, "perfect", color=SUB, fontsize=9, ha="right", va="bottom", rotation=45, rotation_mode="anchor")
    for k in ("volee_glicko", "logistic_regression"):
        frac, mean = calibration_curve(test["y"], preds[k], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "-o", color=COLOR[k], linewidth=2, markersize=7, markeredgecolor=SURFACE, label=NAME[k])
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9)
    ax.set_xlabel("Forecast chance of winning")
    ax.set_ylabel("How often that player actually won")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Its probabilities mean what they say")
    subtitle(ax, "Test matches grouped by forecast · on the line = well calibrated")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.85, bottom=0.12)
    footer(fig, "Volee-ML · 16,107 held-out matches, 2023–2026")
    save(fig, "4_calibration.png")


# ---------------------------------------------------------------------------
# 5. What the model learned
# ---------------------------------------------------------------------------
PLAIN = {
    "games_share_diff": "How convincingly they've been winning",
    "margin_prob": "Margin-aware rating forecast",
    "glicko_prob": "App's rating forecast",
    "rating_diff": "Rating gap",
    "form_diff": "Recent form",
    "momentum_diff": "Momentum",
    "experience_diff": "Experience gap",
    "age_a": "Age (player A)", "age_b": "Age (player B)",
    "rest_days_a": "Days of rest (A)", "rest_days_b": "Days of rest (B)",
    "rd_a": "Rating uncertainty (A)", "rd_b": "Rating uncertainty (B)",
    "h2h_meetings": "Head-to-head meetings", "h2h_edge": "Head-to-head edge",
    "is_womens": "Women's match",
}


def features_chart(model) -> None:
    weights = explain_logistic(model)
    top = weights.reindex(weights.abs().sort_values(ascending=False).index)[:8][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(range(len(top)), top.abs().to_numpy(), color=COLOR["logistic_regression"], height=0.62,
            edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(top.abs().to_numpy()):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([PLAIN.get(f, f) for f in top.index], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Size of the model's weight (features scaled to the same spread)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("What the model learned: how you win matters")
    subtitle(ax, "Largest weights in the logistic regression · only data the app already collects")
    fig.subplots_adjust(left=0.33, right=0.97, top=0.82, bottom=0.14)
    footer(fig, "Volee-ML · trained on 2000–2019, 16 features")
    save(fig, "5_what_it_learned.png")


# ---------------------------------------------------------------------------
# 6. One summary card
# ---------------------------------------------------------------------------
def summary_card(test: pd.DataFrame, preds: dict) -> None:
    y = test["y"]
    unsure = ((test["rd_a"] > 90) | (test["rd_b"] > 90)).to_numpy()
    est, lo, hi = bootstrap_improvement(test, preds["logistic_regression"], preds["volee_glicko"])
    base_new = log_loss(y[unsure], preds["volee_glicko"][unsure])
    new_gain = (base_new - log_loss(y[unsure], preds["logistic_regression"][unsure])) / base_new
    acc_base = accuracy_score(y, preds["volee_glicko"] > 0.5)
    acc_lr = accuracy_score(y, preds["logistic_regression"] > 0.5)

    tiles = [
        ("164k", "pro matches trained on,\nlimited to data the app collects"),
        (f"+{est:.2%}", f"log loss vs the app's Glicko-2\n95% CI +{lo:.2%} to +{hi:.2%}"),
        (f"+{new_gain:.1%}", "for newer players with\nuncertain ratings"),
        (f"{acc_lr:.1%}", f"accuracy on 16,107 held-out\nmatches, up from {acc_base:.1%}"),
    ]
    fig = plt.figure(figsize=(12, 4.2))
    fig.text(0.04, 0.86, "Volee-ML: forecasting tennis matches better than the app's rating system",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(0.04, 0.78, "Python · scikit-learn · SQL · deployed in shadow mode in the app's production database",
             fontsize=11, color=SUB)
    for i, (big, small) in enumerate(tiles):
        x = 0.04 + i * 0.235
        fig.patches.append(plt.Rectangle((x, 0.14), 0.215, 0.52, transform=fig.transFigure,
                                         facecolor="#ffffff", edgecolor=GRID, linewidth=1.2))
        fig.text(x + 0.018, 0.46, big, fontsize=24, fontweight="bold",
                 color=COLOR["logistic_regression"] if i else INK)
        fig.text(x + 0.018, 0.22, small, fontsize=10.5, color=SUB, linespacing=1.4)
    save(fig, "0_summary_card.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    features = load_features()
    _, valid, test = split(features)
    models = {n: joblib.load(config.MODELS_DIR / f"{n}.joblib") for n in ("logistic_regression", "gradient_boosting")}
    valid_preds, test_preds = predict(models, valid), predict(models, test)
    both = pd.concat([valid, test])
    both_preds = {k: np.concatenate([valid_preds[k], test_preds[k]]) for k in valid_preds}

    summary_card(test, test_preds)
    headline(test, test_preds)
    new_players(test, test_preds)
    by_year(both, both_preds)
    calibration(test, test_preds)
    features_chart(models["logistic_regression"])
