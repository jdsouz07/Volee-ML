"""
features.py — STEP 2: turn match results into model-ready rows.

Run it:  python -m volee_ml.features

THE CORE IDEA
A model can't read "Nadal vs Federer". It needs numbers that describe the
two players *as they were just before the match*. So we replay history one
match at a time, like watching every match since 1991 in order:

    for each match, oldest first:
        1. look up both players' current state (rating, recent results, ...)
        2. write down the FEATURES — what we knew before the first point
        3. only then learn the result and update both players' state

Step 2 always happens before step 3. If it didn't, the model would be
peeking at the answer ("data leakage") and would look far better in testing
than it could ever be in real life. `tests/test_no_leakage.py` checks this.

WHY A COIN FLIP
The raw data always lists the winner first. If player A were always the
winner, the model would just learn "A wins". So for each match we flip a
(repeatable) coin to decide who is "A" and who is "B", and the label is
y = 1 if A won. About half the rows end up with y = 1.

EVERY FEATURE HAS A VOLEE EQUIVALENT
Each one is computed only from dates, results, scores, ages and ratings —
all things Volee stores. See the FEATURES dictionary below.
"""

from __future__ import annotations

import math
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

from volee_ml import config
from volee_ml.data import load_matches
from volee_ml.glicko2 import Rating, inflate_for_inactivity, update, win_probability

# The features the models are allowed to use, and what each one means.
# (A = the player we're predicting for, B = their opponent.)
FEATURES = {
    "glicko_prob":      "Volee's own Glicko-2 forecast that A wins (0-1). Also the baseline.",
    "rating_diff":      "A's rating minus B's rating, in Glicko points.",
    "rd_a":             "How unsure A's rating is (Glicko RD). New or rusty players are high.",
    "rd_b":             "Same for B.",
    "experience_diff":  "log(1 + matches A has played) minus the same for B.",
    "rest_days_a":      "log(1 + days since A's last match), capped at a year.",
    "rest_days_b":      "Same for B.",
    "form_diff":        "A's win rate over their last 10 matches minus B's (smoothed towards 50%).",
    "games_share_diff": "Share of games A won over their last 10 matches, minus B's. Captures dominance, not just wins.",
    "momentum_diff":    "How much A's rating moved over their last 5 matches, minus B's.",
    "h2h_edge":         "Head-to-head record between these two, smoothed: (A wins - B wins) / (meetings + 2).",
    "h2h_meetings":     "log(1 + times they've played each other before).",
    "age_a":            "A's age in years (Volee: from date of birth).",
    "age_b":            "Same for B.",
    "is_womens":        "1 for the women's ladder, 0 for the men's.",
}


@dataclass
class PlayerState:
    """Everything we remember about one player between matches."""

    volee: Rating = field(default_factory=lambda: Rating(**_rating_args(config.VOLEE_GLICKO)))
    standard: Rating = field(default_factory=lambda: Rating(**_rating_args(config.STANDARD_GLICKO)))
    last_date: pd.Timestamp | None = None
    matches_played: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=config.FORM_WINDOW))      # (won, games_won, games_total)
    rating_history: deque = field(default_factory=lambda: deque(maxlen=config.MOMENTUM_WINDOW + 1))


def _rating_args(params: dict) -> dict:
    return {"rating": params["initial_rating"], "rd": params["initial_rd"], "vol": params["initial_vol"]}


def coin_flip(match_id: str) -> bool:
    """Repeatable 50/50 decision: is the winner listed as player A?

    Uses a checksum of the match id rather than random numbers, so the same
    match always gets the same answer no matter how the data is sliced.
    """
    return zlib.crc32(match_id.encode()) % 2 == 0


# ---------------------------------------------------------------------------
# Small helpers, one per feature family
# ---------------------------------------------------------------------------

def _rest_days(state: PlayerState, today: pd.Timestamp) -> float:
    if state.last_date is None:
        days = config.DAYS_CAP                       # never played: treat as long rest
    else:
        days = min((today - state.last_date).days, config.DAYS_CAP)
    return math.log1p(max(days, 0))


def _form(state: PlayerState) -> float:
    """Win rate over the recent window, pulled towards 50% when there's little data.

    With 0 matches this is exactly 0.5; with 10 it's mostly the real rate.
    (Adding 2.5 wins and 2.5 losses is a simple 'prior'.)
    """
    wins = sum(won for won, _, _ in state.recent)
    return (wins + 2.5) / (len(state.recent) + 5)


def _games_share(state: PlayerState) -> float:
    won = sum(g for _, g, _ in state.recent)
    total = sum(t for _, _, t in state.recent)
    return (won + 6) / (total + 12)                  # same idea: starts at 50%


def _momentum(state: PlayerState) -> float:
    if len(state.rating_history) < 2:
        return 0.0
    return state.rating_history[-1] - state.rating_history[0]


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------

def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Replay `matches` in order and return one feature row per match."""
    players: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h: dict[tuple[str, str], int] = defaultdict(int)   # (winner, loser) -> times
    rows = []

    for m in matches.itertuples(index=False):
        winner, loser = players[m.winner_id], players[m.loser_id]

        # Time away makes us less sure of a rating (Volee's decay rule).
        for state in (winner, loser):
            if state.last_date is not None:
                idle = (m.date - state.last_date).days
                state.volee = inflate_for_inactivity(state.volee, idle, config.INACTIVITY_DAYS, config.MAX_RD)
                state.standard = inflate_for_inactivity(state.standard, idle, config.INACTIVITY_DAYS, config.MAX_RD)

        # ---- 1 + 2: who is A, and what did we know before the match? ----
        a_is_winner = coin_flip(m.match_id)
        if a_is_winner:
            a_id, b_id, a, b = m.winner_id, m.loser_id, winner, loser
            a_age, b_age = m.winner_age, m.loser_age
        else:
            a_id, b_id, a, b = m.loser_id, m.winner_id, loser, winner
            a_age, b_age = m.loser_age, m.winner_age

        a_wins_vs_b, b_wins_vs_a = h2h[(a_id, b_id)], h2h[(b_id, a_id)]
        rows.append({
            # bookkeeping (not given to the model)
            "match_id": m.match_id,
            "date": m.date,
            "year": m.date.year,
            "tour": m.tour,
            "a_name": m.winner_name if a_is_winner else m.loser_name,
            "b_name": m.loser_name if a_is_winner else m.winner_name,
            "glicko_std_prob": win_probability(a.standard, b.standard),   # reference baseline only
            # features
            "glicko_prob": win_probability(a.volee, b.volee),
            "rating_diff": a.volee.rating - b.volee.rating,
            "rd_a": a.volee.rd,
            "rd_b": b.volee.rd,
            "experience_diff": math.log1p(a.matches_played) - math.log1p(b.matches_played),
            "rest_days_a": _rest_days(a, m.date),
            "rest_days_b": _rest_days(b, m.date),
            "form_diff": _form(a) - _form(b),
            "games_share_diff": _games_share(a) - _games_share(b),
            "momentum_diff": _momentum(a) - _momentum(b),
            "h2h_edge": (a_wins_vs_b - b_wins_vs_a) / (a_wins_vs_b + b_wins_vs_a + 2),
            "h2h_meetings": math.log1p(a_wins_vs_b + b_wins_vs_a),
            "age_a": a_age if pd.notna(a_age) else config.DEFAULT_AGE,
            "age_b": b_age if pd.notna(b_age) else config.DEFAULT_AGE,
            "is_womens": int(m.ladder == "Womens"),
            # the answer
            "y": int(a_is_winner),
        })

        # ---- 3: NOW learn the result and update both players ----
        new_w_volee = update(winner.volee, [(loser.volee, 1)], config.VOLEE_GLICKO["tau"])
        new_l_volee = update(loser.volee, [(winner.volee, 0)], config.VOLEE_GLICKO["tau"])
        new_w_std = update(winner.standard, [(loser.standard, 1)], config.STANDARD_GLICKO["tau"])
        new_l_std = update(loser.standard, [(winner.standard, 0)], config.STANDARD_GLICKO["tau"])
        # (Both new ratings are computed from the OLD ratings before either is
        # saved — otherwise the loser's update would see the winner's new
        # rating. Volee's SQL does the same.)
        winner.volee, loser.volee = new_w_volee, new_l_volee
        winner.standard, loser.standard = new_w_std, new_l_std

        total_games = m.winner_games + m.loser_games
        winner.recent.append((1, m.winner_games, total_games))
        loser.recent.append((0, m.loser_games, total_games))
        for state in (winner, loser):
            state.matches_played += 1
            state.last_date = m.date
            state.rating_history.append(state.volee.rating)
        h2h[(m.winner_id, m.loser_id)] += 1

    return pd.DataFrame(rows)


def split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut the rows into train / validation / test by year (see config.py)."""
    def between(years):
        return features[(features["year"] >= years[0]) & (features["year"] <= years[1])]
    return between(config.TRAIN_YEARS), between(config.VALID_YEARS), between(config.TEST_YEARS)


def load_features() -> pd.DataFrame:
    return pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])


if __name__ == "__main__":
    features = build_features(load_matches())
    features.to_csv(config.FEATURES_CSV, index=False)
    train, valid, test = split(features)
    print(f"Saved {len(features):,} rows to {config.FEATURES_CSV}")
    print(f"  train {len(train):,}  |  validation {len(valid):,}  |  test {len(test):,}")
    print(f"  share where A won: {features['y'].mean():.3f} (should be close to 0.5)")
