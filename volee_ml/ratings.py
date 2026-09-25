"""
ratings.py — replay history through a rating system with any settings.

features.py builds every model input in one careful pass. This file does one
narrower job, fast: run a Glicko-2 rating system over all matches with a
given set of settings and report its pre-match forecast for every match.
tune.py calls it dozens of times to find the best settings; features.py
calls it once more with the winners.

THE MARGIN-AWARE IDEA (this project's own rating system)
Standard Glicko-2 only hears "won" or "lost": it scores the winner 1 and the
loser 0. But 6-1 6-1 and 7-6 7-6 are very different wins, and the model
found that recent *game share* was its most useful addition to Glicko. So
what if the rating system itself heard the margin?

    score for the winner = w * 1  +  (1 - w) * (winner's share of games)

  * w = 1.0  -> ordinary Glicko-2 (only the win counts)
  * w = 0.0  -> only the games count (a 7-6 6-7 7-6 winner with fewer games
               than the loser would even be scored below 0.5)
  * in between -> a blend. tune.py finds the best w on the validation years.

The loser's score is 1 minus the winner's. Glicko-2 accepts fractional
scores (it already handles draws as 0.5), so nothing else changes. That
matters for Volee: it's a one-line change to the score passed into the
existing database function, with no model to run in the app.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volee_ml import config
from volee_ml.glicko2 import Rating, inflate_for_inactivity, update, win_probability


@dataclass(frozen=True)
class RatingSettings:
    """Everything that defines one variant of the rating system."""

    initial_rd: float = 100.0      # how unsure we are about a brand-new player
    initial_vol: float = 0.06      # starting volatility
    tau: float = 0.5               # how fast volatility can change
    margin_weight: float = 1.0     # w above: 1.0 = standard Glicko-2

    @property
    def label(self) -> str:
        return f"RD {self.initial_rd:g}, vol {self.initial_vol:g}, tau {self.tau:g}, w {self.margin_weight:g}"


VOLEE = RatingSettings()   # Volee's real settings (see config.VOLEE_GLICKO)


def winner_score(winner_games: int, loser_games: int, margin_weight: float) -> float:
    """The winner's Glicko 'score' for one match, blending win and game share."""
    share = winner_games / max(winner_games + loser_games, 1)
    return margin_weight * 1.0 + (1.0 - margin_weight) * share


def replay(matches: pd.DataFrame, settings: RatingSettings, coin_flips: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run the rating system over `matches` in order.

    Returns:
      probs   — for each match, the pre-match chance that player A wins,
                where A is the winner when coin_flips[i] is True (the same
                A/B assignment features.py uses).
      players — every player's rating after the last match (used by the demo).
    """
    players: dict[str, Rating] = defaultdict(
        lambda: Rating(1500.0, settings.initial_rd, settings.initial_vol)
    )
    last_seen: dict[str, pd.Timestamp] = {}
    probs = np.empty(len(matches))

    winners = matches["winner_id"].to_numpy()
    losers = matches["loser_id"].to_numpy()
    dates = matches["date"].to_numpy()
    w_games = matches["winner_games"].to_numpy()
    l_games = matches["loser_games"].to_numpy()

    for i in range(len(matches)):
        w_id, l_id, today = winners[i], losers[i], pd.Timestamp(dates[i])

        # Rust: time away makes the rating less certain (Volee's decay rule).
        for pid in (w_id, l_id):
            if pid in last_seen:
                idle = (today - last_seen[pid]).days
                players[pid] = inflate_for_inactivity(players[pid], idle, config.INACTIVITY_DAYS, config.MAX_RD)

        w, l = players[w_id], players[l_id]
        p_winner = win_probability(w, l)                 # forecast BEFORE the result
        probs[i] = p_winner if coin_flips[i] else 1.0 - p_winner

        s = winner_score(w_games[i], l_games[i], settings.margin_weight)
        players[w_id] = update(w, [(l, s)], settings.tau)
        players[l_id] = update(l, [(w, 1.0 - s)], settings.tau)
        last_seen[w_id] = last_seen[l_id] = today

    return probs, dict(players)
