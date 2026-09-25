"""Checks the Volee adapter reads Volee's own formats correctly (volee.py)."""

import pandas as pd

from volee_ml.data import parse_score
from volee_ml.volee import normalise_volee_score, volee_to_matches


def test_volee_scores_become_pro_style():
    assert normalise_volee_score("6-4, 6-3") == "6-4 6-3"
    assert normalise_volee_score("6-4, 3-6, 10-8") == "6-4 3-6 [10-8]"       # match tiebreak
    assert normalise_volee_score("6-4, 3-6, 7-5") == "6-4 3-6 7-5"           # full third set
    assert parse_score(normalise_volee_score("6-4, 3-6, 10-8")) == (2, 1, 10, 10)


def test_winner_and_age_come_out_right():
    export = pd.DataFrame([{
        "played_at": "2026-07-21T15:00:00+00:00", "p1": "x", "p2": "y", "winner": "y",
        "p1_name": "Ann", "p2_name": "Bea", "score": "6-2, 6-1",
        "p1_dob": "2000-07-21", "p2_dob": "1996-01-01", "gender": "Womens",
    }])
    m = volee_to_matches(export).iloc[0]
    assert (m.winner_id, m.loser_id) == ("y", "x")
    assert round(m.loser_age) == 26              # Ann, born 2000-07-21, on 2026-07-21
    assert (m.winner_games, m.loser_games) == (12, 3)
