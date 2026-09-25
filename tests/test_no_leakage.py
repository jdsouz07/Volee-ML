"""
The most important test in the project: features must never know the result.

If a match's features were built using its own result (or any later match),
the model would be cheating and the test score would be a lie. This test
builds features twice — once from ALL matches, once from only the matches
up to and including match k — and checks match k's row is identical. If any
future information leaked in, the two rows would differ.
"""

import pandas as pd

from volee_ml.features import FEATURES, build_features


def _tiny_history() -> pd.DataFrame:
    """Five made-up matches between three players, in date order."""
    rows = [
        ("m1", "2020-01-01", "p1", "p2", "6-4 6-4", 2, 0, 12, 8),
        ("m2", "2020-01-10", "p2", "p3", "6-3 3-6 6-2", 2, 1, 15, 11),
        ("m3", "2020-02-20", "p1", "p3", "7-5 6-2", 2, 0, 13, 7),
        ("m4", "2020-03-01", "p3", "p1", "6-4 6-4", 2, 0, 12, 8),
        ("m5", "2020-03-02", "p1", "p2", "6-1 6-1", 2, 0, 12, 2),
    ]
    df = pd.DataFrame(rows, columns=["match_id", "date", "winner_id", "loser_id", "score",
                                     "winner_sets", "loser_sets", "winner_games", "loser_games"])
    df["date"] = pd.to_datetime(df["date"])
    df["tour"], df["ladder"] = "atp", "Mens"
    df["winner_name"], df["loser_name"] = df["winner_id"], df["loser_id"]
    df["winner_age"], df["loser_age"] = 25.0, 27.0
    df["order"] = range(len(df))
    return df


def test_features_only_use_the_past():
    history = _tiny_history()
    full = build_features(history)
    for k in range(len(history)):
        prefix = build_features(history.iloc[: k + 1])
        pd.testing.assert_series_equal(
            full.iloc[k][list(FEATURES)], prefix.iloc[k][list(FEATURES)], check_names=False
        )


def test_first_meeting_has_no_head_to_head_and_even_form():
    row = build_features(_tiny_history()).iloc[0]
    assert row["h2h_meetings"] == 0
    assert row["form_diff"] == 0
    assert abs(row["glicko_prob"] - 0.5) < 1e-12      # two brand-new players: a coin flip


def test_label_matches_coin_flip():
    feats = build_features(_tiny_history())
    # y=1 means A won; A is the winner exactly when a_name is the recorded winner.
    history = _tiny_history()
    for (_, f), (_, m) in zip(feats.iterrows(), history.iterrows()):
        assert f["y"] == int(f["a_name"] == m["winner_id"])
