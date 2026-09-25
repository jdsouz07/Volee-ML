"""Checks that tennis scores are read correctly (data.py)."""

import pytest

from volee_ml.data import parse_score


@pytest.mark.parametrize(
    "score, expected",
    [
        ("6-4 6-3", (2, 0, 12, 7)),
        ("7-6(5) 3-6 6-4", (2, 1, 16, 16)),           # tiebreak points (5) are ignored
        ("4-6 6-3 [10-8]", (2, 1, 11, 9)),            # match tiebreak counts as one set and one game
        ("6-0 6-0", (2, 0, 12, 0)),
    ],
)
def test_completed_matches(score, expected):
    assert parse_score(score) == expected


@pytest.mark.parametrize(
    "score",
    ["W/O", "6-4 2-1 RET", "6-3 DEF", "", None, "6-4", "6-4 6-3 6-2"],
)
def test_unfinished_or_not_best_of_3_is_dropped(score):
    assert parse_score(score) is None
