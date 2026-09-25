"""
Proves glicko2.py matches Volee.

The expected numbers below were produced by Volee's real database function
`glicko2_update_after_period` on 2026-09-24. If this test passes, the
baseline in this project is the exact rating system the app runs.
"""

import pytest

from volee_ml.glicko2 import Rating, update, win_probability, inflate_for_inactivity


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_glickmans_paper_example():
    # Glickman's own worked example: a 1500 player (RD 200) beats a 1400,
    # loses to a 1550 and a 1700. Paper answer: 1464.06 / 151.52 / 0.05999.
    # Volee's SQL returned exactly these digits:
    new = update(
        Rating(1500, 200, 0.06),
        [(Rating(1400, 30), 1), (Rating(1550, 100), 0), (Rating(1700, 300), 0)],
    )
    assert close(new.rating, 1464.0506705393013)
    assert close(new.rd, 151.5165241238573)
    assert close(new.vol, 0.05999598428648851, 1e-9)


@pytest.mark.parametrize(
    "player, opponent, score, expected",
    [
        # (rating, rd, vol) of player, (rating, rd) of opponent, result, Volee's answer
        ((350, 100, 0.06), (420, 100), 1, (380.7385165657183, 97.0459853378623, 0.06000101325916215)),
        ((500, 80, 0.06), (430, 150), 0, (480.8562464400951, 79.01329291133162, 0.06000097078220604)),
        ((300, 350, 0.06), (300, 100), 1, (474.9363013666434, 252.52055490781143, 0.0599992347159252)),
    ],
)
def test_matches_volee_database(player, opponent, score, expected):
    new = update(Rating(*player), [(Rating(*opponent), score)])
    assert close(new.rating, expected[0])
    assert close(new.rd, expected[1])
    assert close(new.vol, expected[2], 1e-9)


def test_win_probability_is_symmetric_and_sensible():
    a, b = Rating(1700, 60), Rating(1500, 60)
    p = win_probability(a, b)
    assert 0.5 < p < 1.0                                  # the better player is favoured
    assert close(p + win_probability(b, a), 1.0)          # A-beats-B + B-beats-A = 100%
    assert close(win_probability(a, a), 0.5)              # equal players = coin flip


def test_uncertainty_pulls_forecast_towards_50_50():
    sure = win_probability(Rating(1700, 50), Rating(1500, 50))
    unsure = win_probability(Rating(1700, 300), Rating(1500, 300))
    assert sure > unsure > 0.5


def test_inactivity_grows_rd_in_4_week_steps_and_caps():
    p = Rating(1600, 80, 0.06)
    assert inflate_for_inactivity(p, 27).rd == 80          # under 4 weeks: no change
    assert inflate_for_inactivity(p, 28).rd > 80           # one step
    # Each step adds only (0.06 * 173.7)^2 to RD^2, so growth is slow: about
    # 10 years away takes RD from 80 to only about 143. It can never pass 350.
    assert 130 < inflate_for_inactivity(p, 3650).rd < 160
    assert inflate_for_inactivity(p, 10_000_000).rd == 350  # capped like Volee
