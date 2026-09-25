"""
glicko2.py — the Glicko-2 rating system, written to match Volee exactly.

WHAT GLICKO-2 IS, IN ONE PARAGRAPH
Every player has three numbers:
  * rating      — how good we think they are (1500 is average here)
  * RD          — "rating deviation": how UNSURE we are. A new player has a
                  big RD; someone who plays every week has a small one.
  * volatility  — how erratic their results are.
After a match, the winner's rating goes up and the loser's goes down. HOW
MUCH depends on how surprising the result was (beating a much better player
moves you a lot) and on the RD (we change our mind faster about players
we're unsure of). Elo is the simpler ancestor; Glicko adds the RD and
volatility.

WHY IT'S HERE
Volee already uses Glicko-2 to rank players. It's the thing our machine-
learning model has to beat, so it has to be Volee's version, not "a"
Glicko. `tests/test_glicko2.py` feeds this file the same inputs Volee's
database function received and checks the outputs match to many decimals.

The maths follows Glickman's paper "Example of the Glicko-2 system"
(http://www.glicko.net/glicko/glicko2.pdf). Step numbers in the comments
refer to that paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Glicko-2 does its maths on a squashed scale. This constant converts between
# "rating points" (1500, 1650, ...) and that internal scale. Volee uses the
# same value.
SCALE = 173.7178
CONVERGENCE = 0.000001  # stop the volatility search when this precise (Volee: v_eps)


@dataclass
class Rating:
    """One player's Glicko-2 state, in ordinary rating points."""

    rating: float = 1500.0
    rd: float = 100.0
    vol: float = 0.06


# ---------------------------------------------------------------------------
# The two small helper functions everything else is built from
# ---------------------------------------------------------------------------

def _g(phi: float) -> float:
    """Shrinks the impact of an opponent we're unsure about (paper step 3).

    If we barely know how good the opponent is (big phi), beating them
    tells us less, so g() is smaller.
    """
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))


def _expected(mu: float, mu_opp: float, phi_opp: float) -> float:
    """Chance of beating this opponent, on Glicko's internal scale (step 3)."""
    return 1.0 / (1.0 + math.exp(-_g(phi_opp) * (mu - mu_opp)))


# ---------------------------------------------------------------------------
# Predicting a match BEFORE it's played
# ---------------------------------------------------------------------------

def win_probability(a: Rating, b: Rating) -> float:
    """Glicko-2's forecast that player A beats player B.

    This is our BASELINE: the prediction Volee's rating system already
    makes. It uses both players' uncertainty — if either rating is shaky,
    the forecast is pulled towards 50/50.
    """
    mu_a = (a.rating - 1500.0) / SCALE
    mu_b = (b.rating - 1500.0) / SCALE
    phi_combined = math.sqrt((a.rd / SCALE) ** 2 + (b.rd / SCALE) ** 2)
    return 1.0 / (1.0 + math.exp(-_g(phi_combined) * (mu_a - mu_b)))


# ---------------------------------------------------------------------------
# Updating ratings AFTER a match
# ---------------------------------------------------------------------------

def update(player: Rating, opponents: list[tuple[Rating, float]], tau: float = 0.5) -> Rating:
    """New rating for `player` after playing `opponents`.

    `opponents` is a list of (opponent_rating, score) where score is 1 for a
    win and 0 for a loss. Volee (and this project) always passes exactly one
    opponent, because every match is its own rating period. The list form is
    kept so we can run Glickman's three-opponent example as a test.

    This is a line-for-line port of Volee's SQL function
    `glicko2_update_after_period`.
    """
    # Step 2: convert to the internal scale.
    mu = (player.rating - 1500.0) / SCALE
    phi = player.rd / SCALE
    phi2 = phi * phi

    # Steps 3 and 4: how much this period's results should move us.
    sum_v = 0.0   # builds v, the "estimated variance" of the rating
    sum_d = 0.0   # builds delta, the "estimated improvement"
    for opp, score in opponents:
        mu_o = (opp.rating - 1500.0) / SCALE
        phi_o = opp.rd / SCALE
        g = _g(phi_o)
        e = _expected(mu, mu_o, phi_o)
        sum_v += g * g * e * (1.0 - e)
        sum_d += g * (score - e)
    v = 1.0 / sum_v
    delta = v * sum_d
    delta2 = delta * delta

    # Step 5: find the new volatility with the Illinois root-finding method.
    # (This is the fiddly part of Glicko-2. You don't need to follow every
    # line — it's solving an equation numerically — but it has to match
    # Volee's version exactly, so it's ported as-is.)
    a = math.log(player.vol * player.vol)

    def f(x: float) -> float:
        ex = math.exp(x)
        return (ex * (delta2 - phi2 - v - ex)) / (2.0 * (phi2 + v + ex) ** 2) - (x - a) / (tau * tau)

    lo = a
    if delta2 > phi2 + v:
        hi = math.log(delta2 - phi2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0 and k <= 100:
            k += 1
        hi = a - k * tau

    f_lo, f_hi = f(lo), f(hi)
    iterations = 0
    while abs(hi - lo) > CONVERGENCE and iterations < 100:
        iterations += 1
        mid = lo + (lo - hi) * f_lo / (f_hi - f_lo)
        f_mid = f(mid)
        if f_mid * f_hi <= 0:
            lo, f_lo = hi, f_hi
        else:
            f_lo /= 2.0
        hi, f_hi = mid, f_mid
    new_vol = math.exp(lo / 2.0)

    # Steps 6 and 7: new uncertainty and new rating.
    phi_star = math.sqrt(phi2 + new_vol * new_vol)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = mu + new_phi * new_phi * sum_d

    # Step 8: back to rating points.
    return Rating(rating=new_mu * SCALE + 1500.0, rd=new_phi * SCALE, vol=new_vol)


def inflate_for_inactivity(player: Rating, idle_days: float, every_days: int = 28, max_rd: float = 350.0) -> Rating:
    """Grow a player's uncertainty after time away, like Volee's decay job.

    Volee's `apply_activity_decay` adds one step of uncertainty for each
    block of 4 idle weeks: RD becomes sqrt(RD^2 + (vol * SCALE)^2), capped
    at 350. We don't copy the other half of that job (shrinking the rating
    towards Volee's floor of 100), because that floor only exists on Volee's
    100–700 scale and has no equivalent for pro players.
    """
    steps = int(idle_days // every_days)
    rd = player.rd
    for _ in range(steps):
        rd = min(max_rd, math.sqrt(rd * rd + (player.vol * SCALE) ** 2))
        if rd >= max_rd:
            break
    return Rating(rating=player.rating, rd=rd, vol=player.vol)
