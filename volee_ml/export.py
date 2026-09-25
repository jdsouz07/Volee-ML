"""
export.py — STEP 6: package the model for the live demo page.

Run it:  python -m volee_ml.export

The demo (docs/index.html, published on GitHub Pages) forecasts a match
between any two current players, right in the browser. No server is
needed because logistic regression is just arithmetic:

    probability = squash( intercept + sum of weight_i * scaled_feature_i )

So we export three things to docs/demo_data.json:
  1. the model  — its weights, intercept, and the mean/spread used to scale
                  each feature (the StandardScaler);
  2. the players — every player active since ACTIVE_SINCE, with their state
                  as of the last match in the data (ratings, form, age...);
  3. head-to-head records between those players.

The page rebuilds each feature from those pieces exactly as features.py
does, then applies the weights. `tests/test_export.py` checks that the
page's arithmetic (re-done here in Python) gives the same answer as the
real sklearn model.
"""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd

from volee_ml import config
from volee_ml.data import load_matches
from volee_ml.features import FEATURES, _form, _games_share, _momentum, build_features
from volee_ml.glicko2 import Rating, inflate_for_inactivity, win_probability
from volee_ml.ratings import replay
from volee_ml.features import coin_flip
from volee_ml.tune import load_tuned

ACTIVE_SINCE = pd.Timestamp("2025-01-01")
DEMO_JSON = config.ROOT / "docs" / "demo_data.json"


def model_payload(model) -> dict:
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    return {
        "features": list(FEATURES),
        "mean": scaler.mean_.round(8).tolist(),
        "scale": scaler.scale_.round(8).tolist(),
        "weights": lr.coef_[0].round(8).tolist(),
        "intercept": round(float(lr.intercept_[0]), 8),
        "logit_features": ["glicko_prob", "margin_prob"],   # converted to log-odds first (train.py)
    }


def rating_json(r: Rating) -> list[float]:
    return [round(r.rating, 3), round(r.rd, 3), round(r.vol, 6)]


def build_payload() -> dict:
    matches = load_matches()
    _, players, h2h = build_features(matches, return_state=True)
    flips = np.array([coin_flip(mid) for mid in matches["match_id"]])
    _, margin_players = replay(matches, load_tuned()["margin_glicko"], flips)
    as_of = matches["date"].max()

    # Latest age and name per player, from their most recent match.
    latest = {}
    for m in matches.itertuples(index=False):
        latest[m.winner_id] = (m.winner_name, m.winner_age, m.date, m.tour)
        latest[m.loser_id] = (m.loser_name, m.loser_age, m.date, m.tour)

    out_players = {}
    for pid, state in players.items():
        name, age, last_date, tour = latest[pid]
        if last_date < ACTIVE_SINCE:
            continue
        idle = (as_of - last_date).days
        volee = inflate_for_inactivity(state.volee, idle, config.INACTIVITY_DAYS, config.MAX_RD)
        margin = inflate_for_inactivity(margin_players[pid], idle, config.INACTIVITY_DAYS, config.MAX_RD)
        age_now = (age if pd.notna(age) else config.DEFAULT_AGE) + idle / 365.25
        out_players[pid] = {
            "name": name,
            "tour": tour,
            "volee": rating_json(volee),
            "margin": rating_json(margin),
            "matches": state.matches_played,
            "rest": math.log1p(min(idle, config.DAYS_CAP)),
            "form": round(_form(state), 6),
            "games_share": round(_games_share(state), 6),
            "momentum": round(_momentum(state), 4),
            "age": round(age_now, 3),
        }

    out_h2h = {f"{w}|{l}": n for (w, l), n in h2h.items() if w in out_players and l in out_players}
    model = joblib.load(config.MODELS_DIR / "logistic_regression.joblib")
    return {
        "as_of": str(as_of.date()),
        "model": model_payload(model),
        "players": out_players,
        "h2h": out_h2h,
    }


# ---------------------------------------------------------------------------
# The demo's arithmetic, in Python — used by the test to check the page
# ---------------------------------------------------------------------------

def demo_features(payload: dict, a_id: str, b_id: str) -> dict:
    """Rebuild the model's features for a match A vs B from the exported state."""
    a, b = payload["players"][a_id], payload["players"][b_id]
    a_wins = payload["h2h"].get(f"{a_id}|{b_id}", 0)
    b_wins = payload["h2h"].get(f"{b_id}|{a_id}", 0)
    ra, rb = Rating(*a["volee"]), Rating(*b["volee"])
    return {
        "glicko_prob": win_probability(ra, rb),
        "margin_prob": win_probability(Rating(*a["margin"]), Rating(*b["margin"])),
        "rating_diff": ra.rating - rb.rating,
        "rd_a": ra.rd,
        "rd_b": rb.rd,
        "experience_diff": math.log1p(a["matches"]) - math.log1p(b["matches"]),
        "rest_days_a": a["rest"],
        "rest_days_b": b["rest"],
        "form_diff": a["form"] - b["form"],
        "games_share_diff": a["games_share"] - b["games_share"],
        "momentum_diff": a["momentum"] - b["momentum"],
        "h2h_edge": (a_wins - b_wins) / (a_wins + b_wins + 2),
        "h2h_meetings": math.log1p(a_wins + b_wins),
        "age_a": a["age"],
        "age_b": b["age"],
        "is_womens": int(a["tour"] == "wta"),
    }


def demo_probability(payload: dict, features: dict) -> float:
    m = payload["model"]
    total = m["intercept"]
    for name, mean, scale, weight in zip(m["features"], m["mean"], m["scale"], m["weights"]):
        x = features[name]
        if name in m["logit_features"]:
            p = min(max(x, 1e-6), 1 - 1e-6)
            x = math.log(p / (1 - p))
        total += weight * (x - mean) / scale
    return 1 / (1 + math.exp(-total))


if __name__ == "__main__":
    payload = build_payload()
    DEMO_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEMO_JSON.write_text(json.dumps(payload, separators=(",", ":")))
    tours = pd.Series([p["tour"] for p in payload["players"].values()]).value_counts().to_dict()
    print(f"Exported {len(payload['players'])} players {tours}, {len(payload['h2h'])} head-to-head pairs, "
          f"as of {payload['as_of']} -> {DEMO_JSON} ({DEMO_JSON.stat().st_size / 1024:.0f} KB)")
