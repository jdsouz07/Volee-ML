"""
volee.py — STEP 5: run the trained model on matches exported from Volee.

Run it:  python -m volee_ml.volee examples/volee_demo_matches.csv

This is the bridge that makes the project about Volee and not just about
pro tennis. It takes matches in Volee's own format, turns them into the
exact same columns step 1 produces, builds features with the exact same
code as step 2, and asks the trained model for a forecast.

VOLEE EXPORT FORMAT (one row per match; see examples/volee_demo_matches.csv)
    played_at   when the match was played           (matches.played_at)
    p1, p2      the two players' ids                (matches.p1 / p2)
    winner      the winner's id                     (matches.winner)
    p1_name ... display names                       (profiles.display_name)
    score       '6-4, 3-6, 10-8' — Volee's format   (matches.score)
    p1_dob ...  dates of birth                      (profiles.date_of_birth)
    gender      'Mens' or 'Womens'                  (profiles.gender)

The demo file holds the 18 matches between Volee's made-up seed players
(Marcus Reed, Sam Ellis, ...). No real user's data is in this repository.

HONEST LIMITS (worth being able to explain in an interview)
  * The model learned from professionals. Club players are younger AND older,
    play less often, and are far less consistent, so the relationships it
    learned may be off for them. This is called DOMAIN SHIFT. The fix is to
    retrain (or fine-tune) on Volee's own matches once there are enough.
  * Here every player starts at the default rating. In Volee they'd start
    from the rating their onboarding answers gave them, which is better.
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd

from volee_ml import config
from volee_ml.data import parse_score
from volee_ml.features import build_features
from volee_ml.train import model_inputs


def normalise_volee_score(score: str) -> str:
    """'6-4, 3-6, 10-8' -> '6-4 3-6 [10-8]'.

    Volee writes a deciding match tiebreak as a plain '10-8'. The pro data
    marks it with brackets, and parse_score() relies on that to count it as
    one game rather than eighteen.
    """
    sets = [s.strip() for s in str(score).split(",") if s.strip()]
    if len(sets) == 3:
        w, l = (int(x) for x in sets[2].split("-"))
        if max(w, l) >= 10:
            sets[2] = f"[{sets[2]}]"
    return " ".join(sets)


def age_on(date: pd.Timestamp, dob) -> float:
    if pd.isna(dob):
        return config.DEFAULT_AGE
    return (date - pd.Timestamp(dob)).days / 365.25


def volee_to_matches(export: pd.DataFrame) -> pd.DataFrame:
    """Convert a Volee export into the same table step 1 (data.py) produces."""
    rows = []
    for r in export.itertuples(index=False):
        date = pd.Timestamp(r.played_at).tz_localize(None).normalize()
        p1_won = r.winner == r.p1
        score = normalise_volee_score(r.score)
        parsed = parse_score(score)
        if parsed is None:
            continue  # unfinished or not best-of-3, same rule as the pro data
        rows.append({
            "match_id": f"volee_{r.p1}_{r.p2}_{date.date()}",
            "date": date,
            "order": len(rows),
            "tour": "volee",
            "ladder": r.gender,
            "winner_id": r.p1 if p1_won else r.p2,
            "winner_name": r.p1_name if p1_won else r.p2_name,
            "winner_age": age_on(date, r.p1_dob if p1_won else r.p2_dob),
            "loser_id": r.p2 if p1_won else r.p1,
            "loser_name": r.p2_name if p1_won else r.p1_name,
            "loser_age": age_on(date, r.p2_dob if p1_won else r.p1_dob),
            "score": score,
            "winner_sets": parsed[0], "loser_sets": parsed[1],
            "winner_games": parsed[2], "loser_games": parsed[3],
        })
    return pd.DataFrame(rows).sort_values(["date", "order"]).reset_index(drop=True)


def forecast(export_path: str) -> pd.DataFrame:
    matches = volee_to_matches(pd.read_csv(export_path))
    features = build_features(matches)                       # identical code to the pro pipeline
    model = joblib.load(config.MODELS_DIR / "logistic_regression.joblib")
    features["model_prob"] = model.predict_proba(model_inputs(features, for_linear=True))[:, 1]
    return features


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(config.ROOT / "examples" / "volee_demo_matches.csv")
    out = forecast(path)
    pd.set_option("display.width", 140)
    view = out[["date", "a_name", "b_name", "glicko_prob", "model_prob", "y"]].copy()
    view["date"] = view["date"].dt.date
    view.columns = ["date", "player A", "player B", "Glicko: A wins", "model: A wins", "A won?"]
    print(view.round(3).to_string(index=False))
    print(f"\n{len(out)} Volee matches scored. (Tiny sample — this shows the pipeline runs on Volee's data, not how accurate it is.)")
