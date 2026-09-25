"""
tune.py — STEP 2a: give the baseline a fair chance, then test the
margin-aware rating system.

Run it:  python -m volee_ml.tune

WHY THIS STEP EXISTS
"My model beat Glicko" means little if Glicko was set up badly. So before
comparing, we search for Glicko's best settings — tuned on the VALIDATION
years (2020–22) only, never the test years — and use that as a second, much
tougher baseline. Only then is a win clearly the model's doing.

The same search then tries the margin-aware rating system (see ratings.py)
across several margin weights. It gets exactly the same tuning budget as
plain Glicko, so the comparison is fair.

HOW THE SEARCH WORKS
  1. Try every combination of starting uncertainty (RD) and volatility,
     with tau fixed. (Those two matter most.)
  2. Keep the best pair and try a few values of tau.
  3. Keep the best of those and try margin weights from 1.0 (plain
     Glicko) down to 0.3.
  4. Give the margin-aware system the same RD x volatility search plain
     Glicko got in step 1 (its best starting values might differ), then
     try weights just either side of the winner.
Each try replays all 164k matches and scores its forecasts on 2020–22 by
log loss. Searching one or two settings at a time like this is called
coordinate search: much cheaper than trying every combination at once.

Output: reports/tuned_ratings.json (read by features.py) and
reports/tuning.md (the full table, so you can see what mattered).
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
from sklearn.metrics import log_loss

from volee_ml import config
from volee_ml.data import load_matches
from volee_ml.features import coin_flip
from volee_ml.ratings import VOLEE, RatingSettings, replay

RD_GRID = [60, 80, 100, 150, 200, 250, 350]
VOL_GRID = [0.03, 0.06, 0.09]
TAU_GRID = [0.3, 0.5, 0.8, 1.2]
MARGIN_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
MARGIN_FINE = [-0.05, +0.05]   # step 4: nudge the best weight either way


def validation_log_loss(matches, flips, y, valid_mask, settings: RatingSettings) -> float:
    probs, _ = replay(matches, settings, flips)
    return log_loss(y[valid_mask], np.clip(probs[valid_mask], 1e-6, 1 - 1e-6))


def load_tuned() -> dict[str, RatingSettings]:
    """The winning settings, or Volee's defaults if tune.py hasn't been run."""
    path = config.REPORTS_DIR / "tuned_ratings.json"
    if not path.exists():
        return {"tuned_glicko": VOLEE, "margin_glicko": VOLEE}
    saved = json.loads(path.read_text())
    return {name: RatingSettings(**values) for name, values in saved.items()}


if __name__ == "__main__":
    matches = load_matches()
    flips = np.array([coin_flip(mid) for mid in matches["match_id"]])
    y = flips.astype(int)                       # A won exactly when A is the winner
    years = matches["date"].dt.year.to_numpy()
    valid = (years >= config.VALID_YEARS[0]) & (years <= config.VALID_YEARS[1])

    results = []

    def run(settings: RatingSettings, stage: str) -> float:
        loss = validation_log_loss(matches, flips, y, valid, settings)
        results.append((stage, settings, loss))
        print(f"  {stage:<8} {settings.label:<42} validation log loss {loss:.5f}")
        return loss

    print("Volee's current settings:")
    volee_loss = run(VOLEE, "volee")

    print("\n1. starting RD x volatility (plain Glicko, tau 0.5):")
    best = min((RatingSettings(initial_rd=rd, initial_vol=v) for rd in RD_GRID for v in VOL_GRID),
               key=lambda s: run(s, "rd/vol"))

    print("\n2. tau:")
    best = min((replace(best, tau=t) for t in TAU_GRID), key=lambda s: run(s, "tau"))
    tuned_glicko = best
    tuned_loss = min(loss for stage, s, loss in results if s == tuned_glicko)

    print("\n3. margin weight (1.0 = plain Glicko):")
    margin = min((replace(tuned_glicko, margin_weight=w) for w in MARGIN_GRID), key=lambda s: run(s, "margin"))

    print("\n4. margin-aware: RD x volatility again, then finer weights:")
    margin = min((replace(margin, initial_rd=rd, initial_vol=v) for rd in RD_GRID for v in VOL_GRID),
                 key=lambda s: run(s, "m-rd/vol"))
    margin = min([margin] + [replace(margin, margin_weight=round(margin.margin_weight + d, 2)) for d in MARGIN_FINE],
                 key=lambda s: run(s, "m-fine"))
    margin_loss = min(loss for stage, s, loss in results if s == margin)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "tuned_ratings.json").write_text(json.dumps(
        {"tuned_glicko": asdict(tuned_glicko), "margin_glicko": asdict(margin)}, indent=2))

    lines = [
        "# Rating-system tuning",
        "",
        f"Every setting scored by log loss on the validation years {config.VALID_YEARS[0]}–{config.VALID_YEARS[1]} "
        "(lower is better). The test years are not used here.",
        "",
        f"* Volee's current settings: **{volee_loss:.5f}** ({VOLEE.label})",
        f"* Best plain Glicko-2: **{tuned_loss:.5f}** ({tuned_glicko.label})",
        f"* Best margin-aware Glicko-2: **{margin_loss:.5f}** ({margin.label})",
        "",
        "| Stage | Settings | Validation log loss |",
        "|---|---|---|",
    ] + [f"| {stage} | {s.label} | {loss:.5f} |" for stage, s, loss in results]
    (config.REPORTS_DIR / "tuning.md").write_text("\n".join(lines) + "\n")
    print(f"\nBest plain Glicko:  {tuned_glicko.label}  ({tuned_loss:.5f})")
    print(f"Best margin-aware:  {margin.label}  ({margin_loss:.5f})")
    print(f"Wrote {config.REPORTS_DIR / 'tuned_ratings.json'} and tuning.md")
