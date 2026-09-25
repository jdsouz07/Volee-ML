"""
config.py — every setting the project uses, in one place.

If you want to change how the project behaves, you should only ever have to
edit this file. Each setting says what it does and, where it matters, why it
has the value it has.

The big rule for this project: **only use information Volee also has.**
Pro tennis data has rankings, surfaces, serve stats, seeds, heights and so
on. Volee doesn't collect any of that, so a model trained on it could never
run inside Volee. Every column we keep has a Volee equivalent — see
`VOLEE_EQUIVALENTS` below.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # downloaded + intermediate files (gitignored)
RAW_DIR = DATA_DIR / "raw"          # the untouched CSVs from the internet
MATCHES_CSV = DATA_DIR / "matches.csv"      # step 1 output: cleaned matches
FEATURES_CSV = DATA_DIR / "features.csv"    # step 2 output: one row per match, model-ready
MODELS_DIR = ROOT / "models"        # step 3 output: trained models (gitignored)
REPORTS_DIR = ROOT / "reports"      # step 4 output: results table + charts (committed)

# ---------------------------------------------------------------------------
# Where the tennis data comes from
# ---------------------------------------------------------------------------
# Jeff Sackmann's ATP/WTA match files, via an archival mirror (his original
# GitHub repos were taken down in 2026). We pin an exact commit so that
# anyone who runs this project downloads byte-identical files and gets the
# same numbers. License: CC BY-NC-SA 4.0 (non-commercial use is fine).

MIRROR_REPO = "Aneeshers/tennis-sackmann-archive"
MIRROR_COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
RAW_URL = f"https://raw.githubusercontent.com/{MIRROR_REPO}/{MIRROR_COMMIT}"

TOURS = {
    # tour folder/prefix -> the Volee ladder it corresponds to
    "atp": "Mens",
    "wta": "Womens",
}

FIRST_YEAR = 1991   # ratings need history to warm up; see the split below
LAST_YEAR = 2026    # the mirror includes the current season so far

# Which events count. Tour-level singles only:
#   G = Grand Slam, M = Masters / Premier Mandatory, A/P/I/T = regular tour
#   events (ATP and WTA use different letters), F = season finals,
#   O = Olympics, W/C = older WTA tiers.
# Left out: D = Davis Cup / Billie Jean King Cup (team events, dead rubbers).
KEEP_LEVELS = {"G", "M", "A", "P", "PM", "I", "T1", "T2", "T3", "T4", "T5", "F", "O", "W", "C"}

# Volee matches are best of 3 sets. Men's Grand Slams are best of 5, which is
# a different game (more sets = the better player wins more often), so we
# drop them rather than let them skew the model.
BEST_OF = 3

# ---------------------------------------------------------------------------
# Train / validation / test split — BY TIME, never randomly
# ---------------------------------------------------------------------------
# The model must only ever learn from the past and be judged on the future,
# exactly like it would be used for real. 1991–1999 is "warm-up": ratings
# are being built up, but no rows from those years are used for training.

TRAIN_YEARS = (2000, 2019)
VALID_YEARS = (2020, 2022)   # used to compare models and pick settings
TEST_YEARS = (2023, 2026)    # touched once, at the end, for the final number

# ---------------------------------------------------------------------------
# Glicko-2 — the rating system Volee uses
# ---------------------------------------------------------------------------
# These are Volee's real settings, copied from its database:
#   glicko2_update_after_period()  -> tau 0.5, scale 173.7178
#   profiles.rating_deviation      -> default 100
#   profiles.rating_volatility     -> default 0.06
#   apply_activity_decay()         -> after 4 idle weeks, uncertainty grows
#
# Volee runs one rating update per match ("each match is its own rating
# period"), and so do we.

VOLEE_GLICKO = {
    "initial_rating": 1500.0,   # Volee starts from a skill-based rating; pros all start equal
    "initial_rd": 100.0,        # Volee's default rating deviation (how unsure we are)
    "initial_vol": 0.06,        # volatility: how erratic a player's results are
    "tau": 0.5,                 # how fast volatility itself can change
}

# The textbook starting values, used only as a second, stronger baseline.
# Glickman recommends RD 350 for a brand-new player (very unsure). If our
# model only beats Volee's settings but not these, the "win" was really just
# better Glicko tuning, not machine learning — so we report both honestly.
STANDARD_GLICKO = {**VOLEE_GLICKO, "initial_rd": 350.0}

INACTIVITY_DAYS = 28        # Volee's decay threshold (4 weeks)
MAX_RD = 350.0              # Volee caps uncertainty here

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

FORM_WINDOW = 10            # "recent form" = the last 10 matches
MOMENTUM_WINDOW = 5         # "rating momentum" = rating change over last 5 matches
DAYS_CAP = 365              # treat any gap over a year as "a year"
DEFAULT_AGE = 25.0          # used when a birth date is missing (20 of 164k pro rows;
                            # Volee has the same gap for accounts made before it asked for DOB)
RANDOM_SEED = 7             # makes the coin-flip (see features.py) repeatable

# ---------------------------------------------------------------------------
# What each kept column means in Volee (the whole point of the project)
# ---------------------------------------------------------------------------

VOLEE_EQUIVALENTS = {
    "date": "matches.played_at",
    "ladder": "profiles.gender -> Mens / Womens ladder",
    "winner_id / loser_id": "matches.winner, matches.p1 / p2",
    "score (sets, games)": "matches.score, e.g. '6-4, 3-6, 10-8'",
    "age": "profiles.date_of_birth",
    "rating, rating deviation, volatility": "profiles.rating / rating_deviation / rating_volatility (Glicko-2)",
}
