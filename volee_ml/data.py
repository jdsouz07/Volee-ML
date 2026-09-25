"""
data.py — STEP 1: download pro tennis results and clean them into
"Volee-shaped" matches.

Run it:  python -m volee_ml.data

WHAT COMES IN
One CSV per tour per year (e.g. atp_matches_2019.csv) with ~50 columns:
players, score, rankings, surface, serve statistics, heights, and so on.

WHAT GOES OUT
data/matches.csv — one row per completed best-of-3 match, keeping ONLY the
columns Volee also records:

    match_id, date, order, tour, ladder,
    winner_id, winner_name, winner_age,
    loser_id,  loser_name,  loser_age,
    score, winner_sets, loser_sets, winner_games, loser_games

Everything else (ranking, surface, serve stats, seeds...) is thrown away on
purpose. A model can only be used in Volee if Volee can give it the same
inputs.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

from volee_ml import config

# Only these columns are read from the raw files. Anything not listed here
# never even enters the program.
RAW_COLUMNS = [
    "tourney_id", "tourney_date", "tourney_level", "match_num", "round", "best_of",
    "winner_id", "winner_name", "winner_age",
    "loser_id", "loser_name", "loser_age",
    "score",
]

# Matches in one tournament share a start date, so we order them by round.
# Qualifying (Q1..Q4) comes first, then the main draw from the biggest round
# down to the final. RR = round robin (season finals), BR = bronze medal match.
ROUND_ORDER = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3,
    "R128": 4, "R64": 5, "R32": 6, "R16": 7, "RR": 7,
    "QF": 8, "SF": 9, "BR": 10, "F": 10,
}


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def download(tour: str, year: int) -> pd.DataFrame | None:
    """Fetch one season's file, using a local copy if we already have it."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{tour}_matches_{year}.csv"
    if not path.exists():
        url = f"{config.RAW_URL}/{tour}/{tour}_matches_{year}.csv"
        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            return None  # that season isn't in the archive
        response.raise_for_status()
        path.write_bytes(response.content)
    return pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, encoding="latin-1", low_memory=False)


# ---------------------------------------------------------------------------
# Reading a tennis score
# ---------------------------------------------------------------------------

SET_PATTERN = re.compile(r"\[?(\d+)-(\d+)(?:\(\d+\))?\]?")
UNFINISHED = ("RET", "W/O", "DEF", "ABD", "ABN", "Def", "unfinished", "In Progress", "NA")


def parse_score(score: str) -> tuple[int, int, int, int] | None:
    """Turn '7-6(5) 3-6 6-4' into (winner_sets, loser_sets, winner_games, loser_games).

    The score is always written from the WINNER's side. Tiebreak points in
    brackets like (5) are ignored — Volee stores them separately. A deciding
    match tiebreak written as [10-8] counts as one set and one game.

    Returns None for anything that wasn't played to the end (retirement,
    walkover, default). Those tell us little about who is better, and Volee
    handles its own no-shows separately.
    """
    if not isinstance(score, str) or any(word in score for word in UNFINISHED):
        return None
    sets = SET_PATTERN.findall(score)
    if not sets:
        return None

    winner_sets = loser_sets = winner_games = loser_games = 0
    for w_str, l_str in sets:
        w, l = int(w_str), int(l_str)
        if w == l:
            return None  # a set can't end level; something is wrong with this row
        if w > l:
            winner_sets += 1
        else:
            loser_sets += 1
        if max(w, l) >= 10 and "[" in score:  # match tiebreak: count as one game
            w, l = int(w > l), int(l > w)
        winner_games += w
        loser_games += l

    if winner_sets != 2 or loser_sets > 1:
        return None  # not a completed best-of-3 win
    return winner_sets, loser_sets, winner_games, loser_games


# ---------------------------------------------------------------------------
# Building the clean table
# ---------------------------------------------------------------------------

def clean(raw: pd.DataFrame, tour: str) -> pd.DataFrame:
    """Keep completed best-of-3 tour matches and shape them like Volee's."""
    df = raw[raw["best_of"] == config.BEST_OF]
    df = df[df["tourney_level"].isin(config.KEEP_LEVELS)]

    parsed = df["score"].map(parse_score)
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df[["winner_sets", "loser_sets", "winner_games", "loser_games"]] = pd.DataFrame(
        parsed.tolist(), index=df.index
    )

    # Date: the tournament start, plus one day per round, so a final is
    # dated after the semi-final. It's an approximation (the archive only
    # records the tournament's start date) but it keeps the order right and
    # the "days since last match" feature roughly honest.
    start = pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    round_index = df["round"].map(ROUND_ORDER).fillna(5).astype(int)
    df["date"] = start + pd.to_timedelta(round_index, unit="D")
    df["order"] = round_index * 1000 + df["match_num"].fillna(0).astype(int)

    df["tour"] = tour
    df["ladder"] = config.TOURS[tour]
    # Player ids are only unique within a tour, so prefix them.
    df["winner_id"] = tour + "_" + df["winner_id"].astype(str)
    df["loser_id"] = tour + "_" + df["loser_id"].astype(str)
    df["match_id"] = tour + "_" + df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)

    return df[[
        "match_id", "date", "order", "tour", "ladder",
        "winner_id", "winner_name", "winner_age",
        "loser_id", "loser_name", "loser_age",
        "score", "winner_sets", "loser_sets", "winner_games", "loser_games",
    ]]


def build_matches() -> pd.DataFrame:
    """Download every season, clean it, and save data/matches.csv."""
    frames = []
    for tour in config.TOURS:
        for year in range(config.FIRST_YEAR, config.LAST_YEAR + 1):
            raw = download(tour, year)
            if raw is None:
                print(f"  {tour} {year}: not in the archive, skipped")
                continue
            cleaned = clean(raw, tour)
            frames.append(cleaned)
            print(f"  {tour} {year}: {len(raw):>5} raw rows -> {len(cleaned):>5} kept")

    matches = pd.concat(frames, ignore_index=True)
    # Chronological order is essential: every later step walks through the
    # matches one at a time, as if living through history.
    matches = matches.sort_values(["date", "tour", "order"], kind="stable").reset_index(drop=True)
    matches = matches.drop_duplicates("match_id")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(config.MATCHES_CSV, index=False)
    print(f"\nSaved {len(matches):,} matches to {config.MATCHES_CSV}")
    return matches


def load_matches() -> pd.DataFrame:
    return pd.read_csv(config.MATCHES_CSV, parse_dates=["date"])


if __name__ == "__main__":
    build_matches()
