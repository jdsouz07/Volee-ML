# Volee-ML — forecasting tennis matches better than the rating system

[![tests](https://github.com/jdsouz07/Volee-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/jdsouz07/Volee-ML/actions/workflows/tests.yml)

**[Try the live demo →](https://jdsouz07.github.io/Volee-ML/)** Pick any two current pros and compare the forecasts.

A match-outcome model and a new rating system, trained on 164,000 professional
matches and built to run on data from [Volee](https://apps.apple.com/us/app/volee-app/id6762168017),
a live iOS app for club tennis ladders.

**The rule that shapes everything:** only use information Volee also collects:
match dates, scores, results, ages and ratings. Pro data also has rankings,
surfaces and serve statistics, and all of that is deliberately thrown away, so
everything here can run on Volee's own matches.

## Results

Test set: 16,107 matches from 2023–2026, never seen while building or tuning anything.

| | Log loss ↓ | Accuracy | Improvement vs Volee (95% interval) |
|---|---|---|---|
| Volee's Glicko-2, exact parity with the app | 0.6279 | 64.0% | — |
| **Margin-aware Glicko-2** (new rating system, no ML at runtime) | 0.6257 | 64.3% | **+0.35%** (+0.16 to +0.54) |
| **Logistic regression** (16 features) | **0.6170** | **65.2%** | **+1.74%** (+1.41 to +2.09) |
| Gradient boosting | 0.6207 | 64.9% | +1.15% (+0.74 to +1.61) |

* **The baseline is strong.** A 21-setting search on the validation years found
  Volee's current Glicko-2 settings are already the best, so the gains above
  aren't the result of beating a badly tuned rating system.
* **The gains are real.** The 95% intervals come from a tournament-level cluster
  bootstrap and sit entirely above zero, and the model beats the baseline in all
  seven validation and test years ([chart](reports/by_year.png)).
* **Biggest where a club ladder needs it.** When a player's rating is still
  uncertain (new or returning players), the model gains **+3.1%** and 66.1% → 68.3%
  accuracy, and the margin-aware rating alone gains +1.0%.
* **A new rating system Volee could adopt with a one-line change.** Standard
  Glicko only hears win or loss. The margin-aware version scores a win as
  `0.75 + 0.25 × share of games won`, so 6-1 6-1 counts for more than 7-6 7-6.

Full tables, calibration and feature importance: [`reports/results.md`](reports/results.md) ·
tuning search: [`reports/tuning.md`](reports/tuning.md).

## How it works

```
 pro match files ─► 1 data ─► 2a tune ─► 2 features ─► 3 train ─► 4 evaluate ─► 6 export ─► live demo
 (ATP + WTA)        Volee-     rating     replay         logistic    bootstrap CIs,  weights +
                    shaped     settings   history,       regression  calibration,    player state
                    columns    on 2020-22 no peeking     + boosting  by-year chart   → docs/

 Volee export ────► 5 volee ─► same features ─► same model ─► forecasts for Volee matches
```

* **Baseline = Volee itself.** `glicko2.py` is a line-for-line port of Volee's
  database function, and tests confirm it reproduces the app's outputs to 6+ decimals.
* **No leakage.** Features are recorded before each match's result is applied.
  A test rebuilds every row from a truncated history to prove it.
* **Split by time**, never randomly: train 2000–2019, tune and compare on
  2020–22, report on 2023–26.
* **The demo runs the real model** in the browser. A test checks that the page's
  JavaScript, the Python export and the sklearn model agree to 1e-9.

## Run it

```bash
make setup     # Python 3.12 virtual environment + pinned packages
make all       # download (~40 MB) → tune → features → train → evaluate → Volee demo → export (~2–3 min)
make test      # 29 tests
make demo      # the forecaster at http://localhost:8766
```

## Learn it

Start with **[GUIDE.md](GUIDE.md)**. It covers the reading order, every concept in
plain words, how to read the results, exercises and interview questions.

## Data

Match results by Jeff Sackmann (`tennis_atp`, `tennis_wta`), via the archival
mirror [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive),
pinned to one commit. Licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
this project is non-commercial. The Volee demo file contains only Volee's
fictional seed players.
