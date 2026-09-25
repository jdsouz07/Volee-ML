# Volee-ML — predicting tennis matches better than the rating system

A machine-learning model that forecasts who wins a tennis match, trained on
164,000 professional matches and built to run on data from
[Volee](https://apps.apple.com/us/app/volee-app/id6762168017), a live iOS app for
club-level tennis ladders.

**The rule that shapes everything:** the model may only use information Volee
also collects — match dates, scores, results, ages and ratings. Pro data has
rankings, surfaces and serve statistics too; all of that is deliberately
thrown away, so the same model can run on Volee's own matches.

## Result

Test set: 16,107 matches from 2023–2026 that the models never saw while training.

| Model | Log loss ↓ | Accuracy | vs Volee |
|---|---|---|---|
| Volee's current rating system (Glicko-2, exact parity with the app) | 0.6279 | 64.0% | — |
| Glicko-2 with textbook start values | 0.6333 | 64.2% | −0.8% |
| **Logistic regression** | **0.6169** | **65.2%** | **+1.8%** |
| Gradient boosting | 0.6206 | 64.7% | +1.2% |

On matches where a player's rating is still **uncertain** — new or returning
players, which describes most of a club ladder — the gain nearly doubles:
**+3.1% log loss, 66.1% → 68.5% accuracy** (2,629 matches).

Full results, calibration chart and feature importance: [`reports/results.md`](reports/results.md).

## How it works

```
 pro match files ──► 1. data.py ──► 2. features.py ──► 3. train.py ──► 4. evaluate.py
 (ATP + WTA)         keep only       replay history,     logistic reg.     log loss, calibration,
                     Volee-shaped    one match at a      + gradient        importance → reports/
                     columns         time, no peeking    boosting

 Volee export ─────► 5. volee.py ──► same features ──► same model ──► forecasts for Volee matches
```

* **Baseline = Volee itself.** `glicko2.py` is a line-for-line port of Volee's
  database function; tests confirm it produces the app's outputs to 6+ decimals.
* **No leakage.** Features are written down before each match's result is
  applied; a test rebuilds every row from a truncated history to prove it.
* **Split by time**, never randomly: train 2000–2019, validate 2020–22, test 2023–26.

## Run it

```bash
make setup     # Python 3.12 virtual environment + packages
make all       # download (~40 MB), build features, train, evaluate, Volee demo — a few minutes the first time, ~20 s after
make test      # 23 tests
```

## Learn it

Start with **[GUIDE.md](GUIDE.md)**: the reading order, every concept explained
in plain words, how to read the results, and exercises.

## Data

Match results by Jeff Sackmann (`tennis_atp`, `tennis_wta`), via the archival
mirror [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive),
pinned to one commit. Licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
this project is non-commercial. The Volee demo file contains only Volee's
fictional seed players.
