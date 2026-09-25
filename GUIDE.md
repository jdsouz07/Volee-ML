# Guide — understanding every part of this project

This guide is meant to be read with the code open beside it. Plan on about two
hours to go through it properly. By the end you should be able to explain any
line of the project and answer the questions an interviewer would ask.

---

## 1. The question

> Before a tennis match starts, how likely is each player to win?

Volee already answers this. Its rating system, Glicko-2, gives every player a
rating, and from two ratings it can compute a win probability. The question
this project asks is:

> Can machine learning, using only data Volee already has, forecast better
> than Volee's rating system?

The answer is yes: about 2% better overall and 3% better for new or returning
players. Everything below explains how we know that, and why you can trust it.

---

## 2. Reading order

Read the files in this order. Each one starts with a comment block explaining
what it does. Run each step as you go and look at what it produces.

| # | File | What to look for | Run |
|---|---|---|---|
| 1 | `volee_ml/config.py` | Every setting in one place. The comments explain every choice. | — |
| 2 | `volee_ml/glicko2.py` + `tests/test_glicko2.py` | How a rating system works, and how we proved it matches Volee. | `make test` |
| 3 | `volee_ml/data.py` + `tests/test_score_parser.py` | Downloading, reading tennis scores, and throwing away everything Volee doesn't have. | `make data`, then open `data/matches.csv` |
| 4 | `volee_ml/features.py` + `tests/test_no_leakage.py` | **The most important file.** How history gets turned into numbers without cheating. | `make features`, open `data/features.csv` |
| 5 | `volee_ml/train.py` | Two models, and how to read what the simpler one learned. | `make train` |
| 6 | `volee_ml/evaluate.py` | How to score forecasts honestly. | `make evaluate`, read `reports/results.md` |
| 7 | `volee_ml/volee.py` + `tests/test_volee_adapter.py` | Running the same model on Volee's data. | `make volee` |

---

## 3. Concepts, in plain words

**Feature.** A number describing the situation before the match, like "A's
rating minus B's rating". The model only ever sees features. It never sees
player names.

**Label.** The answer we're trying to predict. Here `y = 1` if player A won and
`0` if not.

**Baseline.** The thing to beat. If you can't beat the simple method already in
use, the ML isn't worth adding. Our baseline is Volee's own forecast, so
"better than the baseline" means "better than the app today".

**Data leakage.** When features accidentally contain information from the
future, for example building a player's form from a window that includes
the match being predicted. The model then looks brilliant in testing and
fails in real life. `features.py` avoids it by always writing features down
before applying the result. `test_no_leakage.py` proves it by rebuilding each
row from a history that stops at that match.

**Why the coin flip.** The raw data lists the winner first. Without the flip,
"A" would always be the winner and the model would just learn to say "A". The
flip makes the winner A about half the time. The checksum of the match id
decides it, so it's the same on every run.

**Time-based split.** We train on 2000–2019, compare models on 2020–2022,
and report the final number on 2023–2026. A random split would let the model
train on 2024 and test on 2021, which is using the future to predict the past.
It would also flatter the model.

**Log loss.** The main score. It rewards being confident and right, and punishes
being confident and wrong much more than being hesitant and wrong. A coin flip
scores 0.693 and lower is better. It's the right metric because we care about
the *probability*, not just who's favoured. "Sam has a 70% chance" should mean
Sam wins 7 times in 10.

**Calibration.** Whether the probabilities mean what they say. In
`reports/calibration.png`, each dot groups matches the model called roughly
the same, like "about 70%". If the dots sit on the diagonal, 70% forecasts
came true about 70% of the time. Ours do.

**Logistic regression.** Multiplies each feature by a learned weight, adds them
up, and squashes the total into 0–1. Its weights are its explanation. It can
only learn straight-line effects.

**Gradient boosting.** Hundreds of small decision trees, each correcting the last.
It can learn curves and interactions, but it's harder to explain and more
prone to learning noise.

**Overfitting.** Learning patterns that were just luck in the training data.
`min_samples_leaf=200` and `l2_regularization` in `train.py` are brakes on it.

**Permutation importance.** Scramble one feature's column and measure how much
worse the model gets. A big drop means the model relies on that feature.

**Domain shift.** When the data a model is used on differs from what it learned
from. Pros play weekly and are consistent, while club players are neither. This is
the main risk in applying the model to Volee, covered in section 6.

---

## 4. Reading the results

Open `reports/results.md`.

**Logistic regression wins, at +1.8% log loss on test.** That sounds small, but
in forecasting, small log-loss gains are meaningful: the baseline is already a
decent model, and it is what the published tennis-forecasting work competes
against. Accuracy moves from 64.0% to 65.2%, which is about 190 more correct
calls across the 16,107 test matches.

**The simpler model beats the fancier one.** Gradient boosting did slightly
better on validation accuracy but worse on log loss, and worse on test. With
features that are already clean and mostly linear, the extra flexibility
mostly learns noise. The lesson: always try the simple model first, and only
keep the complex one if it earns its place.

**Textbook Glicko settings do *worse* than Volee's.** Starting new players with
RD 350 (very unsure) makes their early forecasts swing too much. Volee's
default of 100 is better on this data. We report both so no one can say the
ML gain was just better Glicko tuning.

**The gain is biggest where Volee needs it.** On matches involving an uncertain
rating, the model improves log loss by 3.1% and accuracy by 2.4 points.
Glicko only knows the rating, so it's weakest when the rating hasn't settled.
The model adds recent form, game margins, rest and age. Most Volee players are
new or play rarely, so this is the slice that matters most for the app.

**What the model relies on** (`feature_importance.png`):
1. `rating_diff` and `glicko_prob`: the rating system still does most of the work.
2. `games_share_diff`: *how* you've been winning, not just whether. Winning
   6-1 6-1 says more than 7-6 7-6. **This is the model's biggest addition
   over Glicko, which only records win or loss.**
3. Form, age and rest follow. Head-to-head adds almost nothing once ratings
   are known.

**A puzzle in the logistic weights.** `form_diff` has a *negative* weight, as
if winning recently hurts. It doesn't. Form, games share and rating all
measure similar things, so the model splits the credit among them, and one can
end up negative after the others take their share. This is called
**multicollinearity**, and it's a classic interview question about why a
linear model's weights can't always be read one at a time.

---

## 5. Exercises (learn by changing things)

Each takes 5–20 minutes. Run `make features train evaluate` after each change
unless noted.

1. **Break it on purpose.** In `features.py`, move the `rows.append({...})` block
   *below* the rating updates. Run `make test` to watch the leakage test fail,
   then `make features train evaluate` to see the too-good-to-be-true score.
   Undo it afterwards.
2. **Remove the best new feature.** Delete `games_share_diff` from `FEATURES`.
   How much of the gain disappears?
3. **Tune the baseline.** In `config.py`, try `initial_rd` of 60 and 150. Does
   Volee's rating system get better or worse? Should Volee change its default?
4. **Change the form window.** `FORM_WINDOW = 5` and `20`. Is recent-recent form
   more useful than long-term form?
5. **Split by tour.** In `evaluate.py`, add slices for `test["tour"] == "atp"` and
   `"wta"`. Is women's tennis more or less predictable?
6. **Add a feature Volee has.** Volee knows if a match went to a third set.
   Add "share of A's last 10 matches that went three sets" and see whether
   it helps.

---

## 6. Using it in Volee (what would really need to happen)

`volee.py` already runs the trained model on Volee-format matches. To use it in
the app for real:

1. **Seed ratings properly.** In `volee.py`, every player starts at the
   default rating. In Volee they'd start from their real rating.
2. **Deal with domain shift.** Once Volee has a few hundred matches, measure
   the pro-trained model on them against Volee's Glicko. Then retrain on pro
   and Volee data together, weighting Volee's more.
3. **Serve it without slowing the app.** Run a nightly job that scores
   upcoming challenges and writes the probabilities to a table the app reads.
   Nothing runs in the request path.
4. **Show it well.** Only show a percentage if it's calibrated. Otherwise show
   "favourite", "close" or "underdog".

---

## 7. Questions you should be able to answer

* *Why log loss and not accuracy?* Because we care about the probabilities.
  See section 3.
* *How do you know there's no leakage?* Features are recorded before the
  result is applied, and a test rebuilds each row from truncated history.
* *Why not a random train/test split?* It would train on the future.
* *Why did the simpler model win?* The features are clean and mostly linear,
  so extra flexibility mostly fit noise.
* *What would you do with more time?* Retrain on Volee data, add Challenger
  and ITF matches (closer to club level), and test a model that updates
  ratings using game margins instead of only win or loss.
* *Why is your baseline the one you chose?* It's what the app actually uses,
  proven identical by the parity tests.

---

## 8. Resume lines

> **Match-outcome model for Volee (Python, scikit-learn).** Trained on 164k
> professional matches, restricted to data the Volee app collects, and beat
> the app's Glicko-2 rating system by 1.8% log loss on held-out 2023–26
> matches (3.1% for new players). Built a leakage-tested, time-split pipeline
> with a parity-tested port of the app's rating engine, plus an adapter that
> scores the app's own match exports.
