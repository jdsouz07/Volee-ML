# Guide — understanding every part of this project

Read this with the code open beside it. Plan on about three hours to go through
it properly. By the end you should be able to explain any line of the project
and answer the questions an interviewer would ask.

---

## 1. The question

> Before a tennis match starts, how likely is each player to win?

Volee already answers this. Its rating system, Glicko-2, gives every player a
rating, and from two ratings it computes a win probability. This project asks
two questions:

1. **Can machine learning, using only data Volee already has, forecast better
   than Volee's rating system?** Yes: 1.74% better log loss, 95% interval
   +1.41% to +2.09%.
2. **Can the rating system itself be improved, with no ML needed in the app?**
   Yes: a margin-aware Glicko-2 that also learns from game scores is 0.35%
   better, interval +0.16% to +0.54%.

Everything below explains how we know, and why you can trust it.

---

## 2. Reading order

Each file opens with a comment block explaining what it does. Run each step as
you go and look at what it produces.

| # | File | What to look for | Run |
|---|---|---|---|
| 1 | `volee_ml/config.py` | Every setting in one place, with the reason for each. | — |
| 2 | `volee_ml/glicko2.py` + `tests/test_glicko2.py` | How a rating system works, and how we proved it matches Volee. | `make test` |
| 3 | `volee_ml/data.py` + `tests/test_score_parser.py` | Downloading, reading tennis scores, and throwing away everything Volee doesn't have. | `make data`, open `data/matches.csv` |
| 4 | `volee_ml/ratings.py` + `tests/test_ratings.py` | The margin-aware rating idea, in about 20 lines. | — |
| 5 | `volee_ml/tune.py` | Giving the baseline a fair chance, then tuning the new rating system. | `make tune`, read `reports/tuning.md` |
| 6 | `volee_ml/features.py` + `tests/test_no_leakage.py` | **The most important file.** Turning history into numbers without cheating. | `make features`, open `data/features.csv` |
| 7 | `volee_ml/train.py` | Two models, and how to read what the simpler one learned. | `make train` |
| 8 | `volee_ml/evaluate.py` | Scoring forecasts honestly, and proving the gain isn't luck. | `make evaluate`, read `reports/results.md` |
| 9 | `volee_ml/volee.py` + `tests/test_volee_adapter.py` | Running the same model on Volee's data. | `make volee` |
| 10 | `volee_ml/export.py`, `docs/model.js`, `docs/index.html`, `tests/test_export.py` | Shipping the model to a web page with no server. | `make export demo` |

---

## 3. Concepts, in plain words

**Feature.** A number describing the situation before the match, like "A's
rating minus B's rating". The model only ever sees features, never player names.

**Label.** The answer we're predicting: `y = 1` if player A won, `0` if not.

**Baseline.** The thing to beat. Our baseline is Volee's own forecast, so
"better than the baseline" means "better than the app today".

**Data leakage.** When features accidentally contain information from the
future, for example building form from a window that includes the match being
predicted. The model then looks brilliant in testing and fails in real life.
`features.py` writes features down before applying the result.
`test_no_leakage.py` proves it by rebuilding each row from a history that
stops at that match.

**Why the coin flip.** The raw data lists the winner first. Without the flip,
"A" would always be the winner. A checksum of the match id decides who is A,
so it's the same on every run.

**Time-based split.** Train on 2000–2019, tune and compare on 2020–2022, report
on 2023–2026. A random split would train on the future to predict the past.

**Hyperparameters and tuning.** A model's weights are learned from data.
*Hyperparameters* are the settings you choose, like Glicko's starting RD. We
choose them by trying options and scoring each on the **validation** years,
never the test years. If you tuned on the test years, the test score would
stop being an honest estimate. `tune.py` tries one or two settings at a time,
which is called **coordinate search**.

**Log loss.** The main score. It rewards being confident and right, and punishes
being confident and wrong far more than being hesitant and wrong. A coin flip
scores 0.693, and lower is better. It's the right metric because we care about
the *probability*: "Sam has a 70% chance" should mean Sam wins 7 times in 10.

**Calibration.** Whether probabilities mean what they say. In
`reports/calibration.png`, each dot groups matches forecast at about the same
level. On the diagonal means "70%" came true 70% of the time.

**Bootstrap confidence interval.** One test set is one sample of history. To see
how much the improvement could wobble by luck, we re-draw the test set from
itself 2,000 times, with replacement, and recompute the improvement each time.
The middle 95% of those values is the 95% interval. If it's all above zero,
the gain isn't luck.

**Cluster bootstrap.** Matches in the same tournament aren't independent: same
week, same conditions, often the same players. Re-drawing single matches
would pretend we have more independent evidence than we do. So we re-draw
whole tournaments. That gives honestly wider intervals.

**Logistic regression.** Multiplies each feature by a learned weight, adds them
up, and squashes the total into 0–1. Its weights are its explanation.

**Gradient boosting.** Hundreds of small decision trees, each correcting the last.
It can learn curves and interactions, but it's harder to explain and more
prone to learning noise.

**Overfitting.** Learning patterns that were just luck in the training data.
`min_samples_leaf` and `l2_regularization` in `train.py` are brakes on it.

**Permutation importance.** Scramble one feature's column and measure how much
worse the model gets.

**Multicollinearity.** When features overlap, like form, game share and rating, a
linear model splits the credit between them, and a single weight can come out
with a surprising sign. The demo groups related features for that reason.

**Domain shift.** When the data a model is used on differs from what it learned
from. Pros are consistent and play weekly, while club players are neither. See section 6.

---

## 4. Reading the results

Open `reports/results.md`, `reports/tuning.md` and `reports/by_year.png`.

**Volee's settings were already optimal.** The tuning search tried 21
combinations of starting RD and volatility plus four values of tau. Volee's
RD 100 and volatility 0.06 won. Tau made no difference past the fifth decimal.
The textbook RD 350 was clearly worse. So the baseline is strong, and every gain below
is against the best plain Glicko available.

**The margin-aware rating system works.** With a win scored as
`0.75 + 0.25 × game share`, it beats plain Glicko by 0.35% overall and 1.0% on
uncertain ratings, in every year of the chart. The tuning also found it
prefers more starting uncertainty (RD 150) and more volatility (0.09). With
richer information per match, it can afford to move ratings faster. This needs
no model in the app: Volee's database function already accepts fractional
scores, since draws are 0.5, so it's a change to one input.

**Logistic regression wins overall, at +1.74%.** Accuracy goes from 64.0% to 65.2%,
about 190 more correct calls across 16,107 matches. That sounds small, but the
baseline is a well-tuned rating system, and small log-loss gains over a strong
baseline are what forecasting work competes on.

**The simpler model beats the fancier one.** Gradient boosting is close on
validation but worse on test. With clean, mostly linear features, extra
flexibility mostly learns noise. Always try the simple model first.

**The gain holds every year.** In `by_year.png` the model's improvement is
positive in all seven years, and its interval excludes zero in each. A result
from one lucky season wouldn't look like that.

**The gain is biggest where Volee needs it.** On matches with an uncertain rating,
the model gains 3.1%. Glicko only knows the rating, so it's weakest before the
rating settles. Most Volee players are new or play rarely.

**What the model relies on:** the ratings first, then `games_share_diff`.
*How* you've been winning is the model's biggest addition over plain Glicko.
That observation is what led to the margin-aware rating system. Then come form,
age and rest. Head-to-head adds almost nothing once ratings are known.

**The negative `form_diff` weight** is the multicollinearity effect from
section 3, not "winning recently hurts".

---

## 5. Exercises (learn by changing things)

Run `make tune features train evaluate` after each change unless noted. Each
takes 5–20 minutes.

1. **Break it on purpose.** In `features.py`, move the `rows.append({...})` block
   below the rating updates. Watch `make test` fail and the test score become
   too good to be true. Undo it afterwards.
2. **Remove the best new feature.** Delete `games_share_diff` from `FEATURES`.
   How much of the gain disappears?
3. **Push the margin idea further.** In `ratings.py`, try scoring by *sets* won
   instead of games, or a curve instead of a straight blend. Does it beat 0.75?
4. **Watch the interval move.** Set `BOOTSTRAP_ROUNDS = 200` and then `5000`. The
   estimate stays the same. Why do the interval edges move slightly?
5. **Single-match bootstrap.** Change `tournament_of` to return each match id
   unchanged. The intervals get narrower. Explain why that's *less* honest.
6. **Split by tour.** Add slices for `test["tour"] == "atp"` and `"wta"` in
   `evaluate.py`. Is women's tennis more or less predictable?

---

## 6. Using it in Volee

1. **The margin-aware rating first.** It's the cheapest win: change the score
   passed to `glicko2_update_after_period` from `1` / `0` to the blended score,
   and move the default RD and volatility to the tuned values. Nothing new runs
   in the app.
2. **Seed ratings properly.** `volee.py` starts everyone at the default rating.
   In Volee they'd start from their onboarding rating.
3. **Deal with domain shift.** Once Volee has a few hundred matches, measure the
   pro-trained model against Volee's Glicko on them, then retrain on both,
   weighting Volee's data more.
4. **Serve it without slowing the app.** A nightly job scores upcoming
   challenges and writes probabilities to a table the app reads. That's the
   same idea as the demo: the model is small enough to be just weights.

---

## 7. Questions you should be able to answer

* *Why log loss and not accuracy?* We care about the probabilities, not just
  who's favoured.
* *How do you know there's no leakage?* Features are recorded before the
  result, and a test rebuilds each row from truncated history.
* *How do you know the improvement isn't luck?* A 95% cluster-bootstrap interval
  entirely above zero, and a positive gain in every year.
* *Couldn't you just have tuned Glicko better?* We did, on validation, and
  Volee's settings were already the best, so the gain isn't a tuning artefact.
* *Why cluster by tournament?* Matches within a tournament are correlated, so
  treating them as independent would overstate certainty.
* *Why did the simpler model win?* The features are clean and mostly linear,
  so extra flexibility fit noise.
* *How does the demo run the model without a server?* Logistic regression is
  just weights. We export them with each player's current state, and the page
  does the same arithmetic, tested to match sklearn.
* *What would you do next?* Retrain on Volee's own matches, add Challenger and
  ITF matches as a closer proxy for club tennis, and try margin by sets.

---

## 8. Resume lines

> **Volee-ML: match forecasting for a live tennis app** (Python, scikit-learn, JS).
> Trained on 164k pro matches restricted to data the app collects. Beat the app's
> Glicko-2 rating system by **1.74% log loss (95% CI 1.41–2.09%)** on held-out
> 2023–26 matches, and by 3.1% for new players. Designed a **margin-aware Glicko-2**
> that improves the rating system itself by 0.35% with a one-line change.
> Leakage-tested, time-split pipeline with cluster-bootstrap evaluation and a
> browser demo that runs the model client-side.
