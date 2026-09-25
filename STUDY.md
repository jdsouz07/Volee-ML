# Study plan — learn this project well enough to defend it

Nine lessons, about 30–60 minutes each. Every lesson has the same shape:

* **Goal**: what you should be able to do afterwards
* **Read**: the files, in order
* **Do**: something hands-on (run it, change it, break it)
* **Key ideas**: the points to remember
* **Practice**: questions with answers hidden. Answer out loud or on paper *before* you click.
* **Checkpoint**: if you can do this, move on

[GUIDE.md](GUIDE.md) is the reference; this file is the course. Do the lessons in
order, since each builds on the last.

---

## Lesson 0 — Set up and get the map (15 min)

**Goal:** run everything once and know where each piece lives.

**Do:**
```bash
make setup
make all        # ~2–3 minutes the first time
make test       # 29 passed
make demo       # open http://localhost:8766
```
Then open `reports/results.md` and just look. Don't try to understand it yet.

**Key ideas:** the project is seven steps, each a file in `volee_ml/`:
`data` → `tune` → `features` → `train` → `evaluate` → `volee` → `export`.
Every step writes a file the next step reads.

**Practice**

1. Which folder holds things you can delete and rebuild, and which holds things committed to GitHub?
   <details><summary>Answer</summary>

   `data/` and `models/` are rebuilt by `make all`, and `.gitignore` excludes them. `reports/` and `docs/` are committed, because they're the results and the demo people see.
   </details>

2. What single command would you run after changing a feature?
   <details><summary>Answer</summary>

   `make features train evaluate`. Tuning only needs rerunning if you change the rating system. Data only needs rerunning if you change what's downloaded or kept.
   </details>

**Checkpoint:** you can name the seven steps in order without looking.

---

## Lesson 1 — The problem, and how to score a forecast (40 min)

**Goal:** explain log loss, accuracy, calibration and "baseline" to a non-technical friend.

**Read:** GUIDE.md sections 1 and 3 (Log loss, Calibration, Baseline). Then the top of `volee_ml/evaluate.py`.

**Do:** in a Python shell,
```python
import math
for p in (0.9, 0.6, 0.5):
    print(p, "right:", -math.log(p), "wrong:", -math.log(1 - p))
```

**Key ideas**
* We predict a **probability**, not a winner. "70%" should come true 7 times in 10.
* **Log loss** is the average of `−log(probability you gave to what actually happened)`. It punishes confident mistakes hardest.
* **Accuracy** only asks "was the favourite right?". It ignores whether you said 51% or 99%.
* The **baseline** is what you must beat. Ours is Volee's own rating system, so beating it means beating the app today.

**Practice**

1. You say 90% and you're right. You say 90% and you're wrong. What's the log loss of each?
   <details><summary>Answer</summary>

   Right: −ln(0.9) = **0.105**. Wrong: −ln(0.1) = **2.303**. One confident miss costs as much as about 22 confident hits earn.
   </details>

2. Why does a coin flip score 0.693?
   <details><summary>Answer</summary>

   Always saying 50% gives −ln(0.5) = ln 2 = 0.693 on every match. It's the "knows nothing" reference point. Our models score about 0.617.
   </details>

3. Model X has higher accuracy than model Y but worse log loss. How is that possible, and which would you ship?
   <details><summary>Answer</summary>

   X picks the favourite right slightly more often, but its probabilities are badly sized. It's overconfident when wrong, or timid when right. For showing users a percentage, ship Y: its numbers mean what they say.
   </details>

4. Looking at `reports/calibration.png`: what would a curve *below* the diagonal on the right-hand side mean?
   <details><summary>Answer</summary>

   When the model says around 80%, the player wins less than 80% of the time. The model is **overconfident** about favourites.
   </details>

5. Why is "better than Volee's Glicko-2" a more convincing claim than "65% accurate"?
   <details><summary>Answer</summary>

   65% means nothing without context. Tennis favourites win most of the time anyway. Beating the system actually in use is a direct, comparable improvement.
   </details>

**Checkpoint:** explain in two sentences why we optimise log loss rather than accuracy.

---

## Lesson 2 — Glicko-2, the rating system (60 min)

**Goal:** explain rating, RD and volatility, and how we *proved* our copy matches Volee.

**Read:** `volee_ml/glicko2.py` (the top comment first, then `win_probability`, then skim `update`), then `tests/test_glicko2.py`.

**Do:**
```python
from volee_ml.glicko2 import Rating, win_probability, update
print(win_probability(Rating(1700, 60), Rating(1500, 60)))    # sure about both
print(win_probability(Rating(1700, 300), Rating(1500, 300)))  # unsure about both
print(update(Rating(1500, 100), [(Rating(1500, 100), 1)]))    # settled player wins
print(update(Rating(1500, 350), [(Rating(1500, 100), 1)]))    # brand-new player wins
```

**Key ideas**
* **Rating** means how good we think the player is. **RD** means how unsure we are. **Volatility** means how erratic their results are.
* Bigger RD pulls forecasts towards 50/50 *and* makes ratings move more after a result.
* Volee runs one update per match, and so do we.
* **Parity test:** we ran Volee's real database function on fixed inputs, saved its outputs, and our Python must reproduce them to 6+ decimals. That's how you know the baseline really is the app's system.

**Practice**

1. The two `win_probability` calls give 0.752 and 0.665. Same rating gap, so why different?
   <details><summary>Answer</summary>

   With RD 300 we're far less sure either rating is right, so the forecast is pulled towards 50%. Uncertainty shrinks confidence.
   </details>

2. The settled player (RD 100) gains about 26 points for the win. The new player (RD 350) gains about 175. Why?
   <details><summary>Answer</summary>

   With RD 350 we barely know the new player, so one result tells us a lot and the rating jumps. The settled player's rating is already well supported, so one win barely moves it. RD also drops, from 350 to 253, because we now know more.
   </details>

3. Why test against Glickman's own worked example *and* against Volee's database?
   <details><summary>Answer</summary>

   Glickman's example proves the maths is textbook-correct. The Volee cases prove it matches the app's exact implementation. Both could have a bug the other wouldn't catch.
   </details>

4. `inflate_for_inactivity` copies half of Volee's decay job. Which half is left out, and why?
   <details><summary>Answer</summary>

   Volee also shrinks inactive ratings towards its floor of 100. That floor only exists on Volee's 100–700 scale and has no meaning for pros on a 1500-centred scale. RD growth is kept because it applies to any player.
   </details>

5. In one sentence: why is Glicko-2 better than plain Elo for a club ladder?
   <details><summary>Answer</summary>

   It tracks how *sure* it is about each player, so it treats a newcomer's first results differently from a regular's, and most club players are newcomers or irregulars.
   </details>

**Checkpoint:** explain what RD is and name two effects it has, without looking.

---

## Lesson 3 — Data: keeping only what Volee has (40 min)

**Goal:** explain every filter in `data.py` and why it exists.

**Read:** `volee_ml/config.py` (all of it), then `volee_ml/data.py`, then `tests/test_score_parser.py`.

**Do:** open `data/matches.csv` in a spreadsheet and look at a dozen rows. Then:
```python
from volee_ml.data import parse_score
for s in ["6-4 6-3", "6-7(4) 6-3 [10-7]", "6-4 5-3 RET", "6-4 6-3 6-2"]:
    print(s, "->", parse_score(s))
```

**Key ideas**
* **The project's big rule:** only use columns Volee also collects. Rankings, surfaces, serve stats, seeds and heights never enter the program. See `RAW_COLUMNS` and `VOLEE_EQUIVALENTS`.
* Keep only completed best-of-3 tour matches, since that's what Volee matches are.
* Tournaments only record a start date, so rounds are spaced a day apart to keep the order right.
* The data comes from a mirror pinned to one commit, so anyone gets identical files and results.

**Practice**

1. Why drop men's Grand Slam matches?
   <details><summary>Answer</summary>

   They're best of 5. More sets means the better player wins more often, a different game from Volee's best of 3. Mixing them in would teach the model the wrong relationship between skill gap and win chance.
   </details>

2. What does `parse_score("6-7(4) 6-3 [10-7]")` return, and why is the last set counted as one game?
   <details><summary>Answer</summary>

   `(2, 1, 13, 10)`. The `[10-7]` is a match tiebreak, a single deciding "set" played to 10 points. Counting it as 10 and 7 games would wildly inflate the game totals, so it counts as one game won.
   </details>

3. Why throw away retirements and walkovers instead of counting them as wins?
   <details><summary>Answer</summary>

   They say little about who's better, often just injury or illness. Volee also handles no-shows separately from real results.
   </details>

4. Why pin the data to one commit instead of always downloading the latest?
   <details><summary>Answer</summary>

   **Reproducibility.** If the data changes, the numbers in the README stop matching what someone gets when they run it. That already mattered once: the original repositories disappeared in 2026.
   </details>

5. Name one column we *could* have used that would improve accuracy, and say why we didn't.
   <details><summary>Answer</summary>

   Ranking points, surface or serve stats. Volee doesn't collect them, so a model that needed them could never run in the app.
   </details>

**Checkpoint:** list three filters `data.py` applies and justify each.

---

## Lesson 4 — Features, and not cheating (60 min — the most important lesson)

**Goal:** explain how history becomes numbers, and what data leakage is and how the project prevents it.

**Read:** `volee_ml/features.py` (read the top comment twice), then `tests/test_no_leakage.py`.

**Do, then undo:** in `features.py`, cut the whole `rows.append({...})` block and paste it *after* the rating updates, the lines starting `new_w_volee = ...`. Then run:
```bash
make test                       # the leakage test fails
make features train evaluate    # the score becomes too good to be true
git checkout volee_ml/features.py
```

**Key ideas**
* Replay history match by match. Record features first, then apply the result. That order is the whole defence against leakage.
* The **coin flip** decides who is "A", so the label isn't always 1.
* **Smoothing priors:** form starts at 50% and moves towards the real rate as matches pile up, so one lucky win doesn't read as a 100% form.
* The leakage test rebuilds each row from a history that *stops at that match*. If the rows differ, the future leaked in.

**Practice**

1. In the "do" exercise, why did the score jump?
   <details><summary>Answer</summary>

   The features now included the rating *after* the match, which already knows who won. The model learned to read the answer.
   </details>

2. What would the model learn without the coin flip?
   <details><summary>Answer</summary>

   A is always the winner, so it would always predict "A wins" at 100% and score perfectly in training. It would be useless on real matches, where you don't know who the winner is.
   </details>

3. A player has 1 win from 1 match. What's their `_form`? And after 7 wins from 10?
   <details><summary>Answer</summary>

   1 from 1: (1 + 2.5) / (1 + 5) = **0.583**, not 100%. 7 from 10: (7 + 2.5) / (10 + 5) = **0.633**. The 2.5-and-2.5 "prior" keeps tiny samples from looking extreme.
   </details>

4. Why are most features *differences* (A minus B) rather than two separate values?
   <details><summary>Answer</summary>

   Who wins depends on how the players *compare*. A difference encodes that directly and keeps the feature symmetric: swap A and B and it just flips sign.
   </details>

5. Why does the leakage test compare against a *truncated* history rather than just reading the code?
   <details><summary>Answer</summary>

   Reading can miss subtle bugs. Truncation is a mechanical proof: if the row for match k is the same whether or not later matches exist, no later information can have influenced it.
   </details>

**Checkpoint:** explain leakage and the project's two defences, the ordering and the test, to someone who's never heard the word.

---

## Lesson 5 — Tuning fairly, and the margin-aware rating system (50 min)

**Goal:** explain validation vs test, coordinate search, and the new rating idea.

**Read:** `volee_ml/tune.py`, `volee_ml/ratings.py`, `reports/tuning.md`, `tests/test_ratings.py`.

**Do:**
```python
from volee_ml.ratings import winner_score
print(winner_score(12, 2, 0.75))    # won 6-1 6-1
print(winner_score(14, 12, 0.75))   # won 7-6 7-6
print(winner_score(12, 2, 1.0))     # plain Glicko: always 1
```

**Key ideas**
* **Hyperparameters** are settings you choose, not learn. Choose them on the **validation** years. The **test** years are touched once, at the end.
* The search found **Volee's current settings were already the best**. So the baseline is strong, and the model's gain isn't a tuning trick.
* **Margin-aware Glicko-2:** score a win as `0.75 + 0.25 × game share`, so dominant wins count for more. It needs no ML at runtime, just a different input to the existing update.
* The new system got the *same* search budget as plain Glicko. That's what makes the comparison fair.

**Practice**

1. Why is tuning on the test years cheating, even if you never train a model on them?
   <details><summary>Answer</summary>

   Choosing whichever setting scores best on the test set fits the test set. The test score then overstates how well it would do on truly new matches.
   </details>

2. What scores do 6-1 6-1 and 7-6 7-6 wins get with w = 0.75? And with w = 1?
   <details><summary>Answer</summary>

   0.964 and 0.885 with w = 0.75. Both are 1.0 with w = 1, since plain Glicko can't tell them apart.
   </details>

3. Why not use w = 0, pure game share?
   <details><summary>Answer</summary>

   A player can win 7-6 6-7 7-6 with fewer total games. Pure game share would score that *winner* below 0.5, as if they lost. The tuning also shows w = 0.3 is already much worse than 0.75.
   </details>

4. The margin version preferred RD 150 and volatility 0.09, higher than plain Glicko's best. Give a plausible reason.
   <details><summary>Answer</summary>

   Each match now carries more information, not just win or loss, so ratings can safely move faster without chasing noise.
   </details>

5. Why is the margin-aware rating the easiest thing for Volee to adopt?
   <details><summary>Answer</summary>

   Volee's database function already accepts fractional scores, which is how draws work. It's one changed input and two new defaults: no model, no server, no app release for the maths.
   </details>

**Checkpoint:** explain the margin-aware rating in 30 seconds, including why it's fair to compare it with plain Glicko.

---

## Lesson 6 — The models (50 min)

**Goal:** explain both models, why the simpler one won, and how to read its weights.

**Read:** `volee_ml/train.py`, then the weights and importance tables at the bottom of `reports/results.md`, then `reports/feature_importance.png`.

**Do:** in `features.py`, delete `games_share_diff` from `FEATURES` and run `make features train evaluate`. Note how much the logistic regression's gain drops. Then `git checkout volee_ml/features.py`.

**Key ideas**
* **Logistic regression** is weights times scaled features, summed, then squashed to 0–1. Its weights are its explanation.
* **Scaling** with StandardScaler puts every feature on the same spread, so weights are comparable.
* The rating forecasts are converted to **log-odds** first, because that's the scale logistic regression adds on.
* **Gradient boosting** is many small trees, each correcting the last. It's more flexible, and here it was worse. Try simple first.
* **Multicollinearity**: overlapping features split credit, so single weights can have surprising signs.

**Practice**

1. `form_diff` has a negative weight. Does winning recently make you less likely to win?
   <details><summary>Answer</summary>

   No. Form, game share and rating overlap heavily. Game share takes most of the credit, and form's weight ends up correcting for the overlap. Read the group, not a single weight. That's why the demo shows grouped reasons.
   </details>

2. Why did the simpler model beat gradient boosting on the test set?
   <details><summary>Answer</summary>

   The features are already clean, mostly linear summaries. The extra flexibility mostly fit noise in 2000–2019 that didn't carry over to 2023–26.
   </details>

3. Why convert `glicko_prob` to log-odds for logistic regression but not for boosting?
   <details><summary>Answer</summary>

   Logistic regression adds effects on the log-odds scale, so a probability fits naturally once converted. Trees split on thresholds, and any order-preserving transform gives them the same splits.
   </details>

4. What do `min_samples_leaf=200` and `l2_regularization=1.0` protect against?
   <details><summary>Answer</summary>

   **Overfitting.** They stop the trees carving out tiny groups of matches and memorising their quirks.
   </details>

5. Permutation importance says `games_share_diff` matters a lot. How was that measured?
   <details><summary>Answer</summary>

   Shuffle that column on the validation set, breaking its link to the result, and measure how much log loss gets worse. A big increase means the model relied on it.
   </details>

**Checkpoint:** explain why the best model is the simplest one, and what the top three features tell you about tennis.

---

## Lesson 7 — Proving it isn't luck (45 min)

**Goal:** explain confidence intervals, the cluster bootstrap, and the by-year check.

**Read:** the top comment and `bootstrap_improvement` in `volee_ml/evaluate.py`, then `reports/by_year.png`.

**Do:** exercises 4 and 5 in GUIDE.md section 5. Change `BOOTSTRAP_ROUNDS`, then make `tournament_of` return each match id unchanged. Watch the intervals move. Revert both.

**Key ideas**
* **Bootstrap:** re-draw the test set from itself thousands of times and see how much the result wobbles. The middle 95% is the interval.
* **Cluster bootstrap:** re-draw whole tournaments, because matches in one event are correlated. The intervals come out wider and more honest.
* An interval entirely above zero means the gain is real. Positive in every year means it isn't one lucky season.

**Practice**

1. The model's gain is +1.74% with a 95% interval of +1.41% to +2.09%. What does that mean in plain English?
   <details><summary>Answer</summary>

   Across many plausible re-draws of the test matches, the improvement stayed between about 1.4% and 2.1%, and was never near zero. It's very unlikely to be luck.
   </details>

2. Why do intervals get *narrower* when you bootstrap single matches, and why is that worse?
   <details><summary>Answer</summary>

   Treating correlated matches as independent pretends you have more independent evidence than you do. The interval shrinks, but it's overconfident, so it's less honest.
   </details>

3. The margin-aware rating's interval in 2025 alone includes zero. Is the method broken?
   <details><summary>Answer</summary>

   No. One year is a small sample, so its interval is wide. Over the whole test set, the interval of +0.16% to +0.54% excludes zero, and its estimate is positive every year.
   </details>

4. About how many more matches did the model call correctly than Volee's system on the test set?
   <details><summary>Answer</summary>

   Accuracy went from 64.0% to 65.2% over 16,107 matches: about **190 more** correct calls.
   </details>

**Checkpoint:** explain the cluster bootstrap without using the word "bootstrap" first.

---

## Lesson 8 — Taking it to Volee and to the web (45 min)

**Goal:** explain how the model runs on Volee data and in a browser, and the risks in doing so.

**Read:** `volee_ml/volee.py`, `volee_ml/export.py`, `docs/model.js`, `tests/test_export.py`, and the "Using it in Volee" section of GUIDE.md.

**Do:** run `make volee` and read the table. Then `make demo`, pick two players, and compare the two bars.

**Key ideas**
* **The adapter** turns Volee's export into the same columns as step 1, then reuses *identical* feature code. One code path, two datasets.
* **Domain shift:** pros aren't club players. The fix is retraining on Volee's own matches once there are enough.
* **The demo runs the real model with no server:** logistic regression is just weights, so the page repeats the arithmetic. A test proves the JavaScript, the Python and sklearn agree to 1e-9.
* **CI:** GitHub Actions runs the tests on every push, and the badge shows the result.

**Practice**

1. Why does `volee.py` convert `"6-4, 3-6, 10-8"` into `"6-4 3-6 [10-8]"`?
   <details><summary>Answer</summary>

   Volee writes the match tiebreak without brackets. `parse_score` uses the brackets to count it as one game rather than 18.
   </details>

2. Name two ways Volee's players differ from pros that could make the model less accurate there.
   <details><summary>Answer</summary>

   A much wider age range, from 7 to 70+. Far fewer matches per player. Less consistent play. Self-reported starting ratings. All of these are domain shift.
   </details>

3. How would you measure whether the pro-trained model helps Volee, once real data exists?
   <details><summary>Answer</summary>

   Take Volee matches the model has never seen, in time order, and compare its log loss with Volee's Glicko on exactly those matches, with a bootstrap interval, the same method as here.
   </details>

4. Why is the demo's "why" list grouped?
   <details><summary>Answer</summary>

   Multicollinearity. Single weights can have misleading signs, like form appearing to favour the player with the worse form. Summing related features gives reasons that read correctly.
   </details>

5. Why keep the model out of the app's request path, even though it's tiny?
   <details><summary>Answer</summary>

   So nothing in the app waits on it. A nightly job writes forecasts to a table, and the app just reads a number.
   </details>

**Checkpoint:** explain domain shift and your plan for it in under a minute.

---

## Lesson 9 — Put it together

**A 60-second pitch.** Practise until you can say it without notes.

> "Volee is a tennis ladder app I built; it ranks players with Glicko-2. I wanted
> to know if machine learning could forecast matches better, using only data the
> app actually collects. I trained on 164,000 professional matches, stripped of
> anything Volee doesn't have, with a strict time split and a test that proves no
> future data leaks in. My baseline is a line-for-line copy of the app's rating
> code, verified against its database, and tuning showed its settings were already
> optimal. Logistic regression beat it by 1.74% log loss, with a 95% cluster-bootstrap
> interval of 1.4 to 2.1%, and by 3% for new players. The biggest signal was
> margin of victory, so I built a margin-aware Glicko that improves the rating
> system on its own with a one-line change. There's a live demo that runs the
> model in the browser."

**Mock interview.** Answer each in under a minute, then check GUIDE.md section 7.
1. Walk me through your pipeline.
2. How do you know there's no leakage?
3. Why log loss?
4. How do you know the improvement is real?
5. Couldn't you just have tuned Glicko better?
6. Why did logistic regression beat gradient boosting?
7. What's the biggest weakness of the project?
   <details><summary>A strong answer</summary>

   Domain shift: it's trained on pros and meant for club players, and Volee doesn't have enough matches yet to measure the gap. Then say how you'd fix it: retrain on Volee data and evaluate the same way.
   </details>
8. If you had another week, what would you do?
   <details><summary>A strong answer</summary>

   Test on Challenger and ITF matches as a closer stand-in for club tennis. Try margin by sets. Retrain on Volee's data as it accumulates, and propose the margin-aware rating to the app.
   </details>

**Capstone (half a day, on your own).** Add one new feature from data Volee has.
For example: the share of A's last 10 matches that went to three sets. Build it
in `features.py`, add a line to `FEATURES`, run the full pipeline, and report
the change in log loss **with its bootstrap interval**. Then decide honestly
whether to keep it. If you can do this end to end without help, you
understand the project.

---

## Answer key for self-grading

After all nine lessons, you should be able to, without notes:

- [ ] Name the seven pipeline steps and what each produces
- [ ] Explain log loss, calibration and why the baseline matters
- [ ] Explain rating, RD and volatility, and how parity was proven
- [ ] Justify each data filter
- [ ] Explain leakage and both defences, and break it on purpose
- [ ] Explain validation vs test, and why the baseline is strong
- [ ] Explain the margin-aware rating and why the comparison is fair
- [ ] Explain why the simple model won, and read the weights correctly
- [ ] Explain the cluster bootstrap and read the by-year chart
- [ ] Explain domain shift and the plan for Volee
- [ ] Give the 60-second pitch
