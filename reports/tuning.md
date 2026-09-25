# Rating-system tuning

Every setting scored by log loss on the validation years 2020–2022 (lower is better). The test years are not used here.

* Volee's current settings: **0.63024** (RD 100, vol 0.06, tau 0.5, w 1)
* Best plain Glicko-2: **0.63023** (RD 100, vol 0.06, tau 1.2, w 1)
* Best margin-aware Glicko-2: **0.62556** (RD 150, vol 0.09, tau 1.2, w 0.75)

| Stage | Settings | Validation log loss |
|---|---|---|
| volee | RD 100, vol 0.06, tau 0.5, w 1 | 0.63024 |
| rd/vol | RD 60, vol 0.03, tau 0.5, w 1 | 0.63821 |
| rd/vol | RD 60, vol 0.06, tau 0.5, w 1 | 0.63142 |
| rd/vol | RD 60, vol 0.09, tau 0.5, w 1 | 0.63098 |
| rd/vol | RD 80, vol 0.03, tau 0.5, w 1 | 0.63668 |
| rd/vol | RD 80, vol 0.06, tau 0.5, w 1 | 0.63066 |
| rd/vol | RD 80, vol 0.09, tau 0.5, w 1 | 0.63052 |
| rd/vol | RD 100, vol 0.03, tau 0.5, w 1 | 0.63590 |
| rd/vol | RD 100, vol 0.06, tau 0.5, w 1 | 0.63024 |
| rd/vol | RD 100, vol 0.09, tau 0.5, w 1 | 0.63026 |
| rd/vol | RD 150, vol 0.03, tau 0.5, w 1 | 0.63566 |
| rd/vol | RD 150, vol 0.06, tau 0.5, w 1 | 0.63024 |
| rd/vol | RD 150, vol 0.09, tau 0.5, w 1 | 0.63037 |
| rd/vol | RD 200, vol 0.03, tau 0.5, w 1 | 0.63635 |
| rd/vol | RD 200, vol 0.06, tau 0.5, w 1 | 0.63100 |
| rd/vol | RD 200, vol 0.09, tau 0.5, w 1 | 0.63115 |
| rd/vol | RD 250, vol 0.03, tau 0.5, w 1 | 0.63724 |
| rd/vol | RD 250, vol 0.06, tau 0.5, w 1 | 0.63203 |
| rd/vol | RD 250, vol 0.09, tau 0.5, w 1 | 0.63224 |
| rd/vol | RD 350, vol 0.03, tau 0.5, w 1 | 0.63908 |
| rd/vol | RD 350, vol 0.06, tau 0.5, w 1 | 0.63441 |
| rd/vol | RD 350, vol 0.09, tau 0.5, w 1 | 0.63498 |
| tau | RD 100, vol 0.06, tau 0.3, w 1 | 0.63024 |
| tau | RD 100, vol 0.06, tau 0.5, w 1 | 0.63024 |
| tau | RD 100, vol 0.06, tau 0.8, w 1 | 0.63024 |
| tau | RD 100, vol 0.06, tau 1.2, w 1 | 0.63023 |
| margin | RD 100, vol 0.06, tau 1.2, w 1 | 0.63023 |
| margin | RD 100, vol 0.06, tau 1.2, w 0.9 | 0.62883 |
| margin | RD 100, vol 0.06, tau 1.2, w 0.8 | 0.62841 |
| margin | RD 100, vol 0.06, tau 1.2, w 0.7 | 0.62890 |
| margin | RD 100, vol 0.06, tau 1.2, w 0.6 | 0.63022 |
| margin | RD 100, vol 0.06, tau 1.2, w 0.5 | 0.63231 |
| margin | RD 100, vol 0.06, tau 1.2, w 0.4 | 0.63513 |
| margin | RD 100, vol 0.06, tau 1.2, w 0.3 | 0.63864 |
| m-rd/vol | RD 60, vol 0.03, tau 1.2, w 0.8 | 0.63841 |
| m-rd/vol | RD 60, vol 0.06, tau 1.2, w 0.8 | 0.62998 |
| m-rd/vol | RD 60, vol 0.09, tau 1.2, w 0.8 | 0.62736 |
| m-rd/vol | RD 80, vol 0.03, tau 1.2, w 0.8 | 0.63665 |
| m-rd/vol | RD 80, vol 0.06, tau 1.2, w 0.8 | 0.62905 |
| m-rd/vol | RD 80, vol 0.09, tau 1.2, w 0.8 | 0.62677 |
| m-rd/vol | RD 100, vol 0.03, tau 1.2, w 0.8 | 0.63558 |
| m-rd/vol | RD 100, vol 0.06, tau 1.2, w 0.8 | 0.62841 |
| m-rd/vol | RD 100, vol 0.09, tau 1.2, w 0.8 | 0.62633 |
| m-rd/vol | RD 150, vol 0.03, tau 1.2, w 0.8 | 0.63457 |
| m-rd/vol | RD 150, vol 0.06, tau 1.2, w 0.8 | 0.62778 |
| m-rd/vol | RD 150, vol 0.09, tau 1.2, w 0.8 | 0.62589 |
| m-rd/vol | RD 200, vol 0.03, tau 1.2, w 0.8 | 0.63455 |
| m-rd/vol | RD 200, vol 0.06, tau 1.2, w 0.8 | 0.62791 |
| m-rd/vol | RD 200, vol 0.09, tau 1.2, w 0.8 | 0.62606 |
| m-rd/vol | RD 250, vol 0.03, tau 1.2, w 0.8 | 0.63483 |
| m-rd/vol | RD 250, vol 0.06, tau 1.2, w 0.8 | 0.62834 |
| m-rd/vol | RD 250, vol 0.09, tau 1.2, w 0.8 | 0.62653 |
| m-rd/vol | RD 350, vol 0.03, tau 1.2, w 0.8 | 0.63558 |
| m-rd/vol | RD 350, vol 0.06, tau 1.2, w 0.8 | 0.62953 |
| m-rd/vol | RD 350, vol 0.09, tau 1.2, w 0.8 | 0.62794 |
| m-fine | RD 150, vol 0.09, tau 1.2, w 0.8 | 0.62589 |
| m-fine | RD 150, vol 0.09, tau 1.2, w 0.75 | 0.62556 |
| m-fine | RD 150, vol 0.09, tau 1.2, w 0.85 | 0.62652 |
