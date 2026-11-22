# Written analysis

This file holds the explanations that accompany the notebooks (which contain only code and outputs): data-quality
decisions, the key findings of each notebook, the discussion of **why the models behaved as they did**, research notes
(Random Forest vs XGBoost / LightGBM / CatBoost; tree ensembles on tabular vs image/text data) and limitations.
All numbers come from the executed notebooks and `models/results.csv`.

> AI-use disclosure: drafted with the help of Claude Code (Anthropic) and reviewed by B. G. M. Prabuddhi
> (see [AI_DISCLOSURE.md](../AI_DISCLOSURE.md)).


## 01 · Data collection, cleaning & exploratory analysis
*Notebook:* [`01_data_collection_EDA.ipynb`](../notebooks/01_data_collection_EDA.ipynb)

#### Values that are really *reporting errors* or *missing reports*
Some numbers are not "no dengue" or "a real count" but artefacts of reporting. Five patterns were found
(each rule is in `flag_missing_reports()` and was checked against the full 20-year history):

| Reason | Rule | Examples |
|---|---|---|
| `full_year` | a region reports **0 for a whole year** but has cases in other years | Kilinochchi & Mullaitivu 2007-09 (end of the civil war), Monaragala 2020 |
| `gap_week` | **≥ 20 of 26 regions report 0** in one week while neighbouring weeks have only a few | 10 Sep 2022 (23 regions vs 1-6 nearby). Mid-2012 has a week with 20 zeros, but its neighbours have 13-19, so it is *not* flagged |
| `dropout` | one region reports **0 between two weeks with ≥ 10 cases** (a partial report) | Kandy 55 → 1 → 92 type patterns, e.g. 22 Apr 2023 |
| `scrambled` | **≥ 4 regions jump above 4× both neighbouring weeks** while large regions collapse: counts were put against the wrong regions | 30 Mar 2026 and 8 Jun 2026 (no other week in 20 years has more than 2 such regions) |
| `duplicate` | **every** region's count is identical to the previous week (a copied report) | 21 Jan 2012, 18 Dec 2021, 8 Feb 2025, 22 Jun 2026 |

These rows are set to `NaN` with `missing_report = True` and a `missing_reason`. We do **not** impute or re-assign
values (that would invent data); missing lags are handled inside the model pipeline (notebook 02).

#### Outliers: keep the epidemics, flag only the reporting errors
Weekly counts are extremely skewed (median 9, maximum 2,631), so a standard rule flags many weeks as "outliers".
For an early-warning system those weeks are **the signal, not noise**: deleting or capping them would teach the models
that outbreaks never happen. We check whether the extreme weeks are *genuine* using a per-district IQR rule
(value > Q3 + 3·IQR of that district's weekly cases):
* if they cluster in known epidemic years and occur in **many districts at the same time**, they are real outbreaks → **kept**;
* values that jump in one week and fall straight back, or that look like counts moved between districts, are
  **reporting errors**; they were already flagged as missing by `flag_missing_reports()` above.
Skewness is handled later with a log1p target (notebook 02), not by removing data.

### Key findings

* **Data obtained automatically and unaltered:** 26,753 weekly reports for **26 RDHS regions** from
  **23 Dec 2006 to 13 Sep 2026**, plus Open-Meteo weather for every region-week (no gaps).
* **Cleaning:** 3 spelling variants fixed, 2 date formats unified, and year/week rebuilt from dates
  (the source labels are inconsistent at New Year). **570 district-weeks (2.13%) flagged as reporting errors /
  missing reports**: whole missing years (366: northern regions 2007-2009, Monaragala 2020), copied weeks (104: 4 weeks
  identical to the week before), counts assigned to the wrong regions (52: 30 Mar and 8 Jun 2026), single-week
  dropouts (25) and the 10 Sep 2022 national gap (23). They are left as missing, not imputed or re-assigned.
* **Outliers are epidemics, so they are kept.** 1,000 district-weeks (3.8%) are extreme by a per-district IQR rule
  (> Q3 + 3·IQR), but 77% of them fall in the four epidemic years (2017, 2019, 2023, 2026) and 79% occur while at least
  3 regions are extreme in the same week. They are real outbreaks, the very thing an early-warning system must learn,
  so they are not removed or capped; skewness is handled with a log1p target instead. Only reporting errors are flagged.
* **Burden is concentrated:** Colombo (21.8%), Gampaha (12.8%) and Kandy (8.3%) account for ~43% of all
  2007-2025 cases, so absolute errors will be dominated by the Western Province.
* **Strong outbreaks / non-stationarity:** the record epidemic peaked at **10,590 cases in the week of 15 Jul 2017**
  (174,687 cases that year). 2019, 2023 and 2026 were also large outbreak years (2026 already has 83,015 reliably reported cases by mid-September).
* **Seasonality:** the typical year peaks around **week 28 (mid-July, South-West monsoon)** with a second rise in
  Dec-Jan after the North-East monsoon rains (weeks 41-48), and is lowest around **week 15 (April)**, which motivates `week_sin/week_cos`.
* **Persistence is the strongest signal:** median lag-1 autocorrelation is **0.893** (lag-4: 0.66), so lagged cases
  will dominate, and the naive "same as last week" baseline will be hard to beat.
* **Weather leads cases by ~2 months:** after removing district averages, rainfall correlates most with cases
  **8-9 weeks later** (ρ ≈ 0.133) and humidity at **7 weeks** (ρ ≈ 0.139); at the 1-4 week lags in the brief the
  correlation is only 0.01-0.06. Weather is a weak but real, biologically plausible signal (breeding sites + incubation).
* **Split:** train = 2007-2021, validation = 2022, test = 31 Dec 2022 to 13 Sep 2026 (3.7 years). The test period
  is an *outbreak-heavy* period (mean 52.6 cases/region-week vs 29.9 in training and 23.9 in validation), a tough but realistic test.

---

## 02 · Feature engineering, chronological split & baselines
*Notebook:* [`02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb)

### Key findings

* **Feature table:** 35 numeric features (4 case lags, 3 rolling stats, the `lags_missing` flag, 24 weather lags,
  3 calendar) + district one-hot → **64 model columns** after preprocessing. Leakage checks pass: every feature uses week t
  or earlier, and `target(t) == cases_lag1(t+1)` for every reported week in all 26 regions.
* **Split:** train **19,731** rows (2007-02-24 → 2021-12-25), validation **1,328** (2022), test **4,854**
  (2022-12-31 → 2026-08-24). The 2 most recent weeks (52 rows) are excluded from test as possibly incomplete; the latest week
  (26 rows) is the dashboard's forecast input for the week of **2026-09-14**.
* **Missing reports:** only rows whose **target** is a missing report were dropped (554). Gaps inside the lag window are
  forward-filled within the district and counted in `lags_missing`; the median imputer is only a fallback (≤ 12 training rows).
* **Baselines on test:** naive (last week) **MAE 14.95**, RMSE 37.59, R² 0.867. The seasonal naive (same week last year) is
  far worse (MAE 37.75, **153% worse than naive**) because outbreak timing changes from year to year. Every model is reported
  as % improvement over the naive MAE.
* **Target transform:** the target is extremely skewed (skewness 10.8 raw vs 0.34 after log1p). On validation, log1p lowered
  MAE for both ensembles (RF 10.67 → 10.33, LightGBM 10.70 → 10.53), so **all models are trained on log1p(cases)** with
  `expm1` back-transform. (Notebook 07 shows the price of this choice in outbreak weeks.)
* **Risk thresholds** are per-district 50th/75th percentiles of 2007-2021 weekly cases (e.g. Colombo: Medium > 110,
  High > 217). The 2023-2026 test period is outbreak-heavy: **55% of test district-weeks exceed the High threshold.**

---

## 03 · Bagging ensembles: Bagged Trees, Random Forest, Extra Trees
*Notebook:* [`03_bagging_models.ipynb`](../notebooks/03_bagging_models.ipynb)

### Key findings

* **All three bagging ensembles beat the naive baseline on test:** Extra Trees MAE **14.51** (+3.0%), Random Forest
  **14.52** (+2.9%, and the only one that also beats naive on RMSE, 37.47 vs 37.59), bagged trees 14.70 (+1.7%).
* **The gain comes from normal weeks.** Normal-week MAE drops from 7.43 (naive) to **6.03–6.20 (17–19% better)**; in
  outbreak weeks the forests are about level with naive (21.26–21.75 vs 21.06). Averages of past similar weeks move more
  slowly than a fast-rising epidemic.
* **Variance reduction is visible:** averaging trees lowered validation MAE by 14–27% compared with a single tree (bagged
  trees 14.34 → 10.45, RF 13.16 → 10.39, Extra Trees 12.08 → 10.34). Bagged trees gain most because a single unrestricted
  tree has the highest variance; random feature/threshold choices (RF, Extra Trees) start from a better single tree.
* **OOB vs time-ordered validation:** RF OOB R² (log scale) = **0.842**, but on the time-ordered 2022 validation year R² =
  **0.746**. OOB samples random weeks *inside* the training period, so it is optimistic for forecasting the future.
* **Recent cases dominate:** lagged/rolling cases carry **89%** of RF importance, weather 10%, calendar 1%.
* **Selection by validation:** Extra Trees has the lowest validation MAE (10.34 vs RF 10.39) and is saved as the best
  bagging model; Random Forest is saved too (used by the hybrids). Time-series CV over 2007-2021 instead ranks RF first
  (13.56 vs Extra Trees 14.89). Files: RF 15.4 MB, Extra Trees 15.6 MB.

---

## 04 · Boosting ensembles: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
*Notebook:* [`04_boosting_models.ipynb`](../notebooks/04_boosting_models.ipynb)

### Key findings

* **On validation (2022) boosting looks as good as bagging** (val MAE 10.43–10.58 vs 10.34–10.45), but **on the
  outbreak-heavy test period every boosting model is worse than the naive baseline**: Gradient Boosting 15.45 (−3.3%),
  CatBoost 15.71 (−5.1%), AdaBoost 16.02 (−7.2%), XGBoost 16.51 (−10.4%), LightGBM 18.59 (−24.4%), vs naive 14.95 and
  bagging 14.51–14.70.
* **Boosting wins calm weeks and loses outbreak weeks.** Normal-week MAE is the best of any family (5.89–6.33 vs naive
  7.43), but outbreak-week MAE is 23.2–28.8 vs 21.06 for naive. Outbreak weeks carry most of the absolute error, so the
  overall ranking flips.
* **Bias reduction is visible round by round:** Gradient Boosting's training MAE falls from 28.0 (1 round) to 8.6 (400
  rounds), while validation MAE is lowest at round 35 (10.43) and then rises (10.68 at round 400), classic over-fitting
  that early stopping prevents.
* **Early stopping on a calm year is only part of the problem:** with full rounds instead of early-stopped ones, XGBoost
  (16.51 → 15.57) and LightGBM (18.59 → 17.47) improve but stay worse than naive, while AdaBoost, Gradient Boosting and
  CatBoost get worse. Notebook 07 identifies the main cause: the log1p target's retransformation bias in outbreak weeks.
* **Same features drive both families:** XGBoost puts most of its gain on recent cases, like RF.
* **Speed and size:** boosted models train in 0.3–4 s (forests 5–8 s) and are tiny on disk (≤ 0.2 MB vs ~15 MB).
* Best boosting model on validation: **Gradient Boosting** (val MAE 10.430).

---

## 05 · Stacking and voting
*Notebook:* [`05_stacking.ipynb`](../notebooks/05_stacking.ipynb)

### Key findings

* **The base learners are not diverse:** their 2022 validation errors are **97.8% correlated** on average (0.970–0.985).
  They lean on the same lagged-case features and miss the same surges; even the naive baseline's errors correlate 0.88 with
  theirs. In theory, stacking helps most when errors differ, so a large "diversity bonus" was not expected.
* **Stacking (RF + XGBoost + LightGBM + CatBoost → RidgeCV) still beat every single model on test:** MAE **14.23**
  (**+4.8%** vs naive; RMSE 36.88, +1.9%), with the best outbreak-week MAE of any pure ensemble (20.37 vs 21.06 naive), at
  the cost of slightly worse normal weeks (6.69).
* **Why it helped: a learned bias correction.** Meta-weights: RF **0.37**, XGBoost **0.40**, CatBoost **0.25**,
  LightGBM 0.02. They sum to **1.05**, so the stack scales the (log) forecast up slightly, partly offsetting the base models'
  shared under-prediction of surges.
* **Voting (simple mean) did not help:** −4.4% vs naive (MAE 15.61). Three of its four members are boosters that
  under-predict outbreaks, and equal weights cannot correct that. It has the better *validation* MAE of the two (10.41 vs
  10.46), so the pre-set rule saves it as the stacking-family winner.
* **Validation vs test disagree:** on calm 2022, Voting < Stacking; on the outbreak-heavy test, Stacking ≪ Voting.
* Training cost: stacking 27 s (4 base models × 4 time-ordered folds + refit), voting 10 s; files ~16 MB each.

---

## 06 · Hybrid models: RF + SARIMA (time series) and RF + ANN (tabular)
*Notebook:* [`06_hybrid_RF_ARIMA_RF_ANN.ipynb`](../notebooks/06_hybrid_RF_ARIMA_RF_ANN.ipynb)

### Key findings

**RF + SARIMA (single weekly series, walk-forward one-step-ahead on 2023–2026)**
* SARIMA(1,0,1)(1,0,1)₅₂ was chosen for both series. The ADF test rejects a unit root for log cases (p = 0.016 national,
  0.008 Colombo), so no differencing; yearly seasonality lowered AIC from 328.5 → 291.5 (national) and 1151 → 1084 (Colombo).
* **National total: the hybrid is the best model** with MAE **202** cases/week (**+4.0% vs naive** 211), RMSE 398 vs 412,
  R² 0.872. Both components alone are *worse* than naive (SARIMA −2.0%, RF −17.9%). SARIMA supplies a smooth seasonal level;
  the RF corrects its residuals using weather and recent dynamics, which is the textbook case for a hybrid.
* **Colombo: nothing beats naive.** Hybrid MAE 62.0 vs naive 58.1 (−6.6%); SARIMA alone 60.5 is closest. Colombo's June 2026
  surge rose faster than any one-week-ahead model could follow.

**RF + ANN (all regions)**
* With log-scaled inputs the **ANN alone** reaches test MAE 17.44 (−16.7% vs naive). It is the only model that forecasts
  *above* past maxima, which helps at peaks but over-shoots elsewhere (RMSE 65.9).
* **(a) Weighted average: the best model of the whole project on test.** Weight tuned on 2022 = **0.80 RF + 0.20 ANN**,
  test MAE **14.19 (+5.1% vs naive)**, RMSE 36.89, outbreak-week MAE 20.85 (better than naive's 21.06). RF and ANN errors are
  0.93 correlated, but in outbreaks their **biases are opposite** (forest too low, network can over-shoot), so blending cancels
  part of the error. (Diagnostic: 0.80 is also the best weight on test.)
* **(b) RF → ANN** (forest forecast as an input) is worse than naive (MAE 16.38, −9.5%): the ANN trusts the forest feature
  and amplifies its outbreak errors.
* Lesson: a hybrid helps when its components **fail in different ways** (SARIMA + RF on the national series; RF + a small
  share of ANN), not when one model is simply fed into another.

---

## 07 · Final comparison: which ensemble approach works best for Sri Lankan dengue forecasting, and why?
*Notebook:* [`07_final_comparison.ipynb`](../notebooks/07_final_comparison.ipynb)

### Core comparison: Bagging vs Boosting vs Stacking
One representative of each ensemble strategy, all on identical features, target, split and test weeks:
* **Bagging → Random Forest**: deep trees trained in parallel on bootstrap samples, predictions averaged (reduces **variance**).
* **Boosting → XGBoost**: shallow trees trained in sequence, each correcting the current errors (reduces **bias**).
* **Stacking → RF + XGBoost + LightGBM + CatBoost → RidgeCV**: a meta-learner, trained on time-ordered out-of-fold
  predictions, learns how to combine the base models.

### Why did the models behave this way?

#### The naive baseline is strong because dengue is persistent
Weekly cases have a median lag-1 autocorrelation of **0.893** (notebook 01): this week is the best single guide to next
week. "Same as last week" therefore scores MAE **14.95** on test, and SHAP shows every tree model also spends ~**85%** of
its attention on the last 4 weeks of cases and their averages. The realistic goal is not to replace persistence but to
*improve on it*: forecasting the direction of change and smoothing out reporting noise. Every percentage point over
naive is hard-won.

#### Bagging reduces variance, so it is the most reliable single family
A single deep tree reacts to noise; averaging many de-correlated trees cancels that noise (validation MAE fell 14–27% from
1 tree to the full ensemble, notebook 03). In normal weeks this gives forecasts **17–19% more accurate than naive**
(MAE 6.0–6.2 vs 7.43). In outbreak weeks bagging is about level with naive (21.3–21.8 vs 21.06): an average of past
similar situations moves more slowly than a fast-rising epidemic. Net result: Extra Trees **+3.0%**, Random Forest **+2.9%**,
bagged trees +1.7%. Random Forest also has the lowest time-series CV error over 2007–2021 (13.56), which includes the 2017
and 2019 epidemics.

#### Boosting reduces bias, but here that bias reduction did not survive the outbreaks
Boosting fits calm, typical weeks best of all (normal-week MAE **5.89–6.33**), which is what sequential bias reduction
should do. But every booster was worse than naive overall (**−3% to −24%**), because outbreak-week errors were 23–29.
Three reasons, each checked:
1. **Log target → retransformation bias.** Models trained on log1p(cases) predict something close to a geometric
   mean; `expm1` turns that into a forecast that is systematically **too low when values are large and volatile**
   (Jensen's inequality). The raw-vs-log1p diagnostic in notebook 07 shows it: **XGBoost on the raw target scores test MAE 14.37**
   (vs 16.51 with log1p, outbreak-week bias −6.3 instead of −20.4), which is *better than Random Forest and +3.9% vs
   naive*; LightGBM improves from 18.59 to 14.60. Random Forest is hardly affected (14.54 vs 14.52) because its leaves
   average whole groups of similar rows, while boosting's additive, shrunken steps in log space amplify the effect.
   So boosting's poor ranking is caused by the target transformation, **not by boosting itself**.
2. **Calm validation year.** Early stopping picked few rounds (10–104) on 2022, a low-incidence year. Using full rounds
   helped XGBoost and LightGBM but hurt AdaBoost, Gradient Boosting and CatBoost (notebook 04), so this is only part of the story.
3. **Shallow trees + shrinkage** need many rounds to reach extreme values; under strong regularisation they stay close
   to typical levels, which is excellent for normal weeks and poor for surges.

The log1p decision was made correctly on validation (it lowered 2022 MAE for RF, XGBoost and LightGBM), which is exactly
why it is instructive: the validation year did not contain the kind of week where its cost appears. We do **not** switch
to the raw target after seeing the test results, because that would be choosing a model on the test set; it is reported as
a finding and as future work (choose the target transform on validation data that include an epidemic year).

#### Stacking helps only if base learners make different errors, and here they mostly did not
The four base models' validation errors are **97.8% correlated**: they rely on the same lag features and miss the same
surges. Simple **voting** therefore gained nothing (−4.4%), because it averaged three under-forecasting boosters with RF.
**Stacking** did better (+4.8%, the best single "pure ensemble"), for a specific reason: the Ridge meta-learner's weights
(RF 0.37, XGBoost 0.40, CatBoost 0.25, LightGBM 0.02) sum to **1.05**, so it scales the forecast up slightly, correcting part
of the shared under-prediction. It learned a *bias correction* more than a *diversity bonus*.

#### Hybrids work when the components fail in different ways
* **RF + ANN (weighted 0.8 / 0.2) was the best model on test (+5.1%).** Their errors are correlated (0.93), but their
  **biases in outbreaks are opposite**: trees cannot easily push forecasts upward (Extra Trees mean outbreak error −12.4),
  while the neural network, which extrapolates linearly, *over-shoots* (its largest forecast was 3,706 vs a maximum actual of
  1,625; outbreak bias −1.0). A small dose of the ANN lifts the forest's outbreak forecasts; for Colombo's 2026 peak it
  forecast 1,420 vs 1,003 (Extra Trees) and an actual of 1,625.
* **RF → ANN** (forest forecast as an input) did worse (−9.5%): the ANN learned to trust the forest feature and then
  amplified its errors.
* **RF + SARIMA** beat naive on the national total (+4.0%) while SARIMA alone (−2.0%) and RF alone (−17.9%) did not: the
  linear seasonal model supplies a smooth level, the forest corrects its residuals with weather and recent dynamics. For
  Colombo no model beat naive; its June 2026 surge outran every one-week-ahead forecast.

#### Distribution shift: outbreak-heavy test years, not unseen magnitudes
The test period is very different from training (mean 52.9 vs 30.1 cases per region-week; **55% of test weeks are
"outbreak" weeks**). However, only **0.4%** of test weeks exceed their region's 2007–2021 maximum (Colombo's 2026 peak of
1,625 is below its 2017 record of 1,972). So "trees cannot extrapolate" is true but is **not** the main cause here. The
problem is that outbreak-level weeks were *rare* in training, so trees' leaves average them with calmer weeks, and the log
target pulls forecasts down further. For the best model of each family, error is 3.4–4.0× larger in outbreak weeks than in
normal weeks (naive: 2.8×).

#### Weather matters a little, at the lag biology predicts
Weather accounts for ~**8%** of the SHAP attribution. The largest weather effects are at lags of **5 and 7 weeks** (rain and humidity),
matching notebook 01 (correlation peaking at 7–9 weeks) and the biology: rain creates breeding sites → eggs to adult
*Aedes* in ~1–2 weeks → the mosquito must bite an infected person and incubate the virus (~8–12 days) → human incubation
(~4–7 days) → symptoms, clinic visit and notification (~1 week). Weather is useful mainly to anticipate *turning points*,
but one week ahead, recent case counts already contain most of that information.

#### Validation vs test
Validation and test rankings agree only moderately (Spearman 0.74). Selecting on a single calm year (2022) chose
Extra Trees (test +3.0%), while multi-year time-series CV (2007–2021, including epidemics) ranks Random Forest first, and
the test favours combinations that correct outbreak under-prediction. For deployment, validation should cover outbreak years
(e.g. rolling-origin evaluation over several seasons).

### Research notes

#### Random Forest vs gradient boosting (XGBoost, LightGBM, CatBoost)

| | **Random Forest** (bagging) | **XGBoost** | **LightGBM** | **CatBoost** |
|---|---|---|---|---|
| How trees are built | Deep trees, **in parallel**, each on a bootstrap sample with random feature subsets; predictions **averaged** | Shallow trees **in sequence**; each fits the gradient of the loss of the current ensemble; L1/L2-regularised objective, second-order (Newton) steps | Same idea, but trees grow **leaf-wise** on feature **histograms**; GOSS row sampling, EFB feature bundling | Same idea, with **ordered boosting** (residuals computed by models that never saw the row) and **symmetric (oblivious) trees** |
| Main error it attacks | **Variance** (averaging de-correlated trees) | **Bias** (each round corrects the previous) | Bias | Bias, with less over-fitting on small data |
| Tuning effort | Low: works well near defaults; more trees never hurt accuracy | High: learning rate, depth, rounds, regularisation interact; needs early stopping | High; `num_leaves` must be limited or it over-fits | Moderate; strong defaults |
| Speed | Parallel, but deep trees → large models (MBs per hundred trees) | Fast (histogram method) | Usually the fastest | Slower to train, fast to predict |
| Categorical features | Need encoding (one-hot here) | Need encoding (native support is newer/experimental) | Native categorical splits | Native, via ordered target statistics, its signature feature |
| Over-fitting behaviour | Hard to over-fit by adding trees; OOB estimate for free | Over-fits with too many rounds → early stopping essential | Same; leaf-wise trees over-fit small data | More robust by design |
| Typical result on tabular benchmarks | Strong, robust baseline | Often top accuracy after tuning | Similar accuracy, faster | Often best with many categorical features / little tuning |

In this project, the Random Forest vs gradient boosting head-to-head table in notebook 07 shows how the four compare on the same data. *Why did the models behave this way?* above
explains why the ranking came out as it did here.

#### Strengths and limitations of tree ensembles: tabular vs image/text data

**Tabular data (like ours): tree ensembles are usually the method of choice.**
* Columns are heterogeneous (counts, rainfall, a district ID, a week number) with different scales and meanings.
  Trees split on one feature at a time with thresholds, so they need **no scaling**, are robust to skew and outliers,
  and handle interactions ("lots of rain *and* July *and* Colombo") automatically.
* They cope with missing values (natively in XGBoost/LightGBM/CatBoost, via imputation + flags here), and they
  give feature importances and exact SHAP values, which matter when a ministry must trust a forecast.
* Benchmarks such as Grinsztajn et al. (2022), *"Why do tree-based models still outperform deep learning on tabular
  data?"*, find that gradient-boosted trees and random forests still beat deep networks on typical medium-sized tabular
  datasets, partly because tabular targets are often irregular functions of a few informative features, which trees
  fit well and neural nets (biased towards smooth functions) do not.
* **Limitations:** trees produce step-wise predictions and **cannot extrapolate** beyond the target range seen in
  training (distribution-shift check in notebook 07); they do not model time explicitly (we must hand-craft lags); large forests are big files.

**Images and text: deep learning wins.**
* Pixels and words have *spatial or sequential structure*: meaning comes from local patterns (edges, shapes, n-grams,
  word order) that are the **same wherever they appear**. CNNs (weight sharing, translation invariance) and
  Transformers (attention over sequences, pre-trained on huge corpora) are built for this; trees are not.
* A single pixel or token column is almost meaningless on its own, so axis-aligned threshold splits on raw inputs are
  very inefficient, and the input dimension is huge (a 224×224 RGB image has ~150,000 features).
* Trees can still be useful **on top of learned representations**, e.g. gradient boosting on CNN or Transformer
  embeddings, or for structured metadata alongside images or text.

### Limitations
* **Reporting delays and revisions:** the most recent weeks may still increase as late notifications arrive (we
  excluded the last 2 weeks from testing for this reason); the dashboard's latest week can be an under-count.
* **Reporting errors in the source:** 570 district-weeks (2.1%) had to be flagged as missing, including whole missing
  years, copied weeks and two 2026 weeks whose counts were assigned to the wrong regions. Detection rules can miss
  subtler errors. We did not impute targets; lag gaps are median-imputed with was-missing flags.
* **Only one week ahead:** health planning often needs 2-4 weeks of lead time; accuracy will drop at longer horizons.
* **No mobility, serotype, vector-index or population data:** serotype switches (e.g. DENV-3 re-emergence) and human
  movement drive large outbreaks but are not in the features; cases are counts, not incidence per 100,000.
* **Weather at one point per district:** a single grid point (ERA5 reanalysis) cannot capture local rainfall/breeding
  conditions across a whole district.
* **Models trained on 2007-2021 only:** for real deployment they should be retrained on all data up to the latest week.
* **Single test period:** 2023-2026 is dominated by outbreak years; rankings could differ in a calmer period.

### Key findings

1. **Core comparison (bagging vs boosting vs stacking):** Stacking (RF + XGBoost + LightGBM + CatBoost → Ridge) is best
   (MAE 14.23, +4.8% vs naive), then Bagging / Random Forest (14.52, +2.9%), then Boosting / XGBoost (16.51, −10.4%).
2. **Persistence is king.** The naive "same as last week" forecast (MAE 14.95) is hard to beat; recent cases carry ~85% of
   the models' SHAP attribution.
3. **Bagging is the most reliable single family:** 17–19% better than naive in normal weeks, level with naive in outbreak
   weeks. Averaging reduces variance.
4. **Boosting is best in calm weeks but worst in outbreaks, and the cause is the log1p target, not boosting itself.**
   Trained on raw cases, XGBoost scores 14.37 (better than Random Forest) and LightGBM 14.60, instead of 16.51 and 18.59.
   The log1p choice was made (correctly) on the calm 2022 validation year, which hid this cost.
5. **Combining helps when it corrects the shared under-prediction:** stacking's meta-weights sum to 1.05 (+4.8%); the
   RF + ANN weighted hybrid (+5.1%, best on test) blends a forest that forecasts too low with a network that can
   over-shoot. Voting (−4.4%) and RF → ANN (−9.5%) did not help. RF + SARIMA beat naive on the national total (+4.0%).
6. **Model selection:** by the pre-set rule (validation MAE on 2022) the deployed model is **Extra Trees**; the best test
   model is the **RF + ANN hybrid**. A calm validation year cannot reveal outbreak behaviour, so validation should span
   several seasons including epidemics.
