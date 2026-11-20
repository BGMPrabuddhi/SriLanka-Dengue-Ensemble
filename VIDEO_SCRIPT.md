# Video script: Sri Lanka Dengue Early-Warning (≤ 3 minutes, ~400 words)

> Speak at a calm pace (~140 words/minute). **[Show: …]** tells you what to have on screen.
> All numbers come from `models/results.csv` and `models/summary.json`.

---

### 0:00 – 0:20 · The problem
**[Show: Dashboard → Home, national trend chart]**

"In 2017, Sri Lanka recorded more than one hundred and seventy thousand dengue cases, over ten thousand in a single week.
Health officials must decide *this* week where to send fogging teams and hospital resources for *next* week. So I asked:
can ensemble machine learning forecast next week's dengue cases in every region?"

### 0:20 – 0:45 · The data
**[Show: Home KPI cards, then scroll the trend; briefly open notebook 01 heatmap]**

"I used real Sri Lankan data: weekly cases for twenty-six health regions from 2006 to 2026, from the denguedatahub
package, plus rainfall, temperature and humidity from the Open-Meteo archive. Cleaning revealed real reporting errors,
such as copied weeks and counts assigned to the wrong region. I flagged them as missing instead of inventing values."

### 0:45 – 1:30 · The methods
**[Show: Model comparison → 'The ensemble families' tab]**

"Each forecast uses the last four weeks of cases, rainfall and humidity up to eight weeks earlier, and the season. I split
the data by time: train to 2021, validate on 2022, test on 2023 to 2026, so no model ever sees the future.
I compared bagging (bagged trees, Random Forest, Extra Trees), which reduces variance; boosting (AdaBoost, Gradient
Boosting, XGBoost, LightGBM, CatBoost), which reduces bias; stacking; and two hybrids: Random Forest with SARIMA, and
Random Forest with a neural network. Every model is measured against the simple rule 'next week equals this week'."

### 1:30 – 2:30 · Live dashboard demo
**[Click: District forecast & risk → choose Colombo]**
"Here is next week's forecast for Colombo with a colour-coded risk level, and below it all twenty-six regions, ranked."

**[Click: Model comparison → Accuracy tab, then Normal vs outbreak weeks tab]**
"Bagging beat the naive rule by about three percent. Boosting was the best in calm weeks but under-predicted outbreaks,
ending up to twenty-four percent worse. My tests traced this to the log transformation of the target."

**[Click: Actual vs predicted tab → Colombo]**
"You can see every model lagging behind the sudden 2026 surge."

**[Click: Explainability & what-if → move 'Cases reported last week' slider]**
"Recent cases drive about eighty-five percent of the forecast; rain five to seven weeks earlier adds a little. Moving the
slider updates the forecast live."

### 2:30 – 3:00 · Results and conclusion
**[Show: Model comparison → 'Which approach performed best, and why' panel]**

"The best test result came from combining models whose errors cancel: eighty percent Random Forest plus twenty percent
neural network, five point one percent better than naive, closely followed by stacking at four point eight percent.
The lesson: in outbreak-heavy years, diversity in *how* models fail matters more than any single algorithm.
This project was built with the help of Claude Code, an AI assistant, which I have disclosed in the repository. Thank you."
