# Simply-Fit — Project Documentation

## What this is

Simply-Fit is a weight management system built around one observation:
people stop logging food within a week. The app removes that requirement
entirely. The user logs their weight once a day. Everything else is inferred.

The daily weight reading is the net result of everything the body consumed
and burned. If you can extract the signal from the noise in that number,
you do not need a food log. That is the premise this project is built on.

---

## How it evolved from HealthLens

HealthLens was my first version of this idea. It worked, but it was
essentially a calculator — you entered your goal, it returned a fixed
calorie target, and that number never changed regardless of what was
actually happening to your weight.

The core problem with that approach is the direction of the calculation.
HealthLens said: here is your deficit, this is what should happen.
Simply-Fit flips this. It observes what is actually happening and works
backward to infer the calorie balance. Same 7700 kcal/kg constant,
opposite direction.

The practical difference is significant. HealthLens gave the same
recommendation on day 1 and day 30. Simply-Fit's recommendation updates
every day based on the real trend in the user's weight data.

A second problem with HealthLens was that it treated every weight reading
as equally valid. A 1.5 kg spike from a salty dinner would shift the
calculations the same way a real fat gain would. Simply-Fit filters these
out before doing any calculation.

---

## What is actually different under the hood

**Noise filtering.** Raw daily weight is too noisy to act on directly.
Before any calculation runs, the weight log passes through an Exponential
Weighted Moving Average filter that separates the true fat-change trend
from daily water and glycogen fluctuation.

**Backward calorie inference.** A linear regression is fitted on the
smoothed trend. The slope gives kg change per day. Multiplied by 7700,
this is the user's actual daily calorie balance — inferred from their
body, not entered manually. A nominal 95% confidence interval is computed
from the slope's standard error (anti-conservative, because EWMA
pre-smoothing autocorrelates the residuals — documented in
src/ml_engine.py).

On simulated users with known ground truth, this inference recovers the
user's realized energy balance with a median error of ~8 kcal/day
(notebook 02).

**Anomaly detection.** Isolation Forest runs on the residuals between raw
readings and the smoothed trend. Flagged readings are replaced by their
smoothed value before the calorie calculation. Roughly 10% of readings are
flagged by design (contamination=0.1) — a deliberate choice matching
typical daily water-weight volatility.

**A note on the 7700 kcal/kg constant.** This is the classic Wishnofsky
rule and is applied uniformly. A truly personal kcal/kg coefficient cannot
be identified from weight data alone — the same weight trend is consistent
with many intake/expenditure combinations — so personalization in this
system comes from fitting the trend to the user's own data, not from
learning a per-user constant.

**7-day forecasting.** The forecaster predicts the next 7 days of weight
from the last 14. Two paths exist: an LSTM trained on the synthetic cohort
and a linear extrapolation fallback. On held-out (leakage-free,
walk-forward) evaluation the linear path actually wins on this data
(MAE 0.40 vs 0.54 kg — see results/model_evaluation.json and notebook 03);
the simulated trajectories are near-linear by construction. The LSTM is
kept as the research path for real-world data with non-linear patterns,
and the app degrades to the linear path transparently when TensorFlow is
absent.

**AI coach.** The user can ask questions about their progress. The coach
is an intent-routed, tool-calling pipeline (src/agent_coach.py), not an
LLM agent loop: a keyword-scoring intent classifier routes the question,
deterministic tools always run (calorie inference, 7-day forecast,
clinical guideline retrieval, plateau detection), and the final response
is synthesized by the Google Gemini API when a key is available —
otherwise by grounded response templates that use the same tool outputs.
The design is deliberate: behaviour is reproducible and testable with
zero API cost, and the numbers the LLM sees are always computed by tested
code.

---

## Data and validation

There is no large public dataset of daily weight logs with disease
conditions, so I generated synthetic training data — 500 users across
90 days each, using real physiological equations for BMR, TDEE, and noise
parameters from clinical weight variability literature.

The generator's demographics (age, gender, height, start weight) are
fitted to a real, committed extract of CDC NHANES 2017-2018 — 5,434 US
adults built directly from the CDC public files (DEMO_J.XPT and BMX_J.XPT).
Distribution alignment is checked with two-sample Kolmogorov-Smirnov tests
(notebook 05, src/calibration.py):

- The original naive parameterization (age ~ Uniform(18, 60), weight ~
  N(88.5, 25)) was ~7 kg heavier and ~10 years younger than the real
  population (weight KS 0.178, age KS 0.361).
- The calibrated generator achieves weight KS 0.059 (a 67% reduction —
  the distributions are no longer statistically distinguishable,
  p = 0.082) and age KS 0.085 (a 76% reduction).
- Switching weight sampling from Gaussian to gender-conditional
  lognormal did most of the weight-alignment work, because NHANES weight
  is right-skewed.

This validation step matters because it shows the training population
reflects real US adult demographics rather than arbitrary assumptions —
while being honest that it validates the marginals, not the dynamics.

---

## Known limitations

The LSTM is trained on synthetic data and does not beat linear
extrapolation there — the honest conclusion of the evaluation in
notebook 03. It may not fully generalise to real users with illness,
medication effects, or hormonal variability; the inference layer
compensates by adapting to each user's individual data over time.

The inference's 95% CI is anti-conservative (EWMA pre-smoothing), the
7700 kcal/kg constant is an approximation, and the KS comparison is
unweighted (NHANES survey weights not applied).

The coach's clinical guidelines (data/public/clinical_guidelines.json)
are a small curated set used for grounding — not a medical database, and
not a substitute for professional medical advice.

---

## Tech stack

Python, Pandas, NumPy, Scikit-learn (EWMA pipeline, Isolation Forest,
TF-IDF retrieval), SciPy (KS test), TensorFlow/Keras (LSTM),
Streamlit, Google Gemini API (optional), CDC NHANES 2017-2018.

---

## How to run

```bash
git clone https://github.com/tyagiji369/Simply-Fit.git
cd Simply-Fit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # core: runs the full app
streamlit run app/streamlit_app.py
```

For training, tests and notebook execution:

```bash
pip install -r requirements-full.txt
pytest -q                                # 22 tests
python -m src.lstm_forecaster            # retrain + re-evaluate
jupyter nbconvert --to notebook --inplace --execute notebooks/*.ipynb
```

Optional: copy `.env.example` to `.env` and add a free Gemini API key
(https://aistudio.google.com) to enable LLM coach responses.

Not a substitute for professional medical advice.
