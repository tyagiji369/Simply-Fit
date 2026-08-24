# Simply-Fit — Project Documentation

## What this is

Simply-Fit is a weight-management system built around one observation:
people stop logging food within a week. The app removes that requirement
entirely. The user logs their weight once a day. Everything else is inferred.

The daily weight reading is the net result of everything the body consumed
and burned. If you can extract the signal from the noise in that number,
you don't need a food log. That is the premise this project is built on.

## How it evolved from HealthLens

HealthLens was the first version of this idea: it essentially worked
forward — you entered a goal, it returned a fixed calorie target, and that
number never changed regardless of what was happening to your weight.

Simply-Fit flips the direction. It observes what is actually happening and
works backward to infer the calorie balance. Same 7700 kcal/kg constant,
opposite direction. The recommendation updates daily from the real trend,
and water-retention spikes are filtered out before any calculation.

## What is inside (current architecture)

**Noise filtering.** Before any calculation, the weight log passes through
an Exponential Weighted Moving Average (span 7) that separates the true
fat-change trend from daily water and glycogen fluctuation.

**Anomaly detection.** Residuals between raw readings and the EWMA trend
are examined. For logs ≥ 30 days an Isolation Forest is used with a
*hard-capped* contamination (≤ 3 flags or 8% — it can never flag ~10% of
the points "by construction" for no reason). For shorter logs a robust
MAD-based z-score is used instead, because a contamination fraction on
7–20 points is statistically meaningless. Flat or near-flat logs produce
zero flags. Flagged readings are excluded from the calorie estimate.

**Backward calorie inference.** OLS on the *cleaned raw* series (the EWMA
smoothed series is used for the chart, not for the regression: a moving
average attenuates the slope by ~6%, measured on our own synthetic data).
The slope × 7700 gives the inferred kcal/day balance, reported **with a
95% confidence interval** built from the residual scatter, plus separate
28-day and full-trend estimates so a changing rate is visible.

**LSTM forecasting.** A 2-layer LSTM (64→32 units, dropout 0.2) maps
14 days of history to a 7-day forecast. Fixes over the original version:
each window is normalised by *its own* min/max at both train and inference
time (the old code fit the scaler on the full 90-day series at training,
but on arbitrary history at inference — a scale mismatch), and users are
split into train/val/test groups so no user's data leaks across splits.
The model is evaluated against the deterministic linear-extrapolation
baseline on held-out users: **MAE 0.375 kg vs 0.664 kg (43.6% better)**
(see `data/synthetic/lstm_evaluation.json`, reproducible via
`scripts/retrain_and_evaluate.py`). The forecast is now shown in the UI.

**AI coach.** The coach is **intent-routed and tool-assisted** (not a
self-directed ReAct agent — the docs used to say that, which overstated
it). The pipeline classifies the question, computes the relevant metrics,
forecast and retrieved guidelines, and only then either answers
deterministically from those metrics or sends them as grounded JSON to a
Gemini model. The LLM can never invent user numbers: they are computed
before the call, and the prompt is explicit that only measured data may
be used. Responses are capped at ~150 words, never diagnose, and defer
to clinicians when guidelines conflict.

**Clinical RAG.** TF-IDF + cosine retrieval over a curated 7-document
guideline corpus. When the user has a condition (e.g. CKD), retrieval is
restricted to condition-matched guidelines first, so a renal user never
gets a generic sodium answer instead of the renal-specific one.

## Data & validation

There is no large public dataset of daily weight logs with disease
conditions, so the training data is synthetic: 500 users × 90 days,
generated from Mifflin-St Jeor BMR → TDEE → daily weight delta, with
glycogen/sodium/measurement noise that is amplified for relevant
conditions (e.g. hypertension, PCOS).

The synthetic population is validated against a **real NHANES sample that
ships in the repository**: `data/public/nhanes_reference.csv` (7,420 adults
18–80 from CDC NHANES cycles 2009–10 and 2011–12, public-domain data
mirrored by ProjectMOSAIC). The old in-app "validation" generated
Gaussian samples and *called them* NHANES — that has been removed.

Current KS-test result for 500 synthetic users (`src/calibration.py`,
also shown in the app's "Validation" expander):

| Variable | KS before | KS after | Improvement | p-value |
|---|---|---|---|---|
| Weight | 0.185 | 0.065 | 64.9% | 0.037 |
| Age | 0.283 | 0.044 | 84.5% | 0.314 |

Means: synthetic **79.6 kg / 45.7 yrs** vs NHANES **82.0 kg / 46.2 yrs**.

**Honest reading of these numbers:** calibration substantially reduced the
population-level bias; age is no longer statistically different (p = 0.31),
and weight is much closer but still significantly different (p = 0.037).
Earlier versions of this document and the README claimed otherwise
("62%/75%", "82%/76%", "zero population bias"); all of those have been
corrected to a single number set generated by the code that ships here.

## Safety decisions

- Calorie estimates are shown with a CI, not as exact values.
- If CKD is selected, the generic 1.6–2.0 g/kg protein recommendation is
  **replaced** by the renal guidance (0.6–0.8 g/kg, KDOQI) and the generic
  fluid target is replaced by "follow your clinician".
- Meal plans are scaled to the user's actual calorie target instead of
  silently showing a fixed menu under a different label.
- Plateau advice is evidence-caveated ("if it persists 3–4 weeks…") rather
  than asserting that a refeed "restores leptin and T3".

## Known limitations

- The LSTM is trained on synthetic data; real users with illness,
  medication or hormonal effects aren't fully represented. The evaluation
  gives the model's error on test users (~0.37 kg over 7 days) rather than
  an unstated claim.
- Calorie inference uncertainty is typically ±100–200 kcal/day — shown in
  the UI.
- The guideline corpus is small (7 entries). The coach flags this and
  defers to clinicians.
- No persistence layer yet: data lives in Streamlit session state.

## Tech stack

Python, Pandas, NumPy, SciPy, Scikit-learn, TensorFlow/Keras, Streamlit,
Google Gemini API (`google-genai`), real NHANES reference data, pytest + CI.

## How to run

```bash
git clone https://github.com/tyagiji369/Simply-Fit.git
cd Simply-Fit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Not a substitute for professional medical advice.
