# Simply-Fit: Interview Defense Cheat Sheet

Concise, honest 30-second answers for AI/ML and Data Science interviews.
**Only claim numbers that the code in this repo actually produces** — the
docs have been aligned so every figure below is reproducible
(`src/calibration.py`, `scripts/retrain_and_evaluate.py`, `pytest -q`).

---

### 1. Project Pitch & Core Motivation (30 Seconds)
**Q:** *"Can you explain what Simply-Fit is and why you built it?"*
> "Most diet apps rely on manual food logging, which users abandon within
> days and which is systematically misreported. Simply-Fit treats the
> scale as a passive physiological sensor: under the energy-balance
> equation (~7700 kcal/kg), daily weight reflects net calorie balance.
> The system extracts that signal with EWMA filtering, drops water spikes
> with anomaly detection, infers the user's actual calorie balance with a
> confidence interval, forecasts 7 days with an LSTM (evaluated against a
> linear baseline), and answers questions through an intent-routed coach
> grounded in the user's measured data — no food log required."

### 2. Signal Processing & Anomaly Filtering (30 Seconds)
**Q:** *"How do you handle daily weight fluctuations?"*
> "I apply an EWMA (span=7) filter to isolate the fat-mass trend. Then I
> look at residuals. For logs of 30+ days I use Isolation Forest with a
> capped contamination (at most ~8% or 3 points), and for shorter logs a
> robust MAD-based z-score — because a fixed 10% contamination on 14
> points forces ~1.4 'anomalies' by construction, even on pure noise. My
> tests assert that a log with one obvious +1.5 kg spike gets exactly one
> flag, and flat logs get zero."

### 3. Regression & calibration of the estimate (30 Seconds)
**Q:** *"Why do you regress on the raw series instead of the smoothed one?"*
> "Good catch — that was a real bug I fixed. A moving-average filter
> contracts the series, which attenuates the regression slope by about 6%
> at 30 days. So the EWMA is used for the chart and for anomaly
> detection, but the calorie regression runs on the anomaly-cleaned raw
> readings. I also report the 95% CI from the residual scatter instead of
> presenting one precise number; typical uncertainty is ±100–200 kcal/day."

### 4. Deep Learning & LSTM Forecasting (30 Seconds)
**Q:** *"Why an LSTM, and how do you know it works?"*
> "Linear projections miss plateaus and metabolic slowdowns. I use a
> 2-layer LSTM (64→32, dropout 0.2) with 14-day input → 7-day output.
> Two reproducibility fixes: each window is normalised by its own min/max
> at train *and* inference time (previously the scaler was fit on the full
> 90-day series for training but re-fit on arbitrary history at inference),
> and users are split into train/val/test groups. On 500 held-out users
> forecasting their final 7 days, the LSTM has **MAE 0.375 kg vs 0.664 kg
> for the linear baseline — 43.6% better**. The metrics JSON and the
> training script are in the repo."

### 5. Data Realism & NHANES Calibration (30 Seconds)
**Q:** *"Since you trained on synthetic data, how do you know it's realistic?"*
> "There's a real NHANES sample in the repo — 7,420 adults from CDC
> cycles 2009–10 and 2011–12 — and the generator is checked against it
> with a two-sample KS test. Current results: weight KS 0.185 → 0.065
> (64.9% improvement), age KS 0.283 → 0.044 (84.5% improvement). Means are
> now 79.6 kg / 45.7 yrs vs 82.0 kg / 46.2 yrs. I'm careful about the
> wording: age no longer differs significantly (p=0.31) but weight still
> does (p=0.037). Older versions of my docs claimed 'zero population
> bias' — that was wrong and I removed it. The honest framing is:
> calibration reduced the bias substantially but did not eliminate it."

### 6. AI Coach Architecture (30 Seconds)
**Q:** *"How does the coach work under the hood?"*
> "It is intent-routed and tool-assisted — I don't call it a ReAct agent
> any more, because that overstated the design. We classify the question
> (time-to-goal, calories, water weight, plateau, medical, general), the
> ML pipeline + forecast + clinical RAG are executed first, and then the
> LLM receives those precomputed metrics as grounding and must answer the
> exact question in under 150 words with no inventing or diagnosing. If
> no Gemini key is configured there is a deterministic synthesis that
> answers from the same metrics. Retrieval is TF-IDF + cosine over seven
> clinical guidelines, with condition-matched filtering (a CKD user gets
> renal guidance, not a generic answer)."

### 7. Safety design decisions (30 Seconds)
**Q:** *"What about contradicting advice for people with conditions?"*
> "Two examples I fixed: if CKD is selected, the generic 1.6–2.0 g/kg
> protein recommendation is replaced by the KDOQI renal range (0.6–0.8
> g/kg) and the fluid target becomes 'follow your clinician'. And meal
> plans are scaled to the user's actual target intake instead of showing a
> fixed 1170 kcal menu under an 1815 kcal label."

### 8. Tech Stack Overview
* **Languages & ML:** Python, Scikit-Learn (Isolation Forest, Linear Regression, TF-IDF), SciPy (KS test), Pandas, NumPy.
* **Deep Learning:** TensorFlow / Keras (LSTM, causal per-window scaling, group-aware splits).
* **GenAI & NLP:** Google Gemini API (`google-genai`), cosine-vector RAG, intent-routed tool-assisted coach.
* **Frontend:** Streamlit.
* **Quality:** 25 pytest cases, GitHub Actions CI, reproducible training & validation scripts.
