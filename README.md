# Simply-Fit

**Live app:** https://simply-fit-apk.streamlit.app/

A personal weight-management system built around one core idea — the user should not have to log their food. Daily weight is already the net result of everything the body consumed and burned. Simply-Fit extracts that signal and builds everything else from it.

## Background

Food logging is the standard approach in most diet apps, but people abandon it within days and self-reported calorie intake is systematically unreliable. Simply-Fit treats the scale as a passive sensor and infers calorie balance from the weight trend. The mathematical basis is the energy balance equation — a kilogram of fat tissue represents roughly 7700 kcal.

## What it does

1. **Noise filtering** — EWMA (span 7) separates the true fat-change trend from daily water/glycogen fluctuation.
2. **Anomaly detection** — robust residual z-score (MAD-based) for short logs, Isolation Forest with capped contamination for logs ≥ 30 days. No reading count is ever forced to be "anomalous".
3. **Backward calorie inference** — OLS on the cleaned series (note: regression is run on the raw cleaned readings, *not* the EWMA-smoothed series — smoothing attenuates the slope ~6%; the trend chart still shows the EWMA). Result is reported with a **95% confidence interval** plus separate 28-day and full-trend estimates.
4. **7-day forecast** — 2-layer LSTM (14-day input → 7-day output), trained with causal per-window scaling and **user-grouped** splits, compared against the deterministic linear baseline. Current result on 500 held-out users: **LSTM MAE 0.375 kg vs linear 0.664 kg (43.6% better)** — metrics in `data/synthetic/lstm_evaluation.json`, reproducible with `python scripts/retrain_and_evaluate.py`. A linear fallback is used whenever TensorFlow is unavailable.
5. **AI coach** — intent-routed and tool-assisted: your question is classified, then the ML metrics + forecast + retrieved clinical guidelines are computed first and only *then* passed to Gemini (or a deterministic local synthesis if no key is configured). The coach is grounded in your data, never generic.

## Data & validation (honest numbers)

- The synthetic generator (500 users × 90 days) is validated against a **real NHANES sample shipped in the repo** — `data/public/nhanes_reference.csv`, 7,420 adults aged 18–80 from CDC NHANES cycles 2009–10 and 2011–12 (public-domain data).
- Two-sample KS test, current run (500 synthetic users, `src/calibration.py`):

| Variable | Before | After | Improvement | p-value |
|---|---|---|---|---|
| Weight | 0.185 | 0.065 | 64.9% | 0.037 |
| Age | 0.283 | 0.044 | 84.5% | 0.314 |

- Means: synthetic **79.6 kg / 45.7 yrs** vs NHANES **82.0 kg / 46.2 yrs**.
- **Honest framing:** calibration reduced the bias substantially (age is no longer significantly different), but weight is *still* significantly different (p < 0.05). The app says exactly that — it does not claim "zero bias". (Earlier versions of the docs did; they have been corrected.)

## AI coach — real API key setup

The coach uses the Google Gemini API when a key is available, and falls back to a deterministic answer built from the same metrics otherwise.

> **Key formats:** Google currently issues two kinds of Gemini API key — legacy Standard keys (`AIza…`) and the new Auth keys (`AQ.Ab…`) that AI Studio now creates by default. Both work with this project (the official `google-genai` SDK handles both via the native `x-goog-api-key` header).

Add the key in **any one** of these ways:

1. **Streamlit Cloud (recommended for the deployed app):** Settings → Secrets → add `GEMINI_API_KEY = "..."`.
2. **Local:** copy `.env.example` to `.env` and paste the key from https://aistudio.google.com.
3. **In-app:** paste it in the sidebar ("API Settings") at runtime.

The coach box labels its source after every answer: **"✅ Answered with Gemini (live API)"** when the key works, or a warning with the exact API error when it doesn't. `scripts/verify_coach.py` does the same check from the terminal.

## Repo layout

```
app/streamlit_app.py      # Streamlit UI (7-step wizard, charts, coach)
src/
  data_generator.py       # synthetic population (NHANES-calibrated)
  calibration.py          # real-NHANES KS validation (single source of truth)
  ml_engine.py            # EWMA + anomaly detection + calorie inference (± CI)
  lstm_forecaster.py      # causal per-window LSTM + linear baseline + eval
  rag_pipeline.py         # TF-IDF + cosine retrieval over clinical guidelines
  agent_coach.py          # intent-routed, tool-assisted coach (Gemini or local)
  genai_coach.py          # legacy shim (delegates to agent_coach)
scripts/retrain_and_evaluate.py  # reproducible LSTM train/eval
data/public/nhanes_reference.csv # real NHANES reference (7,420 adults)
data/synthetic/lstm_evaluation.json
tests/                    # 25 unit tests (pytest)
(CI workflow ready locally — push blocked by GitHub App `workflows` permission)
```

## How to run

```bash
git clone https://github.com/tyagiji369/Simply-Fit.git
cd Simply-Fit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add GEMINI_API_KEY
streamlit run app/streamlit_app.py
```

```bash
pytest -q                                   # tests
python scripts/retrain_and_evaluate.py      # re-train + evaluate the LSTM
python -m src.calibration                   # re-run the NHANES validation
```

## Known limitations (stated honestly)

- The LSTM is trained on synthetic data; it generalises to real users only partially. We quantify its error (≈0.37 kg over 7 days on test users) instead of claiming perfection.
- The calorie inference has real uncertainty — typically **±100–200 kcal/day**. The app shows the interval; a single week's estimate should be treated as guidance, not precision.
- Clinical guidelines are a small curated corpus (7 entries); the coach flags conflicts and always defers to a clinician.

Not a substitute for professional medical advice.
