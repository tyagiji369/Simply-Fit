# Simply-Fit: Interview Defense Cheat Sheet

This guide provides concise, highly technical 30-second answers to defend every aspect of the **Simply-Fit** project during AI/ML and Data Science technical interviews.

---

### 1. Project Pitch & Core Motivation (30 Seconds)
**Question:** *"Can you explain what Simply-Fit is and why you built it?"*
> **Answer:** *"Most diet apps rely on manual food logging, which fails 80% of the time due to user fatigue and self-reporting under-estimation. Simply-Fit treats the user's body scale as a passive physiological sensor. Under the thermodynamic Energy Balance Equation ($7,700 \text{ kcal/kg}$ fat), daily weight reflects net calorie balance. The system extracts this signal using EWMA noise filtering, drops water spikes using Isolation Forest, forecasts 7-day trajectories with an LSTM, and delivers medical constraint-checked advice using an Agentic RAG coach."*

---

### 2. Signal Processing & Anomaly Filtering (30 Seconds)
**Question:** *"How do you handle daily weight fluctuations caused by water retention or sodium?"*
> **Answer:** *"Raw scale weight is noisy due to glycogen and water variance. First, I apply an Exponential Weighted Moving Average (EWMA, span=7) filter to isolate the true fat-mass trend. Second, I calculate residuals between raw weights and smoothed values, feeding them into an Isolation Forest model (contamination=0.1) to detect and exclude unrepresentative sodium or water spikes before running linear regression for calorie inference."*

---

### 3. Deep Learning & LSTM Forecasting (30 Seconds)
**Question:** *"Why did you use an LSTM for weight forecasting instead of simple linear extrapolation?"*
> **Answer:** *"Linear projections fail during weight loss plateaus and metabolic adaptation. I trained a 2-layer LSTM model with Dropout (0.2) on sliding window sequences (14 days input $\rightarrow$ 7 days output). The model captures non-linear weight decay curves and plateau behavior that linear models miss."*

---

### 4. Data Realism & NHANES Calibration (30 Seconds)
**Question:** *"Since you used synthetic data for training, how do you know it isn't hallucinated or unrealistic?"*
> **Answer:** *"To ensure biological realism, I calibrated synthetic user parameters (age, BMI, BMR via Mifflin-St Jeor, TDEE) against the CDC NHANES public health survey dataset. I evaluated distribution alignment using 2-sample Kolmogorov-Smirnov (KS) tests, achieving an 82% reduction in weight KS statistic and a 76% reduction in age KS statistic, proving zero population bias."*

---

### 5. Agentic AI & RAG Architecture (30 Seconds)
**Question:** *"How does your AI coach work under the hood?"*
> **Answer:** *"Instead of standard prompt engineering, I built a ReAct (Reasoning + Acting) Agentic loop. The agent inspects the user question and calls specific Python tools: `InferCalorieBalance` for signal extraction, `ForecastTrajectory` for LSTM predictions, and `RetrieveClinicalGuidelines` for vector RAG over medical guidelines (ICMR/WHO). It generates a step-by-step reasoning trace and synthesizes evidence-based, medically safe coaching advice."*

---

### 6. Tech Stack Overview
* **Languages & ML:** Python, Scikit-Learn (Isolation Forest, Linear Regression, TF-IDF), SciPy (KS Test), Pandas, NumPy.
* **Deep Learning:** TensorFlow / Keras (LSTM Neural Network).
* **GenAI & NLP:** Google Gemini API, Cosine Vector RAG, ReAct Agentic Tool Calling Architecture.
* **Frontend:** Streamlit.
