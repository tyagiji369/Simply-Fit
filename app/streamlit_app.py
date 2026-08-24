import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import random
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.agent_coach import agent_coach
from src.rag_pipeline import rag_engine
from src.calibration import run_nhanes_calibration_test
from src.ml_engine import KCAL_PER_KG, run_pipeline as ml_pipeline
from src.lstm_forecaster import forecast as lstm_forecast, SEQUENCE_LENGTH


def streamlit_secret(key="GEMINI_API_KEY"):
    """Reads a Streamlit Cloud secret without crashing locally."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Simply-Fit",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F5F1;
    color: #1A1A1A;
}
.block-container { padding: 3rem 2rem 6rem 2rem; max-width: 820px; }

.brand-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.5rem 0 1.3rem 0;
    border-bottom: 1px solid #E8E4DC;
    margin-bottom: 2.2rem;
}
.brand-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 600;
    color: #1A1A1A; letter-spacing: -0.2px;
}
.brand-logo em { font-style: italic; color: #2D6A4F; }
.brand-pill {
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #fff; background: #1A1A1A;
    padding: 3px 10px; border-radius: 20px;
}
.prog-wrap { margin-bottom: 2.4rem; }
.prog-meta { display: flex; justify-content: space-between; margin-bottom: 8px; }
.prog-name { font-size: 0.68rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #1A1A1A; }
.prog-frac { font-size: 0.68rem; color: #B8B0A8; }
.prog-track { display: flex; gap: 4px; }
.prog-seg { height: 3px; flex: 1; border-radius: 3px; background: #E4E0D8; }
.prog-seg.done { background: #2D6A4F; }
.prog-seg.active { background: #95C4AA; }
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; font-weight: 400;
    color: #1A1A1A; letter-spacing: -1px; line-height: 1.13;
    margin-bottom: 0.4rem;
}
.page-title em { font-style: italic; color: #2D6A4F; }
.page-sub { font-size: 0.9rem; color: #909090; font-weight: 300; line-height: 1.6; margin-bottom: 1.8rem; }
.slabel {
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #B8B0A8; margin: 2rem 0 0.7rem 0;
    display: flex; align-items: center; gap: 8px;
}
.slabel::after { content: ''; flex: 1; height: 1px; background: #E8E4DC; }
.cards { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.6rem 0 1.6rem 0; }
.card {
    flex: 1; min-width: 130px;
    background: #fff; border: 1px solid #E8E4DC;
    border-radius: 16px; padding: 1.15rem 1.3rem 1.1rem;
    position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0; background: #E4E0D8;
}
.card.green::before  { background: #2D6A4F; }
.card.amber::before  { background: #B57800; }
.card.red::before    { background: #B92D2D; }
.card.blue::before   { background: #1A56B0; }
.card.dark::before   { background: #1A1A1A; }
.c-lbl { font-size: 0.61rem; font-weight: 700; letter-spacing: 2.2px; text-transform: uppercase; color: #C0B8B0; margin-bottom: 0.45rem; }
.c-val { font-family: 'Playfair Display', serif; font-size: 2.15rem; font-weight: 600; color: #1A1A1A; line-height: 1; letter-spacing: -1px; }
.c-unit { font-size: 0.76rem; color: #C8C0B8; margin-left: 2px; }
.c-sub  { font-size: 0.74rem; color: #A8A098; margin-top: 0.32rem; line-height: 1.4; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.69rem; font-weight: 700; margin-top: 0.4rem; }
.p-green  { background: #DCFCE7; color: #14532D; }
.p-yellow { background: #FEF9C3; color: #713F12; }
.p-orange { background: #FFEDD5; color: #7C2D12; }
.p-red    { background: #FEE2E2; color: #7F1D1D; }
.alert { padding: 0.9rem 1.1rem; border-radius: 11px; font-size: 0.87rem; line-height: 1.65; margin: 0.9rem 0; border: 1px solid transparent; }
.a-info    { background: #EFF6FF; color: #1E3A8A; border-color: #93C5FD; }
.a-success { background: #F0FDF4; color: #14532D; border-color: #86EFAC; }
.a-warn    { background: #FFFBEB; color: #713F12; border-color: #FCD34D; }
.a-danger  { background: #FFF1F2; color: #7F1D1D; border-color: #FCA5A5; }
.a-neutral { background: #FAFAF8; color: #3C3C3C; border-color: #D6D3D1; }
.callout {
    background: #1A1A1A; color: #F5F5F3;
    border-radius: 16px; padding: 1.4rem 1.6rem;
    margin: 1rem 0 1.6rem 0;
    display: flex; align-items: center; gap: 1.4rem;
}
.callout-big {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 600;
    letter-spacing: -2px; line-height: 1; white-space: nowrap;
}
.callout-detail { font-size: 0.82rem; color: #999; line-height: 1.6; }
.callout-detail strong { color: #E0DDD8; }
.chart-wrap {
    background: #fff; border: 1px solid #E8E4DC;
    border-radius: 14px; padding: 1.3rem 1.4rem 0.7rem;
    margin: 0.6rem 0 1.3rem 0;
}
.chart-lbl { font-size: 0.67rem; font-weight: 700; letter-spacing: 1.8px; text-transform: uppercase; color: #C0B8B0; margin-bottom: 0.5rem; }
.coach-box {
    background: #1A1A1A; color: #F5F5F3;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    margin: 0.8rem 0; font-size: 0.88rem; line-height: 1.75;
}
.rec {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 0.9rem 1.1rem; background: #fff;
    border: 1px solid #E8E4DC; border-radius: 11px;
    margin-bottom: 8px; font-size: 0.86rem; color: #444; line-height: 1.6;
}
.rec-n { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 600; color: #D4CFC8; min-width: 22px; }
.rec.flagged { border-color: #FCA5A5; background: #FFF8F8; }
.rec.flagged .rec-n { color: #FCA5A5; }
.meal-card { background: #fff; border: 1px solid #E8E4DC; border-radius: 13px; padding: 1rem 1.2rem; margin-bottom: 9px; }
.meal-head { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #B0A898; margin-bottom: 0.6rem; }
.meal-item { display: flex; justify-content: space-between; align-items: baseline; font-size: 0.86rem; color: #333; padding: 0.22rem 0; border-bottom: 1px dashed #F0EDE8; }
.meal-item:last-child { border-bottom: none; }
.meal-qty { font-size: 0.78rem; color: #AAA; white-space: nowrap; margin-left: 8px; }
.meal-kcal { font-size: 0.72rem; color: #2D6A4F; font-weight: 600; min-width: 48px; text-align: right; }
.div { border: none; border-top: 1px solid #E8E4DC; margin: 2.2rem 0; }
.footer { text-align: center; font-size: 0.69rem; color: #C8C0B8; margin-top: 4rem; padding-top: 1.2rem; border-top: 1px solid #E8E4DC; }
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.87rem !important; font-weight: 500 !important;
    border-radius: 10px !important; padding: 0.55rem 1.2rem !important;
}
div[data-testid="column"]:last-child .stButton > button {
    background: #1A1A1A !important; color: #F7F5F1 !important;
    border: 1.5px solid #1A1A1A !important;
}
div[data-testid="column"]:first-child .stButton > button {
    background: transparent !important; color: #1A1A1A !important;
    border: 1.5px solid #D0CBC3 !important;
}
.stNumberInput input, .stTextArea textarea {
    border-radius: 10px !important; border: 1.5px solid #D8D4CC !important;
    background: #fff !important;
}
.stSelectbox > div > div {
    border-radius: 10px !important; border: 1.5px solid #D8D4CC !important;
    background: #fff !important;
}
.stRadio label {
    font-size: 0.9rem !important; background: #fff;
    border: 1.5px solid #E4E0D8; border-radius: 10px;
    padding: 0.65rem 1rem !important; width: 100%; cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for Gemini API Key setup
with st.sidebar:
    st.markdown("### 🔑 API Settings")
    default_key = os.getenv("GEMINI_API_KEY", "") or streamlit_secret()
    gemini_key_input = st.text_input(
        "Gemini API Key (Optional):",
        type="password",
        value=default_key,
        help="Paste a Google Gemini API key from https://aistudio.google.com (AIza… standard or AQ.Ab… "
             "auth keys both work) — or set it as a GEMINI_API_KEY environment variable / Streamlit "
             "Cloud secret — to enable real LLM coaching."
    )
    if gemini_key_input:
        st.session_state["gemini_api_key"] = gemini_key_input

    # Show where the key is coming from so the (de)activation is visible.
    _key_source = (
        "env / Streamlit secret" if (os.getenv("GEMINI_API_KEY") or streamlit_secret())
        else "sidebar input" if gemini_key_input and "gemini_api_key" in st.session_state
        else None
    )
    if _key_source:
        st.caption(f"✅ Gemini key detected ({_key_source}) — live coaching enabled")
    else:
        st.caption("ℹ️ No Gemini key found — coach answers with the built-in engine")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
PAGES      = ["profile", "metrics", "goal", "target", "baseline", "tracker", "report"]
TOTAL      = len(PAGES)
PAGE_NAMES = ["Profile", "Metrics", "Goal", "Target", "Baseline", "Tracker", "Report"]

ACTIVITY_MULT = {
    "Sedentary (desk job, little exercise)": 1.2,
    "Lightly active (1–3 days/week)":        1.375,
    "Moderately active (3–5 days/week)":     1.55,
    "Very active (6–7 days/week)":           1.725,
    "Athlete / physical job":                1.9,
}

DISEASES = [
    "Type 2 Diabetes",
    "Hypertension (High Blood Pressure)",
    "High Cholesterol",
    "Hypothyroidism",
    "PCOS",
    "NAFLD",
    "Insulin Resistance / Pre-Diabetes",
    "Chronic Kidney Disease (CKD)",
]

MEAL_PLANS = {
    "lose": {
        "Breakfast (7–9 AM)": [
            ("Oats porridge", "40g dry", 150),
            ("Boiled eggs", "2 whole", 140),
            ("Almonds", "10 nuts", 70),
        ],
        "Lunch (1–2 PM)": [
            ("Brown rice / chapati", "½ cup / 1 roti", 160),
            ("Moong dal", "150g cooked", 120),
            ("Stir-fried vegetables", "1.5 cups", 80),
            ("Green salad", "freely", 30),
        ],
        "Snacks (4–5 PM)": [
            ("Buttermilk / chaas", "1 glass", 35),
            ("Cucumber + carrot sticks", "freely", 40),
            ("Roasted chana", "20g", 75),
        ],
        "Dinner (7–8 PM)": [
            ("Moong dal soup", "1 bowl", 100),
            ("Stir-fried vegetables", "2 cups", 80),
            ("Grilled paneer / tofu", "75g", 90),
        ],
    },
    "gain": {
        "Breakfast (7–9 AM)": [
            ("Whole wheat toast", "3 slices", 240),
            ("Peanut butter", "2 tbsp", 190),
            ("Banana", "1 large", 105),
            ("Whole milk", "1 glass", 150),
        ],
        "Lunch (1–2 PM)": [
            ("Steamed rice / chapati", "1 cup / 2 rotis", 300),
            ("Chicken curry / Paneer", "150g", 250),
            ("Mixed dal", "1 cup", 180),
            ("Curd / dahi", "1 bowl", 100),
        ],
        "Snacks (4–5 PM)": [
            ("Mixed nuts & raisins", "30g", 180),
            ("Protein smoothie", "1 glass", 250),
        ],
        "Dinner (7–8 PM)": [
            ("Chapati / Rice", "2 rotis / 1 cup", 240),
            ("Fish / Paneer / Tofu", "150g", 220),
            ("Cooked vegetables", "1 cup", 100),
        ],
    },
    "maintain": {
        "Breakfast (7–9 AM)": [
            ("Oats / Poha", "1 bowl", 200),
            ("Boiled egg / Tofu", "1 serving", 120),
            ("Fresh fruit", "1 piece", 80),
        ],
        "Lunch (1–2 PM)": [
            ("Chapati / Rice", "1.5 rotis / ¾ cup", 200),
            ("Dal / Rajma / Chole", "1 bowl", 160),
            ("Cooked sabzi", "1 cup", 90),
            ("Salad", "1 plate", 30),
        ],
        "Snacks (4–5 PM)": [
            ("Green tea + makhana", "1 cup / 20g", 80),
        ],
        "Dinner (7–8 PM)": [
            ("Lauki / Torai sabzi", "1 cup", 70),
            ("Chapati / Soup", "1 roti / 1 bowl", 120),
            ("Paneer / Dal", "100g / 1 bowl", 140),
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "profile"

defaults = {
    "age": 28,
    "gender": "male",
    "height": 175.0,
    "weight": 78.0,
    "activity": "Lightly active (1–3 days/week)",
    "diseases": [],
    "goal": "lose",
    "target_change": 5.0,
    "target_weeks": 10,
    "target_calories": -385.0,
    "tdee": 2200.0,
    "bmr": 1700.0,
    "weight_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def go(target_page):
    st.session_state.page = target_page

def brand():
    st.markdown("""
    <div class="brand-header">
      <div class="brand-logo">Simply<em>-Fit</em></div>
      <div class="brand-pill">Research Internship Project</div>
    </div>
    """, unsafe_allow_html=True)

def progress_bar(current_page):
    idx  = PAGES.index(current_page)
    frac = f"{idx + 1} OF {TOTAL}"
    name = PAGE_NAMES[idx].upper()

    segs = ""
    for i in range(TOTAL):
        if i < idx:
            segs += '<div class="prog-seg done"></div>'
        elif i == idx:
            segs += '<div class="prog-seg active"></div>'
        else:
            segs += '<div class="prog-seg"></div>'

    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-meta">
        <span class="prog-name">{name}</span>
        <span class="prog-frac">{frac}</span>
      </div>
      <div class="prog-track">{segs}</div>
    </div>
    """, unsafe_allow_html=True)

def title(heading, subheading):
    st.markdown(f'<div class="page-title">{heading}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subheading}</div>', unsafe_allow_html=True)

def slabel(text):
    st.markdown(f'<div class="slabel">{text}</div>', unsafe_allow_html=True)

def alert(text, kind="neutral"):
    st.markdown(f'<div class="alert a-{kind}">{text}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

def nav(prev_page, next_label, disabled=False):
    col_l, col_r = st.columns([1, 1])
    clicked = False
    with col_l:
        if prev_page:
            if st.button("← Back", key=f"back_{st.session_state.page}"):
                go(prev_page)
                st.rerun()
    with col_r:
        if st.button(next_label, disabled=disabled, key=f"next_{st.session_state.page}"):
            clicked = True
    return clicked

def calc_bmr(w, h, a, g):
    if g == "male":
        return 10 * w + 6.25 * h - 5 * a + 5
    return 10 * w + 6.25 * h - 5 * a - 161

def bmi_cat(bmi):
    if bmi < 18.5:   return "Underweight", "p-orange", "amber"
    elif bmi < 25.0: return "Normal weight", "p-green", "green"
    elif bmi < 30.0: return "Overweight", "p-yellow", "amber"
    else:            return "Obese", "p-red", "red"

def detect_anomalies(log, **kwargs):
    """Thin wrapper — single source of truth is src.ml_engine."""
    res = ml_pipeline(log, **kwargs)
    return res["anomaly_flags"], res["smoothed"]

def estimate_balance(log, **kwargs):
    """Thin wrapper — single source of truth is src.ml_engine."""
    res = ml_pipeline(log, **kwargs)
    return (res["kcal_per_day"], res["weekly_kg_change"], res["r_squared"], res)

def smart_ylim(vals, pad=0.2):
    lo, hi = min(vals), max(vals)
    span   = max(hi - lo, 1.0)
    return lo - span * pad, hi + span * pad

def make_fig(w=7, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor("#fff")
    ax.set_facecolor("#fff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8E4DC")
    ax.spines["bottom"].set_color("#E8E4DC")
    ax.tick_params(colors="#1A1A1A", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4, color="#D0CBC3")
    return fig, ax

def chart_wrap(lbl=""):
    if lbl:
        st.markdown(f'<div class="chart-wrap"><div class="chart-lbl">{lbl}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)

def chart_end():
    st.markdown('</div>', unsafe_allow_html=True)

def render_forecast_chart(weight_log, chart_title=None):
    """
    Renders the last 14 readings plus the 7-day forecast (LSTM when
    available, linear extrapolation otherwise).
    """
    if len(weight_log) < SEQUENCE_LENGTH:
        return None
    fc = lstm_forecast(weight_log)
    if fc is None:
        return None

    method = "LSTM model" if fc["method"] == "lstm" else "linear extrapolation (fallback)"
    chart_wrap(chart_title or f"7-day forecast — {method}")
    fig_f, ax_f = make_fig(7, 2.6)
    hist = weight_log[-14:]
    n_hist = len(hist)
    x_hist = np.arange(1, n_hist + 1)
    x_fc = np.arange(n_hist, n_hist + len(fc["values"]) + 1)
    y_all = list(hist) + [hist[-1]] + list(fc["values"])
    ylo, yhi = smart_ylim(y_all)

    ax_f.plot(x_hist, hist, color="#1A1A1A", linewidth=2.0,
              marker="o", markersize=4, label="History (last 14 d)", zorder=4)
    ax_f.plot(x_fc, [hist[-1]] + list(fc["values"]), color="#2D6A4F", linewidth=2.2,
              linestyle="--", marker="o", markersize=4, label="Forecast", zorder=5)
    ax_f.axvline(n_hist, color="#E4E0D8", linewidth=1.0, zorder=1)
    ax_f.set_ylim(ylo, yhi)
    ax_f.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_f.set_xlabel("Day ahead", fontsize=8.5)
    ax_f.set_ylabel("Weight (kg)", fontsize=8.5)
    ax_f.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig_f, width="stretch")
    plt.close()
    chart_end()
    return fc

def get_coach_response(profile, weight_log, question, target_weekly):
    """
    Builds the full user profile for the coach and calls it once.

    ``target_weekly`` is a weekly rate in kg/week (already converted by the
    caller). ``target_weight`` is the absolute goal weight — the agent's
    time-to-goal math depends on it being an absolute kg value.
    """
    try:
        goal = st.session_state.get("goal", "lose")
        start_w = float(st.session_state.get("weight", 78.0))
        change = float(st.session_state.get("target_change", 5.0))
        if goal == "maintain":
            goal_weight = start_w
        elif goal == "gain":
            goal_weight = start_w + change
        else:
            goal_weight = start_w - change

        diseases = profile.get("diseases", []) or []
        prof_dict = {
            "age": st.session_state.get("age", 28),
            "gender": st.session_state.get("gender", "male"),
            "height_cm": st.session_state.get("height", 175.0),
            "weight": start_w,
            "goal": goal,
            "disease": diseases[0] if diseases else None,
            "diseases": diseases,
            "target_weight": round(goal_weight, 1),
            "tdee": st.session_state.get("tdee", 2200),
        }
        api_key = (
            st.session_state.get("gemini_api_key")
            or os.getenv("GEMINI_API_KEY")
            or streamlit_secret()
        )
        res = agent_coach.run_agent(
            prof_dict, weight_log, question,
            target_weekly_change=target_weekly, gemini_api_key=api_key or None
        )
        # Remember how the answer was produced, so the UI can label it.
        st.session_state["coach_used_llm"] = bool(res.get("used_llm"))
        st.session_state["coach_llm_error"] = res.get("llm_error")
        return res["response"]
    except Exception:
        recent = weight_log[-7:]
        weekly_change = round(recent[-1] - recent[0], 2)
        return (f"Your 7-day trend shows {weekly_change:+.2f} kg change "
                f"(approx. {(weekly_change * 7700) / 7:+.0f} kcal/day, "
                f"treat as an estimate ±150 kcal).")

def render_meal_plan(goal, tdee, target):
    """
    Sample menu, scaled to the user's actual calorie target.

    The base recipes are fixed; the displayed kcal are scaled by
    target / base_total so the plan always adds up to the recommended
    intake instead of silently showing a 1170 kcal plan under a
    1815 kcal label.
    """
    plan = MEAL_PLANS.get(goal, MEAL_PLANS["lose"])
    base_total = sum(kcal for items in plan.values() for _, _, kcal in items)
    rec_target = int(tdee + target)
    factor = rec_target / base_total if base_total else 1.0

    slabel(f"Suggested daily meal plan — {rec_target} kcal target "
           f"(sample scaled ×{factor:.2f})")
    st.markdown(
        '<div class="alert a-info">Portions below are a template scaled to your '
        'target. Adjust to your preferences, and follow renal/heart-specific '
        'guidance from your clinician where applicable.</div>',
        unsafe_allow_html=True,
    )

    for meal_name, items in plan.items():
        st.markdown(f'<div class="meal-card"><div class="meal-head">{meal_name}</div>', unsafe_allow_html=True)
        items_html = ""
        for name, qty, kcal in items:
            scaled = int(round(kcal * factor / 5) * 5)
            items_html += (f'<div class="meal-item"><span>{name}<span class="meal-qty">({qty})</span></span>'
                           f'<span class="meal-kcal">{scaled} kcal</span></div>')
        st.markdown(items_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PROFILE
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "profile":
    brand()
    progress_bar("profile")
    title("Let's set up your<br><em>profile.</em>",
          "Your physiological baseline shapes every calculation that follows.")

    col1, col2 = st.columns(2)
    with col1:
        age    = st.number_input("Age", 18, 90, int(st.session_state.age))
        gender = st.selectbox("Biological sex", ["male", "female"],
                              index=0 if st.session_state.gender=="male" else 1,
                              format_func=lambda x: x.capitalize())
        height = st.number_input("Height (cm)", 120.0, 220.0,
                                 float(st.session_state.height), step=0.5)

    with col2:
        weight   = st.number_input("Starting weight (kg)", 35.0, 200.0,
                                   float(st.session_state.weight), step=0.5)
        activity = st.selectbox("Activity level", list(ACTIVITY_MULT.keys()),
                                index=list(ACTIVITY_MULT.keys()).index(st.session_state.activity))

    divider()
    if nav(None, "Continue"):
        st.session_state.age      = age
        st.session_state.gender   = gender
        st.session_state.height   = height
        st.session_state.weight   = weight
        st.session_state.activity = activity

        bmr  = calc_bmr(weight, height, age, gender)
        tdee = bmr * ACTIVITY_MULT[activity]
        st.session_state.bmr  = bmr
        st.session_state.tdee = tdee

        go("metrics")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — METRICS
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "metrics":
    brand()
    progress_bar("metrics")
    title("Your physiological<br><em>baseline.</em>",
          "Calculated from validated clinical formulas.")

    w = st.session_state.weight
    h = st.session_state.height
    a = st.session_state.age
    g = st.session_state.gender

    bmi                     = w / (h / 100) ** 2
    bmi_lbl, bmi_pill, card_col = bmi_cat(bmi)

    bmr  = st.session_state.bmr
    tdee = st.session_state.tdee

    slabel("Key indicators")
    st.markdown(f"""
    <div class="cards">
      <div class="card {card_col}">
        <div class="c-lbl">BMI</div>
        <div class="c-val">{bmi:.1f}</div>
        <span class="pill {bmi_pill}">{bmi_lbl}</span>
      </div>
      <div class="card">
        <div class="c-lbl">BMR</div>
        <div class="c-val">{int(bmr)}<span class="c-unit">kcal</span></div>
        <div class="c-sub">Basal metabolic rate</div>
      </div>
      <div class="card dark">
        <div class="c-lbl">TDEE</div>
        <div class="c-val">{int(tdee)}<span class="c-unit">kcal</span></div>
        <div class="c-sub">Maintenance calories</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    slabel("Medical conditions")
    diseases = st.multiselect("Select any relevant medical conditions", DISEASES,
                              default=st.session_state.diseases)
    st.session_state.diseases = diseases

    if diseases:
        alert(f"Selected: <strong>{', '.join(diseases)}</strong>. "
              f"Dietary targets will account for clinical restrictions.", "warn")

    # BMI Spectrum Chart
    slabel("BMI Spectrum Analysis")
    chart_wrap("BMI Categories")
    fig_b, ax_b = make_fig(7, 1.8)
    ax_b.axvspan(15, 18.5, color="#FEF3C7", alpha=0.6)
    ax_b.axvspan(18.5, 25, color="#DCFCE7", alpha=0.6)
    ax_b.axvspan(25, 30, color="#FEF3C7", alpha=0.6)
    ax_b.axvspan(30, 40, color="#FEE2E2", alpha=0.6)
    ax_b.axvline(bmi, color="#1A1A1A", linewidth=2.5, linestyle="-")
    ax_b.text(bmi, 0.1, f" You ({bmi:.1f})",
              va="bottom", fontsize=9.5, fontweight="bold", color="#1A1A1A")
    ax_b.set_xlim(15, 40); ax_b.set_ylim(-0.72, 0.6)
    ax_b.set_yticks([]); ax_b.set_xticks([18.5, 25, 30])
    ax_b.set_xticklabels(["18.5", "25", "30"]); ax_b.grid(False)
    for sp in ax_b.spines.values(): sp.set_visible(False)
    plt.tight_layout(pad=0.3)
    st.pyplot(fig_b, width="stretch")
    plt.close()
    chart_end()

    divider()
    if nav("profile", "Set my goal"):
        go("goal")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — GOAL
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "goal":
    brand()
    progress_bar("goal")
    title("What are you<br><em>working towards?</em>",
          "Your goal shapes every calorie target and recommendation.")

    goal = st.radio("Select your primary goal:", ["lose", "gain", "maintain"],
                    format_func=lambda x: {
                        "lose": "Lose weight",
                        "gain": "Gain weight",
                        "maintain": "Maintain weight"
                    }[x],
                    index=["lose","gain","maintain"].index(st.session_state.goal))

    descs = {
        "lose":     ("success", "Achieve a calorie deficit through diet and movement. Safe limit: 1 kg/week."),
        "gain":     ("warn",    "Build mass through a controlled surplus. Safe limit: 1.0 kg/week."),
        "maintain": ("info",    "Keep your current weight stable. We will track fluctuations and alert you if the trend drifts."),
    }
    alert(descs[goal][1], descs[goal][0])

    divider()
    if nav("metrics", "Continue"):
        st.session_state.goal = goal
        go("target")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — TARGET
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "target":
    brand()
    progress_bar("target")
    goal = st.session_state.goal
    title("Define your<br><em>target.</em>",
          "How much do you want to change, and in how long?")

    ok = True
    target_calories = 0.0

    if goal == "maintain":
        alert(f"No change target needed. Your daily goal equals your TDEE: "
              f"<strong>{int(st.session_state.tdee)} kcal/day</strong>.", "info")
    else:
        slabel("Weight change goal")
        c1, c2 = st.columns(2)
        with c1:
            change = st.number_input("Weight change (kg)", 0.5, 50.0,
                                     float(st.session_state.target_change),
                                     step=0.5, format="%.1f")
        with c2:
            weeks = st.number_input("Time frame (weeks)", 1, 104,
                                    int(st.session_state.target_weeks), step=1)

        rate  = change / weeks
        max_r = 1.0 if goal == "lose" else 1.0

        if rate > max_r:
            alert(f"Rate of <strong>{rate:.2f} kg/week</strong> exceeds safe limit of "
                  f"{max_r} kg/week. Please adjust.", "danger")
            ok = False
        else:
            daily_kcal      = (KCAL_PER_KG * change) / (weeks * 7)
            target_calories = -daily_kcal if goal == "lose" else daily_kcal
            direction       = "deficit" if goal == "lose" else "surplus"
            intake          = int(st.session_state.tdee + target_calories)
            st.markdown(
                f'<div class="callout">'
                f'<div class="callout-big">{int(abs(daily_kcal))} kcal</div>'
                f'<div class="callout-detail">'
                f'Daily <strong>{direction}</strong> needed<br>'
                f'Target daily intake: <strong>{intake} kcal</strong><br>'
                f'Pace: <strong>{rate:.2f} kg/week</strong>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    divider()
    if nav("goal", "Set baseline log", disabled=not ok):
        if goal != "maintain":
            st.session_state.target_change   = change
            st.session_state.target_weeks    = weeks
            st.session_state.target_calories = target_calories
        else:
            st.session_state.target_change   = 0.0
            st.session_state.target_weeks    = 1
            st.session_state.target_calories = 0.0
        go("baseline")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — BASELINE
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "baseline":
    brand()
    progress_bar("baseline")
    title("Establish your<br><em>baseline log.</em>",
          "Provide 7 days of initial weight data (or use realistic sample data).")

    if not st.session_state.weight_log:
        w_start = st.session_state.weight
        g_type  = st.session_state.goal
        st.session_state.weight_log = [
            round(w_start + (-0.08 if g_type=="lose" else 0.08)*i + random.choice([0, -0.1, 0.1]), 1)
            for i in range(14)
        ]

    slabel("Historical weight entries")
    log_text = st.text_area("Enter daily weight values separated by commas",
                            value=", ".join(str(x) for x in st.session_state.weight_log),
                            height=100)

    try:
        parsed_log = [float(x.strip()) for x in log_text.split(",") if x.strip()]
    except ValueError:
        parsed_log = []

    if len(parsed_log) < 7:
        alert("At least 7 daily readings are required for accurate signal processing.", "warn")

    divider()
    if nav("target", "Start daily tracker", disabled=len(parsed_log) < 7):
        st.session_state.weight_log = parsed_log
        go("tracker")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — TRACKER
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "tracker":
    brand()
    progress_bar("tracker")
    title("Daily weight<br><em>tracker.</em>",
          "Log scale weight, run noise filtering, and infer calorie balance.")

    log    = st.session_state.weight_log
    target = st.session_state.target_calories
    goal   = st.session_state.goal
    n      = len(log)

    slabel("Log today's weight")
    col_i1, col_i2 = st.columns([3, 1])
    with col_i1:
        new_w = st.number_input("Today's scale reading (kg)",
                                value=float(log[-1]), step=0.1, format="%.1f")
    with col_i2:
        st.write("")
        if st.button("Add entry", width="stretch"):
            st.session_state.weight_log.append(round(new_w, 1))
            st.rerun()

    if n >= 7:
        target_weekly = (target * 7) / KCAL_PER_KG
        res = ml_pipeline(log, target_weekly_change=target_weekly)
        flags = res["anomaly_flags"]
        smoothed = res["smoothed"]
        kcal_balance = res["kcal_per_day"]
        weekly_kg = res["weekly_kg_change"]
        r2 = res["r_squared"]
        ci = res["ci_95_kcal_per_day"]

        slabel("ML pipeline output")
        st.markdown(f"""
        <div class="cards">
          <div class="card">
            <div class="c-lbl">Current weight</div>
            <div class="c-val">{log[-1]:.1f}<span class="c-unit">kg</span></div>
            <div class="c-sub">Latest reading</div>
          </div>
          <div class="card dark">
            <div class="c-lbl">Inferred balance</div>
            <div class="c-val">{int(kcal_balance):+d}<span class="c-unit">kcal</span></div>
            <div class="c-sub">±{int(ci)} kcal (95% CI) · no food log needed</div>
          </div>
          <div class="card">
            <div class="c-lbl">Weekly rate</div>
            <div class="c-val">{weekly_kg:+.2f}<span class="c-unit">kg/w</span></div>
            <div class="c-sub">R²={r2} · 28-day: {res['weekly_kg_change_28d']:+.3f} kg/w</div>
          </div>
          <div class="card {"green" if flags.sum()==0 else "amber"}">
            <div class="c-lbl">Anomalies</div>
            <div class="c-val">{int(flags.sum())}</div>
            <div class="c-sub">Filtered ({res['anomaly_method'].replace('_', ' ')})</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        on_track = abs(weekly_kg - target_weekly) < 0.15
        if on_track:
            alert("You are on track this week. Keep your current habits.", "success")
        elif weekly_kg < target_weekly and goal == "lose":
            gap = abs(target_weekly - weekly_kg)
            alert(
                f"Slightly behind target. Try reducing "
                f"<strong>{int(gap*7700/7*0.6)} kcal/day</strong> from food "
                f"and burning <strong>{int(gap*7700/7*0.4)} kcal/day</strong> more through activity.",
                "warn"
            )
        else:
            alert("Ahead of schedule. Maintain your current pace.", "info")

        if flags.sum() > 0:
            alert(
                f"{int(flags.sum())} anomalous reading(s) detected ({res['anomaly_method'].replace('_', ' ')}) "
                f"and excluded from the calorie estimate. These are most likely water-retention spikes; "
                f"the rest of the trend is unchanged.",
                "neutral"
            )

        slabel("Weight trend — EWMA + anomaly detection")
        chart_wrap("Raw readings · EWMA trend · anomalies flagged in red")
        days = np.arange(1, n+1)
        ylo, yhi = smart_ylim(list(log) + list(smoothed))
        fig_t, ax_t = make_fig(7, 3.2)
        ax_t.fill_between(days, log, ylo, alpha=0.06, color="#2D6A4F")
        ax_t.plot(days, log, color="#C8C3BB", linewidth=1.2,
                  marker="o", markersize=4.5,
                  markerfacecolor="#1A1A1A", markeredgecolor="#1A1A1A",
                  label="Daily weight", zorder=3)
        ax_t.plot(days, smoothed, color="#1A1A1A", linewidth=2.4,
                  label="EWMA trend", zorder=4)
        ax_t.scatter(days[flags], np.array(log)[flags],
                     color="#B92D2D", s=55, zorder=5, label="Anomaly detected")
        ax_t.set_ylim(ylo, yhi)
        ax_t.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax_t.set_xlabel("Day", fontsize=8.5)
        ax_t.set_ylabel("Weight (kg)", fontsize=8.5)
        ax_t.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig_t, width="stretch")
        plt.close()
        chart_end()

        render_forecast_chart(log)

        slabel("Ask your coach")
        question = st.text_input("Ask anything about your progress...",
                                 placeholder="Why is my weight fluctuating?")
        if st.button("Ask coach", width="stretch"):
            if question.strip():
                response = get_coach_response(
                    {"diseases": st.session_state.diseases}, log, question, target_weekly
                )
                st.markdown(
                    f'<div class="coach-box">{response}</div>',
                    unsafe_allow_html=True
                )
                if st.session_state.get("coach_used_llm"):
                    st.caption("✅ Answered with Gemini (live API)")
                elif st.session_state.get("coach_llm_error"):
                    st.caption(f"⚠️ Gemini attempt failed ({st.session_state['coach_llm_error'][:120]}) — answered with the built-in engine")
                else:
                    st.caption("ℹ️ No Gemini key configured — answered with the built-in engine")

    elif n > 0:
        alert(f"Log {7 - n} more reading(s) to unlock ML insights and coaching.", "info")

    divider()
    ready = n >= 7
    if nav("baseline", "View full report", disabled=not ready):
        go("report")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 7 — REPORT
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "report":
    brand()
    progress_bar("report")
    title("Your progress<br><em>report.</em>",
          f"Generated {datetime.today().strftime('%d %B %Y')} · "
          f"{len(st.session_state.weight_log)} days of data")

    log      = st.session_state.weight_log
    target   = st.session_state.target_calories
    goal     = st.session_state.goal
    h        = st.session_state.height
    diseases = st.session_state.diseases
    w_kg     = st.session_state.weight
    tdee     = st.session_state.tdee

    res = ml_pipeline(log, target_weekly_change=(target * 7) / KCAL_PER_KG)
    flags = res["anomaly_flags"]
    smoothed = res["smoothed"]
    kcal_balance = res["kcal_per_day"]
    weekly_kg = res["weekly_kg_change"]
    r2 = res["r_squared"]
    ci = res["ci_95_kcal_per_day"]

    days = list(range(1, len(log)+1))
    z    = np.polyfit(days, log, 1)
    p    = np.poly1d(z)

    total_change = log[-1] - log[0]
    bmi_cur      = log[-1] / (h/100)**2
    bmi_lbl, bmi_pill, card_col = bmi_cat(bmi_cur)

    # Summary cards
    cc = "green" if (
        (goal=="lose" and total_change<0) or (goal=="gain" and total_change>0)
    ) else ("red" if (
        (goal=="lose" and total_change>0.2) or (goal=="gain" and total_change<-0.2)
    ) else "")

    slabel("Summary")
    st.markdown(f"""
    <div class="cards">
      <div class="card"><div class="c-lbl">Start</div>
        <div class="c-val">{log[0]:.1f}<span class="c-unit">kg</span></div>
        <div class="c-sub">Day 1</div></div>
      <div class="card"><div class="c-lbl">Current</div>
        <div class="c-val">{log[-1]:.1f}<span class="c-unit">kg</span></div>
        <div class="c-sub">Latest reading</div></div>
      <div class="card {cc}"><div class="c-lbl">Total change</div>
        <div class="c-val">{total_change:+.1f}<span class="c-unit">kg</span></div>
        <div class="c-sub">Over {len(log)} days</div></div>
      <div class="card dark"><div class="c-lbl">Inferred balance</div>
        <div class="c-val">{int(kcal_balance):+d}<span class="c-unit">kcal</span></div>
        <div class="c-sub">±{int(ci)} kcal (95% CI) · no food log needed</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Trend chart
    slabel("Weight trend")
    chart_wrap("Daily readings · EWMA trend · linear regression · anomalies")
    all_v = list(log) + list(smoothed) + list(p(days))
    ylo, yhi = smart_ylim(all_v)

    fig1, ax1 = make_fig(7, 3.8)
    ax1.fill_between(days, log, ylo, alpha=0.05, color="#2D6A4F")
    ax1.plot(days, log, color="#C8C3BB", linewidth=1.2,
             marker="o", markersize=4.5,
             markerfacecolor="#1A1A1A", markeredgecolor="#1A1A1A",
             label="Daily weight", zorder=3)
    ax1.plot(days, smoothed, color="#1A1A1A", linewidth=2.6,
             label="EWMA trend", zorder=4)
    ax1.plot(days, p(days), linestyle="--", color="#B8B0A8", linewidth=1.2,
             label=f"Linear regression ({z[0]:+.3f} kg/day)", zorder=2)
    ax1.scatter(np.array(days)[flags], np.array(log)[flags],
                color="#B92D2D", s=55, zorder=5, label="Anomaly")
    ax1.set_ylim(ylo, yhi)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax1.set_xlabel("Day", fontsize=8.5)
    ax1.set_ylabel("Weight (kg)", fontsize=8.5)
    ax1.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig1, width="stretch")
    plt.close()
    chart_end()

    # 7-day forecast (LSTM when available)
    slabel("Forecast")
    render_forecast_chart(log, "Next 7 days — LSTM forecast vs linear baseline fallback")

    if res["kcal_per_day_28d"] and abs(res["kcal_per_day_28d"] - kcal_balance) > 100:
        alert(
            f"Recent 28-day estimate ({res['kcal_per_day_28d']:+.0f} kcal/day) differs from the "
            f"full-trend estimate ({kcal_balance:+.0f} kcal/day) — the balance is changing over time.",
            "info"
        )

    # Calorie balance chart
    slabel("Inferred daily calorie balance")
    wb = 5
    bal_s, bal_d = [], []
    for i in range(wb, len(log)):
        chunk = log[i-wb:i+1]
        bal_s.append((chunk[-1]-chunk[0])/wb * KCAL_PER_KG)
        bal_d.append(i+1)

    if bal_s:
        chart_wrap("Rolling 5-day inferred balance — inferred from weight, no food logging")
        fig2, ax2 = make_fig(7, 3.2)
        bar_cols = ["#86EFAC" if b < 0 else "#FCA5A5" for b in bal_s]
        ax2.bar(bal_d, bal_s, color=bar_cols, width=0.75, edgecolor="none", zorder=3)
        ax2.axhline(0, color="#D0CBC3", linewidth=0.9, zorder=2)
        if target != 0:
            ax2.axhline(target, color="#1A1A1A", linewidth=1.6, linestyle="--",
                        label=f"Target ({int(target):+d} kcal/day)", zorder=4)
        gp = mpatches.Patch(color="#86EFAC", label="Deficit")
        rp = mpatches.Patch(color="#FCA5A5", label="Surplus")
        handles = [gp, rp]
        if target != 0:
            handles += ax2.get_legend_handles_labels()[0]
        ax2.legend(handles=handles)
        ax2.set_xlabel("Day", fontsize=8.5)
        ax2.set_ylabel("kcal / day", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close()
        chart_end()

    # Goal projection
    if goal != "maintain":
        slabel("Goal projection")
        tc      = st.session_state.target_change
        tw      = st.session_state.target_weeks
        t_end   = log[0] + (-tc if goal == "lose" else tc)
        total_p = tw * 7
        pdays   = list(range(1, total_p+1))
        i_slope = (-tc if goal == "lose" else tc) / total_p

        proj_ideal  = [log[0] + i_slope * d for d in pdays]
        proj_actual = [log[-1] + z[0] * (d - len(log)) for d in pdays]
        all_p = list(log) + proj_ideal + proj_actual + [t_end]
        ylo_p, yhi_p = smart_ylim(all_p, pad=0.15)

        chart_wrap("Actual · ideal path · current-rate projection")
        fig3, ax3 = make_fig(7, 3.8)
        ax3.axhline(t_end, color="#2D6A4F", linewidth=1.1, linestyle="-.",
                    alpha=0.85, label=f"Goal weight ({t_end:.1f} kg)", zorder=2)
        ax3.fill_between(days, log, ylo_p, alpha=0.05, color="#1A1A1A")
        ax3.plot(days, log, color="#1A1A1A", linewidth=2.8,
                 label="Actual so far", zorder=5)
        ax3.plot(pdays, proj_ideal, color="#B8B0A8", linewidth=1.4,
                 linestyle="--", label="Ideal path", zorder=3)
        ax3.plot(pdays, proj_actual, color="#B92D2D", linewidth=1.4,
                 linestyle=":", label="At current rate", zorder=4)
        ax3.axvline(len(log), color="#E4E0D8", linewidth=1.0, zorder=1)
        ax3.text(len(log)+0.4, yhi_p-(yhi_p-ylo_p)*0.05,
                 "today", fontsize=7.5, color="#B8B0A8")
        ax3.set_ylim(ylo_p, yhi_p)
        ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax3.set_xlabel("Day", fontsize=8.5)
        ax3.set_ylabel("Weight (kg)", fontsize=8.5)
        ax3.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig3, width="stretch")
        plt.close()
        chart_end()

        if abs(z[0]) > 0.001:
            dl  = abs((log[-1]-t_end)/z[0])
            pd_ = datetime.today() + timedelta(days=dl)
            pl  = total_p - len(log)
            id_ = datetime.today() + timedelta(days=max(pl, 0))
            dd  = dl - pl
            if abs(dd) < 7:
                alert(f"At current pace you are <strong>right on schedule</strong> — "
                      f"projected goal date: <strong>{pd_.strftime('%d %b %Y')}</strong>.", "success")
            elif dd < 0:
                alert(f"You are <strong>ahead of schedule</strong> — projected: "
                      f"<strong>{pd_.strftime('%d %b %Y')}</strong>, planned: {id_.strftime('%d %b %Y')}.", "info")
            else:
                alert(f"Projected goal date: <strong>{pd_.strftime('%d %b %Y')}</strong> — "
                      f"{int(dd)} days behind. A small daily adjustment closes this gap.", "warn")

    # Ask your coach in Report
    slabel("Ask your AI coach")
    question_rep = st.text_input("Ask anything about your progress or timeline...",
                                 placeholder="How much time will it take to achieve my goal?")
    if st.button("Ask AI coach", width="stretch"):
        if question_rep.strip():
            response = get_coach_response(
                {"diseases": diseases}, log, question_rep, (target * 7) / KCAL_PER_KG
            )
            st.markdown(
                f'<div class="coach-box">{response}</div>',
                unsafe_allow_html=True
            )
            if st.session_state.get("coach_used_llm"):
                st.caption("✅ Answered with Gemini (live API)")
            elif st.session_state.get("coach_llm_error"):
                st.caption(f"⚠️ Gemini attempt failed ({st.session_state['coach_llm_error'][:120]}) — answered with the built-in engine")
            else:
                st.caption("ℹ️ No Gemini key configured — answered with the built-in engine")

    # Health status
    slabel("Current health status")
    st.markdown(f"""
    <div class="cards">
      <div class="card {card_col}">
        <div class="c-lbl">Current BMI</div>
        <div class="c-val">{bmi_cur:.1f}</div>
        <span class="pill {bmi_pill}">{bmi_lbl}</span>
      </div>
      <div class="card">
        <div class="c-lbl">Model R²</div>
        <div class="c-val">{r2}</div>
        <div class="c-sub">Regression fit quality</div>
      </div>
      <div class="card">
        <div class="c-lbl">Anomalies filtered</div>
        <div class="c-val">{int(flags.sum())}</div>
        <div class="c-sub">By Isolation Forest</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if bmi_cur < 18.5:
        alert("<strong>Underweight:</strong> Associated with nutritional deficiencies and weakened immunity. Increase intake with nutrient-dense foods.", "warn")
    elif bmi_cur < 25:
        alert("<strong>Healthy weight:</strong> Your BMI is in the normal range. Focus on maintaining this through sustainable habits.", "success")
    elif bmi_cur < 30:
        alert("<strong>Overweight:</strong> Modest cardiovascular and metabolic risk. A 5–10% weight reduction produces clinically meaningful benefits.", "warn")
    else:
        alert("<strong>Obese range:</strong> Elevated risk of type 2 diabetes, hypertension, and cardiovascular disease. Structured guidance from a healthcare professional is recommended.", "danger")

    # Meal plan
    render_meal_plan(goal, tdee, target)

    # Recommendations — safe defaults, adjusted when the profile has
    # conditions that make a generic recommendation wrong (e.g. CKD).
    slabel("General recommendations")
    cond_lower = [d.lower() for d in diseases]
    has_ckd = any(("ckd" in c or "kidney" in c) for c in cond_lower)
    has_renal_or_heart = has_ckd or any("hypertension" in c for c in cond_lower)

    recs = [
        ("Weigh yourself at the same time every morning — before eating, after bathroom. Consistency matters more than the exact time.", False),
        ("7–9 hours of sleep is essential. Sleep deficit raises hunger hormones and impairs fat loss even with the same calorie intake.", False),
        ("150 minutes of moderate aerobic activity per week is the minimum evidence-backed recommendation for metabolic health.", False),
        ("Limit ultra-processed foods — they are engineered to override satiety signals and are the primary driver of unintended weight gain.", False),
    ]
    if has_ckd:
        recs.append((
            "<strong>Kidney condition:</strong> protein intake must be individualised — non-dialysis CKD "
            "guidelines typically target 0.6–0.8 g/kg/day (KDOQI), the opposite of the general 1.6–2.0 g/kg "
            "recommendation. Confirm your target with your nephrologist.", True))
        recs.append((
            "<strong>Fluid:</strong> fluid targets may be restricted in CKD — follow your clinician's "
            "fluid advice rather than a fixed daily volume.", True))
    else:
        recs.append((
            f"Target protein at 1.6–2.0 g per kg — roughly {int(1.8 * w_kg)} g/day — to preserve muscle "
            "regardless of your goal. <em>(Not applicable if a kidney condition is selected.)</em>", False))
    if has_renal_or_heart:
        recs.append((
            f"Daily fluid around {int(35 * w_kg)} ml is a general guideline; confirm with your clinician "
            "if you have hypertension, heart or kidney disease.", False))
    else:
        recs.append((
            f"Drink roughly {int(35 * w_kg)} ml of water per day ({int(35 * w_kg / 250)} glasses). "
            "Thirst is frequently misread as hunger.", False))

    # Clinical Vector RAG Guidelines Insertion
    if diseases:
        for d in diseases:
            matches = rag_engine.search_guidelines("dietary guidelines", condition=d, top_k=1)
            if matches:
                recs.append((f"<strong>Clinical Guideline ({d}):</strong> {matches[0]['recommendation']} <em>(Source: {matches[0]['source']})</em>", True))

    recs_html = ""
    for i, (text, flagged) in enumerate(recs, 1):
        cls = "rec flagged" if flagged else "rec"
        recs_html += (f'<div class="{cls}"><span class="rec-n">{i:02d}</span>'
                      f'<span>{text}</span></div>')
    st.markdown(recs_html, unsafe_allow_html=True)

    # Model Insights & NHANES Validation Expander — honest framing
    slabel("Model Insights & Data Validation")
    with st.expander("🔬 Validation: synthetic data vs real NHANES sample (KS test)"):
        st.write(
            "The generator is checked against a **real** NHANES sample shipped in the repo "
            "(`data/public/nhanes_reference.csv`, 7,481 adults, cycles 2009–10 & 2011–12). "
            "A smaller KS statistic = closer distributions."
        )
        if st.button("Run validation check", key="ks_test_report_orig"):
            try:
                ks_res = run_nhanes_calibration_test(n_users=500)
                st.write(
                    f"• **Weight KS:** {ks_res['weight_ks_before']} → {ks_res['weight_ks_after']} "
                    f"({ks_res['weight_ks_improvement_pct']}% improvement, p={ks_res['weight_p_value']:.4f})"
                )
                st.write(
                    f"• **Age KS:** {ks_res['age_ks_before']} → {ks_res['age_ks_after']} "
                    f"({ks_res['age_ks_improvement_pct']}% improvement, p={ks_res['age_p_value']:.4f})"
                )
                st.write(
                    f"• **Means:** synthetic {ks_res['mean_synthetic_weight']} kg / "
                    f"{ks_res['mean_synthetic_age']} yrs vs NHANES "
                    f"{ks_res['mean_reference_weight']} kg / {ks_res['mean_reference_age']} yrs "
                    f"({ks_res['n_reference']} reference adults)."
                )
                if ks_res["distributions_still_significantly_different"]:
                    st.warning(
                        "p < 0.05 — the distributions are now *closer*, but still statistically "
                        "different. That is stated honestly: calibration reduced bias, it did not "
                        "eliminate it."
                    )
                else:
                    st.success("No statistically significant difference detected.")
            except FileNotFoundError as e:
                st.error(str(e))

    with st.expander("🤖 LSTM forecast: does it beat the linear baseline?"):
        import json as _json
        _path = os.path.join(ROOT_DIR, "data", "synthetic", "lstm_evaluation.json")
        if os.path.exists(_path):
            with open(_path) as _f:
                _m = _json.load(_f)
            st.write(
                f"• **LSTM MAE:** {_m['mae_lstm_kg']:.3f} kg over the last 7 days "
                f"({_m['n_users']} held-out users)"
            )
            st.write(f"• **Linear baseline MAE:** {_m['mae_linear_kg']:.3f} kg")
            st.write(f"• **Improvement:** {_m['improvement_pct_vs_linear']:+.1f}% vs linear")
            st.caption(_m.get("note", ""))
        else:
            st.write(
                "No evaluation file found. Run `python scripts/retrain_and_evaluate.py` to train "
                "and compare the LSTM against the linear baseline."
            )

    divider()
    _, col_r = st.columns([1, 1])
    with col_r:
        if st.button("Start over", width="stretch"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown(
        '<div class="footer">SIMPLY-FIT &nbsp;·&nbsp; Internship Project &nbsp;·&nbsp; '
        'Not a substitute for professional medical advice</div>',
        unsafe_allow_html=True,
    )
