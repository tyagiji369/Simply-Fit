import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

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

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
PAGES      = ["profile", "metrics", "goal", "target", "baseline", "tracker", "report"]
TOTAL      = len(PAGES)
PAGE_NAMES = ["Profile", "Metrics", "Goal", "Target", "Baseline", "Tracker", "Report"]
KCAL_PER_KG = 7700

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
            ("Banana", "1 large", 120),
            ("Whole milk", "1 glass", 150),
        ],
        "Lunch (1–2 PM)": [
            ("Rice / chapati", "1 cup / 3 roti", 320),
            ("Dal / chicken", "200g", 240),
            ("Sabzi", "1.5 cups", 80),
            ("Curd", "1 bowl", 100),
        ],
        "Snacks (4–5 PM)": [
            ("Banana + peanut butter", "1 banana + 1 tbsp", 195),
            ("Mixed nuts", "30g", 180),
            ("Whole milk", "1 glass", 150),
        ],
        "Dinner (7–8 PM)": [
            ("Rice / chapati", "1 cup / 2 roti", 260),
            ("Dal / paneer", "150g", 200),
            ("Rajma / chickpeas", "100g", 170),
        ],
    },
    "maintain": {
        "Breakfast (7–9 AM)": [
            ("Whole wheat toast / poha", "2 slices / 1 cup", 180),
            ("Boiled egg", "1 whole", 70),
            ("Mixed fruit", "1 cup", 80),
        ],
        "Lunch (1–2 PM)": [
            ("Chapati / rice", "2 roti / ½ cup", 200),
            ("Dal", "150g", 120),
            ("Sabzi", "1.5 cups", 80),
            ("Salad", "freely", 30),
        ],
        "Snacks (4–5 PM)": [
            ("Fruit", "1 medium", 80),
            ("Walnuts", "4 halves", 98),
        ],
        "Dinner (7–8 PM)": [
            ("Chapati / rice", "2 roti / ½ cup", 200),
            ("Dal / sabzi", "150g", 120),
            ("Curd", "1 bowl", 100),
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
DEFAULTS = {
    "page":             "profile",
    "height":           170,
    "weight":           70.0,
    "age":              25,
    "gender":           "Male",
    "activity":         "Sedentary (desk job, little exercise)",
    "diseases":         [],
    "goal":             "lose",
    "target_change":    5.0,
    "target_weeks":     10,
    "target_calories":  0.0,
    "bmr":              0.0,
    "tdee":             0.0,
    "baseline_weights": [],
    "weight_log":       [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════
def go(page):
    st.session_state.page = page
    st.rerun()

def brand():
    st.markdown(
        '<div class="brand-header">'
        '<div class="brand-logo">Simply<em>Fit</em></div>'
        '<span class="brand-pill">Internship Project</span>'
        '</div>',
        unsafe_allow_html=True,
    )

def progress_bar(current):
    idx  = PAGES.index(current)
    segs = "".join(
        f'<div class="prog-seg {"done" if i < idx else "active" if i == idx else ""}"></div>'
        for i in range(TOTAL)
    )
    st.markdown(
        f'<div class="prog-wrap"><div class="prog-meta">'
        f'<span class="prog-name">{PAGE_NAMES[idx]}</span>'
        f'<span class="prog-frac">{idx+1} / {TOTAL}</span>'
        f'</div><div class="prog-track">{segs}</div></div>',
        unsafe_allow_html=True,
    )

def title(main, sub=""):
    st.markdown(f'<div class="page-title">{main}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="page-sub">{sub}</p>', unsafe_allow_html=True)

def slabel(text):
    st.markdown(f'<div class="slabel">{text}</div>', unsafe_allow_html=True)

def alert(msg, kind="neutral"):
    st.markdown(f'<div class="alert a-{kind}">{msg}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="div">', unsafe_allow_html=True)

def nav(back, next_label="Continue", disabled=False):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", use_container_width=True):
            go(back)
    with c2:
        return st.button(next_label, use_container_width=True, disabled=disabled)

def chart_wrap(label=""):
    st.markdown(f'<div class="chart-wrap"><div class="chart-lbl">{label}</div>',
                unsafe_allow_html=True)

def chart_end():
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ML HELPERS
# ═══════════════════════════════════════════════════════════════
def calc_bmr(w, h, age, gender):
    return (10*w + 6.25*h - 5*age + 5) if gender == "Male" \
           else (10*w + 6.25*h - 5*age - 161)

def bmi_cat(bmi):
    if bmi < 18.5: return "Underweight",  "p-yellow", "amber"
    if bmi < 25:   return "Normal weight", "p-green",  "green"
    if bmi < 30:   return "Overweight",   "p-orange", "amber"
    return               "Obese",         "p-red",    "red"

def ewma(weights, span=7):
    return pd.Series(weights).ewm(span=span, adjust=False).mean().values

def detect_anomalies(weights):
    smoothed  = ewma(weights)
    residuals = np.array(weights) - smoothed
    iso       = IsolationForest(contamination=0.1, random_state=42)
    flags     = iso.fit_predict(residuals.reshape(-1, 1)) == -1
    return flags, smoothed

def estimate_balance(weights):
    smoothed     = ewma(weights)
    days         = np.arange(len(smoothed)).reshape(-1, 1)
    model        = LinearRegression()
    model.fit(days, smoothed)
    kg_per_day   = model.coef_[0]
    kcal_per_day = kg_per_day * KCAL_PER_KG
    r2           = model.score(days, smoothed)
    return round(kcal_per_day, 1), round(kg_per_day * 7, 3), round(r2, 3)

def rolling_avg(data, w=7):
    return [float(np.mean(data[max(0, i-w+1):i+1])) for i in range(len(data))]

def smart_ylim(data, pad=0.35):
    mn, mx = min(data), max(data)
    span   = max(mx - mn, 0.8)
    return mn - span*pad, mx + span*pad

def mpl_style():
    plt.rcParams.update({
        "figure.facecolor": "#fff", "axes.facecolor": "#fff",
        "axes.edgecolor": "#E8E4DC", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": "#F0EDE7",
        "grid.linewidth": 0.7, "xtick.color": "#B8B0A8",
        "ytick.color": "#B8B0A8", "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "font.family": "sans-serif",
        "legend.framealpha": 1, "legend.edgecolor": "#E8E4DC",
        "legend.facecolor": "#fff", "legend.fontsize": 8,
    })

def make_fig(w=7, h=3.5):
    mpl_style()
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#fff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

def get_coach_response(profile, weight_log, question, target_weekly):
    recent        = weight_log[-7:]
    weekly_change = round(recent[-1] - recent[0], 2)
    on_track      = abs(weekly_change - target_weekly) < 0.15
    disease       = profile.get("diseases", [])

    if on_track:
        response = (
            f"You are right on track. Your weight changed by "
            f"{weekly_change:+.2f} kg this week against a target of "
            f"{target_weekly:+.1f} kg. Keep doing what you are doing. "
            f"Consistency is your biggest asset right now."
        )
    else:
        gap = round(target_weekly - weekly_change, 2)
        food_adj = abs(int(gap * 7700 / 7 * 0.6))
        act_adj  = abs(int(gap * 7700 / 7 * 0.4))
        response = (
            f"You are slightly off track — {weekly_change:+.2f} kg this week "
            f"vs target of {target_weekly:+.1f} kg. "
            f"Try reducing intake by {food_adj} kcal/day "
            f"and increasing activity by {act_adj} kcal/day."
        )
    if disease:
        response += (
            f" Given your condition(s), please verify any "
            f"dietary changes with your doctor."
        )

    # TODO: replace with real API when credits available
    # from google import genai
    # client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # response = client.models.generate_content(
    #     model="gemini-1.5-flash", contents=prompt).text

    return response

def render_meal_plan(goal, tdee, target_cal):
    intake  = int(tdee + target_cal)
    plan    = MEAL_PLANS.get(goal, MEAL_PLANS["maintain"])
    slabel("Recommended daily meal plan")
    st.markdown(
        f'<div class="callout">'
        f'<div class="callout-big">{intake}</div>'
        f'<div class="callout-detail">Target daily intake in <strong>kcal</strong><br>'
        f'Spread across 4 meal slots. Scale quantities to your weight.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for meal_name, items in plan.items():
        meal_total = sum(k for _, _, k in items)
        html = (f'<div class="meal-card">'
                f'<div class="meal-head">{meal_name} &nbsp;·&nbsp; ~{meal_total} kcal</div>')
        for food, qty, kcal in items:
            html += (f'<div class="meal-item"><span>{food}</span>'
                     f'<span style="display:flex;gap:10px">'
                     f'<span class="meal-qty">{qty}</span>'
                     f'<span class="meal-kcal">{kcal} kcal</span>'
                     f'</span></div>')
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PROFILE
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "profile":
    brand()
    progress_bar("profile")
    title("Tell us about<br><em>yourself.</em>",
          "Your height, weight, age and activity level form the metabolic baseline for every recommendation.")

    slabel("Body measurements")
    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Height (cm)", 100, 230,
                                 int(st.session_state.height), step=1)
        age    = st.number_input("Age", 10, 100,
                                 int(st.session_state.age), step=1)
    with c2:
        weight = st.number_input("Current weight (kg)", 20.0, 300.0,
                                 float(st.session_state.weight),
                                 step=0.1, format="%.1f")
        gender = st.selectbox("Biological sex", ["Male", "Female"],
                              index=0 if st.session_state.gender == "Male" else 1)

    slabel("Activity level")
    activity = st.selectbox("Typical activity level",
                            list(ACTIVITY_MULT.keys()))

    slabel("Medical conditions — select all that apply")
    diseases = st.multiselect("Conditions", DISEASES,
                              default=st.session_state.diseases,
                              label_visibility="collapsed")
    if diseases:
        alert(f"{len(diseases)} condition(s) selected — recommendations will be personalised accordingly.", "warn")
    else:
        alert("No conditions selected — standard recommendations will apply.", "neutral")

    divider()
    _, c2 = st.columns([1, 1])
    with c2:
        if st.button("Continue", use_container_width=True):
            st.session_state.height   = height
            st.session_state.weight   = weight
            st.session_state.age      = age
            st.session_state.gender   = gender
            st.session_state.activity = activity
            st.session_state.diseases = diseases
            go("metrics")

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — METRICS
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "metrics":
    brand()
    progress_bar("metrics")

    h    = st.session_state.height
    w    = st.session_state.weight
    a    = st.session_state.age
    g    = st.session_state.gender
    mult = ACTIVITY_MULT.get(st.session_state.activity, 1.2)

    bmi  = w / (h/100)**2
    bmr  = calc_bmr(w, h, a, g)
    tdee = bmr * mult
    st.session_state.bmr  = bmr
    st.session_state.tdee = tdee

    bmi_lbl, bmi_pill, card_col = bmi_cat(bmi)
    title("Your metabolic<br><em>baseline.</em>",
          "Three numbers that underpin every recommendation Simply-Fit makes.")

    st.markdown(f"""
    <div class="cards">
      <div class="card {card_col}">
        <div class="c-lbl">BMI</div>
        <div class="c-val">{bmi:.1f}</div>
        <span class="pill {bmi_pill}">{bmi_lbl}</span>
        <div class="c-sub">Body mass index</div>
      </div>
      <div class="card">
        <div class="c-lbl">Resting BMR</div>
        <div class="c-val">{int(bmr)}<span class="c-unit">kcal</span></div>
        <div class="c-sub">Calories at complete rest</div>
      </div>
      <div class="card blue">
        <div class="c-lbl">TDEE</div>
        <div class="c-val">{int(tdee)}<span class="c-unit">kcal</span></div>
        <div class="c-sub">Total daily energy expenditure</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    slabel("BMI spectrum")
    chart_wrap()
    fig_b, ax_b = make_fig(7, 1.05)
    zones = [(15,18.5,"#FEF9C3","Underweight"),(18.5,25,"#D1FAE5","Normal"),
             (25,30,"#FFEDD5","Overweight"),(30,40,"#FEE2E2","Obese")]
    for lo, hi, col, lbl in zones:
        ax_b.barh(0, hi-lo, left=lo, color=col, height=0.55,
                  edgecolor="#E8E4DC", linewidth=0.6)
        ax_b.text((lo+hi)/2, -0.5, lbl, ha="center", va="top",
                  fontsize=7.5, color="#A09888")
    ax_b.axvline(bmi, color="#1A1A1A", linewidth=2.5, zorder=5)
    ax_b.scatter([bmi], [0], s=80, color="#1A1A1A", zorder=6)
    ax_b.text(bmi, 0.38, f"  {bmi:.1f}", ha="left" if bmi < 36 else "right",
              va="bottom", fontsize=9.5, fontweight="bold", color="#1A1A1A")
    ax_b.set_xlim(15, 40); ax_b.set_ylim(-0.72, 0.6)
    ax_b.set_yticks([]); ax_b.set_xticks([18.5, 25, 30])
    ax_b.set_xticklabels(["18.5", "25", "30"]); ax_b.grid(False)
    for sp in ax_b.spines.values(): sp.set_visible(False)
    plt.tight_layout(pad=0.3)
    st.pyplot(fig_b, use_container_width=True)
    plt.close()
    chart_end()

    divider()
    if nav("profile", "Set my goal"):
        go("goal")

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — GOAL
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "goal":
    brand()
    progress_bar("goal")
    title("What are you<br><em>working towards?</em>",
          "Your goal shapes every calorie target and recommendation.")

    goal = st.radio("", ["lose", "gain", "maintain"],
                    format_func=lambda x: {
                        "lose": "Lose weight",
                        "gain": "Gain weight",
                        "maintain": "Maintain weight"
                    }[x],
                    index=["lose","gain","maintain"].index(st.session_state.goal),
                    label_visibility="collapsed")

    descs = {
        "lose":     ("success", "Achieve a calorie deficit through diet and movement. Safe limit: 1 kg/week."),
        "gain":     ("warn",    "Build mass through a controlled surplus. Safe limit: 0.5 kg/week."),
        "maintain": ("info",    "Keep your current weight stable. We will track fluctuations and alert you if the trend drifts."),
    }
    alert(descs[goal][1], descs[goal][0])

    divider()
    if nav("metrics", "Continue"):
        st.session_state.goal = goal
        go("target")

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
        max_r = 1.0 if goal == "lose" else 0.5

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
                f'Rate: {rate:.2f} kg/week</div></div>',
                unsafe_allow_html=True,
            )
            st.session_state.target_change   = change
            st.session_state.target_weeks    = weeks
            st.session_state.target_calories = target_calories

    divider()
    if nav("goal", "Continue", disabled=not ok):
        if goal == "maintain":
            st.session_state.target_calories = 0.0
        go("baseline")

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — BASELINE
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "baseline":
    brand()
    progress_bar("baseline")
    title("Enter your<br><em>baseline weights.</em>",
          "7–14 days of past readings calibrate your trend accurately. Oldest reading first.")

    alert("Weigh at the same time daily — ideally morning, after bathroom, before eating.", "neutral")

    slabel("Daily readings — comma separated, oldest first")
    stored = ", ".join(str(x) for x in st.session_state.baseline_weights)
    raw    = st.text_area("Readings (kg):", value=stored,
                          placeholder="e.g.  73.4, 73.0, 72.8, 73.1",
                          height=75, label_visibility="collapsed")

    cg, _ = st.columns([1, 2])
    with cg:
        if st.button("Generate sample data", use_container_width=True):
            base  = st.session_state.weight
            trend = -0.07 if st.session_state.goal == "lose" else \
                     0.04 if st.session_state.goal == "gain" else 0.0
            sample = [round(base + trend*i + random.gauss(0, 0.22), 1)
                      for i in range(14)]
            st.session_state.baseline_weights = sample
            st.rerun()

    weights, err = [], False
    if raw.strip():
        try:
            weights = [float(x.strip()) for x in raw.split(",") if x.strip()]
            if any(w < 20 or w > 300 for w in weights):
                alert("Some values look unrealistic.", "danger"); err = True
        except ValueError:
            alert("Could not parse. Use numbers separated by commas.", "danger"); err = True

    if weights and not err:
        n = len(weights)
        st.caption(f"{n} reading{'s' if n!=1 else ''} loaded.")
        if n < 7:
            alert("At least 7 readings give a more reliable trend.", "warn")

        slabel("Baseline preview — EWMA noise filter applied")
        chart_wrap("Daily readings + exponential weighted moving average")
        smoothed = ewma(weights)
        d        = list(range(1, n+1))
        ylo, yhi = smart_ylim(list(weights) + list(smoothed))

        fig_p, ax_p = make_fig(7, 2.5)
        ax_p.fill_between(d, weights, ylo, alpha=0.06, color="#2D6A4F")
        ax_p.plot(d, weights, color="#C8C3BB", linewidth=1.2, marker="o",
                  markersize=4, markerfacecolor="#2D6A4F",
                  markeredgecolor="#2D6A4F", label="Raw readings")
        ax_p.plot(d, smoothed, color="#1A1A1A", linewidth=2.0, label="EWMA trend")
        ax_p.set_ylim(ylo, yhi)
        ax_p.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax_p.set_xlabel("Day", fontsize=8.5)
        ax_p.set_ylabel("Weight (kg)", fontsize=8.5)
        ax_p.legend()
        plt.tight_layout()
        st.pyplot(fig_p, use_container_width=True)
        plt.close()
        chart_end()

    divider()
    can_go = len(weights) >= 3 and not err
    if nav("target", "Continue", disabled=not can_go):
        st.session_state.baseline_weights = weights
        if not st.session_state.weight_log:
            st.session_state.weight_log = list(weights)
        go("tracker")

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — TRACKER
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "tracker":
    brand()
    progress_bar("tracker")
    title("Daily<br><em>tracker.</em>",
          "Log your weight every morning. Simply-Fit figures out the rest.")

    log      = st.session_state.weight_log
    target   = st.session_state.target_calories
    goal     = st.session_state.goal
    diseases = st.session_state.diseases

    slabel("Today's weight")
    c_inp, c_btn = st.columns([3, 1])
    with c_inp:
        today_w = st.number_input(
            "Weight (kg)", 20.0, 300.0,
            value=float(log[-1]) if log else float(st.session_state.weight),
            step=0.1, format="%.1f", label_visibility="collapsed",
        )
    with c_btn:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Log", use_container_width=True):
            st.session_state.weight_log.append(round(today_w, 1))
            st.rerun()

    n = len(log)
    st.caption(f"{n} reading{'s' if n!=1 else ''} logged")

    if n >= 7:
        flags, smoothed      = detect_anomalies(log)
        kcal_balance, weekly_kg, r2 = estimate_balance(log)
        target_weekly        = (target * 7) / KCAL_PER_KG

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
            <div class="c-sub">From weight trend — no food log needed</div>
          </div>
          <div class="card">
            <div class="c-lbl">Weekly rate</div>
            <div class="c-val">{weekly_kg:+.2f}<span class="c-unit">kg/w</span></div>
            <div class="c-sub">Personal regression R²={r2}</div>
          </div>
          <div class="card {"green" if flags.sum()==0 else "amber"}">
            <div class="c-lbl">Anomalies</div>
            <div class="c-val">{int(flags.sum())}</div>
            <div class="c-sub">Noisy readings filtered out</div>
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
                f"{int(flags.sum())} anomalous reading(s) detected by Isolation Forest "
                f"and filtered from calorie calculations. These are likely water retention spikes.",
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
        st.pyplot(fig_t, use_container_width=True)
        plt.close()
        chart_end()

        slabel("Ask your coach")
        question = st.text_input("Ask anything about your progress...",
                                 placeholder="Why is my weight fluctuating?")
        if st.button("Ask coach", use_container_width=True):
            if question.strip():
                response = get_coach_response(
                    {"diseases": diseases}, log, question, target_weekly
                )
                st.markdown(
                    f'<div class="coach-box">{response}</div>',
                    unsafe_allow_html=True
                )

    elif n > 0:
        alert(f"Log {7 - n} more reading(s) to unlock ML insights and coaching.", "info")

        if n >= 2:
            slabel("Early trend preview")
            chart_wrap()
            ra  = rolling_avg(log, min(3, n))
            d   = list(range(1, n+1))
            ylo, yhi = smart_ylim(log + ra)
            fig_e, ax_e = make_fig(7, 2.5)
            ax_e.plot(d, log, color="#C8C3BB", linewidth=1.2,
                      marker="o", markersize=5,
                      markerfacecolor="#2D6A4F", markeredgecolor="#2D6A4F",
                      label="Daily weight")
            ax_e.plot(d, ra, color="#1A1A1A", linewidth=2.0, label="Rolling avg")
            ax_e.set_ylim(ylo, yhi)
            ax_e.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax_e.set_xlabel("Day", fontsize=8.5)
            ax_e.set_ylabel("Weight (kg)", fontsize=8.5)
            ax_e.legend()
            plt.tight_layout()
            st.pyplot(fig_e, use_container_width=True)
            plt.close()
            chart_end()

    divider()
    ready = n >= 7
    if nav("baseline", "View full report", disabled=not ready):
        go("report")
    if not ready:
        st.caption(f"Log {max(0, 7-n)} more reading(s) to unlock the full report.")

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

    flags, smoothed             = detect_anomalies(log)
    kcal_balance, weekly_kg, r2 = estimate_balance(log)

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
        <div class="c-sub">No food log needed</div></div>
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
    st.pyplot(fig1, use_container_width=True)
    plt.close()
    chart_end()

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
        st.pyplot(fig2, use_container_width=True)
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
        st.pyplot(fig3, use_container_width=True)
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

    # Recommendations
    slabel("General recommendations")
    recs = [
        ("Weigh yourself at the same time every morning — before eating, after bathroom. Consistency matters more than the exact time.", False),
        (f"Target protein at 1.6–2.0 g per kg — roughly {int(1.8 * w_kg)} g/day — to preserve muscle regardless of your goal.", False),
        ("7–9 hours of sleep is essential. Sleep deficit raises hunger hormones and impairs fat loss even with the same calorie intake.", False),
        ("150 minutes of moderate aerobic activity per week is the minimum evidence-backed recommendation for metabolic health.", False),
        ("Limit ultra-processed foods — they are engineered to override satiety signals and are the primary driver of unintended weight gain.", False),
        (f"Drink roughly {int(35 * w_kg)} ml of water per day ({int(35 * w_kg / 250)} glasses). Thirst is frequently misread as hunger.", False),
    ]
    if diseases:
        recs.append(("Your medical condition(s) may require specific dietary adjustments beyond these general recommendations. Always verify changes with your doctor.", True))

    recs_html = ""
    for i, (text, flagged) in enumerate(recs, 1):
        cls = "rec flagged" if flagged else "rec"
        recs_html += (f'<div class="{cls}"><span class="rec-n">{i:02d}</span>'
                      f'<span>{text}</span></div>')
    st.markdown(recs_html, unsafe_allow_html=True)

    divider()
    _, col_r = st.columns([1, 1])
    with col_r:
        if st.button("Start over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown(
        '<div class="footer">SIMPLY-FIT &nbsp;·&nbsp; Internship Project &nbsp;·&nbsp; '
        'Not a substitute for professional medical advice</div>',
        unsafe_allow_html=True,
    )