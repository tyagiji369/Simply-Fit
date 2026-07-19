import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Simply-Fit",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Styles ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F5F1;
    color: #1A1A1A;
}
.block-container { padding: 2rem 2rem 4rem 2rem; max-width: 800px; }
.brand { font-size: 1.6rem; font-weight: 600; color: #1A1A1A; margin-bottom: 0.2rem; }
.brand span { color: #2D6A4F; }
.metric-card {
    background: #fff; border: 1px solid #E8E4DC;
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-bottom: 8px;
}
.metric-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #B8B0A8; margin-bottom: 0.3rem; }
.metric-value { font-size: 1.8rem; font-weight: 600; color: #1A1A1A; }
.metric-sub { font-size: 0.78rem; color: #A8A098; margin-top: 0.2rem; }
.alert-success { background: #F0FDF4; color: #14532D; border: 1px solid #86EFAC;
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.alert-warn { background: #FFFBEB; color: #713F12; border: 1px solid #FCD34D;
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.alert-info { background: #EFF6FF; color: #1E3A8A; border: 1px solid #93C5FD;
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.coach-box { background: #1A1A1A; color: #F5F5F3; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin: 1rem 0; font-size: 0.9rem; line-height: 1.7; }
.section-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #B8B0A8; margin: 1.8rem 0 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
KCAL_PER_KG = 7700
ACTIVITY_MULT = {
    "Sedentary":        1.2,
    "Lightly active":   1.375,
    "Moderately active":1.55,
    "Very active":      1.725,
    "Athlete":          1.9
}
DISEASES = [
    "None", "Type 2 Diabetes", "Hypertension",
    "High Cholesterol", "Hypothyroidism", "PCOS",
    "NAFLD", "Insulin Resistance", "CKD"
]

# ── Session state ──────────────────────────────────────────────
DEFAULTS = {
    "page":         "profile",
    "weight_log":   [],
    "profile":      {},
    "goal":         "lose",
    "target_weekly":-0.5,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helper functions ───────────────────────────────────────────
def go(page):
    st.session_state.page = page
    st.rerun()

def calc_bmr(w, h, age, gender):
    if gender == "Male":
        return 10*w + 6.25*h - 5*age + 5
    return 10*w + 6.25*h - 5*age - 161

def ewma(weights, span=7):
    return pd.Series(weights).ewm(span=span, adjust=False).mean().values

def detect_anomalies(weights):
    smoothed  = ewma(weights)
    residuals = np.array(weights) - smoothed
    iso       = IsolationForest(contamination=0.1, random_state=42)
    flags     = iso.fit_predict(residuals.reshape(-1, 1)) == -1
    return flags, smoothed

def estimate_balance(weights):
    smoothed = ewma(weights)
    days     = np.arange(len(smoothed)).reshape(-1, 1)
    model    = LinearRegression()
    model.fit(days, smoothed)
    kg_per_day   = model.coef_[0]
    kcal_per_day = kg_per_day * KCAL_PER_KG
    return round(kcal_per_day, 1), round(kg_per_day * 7, 3)

def get_coach_response(profile, weight_log, question, target_weekly):
    recent        = weight_log[-7:]
    weekly_change = round(recent[-1] - recent[0], 2)
    on_track      = abs(weekly_change - target_weekly) < 0.15
    disease       = profile.get("disease", "None")

    if on_track:
        response = (
            f"You are right on track. Your weight changed by "
            f"{weekly_change:+.2f} kg this week against a target of "
            f"{target_weekly:+.1f} kg. Keep doing what you are doing. "
            f"Consistency is your biggest asset right now."
        )
    else:
        gap = round(target_weekly - weekly_change, 2)
        response = (
            f"You are slightly off track — {weekly_change:+.2f} kg this week "
            f"vs target of {target_weekly:+.1f} kg. "
            f"Try reducing intake by {abs(int(gap*7700/7*0.6))} kcal/day "
            f"and increasing activity by {abs(int(gap*7700/7*0.4))} kcal/day."
        )
    if disease != "None":
        response += (
            f" Given your {disease} condition, please verify any "
            f"dietary changes with your doctor."
        )

    # TODO: uncomment when API credits available
    # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # model = genai.GenerativeModel("gemini-1.5-flash")
    # response = model.generate_content(prompt).text

    return response

def make_chart(weights):
    flags, smoothed = detect_anomalies(weights)
    days = np.arange(1, len(weights)+1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#fff")
    ax.set_facecolor("#fff")
    ax.plot(days, weights, color="#C8C3BB", linewidth=1.2,
            marker="o", markersize=3.5, label="Daily weight")
    ax.plot(days, smoothed, color="#1A1A1A", linewidth=2.2,
            label="Trend (EWMA)")
    ax.scatter(days[flags], np.array(weights)[flags],
               color="#B92D2D", s=45, zorder=5, label="Anomaly")
    ax.set_xlabel("Day", fontsize=8.5)
    ax.set_ylabel("Weight (kg)", fontsize=8.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PROFILE
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "profile":
    st.markdown('<div class="brand">Simply<span>Fit</span></div>', 
                unsafe_allow_html=True)
    st.markdown("#### Tell us about yourself")
    st.markdown("We use this to calculate your metabolic baseline.")

    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Height (cm)", 100, 230, 170)
        age    = st.number_input("Age", 10, 100, 25)
    with c2:
        weight = st.number_input("Current weight (kg)", 20.0, 300.0, 70.0, 0.1)
        gender = st.selectbox("Gender", ["Male", "Female"])

    activity = st.selectbox("Activity level", list(ACTIVITY_MULT.keys()))
    disease  = st.selectbox("Medical condition (if any)", DISEASES)

    if st.button("Continue", use_container_width=True):
        bmr  = calc_bmr(weight, height, age, gender)
        tdee = bmr * ACTIVITY_MULT[activity]
        st.session_state.profile = {
            "height": height, "weight": weight,
            "age": age, "gender": gender,
            "activity": activity, "disease": disease,
            "bmr": round(bmr, 1), "tdee": round(tdee, 1)
        }
        go("goal")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — GOAL
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "goal":
    st.markdown('<div class="brand">Simply<span>Fit</span></div>',
                unsafe_allow_html=True)
    st.markdown("#### What are you working towards?")

    goal = st.radio("", ["Lose weight", "Gain weight", "Maintain weight"],
                    label_visibility="collapsed")

    if goal != "Maintain weight":
        change = st.number_input("How much? (kg)", 0.5, 50.0, 5.0, 0.5)
        weeks  = st.number_input("In how many weeks?", 1, 104, 10, 1)
        rate   = change / weeks
        max_r  = 1.0 if "Lose" in goal else 0.5
        if rate > max_r:
            st.markdown(
                f'<div class="alert-warn">Rate of {rate:.2f} kg/week exceeds '
                f'safe limit of {max_r} kg/week. Please adjust.</div>',
                unsafe_allow_html=True
            )
        else:
            target_weekly = -round(change/weeks, 3) if "Lose" in goal \
                             else round(change/weeks, 3)
            st.session_state.target_weekly = target_weekly
    else:
        st.session_state.target_weekly = 0.0

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", use_container_width=True):
            go("profile")
    with c2:
        if st.button("Continue", use_container_width=True):
            st.session_state.goal = goal
            go("tracker")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TRACKER
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "tracker":
    st.markdown('<div class="brand">Simply<span>Fit</span></div>',
                unsafe_allow_html=True)
    st.markdown("#### Daily weight tracker")
    st.markdown("Log your weight every morning. That's all you need to do.")

    profile = st.session_state.profile
    log     = st.session_state.weight_log

    # Log today's weight
    c1, c2 = st.columns([3, 1])
    with c1:
        today = st.number_input(
            "Today's weight (kg)", 20.0, 300.0,
            float(log[-1]) if log else float(profile.get("weight", 70)),
            0.1, label_visibility="collapsed"
        )
    with c2:
        if st.button("Log", use_container_width=True):
            st.session_state.weight_log.append(round(today, 1))
            st.rerun()

    st.caption(f"{len(log)} reading{'s' if len(log)!=1 else ''} logged")

    if len(log) >= 7:
        # ML pipeline
        flags, smoothed = detect_anomalies(log)
        kcal_balance, weekly_kg = estimate_balance(log)
        target = st.session_state.target_weekly

        # Metrics
        st.markdown('<div class="section-label">This week</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current</div>
                <div class="metric-value">{log[-1]:.1f}<small> kg</small></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Weekly rate</div>
                <div class="metric-value">{weekly_kg:+.2f}<small> kg</small></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cal balance</div>
                <div class="metric-value">{int(kcal_balance):+d}<small> kcal</small></div>
            </div>""", unsafe_allow_html=True)

        # Recommendation
        on_track = abs(weekly_kg - target) < 0.15
        if on_track:
            st.markdown(
                '<div class="alert-success">You are on track this week. '
                'Keep your current habits.</div>',
                unsafe_allow_html=True
            )
        elif weekly_kg < target:
            gap = abs(target - weekly_kg)
            st.markdown(
                f'<div class="alert-warn">Slightly behind. Try reducing '
                f'{int(gap*7700/7*0.6):.0f} kcal/day from food and burning '
                f'{int(gap*7700/7*0.4):.0f} kcal/day more through activity.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="alert-info">Ahead of schedule. '
                'Maintain your current pace.</div>',
                unsafe_allow_html=True
            )

        if flags.sum() > 0:
            st.caption(f"{flags.sum()} anomalous reading(s) detected and "
                       f"filtered from calculations.")

        # Chart
        st.markdown('<div class="section-label">Weight trend</div>',
                    unsafe_allow_html=True)
        st.pyplot(make_chart(log), use_container_width=True)

        # AI Coach
        st.markdown('<div class="section-label">Ask your coach</div>',
                    unsafe_allow_html=True)
        question = st.text_input("Ask anything about your progress...",
                                 placeholder="Why is my weight fluctuating?")
        if st.button("Ask coach", use_container_width=True):
            if question.strip():
                response = get_coach_response(
                    profile, log, question,
                    st.session_state.target_weekly
                )
                st.markdown(
                    f'<div class="coach-box">{response}</div>',
                    unsafe_allow_html=True
                )

    elif len(log) > 0:
        st.markdown(
            f'<div class="alert-info">Log {7 - len(log)} more '
            f'reading(s) to unlock insights and coaching.</div>',
            unsafe_allow_html=True
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", use_container_width=True):
            go("goal")
    with c2:
        if st.button("View full report", use_container_width=True,
                     disabled=len(log) < 7):
            go("report")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — REPORT
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "report":
    st.markdown('<div class="brand">Simply<span>Fit</span></div>',
                unsafe_allow_html=True)
    st.markdown("#### Your progress report")

    profile = st.session_state.profile
    log     = st.session_state.weight_log
    target  = st.session_state.target_weekly

    flags, smoothed = detect_anomalies(log)
    kcal_balance, weekly_kg = estimate_balance(log)

    total_change = round(log[-1] - log[0], 2)
    bmi = round(profile["weight"] / (profile["height"]/100)**2, 1)

    # Summary cards
    st.markdown('<div class="section-label">Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Start</div>
            <div class="metric-value">{log[0]:.1f}</div>
            <div class="metric-sub">kg</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current</div>
            <div class="metric-value">{log[-1]:.1f}</div>
            <div class="metric-sub">kg</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Change</div>
            <div class="metric-value">{total_change:+.1f}</div>
            <div class="metric-sub">kg total</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">BMI</div>
            <div class="metric-value">{bmi}</div>
            <div class="metric-sub">current</div>
        </div>""", unsafe_allow_html=True)

    # Chart
    st.markdown('<div class="section-label">Full trend</div>',
                unsafe_allow_html=True)
    st.pyplot(make_chart(log), use_container_width=True)

    # Stats
    st.markdown('<div class="section-label">Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    - Estimated daily calorie balance: **{int(kcal_balance):+d} kcal/day**
    - Weekly weight change rate: **{weekly_kg:+.3f} kg/week**
    - Anomalous readings filtered: **{flags.sum()}**
    - Days tracked: **{len(log)}**
    - TDEE: **{int(profile['tdee'])} kcal/day**
    - BMR: **{int(profile['bmr'])} kcal/day**
    """)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to tracker", use_container_width=True):
            go("tracker")
    with c2:
        if st.button("Start over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()