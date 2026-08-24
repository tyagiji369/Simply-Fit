import sys
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv()

from src.ml_engine import run_pipeline as run_ml_pipeline
from src.lstm_forecaster import forecast as forecast_lstm
from src.agent_coach import agent_coach
from src.rag_pipeline import rag_engine
from src.calibration import run_nhanes_calibration_test

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Simply-Fit | Personal Weight & Calorie Tracker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F8FAFC;
    color: #0F172A;
}

.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 4rem;
    max-width: 1080px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

/* User Dashboard Cards */
.user-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0F172A;
}

.metric-sub {
    font-size: 0.78rem;
    font-weight: 600;
    color: #2563EB;
    margin-top: 0.35rem;
}

.sub-green { color: #16A34A; }
.sub-red { color: #DC2626; }

/* Chat Bubbles Styling */
.stChatMessage {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR: USER PROFILE & SETTINGS
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚖️ Simply-Fit")
    st.caption("AI-Powered Calorie Inference")
    st.divider()

    st.subheader("👤 Your Profile")
    user_age = st.number_input("Age:", min_value=18, max_value=90, value=34)
    user_gender = st.selectbox("Gender:", ["Male", "Female"], index=0)
    user_height = st.number_input("Height (cm):", min_value=120, max_value=220, value=178)
    user_disease = st.selectbox(
        "Medical Condition:",
        ["Hypertension", "Type 2 Diabetes", "PCOS", "CKD", "Hypothyroidism", "NAFLD", "None"],
        index=0
    )
    target_weekly_change = st.slider("Target Weight Loss (kg/week):", min_value=-1.2, max_value=0.5, value=-0.5, step=0.1)

    st.divider()
    st.subheader("⚙️ API Configuration")
    user_gemini_key = st.text_input(
        "Gemini API Key (Optional):",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Optional: Enter a Gemini API key from Google AI Studio. If left blank, the app uses its offline intelligent coach engine."
    )

    st.divider()
    with st.expander("🔬 Model Insights & Data Calibration"):
        st.caption("CDC NHANES Dataset Benchmarking")
        if st.button("Run KS Test Calibration"):
            ks_res = run_nhanes_calibration_test(n_users=500)
            st.success(f"Weight KS Improvement: +{ks_res['weight_ks_improvement_pct']}%")
            st.info(f"Age KS Improvement: +{ks_res['age_ks_improvement_pct']}%")

# ═══════════════════════════════════════════════════════════════
# SESSION STATE SETUP
# ═══════════════════════════════════════════════════════════════
if "weight_log" not in st.session_state:
    base = 82.0
    st.session_state["weight_log"] = [round(base - (i * 0.08) + random.choice([0.0, -0.1, 0.1, -0.2, 1.2 if i == 12 else 0.0]), 1) for i in range(21)]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {
            "role": "assistant",
            "content": "Hi! I am your Simply-Fit AI Coach. I analyze your scale weights to infer your true daily calorie deficit without food logging. How are you feeling about your progress today?",
            "trace": None
        }
    ]

# Header Bar
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
    <div>
        <h2 style="font-weight: 800; font-size: 1.8rem; margin: 0; color: #0F172A;">Simply-Fit</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Passive Calorie Inference & Personal Health Dashboard</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Calculate ML Pipeline Outputs
weights = st.session_state["weight_log"]
ml_output = run_ml_pipeline(weights, target_weekly_change=target_weekly_change)

# ═══════════════════════════════════════════════════════════════
# SECTION 1: USER DASHBOARD METRICS
# ═══════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="user-card">
        <div class="metric-label">Current Weight</div>
        <div class="metric-value">{weights[-1]} <span style="font-size:0.9rem;">kg</span></div>
        <div class="metric-sub">Start: {weights[0]} kg ({weights[-1] - weights[0]:+.1f} kg)</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="user-card">
        <div class="metric-label">Daily Calorie Balance</div>
        <div class="metric-value">{ml_output['kcal_per_day']:+.0f} <span style="font-size:0.9rem;">kcal/d</span></div>
        <div class="metric-sub sub-green">Target Deficit: {(target_weekly_change * 7700) / 7:+.0f} kcal/d</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="user-card">
        <div class="metric-label">Water Spikes Filtered</div>
        <div class="metric-value">{ml_output['anomalies_detected']} <span style="font-size:0.9rem;">spikes</span></div>
        <div class="metric-sub sub-red">Isolation Forest Active</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="user-card">
        <div class="metric-label">7-Day Trend Pace</div>
        <div class="metric-value">{ml_output['weekly_kg_change']:+.2f} <span style="font-size:0.9rem;">kg/wk</span></div>
        <div class="metric-sub">Target Pace: {target_weekly_change:+.1f} kg/wk</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ═══════════════════════════════════════════════════════════════
# SECTION 2: VISUAL WEIGHT TRAJECTORY & FORECAST CHART
# ═══════════════════════════════════════════════════════════════
col_chart, col_form = st.columns([2.8, 1.2])

with col_chart:
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    days = list(range(1, len(weights) + 1))
    ax.plot(days, weights, color="#94A3B8", linestyle="--", marker="o", markersize=4, label="Raw Scale Log", alpha=0.7)
    ax.plot(days, ml_output["smoothed"], color="#2563EB", linewidth=2.4, label="True Fat Mass Trend (EWMA)")

    # Anomalies
    anomaly_indices = [i + 1 for i, flag in enumerate(ml_output["anomaly_flags"]) if flag]
    if anomaly_indices:
        anomaly_weights = [weights[i - 1] for i in anomaly_indices]
        ax.scatter(anomaly_indices, anomaly_weights, color="#EF4444", s=80, zorder=5, label="Filtered Water Spike")

    # 7-Day Forecast
    lstm_pred = forecast_lstm(weights)
    if lstm_pred is not None:
        future_days = list(range(len(weights) + 1, len(weights) + 8))
        ax.plot(future_days, lstm_pred, color="#16A34A", linestyle="-.", linewidth=2, marker="s", markersize=4, label="7-Day LSTM Forecast")

    ax.set_title("Weight Trajectory & 7-Day Forecast", fontsize=10.5, fontweight="bold", pad=10, color="#0F172A")
    ax.set_xlabel("Day", fontsize=8.5, color="#475569")
    ax.set_ylabel("Weight (kg)", fontsize=8.5, color="#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E1")
    ax.legend(fontsize=8, loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")

    st.pyplot(fig)

with col_form:
    st.markdown("#### ➕ Log Weight")
    st.caption("Record today's scale reading:")
    new_val = st.number_input("Today's Scale Weight (kg):", value=round(weights[-1] - 0.1, 1), step=0.1)
    if st.button("Save Daily Reading", use_container_width=True, type="primary"):
        st.session_state["weight_log"].append(round(new_val, 1))
        st.success("Log updated!")
        st.rerun()

    st.write("")
    st.markdown("#### 🔄 Quick Actions")
    if st.button("Load Normal Progress", use_container_width=True):
        base = 82.0
        st.session_state["weight_log"] = [round(base - (i * 0.08) + random.choice([0.0, -0.1, 0.1, -0.2]), 1) for i in range(21)]
        st.session_state["chat_history"] = []
        st.rerun()

    if st.button("Simulate Sodium Spike", use_container_width=True):
        base = 82.0
        st.session_state["weight_log"] = [round(base - (i * 0.08) + (1.4 if i == 14 else random.choice([0.0, -0.1, 0.1])), 1) for i in range(21)]
        st.session_state["chat_history"] = []
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════
# SECTION 3: CONVERSATIONAL AI HEALTH COACH
# ═══════════════════════════════════════════════════════════════
st.markdown("### 💬 Ask Simply-Fit AI Coach")
st.caption("Get personalized dietary feedback based on your actual scale data, inferred calorie deficit, and medical conditions.")

profile = {
    "age": user_age,
    "gender": user_gender,
    "height_cm": user_height,
    "disease": user_disease
}

# Display Chat Messages
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 How the AI calculated this (Technical Execution Trace)"):
                for step in msg["trace"]:
                    if "thought" in step:
                        st.markdown(f"**Step {step['step']}: Reasoning**")
                        st.write(f"💭 {step['thought']}")
                    else:
                        st.markdown(f"**Step {step['step']}: Tool Invoked — `{step['tool_call']}`**")
                        st.json(step["observation"])

# Chat Input
if prompt := st.chat_input("Type your question (e.g. 'How is my weight progress given my hypertension?')"):
    st.session_state["chat_history"].append({"role": "user", "content": prompt, "trace": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your trend and medical guidelines..."):
            agent_res = agent_coach.run_agent(
                profile,
                st.session_state["weight_log"],
                prompt,
                target_weekly_change=target_weekly_change,
                gemini_api_key=user_gemini_key
            )

        st.markdown(agent_res["response"])
        with st.expander("🔍 How the AI calculated this (Technical Execution Trace)"):
            for step in agent_res["trace"]:
                if "thought" in step:
                    st.markdown(f"**Step {step['step']}: Reasoning**")
                    st.write(f"💭 {step['thought']}")
                else:
                    st.markdown(f"**Step {step['step']}: Tool Invoked — `{step['tool_call']}`**")
                    st.json(step["observation"])

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": agent_res["response"],
        "trace": agent_res["trace"]
    })
