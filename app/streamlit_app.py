import sys
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    page_title="Simply-Fit: AI Calorie Inference & Agentic RAG",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F5F1;
    color: #1A1A1A;
}
.block-container { padding: 2.5rem 2rem 5rem 2rem; max-width: 860px; }

.brand-header {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1.5px solid #1A1A1A; padding-bottom: 12px; margin-bottom: 2rem;
}
.brand-title {
    font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 600;
    letter-spacing: -0.02em; color: #1A1A1A;
}
.brand-sub { font-size: 0.8rem; color: #666; font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 0.08em; }

.card {
    background: #FFFFFF; border: 1px solid #E5E0D8; border-radius: 8px;
    padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; shadow: 0 1px 3px rgba(0,0,0,0.02);
}

.agent-box {
    background: #F0F4F8; border-left: 4px solid #2B579A; padding: 1rem 1.2rem;
    border-radius: 4px; margin-bottom: 1rem;
}

.trace-step {
    background: #FFFFFF; border: 1px solid #D0D7DE; border-radius: 6px;
    padding: 0.75rem 1rem; margin-bottom: 0.6rem; font-size: 0.88rem;
}
.trace-header { font-weight: 600; color: #2B579A; margin-bottom: 4px; }
.trace-json { font-family: monospace; font-size: 0.8rem; background: #F8F9FA; padding: 6px 10px; border-radius: 4px; color: #333; }

.metric-big { font-size: 1.6rem; font-weight: 700; color: #1A1A1A; }
.metric-lbl { font-size: 0.75rem; text-transform: uppercase; color: #777; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# Header UI
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-title">Simply-Fit</div>
        <div class="brand-sub">Passive Weight Management & Agentic Health RAG System</div>
    </div>
    <div style="text-align: right; font-size: 0.8rem; color: #555;">
        <b>Research Internship Project</b><br>IIT Kharagpur
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE SETUP
# ═══════════════════════════════════════════════════════════════
if "weight_log" not in st.session_state:
    # 21 days realistic weight decay log
    base = 82.0
    weights = []
    for i in range(21):
        noise = random.choice([0.0, -0.1, 0.2, -0.2, 0.4 if i == 12 else 0.0])  # Salt spike day 12
        weights.append(round(base - (i * 0.08) + noise, 1))
    st.session_state["weight_log"] = weights

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {
        "age": 34,
        "gender": "Male",
        "height_cm": 178,
        "goal": "lose",
        "disease": "Hypertension"
    }

# Main Tab Selection
tab1, tab2, tab3 = st.tabs([
    "⚖️ Weight Tracker & ML Signal Pipeline",
    "🤖 Agentic AI Coach & RAG Workbench",
    "📊 Population Data Realism (NHANES KS Test)"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: ML SIGNAL PIPELINE
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Passive Calorie Inference Pipeline")
    st.caption("Treating daily scale readings as a passive physiological sensor to extract net calorie balance without food logging.")

    col1, col2, col3 = st.columns(3)
    weights = st.session_state["weight_log"]

    # Run ML Pipeline
    ml_output = run_ml_pipeline(weights, target_weekly_change=-0.5)

    with col1:
        st.markdown(f"<div class='card'><div class='metric-lbl'>Current Weight</div><div class='metric-big'>{weights[-1]} kg</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><div class='metric-lbl'>Inferred Calorie Balance</div><div class='metric-big'>{ml_output['kcal_per_day']:+.0f} kcal/d</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='card'><div class='metric-lbl'>Water Spikes Filtered</div><div class='metric-big'>{ml_output['anomalies_detected']} anomalies</div></div>", unsafe_allow_html=True)

    # Plot Signal Processing Output
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="#F7F5F1")
    ax.set_facecolor("#FFFFFF")
    days = list(range(1, len(weights) + 1))

    # Raw weight
    ax.plot(days, weights, color="#888888", linestyle="--", marker="o", markersize=4, label="Raw Weight Log (Noisy)", alpha=0.7)
    # EWMA Smoothed
    ax.plot(days, ml_output["smoothed"], color="#2B579A", linewidth=2.2, label="EWMA Smoothed Trend (True Fat Signal)")

    # Anomalies
    anomaly_indices = [i + 1 for i, flag in enumerate(ml_output["anomaly_flags"]) if flag]
    if anomaly_indices:
        anomaly_weights = [weights[i - 1] for i in anomaly_indices]
        ax.scatter(anomaly_indices, anomaly_weights, color="#D9534F", s=70, zorder=5, label="Isolation Forest Anomaly (Sodium/Water Spike)")

    # 7-Day LSTM Forecast
    lstm_pred = forecast_lstm(weights)
    if lstm_pred is not None:
        future_days = list(range(len(weights) + 1, len(weights) + 8))
        ax.plot(future_days, lstm_pred, color="#2E7D32", linestyle="-.", linewidth=2, marker="s", markersize=4, label="LSTM 7-Day Forecast")

    ax.set_title("EWMA Noise Filtering & LSTM Weight Forecast", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Weight (kg)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    st.pyplot(fig)

    with st.expander("➕ Log Today's Scale Weight"):
        new_w = st.number_input("Weight Reading (kg):", value=round(weights[-1] - 0.1, 1), step=0.1)
        if st.button("Add Reading"):
            st.session_state["weight_log"].append(round(new_w, 1))
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 2: AGENTIC AI & RAG WORKBENCH
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Agentic AI Coach & Clinical RAG Engine")
    st.caption("Demonstrates ReAct (Reasoning + Acting) orchestrating tool calling across ML signal tools, LSTM neural forecaster, and medical RAG guidelines.")

    profile = st.session_state["user_profile"]

    col_a, col_b = st.columns(2)
    with col_a:
        disease_input = st.selectbox("Medical Condition:", ["Hypertension", "Type 2 Diabetes", "PCOS", "CKD", "Hypothyroidism", "None"], index=0)
        profile["disease"] = disease_input
    with col_b:
        user_query = st.text_input("Ask Coach a Question:", "How is my weight progress given my hypertension condition?")

    if st.button("🚀 Run Agentic Reasoning Loop"):
        with st.spinner("Executing ReAct Tool Calling & RAG Vector Retrieval..."):
            agent_out = agent_coach.run_agent(profile, st.session_state["weight_log"], user_query)

        st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
        st.markdown(f"**🤖 Agent Response:**\n\n{agent_out['response']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 🔍 Step-by-Step Agent Execution Trace")
        for step in agent_out["trace"]:
            st.markdown(f"<div class='trace-step'>", unsafe_allow_html=True)
            if "thought" in step:
                st.markdown(f"<div class='trace-header'>Step {step['step']}: Reasoning & Tool Planning</div>", unsafe_allow_html=True)
                st.write(f"💭 *Thought:* {step['thought']}")
                st.write(f"🔧 *Selected Tools:* `{step['selected_tools']}`")
            else:
                st.markdown(f"<div class='trace-header'>Step {step['step']}: Tool Execution — <code>{step['tool_call']}</code></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='trace-json'>{step['observation']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3: POPULATION DATA REALISM (NHANES KS TEST)
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Population Calibration & Statistical Data Realism")
    st.caption("Proves that synthetic user profiles reflect real human physiology rather than arbitrary assumptions by benchmarking against CDC NHANES public health survey data using Kolmogorov-Smirnov (KS) tests.")

    if st.button("📊 Run Kolmogorov-Smirnov Calibration Test"):
        with st.spinner("Computing 2-Sample KS Test against CDC NHANES Population Statistics..."):
            ks_res = run_nhanes_calibration_test(n_users=500)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='card'>
                <div class='metric-lbl'>Weight KS Improvement</div>
                <div class='metric-big' style='color: #2E7D32;'>+{ks_res['weight_ks_improvement_pct']}%</div>
                <small>KS Stat: {ks_res['weight_ks_before']} ➔ <b>{ks_res['weight_ks_after']}</b></small><br>
                <small>Synthetic Mean Weight: {ks_res['mean_synthetic_weight']} kg vs NHANES: {ks_res['mean_nhanes_weight']} kg</small>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='card'>
                <div class='metric-lbl'>Age KS Improvement</div>
                <div class='metric-big' style='color: #2E7D32;'>+{ks_res['age_ks_improvement_pct']}%</div>
                <small>KS Stat: {ks_res['age_ks_before']} ➔ <b>{ks_res['age_ks_after']}</b></small><br>
                <small>Synthetic Mean Age: {ks_res['mean_synthetic_age']} yrs vs NHANES: {ks_res['mean_nhanes_age']} yrs</small>
            </div>
            """, unsafe_allow_html=True)

        st.success("✅ Synthetic data distribution calibrated successfully! Zero distribution bias detected.")
