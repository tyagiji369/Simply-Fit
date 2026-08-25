"""
Conversational AI coach for Simply-Fit.

Design — intent-routed tool calling (NOT an LLM-driven agent loop):
  1. A keyword-based intent classifier routes the question to one of six
     intents (time-to-goal, calories/nutrition, fluctuations, plateau,
     medical guidelines, general progress) and computes the numbers the
     answer needs.
  2. Deterministic Python tools always run for every question: calorie
     inference (src/ml_engine), 7-day forecasting (src/lstm_forecaster),
     clinical guideline retrieval (src/rag_pipeline) and plateau detection.
  3. The final response is written by the Google Gemini API when a key is
     available; otherwise deterministic per-intent templates built from the
     same tool outputs are used, so the coach works with zero API cost.

The pipeline is fixed on purpose — the LLM never selects which tools run —
which makes behaviour reproducible and testable without any API key. The
`trace` returned with each response is an execution log of what actually
ran, not simulated reasoning.
"""

import os
import json
import math
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv
from src.ml_engine import run_pipeline as run_ml_pipeline
from src.lstm_forecaster import forecast as forecast_lstm
from src.rag_pipeline import rag_engine

load_dotenv()


class SimplyFitAgent:
    """
    Conversational AI Coach for Simply-Fit.
    Intent classification, deterministic tool calls (calorie inference,
    forecasting, clinical RAG, plateau detection), Gemini API synthesis
    when available, deterministic fallback responses otherwise.
    """

    def __init__(self):
        self.tools = {
            "infer_calorie_balance": self._tool_infer_calorie_balance,
            "forecast_trajectory": self._tool_forecast_trajectory,
            "retrieve_clinical_guidelines": self._tool_retrieve_clinical_guidelines,
            "check_plateau_status": self._tool_check_plateau_status
        }

    # ── Tool Definitions ──────────────────────────────────────
    def _tool_infer_calorie_balance(self, weight_log, target_weekly_change):
        return run_ml_pipeline(weight_log, target_weekly_change=target_weekly_change)

    def _tool_forecast_trajectory(self, weight_log):
        try:
            pred = forecast_lstm(weight_log)
            if pred is not None:
                return [round(float(x), 1) for x in pred]
        except Exception:
            pass
        recent = weight_log[-7:]
        daily_rate = (recent[-1] - recent[0]) / 7.0
        return [round(recent[-1] + daily_rate * i, 1) for i in range(1, 8)]

    def _tool_retrieve_clinical_guidelines(self, query, disease):
        return rag_engine.search_guidelines(query, condition=disease, top_k=2)

    def _tool_check_plateau_status(self, weight_log):
        if len(weight_log) < 14:
            return {"is_plateau": False, "change_14d": round(weight_log[-1] - weight_log[0], 2)}
        recent_14 = weight_log[-14:]
        total_delta = recent_14[-1] - recent_14[0]
        is_plateau = abs(total_delta) < 0.2
        return {
            "is_plateau": is_plateau,
            "change_14d": round(total_delta, 2),
            "recommendation": "Introduce 2-day maintenance refeed to restore leptin and T3 levels" if is_plateau else "Maintain current deficit"
        }

    # ── Query Intent & Semantics Analyzer ──────────────────────
    # Keyword groups per intent. The intent with the MOST keyword matches
    # wins (ties broken by group order), so a question like "what should my
    # doctor know about my hypertension diet" routes to medical_guideline
    # (two medical hits) rather than calorie_nutrition (one hit).
    INTENT_KEYWORDS = {
        "time_to_goal":     ["time", "long", "reach", "achieve", "goal", "when",
                             "weeks", "days", "schedule", "finish", "date"],
        "calorie_nutrition": ["calorie", "eat", "intake", "deficit", "macro",
                              "food", "diet", "tdee", "protein", "meal"],
        "water_weight":     ["fluctuat", "water", "spike", "salt", "sodium",
                             "scale", "anomaly", "retention", "bounce",
                             "up and down", "vary"],
        "plateau":          ["plateau", "stuck", "slow", "stopped",
                             "not moving", "stagnant", "stagnate"],
        "medical_guideline": ["disease", "hypertension", "diabetes", "pcos",
                              "ckd", "pressure", "sugar", "condition",
                              "doctor", "health"],
    }

    def analyze_query_intent(self, question, profile, current_weight, ml_res, target_weekly_change):
        q_lower = question.lower()
        target_weight = profile.get("target_weight", current_weight - 5.0)

        scores = {
            intent: sum(1 for kw in kws if kw in q_lower)
            for intent, kws in self.INTENT_KEYWORDS.items()
        }
        best_intent = max(scores, key=lambda k: (scores[k], -list(scores).index(k)))
        if scores[best_intent] == 0:
            best_intent = "general_progress"

        # 1. Time to goal / duration
        if best_intent == "time_to_goal":
            remaining_kg = abs(current_weight - target_weight)
            weekly_rate = abs(target_weekly_change) if target_weekly_change != 0 else 0.5
            weeks_needed = remaining_kg / weekly_rate
            days_needed = int(round(weeks_needed * 7))
            return {
                "intent": "time_to_goal",
                "remaining_kg": round(remaining_kg, 1),
                "target_weight": target_weight,
                "weeks_needed": round(weeks_needed, 1),
                "days_needed": days_needed
            }

        # 2. Calorie / Food / Nutrition / Protein
        if best_intent == "calorie_nutrition":
            tdee = profile.get("tdee", 2200)
            inferred_intake = tdee + ml_res["kcal_per_day"]
            target_intake = tdee + (target_weekly_change * 7700) / 7.0
            protein_g = int(round(current_weight * 1.8))
            return {
                "intent": "calorie_nutrition",
                "tdee": int(tdee),
                "inferred_intake": int(round(inferred_intake)),
                "target_intake": int(round(target_intake)),
                "protein_g": protein_g
            }

        # 3. Fluctuation / Scale noise / Water weight / Salt
        if best_intent == "water_weight":
            return {
                "intent": "water_weight",
                "anomalies": ml_res["anomalies_detected"]
            }

        # 4. Plateau / Slowdown
        if best_intent == "plateau":
            return {
                "intent": "plateau",
                "anomalies": ml_res["anomalies_detected"]
            }

        # 5. Disease / Medical guideline
        if best_intent == "medical_guideline":
            return {
                "intent": "medical_guideline",
                "disease": profile.get("disease", "none")
            }

        # 6. General progress
        return {"intent": "general_progress"}

    # ── Tool execution pipeline ───────────────────────────────
    def run_agent(self, profile, weight_log, question, target_weekly_change=-0.5, gemini_api_key=None):
        trace = []
        disease = profile.get("disease", "none")
        if disease == "none" and profile.get("diseases"):
            disease = profile["diseases"][0] if profile["diseases"] else "none"

        ml_res = self._tool_infer_calorie_balance(weight_log, target_weekly_change)
        forecast_res = self._tool_forecast_trajectory(weight_log)
        rag_res = self._tool_retrieve_clinical_guidelines(question, disease)
        plateau_res = self._tool_check_plateau_status(weight_log)

        current_weight = round(ml_res["smoothed"][-1], 1)
        intent_info = self.analyze_query_intent(question, profile, current_weight, ml_res, target_weekly_change)

        trace.append({
            "step": 1,
            "action": "classify_intent",
            "intent": intent_info["intent"],
            "tools_executed": ["infer_calorie_balance", "forecast_trajectory",
                               "retrieve_clinical_guidelines", "check_plateau_status"],
            "note": "fixed pipeline: all tools run for every question (no LLM tool selection)"
        })

        trace.append({
            "step": 2,
            "tool_call": "infer_calorie_balance",
            "observation": {
                "current_smoothed_weight": current_weight,
                "inferred_kcal_per_day": ml_res["kcal_per_day"],
                "weekly_kg_change": ml_res["weekly_kg_change"],
                "water_spikes_filtered": ml_res["anomalies_detected"]
            }
        })

        trace.append({
            "step": 3,
            "tool_call": "forecast_trajectory",
            "observation": {"7_day_lstm_forecast": forecast_res}
        })

        trace.append({
            "step": 4,
            "tool_call": "retrieve_clinical_guidelines",
            "observation": [
                {"condition": g["condition"], "topic": g["topic"], "rule": g["recommendation"]}
                for g in rag_res
            ]
        })

        # Gemini API or Conversational Synthesis
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                system_prompt = f"""
                You are Simply-Fit AI Coach. Answer the user's specific question directly, empathetically, and accurately (<120 words).
                
                USER PROFILE & LIVE METRICS:
                - Question: "{question}"
                - Question Intent: {intent_info['intent']}
                - Calculated Intent Metrics: {json.dumps(intent_info)}
                - Current Smoothed Fat Weight: {current_weight} kg
                - Inferred Daily Calorie Deficit/Surplus: {ml_res['kcal_per_day']} kcal/day
                - 7-Day Predicted Trajectory: {forecast_res}
                - Medical Condition: {disease}
                - Retrieved Clinical RAG Guidelines: {json.dumps(rag_res)}

                Rules:
                1. Answer the exact question asked first! If asked why weight fluctuates, explain water/sodium noise. If asked about time to goal, give exact weeks/days calculation.
                2. Use the calculated intent data provided above.
                3. Keep response concise, warm, professional, and medically safe.
                """
                try:
                    res = client.models.generate_content(model="gemini-2.5-flash", contents=system_prompt)
                    response_text = res.text
                except Exception:
                    res = client.models.generate_content(model="gemini-1.5-flash", contents=system_prompt)
                    response_text = res.text
            except Exception as e:
                response_text = self._smart_intent_synthesis(intent_info, question, current_weight, ml_res, target_weekly_change, profile, plateau_res, rag_res)
        else:
            response_text = self._smart_intent_synthesis(intent_info, question, current_weight, ml_res, target_weekly_change, profile, plateau_res, rag_res)

        return {
            "trace": trace,
            "response": response_text,
            "ml_output": ml_res,
            "forecast": forecast_res,
            "intent": intent_info
        }

    def _smart_intent_synthesis(self, intent_info, question, current_weight, ml_res, target_weekly_change, profile, plateau_res, rag_res):
        intent = intent_info["intent"]

        # Intent 1: Time to Goal
        if intent == "time_to_goal":
            rem_kg = intent_info["remaining_kg"]
            target_w = intent_info["target_weight"]
            weeks = intent_info["weeks_needed"]
            days = intent_info["days_needed"]
            target_date = (datetime.now() + timedelta(days=days)).strftime("%d %b %Y")

            return (
                f"🎯 **Estimated Timeline to Reach Goal:**\n\n"
                f"To reach your goal weight of **{target_w} kg** from your current smoothed weight of **{current_weight} kg** ({rem_kg} kg remaining), "
                f"at your target rate of **{abs(target_weekly_change):.1f} kg/week**, it will take approximately **{weeks} weeks** (~**{days} days**).\n\n"
                f"📅 **Estimated Goal Completion Date:** **{target_date}**."
            )

        # Intent 2: Calorie & Nutrition
        if intent == "calorie_nutrition":
            tdee = intent_info["tdee"]
            inferred = intent_info["inferred_intake"]
            target_intake = intent_info["target_intake"]
            protein_g = intent_info["protein_g"]

            return (
                f"🥗 **Calorie & Macro Target Breakdown:**\n\n"
                f"• **Maintenance (TDEE):** {tdee} kcal/day\n"
                f"• **Inferred Current Intake:** ~{inferred} kcal/day (*{ml_res['kcal_per_day']:+.0f} kcal balance*)\n"
                f"• **Recommended Target Intake:** **{target_intake} kcal/day** for {target_weekly_change:+.1f} kg/week pace.\n"
                f"• **Recommended Daily Protein:** **{protein_g}g / day** (1.8 g/kg) to preserve lean muscle."
            )

        # Intent 3: Fluctuation / Scale Noise / Water Weight
        if intent == "water_weight":
            anomalies = intent_info["anomalies"]
            return (
                f"🌊 **Why Daily Scale Weight Fluctuates:**\n\n"
                f"Daily scale readings fluctuate by 0.5–1.5 kg due to sodium retention, glycogen storage, and digestive hydration—not actual fat gain.\n\n"
                f"• Isolation Forest flagged **{anomalies} temporary water spikes** in your log.\n"
                f"• Your true **EWMA smoothed fat mass trend** is **{current_weight} kg**."
            )

        # Intent 4: Plateau
        if intent == "plateau":
            is_p = plateau_res.get("is_plateau")
            change_14 = plateau_res.get("change_14d", 0.0)
            if is_p:
                return (
                    f"⚠️ **Metabolic Plateau Analysis:**\n\n"
                    f"Your 14-day weight change is **{change_14:+.2f} kg**, indicating a weight plateau due to metabolic slowdown.\n\n"
                    f"💡 **Recommendation:** Introduce a 2-day diet break at maintenance calories (~{profile.get('tdee', 2200)} kcal/day) to restore leptin and thyroid hormone (T3) levels."
                )
            else:
                return (
                    f"✅ **No Plateau Detected:**\n\n"
                    f"Your weight is steadily changing ({change_14:+.2f} kg over 14 days). "
                    f"Your current daily calorie balance is **{ml_res['kcal_per_day']:+.0f} kcal/day**."
                )

        # Intent 5: Medical Guidelines
        if intent == "medical_guideline":
            disease = profile.get("disease", "none")
            msg = f"🩺 **Medical Guidelines for {disease}:**\n\n"
            if rag_res and disease != "none":
                top = rag_res[0]
                msg += f"• **Clinical Protocol:** {top['recommendation']}\n• **Source:** {top['source']}"
            else:
                msg += f"• No specific medical conditions flagged. Ensure adequate hydration and balanced micronutrients."
            return msg

        # Intent 6: General Progress
        weekly = ml_res["weekly_kg_change"]
        on_track = abs(weekly - target_weekly_change) < 0.15

        msg = f"📊 **Your Weight Trend & Progress:**\n\n"
        msg += f"• **Current Smoothed Weight:** {current_weight} kg\n"
        msg += f"• **Weekly Rate:** {weekly:+.2f} kg/week (Target: {target_weekly_change:+.1f} kg/week)\n"
        msg += f"• **Inferred Daily Balance:** {ml_res['kcal_per_day']:+.0f} kcal/day\n\n"
        msg += "You are right on pace! Keep going." if on_track else "You are slightly off pace. A small daily adjustment of ~150 kcal will bring you back on schedule."
        
        return msg


# Global Agent instance
agent_coach = SimplyFitAgent()
