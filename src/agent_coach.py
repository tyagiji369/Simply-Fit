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
    Intelligent AI Coach for Simply-Fit.
    Supports intent classification, dynamic calculations (time to goal, calorie targets,
    water retention, medical RAG), Google Gemini API, and smart fallback synthesis.
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

    # ── Intent Analysis & Mathematical Calculations ────────────
    def analyze_query_intent(self, question, profile, current_weight, ml_res, target_weekly_change):
        q_lower = question.lower()
        target_weight = profile.get("target_weight", current_weight - 5.0)
        
        # 1. Time to goal intent
        if any(w in q_lower for w in ["time", "long", "reach", "achieve", "goal", "when", "weeks", "days"]):
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
        
        # 2. Calorie / Nutrition intake intent
        if any(w in q_lower for w in ["calorie", "eat", "intake", "deficit", "macro", "food", "diet", "tdee"]):
            tdee = profile.get("tdee", 2200)
            inferred_intake = tdee + ml_res["kcal_per_day"]
            target_intake = tdee + (target_weekly_change * 7700) / 7.0
            return {
                "intent": "calorie_nutrition",
                "tdee": int(tdee),
                "inferred_intake": int(round(inferred_intake)),
                "target_intake": int(round(target_intake))
            }
        
        # 3. Water weight / Anomaly intent
        if any(w in q_lower for w in ["water", "spike", "salt", "sodium", "fluctuate", "anomaly", "retention"]):
            return {
                "intent": "water_weight",
                "anomalies": ml_res["anomalies_detected"]
            }

        # 4. Disease / Medical guideline intent
        if any(w in q_lower for w in ["disease", "hypertension", "diabetes", "pcos", "ckd", "pressure", "sugar", "condition", "doctor"]):
            return {
                "intent": "medical_guideline",
                "disease": profile.get("disease", "none")
            }

        # 5. General progress intent
        return {"intent": "general_progress"}

    # ── Agentic Execution Loop ────────────────────────────────
    def run_agent(self, profile, weight_log, question, target_weekly_change=-0.5, gemini_api_key=None):
        trace = []
        disease = profile.get("disease", "none")
        if disease == "none" and profile.get("diseases"):
            disease = profile["diseases"][0] if profile["diseases"] else "none"

        # Step 1: Tool Executions
        ml_res = self._tool_infer_calorie_balance(weight_log, target_weekly_change)
        forecast_res = self._tool_forecast_trajectory(weight_log)
        rag_res = self._tool_retrieve_clinical_guidelines(question, disease)
        plateau_res = self._tool_check_plateau_status(weight_log)

        current_weight = round(ml_res["smoothed"][-1], 1)
        intent_info = self.analyze_query_intent(question, profile, current_weight, ml_res, target_weekly_change)

        trace.append({
            "step": 1,
            "thought": f"Query intent classified as '{intent_info['intent']}'. Executing EWMA signal processing, Isolation Forest anomaly filter, LSTM forecasting, and Clinical RAG vector retrieval.",
            "selected_tools": ["infer_calorie_balance", "forecast_trajectory", "retrieve_clinical_guidelines", "check_plateau_status"]
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

        # Gemini API or Intent-Matched Synthesis
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                system_prompt = f"""
                You are Simply-Fit AI Coach. Answer the user's specific question directly and concisely (<100 words).
                
                USER PROFILE & Intent Data:
                - Question: "{question}"
                - Question Intent: {intent_info['intent']}
                - Calculated Intent Data: {json.dumps(intent_info)}
                - Current Smoothed Weight: {current_weight} kg
                - Inferred Daily Deficit/Surplus: {ml_res['kcal_per_day']} kcal/day
                - 7-Day LSTM Forecast: {forecast_res}
                - Medical Condition: {disease}
                - Retrieved Medical Guidelines: {json.dumps(rag_res)}

                Rules:
                1. Answer the exact question asked first! (If asked about time to goal, give exact weeks/days calculation. If asked about calories, give intake numbers).
                2. Be encouraging, precise, and concise.
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
                f"🎯 **Estimated Time to Reach Goal:**\n\n"
                f"• **Current Smoothed Weight:** {current_weight} kg\n"
                f"• **Target Goal Weight:** {target_w} kg (*{rem_kg} kg remaining*)\n"
                f"• **Target Pace:** {abs(target_weekly_change):.1f} kg/week\n\n"
                f"At your target rate of **{abs(target_weekly_change):.1f} kg/week**, it will take approximately **{weeks} weeks** (~**{days} days**). "
                f"Your estimated goal completion date is **{target_date}**."
            )

        # Intent 2: Calorie & Nutrition
        if intent == "calorie_nutrition":
            tdee = intent_info["tdee"]
            inferred = intent_info["inferred_intake"]
            target_intake = intent_info["target_intake"]

            return (
                f"🥗 **Daily Calorie & Nutrition Breakdown:**\n\n"
                f"• **Estimated TDEE (Maintenance):** {tdee} kcal/day\n"
                f"• **Inferred Current Intake:** ~{inferred} kcal/day (*{ml_res['kcal_per_day']:+.0f} kcal balance*)\n"
                f"• **Recommended Daily Intake:** **{target_intake} kcal/day** to achieve your pace of {target_weekly_change:+.1f} kg/week."
            )

        # Intent 3: Water Weight
        if intent == "water_weight":
            anomalies = intent_info["anomalies"]
            return (
                f"💧 **Water Weight & Sodium Fluctuation:**\n\n"
                f"Isolation Forest detected **{anomalies} temporary water spikes** in your log. "
                f"Daily scale weight varies by 0.5–1.5 kg due to sodium retention and glycogen storage. "
                f"Your true EWMA smoothed fat mass trend is **{current_weight} kg**."
            )

        # Intent 4: Medical Guidelines
        if intent == "medical_guideline":
            disease = profile.get("disease", "none")
            msg = f"🩺 **Medical Guidance for {disease}:**\n\n"
            if rag_res and disease != "none":
                top = rag_res[0]
                msg += f"• **Clinical Protocol:** {top['recommendation']} *(Source: {top['source']})*\n"
            else:
                msg += f"• No specific medical restrictions set. Ensure balanced micronutrient intake and stay hydrated."
            return msg

        # Intent 5: General Progress
        recent = ml_res["smoothed"]
        weekly = ml_res["weekly_kg_change"]
        on_track = abs(weekly - target_weekly_change) < 0.15

        msg = f"📊 **Your Progress Summary:**\n\n"
        msg += f"• **Current Smoothed Weight:** {current_weight} kg\n"
        msg += f"• **7-Day Trend:** {weekly:+.2f} kg/week (Target: {target_weekly_change:+.1f} kg/week)\n"
        msg += f"• **Inferred Daily Balance:** {ml_res['kcal_per_day']:+.0f} kcal/day\n\n"
        msg += "You are right on track! Keep up your current routine." if on_track else "You are slightly off pace. Consider adjusting daily intake by ~150 kcal."
        
        return msg


# Global Agent instance
agent_coach = SimplyFitAgent()
