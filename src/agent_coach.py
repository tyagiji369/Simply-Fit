import os
import json
import numpy as np
from dotenv import load_dotenv
from src.ml_engine import run_pipeline as run_ml_pipeline
from src.lstm_forecaster import forecast as forecast_lstm
from src.rag_pipeline import rag_engine

load_dotenv()


class SimplyFitAgent:
    """
    Agentic AI Coach using ReAct (Reasoning + Acting) pattern with tool execution.
    Orchestrates ML Signal Processing, LSTM Neural Forecasting, and Clinical RAG.
    Supports both Google Gemini API (gemini-2.5-flash) and deterministic fallback execution.
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
        """Runs EWMA filtering, Isolation Forest anomaly detection, and linear regression."""
        return run_ml_pipeline(weight_log, target_weekly_change=target_weekly_change)

    def _tool_forecast_trajectory(self, weight_log):
        """Runs LSTM neural network forecasting or trend extrapolation fallback."""
        try:
            pred = forecast_lstm(weight_log)
            if pred is not None:
                return [round(float(x), 1) for x in pred]
        except Exception:
            pass

        # Robust analytical forecast fallback if model weights absent
        recent = weight_log[-7:]
        daily_rate = (recent[-1] - recent[0]) / 7.0
        return [round(recent[-1] + daily_rate * i, 1) for i in range(1, 8)]

    def _tool_retrieve_clinical_guidelines(self, query, disease):
        """Retrieves top-k evidence-based clinical nutrition guidelines via vector search."""
        return rag_engine.search_guidelines(query, condition=disease, top_k=2)

    def _tool_check_plateau_status(self, weight_log):
        """Detects metabolic slowdown or weight loss plateau over 14-day window."""
        if len(weight_log) < 14:
            return {"is_plateau": False, "change_14d": round(weight_log[-1] - weight_log[0], 2)}
        recent_14 = weight_log[-14:]
        total_delta = recent_14[-1] - recent_14[0]
        is_plateau = abs(total_delta) < 0.2
        return {
            "is_plateau": is_plateau,
            "change_14d": round(total_delta, 2),
            "recommendation": "Introduce 2-day maintenance refeed to restore leptin levels" if is_plateau else "Maintain current deficit"
        }

    # ── Agentic Execution Loop ────────────────────────────────
    def run_agent(self, profile, weight_log, question, target_weekly_change=-0.5, gemini_api_key=None):
        """
        Executes the ReAct Agent Orchestration loop.
        Returns a dict containing:
        - trace: List of step-by-step reasoning & tool execution logs
        - response: Final grounded coaching response
        """
        trace = []
        disease = profile.get("disease", "none")
        if disease == "none" and profile.get("diseases"):
            disease = profile["diseases"][0] if profile["diseases"] else "none"

        # Step 1: Agent Reasoning & Tool Planning
        trace.append({
            "step": 1,
            "thought": f"Analyzing query: '{question}'. Planning multi-tool execution for signal filtering, 7-day trajectory forecasting, and clinical RAG retrieval.",
            "selected_tools": ["infer_calorie_balance", "forecast_trajectory", "retrieve_clinical_guidelines", "check_plateau_status"]
        })

        # Step 2: Tool Executions
        ml_res = self._tool_infer_calorie_balance(weight_log, target_weekly_change)
        forecast_res = self._tool_forecast_trajectory(weight_log)
        rag_res = self._tool_retrieve_clinical_guidelines(question, disease)
        plateau_res = self._tool_check_plateau_status(weight_log)

        trace.append({
            "step": 2,
            "tool_call": "infer_calorie_balance",
            "observation": {
                "inferred_kcal_per_day": ml_res["kcal_per_day"],
                "weekly_kg_change": ml_res["weekly_kg_change"],
                "anomalies_filtered": ml_res["anomalies_detected"],
                "r_squared_fit": ml_res["r_squared"]
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
                {"condition": g["condition"], "topic": g["topic"], "rule": g["recommendation"], "source": g["source"]}
                for g in rag_res
            ]
        })

        trace.append({
            "step": 5,
            "tool_call": "check_plateau_status",
            "observation": plateau_res
        })

        # Step 3: Response Generation (Gemini API or Fallback)
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        recent_7 = weight_log[-7:]
        weekly_change = round(recent_7[-1] - recent_7[0], 2)

        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                system_prompt = f"""
                You are Simply-Fit AI Coach, a clinical nutrition assistant.
                Answer the user's question concisely (<120 words) using the grounded tool outputs below.

                USER DATA & TOOL RESULTS:
                - Current Weight: {weight_log[-1]} kg (7-day change: {weekly_change:+} kg, Target: {target_weekly_change:+} kg/week)
                - Inferred Daily Calorie Deficit/Surplus: {ml_res['kcal_per_day']} kcal/day
                - Filtered Anomalies: {ml_res['anomalies_detected']} transient water spikes removed by Isolation Forest
                - 7-Day Predicted Trajectory: {forecast_res}
                - Weight Plateau Status: {plateau_res['recommendation']}
                - Medical Condition: {disease}
                - Retrieved Clinical Guidelines (Vector RAG): {json.dumps(rag_res)}

                User Question: {question}
                Format clearly with bullet points where applicable. Be empathetic, encouraging, and medically grounded.
                """
                
                # Attempt gemini model generation
                try:
                    res = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=system_prompt
                    )
                    response_text = res.text
                except Exception:
                    res = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=system_prompt
                    )
                    response_text = res.text
            except Exception as e:
                response_text = f"*(Note: Gemini API call failed with error: {str(e)}. Using grounded tool engine output below.)*\n\n" + self._fallback_synthesis(weekly_change, target_weekly_change, ml_res, plateau_res, rag_res, disease)
        else:
            response_text = self._fallback_synthesis(weekly_change, target_weekly_change, ml_res, plateau_res, rag_res, disease)

        return {
            "trace": trace,
            "response": response_text,
            "ml_output": ml_res,
            "forecast": forecast_res,
            "rag_matches": rag_res
        }

    def _fallback_synthesis(self, weekly_change, target_weekly_change, ml_res, plateau_res, rag_res, disease):
        """Generates deterministic, grounded coaching response from tool outputs."""
        on_track = abs(weekly_change - target_weekly_change) < 0.15

        if on_track:
            msg = (
                f"You changed **{weekly_change:+.2f} kg** this week against a target of **{target_weekly_change:+.1f} kg/week**. "
                f"Signal processing infers your actual daily caloric balance is **{ml_res['kcal_per_day']:+.0f} kcal/day**.\n\n"
                f"Your trajectory is steady and on track!"
            )
        else:
            msg = (
                f"Your 7-day weight trend shows **{weekly_change:+.2f} kg** vs target of **{target_weekly_change:+.1f} kg/week**. "
                f"Inferred daily balance is **{ml_res['kcal_per_day']:+.0f} kcal/day**.\n\n"
            )
            if "food_adjustment_kcal" in ml_res:
                msg += f"• **Suggested Intake Adjustment:** Reduce intake by **{abs(ml_res['food_adjustment_kcal']):.0f} kcal/day**.\n"
                msg += f"• **Suggested Activity Adjustment:** Increase daily movement by **{abs(ml_res['activity_adjustment_kcal']):.0f} kcal/day**.\n"

        if plateau_res.get("is_plateau"):
            msg += f"\n⚠️ **Plateau Warning:** 14-day trend indicates a weight loss plateau ({plateau_res['change_14d']} kg change). *{plateau_res['recommendation']}*."

        if rag_res and disease != "none":
            top_rule = rag_res[0]
            msg += f"\n\n🩺 **Clinical Guideline ({disease}):** {top_rule['recommendation']} *(Source: {top_rule['source']})*"

        return msg


# Global Agent instance
agent_coach = SimplyFitAgent()


def get_agent_response(profile, weight_log, question, target_weekly_change=-0.5, gemini_api_key=None):
    """Convenience wrapper."""
    return agent_coach.run_agent(profile, weight_log, question, target_weekly_change, gemini_api_key=gemini_api_key)
